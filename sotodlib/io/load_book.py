"""load_obs_book

This module supports loading SO "obs/oper" books (i.e. bookbound
detector data with encoder and timestamp information) into AxisManager
data structure.

The two access points in this submodule are:

  load_obs_book

    This is a Context-compatible loader function, to be used when a
    full obsfiledb is available.

  load_book_file

    This can be used to load single G3 files (or a set of
    sample-contiguous G3 files) from a book, by filename.

"""

import so3g
from spt3g import core as spt3g_core
import numpy as np

from glob import glob
import itertools
import logging
import os
import re
import yaml

import sotodlib
from sotodlib import core
from .check_book import _compact_list  # just a list with a limited repr
from . import load_smurf


logger = logging.getLogger(__name__)

_TES_BIAS_COUNT = 12  # per detset / primary file group

#: Signal DAC units are rescaled to phase before returning.
SIGNAL_RESCALE = np.pi / 2**15

DEG = np.pi / 180


def load_obs_book(db, obs_id, dets=None, prefix=None, samples=None,
                  no_signal=None,
                  **kwargs):
    """Obsloader function for SO "Level 3" obs/oper Books.

    See API template, `sotodlib.core.context.obsloader_template`, for
    details of all supported arguments.

    """
    if any([v is not None for v in kwargs.values()]):
        raise RuntimeError(
            f"This loader function does not understand these kwargs: f{kwargs}")

    if prefix is None:
        prefix = db.prefix
        if prefix is None:
            prefix = './'

    if no_signal is None:
        no_signal = False  # from here, assume no_signal in [True, False]

    # Regardless of what dets have been asked for (maybe none), get
    # the list of detsets implicated in this observation.  Make sure
    # this list is ordered by detset or the signal might not be
    # ordered properly (checked later).
    c = db.conn.execute('select distinct DS.name, DS.det from detsets DS '
                        'join files on DS.name=files.detset '
                        'where obs_id=? '
                        'order by DS.name',
                        (obs_id,))
    all_pairs = [tuple(r) for r in c.fetchall()]

    # Now filter to only the dets requested.
    if dets is None:
        pairs_req = all_pairs
    else:
        pairs_req = [p for p in all_pairs if p[1] in dets]
        dets_req = [p[1] for p in pairs_req]
        unmatched = [d for d in dets if d not in dets_req]
        if len(unmatched):
            raise RuntimeError("User requested invalid dets (e.g. %s) "
                               "for obs_id=%s" % (unmatched[0], obs_id))
        del dets_req, unmatched
    del all_pairs, dets

    # Make sure "pairs" is sorted, at _least_ at the level of grouping
    # detsets together; then make sure the detsets are processed in
    # that order.
    detsets_req = sorted(set([p[0] for p in pairs_req]))
    dets_req = []
    for _ds in detsets_req:
        dets_req.extend([p[1] for p in pairs_req if p[0] == _ds])
    del pairs_req

    file_map = db.get_files(obs_id, detsets=detsets_req)
    one_group = list(file_map.values())[0]  # [('file0', 0, 1000), ('file1', 1000, 2000), ...]
    
    # Figure out how many samples we're loading.
    sample_range = one_group[0][1], one_group[-1][2]
    if samples is None:
        samples = None, None
    samples = list(samples)
    if samples[0] is None:
        samples[0] = sample_range[0]
    if samples[1] is None:
        samples[1] = sample_range[1]
    elif samples[1] < 0:
        samples[1] = sample_range[1] + samples[1]
    samples[0] = min(max(0, samples[0]), sample_range[1])
    samples[1] = min(max(samples), sample_range[1])

    # Consider pre-allocating the signal buffer.
    signal_buffer = None
    if samples[1] is not None and not no_signal:
        signal_buffer = np.empty((len(dets_req), samples[1] - samples[0]),
                                 dtype='float32')

    ancil = None
    timestamps = None
    results = {}

    for detset in detsets_req:
        files = file_map[detset]
        results[detset] = _load_book_detset(
            files, prefix=prefix, load_ancil=(ancil is None),
            samples=samples, dets=dets_req, no_signal=no_signal,
            signal_buffer=signal_buffer)
        if ancil is None:
            ancil = results[detset]['ancil']
            timestamps = results[detset]['timestamps']

    if len(results) == 0:
        # Load the ancil files, to get ancil stuff.
        _one_fileset = next(iter(file_map.values()))
        ancil_files = _get_ancil_files(_one_fileset)
        _res = _load_book_detset(ancil_files, prefix=prefix, load_ancil=True,
                                 samples=samples, dets=[])
        ancil = _res['ancil']
        timestamps = _res['timestamps']

    obs = _concat_filesets(results, ancil, timestamps,
                           sample0=samples[0], obs_id=obs_id,
                           signal_buffer=signal_buffer,
                           get_frame_det_info=False)
    if signal_buffer is not None:
        # Make sure that, whatever happened during concatenation, the
        # dets are still ordered as was assumed by signal_buffer.
        assert(np.all(obs.dets.vals == dets_req))
    return obs

def load_book_file(filename, dets=None, samples=None, no_signal=False):
    """Load one or more g3 files (from an obs/oper book) and return the
    contents as an AxisManager.

    Args:

      filename (str or list of str): The g3 file(s) to load.  See
        Notes.
      dets (None, or list of str): The detectors (readout_id) to load.
      samples (tuple or None): Sample range to load.
      no_signal (bool): If True, the signal data are not read and
        .signal is set to None.

    Notes:

      The filename argument can be a string or a list of strings.  If
      it's a single string, and contains a wildcard (*), then the
      pattern will be globbed and sorted and all those files will be
      loaded.

      When loading multiple files, they must be from the **same
      fileset / stream_id** (i.e. contain the same detectors).

      The "samples" argument should be a tuple (start, stop), and will
      be interpreted as sample counts within the stream of samples
      presented in the requested files, starting from 0.  The "dets"
      argument, if passed, must be a list of strings; elements in that
      list that aren't found in the data will simply be ignored (so
      the returned signal may have fewer dets than were requested).

    """
    if samples is None:
        samples = [0, None]
    else:
        samples = list(samples)
        if samples[0] is None:
            samples[0] = 0

    if isinstance(filename, str):
        if '*' in filename:
            filename = sorted(glob(filename))
        else:
            filename = [filename]

    files = [(f, None, None) for f in filename]
    this_detset = _load_book_detset(
        files, load_ancil=True,
        samples=samples, dets=dets, no_signal=no_signal)

    return _concat_filesets({'?': this_detset},
                            this_detset['ancil'],
                            this_detset['timestamps'],
                            sample0=samples[0],
                            no_signal=no_signal)


def load_smurf_npy_data(ctx, obs_id, substr):
    """
    Loads an sodetlib npy file from Z_smurf archive of book.

    Args
    _____
    obs_id: str
        obs-id of book to load file from
    substr: str
        substring to use to find numpy file in Z_smurf
    """
    files = ctx.obsfiledb.get_files(obs_id)
    book_dir = os.path.dirname(list(files.values())[0][0][0])
    smurf_dir = os.path.join(book_dir, 'Z_smurf')
    for f in os.listdir(smurf_dir):
        if substr in f:
            fpath = os.path.join(smurf_dir, f)
            break
    else:
        raise FileNotFoundError("Could not find npy file")
    res = np.load(fpath, allow_pickle=True).item()
    return res


def _load_book_detset(files, prefix='', load_ancil=True,
                      dets=None, samples=None, no_signal=False,
                      signal_buffer=None):
    """Read data from a single detset.

    If a list of dets is specified, it may include dets that aren't
    found in this set of files.  It is thus used only to screen
    detectors and to set the order they appear in the output.  The
    actual detectors found and loaded are returned as 'dets' in the
    output.

    """
    stream_id = None
    ancil_acc = None
    times_acc = None
    if load_ancil:
        times_acc = Accumulator1d(samples=samples)
        ancil_acc = AccumulatorTimesampleMap(samples=samples)
    primary_acc = AccumulatorNamed(samples=samples)
    bias_names = []
    bias_acc = Accumulator2d(samples=samples)
    signal_acc = None
    this_stream_dets = None

    if no_signal:
        signal_acc = None
    elif signal_buffer is not None:
        signal_acc = Accumulator2d(
            samples=samples,
            insert_at=signal_buffer,
            keys_to_keep=dets,
            calibrate=SIGNAL_RESCALE)
    else:
        signal_acc = Accumulator2d(
            samples=samples,
            keys_to_keep=dets,
            calibrate=SIGNAL_RESCALE)

    # Sniff out a smurf status frame.
    smurf_proc = load_smurf.SmurfStatus._get_frame_processor()

    for frame, frame_offset in _frames_iterator(files, prefix, samples,
                                                smurf_proc=smurf_proc):
        more_data = True

        # Anything in ancil should be identical across
        # filesets, so only process it once.
        if load_ancil:
            more_data &= times_acc.append(frame['ancil'].times, frame_offset)
            more_data &= ancil_acc.append(frame['ancil'], frame_offset)

        if 'stream_id' in frame:
            if stream_id is None:
                stream_id = frame['stream_id']
            assert (stream_id == frame['stream_id'])  # check your data files

        if 'primary' in frame:
            more_data &= primary_acc.append(frame['primary'], frame_offset)
            bias_names = _check_bias_names(frame)[:_TES_BIAS_COUNT]
            more_data &= bias_acc.append(frame['tes_biases'], frame_offset)

        if 'signal' in frame:
            # Even if no_signal, we need the det list.
            if this_stream_dets is None:
                this_stream_dets = _compact_list(frame['signal'].names)

            # Extract the main signal
            if not no_signal:
                more_data &= signal_acc.append(frame['signal'], frame_offset)

        if not more_data:
            break

    if times_acc is not None:
        times_acc = times_acc.finalize() / spt3g_core.G3Units.sec

    req_dets_in_stream = None
    det_idx_in_stream = None
    if this_stream_dets:
        # Of the requested detectors, what ones were actually found in
        # this stream?  Keep ordering same as the request, which
        # should be the same as the populated data.
        if dets is None:
            req_dets_in_stream = this_stream_dets
        else:
            req_dets_in_stream = _compact_list(
                [d for d in dets if d in this_stream_dets])

        # For each loaded detector, what was its index within the stream's
        # dets?
        det_idx_in_stream = _compact_list([this_stream_dets.index(d)
                                           for d in req_dets_in_stream])

    stat = smurf_proc.get_status()
    ch_info = None
    iir_params = None
    if stat.num_chans is None:
        # Try to grab it from file 000?
        try_file = re.sub(r'_\d\d\d\.g3', '_000.g3', files[0][0])
        if try_file != files[0][0] and os.path.exists(try_file):
            logger.warning(f'Trying to get SmurfStatus from {try_file} ...')
            stat = load_smurf.SmurfStatus.from_file(try_file)
            if stat.num_chans is None:
                logger.warning('... it did not work.')
                # No wiring frames probably means it's an A_* (ancil) file.

    if stat.num_chans is not None:
        # This is an AxisManager, with dets axis for just this stream
        # ... extract and stack data later.
        ch_info = load_smurf.get_channel_info(stat, mask=det_idx_in_stream)
        # And this stuff is per stream, so keep it separate.
        iir_params = {'enabled': stat.filter_enabled,
                      'b': stat.filter_b,
                      'a': stat.filter_a,
                      'fscale': 1. / stat.flux_ramp_rate_hz}

    return {
        'stream_id': stream_id,
        'signal': signal_acc,
        'dets': req_dets_in_stream,
        'primary': primary_acc,
        'biases': bias_acc,
        'bias_names': bias_names,
        'smurf_ch_info': ch_info,
        'iir_params': iir_params,
        'ancil': ancil_acc,
        'timestamps': times_acc,
    }


def _concat_filesets(results, ancil=None, timestamps=None,
                     sample0=0, obs_id=None, dets=None,
                     no_signal=False, signal_buffer=None,
                     get_frame_det_info=True):
    """Assemble multiple detset results (as returned by _load_book_detset)
    into a full AxisManager.

    """
    if ancil is None:
        ancil = next(iter(results))['ancil']
    if timestamps is None:
        timestamps = next(iter(results))['timestamps']

    if dets is None:
        dets = list(itertools.chain(*[r['dets'] for r in results.values()]))

    aman = core.AxisManager(
        core.LabelAxis('dets', dets),
        core.OffsetAxis('samps',
                        count=len(timestamps),
                        offset=sample0,
                        origin_tag=obs_id))

    aman.wrap('timestamps', timestamps, axis_map=[(0, 'samps')])

    if ancil is not None:
        # Handle ancillary fields, a.k.a. boresight pointing /
        # rotation / corotator.
        aman.wrap('ancil', core.AxisManager(aman.samps))

        # Put all fields into 'ancil'.
        _a = aman['ancil']
        for k, v in ancil.finalize().items():
            _a.wrap(k, v, [(0, 'samps')])

        # Transform some fields into 'boresight'.
        if 'az_enc' in _a and 'el_enc' in _a:
            aman.wrap('boresight', core.AxisManager(aman.samps))
            _b = aman['boresight']
            _b.wrap('az', _a['az_enc'] * DEG, [(0, 'samps')])
            _b.wrap('el', _a['el_enc'] * DEG, [(0, 'samps')])

            roll = None
            if 'boresight_enc' in _a:
                roll = -1*_a['boresight_enc']
            elif 'corotator_enc' in _a:
                roll = _a['el_enc'] - 60 - _a['corotator_enc']
            if roll is None:
                _b.wrap('roll', None)
            else:
                _b.wrap('roll', roll * DEG, [(0, 'samps')])

    if len(results) == 0:
        return aman

    one_result = next(iter(results.values()))
    no_signal = one_result['signal'] is None
    if no_signal:
        aman.wrap('signal', None)
    else:
        if signal_buffer is not None:
            aman.wrap('signal', signal_buffer,
                      [(0, 'dets'), (1, 'samps')])
        else:
            aman.wrap_new('signal', shape=('dets', 'samps'), dtype='float32')
            dets_ofs = 0
            for v in results.values():
                d = v['signal'].finalize()
                aman['signal'][dets_ofs:dets_ofs + len(d)] = d
                dets_ofs += len(d)

    # In sims, the whole primary block may be unpopulated.
    if any([v['primary'].data is not None for v in results.values()]):
        # Biases
        all_bias_names = []
        for v in results.values():
            all_bias_names.extend(v['bias_names'][:_TES_BIAS_COUNT])
        aman.merge(core.AxisManager(core.LabelAxis('bias_lines', all_bias_names)))
        aman.wrap_new('biases', shape=('bias_lines', 'samps'), dtype='int32')
        for i, v in enumerate(results.values()):
            aman['biases'][i * _TES_BIAS_COUNT:(i + 1) * _TES_BIAS_COUNT, :] = \
                v['biases'].finalize()[:_TES_BIAS_COUNT, :]

        # Primary (and other stuff to group per-stream)
        aman.wrap('primary', core.AxisManager(aman.samps))
        aman.wrap('iir_params', core.AxisManager())
        aman['iir_params'].wrap('per_stream', True)
        for r in results.values():
            # Primary.
            _prim = core.AxisManager(aman.samps)
            for k, v in r['primary'].finalize().items():
                _prim.wrap(k, v, [(0, 'samps')])
            aman['primary'].wrap(r['stream_id'], _prim)
            # Filter parameters
            _iir = None
            if r.get('iir_params') is not None:
                _iir = core.AxisManager()
                for k, v in r['iir_params'].items():
                    _iir.wrap(k, v)
            aman['iir_params'].wrap(r['stream_id'], _iir)

    # flags place
    aman.wrap("flags", core.FlagManager.for_tod(aman, "dets", "samps"))

    if not get_frame_det_info:
        return aman

    # The detset, stream_id, and smurf.* channel info will normally be
    # populated by a downstream data product, so that they are
    # available without having to read the main G3 data (and thus with
    # get_meta).  But the block below should be maintained for use
    # with load_book_file, where the user is unlikely to also have
    # good metadata ready to go.
    #
    # Even if the smurf info isn't merged in here, it still gets
    # parsed.  The main need seems to be to populate the iir_params.

    # det_info
    det_info = core.metadata.ResultSet(
        ['detset', '_readout_id', 'stream_id'])
    ch_info = None  # or False, or dict.
    for detset, r in results.items():
        det_info.rows.extend(
            [(detset, _d, r['stream_id']) for _d in r['dets']])

        if r['smurf_ch_info'] is None:
            ch_info = False
            break

        _ch_info_keys = list(r['smurf_ch_info']._fields.keys())

        if ch_info is None:
            ch_info = {k: [] for k in _ch_info_keys}
        elif set(ch_info.keys()) != set(_ch_info_keys):
            ch_info = False
            break

    if ch_info is False:
        logger.warning('Missing or inconsistent smurf status fields; '
                       'dropping det_info.smurf.')

    if ch_info:
        for detset, r in results.items():
            if r['smurf_ch_info'] is None:
                break
            for k, v in r['smurf_ch_info']._fields.items():
                if k not in ch_info:
                    ch_info[k] = []
                ch_info[k].extend(v)

    aman.wrap('det_info', det_info.to_axismanager(axis_key='_readout_id'))

    if ch_info:
        smurf = core.AxisManager(aman.dets)
        for k, v in ch_info.items():
            smurf.wrap(k, np.array(v), [(0, 'dets')])
        aman['det_info'].wrap('smurf', smurf)

    return aman


def _check_bias_names(frame):
    """Verify the frame has TES biases with expected names; return the
    modified names that include stream_id.

    """
    for i, name in enumerate(frame['tes_biases'].names):
        if name != 'bias%02i' % i:
            raise RuntimeError(f'Bias at index {i} has unexpected name "{name}"!')
    stream_id = frame['stream_id']
    return [f'{stream_id}_b{_i:02d}' for _i in range(i+1)]


def _get_ancil_files(non_ancil_files):
    def _rewrite_entry(row):
        f, etc = row[0], row[1:]
        p, b = os.path.split(f)
        tokens = b.split('_')
        a = os.path.join(p, 'A_ancil_' + tokens[-1])
        return tuple([a] + list(etc))
    return [_rewrite_entry(x) for x in non_ancil_files]


class Accumulator:
    def __init__(self, shape=None, samples=None, preconsumed=None):
        if samples is None:
            samples = None, None
        samples = list(samples)
        if samples[0] == None:
            samples[0] = 0
        self.samples = samples

        self.shape = None
        self.data = None
        self.consumed = preconsumed

    def append(self, data, preconsumed=None):
        if self.consumed is None:
            if preconsumed is None:
                preconsumed = 0
            self.consumed = preconsumed
        if preconsumed is not None:
            assert(self.consumed == preconsumed)

        data_count = self._sample_count(data)

        # global sample indices of our destination buffer are
        # self.samples; global sample indices of this block are:
        block_samples = [self.consumed, self.consumed + data_count]

        # Does this block precede our area of interest?
        if block_samples[1] <= self.samples[0]:
            self.consumed += data_count
            return True

        # Overlap?
        over_samples = [max(block_samples[0], self.samples[0]),
                        block_samples[1]]
        if self.samples[1] is not None:
            over_samples[1] = min(over_samples[1], self.samples[1])

        # Size of overlap
        if over_samples[1] - over_samples[0] <= 0:
            return False

        # Extraction slice
        src_slice = slice(over_samples[0] - self.consumed,
                          over_samples[1] - self.consumed)

        # Insertion slice
        dest_slice = slice(over_samples[0] - self.samples[0],
                           over_samples[1] - self.samples[0])

        # Specialization ...
        self._extract(data, src_slice, dest_slice)

        if over_samples[1] != block_samples[1]:
            return False

        self.consumed += data_count
        return True


class Accumulator1d(Accumulator):
    def _sample_count(self, _data):
        return len(_data)

    def _extract(self, data, src_slice, dest_slice):
        _data = np.asarray(data[src_slice])

        # On first frame, check if we know the final data shape.
        if self.data is None:
            if self.samples[1] is not None:
                self.shape = (self.samples[1] - self.samples[0], )

        if self.shape is not None:
            # Determinate.
            if self.data is None:
                self.data = np.empty(self.shape, dtype=_data.dtype)
            self.data[dest_slice] = _data
        else:
            # Indeterminate
            if self.data is None:
                self.data = []
            self.data.append(_data)

    def finalize(self):
        if self.shape is None:
            return np.hstack(self.data)
        else:
            return self.data


class AccumulatorNamed(Accumulator):
    """Accumulator for unpacking 2-d data (G3SuperTimestream) into
    individual (named) 1-d vectors.

    """
    def _sample_count(self, _data):
        return len(_data.times)

    def _extract(self, data, src_slice, dest_slice):
        # On first frame, check if we know the final data shape.
        if self.data is None:
            if self.samples[1] is not None:
                self.shape = (self.samples[1] - self.samples[0], )
            if hasattr(data, 'names'):
                # G3SuperTimestream ...
                self.keys = [k for k in data.names]
            else:
                # G3TimesampleMap ...
                self.keys = [k for k in data.keys()]

        if self.shape is not None:
            # Determinate.
            if self.data is None:
                self.data = {k: np.empty(self.shape[-1], dtype=data.data.dtype)
                             for k in self.keys}
            for i, k in enumerate(self.keys):
                self.data[k][dest_slice] = data.data[i][src_slice]
        else:
            # Indeterminate
            if self.data is None:
                self.data = {k: [] for k in self.keys}
            for i, k in enumerate(self.keys):
                self.data[k].append(data.data[i][src_slice])

    def finalize(self):
        if self.shape is None:
            self.data = {k: np.hstack(self.data[k]) for k in self.keys}
        return self.data


class AccumulatorTimesampleMap(AccumulatorNamed):
    """Accumulator for unpacking 2-d data (G3TimestampleMap) into
    individual (named) 1-d vectors.

    """
    def _extract(self, data, src_slice, dest_slice):
        # On first frame, check if we know the final data shape.
        if self.data is None:
            if self.samples[1] is not None:
                self.shape = (self.samples[1] - self.samples[0], )
            self.keys = [k for k in data.keys()]

        if self.shape is not None:
            # Determinate.
            if self.data is None:
                self.data = {k: np.empty(self.shape[-1], dtype=np.asarray(data[k]).dtype)
                             for k in self.keys}
            for i, k in enumerate(self.keys):
                self.data[k][dest_slice] = data[k][src_slice]
        else:
            # Indeterminate
            if self.data is None:
                self.data = {k: [] for k in self.keys}
            for i, k in enumerate(self.keys):
                self.data[k].append(data[k][src_slice])


class Accumulator2d(Accumulator):
    """Accumulator for unpacking 2-d data (G3SuperTimestream) into
    2-d array (preserving first axis labels).

    """
    def __init__(self, *args, insert_at=None, keys_to_keep=None,
                 calibrate=None, **kwargs):
        super().__init__(*args, **kwargs)
        # An optional destination buffer for the data.
        self.insert_at = insert_at
        self.keys_to_keep = keys_to_keep
        self.insert_at_idx = None
        self.extract_at_idx = None
        if self.insert_at is not None and self.samples[1] is None:
            self.samples[1] = self.insert_at.shape[-1] + self.samples[0]
        self.calibrate = calibrate

    def _sample_count(self, _data):
        return len(_data.times)

    def _extract(self, data, src_slice, dest_slice):

        if self.calibrate is not None:
            # This is a low cost operation if you do it before
            # decompression.  (Also do it before you use data.dtype,
            # in "first frame stuff".)
            data.calibrate(np.array([self.calibrate] * len(data.names)))

        # First frame stuff ...
        if self.data is None:
            if self.insert_at is not None:
                # We have a destination buffer
                self.keys, self.extract_at_idx, self.insert_at_idx = \
                    core.util.get_coindices(
                        data.names, self.keys_to_keep)
                self.data = self.insert_at  # place-holder
            else:
                if self.keys_to_keep is not None:
                    self.keys, _, self.extract_at_idx = \
                        core.util.get_coindices(self.keys_to_keep, data.names)
                else:
                    self.keys = list(data.names)
                # Set shape, if we know it.
                if self.samples[1] is not None:
                    self.shape = (len(self.keys), self.samples[1] - self.samples[0])
                    self.data = np.empty(self.shape, data.dtype)
                else:
                    self.data = []

        # G3SuperTimestream.extract() is available from so3g v0.1.13
        # (April 2024).  The previous handling (below this block) can
        # be removed in a few months.
        if hasattr(data, 'extract'):
            if self.insert_at is not None:
                data.extract(self.insert_at[:, dest_slice],
                             self.insert_at_idx,
                             self.extract_at_idx,
                             src_slice.start, src_slice.stop)
            elif self.shape is not None:
                data.extract(self.data[:, dest_slice], None, self.extract_at_idx,
                             src_slice.start, src_slice.stop)
            else:
                _sh = [len(data.names), len(data.times)]
                if self.extract_at_idx is not None:
                    _sh[0] = len(self.extract_at_idx)
                _dest = np.empty(_sh, dtype=data.dtype)
                data.extract(_dest, None, self.extract_at_idx,
                             src_slice.start, src_slice.stop)
                self.data.append(_dest)
            return

        # Store data from this frame.
        if self.insert_at is not None:
            # Indexed by name
            for i0, i1 in zip(self.insert_at_idx, self.extract_at_idx):
                self.insert_at[i0, dest_slice] = data.data[i1, src_slice]

        elif self.shape is not None:
            # Full array to hold data.
            if self.extract_at_idx is not None:
                for i, j in enumerate(self.extract_at_idx):
                    self.data[i, dest_slice] = data.data[j, src_slice]
            else:
                self.data[:, dest_slice] = data.data[:, src_slice]

        else:
            # List of arrays, to be hstacked later.
            if self.extract_at_idx is not None:
                self.data.append(data.data[self.extract_at_idx, src_slice])
            else:
                self.data.append(data.data[:, src_slice])

    def finalize(self):
        if self.insert_at is not None:
            pass
        elif self.shape is None:
            self.data = np.hstack(self.data)
        return self.data


def _frames_iterator(files, prefix, samples, smurf_proc=None):
    """Iterates over frames in files.  yields only frames that might be of
    interest for timestream unpacking.

    Yields each (frame, offset).  The offset is the global offset
    associated with the start of the frame.

    """
    offset = 0
    for f, i0, i1 in files:
        if i0 is None:
            i0 = offset

        if smurf_proc is not None:
            if (i1 is not None) and (i1 <= samples[0]):
                continue
            if samples[1] is not None and i0 >= samples[1]:
                break

        filename = os.path.join(prefix, f)
        offset = i0

        for frame in spt3g_core.G3File(filename):
            if smurf_proc is not None and smurf_proc.process(frame):
                # We found a dump frame, so stop looking.
                smurf_proc = None
            if frame.type is not spt3g_core.G3FrameType.Scan:
                continue
            yield frame, offset
            offset += len(frame['ancil'].times)
            # Alternately, use frame['sample_range']


def get_cal_obsids(ctx, obs_id, cal_type):
    """
    Returns set of obs-ids corresponding to the most recent calibration
    operations for a given obsid.

    Args
    ------
    ctx: core.Context
        Context object
    obs_id: str
        obs_id for which you want to get relevant calibration info
    cal_type: str
        Calibration subtype to use in the obsdb query. For example: 'iv' or
        'bias_steps'.

    Returns
    ----------
        obs_ids: dict
            Dict of obs_ids for each detset in specified operation
    """
    obs = ctx.obsdb.query(f"obs_id == '{obs_id}'")[0]
    detsets = ctx.obsfiledb.get_detsets(obs_id)
    min_ct = obs['start_time'] - 3600*24*7
    cal_all = ctx.obsdb.query(
        f"""
        start_time <= {obs['start_time']} and subtype=='{cal_type}'
        and start_time > {min_ct}
        """, sort=['start_time']
    )[::-1]

    obs_ids = {
        ds: None for ds in detsets
    }
    ids_to_find = len(obs_ids)
    ids_found = 0

    for o in cal_all:
        dsets = ctx.obsfiledb.get_files(o['obs_id']).keys()
        for ds in dsets:
            if ds in obs_ids:
                if obs_ids[ds] is None:
                    obs_ids[ds] = o['obs_id']
                    ids_found += 1
        if ids_to_find == ids_found:
            break

    return obs_ids


core.OBSLOADER_REGISTRY['obs-book'] = load_obs_book

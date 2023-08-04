import so3g
from spt3g import core as spt3g_core
import yaml
import os

import sotodlib
from sotodlib import core

import numpy as np


_TES_BIAS_COUNT = 12  # per detset / primary file group


def _extract_1d(src, src_offset, dest, dest_offset):
    """Assist with unloading frames into a destination array.  Copies as
    much data as possible from src[src_offset:] into
    dest[dest_offset:].  "As possible" accounts for the possibility
    that src_offset is beyond the end of src.  The return value is
    used to update src_offset (and, with a bit of additional care,
    dest_offset) for next frame.

    Args:
      src (sliceable): source vector.
      src_offset (int): starting index into source.
      dest (ndarray): destination buffer.
      dest_offset (int): offset into dest samples axis.

    Returns:
      Number of samples consumed from src; this includes any samples
      discarded in order to "seek" to src_offset.

    """
    count = min(len(src) - src_offset, len(dest) - dest_offset)
    if count < 0:
        return len(src)
    samp_slice = slice(src_offset, src_offset+count)
    dest[dest_offset:dest_offset+count] = src[samp_slice]
    return src_offset + count


def _extract_2d(src, src_offset, dest, dest_offset, dets=None):
    """Unpack certain elements of a G3SuperTimestream into a 2d array.
    Equivalent to running _extract_1d, row by row, on src and dest.

    Args:
      src (G3SuperTimestream): source frame object.
      src_offset (int): starting index into source.
      dest (ndarray): destination buffer.
      dest_offset (int): offset into dest samples axis.
      dets (list of str): detector names for each index of first axis
        of dest.  If None, simple one-to-one match-up is assumed.

    Returns:
      Number of samples consumed from src; this includes any samples
      discarded in order to "seek" to src_offset.

    """
    # What dets do we have here and where do they belong?
    count = min(len(src.times) - src_offset, dest.shape[-1] - dest_offset)
    if count < 0:
        return len(src.times)
    samp_slice = slice(src_offset, src_offset+count)
    if dets is None:
        # Straight copy should work.
        dest[:,dest_offset:dest_offset+count] = src.data[:,samp_slice]
    else:
        # Copy index-to-index.
        _, src_idx, dest_idx = core.util.get_coindices(src.names, dets)
        for i0, i1 in zip(src_idx, dest_idx):
            dest[i1,dest_offset:dest_offset+count] = src.data[i0,samp_slice]
    return src_offset + count


def _check_bias_names(frame):
    """Verify the frame has TES biases with expected names; return the
    modified names that include stream_id.

    """
    for i, name in enumerate(frame['tes_biases'].names):
        if name != 'bias%02i' % i:
            raise RuntimeError(f'Bias at index {i} has unexpected name "{name}"!')
    stream_id = frame['stream_id']
    return [f'{stream_id}_b{_i:02d}' for _i in range(i+1)]


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
    # the list of detsets implicated in this observation.
    c = db.conn.execute('select distinct DS.name, DS.det from detsets DS '
                        'join files on DS.name=files.detset '
                        'where obs_id=?', (obs_id,))
    pairs = [tuple(r) for r in c.fetchall()]

    # Now filter to only the dets requested.
    if dets is None:
        pairs_req = pairs
    else:
        pairs_req = [p for p in pairs if p[1] in dets]
        # Use sets for this...
        dets_req = [p[1] for p in pairs_req]
        unmatched = [d for d in dets if d not in dets_req]
        if len(unmatched):
            raise RuntimeError("User requested invalid dets (e.g. %s) "
                               "for obs_id=%s" % (unmatched[0], obs_id))
    detsets_req = set(p[0] for p in pairs_req)

    file_map = db.get_files(obs_id)
    one_group = list(file_map.values())[0]  # [('file0', 0, 1000), ('file1', 1000, 2000), ...]
    num_files = len(one_group)
    book_root = os.path.split(one_group[0][0])[0]
    
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

    file_map = db.get_files(obs_id)

    # Prepare the output structure.
    aman = core.AxisManager(
        core.LabelAxis('dets', [p[1] for p in pairs_req]),
        core.OffsetAxis('samps', samples[1] - samples[0], samples[0], obs_id))
    aman.wrap('primary', core.AxisManager(aman.samps))
    if no_signal:
        aman.wrap('signal', None)
    else:
        aman.wrap_new('signal', ('dets', 'samps'), dtype='float32')

    # det_info begins
    det_info = {p[1]: [p[0], p[1]] for p in pairs_req}

    primary_fields = None
    bias_dest = np.zeros((len(detsets_req), _TES_BIAS_COUNT,
                          samples[1] - samples[0]), dtype='int32')
    all_bias_names = []

    # Temporary storage of G3 timestamp vector
    g3_times = np.zeros(aman.samps.count, 'int64')

    # Read the signal data, by detset.
    detset_index = -1

    # When there are no dets being loaded, get "ancil" information from the ancil files.
    if len(detsets_req) == 0:
        def _rewrite_entry(row):
            f, etc = row[0], row[1:]
            p, b = os.path.split(f)
            tokens = b.split('_')
            a = os.path.join(p, 'A_ancil_' + tokens[-1])
            return tuple([a] + list(etc))
        _one_fileset = next(iter(file_map.values()))
        file_map = {'(ancil)': [_rewrite_entry(x) for x in _one_fileset]}

    # On the first pass through the loop we will load ancillary info.
    first_pass = True

    for detset, files in file_map.items():
        if detset not in detsets_req and detset != '(ancil)':
            continue
        detset_index += 1

        # Target AxisManagers for special primary fields, to be
        # populated once we can inspect the frames.
        primary_dest = None
        
        # dest_offset is the index, along samples axis, where next
        # data will be written.
        dest_offset = 0

        stream_id = None
        bias_names = []

        for frame, start in _frames_iterator(files, prefix, samples):

            # This should get set in at least one block below...
            delta = None

            # Anything in ancil should be identical across
            # filesets, so only process it once.
            if first_pass:
                delta = _extract_1d(frame['ancil'].times, start,
                                    g3_times, dest_offset)

            if 'primary' in frame:
                if stream_id is None:
                    stream_id = frame['stream_id']
                    for p in pairs_req:
                        if p[0] == detset:
                            det_info[p[1]].append(stream_id)

                # Extract "primary" fields, organize by detset.
                if primary_fields is None:
                    primary_fields = list(frame['primary'].names)
                else:
                    assert(primary_fields == list(frame['primary'].names))

                if primary_dest is None:
                    primary_dest = core.AxisManager(aman.samps)
                    for f in primary_fields:
                        primary_dest.wrap_new(f, ('samps', ), dtype='uint64')
                    aman['primary'].wrap(stream_id, primary_dest)

                primary_block = frame['primary'].data
                for i, f in enumerate(primary_fields):
                    delta = _extract_1d(primary_block[i], start,
                                        primary_dest[f], dest_offset)

                # Extract "bias", organize by detset (note input bias
                # arrays might have 20 instead of 12 entries...)
                if len(bias_dest):
                    bias_names = _check_bias_names(frame)[:_TES_BIAS_COUNT]
                    delta = _extract_2d(frame['tes_biases'], start,
                                        bias_dest[detset_index], dest_offset,
                                        bias_names)

                # Extract the main signal
                if not no_signal:
                    delta = _extract_2d(
                        frame['signal'],
                        start,
                        aman['signal'],
                        dest_offset,
                        aman.dets.vals,
                    )
                
            # How many samples were added to dest?
            if delta > start:
                dest_offset += (delta - start)
                start = 0
            else:
                start -= delta

            if dest_offset >= samples[1] - samples[0]:
                break

        # Wrap up for detset ...
        all_bias_names.extend(bias_names)

        first_pass = False

    # Convert timestamps
    aman.wrap('timestamps', g3_times / spt3g_core.G3Units.sec,
              [(0, 'samps')])

    # Merge the bias lines.
    aman.merge(core.AxisManager(core.LabelAxis('bias_lines', all_bias_names)))
    aman.wrap('biases', bias_dest.reshape((-1, samples[1] - samples[0])),
              [(0, 'bias_lines'), (1, 'samps')])

    det_info = core.metadata.ResultSet(
        ['detset', '_readout_id', 'stream_id'],
        src=list(det_info.values()))
    aman.wrap('det_info', det_info.to_axismanager(axis_key='_readout_id'))
    aman.wrap("flags", core.FlagManager.for_tod(aman, "dets", "samps"))

    return aman


def _frames_iterator(files, prefix, samples):
    """Iterates over all the frames in files.

    Yields each (frame, start), start is the offset into the frame
    from which data should be taken.  Note this offset could be beyond
    the end of the frame, indicating the frame should be ignored.

    """
    for f, i0, i1 in files:
        if i1 <= samples[0]:
            continue
        if i0 >= samples[1]:
            break

        # "start" is the offset from which we should start copying
        # data.  It is updated after each frame is processed.
        # Initially, it could be larger than the number of samples
        # in the frame we're looking at.  Eventually, it could be
        # zero.
        start = max(0, samples[0] - i0)

        filename = os.path.join(prefix, f)
        for frame in spt3g_core.G3File(filename):
            if frame.type is not spt3g_core.G3FrameType.Scan:
                continue
            yield frame, start

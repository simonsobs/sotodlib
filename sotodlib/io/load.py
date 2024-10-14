"""Functions and classes to support the loading of TOD data from files
on disk.  The routines provide the standard no-TOAST representation of
SO TOD data, and also should be leveraged to populate TOAST TOD
objects.

The basic routines here provide structures for automatically unpacking
series of G3 frames into coherent data structures.

Additional routines use an ObsFileDb (provided by the user) to
optimize disk reads for loading particular data of interest to a user.
These routines rely on the completeness and accuracy of the ObsFileDb,
and are appropriate for large archives of files that have been
pre-scanned to log their contents.

There will inevitably be a few different "file formats" out there, at
the very least some variations in the schema used on top of G3.
Presently the loader targets the pipe-s0001 "1day" TOD sim for the
LAT.

To support different loaders, we will host (here) a registry of named
loader functions.

"""

import so3g
from spt3g import core as g3core

import numpy as np

from collections import OrderedDict

from .. import core
from .load_toast_h5 import load_toast_h5_obs, load_toast_h5_dichroic_hack


# Interim, Oct. 2021.  Wait a year, remove this caution.
if hasattr(so3g, 'G3SuperTimestream'):
    from so3g import G3SuperTimestream
else:
    class MockG3SuperTimestream(object):
        pass
    G3SuperTimestream = MockG3SuperTimestream


class FieldGroup(list):
    """This is essentially a roadmap for decoding data from a
    G3FrameObject.  Each entry in this list is either a string, giving
    the key of a vector that should be loaded from the frame object,
    or a FieldGroup requesting a recursive decent into a "map" defined
    in the FrameObject.

    It is used to formulate decoding requests for a frame.  For
    example, a frame structured like this::

      Frame (Scan) [
        "boresight" (spt3g.core.G3TimestreamMap) => Timestreams from 3 detectors
        "signal" (spt3g.core.G3TimestreamMap) => Timestreams from 794 detectors
        "site_velocity" (spt3g.core.G3VectorDouble) => 3327 elements
      ]

    might have a roadmap defined like this::

      request = FieldGroup('root', [
        FieldGroup('boresight', ['az', 'el', 'roll']),
        FieldGroup('signal', ['det1', 'det2', 'det3']),
        'site_velocity'])

    Note that the name 'root', of the outermost FieldGroup, is not
    needed to decode a G3FrameObject... but you might use it for
    something else.

    When a string appears in the list, it is promoted to a "Field",
    internally.  If a field requires special processing, instantiate
    it directly to set options; e.g.::

      Field('site_velocity', optional=True)

    """
    def __init__(self, name, items=None, timestamp_field=None,
                 compression=False, refs_ok=True):
        """Arguments:
          name (str): The key in the parent G3FrameObject at which to
            find the specified items.
          items (list, optional): A list of items.
          timestamp_field (str): If not None, then the timestamps will
            be extracted from the indicated G3TimestreamMap and given
            the name specified here.
          compression (bool): If the requested item is a
            G3TimestreamMap with offset+gain compression implemented,
            then say so here.
          refs_ok (bool): If True, then extraction code will be
            instructed that it is ok to collect a bunch of 1-d
            references to the same large 2-d array, rather than
            copying out the 1-d subarrays of interest so big 2-d
            FrameObjects can be freed.  If False, copies will be
            forced (this is efficient in the limit that small number
            of available channels is being collected).

        """
        super().__init__()
        self.name = name
        self.compression = compression
        self.timestamp_field = timestamp_field
        self.refs_ok = refs_ok
        if items is not None:
            self.extend(items)

    def empty(self):
        """Returns an empty data structure (nested OrderederdDicts of lists)
        suitable for containing the data that this FieldGroup would
        decode.

        """
        output = OrderedDict()
        for item in self:
            if isinstance(item, FieldGroup):
                output[item.name] = item.empty()
                if item.timestamp_field:
                    output[item.timestamp_field] = []
            else:
                item = Field.as_field(item)
                output[item.name] = []
        return output

    @staticmethod
    def hstack_result(streams):
        """Recursively descends in streams (nested OrderedDicts) until it
        finds a list; assumes the list is a list of 1-d numpy arrays,
        and concatenates them together (in place).

        """
        keys = list(streams.keys())
        for k in keys:
            if isinstance(streams[k], list):
                streams[k] = np.hstack(streams[k])
            else:
                FieldGroup.hstack_result(streams[k])

    @staticmethod
    def merge_result(dest, src):
        """Recursively descends through src (nested OrderedDicts), and
        compares to dest.  In places where dest has the same keys, an
        update is performed on dest.  Any duplicate keys that are not
        OrderedDicts will be clobbered.

        For example, if the input is::

           dest = {'signal': {'a': [0,1,2,3],
                              'b': [1,2,3,4]},
                   'hk': {'temp1': [9,11,13,19]}}
           src =  {'signal': {'c': [2,3,4,5],
                              'd': [3,4,5,6]},
                   'hk': {'temp1': [9,11,13,19]}}

        Then dest will be updated to::

           dest = {'signal': {'a': [0,1,2,3],
                              'b': [1,2,3,4],
                              'c': [2,3,4,5],
                              'd': [3,4,5,6]},
                   'hk': {'temp1': [9,11,13,19]}}

        """
        keys = list(src.keys())
        for k in keys:
            if isinstance(src[k], dict):
                if not k in dest:
                    dest[k] = src[k]
                else:
                    FieldGroup.merge_result(dest[k], src[k])
            else:
                dest[k] = src[k]

class Field:
    def __init__(self, name, optional=False, wildcard=False, oversample=1):
        self.name = name
        self.wildcard = wildcard
        self.opts = {'optional': optional}
        self.oversample = oversample

    @staticmethod
    def as_field(item):
        if isinstance(item, Field):
            return item
        if isinstance(item, str):
            return Field(item)
        raise TypeError('Cannot promote %s to Field.' % item)


def unpack_frame_object(fo, field_request, streams, compression_info=None,
                        offset=0, max_count=None, refs_ok=True):
    """Unpack requested fields from a G3FrameObject, and update a data structure.
    
    Arguments:
      fo (G3FrameObject): The source object.
      field_request (FieldGroup): Description of what fields to unpack.
      streams (nested OrderedDicts): The data structure to unpack the data into.
      compression_info: The gain and offset dicts in the case of
        compression (or None to disable).

    """
    # Expand wildcards?
    to_remove = []
    to_add = []
    for item in field_request:
        if getattr(item, 'wildcard', False):
            assert(item.name == '*')  # That's the only wildcard we allow right now...
            to_remove.append(item)
            del streams[item.name]
            if isinstance(fo, G3SuperTimestream):
                for k in fo.names:
                    to_add.append(Field(k))
                    to_add[-1].opts = item.opts
                    streams[k] = []
            else:
                for k in fo.keys():
                    to_add.append(Field(k))
                    to_add[-1].opts = item.opts
                    streams[k] = []
    for item in to_remove:
        field_request.remove(item)
    field_request.extend(to_add)
    
    if isinstance(fo, G3SuperTimestream):
        key_map = {k: i for i, k in enumerate(fo.names)}
        
    def our_slice(n, oversamp):
        # returns (range size, slice)
        if n <= offset:
            return 0, slice(0, 0)
        n = n - offset
        if max_count is not None:
            n = min(n, max_count)
        return n, slice(offset * oversamp, (offset + n) * oversamp)

    _consumed = 0
    for item in field_request:
        if isinstance(item, FieldGroup):
            if item.compression:
                _gain = fo.get('compressor_gain_%s' % item.name, {})
                _offset = fo.get('compressor_offset_%s' % item.name, {})
                comp_info = (_gain, _offset)
            else:
                comp_info = None
            target = fo[item.name]
            _consumed = unpack_frame_object(target, item, streams[item.name], comp_info,
                                            offset=offset, max_count=max_count,
                                            refs_ok=item.refs_ok)
            _n, sl = our_slice(_consumed, 1)
            # Check and slice timestamp field independently -- must
            # work even if no dets requested led to _n=0 above.
            if item.timestamp_field is not None:
                if isinstance(target, G3SuperTimestream):
                    timesv = np.array(target.times) / g3core.G3Units.sec
                else:
                    t0, t1, ns = target.start, target.stop, target.n_samples
                    t0, t1 = t0.time / g3core.G3Units.sec, t1.time / g3core.G3Units.sec
                    timesv = np.linspace(t0, t1, ns)
                _consumed = len(timesv)
                _n, sl = our_slice(_consumed, 1)
                streams[item.timestamp_field].append(timesv[sl])
            continue
        # This is a simple field.
        item = Field.as_field(item)
        key = item.name
        if item.opts['optional']:
            if not key in fo.keys():
                if key in streams:
                    assert(len(streams[key]) == 0) # field went missing?
                    del streams[key]
                assert(key not in streams)
                continue
        if compression_info is not None:
            _gain, _offset = compression_info
            m, b = _gain.get(key, 1.), _offset.get(key, 0.)
            v = np.array(fo[key], dtype='float32') / m + b
        else:
            if isinstance(fo, G3SuperTimestream):
                v = fo.data[key_map[key]]
            else:
                v = np.array(fo[key])
        _consumed = len(v) // item.oversample
        _n, sl = our_slice(_consumed, item.oversample)

        if _n:
            if refs_ok:
                streams[key].append(v[sl])
            else:
                streams[key].append(np.copy(v[sl]))
    return _consumed

def unpack_frames(filename, field_request, streams, samples=None):
    """Read frames from the specified file and expand the data by stream.
    Only the requested fields, specified through *_fields arguments,
    are expanded.

    Arguments:
      filename (str): Full path to the file to load.
      field_request: Instructions for what fields to load.
      streams: Structure to which to append the
        streams from this file (perhaps obtained from running
        unpack_frames on a preceding file).
      samples (int, int): Start and end of sample range to unpack
        *from this file*.  First argument must be non-negative.  Second
        argument may be None, indicating to read forever.

    Returns:
      streams (structure containing lists of numpy arrays).

    """
    if streams is None:
        streams = field_request.empty()
    if samples is None:
        offset = 0
        to_read = None
    else:
        offset, to_read = samples
        if to_read is not None:
            to_read -= offset

    reader = so3g.G3IndexedReader(filename)
    while to_read is None or to_read > 0:
        frames = reader.Process(None)
        if len(frames) == 0:
            break
        frame = frames[0]
        if frame.type != g3core.G3FrameType.Scan:
            continue
        _consumed = unpack_frame_object(
            frame, field_request, streams, offset=offset, max_count=to_read)
        offset -= _consumed
        if offset < 0:
            if to_read is not None:
                to_read += offset
            offset = 0

    return streams


def load_file(filename, dets=None, signal_only=False):
    """Load data from file where there is no supporting obsfiledb.

    Args:
      filename (str or list): A filename or list of filenames (to be
        loaded in order).
      dets (list of str): The detector names of interest.  If None,
        loads all dets present in this file.  To load only the
        ancillary data, pass an empty list.
      signal_only (bool): If set, then only 'signal' is collected and
        other stuff is ignored.

    """
    if isinstance(filename, str):
        filenames = [filename]
    else:
        filenames = filename

    subreq = [
        FieldGroup('signal', [Field('*', wildcard=True)],
                   timestamp_field='timestamps', compression=True),
        ]
    subreq.extend([
        FieldGroup('boresight', ['az', 'el', 'roll']),
        Field('hwp_angle', optional=True),
        Field('corotator_angle', optional=True),
        Field('site_position', oversample=3),
        Field('site_velocity', oversample=3),
        Field('boresight_azel', oversample=4),
        Field('boresight_radec', oversample=4),
    ])
    request = FieldGroup('root', subreq)

    streams = None
    for filename in filenames:
        streams = unpack_frames(filename, request, streams)

    # Do we need to update that dets list?
    if dets is None:
        dets = list(streams['signal'].keys())

    # Create AxisManager now that we know the sample count.
    if len(dets) == 0:
        count = sum(map(len,streams['timestamps']))
    else:
        count = sum(map(len,streams['signal'][dets[0]]))
    aman = core.AxisManager(
        core.LabelAxis('dets', dets),
        core.OffsetAxis('samps', count, 0),
    )
    aman.wrap('signal', np.zeros(aman.shape, 'float32'),
              [(0, 'dets'), (1, 'samps')])

    if not signal_only:
        # The non-signal fields are duplicated across files so you
        # can just absorb them once.
        aman.wrap('timestamps', hstack_into(None, streams['timestamps']),
                      [(0, 'samps')])
        bman = core.AxisManager(aman.samps.copy())
        for k in ['az', 'el', 'roll']:
            bman.wrap(k, hstack_into(None, streams['boresight'][k]),
                      [(0, 'samps')])
        aman.wrap('boresight', bman)
        aman.wrap('qboresight_azel',
                  hstack_into(None, streams['boresight_azel']).reshape((-1, 4)),
                  [(0, 'samps')])
        aman.wrap('qboresight_radec',
                  hstack_into(None, streams['boresight_radec']).reshape((-1, 4)),
                  [(0, 'samps')])
        site = core.AxisManager(aman.samps.copy())
        for k in ['position', 'velocity']:
            site.wrap(k, hstack_into(None, streams['site_' + k]).reshape(-1, 3),
                      [(0, 'samps')])
        aman.wrap('site', site)

        if 'hwp_angle' in streams:
            aman.wrap('hwp_angle', hstack_into(None, streams['hwp_angle']),
                      [(0, 'samps')])

        if 'corotator_angle' in streams:
            aman.wrap('corotator_angle', hstack_into(None, streams['corotator_angle']),
                      [(0, 'samps')])

    # Copy in the signal, for each file.
    for det_name, arrs in streams['signal'].items():
        if dets is None or det_name in dets:
            i = list(aman.dets.vals).index(det_name)
            hstack_into(aman.signal[i], arrs)

    del streams
    return aman


def load_observation(db, obs_id, dets=None, samples=None, prefix=None,
                     no_signal=None,
                     **kwargs):
    """Obsloader function for TOAST simulate data -- this function matches
    output from pipe-s0001/s0002 (and SSO sims in 2019-2021).

    See API template, `sotodlib.core.context.obsloader_template`, for
    details.

    """
    if any([v is not None for v in kwargs.values()]):
        raise RuntimeError(
            f"This loader function does not understand kwargs: f{kwargs}")

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

    # Group by detset.
    dets_by_detset = OrderedDict([(p[0], []) for p in pairs_req])
    for p in pairs_req:
        dets_by_detset[p[0]].append(p[1])

    # If user requested no dets, include one detset to get ancil.
    if len(dets_by_detset) == 0:
        dets_by_detset[pairs[0][0]] = []

    # Loop through relevant files, in sample order, and accumulate
    # lists of stream segments.
    aman = None

    for detset, dets in dets_by_detset.items():
        detset_files = db.get_files(obs_id, detsets=[detset], prefix=prefix)[detset]
        subreq = []
        if no_signal:
            if aman is None:
                subreq.extend([
                    FieldGroup('signal', [], timestamp_field='timestamps',
                               compression=True),
                ])
        else:
            subreq.extend([
                FieldGroup('signal', dets, timestamp_field='timestamps',
                           compression=True),
            ])
        if aman is None:
            subreq.extend([
                FieldGroup('boresight', ['az', 'el', 'roll']),
                Field('hwp_angle', optional=True),
                Field('corotator_angle', optional=True),
                Field('site_position', oversample=3),
                Field('site_velocity', oversample=3),
                Field('boresight_azel', oversample=4),
                Field('boresight_radec', oversample=4),
            ])
        if len(subreq) == 0:
            continue

        request = FieldGroup('root', subreq)

        if samples:
            sample_start, sample_stop = samples
        else:
            sample_start, sample_stop = 0, None

        stop = sample_stop

        file_list = db.get_files(obs_id, [detset], prefix=prefix)[detset]
        streams = None
        for row in detset_files:
            filename, file_start, file_stop = row
            assert(file_start is not None)
            start = max(0, sample_start - file_start)
            if sample_stop is not None:
                stop = sample_stop - file_start
            streams = unpack_frames(
                filename, request, streams, samples=(start, stop))

        if aman is None:
            # Create AxisManager now that we know the sample count.
            count = sum(map(len,streams['timestamps']))
            aman = core.AxisManager(
                core.LabelAxis('dets', [p[1] for p in pairs_req]),
                core.OffsetAxis('samps', count, sample_start, obs_id),
            )
            if no_signal:
                aman.wrap('signal', None)
            else:
                aman.wrap('signal', np.zeros(aman.shape, 'float32'),
                          [(0, 'dets'), (1, 'samps')])

            # The non-signal fields are duplicated across files so you
            # can just absorb them once.
            aman.wrap('timestamps', hstack_into(None, streams['timestamps']),
                          [(0, 'samps')])
            bman = core.AxisManager(aman.samps.copy())
            for k in ['az', 'el', 'roll']:
                bman.wrap(k, hstack_into(None, streams['boresight'][k]),
                          [(0, 'samps')])
            aman.wrap('boresight', bman)
            aman.wrap('qboresight_azel',
                      hstack_into(None, streams['boresight_azel']).reshape((-1, 4)),
                      [(0, 'samps')])
            aman.wrap('qboresight_radec',
                      hstack_into(None, streams['boresight_radec']).reshape((-1, 4)),
                      [(0, 'samps')])
            site = core.AxisManager(aman.samps.copy())
            for k in ['position', 'velocity']:
                site.wrap(k, hstack_into(None, streams['site_' + k]).reshape(-1, 3),
                          [(0, 'samps')])
            aman.wrap('site', site)

            if 'hwp_angle' in streams:
                aman.wrap('hwp_angle', hstack_into(None, streams['hwp_angle']),
                          [(0, 'samps')])

            if 'corotator_angle' in streams:
                aman.wrap('corotator_angle', hstack_into(None, streams['corotator_angle']),
                          [(0, 'samps')])

        # Copy in the signal, for each file.
        for det_name, arrs in streams['signal'].items():
            i = list(aman.dets.vals).index(det_name)
            hstack_into(aman.signal[i], arrs)

        del streams
    
    aman.wrap('flags', core.FlagManager.for_tod(aman))
    return aman


def hstack_into(dest, src_arrays):
    if dest is None:
        return np.hstack(src_arrays)
    offset = 0
    for ar in src_arrays:
        n = len(ar)
        dest[offset:offset+n] = ar
        offset += n
    return dest


# Register the loaders defined here.
core.OBSLOADER_REGISTRY.update(
    {
        'pipe-s0001': load_observation,
        'toast3-hdf': load_toast_h5_obs,
        'toast3-hdf-dichroic-hack': load_toast_h5_dichroic_hack,
        'default': load_observation,
    }
)


# Deprecated alias (used to live here...)
OBSLOADER_REGISTRY = core.OBSLOADER_REGISTRY

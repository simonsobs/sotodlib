"""Functions and classes to support the loading of TOD data from files
on disk.  The routines provide the standard no-TOAST representation of
SO TOD data, and also should be leveraged to populate TOAST TOD
objects.

The basic routines here provide structures for automatically unpacking
series of G3 frames into coherent data structures.

Additional routines use an ObsFileDB (provided by the user) to
optimize disk reads for loading particular data of interest to a user.
These routines rely on the completeness and accuracy of the ObsFileDB,
and are appropriate for large archives of files that have been
pre-scanned to log their contents.

There will inevitably be a few different "file formats" out there, at
the very least some variations in the schema used on top of G3.
Presently the loader targets the pipe-s0001 "1day" TOD sim for the
LAT.

"""

import so3g
from spt3g import core

import numpy as np

from collections import OrderedDict

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
                 compression=False):
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

        """
        super().__init__()
        self.name = name
        self.compression = compression
        self.timestamp_field = timestamp_field
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
    def __init__(self, name, optional=False):
        self.name = name
        self.opts = {'optional': optional}

    @staticmethod
    def as_field(item):
        if isinstance(item, Field):
            return item
        if isinstance(item, str):
            return Field(item)
        raise TypeError('Cannot promote %s to Field.' % item)


def unpack_frame_object(fo, field_request, streams, compression_info=None):
    """Unpack requested fields from a G3FrameObject, and update a data structure.
    
    Arguments:
      fo (G3FrameObject): The source object.
      field_request (FieldGroup): Description of what fields to unpack.
      streams (nested OrderedDicts): The data structure to unpack the data into.
      compression_info: The gain and offset dicts in the case of
        compression (or None to disable).

    """
    for item in field_request:
        if isinstance(item, FieldGroup):
            if item.compression:
                gain = fo.get('compressor_gain_%s' % item.name, {})
                offset = fo.get('compressor_offset_%s' % item.name, {})
                comp_info = (gain, offset)
            else:
                comp_info = None
            target = fo[item.name]
            unpack_frame_object(target, item, streams[item.name], comp_info)
            if item.timestamp_field is not None:
                t0, t1, ns = target.start, target.stop, target.n_samples
                t0, t1 = t0.time / core.G3Units.sec, t1.time / core.G3Units.sec
                streams[item.timestamp_field].append(np.linspace(t0, t1, ns))
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
            gain, offset = compression_info
            m, b = gain.get(key, 1.), offset.get(key, 0.)
            v = np.array(fo[key], dtype='float32') / m + b
        else:
            v = np.array(fo[key])
        streams[key].append(v)

def unpack_frames(filename, field_request, streams):
    """Read frames from the specified file and expand the data by stream.
    Only the requested fields, specified through *_fields arguments,
    are expanded.

    Arguments:
      filename (str): Full path to the file to load.
      field_request: Instructions for what fields to load.
      streams: Structure to which to append the
        streams from this file (perhaps obtained from running
        unpack_frames on a preceding file).

    Returns:
      streams (structure containing lists of numpy arrays).

    """
    if streams is None:
        streams = field_request.empty()
    
    reader = so3g.G3IndexedReader(filename)
    while True:
        frames = reader.Process(None)
        if len(frames) == 0:
            break
        frame = frames[0]
        if frame.type == core.G3FrameType.Scan:
            unpack_frame_object(frame, field_request, streams)
    return streams

def load_observation(db, obs_id, dets=None, prefix=None):
    """Load the data for some observation.  You can restrict to only some
    detectors. Coming soon: also restrict by time range / sample
    index.


    This specifically targets the pipe-s0001 sim format.

    Arguments:

      db (ObsFileDB): The database describing this observation file
        set.
      obs_id (str): The identifier of the observation.
      dets (list of str): The detector names of interest.  If None,
        loads all dets present in this observation.
      prefix (str): The root address of the data files.  If not
        specified, the prefix is taken from the ObsFileDB.

    Returns:
      (signal_streams, other_streams).

    """
    if prefix is None:
        prefix = db.prefix
        if prefix is None:
            prefix = './'

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
        unmatched = [d for d in dets if not d in dets_req]
        if len(unmatched):
            raise RuntimeError("User requested invalid dets (e.g. %s) "
                               "for obs_id=%s" % (unmatched[0], obs_id))

    # Group by detset.
    dets_by_detset = OrderedDict([(p[0],[]) for p in pairs_req])
    for p in pairs_req:
        dets_by_detset[p[0]].append(p[1])

    # Loop through relevant files, in sample order, and accumulate
    # lists of stream segments.

    stream_groups = []
    for detset, dets in dets_by_detset.items():
        c = db.conn.execute('select name from files '
                            'where obs_id=? and detset=? ' +
                            'order by sample_start', (obs_id, detset))
        streams = None
        request = FieldGroup('root', [
            FieldGroup('signal', dets, timestamp_field='timestamps',
                       compression=True),
            FieldGroup('boresight', ['az', 'el', 'roll']),
            Field('hwp_angle', optional=True),
            'site_position',
            'site_velocity',
            'boresight_azel',
            'boresight_radec',
        ])
        for row in c:
            f = row[0]
            streams = unpack_frames(prefix+f, request, streams)
        stream_groups.append((request, streams))

    # Merge the groups.
    streams = OrderedDict()
    for request, s in stream_groups:
        request.merge_result(streams, s)
        
    FieldGroup.hstack_result(streams)

    # Re-pack into something plausibly standardizable.  In first sims,
    # everything is co-sampled by design, and thus "primary".  But we
    # return an object that includes a place for "secondary"
    # (non-cosampled) timestreams.

    streams_out = OrderedDict([
        ('primary', {
            'signal': streams['signal'],
            'timestamps': streams['timestamps'],
            'boresight': streams['boresight'],
            'qboresight_azel': streams['boresight_azel'].reshape((-1, 4)),
            'qboresight_radec': streams['boresight_radec'].reshape((-1, 4)),
            'site': {'position': streams['site_position'].reshape((-1, 3)),
                     'velocity': streams['site_velocity'].reshape((-1, 3))},
            }
         ),
        ('secondary', {})
        ])

    # Optional fields.

    if 'hwp_angle' in streams:
        streams_out['primary']['hwp_angle'] = streams['hwp_angle']

    return streams_out

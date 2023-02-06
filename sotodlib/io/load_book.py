import so3g
from spt3g import core as spt3g_core
import yaml
import os

import sotodlib
from sotodlib import core

#from . import load

import numpy as np

def _extract_1d(src, src_offset, dest, dest_offset):
    """Args:
      src (sliceable)
      src_offset (int): starting index into source.
      dest (ndarray)
      dest_offset (int): offset into dest samples axis.

    Returns:
      Number of samples consumed from src; this includes any samples
      discarded in order to "seek" to src_offset.

    """
    # What dets do we have here and where do they belong?
    count = min(len(src) - src_offset, dest.shape[-1] - dest_offset)
    if count < 0:
        return len(src)
    samp_slice = slice(src_offset, src_offset+count)
    dest[dest_offset:dest_offset+count] = src[samp_slice]
    return src_offset + count

def _extract_2d(src, src_offset, dest, dest_offset, dets):
    """Args:
      src (G3SuperTimestream)
      src_offset (int): starting index into source.
      dest (ndarray)
      dest_offset (int): offset into dest samples axis.
      dets (list of str): detector names for each index of first axis
        of dest.

    Returns:
      Number of samples consumed from src; this includes any samples
      discarded in order to "seek" to src_offset.

    """
    # What dets do we have here and where do they belong?
    _, src_idx, dest_idx = core.util.get_coindices(src.names, dets)
    count = min(len(src.times) - src_offset, dest.shape[-1] - dest_offset)
    if count < 0:
        return len(src.times)
    samp_slice = slice(src_offset, src_offset+count)
    for i0, i1 in zip(src_idx, dest_idx):
        dest[i1,dest_offset:dest_offset+count] = src.data[i0,samp_slice]
    return src_offset + count

def load_obs_book(db, obs_id, dets=None, prefix=None, samples=None,
                  no_signal=None,
                  **kwargs):
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
    aman.wrap('tes_bias', core.AxisManager(aman.samps))
    if no_signal:
        aman.wrap('signal', None)
    else:
        aman.wrap_new('signal', ('dets', 'samps'), dtype='float32')

    # det_info begins
    det_info = {p[1]: [p[0], p[1]] for p in pairs_req}

    primary_fields = None
    tes_fields = None

    # Read the signal data, by detset.
    for detset, files in file_map.items():
        if no_signal or detset not in detsets_req:
            continue

        # Target AxisManagers for special primary fields, to be
        # populated once we can inspect the frames.
        primary_dest = None
        tes_dest = None
        
        # dest_offset is the index, along samples axis, where next
        # data will be written.
        dest_offset = 0

        stream_id = None

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
                        primary_dest.wrap_new(f, ('samps', ), dtype='int64')
                    aman['primary'].wrap(detset, primary_dest)

                primary_block = frame['primary'].data
                for i, f in enumerate(primary_fields):
                    delta = _extract_1d(primary_block[i], start,
                                        primary_dest[f], dest_offset)

                # Extract "tes_bias", organize by detset.
                if tes_fields is None:
                    tes_fields = list(frame['tes_bias'].names)
                else:
                    assert(tes_fields == list(frame['tes_bias'].names))

                if tes_dest is None:
                    tes_dest = core.AxisManager(aman.samps)
                    for f in tes_fields:
                        tes_dest.wrap_new(f, ('samps', ), dtype='int32')
                    aman['tes_bias'].wrap(detset, tes_dest)

                tes_block = frame['tes_bias'].data
                for i, f in enumerate(tes_fields):
                    delta = _extract_1d(tes_block[i], start,
                                        tes_dest[f], dest_offset)

                # Extract the main signal
                if not no_signal:
                    delta = _extract_2d(frame['signal'], start,
                                        aman['signal'], dest_offset, aman.dets.vals)
                
                # How many samples were added to dest?
                if delta > start:
                    dest_offset += (delta - start)
                    start = 0
                else:
                    start -= delta

                if dest_offset >= samples[1] - samples[0]:
                    break

    det_info = core.metadata.ResultSet(
        ['detset', 'readout_id', 'stream_id'],
        src=list(det_info.values()))
    aman.wrap('det_info', det_info.to_axismanager(axis_key='readout_id'))
    return aman

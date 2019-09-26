# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST frame conversion utilities.
"""
import sys
import re

import itertools
import operator

import numpy as np

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g

from toast.mpi import MPI

import toast.qarray as qa
from toast.tod import spt3g_utils as s3utils
from toast.tod import TOD
from toast.utils import Logger, Environment, memreport
from toast.timing import Timer

# Mapping between TOAST cache names and so3g fields

FIELD_TO_CACHE = {
    #"timestamps" : TOD.TIMESTAMP_NAME,
    "flags_common" : TOD.COMMON_FLAG_NAME,
    "site_velocity" : TOD.VELOCITY_NAME,
    "site_position" : TOD.POSITION_NAME,
    "flags" : TOD.FLAG_NAME,
    "signal" : TOD.SIGNAL_NAME,
}


def recode_timestream(ts, params, rmstarget=2 ** 10, rmsmode="white"):
    """ts is a G3Timestream.  Returns a new
    G3Timestream for same samples as ts, but with data
    scaled and translated with gain and offset,
    rounded, and with FLAC compression enabled.

    Args:
        ts (G3Timestream) :  Input signal
        params (bool or dict) :  if True, compress with default
            parameters.  If dict with 'rmstarget' member, override
            default `rmstarget`.  If dict with `gain` and `offset`
            members, use those instead.
        params (None, bool or dict) :  If None, False or an empty dict,
             no compression or casting to integers.  If True or
             non-empty dictionary, enable compression.  Expected fields
             in the dictionary ('rmstarget', 'gain', 'offset', 'rmsmode')
             allow overriding defaults.
        rmstarget (float) :  Scale the iput signal to have this RMS.
            Should be much smaller then the 24-bit integer range:
            [-2 ** 23 : 2 ** 23] = [-8,388,608 : 8,388,608].
            The gain will be reduced if the scaled signal does
            not fit within the range of allowed values.
        rmsmode (string) : "white" or "full", determines how the
            signal RMS is measured.
    Returns:
        new_ts (G3Timestream) :  Scaled and translated timestream
            with the FLAC compression enabled
        gain (float) :  The applied gain
        offset (float) :  The removed offset

    """
    if not params:
        return ts, 1, 0
    gain = None
    offset = None
    if isinstance(params, dict):
        if "rmsmode" in params:
            rmsmode = params["rmsmode"]
        if "rmstarget" in params:
            rmstarget = params["rmstarget"]
        if "gain" in params:
            gain = params["gain"]
        if "offset" in params:
            offset = params["offset"]
    v = np.array(ts)
    vmin = np.amin(v)
    vmax = np.amax(v)
    if offset is None:
        offset = 0.5 * (vmin + vmax)
        amp = vmax - offset
    else:
        amp = np.max(np.abs(vmin - offset), np.abs(vmax - offset))
    if gain is None:
        if rmsmode == "white":
            rms = np.std(np.diff(v)) / np.sqrt(2)
        elif rmsmode == "full":
            rms = np.std(v)
        else:
            raise RuntimeError("Unrecognized RMS mode = '{}'".format(rmsmode))
        if rms == 0:
            gain = 1
        else:
            gain = rmstarget / rms
        # If the data have extreme outliers, we have to reduce the gain
        # to fit the 24-bit signed integer range
        while amp * gain >= 2 ** 23:
            gain *= 0.5
    elif amp * gain >= 2 ** 23:
        raise RuntimeError("The specified gain and offset saturate the band.")
    v = np.round((v - offset) * gain)
    new_ts = core3g.G3Timestream(v)
    new_ts.units = core3g.G3TimestreamUnits.Counts
    new_ts.SetFLACCompression(True)
    new_ts.start = ts.start
    new_ts.stop = ts.stop
    return new_ts, gain, offset


def frame_to_tod(
        tod,
        frame_offset,
        frame_size,
        frame_data=None,
        detector_map="signal",
        flag_map="flags",
        all_flavors=False,
):
    """Distribute a frame from the rank zero process.

    Args:
        tod (toast.TOD): instance of a TOD class.
        frame_offset (int): the first sample of the frame.
        frame_size (int): the number of samples in the the frame.
        frame_data (G3Frame): the input frame (only on rank zero).
        detector_map (str): the name of the frame timestream map.
        flag_map (str): then name of the frame flag map.
        all_flavors (bool):  Return all signal flavors that start
             with `detector_map`

    Returns:
        None

    """
    log = Logger.get()
    comm = tod.mpicomm
    if comm is None:
        rank = 0
    else:
        rank = comm.rank

    # First broadcast the frame data.
    if comm is not None:
        frame_data = comm.bcast(frame_data, root=0)

    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # Compute overlap of the frame with the local samples.
    cacheoff, froff, nfr = s3utils.local_frame_indices(
        local_first, nlocal, frame_offset, frame_size)

    # Helper function to expand time stamps from a core3g.G3Timestream object
    def copy_times(fieldval):
        ref = tod.local_times()
        if ref[cacheoff] != 0 and ref[cacheoff + nfr - 1] != 0:
            # the timestamps were already cached
            return
        tstart = fieldval.start.mjd
        tstop = fieldval.stop.mjd
        # Convert the MJD time stamps to UNIX time
        tstart = (tstart - 2440587.5 + 2400000.5) * 86400
        tstop = (tstop - 2440587.5 + 2400000.5) * 86400
        n = len(fieldval)
        ref[cacheoff : cacheoff + nfr] \
            = np.linspace(tstart, tstop, n)[froff : froff + nfr]
        del ref
        return

    def get_gain_and_offset(frame_data, tsmap, field):
        """ Check if the timestream in framedata[tsmap][field] was
        scaled and translated for compression.
        """
        try:
            gain = frame_data["compressor_gain_" + tsmap][field]
            offset = frame_data["compressor_offset_" + tsmap][field]
        except:
            gain, offset = None, None
        return gain, offset

    # Helper function to actually copy a slice of data into cache.
    def copy_slice(data, field, cache_prefix, gain=None, offset=None):
        if isinstance(data[field], core3g.G3Timestream):
            # every G3Timestream has time stamps, we'll only use the
            # first one
            copy_times(data[field])
        if cache_prefix is None:
            if field in FIELD_TO_CACHE:
                cache_field = FIELD_TO_CACHE[field]
            else:
                cache_field = field
        else:
            cache_field = "{}{}".format(cache_prefix, field)
        # Check field type and data shape
        ftype = s3utils.g3_dtype(data[field])
        flen = len(data[field])
        nnz = flen // frame_size
        if nnz * frame_size != flen:
            msg = "field {} has length {} which is not "\
                "divisible by size {}".format(field, flen, frame_size)
            raise RuntimeError(msg)
        if not tod.cache.exists(cache_field):
            # The field does not yet exist in cache, so create it.
            # print("proc {}:  create cache field {}, {}, ({},
            # {})".format(tod.mpicomm.rank, field, ftype, tod.local_samples[1],
            # nnz), flush=True)
            if nnz == 1:
                tod.cache.create(cache_field, ftype,
                                      (tod.local_samples[1],))
            else:
                tod.cache.create(cache_field, ftype,
                                      (tod.local_samples[1], nnz))
        # print("proc {}: get cache ref for {}".format(tod.mpicomm.rank,
        # cache_field), flush=True)
        ref = tod.cache.reference(cache_field)
        # Verify that the dimensions of the cache object are what we expect,
        # then copy the data.
        cache_samples = None
        cache_nnz = None
        if (len(ref.shape) > 1) and (ref.shape[1] > 0):
            # We have a packed 2D array
            cache_samples = ref.shape[0]
            cache_nnz = ref.shape[1]
        else:
            cache_nnz = 1
            cache_samples = len(ref)

        if cache_samples != tod.local_samples[1]:
            msg = "field {}: cache has {} samples, which is"
            " different from local TOD size {}"\
                .format(field, cache_samples, tod.local_samples[1])
            raise RuntimeError(msg)

        if cache_nnz != nnz:
            msg = "field {}: cache has nnz = {}, which is"\
                " different from frame nnz {}"\
                .format(field, cache_nnz, nnz)
            raise RuntimeError(msg)

        if cache_nnz > 1:
            slc = \
                np.array(data[field][nnz * froff : nnz * (froff + nfr)],
                         copy=False).reshape((-1, nnz))
            if gain is not None:
                slc /= gain
            if offset is not None:
                slc += offset
            ref[cacheoff : cacheoff + nfr, :] = slc
        else:
            slc = np.array(data[field][froff : froff + nfr], copy=False)
            if gain is not None:
                slc /= gain
            if offset is not None:
                slc += offset
            ref[cacheoff : cacheoff + nfr] = slc
        del ref
        return

    def copy_flags(chunks, field, cache_prefix):
        """ Translate flag intervals into sample ranges
        and record them in the TOD cache

        FIXME: uses sample indices instead of time stamps
        """
        ndata = np.zeros(froff + nfr, dtype=np.uint8)
        for beg, end in chunks.array():
            ndata[beg : end + 1] = 1
        if cache_prefix is None:
            if field in FIELD_TO_CACHE:
                cache_field = FIELD_TO_CACHE[field]
            else:
                cache_field = field
        else:
            cache_field = "{}{}".format(cache_prefix, field)
        # Check field type and data shape
        ftype = np.dtype(np.uint8)
        flen = len(ndata)
        nnz = flen // frame_size
        if nnz * frame_size != flen:
            msg = "field {} has length {} which is not "\
                "divisible by size {}".format(field, flen, frame_size)
            raise RuntimeError(msg)
        if not tod.cache.exists(cache_field):
            tod.cache.create(cache_field, ftype, (tod.local_samples[1],))
        ref = tod.cache.reference(cache_field)
        # Verify that the dimensions of the cache object are what we expect,
        # then copy the data.
        cache_samples = len(ref)

        if cache_samples != tod.local_samples[1]:
            msg = "field {}: cache has {} samples, which is"
            " different from local TOD size {}"\
                .format(field, cache_samples, tod.local_samples[1])
            raise RuntimeError(msg)

        slc = np.array(ndata[froff : froff + nfr], copy=False)
        ref[cacheoff : cacheoff + nfr] = slc
        del ref
        return

    if cacheoff is not None:
        # print("proc {} has overlap with frame {}:  {} {} \
        # {}".format(tod.mpicomm.rank, frame, cacheoff, froff, nfr),
        # flush=True)

        # This process has some overlap with the frame.
        # FIXME:  need to account for multiple timestream maps.
        for field, fieldval in frame_data.iteritems():
            # Skip over maps
            if isinstance(fieldval, core3g.G3TimestreamMap) or \
               isinstance(fieldval, core3g.G3MapVectorDouble) or \
               isinstance(fieldval, core3g.G3MapVectorInt) or \
               isinstance(fieldval, so3g.MapIntervalsInt) or \
               isinstance(fieldval, core3g.G3MapDouble):
                continue
            if isinstance(fieldval, so3g.IntervalsInt):
                copy_flags(fieldval, field, None)
            else:
                try:
                    copy_slice(frame_data, field, None)
                except TypeError:
                    # scalar metadata instead of vector
                    continue
        dpats = None
        if (detector_map is not None) or (flag_map is not None):
            # Build our list of regex matches
            dpats = [re.compile(".*{}.*".format(d)) for d in tod.local_dets]

        if detector_map is not None:
            # If the field name contains any of our local detectors,
            # then cache it.
            for field in frame_data[detector_map].keys():
                for dp in dpats:
                    if dp.match(field) is not None:
                        # print("proc {} copy frame {}, field
                        # {}".format(tod.mpicomm.rank, frame, field),
                        # flush=True)
                        gain, offset = get_gain_and_offset(
                            frame_data, detector_map, field)
                        copy_slice(frame_data[detector_map],
                                   field,
                                   TOD.SIGNAL_NAME + "_",
                                   gain=gain, offset=offset,
                        )
                        break
            # Look for additional signal flavors and cache them as well
            if all_flavors:
                prefix = detector_map + "_"
                for key in frame_data.keys():
                    if prefix in key:
                        flavor = key.replace(prefix, "")
                        for field in frame_data[key].keys():
                            for dp in dpats:
                                if dp.match(field) is not None:
                                    gain, offset = get_gain_and_offset(
                                        frame_data, key, field)
                                    copy_slice(frame_data[key],
                                               field,
                                               flavor + "_",
                                               gain=gain, offset=offset,
                                    )
                                    break

        if flag_map is not None:
            # If the field name contains any of our local detectors,
            # then cache it.
            for field in frame_data[flag_map].keys():
                for dp in dpats:
                    if dp.match(field) is not None:
                        chunks = frame_data[flag_map][field]
                        copy_flags(chunks, field, TOD.FLAG_NAME + "_")
                        break
    return


def tod_to_frames(
        tod,
        start_frame,
        n_frames,
        frame_offsets,
        frame_sizes,
        cache_signal=None,
        cache_flags=None,
        cache_common_flags=None,
        copy_common=None,
        copy_detector=None,
        mask_flag_common=255,
        mask_flag=255,
        units=None,
        dets=None,
        compress=False,
):
    """Gather all data from the distributed TOD cache for a set of frames.

    Args:
        tod (toast.TOD): instance of a TOD class.
        start_frame (int): the first frame index.
        n_frames (int): the number of frames.
        frame_offsets (array_like): list of the first samples of all frames.
        frame_sizes (list): list of the number of samples in each frame.
        cache_signal (str): if None, read signal from TOD.  Otherwise use this
            cache prefix for the detector signal timestreams.
        cache_flags (str): if None read det flags from TOD.  Otherwise use
            this cache prefix for the detector flag timestreams.
        cache_common_flags (str): if None, read common flags from TOD.
            Otherwise use this cache prefix.
        copy_common (tuple): (cache name, G3 type, frame name) of each extra
            common field to copy from cache.
        copy_detector (tuple): (cache name prefix, G3 type, G3 map type,
            frame name) of each distributed detector field (excluding the
            "signal") to copy from cache.
        mask_flag_common (int):  Bitmask to apply to common flags.
        mask_flag (int):  Bitmask to apply to per-detector flags.
        units: G3 units of the detector data.
        dets (list):  List of detectors to include in the frame.  If None,
            use all of the detectors in the TOD object.
        compress (bool or dict):  If True or a dictionary of compression parameters,
            store the timestreams as FLAC-compressed, 24-bit integers instead of
            uncompressed doubles.

    Returns:
        (list): List of frames on rank zero.  Other processes have a list of
            None values.

    """
    comm = tod.mpicomm
    rank = 0
    if comm is not None:
        rank = comm.rank
    comm_row = tod.grid_comm_row

    # Detector names
    if dets is None:
        detnames = tod.detectors
    else:
        detnames = []
        use_dets = set(dets)
        for det in tod.detectors:
            if det in use_dets:
                detnames.append(det)

    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # The process grid
    detranks, sampranks = tod.grid_size
    rankdet, ranksamp = tod.grid_ranks

    def get_local_cache(prow, field, cacheoff, ncache):
        """Read a local slice of a cache field.
        """
        mtype = None
        pdata = None
        nnz = 0
        if rankdet == prow:
            ref = tod.cache.reference(field)
            nnz = 1
            if (len(ref.shape) > 1) and (ref.shape[1] > 0):
                nnz = ref.shape[1]
            if ref.dtype == np.dtype(np.float64):
                mtype = MPI.DOUBLE
            elif ref.dtype == np.dtype(np.int64):
                mtype = MPI.INT64_T
            elif ref.dtype == np.dtype(np.int32):
                mtype = MPI.INT32_T
            elif ref.dtype == np.dtype(np.uint8):
                mtype = MPI.UINT8_T
            else:
                msg = "Cannot use cache field {} of type {}"\
                    .format(field, ref.dtype)
                raise RuntimeError(msg)
            if cacheoff is not None:
                pdata = ref.flatten()[nnz * cacheoff : nnz * (cacheoff + ncache)]
            else:
                pdata = np.zeros(0, dtype=ref.dtype)
        return (pdata, nnz, mtype)

    def gather_field(prow, pdata, nnz, mpitype, cacheoff, ncache, tag):
        """Gather a single timestream buffer to the root process.
        """
        is_none = pdata is None
        all_none = comm.allreduce(is_none, MPI.LAND)
        if all_none:
            # This situation arises at least when gathering HWP angle from LAT
            return None
        gdata = None
        # We are going to allreduce this later, so that every process
        # knows the dimensions of the field.
        gproc = 0
        allnnz = 0

        # Size of the local buffer
        pz = 0
        if pdata is not None:
            pz = len(pdata)

        if rankdet == prow:
            psizes = None
            if comm_row is None:
                psizes = [pz]
            else:
                psizes = comm_row.gather(pz, root=0)
            disp = None
            totsize = None
            if ranksamp == 0:
                # We are the process collecting the gathered data.
                allnnz = nnz
                gproc = rank
                # Compute the displacements into the receive buffer.
                disp = [0]
                for ps in psizes[:-1]:
                    last = disp[-1]
                    disp.append(last + ps)
                totsize = np.sum(psizes)
                # allocate receive buffer
                gdata = np.zeros(totsize, dtype=pdata.dtype)

            if comm_row is None:
                pdata[:] = gdata
            else:
                comm_row.Gatherv(pdata, [gdata, psizes, disp, mpitype], root=0)
            del disp
            del psizes

        # Now send this data to the root process of the whole communicator.
        # Only one process (the first one in process row "prow") has data
        # to send.

        if comm is not None:
            # All processes find out which one did the gather
            gproc = comm.allreduce(gproc, MPI.SUM)
            # All processes find out the field dimensions
            allnnz = comm.allreduce(allnnz, MPI.SUM)

        mtag = 10 * tag

        rdata = None
        if gproc == 0:
            if gdata is not None:
                if allnnz == 1:
                    rdata = gdata
                else:
                    rdata = gdata.reshape((-1, allnnz))
        else:
            # Data not yet on rank 0
            if rank == 0:
                # Receive data from the first process in this row
                rtype = comm.recv(source=gproc, tag=(mtag+1))
                rsize = comm.recv(source=gproc, tag=(mtag+2))
                rdata = np.zeros(rsize, dtype=np.dtype(rtype))
                comm.Recv(rdata, source=gproc, tag=mtag)
                # Reshape if needed
                if allnnz > 1:
                    rdata = rdata.reshape((-1, allnnz))
            elif (rank == gproc):
                # Send our data
                comm.send(gdata.dtype.char, dest=0, tag=(mtag+1))
                comm.send(len(gdata), dest=0, tag=(mtag+2))
                comm.Send(gdata, 0, tag=mtag)
        return rdata

    # For efficiency, we are going to gather the data for all frames at once.
    # Then we will split those up when doing the write.

    # Frame offsets relative to the memory buffers we are gathering
    fdataoff = [0]
    for f in frame_sizes[:-1]:
        last = fdataoff[-1]
        fdataoff.append(last + f)

    # The list of frames- only on the root process.
    fdata = None
    if rank == 0:
        fdata = [core3g.G3Frame(core3g.G3FrameType.Scan)
                 for f in range(n_frames)]
    else:
        fdata = [None for f in range(n_frames)]

    def split_field(data, g3t, framefield, mapfield=None, g3units=units,
                    times=None):
        """Split a gathered data buffer into frames- only on root process.
        """
        if data is None:
            return
        if g3t == core3g.G3VectorTime:
            # Special case for time values stored as int64_t, but
            # wrapped in a class.
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                g3times = list()
                for t in range(ndata):
                    g3times.append(core3g.G3Time(data[dataoff + t]))
                if mapfield is None:
                    fdata[f][framefield] = core3g.G3VectorTime(g3times)
                else:
                    fdata[f][framefield][mapfield] = \
                        core3g.G3VectorTime(g3times)
                del g3times
        elif g3t == so3g.IntervalsInt:
            # Flag vector is written as a simple boolean.
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                # Extract flag vector (0 or 1) for this frame
                frame_flags = (data[dataoff : dataoff + ndata] != 0).astype(int)
                # Convert bit 0 to an IntervalsInt.
                ival = so3g.IntervalsInt.from_mask(frame_flags, 1)[0]
                if mapfield is None:
                    fdata[f][framefield] = ival
                else:
                    fdata[f][framefield][mapfield] = ival
        elif g3t == core3g.G3Timestream:
            if times is None:
                raise RuntimeError(
                    "You must provide the time stamp vector with a "
                    "Timestream object")
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                timeslice = times[cacheoff + dataoff : cacheoff + dataoff + ndata]
                tstart = timeslice[0] * 1e8
                tstop = timeslice[-1] * 1e8
                if mapfield is None:
                    if g3units is None:
                        fdata[f][framefield] = \
                            g3t(data[dataoff : dataoff + ndata])
                    else:
                        fdata[f][framefield] = \
                            g3t(data[dataoff : dataoff + ndata], g3units)
                    fdata[f][framefield].start = core3g.G3Time(tstart)
                    fdata[f][framefield].stop = core3g.G3Time(tstop)
                else:
                    # Individual detector data.  The only fields that
                    # we (optionally) compress.
                    if g3units is None:
                        tstream = g3t(data[dataoff : dataoff + ndata])
                    else:
                        tstream = g3t(data[dataoff : dataoff + ndata], g3units)
                    if compress and "compressor_gain_" + framefield in fdata[f]:
                        (tstream, gain, offset) = recode_timestream(tstream, compress)
                        fdata[f]["compressor_gain_" + framefield][mapfield] = gain
                        fdata[f]["compressor_offset_" + framefield][mapfield] = offset
                    fdata[f][framefield][mapfield] = tstream
                    fdata[f][framefield][mapfield].start = core3g.G3Time(tstart)
                    fdata[f][framefield][mapfield].stop = core3g.G3Time(tstop)
        else:
            # The bindings of G3Vector seem to only work with
            # lists.  This is probably horribly inefficient.
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                if len(data.shape) == 1:
                    fdata[f][framefield] = \
                        g3t(data[dataoff : dataoff + ndata].tolist())
                else:
                    # We have a 2D quantity
                    fdata[f][framefield] = \
                        g3t(data[dataoff : dataoff + ndata, :].flatten()
                            .tolist())
        return

    # Compute the overlap of all frames with the local process.  We want to
    # to find the full sample range that this process overlaps the total set
    # of frames.

    cacheoff = None
    ncache = 0

    for f in range(n_frames):
        # Compute overlap of the frame with the local samples.
        fcacheoff, froff, nfr = s3utils.local_frame_indices(
            local_first, nlocal, frame_offsets[f], frame_sizes[f])
        if fcacheoff is not None:
            if cacheoff is None:
                cacheoff = fcacheoff
                ncache = nfr
            else:
                ncache += nfr

    # Now gather the full sample data one field at a time.  The root process
    # splits up the results into frames.

    # First collect boresight data.  In addition to quaternions for the Az/El
    # pointing, we convert this back into angles that follow the specs
    # for telescope pointing.

    times = None
    if rankdet == 0:
        times = tod.local_times()
    if comm is not None:
        times = gather_field(0, times, 1, MPI.DOUBLE, cacheoff, ncache, 0)

    bore = None
    if rankdet == 0:
        bore = tod.read_boresight(local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        bore = gather_field(0, bore, 4, MPI.DOUBLE, cacheoff, ncache, 0)
    if rank == 0:
        split_field(bore.reshape(-1, 4), core3g.G3VectorDouble,
                    "boresight_radec")

    bore = None
    if rankdet == 0:
        bore = tod.read_boresight_azel(
            local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        bore = gather_field(0, bore, 4, MPI.DOUBLE, cacheoff, ncache, 1)
    if rank == 0:
        split_field(bore.reshape(-1, 4), core3g.G3VectorDouble,
                    "boresight_azel")

    if rank == 0:
        for f in range(n_frames):
            fdata[f]["boresight"] = core3g.G3TimestreamMap()
        ang_theta, ang_phi, ang_psi = qa.to_angles(bore)
        ang_az = ang_phi
        ang_el = (np.pi / 2.0) - ang_theta
        ang_roll = ang_psi
        split_field(ang_az, core3g.G3Timestream, "boresight", "az", None,
                    times=times)
        split_field(ang_el, core3g.G3Timestream, "boresight", "el", None,
                    times=times)
        split_field(ang_roll, core3g.G3Timestream, "boresight", "roll", None,
                    times=times)

    hwp_angle = None
    if rankdet == 0:
        hwp_angle = tod.local_hwp_angle()
    if comm is not None:
        hwp_angle = gather_field(0, hwp_angle, 1, MPI.DOUBLE, cacheoff, ncache, 0)
    if rank == 0:
        split_field(hwp_angle, core3g.G3VectorDouble, "hwp_angle", times=times)

    # Now the position and velocity information

    pos = None
    if rankdet == 0:
        pos = tod.read_position(local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        pos = gather_field(0, pos, 3, MPI.DOUBLE, cacheoff, ncache, 2)
    if rank == 0:
        split_field(pos.reshape(-1, 3), core3g.G3VectorDouble, "site_position")

    vel = None
    if rankdet == 0:
        vel = tod.read_velocity(local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        vel = gather_field(0, vel, 3, MPI.DOUBLE, cacheoff, ncache, 3)
    if rank == 0:
        split_field(vel.reshape(-1, 3), core3g.G3VectorDouble, "site_velocity")

    # Now handle the common flags- either from a cache object or from the
    # TOD methods

    cflags = None
    nnz = 1
    if cache_common_flags is None:
        if rankdet == 0:
            cflags = tod.read_common_flags(local_start=cacheoff, n=ncache)
            cflags &= mask_flag_common
    else:
        cflags, nnz, mtype = get_local_cache(0, cache_common_flags, cacheoff,
                                             ncache)
        if cflags is not None:
            cflags &= mask_flag_common
    if comm is not None:
        mtype = MPI.UINT8_T
        cflags = gather_field(0, cflags, nnz, mtype, cacheoff, ncache, 4)
    if rank == 0:
        split_field(cflags, so3g.IntervalsInt, "flags_common")

    # Any extra common fields

    if comm is not None:
        comm.barrier()

    if copy_common is not None:
        for cindx, (cname, g3typ, fname) in enumerate(copy_common):
            cdata, nnz, mtype = get_local_cache(0, cname, cacheoff, ncache)
            cdata = gather_field(0, cdata, nnz, mtype, cacheoff, ncache, cindx)
            if rank == 0:
                split_field(cdata, g3typ, fname)

    # Now read all per-detector quantities.

    # For each detector field, processes which have the detector
    # in their local_dets should be in the same process row.

    if rank == 0:
        for f in range(n_frames):
            fdata[f]["signal"] = core3g.G3TimestreamMap()
            if compress:
                fdata[f]["compressor_gain_signal"] = core3g.G3MapDouble()
                fdata[f]["compressor_offset_signal"] = core3g.G3MapDouble()
            fdata[f]["flags"] = so3g.MapIntervalsInt()
            if copy_detector is not None:
                for cname, g3typ, g3maptyp, fnm in copy_detector:
                    fdata[f][fnm] = g3maptyp()
                    if compress:
                        fdata[f]["compressor_gain_" + fnm] = core3g.G3MapDouble()
                        fdata[f]["compressor_offset_" + fnm] = core3g.G3MapDouble()

    for dindx, dname in enumerate(detnames):
        drow = -1
        if dname in tod.local_dets:
            drow = rankdet
        # As a sanity check, verify that every process which
        # has this detector is in the same process row.
        rowcheck = None
        if comm is None:
            rowcheck = [drow]
        else:
            rowcheck = comm.gather(drow, root=0)
        prow = 0
        if rank == 0:
            rc = np.array([x for x in rowcheck if (x >= 0)],
                          dtype=np.int32)
            prow = np.max(rc)
            if np.min(rc) != prow:
                msg = "Processes with detector {} are not in the "\
                    "same row of the process grid\n".format(dname)
                sys.stderr.write(msg)
                if comm is not None:
                    comm.abort()

        # Every process finds out which process row is participating.
        if comm is not None:
            prow = comm.bcast(prow, root=0)

        # "signal"

        detdata = None
        nnz = 1
        if cache_signal is None:
            if rankdet == prow:
                detdata = tod.local_signal(dname)[cacheoff : cacheoff + ncache]
        else:
            cache_det = "{}_{}".format(cache_signal, dname)
            detdata, nnz, mtype = get_local_cache(prow, cache_det, cacheoff,
                                                  ncache)
        if comm is not None:
            mtype = MPI.DOUBLE
            detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                   ncache, dindx)
        if rank == 0:
            split_field(detdata, core3g.G3Timestream, "signal",
                        mapfield=dname, times=times)

        # "flags"

        detdata = None
        nnz = 1
        if cache_flags is None:
            if rankdet == prow:
                detdata = tod.local_flags(dname)[cacheoff : cacheoff + ncache]
                detdata &= mask_flag
        else:
            cache_det = "{}_{}".format(cache_flags, dname)
            detdata, nnz, mtype = get_local_cache(prow, cache_det, cacheoff,
                                                  ncache)
            if detdata is not None:
                detdata &= mask_flag
        if comm is not None:
            mtype = MPI.UINT8_T
            detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                   ncache, dindx)
        if rank == 0:
            split_field(detdata, so3g.IntervalsInt, "flags", mapfield=dname)

        # Now copy any additional fields.

        if copy_detector is not None:
            for cname, g3typ, g3maptyp, fnm in copy_detector:
                cache_det = "{}_{}".format(cname, dname)
                detdata, nnz, mtype = get_local_cache(prow, cache_det,
                                                      cacheoff, ncache)
                detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                       ncache, dindx)
                if rank == 0:
                    split_field(detdata, g3typ, fnm, mapfield=dname,
                                times=times)

    return fdata

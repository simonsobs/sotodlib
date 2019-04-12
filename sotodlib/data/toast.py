# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST interface tools.

This module contains code for interfacing with TOAST data representations.

"""
import os
import sys
import re

import itertools
import operator

import numpy as np

import toast
from toast.mpi import MPI
from toast.tod.interval import intervals_to_chunklist
import toast.qarray as qa

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g
from toast.tod import spt3g_utils as s3utils


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
        units=None):
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

    Returns:
        (list): List of frames on rank zero.  Other processes have a list of
            None values.

    """
    # Detector names
    detnames = tod.detectors

    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # The process grid
    detranks, sampranks = tod.grid_size
    rankdet, ranksamp = tod.grid_ranks

    def get_local_cache(prow, fld, cacheoff, ncache):
        """Read a local slice of a cache field.
        """
        mtype = None
        pdata = None
        if rankdet == prow:
            ref = tod.cache.reference(fld)
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
                    .format(fld, ref.dtype)
                raise RuntimeError(msg)
            if cacheoff is not None:
                pdata = ref.flatten()[nnz*cacheoff:nnz*(cacheoff+ncache)]
            else:
                pdata = np.zeros(0, dtype=ref.dtype)
        return (pdata, nnz, mtype)

    def gather_field(prow, pdata, nnz, mpitype, cacheoff, ncache, tag):
        """Gather a single timestream buffer to the root process.
        """
        gdata = None
        # We are going to allreduce this later, so that every process
        # knows the dimensions of the field.
        gproc = 0
        allnnz = 0

        # Size of the local buffer
        pz = len(pdata)

        if rankdet == prow:
            psizes = tod.grid_comm_row.gather(pz, root=0)
            disp = None
            totsize = None
            if ranksamp == 0:
                # We are the process collecting the gathered data.
                allnnz = nnz
                gproc = tod.mpicomm.rank
                # Compute the displacements into the receive buffer.
                disp = [0]
                for ps in psizes[:-1]:
                    last = disp[-1]
                    disp.append(last + ps)
                totsize = np.sum(psizes)
                # allocate receive buffer
                gdata = np.zeros(totsize, dtype=pdata.dtype)

            tod.grid_comm_row.Gatherv(pdata, [gdata, psizes, disp, mpitype],
                                      root=0)
            del disp
            del psizes

        # Now send this data to the root process of the whole communicator.
        # Only one process (the first one in process row "prow") has data
        # to send.

        # All processes find out which one did the gather
        gproc = tod.mpicomm.allreduce(gproc, MPI.SUM)
        # All processes find out the field dimensions
        allnnz = tod.mpicomm.allreduce(allnnz, MPI.SUM)

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
            if tod.mpicomm.rank == 0:
                # Receive data from the first process in this row
                rtype = tod.mpicomm.recv(source=gproc, tag=(mtag+1))
                rsize = tod.mpicomm.recv(source=gproc, tag=(mtag+2))
                rdata = np.zeros(rsize, dtype=np.dtype(rtype))
                tod.mpicomm.Recv(rdata, source=gproc, tag=mtag)
                # Reshape if needed
                if allnnz > 1:
                    rdata = rdata.reshape((-1, allnnz))
            elif (tod.mpicomm.rank == gproc):
                # Send our data
                tod.mpicomm.send(gdata.dtype.char, dest=0, tag=(mtag+1))
                tod.mpicomm.send(len(gdata), dest=0, tag=(mtag+2))
                tod.mpicomm.Send(gdata, 0, tag=mtag)
        return rdata

    # For efficiency, we are going to gather the data for all frames at once.
    # Then we will split those up when doing the write.

    # Frame offsets relative to the memory buffers we are gathering
    fdataoff = [0]
    for f in frame_sizes[:-1]:
        last = fdataoff[-1]
        fdataoff.append(last+f)

    # The list of frames- only on the root process.
    fdata = None
    if tod.mpicomm.rank == 0:
        fdata = [core3g.G3Frame(core3g.G3FrameType.Scan)
                 for f in range(n_frames)]
    else:
        fdata = [None for f in range(n_frames)]

    def flags_to_intervals(flgs):
        """Convert a flag vector to an interval list.
        """
        groups = [
            [i for i, value in it] for key, it in
            itertools.groupby(enumerate(flgs), key=operator.itemgetter(1))
            if key != 0]
        chunks = list()
        for grp in groups:
            chunks.append([grp[0], grp[-1]])
        return chunks

    def split_field(data, g3t, framefield, mapfield=None):
        """Split a gathered data buffer into frames.
        """
        if tod.mpicomm.rank == 0:
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
                # This means that the data is actually flags
                # and we should convert it into a list of intervals.
                fint = flags_to_intervals(data)
                for f in range(n_frames):
                    dataoff = fdataoff[f]
                    ndata = frame_sizes[f]
                    datalast = dataoff + ndata
                    chunks = list()
                    idomain = (0, ndata-1)
                    for intr in fint:
                        # Interval sample ranges are defined relative to the
                        # frame itself.
                        cfirst = None
                        clast = None
                        if (intr[0] < datalast) and (intr[1] >= dataoff):
                            # there is some overlap...
                            if intr[0] < dataoff:
                                cfirst = 0
                            else:
                                cfirst = intr[0] - dataoff
                            if intr[1] >= datalast:
                                clast = ndata - 1
                            else:
                                clast = intr[1] - dataoff
                            chunks.append([cfirst, clast])
                    if mapfield is None:
                        if len(chunks) == 0:
                            fdata[f][framefield] = \
                                so3g.IntervalsInt()
                        else:
                            fdata[f][framefield] = \
                                so3g.IntervalsInt.from_array(
                                    np.array(chunks, dtype=np.int64))
                        fdata[f][framefield].domain = idomain
                    else:
                        if len(chunks) == 0:
                            fdata[f][framefield][mapfield] = \
                                so3g.IntervalsInt()
                        else:
                            fdata[f][framefield][mapfield] = \
                                so3g.IntervalsInt.from_array(
                                    np.array(chunks, dtype=np.int64))
                            fdata[f][framefield][mapfield].domain = idomain
                del fint
            elif g3t == core3g.G3Timestream:
                for f in range(n_frames):
                    dataoff = fdataoff[f]
                    ndata = frame_sizes[f]
                    if mapfield is None:
                        if units is None:
                            fdata[f][framefield] = \
                                g3t(data[dataoff:dataoff+ndata])
                        else:
                            fdata[f][framefield] = \
                                g3t(data[dataoff:dataoff+ndata], units)
                    else:
                        if units is None:
                            fdata[f][framefield][mapfield] = \
                                g3t(data[dataoff:dataoff+ndata])
                        else:
                            fdata[f][framefield][mapfield] = \
                                g3t(data[dataoff:dataoff+ndata], units)
            else:
                # The bindings of G3Vector seem to only work with
                # lists.  This is probably horribly inefficient.
                for f in range(n_frames):
                    dataoff = fdataoff[f]
                    ndata = frame_sizes[f]
                    if len(data.shape) == 1:
                        fdata[f][framefield] = \
                            g3t(data[dataoff:dataoff+ndata].tolist())
                    else:
                        # We have a 2D quantity
                        fdata[f][framefield] = \
                            g3t(data[dataoff:dataoff+ndata, :].flatten()
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

    bore = None
    if rankdet == 0:
        bore = tod.read_boresight(local_start=cacheoff, n=ncache)
    bore = gather_field(0, bore.flatten(), 4, MPI.DOUBLE, cacheoff, ncache, 0)
    split_field(bore.reshape(-1, 4), core3g.G3VectorDouble, "qboresight_radec")

    bore = None
    if rankdet == 0:
        bore = tod.read_boresight_azel(local_start=cacheoff, n=ncache)
    bore = gather_field(0, bore.flatten(), 4, MPI.DOUBLE, cacheoff, ncache, 1)
    split_field(bore.reshape(-1, 4), core3g.G3VectorDouble, "qboresight_azel")

    if tod.mpicomm.rank == 0:
        for f in range(n_frames):
            fdata[f]["boresight"] = core3g.G3TimestreamMap()

    ang_az, ang_el, ang_roll = qa.to_angles(bore)
    split_field(ang_az, core3g.G3Timestream, "boresight", "az")
    split_field(ang_el, core3g.G3Timestream, "boresight", "el")
    split_field(ang_roll, core3g.G3Timestream, "boresight", "roll")

    # Now the position and velocity information

    pos = None
    if rankdet == 0:
        pos = tod.read_position(local_start=cacheoff, n=ncache)
    pos = gather_field(0, pos.flatten(), 3, MPI.DOUBLE, cacheoff, ncache, 2)
    split_field(pos.reshape(-1, 3), core3g.G3VectorDouble, "site_position")

    vel = None
    if rankdet == 0:
        vel = tod.read_velocity(local_start=cacheoff, n=ncache)
    vel = gather_field(0, vel.flatten(), 3, MPI.DOUBLE, cacheoff, ncache, 3)
    split_field(vel.reshape(-1, 3), core3g.G3VectorDouble, "site_velocity")

    # Now handle the common flags- either from a cache object or from the
    # TOD methods

    cflags = None
    nnz = 1
    mtype = MPI.UINT8_T
    if cache_common_flags is None:
        if rankdet == 0:
            cflags = tod.read_common_flags(local_start=cacheoff, n=ncache)
            cflags &= mask_flag_common
    else:
        cflags, nnz, mtype = get_local_cache(0, cache_common_flags, cacheoff,
                                             ncache)
        cflags &= mask_flag_common
    cflags = gather_field(0, cflags, nnz, mtype, cacheoff, ncache, 4)
    split_field(cflags, so3g.IntervalsInt, "flags_common")

    # Any extra common fields

    tod.mpicomm.barrier()

    if copy_common is not None:
        for cindx, (cname, g3typ, fname) in enumerate(copy_common):
            cdata, nnz, mtype = get_local_cache(0, cname, cacheoff, ncache)
            cdata = gather_field(0, cdata, nnz, mtype, cacheoff, ncache, cindx)
            split_field(cdata, g3typ, fname)

    # Now read all per-detector quantities.

    # For each detector field, processes which have the detector
    # in their local_dets should be in the same process row.

    if tod.mpicomm.rank == 0:
        for f in range(n_frames):
            fdata[f]["signal"] = core3g.G3TimestreamMap()
            fdata[f]["flags"] = so3g.MapIntervalsInt()
            if copy_detector is not None:
                for cname, g3typ, g3maptyp, fnm in copy_detector:
                    fdata[f][fnm] = g3maptyp()

    for dindx, dname in enumerate(detnames):
        drow = -1
        if dname in tod.local_dets:
            drow = rankdet
        # As a sanity check, verify that every process which
        # has this detector is in the same process row.
        rowcheck = tod.mpicomm.gather(drow, root=0)
        prow = 0
        if tod.mpicomm.rank == 0:
            rc = np.array([x for x in rowcheck if (x >= 0)],
                          dtype=np.int32)
            prow = np.max(rc)
            if np.min(rc) != prow:
                msg = "Processes with detector {} are not in the "\
                    "same row of the process grid\n".format(dname)
                sys.stderr.write(msg)
                tod.mpicomm.abort()

        # Every process finds out which process row is participating.
        prow = tod.mpicomm.bcast(prow, root=0)

        # "signal"

        detdata = None
        nnz = 1
        mtype = MPI.DOUBLE
        if cache_signal is None:
            if rankdet == prow:
                detdata = tod.read(detector=dname, local_start=cacheoff,
                                   n=ncache)
        else:
            cache_det = "{}_{}".format(cache_signal, dname)
            detdata, nnz, mtype = get_local_cache(prow, cache_det, cacheoff,
                                                  ncache)
        detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                               ncache, dindx)
        split_field(detdata, core3g.G3Timestream, "signal", mapfield=dname)

        # "flags"

        detdata = None
        nnz = 1
        mtype = MPI.UINT8_T
        if cache_flags is None:
            if rankdet == prow:
                detdata = tod.read_flags(detector=dname, local_start=cacheoff,
                                         n=ncache)
                detdata &= mask_flag
        else:
            cache_det = "{}_{}".format(cache_flags, dname)
            detdata, nnz, mtype = get_local_cache(prow, cache_det, cacheoff,
                                                  ncache)
            detdata &= mask_flag
        detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                               ncache, dindx)
        split_field(detdata, so3g.IntervalsInt, "flags", mapfield=dname)

        # Now copy any additional fields.

        if copy_detector is not None:
            for cname, g3typ, g3maptyp, fnm in copy_detector:
                cache_det = "{}_{}".format(cname, dname)
                detdata, nnz, mtype = get_local_cache(prow, cache_det,
                                                      cacheoff, ncache)
                detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                       ncache, dindx)
                split_field(detdata, g3typ, fnm, mapfield=dname)

    return fdata


class ToastExport(toast.Operator):
    """Operator which writes data to a directory tree of frame files.

    The top level directory will contain one subdirectory per observation.
    Each observation directory will contain frame files of the approximately
    the specified size.  A single frame file will contain multiple frames.
    The size of each frame is determined by either the TOD distribution
    chunks or the separate time intervals for the observation.

    Args:
        outdir (str): the top-level output directory.
        prefix (str): the file name prefix for each frame file.
        use_todchunks (bool): if True, use the chunks of the original TOD for
            data distribution.
        use_intervals (bool): if True, use the intervals in the observation
            dictionary for data distribution.
        cache_name (str):  The name of the cache object (<name>_<detector>) in
            the existing TOD to use for the detector timestream.  If None, use
            the read* methods from the existing TOD.
        cache_common (str):  The name of the cache object in the existing TOD
            to use for common flags.  If None, use the read* methods from the
            existing TOD.
        cache_flag_name (str):   The name of the cache object
            (<name>_<detector>) in the existing TOD to use for the flag
            timestream.  If None, use the read* methods from the existing TOD.
        mask_flag_common (int):  Bitmask to apply to common flags.
        mask_flag (int):  Bitmask to apply to per-detector flags.
        filesize (int):  The approximate file size of each frame file in
            bytes.
        units (G3TimestreamUnits):  The units of the detector data.

    """
    def __init__(self, outdir, prefix="so", use_todchunks=False,
                 use_intervals=False, cache_name=None, cache_common=None,
                 cache_flag_name=None, mask_flag_common=255, mask_flag=255,
                 filesize=500000000, units=None):
        self._outdir = outdir
        self._prefix = prefix
        self._cache_common = cache_common
        self._cache_name = cache_name
        self._cache_flag_name = cache_flag_name
        self._mask_flag = mask_flag
        self._mask_flag_common = mask_flag_common
        if use_todchunks and use_intervals:
            raise RuntimeError("cannot use both TOD chunks and Intervals")
        self._usechunks = use_todchunks
        self._useintervals = use_intervals
        self._target_framefile = filesize
        self._units = units
        # We call the parent class constructor
        super().__init__()

    def _write_obs(self, writer, props, detindx):
        """Write an observation frame.

        Given a dictionary of scalars, write these to an observation frame.

        Args:
            writer (G3Writer): The writer instance.
            props (dict): Dictionary of properties.
            detindx (dict): Dictionary of UIDs for each detector.

        Returns:
            None

        """
        f = core3g.G3Frame(core3g.G3FrameType.Observation)
        for k, v in props.items():
            f[k] = s3utils.to_g3_type(v)
        indx = core3g.G3MapInt()
        for k, v in detindx.items():
            indx[k] = int(v)
        f["detector_uid"] = indx
        writer(f)
        return

    def _write_precal(self, writer, dets, noise):
        """Write the calibration frame at the start of an observation.

        This frame nominally contains "preliminary" values for the detectors.
        For simulations, this contains the true detector offsets and noise
        properties.


        """
        qname = "detector_offset"
        f = core3g.G3Frame(core3g.G3FrameType.Calibration)
        # Add a vector map for quaternions
        f[qname] = core3g.G3MapVectorDouble()
        for k, v in dets.items():
            f[qname][k] = core3g.G3VectorDouble(v)
        if noise is not None:
            kfreq = "noise_stream_freq"
            kpsd = "noise_stream_psd"
            kindx = "noise_stream_index"
            dstr = "noise_detector_streams"
            dwt = "noise_detector_weights"
            f[kfreq] = core3g.G3MapVectorDouble()
            f[kpsd] = core3g.G3MapVectorDouble()
            f[kindx] = core3g.G3MapInt()
            f[dstr] = core3g.G3MapVectorInt()
            f[dwt] = core3g.G3MapVectorDouble()
            nse_dets = list(noise.detectors)
            nse_keys = list(noise.keys)
            st = dict()
            wts = dict()
            for d in nse_dets:
                st[d] = list()
                wts[d] = list()
            for k in nse_keys:
                f[kfreq][k] = core3g.G3VectorDouble(noise.freq(k).tolist())
                f[kpsd][k] = core3g.G3VectorDouble(noise.psd(k).tolist())
                f[kindx][k] = int(noise.index(k))
                for d in nse_dets:
                    wt = noise.weight(d, k)
                    if wt > 0:
                        st[d].append(noise.index(k))
                        wts[d].append(wt)
            for d in nse_dets:
                f[dstr][d] = core3g.G3VectorInt(st[d])
                f[dwt][d] = core3g.G3VectorDouble(wts[d])
        writer(f)
        return

    def _bytes_per_sample(self, ndet, nflavor):
        # For each sample we have:
        #   - 1 x 8 bytes for timestamp
        #   - 4 x 8 bytes for boresight RA/DEC quats
        #   - 4 x 8 bytes for boresight Az/El quats
        #   - 2 x 8 bytes for boresight Az/El angles
        #   - 3 x 8 bytes for telescope position
        #   - 3 x 8 bytes for telescope velocity
        #   - 1 x 8 bytes x number of dets x number of flavors
        persample = 8 + 1 + 32 + 48 + 24 + 24 + 8 * ndet * nflavor
        return persample

    def exec(self, data):
        """Export data to a directory tree of so3g frames.

        For errors that prevent the export, this function will directly call
        MPI Abort() rather than raise exceptions.  This could be changed in
        the future if additional logic is implemented to ensure that all
        processes raise an exception when one process encounters an error.

        Args:
            data (toast.Data): The distributed data.

        """
        # the two-level toast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        # One process checks the path
        if cworld.rank == 0:
            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir)
        cworld.barrier()

        for obs in data.obs:
            # Observation information.  Anything here that is a simple data
            # type will get written to the observation frame.
            props = dict()
            for k, v in obs.items():
                if isinstance(v, (int, str, bool, float)):
                    props[k] = v

            # Every observation must have a name...
            obsname = obs["name"]

            # The TOD
            tod = obs["tod"]
            nsamp = tod.total_samples
            detquat = tod.detoffset()
            detindx = tod.detindx
            ndets = len(detquat)
            detnames = tod.detectors

            # Get any other metadata from the TOD
            props.update(tod.meta())

            # First process in the group makes the output directory
            obsdir = os.path.join(self._outdir, obsname)
            if cgroup.rank == 0:
                if not os.path.isdir(obsdir):
                    os.makedirs(obsdir)
            cgroup.barrier()

            detranks, sampranks = tod.grid_size

            # Determine frame sizes based on the data distribution
            framesizes = None
            if self._usechunks:
                framesizes = tod.total_chunks
            elif self._useintervals:
                if "intervals" not in obs:
                    raise RuntimeError(
                        "Observation does not contain intervals, cannot \
                        distribute using them")
                framesizes = intervals_to_chunklist(obs["intervals"], nsamp)
            if framesizes is None:
                framesizes = [nsamp]

            # Examine all the cache objects and find the set of prefixes
            flavors = set()
            flavor_type = dict()
            flavor_maptype = dict()
            pat = re.compile(r"^(.*?)_(.*)")
            for nm in list(tod.cache.keys()):
                mat = pat.match(nm)
                if mat is not None:
                    pref = mat.group(1)
                    md = mat.group(2)
                    if md in detnames:
                        # This cache field has the form <prefix>_<det>
                        if pref not in flavor_type:
                            ref = tod.cache.reference(nm)
                            if ref.dtype == np.dtype(np.float64):
                                flavors.add(pref)
                                flavor_type[pref] = core3g.G3Timestream
                                flavor_maptype[pref] = core3g.G3TimestreamMap
                            elif ref.dtype == np.dtype(np.int32):
                                flavors.add(pref)
                                flavor_type[pref] = core3g.G3VectorInt
                                flavor_maptype[pref] = core3g.G3MapVectorInt
                            elif ref.dtype == np.dtype(np.uint8):
                                flavors.add(pref)
                                flavor_type[pref] = so3g.IntervalsInt
                                flavor_maptype[pref] = so3g.MapIntervalsInt
                            else:
                                msg = "Cache prefix {} has unsupported \
                                    data type.  Skipping export"
                                raise RuntimeError(msg)
            flavors.discard(self._cache_name)
            flavors.discard(self._cache_flag_name)
            copy_flavors = [
                (x, flavor_type[x], flavor_maptype[x], "signal_{}".format(x))
                for x in flavors]

            print("found cache flavors ", flavors, flush=True)

            # Given the dimensions of this observation, compute the frame
            # file sizes and all relevant offsets.

            frame_sample_offs = None
            file_sample_offs = None
            file_frame_offs = None
            if cgroup.rank == 0:
                # Compute the frame file breaks.  We ignore the observation
                # and calibration frames since they are small.
                sampbytes = self._bytes_per_sample(len(detquat), len(flavors))

                file_sample_offs, file_frame_offs, frame_sample_offs = \
                    s3utils.compute_file_frames(
                        sampbytes, framesizes,
                        file_size=self._target_framefile)

            file_sample_offs = cgroup.bcast(file_sample_offs, root=0)
            file_frame_offs = cgroup.bcast(file_frame_offs, root=0)
            frame_sample_offs = cgroup.bcast(frame_sample_offs, root=0)

            ex_files = [os.path.join(obsdir,
                        "{}_{:08d}.g3".format(self._prefix, x))
                        for x in file_sample_offs]

            # Loop over each frame file.  Write the header frames and then
            # gather the data from all processes before writing the scan
            # frames.

            for ifile, (ffile, foff) in enumerate(zip(ex_files,
                                                  file_frame_offs)):
                nframes = None
                print("  ifile = {}, ffile = {}, foff = {}".format(ifile, ffile, foff), flush=True)
                if ifile == len(ex_files) - 1:
                    # we are at the last file
                    nframes = len(framesizes) - foff
                else:
                    # get number of frames in this file
                    nframes = file_frame_offs[ifile+1] - foff

                writer = None
                if cgroup.rank == 0:
                    writer = core3g.G3Writer(ffile)
                    self._write_obs(writer, props, detindx)
                    if "noise" in obs:
                        self._write_precal(writer, detquat, obs["noise"])
                    else:
                        self._write_precal(writer, detquat, None)

                # Collect data for all frames in the file in one go.

                frm_offsets = [frame_sample_offs[foff+f]
                               for f in range(nframes)]
                frm_sizes = [framesizes[foff+f] for f in range(nframes)]

                if cgroup.rank == 0:
                    print("  {} file {}".format(obsdir, ifile), flush=True)
                    print("    start frame = {}, nframes = {}".format(foff, nframes), flush=True)
                    print("    frame offs = ", frm_offsets, flush=True)
                    print("    frame sizes = ", frm_sizes, flush=True)

                fdata = tod_to_frames(
                    tod, foff, nframes, frm_offsets, frm_sizes,
                    cache_signal=self._cache_name,
                    cache_flags=self._cache_flag_name,
                    cache_common_flags=self._cache_common,
                    copy_common=None,
                    copy_detector=copy_flavors,
                    units=self._units)

                if cgroup.rank == 0:
                    for fdt in fdata:
                        writer(fdt)
                    del writer
                del fdata

        return

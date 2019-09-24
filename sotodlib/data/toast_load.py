# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST interface tools.

This module contains code for interfacing with TOAST data representations.

"""
import os
import sys
import re

import traceback

import numpy as np

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g

from toast.dist import Data, distribute_discrete

import toast.qarray as qa
from toast.tod import TOD, Noise
from toast.tod import spt3g_utils as s3utils
from toast.utils import Logger, Environment, memreport
from toast.timing import Timer
from toast.mpi import MPI

from ..hardware import Hardware

from .toast_frame_utils import frame_to_tod


# FIXME:  This CamelCase name is ridiculous in all caps...

class SOTOD(TOD):
    """This class contains the timestream data.

    An instance of this class loads a directory of frame files into memory.
    Filtering by detector properties is done at construction time.

    Args:
        path (str):  The path to this observation directory.
        file_names:
        file_nframes:
        file_sample_offs:
        frame_sizes:
        frame_sizes_by_offset:
        frame_sample_offs:
        detquats:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which this
            observation data is distributed.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        all_flavors (bool):  Return all signal flavors

    """
    def __init__(self, path, file_names, file_nframes, file_sample_offs,
                 frame_sizes, frame_sizes_by_offset,
                 frame_sample_offs, detquats,
                 mpicomm, detranks=1, all_flavors=False):
        self._path = path
        self._units = None
        self._detquats = detquats
        self._frame_sizes = frame_sizes
        self._frame_sample_offs = frame_sample_offs
        self._file_names = file_names
        self._file_nframes = file_nframes
        self._file_sample_offs = file_sample_offs
        self._all_flavors = all_flavors

        sampsizes = []
        nsamp = 0
        for sz in frame_sizes_by_offset.values():
            sampsizes.append(sz)
            nsamp += sz

        # We need to assign a unique integer index to each detector.  This
        # is used when seeding the streamed RNG in order to simulate
        # timestreams.  For simplicity, and assuming that detector names
        # are not too long, we can convert the detector name to bytes and
        # then to an integer.
        # FIXME: these numbers will not agree with the ones synthesized
        # FIXME:     in toast_so_sim.py.  Instead, they should be made
        # FIXME:     part of the hardware model.

        self._detindx = {}
        for det in detquats.keys():
            bdet = det.encode("utf-8")
            uid = None
            try:
                ind = int.from_bytes(bdet, byteorder="little")
                uid = int(ind & 0xFFFFFFFF)
            except:
                raise RuntimeError(
                    "Cannot convert detector name {} to a unique integer-\
                    maybe it is too long?".format(det))
            self._detindx[det] = uid

        # call base class constructor to distribute data
        super().__init__(
            mpicomm, list(sorted(detquats.keys())), nsamp,
            detindx=self._detindx, detranks=detranks,
            sampsizes=sampsizes, meta=dict())

        # Now that the data distribution is set, read frames into memory.
        self.load_frames()

        self.get_azel()
        return

    def get_azel(self):
        """ Translate the boresight az/el quaternions into azimuth and
        elevation
        """
        quats = self.cache.reference("boresight_azel")
        theta, phi = qa.to_position(quats)
        self._az = (-phi % (2 * np.pi))
        self._el = np.pi / 2 - theta
        if len(self._az) > 0:
            azmin = np.amin(self._az)
            azmax = np.amax(self._az)
            elmin = np.amin(self._el)
            elmax = np.amax(self._el)
        else:
            azmin = 1000
            azmax = -1000
            elmin = 1000
            elmax = -1000
        if self.mpicomm is not None:
            azmin = self.mpicomm.allreduce(azmin, MPI.MIN)
            azmax = self.mpicomm.allreduce(azmax, MPI.MAX)
            elmin = self.mpicomm.allreduce(elmin, MPI.MIN)
            elmax = self.mpicomm.allreduce(elmax, MPI.MAX)
        self.scan_range = (azmin, azmax, elmin, elmax)
        return

    def load_frames(self):
        log = Logger.get()
        rank = 0
        if self.mpicomm is not None:
            rank = self.mpicomm.rank

        # Timestamps
        self.cache.create(self.TIMESTAMP_NAME, np.float64,
                          (self.local_samples[1],))

        # Boresight pointing
        self.cache.create("boresight_radec", np.float64,
                          (self.local_samples[1], 4))
        self.cache.create("boresight_azel", np.float64,
                          (self.local_samples[1], 4))
        self.cache.create(self.HWP_ANGLE_NAME, np.float64,
                          (self.local_samples[1],))

        # Common flags
        self.cache.create(self.COMMON_FLAG_NAME, np.uint8,
                          (self.local_samples[1],))

        # Telescope position and velocity
        self.cache.create(self.POSITION_NAME, np.float64,
                          (self.local_samples[1], 3))
        self.cache.create(self.VELOCITY_NAME, np.float64,
                          (self.local_samples[1], 3))

        # Detector data and flags
        for det in self.local_dets:
            name = "{}_{}".format(self.SIGNAL_NAME, det)
            self.cache.create(name, np.float64, (self.local_samples[1],))
            name = "{}_{}".format(self.FLAG_NAME, det)
            self.cache.create(name, np.uint8, (self.local_samples[1],))

        timer = Timer()
        for ffile in self._file_names:
            fnf = self._file_nframes[ffile]
            frame_offsets = self._frame_sample_offs[ffile]
            frame_sizes = self._frame_sizes[ffile]
            if rank == 0:
                log.debug("Loading {} frames from {}".format(fnf, ffile))
            # Loop over all frames- only the root process will actually
            # read data from disk.
            if rank == 0:
                gfile = core3g.G3File(ffile)
            else:
                gfile = [None] * fnf

            timer.clear()
            timer.start()
            for fdata, frame_offset, frame_size in zip(
                    gfile, frame_offsets, frame_sizes):
                is_scan = True
                if rank == 0:
                    if fdata.type != core3g.G3FrameType.Scan:
                        is_scan = False
                if self.mpicomm is not None:
                    is_scan = self.mpicomm.bcast(is_scan, root=0)
                if not is_scan:
                    continue

                frame_to_tod(
                    self,
                    frame_offset,
                    frame_size,
                    frame_data=fdata,
                    all_flavors=self._all_flavors,
                )
                if self.mpicomm is not None:
                    self.mpicomm.barrier()
            timer.stop()
            if rank == 0:
                log.debug("Translated frames in {}s".format(timer.seconds()))
            del gfile
        return

    def detoffset(self):
        return dict(self._detquats)

    def _get_boresight(self, start, n):
        ref = self.cache.reference("boresight_radec")[start:start+n, :]
        return ref

    def _put_boresight(self, start, data):
        ref = self.cache.reference("boresight_radec")
        ref[start:(start+data.shape[0]), :] = data
        del ref
        return

    def _get_boresight_azel(self, start, n):
        ref = self.cache.reference("boresight_azel")[start:start+n, :]
        return ref

    def _put_boresight_azel(self, start, data):
        ref = self.cache.reference("boresight_azel")
        ref[start:(start+data.shape[0]), :] = data
        del ref
        return

    def _get(self, detector, start, n):
        name = "{}_{}".format("signal", detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref

    def _put(self, detector, start, data):
        name = "{}_{}".format("signal", detector)
        ref = self.cache.reference(name)
        ref[start:(start+data.shape[0])] = data
        del ref
        return

    def _get_flags(self, detector, start, n):
        name = "{}_{}".format("flags", detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref

    def _put_flags(self, detector, start, flags):
        name = "{}_{}".format("flags", detector)
        ref = self.cache.reference(name)
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return

    def _get_common_flags(self, start, n):
        ref = self.cache.reference("flags_common")[start:start+n]
        return ref

    def _put_common_flags(self, start, flags):
        ref = self.cache.reference("flags_common")
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return

    def _get_hwp_angle(self, start, n):
        if self.cache.exists(self.HWP_ANGLE_NAME):
            hwpang = self.cache.reference(self.HWP_ANGLE_NAME)[start:start+n]
        else:
            hwpang = None
        return hwpang

    def _put_hwp_angle(self, start, hwpang):
        ref = self.cache.reference(self.HWP_ANGLE_NAME)
        ref[start:(start + hwpang.shape[0])] = hwpang
        del ref
        return

    def _get_times(self, start, n):
        ref = self.cache.reference("timestamps")[start:start+n]
        tm = 1.0e-9 * ref.astype(np.float64)
        del ref
        return tm

    def _put_times(self, start, stamps):
        ref = self.cache.reference("timestamps")
        ref[start:(start+stamps.shape[0])] = np.array(1.0e9 * stamps,
                                                      dtype=np.int64)
        del ref
        return

    def _get_pntg(self, detector, start, n):
        # Get boresight pointing (from disk or cache)
        bore = self._get_boresight(start, n)
        # Apply detector quaternion and return
        return qa.mult(bore, self._detquats[detector])

    def _put_pntg(self, detector, start, data):
        raise RuntimeError("SOTOD computes detector pointing on the fly."
                           " Use the write_boresight() method instead.")
        return

    def _get_position(self, start, n):
        ref = self.cache.reference("site_position")[start:start+n, :]
        return ref

    def _put_position(self, start, pos):
        ref = self.cache.reference("site_position")
        ref[start:(start+pos.shape[0]), :] = pos
        del ref
        return

    def _get_velocity(self, start, n):
        ref = self.cache.reference("site_velocity")[start:start+n, :]
        return ref

    def _put_velocity(self, start, vel):
        ref = self.cache.reference("site_velocity")
        ref[start:(start+vel.shape[0]), :] = vel
        del ref
        return

    def read_boresight_az(self, local_start=0, n=0):
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read boresight azimuth - process "
                "has no assigned local samples"
            )
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid".format(
                    local_start, local_start + n - 1
                )
            )
        return self._az[local_start : local_start + n]


def parse_cal_frames(calframes, dets):
    detlist = None
    if isinstance(dets, Hardware):
        detlist = list(sorted(dets.data["detectors"].keys()))
    elif isinstance(dets, list):
        detlist = dets
    qname = "detector_offset"
    detoffset = dict()
    kfreq = "noise_stream_freq"
    kpsd = "noise_stream_psd"
    kindx = "noise_stream_index"
    dstr = "noise_detector_streams"
    dwt = "noise_detector_weights"
    noise_freq = dict()
    noise_index = dict()
    noise_psds = dict()
    mixing = dict()
    detstrms = dict()
    detwghts = dict()
    detnames = []
    for calframe in calframes:
        if detlist is None:
            for d, q in calframe[qname].iteritems():
                detoffset[d] = np.array(q)
        else:
            for d, q in calframe[qname].iteritems():
                if d in detlist:
                    detoffset[d] = np.array(q)
        detnames += list(detoffset.keys())
        for k, v in calframe[kfreq].iteritems():
            noise_freq[k] = np.array(v)
        for k, v in calframe[kpsd].iteritems():
            noise_psds[k] = np.array(v)
        for k, v in calframe[kindx].iteritems():
            noise_index[k] = int(v)
        for k, v in calframe[dstr].iteritems():
            detstrms[k] = np.array(v)
        for k, v in calframe[dwt].iteritems():
            detwghts[k] = np.array(v)
    detnames = sorted(detnames)
    for det in detnames:
        mixing[det] = dict()
        for st, wt in zip(detstrms[det], detwghts[det]):
            mixing[det][st] = wt
    #print(detnames, noise_freq, noise_psds, noise_index, mixing, flush=True)
    # FIXME:  The original data dump should have specified the mixing matrix
    # explicitly.
    nse = Noise(detectors=detnames, freqs=noise_freq, psds=noise_psds)
    # nse = Noise(detectors=detnames, freqs=noise_freq, psds=noise_psds,
    #             mixmatrix=mixing, indices=noise_index)
    return detoffset, nse


def load_observation(path, dets=None, mpicomm=None, prefix=None, **kwargs):
    """Loads an observation into memory.

    Given an observation directory, load frame files into memory.  Observation
    and Calibration frames are stored in corresponding toast objects and Scan
    frames are loaded and distributed.  Further selection of a subset of
    detectors is done based on an explicit list or a Hardware object.

    Additional keyword arguments are passed to the SOTOD constructor.

    Args:
        path (str):  The path to the observation directory.
        dets (list):  Either a list of detectors, a Hardware object, or None.
        mpicomm (mpi4py.MPI.Comm):  The communicator.
        prefix (str):  Only consider frame files with this prefix.

    Returns:
        (dict):  The observation dictionary.

    """
    log = Logger.get()
    rank = 0
    if mpicomm is not None:
        rank = mpicomm.rank
    frame_sizes = {}
    frame_sizes_by_offset = {}
    frame_sample_offs = {}
    file_names = list()
    file_sample_offs = {}
    nframes = {}

    obs = dict()

    latest_obs = None
    latest_cal_frames = []
    first_offset = None

    if rank == 0:
        pat = None
        if prefix is None:
            pat = re.compile(r".*_(\d{8}).g3")
        else:
            pat = re.compile(r"{}_(\d{{8}}).g3".format(prefix))
        for root, dirs, files in os.walk(path, topdown=True):
            for f in sorted(files):
                fmat = pat.match(f)
                if fmat is not None:
                    ffile = os.path.join(path, f)
                    fsampoff = int(fmat.group(1))
                    if first_offset is None:
                        first_offset = fsampoff
                    file_names.append(ffile)
                    allframes = 0
                    file_sample_offs[ffile] = fsampoff
                    frame_sizes[ffile] = []
                    frame_sample_offs[ffile] = []
                    for frame in core3g.G3File(ffile):
                        allframes += 1
                        if frame.type == core3g.G3FrameType.Scan:
                            # This is a scan frame, process it.
                            fsz = len(frame["boresight"]["az"])
                            if fsampoff not in frame_sizes_by_offset:
                                frame_sizes_by_offset[fsampoff] = fsz
                            else:
                                if frame_sizes_by_offset[fsampoff] != fsz:
                                    raise RuntimeError(
                                        "Frame size at {} changes. {} != {}"
                                        "".format(
                                            fsampoff,
                                            frame_sizes_by_offset[fsampoff],
                                            fsz))
                            frame_sample_offs[ffile].append(fsampoff)
                            frame_sizes[ffile].append(fsz)
                            fsampoff += fsz
                        else:
                            frame_sample_offs[ffile].append(0)
                            frame_sizes[ffile].append(0)
                            if frame.type == core3g.G3FrameType.Observation:
                                latest_obs = frame
                            elif frame.type == core3g.G3FrameType.Calibration:
                                if fsampoff == first_offset:
                                    latest_cal_frames.append(frame)
                            else:
                                # Unknown frame type- skip it.
                                pass
                    frame_sample_offs[ffile] = np.array(frame_sample_offs[ffile], dtype=np.int64)
                    nframes[ffile] = allframes
                    log.debug("{} starts at {} and has {} frames".format(
                        ffile, file_sample_offs[ffile], nframes[ffile]))
            break
        if len(file_names) == 0:
            raise RuntimeError(
                "No frames found at '{}' with prefix '{}'"
                .format(path, prefix))

    if mpicomm is not None:
        latest_obs = mpicomm.bcast(latest_obs, root=0)
        latest_cal_frames = mpicomm.bcast(latest_cal_frames, root=0)
        nframes = mpicomm.bcast(nframes, root=0)
        file_names = mpicomm.bcast(file_names, root=0)
        file_sample_offs = mpicomm.bcast(file_sample_offs, root=0)
        frame_sizes = mpicomm.bcast(frame_sizes, root=0)
        frame_sizes_by_offset = mpicomm.bcast(frame_sizes_by_offset, root=0)
        frame_sample_offs = mpicomm.bcast(frame_sample_offs, root=0)

    if latest_obs is None:
        raise RuntimeError("No observation frame was found!")
    for k, v in latest_obs.iteritems():
        obs[k] = s3utils.from_g3_type(v)

    if len(latest_cal_frames) == 0:
        raise RuntimeError("No calibration frame with detector offsets!")
    detoffset, noise = parse_cal_frames(latest_cal_frames, dets)

    obs["noise"] = noise

    obs["tod"] = SOTOD(path, file_names, nframes, file_sample_offs,
                       frame_sizes, frame_sizes_by_offset,
                       frame_sample_offs, detquats=detoffset,
                       mpicomm=mpicomm, **kwargs)
    return obs


def obsweight(path, prefix=None):
    """Compute frame file sizes.

    This uses the sizes of the frame files in an observation as a proxy for
    the amount of data in that observation.  This allows us to approximately
    load balance the observations across process groups.

    Args:
        path (str):  The directory of frame files.

    Returns:
        (float):  Approximate total size in MB.

    """
    pat = None
    if prefix is None:
        pat = re.compile(r".*_\d{8}.g3")
    else:
        pat = re.compile(r"{}_\d{{8}}.g3".format(prefix))
    total = 0
    for root, dirs, files in os.walk(path, topdown=True):
        for f in files:
            mat = pat.match(f)
            if mat is not None:
                statinfo = os.stat(os.path.join(root, f))
                total += statinfo.st_size
        break
    return float(total) / 1.0e6


def load_data(dir, obs=None, comm=None, prefix=None, **kwargs):
    """Loads data into memory.

    Given a directory tree of observations, load one or more observations.
    The observations are distributed among groups in the toast communicator.
    Additional keyword arguments are passed to the load_observation()
    function.

    Args:
        dir (str):  The top-level directory that contains subdirectories (one
            per observation).
        obs (list):  The list of observations to load.
        comm (toast.Comm): the toast Comm class for distributing the data.
        prefix (str):  Only consider frame files with this prefix.

    Returns:
        (toast.Data):  The distributed data object.

    """
    log = Logger.get()
    # the global communicator
    cworld = comm.comm_world
    # the communicator within the group
    cgroup = comm.comm_group

    # One process gets the list of observation directories
    obslist = list()
    weight = dict()

    worldrank = 0
    if cworld is not None:
        worldrank = cworld.rank

    if worldrank == 0:
        for root, dirs, files in os.walk(dir, topdown=True):
            for d in dirs:
                # FIXME:  Add some check here to make sure that this is a
                # directory of frame files.
                obslist.append(d)
                weight[d] = obsweight(os.path.join(root, dir), prefix=prefix)
            break
        obslist = sorted(obslist)
        # Filter by the requested obs
        fobs = list()
        if obs is not None:
            for ob in obslist:
                if ob in obs:
                    fobs.append(ob)
            obslist = fobs

    if cworld is not None:
        obslist = cworld.bcast(obslist, root=0)
        weight = cworld.bcast(weight, root=0)

    # Distribute observations based on approximate size
    dweight = [weight[x] for x in obslist]
    distobs = distribute_discrete(dweight, comm.ngroups)

    # Distributed data
    data = Data(comm)

    # Now every group adds its observations to the list

    firstobs = distobs[comm.group][0]
    nobs = distobs[comm.group][1]
    for ob in range(firstobs, firstobs+nobs):
        opath = os.path.join(dir, obslist[ob])
        log.debug("Loading {}".format(opath))
        # In case something goes wrong on one process, make sure the job
        # is killed.
        try:
            data.obs.append(
                load_observation(opath, mpicomm=cgroup, prefix=prefix, **kwargs)
            )
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value,
                                               exc_traceback)
            lines = ["Proc {}: {}".format(worldrank, x)
                     for x in lines]
            print("".join(lines), flush=True)
            if cworld is not None:
                cworld.Abort()

    return data

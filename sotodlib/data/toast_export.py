# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST interface tools.

This module contains code for interfacing with TOAST data representations.

"""
import os
import re

import numpy as np

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g

import toast
from toast.tod.interval import intervals_to_chunklist
from toast.tod import spt3g_utils as s3utils

from .toast_frame_utils import tod_to_frames


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
        cache_copy (list):  A list of cache names (<name>_<detector>) that
            contain additional detector signals to be copied to the frames.
        mask_flag_common (int):  Bitmask to apply to common flags.
        mask_flag (int):  Bitmask to apply to per-detector flags.
        filesize (int):  The approximate file size of each frame file in
            bytes.
        units (G3TimestreamUnits):  The units of the detector data.
        detgroups (dict):  Dictionary of groups to arrange the detectors into.
            Each key is a name of the group and each value is a list of
            detector names
        compress (bool or dict):  If True or a dictionary of compression parameters,
            store the timestreams as FLAC-compressed, 24-bit integers instead of
            uncompressed doubles.
        verbose (bool):  Verbose reporting

    """
    def __init__(self, outdir, prefix="so", use_todchunks=False,
                 use_intervals=False, cache_name=None, cache_common=None,
                 cache_flag_name=None, cache_copy=None, mask_flag_common=255,
                 mask_flag=255, filesize=500000000, units=None,
                 detgroups=None,
                 compress=False, verbose=True,
    ):
        self._outdir = outdir
        self._prefix = prefix
        self._cache_common = cache_common
        self._cache_name = cache_name
        self._cache_flag_name = cache_flag_name
        self._cache_copy = cache_copy
        self._mask_flag = mask_flag
        self._mask_flag_common = mask_flag_common
        if use_todchunks and use_intervals:
            raise RuntimeError("cannot use both TOD chunks and Intervals")
        self._usechunks = use_todchunks
        self._useintervals = use_intervals
        self._target_framefile = filesize
        self._units = units
        self._detgroups = detgroups
        self._compress = compress
        self._verbose = verbose
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
        #   - 1 x 1 bytes for common flags
        #   - 4 x 8 bytes for boresight RA/DEC quats
        #   - 4 x 8 bytes for boresight Az/El quats
        #   - 2 x 8 bytes for boresight Az/El angles
        #   - 3 x 8 bytes for telescope position
        #   - 3 x 8 bytes for telescope velocity
        #   - 1 x 8 bytes x number of dets x number of flavors
        persample = 8 + 1 + 32 + 48 + 24 + 24 + 8 * ndet * nflavor
        return persample

    def _get_framesizes(self, tod, obs, nsamp, keep_offsets):
        """ Determine frame sizes based on the data distribution
        """
        if keep_offsets:
            framesizes = self._framesizes
        else:
            framesizes = None
            if self._usechunks:
                framesizes = tod.total_chunks
            elif self._useintervals:
                if "intervals" not in obs:
                    raise RuntimeError(
                        "Observation does not contain intervals, cannot "
                        "distribute using them")
                framesizes = intervals_to_chunklist(obs["intervals"], nsamp)
            if framesizes is None:
                framesizes = [nsamp]
            self._framesizes = framesizes
        return framesizes

    def _get_flavors(self, tod, detnames, grouprank):
        """ Examine all the cache objects and find the set of prefixes
        """
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
        # If the main signals and flags are coming from the cache, remove
        # them from consideration here.
        if self._cache_name is not None:
            flavors.discard(self._cache_name)
        if self._cache_flag_name is not None:
            flavors.discard(self._cache_flag_name)

        # Restrict this list of available flavors to just those that
        # we want to export.
        copy_flavors = []
        if self._cache_copy is not None:
            copy_flavors = list()
            for flv in flavors:
                if flv in self._cache_copy:
                    copy_flavors.append(
                        (flv, flavor_type[flv], flavor_maptype[flv],
                         "signal_{}".format(flv)))
            if grouprank == 0 and len(copy_flavors) > 0 and self._verbose:
                print("Found {} extra TOD flavors: {}".format(
                    len(copy_flavors), copy_flavors), flush=True)

        return flavors, flavor_type, flavor_maptype, copy_flavors

    def _get_offsets(self, cgroup, grouprank, keep_offsets, ndet, nflavor, framesizes):
        """Given the dimensions of this observation, compute the frame
        file sizes and all relevant offsets.
        """

        frame_sample_offs = None
        file_sample_offs = None
        file_frame_offs = None
        if grouprank == 0:
            # Compute the frame file breaks.  We ignore the observation
            # and calibration frames since they are small.
            if keep_offsets:
                file_sample_offs = self._file_sample_offs
                file_frame_offs = self._file_frame_offs
                frame_sample_offs = self._frame_sample_offs
            else:
                sampbytes = self._bytes_per_sample(ndet, nflavor + 1)
                file_sample_offs, file_frame_offs, frame_sample_offs = \
                    s3utils.compute_file_frames(
                        sampbytes, framesizes,
                        file_size=self._target_framefile)
                self._file_sample_offs = file_sample_offs
                self._file_frame_offs = file_frame_offs
                self._frame_sample_offs = frame_sample_offs

        if cgroup is not None:
            file_sample_offs = cgroup.bcast(file_sample_offs, root=0)
            file_frame_offs = cgroup.bcast(file_frame_offs, root=0)
            frame_sample_offs = cgroup.bcast(frame_sample_offs, root=0)
        return frame_sample_offs, file_sample_offs, file_frame_offs

    def _export_observation(self, obs, cgroup,
                            detgroup=None, detectors=None, keep_offsets=False):
        """ Export observation in one or more frame files
        """

        grouprank = 0
        if cgroup is not None:
            grouprank = cgroup.rank

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
        if detectors is None:
            detquat = tod.detoffset()
            detindx = tod.detindx
            detnames = tod.detectors
        else:
            detquat_temp = tod.detoffset()
            detindx_temp = tod.detindx
            detquat = {}
            detindx = {}
            detnames = []
            toddets = set(tod.detectors)
            for det in detectors:
                if det not in toddets:
                    continue
                detnames.append(det)
                detquat[det] = detquat_temp[det]
                detindx[det] = detindx_temp[det]
        ndets = len(detquat)

        # Get any other metadata from the TOD
        props.update(tod.meta)

        # First process in the group makes the output directory
        obsdir = os.path.join(self._outdir, obsname)
        if cgroup.rank == 0:
            if not os.path.isdir(obsdir):
                os.makedirs(obsdir)
        cgroup.barrier()

        detranks, sampranks = tod.grid_size

        framesizes = self._get_framesizes(tod, obs, nsamp, keep_offsets)

        (flavors, flavor_type, flavor_maptype, copy_flavors
        ) = self._get_flavors(tod, detnames, grouprank)

        (frame_sample_offs, file_sample_offs, file_frame_offs
        ) = self._get_offsets(cgroup, grouprank, keep_offsets, len(detquat),
                              len(copy_flavors), framesizes)

        if detgroup is None:
            prefix = self._prefix
        else:
            prefix = "{}_{}".format(self._prefix, detgroup)
        ex_files = [os.path.join(obsdir,
                    "{}_{:08d}.g3".format(prefix, x))
                    for x in file_sample_offs]

        # Loop over each frame file.  Write the header frames and then
        # gather the data from all processes before writing the scan
        # frames.

        for ifile, (ffile, foff) in enumerate(zip(ex_files,
                                              file_frame_offs)):
            nframes = None
            # print("  ifile = {}, ffile = {}, foff = {}"
            #       .format(ifile, ffile, foff), flush=True)
            if ifile == len(ex_files) - 1:
                # we are at the last file
                nframes = len(framesizes) - foff
            else:
                # get number of frames in this file
                nframes = file_frame_offs[ifile + 1] - foff

            writer = None
            if grouprank == 0:
                writer = core3g.G3Writer(ffile)
                self._write_obs(writer, props, detindx)
                if "noise" in obs:
                    self._write_precal(writer, detquat, obs["noise"])
                else:
                    self._write_precal(writer, detquat, None)

            # Collect data for all frames in the file in one go.

            frm_offsets = [frame_sample_offs[foff + f]
                           for f in range(nframes)]
            frm_sizes = [framesizes[foff + f] for f in range(nframes)]

            if grouprank == 0 and self._verbose:
                print("  {} file {} detector group {}".format(
                    obsdir, ifile, detgroup), flush=True)
                print("    start frame = {}, nframes = {}"
                      .format(foff, nframes), flush=True)
                print("    frame offs = ", frm_offsets, flush=True)
                print("    frame sizes = ", frm_sizes, flush=True)

            fdata = tod_to_frames(
                tod, foff, nframes, frm_offsets, frm_sizes,
                cache_signal=self._cache_name,
                cache_flags=self._cache_flag_name,
                cache_common_flags=self._cache_common,
                copy_common=None,
                copy_detector=copy_flavors,
                units=self._units,
                dets=detnames,
                mask_flag_common=self._mask_flag_common,
                mask_flag=self._mask_flag,
                compress=self._compress,
            )

            if grouprank == 0:
                for fdt in fdata:
                    writer(fdt)
                del writer
            del fdata

        return

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

        worldrank = 0
        if cworld is not None:
            worldrank = cworld.rank

        # One process checks the path
        if worldrank == 0:
            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir)
        cworld.barrier()

        for obs in data.obs:
            if self._detgroups is None:
                detgroup = None
                detectors = obs["tod"].detectors
                self._export_observation(obs, cgroup)
            else:
                keep_offsets = False
                for detgroup, detectors in self._detgroups.items():
                    self._export_observation(obs, cgroup, detgroup, detectors,
                                             keep_offsets)
                    keep_offsets = True

        return

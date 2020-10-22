# Copyright (c) 2018-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST export tools.

This module contains code for dumping "raw" simulated data.

"""
import os
import re

import numpy as np

import traitlets

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g

import toast

from toast.timing import function_timer, Timer

import toast.traits as trts

from toast import Operator

from toast.tod.interval import intervals_to_chunklist

from toast import spt3g as toast3g

from .frame_utils import tod_to_frames


@trts.trait_docs
class Export3G(Operator):
    """Operator which writes data to a directory tree of frame files.

    The top level directory will contain one subdirectory per observation.
    Each observation directory will contain frame files of approximately
    the specified size.  A single frame file will contain multiple frames.
    The size of each frame is determined by either the observation sample
    sets or the specified interval name.

    """

    # Class traits

    API = trts.Int(0, help="Internal interface version for this operator")

    out_dir = trts.Unicode("export3g", help="Top-level directory for the export")

    file_prefix = trts.Unicode("so", help="Prefix for each frame file.")

    file_size = trts.Int(500000000, help="Approximate frame file size in bytes")

    frame_intervals = trts.Unicode(
        None, allow_none=True, help="Name of intervals to use for frame boundaries."
    )

    g3units = trts.Instance(
        klass=core3g.G3TimestreamUnits,
        allow_none=True,
        help="The G3 units of the timestream data",
    )

    compress = trts.Bool(False, help="Observation shared key for timestamps")

    compress_params = trts.Dict(
        None, allow_none=True, help="FLAC compression parameters"
    )

    verbose = trts.Bool(False, help="If True, write excessive information")

    @traitlets.validate("file_size")
    def _check_realization(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Frame file size must be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        # The two-level toast communicator
        comm = data.comm
        # The global communicator
        cworld = comm.comm_world
        # The communicator within the group
        cgroup = comm.comm_group
        # The communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        worldrank = 0
        if cworld is not None:
            worldrank = cworld.rank

        # One process checks the path
        if worldrank == 0:
            if not os.path.isdir(self.out_dir):
                os.makedirs(self.out_dir)
        cworld.barrier()

        for ob in data.obs:
            # To allow parallel writing of frame files, we first redistribute so that
            # every process has all detectors for a slice of time.
            ob.redistribute(1)

            # Export this observation.  The grouping of detectors is set by the
            # observation detector_sets, and the frame sizes are set by the observation
            # sample_sets.

            # Compute the frame file boundaries for this process.

            # First export shared data

        return

    def _finalize(self, data, **kwargs):
        pass

    def _requires(self):
        return dict()

    def _provides(self):
        return {
            "shared": [
                self.times,
                self.flags,
                self.boresight,
                self.hwp_angle,
                self.position,
                self.velocity,
            ]
        }

    def _accelerators(self):
        return list()

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
            if k == "detector_uid":
                # old indices loaded from so3g file
                continue
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
                    if wt != 0:
                        st[d].append(noise.index(k))
                        wts[d].append(wt)
            for d in nse_dets:
                f[dstr][d] = core3g.G3VectorInt(st[d])
                f[dwt][d] = core3g.G3VectorDouble(wts[d])
        writer(f)
        return

    # def _get_framesizes(self, obs, nsamp, keep_offsets):
    #     """ Determine frame sizes based on the data distribution
    #     """
    #     if keep_offsets:
    #         framesizes = self._framesizes
    #     else:
    #         framesizes = None
    #         if self._usechunks:
    #             framesizes = tod.total_chunks
    #         elif self._useintervals:
    #             if "intervals" not in obs:
    #                 raise RuntimeError(
    #                     "Observation does not contain intervals, cannot "
    #                     "distribute using them")
    #             framesizes = intervals_to_chunklist(obs["intervals"], nsamp)
    #         if framesizes is None:
    #             framesizes = [nsamp]
    #         self._framesizes = framesizes
    #     return framesizes
    #
    # def _get_flavors(self, tod, detnames, grouprank):
    #     """ Examine all the cache objects and find the set of prefixes
    #     """
    #     flavors = set()
    #     flavor_type = dict()
    #     flavor_maptype = dict()
    #     pat = re.compile(r"^(.*?)_(.*)")
    #     for nm in list(tod.cache.keys()):
    #         mat = pat.match(nm)
    #         if mat is not None:
    #             pref = mat.group(1)
    #             md = mat.group(2)
    #             if md in detnames:
    #                 # This cache field has the form <prefix>_<det>
    #                 if pref not in flavor_type:
    #                     ref = tod.cache.reference(nm)
    #                     if ref.dtype == np.dtype(np.float64):
    #                         flavors.add(pref)
    #                         flavor_type[pref] = core3g.G3Timestream
    #                         flavor_maptype[pref] = core3g.G3TimestreamMap
    #                     elif ref.dtype == np.dtype(np.int32):
    #                         flavors.add(pref)
    #                         flavor_type[pref] = core3g.G3VectorInt
    #                         flavor_maptype[pref] = core3g.G3MapVectorInt
    #                     elif ref.dtype == np.dtype(np.uint8):
    #                         flavors.add(pref)
    #                         flavor_type[pref] = so3g.IntervalsInt
    #                         flavor_maptype[pref] = so3g.MapIntervalsInt
    #     # If the main signals and flags are coming from the cache, remove
    #     # them from consideration here.
    #     if self._cache_name is not None:
    #         flavors.discard(self._cache_name)
    #     if self._cache_flag_name is not None:
    #         flavors.discard(self._cache_flag_name)
    #
    #     # Restrict this list of available flavors to just those that
    #     # we want to export.
    #     copy_flavors = []
    #     if self._cache_copy is not None:
    #         copy_flavors = list()
    #         for flv in flavors:
    #             if flv in self._cache_copy:
    #                 copy_flavors.append(
    #                     (flv, flavor_type[flv], flavor_maptype[flv],
    #                      "signal_{}".format(flv)))
    #         if grouprank == 0 and len(copy_flavors) > 0 and self._verbose:
    #             print("Found {} extra TOD flavors: {}".format(
    #                 len(copy_flavors), copy_flavors), flush=True)
    #
    #     return flavors, flavor_type, flavor_maptype, copy_flavors
    #
    # def _get_offsets(self, cgroup, grouprank, keep_offsets, ndet, nflavor, framesizes):
    #     """Given the dimensions of this observation, compute the frame
    #     file sizes and all relevant offsets.
    #     """
    #
    #     frame_sample_offs = None
    #     file_sample_offs = None
    #     file_frame_offs = None
    #     if grouprank == 0:
    #         # Compute the frame file breaks.  We ignore the observation
    #         # and calibration frames since they are small.
    #         if keep_offsets:
    #             file_sample_offs = self._file_sample_offs
    #             file_frame_offs = self._file_frame_offs
    #             frame_sample_offs = self._frame_sample_offs
    #         else:
    #             sampbytes = self._bytes_per_sample(ndet, nflavor + 1)
    #             file_sample_offs, file_frame_offs, frame_sample_offs = \
    #                 s3utils.compute_file_frames(
    #                     sampbytes, framesizes,
    #                     file_size=self._target_framefile)
    #             self._file_sample_offs = file_sample_offs
    #             self._file_frame_offs = file_frame_offs
    #             self._frame_sample_offs = frame_sample_offs
    #
    #     if cgroup is not None:
    #         file_sample_offs = cgroup.bcast(file_sample_offs, root=0)
    #         file_frame_offs = cgroup.bcast(file_frame_offs, root=0)
    #         frame_sample_offs = cgroup.bcast(frame_sample_offs, root=0)
    #     return frame_sample_offs, file_sample_offs, file_frame_offs

    def _bytes_per_sample(self, obs, dets):
        """Compute the bytes per time sample.

        For the local observation data, compute the bytes per sample needed to export
        all shared and detdata objects.
        """
        total = 0
        n_det = len(dets)
        for key in obs.shared.keys():
            shp = obs.shared[key].shape
            if shp[0] != obs.n_local_samples:
                # This is not a timestream quantity
                continue
            detelem = 1
            if len(shp) > 1:
                for s in shp[1:]:
                    detelem *= s
            dt = obs.shared[key].dtype
            total += detelem * dt.itemsize
        for key in obs.detdata.keys():
            shp = obs.detdata[key].shape
            n_samp = shp[1]
            detelem = 1
            if len(shp) > 2:
                for s in shp[2:]:
                    detelem *= s
            dt = obs.detdata[key].dtype
            total += n_det * detelem * dt.itemsize
        return total

    def _frame_file_offsets(self, obs):
        """Compute the frame file boundaries and frame offsets.

        Given the local observation data (this assumes that the observation has
        been redistributed), compute the frame and file offsets for every detector set.

        """
        offsets = list()
        for ds in obs.local_detector_sets:
            bps = self._bytes_per_sample(obs, ds)

    def _export_observation(
        self, obs, cgroup, detgroup=None, detectors=None, keep_offsets=False
    ):
        """Export observation in one or more frame files"""

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

        # Some fields are specific to telescope
        telescope_name = obs["telescope"]

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

        (flavors, flavor_type, flavor_maptype, copy_flavors) = self._get_flavors(
            tod, detnames, grouprank
        )

        (frame_sample_offs, file_sample_offs, file_frame_offs) = self._get_offsets(
            cgroup,
            grouprank,
            keep_offsets,
            len(detquat),
            len(copy_flavors),
            framesizes,
            telescope_name,
        )

        if detgroup is None:
            prefix = self._prefix
        else:
            prefix = "{}_{}".format(self._prefix, detgroup)
        ex_files = [
            os.path.join(obsdir, "{}_{:08d}.g3".format(prefix, x))
            for x in file_sample_offs
        ]

        # Loop over each frame file.  Write the header frames and then
        # gather the data from all processes before writing the scan
        # frames.

        for ifile, (ffile, foff) in enumerate(zip(ex_files, file_frame_offs)):
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

            frm_offsets = [frame_sample_offs[foff + f] for f in range(nframes)]
            frm_sizes = [framesizes[foff + f] for f in range(nframes)]

            # if grouprank == 0 and self._verbose:
            #     print("  {} file {} detector group {}".format(obsdir, ifile, detgroup), flush=True)
            #     print("    start frame = {}, nframes = {}".format(foff, nframes), flush=True)
            #     print("    frame offs = ", frm_offsets, flush=True)
            #     print("    frame sizes = ", frm_sizes, flush=True)

            fdata = tod_to_frames(
                tod,
                foff,
                nframes,
                frm_offsets,
                frm_sizes,
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

# Copyright (c) 2025-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for interfacing with AxisManagers."""

import re

import numpy as np
# from astropy import units as u
# from astropy.table import Column, QTable

import toast
from toast.mpi import MPI

# from toast.timing import function_timer, Timer
# from toast.utils import Logger
from toast.observation import default_values as defaults

# from ..core import AxisManager
# from ..core.axisman import AxisInterface
# from .instrument import SOSite

# import copy
# import re
# from collections.abc import MutableMapping

# import numpy as np
# from astropy.table import Column

# import toast
# from toast.mpi import MPI
# from toast.timing import function_timer, Timer
# from toast.utils import Logger
# from toast.dist import distribute_uniform
# from toast.observation import default_values as defaults
# from toast.utils import replace_byte_arrays, byte_array_to_unicode


from ..core.axisman import AxisManager, IndexAxis, LabelAxis


demod_toast_prefixes = ["demod0", "demod4r", "demod4i"]

demod_axmgr_names = ["dsT", "demodQ", "demodU"]


# class AxNames:
#     """Class representing naming conventions to use with AxisManagers.

#     Args:
#         override (dict):  Dictionary of overrides.

#     """
#     def __init__(self, override={}):
#         self._ax_names = {
#             "pathsep": ":",
#             "times": "timestamps",
#             "det_data": "signal",
#             "azimuth": "boresight:az",
#             "elevation": "boresight:el",
#             "roll": "boresight:roll",
#             "hwp_angle": "hwp_angle",
#             "wafer_key": "stream_id",
#             "axis_dets": "dets",
#             "axis_samps": "samps",
#         }
#         self._ax_names.update(override)

#     def __getitem__(self, key):
#         if key in self._ax_names:
#             return self._ax_names[key]
#         else:
#             return None


def obs_is_demod(obs, signal=defaults.det_data):
    """Determine if observation has demodulated data.

    Args:
        obs (Observation):  The observation.
        signal (str):  The name of the signal to examine.

    Returns:
        (bool):  True if the data is demodulated.

    """
    local_dets = obs.select_local_detectors()
    local_demod = 0

    if len(local_dets) > 0 and local_dets[0].startswith("demod"):
        local_demod = 1

    if obs.comm.comm_group is not None:
        use_demod = obs.comm.comm_group.allreduce(local_demod, op=MPI.LOR)
    else:
        use_demod = local_demod
    return use_demod


def axmgr_is_demod(axmgr, signal="demodQ"):
    """Determine if an AxisManager has demodulated data.

    Args:
        axmgr (AxisManager):  The observation.
        signal (str):  The name of the signal to examine.

    Returns:
        (list):  The demodulated data prefixes or None

    """
    return signal in axmgr


def obs_sample_axis(obs, name="samps"):
    """Given an Observation, return an Axis for the sample direction."""
    return IndexAxis(name, count=obs.n_all_samples)


def obs_det_axis(obs, signal=defaults.det_data, name="dets"):
    """Given an Observation, return an Axis for the detector direction.

    If the data is demodulated, return a LabelAxis with the root detector
    names.

    Args:

    Returns:


    """
    if obs_is_demod(obs, signal=signal):
        # We have demodulated data.  Get the root detector names.
        detpat = re.compile(r"(demod[024ri]+)_(.*)")
        detnames = set()
        for det in obs.all_detectors:
            mat = detpat.match(det)
            if mat is None:
                msg = f"Detector name '{det}' does not follow expected"
                msg += " convention for demodulated detectors"
                raise RuntimeError(msg)
            detnames.add(mat.group(2))
        return LabelAxis(name, vals=sorted(detnames))
    else:
        return LabelAxis(name, vals=obs.all_detectors)


def _obs_parse_session(session):
    """Given an observing session with standard naming, return properties."""
    name_pat = re.compile(r"obs_(\d+)_(.*)_(\d+)")
    name_mat = name_pat.match(session.name)
    if name_mat is None:
        msg = f"Session name '{session.name}' does not match expected format"
        raise RuntimeError(msg)
    time = int(name_mat.group(1))
    tele = name_mat.group(2)
    wafer_mask = [int(x) for x in list(name_mat.group(3))]
    wafer_count = np.count_nonzero(wafer_mask)
    return time, tele, wafer_count


def obs_meta_other(obs, root_axmgr, pathsep):
    """Extract other metadata into nested AxisManagers.

    This extracts the "top-level" metadata into the root AxisManager.  It
    ignores any objects that already exist, to avoid overwriting information
    extracted from the focalplane table.

    """
    for k, v in obs.items():
        if k == "obs_info":
            # This is handled separately
            continue
        if


def obs_meta_info(obs, root_axmgr, pathsep, recompute=False):
    """Extract the obs_info AxisManager and place it in the root.

    The `obs_info` structure includes a combination of fields from the toast
    Observation and its parent Session.  If the original `obs_info` structure
    is found in the Observation metadata, it is returned unless recompute is
    True.

    Args:


    """
    # See if we already have the original obs_info in the metadata.  If so, just return
    # that original thing
    if "obs_info" in obs and not recompute:
        obs_info = AxisManager()
        for k, v in obs["obs_info"].items():
            obs_info.wrap(k, v)
        root_axmgr.wrap("obs_info", obs_info)
        return

    if not obs.is_distributed_by_detector:
        msg = f"Observation {obs.name} is not distributed by detector. "
        msg += "Other distributions not implemented in this function."
        raise RuntimeError(msg)

    # The "obs_id" is actually the Session name
    obs_id = obs.session.name

    # Toast does not track the list of wafer slots and stream IDs within the session.
    # We also do not save separately the telescope tube slot and tube "flavor".  We
    # reconstruct this information from the focalplane information for each observation.
    bands_to_flavor = {
        "f030": "lf",
        "f040": "lf",
        "f090": "mf",
        "f150": "mf",
        "f220": "uhf",
        "f280": "uhf",
    }

    fp = obs.telescope.focalplane.detector_data
    wafer_slots = set(np.unique(fp[f"det_info{pathsep}wafer_slot"]))
    stream_ids = set(np.unique(fp[f"det_info{pathsep}stream_id"]))
    tel_tubes = set(np.unique(fp[f"det_info{pathsep}tel_tube"]))
    det_bands = set(np.unique(fp[f"det_info{pathsep}wafer{pathsep}bandpass"]))

    tube_flavors = {bands_to_flavor[x] for x in det_bands}
    if len(tube_flavors) != 1:
        msg = f"Observation '{obs.name}' does not have"
        msg += f" exactly one tube flavor ({tube_flavors})"
        raise RuntimeError(msg)
    tube_flavor = tube_flavors.pop()
    if len(tel_tubes) != 1:
        msg = f"Observation '{obs.name}' does not have"
        msg += f" exactly one tube slot ({tel_tubes})"
        raise RuntimeError(msg)

    wafer_slots_list = sorted(wafer_slots)
    stream_ids_list = sorted(stream_ids)

    # The "tube_slot" for SATs in the obsdb is somewhat random, but in det_info
    # it matches the normal telescope name.  That is what we use here.
    tel_tube = tel_tubes.pop()

    # These correspond to the session time range
    timestamp = obs.session.start.timestamp()
    start_time = obs.session.start.timestamp()
    stop_time = obs.session.end.timestamp()
    duration = stop_time - start_time

    # The number of total samples in the observation
    n_samples = obs.n_all_samples

    # Extract info from the session name
    int_start, tele_name, wafer_count = _obs_parse_session(obs.session)
    tele_flavor = tele_name[:3]

    # We are only concerned with converting science obs
    obs_type = "obs"
    obs_subtype = "cmb"

    # To compute metadata about the boresight az / el / roll range, we look at the
    # first observation in the session, since this is the same for all wafers.

    def _throw_center(raw_ang):
        ang = np.unwrap(raw_ang)
        angmin = np.fmod(np.amin(ang), 2 * np.pi)
        angmax = np.fmod(np.amax(ang), 2 * np.pi)
        if angmax >= angmin:
            ang_throw = 0.5 * (angmax - angmin)
            ang_center = angmin + ang_throw
        else:
            ang_throw = 0.5 * (angmax + (2.0 * np.pi - angmin))
            ang_center = angmin - ang_throw
            if ang_center < 0:
                ang_center += 2 * np.pi
        ang_throw = np.degrees(ang_throw)
        ang_center = np.degrees(ang_center)
        return ang_center, ang_throw

    az_center = None
    az_throw = None
    el_center = None
    el_throw = None
    roll_center = None
    roll_throw = None
    hwp_freq_mean = None
    hwp_freq_stdev = None

    if obs.comm.group_rank == 0:
        bad = obs.shared[defaults.shared_flags].data & defaults.shared_mask_nonscience
        good = np.logical_not(bad)
        good_times = obs.shared[defaults.times].data[good]

        good_az = obs.shared[defaults.azimuth].data[good]
        az_center, az_throw = _throw_center(good_az)

        good_el = obs.shared[defaults.elevation].data[good]
        elmin = np.amin(good_el)
        elmax = np.amax(good_el)
        el_throw = 0.5 * np.degrees(elmax - elmin)
        el_center = np.degrees(elmin + 0.5 * el_throw)

        good_roll = obs.shared[defaults.roll].data[good]
        roll_center, roll_throw = _throw_center(good_roll)

        if defaults.hwp_angle in obs.shared:
            # We have HWP data
            good_hwp = obs.shared[defaults.hwp_angle].data[good]
            hwp = np.unwrap(good_hwp)
            hwp_rate = np.diff(hwp) / np.diff(good_times)
            hwp_freq_mean = np.mean(hwp_rate) / (2 * np.pi)
            hwp_freq_stdev = np.std(hwp_rate) / (2 * np.pi)
    if obs.comm.comm_group is not None:
        az_center = obs.comm.comm_group.bcast(az_center, root=0)
        az_throw = obs.comm.comm_group.bcast(az_throw, root=0)
        el_center = obs.comm.comm_group.bcast(el_center, root=0)
        el_throw = obs.comm.comm_group.bcast(el_throw, root=0)
        roll_center = obs.comm.comm_group.bcast(roll_center, root=0)
        roll_throw = obs.comm.comm_group.bcast(roll_throw, root=0)
        hwp_freq_mean = obs.comm.comm_group.bcast(hwp_freq_mean, root=0)
        hwp_freq_stdev = obs.comm.comm_group.bcast(hwp_freq_stdev, root=0)

    obs_info = AxisManager()

    obs_info.wrap("obs_id", obs_id)
    obs_info.wrap("timestamp", timestamp)
    obs_info.wrap("start_time", start_time)
    obs_info.wrap("n_samples", n_samples)
    obs_info.wrap("telescope", tele_name)
    obs_info.wrap("telescope_flavor", tele_flavor)
    obs_info.wrap("tube_slot", tel_tube)
    obs_info.wrap("tube_flavor", tube_flavor)
    obs_info.wrap("type", obs_type)
    obs_info.wrap("subtype", obs_subtype)
    obs_info.wrap("wafer_count", wafer_count)

    # FIXME: what about KIDs?
    obs_info.wrap("detector_flavor", "tes")

    obs_info.wrap("az_center", az_center)
    obs_info.wrap("az_throw", az_throw)
    obs_info.wrap("el_center", el_center)
    obs_info.wrap("el_throw", el_throw)

    obs_info.wrap("stop_time", stop_time)
    obs_info.wrap("duration", duration)

    obs_info.wrap("roll_center", roll_center)
    obs_info.wrap("roll_throw", roll_throw)

    if hwp_freq_mean is not None:
        obs_info.wrap("hwp_freq_mean", hwp_freq_mean)
        obs_info.wrap("hwp_freq_stdev", hwp_freq_stdev)

    obs_info.wrap("wafer_slots_list", ",".join(wafer_slots_list))
    obs_info.wrap("stream_ids_list", ",".join(stream_ids_list))

    root_axmgr.wrap("obs_info", obs_info)


def obs_metadata(obs, pathsep, recompute_obsinfo=False):
    """Create the top-level AxisManager.

    This function creates the AxisManager representing the observation and sets
    up the detector and sample axes.  Then it extracts all general metadata and
    detector properties.




    """
    det_axis = obs_det_axis(obs)
    samp_axis = obs_sample_axis(obs)
    root_axmgr = AxisManager(det_axis, samp_axis)

    obs_meta_info(obs, root_axmgr, pathsep, recompute=recompute_obsinfo)

    obs_meta_other(obs, root_axmgr, pathsep)

    return root_axmgr


def observation_to_axmgr(
    obs, pathsep=":", require_groupsize_one=False, recompute_obsinfo=False
):
    """Create an AxisManager from an Observation.

    The input Observation is distributed over a process group.  The resulting
    AxisManager result will only exist on the first process in the group.  Other
    processes in the group will have `None` returned.

    Args:
        obs (Observation):  The observation.
        require_groupsize_one (bool):  If True require the input process group to have
            a single process and raise an exception if not.

    Returns:
        (AxisManager):  The resulting AxisManager on the first process in the group
            and None on other processes.

    """
    if require_groupsize_one and obs.comm.group_size != 1:
        msg = f"Observation {obs.name} has process group size of {obs.comm.group_size}"
        msg += ", but a group size of one is required."
        raise RuntimeError(msg)

    # Get the top-level AxisManager and metadata
    root_axmgr = obs_metadata(obs, pathsep, recompute_obsinfo=False)

    # Extract shared data

    # Extract detector data

    return root_axmgr


# def _wrap_obs(self, ob, dets, passinfo):
#     """ Prepare data for the mapmaker """

#     # Get the focalplane for this observation
#     fp = ob.telescope.focalplane

#     # Get the sample rate from the data.  We also have nominal sample rates
#     # from the noise model and also from the focalplane.
#     # (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
#     #     ob.shared[self.times].data
#     # )

#     axdets = LabelAxis("dets", dets)
#     nsample = int(ob.n_local_samples)

#     axsamps = OffsetAxis(
#         "samps",
#         count=nsample,
#         offset=ob.local_index_offset,
#         origin_tag=ob.name,
#     )

#     # Convert the data view into a RangesMatrix
#     ranges = so3g.proj.ranges.RangesMatrix.zeros((len(dets), nsample))
#     if self.view is not None:
#          view_ranges = np.array(
#              [[x.first, x.last] for x in ob.intervals[self.view]]
#          )
#          ranges += so3g.proj.ranges.Ranges.from_array(view_ranges.astype(np.int32), nsample)

#     # Convert the focalplane offsets into the expected form
#     det_to_row = {y["name"]: x for x, y in enumerate(fp.detector_data)}
#     det_quat = np.array([fp.detector_data["quat"][det_to_row[x]] for x in dets])
#     xi, eta, gamma = quat_to_xieta(det_quat)

#     axfp = AxisManager()
#     axfp.wrap("xi", xi, axis_map=[(0, axdets)])
#     axfp.wrap("eta", eta, axis_map=[(0, axdets)])
#     axfp.wrap("gamma", gamma, axis_map=[(0, axdets)])

#     # Convert Az/El quaternion of the detector back into
#     # angles from the simulation.
#     theta, phi, pa = toast.qarray.to_iso_angles(ob.shared[self.boresight])

#     # Azimuth is measured in the opposite direction from longitude
#     az = 2 * np.pi - phi
#     el = np.pi / 2 - theta
#     roll = pa

#     axbore = AxisManager()
#     axbore.wrap("az", az, axis_map=[(0, axsamps)])
#     axbore.wrap("el", el, axis_map=[(0, axsamps)])
#     axbore.wrap("roll", roll, axis_map=[(0, axsamps)])

#     axobs = AxisManager()
#     axobs.wrap("focal_plane", axfp)
#     axobs.wrap("timestamps", ob.shared[self.times], axis_map=[(0, axsamps)])
#     axobs.wrap(
#         "signal",
#         np.vstack(ob.detdata[self.det_data][dets, :]),
#         axis_map=[(0, axdets), (1, axsamps)],
#     )
#     axobs.wrap("boresight", axbore)
#     axobs.wrap('flags', FlagManager.for_tod(axobs))
#     axobs.flags.wrap("glitch_flags", ranges, axis_map=[(0, axdets), (1, axsamps)])
#     axobs.wrap("weather", np.full(1, self.weather))
#     axobs.wrap("site",    np.full(1, "so"))

#     # median detrend added to TOAST workflow script
#     #axobs.signal -= np.median(axobs.signal, axis=-1, keepdims=True)

#     if self.truncate_tod:
#         # FFT-truncate for faster fft ops
#         axobs.restrict("samps", [0, fft.fft_len(axobs.samps.count)])

#     # MLMapmaker.add_obs will apply deslope
#     if self.deslope:
#         utils.deslope(axobs.signal, w=5, inplace=True)

#     if self.downsample != 1:
#         axobs = mm.downsample_obs(axobs, passinfo.downsample)

#     # NOTE:  Expected contents look like:
#     # >>> tod
#     # AxisManager(signal[dets,samps], timestamps[samps], readout_filter_cal[dets],
#     # mce_filter_params[6], iir_params[3,5], flags*[samps], boresight*[samps],
#     # array_data*[dets], pointofs*[dets], focal_plane*[dets], abscal[dets],
#     # timeconst[dets], glitch_flags[dets,samps], source_flags[dets,samps],
#     # relcal[dets], dets:LabelAxis(63), samps:OffsetAxis(372680))
#     # >>> tod.focal_plane
#     # AxisManager(xi[dets], eta[dets], gamma[dets], dets:LabelAxis(63))
#     # >>> tod.boresight
#     # AxisManager(az[samps], el[samps], roll[samps], samps:OffsetAxis(372680))

#     return axobs


# def ax_name_fp_subst(var, fp_array, det=None):
#     """Substitute a focalplane column value into a string.

#     If det is not specified, then there should be a unique name for the whole table
#     column when substituting.

#     Args:
#         var (str):  The input variable string
#         fp_array (Table):  The focalplane astropy Table of detector properties.
#         det (str):  If not None, use the value for this row of the table.

#     Returns:
#         (str):  The string with substitutions made.

#     """
#     if var is None:
#         return None
#     out = ""
#     last = 0
#     for match in re.finditer(r"(\{.*\})", var):
#         out += var[last : match.start()]
#         colname = match.group()
#         colname = colname.replace("{", "")
#         colname = colname.replace("}", "")
#         if colname not in fp_array.keys():
#             msg = f"Column '{colname}' not in focalplane table"
#             raise RuntimeError(msg)
#         if det is None:
#             vals = np.unique(fp_array[colname])
#             if len(vals) == 1:
#                 out += f"{vals[0]}"
#             else:
#                 msg = f"Column '{colname}' has multiple values ({vals}), cannot"
#                 msg += " use in name"
#                 raise RuntimeError(msg)
#         else:
#             mask = fp_array[colname] == det
#             out += f"{fp_array[colname][mask][0]}"
#         last = match.end()
#     out += var[last:]
#     return out


# def parse_axmgr(
#     axman,
#     obs_meta,
#     fp_cols,
#     ax_names,
#     obs_base=None,
#     fp_base=None,
# ):
#     """Parse AxisManager metadata.

#     This is a recursive function which descends the AxisManager hierarchy and splits
#     meta data into "detector properties" appropriate for a TOAST Focalplane and "other
#     metadata" which will be stored in the Observation dictionary.

#     For each element of the input AxisManager we do the following:

#     - If the object has one axis and it is the detector axis,
#       treat it as a column in the detector property table
#     - If the object has one axis and it is the sample axis,
#       it will be loaded later as shared data.
#     - If the object has multiple axes, load it later
#     - If the object has no axes, then treat it as observation
#       metadata
#     - If the object is a nested AxisManager, descend and apply
#       the same steps as above.

#     Args:
#         axman (AxisManager):  The input axismanager.
#         obs_meta (dict):  The working dictionary of observation metadata.
#         fp_cols (dict):  Dictionary of table Columns of detector properties.
#         ax_names (AxNames):  The AxisManager naming conventions to use.
#         obs_base (str):  The current key in the nested observation metadata.
#         fp_base (str):  The current key to use when appending focalplane columns.

#     Returns:
#         None

#     """
#     path_sep = ax_names["path_sep"]
#     det_axis = ax_names["axis_dets"]
#     if obs_base is None:
#         om = obs_meta
#     else:
#         obs_meta[obs_base] = dict()
#         om = obs_meta[obs_base]
#     for key in axman.keys():
#         if fp_base is not None:
#             fp_key = f"{fp_base}{path_sep}{key}"
#         else:
#             fp_key = key
#         if isinstance(axman[key], AxisInterface):
#             # This is one of the axes
#             continue
#         if isinstance(axman[key], AxisManager):
#             # Descend
#             parse_axmgr(
#                 axman[key],
#                 obs_meta,
#                 fp_cols,
#                 ax_names,
#                 obs_base=key,
#                 fp_base=fp_key,
#             )
#         else:
#             # FIXME:  It would be nicer if this information was available
#             # through a public member...
#             field_axes = axman._assignments[key]
#             if len(field_axes) == 0:
#                 # This data is not associated with an axis.
#                 om[key] = axman[key]
#             elif len(field_axes) == 1 and field_axes[0] == det_axis:
#                 # This is a detector property
#                 if fp_key in fp_cols:
#                     msg = f"Metadata key '{fp_key}' is duplicated in nested"
#                     msg += " AxisManagers"
#                     raise RuntimeError(msg)
#                 fp_cols[fp_key] = Column(name=fp_key, data=np.array(axman[key]))
#     if obs_base is not None and len(om) == 0:
#         # There were no meta data keys- delete this dict
#         del obs_meta[obs_base]


# def obs_properties(
#     axmgr,
#     ax_names,
#     analytic_bandwidth=None,
# ):
#     """Parse information needed to construct an Observation.

#     Args:
#         axmgr
#         ax_names (AxNames):  The AxisManager naming conventions to use.
#         analytic_bandwith (float):  The fractional...

#     """

#     ax_pathsep=ax_names["pathsep"],
#     axis_detector=ax_names["axis_dets"],
#     ax_wafer_key=ax_names["wafer_key"],

#     n_samp = axmgr["samps"].count
#     obs_meta = dict()
#     fp_cols = dict()

#     # Parse the axis manager metadata into observation metadata
#     # and detector properties.
#     parse_axmgr(
#         axmgr,
#         obs_meta,
#         fp_cols,
#         ax_pathsep,
#         axis_detector,
#         None,
#         None,
#     )

#     # Set the column used by toast as the detector name.
#     fp_cols["name"] = Column(
#         name="name",
#         data=fp_cols[f"det_info{ax_pathsep}readout_id"].data,
#     )

#     if analytic_bandwidth is not None:
#         # Add bandpass information to the focalplane
#         try:
#             band = fp_cols[f"det_info{ax_pathsep}band"].data
#         except KeyError:
#             band = fp_cols[f"det_info{ax_pathsep}wafer{ax_pathsep}bandpass"].data
#         freq = [float(b[1:]) for b in band]
#         bandcenter = np.array(freq) * u.GHz
#         bandwidth = bandcenter * analytic_bandwidth
#         fp_cols["bandcenter"] = Column(name="bandcenter", data=bandcenter)
#         fp_cols["bandwidth"] = Column(name="bandwidth", data=bandwidth)

#     # Construct detector properties table
#     det_props = QTable(fp_cols)

#     # Detector ordering.  When distributing and reading data, we
#     # need to have the detectors sorted into contiguous blocks of
#     # of detectors.  We sort the table now by wafer and then by name.
#     wafer_key = f"det_info{ax_pathsep}{ax_wafer_key}"
#     det_props.sort([wafer_key, "name"])

#     return (obs_meta, det_props, n_samp)


# def compare_obs_properties(comm, local_props):
#     pass


# @function_timer
# def create_obs_instrument(
#     telescope_name, obs_name, gcomm, det_props, ax_names
# ):
#     """Create telescope, session, and focalplane flags.

#     These are objects needed prior to instantiating the Observation.

#     Args:
#         telescope_name (str):  The name of the telescope.
#         obs_name (str):  The observation name
#         gcomm (MPI.Comm):  The group communicator or None.
#         det_props (QTable):  The astropy table of detector properties.
#         ax_names (AxNames):  The AxisManager naming conventions to use.

#     Returns:
#         (tuple):  The (Telescope, per-detector flags) where the flags
#             cut detectors with the processing bit if they have no
#             pointing reconstruction.

#     """
#     log = Logger.get()

#     ax_pathsep=ax_names["pathsep"]

#     # Convert any focalplane quaternion offsets to toast format.  We also
#     # look for any detectors with NaN values and flag those with the
#     # processing bit.

#     xi_key = f"focal_plane{ax_pathsep}xi"
#     eta_key = f"focal_plane{ax_pathsep}eta"
#     gamma_key = f"focal_plane{ax_pathsep}gamma"
#     wafer_type_key = f"det_info{ax_pathsep}wafer{ax_pathsep}type"

#     fp_flags = dict()
#     if xi_key in det_props.colnames:
#         fp_bad = np.logical_or(
#             np.isnan(det_props[xi_key]),
#             np.logical_or(
#                 np.isnan(det_props[eta_key]),
#                 np.isnan(det_props[gamma_key]),
#             ),
#         )
#         fp_flags = {
#             x: defaults.det_mask_processing
#             for x, y in zip(det_props["name"], fp_bad)
#             if y
#         }
#         det_props[xi_key][fp_bad] = 0
#         det_props[eta_key][fp_bad] = 0
#         det_props[gamma_key][fp_bad] = 0
#         quat_data = toast.instrument_coords.xieta_to_quat(
#             det_props[xi_key],
#             det_props[eta_key],
#             det_props[gamma_key],
#         )
#     else:
#         # No detector offsets yet
#         quat_data = np.tile(
#             np.array([0, 0, 0, 1], dtype=np.float64),
#             len(det_props[f"det_info{ax_pathsep}readout_id"]),
#         ).reshape((-1, 4))

#     # Flag non-optical detectors with the processing mask
#     if wafer_type_key in det_props.colnames:
#         fp_dark = det_props[wafer_type_key] != "OPTC"
#         for dname, dark in zip(det_props["name"], fp_dark):
#             if dark:
#                 fp_flags[dname] |= defaults.det_mask_processing

#     # Do we have any good detectors left?
#     n_good = 0
#     for det in det_props["name"]:
#         if det not in fp_flags or fp_flags[det] == 0:
#             n_good += 1
#     if n_good == 0:
#         log.warning_rank(
#             f"create_obs_instrument: {obs_name} has no unflagged detectors, skipping!",
#             comm=gcomm,
#         )
#         return (None, None)

#     # Add a column for the quaternion offset of each detector, as well
#     # as the gamma angle with standard naming.
#     quat_col = Column(
#         name="quat",
#         data=quat_data,
#     )
#     det_props.add_column(quat_col, index=0)
#     gamma_col = Column(
#         name="gamma",
#         data=det_props[gamma_key] * u.radian,
#     )
#     det_props.add_column(gamma_col)

#     focalplane = toast.instrument.Focalplane(
#         detector_data=det_props,
#         sample_rate=1.0 * u.Hz,
#     )

#     # For now, this should be good enough position for instruments near the
#     # S.O. location.
#     site = SOSite()

#     telescope = toast.instrument.Telescope(
#         telescope_name, focalplane=focalplane, site=site
#     )
#     return telescope, fp_flags


# @function_timer
# def create_observation(
#     obs_name,
#     session_name,
#     comm,
#     telescope,
#     n_samp,
#     fp_flags,
#     meta,
#     ax_names,
#     azimuth=defaults.azimuth,
#     elevation=defaults.elevation,
#     boresight_azel=defaults.boresight_azel,
#     boresight_radec=defaults.boresight_radec,
#     shared_flags=defaults.shared_flags,
#     det_data=defaults.det_data,
#     det_flags=defaults.det_flags,
#     det_data_units=defaults.det_data_units,
#     hwp_angle=None,
#     boresight_angle=None,
#     corotator_angle=None,
#     detset_key=None,
# ):
#     """Create the observation.

#     Note that the focalplane table has already been sorted by wafer
#     name and then detector name.

#     Args:
#         obs_name (str):  The observation name
#         session_name (str):  The observing session name (context obs ID).
#         comm (toast.Comm):  The toast communicator.
#         telescope (Telescope):  The Telescope for this observation.
#         n_samp (int):  The number of samples in the observation.
#         fp_flags (dict):  The per-detector flags to apply.
#         meta (dict):  The observation metadata to populate.
#         ax_names (AxNames):  The AxisManager naming conventions to use.

#         azimuth=defaults.azimuth,
#         elevation=defaults.elevation,
#         boresight_azel=defaults.boresight_azel,
#         boresight_radec=defaults.boresight_radec,
#         shared_flags=defaults.shared_flags,
#         det_data=defaults.det_data,
#         det_flags=defaults.det_flags,
#         det_data_units=defaults.det_data_units,
#         hwp_angle=None,
#         boresight_angle=None,
#         corotator_angle=None,
#         detset_key=None,

#     Returns:
#         (tuple):  The (Observation, have_pointing), where the second
#             element is True if the observation has boresight pointing.

#     """
#     log = Logger.get()
#     timer = Timer()
#     timer.start()

#     times=ax_names["times"]
#     ax_boresight_az=ax_names["azimuth"]
#     ax_boresight_el=ax_names["elevation"]
#     ax_boresight_roll=ax_names["roll"]
#     ax_hwp_angle=ax_names["hwp_angle"]
#     ax_det_data=ax_names["det_data"]
#     roll=ax_names["roll"]

#     # Computer the detector sets
#     if detset_key is None:
#         detsets = None
#     else:
#         detsets = telescope.focalplane.detector_groups(detset_key)

#     # Note:  the session times will be updated later when reading timestamps
#     session = toast.instrument.Session(session_name)

#     # Create the observation
#     ob = toast.Observation(
#         comm,
#         telescope,
#         n_samp,
#         name=obs_name,
#         session=session,
#         detector_sets=detsets,
#         sample_sets=None,
#         process_rows=comm.group_size,
#     )

#     # Apply detector flags for bad pointing reconstruction
#     local_dets = set(ob.local_detectors)
#     local_fp_flags = {x: y for x, y in fp_flags.items() if x in local_dets}
#     ob.update_local_detector_flags(local_fp_flags)

#     # Create observation fields
#     ob.shared.create_column(
#         times,
#         shape=(ob.n_local_samples,),
#         dtype=np.float64,
#     )

#     fp_array = telescope.focalplane.detector_data
#     ax_boresight_az = ax_name_fp_subst(ax_boresight_az, fp_array)
#     ax_boresight_el = ax_name_fp_subst(ax_boresight_el, fp_array)
#     ax_boresight_roll = ax_name_fp_subst(ax_boresight_roll, fp_array)
#     ax_hwp_angle = ax_name_fp_subst(ax_hwp_angle, fp_array)
#     ax_det_signal = ax_name_fp_subst(ax_det_data, fp_array)

#     have_pointing = True
#     if ax_boresight_az is None:
#         have_pointing = False
#     else:
#         ob.shared.create_column(
#             azimuth,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )
#     if ax_boresight_el is None:
#         have_pointing = False
#     else:
#         ob.shared.create_column(
#             elevation,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )
#     if ax_boresight_roll is None:
#         have_pointing = False
#     else:
#         ob.shared.create_column(
#             roll,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )

#     if have_pointing:
#         ob.shared.create_column(
#             boresight_azel,
#             shape=(ob.n_local_samples, 4),
#             dtype=np.float64,
#         )
#         ob.shared.create_column(
#             boresight_radec,
#             shape=(ob.n_local_samples, 4),
#             dtype=np.float64,
#         )
#     if hwp_angle is not None and ax_hwp_angle is not None:
#         ob.shared.create_column(
#             hwp_angle,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )
#     if boresight_angle is not None:
#         ob.shared.create_column(
#             boresight_angle,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )
#     if corotator_angle is not None:
#         ob.shared.create_column(
#             corotator_angle,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )
#     ob.shared.create_column(
#         shared_flags,
#         shape=(ob.n_local_samples,),
#         dtype=np.uint8,
#     )
#     if have_pointing:
#         ob.shared.create_column(
#             defaults.position,
#             shape=(ob.n_local_samples, 3),
#             dtype=np.float64,
#         )
#         ob.shared.create_column(
#             defaults.velocity,
#             shape=(ob.n_local_samples, 3),
#             dtype=np.float64,
#         )
#     if ax_det_signal is not None:
#         ob.detdata.create(det_data, dtype=np.float64, units=det_data_units)
#         ob.detdata.create(det_flags, dtype=np.uint8)

#     if meta is not None:
#         ob.update(meta)

#     log.debug_rank(
#         f"create_observation: {obs_name} allocate Observation in",
#         comm=comm.comm_group,
#         timer=timer,
#     )
#     return ob, have_pointing


# def obs_to_axmgr():
#     """Convert local Observation data into an AxisManager on each process."""

#     # each process builds axman hierarchy for local detectors

#     # wrap signal and other pieces

#     pass


# def axmgr_to_obs(axmgr):
#     """Convert AxisManager data on each process into an Observation."""

#     # parse_meta from either full or partial axis manager to get telescope / sample info

#     obs_properties(
#         axmgr,
#         ax_pathsep=axmgr_defaults["pathsep"],
#         axis_detector=axmgr_defaults["axis_dets"],
#         axis_samps=axmgr_defaults["axis_samps"],
#         ax_wafer_key=axmgr_defaults["wafer_key"],
#         analytic_bandwidth=None,
#     )

#     # verify metadata consistency across processes in the group

#     # Create empty observation or overwrite existing
#     #   - what if we are wrapping the original input observation?

#     pass


# def scatter_axmgr(
#     axmgr,
#     obs_name,
#     session_name,
#     telescope_name,
#     comm,
#     ax_names,
#     analytic_bandwidth=None,
#     azimuth=defaults.azimuth,
#     elevation=defaults.elevation,
#     boresight_azel=defaults.boresight_azel,
#     boresight_radec=defaults.boresight_radec,
#     shared_flags=defaults.shared_flags,
#     det_data=defaults.det_data,
#     det_flags=defaults.det_flags,
#     det_data_units=defaults.det_data_units,
#     hwp_angle=None,
#     boresight_angle=None,
#     corotator_angle=None,
#     detset_key=None,
# ):
#     """Convert an AxisManager on one process into a distributed Observation.


#     """

#     (obs_meta, det_props, n_samp) = obs_properties(
#         axmgr,
#         ax_names=ax_names,
#         analytic_bandwidth=analytic_bandwidth,
#     )

#     telescope, fp_flags = create_obs_instrument(
#         telescope_name, obs_name, comm.comm_group, det_props, ax_names=ax_names
#     )

#     obs, have_pointing = create_observation(
#         obs_name,
#         session_name,
#         comm,
#         telescope,
#         n_samp,
#         fp_flags,
#         obs_meta,
#         ax_names,
#         azimuth=azimuth,
#         elevation=elevation,
#         boresight_azel=boresight_azel,
#         boresight_radec=boresight_radec,
#         shared_flags=shared_flags,
#         det_data=det_data,
#         det_flags=det_flags,
#         det_data_units=det_data_units,
#         hwp_angle=hwp_angle,
#         boresight_angle=boresight_angle,
#         corotator_angle=corotator_angle,
#         detset_key=detset_key,
#     )

#     # Read and communicate data
#     self._load_data(ob, have_pointing, preproc_conf)

#     # Compute the boresight pointing and observatory position
#     if have_pointing:
#         compute_boresight_pointing(
#             ob,
#             self.times,
#             self.azimuth,
#             self.elevation,
#             self.roll,
#             self.boresight_azel,
#             self.boresight_radec,
#             defaults.position,
#             defaults.velocity,
#             self.shared_flags,
#             defaults.shared_mask_processing,
#         )
#     data.obs.append(ob)
#     log.info_rank(
#         f"LoadContext {obs_name} loaded in",
#         comm=comm.comm_group,
#         timer=otimer,
#     )


# def gather_axmgr():
#     """Collect a distributed Observation into an AxisManager on one process."""

#     # rank zero builds high-level axman metadata and hierarchy

#     # communicate per-process signal pieces

#     pass


# def parse_metadata(axman, obs_meta, fp_cols, path_sep, det_axis, obs_base, fp_base):
#     """Parse Context metadata.

#     This is a recursive function which starts with the AxisManger returned
#     by `Context.get_meta()` and splits meta data into "detector properties"
#     which will be placed in the focalplane model and "other metadata" which
#     will be stored in the Observation dictionary.

#     For each element of the input AxisManager we do the following:

#     - If the object has one axis and it is the detector axis,
#       treat it as a column in the detector property table
#     - If the object has one axis and it is the sample axis,
#       it will be loaded later as shared data.
#     - If the object has multiple axes, load it later
#     - If the object has no axes, then treat it as observation
#       metadata
#     - If the object is a nested AxisManager, descend and apply
#       the same steps as above.

#     Args:
#         axman (AxisManager):  The input axismanager.
#         obs_meta (dict):  The working dictionary of observation metadata.
#         fp_cols (dict):  Dictionary of table Columns of detector properties.
#         path_sep (str):  The path separation char when building flattened names.
#         det_axis (str):  The name of the detector axis.
#         obs_base (str):  The current key in the nested observation metadata.
#         fp_base (str):  The current key to use when appending focalplane columns.

#     Returns:
#         None

#     """
#     if obs_base is None:
#         om = obs_meta
#     else:
#         obs_meta[obs_base] = dict()
#         om = obs_meta[obs_base]
#     for key in axman.keys():
#         if fp_base is not None:
#             fp_key = f"{fp_base}{path_sep}{key}"
#         else:
#             fp_key = key
#         if isinstance(axman[key], AxisInterface):
#             # This is one of the axes
#             continue
#         if isinstance(axman[key], AxisManager):
#             # Descend
#             parse_metadata(
#                 axman[key], obs_meta, fp_cols, path_sep, det_axis, key, fp_key
#             )
#         else:
#             # FIXME:  It would be nicer if this information was available
#             # through a public member...
#             field_axes = axman._assignments[key]
#             if len(field_axes) == 0:
#                 # This data is not associated with an axis.
#                 om[key] = replace_byte_arrays(axman[key])
#             elif len(field_axes) == 1 and field_axes[0] == det_axis:
#                 # This is a detector property
#                 if fp_key in fp_cols:
#                     msg = f"Context meta key '{fp_key}' is duplicated in nested"
#                     msg += " AxisManagers"
#                     raise RuntimeError(msg)
#                 fp_cols[fp_key] = Column(
#                     name=fp_key, data=np.array(byte_array_to_unicode(axman[key]))
#                 )
#     if obs_base is not None and len(om) == 0:
#         # There were no meta data keys- delete this dict
#         del obs_meta[obs_base]


# def _local_process_dets(obs, rank):
#     """Get the list of local detectors for a specific rank."""
#     # Full detector list
#     det_names = obs.all_detectors
#     # The range of full detector indices on this process.
#     full_indices = obs.dist.det_indices[rank]
#     # The same info, as a slice.
#     full_slc = slice(full_indices.offset, full_indices.offset + full_indices.n_elem, 1)
#     # The names of the detectors on this process.
#     return det_names[full_slc]


# def distribute_detector_props(obs, wafer_key):
#     """Compute the communication patterns for detector data.

#     This computes the range of local detector indices for each wafer on all
#     processes, and also the corresponding range of indices in the wafer data
#     on the reading process.  For example, imagine we have 2 wafers and 4
#     processes:

#     +====================+    +====================+
#     |       Wafer 0      |    |     Process 0      |
#     |                    |    |--------------------|
#     |                    |    |     Process 1      |
#     +====================+    |                    |
#     |       Wafer 1      |    |--------------------|
#     |                    |    |     Process 2      |
#     |                    |    |--------------------|
#     |                    |    |     Process 3      |
#     +====================+    +--------------------+

#     In this case, Process 0 will read Wafer 0 and Process 1 will read Wafer 1.
#     The data from Wafer 0 needs to be sent to Processes 0 and 1, and the data
#     from Wafer 1 needs to be sent to Processes 1, 2, and 3.

#     Args:
#         obs (Observation):  The observation.
#         wafer_key (str):  The column of the focalplane table with the wafer name.

#     Returns:
#         (tuple):  The (wafer_dets, wafer_readers, wafer_proc_dets, proc_wafer_dets)

#     """
#     gcomm = obs.comm.comm_group
#     rank = obs.comm.group_rank
#     gsize = obs.comm.group_size

#     det_table = obs.telescope.focalplane.detector_data

#     # Wafer names for all detectors
#     det_wafers = list(det_table[wafer_key])

#     # For each process, this has a key for each wafer on that process.
#     # The value for each wafer key is the destination local detector indices for
#     # that wafer.
#     proc_wafer_dets = dict()

#     # For each wafer, this has a key for each destination process.  The value
#     # for each process key is the range of detector indices in the wafer data
#     # that should be sent to that process.
#     wafer_proc_dets = dict()

#     # For each wafer, the process that should do the reading.  This is the first
#     # process rank that has any data from that wafer.
#     wafer_readers = dict()

#     # For each wafer, the list of local detector names within that wafer.
#     wafer_dets = dict()

#     wafer_off = 0
#     cur_wafer = det_wafers[0]
#     wafer_proc_dets[cur_wafer] = dict()
#     wafer_readers[cur_wafer] = 0
#     wafer_dets[cur_wafer] = list()
#     for proc in range(gsize):
#         proc_wafer_dets[proc] = dict()
#         # The range of full detector indices on this process.
#         full_indices = obs.dist.det_indices[proc]
#         # The same info, as a slice.
#         full_slc = slice(
#             full_indices.offset, full_indices.offset + full_indices.n_elem, 1
#         )
#         # The names of the detectors on this process.
#         local_det_names = _local_process_dets(obs, proc)
#         # The wafer names of the detectors on this process.
#         local_det_wafers = det_wafers[full_slc]

#         if local_det_wafers[0] != cur_wafer:
#             # This process starts on a new wafer
#             cur_wafer = local_det_wafers[0]
#             wafer_off = 0
#         # Check for wafer transitions in this process's data
#         wafer_change = [
#             x
#             for x, (y, z) in enumerate(zip(local_det_wafers[:-1], local_det_wafers[1:]))
#             if y != z
#         ]
#         # Add the detector ranges to the mapping dictionaries
#         wfirst = 0
#         for wc in wafer_change:
#             wlast = wc
#             nwblock = wlast - wfirst + 1
#             # Extend list of detectors for this wafer
#             wafer_dets[cur_wafer].extend(local_det_names[wfirst : wlast + 1])
#             # Detector indices for this wafer in the local data
#             proc_wafer_dets[proc][cur_wafer] = (wfirst, wlast + 1)
#             # Detector indices for this process's local data within the full
#             # wafer data.
#             wafer_proc_dets[cur_wafer][proc] = (wafer_off, wafer_off + nwblock)
#             wfirst = wlast + 1
#             # We are now on a new wafer.  Reset the wafer detector offset and
#             # also assign the reading to this process.
#             cur_wafer = local_det_wafers[wlast + 1]
#             wafer_off = 0
#             wafer_dets[cur_wafer] = list()
#             wafer_readers[cur_wafer] = proc
#             wafer_proc_dets[cur_wafer] = dict()
#         # Handle final block
#         wlast = len(local_det_wafers) - 1
#         nwblock = wlast - wfirst + 1
#         wafer_dets[cur_wafer].extend(local_det_names[wfirst : wlast + 1])
#         proc_wafer_dets[proc][cur_wafer] = (wfirst, wlast + 1)
#         wafer_proc_dets[cur_wafer][proc] = (wafer_off, wafer_off + nwblock)
#         wafer_off += nwblock

#     return (wafer_dets, wafer_readers, wafer_proc_dets, proc_wafer_dets)


# @function_timer
# def distribute_detector_data(
#     obs,
#     field,
#     axwafers,
#     axis_dets,
#     axis_samples,
#     axfield,
#     axdtype,
#     wafer_readers,
#     wafer_proc_dets,
#     proc_wafer_dets,
#     path_sep,
#     is_flag=False,
#     flag_invert=False,
#     flag_mask=None,
#     daq_units=False,
# ):
#     """Communicate detector data from the reading processes to the destinations.

#     Data for each wafer is loaded on exactly one process.  This function uses the
#     pre-computed data communication pattern built by `distribute_detector_props()`.
#     However, the loaded wafer data AxisManager may have been truncated with a
#     symmetric sample buffer on either side and some detectors may have been removed.
#     This function must account for this when copying data from the AxisManager into
#     The full-size observation.

#     The observation data is modified in-place.

#     Args:
#         obs (Observation):  The observation.
#         field (str):  The detdata field in the Observation.
#         axwafers (dict):  The dictionary of wafer data loaded on this process.
#         axis_dets (str):  The name of the detector LabelAxis in the AxisManagers.
#         axis_samples (str):  The name of the sample OffsetAxis in the AxisManagers.
#         axfield (str):  The name of the data field in the AxisManagers.
#         axdtype (np.dtype):  The dtype of the source buffer for sending the data
#             from the AxisManager.
#         wafer_readers (dict):  The reading process for each wafer.
#         wafer_proc_dets (dict):  For each wafer, the dictionary of destination
#             processes and the detector indices to send.
#         proc_wafer_dets (dict):  For each process, the dictionary of wafer name to
#             local detector indices to receive.
#         path_sep (str):  The path separation char when building flattened names.
#         is_flag (bool):  If True, this field is a flag.
#         flag_invert (bool):  If True, invert the meaning of the flag values.
#         flag_mask (np.uint8):  The flag mask (or None).
#         daq_units (bool):  If True, convert AxisManager signal to DAQ units.

#     Returns:
#         None

#     """
#     log = Logger.get()
#     gcomm = obs.comm.comm_group
#     rank = obs.comm.group_rank
#     gsize = obs.comm.group_size

#     rescale_to_daq = float(2**15) / np.pi

#     # If the blocks of detector data exceed 2^30 elements in total, they might hit
#     # MPI limitations on the communication message sizes.  Work around that here.
#     try:
#         from mpi4py.util import pkl5

#         if gcomm is not None:
#             gcomm = pkl5.Intracomm(gcomm)
#     except Exception:
#         pass

#     def _process_flag(mask, inbuf, detdata, det_begin, det_end, invert):
#         n_det = det_end - det_begin
#         temp = mask * np.ones_like(inbuf)
#         if invert:
#             temp[inbuf != 0] = 0
#         else:
#             temp[inbuf == 0] = 0
#         for idet in range(n_det):
#             detdata[idet + det_begin, :] |= temp[idet]

#     # Keep a handle to our send buffers to ensure that any temporary objects
#     # remain in existance until after all receives have happened.
#     send_data = dict()
#     send_req = list()
#     recv_data = None

#     # The tag value stride, to ensure unique tags for every sender / receiver / wafer
#     # combination.
#     tag_stride = 2000

#     wf_index = {y: x for x, y in enumerate(wafer_readers.keys())}

#     for wafer, reader in wafer_readers.items():
#         if reader == rank:
#             send_data[wafer] = dict()
#             # The wafer axis manager may have a different set of detectors than
#             # the full data, if some detectors have been cut.  It may also have
#             # a symmetric buffer of samples cut on either side of the observation.
#             # here we construct the mapping from full data into this restricted
#             # axis manager.
#             restricted_dets = axwafers[wafer][axis_dets].vals
#             restricted_indices = {y: x for x, y in enumerate(restricted_dets)}

#             # If the field is nested, descend
#             field_path = axfield.split(path_sep)
#             axroot = axwafers[wafer]
#             axobject = axroot[field_path[0]]
#             for ch in field_path[1:]:
#                 axobject = axobject[ch]

#             for receiver, send_dets in wafer_proc_dets[wafer].items():
#                 # "send_dets" is the un-restricted range of wafer detectors.
#                 # We will send a full-size (unrestricted) buffer to ease
#                 # bookkeeping, and just copy un-cut detector data into that
#                 # buffer.
#                 n_send_det = send_dets[1] - send_dets[0]

#                 # The names of the detectors on the receiving process.
#                 local_det_names = _local_process_dets(obs, receiver)

#                 # The receiving detector names and their relative indices
#                 recv_dets = proc_wafer_dets[receiver][wafer]
#                 n_recv_det = recv_dets[1] - recv_dets[0]
#                 recv_det_names = local_det_names[recv_dets[0] : recv_dets[1]]
#                 recv_det_indices = {y: x for x, y in enumerate(recv_det_names)}

#                 # Build the mapping of restricted indices to send buffer indices
#                 restrict_to_send = {
#                     restricted_indices[x]: y
#                     for x, y in recv_det_indices.items()
#                     if x in restricted_indices
#                 }

#                 # The per-detector flags, so that detectors are cut properly on the
#                 # receiver.
#                 det_flags = np.ones(
#                     n_send_det,
#                     dtype=np.uint8,
#                 )
#                 for idet_ax, idet_send in restrict_to_send.items():
#                     det_flags[idet_send] = 0

#                 # Does the axis manager have a truncated number of samples?
#                 restricted_samps = axwafers[wafer][axis_samples].count
#                 if restricted_samps != obs.n_local_samples:
#                     ax_shift = axwafers[wafer][axis_samples].offset
#                 else:
#                     ax_shift = 0

#                 # flat-packed buffer size of send buffer
#                 flat_size = n_send_det * obs.n_local_samples

#                 # Is this some detector flag data using ranges instead of samples?
#                 # If so, we construct a temporary buffer and build sample flags
#                 # from the ranges.
#                 if isinstance(axobject, so3g.proj.RangesMatrix):
#                     # Yes, flagged ranges.  We may have restricted sample ranges
#                     # for our flags, and so we initialize the full buffer to
#                     # the invalid mask.
#                     sdata = defaults.det_mask_invalid * np.ones(
#                         flat_size,
#                         dtype=np.uint8,
#                     )
#                     if flag_invert:
#                         for idet_ax, idet_send in restrict_to_send.items():
#                             # Set this detector's values to the mask, and then
#                             # we will "unflag" the specified ranges.
#                             off = idet_send * obs.n_local_samples + ax_shift
#                             sdata[off : off + restricted_samps] = flag_mask
#                             for rg in axobject[idet_ax].ranges():
#                                 sdata[off + rg[0] : off + rg[1]] = 0
#                     else:
#                         for idet_ax, idet_send in restrict_to_send.items():
#                             # Set this detector's values to good, and then
#                             # we will flag the specified ranges.
#                             off = idet_send * obs.n_local_samples + ax_shift
#                             sdata[off : off + restricted_samps] = 0
#                             for rg in axobject[idet_ax].ranges():
#                                 sdata[off + rg[0] : off + rg[1]] = flag_mask
#                 else:
#                     # Either normal sample flags or signal data
#                     if is_flag:
#                         # We may have restricted sample ranges for our flags, and so
#                         # we initialize the full buffer to the invalid mask.
#                         sdata = defaults.det_mask_invalid * np.ones(
#                             flat_size,
#                             dtype=axdtype,
#                         )
#                     else:
#                         # Signal data is initialized to zero.  Restricted / missing
#                         # samples will be indicated by flags.
#                         sdata = np.zeros(
#                             flat_size,
#                             dtype=axdtype,
#                         )
#                     for idet_ax, idet_send in restrict_to_send.items():
#                         off = idet_send * obs.n_local_samples + ax_shift
#                         sdata[off : off + restricted_samps] = axobject[idet_ax, :]

#                 if receiver == rank:
#                     # We just need to process the data locally
#                     sdata_2d = sdata.reshape((n_send_det, -1))
#                     if is_flag:
#                         _process_flag(
#                             flag_mask,
#                             sdata_2d,
#                             obs.detdata[field],
#                             recv_dets[0],
#                             recv_dets[1],
#                             flag_invert,
#                         )
#                     else:
#                         for idet in range(n_send_det):
#                             if daq_units:
#                                 nan_mask = np.isnan(sdata_2d[idet])
#                                 if np.count_nonzero(nan_mask) > 0:
#                                     det_flags[idet] = 1
#                                 np.rint(
#                                     rescale_to_daq * sdata_2d[idet],
#                                     out=obs.detdata[field][idet + recv_dets[0], :],
#                                     casting="unsafe",
#                                 )
#                             else:
#                                 obs.detdata[field][idet + recv_dets[0], :] = sdata_2d[
#                                     idet
#                                 ]
#                     # Update per-detector flags
#                     dflags = {
#                         obs.local_detectors[recv_dets[0] + x]: defaults.det_mask_invalid
#                         for x in range(n_recv_det)
#                         if det_flags[x] != 0
#                     }
#                     obs.update_local_detector_flags(dflags)
#                 else:
#                     # Send asynchronously.
#                     tag = (rank * gsize + receiver) * tag_stride + 2 * wf_index[wafer]
#                     flag_tag = tag + 1
#                     req = gcomm.isend(sdata, dest=receiver, tag=tag)
#                     req_flags = gcomm.isend(det_flags, dest=receiver, tag=flag_tag)
#                     # Save a handle to this buffer while send operation is in progress
#                     send_data[wafer][axfield] = sdata
#                     send_data[wafer][f"{axfield}_flags"] = det_flags
#                     send_req.append(req)
#                     send_req.append(req_flags)

#     my_wafer_dets = proc_wafer_dets[rank]
#     for wafer, recv_dets in my_wafer_dets.items():
#         n_recv_det = recv_dets[1] - recv_dets[0]
#         sender = wafer_readers[wafer]
#         if sender == rank:
#             # This data was already copied locally above
#             continue
#         else:
#             # Receive from sender.  We always allocate a contiguous temporary buffer.
#             tag = (sender * gsize + rank) * tag_stride + 2 * wf_index[wafer]
#             flag_tag = tag + 1
#             recv_data = gcomm.recv(source=sender, tag=tag)
#             recv_2d = recv_data.reshape((n_recv_det, -1))
#             det_flags = gcomm.recv(source=sender, tag=flag_tag)
#             if is_flag:
#                 _process_flag(
#                     flag_mask,
#                     recv_2d,
#                     obs.detdata[field],
#                     recv_dets[0],
#                     recv_dets[1],
#                     flag_invert,
#                 )
#             else:
#                 # Just assign
#                 for idet in range(n_recv_det):
#                     if daq_units:
#                         nan_mask = np.isnan(recv_2d[idet])
#                         if np.count_nonzero(nan_mask) > 0:
#                             det_flags[idet] = 1
#                         np.rint(
#                             rescale_to_daq * recv_2d[idet],
#                             out=obs.detdata[field][idet + recv_dets[0], :],
#                             casting="unsafe",
#                         )
#                     else:
#                         obs.detdata[field][idet + recv_dets[0], :] = recv_2d[idet]
#             # Update per-detector flags
#             dflags = {
#                 obs.local_detectors[recv_dets[0] + x]: defaults.det_mask_invalid
#                 for x in range(n_recv_det)
#                 if det_flags[x] != 0
#             }
#             obs.update_local_detector_flags(dflags)
#             del recv_2d
#             del recv_data
#             del det_flags

#     # Wait for communication
#     for req in send_req:
#         req.wait()
#     if gcomm is not None:
#         gcomm.barrier()

#     # Now safe to delete our dictionary of isend buffer handles, which might include
#     # temporary buffers of ranges flags.
#     del send_data

#     # Every process checks its local data for NaN values.  If any are found, a warning
#     # is printed and the detector is cut.  If we are using DAQ units, those detectors
#     # with NaNs were already cut during the float to int32 conversion above.
#     if not daq_units:
#         dflags = dict()
#         for det in obs.local_detectors:
#             nnan = np.count_nonzero(np.isnan(obs.detdata[field][det]))
#             if nnan > 0:
#                 msg = f"{obs.name}:{det} has {nnan} NaN values.  Cutting."
#                 log.warning(msg)
#                 dflags[det] = defaults.det_mask_invalid
#         obs.update_local_detector_flags(dflags)


# @function_timer
# def _create_observation(
#     self, obs_name, session_name, comm, telescope, n_samp, fp_flags, meta
# ):
#     """Create the observation.

#     Note that the focalplane table has already been sorted by wafer
#     name and then detector name.

#     Args:
#         obs_name (str):  The observation name
#         session_name (str):  The observing session name (context obs ID).
#         gcomm (MPI.Comm):  The group communicator or None.
#         telescope (Telescope):  The Telescope for this observation.
#         n_samp (int):  The number of samples in the observation.
#         fp_flags (dict):  The per-detector flags to apply.
#         meta (dict):  The observation metadata to populate.

#     Returns:
#         (tuple):  The (Observation, have_pointing), where the second
#             element is True if the observation has boresight pointing.

#     """


# def axman_to_observation(
#     aman,
#     telescope_name="UNKNOWN",
#     ax_times="timestamps",
#     ax_flags=[
#         ("smurfgaps_ufm_{det_info:wafer:array}", defaults.shared_mask_invalid),
#         ("acu_drops", defaults.shared_mask_invalid)
#     ],
#     ax_det_signal="signal",
#     ax_det_flags=[],
#     ax_boresight_az="boresight:az",
#     ax_boresight_el="boresight:el",
#     ax_boresight_roll="boresight:roll",
#     ax_boresight_angle="ancil:boresight_enc",
#     ax_hwp_angle="hwp_angle",
#     ax_corotator_angle="ancil:corotator_enc",
#     ax_pathsep=":",
#     ax_detinfo_wafer_key="stream_id",
#     axis_detector="dets",
#     axis_sample="samps",
#     detset_key="pixel",
#     times=defaults.times,
#     det_data_units=defaults.det_data_units,
#     shared_flags=defaults.shared_flags,
#     det_data=defaults.det_data,
#     det_flags=defaults.det_flags,
#     azimuth=defaults.azimuth,
#     elevation=defaults.elevation,
#     roll=defaults.roll,
#     boresight_azel=defaults.boresight_azel,
#     boresight_radec=defaults.boresight_radec,
#     corotator_angle="corotator_angle",
#     boresight_angle="boresight_angle",
#     hwp_angle=defaults.hwp_angle,
#     analytic_bandpass=False,
#     bandwidth=0.2,
#     daq_units=False,
#     comm=None
# ):
#     """Create an Observation from an AxisManager.

#     The AxisManager should exist on a single process.  The Observation may be distributed over a process group.  The AxisManager may have

#     """
#     log = Logger.get()
#         timer = Timer()
#         timer.start()

#         # Computer the detector sets
#         if self.detset_key is None:
#             detsets = None
#         else:
#             detsets = telescope.focalplane.detector_groups(self.detset_key)

#         # Note:  the session times will be updated later when reading timestamps
#         session = toast.instrument.Session(session_name)

#         # Create the observation
#         ob = toast.Observation(
#             comm,
#             telescope,
#             n_samp,
#             name=obs_name,
#             session=session,
#             detector_sets=detsets,
#             sample_sets=None,
#             process_rows=comm.group_size,
#         )

#         # Apply detector flags for bad pointing reconstruction
#         local_dets = set(ob.local_detectors)
#         local_fp_flags = {x: y for x, y in fp_flags.items() if x in local_dets}
#         ob.update_local_detector_flags(local_fp_flags)

#         # Create observation fields
#         ob.shared.create_column(
#             self.times,
#             shape=(ob.n_local_samples,),
#             dtype=np.float64,
#         )

#         fp_array = telescope.focalplane.detector_data
#         ax_boresight_az = ax_name_fp_subst(self.ax_boresight_az, fp_array)
#         ax_boresight_el = ax_name_fp_subst(self.ax_boresight_el, fp_array)
#         ax_boresight_roll = ax_name_fp_subst(self.ax_boresight_roll, fp_array)
#         ax_det_signal = ax_name_fp_subst(self.ax_det_signal, fp_array)

#         have_pointing = True
#         if ax_boresight_az is None:
#             have_pointing = False
#         else:
#             ob.shared.create_column(
#                 self.azimuth,
#                 shape=(ob.n_local_samples,),
#                 dtype=np.float64,
#             )
#         if ax_boresight_el is None:
#             have_pointing = False
#         else:
#             ob.shared.create_column(
#                 self.elevation,
#                 shape=(ob.n_local_samples,),
#                 dtype=np.float64,
#             )
#         if ax_boresight_roll is None:
#             have_pointing = False
#         else:
#             ob.shared.create_column(
#                 self.roll,
#                 shape=(ob.n_local_samples,),
#                 dtype=np.float64,
#             )

#         if have_pointing:
#             ob.shared.create_column(
#                 self.boresight_azel,
#                 shape=(ob.n_local_samples, 4),
#                 dtype=np.float64,
#             )
#             ob.shared.create_column(
#                 self.boresight_radec,
#                 shape=(ob.n_local_samples, 4),
#                 dtype=np.float64,
#             )
#         ob.shared.create_column(
#             self.shared_flags,
#             shape=(ob.n_local_samples,),
#             dtype=np.uint8,
#         )
#         if have_pointing:
#             ob.shared.create_column(
#                 defaults.position,
#                 shape=(ob.n_local_samples, 3),
#                 dtype=np.float64,
#             )
#             ob.shared.create_column(
#                 defaults.velocity,
#                 shape=(ob.n_local_samples, 3),
#                 dtype=np.float64,
#             )
#         if ax_det_signal is not None:
#             if self.daq_units:
#                 ob.detdata.create(
#                     self.det_data, dtype=np.int32, units=u.dimensionless_unscaled
#                 )
#             else:
#                 ob.detdata.create(
#                     self.det_data, dtype=np.float64, units=self.det_data_units
#                 )
#             ob.detdata.create(self.det_flags, dtype=np.uint8)

#         if meta is not None:
#             ob.update(meta)

#         log.debug_rank(
#             f"LoadContext {obs_name} allocate Observation in",
#             comm=comm.comm_group,
#             timer=timer,
#         )
#         return ob, have_pointing


#         # One process in the group loads the metadata, builds the focalplane
#         # model, and broadcasts to the rest of the group.
#         #
#         # FIXME: in the case of multiple wafers per observation, each wafer
#         # reader needs to do this step, in order to pull in unique metadata
#         # that is per-wafer and merge.
#         #
#         obs_meta, det_props = self._load_metadata(
#             obs_name,
#             obs_props[obindx]["session_name"],
#             comm.comm_group,
#             obs_dets_select,
#             preproc_conf,
#         )

#         # Create the instrument model for the observation
#         telescope, fp_flags = self._create_obs_instrument(
#             obs_name,
#             comm.comm_group,
#             det_props,
#         )

#         # Create the observation and allocate data objects
#         ob, have_pointing = self._create_observation(
#             obs_name,
#             obs_props[obindx]["session_name"],
#             comm,
#             telescope,
#             n_samp,
#             fp_flags,
#             obs_meta,
#         )

#         # Read and communicate data
#         self._load_data(ob, have_pointing, preproc_conf)

#         # Report number of cut detectors
#         local_cut = np.count_nonzero(
#             [y for x, y in ob.local_detector_flags.items()]
#         )
#         if ob.comm.comm_group is None:
#             all_cut = local_cut
#         else:
#             all_cut = ob.comm.comm_group.gather(local_cut, root=0)
#             if ob.comm.group_rank == 0:
#                 tot_cut = np.sum(all_cut)
#                 msg = f"LoadContext {ob.name} {tot_cut} / "
#                 msg += f"{len(ob.all_detectors)} dets trimmed on read"
#                 log.debug(msg)

#         # Now that all metadata has been loaded, ensure that all byte strings
#         # are converted to unicode arrays.
#         ob._internal = replace_byte_arrays(ob._internal)

#         # Optionally load housekeeping data
#         if self.hk_site_root is not None or self.hk_platform_root is not None:
#             hk_timer = Timer()
#             hk_timer.start()
#             ob.hk = HKManager(
#                 ob.comm.comm_group,
#                 ob.shared[self.times].data,
#                 site_root=self.hk_site_root,
#                 site_db=self.hk_site_db,
#                 site_fields=self.hk_site_fields,
#                 site_aliases=self.hk_site_aliases,
#                 plat_root=self.hk_platform_root,
#                 plat_db=self.hk_platform_db,
#                 plat_fields=self.hk_platform_fields,
#                 plat_aliases=self.hk_platform_aliases,
#             )
#             log.debug_rank(
#                 f"LoadContext {obs_name} load House Keeping in",
#                 comm=comm.comm_group,
#                 timer=hk_timer,
#             )

#         # Compute the boresight pointing and observatory position
#         if have_pointing:
#             compute_boresight_pointing(
#                 ob,
#                 self.times,
#                 self.azimuth,
#                 self.elevation,
#                 self.roll,
#                 self.boresight_azel,
#                 self.boresight_radec,
#                 defaults.position,
#                 defaults.velocity,
#                 self.shared_flags,
#                 defaults.shared_mask_processing,
#             )
#         data.obs.append(ob)
#         log.info_rank(
#             f"LoadContext {obs_name} loaded in",
#             comm=comm.comm_group,
#             timer=otimer,
#         )

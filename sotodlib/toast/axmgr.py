# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for interfacing with AxisManagers."""

import re

import numpy as np
from astropy import units as u
from astropy.table import Column, QTable

import toast
from toast.mpi import MPI
from toast.timing import function_timer, Timer
from toast.utils import Logger
from toast.observation import default_values as defaults

from ..core import AxisManager
from ..core.axisman import AxisInterface
from .instrument import SOSite


class AxNames(object):
    """Class representing naming conventions to use with AxisManagers.

    Args:
        override (dict):  Dictionary of overrides.

    """
    def __init__(self, override={}):
        self._ax_names = {
            "pathsep": ":",
            "times": "timestamps",
            "det_data": "signal",
            "azimuth": "boresight:az",
            "elevation": "boresight:el",
            "roll": "boresight:roll",
            "hwp_angle": "hwp_angle",
            "wafer_key": "stream_id",
            "axis_dets": "dets",
            "axis_samps": "samps",
        }
        self._ax_names.update(override)

    def __getitem__(self, key):
        if key in self._ax_names:
            return self._ax_names[key]
        else:
            return None


def ax_name_fp_subst(var, fp_array, det=None):
    """Substitute a focalplane column value into a string.

    If det is not specified, then there should be a unique name for the whole table
    column when substituting.

    Args:
        var (str):  The input variable string
        fp_array (Table):  The focalplane astropy Table of detector properties.
        det (str):  If not None, use the value for this row of the table.

    Returns:
        (str):  The string with substitutions made.

    """
    if var is None:
        return None
    out = ""
    last = 0
    for match in re.finditer(r"(\{.*\})", var):
        out += var[last : match.start()]
        colname = match.group()
        colname = colname.replace("{", "")
        colname = colname.replace("}", "")
        if colname not in fp_array.keys():
            msg = f"Column '{colname}' not in focalplane table"
            raise RuntimeError(msg)
        if det is None:
            vals = np.unique(fp_array[colname])
            if len(vals) == 1:
                out += f"{vals[0]}"
            else:
                msg = f"Column '{colname}' has multiple values ({vals}), cannot"
                msg += " use in name"
                raise RuntimeError(msg)
        else:
            mask = fp_array[colname] == det
            out += f"{fp_array[colname][mask][0]}"
        last = match.end()
    out += var[last:]
    return out


def parse_axmgr(
    axman,
    obs_meta,
    fp_cols,
    ax_names,
    obs_base=None,
    fp_base=None,
):
    """Parse AxisManager metadata.

    This is a recursive function which descends the AxisManager hierarchy and splits
    meta data into "detector properties" appropriate for a TOAST Focalplane and "other
    metadata" which will be stored in the Observation dictionary.

    For each element of the input AxisManager we do the following:

    - If the object has one axis and it is the detector axis,
      treat it as a column in the detector property table
    - If the object has one axis and it is the sample axis,
      it will be loaded later as shared data.
    - If the object has multiple axes, load it later
    - If the object has no axes, then treat it as observation
      metadata
    - If the object is a nested AxisManager, descend and apply
      the same steps as above.

    Args:
        axman (AxisManager):  The input axismanager.
        obs_meta (dict):  The working dictionary of observation metadata.
        fp_cols (dict):  Dictionary of table Columns of detector properties.
        ax_names (AxNames):  The AxisManager naming conventions to use.
        obs_base (str):  The current key in the nested observation metadata.
        fp_base (str):  The current key to use when appending focalplane columns.

    Returns:
        None

    """
    path_sep = ax_names["path_sep"]
    det_axis = ax_names["axis_dets"]
    if obs_base is None:
        om = obs_meta
    else:
        obs_meta[obs_base] = dict()
        om = obs_meta[obs_base]
    for key in axman.keys():
        if fp_base is not None:
            fp_key = f"{fp_base}{path_sep}{key}"
        else:
            fp_key = key
        if isinstance(axman[key], AxisInterface):
            # This is one of the axes
            continue
        if isinstance(axman[key], AxisManager):
            # Descend
            parse_axmgr(
                axman[key],
                obs_meta,
                fp_cols,
                ax_names,
                obs_base=key,
                fp_base=fp_key,
            )
        else:
            # FIXME:  It would be nicer if this information was available
            # through a public member...
            field_axes = axman._assignments[key]
            if len(field_axes) == 0:
                # This data is not associated with an axis.
                om[key] = axman[key]
            elif len(field_axes) == 1 and field_axes[0] == det_axis:
                # This is a detector property
                if fp_key in fp_cols:
                    msg = f"Metadata key '{fp_key}' is duplicated in nested"
                    msg += " AxisManagers"
                    raise RuntimeError(msg)
                fp_cols[fp_key] = Column(name=fp_key, data=np.array(axman[key]))
    if obs_base is not None and len(om) == 0:
        # There were no meta data keys- delete this dict
        del obs_meta[obs_base]


def obs_properties(
    axmgr,
    ax_names,
    analytic_bandwidth=None,
):
    """Parse information needed to construct an Observation.

    Args:
        axmgr
        ax_names (AxNames):  The AxisManager naming conventions to use.
        analytic_bandwith (float):  The fractional...

    """

    ax_pathsep=ax_names["pathsep"],
    axis_detector=ax_names["axis_dets"],
    ax_wafer_key=ax_names["wafer_key"],

    n_samp = axmgr["samps"].count
    obs_meta = dict()
    fp_cols = dict()

    # Parse the axis manager metadata into observation metadata
    # and detector properties.
    parse_axmgr(
        axmgr,
        obs_meta,
        fp_cols,
        ax_pathsep,
        axis_detector,
        None,
        None,
    )

    # Set the column used by toast as the detector name.
    fp_cols["name"] = Column(
        name="name",
        data=fp_cols[f"det_info{ax_pathsep}readout_id"].data,
    )

    if analytic_bandwidth is not None:
        # Add bandpass information to the focalplane
        try:
            band = fp_cols[f"det_info{ax_pathsep}band"].data
        except KeyError:
            band = fp_cols[f"det_info{ax_pathsep}wafer{ax_pathsep}bandpass"].data
        freq = [float(b[1:]) for b in band]
        bandcenter = np.array(freq) * u.GHz
        bandwidth = bandcenter * analytic_bandwidth
        fp_cols["bandcenter"] = Column(name="bandcenter", data=bandcenter)
        fp_cols["bandwidth"] = Column(name="bandwidth", data=bandwidth)

    # Construct detector properties table
    det_props = QTable(fp_cols)

    # Detector ordering.  When distributing and reading data, we
    # need to have the detectors sorted into contiguous blocks of
    # of detectors.  We sort the table now by wafer and then by name.
    wafer_key = f"det_info{ax_pathsep}{ax_wafer_key}"
    det_props.sort([wafer_key, "name"])

    return (obs_meta, det_props, n_samp)


def compare_obs_properties(comm, local_props):
    pass


@function_timer
def create_obs_instrument(
    telescope_name, obs_name, gcomm, det_props, ax_names
):
    """Create telescope, session, and focalplane flags.

    These are objects needed prior to instantiating the Observation.

    Args:
        telescope_name (str):  The name of the telescope.
        obs_name (str):  The observation name
        gcomm (MPI.Comm):  The group communicator or None.
        det_props (QTable):  The astropy table of detector properties.
        ax_names (AxNames):  The AxisManager naming conventions to use.

    Returns:
        (tuple):  The (Telescope, per-detector flags) where the flags
            cut detectors with the processing bit if they have no
            pointing reconstruction.

    """
    log = Logger.get()

    ax_pathsep=ax_names["pathsep"]

    # Convert any focalplane quaternion offsets to toast format.  We also
    # look for any detectors with NaN values and flag those with the
    # processing bit.

    xi_key = f"focal_plane{ax_pathsep}xi"
    eta_key = f"focal_plane{ax_pathsep}eta"
    gamma_key = f"focal_plane{ax_pathsep}gamma"
    wafer_type_key = f"det_info{ax_pathsep}wafer{ax_pathsep}type"

    fp_flags = dict()
    if xi_key in det_props.colnames:
        fp_bad = np.logical_or(
            np.isnan(det_props[xi_key]),
            np.logical_or(
                np.isnan(det_props[eta_key]),
                np.isnan(det_props[gamma_key]),
            ),
        )
        fp_flags = {
            x: defaults.det_mask_processing
            for x, y in zip(det_props["name"], fp_bad)
            if y
        }
        det_props[xi_key][fp_bad] = 0
        det_props[eta_key][fp_bad] = 0
        det_props[gamma_key][fp_bad] = 0
        quat_data = toast.instrument_coords.xieta_to_quat(
            det_props[xi_key],
            det_props[eta_key],
            det_props[gamma_key],
        )
    else:
        # No detector offsets yet
        quat_data = np.tile(
            np.array([0, 0, 0, 1], dtype=np.float64),
            len(det_props[f"det_info{ax_pathsep}readout_id"]),
        ).reshape((-1, 4))

    # Flag non-optical detectors with the processing mask
    if wafer_type_key in det_props.colnames:
        fp_dark = det_props[wafer_type_key] != "OPTC"
        for dname, dark in zip(det_props["name"], fp_dark):
            if dark:
                fp_flags[dname] |= defaults.det_mask_processing

    # Do we have any good detectors left?
    n_good = 0
    for det in det_props["name"]:
        if det not in fp_flags or fp_flags[det] == 0:
            n_good += 1
    if n_good == 0:
        log.warning_rank(
            f"create_obs_instrument: {obs_name} has no unflagged detectors, skipping!",
            comm=gcomm,
        )
        return (None, None)

    # Add a column for the quaternion offset of each detector, as well
    # as the gamma angle with standard naming.
    quat_col = Column(
        name="quat",
        data=quat_data,
    )
    det_props.add_column(quat_col, index=0)
    gamma_col = Column(
        name="gamma",
        data=det_props[gamma_key] * u.radian,
    )
    det_props.add_column(gamma_col)

    focalplane = toast.instrument.Focalplane(
        detector_data=det_props,
        sample_rate=1.0 * u.Hz,
    )

    # For now, this should be good enough position for instruments near the
    # S.O. location.
    site = SOSite()

    telescope = toast.instrument.Telescope(
        telescope_name, focalplane=focalplane, site=site
    )
    return telescope, fp_flags


@function_timer
def create_observation(
    obs_name,
    session_name,
    comm,
    telescope,
    n_samp,
    fp_flags,
    meta,
    ax_names,
    azimuth=defaults.azimuth,
    elevation=defaults.elevation,
    boresight_azel=defaults.boresight_azel,
    boresight_radec=defaults.boresight_radec,
    shared_flags=defaults.shared_flags,
    det_data=defaults.det_data,
    det_flags=defaults.det_flags,
    det_data_units=defaults.det_data_units,
    hwp_angle=None,
    boresight_angle=None,
    corotator_angle=None,
    detset_key=None,
):
    """Create the observation.

    Note that the focalplane table has already been sorted by wafer
    name and then detector name.

    Args:
        obs_name (str):  The observation name
        session_name (str):  The observing session name (context obs ID).
        comm (toast.Comm):  The toast communicator.
        telescope (Telescope):  The Telescope for this observation.
        n_samp (int):  The number of samples in the observation.
        fp_flags (dict):  The per-detector flags to apply.
        meta (dict):  The observation metadata to populate.
        ax_names (AxNames):  The AxisManager naming conventions to use.

        azimuth=defaults.azimuth,
        elevation=defaults.elevation,
        boresight_azel=defaults.boresight_azel,
        boresight_radec=defaults.boresight_radec,
        shared_flags=defaults.shared_flags,
        det_data=defaults.det_data,
        det_flags=defaults.det_flags,
        det_data_units=defaults.det_data_units,
        hwp_angle=None,
        boresight_angle=None,
        corotator_angle=None,
        detset_key=None,

    Returns:
        (tuple):  The (Observation, have_pointing), where the second
            element is True if the observation has boresight pointing.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    times=ax_names["times"]
    ax_boresight_az=ax_names["azimuth"]
    ax_boresight_el=ax_names["elevation"]
    ax_boresight_roll=ax_names["roll"]
    ax_hwp_angle=ax_names["hwp_angle"]
    ax_det_data=ax_names["det_data"]
    roll=ax_names["roll"]

    # Computer the detector sets
    if detset_key is None:
        detsets = None
    else:
        detsets = telescope.focalplane.detector_groups(detset_key)

    # Note:  the session times will be updated later when reading timestamps
    session = toast.instrument.Session(session_name)

    # Create the observation
    ob = toast.Observation(
        comm,
        telescope,
        n_samp,
        name=obs_name,
        session=session,
        detector_sets=detsets,
        sample_sets=None,
        process_rows=comm.group_size,
    )

    # Apply detector flags for bad pointing reconstruction
    local_dets = set(ob.local_detectors)
    local_fp_flags = {x: y for x, y in fp_flags.items() if x in local_dets}
    ob.update_local_detector_flags(local_fp_flags)

    # Create observation fields
    ob.shared.create_column(
        times,
        shape=(ob.n_local_samples,),
        dtype=np.float64,
    )

    fp_array = telescope.focalplane.detector_data
    ax_boresight_az = ax_name_fp_subst(ax_boresight_az, fp_array)
    ax_boresight_el = ax_name_fp_subst(ax_boresight_el, fp_array)
    ax_boresight_roll = ax_name_fp_subst(ax_boresight_roll, fp_array)
    ax_hwp_angle = ax_name_fp_subst(ax_hwp_angle, fp_array)
    ax_det_signal = ax_name_fp_subst(ax_det_data, fp_array)

    have_pointing = True
    if ax_boresight_az is None:
        have_pointing = False
    else:
        ob.shared.create_column(
            azimuth,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if ax_boresight_el is None:
        have_pointing = False
    else:
        ob.shared.create_column(
            elevation,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if ax_boresight_roll is None:
        have_pointing = False
    else:
        ob.shared.create_column(
            roll,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )

    if have_pointing:
        ob.shared.create_column(
            boresight_azel,
            shape=(ob.n_local_samples, 4),
            dtype=np.float64,
        )
        ob.shared.create_column(
            boresight_radec,
            shape=(ob.n_local_samples, 4),
            dtype=np.float64,
        )
    if hwp_angle is not None and ax_hwp_angle is not None:
        ob.shared.create_column(
            hwp_angle,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if boresight_angle is not None:
        ob.shared.create_column(
            boresight_angle,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if corotator_angle is not None:
        ob.shared.create_column(
            corotator_angle,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    ob.shared.create_column(
        shared_flags,
        shape=(ob.n_local_samples,),
        dtype=np.uint8,
    )
    if have_pointing:
        ob.shared.create_column(
            defaults.position,
            shape=(ob.n_local_samples, 3),
            dtype=np.float64,
        )
        ob.shared.create_column(
            defaults.velocity,
            shape=(ob.n_local_samples, 3),
            dtype=np.float64,
        )
    if ax_det_signal is not None:
        ob.detdata.create(det_data, dtype=np.float64, units=det_data_units)
        ob.detdata.create(det_flags, dtype=np.uint8)

    if meta is not None:
        ob.update(meta)

    log.debug_rank(
        f"create_observation: {obs_name} allocate Observation in",
        comm=comm.comm_group,
        timer=timer,
    )
    return ob, have_pointing


def obs_to_axmgr():
    """Convert local Observation data into an AxisManager on each process."""

    # each process builds axman hierarchy for local detectors

    # wrap signal and other pieces

    pass


def axmgr_to_obs(axmgr):
    """Convert AxisManager data on each process into an Observation."""

    # parse_meta from either full or partial axis manager to get telescope / sample info

    obs_properties(
        axmgr,
        ax_pathsep=axmgr_defaults["pathsep"],
        axis_detector=axmgr_defaults["axis_dets"],
        axis_samps=axmgr_defaults["axis_samps"],
        ax_wafer_key=axmgr_defaults["wafer_key"],
        analytic_bandwidth=None,
    )

    # verify metadata consistency across processes in the group

    # Create empty observation or overwrite existing
    #   - what if we are wrapping the original input observation?

    pass


def scatter_axmgr(
    axmgr,
    obs_name,
    session_name,
    telescope_name,
    comm,
    ax_names,
    analytic_bandwidth=None,
    azimuth=defaults.azimuth,
    elevation=defaults.elevation,
    boresight_azel=defaults.boresight_azel,
    boresight_radec=defaults.boresight_radec,
    shared_flags=defaults.shared_flags,
    det_data=defaults.det_data,
    det_flags=defaults.det_flags,
    det_data_units=defaults.det_data_units,
    hwp_angle=None,
    boresight_angle=None,
    corotator_angle=None,
    detset_key=None,
):
    """Convert an AxisManager on one process into a distributed Observation.



    """

    (obs_meta, det_props, n_samp) = obs_properties(
        axmgr,
        ax_names=ax_names,
        analytic_bandwidth=analytic_bandwidth,
    )

    telescope, fp_flags = create_obs_instrument(
        telescope_name, obs_name, comm.comm_group, det_props, ax_names=ax_names
    )

    obs, have_pointing = create_observation(
        obs_name,
        session_name,
        comm,
        telescope,
        n_samp,
        fp_flags,
        obs_meta,
        ax_names,
        azimuth=azimuth,
        elevation=elevation,
        boresight_azel=boresight_azel,
        boresight_radec=boresight_radec,
        shared_flags=shared_flags,
        det_data=det_data,
        det_flags=det_flags,
        det_data_units=det_data_units,
        hwp_angle=hwp_angle,
        boresight_angle=boresight_angle,
        corotator_angle=corotator_angle,
        detset_key=detset_key,
    )

    # Read and communicate data
    self._load_data(ob, have_pointing, preproc_conf)

    # Compute the boresight pointing and observatory position
    if have_pointing:
        compute_boresight_pointing(
            ob,
            self.times,
            self.azimuth,
            self.elevation,
            self.roll,
            self.boresight_azel,
            self.boresight_radec,
            defaults.position,
            defaults.velocity,
            self.shared_flags,
            defaults.shared_mask_processing,
        )
    data.obs.append(ob)
    log.info_rank(
        f"LoadContext {obs_name} loaded in",
        comm=comm.comm_group,
        timer=otimer,
    )


def gather_axmgr():
    """Collect a distributed Observation into an AxisManager on one process."""

    # rank zero builds high-level axman metadata and hierarchy

    # communicate per-process signal pieces

    pass

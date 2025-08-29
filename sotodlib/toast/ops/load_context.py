# Copyright (c) 2022-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import re
import sqlite3
import yaml
from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from astropy.table import Column, QTable

import toast
from toast.timing import function_timer, Timer
from toast.traits import (
    trait_docs,
    Int,
    Unicode,
    Instance,
    List,
    Unit,
    Dict,
    Bool,
    Float,
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.dist import distribute_discrete
from toast.observation import default_values as defaults

import so3g

from ...core import Context, AxisManager, FlagManager
from ...core.axisman import AxisInterface

from ..instrument import SOSite

from .load_context_utils import (
    compute_boresight_pointing,
    parse_metadata,
    read_and_preprocess_wafers,
    open_context,
    distribute_detector_data,
    distribute_detector_props,
    ax_name_fp_subst,
)


@trait_docs
class LoadContext(Operator):
    """Load one or more observations from a Context into observing sessions.

    Given a context, load one or more observations.  If specified, the context
    should exist on all processes in the group.  The `readout_ids`, `detsets`,
    and `bands` traits are used to select subsets of detectors.  Alternatively
    the context file can be specified, in which case this is passed to the
    context constructor.

    The traits starting with an "ax_*" prefix indicate the axis manager names
    that should be mapped to the standard toast shared and detdata objects.
    For `ax_flags` tuples, a negative bit value indicates that the flag data
    should first be inverted before combined.

    For all of the "ax_*" traits, the names may additionally include substitution
    strings enclosed in '{}' that contain the name of a focalplane detector_data
    key.  For example, the shared smurfgap flags are named after the UFM (array
    name).  So you can import the correct field name for all observations with:

        ax_flags=[
            (
                "smurfgaps_ufm_{det_info:wafer:array}",
                defaults.shared_mask_invalid
            ),
        ]

    Additional nested axismanager data is grouped into detdata, shared, or
    metadata objects based on whether they use detector / sample axes.  The
    nested structure is flattened into names built from the keys in the
    hierarchy.

    Important considerations for data distribution:

    - Each "observation" in the Context system maps to a toast "observing
      session".  By default, a Context observation is turned into a toast
      observing session with one observation per wafer.  If `combine_wafers`
      is True, then all wafers will be combined into a single toast
      observation.

    - In the default case of one observation per wafer, recall that only
      one process in a group loads data from disk.  For jobs that are I/O
      bound and loading multiple wafers, you should use multiple process
      groups to parallelize the data loading (which is the dominant cost).

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    context = Instance(
        klass=Context,
        allow_none=True,
        help="The Context, which should exist on all processes",
    )

    context_file = Unicode(
        None,
        allow_none=True,
        help="Create a context from this file",
    )

    observation_regex = Unicode(
        None,
        allow_none=True,
        help="Regular expression match to apply to observation IDs",
    )

    observation_file = Unicode(
        None,
        allow_none=True,
        help="Text file containing observation IDs to load",
    )

    preprocess_config = Unicode(
        None,
        allow_none=True,
        help="Apply pre-processing with this configuration",
    )

    observations = List(list(), help="List of observation IDs to load")

    readout_ids = List(list(), help="Only load this list of readout_id values")

    stream_ids = List(list(), help="Only load this list of stream_id values")

    detsets = List(list(), help="Only load this list of detset values")

    bands = List(list(), help="Only load this list of band values")

    dets_select = Dict(
        dict(), help="The `dets` selection dictionary to pass to get_obs()"
    )

    combine_wafers = Bool(
        False,
        help="If True, combine all wafers into a single observation",
    )

    ax_times = Unicode(
        "timestamps",
        help="Name of field to associate with times",
    )

    ax_flags = List(
        [],
        help="Tuples of (field, bit value) merged to shared_flags",
    )

    ax_det_signal = Unicode(
        "signal",
        allow_none=True,
        help="Name of field to associate with det_data",
    )

    ax_det_flags = List(
        [],
        help="Tuples of (field, bit_value) merged to det_flags",
    )

    ax_boresight_az = Unicode(
        "boresight:az",
        allow_none=True,
        help="Field with boresight Az",
    )

    ax_boresight_el = Unicode(
        "boresight:el",
        allow_none=True,
        help="Field with boresight El",
    )

    ax_boresight_roll = Unicode(
        "boresight:roll",
        allow_none=True,
        help="Field with boresight Roll",
    )

    ax_hwp_angle = Unicode(
        "hwp_angle",
        allow_none=True,
        help="Field with HWP angle",
    )

    ax_pathsep = Unicode(
        ":",
        help="Path separator when flattening nested fields",
    )

    ax_detinfo_wafer_key = Unicode(
        "stream_id",
        help="Name of the det_info property containing the wafer ID",
    )

    axis_detector = Unicode(
        "dets", help="Name of the LabelAxis for the detector direction"
    )

    axis_sample = Unicode(
        "samps", help="Name of the OffsetAxis for the sample direction"
    )

    telescope_name = Unicode("UNKNOWN", help="Name of the telescope")

    detset_key = Unicode(
        None,
        allow_none=True,
        help="Column of the focalplane detector_data to use for data distribution",
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key for detector signal",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for detector flags",
    )

    azimuth = Unicode(
        defaults.azimuth, help="Observation shared key for boresight Azimuth"
    )

    elevation = Unicode(
        defaults.elevation, help="Observation shared key for boresight Elevation"
    )

    roll = Unicode(defaults.roll, help="Observation shared key for boresight Roll")

    boresight_azel = Unicode(
        defaults.boresight_azel,
        help="Observation shared key for boresight Az/El quaternions",
    )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        help="Observation shared key for boresight RA/DEC quaternions",
    )

    corotator_angle = Unicode(
        None,
        allow_none=True,
        help="Observation shared key for corotator_angle (if it is used)",
    )

    boresight_angle = Unicode(
        None,
        allow_none=True,
        help="Observation shared key for boresight rotation angle (if it is used)",
    )

    hwp_angle = Unicode(
        None,
        allow_none=True,
        help="Observation shared key for HWP angle (if it is used)",
    )

    analytic_bandpass = Bool(False, help="Add analytic bandpass to each detector")

    bandwidth = Float(0.2, help="Fractional bandwith used in analytic bandpass")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        comm = data.comm

        if self.combine_wafers:
            msg = "Combining multiple wafers in a single observation is"
            msg += " currently broken"
            raise NotImplementedError(msg)

        if self.context is None:
            if self.context_file is None:
                msg = "Either the context or context_file must be specified"
                raise RuntimeError(msg)
        else:
            if self.context_file is not None:
                msg = "Only one of the context or context_file should be specified"
                raise RuntimeError(msg)

        # Build our detector selection dictionary.  Merge our explicit traits
        # with any pre-existing detector selection.
        dets_select = None
        if len(self.dets_select) > 0:
            # Use the full dictionary provided by the user
            dets_select = self.dets_select
        elif (
            len(self.readout_ids) > 0
            or len(self.bands) > 0
            or len(self.detsets) > 0
            or len(self.stream_ids) > 0
        ):
            # We have some selection, build the dictionary
            dets_select = dict()
            if len(self.readout_ids) > 0:
                dets_select["readout_id"] = list(self.readout_ids)
            if len(self.bands) > 0:
                dets_select["band"] = list(self.bands)
            if len(self.detsets) > 0:
                dets_select["detset"] = list(self.detsets)
            if len(self.stream_ids) > 0:
                dets_select["stream_id"] = list(self.stream_ids)

        # Get any selection of wafers
        keep_wafers = None
        if dets_select is not None and "stream_id" in dets_select:
            if isinstance(dets_select["stream_id"], (list, tuple, set)):
                keep_wafers = set(dets_select["stream_id"])
            else:
                # Scalar
                keep_wafers = set()
                keep_wafers.add(dets_select["stream_id"])

        # One global process queries the observation metadata and computes
        # the observation distribution among process groups.
        obs_check = (
            (self.observation_regex is not None)
            + (len(self.observations) > 0)
            + (self.observation_file is not None)
        )
        if obs_check != 1:
            msg = "Exactly one of observation_regex, observation_file"
            msg += " and observations should be specified"
            raise RuntimeError(msg)
        if self.observation_file is not None:
            olist = None
            if comm.world_rank == 0:
                olist = list()
                with open(self.observation_file, "r") as f:
                    for line in f:
                        if re.match(r"^#.*", line) is None:
                            olist.append(line.strip())
            if comm.comm_world is not None:
                olist = comm.comm_world.bcast(olist, root=0)
            self._observations = olist
        else:
            self._observations = self.observations

        obs_props = None
        preproc_conf = None
        if comm.world_rank == 0:
            obs_list = list(sorted(self._observations))
            if self.preprocess_config is not None:
                with open(self.preprocess_config, "r") as f:
                    preproc_conf = yaml.safe_load(f)

            # Query the obsdb directly so that we only have one DB query,
            # rather than doing one query per observation.
            sort_by = ["obs_id"]
            if len(obs_list) > 0:
                query_str = "obs.obs_id IN ("
                for oid in obs_list:
                    query_str += f"'{oid}',"
                query_str = query_str.rstrip(",")
                query_str += ")"
            else:
                query_str = "1"

            # If we are using preprocessing, load the archive DB and cut any
            # observations or wafer slots that do not exist.
            preproc_lookup = None
            if preproc_conf is not None:
                preproc_lookup = dict()
                preproc_db = preproc_conf["archive"]["index"]
                con = sqlite3.connect(preproc_db)
                cur = con.cursor()
                res = cur.execute('select "obs:obs_id","dets:wafer_slot" from map')
                for row in res.fetchall():
                    obid = row[0]
                    wslot = row[1]
                    if obid not in preproc_lookup:
                        preproc_lookup[obid] = set()
                    preproc_lookup[obid].add(wslot)
                del res
                del cur
                con.close()
                del con

            # Open the databases and query
            ctx = open_context(context=self.context, context_file=self.context_file)
            query_result = ctx.obsdb.query(query_text=query_str, sort=sort_by)

            # Build up the list of observing session properties, and optionally
            # only keep observations that match a regex.
            pat = None
            if self.observation_regex is not None:
                pat = re.compile(self.observation_regex)
            session_props = list()
            for row in query_result:
                obs_id = str(row["obs_id"])
                if pat is not None and pat.match(obs_id) is None:
                    continue
                # If we are using preprocessing, only keep obs_ids and wafer slots that
                # exist in the preprocessing archive.
                raw_wafer_slots = list(row["wafer_slots_list"].split(","))
                raw_stream_ids = list(row["stream_ids_list"].split(","))
                if preproc_lookup is not None:
                    if obs_id not in preproc_lookup:
                        msg = f"Requested obs_id {obs_id} does not exist in "
                        msg += f"preprocessing archive {preproc_db}.  Skipping."
                        log.warning(msg)
                        continue
                    wafer_slots = list()
                    stream_ids = list()
                    for ws, sid in zip(raw_wafer_slots, raw_stream_ids):
                        if ws not in preproc_lookup[obs_id]:
                            msg = f"Wafer slot {obs_id}:{ws} not in preprocessing"
                            msg += f" archive {preproc_db}.  Skipping."
                            log.warning(msg)
                        else:
                            wafer_slots.append(ws)
                            stream_ids.append(sid)
                else:
                    wafer_slots = raw_wafer_slots
                    stream_ids = raw_stream_ids
                sprops = dict()
                sprops["session_name"] = obs_id
                sprops["session_start"] = float(row["start_time"])
                sprops["session_stop"] = float(row["stop_time"])
                sprops["n_samples"] = int(row["n_samples"])
                sprops["tele_name"] = str(row["telescope"])
                sprops["n_wafers"] = int(row["wafer_count"])
                sprops["wafer_slots"] = wafer_slots
                sprops["wafers"] = stream_ids
                session_props.append(sprops)

            # Close the databases
            if self.context_file is not None:
                del ctx

            # Now construct the list of observation properties, either with all wafers
            # in a single observation or one observation per wafer.
            obs_props = list()
            for sprops in session_props:
                if self.combine_wafers:
                    # Place all wafers into a single observation
                    oprops = dict(sprops)
                    oprops["name"] = oprops["session_name"]
                    oprops["wafer"] = "all"
                    if keep_wafers is not None:
                        new_wafers = list()
                        for wf in oprops["wafers"]:
                            if wf in keep_wafers:
                                new_wafers.append(wf)
                        oprops["n_wafers"] = len(new_wafers)
                        oprops["wafers"] = new_wafers
                    if oprops["n_wafers"] > 0:
                        obs_props.append(oprops)
                else:
                    # One observation per wafer
                    for wf in sprops["wafers"]:
                        if (keep_wafers is None) or (wf in keep_wafers):
                            oprops = dict(sprops)
                            oprops["n_wafers"] = 1
                            oprops["name"] = f"{oprops['session_name']}_{wf}"
                            oprops["wafer"] = wf
                            obs_props.append(oprops)

        if comm.comm_world is not None:
            obs_props = comm.comm_world.bcast(obs_props, root=0)
            if self.preprocess_config is not None:
                preproc_conf = comm.comm_world.bcast(preproc_conf, root=0)

        log.info_rank(
            "LoadContext parsed observation sizes in", comm=comm.comm_world, timer=timer
        )

        if len(obs_props) == 0:
            msg = "No observation IDs specified or matched the regex"
            raise RuntimeError(msg)

        # Distribute observations among groups- we use the detector-seconds
        # for load balancing.

        obs_sizes = [int(x["n_wafers"] * x["n_samples"]) for x in obs_props]
        groupdist = distribute_discrete(obs_sizes, comm.ngroups)
        group_firstobs = groupdist[comm.group].offset
        group_numobs = groupdist[comm.group].n_elem

        # Every group loads its observations
        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            obs_name = obs_props[obindx]["name"]
            otimer = Timer()
            otimer.start()

            # If we have an observation with a specific wafer, modify the detector
            # selection dictionary to include only that wafer.
            obs_dets_select = dets_select
            if obs_props[obindx]["wafer"] != "all":
                if obs_dets_select is None:
                    obs_dets_select = dict()
                obs_dets_select[self.ax_detinfo_wafer_key] = obs_props[obindx]["wafer"]

            # One process in the group loads the metadata, builds the focalplane
            # model, and broadcasts to the rest of the group.
            #
            # FIXME: in the case of multiple wafers per observation, each wafer
            # reader needs to do this step, in order to pull in unique metadata
            # that is per-wafer and merge.
            #
            obs_meta, det_props, n_samp = self._load_metadata(
                obs_name,
                obs_props[obindx]["session_name"],
                comm.comm_group,
                obs_dets_select,
                preproc_conf,
            )

            # Create the instrument model for the observation
            telescope, fp_flags = self._create_obs_instrument(
                obs_name,
                comm.comm_group,
                det_props,
            )

            # Create the observation and allocate data objects
            ob, have_pointing = self._create_observation(
                obs_name,
                obs_props[obindx]["session_name"],
                comm,
                telescope,
                n_samp,
                fp_flags,
                obs_meta,
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

    @function_timer
    def _load_metadata(self, obs_name, session_name, gcomm, dets_select, preproc_conf):
        """Load observation metadata and the focalplane properties.

        One process in the group loads the metadata, builds the focalplane
        model, and broadcasts to the rest of the group.

        Args:
            obs_name (str):  The observation name
            session_name (str):  The observing session name (context obs ID).
            gcomm (MPI.Comm):  The group communicator or None.
            dets_select (dict):  The detector selection dictionary passed to
                Context.get_meta()
            preproc_conf:  If not None, the preprocessing config dictionary used
                to cut detectors when loading.

        Returns:
            (tuple):  The (observation metadata, detector property table, samples)
                for the observation.

        """
        log = Logger.get()
        timer = Timer()
        timer.start()
        if gcomm is None:
            rank = 0
        else:
            rank = gcomm.rank

        det_props = None
        obs_meta = None
        n_samp = None

        # FIXME: This only works if there is one wafer per observation (and
        # hence one reader).
        if rank == 0:
            # Load metadata
            ctx = open_context(context=self.context, context_file=self.context_file)
            meta = ctx.get_meta(session_name, dets=dets_select)
            if self.context_file is not None:
                del ctx
            n_samp = meta["samps"].count

            # Parse the axis manager metadata into observation metadata
            # and detector properties.
            obs_meta = dict()
            fp_cols = dict()
            parse_metadata(
                meta,
                obs_meta,
                fp_cols,
                self.ax_pathsep,
                self.axis_detector,
                None,
                None,
            )

            # Set the column used by toast as the detector name.
            fp_cols["name"] = Column(
                name="name",
                data=fp_cols[f"det_info{self.ax_pathsep}readout_id"].data,
            )

            if self.analytic_bandpass:
                # Add bandpass information to the focalplane
                try:
                    band = fp_cols[f"det_info{self.ax_pathsep}band"].data
                except KeyError:
                    band = fp_cols[
                        f"det_info{self.ax_pathsep}wafer{self.ax_pathsep}bandpass"
                    ].data
                freq = [float(b[1:]) for b in band]
                bandcenter = np.array(freq) * u.GHz
                bandwidth = bandcenter * self.bandwidth
                fp_cols["bandcenter"] = Column(name="bandcenter", data=bandcenter)
                fp_cols["bandwidth"] = Column(name="bandwidth", data=bandwidth)

            # Construct table
            det_props = QTable(fp_cols)
            del meta

            # Detector ordering.  When distributing and reading data, we
            # need to have the detectors sorted into contiguous blocks of
            # of detectors.  We sort the table now by wafer and then by name.
            wafer_key = f"det_info{self.ax_pathsep}{self.ax_detinfo_wafer_key}"
            det_props.sort([wafer_key, "name"])

        log.debug_rank(
            f"LoadContext {obs_name} metadata loaded in",
            comm=gcomm,
            timer=timer,
        )

        if gcomm is not None:
            obs_meta = gcomm.bcast(obs_meta, root=0)
            det_props = gcomm.bcast(det_props, root=0)
            n_samp = gcomm.bcast(n_samp, root=0)

        log.debug_rank(
            f"LoadContext {obs_name} metadata bcast took",
            comm=gcomm,
            timer=timer,
        )
        return (obs_meta, det_props, n_samp)

    @function_timer
    def _create_obs_instrument(self, obs_name, gcomm, det_props):
        """Create telescope, session, and focalplane flags.

        These are objects needed prior to instantiating the Observation.

        Args:
            obs_name (str):  The observation name
            gcomm (MPI.Comm):  The group communicator or None.
            det_props (QTable):  The astropy table of detector properties.

        Returns:
            (tuple):  The (Telescope, per-detector flags) where the flags
                cut detectors with the processing bit if they have no
                pointing reconstruction.

        """
        log = Logger.get()

        # Convert any focalplane quaternion offsets to toast format.  We also
        # look for any detectors with NaN values and flag those with the
        # processing bit.

        xi_key = f"focal_plane{self.ax_pathsep}xi"
        eta_key = f"focal_plane{self.ax_pathsep}eta"
        gamma_key = f"focal_plane{self.ax_pathsep}gamma"
        wafer_type_key = f"det_info{self.ax_pathsep}wafer{self.ax_pathsep}type"

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
                len(det_props[f"det_info{self.ax_pathsep}readout_id"]),
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
                f"LoadContext {obs_name} has no unflagged detectors, skipping!",
                comm=gcomm,
            )
            return

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

        # Use the default observatory location
        raw_site = so3g.proj.coords.SITES["_default"]
        tele_lon_deg = raw_site.lon
        tele_lat_deg = raw_site.lat
        tele_alt_m = raw_site.elev

        # Construct a toast weather object.  If we had metadata containing the
        # current conditions (temperature, PWV, wind, etc) we could use it here.
        # For now, we use the EarthlySite.typical_weather so that our pointing
        # corrections are the same as other parts of the code.  For the weather
        # parameters not in this dictionary, we use the toast-bundled MERRA-2
        # data for the Atacama.  We add this MERRA-2 data later once we have the
        # observing session start time.

        site_temp_celsius = raw_site.typical_weather["temperature"]
        site_humidity = raw_site.typical_weather["humidity"]
        site_pressure = raw_site.typical_weather["pressure"]

        site_weather = toast.weather.Weather(
            time=None,
            ice_water=None,
            liquid_water=None,
            pwv=None,
            humidity=None,
            surface_pressure=site_pressure * u.mbar,
            surface_temperature=site_temp_celsius * u.Celsius,
            air_temperature=site_temp_celsius * u.Celsius,
            west_wind=None,
            south_wind=None,
        )
        site_weather.relative_humidity = site_humidity

        site = SOSite(
            name="SO",
            lat=tele_lat_deg * u.degree,
            lon=tele_lon_deg * u.degree,
            alt=tele_alt_m * u.meter,
            weather=site_weather,
        )

        telescope = toast.instrument.Telescope(
            self.telescope_name, focalplane=focalplane, site=site
        )
        return telescope, fp_flags

    @function_timer
    def _create_observation(
        self, obs_name, session_name, comm, telescope, n_samp, fp_flags, meta
    ):
        """Create the observation.

        Note that the focalplane table has already been sorted by wafer
        name and then detector name.

        Args:
            obs_name (str):  The observation name
            session_name (str):  The observing session name (context obs ID).
            gcomm (MPI.Comm):  The group communicator or None.
            telescope (Telescope):  The Telescope for this observation.
            n_samp (int):  The number of samples in the observation.
            fp_flags (dict):  The per-detector flags to apply.
            meta (dict):  The observation metadata to populate.

        Returns:
            (tuple):  The (Observation, have_pointing), where the second
                element is True if the observation has boresight pointing.

        """
        log = Logger.get()
        timer = Timer()
        timer.start()

        # Computer the detector sets
        if self.detset_key is None:
            detsets = None
        else:
            detsets = telescope.focalplane.detector_groups(self.detset_key)

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
            self.times,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )

        fp_array = telescope.focalplane.detector_data
        ax_boresight_az = ax_name_fp_subst(self.ax_boresight_az, fp_array)
        ax_boresight_el = ax_name_fp_subst(self.ax_boresight_el, fp_array)
        ax_boresight_roll = ax_name_fp_subst(self.ax_boresight_roll, fp_array)
        ax_hwp_angle = ax_name_fp_subst(self.ax_hwp_angle, fp_array)
        ax_det_signal = ax_name_fp_subst(self.ax_det_signal, fp_array)

        have_pointing = True
        if ax_boresight_az is None:
            have_pointing = False
        else:
            ob.shared.create_column(
                self.azimuth,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
        if ax_boresight_el is None:
            have_pointing = False
        else:
            ob.shared.create_column(
                self.elevation,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
        if ax_boresight_roll is None:
            have_pointing = False
        else:
            ob.shared.create_column(
                self.roll,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )

        if have_pointing:
            ob.shared.create_column(
                self.boresight_azel,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.boresight_radec,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
            )
        if self.hwp_angle is not None and ax_hwp_angle is not None:
            ob.shared.create_column(
                self.hwp_angle,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
        if self.boresight_angle is not None:
            ob.shared.create_column(
                self.boresight_angle,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
        if self.corotator_angle is not None:
            ob.shared.create_column(
                self.corotator_angle,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
        ob.shared.create_column(
            self.shared_flags,
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
            ob.detdata.create(
                self.det_data, dtype=np.float64, units=self.det_data_units
            )
            ob.detdata.create(self.det_flags, dtype=np.uint8)

        if meta is not None:
            ob.update(meta)

        log.debug_rank(
            f"LoadContext {obs_name} allocate Observation in",
            comm=comm.comm_group,
            timer=timer,
        )
        return ob, have_pointing

    @function_timer
    def _load_data(self, ob, have_pointing, pconf):
        """Recursively load AxisManager data.

        The first process with detectors from a given wafer will read
        all data for that wafer.  Some examples:

        - 1 processes, 7 wafers:  rank 0 reads all wafers
        - 2 processes, 7 wafers:  rank 0 reads 3 wafers, rank 1 reads 4 wafers
        - 16 processes, 7 wafers:  ranks 0, 2, 4, 6, 9, 11, 13 read one wafer each

        Each reading process will optionally apply the preprocessing
        to the loaded data prior to distribution. The rank zero process (which
        is always a reader) extracts any shared data and additional metadata that
        is found.  Reading processes communicate slices of detector data to all
        other processes.

        Args:
            ob (Observation):  The observation to populate.
            have_pointing (bool):  True if the data contains boresight pointing.
            pconf (dict):  The optional preprocessing config.

        Returns:
            None

        """
        log = Logger.get()
        timer = Timer()
        timer.start()

        comm = ob.comm
        gcomm = comm.comm_group
        rank = ob.comm.group_rank

        # Compute the data distribution of detectors
        wafer_key = f"det_info{self.ax_pathsep}{self.ax_detinfo_wafer_key}"
        wafer_dets, wafer_readers, wafer_proc_dets, proc_wafer_dets = (
            distribute_detector_props(ob, wafer_key)
        )

        # Load data and apply preprocessing.
        axwafers = read_and_preprocess_wafers(
            ob.name,
            ob.session.name,
            gcomm,
            wafer_readers,
            wafer_dets,
            preconfig=pconf,
            context=self.context,
            context_file=self.context_file,
        )

        log.debug_rank(
            f"LoadContext {ob.name} AxisManager data loaded in",
            comm=gcomm,
            timer=timer,
        )

        # Track the fields we are extracting
        fp_array = ob.telescope.focalplane.detector_data
        ax_times = ax_name_fp_subst(self.ax_times, fp_array)
        ax_boresight_az = ax_name_fp_subst(self.ax_boresight_az, fp_array)
        ax_boresight_el = ax_name_fp_subst(self.ax_boresight_el, fp_array)
        ax_boresight_roll = ax_name_fp_subst(self.ax_boresight_roll, fp_array)
        ax_hwp_angle = ax_name_fp_subst(self.ax_hwp_angle, fp_array)
        ax_det_signal = ax_name_fp_subst(self.ax_det_signal, fp_array)
        ax_flags = list()
        for axname, bit in self.ax_flags:
            full_name = ax_name_fp_subst(axname, fp_array)
            ax_flags.append((full_name, bit))
        ax_det_flags = list()
        for axname, bit in self.ax_det_flags:
            full_name = ax_name_fp_subst(axname, fp_array)
            ax_det_flags.append((full_name, bit))

        shared_ax_to_obs = {ax_times: self.times}
        if have_pointing:
            shared_ax_to_obs[ax_boresight_az] = self.azimuth
            shared_ax_to_obs[ax_boresight_el] = self.elevation
            shared_ax_to_obs[ax_boresight_roll] = self.roll
        if self.hwp_angle is not None and ax_hwp_angle is not None:
            shared_ax_to_obs[ax_hwp_angle] = self.hwp_angle
        shared_flag_invert = {x[0]: (x[1] < 0) for x in ax_flags}
        shared_flag_fields = {x[0]: abs(x[1]) for x in ax_flags}
        det_flag_invert = {x[0]: (x[1] < 0) for x in ax_det_flags}
        det_flag_fields = {x[0]: abs(x[1]) for x in ax_det_flags}

        # The results of the recursive parsing
        extra_meta = dict()
        shared_data = dict()
        det_data = dict()
        interval_data = dict()

        # Recursively parse axis managers and build list of data to
        # populate in the observation.  Since all wafers have the same
        # ancil data (and the same field names, just with different
        # detectors), we can just do this on one process and broadcast
        # the result.
        #
        # FIXME:  In the case of multiple wafers (and readers) per
        # observation, every reader must parse every wafer and merge
        # all the metadata information.
        #
        temp_shared = None
        restricted_samps = None
        ax_shift = None
        if ob.comm.group_rank == 0:
            first_wafer = list(axwafers.keys())[0]
            self._parse_data(
                ob,
                axwafers[first_wafer],
                shared_ax_to_obs,
                shared_flag_fields,
                shared_flag_invert,
                det_flag_fields,
                det_flag_invert,
                extra_meta,
                shared_data,
                det_data,
                interval_data,
                None,
            )
            temp_shared = {x: None for x, y in shared_data.items()}

            # Does the axis manager have a truncated number of samples?
            restricted_samps = axwafers[first_wafer][self.axis_sample].count
            if restricted_samps != ob.n_local_samples:
                ax_shift = axwafers[first_wafer][self.axis_sample].offset
            else:
                ax_shift = 0

        if gcomm is not None:
            extra_meta = gcomm.bcast(extra_meta, root=0)
            temp_shared = gcomm.bcast(temp_shared, root=0)
            if ob.comm.group_rank != 0:
                shared_data = temp_shared
            det_data = gcomm.bcast(det_data, root=0)
            interval_data = gcomm.bcast(interval_data, root=0)
            restricted_samps = gcomm.bcast(restricted_samps, root=0)
            ax_shift = gcomm.bcast(ax_shift, root=0)

        # Add extra metadata that was discovered.
        ob.update(extra_meta)

        log.debug_rank(
            f"LoadContext {ob.name} AxisManager field parsing took",
            comm=gcomm,
            timer=timer,
        )

        # Create the intervals
        for intr_name, sampspans in interval_data.items():
            ob.intervals[intr_name] = toast.intervals.IntervalList(
                ob.shared[self.times], samplespans=sampspans
            )

        # Collectively store shared data.  All readers have a full copy of
        # this data, but we only set this from rank zero.  If there are
        # cut samples at the beginning and end, ensure that timestamps are
        # always valid.
        for shr_obs_name, shrbuf in shared_data.items():
            bf = shrbuf
            if shr_obs_name == self.times and restricted_samps != ob.n_local_samples:
                if rank == 0:
                    msg = f"Axman samples {restricted_samps} != {ob.n_local_samples}"
                    msg += ", extrapolating timestamps"
                    log.debug(msg)
                    bf = np.zeros(ob.n_local_samples, dtype=np.float64)
                    (rate, dt, _, _, _) = toast.utils.rate_from_times(shrbuf)
                    bf[ax_shift : ax_shift + restricted_samps] = shrbuf
                    bf[0:ax_shift] = shrbuf[0] + dt * np.arange(
                        -ax_shift, 0, 1, dtype=np.float64
                    )
                    end_gap = ob.n_local_samples - restricted_samps - ax_shift
                    bf[ax_shift + restricted_samps :] = shrbuf[-1] + dt * np.arange(
                        1, end_gap + 1, 1, dtype=np.float64
                    )
            ob.shared[shr_obs_name].set(bf, fromrank=0)

        log.debug_rank(
            f"LoadContext {ob.name} Shared data copy took",
            comm=gcomm,
            timer=timer,
        )

        # Now that we have timestamps loaded, update our focalplane sample rate
        # and observing session times.
        (rate, dt, dt_min, dt_max, dt_std) = toast.utils.rate_from_times(
            ob.shared[self.times].data
        )
        ob.telescope.focalplane.sample_rate = rate * u.Hz

        ob.session.start = datetime.fromtimestamp(
            ob.shared[self.times].data[0]
        ).astimezone(timezone.utc)
        ob.session.end = datetime.fromtimestamp(
            ob.shared[self.times].data[-1]
        ).astimezone(timezone.utc)

        # Update site weather, now that we have the observing time.
        sim_atacama = toast.weather.SimWeather(
            time=ob.session.start,
            name="atacama",
            median_weather=True,
        )
        old_weather = ob.telescope.site.weather
        new_weather = toast.weather.Weather(
            time=sim_atacama.time,
            ice_water=sim_atacama.ice_water,
            liquid_water=sim_atacama.liquid_water,
            pwv=sim_atacama.pwv,
            humidity=sim_atacama.humidity,
            surface_pressure=old_weather.surface_pressure,
            surface_temperature=old_weather.surface_temperature,
            air_temperature=old_weather.air_temperature,
            west_wind=sim_atacama.west_wind,
            south_wind=sim_atacama.south_wind,
        )
        new_weather.relative_humidity = old_weather.relative_humidity
        ob.telescope.site.weather = new_weather

        log.debug_rank(
            f"LoadContext {ob.name} sample rate calculation took",
            comm=gcomm,
            timer=timer,
        )

        # Distribute detector data
        for field, merge_list in det_data.items():
            for ax_field, ax_dtype, mask in merge_list:
                do_invert = False
                if ax_field in det_flag_invert:
                    do_invert = det_flag_invert[ax_field]
                distribute_detector_data(
                    ob,
                    field,
                    axwafers,
                    self.axis_detector,
                    self.axis_sample,
                    ax_field,
                    ax_dtype,
                    wafer_readers,
                    wafer_proc_dets,
                    proc_wafer_dets,
                    str(self.ax_pathsep),
                    is_flag=(mask is not None),
                    flag_invert=do_invert,
                    flag_mask=mask,
                )

        # Original wafer data no longer needed.  AxisManager does not seem to
        # have a clean destructor, so do it manually.
        def _ax_del_children(ax):
            for k in list(ax.keys()):
                if isinstance(ax[k], AxisManager):
                    _ax_del_children(ax[k])
                del ax[k]

        for wfname in list(axwafers.keys()):
            _ax_del_children(axwafers[wfname])
            del axwafers[wfname]
        del axwafers

        log.debug_rank(
            f"LoadContext {ob.name} Detector data distribution took",
            comm=gcomm,
            timer=timer,
        )

    def _parse_data(
        self,
        obs,
        axman,
        shared_ax_to_obs,
        shared_flag_fields,
        shared_flag_invert,
        det_flag_fields,
        det_flag_invert,
        extra_meta,
        shared_data,
        det_data,
        interval_data,
        base,
    ):
        """Recursively parse AxisManager data products.

        This (mostly) does not actually extract the data, but instead builds
        up information needed to extract that data in following steps.

        Args:
            obs (Observation):  The observation
            axman (AxisManager):  The top-level axis manager
            shared_ax_to_obs (dict):  For each ancil axis manager field, the
                corresponding observation shared key.
            shared_flag_fields (dict):  For each shared axis manager field, the
                (target observation flag field, bit mask) where this should be
                merged.
            shared_flag_invert (dict):  For each shared axis manager flag field,
                whether the meaning of the flag should be inverted.
            det_flag_fields (dict):  For each axis manager detector flag field,
                the (target observation detdata field, bit mask) where this
                should be merged.
            det_flag_invert (dict):  For each axis manager detector flag field,
                whether the meaning of the flag should be inverted.
            extra_meta (dict):  Any new / additional metadata found on the root
                process.
            shared_data (dict):  For each shared observation field, a handle to
                the axis manager data buffer to use.  This is only significant
                on the rank zero process.
            det_data (dict):  For each detector data field, a tuple of (axman
                field, flag mask).  This is significant on the reading processes.
                If the data is signal then flag mask will be None.
            interval_data (dict):  For each observation interval field, the
                sample spans to use.  Only significant on the rank zero process.
            base (str):  The current dictionary key for this recursion level.

        Returns:
            None

        """

        def _valid_meta(obj):
            # Check that the object is an allowed type before adding it to the metadata
            if np.isscalar(obj):
                return True
            if isinstance(obj, np.ndarray):
                return True
            if isinstance(obj, (list, set, dict, tuple, str)):
                return True
            return False

        # Find the per-observation names we might be using
        fp_array = obs.telescope.focalplane.detector_data
        ax_det_signal = ax_name_fp_subst(self.ax_det_signal, fp_array)

        # Some metadata has already been parsed, but some new values
        # may only show up when reading data, so we need to handle those
        # as well.
        rank = obs.comm.group_rank

        created_meta_extra = False
        created_meta_obs = False

        if base is None:
            mcur = obs
            mext = extra_meta
        else:
            if base not in obs:
                created_meta_obs = True
                obs[base] = dict()
            mcur = obs[base]
            if base not in extra_meta:
                created_meta_extra = True
                extra_meta[base] = dict()
            mext = extra_meta[base]
        for key in axman.keys():
            if base is not None:
                data_key = f"{base}{self.ax_pathsep}{key}"
            else:
                data_key = key
            if isinstance(axman[key], AxisInterface):
                # This is one of the axes
                continue
            if isinstance(axman[key], FlagManager):
                # Descend- Anything different we need to do here?
                self._parse_data(
                    obs,
                    axman[key],
                    shared_ax_to_obs,
                    shared_flag_fields,
                    shared_flag_invert,
                    det_flag_fields,
                    det_flag_invert,
                    extra_meta,
                    shared_data,
                    det_data,
                    interval_data,
                    key,
                )
            elif isinstance(axman[key], AxisManager):
                # Descend
                self._parse_data(
                    obs,
                    axman[key],
                    shared_ax_to_obs,
                    shared_flag_fields,
                    shared_flag_invert,
                    det_flag_fields,
                    det_flag_invert,
                    extra_meta,
                    shared_data,
                    det_data,
                    interval_data,
                    key,
                )
            else:
                # FIXME:  It would be nicer if this information was available
                # through a public member...
                field_axes = axman._assignments[key]
                if len(field_axes) == 0:
                    # This data is not associated with an axis.  If it does not
                    # yet exist in the observation metadata, then add it.
                    if _valid_meta(axman[key]):
                        mext[key] = axman[key]
                elif field_axes[0] == self.axis_detector:
                    if len(field_axes) == 1:
                        # This is a detector property
                        if data_key in obs.telescope.focalplane.detector_data.colnames:
                            # We already included this in the detector properties
                            continue
                        # This must be some per-detector derived data- add to the
                        # observation dictionary
                        if data_key not in mcur:
                            if _valid_meta(axman[key]):
                                mext[key] = axman[key]
                    elif field_axes[1] == self.axis_sample:
                        # This is detector data.  See if it is one of the standard
                        # fields we are parsing.
                        if hasattr(axman[key], "dtype"):
                            # This is an array
                            dt = axman[key].dtype
                        else:
                            # This is a RangesMatrix of flags
                            dt = np.dtype(np.uint8)
                        if data_key == ax_det_signal:
                            # Detector signal
                            det_data[self.det_data] = [(data_key, dt, None)]
                        elif data_key in det_flag_fields:
                            # One of the flag fields
                            if self.det_flags not in det_data:
                                det_data[self.det_flags] = list()
                            det_data[self.det_flags].append(
                                (
                                    data_key,
                                    dt,
                                    det_flag_fields[data_key],
                                )
                            )
                        else:
                            # Some other kind of detector data.  Ignore this for now,
                            # since this needs to be pre-created in the observation
                            # for all processes (not just the readers).  If we need
                            # to read other detector data fields we should add a trait
                            # for those specifying the shape and type.
                            pass
                    else:
                        # Must be some other type of object...
                        if data_key not in mcur:
                            if _valid_meta(axman[key]):
                                mext[key] = axman[key]
                elif field_axes[0] == self.axis_sample:
                    # This is shared data
                    if isinstance(axman[key], so3g.proj.Ranges):
                        # This is a set of 1D shared ranges.  Translate this to a
                        # toast interval list.
                        if rank == 0:
                            samplespans = list()
                            for rg in axman[key].ranges():
                                samplespans.append((rg[0], rg[1] - 1))
                            interval_data[data_key] = samplespans
                    elif data_key in shared_ax_to_obs:
                        # One of our selected shared data fields
                        if rank == 0:
                            shared_data[shared_ax_to_obs[data_key]] = axman[key]
                    elif data_key in shared_flag_fields:
                        # One of our selected flag fields
                        if obs.rank == 0:
                            if self.shared_flags not in shared_data:
                                shared_data[self.shared_flags] = np.array(
                                    obs.shared[self.shared_flags].data
                                )
                            temp = shared_flag_fields[data_key] * np.ones_like(
                                axman[key]
                            )
                            if shared_flag_invert[data_key]:
                                temp[axman[key] != 0] = 0
                            else:
                                temp[axman[key] == 0] = 0
                            shared_data[self.shared_flags] |= temp
                            del temp
                    else:
                        # This is some other shared data.
                        # FIXME: we skip this for now.
                        continue
                else:
                    # Some other object...
                    if data_key not in mcur:
                        if _valid_meta(axman[key]):
                            mext[key] = axman[key]

        # Clean up any dictionaries that we created if they were empty
        if created_meta_extra and len(mext) == 0:
            del extra_meta[base]
        if created_meta_obs and len(mcur) == 0:
            del obs[base]

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {
            "shared": [
                self.times,
                self.boresight_azel,
            ],
            "detdata": [self.det_data],
        }
        if self.boresight_radec is not None:
            prov["shared"].append(self.boresight_radec)
        if self.boresight_angle is not None:
            prov["shared"].append(self.boresight_angle)
        if self.corotator_angle is not None:
            prov["shared"].append(self.corotator_angle)
        if self.hwp_angle is not None:
            prov["shared"].append(self.hwp_angle)
        return prov

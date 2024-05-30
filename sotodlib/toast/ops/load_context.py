# Copyright (c) 2022-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import re
import datetime
import yaml

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import Column, QTable

import toast
from toast.dist import distribute_uniform
from toast.timing import function_timer, Timer
from toast.traits import (
    trait_docs, Int, Unicode, Instance, List, Unit, Dict, Bool, Float
)
from toast.ops.operator import Operator
from toast.utils import Environment, Logger
from toast.dist import distribute_discrete
from toast.observation import default_values as defaults

import so3g

from ...core import Context, AxisManager, FlagManager
from ...core.axisman import AxisInterface
from ...preprocess import Pipeline as PreProcPipe

from ..instrument import SOFocalplane, SOSite


@trait_docs
class LoadContext(Operator):
    """Load one or more observations from a Context.

    NOTE:  each "observation" in the Context system maps to a toast "observing
    session".

    Given a context, load one or more observations.  If specified, the context
    should exist on all processes in the group.  The `readout_ids`, `detsets`,
    and `bands` traits are used to select subsets of detectors.  Alternatively
    the context file can be specified, in which case this is passed to the
    context constructor.

    The traits starting with an "ax_*" prefix indicate the axis manager names
    that should be mapped to the standard toast shared and detdata objects.
    For ax_flags tuples, a negative bit value indicates that the flag data
    should first be inverted before combined.

    Additional nested axismanager data is grouped into detdata, shared, or
    metadata objects based on whether they use detector / sample axes.  The
    nested structure is flattened into names built from the keys in the
    hierarchy.

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

    read_independent = Bool(
        False, help="If True, all processes read det data independently"
    )

    observations = List(list(), help="List of observation IDs to load")

    readout_ids = List(list(), help="Only load this list of readout_id values")

    detsets = List(list(), help="Only load this list of detset values")

    bands = List(list(), help="Only load this list of band values")

    dets_select = Dict(
        dict(), help="The `dets` selection dictionary to pass to get_obs()"
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
        "boresight_az",
        allow_none=True,
        help="Field with boresight Az",
    )

    ax_boresight_el = Unicode(
        "boresight_el",
        allow_none=True,
        help="Field with boresight El",
    )

    ax_boresight_roll = Unicode(
        "boresight_roll",
        allow_none=True,
        help="Field with boresight Roll",
    )

    ax_hwp_angle = Unicode(
        "hwp_angle",
        allow_none=True,
        help="Field with HWP angle",
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

    def _open_context(self):
        if self.context is None:
            # The user did not specify a context- create a temporary
            # one from the file
            return Context(self.context_file)
        else:
            # Just return the user-specified context.
            return self.context

    def _close_context(self, ctx):
        if self.context is None:
            # We previously created a context from a file.  Close
            # / delete that now.
            del ctx

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        comm = data.comm

        if self.context is None:
            if self.context_file is None:
                msg = "Either the context or context_file must be specified"
                raise RuntimeError(msg)
        else:
            if self.context_file is not None:
                msg = "Only one of the context or context_file should be specified"
                raise RuntimeError(msg)

        if not self.read_independent:
            msg = "Parallel reading from a context is temporarily broken."
            msg += " Forcing read_independent=True, but this will be very"
            msg += " slow for more that ~64 processes."
            self.read_independent = True
            log.warning_rank(msg, comm=comm.comm_world)

        # Build our detector selection dictionary
        dets_select = None
        if len(self.dets_select) > 0:
            # Use the full dictionary provided by the user
            dets_select = self.dets_select
        elif len(self.readout_ids) > 0 or len(self.bands) > 0 or len(self.detsets) > 0:
            # We have some selection, build the dictionary
            dets_select = dict()
            if len(self.readout_ids) > 0:
                dets_select["readout_id"] = list(self.readout_ids)
            if len(self.bands) > 0:
                dets_select["band"] = list(self.bands)
            if len(self.detsets) > 0:
                dets_select["detset"] = list(self.detsets)

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
            self.observations = olist

        obs_props = None
        preproc_conf = None
        if comm.world_rank == 0:
            obs_props = list()
            obs_list = self.observations
            if self.preprocess_config is not None:
                with open(self.preprocess_config, "r") as f:
                    preproc_conf = yaml.safe_load(f)
            ctx = self._open_context()
            if self.observation_regex is not None:
                # Match against the full list of observation IDs
                obs_list = []
                pat = re.compile(self.observation_regex)
                all_obs = ctx.obsdb.query()
                for result in all_obs:
                    if pat.match(result["obs_id"]) is not None:
                        obs_list.append(result["obs_id"])

            for iobs, obs_id in enumerate(obs_list):
                meta = ctx.get_meta(obs_id=obs_id, dets=dets_select)
                oprops = dict()
                oprops["name"] = obs_id
                oprops["duration"] = meta["obs_info"]["duration"]
                oprops["n_det"] = len(meta["dets"].vals)
                obs_props.append(oprops)
            self._close_context(ctx)

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

        obs_sizes = [int(x["n_det"] * x["duration"]) for x in obs_props]
        groupdist = distribute_discrete(obs_sizes, comm.ngroups)
        group_firstobs = groupdist[comm.group].offset
        group_numobs = groupdist[comm.group].n_elem

        # Every group loads its observations
        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            obs_name = obs_props[obindx]["name"]
            otimer = Timer()
            otimer.start()

            # One process in the group loads the metadata, builds the focalplane
            # model, and broadcasts to the rest of the group.

            det_props = None
            obs_meta = None
            n_samp = None
            if comm.group_rank == 0:
                # Load metadata
                ctx = self._open_context()
                meta = ctx.get_meta(obs_name, dets=dets_select)
                self._close_context(ctx)
                n_samp = meta["samps"].count

                if self.preprocess_config is not None:
                    # Cut detectors with preprocessing
                    prepipe = PreProcPipe(
                        preproc_conf["process_pipe"],
                        logger=log,
                    )
                    for process in prepipe:
                        log.debug(f"Preprocess selecting on {process.name}")
                        process.select(meta)

                # For each element of meta we do the following:
                # - If the object has one axis and it is the detector axis,
                #   treat it as a column in the detector property table
                # - If the object has one axis and it is the sample axis,
                #   it will be loaded later as shared data.
                # - If the object has multiple axes, load it later
                # - If the object has no axes, then treat it as observation
                #   metadata
                # - If the object is a nested AxisManager, descend and apply
                #   the same steps as above.
                obs_meta = dict()
                fp_cols = dict()
                self._parse_meta(meta, None, obs_meta, None, fp_cols)

                if self.analytic_bandpass:
                    # Add bandpass information to the focalplane
                    try:
                        band = fp_cols["det_info_band"].data
                    except KeyError:
                        band = fp_cols["det_info_wafer_bandpass"].data
                    freq = [float(b[1:]) for b in band]
                    bandcenter = np.array(freq) * u.GHz
                    bandwidth = bandcenter * self.bandwidth
                    fp_cols["bandcenter"] = Column(name="bandcenter", data=bandcenter)
                    fp_cols["bandwidth"] = Column(name="bandwidth", data=bandwidth)

                # Construct table
                det_props = QTable(fp_cols)
                del meta

            log.info_rank(
                f"LoadContext {obs_name} metadata loaded in",
                comm=comm.comm_group,
                timer=otimer,
            )

            if comm.comm_group is not None:
                obs_meta = comm.comm_group.bcast(obs_meta, root=0)
                det_props = comm.comm_group.bcast(det_props, root=0)
                n_samp = comm.comm_group.bcast(n_samp, root=0)

            log.info_rank(
                f"LoadContext {obs_name} metadata bcast took",
                comm=comm.comm_group,
                timer=otimer,
            )

            # Create the observation.  We intentionally use the generic focalplane
            # and class here, in case we are loading data from legacy experiments.

            # Convert any focalplane quaternion offsets to toast format.  We also
            # look for any detectors with NaN values and flag those with the
            # processing bit.

            name_col = Column(name="name", data=det_props["det_info_readout_id"])
            det_props.add_column(name_col, index=0)

            fp_flags = dict()
            if "focal_plane_xi" in det_props.colnames:
                fp_bad = np.logical_or(
                    np.isnan(det_props["focal_plane_xi"]),
                    np.logical_or(
                        np.isnan(det_props["focal_plane_eta"]),
                        np.isnan(det_props["focal_plane_gamma"]),
                    ),
                )
                fp_flags = {
                    x: defaults.det_mask_processing
                    for x, y in zip(det_props["name"], fp_bad)
                    if y
                }
                det_props["focal_plane_xi"][fp_bad] = 0
                det_props["focal_plane_eta"][fp_bad] = 0
                det_props["focal_plane_gamma"][fp_bad] = 0
                quat_data = toast.instrument_coords.xieta_to_quat(
                    det_props["focal_plane_xi"],
                    det_props["focal_plane_eta"],
                    det_props["focal_plane_gamma"],
                )
            else:
                # No detector offsets yet
                quat_data = np.tile(
                    np.array([0, 0, 0, 1], dtype=np.float64),
                    len(det_props["det_info_readout_id"]),
                ).reshape((-1, 4))

            # Do we have any good detectors left?
            n_good = 0
            for det in det_props["name"]:
                if det not in fp_flags or fp_flags[det] == 0:
                    n_good += 1
            if n_good == 0:
                log.warning_rank(
                    f"LoadContext {obs_name} has no unflagged detectors, skipping!",
                    comm=comm.comm_group,
                )
                continue

            quat_col = Column(
                name="quat",
                data=quat_data,
            )
            det_props.add_column(quat_col, index=0)

            focalplane = toast.instrument.Focalplane(
                detector_data=det_props,
                sample_rate=1.0 * u.Hz,
            )

            if self.detset_key is None:
                detsets = None
            else:
                detsets = focalplane.detector_groups(self.detset_key)

            # For now, this should be good enough position for instruments near the
            # S.O. location.
            site = SOSite()

            telescope = toast.instrument.Telescope(
                self.telescope_name, focalplane=focalplane, site=site
            )

            # Note:  the session times will be updated later when reading timestamps
            session = toast.instrument.Session(obs_name)

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

            log.info_rank(
                f"LoadContext {obs_name} construct Observation in",
                comm=comm.comm_group,
                timer=otimer,
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

            have_pointing = True
            if self.ax_boresight_az is None:
                have_pointing = False
            else:
                ob.shared.create_column(
                    self.azimuth,
                    shape=(ob.n_local_samples,),
                    dtype=np.float64,
                )
            if self.ax_boresight_el is None:
                have_pointing = False
            else:
                ob.shared.create_column(
                    self.elevation,
                    shape=(ob.n_local_samples,),
                    dtype=np.float64,
                )
            if self.ax_boresight_roll is None:
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
            if self.hwp_angle is not None and self.ax_hwp_angle is not None:
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
            if self.ax_det_signal is not None:
                ob.detdata.create(
                    self.det_data, dtype=np.float64, units=self.det_data_units
                )
                ob.detdata.create(self.det_flags, dtype=np.uint8)

            log.info_rank(
                f"LoadContext {obs_name} allocated TOD in",
                comm=comm.comm_group,
                timer=otimer,
            )

            # Load and distribute the detector data
            axtod = self._read_detdata(
                ob, preproc_conf, allread=self.read_independent
            )

            log.info_rank(
                f"LoadContext {obs_name} AxisManager loaded in",
                comm=comm.comm_group,
                timer=otimer,
            )

            self._parse_data(ob, have_pointing, axtod, None)

            # No longer needed
            del axtod

            log.info_rank(
                f"LoadContext {obs_name} AxisManager to Observation conversion took",
                comm=comm.comm_group,
                timer=otimer,
            )

            # Position and velocity of the observatory are simply computed.  Only the
            # first row of the process grid needs to do this.
            if have_pointing:
                position = None
                velocity = None
                if ob.comm_col_rank == 0:
                    position, velocity = site.position_velocity(ob.shared[self.times])
                ob.shared[defaults.position].set(position, offset=(0, 0), fromrank=0)
                ob.shared[defaults.velocity].set(velocity, offset=(0, 0), fromrank=0)
                log.info_rank(
                    f"LoadContext {obs_name} site position/velocity took",
                    comm=comm.comm_group,
                    timer=otimer,
                )

                # Since this step takes some time, all processes in the group
                # contribute
                pnt_dist = distribute_uniform(ob.n_all_samples, comm.group_size)

                bore_azel = None
                bore_radec = None
                bore_flags = None

                slc_off = pnt_dist[comm.group_rank].offset
                slc_n = pnt_dist[comm.group_rank].n_elem
                slc = slice(slc_off, slc_n, 1)

                bore_bad = np.logical_or(
                    np.isnan(ob.shared[self.azimuth].data[slc]),
                    np.logical_or(
                        np.isnan(ob.shared[self.elevation].data[slc]),
                        np.isnan(ob.shared[self.roll].data[slc]),
                    ),
                )
                bore_flags = np.array(ob.shared[self.shared_flags].data[slc])
                bore_flags[bore_bad] |= defaults.shared_mask_processing
                temp_az = np.array(ob.shared[self.azimuth].data[slc])
                temp_el = np.array(ob.shared[self.elevation].data[slc])
                temp_roll = np.array(ob.shared[self.roll].data[slc])
                temp_az[bore_bad] = 0
                temp_el[bore_bad] = 0
                temp_roll[bore_bad] = 0
                bore_azel = toast.qarray.from_lonlat_angles(
                    -temp_az,
                    temp_el,
                    temp_roll,
                )
                bore_radec = toast.coordinates.azel_to_radec(
                    site,
                    ob.shared[self.times].data[slc],
                    bore_azel,
                    use_qpoint=True,
                )

                if comm.comm_group is None:
                    all_bore_flags = bore_flags
                    all_bore_azel = bore_azel
                    all_bore_radec = bore_radec
                else:
                    if ob.comm_col_rank == 0:
                        all_bore_flags = np.concatenate(
                            comm.comm_group.gather(bore_flags, root=0)
                        )
                    else:
                        all_bore_flags = None
                        _ = comm.comm_group.gather(bore_flags, root=0)
                    del bore_flags
                    if ob.comm_col_rank == 0:
                        all_bore_azel = np.concatenate(
                            comm.comm_group.gather(bore_azel, root=0)
                        )
                    else:
                        all_bore_azel = None
                        _ = comm.comm_group.gather(bore_azel, root=0)
                    del bore_azel
                    if ob.comm_col_rank == 0:
                        all_bore_radec = np.concatenate(
                            comm.comm_group.gather(bore_radec, root=0)
                        )
                    else:
                        all_bore_radec = None
                        _ = comm.comm_group.gather(bore_radec, root=0)
                    del bore_radec

                ob.shared[self.shared_flags].set(
                    all_bore_flags, offset=(0,), fromrank=0
                )
                ob.shared[self.boresight_azel].set(
                    all_bore_azel, offset=(0, 0), fromrank=0
                )
                ob.shared[self.boresight_radec].set(
                    all_bore_radec, offset=(0, 0), fromrank=0
                )

                log.info_rank(
                    f"LoadContext {obs_name} boresight pointing conversion took",
                    comm=comm.comm_group,
                    timer=otimer,
                )

            data.obs.append(ob)

    def _read_detdata(self, obs, pconf, allread=False):
        log = Logger.get()
        obs_name = obs.name

        # By default, we attempt to use a single reading process and send
        # restricted axis managers to each process.  Since these are potentially
        # larger than 2GB, we try to use the new mpi4py support for pickle-5.
        # If that is not available, we revert to having every process read
        # independently.
        gcomm = obs.comm.comm_group
        if gcomm is None or gcomm.size == 1:
            allread = True
        elif not allread:
            try:
                from mpi4py.util import pkl5

                gcomm = pkl5.Intracomm(gcomm)
            except Exception:
                msg = "Could not use pickle-5 communication, falling back to"
                msg += " independent det data reading"
                log.warning_rank(msg, comm=gcomm)
                allread = True

        if allread:
            msg = f"LoadContext {obs_name} {obs.comm.group_size} processes "
            msg += "reading independently"
            log.info_rank(msg, comm=gcomm)
            # Independent reading
            ctx = self._open_context()
            axtod = ctx.get_obs(obs_name, dets=obs.local_detectors)
            self._close_context(ctx)
            # Apply preprocessing.
            if self.preprocess_config is not None:
                prepipe = PreProcPipe(pconf["process_pipe"], logger=log)
                prepipe.run(axtod, axtod.preprocess)
        else:
            msg = f"LoadContext {obs_name} one reader sending"
            msg += f" data to {obs.comm.group_size} processes"
            log.info_rank(msg, comm=gcomm)
            # One process loads the whole observation and applies preprocessing
            if gcomm.rank == 0:
                ctx = self._open_context()
                axfull = ctx.get_obs(obs_name, dets=obs.all_detectors)
                self._close_context(ctx)
                # Apply preprocessing.
                if self.preprocess_config is not None:
                    prepipe = PreProcPipe(pconf["process_pipe"], logger=log)
                    prepipe.run(axfull, axfull.preprocess)

            if gcomm.rank == 0:
                # We will keep 2 buffers in memory to better overlap
                # axis manager restriction and communication.
                req_even = None
                req_odd = None
                tod_even = None
                tod_odd = None
                # Loop in reverse order so that the rank zero process
                # can restrict its own data while waiting for communication
                # to finish at the end.
                for grank in range(gcomm.size - 1, -1, -1):
                    rank_dets = obs.dist.dets[grank]
                    if grank == 0:
                        # Extract our own data
                        axtod = axfull.restrict("dets", rank_dets, in_place=False)
                    else:
                        # Restrict and send
                        if grank % 2 == 0:
                            if req_even is not None:
                                req_even.wait()
                                del tod_even
                        else:
                            if req_odd is not None:
                                req_odd.wait()
                                del tod_odd
                        if grank % 2 == 0:
                            tod_even = axfull.restrict(
                                "dets", rank_dets, in_place=False
                            )
                        else:
                            tod_odd = axfull.restrict("dets", rank_dets, in_place=False)
                        if grank % 2 == 0:
                            req_even = gcomm.isend(tod_even, grank, tag=grank)
                        else:
                            req_odd = gcomm.isend(tod_odd, grank, tag=grank)
                if req_even is not None:
                    req_even.wait()
                    del tod_even
                if req_odd is not None:
                    req_odd.wait()
                    del tod_odd
                del axfull
            else:
                # Receive our data
                axtod = gcomm.recv(source=0, tag=gcomm.rank)
            gcomm.barrier()
        return axtod

    def _parse_data(self, obs, have_pointing, axman, base):
        # Some metadata has already been parsed, but some new values
        # may only show up when reading data, so we need to handle those
        # as well.
        shared_ax_to_obs = {self.ax_times: self.times}
        if have_pointing:
            shared_ax_to_obs[self.ax_boresight_az] = self.azimuth
            shared_ax_to_obs[self.ax_boresight_el] = self.elevation
            shared_ax_to_obs[self.ax_boresight_roll] = self.roll
        if self.hwp_angle is not None and self.ax_hwp_angle is not None:
            shared_ax_to_obs[self.ax_hwp_angle] = self.hwp_angle
        shared_flag_invert = {x[0]: (x[1] < 0) for x in self.ax_flags}
        shared_flag_fields = {x[0]: abs(x[1]) for x in self.ax_flags}
        det_flag_invert = {x[0]: (x[1] < 0) for x in self.ax_det_flags}
        det_flag_fields = {x[0]: abs(x[1]) for x in self.ax_det_flags}

        if base is None:
            om = obs
        else:
            if base not in obs:
                obs[base] = dict()
            om = obs[base]
        for key in axman.keys():
            if base is not None:
                data_key = f"{base}_{key}"
            else:
                data_key = key
            if isinstance(axman[key], AxisInterface):
                # This is one of the axes
                continue
            if isinstance(axman[key], FlagManager):
                # Descend- Anything different we need to do here?
                self._parse_data(obs, have_pointing, axman[key], key)
            elif isinstance(axman[key], AxisManager):
                # Descend
                self._parse_data(obs, have_pointing, axman[key], key)
            else:
                # FIXME:  It would be nicer if this information was available
                # through a public member...
                field_axes = axman._assignments[key]
                if len(field_axes) == 0:
                    # This data is not associated with an axis.  If it does not
                    # yet exist in the observation metadata, then add it.
                    if key not in om:
                        om[key] = axman[key]
                elif field_axes[0] == self.axis_detector:
                    if len(field_axes) == 1:
                        # This is a detector property
                        if data_key in obs.telescope.focalplane.detector_data.colnames:
                            # We already included this in the detector properties
                            continue
                        # This must be some per-detector derived data- add to the
                        # observation dictionary
                        if data_key not in om:
                            om[key] = axman[key]
                    elif field_axes[1] == self.axis_sample:
                        # This is detector data.  See if it is one of the standard
                        # fields we are parsing.
                        if data_key == self.ax_det_signal:
                            obs.detdata[self.det_data][:, :] = axman[key]
                        elif data_key in det_flag_fields:
                            if isinstance(axman[key], so3g.proj.RangesMatrix):
                                temp = np.empty(obs.n_local_samples, dtype=np.uint8)
                                if det_flag_invert[data_key]:
                                    for idet, det in enumerate(obs.local_detectors):
                                        temp[:] = det_flag_fields[data_key]
                                        for rg in axman[key][idet].ranges():
                                            temp[rg[0] : rg[1]] = 0
                                        obs.detdata[self.det_flags][det] |= temp
                                else:
                                    for idet, det in enumerate(obs.local_detectors):
                                        temp[:] = 0
                                        for rg in axman[key][idet].ranges():
                                            temp[rg[0] : rg[1]] = det_flag_fields[
                                                data_key
                                            ]
                                        obs.detdata[self.det_flags][det] |= temp
                            else:
                                # Explicit flags per sample
                                temp = det_flag_fields[data_key] * np.ones_like(
                                    obs.detdata[self.det_flags][:]
                                )
                                if det_flag_invert[data_key]:
                                    temp[axman[key] != 0] = 0
                                else:
                                    temp[axman[key] == 0] = 0
                                obs.detdata[self.det_flags][:] |= temp
                        else:
                            # Some other kind of detector data
                            if len(axman[key].shape) > 2:
                                shp = axman[key].shape[2:]
                            else:
                                shp = None
                            obs.detdata.create(
                                data_key,
                                sample_shape=shp,
                                dtype=axman[key].dtype,
                                units=u.dimensionless_unscaled,
                            )
                            for idet, det in enumerate(obs.local_detectors):
                                obs.detdata[data_key][idet] = axman[key][idet]
                    else:
                        # Must be some other type of object...
                        if data_key not in om:
                            om[key] = axman[key]
                elif field_axes[0] == self.axis_sample:
                    # This is shared data
                    if isinstance(axman[key], so3g.proj.Ranges):
                        # This is a set of 1D shared ranges.  Translate this to a
                        # toast interval list.
                        samplespans = list()
                        for rg in axman[key].ranges():
                            samplespans.append((rg[0], rg[1] - 1))
                        obs.intervals[data_key] = toast.intervals.IntervalList(
                            obs.shared[self.times], samplespans=samplespans
                        )
                    elif data_key in shared_ax_to_obs:
                        axbuf = None
                        if obs.comm_col_rank == 0:
                            axbuf = axman[key]
                        obs.shared[shared_ax_to_obs[data_key]].set(
                            axbuf,
                            fromrank=0,
                        )
                    elif data_key in shared_flag_fields:
                        axbuf = None
                        if obs.comm_col_rank == 0:
                            axbuf = np.array(obs.shared[self.shared_flags])
                            temp = shared_flag_fields[data_key] * np.ones_like(axbuf)
                            if shared_flag_invert[data_key]:
                                temp[axman[key] != 0] = 0
                            else:
                                temp[axman[key] == 0] = 0
                            axbuf |= temp
                        obs.shared[self.shared_flags].set(
                            axbuf,
                            fromrank=0,
                        )
                    else:
                        # This is some other shared data.
                        # FIXME: we skip this for now.  If we want to re-enable,
                        # we should pre-create this shared data or debug a
                        # deadlock that occurs when running the code as it is.
                        continue
                        obs.shared.create_column(
                            data_key,
                            shape=axman[key].shape,
                            dtype=axman[key].dtype,
                        )
                        sdata = None
                        if obs.comm_col_rank == 0:
                            sdata = axman[key]
                        obs.shared[data_key].set(sdata)
                else:
                    # Some other object...
                    if data_key not in om:
                        om[key] = axman[key]

        # Now that we have timestamps loaded, update our focalplane sample rate
        (rate, dt, dt_min, dt_max, dt_std) = toast.utils.rate_from_times(
            obs.shared[self.times].data
        )
        obs.telescope.focalplane.rate = rate * u.Hz

        if base is not None and len(om) == 0:
            # We created a dictionary that was not used, clean it up
            del obs[base]

    def _parse_meta(self, axman, obs_base, obs_meta, fp_base, fp_cols):
        if obs_base is None:
            om = obs_meta
        else:
            obs_meta[obs_base] = dict()
            om = obs_meta[obs_base]
        for key in axman.keys():
            if fp_base is not None:
                fp_key = f"{fp_base}_{key}"
            else:
                fp_key = key
            if isinstance(axman[key], AxisInterface):
                # This is one of the axes
                continue
            if isinstance(axman[key], AxisManager):
                # Descend
                self._parse_meta(axman[key], key, obs_meta, fp_key, fp_cols)
            else:
                # FIXME:  It would be nicer if this information was available
                # through a public member...
                field_axes = axman._assignments[key]
                if len(field_axes) == 0:
                    # This data is not associated with an axis.
                    om[key] = axman[key]
                elif len(field_axes) == 1 and field_axes[0] == self.axis_detector:
                    # This is a detector property
                    if fp_key in fp_cols:
                        msg = f"Context meta key '{fp_key}' is duplicated in nested"
                        msg += " AxisManagers"
                        raise RuntimeError(msg)
                    fp_cols[fp_key] = Column(name=fp_key, data=np.array(axman[key]))
        if obs_base is not None and len(om) == 0:
            # There were no meta data keys- delete this dict
            del obs_meta[obs_base]

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

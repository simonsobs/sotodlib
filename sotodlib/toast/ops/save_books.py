# Copyright (c) 2022-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import re

import numpy as np
import traitlets
from astropy import units as u

from toast.timing import function_timer, Timer
from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance
from toast.ops.operator import Operator
from toast.utils import Environment, Logger
from toast.observation import default_values as defaults

from ..io import write_book
from ...sim_hardware import telescope_tube_wafer
from ...core.hardware import build_readout_id
from ..instrument import SOFocalplane, update_creation_time


@trait_docs
class SaveBooks(Operator):
    """Export observations to Level-3 Book format.

    Create one book per tube, per observing session.  Each session consists of
    observations from the same telescope tube.  Basic metadata is written to
    Observation frames at the start of each Primary File Group and the
    M_observation.yaml file.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    book_dir = Unicode("books", help="Top-level export directory")

    focalplane_dir = Unicode(
        None, allow_none=True, help="Directory for focalplane models"
    )

    noise_dir = Unicode(None, allow_none=True, help="Directory for noise models")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

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

    boresight_azel = Unicode(
        defaults.boresight_azel, help="Observation shared key for boresight Az/El"
    )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        allow_none=True,
        help="Observation shared key for boresight RA/DEC",
    )

    corotator_angle = Unicode(
        defaults.corotator_angle,
        allow_none=True,
        help="Observation shared key for corotator_angle (if it is used)",
    )

    boresight_angle = Unicode(
        defaults.boresight_angle,
        allow_none=True,
        help="Observation shared key for boresight rotation angle (if it is used)",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, allow_none=True, help="Observation shared key for HWP angle"
    )

    frame_intervals = Unicode(
        None,
        allow_none=True,
        help="Observation interval key for frame boundaries",
    )

    creation_time = Float(None, allow_none=True, help="Last detector tuning time")

    gzip = Bool(False, help="If True, gzip compress the frame files")

    purge = Bool(False, help="If True, delete observation data as it is saved")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        if data.comm.ngroups != 1:
            msg = "SaveBooks is only usable in workflows with one process group"
            log.error(msg)
            raise RuntimeError(msg)

        out_fp_dir = self.focalplane_dir
        if self.focalplane_dir is None:
            out_fp_dir = self.book_dir

        out_nse_dir = self.noise_dir
        if self.noise_dir is None:
            out_nse_dir = self.book_dir

        create_time = self.creation_time
        if create_time is None:
            # Find the global minimum time
            for ob in data.obs:
                ob_start = ob.session.start.timestamp()
                if create_time is None:
                    create_time = ob_start
                elif ob_start < create_time:
                    create_time = ob_start

        # Update the focalplane readout_ids to match this creation time
        for ob in data.obs:
            update_creation_time(ob.telescope.focalplane.detector_data, create_time)

        # Split data by session
        data_sessions = data.split(obs_session_name=True)

        fp_cache = dict()

        for sname, sdata in data_sessions.items():
            # Create a full-telescope focalplane once (since it is slow), so that we
            # can reference subsets of detectors from that.
            tele_mat = re.match(r"^(.*?)_", sdata.obs[0].telescope.name)
            telename = tele_mat.group(1)
            if telename in fp_cache:
                full_fp = fp_cache[telename]
            else:
                full_fp = SOFocalplane(
                    sample_rate=sdata.obs[0].telescope.focalplane.sample_rate,
                    telescope=telename,
                    creation_time=create_time,
                )
                fp_cache[telename] = full_fp

            # Pass through all observations and:
            # - group observations by optics tube
            # - ensure that frame boundaries are set up

            tube_obs = dict()

            for ob in sdata.obs:
                tbs = set(ob.telescope.focalplane.detector_data["tube_slot"])
                if len(tbs) != 1:
                    msg = f"Observation {ob.name} has multiple optics tubes ({tbs})"
                    raise RuntimeError(msg)
                this_tube = tbs.pop()
                if this_tube not in tube_obs:
                    tube_obs[this_tube] = list()
                tube_obs[this_tube].append(ob)

                # Create frame intervals if not specified
                redist_sampsets = False
                frame_intervals = self.frame_intervals
                if frame_intervals is None:
                    # We are using the sample set distribution for our frame boundaries.
                    frame_intervals = "frames"
                    timespans = list()
                    offset = 0
                    n_frames = 0
                    first_set = ob.dist.samp_sets[ob.comm.group_rank].offset
                    n_set = ob.dist.samp_sets[ob.comm.group_rank].n_elem
                    for sset in range(first_set, first_set + n_set):
                        for chunk in ob.dist.sample_sets[sset]:
                            timespans.append(
                                (
                                    ob.shared[self.times][offset],
                                    ob.shared[self.times][offset + chunk - 1],
                                )
                            )
                            n_frames += 1
                            offset += chunk
                    ob.intervals.create_col(
                        frame_intervals, timespans, ob.shared[self.times]
                    )
                else:
                    # We were given an existing set of frame boundaries.  Compute new
                    # sample sets to use when redistributing.
                    if ob.comm_col_rank == 0:
                        # First row of process grid gets local chunks
                        local_sets = list()
                        offset = 0
                        for intr in ob.intervals[frame_intervals]:
                            chunk = intr.last - offset
                            local_sets.append([chunk])
                            offset += chunk
                        if offset != ob.n_local_samples:
                            local_sets.append([ob.n_local_samples - offset])
                        # Gather across the row
                        all_sets = [
                            local_sets,
                        ]
                        if ob.comm_row is not None:
                            all_sets = ob.comm_row.gather(local_sets, root=0)
                        if ob.comm_row_rank == 0:
                            redist_sampsets = list()
                            for pset in all_sets:
                                redist_sampsets.extend(pset)
                    if ob.comm.comm_group is not None:
                        redist_sampsets = ob.comm.comm_group.bcast(
                            redist_sampsets, root=0
                        )

            # Write observations for each tube

            for tube, tobs in tube_obs.items():
                log.info_rank(
                    f"Writing book for session {sname}, tube {tube}",
                    comm=sdata.comm.comm_group,
                )
                # Save it
                write_book(
                    tobs,
                    full_fp,
                    self.book_dir,
                    out_fp_dir,
                    out_nse_dir,
                    self.times,
                    self.shared_flags,
                    self.det_data,
                    self.det_flags,
                    self.boresight_azel,
                    self.boresight_radec,
                    self.corotator_angle,
                    self.boresight_angle,
                    self.hwp_angle,
                    gzip=self.gzip,
                    frame_intervals=frame_intervals,
                    redist_sampsets=redist_sampsets,
                )
                if ob.comm.comm_group is not None:
                    ob.comm.comm_group.barrier()

            # Delete our temporary frame interval if we created it
            for ob in sdata.obs:
                if self.frame_intervals is None:
                    del ob.intervals[frame_intervals]

                if ob.comm.comm_group is not None:
                    ob.comm.comm_group.barrier()

                if self.purge:
                    ob.clear()

        del data_sessions
        if self.purge:
            data.obs.clear()

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [
                self.times,
                self.boresight_azel,
            ],
            "detdata": [self.det_data],
        }
        if self.boresight_radec is not None:
            req["shared"].append(self.boresight_radec)
        if self.boresight_angle is not None:
            req["shared"].append(self.boresight_angle)
        if self.corotator_angle is not None:
            req["shared"].append(self.corotator_angle)
        if self.hwp_angle is not None:
            req["shared"].append(self.hwp_angle)
        return req

    def _provides(self):
        return dict()

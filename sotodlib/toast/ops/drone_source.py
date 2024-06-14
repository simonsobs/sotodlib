# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from astropy.table import QTable
from astropy import units as u

from toast.utils import Logger
from toast.traits import trait_docs, Int, Unicode, List
from toast.timing import function_timer
from toast.observation import default_values as defaults
from toast.ops import Operator


@trait_docs
class DroneSource(Operator):
    """Convert drone ALT / AZ location files into source coordinates.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_irregular,
        help="Bit mask value for samples outside drone data",
    )

    drone_distance = Unicode(
        "source_distance",
        help="Observation shared key for distance to the drone in meters",
    )

    drone_source = Unicode(
        "source", help="Observation shared key for drone az / el degrees"
    )

    drone_files = List(
        [], help="List of drone alt/az ecsv files"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if len(self.drone_files) == 0:
            raise RuntimeError("You must specify some drone position files")

        drone_times = None
        drone_az = None
        drone_el = None
        drone_dist = None

        if data.comm.world_rank == 0:
            # Load drone position data
            raw = list()
            for file in self.drone_files:
                tbl = QTable.read(file, format="ascii.ecsv")
                tbl_start = tbl["ctime"][0]
                raw.append({"data": tbl, "start": tbl_start})
            # Sort file data based on start time
            file_data = list(sorted(raw, key=lambda x: x["start"]))

            # Concatenate file data into contiguous arrays
            drone_times = np.concatenate(
                [x["data"]["ctime"].to_value(u.s) for x in file_data]
            )
            drone_az = np.concatenate(
                [x["data"]["Az"].to_value(u.degree) for x in file_data]
            )
            drone_el = np.concatenate(
                [x["data"]["Alt"].to_value(u.degree) for x in file_data]
            )
            drone_dist = np.concatenate(
                [x["data"]["Dist"].to_value(u.m) for x in file_data]
            )
            del file_data
            del raw
            print(f"Drone data has {len(drone_times)} samples", flush=True)

        if data.comm.comm_world is not None:
            drone_times = data.comm.comm_world.bcast(drone_times, root=0)
            drone_az = data.comm.comm_world.bcast(drone_az, root=0)
            drone_el = data.comm.comm_world.bcast(drone_el, root=0)
            drone_dist = data.comm.comm_world.bcast(drone_dist, root=0)

        for ob in data.obs:
            # Create shared data objects
            ob.shared.create_column(
                self.drone_source,
                shape=(ob.n_local_samples, 2),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.drone_distance,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            if ob.comm_col_rank == 0:
                print(f"Shared data {self.drone_source} shape {ob.shared[self.drone_source].data.shape}", flush=True)

            # First row of the process grid interpolates drone position to
            # observation timestamps.  Samples outside the drone timestamps are
            # flagged.
            if ob.comm_col_rank == 0:
                times = ob.shared[self.times].data
                az = np.interp(times, drone_times, drone_az)
                el = np.interp(times, drone_times, drone_el)
                print(f"Drone {ob.name} interpolate to {len(times)} samples", flush=True)
                print(f" az = {az} ({az.shape}), original = {drone_az} ({drone_az.shape})", flush=True)
                coord = np.column_stack([-az, el])
                print(f" coord = {coord} ({coord.shape})", flush=True)
                dist = np.interp(times, drone_times, drone_dist)
                outside = np.logical_or(
                    times < drone_times[0],
                    times > drone_times[-1],
                )
                flags = np.array(ob.shared[self.shared_flags].data)
                flags[outside] |= self.shared_flag_mask
            else:
                coord = None
                dist = None
                flags = None
            ob.shared[self.drone_source].set(coord)
            ob.shared[self.drone_distance].set(dist)
            ob.shared[self.shared_flags].set(flags)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.shared_flags],
        }
        return req

    def _provides(self):
        prov = {
            "shared": [self.drone_distance, self.drone_source],
        }
        return prov

    def _accelerators(self):
        return list()

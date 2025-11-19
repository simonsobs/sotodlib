# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

from collections.abc import MutableMapping

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from toast.utils import Logger

from ..io import hkdb


class HKManager(MutableMapping):
    """Class used to manage House Keeping data in an Observation.

    The constructor is used to populate the data given information about the
    housekeeping data and DB locations.  Once constructed, the housekeeping data is
    intended to be immutable.

    After creation, you can access a given HK field by name with standard dictionary
    syntax.  This will return a tuple of the timestamps and data array:

        hk_times, hk_data = ob.hk[field_name]

    For convenience, you can also access an equivalent array of data that is
    interpolated to the detector sampling.  The default interpolation uses the scipy
    PchipInterpolator, but you can also get linear and cubic spline interpolation:

        hk_data = ob.hk.interp(field_name) # PchipInterpolator

        hk_data = ob.hk.interp_linear(field_name) # numpy linear interpolation

        hk_data = ob.hk.interp_cubic(field_name) # cubic spline interpolation

    Args:
        comm (MPI.Comm):  The MPI communicator for the observation.
        timestamps (array):  The observation sample times.
        site_root (str):  The path to the site HK files.
        site_db (str):  The path to the site HK database.
        site_fields (list):  Restrict loading to these fields.
        site_aliases (dict):  Make aliases to fields with these names.
        plat_root (str):  The path to the platform HK files.
        plat_db (str):  The path to the platform HK database.
        plat_fields (list):  Restrict loading to these fields.
        plat_aliases (dict):  Make aliases to fields with these names.

    """

    def __init__(
        self,
        comm,
        timestamps,
        site_root=None,
        site_db=None,
        site_fields=None,
        site_aliases=None,
        plat_root=None,
        plat_db=None,
        plat_fields=None,
        plat_aliases=None,
    ):
        self._internal = dict()
        self._aliases = dict()
        self._stamps = timestamps
        self._start_time = timestamps[0]
        self._stop_time = timestamps[-1]
        rank = 0
        if comm is not None:
            rank = comm.rank

        # Load the site HK data first
        if site_root is not None:
            if site_db is None:
                raise RuntimeError("If site_root is specified, site_db is required")
            if rank == 0:
                # Load data on just one process per group
                self._load_hk_data(site_root, site_db, site_fields, site_aliases)

        # Load the plot HK data
        if plat_root is not None:
            if plat_db is None:
                raise RuntimeError("If plat_root is specified, plat_db is required")
            if rank == 0:
                # Load data on just one process per group
                self._load_hk_data(plat_root, plat_db, plat_fields, plat_aliases)

        if comm is not None:
            # Broadcast the small HK data to all processes
            self._internal = comm.bcast(self._internal, root=0)
            self._aliases = comm.bcast(self._aliases, root=0)

    def _load_hk_data(self, root, db, fields, aliases):
        """Helper function to do the actual loading."""
        log = Logger.get()
        conf = hkdb.HkConfig.from_dict(
            {
                "hk_root": root,
                "db_url": {
                    "database": db,
                    "drivername": "sqlite",
                },
                "aliases": {},
            }
        )
        # Get the full list of fields
        test_spec = hkdb.LoadSpec(
            cfg=conf,
            start=self._start_time,
            end=self._stop_time,
            fields=[],
            downsample_factor=1,
            hkdb=None,
        )
        all_fields = hkdb.get_feed_list(test_spec)
        if fields is None or len(fields) == 0:
            selected = list(all_fields)
        else:
            selected = fields
        # Load the data
        lspec = hkdb.LoadSpec(
            cfg=conf,
            start=self._start_time,
            end=self._stop_time,
            fields=selected,
            downsample_factor=1,
            hkdb=None,
        )
        result = hkdb.load_hk(lspec, show_pb=False)
        self._internal.update(result.data)
        if aliases is not None:
            for k, v in aliases.items():
                if v not in self._internal:
                    msg = f"Skipping alias ('{k}') for field '{v}', "
                    msg += "which does not exist"
                    log.warning(msg)
                else:
                    self._aliases[k] = v

    # Interpolation

    def interp(self, field):
        times, vals = self.__getitem__(field)
        itrp = PchipInterpolator(times, vals, extrapolate=True)
        return itrp(self._stamps)

    def interp_linear(self, field):
        times, vals = self.__getitem__(field)
        return np.interp(self._stamps, times, vals)

    def interp_cubic(self, field):
        times, vals = self.__getitem__(field)
        itrp = CubicSpline(times, vals, extrapolate=True)
        return itrp(self._stamps)

    def memory_use(self):
        bytes = 0
        for k, v in self._internal.items():
            bytes += v[0].nbytes
            bytes += v[1].nbytes
        return bytes

    # Mapping methods

    @property
    def fields(self):
        return list(self._internal.keys())

    @property
    def aliases(self):
        return dict(self._aliases)

    def __getitem__(self, key):
        if key in self._aliases:
            return self._internal[self._aliases[key]]
        else:
            return self._internal[key]

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete housekeeping fields after load")

    def __setitem__(self, key, value):
        raise NotImplementedError("Cannot modify raw housekeeping data")

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def __repr__(self):
        val = f"<HKManager {len(self._internal)} fields, {len(self._aliases)} aliases>"
        return val

    def __eq__(self, other):
        log = Logger.get()
        if self._internal != other._internal:
            log.verbose(f"  data {self._internal} != {other._internal}")
            return False
        if self._aliases != other._aliases:
            log.verbose(f"  aliases {self._aliases} != {other._aliases}")
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

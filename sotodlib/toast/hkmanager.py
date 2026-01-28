# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import copy
import re
from collections.abc import MutableMapping

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from toast.utils import Logger, replace_byte_arrays, array_equal
from toast.io.hdf_utils import (
    load_meta_object,
    save_meta_object,
)

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

    # Exclude the massive HWP and WG housekeeping data by default, since
    # this information is only used to build the HWP angle and WG calibration
    # metadata.  Loading these fields dramatically increases the load time
    # for normal use.  These can still be forcibly loaded by specifying them
    # in the `plat_fields` list.
    exclude_by_default = [
        r"^hwp-.*",
        r"^wg-.*",
        r"^acu.acu_udp_stream.*",
    ]

    def __init__(
        self,
        comm=None,
        timestamps=None,
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
        if timestamps is not None:
            self._start_time = timestamps[0]
            self._stop_time = timestamps[-1]
        elif site_root is not None or plat_root is not None:
            msg = "If loading site or platform data, timestamps are required"
            raise RuntimeError(msg)
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

        # Load the platform HK data
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
        exclude_pat = [re.compile(x) for x in self.exclude_by_default]
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
            # The user is requesting "all" fields.  We will prune this list to exclude
            # fields related to the HWP and wire grid encoders, which are excessively
            # large and rarely used for downstream analysis.
            selected = list()
            for fld in all_fields:
                keep = True
                for pat in exclude_pat:
                    if pat.match(fld) is not None:
                        keep = False
                if keep:
                    selected.append(fld)
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

        converted = replace_byte_arrays(result.data)

        self._internal.update(converted)
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

    def memory_use(self, threshold=0):
        """Return the total size of all fields and a string with the breakdown.

        Args:
            threshold (int):  Only include in the string objects larger than this.

        Returns:
            (tuple):  The (total size, info string)

        """
        bytes = 0
        fstr = ""
        for k, v in self._internal.items():
            fbytes = v[0].nbytes + v[1].nbytes
            bytes += fbytes
            if fbytes > threshold:
                fstr += f"{k}: {fbytes} bytes\n"
        return bytes, fstr

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
        if set(self._internal.keys()) != set(other._internal.keys()):
            log.verbose(f"  keys {self._internal} != {other._internal}")
            return False
        for k, v in self._internal.items():
            times = v[0]
            vals = v[1]
            other_times = other._internal[k][0]
            other_vals = other._internal[k][1]
            if not array_equal(times, other_times, log_prefix=f"HKManager {k} times"):
                return False
            if not array_equal(vals, other_vals, log_prefix=f"HKManager {k} values"):
                return False
        if self._aliases != other._aliases:
            log.verbose(f"  aliases {self._aliases} != {other._aliases}")
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def save_hdf5(self, handle, obs, **kwargs):
        """Save the HKManager object to an HDF5 file.

        Args:
            handle (h5py.Group):  The group to populate.
            obs (Observation):  The parent observation.

        Returns:
            None

        """
        gcomm = obs.comm.comm_group
        if (gcomm is None) or (gcomm.rank == 0):
            # The rank zero process should always be writing
            if handle is None:
                raise RuntimeError("HDF5 group is not open on the root process")
        if handle is not None:
            tdata = handle.create_dataset(
                "timestamps", self._stamps.shape, dtype=self._stamps.dtype
            )
            hslices = [slice(0, x) for x in self._stamps.shape]
            hslices = tuple(hslices)
            tdata.write_direct(self._stamps, hslices, hslices)
            del tdata
            save_meta_object(handle, "data", self._internal)
            save_meta_object(handle, "aliases", self._aliases)

    def load_hdf5(self, handle, obs, **kwargs):
        """Load the HKManager object from an HDF5 file.

        Args:
            handle (h5py.Group):  The group containing noise model.
            obs (Observation):  The parent observation.

        Returns:
            None

        """
        gcomm = obs.comm.comm_group
        if (gcomm is None) or (gcomm.rank == 0):
            # The rank zero process should always be reading
            if handle is None:
                raise RuntimeError("HDF5 group is not open on the root process")
        stamps = None
        data = None
        aliases = None
        if handle is not None:
            tdata = handle["timestamps"]
            stamps = np.copy(tdata[:])
            data = load_meta_object(handle["data"])
            aliases = load_meta_object(handle["aliases"])
            del tdata
        if gcomm is not None:
            stamps = gcomm.bcast(stamps, root=0)
            data = gcomm.bcast(data, root=0)
            aliases = gcomm.bcast(aliases, root=0)

        self._stamps = stamps
        self._start_time = stamps[0]
        self._stop_time = stamps[-1]
        self._internal = data
        self._aliases = aliases

    def duplicate(self):
        """Create a copy of the HKManager"""
        new_hk = HKManager()
        new_hk._stamps = np.copy(self._stamps)
        new_hk._start_time = new_hk._stamps[0]
        new_hk._stop_time = new_hk._stamps[-1]
        new_hk._internal = copy.deepcopy(self._internal)
        new_hk._aliases = copy.deepcopy(self._aliases)
        return new_hk

    def downsample(self, demod_times):
        """Downsample the HKManager

        The demodulation operator with downsample any observation
        attributes with a `downsample()` method.  For this class
        that operation is trivial, since it just involves replacing
        the timestamps.

        Args:
            demod_times (array):  The new downsampled timestamps.

        Returns:
            (HKManager):  The downsampled HKManager.

        """
        new_hk = HKManager()
        new_hk._stamps = demod_times
        new_hk._start_time = new_hk._stamps[0]
        new_hk._stop_time = new_hk._stamps[-1]
        new_hk._internal = copy.deepcopy(self._internal)
        new_hk._aliases = copy.deepcopy(self._aliases)
        return new_hk

# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

import numpy as np

import toast
from toast.utils import Logger
from toast.traits import trait_docs, Int, Unicode, Instance, List
from toast.timing import function_timer
from toast.intervals import IntervalList
from toast.observation import default_values as defaults
from toast.ops import Operator
from toast.pixels_io_healpix import write_healpix_fits, write_healpix_hdf5
from toast.pixels_io_wcs import write_wcs_fits


class Split(object):
    """Base class for objects implementing a particular split."""

    def __init__(self, name="N/A"):
        self._name = name
        self._split_intervals = "split"
        self._saved_det_flags = "saved_det_split_flags"

    @property
    def name(self):
        return self._name

    @property
    def split_intervals(self):
        return self._split_intervals

    def _create_split(self, obs):
        pass

    def _remove_split(self, obs):
        if self._split_intervals in obs.intervals:
            del obs.intervals[self._split_intervals]
        if self._saved_det_flags in obs:
            # We have some detector flags to restore
            obs.set_local_detector_flags(obs[self._saved_det_flags])
            del obs[self._saved_det_flags]

    def create_split(self, obs):
        self._create_split(obs)

    def remove_split(self, obs):
        self._remove_split(obs)


class SplitAll(Split):
    """Split to process all data."""

    def __init__(self, name, interval=defaults.scanning_interval):
        super().__init__(name)
        self._interval = interval

    def name(self):
        return "all"

    def _create_split(self, obs):
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitLeftGoing(Split):
    """Split to process only left-going scans."""

    def __init__(self, name, interval=defaults.scan_rightleft_interval):
        super().__init__(name)
        self._interval = interval

    def name(self):
        return "left_going"

    def _create_split(self, obs):
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitRightGoing(Split):
    """Split to process only right-going scans."""

    def __init__(self, name, interval=defaults.scan_leftright_interval):
        super().__init__(name)
        self._interval = interval

    def name(self):
        return "right_going"

    def _create_split(self, obs):
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


@trait_docs
class Splits(Operator):
    """Apply a sequence of splits to the data, and make a map of each."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    mapmaker = Instance(
        klass=Operator,
        allow_none=True,
        help="The mapmaking operator to use",
    )

    output_dir = Unicode(
        ".",
        help="Top-level output directory",
    )

    splits = List([], help="The list of named splits to apply")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.mapmaker is None:
            msg = "You must specify the mapmaker operator"
            log.error(msg)
            raise RuntimeError(msg)

        # Build splits we are using
        self._split_obj = dict()
        for split_name in self.splits:
            if split_name == "all":
                self._split_obj[split_name] = SplitAll(split_name)
            elif split_name == "left_going":
                self._split_obj[split_name] = SplitLeftGoing(split_name)
            elif split_name == "right_going":
                self._split_obj[split_name] = SplitRightGoing(split_name)
            else:
                msg = f"Unsupported split '{split_name}'"
                log.error(msg)
                raise RuntimeError(msg)

        # Save starting mapmaker parameters, to restore later
        mapmaker_save_traits = dict()
        for trait_name, trait in self.mapmaker.traits().items():
            mapmaker_save_traits[trait_name] = trait.get(self.mapmaker)

        # Possible traits that we want to toggle off
        mapmaker_disable_traits = [
            "write_binmap",
            "write_map",
            "write_noiseweighted_map",
            "write_hits",
            "write_cov",
            "write_invcov",
            "write_rcond",
            "write_solver_products",
            "save_cleaned",
        ]

        # Possible traits we want to enable
        mapmaker_enable_traits = [
            "keep_final_products",
            "reset_pix_dist",
        ]

        for trt in mapmaker_disable_traits:
            if hasattr(self.mapmaker, trt):
                setattr(self.mapmaker, trt, False)

        for trt in mapmaker_enable_traits:
            if hasattr(self.mapmaker, trt):
                setattr(self.mapmaker, trt, True)

        # Loop over splits
        for split_name, spl in self._split_obj.items():
            # Set mapmaker name based on split and the name of this
            # Splits operator.
            mname = f"{self.name}_{split_name}"
            self.mapmaker.name = mname

            # Apply this split
            for ob in data.obs:
                spl.create_split(ob)

            # Set mapmaking tools to use the current split interval list
            map_binner = self.mapmaker.map_binning
            map_binner.pixel_pointing.view = spl.split_intervals
            toast.ops.Delete(
                detdata=[
                    map_binner.pixel_pointing.pixels,
                    map_binner.stokes_weights.weights,
                    map_binner.pixel_pointing.detector_pointing.quats,
                ],
            ).apply(data)

            # Run mapmaking
            self.mapmaker.apply(data)

            # Write
            self.write_splits(data, split_name=split_name)

            # Remove split
            for ob in data.obs:
                spl.remove_split(ob)

        # Restore mapmaker traits
        for k, v in mapmaker_save_traits.items():
            setattr(self.mapmaker, k, v)

    def write_splits(self, data, split_name=None):
        """Write out all split products."""
        if not hasattr(self, "_split_obj"):
            msg = "No splits have been created yet, cannot write"
            raise RuntimeError(msg)

        is_pix_wcs = hasattr(self.mapmaker.map_binning.pixel_pointing, "wcs")
        is_hpix_nest = None
        if not is_pix_wcs:
            is_hpix_nest = self.mapmaker.map_binning.pixel_pointing.nest

        if split_name is None:
            to_write = dict(self._split_obj)
        else:
            to_write = {split_name: self._split_obj[split_name]}

        for spname, spl in to_write.items():
            mname = f"{self.name}_{split_name}"
            for prod in ["hits", "map", "invcov", "noiseweighted_map"]:
                mkey = f"{mname}_{prod}"
                if is_pix_wcs:
                    fname = os.path.join(
                        self.output_dir, f"{self.name}_{split_name}_{prod}.fits"
                    )
                    # FIXME: add single precision option to upstream function
                    write_wcs_fits(data[mkey], fname)
                else:
                    if self.mapmaker.write_hdf5:
                        # Non-standard HDF5 output
                        fname = os.path.join(
                            self.output_dir, f"{self.name}_{split_name}_{prod}.h5"
                        )
                        write_healpix_hdf5(
                            data[mkey],
                            fname,
                            nest=is_hpix_nest,
                            single_precision=True,
                            force_serial=self.mapmaker.write_hdf5_serial,
                        )
                    else:
                        # Standard FITS output
                        fname = os.path.join(
                            self.output_dir, f"{self.name}_{split_name}_{prod}.fits"
                        )
                        write_healpix_fits(
                            data[mkey],
                            fname,
                            nest=is_hpix_nest,
                            single_precision=True,
                        )
            # Clean up all products for this split
            for prod in [
                "hits",
                "cov",
                "invcov",
                "rcond",
                "flags",
                "cleaned",
                "binmap",
                "map",
                "noiseweighted_map",
            ]:
                mkey = f"{mname}_{prod}"
                if mkey in data:
                    del data[mkey]

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "detdata": self.det_data,
        }
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()

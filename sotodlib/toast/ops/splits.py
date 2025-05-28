# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

import astropy.units as u
import numpy as np

import toast
from toast.utils import Logger
from toast.traits import trait_docs, Int, Unicode, Instance, List, Bool
from toast.timing import function_timer
from toast.intervals import IntervalList
from toast.observation import default_values as defaults
from toast.ops import Operator
from toast.pixels_io_healpix import write_healpix_fits, write_healpix_hdf5
from toast.pixels_io_wcs import write_wcs_fits
from toast import qarray as qa


class Split(object):
    """Base class for objects implementing a particular split.

    Derived classes should implement the `_create_split()` method and optionally
    the `_remove_split()` method.

    When the `_create_split` method is called, the detector flags have already
    been backed up and the derived class can modify them to exclude detectors by
    setting the invalid mask bit.  If the derived class does not set the split
    sample intervals, then all samples will be used.  Many splits simply make a
    copy of an existing interval list to use for the split.  However, anything
    could be done here, including logical operations with multiple interval lists
    or constructing a new interval list from other metadata.

    The `_remove_split` method is only needed if any other observation cleanup
    is needed.  After this method is called, the splits interval will be deleted
    and the detector flags restored.

    """
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
        pass

    def _delete_split(self, obs):
        """Remove any split products that exist."""
        if self._split_intervals in obs.intervals:
            del obs.intervals[self._split_intervals]
        if self._saved_det_flags in obs:
            del obs[self._saved_det_flags]

    def create_split(self, obs):
        """Create the split in the given observation.

        Args:
            obs (Observation):  The observation to modify.

        Returns:
            None

        """
        # Clear any existing split products.  These should not exist
        # if previous splits were properly removed.
        self._delete_split(obs)

        # Backup detector flags
        obs[self._saved_det_flags] = dict(obs.local_detector_flags)

        # Call split-specific creation
        self._create_split(obs)

        # If the split interval was not setup by the derived class, set it
        # to None (all samples).
        if self._split_intervals not in obs.intervals:
            obs.intervals[self._split_intervals] = IntervalList(
                intervals=obs.intervals[None]
            )

    def remove_split(self, obs):
        """Remove the split from the given observation.

        Args:
            obs (Observation):  The observation to modify.

        Returns:
            None

        """
        # Perform any split-specific operations
        self._remove_split(obs)

        # Restore detector flags and remove the splits interval.
        obs.set_local_detector_flags(obs[self._saved_det_flags])
        self._delete_split(obs)

    @staticmethod
    def create(name, **kwargs):
        """Factory method to instantiate splits.

        The name is the hard-coded name of each split derived class.
        any kwargs are passed to the constructor.  This function is
        the one place to register new split types and their names.

        Args:
            name (str):  The name of the split.  If a list is passed, then
                the SplitByList split is created.

        """
        log = Logger.get()
        if isinstance(name, list):
            # The "real" name should be in kwargs
            split_name = kwargs["split_name"]
            del kwargs["split_name"]
            kwargs["dets"] = name
            return SplitByList(split_name, **kwargs)
        elif name == "all":
            return SplitAll(name, **kwargs)
        elif name == "left_going":
            return SplitLeftGoing(name, **kwargs)
        elif name == "right_going":
            return SplitRightGoing(name, **kwargs)
        elif name == "outer_detectors":
            return SplitOuterDetectors(name, **kwargs)
        elif name == "inner_detectors":
            return SplitInnerDetectors(name, **kwargs)
        elif name == "polA":
            return SplitPolA(name, **kwargs)
        elif name == "polB":
            return SplitPolB(name, **kwargs)
        else:
            msg = f"Unsupported split '{name}'"
            log.error(msg)
            raise RuntimeError(msg)


class SplitAll(Split):
    """Split to process all data.

    Args:
        name (str):  Name of the split.
        interval (str):  Name of the interval to use.

    """
    def __init__(self, name, interval=defaults.scanning_interval):
        super().__init__(name)
        self._interval = interval

    def _create_split(self, obs):
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitLeftGoing(Split):
    """Split to process only left-going scans.

    Args:
        name (str):  Name of the split.
        interval (str):  Name of the interval to use.

    """
    def __init__(self, name, interval=defaults.scan_rightleft_interval):
        super().__init__(name)
        self._interval = interval

    def _create_split(self, obs):
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitRightGoing(Split):
    """Split to process only right-going scans.

    Args:
        name (str):  Name of the split.
        interval (str):  Name of the interval to use.

    """
    def __init__(self, name, interval=defaults.scan_leftright_interval):
        super().__init__(name)
        self._interval = interval

    def _create_split(self, obs):
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitOuterDetectors(Split):
    """Split to process only detectors on the edge of the focalplane.

    Args:
        name (str):  Name of the split.
        radial_cut (float):  Radial distance from boresight to cut in
            units of FOV/2.  Detectors inside this are cut.
        interval (str):  Name of the interval to use.

    """
    def __init__(
        self,
        name,
        radial_cut=0.7071067811865476,
        interval=defaults.scanning_interval,
    ):
        super().__init__(name)
        self._radial_cut = radial_cut
        self._interval = interval

    def _create_split(self, obs):
        focalplane = obs.telescope.focalplane
        fp_radius = focalplane.field_of_view / 2
        limit = fp_radius.to_value(u.rad) * self._radial_cut

        # Flag detectors
        flags = dict(obs.local_detector_flags)
        for det in obs.select_local_detectors():
            det_quat = focalplane[det]["quat"]
            det_theta, det_phi, det_psi = qa.to_iso_angles(det_quat)
            if det_theta < limit:
                # Cut the detector
                flags[det] |= defaults.det_mask_invalid
        obs.set_local_detector_flags(flags)

        # Set the split intervals to be a simple copy of the input interval list.
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitInnerDetectors(Split):
    """Split to process only detectors on the inner part of the focalplane.

    Args:
        name (str):  Name of the split.
        radial_cut (float):  Radial distance from boresight to cut in
            units of FOV/2.  Detectors outside this are cut.
        interval (str):  Name of the interval to use.

    """
    def __init__(
        self,
        name,
        radial_cut=0.7071067811865476,
        interval=defaults.scanning_interval,
    ):
        super().__init__(name)
        self._radial_cut = radial_cut
        self._interval = interval

    def _create_split(self, obs):
        focalplane = obs.telescope.focalplane
        fp_radius = focalplane.field_of_view / 2
        limit = fp_radius.to_value(u.rad) * self._radial_cut

        # Flag detectors
        flags = dict(obs.local_detector_flags)
        for det in obs.select_local_detectors():
            det_quat = focalplane[det]["quat"]
            det_theta, det_phi, det_psi = qa.to_iso_angles(det_quat)
            if det_theta >= limit:
                # Cut the detector
                flags[det] |= defaults.det_mask_invalid
        obs.set_local_detector_flags(flags)

        # Set the split intervals to be a simple copy of the input interval list.
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitByList(Split):
    """Split to process only an explicit list of detectors.

    If dets is None, all detectors are used.

    Args:
        name (str):  Name of the split.
        dets (list) : List of detectors to process.
        interval (str):  Name of the interval to use.

    """
    def __init__(
        self,
        name,
        dets=None,
        interval=defaults.scanning_interval,
    ):
        super().__init__(name)
        # Convert to a set for faster lookup.
        if dets is None:
            self._dets = None
        else:
            self._dets = set(dets)
        self._interval = interval
        # Add all possible demodulated names of the detector
        demod_dets = set()
        for det in self._dets:
            for prefix in ["demod0", "demo2r", "demod2i", "demod4r", "demod4i"]:
                demod_dets.add(f"{prefix}_{det}")
        self._dets.update(demod_dets)

    def _create_split(self, obs):
        # Flag detectors
        if self._dets is not None:
            # We have some selection
            flags = dict(obs.local_detector_flags)
            for det in obs.select_local_detectors():
                if det not in self._dets:
                    # Not in the list, cut it.
                    flags[det] |= defaults.det_mask_invalid
            obs.set_local_detector_flags(flags)

        # Set the split intervals to be a simple copy of the input interval list.
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitPolA(Split):
    """Split to process only detector with A polarization.

    Args:
        name (str):  Name of the split.
        interval (str):  Name of the interval to use.

    """
    def __init__(self, name, interval=defaults.scanning_interval):
        super().__init__(name)
        self._interval = interval

    def _create_split(self, obs):
        focalplane = obs.telescope.focalplane

        # Flag detectors
        flags = dict(obs.local_detector_flags)
        for det in obs.select_local_detectors():
            pol = focalplane[det]["pol"]
            if pol != "A":
                # Cut the detector
                flags[det] |= defaults.det_mask_invalid
        obs.set_local_detector_flags(flags)

        # Set the split intervals to be a simple copy of the input interval list.
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


class SplitPolB(Split):
    """Split to process only detector with B polarization.

    Args:
        name (str):  Name of the split.
        interval (str):  Name of the interval to use.

    """
    def __init__(self, name, interval=defaults.scanning_interval):
        super().__init__(name)
        self._interval = interval

    def _create_split(self, obs):
        focalplane = obs.telescope.focalplane

        # Flag detectors
        flags = dict(obs.local_detector_flags)
        for det in obs.select_local_detectors():
            pol = focalplane[det]["pol"]
            if pol != "B":
                # Cut the detector
                flags[det] |= defaults.det_mask_invalid
        obs.set_local_detector_flags(flags)

        # Set the split intervals to be a simple copy of the input interval list.
        timespans = [(x.start, x.stop) for x in obs.intervals[self._interval]]
        obs.intervals[self._split_intervals] = IntervalList(
            obs.shared[defaults.times],
            timespans=timespans,
        )


@trait_docs
class Splits(Operator):
    """Apply a sequence of splits to the data, and make a map of each.

    The list of `splits` contains strings, where each string is either
    the name of the split, or is the name of the split followed by a
    string representation of a dictionary to pass as kwargs to the split
    constructor.

    """

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
        for split_arg in self.splits:
            # Each element of the list might be just a name, or it might
            # be a name followed by a dictionary of kwargs.
            split_opts = split_arg.split(":")
            if len(split_opts) > 1:
                split_name = split_opts[0]
                # Join the rest of the string, use python eval to convert
                # to a dictionary.
                split_kw = dict()
                raise NotImplementedError("Parsing of extra split kwargs not complete")
            else:
                split_name = split_opts[0]
                split_kw = dict()
            if os.path.isfile(split_name):
                # This is a file containing the list of detectors to use.
                # Build the split name from the path.
                name, ext = os.path.splitext(os.path.basename(split_name))
                det_list = None
                if data.comm.world_rank == 0:
                    det_list = list()
                    with open(split_name, "r") as f:
                        for line in f:
                            det_list.append(line.strip())
                if data.comm.comm_world is not None:
                    det_list = data.comm.comm_world.bcast(det_list, root=0)
                split_kw["split_name"] = name
                self._split_obj[name] = Split.create(det_list, **split_kw)
            else:
                # Just a normal split type
                self._split_obj[split_name] = Split.create(split_name, **split_kw)

        if data.comm.world_rank == 0:
            msg = "Using splits: "
            for sname in self._split_obj.keys():
                msg += f"{sname}, "
            log.info(msg)

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

        if hasattr(self.mapmaker, "map_binning"):
            map_binner = self.mapmaker.map_binning
        else:
            map_binner = self.mapmaker.binning
        pointing_view_save = map_binner.pixel_pointing.view

        # If the pixel distribution does yet exit, we create it once prior to
        # doing any splits.
        if map_binner.pixel_dist not in data:
            pix_dist = toast.ops.BuildPixelDistribution(
                pixel_dist=map_binner.pixel_dist,
                pixel_pointing=map_binner.pixel_pointing,
                save_pointing=map_binner.full_pointing,
            )
            pix_dist.apply(data)

        for trt in mapmaker_disable_traits:
            if hasattr(self.mapmaker, trt):
                setattr(self.mapmaker, trt, False)

        for trt in mapmaker_enable_traits:
            if hasattr(self.mapmaker, trt):
                setattr(self.mapmaker, trt, True)

        # Loop over splits
        for split_name, spl in self._split_obj.items():
            log.info_rank(f"Running Split '{split_name}'", comm=data.comm.comm_world)
            # Set mapmaker name based on split and the name of this
            # Splits operator.
            mname = f"{self.name}_{split_name}"
            self.mapmaker.name = mname

            # Apply this split
            for ob in data.obs:
                spl.create_split(ob)

            # Set mapmaking tools to use the current split interval list
            map_binner.pixel_pointing.view = spl.split_intervals
            if not map_binner.full_pointing:
                # We are not using full pointing and so we clear the
                # residual pointing for this split
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
        map_binner.pixel_pointing.view = pointing_view_save

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

# Copyright (c) 2020-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

from astropy import units as u
import traitlets
import numpy as np
import healpy as hp
import h5py

from toast.utils import Logger
import toast.qarray as qa
from toast.traits import trait_docs, Int, Unicode, Bool, Float, Instance
from toast.timing import function_timer
from toast.observation import default_values as defaults
from toast.ops import (
    Operator, BuildPixelDistribution, BuildNoiseWeighted, BuildHitMap
)
from toast.pixels import PixelData
from toast.pixels_io_healpix import collect_healpix_submaps
from toast.pixels_io_wcs import collect_wcs_submaps


@trait_docs
class Hn(Operator):
    """Evaluate geometrical h_n factors to support map-based simulations.

    For every detector, the noise-weighted average of exp(i n psi) is binned
    into maps for nmin <= n <= nmax:

        h_n(p) = sum_t w_t exp(i n psi_t) / n_hits(p)

    from psi_t, the orientation of the detector polarization axis in the sky 
    frame at time t, and n_hits, the number of hits in pixel p. 

    The maps are dimensionless.  One HDF5 file is written per detector:
    one group "n,m" per spin (m = 0, the HWP spin, is not used yet) holding
    the datasets "Re" (cos part) and "Im" (sin part).  Group "0,0" (written
    when nmin == 0) holds the hit count in "Re" and zeros in "Im".  Pixels
    with no hits are set to zero. 

    The output files are written as h5, hdf, or hdf5 files, and ordered as 
    - The name of the file contains the detector name, and the pair (A or B) of the detector.
    - Then, the h-map can be accessed by indexing "n,0" for the spin n
    - Finally, the real and imaginary parts of the h-map can be accessed by 
      indexing "Re" and "Im" respectively.  

    Each file also has the following attributes:
    - "pol_ang_rad": polarization angle in the focalplane ("pol_ang").
    - "wafer_x_mm", "wafer_y_mm": position of the detector in its wafer
      ("wafer_x", "wafer_y"), relative to the wafer centre.
    - "duration_s": total duration of the observations of this detector.
    - "sampling_rate_Hz": sampling rate of the observations of this detector.
    - "pixelization": "HEALPIX" or "WCS", depending on the pixelization used.
    - "coordinate_system": the coordinate system of the pixelization.
    
    If the pixelization is HEALPIX, the following attributes are also present:
    - "nside": the nside of the HEALPIX pixelization.
    - "ordering": the ordering of the HEALPIX pixelization ("RING" or "NESTED").

    If the pixelization is WCS, the following attributes are also present:
    - "wcs_header": the WCS header of the pixelization. 
    - "wcs_shape": the shape of the WCS pixelization.

    
    
    Example for building the h-maps of a single detector: 
        >>> name_hmap = 'h_n_{detector_name}.hdf'
        >>> with h5py.File(name_hmap, 'r') as file:
                h_n_map = file["n,0"]["Re"] + 1j * file["n,0"]["Im"] 
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pixel pointing operator",
    )

    hwp_angle = Unicode(
        None, allow_none=True, help="Observation shared key for HWP angle"
    )

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object should be stored",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional telescope flagging",
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    noise_model = Unicode(
        "noise_model",
        allow_None=True,
        help="Observation key containing the noise model",
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    save_pointing = Bool(
        False, help="If True, do not clear detector pointing matrices after use"
    )

    include_pol_angle = Bool(
        False,
        help="If True, include the polarization angle in the angle used to compute h_n",
    )

    use_single_precision = Bool(
        False, help="If True, write output maps in single precision"
    )

    file_format = Unicode(
        "h5", help="Extension of the output HDF5 files: 'h5', 'hdf', or 'hdf5'"
    )

    nmin = Int(0, help="Minimum `n` to evaluate.")
    nmax = Int(4, help="Maximum `n` to evaluate.")

    hweight_name = "h_n_weights"
    ones_name = "h_n_ones"
    h_n_map = "h_n_map"
    detector_hits_map = "h_n_hits"

    @traitlets.validate("nmin")
    def _check_min(self, proposal):
        nmin = proposal["value"]
        if nmin < 0:
            raise traitlets.TraitError("Nmin should be greater than 0")
        return nmin

    @traitlets.validate("nmax")
    def _check_max(self, proposal):
        nmax = proposal["value"]
        if nmax < 1:
            raise traitlets.TraitError("Nmax should be greater than 0")
        return nmax

    @traitlets.validate("file_format")
    def _check_file_format(self, proposal):
        file_format = proposal["value"]
        if file_format not in ("h5", "hdf", "hdf5"):
            raise traitlets.TraitError("File format should be 'h5', 'hdf', or 'hdf5'")
        return file_format

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pixels = proposal["value"]
        if pixels is not None:
            if not isinstance(pixels, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pixels.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pixels

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _spins(self):
        """The spins n >= 1 that need a cos / sin column."""
        return list(range(max(self.nmin, 1), self.nmax + 1))

    @function_timer
    def _get_psi(self, obs, det):
        """Return the psi angle of `det` in the sky frame at every sample of `obs`, in radians.
        psi is also called parallactic angle. If `include_pol_angle` is False, 
        the polarization angle of the detector in the focalplane is subtracted."""
        quats = obs.detdata[self.pixel_pointing.detector_pointing.quats][det]
        _, _, psi = qa.to_iso_angles(quats)
        if not self.include_pol_angle:
            # Remove the polarization angle
            psi -= obs.telescope.focalplane[det]["pol_ang"].to_value(u.rad)
        return psi

    @function_timer
    def _compute_weights(self, obs, det):
        """Store [1, cos(n psi), sin(n psi), ...] as the pointing weights of `det`
        in one observation.

        The binning is then done in a single pass for all spins.  The "TOD"
        is a constant 1 K: BuildNoiseWeighted needs temperature units to apply
        the noise weights, and the unit cancels in the normalization.
        """
        spins = self._spins()
        nnz = 1 + 2 * len(spins)
        obs.detdata.ensure(
            self.ones_name, detectors=[det], create_units=defaults.det_data_units
        )
        obs.detdata[self.ones_name][det] = 1.0
        obs.detdata.ensure(
            self.hweight_name, sample_shape=(nnz,), detectors=[det]
        )
        weights = obs.detdata[self.hweight_name][det]
        weights[:, 0] = 1.0
        if len(spins) > 0:
            psi = self._get_psi(obs, det)
            for i, n in enumerate(spins):
                weights[:, 1 + 2 * i] = np.cos(n * psi)
                weights[:, 2 + 2 * i] = np.sin(n * psi)
        return

    @function_timer
    def _accumulate(self, data, det):
        """Bin the weights and hits of `det`.  Collective over all processes.

        The weights only exist for one observation at a time: each observation
        is binned with `exec()` and its weights deleted before the next one.
        The maps are synchronized once, by `finalize()`.
        """
        # Create the maps on every process: `finalize()` is collective, and
        # only syncs maps that exist.
        dist = data[self.pixel_dist]
        data[self.h_n_map] = PixelData(
            dist,
            np.float64,
            n_value=1 + 2 * len(self._spins()),
            units=1.0 / defaults.det_data_units,
        )
        data[self.detector_hits_map] = PixelData(dist, np.int64, n_value=1)
        flag_kwargs = dict(
            pixel_dist=self.pixel_dist,
            view=self.pixel_pointing.view,
            pixels=self.pixel_pointing.pixels,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )
        build_zmap = BuildNoiseWeighted(
            zmap=self.h_n_map,
            weights=self.hweight_name,
            noise_model=self.noise_model,
            det_data=self.ones_name,
            **flag_kwargs,
        )
        build_hits = BuildHitMap(hits=self.detector_hits_map, **flag_kwargs)
        for obs in data.obs:
            # Same detector selection as the toast binning operators: a flagged
            # detector has no pointing in this observation.
            dets = obs.select_local_detectors(
                selection=[det], flagmask=defaults.det_mask_nonscience
            )
            if det not in dets:
                continue
            obs_data = data.select(obs_uid=obs.uid)
            self.pixel_pointing.detector_pointing.apply(obs_data, detectors=[det])
            self.pixel_pointing.apply(obs_data, detectors=[det])
            self._compute_weights(obs, det)
            build_zmap.exec(obs_data, detectors=[det])
            build_hits.exec(obs_data, detectors=[det])
            for name in self.hweight_name, self.ones_name:
                del obs.detdata[name]
        build_zmap.finalize(data)
        build_hits.finalize(data)
        return

    def _collect(self, pixel_data):
        """Gather a distributed map on the root process.

        Returns an array of shape (n_value, npix) in RING ordering for Healpix
        or (n_value, n_row, n_col) for WCS, and None on the other processes.
        """
        dist = pixel_data.distribution
        if getattr(dist, "wcs", None) is not None:
            return collect_wcs_submaps(pixel_data)
        _, fview = collect_healpix_submaps(pixel_data)
        if fview is None:
            return None
        maps = np.array(fview)
        if dist.nest:
            maps = hp.reorder(maps, n2r=True)
        return maps


    def _focalplane_attrs(self, focalplane, det):
        """Polarization angle and positions of `det` (see the class docstring)."""
        attrs = {
            "pol_ang_rad": float(focalplane[det]["pol_ang"].to_value(u.rad)),
            "wafer_x_mm": float(focalplane[det]["wafer_x"].to_value(u.mm)),
            "wafer_y_mm": float(focalplane[det]["wafer_y"].to_value(u.mm)),
        }
        return attrs

    @function_timer
    def _gather_det_info(self, data):
        """Return {det: attributes} on the root process: the total duration
        and the sampling rate of the observations of each detector, and its
        focalplane properties."""
        scans = dict()
        props = dict()
        for obs in data.obs:
            focalplane = obs.telescope.focalplane
            rate = focalplane.sample_rate.to_value(u.Hz)
            duration = obs.n_all_samples / rate
            for det in obs.local_detectors:
                scans.setdefault(det, dict())[obs.uid] = (duration, rate)
                if det not in props:
                    props[det] = self._focalplane_attrs(focalplane, det)
        comm = data.comm.comm_world
        local = (scans, props)
        all_local = [local] if comm is None else comm.gather(local)
        if all_local is None:
            return None
        merged = dict()
        info = dict()
        for proc_scans, proc_props in all_local:
            for det, obs_info in proc_scans.items():
                merged.setdefault(det, dict()).update(obs_info)
            info.update(proc_props)
        for det, obs_info in merged.items():
            durations, rates = zip(*obs_info.values())
            info[det]["duration_s"] = float(np.sum(durations))
            info[det]["sampling_rate_Hz"] = float(rates[0])
        return info

    def _geometry_attrs(self, dist):
        if getattr(dist, "wcs", None) is not None:
            return {
                "pixelization": "WCS",
                "coordinate_system": str(
                    getattr(self.pixel_pointing, "coord_frame", "unknown")
                ),
                "wcs_header": dist.wcs.to_header_string(),
                "wcs_shape": tuple(dist.wcs_shape),
            }
        coord = self.pixel_pointing.detector_pointing.coord_out
        return {
            "pixelization": "HEALPIX",
            "coordinate_system": "unknown" if coord is None else str(coord),
            "nside": hp.npix2nside(dist.n_pix),
            "ordering": "RING",
        }

    @function_timer
    def _write(self, fname, det, maps, hits, attrs):
        """Write the h_n maps of one detector (root process only)."""
        dtype = np.float32 if self.use_single_precision else np.float64
        norm = maps[0]
        seen = norm != 0
        with h5py.File(fname, "w") as file:
            for key, value in attrs.items():
                file.attrs[key] = value
            if self.nmin == 0:
                group = file.create_group("0,0")
                group.create_dataset("Re", data=hits.astype(dtype))
                group.create_dataset("Im", data=np.zeros(hits.shape, dtype=dtype))
            for i, spin in enumerate(self._spins()):
                group = file.create_group(f"{spin},0")
                for name, col in ("Re", 1 + 2 * i), ("Im", 2 + 2 * i):
                    h_map = np.zeros(norm.shape, dtype=np.float64)
                    h_map[seen] = maps[col][seen] / norm[seen]
                    group.create_dataset(name, data=h_map.astype(dtype))
        return

    @function_timer
    def _get_detectors(self, data):
        """Find a super set of detectors across all processes in the
        communicator.

        The list is sorted so that every process loops over the detectors in
        the same order: the per-detector map accumulation is collective, and a
        set does not have a reproducible iteration order across processes.
        """
        my_detectors = set(data.all_local_detectors())
        comm = data.comm.comm_world
        if comm is not None:
            for detectors in comm.allgather(my_detectors):
                my_detectors |= detectors
        return sorted(my_detectors)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "pixel_dist", "noise_model":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        is_root = data.comm.world_rank == 0
        if is_root:
            os.makedirs(self.output_dir, exist_ok=True)

        # To accumulate, we need the pixel distribution.

        if self.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                shared_flags=self.shared_flags,
                shared_flag_mask=self.shared_flag_mask,
                save_pointing=self.save_pointing,
            )
            log.info_rank("Caching pixel distribution", comm=data.comm.comm_world)
            pix_dist.apply(data)

        all_detectors = self._get_detectors(data)
        det_info = self._gather_det_info(data)
        geometry = self._geometry_attrs(data[self.pixel_dist])

        for det in all_detectors:
            self._accumulate(data, det)
            maps = self._collect(data[self.h_n_map])
            hits = self._collect(data[self.detector_hits_map])

            if is_root:
                attrs = dict(geometry)
                attrs.update(
                    {
                        "det": str(det),
                        "nmin": self.nmin,
                        "nmax": self.nmax,
                        "include_pol_angle": self.include_pol_angle,
                    }
                )
                attrs.update(det_info[det])
                fname = os.path.join(
                    self.output_dir, f"{self.name}_{det}.{self.file_format}"
                )
                self._write(fname, det, maps, hits[0], attrs)
                log.info(f"Wrote h_n maps to {fname}")

        for key in self.h_n_map, self.detector_hits_map:
            if key in data:
                del data[key]
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req["meta"].extend([self.noise_model])
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return {"meta": list(), "shared": list(), "detdata": list()}

    def _accelerators(self):
        return list()

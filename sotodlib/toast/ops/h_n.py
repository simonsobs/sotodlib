# Copyright (c) 2020-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os

from astropy import units as u
import traitlets
import numpy as np
import healpy as hp

from toast.utils import Logger, memreport
from toast.mpi import MPI
import toast.qarray as qa
from toast.traits import trait_docs, Int, Unicode, Bool, Float, Instance
from toast.timing import function_timer, Timer
from toast.covariance import covariance_invert, covariance_apply
from toast.observation import default_values as defaults
from toast.ops import (
    Operator, BuildPixelDistribution, BuildInverseCovariance, BuildNoiseWeighted
)
from toast.pixels_io_healpix import collect_healpix_submaps


@trait_docs
class Hn(Operator):
    """Evaluate geometrical h_n factors to support map-based simulations.

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

    rcond_threshold = Float(
        1.0e-8, help="Minimum value for inverse condition number cut."
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
        "npy", help="File format for output maps: 'npy', 'fits', or 'hdf5'"
    )

    nmin = Int(0, help="Minimum `n` to evaluate.")
    nmax = Int(4, help="Maximum `n` to evaluate.")

    cos_1_name = "cos_1"
    sin_1_name = "sin_1"
    cos_n_name = "cos_n"
    sin_n_name = "sin_n"
    hweight_name = "hweight"
    h_n_map = "h_n_map"
    covariance = "h_n_cov"

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
        if file_format not in ("npy", "fits", "hdf5"):
            raise traitlets.TraitError("File format should be 'npy', 'fits', or 'hdf5'")
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

    def _write_h_n_map_npy(
        self,
        pixel_data,
        path,
        comm_bytes=10000000,
        report_memory=False,
        single_precision=False,
    ):
        """Write distributed PixelData to a numpy NPY file.

        Args:
            path (str): The path to the output file (NPY).
            comm_bytes (int): The approximate message size to use.
            report_memory (bool): Report the amount of available memory on
                the root node just before writing out the map.
            single_precision (bool): If True, write floats and integers in single
                precision.

        Returns:
            None

        """
        log = Logger.get()

        # The distribution
        dist = pixel_data.distribution
        rank = 0
        if dist.comm is not None:
            rank = dist.comm.rank

        # Healpix PixelDistribution should have the nest information.
        nest = dist.nest

        # Unit string to write
        # if pixel_data.units == u.K:
        #     funits = "K"
        # elif pixel_data.units == u.mK:
        #     funits = "mK"
        # elif pixel_data.units == u.uK:
        #     funits = "uK"
        # else:
        #     funits = str(pixel_data.units)

        fdata, fview = collect_healpix_submaps(pixel_data, comm_bytes=comm_bytes)

        if rank == 0:
            if os.path.isfile(path):
                os.remove(path)
            dtype = pixel_data.dtype
            if single_precision:
                if dtype == np.dtype(np.float64):
                    dtype = np.float32
                elif dtype == np.dtype(np.int64):
                    dtype = np.int32
            if report_memory:
                mem = memreport(msg="(root node)", silent=True)
                log.info(f"About to write {path}:  {mem}")
            # extra = [(f"TUNIT{x}", f"{funits}") for x in range(self.n_value)]
            # hp.write_map(
            #     path,
            #     fview,
            #     dtype=dtype,
            #     fits_IDL=False,
            #     nest=nest,
            #     extra_header=extra,
            # )
            # if not os.path.isfile(path):
            #     mode = "w+"
            # else:
            #     # print("N value", n, "path", path, flush=True)
            #     mode = "r+"
            # Create a memmap array to write the data
            file_memmap = np.memmap(
                path,
                mode="w+",
                shape=(fview[0].shape[0]),
                dtype=dtype,
            )
            file_memmap[:] = hp.reorder(fview[:], n2r=nest)
            file_memmap.flush()

            del fview
            # for col in range(self.n_value):
            fdata[0].clear()
            del fdata


    @function_timer
    def _get_h_n(self, data, n, det):
        """ Compute and store the next order of h_n
        """
        for obs in data.obs:
            if det not in obs.local_detectors:
                continue
            # HWP angle is not yet used but will be needed soon
            if self.hwp_angle is not None:
                hwpang = obs.shared[self.hwp_angle]
            else:
                hwpang = None

            if n == self.nmin:
                for name in [
                    self.cos_1_name,
                    self.sin_1_name,
                    self.hweight_name,
                    self.cos_n_name,
                    self.sin_n_name,
                ]:
                    psd_units = obs[self.noise_model].psd(det).unit
                    rate_units = obs[self.noise_model].rate(det).unit
                    tod_units = (psd_units * rate_units) ** 0.5
                    obs.detdata.ensure(name, detectors=[det], create_units=tod_units)

            if n == 0:
                # Compute detector quaternions
                obs_data = data.select(obs_uid=obs.uid)
                self.pixel_pointing.detector_pointing.apply(obs_data, detectors=[det])

                cos_n_new = 1 #np.cos(psi)
                sin_n_new = 0 #np.sin(psi)
                obs.detdata[self.cos_1_name][det] = cos_n_new * u.K
                obs.detdata[self.sin_1_name][det] = sin_n_new * u.K
                obs.detdata[self.hweight_name][det] = 1
            elif n == 1:
                # Compute detector quaternions
                obs_data = data.select(obs_uid=obs.uid)
                self.pixel_pointing.detector_pointing.apply(obs_data, detectors=[det])
                quats = obs.detdata[self.pixel_pointing.detector_pointing.quats][det]
                theta, phi, psi = qa.to_iso_angles(quats)
                if not self.include_pol_angle:
                    # FIXME: temporary hack until instrument classes are also pre-staged
                    # to GPU -- taken from derivatives_and_beams toast3 branch 
                    focalplane = obs.telescope.focalplane
                    focalplane.detector_data
                    psi -= np.deg2rad(focalplane[det]["pol_ang"].value) # Removing the polarization angle
                    # psi -= focalplane[det]["pol_angle"].value # Removing the polarization angle
                    number_step = max(1, psi.shape[0] // 20)
                    # print("########TEST PRINT: Det", det, 
                    #       "pol_ang (deg)", focalplane[det]["pol_ang"].value, focalplane[det]["pol_ang"].value % 180., 
                    #       "pol_angle (deg)", np.rad2deg(focalplane[det]["pol_angle"].value), flush=True)
                    # print("psi (deg) -- shape", np.rad2deg(psi % (2*np.pi)).shape, 
                    #       '-- max', np.max(psi % (2*np.pi))*180./np.pi,
                    #       '-- min', np.min(psi % (2*np.pi))*180./np.pi,
                    #       '-- mean', np.mean(psi % (2*np.pi))*180./np.pi,
                    #       '-- std', np.std(psi % (2*np.pi))*180./np.pi,
                    #       '-- sample (one every 1000)',
                    #       (psi[::number_step] % (2*np.pi))*180./np.pi, flush=True
                    # ) #np.rad2deg(psi))

                cos_n_new = np.cos(psi)
                sin_n_new = np.sin(psi)
                obs.detdata[self.cos_1_name][det] = cos_n_new * u.K
                obs.detdata[self.sin_1_name][det] = sin_n_new * u.K
                obs.detdata[self.hweight_name][det] = 1
            else:
                # Use the angle sum identities to evaluate the
                # next cos(n * psi) and sin(n * psi)
                cos_1 = obs.detdata[self.cos_1_name][det]
                sin_1 = obs.detdata[self.sin_1_name][det]
                cos_n_old = obs.detdata[self.cos_n_name][det].copy()
                sin_n_old = obs.detdata[self.sin_n_name][det].copy()
                cos_n_new = cos_n_old * cos_1 * u.K - sin_n_old * sin_1 * u.K
                sin_n_new = sin_n_old * cos_1 * u.K + cos_n_old * sin_1 * u.K
            obs.detdata[self.cos_n_name][det] = cos_n_new
            obs.detdata[self.sin_n_name][det] = sin_n_new
        return

    @function_timer
    def _get_covariance(self, data, det):
        self.pixel_pointing.apply(data, detectors=[det])
        if self.covariance in data:
            data[self.covariance].reset()
            inv_cov_units = (1.0 / defaults.det_data_units**2)
            data[self.covariance].update_units(inv_cov_units)
        BuildInverseCovariance(
            pixel_dist=self.pixel_dist,
            inverse_covariance=self.covariance,
            view=self.pixel_pointing.view,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            pixels=self.pixel_pointing.pixels,
            weights=self.hweight_name,
            noise_model=self.noise_model,
            sync_type=self.sync_type,
        ).apply(data, detectors=[det])

        covariance_invert(
            data[self.covariance],
            self.rcond_threshold,
            use_alltoallv=(self.sync_type == "alltoallv"),
        )
        return


    @function_timer
    def _save_h_n(self, data, n, det):
        """ Accumulate and save the next order of h_n
        """
        if n < self.nmin:
            return

        log = Logger.get()

        for name, det_data in ("cos", self.cos_n_name), ("sin", self.sin_n_name):

            if self.h_n_map in data:
                data[self.h_n_map].reset()
                h_n_map_units = (1.0 / defaults.det_data_units)
                data[self.h_n_map].update_units(h_n_map_units)

            build_zmap = BuildNoiseWeighted(
                pixel_dist=self.pixel_dist,
                zmap=self.h_n_map,
                view=self.pixel_pointing.view,
                pixels=self.pixel_pointing.pixels,
                weights=self.hweight_name,
                noise_model=self.noise_model,
                det_data=det_data,
                det_flags=self.det_flags,
                det_flag_mask=self.det_flag_mask,
                shared_flags=self.shared_flags,
                shared_flag_mask=self.shared_flag_mask,
                sync_type=self.sync_type,
            )
            build_zmap.apply(data, detectors=[det])

            if n != 0:
                covariance_apply(
                    data[self.covariance],
                    data[self.h_n_map],
                    use_alltoallv=(self.sync_type == "alltoallv"),
                )

            # fname = os.path.join(self.output_dir, f"{self.name}_{det}_{name}_{n}.fits")
            # fname = os.path.join(self.output_dir, f"{self.name}_{det}_{name}_{n}.hdf5")
            fname = os.path.join(self.output_dir, f"{self.name}_{det}_{name}_{n}."+self.file_format)
            # fname = os.path.join(self.output_dir, f"{self.name}_{det}_{name}_{n}.fits")
            if '.fits' in fname or '.hdf5' in fname:
                data[self.h_n_map].write(fname)
            else:
                self._write_h_n_map_npy(
                    data[self.h_n_map],
                    fname,
                    single_precision=self.use_single_precision,
                )
            log.info_rank(f"Wrote h_n map to {fname}", comm=data.comm.comm_world)

    @function_timer
    def _get_detectors(self, data):
        """ Find a super set of detectors across all processes in the
        communicator.
        """
        my_detectors = set(data.all_local_detectors())
        comm = data.comm.comm_world
        if comm is None:
            all_detectors = my_detectors
        else:
            all_detectors = comm.gather(my_detectors)
            if comm.rank == 0:
                for detectors in all_detectors:
                    my_detectors = my_detectors.union(detectors)
            all_detectors = comm.bcast(my_detectors)
        return all_detectors

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "pixel_dist", "noise_model":
            if not hasattr(self, trait):
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if data.comm.world_rank == 0:
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

        detectors = self._get_detectors(data)

        n_min = 1 if self.nmin > 1 else self.nmin
        for det in detectors:
            for n in range(n_min, self.nmax + 1):
                self._get_h_n(data, n, det)
                self._get_covariance(data, det)
                self._save_h_n(data, n, det)

            # Purge the intermediate data objets

            for obs in data.obs:
                for name in (
                        self.hweight_name,
                        self.cos_1_name, self.sin_1_name,
                        self.cos_n_name, self.sin_n_name
                ):
                    del obs.detdata[name]

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req["meta"].extend([self.noise_model])
        req["detdata"].extend([self.det_data])
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"meta": [self.binned], "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()

# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check that pointing expanded with TOAST is compatible with the MLMapmaker

"""

import os
import tempfile
import unittest

import astropy.units as u
import numpy as np
from pixell import utils, enmap, curvedsky

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast
    from toast.observation import default_values as defaults
    toast_available = True
except ImportError as e:
    toast_available = False

if toast_available:
    import healpy as hp

from . import _helpers as helpers


class MapmakerPointingTest(unittest.TestCase):
    def test_mapmaker_pointing(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        log = toast.utils.Logger.get()

        nnz = 3
        nside = 1024
        wpix = hp.nside2pixarea(nside, degrees=True)**.5 * 60
        npix = 12 * nside**2
        testdir = tempfile.TemporaryDirectory()

        comm, procs, rank = toast.get_world()
        data = helpers.simulation_test_data(
            comm,
            telescope_name=None,
            wafer_slot="w00",
            bands="LAT_f230",
            sample_rate=10.0 * u.Hz,
            thin_fp=16,
            cal_schedule=False,
        )

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_radec",
            quats="quats_radec",
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )
        pixels = toast.ops.PixelsHealpix(
            nside=nside,
            create_dist="pixel_dist",
            detector_pointing=pointing,
        )
        weights = toast.ops.StokesWeights(
            name="weights_radec",
            weights="weights_radec",
            mode="IQU",
            detector_pointing=pointing,
        )

        input_map_file = os.path.join(
            testdir.name,
            "pointing_test_map.fits",
        )
        if rank == 0:
            input_map = np.zeros([nnz, npix], dtype=np.float32)
            input_map[1] = 1
            hp.write_map(
                input_map_file, input_map, nest=pixels.nest, column_units="K"
            )
        log.info_rank(f"Wrote test map to {input_map_file}", comm=comm)

        if data.comm.comm_world is not None:
            data.comm.comm_world.Barrier()

        # Scan map into timestreams
        scan_hpix = toast.ops.ScanHealpixMap(
            file=input_map_file,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)
        
        # Bin the map using TOAST
        toast.ops.DefaultNoiseModel(noise_model="noise_model").apply(data)
        binner = toast.ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
            shared_flag_mask=0,
        )
        toast.ops.MapMaker(
            name="mapmaker",
            det_data=defaults.det_data,
            binning=binner,
            write_hits=True,
            write_binmap=True,
            write_rcond=False,
            write_invcov=False,
            write_cov=False,
            write_map=False,
            output_dir=testdir.name,
        ).apply(data)
        file_hits_toast = os.path.join(testdir.name, "mapmaker_hits.fits")
        file_map_toast = os.path.join(testdir.name, "mapmaker_binmap.fits")

        # Create an area file
        areafile = os.path.join(testdir.name, "area_file.fits")
        box = utils.parse_box("-51:-35,35:49") * utils.degree
        res = wpix * utils.arcmin
        shape, wcs = enmap.geometry(box, res=res, proj="car")
        wcs.wcs.crpix[1] -= 0.5
        enmap.write_map(areafile, enmap.zeros(shape, wcs, np.uint8))

        mapmaker = so_ops.MLMapmaker(
            area=areafile,
            name="mlmapmaker",
            out_dir=testdir.name,
            comps="TQU",
            nmat_type="Nmat",
            maxiter=[3],
            truncate_tod=False,
            write_hits=True,
            write_rhs=False,
            write_div=False,
            write_bin=True,
            deslope=False,
        )
        mapmaker.apply(data)

        if rank == 0:
            # Direct comparison of pointing
            obs = data.obs[0]
            pmap = mapmaker.signal_map.data[obs.name].pmap
            fplane = pmap._get_asm().fplane
            coords = np.array(pmap.sight.coords(fplane))
            dets = mapmaker.mapmaker.data[0].dets
            ndet = len(dets)

            pointing.apply(data)
            dlats_mean = np.zeros(ndet)
            dlats_rms = np.zeros(ndet)
            dlons_mean = np.zeros(ndet)
            dlons_rms = np.zeros(ndet)
            dgammas_mean = np.zeros(ndet)
            dgammas_rms = np.zeros(ndet)
            for idet in range(ndet):
                lon, lat, cosgamma, singamma = coords[idet].T
                gamma = np.arctan2(singamma, cosgamma)
                theta, toast_lon, toast_gamma = toast.qarray.to_iso_angles(
                    obs.detdata["quats_radec"][idet]
                )
                toast_lat = np.pi / 2 - theta
                toast_dets = obs.local_detectors
                dlon = np.degrees(lon - toast_lon) * 3600
                dlat = np.degrees(lat - toast_lat) * 3600
                dgamma = np.degrees(np.unwrap(gamma - toast_gamma)) * 3600
                mean_dlon = np.mean(dlon)
                rms_dlon = np.std(dlon)
                mean_dlat = np.mean(dlat)
                rms_dlat = np.std(dlat)
                mean_dgamma = np.mean(dgamma)
                rms_dgamma = np.std(dgamma)
                #log.info(f"dlat = {mean_dlat:.3f} +- {rms_dlat:.3f} arcsec")
                #log.info(f"dlon = {mean_dlon:.3f} +- {rms_dlon:.3f} arcsec")
                #log.info(f"dgamma = {mean_dgamma:.3f} +- {rms_dgamma:.3f} arcsec")
                dlats_mean[idet] = mean_dlat
                dlats_rms[idet] = rms_dlat
                dlons_mean[idet] = mean_dlon
                dlons_rms[idet] = rms_dlon
                dgammas_mean[idet] = mean_dgamma
                dgammas_rms[idet] = rms_dgamma
            log.info(f"dlat = {np.mean(dlats_mean):.3f} +- {np.std(dlats_mean):.3f} arcsec")
            log.info(f"dlon = {np.mean(dlons_mean):.3f} +- {np.std(dlons_mean):.3f} arcsec")
            log.info(f"dgamma = {np.mean(dgammas_mean):.3f} +- {np.std(dgammas_mean):.3f} arcsec")
            tol = 1.0
            if rms_dlon > tol:
                raise RuntimeError(f"RA differs systematically over {tol} arc seconds")
            if rms_dlat > tol:
                raise RuntimeError(f"Dec differs systematically over {tol} arc seconds")
            if rms_dgamma > tol:
                raise RuntimeError(f"PA differs systematically over {tol} arc seconds")

            # Compare binned maps

            file_hits = os.path.join(testdir.name, "mlmapmaker_sky_hits.fits")
            file_map = os.path.join(testdir.name, "mlmapmaker_sky_map.fits")
            hits = enmap.read_map(file_hits).astype(int)
            sky = enmap.read_map(file_map)
            good = hits != 0
            ind = sky[1] != 0

            toast_hits = hp.read_map(file_hits_toast)
            toast_sky = hp.read_map(file_map_toast, None)
            toast_good = toast_hits != 0
            toast_ind = toast_sky[1] != 0

            rms = []
            means = []
            toast_rms = []
            toast_means = []
            log.info(f"            {'MLMapmaker':16}            {'TOAST':16}")
            log.info(f"Hits        {np.sum(hits):8}       {np.sum(toast_hits):16}")
            log.info(f"Hit pixels  {np.sum(good):8}       {np.sum(toast_good):16}")
            log.info(f"After cut   {np.sum(ind):8}       {np.sum(toast_ind):16}")
            for i in range(3):
                rms.append(np.std(sky[i][ind].ravel()))
                means.append(np.mean(sky[i][ind].ravel()))
                toast_rms.append(np.std(toast_sky[i][toast_ind].ravel()))
                toast_means.append(np.mean(toast_sky[i][toast_ind].ravel()))
                stokes = "IQU"[i]
                log.info(
                    f"{stokes:3} {means[i]:12.6f} +- {rms[i]:12.6f}, "
                    f"{toast_means[i]:12.6f} +- {toast_rms[i]:12.6f}"
                )

            tol = 1e-3
            if np.abs(means[0]) > tol:
                raise RuntimeError("Found non-zero I")

            if np.abs(means[1] - 1) > tol:
                raise RuntimeError("Found non-unit Q")

            if np.abs(means[2]) > tol:
                raise RuntimeError("Found non-zero U")

        helpers.close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()

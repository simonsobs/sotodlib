# Copyright (c) 2023-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check that pointing expanded with TOAST is compatible with the MLMapmaker

"""

import os
import shutil
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

from ._helpers import (
    create_outdir, simulation_test_data, close_data_and_comm
)


class MapmakerPointingTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_mapmaker_pointing_wcs(self):
        if not toast_available:
            return
        log = toast.utils.Logger.get()
        world, procs, rank = toast.get_world()
        testdir = os.path.join(self.outdir, "pointing_wcs")
        if world is None or world.rank == 0:
            if os.path.isdir(testdir):
                shutil.rmtree(testdir)
            os.makedirs(testdir)
        if world is not None:
            world.barrier()

        if rank == 0:
            # Create an area file
            areafile = os.path.join(testdir, "wcs_area_file.fits")
            ra_start, ra_stop = 35, 49
            dec_start, dec_stop = -51, -35
            box = np.radians([[dec_start, ra_start], [dec_stop, ra_stop]])
            reso = np.radians(1.0 / 60)
            shape, wcs = enmap.geometry(box, res=reso, proj="car")
            wcs.wcs.crpix[1] -= 0.5
            enmap.write_map_geometry(areafile, shape, wcs)

            # Create our input map with point sources
            input_map = enmap.zeros((3,) + shape, wcs=wcs)
            DEC, RA = input_map.posmap()
            radius = np.radians(15 / 60)
            for ra in np.linspace(ra_start, ra_stop, 10):
                for dec in np.linspace(dec_start, dec_stop, 10):
                    ra0 = np.radians(ra)
                    dec0 = np.radians(dec)
                    input_map[0] += np.exp(
                        -0.5 * ((DEC - dec0)**2 + (RA - ra0)**2)/radius**2
                    )
            input_map_file = os.path.join(testdir, "input_map.fits")
            enmap.write_map(input_map_file, input_map, extra={"BUNIT" : "uK"})
        else:
            areafile = None
            input_map_file = None

        if world is not None:
            areafile = world.bcast(areafile)
            input_map_file = world.bcast(input_map_file)
        log.info_rank(f"Wrote test map to {input_map_file}", comm=world)

        # scan the WCS map with TOAST
        # map the scanned signal using TOAST
        # map the scanned signal using MLMapmaker
        # compare input/TOAST/MLMapmaker

        data = simulation_test_data(
            world,
            telescope_name=None,
            wafer_slot="w00",
            bands="LAT_f230",
            sample_rate=10.0 * u.Hz,
            thin_fp=16,
            cal_schedule=False,
            groupsize=1,  # required by MLMapmaker
        )

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_radec",
            quats="quats_radec",
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )
        pixels = toast.ops.PixelsWCS(
            fits_header=areafile,
            create_dist="pixel_dist",
            detector_pointing=pointing,
        )
        weights = toast.ops.StokesWeights(
            name="weights_radec",
            weights="weights_radec",
            mode="IQU",
            detector_pointing=pointing,
        )

        # Scan map into timestreams
        scan_wcs = toast.ops.ScanWCSMap(
            file=input_map_file,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_wcs.apply(data)

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
            write_rcond=True,
            write_invcov=False,
            write_cov=False,
            write_map=False,
            output_dir=testdir,
            map_rcond_threshold=1e-6,
        ).apply(data)
        file_hits_toast = os.path.join(testdir, "mapmaker_hits.fits")
        file_map_toast = os.path.join(testdir, "mapmaker_binmap.fits")
        file_rcond_toast = os.path.join(testdir, "mapmaker_rcond.fits")

        mapmaker = so_ops.MLMapmaker(
            area=areafile,
            name="mlmapmaker",
            out_dir=testdir,
            comps="TQU",
            nmat_type="Nmat",
            maxiter=[3],
            weather="typical",
            truncate_tod=False,
            write_hits=True,
            write_rhs=False,
            write_div="all",
            write_bin=True,
            deslope=False,
        )
        mapmaker.apply(data)

        if rank == 0:
            # Compare binned maps
            import matplotlib.pyplot as plt

            file_hits = os.path.join(testdir, "mlmapmaker_sky_hits.fits")
            file_map = os.path.join(testdir, "mlmapmaker_sky_map.fits")
            hits = enmap.read_map(file_hits).astype(int)
            sky = enmap.read_map(file_map) * 1e6  # input map is in uK

            toast_hits = enmap.read_map(file_hits_toast)[0]
            toast_sky = enmap.read_map(file_map_toast, None)
            toast_sky *= 1e6  # input map is in uK
            toast_rcond = enmap.read_map(file_rcond_toast, None)[0]

            # build a comparison mask
            mask = toast_rcond > 0

            # There are a tiny number of outliers in the MLMapmaker
            # difference map.  Discard 10 worst offenders.
            dmap = sky[0] - input_map[0]
            sorted_dmap = np.sort(np.abs(dmap[mask]))
            limit = sorted_dmap[-10]
            bad_pix = np.logical_and(np.abs(dmap) > limit, mask)
            log.info(
                f"Masking 10 / {np.sum(mask)} worst pixels in the MLMapmaker "
                "residual map"
            )
            mask[bad_pix] = False

            bad = np.logical_not(mask)

            nrow, ncol = 2, 3
            fig = plt.figure(figsize=[6 * ncol, 4 * nrow])
            axes = [
                fig.add_subplot(nrow, ncol, iplot + 1)
                for iplot in range(nrow * ncol)
            ]

            amp = 1.0
            args = {
                "extent" : [ra_start, ra_stop, dec_stop, dec_start],
                "cmap" : "bwr",
                "vmin" : -amp,
                "vmax" : amp,
                "interpolation" : "none",
            }

            ax = axes[0]
            ax.set_title("Input")
            rms0 = np.std(input_map[0, mask])
            plot = ax.imshow(input_map[0], **args)
            plt.colorbar(plot)

            ax = axes[1]
            ax.set_title("TOAST binned")
            toast_sky[:, bad] = np.nan
            plot = ax.imshow(toast_sky[0], **args)
            rms1 = np.std(toast_sky[0, mask])
            plt.colorbar(plot)

            ax = axes[2]
            ax.set_title("MLMapmaker binned")
            sky[:, bad] = np.nan
            plot = ax.imshow(sky[0], **args)
            rms2 = np.nanstd(sky[0, mask])
            plt.colorbar(plot)

            ax = axes[3]
            ax.set_title("rcond mask")
            plot = ax.imshow(mask, **args)
            plt.colorbar(plot)

            amp = 1e-6
            args["vmin"] = -amp
            args["vmax"] = amp

            ax = axes[4]
            ax.set_title("TOAST - input")
            dmap1 = toast_sky[0] - input_map[0]
            plot = ax.imshow(dmap1, **args)
            rms3 = np.std(dmap1[mask])
            plt.colorbar(plot)

            ax = axes[5]
            ax.set_title("MLMapmaker - input")
            dmap2 = sky[0] - input_map[0]
            plot = ax.imshow(dmap2, **args)
            rms4 = np.std(dmap2[mask])
            plt.colorbar(plot)

            # for ax in axes:
            #     ax.set_xlim([39, 43])
            #     ax.set_ylim([-44, -40])

            fname_plot = os.path.join(testdir, "wcs_diff.png")
            plt.savefig(fname_plot)

            test1 = np.abs(rms1 / rms0 - 1)
            test2 = np.abs(rms2 / rms0 - 1)
            test3 = np.abs(rms3 / rms0)
            test4 = np.abs(rms4 / rms0)
            log.info(f"RMS comparison:")
            log.info(f"           TOAST / input - 1 = {test1:.4e}")
            log.info(f"      MLMapmaker / input - 1 = {test2:.4e}")
            log.info(f"     (TOAST - input) / input = {test3:.4e}")
            log.info(f"(MLMapmaker - input) / input = {test4:.4e}")

            assert test1 < 1e-6
            assert test2 < 1e-6
            assert test3 < 1e-6
            assert test4 < 1e-6

        close_data_and_comm(data)

    def test_mapmaker_pointing(self):
        if not toast_available:
            return

        log = toast.utils.Logger.get()
        world, procs, rank = toast.get_world()
        testdir = os.path.join(self.outdir, "pointing_healpix")
        if world is None or world.rank == 0:
            if os.path.isdir(testdir):
                shutil.rmtree(testdir)
            os.makedirs(testdir)
        if world is not None:
            world.barrier()

        nnz = 3
        nside = 1024
        wpix = hp.nside2pixarea(nside, degrees=True)**.5 * 60
        npix = 12 * nside**2

        data = simulation_test_data(
            world,
            telescope_name=None,
            wafer_slot="w00",
            bands="LAT_f230",
            sample_rate=10.0 * u.Hz,
            thin_fp=16,
            cal_schedule=False,
            groupsize=1,
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

        input_map_file = os.path.join(testdir, "pointing_test_map.fits")
        if rank == 0:
            input_map = np.zeros([nnz, npix], dtype=np.float32)
            input_map[1] = 1
            hp.write_map(
                input_map_file, input_map, nest=pixels.nest, column_units="K"
            )
        log.info_rank(f"Wrote test map to {input_map_file}", comm=world)

        if world is not None:
            world.Barrier()

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
            output_dir=testdir,
        ).apply(data)
        file_hits_toast = os.path.join(testdir, "mapmaker_hits.fits")
        file_map_toast = os.path.join(testdir, "mapmaker_binmap.fits")

        # Create an area file
        areafile = os.path.join(testdir, "area_file.fits")
        box = utils.parse_box("-51:-35,35:49") * utils.degree
        res = wpix * utils.arcmin
        shape, wcs = enmap.geometry(box, res=res, proj="car")
        wcs.wcs.crpix[1] -= 0.5
        enmap.write_map_geometry(areafile, shape, wcs)

        mapmaker = so_ops.MLMapmaker(
            area=areafile,
            name="mlmapmaker",
            out_dir=testdir,
            comps="TQU",
            nmat_type="Nmat",
            maxiter=[3],
            weather="typical",
            truncate_tod=False,
            write_hits=True,
            write_rhs=False,
            write_div="all",
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
                tol = 1.0
                if rms_dlon > tol:
                    msg = f"RA differs systematically over {tol} arc seconds"
                    raise RuntimeError(msg)
                if rms_dlat > tol:
                    msg = f"Dec differs systematically over {tol} arc seconds"
                    raise RuntimeError(msg)
                if rms_dgamma > tol:
                    msg = f"PA differs systematically over {tol} arc seconds"
                    raise RuntimeError(msg)
            log.info(
                f"dlat = {np.mean(dlats_mean):.3f} "
                f"+- {np.std(dlats_mean):.3f} arcsec"
            )
            log.info(
                f"dlon = {np.mean(dlons_mean):.3f} "
                f"+- {np.std(dlons_mean):.3f} arcsec"
            )
            log.info(
                f"dgamma = {np.mean(dgammas_mean):.3f} "
                f"+- {np.std(dgammas_mean):.3f} arcsec"
            )

            # Compare binned maps

            file_hits = os.path.join(testdir, "mlmapmaker_sky_hits.fits")
            file_map = os.path.join(testdir, "mlmapmaker_sky_map.fits")
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
            log.info(
                f"Hits        {np.sum(hits):8}       {np.sum(toast_hits):16}"
            )
            log.info(
                f"Hit pixels  {np.sum(good):8}       {np.sum(toast_good):16}"
            )
            log.info(
                f"After cut   {np.sum(ind):8}       {np.sum(toast_ind):16}"
            )
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

        close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()

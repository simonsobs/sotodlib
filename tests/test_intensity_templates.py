# Copyright (c) 2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the TOAST intensity templates

"""

import os
import unittest

import astropy.units as u
import healpy as hp
import numpy as np

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast
    from toast.observation import default_values as defaults
    toast_available = True
except ImportError as e:
    toast_available = False

from ._helpers import create_outdir, close_data_and_comm, simulation_test_data


class IntensityTemplateTest(unittest.TestCase):
    def test_intensity_templates(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()

        outdir = create_outdir(
            subdir=os.path.splitext(os.path.basename(__file__))[0],
            mpicomm=comm,
        )

        data = simulation_test_data(
            comm,
            telescope_name="SAT4",
            wafer_slot="w42",
            bands="SAT_f030,SAT_f040",
            sample_rate=37.0 * u.Hz,
            thin_fp=64,
            cal_schedule=False,
        )

        # Simulate noise to fill the TOD

        toast.ops.DefaultNoiseModel().apply(data)
        toast.ops.SimNoise().apply(data)

        # Demodulate

        detpointing = toast.ops.PointingDetectorSimple(
            quats="quats_radec",
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )

        weights = toast.ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="weights_radec",
        )

        toast.ops.Demodulate(stokes_weights=weights, in_place=True).apply(data)

        # Generate and cache templates

        so_ops.IntensityTemplates(
            name="intensity_templates",
            fpkeys="wafer_slot,bandcenter",
            template_name="intensity_templates",
            mode="radius:1deg",
            submode="lowpass:1Hz",
        ).apply(data)

        # Replace polarization signal with scaled intensity templates

        for ob in data.obs:
            template_dict = ob["intensity_templates"]
            for det in ob.local_detectors:
                if det.startswith("demod0"):
                    continue
                sig = ob.detdata[defaults.det_data][det]
                sig[:] = 0
                key = template_dict["det_to_key"][det]
                templates = template_dict[key]
                for name, template in templates.items():
                    sig += np.random.randn() * template

        # Now try projecting out the templates in FilterBin

        demod_weights = toast.ops.StokesWeightsDemod()

        pixels = toast.ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )

        binning = toast.ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=demod_weights,
        )

        filterbin = toast.ops.FilterBin(
            binning=binning,
            output_dir=outdir,
            precomputed_templates="intensity_templates",
            precomputed_template_view="scanning",
            write_binmap=True,
            write_map=True,
            poly_filter_order=None,
            ground_filter_order=None,
            ground_filter_bin_width=None,
        )
        filterbin.apply(data)

        # Confirm that the filtered map is consistent with zero

        fname_binned = os.path.join(outdir, "FilterBin_unfiltered_map.fits")
        fname_filtered = os.path.join(outdir, "FilterBin_filtered_map.fits")
        binned = hp.read_map(fname_binned, None)
        filtered = hp.read_map(fname_filtered, None)
        good = binned[0] != 0

        # Intensity should not be filtered
        assert (
            np.abs(
                np.std(filtered[0, good]) / np.std(binned[0, good]) - 1
            ) < 1e-10
        )
        # Polarization should be substantially suppressed
        assert (
            np.abs(np.std(filtered[1, good]) / np.std(binned[1, good])) < 1e-10
        )
        assert (
            np.abs(np.std(filtered[2, good]) / np.std(binned[2, good])) < 1e-10
        )

        close_data_and_comm(data)

    def test_intensity_template_caching(self):
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()

        outdir = create_outdir(
            subdir=os.path.splitext(os.path.basename(__file__))[0],
            mpicomm=comm,
        )

        data = simulation_test_data(
            comm,
            telescope_name="SAT4",
            wafer_slot="w42",
            bands="SAT_f030,SAT_f040",
            sample_rate=37.0 * u.Hz,
            thin_fp=64,
            cal_schedule=False,
        )

        # Simulate noise to fill the TOD

        toast.ops.DefaultNoiseModel().apply(data)
        toast.ops.SimNoise().apply(data)

        # Demodulate

        detpointing = toast.ops.PointingDetectorSimple(
            quats="quats_radec",
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )

        weights = toast.ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="weights_radec",
        )

        toast.ops.Demodulate(stokes_weights=weights, in_place=True).apply(data)

        # Generate and cache templates

        cache_dir = os.path.join(outdir, "template_cache")
        so_ops.IntensityTemplates(
            name="intensity_templates",
            fpkeys="wafer_slot,bandcenter",
            template_name="intensity_templates",
            cache_dir=cache_dir,
            mode="radius:1deg",
            submode="lowpass:1Hz",
        ).apply(data)

        # Try loading the templates from cache and compare to the ones
        # in memory

        so_ops.IntensityTemplates(
            name="intensity_templates_copy",
            fpkeys="wafer_slot,bandcenter",
            template_name="intensity_templates_copy",
            cache_dir=cache_dir,
            mode="radius:1deg",
            submode="lowpass:1Hz",
        ).apply(data)

        for ob in data.obs:
            orig = ob["intensity_templates"]
            cached = ob["intensity_templates_copy"]
            for key1, value1 in orig.items():
                for key2, value2 in value1.items():
                    if isinstance(value2, str):
                        assert value2 == cached[key1][key2]
                    else:
                        assert np.allclose(value2, cached[key1][key2])

        close_data_and_comm(data)

if __name__ == '__main__':
    unittest.main()

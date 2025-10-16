# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test so-toast-run."""

import os
import shutil
import glob
import re

import numpy as np
import astropy.units as u

from unittest import TestCase

# Import so3g before any other packages that import spt3g
import so3g

try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import toast
    from toast.observation import default_values as defaults

    toast_available = True
except ImportError:
    raise
    toast_available = False

from sotodlib.sim_hardware import sim_nominal
from sotodlib.toast.scripts.so_sim_telescope import main as so_sim_tele_main
from sotodlib.toast.scripts.so_run import main as so_run_main

from ._helpers import create_outdir, observing_schedule


class ToastRunTest(TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_sim(self):
        if not toast_available:
            print("TOAST cannot be imported- skipping unit test", flush=True)
            return

        world, procs, rank = toast.get_world()
        testdir = os.path.join(self.outdir, "sim")
        if world is None or world.rank == 0:
            if os.path.isdir(testdir):
                shutil.rmtree(testdir)
            os.makedirs(testdir)
        if world is not None:
            world.barrier()

        instrument_file = None
        schedule_file = None
        config_path = None
        if rank == 0:
            # Simulate an SO-style hardware file and write to disk
            fullhw = sim_nominal()
            sotoast.sim_telescope_detectors(fullhw, "SAT1")
            hw = fullhw.select(
                match={
                    "wafer_slot": [
                        "w25",
                    ],
                }
            )
            hwpath = os.path.join(testdir, "telescope_SAT1_w25.toml.gz")
            hw.dump(hwpath, overwrite=True, compress=True)

            # Convert this to an instrument file loadable by SimGround
            instrument_file = os.path.join(testdir, "telescope.h5")
            opts = [
                "--output",
                instrument_file,
                "--hardware",
                hwpath,
                "--thinfp",
                "10",
            ]
            so_sim_tele_main(opts=opts)

            # Load the telescope back in, and use it for the schedule
            # building
            telescope, _ = toast.io.load_instrument_file(instrument_file)

            # Create a schedule
            _ = observing_schedule(telescope, mpicomm=None, temp_dir=testdir)
            schedule_file = os.path.join(testdir, "schedule.txt")

            # Create the pipelines to run and dump to config file
            conf_pipe = dict()

            sim_list = [
                toast.ops.SimGround(
                    name="sim_ground",
                    hwp_angle="hwp_angle",
                    hwp_rpm=1.0,
                ),
                toast.ops.DefaultNoiseModel(
                    name="noise_model", noise_model="sim_noise_model"
                ),
                toast.ops.SimNoise(name="sim_noise", noise_model="sim_noise_model"),
            ]
            sim_ops = {x.name: x for x in sim_list}

            for op_name, op in sim_ops.items():
                conf_pipe = op.get_config(input=conf_pipe)
            sim_pipe = toast.ops.Pipeline(name="sim_pipe")
            sim_pipe.operators = [y for x, y in sim_ops.items()]
            conf_pipe = sim_pipe.get_config(input=conf_pipe)

            reduce_list = [
                toast.ops.SignalDiffNoiseModel(
                    name="diff_noise", noise_model="noise_model", det_mask=1
                ),
                toast.ops.PointingDetectorSimple(
                    name="det_pointing", boresight="boresight_radec"
                ),
                toast.ops.PixelsHealpix(name="pixels", nside=256),
                toast.ops.StokesWeights(
                    name="weights", hwp_angle="hwp_angle", mode="IQU"
                ),
                toast.ops.Demodulate(
                    name="demodulate",
                    det_mask=1,
                    noise_model="noise_model",
                    nskip=10,
                    in_place=True,
                ),
                toast.ops.StokesWeightsDemod(name="weights_demod"),
                toast.ops.PolyFilter(
                    name="polyfilter", order=5, view="scanning", det_mask=1
                ),
                toast.ops.BinMap(name="binner", full_pointing=True),
                toast.ops.MapMaker(name="mapmaker"),
            ]
            reduce_ops = {x.name: x for x in reduce_list}
            # Set up references
            reduce_ops["pixels"].detector_pointing = reduce_ops["det_pointing"]
            reduce_ops["weights"].detector_pointing = reduce_ops["det_pointing"]
            reduce_ops["demodulate"].stokes_weights = reduce_ops["weights"]
            reduce_ops["binner"].pixel_pointing = reduce_ops["pixels"]
            reduce_ops["binner"].stokes_weights = reduce_ops["weights_demod"]
            reduce_ops["mapmaker"].binning = reduce_ops["binner"]

            for op_name, op in reduce_ops.items():
                conf_pipe = op.get_config(input=conf_pipe)
            reduce_pipe = toast.ops.Pipeline(name="reduce_pipe")
            reduce_pipe.operators = [
                reduce_ops["diff_noise"],
                reduce_ops["demodulate"],
                reduce_ops["polyfilter"],
            ]
            conf_pipe = reduce_pipe.get_config(input=conf_pipe)

            main_pipe = toast.ops.Pipeline(
                name="main",
                operators=[sim_pipe, reduce_pipe, reduce_ops["mapmaker"]],
            )
            conf_pipe = main_pipe.get_config(input=conf_pipe)

            config_path = os.path.join(testdir, "config.yml")
            toast.config.dump_yaml(config_path, conf_pipe)

        if world is not None:
            world.barrier()
            instrument_file = world.bcast(instrument_file, root=0)
            schedule_file = world.bcast(schedule_file, root=0)
            config_path = world.bcast(config_path, root=0)

        # Create the commandline options
        opts = [
            "--worker_size",
            "1",
            "--out_root",
            testdir,
            "--task_per_wafer",
            "--sim_telescope",
            instrument_file,
            "--sim_schedule",
            schedule_file,
            "--config",
            config_path,
            "--sim_ground.telescope_file",
            "{telescope_file}",
            "--sim_ground.schedule_file",
            "{schedule_file}",
            "--mapmaker.output_dir",
            "{run_dir}",
        ]

        # Run it
        so_run_main(opts=opts, comm=world)

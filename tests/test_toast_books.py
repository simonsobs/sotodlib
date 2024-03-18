# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast data export.
"""

import os
import shutil
import glob
import re

import numpy as np
import astropy.units as u

from unittest import TestCase

# Import so3g before any other packages that import spt3g
import so3g

from ._helpers import create_outdir, simulation_test_multitube, close_data_and_comm


try:
    # Import sotodlib toast module first, which sets global toast defaults
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    import toast
    from toast.observation import default_values as defaults

    toast_available = True
except ImportError:
    raise
    toast_available = False


class ToastBooksTest(TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = toast.get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)

    def test_book_saveload(self):
        if not toast_available:
            return
        world, procs, rank = toast.get_world()
        testdir = os.path.join(self.outdir, "save_load")
        if world is None or world.rank == 0:
            if os.path.isdir(testdir):
                shutil.rmtree(testdir)
            os.makedirs(testdir)
        if world is not None:
            world.barrier()

        data = simulation_test_multitube(
            world,
            telescope_name="LAT",
            tubes="c1,i6",
            sample_rate=2.0 * u.Hz,
            detset_key="pixel",
            thin_fp=100,
            temp_dir=self.outdir,
        )

        # Create a noise model from focalplane detector properties
        noise_model = toast.ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate some noise
        sim_noise = toast.ops.SimNoise()
        sim_noise.apply(data)

        # Corotation
        corotator = so_ops.CoRotator()
        corotator.apply(data)

        # Purge intervals.  The book format currently does not support things
        # like this.  We would have to reconstruct those from the flag information
        # for now.  Make a copy for later comparison

        original = list()
        for ob in data.obs:
            interval_names = list(ob.intervals.keys())
            for iname in interval_names:
                del ob.intervals[iname]
            original.append(ob.duplicate(times="times"))

        # Set up the save operator

        book_dir = os.path.join(testdir, "books")
        focalplane_dir = os.path.join(testdir, "focalplanes")
        noise_dir = os.path.join(testdir, "noise_models")

        save = so_ops.SaveBooks(
            book_dir=book_dir,
            focalplane_dir=focalplane_dir,
            noise_dir=noise_dir,
            frame_intervals=None,
            gzip=True,
            hwp_angle=None, # This is LAT data
        )

        # Save the data
        # data.info()
        save.apply(data)

        # Load the data

        new_data = toast.Data(comm=data.comm)
        load = so_ops.LoadBooks(
            books=glob.glob(os.path.join(book_dir, "obs_*")),
            focalplane_dir=focalplane_dir,
            noise_dir=noise_dir,
            detset_key="pixel",
            hwp_angle=None,
            corotator_angle=defaults.corotator_angle,
            boresight_angle=None,
        )
        load.apply(new_data)
        # new_data.info()

        # Check for consistency

        # print(f"Original names:  {[x.name for x in data.obs]}", flush=True)

        # print(f"New names:  {[x.name for x in new_data.obs]}", flush=True)

        new_order = {y.name: x for x, y in enumerate(new_data.obs)}

        for orig in original:
            # The telescope name in the original data is modified by the observing
            # split.  It does not matter in practice, but causes the equality
            # test to fail.
            orig.telescope.name = re.sub(r"^LAT_", "", orig.telescope.name)

            ob = new_data.obs[new_order[orig.name]]

            # FIXME:  Leave these in if we want to check general metadata
            # handling.
            del orig["scan_el"]
            del orig["scan_min_az"]
            del orig["scan_max_az"]
            del orig["scan_min_el"]
            del orig["scan_max_el"]

            # The loaded noise model is a base class instance, and the one
            # written is an analytic model.  Fix this eventually so that
            # the comparison passes below.
            del orig["noise_model"]
            del ob["noise_model"]

            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)

        close_data_and_comm(new_data)
        close_data_and_comm(data)

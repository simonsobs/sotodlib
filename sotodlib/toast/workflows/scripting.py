# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""High-level blocks of workflow tools.
"""
import toast
from toast.observation import default_values as defaults

from .. import workflows as wrk


def setup_load_or_simulate_observing(parser, operators):
    # Loading data from disk
    wrk.setup_load_data_hdf5(operators)
    wrk.setup_load_data_books(operators)
    wrk.setup_load_data_context(operators)
    wrk.setup_act_responsivity_sign(operators)
    wrk.setup_az_intervals(operators)
    wrk.setup_weather_model(operators)
    wrk.setup_diff_noise_estimation(operators)

    # Simulated observing
    wrk.setup_simulate_observing(parser, operators)
    wrk.setup_simple_noise_models(operators)

    wrk.setup_pointing(operators)


def load_or_simulate_observing(job, otherargs, runargs, comm):
    log = toast.utils.Logger.get()
    job_ops = job.operators

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    job_ops.mem_count.prefix = "Before Data Load or Simulate"
    job_ops.mem_count.apply(data)

    if job_ops.sim_ground.enabled:
        data = wrk.simulate_observing(job, otherargs, runargs, comm)
        wrk.simple_noise_models(job, otherargs, runargs, data)
    else:
        group_size = wrk.reduction_group_size(job, runargs, comm)
        toast_comm = toast.Comm(world=comm, groupsize=group_size)
        data = toast.Data(comm=toast_comm)
        # Load data from all formats
        wrk.load_data_hdf5(job, otherargs, runargs, data)
        wrk.load_data_books(job, otherargs, runargs, data)
        wrk.load_data_context(job, otherargs, runargs, data)
        wrk.act_responsivity_sign(job, otherargs, runargs, data)
        wrk.create_az_intervals(job, otherargs, runargs, data)
        # Append a weather model
        wrk.append_weather_model(job, otherargs, runargs, data)
        # Before running simulations based on real data, we need
        # a starting noise estimate
        wrk.diff_noise_estimation(job, otherargs, runargs, data)
        # optionally zero out
        if otherargs.zero_loaded_data:
            toast.ops.Reset(detdata=[defaults.det_data])
    wrk.select_pointing(job, otherargs, runargs, data)

    job_ops.mem_count.prefix = "After Data Load or Simulate"
    job_ops.mem_count.apply(data)

    return data


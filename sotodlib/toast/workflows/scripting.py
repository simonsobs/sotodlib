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
    wrk.setup_readout_filter(operators)
    wrk.setup_deconvolve_detector_timeconstant(operators)

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
        wrk.apply_readout_filter(job, otherargs, runargs, data)
        wrk.deconvolve_detector_timeconstant(job, otherargs, runargs, data)
        # Append a weather model
        wrk.append_weather_model(job, otherargs, runargs, data)
        # Before running simulations based on real data, we need
        # a starting noise estimate
        wrk.diff_noise_estimation(job, otherargs, runargs, data)
        # optionally zero out
        if hasattr(otherargs, "zero_loaded_data") and otherargs.zero_loaded_data:
            toast.ops.Reset(detdata=[defaults.det_data])
    wrk.select_pointing(job, otherargs, runargs, data)

    job_ops.mem_count.prefix = "After Data Load or Simulate"
    job_ops.mem_count.apply(data)

    return data


def setup_preprocess(parser, operators):
    wrk.setup_filter_hwpss_relcal(operators)
    wrk.setup_filter_hwpss(operators)
    wrk.setup_simple_jumpcorrect(operators)
    wrk.setup_simple_deglitch(operators)
    wrk.setup_flag_diff_noise_outliers(operators)
    wrk.setup_flag_noise_outliers(operators)


def preprocess(job, otherargs, runargs, data):
    """Apply flags and cuts based on data quality."""
    log = toast.utils.Logger.get()
    job_ops = job.operators

    # Run these on the original data
    wrk.simple_jumpcorrect(job, otherargs, runargs, data)
    wrk.simple_deglitch(job, otherargs, runargs, data)

    if otherargs.preprocess_copy:
        prename = "preprocess"
        toast.ops.Copy(detdata=[(defaults.det_data, prename)]).apply(data)
    else:
        prename = defaults.det_data

    def _run_pre_op(op, wrkf):
        save_op_detdata = op.det_data
        op.det_data = prename
        wrkf(job, otherargs, runargs, data)
        op.det_data = save_op_detdata

    # _run_pre_op(job_ops.simple_jumpcorrect, wrk.simple_jumpcorrect)

    # _run_pre_op(job_ops.simple_deglitch, wrk.simple_deglitch)

    # Ensure that we run the HWPSS filter for preprocessing, even if we
    # are not using it on the original data.
    if job_ops.hwpfilter.enabled and job_ops.hwpssfilter.enabled:
        raise RuntimeError("Cannot enable both hwpfilter and hwpssfilter")
    # Select which hwp filter we are using
    if job_ops.hwpfilter.enabled:
        hwp_op = job_ops.hwpfilter
        hwp_wrk = wrk.filter_hwpss
    else:
        hwp_op = job_ops.hwpssfilter
        hwp_wrk = wrk.filter_hwpss_relcal
    # Save state, run, and restore
    if otherargs.preprocess_copy:
        save_hwpf_state = hwp_op.enabled
        hwp_op.enabled = True
    _run_pre_op(hwp_op, hwp_wrk)
    if otherargs.preprocess_copy:
        hwp_op.enabled = save_hwpf_state

    _run_pre_op(job_ops.diff_noise_cut, wrk.flag_diff_noise_outliers)

    _run_pre_op(job_ops.noise_cut, wrk.flag_noise_outliers)

    if otherargs.preprocess_copy:
        toast.ops.Delete(detdata=[prename,]).apply(data)
        # FIXME:  Although the glitches / jumps are flagged, the original
        # data has not been filled / corrected.  Not sure how to do this
        # without applying the hwpfilter to the original data, which we
        # might want to avoid if solving for that template in the mapmaking.
        if job_ops.hwpsscal.enabled:
            job_ops.hwpsscal.cal_name = job_ops.hwpssfilter.relcal
            job_ops.hwpsscal.apply(data)

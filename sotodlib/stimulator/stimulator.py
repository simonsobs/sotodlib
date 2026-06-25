import numpy as np
import lmfit
import ruptures as rpt
import astropy

from sotodlib import tod_ops, core
from sotodlib.io import hkdb
from sotodlib.stimulator.utils_stimulator import (
    func_sines,
    func_response_amplitude,
    func_response_phase_with_dt,
)

CHOPPING_FREQS = {"f1": 6, "f2": 15, "f3": 33, "f4": 63, "f5": 93, "f6": 123, "f7": 147}
STM_NORMALIZE_TEMP = (
    750  # Kelvin. Normalization temperature for stimulator signal temperature.
)


def get_hk(hkdb_cfg, aman=None, t_start=None, t_end=None):
    """
    Get housekeeping data for one axis manager.

    Args:
        hkdb_cfg: HK database config.
        aman: Axis manager of detector data.
        t_start: Start time of HK data. If None, use aman's start time.
        t_end: End time of HK data. If None, use aman's end time.

    Return:
        Housekeeping data.
    """
    feed_list = [
        "stimulator-blh.motor.*",
        "stimulator-ds378.relay.*",
        "stimulator-enc.stim_enc.*",
        "stimulator-enc.stim_enc_downsampled.*",
        "stimulator-pcr500ma.heater_source.*",
        "stimulator-thermo.temperatures.*",
    ]

    if t_start is None:
        t_start = aman.timestamps[0]

    if t_end is None:
        t_end = aman.timestamps[-1]

    # Load stimulator's data
    lspec = hkdb.LoadSpec(
        hkdb_cfg,
        fields=feed_list,
        start=t_start,
        end=t_end,
    )
    result = hkdb.load_hk(lspec)

    return result


def preprocessing(aman, hkdata, idxs=None, n_bins=40, delete_filtered_tod=True):
    """
    Preprocessing for stimulator data analysis using HK data.
    This includes getting timing against encoder signal, chopping status,
    timing cut for analysis, signal temperature, filtering of TOD, and making and fitting coadded data.

    For each detector, the raw signal is deconvolved using the IIR filter, and then either high-pass filtered (hpf),
    or both low-pass and high-pass filtered (lpf) to form a bandpass around the stimulator's chopping frequency.
    The filtered signal is co-added into chopper phase bins.
    In former case (hpf), coadded signal is fit to a seven-harmonic sine model to obtain its fundamental amplitude (a0).
    In the latter case (lpf), the co-added waveform is instead fit to a single-mode sine curve.

    Lowest chopping frequency data will be used for gain calculation,
    while higher frequencies will be used for time constant calculation.

    Args:
        aman: Axis manager of detector data.
        hkdata: Housekeeping data for aman.
        idxs: List of detector indices for calculation. If None, all detectors are calculated.
        n_bins: Number of bins for co-adding data.
        delete_filtered_tod: Whether to delete filtered TOD data after processing.
    Return:
        valid_gain: bool, True if gain can be calculated, False otherwise.
        valid_timeconstant: bool, True if time constant can be calculated, False otherwise.
    """
    det_mask = np.full(aman.dets.count, False)
    if idxs is None:
        det_mask[:] = True
    else:
        det_mask[idxs] = True

    valid_data = get_encoder_timing(aman, hkdata)  # Get timing against encoder t0
    aman.stm_cal.wrap(
        "sampling_rate", 1 / np.median(np.diff(aman.timestamps)), overwrite=True
    )
    get_chopping_status(aman)
    get_timing_cut(aman)
    get_chopping_freqs(aman)
    get_signal_temp(aman, hkdata)

    # Check if data is valid for gain or time constant calculation
    if not valid_data:
        valid_gain = False
        valid_timeconstant = False
        return valid_gain, valid_timeconstant
    else:
        valid_gain = True
        if aman.stm_cal.chopping_freqs.shape[0] <= 1:
            valid_timeconstant = False
        else:
            valid_timeconstant = True

    model, params_base = get_fit_params(cal_type="coadd")
    initialize_aman(aman, "coadd", model, n_bins)

    # Calculate filtering frequencies
    if valid_gain:
        filter_freqs = {}
        if round(aman.stm_cal.chopping_freqs[0]) != CHOPPING_FREQS["f1"]:
            filter_freqs["f1_gain"] = round(aman.stm_cal.chopping_freqs[0])
        else:
            filter_freqs["f1_gain"] = CHOPPING_FREQS["f1"]
        filtering(aman, filter_freqs, "gain")

        # Make and fit co-added data
        get_coadd_data(aman, "gain", n_bins, det_mask)
        fit_coadd_data(aman, "gain", det_mask, model, params_base)

        # Delete filtered TOD data if requested.
        # This operation will run for each filtering frequency in near future to save memory usage.
        if delete_filtered_tod:
            keys_to_delete = [key for key in aman._fields if key.startswith("signal_")]
            for key in keys_to_delete:
                aman.move(key, None)

    if valid_timeconstant:
        filter_freqs = {}
        for key, freq in zip(aman.stm_cal.chopping_freq_key.vals, aman.stm_cal.chopping_freqs):
            if key == 'f1_gain':
                continue

            if round(freq) != CHOPPING_FREQS[key]:
                filter_freqs[key] = round(freq)
            else:
                filter_freqs[key] = CHOPPING_FREQS[key]
        filtering(aman, filter_freqs, "timeconstant")

        # Make and fit co-added data
        get_coadd_data(aman, "timeconstant", n_bins, det_mask)
        fit_coadd_data(aman, "timeconstant", det_mask, model, params_base)

        # Delete filtered TOD data if requested.
        # This operation will run for each filtering frequency in near future to save memory usage.
        if delete_filtered_tod:
            keys_to_delete = [key for key in aman._fields if key.startswith("signal_")]
            for key in keys_to_delete:
                aman.move(key, None)

    return valid_gain, valid_timeconstant


def calc_gain(aman):
    """
    Calculate the gain of the detectors.
    Gain is the amplitude of the fundamental sine wave (a0)
    normalized to a typical temperature of 750 K by the measured heater/env temperatures.
    Results are written into aman.stm_cal.

    Args:
        aman: Axis manager of TOD data, including timestamps and raw signal.
    """
    heater_temp = aman.stm_cal.temps[aman.stm_cal.positions.vals == "heater"][0][0]
    env_temp = aman.stm_cal.temps[aman.stm_cal.positions.vals == "env"][0][0]

    arr = abs(aman.stm_cal["fit_coadd"]["lpf"]["f1_gain"]["a0"]) * (
        STM_NORMALIZE_TEMP / (heater_temp - env_temp)
    )
    aman.stm_cal.wrap("stm_gain", arr, [(0, "dets")], overwrite=True)


def calc_timeconstant(aman, idxs=None):
    """
    Calculate the time constant of the detectors and the readout delay.

    For each detector, TOD is co-added and fitted by a seven harmonics sine model to get its fundamental amplitude (a0)
    and phase delay (t0) per chopping frequency during preprocessing() function.
    In this function, fits amplitude (a0) vs. chopping frequency to a single-pole response model to get the time constant (tau),
    then fits phase delay (t0) vs. chopping frequency with tau fixed to get the readout delay (dt).
    Additionally fits phase delay (t0) vs. chopping frequency with both timeconstant and readout delay as free parameters.
    Results are written into aman.stm_cal

    Args:
        aman: Axis manager of TOD data, including timestamps and raw signal.
        hkdata: HK data for aman.
        idxs: List of detector indices for calculation. If None, all detectors are calculated.
    """
    det_mask = np.full(aman.dets.count, False)
    if idxs is None:
        det_mask[:] = True
    else:
        det_mask[idxs] = True

    models, params_bases = get_fit_params(cal_type="timeconstant")
    initialize_aman(aman, "timeconstant", models)
    for i_det, m in enumerate(det_mask):
        if not m:
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            continue

        # Fitting
        for filt_key in aman.stm_cal["fit_coadd"]._fields:
            a0s = [
                aman.stm_cal["fit_coadd"][filt_key][f_key]["a0"][i_det]
                for f_key in aman.stm_cal["fit_coadd"][filt_key]._fields
                if f_key != "f1_gain"
            ]
            t0s = [
                aman.stm_cal["fit_coadd"][filt_key][f_key]["t0"][i_det]
                for f_key in aman.stm_cal["fit_coadd"][filt_key]._fields
                if f_key != "f1_gain"
            ]

            # Edit a0 and t0 result
            # This flip of half period of sin function is needed because limiting a0 range to positive values makes fitting fails sometimes.
            # Fitting function will be changed to a*cos(t)+b*sin(t) instead of a0*sin(t+t0) in the future to prevent this issue.
            for i in range(len(t0s)):
                if a0s[i] < 0:
                    t0s[i] = t0s[i] - 0.5
                if i > 0:
                    if t0s[i] < t0s[i - 1]:
                        t0s[i] = t0s[i] + 1
            a0s = np.abs(a0s)
            t0s = np.array(t0s)

            f = [
                freq
                for key, freq in zip(
                    aman.stm_cal.chopping_freq_key.vals, aman.stm_cal.chopping_freqs
                )
                if key != "f1_gain"
            ]
            for fit_key in ["fit_amp", "fit_phase__fix_tau", "fit_phase__free"]:
                params = params_bases[fit_key].copy()
                if fit_key == "fit_amp":
                    if np.isnan(a0s).all():
                        continue

                    result = models[fit_key].fit(
                        a0s, params, f=f, method="least_squares"
                    )
                    fill_result(aman.stm_cal[fit_key][filt_key], result, i_det)
                else:
                    if np.isnan(t0s).all():
                        continue

                    if fit_key == "fit_phase__fix_tau":
                        if ~np.isnan(aman.stm_cal["fit_amp"][filt_key]["tau"][i_det]):
                            params["tau"].set(
                                value=aman.stm_cal["fit_amp"][filt_key]["tau"][i_det],
                                vary=False,
                            )
                            result = models[fit_key].fit(
                                -np.array(t0s) * 360,
                                params,
                                f=f,
                                method="least_squares",
                            )
                            fill_result(aman.stm_cal[fit_key][filt_key], result, i_det)
                    else:
                        result = models[fit_key].fit(
                            -np.array(t0s) * 360, params, f=f, method="least_squares"
                        )
                        fill_result(aman.stm_cal[fit_key][filt_key], result, i_det)

    aman.stm_cal.wrap(
        "stm_tau", aman.stm_cal["fit_amp"]["lpf"]["tau"], [(0, "dets")], overwrite=True
    )
    aman.stm_cal.wrap(
        "readout_delay",
        aman.stm_cal["fit_phase__fix_tau"]["lpf"]["dt"],
        [(0, "dets")],
        overwrite=True,
    )

    if "stm_gain" in aman.stm_cal:
        f = aman.stm_cal.chopping_freqs[aman.stm_cal.chopping_freq_key.vals == "f1"][0]
        correction_factor = 1 / func_response_amplitude(
            f=f, tau=aman.stm_cal["fit_amp"]["lpf"]["tau"], a=1
        )
        aman.stm_cal.wrap(
            "stm_gain_with_tau_correction",
            aman.stm_cal.stm_gain * correction_factor,
            [(0, "dets")],
            overwrite=True,
        )


def get_encoder_timing(aman, hkdata):
    """
    Get timing during one cycle of the chopping.
    t_enc: Time when the encoder signal is at state 0.
        This is Temps Atomique International(TAI) time with finer resolution than HK time.
    t_hk: HK time when the encoder signal is at state 0. This is UNIX time with 0.1 second resolution.
    frac_timing: Timing of each TOD data against encoder signal.
        The range is [0,1). During this range, the detector views the stimulator's heater and blackbody once each.

    Args:
        aman: Axis manager with the detector data.
        hkdata: The housekeeping data.

    Returns:
        bool: True if timing data is valid, False otherwise.
    """

    t_astropy = astropy.time.Time(
        hkdata.data["stimulator-enc.stim_enc.state"][0][0], format="unix"
    )
    leap_seconds = t_astropy.unix_tai - t_astropy.unix

    state = np.array(hkdata.data["stimulator-enc.stim_enc.state"][1])
    t_enc = (
        np.array(hkdata.data["stimulator-enc.stim_enc.timestamps_tai"][1])[state == 0]
        - leap_seconds
    )
    t_hk = np.array(hkdata.data["stimulator-enc.stim_enc.timestamps_tai"][0])[
        state == 0
    ]

    # Cut Encoder data not to contain next run data
    # Take 30s buffer from last TOD data
    mask = t_hk < aman.timestamps[-1] + 30
    t_enc = t_enc[mask]
    t_hk = t_hk[mask]

    # Get timing against encoder t0
    i_enc = 0
    frac_timing = np.full(aman.samps.count, np.nan)
    for i, t in enumerate(aman.timestamps):
        if i_enc >= len(t_enc) - 2:
            continue
        t1_enc_tmp = t_enc[i_enc]
        t2_enc_tmp = t_enc[i_enc + 1]

        while t2_enc_tmp < t:
            i_enc += 1
            t1_enc_tmp = t_enc[i_enc]
            t2_enc_tmp = t_enc[i_enc + 1]

        frac_timing[i] = (t - t1_enc_tmp) / (t2_enc_tmp - t1_enc_tmp)

    if "stm_samps" not in aman._axes.keys():
        stm_samps = core.IndexAxis("stm_samps", t_enc.size)
        stm_cal = core.AxisManager(stm_samps, aman.samps)
        aman.wrap("stm_cal", stm_cal, overwrite=True)
    aman.stm_cal.wrap("t_enc", t_enc, [(0, "stm_samps")], overwrite=True)
    aman.stm_cal.wrap("t_hk", t_hk, [(0, "stm_samps")], overwrite=True)

    valid_data = True
    if t_enc[-1] < aman.timestamps[-1]:
        print(
            f"Invalid HK data. last t_enc: {t_enc[-1]}, last TOD timestamp: {aman.timestamps[-1]}"
        )
        valid_data = False
    else:
        # Add timing against encoder to axis manager
        aman.stm_cal.wrap('frac_timing', np.array(frac_timing), [(0,'samps')], overwrite=True)

    return valid_data


def get_chopping_status(aman, min_step_duration=10, penalty=100):
    """
    Get timing when the chopping speed changes. Those are saved in the axis manager as aman.stm_cal.t_chopping_change.
    It also include two more timings. One is first timing when TOD starts(t=0).
    The other is last timing when TOD ends or chopping starts to end, whichever comes first.

    Args:
        aman: axis manager with aman.stm_cal field
        min_step_duration: Minimum duration to search for each chopping step.
            The unit is in seconds but it should be an integer.
            This is used to avoid small chopping speed fluctuations that are shorter than this duration.
        penalty: Penalty value for change point detection.
            This is used to control the sensitivity of change point detection.
            A larger penalty will result in fewer detected change points.
    """
    # Get chopping frequency using the encoder data
    xx0 = aman.stm_cal.t_enc[:-1]
    xx1 = aman.stm_cal.t_enc[1:]
    x = (xx0 + xx1) / 2 - aman.timestamps[0]
    y = 1 / (xx1 - xx0)  # Chopping frequency

    x_min = int(x.min())
    x_max = int(min(x.max(), aman.timestamps[-1] - aman.timestamps[0]))
    if x_min < 0:
        x_min = 0

    # Get 1s average data
    n_bins = x_max - x_min  # 1s bin
    data = [[] for _ in range(n_bins)]
    for i_bin in range(n_bins):
        cut = (i_bin <= x) & (x < i_bin + 1)
        data[i_bin] = y[cut]

    x_ave = np.linspace(x_min, x_max - 1, n_bins) + 0.5
    y_ave = np.array([np.nanmean(d) for d in data])

    # Get breakout points where the chopping speed changes
    model1 = rpt.Pelt(model="l2", min_size=min_step_duration).fit(y_ave)
    bkps = model1.predict(pen=penalty)  # index +1

    # Re-calculate the breakout points
    refined_bkps = []
    window = 5
    for b in bkps:
        left = max(0, b - window)
        right = min(len(y_ave), b + window)

        local_diff = np.abs(np.diff(y_ave[left:right]))

        # Cut small fluctuation
        if np.max(local_diff) < 1:
            continue
        idx = np.argmax(local_diff)

        refined_bkps.append(left + idx + 1)

    # Add last point
    refined_bkps.append(x_ave.size)  # index +1

    t_chopping_change = x_ave[np.array(refined_bkps) - 1]
    t_chopping_change = np.concatenate([[0], t_chopping_change], 0)

    aman.stm_cal.wrap("t_chopping_change", t_chopping_change, overwrite=True)


def get_timing_cut(aman, dt_gain=60, dt_timeconstant=10, dt_wait=9, dt_buffer=2):
    """
    Get timing range for analysis.

    Args:
        aman: axis manager with aman.stm_cal field
        dt_gain: Time duration for gain analysis.
        dt_timeconstant: Time duration for time constant analysis from the time when chopping speed changes and dt_wait passed.
        dt_wait: Waiting time after chopping speed changes to avoid chopper's acceleration time.
        dt_buffer: Time buffer to avoid ringing by high-pass filter at the beginning of tod.
    """
    if len(aman.stm_cal.t_chopping_change) == 2:
        cal_type = "gain"
    elif len(aman.stm_cal.t_chopping_change) > 2:
        cal_type = "gain_timeconstant"
    else:
        raise ValueError("Chopping information is incorrect.")

    t_cuts = []

    t_start = dt_buffer + aman.stm_cal.t_chopping_change[0]
    t_end = min(t_start + dt_gain, aman.stm_cal.t_chopping_change[1])
    t_cuts.append((t_start, t_end))

    if cal_type == "gain_timeconstant":
        for i in range(len(aman.stm_cal.t_chopping_change) - 1):
            t_start = aman.stm_cal.t_chopping_change[i] + dt_wait
            if i == 0:
                t_start = (
                    dt_buffer + aman.stm_cal.t_chopping_change[i]
                )  # 2 seconds to avoid ringing by high-pass filter
            t_end = min(
                t_start + dt_timeconstant, aman.stm_cal.t_chopping_change[i + 1]
            )
            t_cuts.append((t_start, t_end))

    keys = ["f1_gain"]
    if cal_type == "gain_timeconstant":
        for i in range(len(aman.stm_cal.t_chopping_change) - 1):
            keys.append(f"f{int(i + 1)}")
    label_freqs = core.LabelAxis("chopping_freq_key", keys)
    aman.stm_cal.add_axis(label_freqs)

    aman.stm_cal.wrap(
        "t_cuts", np.array(t_cuts), [(0, "chopping_freq_key")], overwrite=True
    )


def get_chopping_freqs(aman):
    """
    Get chopping frequencies based on the timing cuts.
    This chopping frequency will be used for analysis.

    Args:
        aman: axis manager with aman.stm_cal.t_cuts field
    """
    # Get chopping frequency using the encoder data
    xx0 = aman.stm_cal.t_enc[:-1]
    xx1 = aman.stm_cal.t_enc[1:]
    x = (xx0 + xx1) / 2 - aman.timestamps[0]
    y = 1 / (xx1 - xx0)  # Chopping frequency

    # Get average of chopping frequency
    chopping_freqs = []
    for t_start, t_end in aman.stm_cal.t_cuts:
        y_ave = y[(x >= t_start) & (x < t_end)]
        if y_ave.size > 0:
            chopping_freqs.append(np.average(y_ave))
        else:
            raise ValueError(
                "No valid chopping frequency data available. Check the timing cuts aman.stm_cal.t_cuts."
            )

    aman.stm_cal.wrap(
        "chopping_freqs",
        np.array(chopping_freqs),
        [(0, "chopping_freq_key")],
        overwrite=True,
    )


def get_signal_temp(aman, hkdata):
    """
    Get stimulator signal temperature at each chopping frequency.

    Args:
        aman: axis manager with aman.stm_cal field
        hkdata: Housekeeping data including temperature data.
    """

    position_keys = ["heater", "chopper_rear", "chopper_front", "air"]
    keys = [
        "stimulator-thermo.temperatures.Channel_0_T",
        "stimulator-thermo.temperatures.Channel_4_T",
        "stimulator-thermo.temperatures.Channel_6_T",
        "stimulator-thermo.temperatures.Channel_5_T",
    ]
    temps = {}
    for position_key in position_keys:
        temps[position_key] = []

    for key, position_key in zip(keys, position_keys):
        x = hkdata.data[key][0] - aman.timestamps[0]
        y = hkdata.data[key][1] + 273.15  # Convert degrees from Celsius to Kelvin

        for t_min, t_max in aman.stm_cal.t_cuts:
            arr = y[(t_min <= x) & (x < t_max)]
            y_mean = np.mean(arr) if arr.size > 0 else np.nan
            temps[position_key].append(y_mean)

        temps[position_key] = np.array(temps[position_key])

    temps["env"] = (temps["chopper_rear"] + temps["chopper_front"] + temps["air"]) / 3
    position_keys.append("env")

    label_positions = core.LabelAxis("positions", position_keys)
    aman.stm_cal.add_axis(label_positions)
    arr = np.array(list(temps.values()))

    aman.stm_cal.wrap(
        "temps", arr, [(0, "positions"), (1, "chopping_freq_key")], overwrite=True
    )


def filtering(
    aman,
    filter_freqs,
    cal_type,
    hpf_cutoff=1,
    hpf_width=2,
    lpf_cutoff_factor=1.5,
    lpf_width_fraction=1 / 5,
):
    """
    Filtering signal data.

    Args:
        aman: axis manager
        cal_type: 'gain' or 'timeconstant'. type of calibration
        filter_freqs: dictionary of chopping frequencies to be used for filtering
        hpf_cutoff: cutoff frequency for high-pass filter in Hz. Default is 1Hz.
        hpf_width: full width of high-pass filter in Hz. Default is 2Hz.
        lpf_cutoff_factor: cutoff frequency for low-pass filter is calculated by multiplying this factor and chopping frequency. Default is 1.5.
        lpf_width_fraction: full width of low-pass filter is calculated by multiplying this factor and cutoff frequency. Default is 1/5.
    """

    # Invert IIR filter
    iirc_filter = tod_ops.filters.iir_filter(aman, invert=True)
    signal_new = tod_ops.fourier_filter(aman, iirc_filter, signal_name="signal")
    aman.wrap("signal_iirc", signal_new, [(0, "dets"), (1, "samps")], overwrite=True)

    # Make HPFed data
    hpf = tod_ops.filters.high_pass_sine2(hpf_cutoff, hpf_width)
    signal_new = tod_ops.fourier_filter(aman, hpf, signal_name="signal_iirc")
    aman.wrap("signal_hpf", signal_new, [(0, "dets"), (1, "samps")], overwrite=True)

    # Make LPFed data
    if cal_type == "gain":
        filter_cutoff = lpf_cutoff_factor * filter_freqs["f1_gain"]
        lpf = tod_ops.filters.low_pass_sine2(
            filter_cutoff, filter_cutoff * lpf_width_fraction
        )
        signal_new = tod_ops.fourier_filter(aman, lpf, signal_name="signal_hpf")
        aman.wrap("signal_lpf", signal_new, [(0, "dets"), (1, "samps")], overwrite=True)

    elif cal_type == "timeconstant":
        for key, chopping_freq in filter_freqs.items():
            filter_cutoff = lpf_cutoff_factor * chopping_freq
            lpf = tod_ops.filters.low_pass_sine2(
                filter_cutoff, filter_cutoff * lpf_width_fraction
            )
            signal_new = tod_ops.fourier_filter(aman, lpf, signal_name="signal_hpf")
            aman.wrap(
                f"signal_lpf_{key}",
                signal_new,
                [(0, "dets"), (1, "samps")],
                overwrite=True,
            )

    else:
        raise ValueError(
            f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'."
        )

    # Fill filtering parameters to axis manager
    if "filtering_params" not in aman.stm_cal.keys():
        aman.stm_cal.wrap("filtering_params", core.AxisManager())

    aman.stm_cal.filtering_params.wrap("hpf_cutoff", hpf_cutoff, overwrite=True)
    aman.stm_cal.filtering_params.wrap("hpf_width", hpf_width, overwrite=True)
    aman.stm_cal.filtering_params.wrap(
        "lpf_cutoff_factor", lpf_cutoff_factor, overwrite=True
    )
    aman.stm_cal.filtering_params.wrap(
        "lpf_width_fraction", lpf_width_fraction, overwrite=True
    )

    if cal_type == "gain":
        aman.stm_cal.filtering_params.wrap(
            "filter_freq_gain",
            np.array(list(filter_freqs.values())),
            overwrite=True,
        )
    elif cal_type == "timeconstant":
        aman.stm_cal.filtering_params.wrap(
            "filter_freq_tau",
            np.array(list(filter_freqs.values())),
            overwrite=True,
        )


def initialize_aman(aman, init_type, model, n_bins=None):
    """
    Initialize the axis manager with the specified type.

    Args:
        aman: axis manager
        init_type: type of initialization. 'coadd' or 'timeconstant'
        model: model for the fit
        n_bins: # of bins for co-added data
    """
    filt_keys = ["iirc", "hpf", "lpf"]
    if init_type == "coadd":
        freq_keys = ["f1_gain"] + list(
            CHOPPING_FREQS.keys()
        )  # Need to be corrected in the future
    elif init_type == "timeconstant":
        freq_keys = CHOPPING_FREQS.keys()
    else:
        raise ValueError(
            f"'{init_type}' is a wrong initialization type. Please specify 'coadd' or 'timeconstant'."
        )

    if init_type == "coadd":
        if n_bins is None:
            raise ValueError("n_bins must be specified for coadd initialization.")

    def ensure_wrapped(parent_field, key, arr=None, axis=None):
        if key not in parent_field:
            if arr is None and axis is None:
                parent_field.wrap(key, core.AxisManager(aman.dets))
            elif arr is None:
                parent_field.wrap(key, axis)
            elif axis is None:
                parent_field.wrap(key, arr, [(0, "dets")])
            else:
                parent_field.wrap(key, arr, axis)
        return parent_field[key]

    if init_type == "coadd":
        # Initalize axis manager for co-added data. Overwrite if already exists
        if "coadd_data" not in aman.stm_cal.keys():
            axis_bins = core.IndexAxis("stm_coadd_bins", n_bins)
            aman.stm_cal.wrap("coadd_data", core.AxisManager(aman.dets, axis_bins))

        for filt_key in filt_keys:
            if filt_key not in aman.stm_cal.coadd_data.keys():
                aman.stm_cal.coadd_data.wrap(filt_key, core.AxisManager())

            for freq_key in freq_keys:
                if freq_key not in aman.stm_cal.coadd_data[filt_key].keys():
                    aman.stm_cal.coadd_data[filt_key].wrap(
                        freq_key,
                        core.AxisManager(
                            aman.dets, aman.stm_cal.coadd_data.stm_coadd_bins
                        ),
                    )

                    for key in ["x", "y", "yerr"]:
                        arr = np.full((aman.dets.count, n_bins), np.nan)
                        aman.stm_cal.coadd_data[filt_key][freq_key].wrap(
                            f"{key}",
                            arr,
                            [(0, "dets"), (1, "stm_coadd_bins")],
                            overwrite=True,
                        )

        # Initialize axis manager for fit results. Do not overwrite if already exists.
        field = ensure_wrapped(aman.stm_cal, "fit_coadd")

        for filt_key in filt_keys:
            if filt_key == "iirc":
                continue
            ensure_wrapped(aman.stm_cal["fit_coadd"], filt_key)

            for freq_key in freq_keys:
                field = ensure_wrapped(
                    aman.stm_cal["fit_coadd"][filt_key],
                    freq_key,
                    axis=core.AxisManager(
                        aman.dets, aman.stm_cal.coadd_data.stm_coadd_bins
                    ),
                )

                for key in ["chisqr", "redchi"]:
                    ensure_wrapped(field, key, np.full(aman.dets.count, np.nan))
                ensure_wrapped(
                    field,
                    "weight",
                    arr=np.full((aman.dets.count, n_bins), np.nan),
                    axis=[(0, "dets"), (1, "stm_coadd_bins")],
                )

                for key in model.param_names:
                    if key == "t":
                        continue
                    ensure_wrapped(field, key, np.full(aman.dets.count, np.nan))
                    ensure_wrapped(
                        field, f"{key}_stderr", np.full(aman.dets.count, np.nan)
                    )

    if init_type == "timeconstant":
        # Initialize axis manager for timeconstant fit results. Do not overwrite if already exists.  # noqa: E115
        for fit_key in ["fit_amp", "fit_phase__fix_tau", "fit_phase__free"]:
            ensure_wrapped(aman.stm_cal, fit_key)

            for filt_key in filt_keys:
                if filt_key == "iirc":
                    continue
                index_axis = core.IndexAxis(
                    "ndata_taufit", aman.stm_cal.filtering_params.filter_freq_tau.size
                )
                field = ensure_wrapped(
                    aman.stm_cal[fit_key],
                    filt_key,
                    axis=core.AxisManager(aman.dets, index_axis),
                )

                for key in ["chisqr", "redchi"]:
                    ensure_wrapped(field, key, np.full(aman.dets.count, np.nan))
                ensure_wrapped(
                    field,
                    "weight",
                    arr=np.full(
                        (
                            aman.dets.count,
                            aman.stm_cal.filtering_params.filter_freq_tau.size,
                        ),
                        np.nan,
                    ),
                    axis=[(0, "dets"), (1, "ndata_taufit")],
                )

                for key in model[fit_key].param_names:
                    if key == "f":
                        continue
                    ensure_wrapped(field, key, np.full(aman.dets.count, np.nan))
                    ensure_wrapped(
                        field, f"{key}_stderr", np.full(aman.dets.count, np.nan)
                    )
                ensure_wrapped(
                    field,
                    "data",
                    arr=np.full(
                        (
                            aman.dets.count,
                            aman.stm_cal.filtering_params.filter_freq_tau.size,
                        ),
                        np.nan,
                    ),
                    axis=[(0, "dets"), (1, "ndata_taufit")],
                )


def get_coadd_data(aman, cal_type, n_bins, det_mask):
    """
    Making co-added data.

    Args:
        aman: axis manager of tod data, including timestamps and tod signal
        cal_type: 'all', 'gain' or 'timeconstant'. type of calibration. If all, co-add data for all frequencies will be calculated.
        n_bins: # of bins for co-added data
        t_min: Minimum time for the data analysis
        t_max: Maximum time for the data analysis
    """
    t0 = aman.timestamps[0]
    bins = np.linspace(0, 1 - 1 / n_bins, n_bins)
    x = bins + 1 / n_bins / 2
    filt_keys = ["iirc", "hpf", "lpf"]
    if cal_type == "gain":
        freq_keys = ["f1_gain"]
    elif cal_type == "timeconstant":
        freq_keys = CHOPPING_FREQS.keys()
    elif cal_type == "all":
        freq_keys = ["f1_gain"] + list(CHOPPING_FREQS.keys())
    else:
        raise ValueError(
            f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'."
        )

    # Get cuts for co-addition
    cuts = {}
    for freq_key in freq_keys:
        idx = np.where(aman.stm_cal.chopping_freq_key.vals == freq_key)[0][0]
        t_min, t_max = aman.stm_cal.t_cuts[idx]

        cut1 = (t_min <= aman.timestamps - t0) & (aman.timestamps - t0 < t_max)

        cuts[freq_key] = []
        for i_bin in range(n_bins):
            cut2 = (i_bin / n_bins <= aman.stm_cal.frac_timing) & (
                aman.stm_cal.frac_timing < (i_bin + 1) / n_bins
            )
            cut = cut1 & cut2
            cuts[freq_key].append(cut)

    # Calculate co-added data
    for i_det, m in enumerate(det_mask):
        if not m:
            continue
        if not np.isfinite(aman.signal[i_det]).all():
            continue

        for filt_key in filt_keys:
            for freq_key in freq_keys:
                if filt_key == "hpf" or filt_key == "iirc":
                    key = f"signal_{filt_key}"
                elif filt_key == "lpf":
                    if freq_key == "f1_gain":
                        key = "signal_lpf"
                    else:
                        key = f"signal_lpf_{freq_key}"

                data = [[] for _ in range(n_bins)]
                for i_bin in range(n_bins):
                    data[int(i_bin)] = aman[key][i_det][cuts[freq_key][i_bin]]

                y = np.array(
                    [np.nanmean(d) if not np.isnan(d).all() else np.nan for d in data]
                )
                yerr = np.array(
                    [
                        np.array(d).std(ddof=1) / np.sqrt(len(d))
                        if len(d) > 0 and not np.isnan(d).all()
                        else np.nan
                        for d in data
                    ]
                )

                aman.stm_cal.coadd_data[filt_key][freq_key]["x"][i_det] = x
                aman.stm_cal.coadd_data[filt_key][freq_key]["y"][i_det] = y
                aman.stm_cal.coadd_data[filt_key][freq_key]["yerr"][i_det] = yerr


def fit_coadd_data(aman, cal_type, det_mask, model, params_base):
    """
    Fit co-added data for each detector.

    Args:
        aman: axis manager containing the coadded data
        cal_type: 'all', 'gain' or 'timeconstant'. type of calibration. If all, co-add data for all frequencies will be calculated.
        det_mask: boolean mask for the detectors to be fitted
        model: lmfit model for fitting
        params_base: base parameters for fitting
    """
    if cal_type == "all":
        freq_keys = aman.stm_cal.coadd_data["hpf"]
    elif cal_type == "gain":
        freq_keys = ["f1_gain"]
    elif cal_type == "timeconstant":
        freq_keys = CHOPPING_FREQS.keys()
    else:
        raise ValueError(
            f"'{cal_type}' is a wrong type. Please specify 'all', 'gain' or 'timeconstant'."
        )

    for i_det, m in enumerate(det_mask):
        if not m:
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            continue

        # Fitting
        for filt_key in [x for x in aman.stm_cal.coadd_data._fields if x != "iirc"]:
            for f_key in freq_keys:
                params = params_base.copy()

                x = aman.stm_cal.coadd_data[filt_key][f_key]["x"][i_det]
                y = aman.stm_cal.coadd_data[filt_key][f_key]["y"][i_det]
                yerr = aman.stm_cal.coadd_data[filt_key][f_key]["yerr"][i_det]

                mask = np.isfinite(y)
                x = x[mask]
                y = y[mask]
                yerr = yerr[mask]

                if y.size == 0:
                    continue

                if filt_key == "lpf":
                    for i in range(1, 7):
                        params[f"a{i}"].set(value=0, vary=False)
                        params[f"t{i}"].set(value=0, vary=False)
                    result = model.fit(y, params, t=x, weights=1 / np.array(yerr))
                else:
                    result = model.fit(y, params, t=x, weights=1 / np.array(yerr))

                fill_result(
                    aman.stm_cal.fit_coadd[filt_key][f_key], result=result, i_det=i_det
                )


def fill_result(field, result, i_det):
    """
    Fill lmfit result into axis manager.

    Args:
        field: field in the axis manager to fill data
        result: lmfit result
        i_det: detector index
        fit_key: type of fit. 'fit_coadd', 'fit_amp', 'fit_phase__fix_tau', or 'fit_phase__free'
    """
    for key in result.params.keys():
        field[key][i_det] = result.params[key].value
        field[f"{key}_stderr"][i_det] = result.params[key].stderr
        field["chisqr"][i_det] = result.chisqr
        field["redchi"][i_det] = result.redchi

    if "tau" in result.params.keys():
        field["data"][i_det] = result.data
    else:
        field["weight"][i_det] = result.weights


def get_fit_params(cal_type):
    """
    Get fit parameters for fitting.

    Args:
        cal_type: 'coadd' or 'timeconstant'. type of calibration

    Return:
        model: lmfit model for fitting
        params_base: lmfit parameters for fitting
    """
    amps_init = [1e-3, 2e-4, 2e-4, 1e-4, 1e-4, 1e-5, 1e-5]

    if cal_type == "coadd":
        model = lmfit.Model(func_sines)

        params_base = lmfit.Parameters()
        for i, a in enumerate(amps_init):
            params_base.add(f"a{i}", value=a)
            params_base.add(f"t{i}", value=0.5, min=0, max=1)
    elif cal_type == "timeconstant":
        model = {}
        model["fit_amp"] = lmfit.Model(func_response_amplitude)
        model["fit_phase__fix_tau"] = lmfit.Model(
            func_response_phase_with_dt, independent_vars=["f"]
        )
        model["fit_phase__free"] = lmfit.Model(
            func_response_phase_with_dt, independent_vars=["f"]
        )

        params_base = {}
        params_base["fit_amp"] = lmfit.Parameters()
        params_base["fit_phase__fix_tau"] = lmfit.Parameters()
        params_base["fit_phase__free"] = lmfit.Parameters()
        params_base["fit_amp"].add("a", value=1e-3, min=0, max=1)
        params_base["fit_amp"].add("tau", value=1e-3, min=0, max=1)
        params_base["fit_phase__fix_tau"].add("tau")
        params_base["fit_phase__free"].add("tau", value=1e-3, min=0, max=0.1)
        for fit_key in ["fit_phase__fix_tau", "fit_phase__free"]:
            params_base[fit_key].add("theta_geo", value=0, min=-90, max=90)
            params_base[fit_key].add("dt", value=0.125 * 1e-3, min=-3e-3, max=3e-3)

    return model, params_base

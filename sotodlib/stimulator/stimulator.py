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


def preprocessing(aman, hkdata):
    """
    Preprocessing for stimulator data analysis using HK data.
    This includes getting timing against encoder signal, chopping status, timing cut for analysis, and signal temperature.

    Args:
        aman: Axis manager of detector data.
        hkdata: Housekeeping data for aman.

    Return:
        valid_gain: bool, True if gain can be calculated, False otherwise.
        valid_timeconstant: bool, True if time constant can be calculated, False otherwise.
    """

    valid_data = get_encoder_timing(aman, hkdata)  # Get timing against encoder t0
    aman.stm_cal.wrap(
        "sampling_rate", 1 / np.median(np.diff(aman.timestamps)), overwrite=True
    )
    get_chopping_status(aman)
    get_timing_cut(aman)
    get_signal_temp(aman, hkdata)

    if not valid_data:
        valid_gain = False
        valid_timeconstant = False
    else:
        valid_gain = True
        if aman.stm_cal.chopping_freqs.shape[0] <= 1:
            valid_timeconstant = False
        else:
            valid_timeconstant = True

    return valid_gain, valid_timeconstant


def calc_gain(aman, hkdata, idxs=None, n_bins=40):
    """
    Calculate the gain of the detectors.

    Args:
        aman: Axis manager of TOD data, including timestamps and raw signal.
        hkdata: HK data for aman.
        idxs: List of detector indices for calculation. If None, all detectors are calculated.
        n_bins: Number of bins for co-adding data.
    """
    det_mask = np.full(aman.dets.count, False)
    if idxs is None:
        det_mask[:] = True
    else:
        det_mask[idxs] = True


    chopping_freqs = {}
    if round(aman.stm_cal.chopping_freqs[0]) != CHOPPING_FREQS["f1"]:
        chopping_freqs["f1_gain"] = round(aman.stm_cal.chopping_freqs[0])
    else:
        chopping_freqs["f1_gain"] = CHOPPING_FREQS["f1"]

    filtering(aman, chopping_freqs, "gain")

    model, params_base = get_fit_params(cal_type="coadd")

    # Make and get co-added data
    #fit_result = get_dicts(aman, "gain")
    initialize_aman(aman, 'coadd', model, n_bins)

    get_coadd_data(aman, "gain", n_bins, det_mask)

    fit_coadd_data(aman, 'gain', det_mask, model, params_base)

    #for i_det, m in enumerate(det_mask):
    #    if not m:
    #        continue

    #    # Strange data check
    #    if not np.isfinite(aman.signal[i_det]).all():
    #        continue

    #    # Fitting
    #    for filt_key in fit_result["fit_coadd"].keys():
    #        params = params_base.copy()

    #        x = aman.stm_cal.coadd_data[filt_key]["f1_gain"]["x"][i_det]
    #        y = aman.stm_cal.coadd_data[filt_key]["f1_gain"]["y"][i_det]
    #        yerr = aman.stm_cal.coadd_data[filt_key]["f1_gain"]["yerr"][i_det]

    #        mask = np.isfinite(y)
    #        x = x[mask]
    #        y = y[mask]
    #        yerr = yerr[mask]

    #        if y.size == 0:
    #            continue

    #        if filt_key == "lpf":
    #            for i in range(1, 7):
    #                params[f"a{i}"].set(value=0, vary=False)
    #                params[f"t{i}"].set(value=0, vary=False)
    #            result = model.fit(y, params, t=x, weights=1 / np.array(yerr))
    #        else:
    #            result = model.fit(y, params, t=x, weights=1 / np.array(yerr))

    #        fit_result["fit_coadd"][filt_key]["f1_gain"][i_det] = result

    fill_data(aman, fit_result, cal_type="gain")


def calc_timeconstant(aman, hkdata, idxs=None, n_bins=40):
    """
    Calculate the time constant of the detectors.

    Args:
        aman: Axis manager of TOD data, including timestamps and raw signal.
        hkdata: HK data for aman.
        idxs: List of detector indices for calculation. If None, all detectors are calculated.
        n_bins: Number of bins for co-adding data.

    Return:
        bool: True if data is valid and calculation is performed, False otherwise.
    """
    det_mask = np.full(aman.dets.count, False)
    if idxs is None:
        det_mask[:] = True
    else:
        det_mask[idxs] = True


    chopping_freqs = {}
    for i, (key, f) in enumerate(CHOPPING_FREQS.items()):
        if round(aman.stm_cal.chopping_freqs[0]) != f:
            chopping_freqs[key] = round(aman.stm_cal.chopping_freqs[i])
        else:
            chopping_freqs[key] = f

    filtering(aman, chopping_freqs, "timeconstant")


    #fit_result = get_dicts(aman, "timeconstant")
    #get_coadd_data(aman, "timeconstant", n_bins, det_mask)
    model, params_base = get_fit_params(cal_type='coadd')
    initialize_aman(aman, 'coadd', model, n_bins)

    get_coadd_data(aman, "timeconstant", n_bins, det_mask)

    fit_coadd_data(aman, 'timeconstant', det_mask, model, params_base)

    models, params_bases = get_fit_params(cal_type="timeconstant")
    for i_det, m in enumerate(det_mask):
        if not m:
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            continue

        # Fitting
        for filt_key in fit_result["fit_coadd"].keys():
            #for f_key in chopping_freqs.keys():
                #params = params_bases["fit_coadd"].copy()

                #x = aman.stm_cal.coadd_data[filt_key][f_key]["x"][i_det]
                #y = aman.stm_cal.coadd_data[filt_key][f_key]["y"][i_det]
                #yerr = aman.stm_cal.coadd_data[filt_key][f_key]["yerr"][i_det]

                #mask = np.isfinite(y)
                #x = x[mask]
                #y = y[mask]
                #yerr = yerr[mask]

                #if y.size == 0:
                #    continue

                #if filt_key == "lpf":
                #    for i in range(1, 7):
                #        params[f"a{i}"].set(value=0, vary=False)
                #        params[f"t{i}"].set(value=0, vary=False)
                #    result = models["fit_coadd"].fit(
                #        y, params, t=x, weights=1 / np.array(yerr)
                #    )
                #else:
                #    result = models["fit_coadd"].fit(
                #        y, params, t=x, weights=1 / np.array(yerr)
                #    )

                #fit_result["fit_coadd"][filt_key][f_key][i_det] = result

            a0s = [
                aman.stm_cal["fit_coadd"][filt_key][f_key]["a0"][i_det]
                for f_key in aman.stm_cal["fit_coadd"][filt_key]
            ]
            t0s = [
                aman.stm_cal["fit_coadd"][filt_key][f_key]["t0"][i_det]
                for f_key in aman.stm_cal["fit_coadd"][filt_key]
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

            f = [chopping_freqs[f_key] for f_key in chopping_freqs.keys()]
            for fit_key in ["fit_amp", "fit_phase__fix_tau", "fit_phase__free"]:
                params = params_bases[fit_key].copy()
                if fit_key == "fit_amp":
                    if np.isnan(a0s).all():
                        continue

                    result = models[fit_key].fit(
                        a0s, params, f=f, method="least_squares"
                    )
                    fill_result(aman, result, i_det, fit_key)
                else:
                    if np.isnan(t0s).all():
                        continue

                    if fit_key == "fit_phase__fix_tau":
                        if ~np.isnan(aman.stm_cal["fit_amp"][filt_key]['tau'][i_det]):
                            params["tau"].set(
                                value=aman.stm_cal["fit_amp"][filt_key]["tau"][i_det],
                                vary=False
                            )
                            result = models[fit_key].fit(
                                -np.array(t0s) * 360,
                                params,
                                f=f,
                                method="least_squares",
                            )
                            fill_result(aman, result, i_det, fit_key)
                    else:
                        result = models[fit_key].fit(
                            -np.array(t0s) * 360, params, f=f, method="least_squares"
                        )
                        fill_result(aman, result, i_det, fit_key)

    #fill_data(aman, fit_result, cal_type="timeconstant")


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
        stm_cal = core.AxisManager(stm_samps)
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
        # aman.stm_cal.wrap('frac_timing', np.array(frac_timing), [(0,'samps')], overwrite=True)
        aman.wrap("frac_timing", np.array(frac_timing), [(0, "samps")], overwrite=True)

    return valid_data


def get_chopping_status(aman, min_step_duration=10, penalty=100, delta_t=5, wait_t=5):
    """
    Get chopping frequency and timing when the chopping speed changes.

    Args:
        aman: axis manager with aman.stm_cal field
        min_step_duration: Minimum duration to search for each chopping step.
            The unit is in seconds but it should be an integer.
            This is used to avoid small chopping speed fluctuations that are shorter than this duration.
        penalty: Penalty value for change point detection.
            This is used to control the sensitivity of change point detection.
            A larger penalty will result in fewer detected change points.
        delta_t: Time duration for averaging chopping frequency in seconds.
        wait_t: Waiting time after chopping speed changes to avoid chopper's acceleration time. The unit is in seconds.
            The averaging of chopping frequency will be calculated after this waiting time.

    """
    xx0 = aman.stm_cal.t_enc[:-1]
    xx1 = aman.stm_cal.t_enc[1:]
    x = (xx0 + xx1) / 2 - aman.timestamps[0]
    y = 1 / (xx1 - xx0)

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

    # Get average of chopping frequency
    wait_t = wait_t
    delta_t = delta_t
    n_freq = len(refined_bkps)
    chopping_freqs = []
    for i_freq in range(n_freq):
        if i_freq == 0:
            i_min = 0
        else:
            i_min = refined_bkps[i_freq - 1] - 1 + wait_t

        chopping_freqs.append(np.average(y_ave[i_min : i_min + delta_t]))

    t_chopping_change = x_ave[np.array(refined_bkps) - 1]
    t_chopping_change = np.concatenate([[0], t_chopping_change], 0)

    aman.stm_cal.wrap("t_chopping_change", t_chopping_change, overwrite=True)
    aman.stm_cal.wrap("chopping_freqs", np.array(chopping_freqs), overwrite=True)


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


def get_signal_temp(aman, hkdata):
    """
    Get signal temperature for chopping frequency.

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
    arr = [x for x in temps.values()]
    arr = np.array(arr)

    aman.stm_cal.wrap(
        "temps", arr, [(0, "positions"), (1, "chopping_freq_key")], overwrite=True
    )


def filtering(
    aman,
    chopping_freqs,
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
        chopping_freqs: dictionary of chopping frequencies
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
        filter_cutoff = lpf_cutoff_factor * chopping_freqs["f1_gain"]
        lpf = tod_ops.filters.low_pass_sine2(
            filter_cutoff, filter_cutoff * lpf_width_fraction
        )
        signal_new = tod_ops.fourier_filter(aman, lpf, signal_name="signal_hpf")
        aman.wrap("signal_lpf", signal_new, [(0, "dets"), (1, "samps")], overwrite=True)

    elif cal_type == "timeconstant":
        for key, chopping_freq in chopping_freqs.items():
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
            "chopping_freqs_gain",
            np.array(list(chopping_freqs.values())),
            overwrite=True,
        )
    elif cal_type == "timeconstant":
        aman.stm_cal.filtering_params.wrap(
            "chopping_freqs_tau",
            np.array(list(chopping_freqs.values())),
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
        freq_keys = ['f1_gain'] + list(CHOPPING_FREQS.keys())# Need to be corrected in the future
    elif init_type == "timeconstant":
        freq_keys = CHOPPING_FREQS.keys()
    else:
        raise ValueError(
            f"'{init_type}' is a wrong initialization type. Please specify 'coadd' or 'timeconstant'."
        )

    if init_type == "coadd":
        if n_bins is None:
            raise ValueError("n_bins must be specified for coadd initialization.")


    def get_or_wrap(parent_field, key, arr=None):
        if key not in parent_field:
            if arr is None:
                parent_field.wrap(key, core.AxisManager(aman.dets))
            else:
                parent_field.wrap(key, arr, [(0, "dets")])
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
                        core.AxisManager(aman.dets, aman.stm_cal.coadd_data.stm_coadd_bins),
                    )

                for key in ["x", "y", "yerr"]:
                    arr = np.full((aman.dets.count, n_bins), np.nan)
                    aman.stm_cal.coadd_data[filt_key][freq_key].wrap(
                        f"{key}", arr, [(0, "dets"), (1, "stm_coadd_bins")], overwrite=True
                    )

        # Initialize axis manager for fit results. Do not overwrite if already exists.
        field = get_or_wrap(aman.stm_cal, "fit_coadd")

        for filt_key in filt_keys:
            if filt_key == 'iirc':
                continue
            field = get_or_wrap(field, filt_key)

            for freq_key in freq_keys:
                field = get_or_wrap(field, freq_key)

                for key in ["chisqr", "redchi", "weight"]:
                    get_or_wrap(field, key, np.full(aman.dets.count, np.nan))

                for key in model.param_names:
                    if key == 't':
                        continue
                    get_or_wrap(field, key, np.full(aman.dets.count, np.nan))
                    get_or_wrap(field, f'{key}_stderr', np.full(aman.dets.count, np.nan))

    if init_type == "timeconstant":
    # Initialize axis manager for timeconstant fit results. Do not overwrite if already exists.  # noqa: E115
        for fit_key in ["fit_amp", "fit_phase__fix_tau", "fit_phase__free"]:
            field = get_or_wrap(aman.stm_cal, fit_key)

            for filt_key in filt_keys:
                if filt_key == 'iirc':
                    continue

                if filt_key not in field:
                    field.wrap(
                        filt_key,
                        core.AxisManager(
                            aman.dets,
                            core.IndexAxis(
                                "ndata_taufit",
                                aman.stm_cal.filtering_params.chopping_freqs_tau.size,
                            ),
                        ),
                        overwrite=True,
                    )
                field = field[filt_key]

                for key in ["chisqr", "redchi", "weight"]:
                    get_or_wrap(field, key, np.full(aman.dets.count, np.nan))

                for key in model[fit_key].param_names:
                    if key == 'f':
                        continue
                    get_or_wrap(field, key, np.full(aman.dets.count, np.nan))
                    get_or_wrap(field, f'{key}_stderr', np.full(aman.dets.count, np.nan))
                field.wrap("data", arr, [(0, "dets"), (1, "ndata_taufit")])


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
        freq_keys = ['f1_gain'] + list(CHOPPING_FREQS.keys())
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
            cut2 = (i_bin / n_bins <= aman.frac_timing) & (
                aman.frac_timing < (i_bin + 1) / n_bins
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
        freq_keys = aman.stm_cal.coadd_data['hpf']
    if cal_type == "gain":
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
        for filt_key in aman.stm_cal.coadd_data:
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
                    result = model.fit(
                        y, params, t=x, weights=1 / np.array(yerr)
                    )
                else:
                    result = model.fit(
                        y, params, t=x, weights=1 / np.array(yerr)
                    )

                fill_result(aman, result=result, i_det=i_det, fit_key="fit_coadd")


def fill_result(aman, result, i_det, fit_key):
    """
    Fill lmfit result into axis manager.

    Args:
        aman: axis manager to fill
        result: lmfit result
        i_det: detector index
        fit_key: type of fit. 'fit_coadd', 'fit_amp', 'fit_phase__fix_tau', or 'fit_phase__free'
    """
    field = aman.stm_cal[fit_key]

    for key in result.params.keys():
        field[key][i_det] = result.params[key].value
        field[f"{key}_stderr"][i_det] = result.params[key].stderr
        field["chisqr"][i_det] = result.chisqr
        field["redchi"][i_det] = result.redchi

        if fit_key == "fit_coadd":
            field["weight"][i_det] = result.weights
        else:
            field["data"][i_det] = result.data


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


def get_dicts(aman, cal_type):
    """
    Get dictionaries for co-added data and fit result.

    Args:
        aman: axis manager
        cal_type: 'gain' or 'timeconstant'. type of calibration

    Return:
        fit_result: Dictionary for fit result
    """
    filt_keys = ["iirc", "hpf", "lpf"]

    fit_result = {}

    if cal_type == "gain":
        freq_keys = ["f1_gain"]
        fit_keys = ["fit_coadd"]
    elif cal_type == "timeconstant":
        freq_keys = CHOPPING_FREQS.keys()
        fit_keys = ["fit_coadd", "fit_amp", "fit_phase__fix_tau", "fit_phase__free"]
    else:
        raise ValueError(
            f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'."
        )

    # Initialize fit_result dictionary
    for fit_key in fit_keys:
        fit_result[fit_key] = {}
        for filt_key in filt_keys:
            if filt_key != "iirc":
                if fit_key == "fit_coadd":
                    fit_result[fit_key][filt_key] = {}
                    for freq_key in freq_keys:
                        fit_result[fit_key][filt_key][freq_key] = np.full(
                            aman.dets.count, np.nan, dtype=object
                        )
                else:
                    fit_result[fit_key][filt_key] = np.full(
                        aman.dets.count, np.nan, dtype=object
                    )

    return fit_result


def fill_data(aman, fit_result, cal_type):
    """
    Fill co-added data and fit result to axis manager.

    Args:
        aman: axis manager
        fit_result: fit result
        n_bins: number of bins
        cal_type: 'gain' or 'timeconstant'. type of calibration
    """

    # Fill fit result to axis manager
    for fit_key in fit_result.keys():
        if fit_key not in aman.stm_cal.keys():
            aman.stm_cal.wrap(fit_key, core.AxisManager(aman.dets))

        for filt_key in fit_result[fit_key].keys():
            if fit_key == "fit_coadd":
                if filt_key not in aman.stm_cal[fit_key].keys():
                    aman.stm_cal[fit_key].wrap(filt_key, core.AxisManager(aman.dets))
                for freq_key in fit_result[fit_key][filt_key].keys():
                    aman.stm_cal[fit_key][filt_key].wrap(
                        freq_key,
                        core.AxisManager(
                            aman.dets, aman.stm_cal.coadd_data.stm_coadd_bins
                        ),
                        overwrite=True,
                    )
            else:
                aman.stm_cal[fit_key].wrap(
                    filt_key,
                    core.AxisManager(
                        aman.dets,
                        core.IndexAxis(
                            "ndata_taufit",
                            aman.stm_cal.filtering_params.chopping_freqs_tau.size,
                        ),
                    ),
                    overwrite=True,
                )

        for filt_key in fit_result[fit_key].keys():

            def fill_field(field, result, weight_axis):
                valid_result = [x for x in result if not isinstance(x, (int, float))]
                if len(valid_result) == 0:
                    return

                for key in valid_result[0].params.keys():
                    arr = np.array(
                        [
                            float(x.params[key].value)
                            if not isinstance(x, (int, float))
                            else np.nan
                            for x in result
                        ]
                    )
                    field.wrap(key, arr, [(0, "dets")], overwrite=True)
                    arr = np.array(
                        [
                            float(x.params[key].stderr)
                            if not isinstance(x, (int, float))
                            and x.params[key].stderr is not None
                            else np.nan
                            for x in result
                        ]
                    )
                    field.wrap(f"{key}_stderr", arr, [(0, "dets")], overwrite=True)
                    arr = np.array(
                        [
                            float(x.chisqr)
                            if not isinstance(x, (int, float))
                            else np.nan
                            for x in result
                        ]
                    )
                    field.wrap("chisqr", arr, [(0, "dets")], overwrite=True)
                    arr = np.array(
                        [
                            float(x.redchi)
                            if not isinstance(x, (int, float))
                            else np.nan
                            for x in result
                        ]
                    )
                    field.wrap("redchi", arr, [(0, "dets")], overwrite=True)
                    if (
                        weight_axis == "stm_coadd_bins"
                    ):  # Because we don't use weight for time-constant fit for the first step analysis
                        arr = np.array(
                            [
                                x.weights
                                if not isinstance(x, (int, float))
                                else np.full(field._axes[weight_axis].count, np.nan)
                                for x in result
                            ]
                        )
                        arr = arr.astype(float)
                        field.wrap(
                            "weights",
                            arr,
                            [(0, "dets"), (1, weight_axis)],
                            overwrite=True,
                        )
                if fit_key != "fit_coadd":
                    data_len = len(valid_result[0].data)
                    arr = np.array(
                        [
                            x.data
                            if not isinstance(x, (int, float))
                            else np.full(data_len, np.nan)
                            for x in result
                        ]
                    )
                    field.wrap("data", arr, [(0, "dets")], overwrite=True)
                    arr = np.array(
                        [
                            x.userkws["f"]
                            if not isinstance(x, (int, float))
                            else np.full(data_len, np.nan)
                            for x in result
                        ]
                    )
                    field.wrap("f", arr, [(0, "dets")], overwrite=True)

            if fit_key == "fit_coadd":
                for freq_key in fit_result[fit_key][filt_key].keys():
                    field = aman.stm_cal.fit_coadd[filt_key][freq_key]
                    result = fit_result[fit_key][filt_key][freq_key]
                    weight_axis = "stm_coadd_bins"

                    fill_field(field, result, weight_axis)
            else:
                field = aman.stm_cal[fit_key][filt_key]
                result = fit_result[fit_key][filt_key]
                weight_axis = "ndata_taufit"
                fill_field(field, result, weight_axis)

    # Fill result for general people
    if cal_type == "gain":
        heater_temp = aman.stm_cal.temps[aman.stm_cal.positions.vals == "heater"][0][0]
        env_temp = aman.stm_cal.temps[aman.stm_cal.positions.vals == "env"][0][0]

        arr = np.array(
            [
                float(x.best_values["a0"])
                if not isinstance(x, (int, float))
                else np.nan
                for x in fit_result["fit_coadd"]["lpf"]["f1_gain"]
            ]
        )
        arr = abs(arr) * (STM_NORMALIZE_TEMP / (heater_temp - env_temp))
        aman.stm_cal.wrap("stm_gain", arr, [(0, "dets")], overwrite=True)
    if cal_type == "timeconstant":
        arr = np.array(
            [
                float(x.best_values["tau"])
                if not isinstance(x, (int, float))
                else np.nan
                for x in fit_result["fit_amp"]["lpf"]
            ]
        )
        aman.stm_cal.wrap("stm_tau", arr, [(0, "dets")], overwrite=True)
        arr = np.array(
            [
                float(x.best_values["dt"])
                if not isinstance(x, (int, float))
                else np.nan
                for x in fit_result["fit_phase__fix_tau"]["lpf"]
            ]
        )
        aman.stm_cal.wrap("readout_delay", arr, [(0, "dets")], overwrite=True)

"""Functions used for demodulating general sinusoidal modulation in
timestreams. Often used in lab testing or with optical choppers."""

import numpy as np
from sotodlib.tod_ops import filters
from scipy.optimize import curve_fit

import logging
logger = logging.getLogger(__name__)


def _remove_demod_placeholders(aman):
    '''Remove some temporary data from aman that was used for demodulation.'''
    if 'highband_sine' in aman:
        aman.move('highband_sine', None)
    if 'highband_cos' in aman:
        aman.move('highband_cos', None)
    if 'mod_sine' in aman:
        aman.move('mod_sine', None)
    if 'mod_cos' in aman:
        aman.move('mod_cos', None)
    return


def demod_sine(aman, freq=8.0, lp_fc=1, signal=None,
               demod_name='demod_signal'):
    '''Demodulate a sinusoidal function from AxisManager.signal_name and then
    place it into AxisManager.demod_name, using a sqrt(sin + cos) wave.

    args:

        aman                 - axis manager to use
        freq                 - demodulation frequency (Hz)
        lp_fc                - low pass filter cutoff frequency to use (Hz)
        signal               - name of axis manager signal to demodulate.
                               If not specified this uses 'aman.signal'
        demod_name           - name of resulting demod signal to put in aman
    '''
    if signal is None:
        signal = aman.signal
    try:
        sinewave = np.sin(2*np.pi*freq*(aman.timestamps-aman.timestamps[0]))
        coswave = np.cos(2*np.pi*freq*(aman.timestamps-aman.timestamps[0]))

        aman.wrap('highband_sine', signal * sinewave,
                  [(0, 'dets'), (1, 'samps')])
        aman.wrap('highband_cos', signal * coswave,
                  [(0, 'dets'), (1, 'samps')])

        aman.wrap('mod_sine', sinewave**2, [(0, 'samps')])
        aman.wrap('mod_cos', coswave**2, [(0, 'samps')])

        div_mod_sin = filters.fourier_filter(aman,  filters.low_pass_butter4(
            fc=lp_fc), detrend=None, signal_name='mod_sine')
        div_mod_cos = filters.fourier_filter(aman,  filters.low_pass_butter4(
            fc=lp_fc), detrend=None, signal_name='mod_cos')

        if demod_name in aman:
            aman.move(demod_name, None)

        demod_sin_signal = filters.fourier_filter(
            aman,  filters.low_pass_butter4(fc=lp_fc), detrend=None,
            signal_name='highband_sine') / div_mod_sin
        demod_cos_signal = filters.fourier_filter(
            aman,  filters.low_pass_butter4(fc=lp_fc), detrend=None,
            signal_name='highband_cos') / div_mod_cos

        demod_signal = np.sqrt(demod_cos_signal**2 + demod_sin_signal**2)

        aman.wrap(demod_name, demod_signal, [(0, 'dets'), (1, 'samps')])
        _remove_demod_placeholders(aman)
    except:
        _remove_demod_placeholders(aman)
        raise
    return


def sinewave(aman, freq=8.0, phase=0):
    return np.sin(
        2 * np.pi * freq * (aman.timestamps-aman.timestamps[0]) + phase)


def demod_single_sine(aman, phase, freq=8.0, lp_fc=1, signal=None,
                      demod_name='demod_signal'):
    '''Demodulate a sinusoidal function from AxisManager.signal_name and then
    place it into AxisManager.demod_name. This only uses a single sine + phase
    in the demodulation with the phase preferably fitted to the data
    beforehand.

    args:

        aman                 - axis manager to use
        phase                - starting phase of demod sinewave to use
                               (this should be fitted or determined beforehand)
        freq                 - demodulation frequency (Hz)
        lp_fc                - low pass filter cutoff frequency to use (Hz)
        signal               - name of axis manager signal to demodulate.
                               If not specified this uses 'aman.signal'
        demod_name           - name of resulting demod signal to put in aman
    '''
    if signal is None:
        signal = aman.signal
    try:
        demod_sinewave = sinewave(aman, phase=phase)
        aman.wrap('highband_sine', signal * demod_sinewave,
                  [(0, 'dets'), (1, 'samps')])

        aman.wrap('mod_sine', demod_sinewave**2, [(0, 'samps')])

        div_mod_sin = filters.fourier_filter(
            aman,  filters.low_pass_butter4(fc=lp_fc), detrend=None,
            signal_name='mod_sine')

        if 'demod_signal' in aman:
            aman.move('demod_signal', None)

        demod_sin_signal = filters.fourier_filter(
            aman,  filters.low_pass_butter4(fc=lp_fc), detrend=None,
            signal_name='highband_sine')/div_mod_sin

        demod_signal = demod_sin_signal

        aman.wrap(demod_name, demod_signal, [(0, 'dets'), (1, 'samps')])
        _remove_demod_placeholders(aman)
    except:
        _remove_demod_placeholders(aman)
        raise
    return


def get_phase_fit_signals(aman, middle_relative_time, index_limit=100,
                          threshold=10., plot=False):
    '''Get the signals we want to fit a phase to. These should have strong power
       in the region we're interested in.

    args:

        aman                 - axis manager to use
        middle_relative_time - (relative) time since the beginning of the
                               timestream that we want to look around
        index_limit          - +/- limits of indices around
                               middle_relative_time that we look
        threshold            - power threshold that we cut off when checking
        plot                 - show plots of data

    returns a list of signals as well as the times they correspond to.

    '''

    phase_fit_signals = []
    if (plot):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in aman.flags.has_cuts(['trends']):
            continue
        relative_time = aman.timestamps - aman.timestamps[0]
        middle_time = np.where(relative_time < middle_relative_time)[0][-1]

        start, stop = (middle_time - index_limit, middle_time + index_limit)

        # We only take specific signals with power below the threshold
        # to avoid weirdly saturated detectors
        if np.max(np.abs(aman.signal[i][start: stop])) < threshold:
            if (plot):
                plt.plot(relative_time[start: stop],
                         aman.signal[i][start: stop])

            phase_fit_signals.append(aman.signal[i][start: stop])
    if (plot):
        plt.grid()
        plt.show()

    return phase_fit_signals, aman.timestamps[start: stop]


def _fit_sin(tt, yy, freq=8, plot=False):
    '''Fit sin to the input time sequence, and return fitting parameters
    "amplitude", "omega", "phase", "offset", "freq"'''
    # Create sine func to fit.
    omega = 2 * np.pi * freq

    def sinfunc(t, A, phi, c):
        return np.abs(A) * np.sin(omega * t + phi) + c

    tt = np.array(tt)
    yy = np.array(yy)
    # guess the amplitude and offset of this sine wave
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 0., guess_offset])

    # fit to the data.
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess, maxfev=10000)

    A, phi, c = popt

    if (plot):
        import matplotlib.pyplot as plt
        plt.plot(tt, yy, label='data')
        plt.plot(tt, sinfunc(tt, *popt), label='fit, phase = %.2f' % phi)
        plt.legend()
        plt.show()

    return {"amplitude": A, "phase": phi, "offset": c, "freq": freq}


def fit_phase(aman, middle_relative_time, index_limit=180, threshold=.8,
              freq=8.0, plot=False):
    '''Fit a sinewave to modulated data, only useful in regions where the
    modulated signal has sufficient signal-to-noise.

    args:

        aman                 - axis manager to use
        middle_relative_time - (relative) time since the beginning of the
                               timestream that we want to fit around
        index_limit          - +/- limits of indices around
                               middle_relative_time that we look
        threshold            - power threshold that we cut off when checking
                               various detectors (if we should use their data)
        freq                 - chopper frequency (Hz)
        plot                 - show plots of data

    returns a phase value to use with demod_single_sine

    '''
    phase_fit_signals, times = get_phase_fit_signals(
        aman, middle_relative_time, threshold=threshold,
        index_limit=index_limit, plot=plot)

    fit_attrs = [_fit_sin(times - times[0], signal, freq=freq,
                          plot=False) for signal in phase_fit_signals]
    phases = np.array([attrs['phase'] for attrs in fit_attrs])
    # convert this phase to a global phase
    global_phases = phases - (2 * np.pi * freq) * (
        times[0] - aman.timestamps[0])
    # convert to usual range
    global_phases = np.unwrap(np.sort(np.mod(global_phases, 2 * np.pi)))
    mean, median, std = (np.mean(global_phases), np.median(global_phases),
                         np.std(global_phases))
    logger.debug('mean, median, std of fitted phases: %.2f, %.2f, %.2f' % (
        mean, median, std))
    phase_to_use = median

    if (plot):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        amplitudes = []
        for i in range(0, aman.dets.count):
            if aman.dets.vals[i] in aman.flags.has_cuts(['trends']):
                continue
            relative_time = aman.timestamps - aman.timestamps[0]
            middle_time = np.where(relative_time < middle_relative_time)[0][-1]

            start, stop = (middle_time - index_limit // 2,
                           middle_time + index_limit // 2)

            amplitude = np.max(np.abs(aman.signal[i][start: stop]))
            amplitudes.append(amplitude)

            if np.max(np.abs(aman.signal[i][start: stop])) < threshold:
                plt.plot(relative_time[start: stop],
                         aman.signal[i][start: stop])
        # Plot a fitted sinewave now
        plt.plot(relative_time[start: stop], np.mean(amplitudes) * sinewave(
            aman, phase=phase_to_use)[start: stop], label='fitted sine',
            linewidth=5, color='black', alpha=.7)

        plt.legend()
        plt.grid()
        plt.show()

    return phase_to_use, global_phases

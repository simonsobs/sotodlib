# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import sys

import numpy as np
from scipy.signal import firwin, fftconvolve

import toast


class Lowpass:
    """ A callable class that applies the low pass filter """
    
    def __init__(self, wkernel, fmax, fsample, offset, nskip, window="hamming"):
        """ Arguments:
        wkernel(int) : width of the filter kernel
        fmax(float) : maximum frequency of the filter
        fsample(float) : signal sampling frequency
        offset(int) : signal index offset for downsampling
        nskip(int) : downsampling factor
        """
        self.lpf = firwin(wkernel, fmax, window=window, pass_zero=True, fs=fsample)
        self._offset = offset
        self._nskip = nskip

    def __call__(self, signal):
        lowpassed = fftconvolve(signal, self.lpf, mode="same").real
        downsampled = lowpassed[self._offset % self._nskip :: self._nskip]
        return downsampled


class OpDemod(toast.Operator):
    """ Demodulate and downsample HWP-modulated data

    """

    def __init__(
        self,
        name="signal",
        wkernel=None,
        fmax=None,
        nskip=3,
        flagmask=1,
        intervals="intervals",
        window="hamming",
        purge=True,
    ):
        """ Arguments:
        name(str) : Cache prefix to operate on
        fmax(float) : max frequency of low-pass filter
        wkernel(int) : kernel size of filter
        flagmask(uint8) : flag bits to raise for invalid samples
        intervals(string) : Name of the intervals in the observation dictionary
        window(string) : Window function name recognized by scipy.signal.firwin
        purge(bool) : Remove inputs after demodulation
        """
        self._name = name
        self._wkernel = wkernel
        self._fmax = fmax
        self._nskip = nskip
        self._flagmask = flagmask
        self._intervals = intervals
        self._window = window
        self._purge = purge

    def _get_fmax(self, tod):
        times = tod.local_times()
        hwp_angle = np.unwrap(tod.local_hwp_angle())
        hwp_rate = np.mean(np.diff(hwp_angle) / np.diff(times)) / (2 * np.pi)
        if self._fmax is not None:
            fmax = self._fmax
        else:
            # set low-pass filter cut-off frequency as same as HWP 1f
            fmax = hwp_rate
        return fmax, hwp_rate

    def _get_wkernel(self, tod, fmax, fsample):
        if self._wkernel is not None:
            wkernel = self._wkernel
        else:
            # set kernel size longer than low-pass filter time scale
            wkernel = (1 << int(np.ceil(np.log(fsample / fmax * 10) / np.log(2)))) - 1
        return wkernel

    def _demodulate_times(self, tod, offset):
        """ Downsample timestamps and adjust sampling rate """
        times = tod.local_times()
        times = times[offset % self._nskip :: self._nskip]
        tod.cache.put(tod.TIMESTAMP_NAME, times, replace=True)
        tod._rate /= self._nskip
        return

    def _demodulate_offsets(self, obs, tod):
        dist_samples = tod._dist_samples
        dist_samples = np.array(dist_samples) // self._nskip
        tod._dist_samples = list(dist_samples)
        offset, nsample = tod.local_samples
        times = tod.local_times()
        for ival in obs[self._intervals]:
            ival.first //= self._nskip
            ival.last //= self._nskip
            local_first = ival.first - offset
            if local_first >= 0 and local_first < nsample:
                ival.start = times[local_first]
            local_last = ival.last - offset
            if local_last >= 0 and local_last < nsample:
                ival.stop = times[local_last]
        return

    def _demodulate_flag(self, flags, wkernel, offset):
        """ Collapse flags inside the filter window and downsample """
        """
        # FIXME: this is horribly inefficient but optimization may require
        # FIXME: a compiled kernel
        n = flags.size
        new_flags = []
        width = wkernel // 2 + 1
        for i in range(0, n, self._nskip):
            ind = slice(max(0, i - width), min(n, i + width + 1))
            buf = flags[ind]
            flag = buf[0]
            for flag2 in buf[1:]:
                flag |= flag2
            new_flags.append(flag)
        new_flags = np.array(new_flags)
        """
        # FIXME: for now, just downsample the flags.  Real data will require
        # FIXME:    measuring the total flag within the filter window
        new_flags = flags[offset % self._nskip::self._nskip]
        return new_flags

    def _demodulate_common_flags(self, tod, wkernel, offset):
        """ Combine and downsample flags in the filter window """
        common_flags = tod.local_common_flags()
        new_flags = self._demodulate_flag(common_flags, wkernel, offset)
        tod.cache.put(tod.COMMON_FLAG_NAME, new_flags, replace=True)
        return

    def _demodulate_signal(self, tod, det, lowpass):
        """ demodulate signal TOD """
        signal = tod.local_signal(det, self._name)
        weights = tod.cache.reference("weights_{}".format(det))
        # iweights = 1
        # qweights = eta * cos(2 * psi_det + 4 * psi_hwp)
        # uweights = eta * sin(2 * psi_det + 4 * psi_hwp)
        iweights, qweights, uweights = weights.T
        etainv = 1 / np.sqrt(qweights ** 2 + uweights ** 2)
        signal_demod0 = lowpass(signal)
        signal_demod4r = lowpass(signal * 2 * qweights * etainv)
        signal_demod4i = lowpass(signal * 2 * uweights * etainv)
        signal_name_0 = "{}_demod0_{}".format(self._name, det)
        signal_name_4r = "{}_demod4r_{}".format(self._name, det)
        signal_name_4i = "{}_demod4i_{}".format(self._name, det)
        tod.cache.put(signal_name_0, signal_demod0, replace=True)
        tod.cache.put(signal_name_4r, signal_demod4r, replace=True)
        tod.cache.put(signal_name_4i, signal_demod4i, replace=True)
        if self._purge:
            if self._name is None:
                tod.cache.destroy("{}_{}".format(tod.SIGNAL_NAME, det))
            else:
                tod.cache.destroy("{}_{}".format(self._name, det))
        return

    def _demodulate_flags(self, tod, det, wkernel, offset):
        """ Demodulate and downsample flags """
        flags = tod.local_flags(det)
        # flag invalid samples in both ends
        flags[: wkernel // 2] |= self._flagmask
        flags[-(wkernel // 2) :] |= self._flagmask

        # Downsample and copy flags
        new_flags = self._demodulate_flag(flags, wkernel, offset)
        for demodkey in ["demod0", "demod4r", "demod4i"]:
            demod_name = tod.FLAG_NAME + "_{}_{}".format(demodkey, det)
            tod.cache.put(demod_name, new_flags, replace=True)
        if self._purge:
            tod.cache.destroy("{}_{}".format(tod.FLAG_NAME, det))
        return

    def _demodulate_pointing(self, tod, det, lowpass, offset):
        """ demodulate pointing matrix """
        weights = tod.cache.reference("weights_{}".format(det))
        iweights, qweights, uweights = weights.T
        # We lowpass even constant-valued vectors to match the
        # normalization and downsampling
        iweights = lowpass(iweights)
        eta = np.sqrt(qweights ** 2 + uweights ** 2)
        eta = lowpass(eta)
        zeros = np.zeros_like(iweights)

        weights_demod0 = np.column_stack([iweights, zeros, zeros])
        weights_name = "weights_demod0_{}".format(det)
        tod.cache.put(weights_name, weights_demod0, replace=True)

        weights_demod4r = np.column_stack([zeros, eta, zeros])
        weights_name_4r = "weights_demod4r_{}".format(det)
        tod.cache.put(weights_name_4r, weights_demod4r, replace=True)

        weights_demod4i = np.column_stack([zeros, zeros, eta])
        weights_name_4i = "weights_demod4i_{}".format(det)
        tod.cache.put(weights_name_4i, weights_demod4i, replace=True)

        # Downsample and copy pixel numbers
        local_pixels = tod.cache.reference("pixels_{}".format(det))
        pixels = local_pixels[offset % self._nskip :: self._nskip]
        for demodkey in ["demod0", "demod4r", "demod4i"]:
            demod_name = "pixels_{}_{}".format(demodkey, det)
            tod.cache.put(demod_name, pixels, replace=True)

        if self._purge:
            tod.cache.destroy("{}_{}".format("weights", det))
            tod.cache.destroy("{}_{}".format("pixels", det))
        return

    def _demodulate_noise(self, noise, det, fsample, hwp_rate, lowpass):
        """ Add Noise objects for the new detectors """
        lpf = lowpass.lpf
        lpf_freq = np.fft.rfftfreq(lpf.size, 1/fsample)
        lpf_value = np.abs(np.fft.rfft(lpf)) ** 2
        # weight -- ignored
        # index  - ignored
        # rate
        rate_in = noise.rate(det)
        # freq
        freq_in = noise.freq(det)
        # psd
        psd_in = noise.psd(det)
        for demodkey in ["demod0", "demod4r", "demod4i"]:
            demod_name = "{}_{}".format(demodkey, det)
            # Lowpass
            if demodkey == "demod0":
                # lowpass psd
                psd_out = psd_in * np.interp(freq_in, lpf_freq, lpf_value)
            else:
                # get noise at 4f
                psd_out = np.zeros_like(psd_in)
                psd_out[:] = np.interp(4 * hwp_rate, freq_in, psd_in)
            # Downsample
            rate_out = rate_in / self._nskip
            ind = freq_in <= rate_out / 2
            freq_out = freq_in[ind]
            psd_out = psd_out[ind] / self._nskip
            # Insert
            noise._rates[demod_name] = rate_out
            noise._freqs[demod_name] = freq_out
            noise._psds[demod_name] = psd_out
        if self._purge:
            del noise._rates[det]
            del noise._freqs[det]
            del noise._psds[det]
        return

    def exec(self, data):
        for obs in data.obs:
            tod = obs["tod"]
            try:
                hwp_angle = tod.local_hwp_angle()
            except:
                continue
            if np.std(hwp_angle) == 0:
                continue
            offset, nsample = tod.local_samples
            noise = obs["noise"]
            fsample = tod._rate
            fmax, hwp_rate = self._get_fmax(tod)
            wkernel = self._get_wkernel(tod, fmax, fsample)
            lowpass = Lowpass(wkernel, fmax, fsample, offset, self._nskip, self._window)

            for det in tod.local_dets:
                if det.startswith("demod"):
                    continue

                self._demodulate_flags(tod, det, wkernel, offset)
                self._demodulate_signal(tod, det, lowpass)
                self._demodulate_pointing(tod, det, lowpass, offset)
                self._demodulate_noise(noise, det, fsample, hwp_rate, lowpass)

            self._demodulate_times(tod, offset)
            self._demodulate_common_flags(tod, wkernel, offset)
            self._demodulate_offsets(obs, tod)

            # change detector list
            tod._dist_dets_copy = tod._dist_dets
            demod_dets = [
                [
                    "{}_{}".format(demodkey, det)
                    for det in localdet
                    if not det.startswith("demod")
                    for demodkey in ["demod0", "demod4r", "demod4i"]
                ]
                for localdet in tod._dist_dets
            ]
            tod._dist_dets = demod_dets

        return

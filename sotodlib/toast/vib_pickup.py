import toast
import numpy as np


class OpVibPickup(toast.Operator):
    def __init__(
            self,
            name="signal",
            out=None,
            vib_freq=8,
            vib_amp_mean=0,
            vib_amp_std=0,
    ):
        """
        A function to add vibrational pickup into the detector timestreams.

        Arguments:
        name(str) : The Cache prefix to operate on
        out(str): Cache prefix to put output. If None, will use name
        vib_freq(float): Frequency [Hz] of the vibrational pickup
        vib_amp_mean: Mean amplitude of the vibrationa pickup
        vib_amp_std: Standard devaition of the amplitude distribution
        """
        self._name = name
        if self._out is None:
            self._out = self._name
        else:
            self._out = out
        self._vib_freq = vib_freq
        self._vib_amp_mean = vib_amp_mean
        self._vib_amp_std = vib_amp_std

    def exec(self, data):
        for obs in data.obs:
            tod: toast.tod.TOD = obs["tod"]
            times = tod.local_times()

            for det in tod.local_dets:
                signal = tod.local_signal(det, self._name)
                # Do we need to make the vibration amplitude constant for the
                # entire pipeline, or is it ok it each observation has a
                # different amplitude?
                amp = np.random.normal(self._vib_amp_mean, self._vib_amp_std)
                if amp < 0:
                    amp = 0

                signal = tod.local_signal(det, self._name)
                out = signal + amp * np.sin(times * self._vib_freq)
                tod.cache.put("{}_{}".format(self._out, det), out, replace=True)

        return

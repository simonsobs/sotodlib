# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

import toast


class OpTimeConst(toast.Operator):
    """ Simple time constant filtering without flag checks.

    No checks are made for TOAST data distribution: distributing the TOD
    in time domain will slightly affect the results.
    """

    def __init__(
            self,
            name="signal",
            tau_name="tau",
            out=None,
            inverse=True,
            tau=None,
            tau_sigma=0,
            realization=0,
    ):
        """
        A function to apply or remove a detector time constant to the data

        Arguments:
        name(str) : Cache prefix to operate on
        tau_name(str) : Where to look in the focal plane for time
            contants to apply / remove
        out(str) : Cache prefix to output. If None, use name
        tau(float) : Time constant (in seconds) to use for all detectors.
            If set, will override `tau_name`
        realization(int) :  Realization ID, only used if tau_sigma is nonzero
        """
        self._name = name
        self._taus = tau_name
        self.inverse = inverse
        if out is None:
            self._out = self._name
        else:
            self._out = out
        self._tau = tau
        self._tau_sigma = tau_sigma
        self._realization = realization

    def exec(self, data):
        for obs in data.obs:
            tod = obs["tod"]
            times = tod.local_times()
            if times.size < 2:
                # Cannot filter
                continue

            if self._tau_sigma:
                obsindx = 0
                if "id" in obs:
                    obsindx = obs["id"]
                else:
                    print("Warning: observation ID is not set, using zero!")

            # Get an approximate time step, even if the sampling is irregular
            tstep = (times[-1] - times[0]) / (times.size - 1)
            freqs = np.fft.rfftfreq(times.size, tstep)

            for det in tod.local_dets:
                signal = tod.local_signal(det, self._name)
                if self._tau is None:
                    if self._taus not in obs["focalplane"][det]:
                        raise RuntimeError(
                            "Cannot apply time constant.  No value specified "
                            "nor found in the focalplane database."
                        )
                    tau = obs["focalplane"][det][self._taus]
                else:
                    tau = self._tau
                if self._tau_sigma:
                    # randomize tau in a reproducible manner
                    seed = 1000000 * self._realization
                    seed = 100000 * obsindx
                    seed += obs["focalplane"][det]["index"]
                    seed %= 2 ** 31
                    np.random.seed(seed)
                    tau *= 1 + np.random.randn() * self._tau_sigma
                if self.inverse:
                    taufilter = (1 + 2.0j * np.pi * freqs * tau)
                else:
                    taufilter = 1.0 / (1 + 2.0j * np.pi * freqs * tau)
                # We filter the entire TOD buffer, even if the filter
                # kernel would fit in a shorter buffer.  This implies a log(n)
                # performance penalty.
                out = np.fft.irfft(taufilter * np.fft.rfft(signal), n=times.size)
                tod.cache.put("{}_{}".format(self._out, det), out, replace=True)

        return

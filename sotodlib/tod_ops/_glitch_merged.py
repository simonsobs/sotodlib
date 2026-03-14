"""Low-level summary statistics for glitch classification.

Each statistic is implemented as a subclass of :class:`GlitchStat` and
registered automatically with the :func:`register_stat` decorator.  The
:attr:`~GlitchStat.requires` attribute declares which inputs (e.g.,
``'signal'``, ``'x_pos'``, ``'y_pos'``) the statistic actually needs,
so callers only pass the relevant data.

Adding a new statistic
----------------------
Define a new subclass at module level in this file or in any
downstream module that imports the registry helpers::

    @register_stat
    class StatName(GlitchStat):
        name = "Name of Statistic"
        requires = ("signal", "x_pos")

        def calc(self, *, signal, x_pos):
            # ... your implementation ...
            return stat

The new statistic will then be available in ``get_all_stat_names()`` and
can be referenced by name.
"""

import numpy as np
import scipy.signal
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GlitchStat(ABC):
    """Base class for a single glitch summary statistic.

    Subclasses must set two class-level attributes:

    Attributes
    ----------
    name : str
        Human-readable label used as the key in the stat registry.
    requires : tuple of str
        Inputs that the :meth:`calc` method needs,
        e.g., ``('signal', 'x_pos', 'y_pos')``.
    """

    name: str
    requires: tuple

    @abstractmethod
    def calc(self, **kwargs):
        """Compute and return the statistic (scalar)."""
        ...

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STAT_REGISTRY: dict = {}


def register_stat(cls):
    """Class decorator: instantiate *cls* and add it to the registry."""
    instance = cls()
    _STAT_REGISTRY[instance.name] = instance
    return cls


def get_stat(name):
    """Return the :class:`GlitchStat` instance registered under *name*.

    Usage::

        get_stat("Name of Statistic").calc(signal=...)
        get_stat("Name of Statistic").calc(**data)

    where ``data`` is a dict containing the keys listed in
    :attr:`GlitchStat.requires`.

    Parameters
    ----------
    name : str
        Name of the statistic (must match :attr:`GlitchStat.name`).

    Returns
    -------
    GlitchStat

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    return _STAT_REGISTRY[name]


def get_all_stat_names():
    """Return the names of all registered stats in registration order.

    Returns
    -------
    list of str
    """
    return list(_STAT_REGISTRY.keys())


def get_stat_registry():
    """Return a copy of the full ``{name: GlitchStat}`` registry.

    Returns
    -------
    dict
    """
    return dict(_STAT_REGISTRY)


# ---------------------------------------------------------------------------
# Stat classes
# ---------------------------------------------------------------------------

@register_stat
class NumDets(GlitchStat):
    """Return the number of detectors affected by the glitch."""

    name = "Number of Detectors"
    requires = ("x_pos",)

    def calc(self, *, x_pos):
        return len(x_pos)


@register_stat
class RatioYXExtent(GlitchStat):
    """Ratio of the focal plane extents in y and x.

    Returns ``inf`` when the x range is zero.
    """

    name = "Y and X Extent Ratio"
    requires = ("x_pos", "y_pos")

    def calc(self, *, x_pos, y_pos):
        x_range = np.max(x_pos) - np.min(x_pos)
        if x_range == 0:
            return np.inf
        return (np.max(y_pos) - np.min(y_pos)) / x_range


@register_stat
class MeanCorrelation(GlitchStat):
    """Mean absolute Pearson correlation between detector pairs.

    Uses :func:`numpy.corrcoef` to compute the full correlation matrix.
    The upper triangle (including the diagonal) is averaged.
    """

    name = "Mean abs(Correlation)"
    requires = ("signal",)

    def calc(self, *, signal):
        if signal.shape[0] < 2 or signal.shape[1] < 2:
            return np.nan
        corr = np.corrcoef(signal)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        return np.nanmean(np.abs(corr[mask]))


@register_stat
class MeanTimeLags(GlitchStat):
    """Mean absolute time lag via FFT cross-correlation.

    Detector-pair time lags are estimated as the argmax of the circular FFT
    cross-correlation (in samples), choosing the smaller of the forward or
    wrap-around shift.

    Returns ``nan`` if fewer than 2 detectors or samples.
    """

    name = "Mean abs(Time Lag)"
    requires = ("signal",)

    def calc(self, *, signal):
        n_dets, n_samps = signal.shape
        if n_dets < 2 or n_samps < 2:
            return np.nan

        ffts = np.fft.fft(signal, axis=1)

        # Compute cross-power spectra for upper-triangle pairs only
        idx_i, idx_j = np.triu_indices(n_dets, k=1)
        cross_power = ffts[idx_i] * np.conj(ffts[idx_j])
        cross_corr = np.fft.ifft(cross_power, axis=1).real

        max_indices = np.argmax(cross_corr, axis=1)
        # Take the smaller absolute lag: forward shift or backward wrap-around
        shifts = max_indices - n_samps
        lags = np.where(np.abs(shifts) < max_indices, shifts, max_indices)

        return np.mean(np.abs(lags))


@register_stat
class FracMaxAdjY(GlitchStat):
    """Fraction of detectors in the peak and adjacent y-histogram bins.

    Computes a histogram of *y_pos*, finds the peak bin, sums that bin
    and its immediate neighbours, and divides by the total detector
    count.
    """

    name = "Y Hist Max and Adjacent/Number of Detectors"
    requires = ("y_pos",)

    def calc(self, *, y_pos):
        counts, _ = np.histogram(y_pos)
        peak_idx = int(np.argmax(counts))
        lo = max(peak_idx - 1, 0)
        hi = min(peak_idx + 1, len(counts) - 1)
        return counts[lo:hi + 1].sum() / len(y_pos)


@register_stat
class FracMaxNearY(GlitchStat):
    """Fraction of detectors within 0.1 of the y-histogram peak.

    Computes the histogram and selects all bins whose left edge is
    within 0.1 of the peak bin's left edge.
    """

    name = "Within 0.1 of Y Hist Max/Number of Detectors"
    requires = ("y_pos",)

    def calc(self, *, y_pos):
        counts, edges = np.histogram(y_pos)
        peak_idx = int(np.argmax(counts))
        peak_edge = edges[peak_idx]
        close = np.where(np.abs(edges - peak_edge) <= 0.1)[0]
        # Edge array is one longer than counts; clip to valid bin indices
        bin_indices = close[close < len(counts)]
        return counts[bin_indices].sum() / len(y_pos)


@register_stat
class NumPeaks(GlitchStat):
    """Number of peaks in the combined (max-across-detectors) TOD.

    Smooths the per-sample maximum/mean across detectors with a 3-wide
    boxcar, selects prominent samples, and counts peaks with
    :func:`scipy.signal.find_peaks`.
    """

    name = "Number of Peaks"
    requires = ("signal",)

    def calc(self, *, signal):
        kernel = np.ones(3) / 3
        max_vals = np.convolve(np.max(signal, axis=0), kernel, mode="same")
        mean_vals = np.convolve(np.mean(signal, axis=0), kernel, mode="same")
        std_val = np.std(signal)

        vals_for_peaks = np.where(
            max_vals >= mean_vals + 3 * std_val, max_vals, mean_vals
        )

        prom = max(1e-12, abs(np.mean(vals_for_peaks)) + 2.0 * std_val)
        peaks, _ = scipy.signal.find_peaks(vals_for_peaks, prominence=prom)
        return len(peaks)

"""Machine-learning classification of glitches.

Glitches are grouped into *snippets*: ranges of samples in which several
detectors are flagged simultaneously, together with the detectors flagged
in each.  Summary statistics of each snippet are then used to classify it
with a random forest as one of:

- ``0``: Point Sources
- ``1``: Point Sources + Other
- ``2``: Cosmic Rays
- ``3``: Electronic Glitches

The typical workflow is::

    snippets = get_snippets_and_classify(aman, glitch_flags, "trained_forest.pkl")

which runs :func:`get_snippets` followed by :func:`classify_snippets`.
The output ``snippets`` holds the sample ranges and detectors affected
along with the summary statistics and predictions for each snippet.

To train a new forest, compute the statistics of labelled snippets with
:func:`compute_summary_stats` and pass them to :func:`train_forest`.

New summary statistics can be added with the :func:`register_stat`
decorator::

    @register_stat("Name of Statistic")
    def name_of_statistic(signal, x_pos):
        # ... your implementation ...
        return stat

where the names of the function's parameters determine the inputs it
receives (see :func:`register_stat`).
"""

import inspect
import logging
import pickle
import warnings

import numpy as np
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from so3g.proj import Ranges

from ..core import AxisManager, IndexAxis, LabelAxis
from .detrend import detrend_tod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snippet operations
# ---------------------------------------------------------------------------

def get_snippets_and_classify(aman, glitch_flags, trained_forest_name,
                              n_thres=2, buffer=5, detrend_method="median", n_jobs=1):
    """Find the glitch snippets and classify them with a trained random forest.

    This is a convenience function that runs :func:`get_snippets` and
    :func:`classify_snippets`.

    Parameters
    ----------
    aman : AxisManager
        The TOD containing the signal, detectors, and timesamples.
    glitch_flags : so3g.proj.RangesMatrix
        Glitch flags, of shape ``(dets, samps)`` matching *aman*.
    trained_forest_name : str
        Path to the pickled trained random forest (the ``.pkl`` extension
        is optional).
    n_thres : int, optional
        Minimum number of simultaneously flagged detectors.  Default is 2.
    buffer : int, optional
        Number of samples by which to pad each snippet on either side.
        Default is 5.
    detrend_method : str, optional
        Method used to detrend each snippet before computing its summary
        statistics: ``"linear"``, ``"mean"`` or ``"median"``.  Default is
        ``"median"``.
    n_jobs : int, optional
        Number of threads the forest uses for prediction.  Default is 1.

    Returns
    -------
    snippets : AxisManager
        The snippets with their summary statistics and predictions.
    """
    snippets = get_snippets(aman, glitch_flags, n_thres=n_thres, buffer=buffer)
    classify_snippets(aman, snippets, trained_forest_name,
                      detrend_method=detrend_method, n_jobs=n_jobs)
    return snippets


def get_snippets(aman, glitch_flags, n_thres=2, buffer=5):
    """Find the glitch snippets in a TOD.

    A snippet comprises a range of samples in which at least *n_thres*
    detectors are flagged simultaneously, padded by *buffer* samples on
    either side, together with the detectors flagged anywhere within that
    range.  Snippets that overlap after padding are merged.

    Parameters
    ----------
    aman : AxisManager
        The TOD, used for its ``dets`` and ``samps`` axes.
    glitch_flags : so3g.proj.RangesMatrix
        Glitch flags, of shape ``(dets, samps)`` matching *aman*.
    n_thres : int, optional
        Minimum number of simultaneously flagged detectors.  Default is 2.
    buffer : int, optional
        Number of samples by which to pad each snippet on either side.
        Default is 5.

    Returns
    -------
    snippets : AxisManager
        With the ``dets`` and ``samps`` axes of *aman*, a ``snippets`` axis,
        and the fields:

        - ``snippet_ranges`` (``samps``): Ranges of the snippets' samples;
          ``snippet_ranges.ranges()`` gives one ``(start, stop)`` pair per
          snippet.
        - ``det_mask`` (``snippets``, ``dets``): whether each detector is
          affected in each snippet.

    Raises
    ------
    ValueError
        If the shape of *glitch_flags* does not match *aman*.
    """
    shape = (aman.dets.count, aman.samps.count)
    if tuple(glitch_flags.shape) != shape:
        raise ValueError(f"glitch_flags has shape {tuple(glitch_flags.shape)}, "
                         f"but aman has (dets, samps) = {shape}")

    mask = glitch_flags.mask()
    snippet_ranges = Ranges.from_bitmask(mask.sum(axis=0) >= n_thres).buffer(buffer)
    intervals = snippet_ranges.ranges()
    det_mask = np.zeros((len(intervals), aman.dets.count), dtype=bool)
    for i, (start, stop) in enumerate(intervals):
        det_mask[i] = mask[:, start:stop].any(axis=1)

    snippets = AxisManager(aman.dets, aman.samps, IndexAxis("snippets", len(intervals)))
    snippets.wrap("snippet_ranges", snippet_ranges, [(0, "samps")])
    snippets.wrap("det_mask", det_mask, [(0, "snippets"), (1, "dets")])
    return snippets


def compute_summary_stats(aman, snippets, stats=None, detrend_method="median"):
    """Compute the summary statistics of glitch snippets.

    For each snippet, the statistics receive the following inputs for the
    affected detectors:

    - ``signal``: their signal over the snippet, of shape
      ``(n_dets, n_samps)``, detrended separately for each detector.
    - ``x_pos``, ``y_pos``: their focal-plane positions in degrees,
      rotated by the mean boresight roll over the snippet.

    Detectors without (finite) focal-plane positions are excluded from all
    statistics.  If a snippet does not have any detectors with a valid
    position, all of its statistics will be NaN.

    Parameters
    ----------
    aman : AxisManager
        The TOD that *snippets* was found in.  Must contain ``signal``,
        ``boresight.roll``, ``focal_plane.xi`` and ``focal_plane.eta``.
    snippets : AxisManager
        Glitch snippets from :func:`get_snippets`.
    stats : list of str, optional
        Names of the statistics to compute.  Default is all registered
        statistics, in registration order (see :func:`get_all_stat_names`).
    detrend_method : str, optional
        Detrending method, passed to
        :func:`~sotodlib.tod_ops.detrend.detrend_tod`: ``"linear"``,
        ``"mean"`` or ``"median"``.  Default is ``"median"``.

    Returns
    -------
    stats_array : numpy.ndarray
        Array of shape ``(n_snippets, n_stats)``, with columns ordered as
        *stats*.
    """
    if stats is None:
        stats = get_all_stat_names()
    funcs = [get_stat(name) for name in stats]

    stats_array = np.full((snippets.snippets.count, len(funcs)), np.nan)
    n_no_positions = 0  # counter for snippets with detectors missing focal-plane positions
    for i, ((start, stop), det_mask) in enumerate(
            zip(snippets.snippet_ranges.ranges(), snippets.det_mask)):
        dets = np.flatnonzero(det_mask)

        roll_corr = -np.mean(aman.boresight.roll[start:stop])
        xi, eta = aman.focal_plane.xi[dets], aman.focal_plane.eta[dets]
        x_pos = np.rad2deg(xi * np.cos(roll_corr) - eta * np.sin(roll_corr))
        y_pos = np.rad2deg(eta * np.cos(roll_corr) + xi * np.sin(roll_corr))

        # Detectors without valid positions can't be used in position-based stats
        has_pos = np.isfinite(x_pos) & np.isfinite(y_pos)
        if not np.any(has_pos):
            n_no_positions += 1
            continue

        # Fancy indexing copies the signal, so detrending won't modify aman
        signal = _detrend(aman.signal[dets[has_pos], start:stop], detrend_method)
        inputs = {"signal": signal, "x_pos": x_pos[has_pos], "y_pos": y_pos[has_pos]}
        # Compute stats
        stats_array[i] = [f(**{k: inputs[k] for k in f.requires}) for f in funcs]

    if n_no_positions:
        logger.warning(f"{n_no_positions} of {len(stats_array)} glitch snippets have no "
                       "detectors with focal-plane positions; their stats are NaN.")
    return stats_array


def _detrend(signal, method):
    """Detrend each row of a ``(dets, samps)`` array in place."""
    tod = AxisManager(IndexAxis("dets", signal.shape[0]), IndexAxis("samps", signal.shape[1]))
    tod.wrap("signal", signal, [(0, "dets"), (1, "samps")])
    return detrend_tod(tod, method=method)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: Column names of the ``predictions`` array (see :func:`classify_snippets`).
#: Column 0 is the predicted class (0-3, as listed in the module
#: description); columns 1-4 are the probabilities of classes 0-3.
PRED_COLUMNS = ['Glitch Prediction', 'Probability of being a Point Source',
                'Probability of being a Point Source + Other',
                'Probability of being a Cosmic Ray',
                'Probability of being an Electronic Glitch']


def classify_snippets(aman, snippets, trained_forest, detrend_method="median", n_jobs=1):
    """Classify glitch snippets with a trained random forest.

    Computes the summary statistics that the forest uses (see
    :func:`compute_summary_stats`) and classifies the snippets whose
    statistics are all finite.

    Parameters
    ----------
    aman : AxisManager
        The TOD that *snippets* was found in (see
        :func:`compute_summary_stats`).
    snippets : AxisManager
        Glitch snippets from :func:`get_snippets`; modified in place.
    trained_forest : sklearn.ensemble.RandomForestClassifier or str
        The trained forest, or the path to a pickled one (the ``.pkl``
        extension is optional).  Its ``feature_names_in_`` must be names of
        registered statistics, and its classes must be those defined in
        :data:`PRED_COLUMNS`.
    detrend_method : str, optional
        See :func:`compute_summary_stats`.  Should match the method used to
        compute the forest's training data.  Default is ``"median"``.
    n_jobs : int or None, optional
        Number of threads the forest uses for prediction; this sets
        ``trained_forest.n_jobs``.  If None, the forest's own setting is
        kept.  Default is 1, as for the number of snippets in a typical TOD,
        the overhead of starting threads outweighs the gain.

    Returns
    -------
    snippets : AxisManager
        The input *snippets*, with the added fields:

        - ``stats`` (``snippets``, ``stat_names``): the summary statistics,
          where the ``stat_names`` axis holds the forest's feature names.
        - ``predictions`` (``snippets``, ``pred_cols``): the predicted class
          and the class probabilities, where the ``pred_cols`` axis holds
          :data:`PRED_COLUMNS`.  Snippets with non-finite statistics are not
          classified, and their predictions are NaN.
        - ``detrend_method``: *detrend_method*.

    Raises
    ------
    ValueError
        If the forest's classes do not match those implied by
        :data:`PRED_COLUMNS`.
    """
    if isinstance(trained_forest, str):
        if not trained_forest.endswith(".pkl"):
            trained_forest += ".pkl"
        with open(trained_forest, "rb") as f:
            trained_forest = pickle.load(f)
    if n_jobs is not None:
        trained_forest.n_jobs = n_jobs

    # predict_proba orders its columns by trained_forest.classes_, so these
    # must be the classes that PRED_COLUMNS assumes
    classes = np.arange(len(PRED_COLUMNS) - 1)
    if not np.array_equal(trained_forest.classes_, classes):
        raise ValueError(f"Expected a forest with classes {classes.tolist()}, "
                         f"but got {trained_forest.classes_.tolist()}")

    # Compute the stats in the order the forest expects
    stat_names = list(trained_forest.feature_names_in_)
    stats = compute_summary_stats(aman, snippets, stats=stat_names,
                                  detrend_method=detrend_method)

    # Classify snippets; those with non-finite statistics are returned as NaN
    preds = np.full((len(stats), len(PRED_COLUMNS)), np.nan)
    valid = np.all(np.isfinite(stats), axis=1)
    if np.any(valid):
        with warnings.catch_warnings():
            # The forest was fit with feature names, which a numpy array
            # doesn't have; its columns are already in the right order
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            probs = trained_forest.predict_proba(stats[valid])
        # Predicted class is the one with the highest probability; following line
        # is same as trained_forest.predict(), without computing probs twice
        preds[valid, 0] = trained_forest.classes_[np.argmax(probs, axis=1)]
        preds[valid, 1:] = probs

    snippets.wrap("stats", stats,
                  [(0, "snippets"), (1, LabelAxis("stat_names", stat_names))])
    snippets.wrap("predictions", preds,
                  [(0, "snippets"), (1, LabelAxis("pred_cols", PRED_COLUMNS))])
    snippets.wrap("detrend_method", detrend_method)
    return snippets


def train_forest(X_train, y_train, stats=None, n_trees=50, max_depth=15, n_jobs=None):
    """Train a random forest to classify glitch snippets.

    Parameters
    ----------
    X_train : numpy.ndarray
        Summary statistics of the training snippets, of shape
        ``(n_snippets, n_stats)`` (see :func:`compute_summary_stats`).
    y_train : numpy.ndarray
        Classes of the training snippets, of shape ``(n_snippets,)``.
    stats : list of str, optional
        Names of the statistics in the columns of *X_train*.  Default is
        all registered statistics, in registration order (see
        :func:`get_all_stat_names`), which is also the default column order
        of :func:`compute_summary_stats`.
    n_trees : int, optional
        Number of trees in the forest.  Default is 50.
    max_depth : int, optional
        Maximum depth of each tree.  Default is 15.
    n_jobs : int or None, optional
        Number of threads used for training (-1 for all cores).  This only
        affects the speed of training, not the trained forest.  Default is
        None, i.e., a single thread.

    Returns
    -------
    forest : sklearn.ensemble.RandomForestClassifier
        The trained forest, with ``feature_names_in_`` set to *stats*.

    Raises
    ------
    ValueError
        If the number of columns of *X_train* does not match the number of
        *stats*.
    """
    if stats is None:
        stats = get_all_stat_names()

    if X_train.shape[1] != len(stats):
        raise ValueError(
            f"Number of columns in X_train ({X_train.shape[1]}) does not "
            f"match number of stat names ({len(stats)})")

    forest = RandomForestClassifier(criterion="entropy", n_estimators=n_trees,
                                    max_depth=max_depth, random_state=1, n_jobs=n_jobs)
    forest.fit(X_train, y_train)
    forest.feature_names_in_ = np.array(stats)

    return forest


# ---------------------------------------------------------------------------
# Summary statistics: registry
# ---------------------------------------------------------------------------

#: Inputs available to the summary statistics (see
#: :func:`compute_summary_stats`).
STAT_INPUTS = ("signal", "x_pos", "y_pos")

_STAT_REGISTRY = {}


def register_stat(name):
    """Decorator registering a function as a summary statistic.

    The function receives, as keyword arguments, the inputs named by its
    parameters, which must be among :data:`STAT_INPUTS` (see
    :func:`compute_summary_stats` for their definitions), and must return
    a scalar.

    Parameters
    ----------
    name : str
        Name of the statistic.  Trained forests identify their features by
        these names, so a statistic must keep its name once a forest has
        been trained with it.

    Raises
    ------
    ValueError
        If the function has parameters not in :data:`STAT_INPUTS`.
    """
    def decorator(func):
        requires = tuple(inspect.signature(func).parameters)
        unknown = [p for p in requires if p not in STAT_INPUTS]
        if unknown:
            raise ValueError(f"Stat {name!r} requires unknown inputs {unknown}; "
                             f"available inputs are {STAT_INPUTS}")
        func.requires = requires
        _STAT_REGISTRY[name] = func
        return func
    return decorator


def get_stat(name):
    """Return the function of the summary statistic registered as *name*.

    Raises
    ------
    KeyError
        If no statistic is registered as *name*.
    """
    return _STAT_REGISTRY[name]


def get_all_stat_names():
    """Return the names of all registered statistics, in registration order.

    Returns
    -------
    list of str
    """
    return list(_STAT_REGISTRY)


# ---------------------------------------------------------------------------
# Summary statistics: built-in
# ---------------------------------------------------------------------------

@register_stat("Number of Detectors")
def num_dets(x_pos):
    """Number of affected detectors."""
    return len(x_pos)


@register_stat("Y and X Extent Ratio")
def yx_extent_ratio(x_pos, y_pos):
    """Ratio of the extents of the affected detectors in y and x.

    Returns ``inf`` if the x extent is zero, or ``nan`` if both extents are
    zero (e.g., a single detector, or detectors in the same pixel).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.max(y_pos) - np.min(y_pos)) / (np.max(x_pos) - np.min(x_pos))


@register_stat("Mean abs(Correlation)")
def mean_abs_correlation(signal):
    """Mean absolute Pearson correlation between the detectors' signals.

    Averages the upper triangle of the correlation matrix *including* the
    diagonal (i.e., each detector's correlation of 1 with itself).  NaN
    correlations (e.g., of a constant signal) are ignored.

    Returns ``nan`` if there are fewer than 2 samples.
    """
    if signal.shape[1] < 2:
        return np.nan
    # corrcoef returns a scalar for a single detector
    corr = np.atleast_2d(np.corrcoef(signal))
    return np.nanmean(np.abs(corr[np.triu_indices_from(corr)]))


@register_stat("Mean abs(Time Lag)")
def mean_abs_time_lag(signal):
    """Mean absolute time lag, in samples, between pairs of detectors.

    The lag between two detectors is the shift that maximizes the circular
    cross-correlation of their signals, taking the shorter of the forward
    and backward shifts.

    Returns ``nan`` if there are fewer than 2 detectors or samples.
    """
    n_dets, n_samps = signal.shape
    if n_dets < 2 or n_samps < 2:
        return np.nan

    ffts = np.fft.fft(signal, axis=1)
    # Cross-correlate each detector with all later detectors
    shifts = np.concatenate([
        np.argmax(np.fft.ifft(ffts[i] * np.conj(ffts[i + 1:]), axis=1), axis=1)
        for i in range(n_dets - 1)])
    # A forward shift by s samples is a backward shift by n_samps - s
    return np.mean(np.minimum(shifts, n_samps - shifts))


@register_stat("Y Hist Max and Adjacent/Number of Detectors")
def frac_max_adjacent_y(y_pos):
    """Fraction of detectors in the peak bin of the y histogram and its neighbours.

    Uses the default 10-bin histogram of :func:`numpy.histogram`.
    """
    counts, _ = np.histogram(y_pos)
    peak_idx = np.argmax(counts)
    lo = max(peak_idx - 1, 0)
    hi = min(peak_idx + 1, len(counts) - 1)
    return counts[lo:hi + 1].sum() / len(y_pos)


@register_stat("Within 0.1 of Y Hist Max/Number of Detectors")
def frac_near_max_y(y_pos):
    """Fraction of detectors in the y-histogram bins near the peak bin.

    Counts the detectors in the bins of the default 10-bin histogram of
    :func:`numpy.histogram` whose edges both lie within 0.1 (deg) of the
    lower edge of the peak bin.  Note the result is 0 if the bins are wider
    than 0.1, i.e., if the y extent exceeds 1 deg.
    """
    counts, edges = np.histogram(y_pos)
    peak_edge = edges[np.argmax(counts)]
    near = np.abs(edges - peak_edge) <= 0.1
    # Bin k spans edges[k] to edges[k + 1]
    return counts[near[:-1] & near[1:]].sum() / len(y_pos)


@register_stat("Number of Peaks")
def num_peaks(signal):
    """Number of peaks in the combined signal of the affected detectors.

    The per-sample maximum and mean of the signal across detectors are
    smoothed with a 3-sample boxcar.  The combined signal is the smoothed
    maximum where it exceeds the smoothed mean by at least 3 times the
    standard deviation of the signal, and the smoothed mean elsewhere.
    Peaks are found with :func:`scipy.signal.find_peaks`, with a minimum
    prominence of the absolute mean of the combined signal plus twice the
    standard deviation of the signal.

    As the smoothing kernel is defined in samples, the result depends on
    the sample rate.
    """
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    max_vals = np.convolve(np.max(signal, axis=0), kernel, mode="same")
    mean_vals = np.convolve(np.mean(signal, axis=0), kernel, mode="same")
    std_val = np.std(signal)

    vals_for_peaks = np.where(max_vals >= mean_vals + 3 * std_val, max_vals, mean_vals)

    prom = np.max([1e-12, np.abs(np.mean(vals_for_peaks)) + 2.0 * std_val])
    peaks, _ = scipy.signal.find_peaks(vals_for_peaks, prominence=prom)
    return len(peaks)

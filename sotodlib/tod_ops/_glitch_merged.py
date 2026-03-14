"""ML-based glitch classification pipeline.

This module provides tools for extracting glitch "snippets" from
time-ordered data, computing summary statistics for each snippet, and
classifying them into physical categories (point sources, cosmic rays,
electronic glitches, etc.) using a random-forest classifier.

The typical workflow is:

1. Detect glitches and extract snippets with :func:`get_snippets`.
2. Compute summary statistics with :func:`compute_summary_stats`.
3. Classify the snippets using a trained forest via
   :func:`classify_snippets` (or train your own with
   :func:`train_forest`).

New summary statistics can be added by subclassing :class:`GlitchStat`
and decorating with :func:`register_stat`::

    @register_stat
    class StatName(GlitchStat):
        name = "Name of Statistic"
        requires = ("signal", "x_pos")

        def calc(self, *, signal, x_pos):
            # ... your implementation ...
            return stat
"""

import pickle as pk
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from so3g.proj import Ranges

from ..core import AxisManager
from sotodlib.tod_ops import detrend_tod


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

def ranges_from_n_flagged(n_flagged, n_thres=2, buffer=5):
    """Build a Ranges object from the number of simultaneously flagged detectors.

    Selects samples where the number of flagged detectors meets or exceeds
    *n_thres*, then buffers each range by *buffer* samples on each side.

    Parameters
    ----------
    n_flagged : numpy.ndarray
        1-D array giving the number of detectors flagged at each sample.
    n_thres : int, optional
        Minimum number of simultaneously flagged detectors to form a
        range.  Default is 2.
    buffer : int, optional
        Number of samples to pad on each side of each range.  Default
        is 5.

    Returns
    -------
    so3g.proj.Ranges
        Ranges where at least *n_thres* detectors are flagged.
    """
    return Ranges.from_bitmask(n_flagged >= n_thres).buffer(buffer)


def get_det_mask(ranges_matrix, ranges):
    """
    Given a per-detector ranges matrix and a second set of ranges, return a
    mask of the detectors that overlap (intersect) the ranges in the latter.

    Parameters
    ----------
    ranges_matrix: RangesMatrix
        The ranges matrix containing a Ranges object for each detector.
    ranges: Ranges
        The ranges to be checked for overlap (intersection) with ranges_matrix.

    Returns
    -------
    det_mask: list
        A list of lists containing boolean values indicating whether each
        detector overlaps (intersects) the ranges in ranges_affected.
        Dimensions:
        len(outer list) = len(ranges)
        len(inner list) = n_detectors
    """
    det_mask = [[len(o.ranges()) > 0 for o in (ranges_matrix * Ranges.from_array(ranges.ranges()[i:i+1], ranges.count))]
                for i in range(len(ranges.ranges()))]
    return det_mask


def ranges2slices(r, offset=0):
    """Convert a Ranges object to a list of Python slices.

    Parameters
    ----------
    r : so3g.proj.Ranges
        The ranges to convert.
    offset : int, optional
        An integer offset added to both the start and stop of every
        range.  Default is 0.

    Returns
    -------
    list of slice
        One slice per range.
    """
    slices = [slice(r_[0]+offset, r_[1]+offset) for r_ in r.ranges()]
    return slices


def build_snippet_layouts(aman, slices, dets_affected):
    """
    Build snippet layouts from lists of the affected detectors and slices.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the data.
    slices: list
        The slices of the data that are affected.
    dets_affected: list
        The affected detectors for each slice.

    Returns
    -------
    snippets: list
        A list of AxisManagers, each containing (only) the affected detectors and
        samples for that snippet, e.g.:
        AxisManager(dets:LabelAxis(n_dets_affected), samps:OffsetAxis(n_samps_affected))
    """
    snippets = [AxisManager(
                    aman.dets.restriction(aman.dets.vals[dets])[0],
                    aman.samps.restriction(sl)[0],
                ) for (sl, dets) in zip(slices, dets_affected)]
    return snippets


def extract_snippet(aman, snippet_layout, in_place=False):
    """Restrict an AxisManager to the detectors and samples in a snippet layout.

    Parameters
    ----------
    aman : AxisManager
        The full data axis manager.
    snippet_layout : AxisManager
        An axis manager whose axes define the restriction.
    in_place : bool, optional
        If *True*, modify *aman* in place.  Default is *False*.

    Returns
    -------
    AxisManager
        The restricted axis manager.
    """
    return aman.restrict_axes(snippet_layout._axes, in_place=in_place)


def extract_snippets(aman, snippet_layouts):
    """
    Helper function to run extract_snippet for a list of snippet_layouts.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the data from which to extract the
        snippets.
    snippet_layouts: list
        List of axis managers containing information on the detectors and
        samples affected by each glitch snippet.

    Returns
    -------
    snippets: list
        A list of AxisManagers, each containing the data for the corresponding
        snippet layout.
    """
    return list(map(partial(extract_snippet, aman), snippet_layouts))


def get_snippets(aman, glitch_ranges, det_mask, offset=0):
    """
    Given the ranges of the glitches and masks of the affected detectors,
    return the snippets of data affected by the glitches.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the data.
    glitch_ranges: Ranges
        The ranges of the glitches.
    det_mask: list
        A list of lists of boolean values identifying which detectors are
        affected by the glitches in glitch_ranges.
    offset: int
        The offset to be added to the start of each range. Default is 0.

    Returns
    -------
    snippets: list
        A list of AxisManagers, each containing the data for the corresponding
        glitch snippet.
    """
    # compile slices for each range
    slices = ranges2slices(glitch_ranges, offset=offset)

    # from the det_mask, get the indices of the affected detectors
    det_idxs = [np.where(det_mask[i])[0] for i in range(len(det_mask))]

    # build snippet layouts, each of which is an axis manager containing restricted axes
    snippet_layouts = build_snippet_layouts(aman, slices, det_idxs)

    # extract the snippets from aman
    snippets = extract_snippets(aman, snippet_layouts)

    return snippets


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def get_default_stats():
    """Return the default set of summary statistics for glitch classification.

    Returns all stat names from the stat registry, in registration order.

    Returns
    -------
    list of str
    """
    return get_all_stat_names()


def compute_summary_stats(snippet, stats=None):
    """Compute the summary statistics for glitch classification.

    Parameters
    ----------
    snippet : AxisManager
        Axis manager containing glitch snippets computed with
        :func:`extract_snippets`.
    stats : list of str, optional
        Summary statistics to compute.  Each entry must be a registered
        stat name.  If *None*, all default statistics are used.

    Returns
    -------
    stats_arr : numpy.ndarray
        Array of shape ``(n_stats,)`` with the computed summary statistics.
    """
    if stats is None:
        stats = get_default_stats()

    signal = detrend_tod(snippet, method='median')

    roll_corr = -np.mean(snippet.boresight.roll)  # roll correction
    xi, eta = snippet.focal_plane.xi, snippet.focal_plane.eta
    x_wnans = np.rad2deg(xi * np.cos(roll_corr) - eta * np.sin(roll_corr))
    y_wnans = np.rad2deg(eta * np.cos(roll_corr) + xi * np.sin(roll_corr))

    x_t = x_wnans[~np.isnan(x_wnans)]
    y_t = y_wnans[~np.isnan(y_wnans)]

    data = {'signal': signal, 'x_pos': x_t, 'y_pos': y_t}

    stats_arr = np.array([
        get_stat(s).calc(**{k: data[k] for k in get_stat(s).requires})
        for s in stats
    ])

    return stats_arr


def train_forest(X_train, y_train, stats=None, n_trees=50, max_depth=15):
    """Train a random forest classifier for glitch classification.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training data of shape ``(n_samples, n_features)``.
    y_train : numpy.ndarray
        Training labels of shape ``(n_samples,)``.
    stats : list of str, optional
        Feature names corresponding to the columns of *X_train*.  If
        *None*, the default statistics from :func:`get_default_stats` are
        used.
    n_trees : int, optional
        Number of trees in the forest.  Default is 50.
    max_depth : int, optional
        Maximum depth of each tree.  Default is 15.

    Returns
    -------
    forest : sklearn.ensemble.RandomForestClassifier
        Trained random forest with ``feature_names_in_`` set to *stats*.

    Raises
    ------
    ValueError
        If the number of columns in *X_train* does not match the length
        of *stats*.
    """
    if stats is None:
        stats = get_default_stats()

    if X_train.shape[1] != len(stats):
        raise ValueError(
            f"Number of columns in X_train ({X_train.shape[1]}) does not "
            f"match number of stat names ({len(stats)})")

    forest = RandomForestClassifier(criterion='entropy', n_estimators=n_trees, random_state=1, n_jobs=2, max_depth=max_depth)

    forest.fit(X_train, y_train)
    forest.feature_names_in_ = np.array(stats)

    return forest


#: Column names for the prediction arrays returned by :func:`classify_data_forest`
#: and :func:`classify_snippets`.  Column 0 is the predicted class label (0–3);
#: columns 1–4 are the class probabilities.
PRED_COLUMNS = ['Glitch Prediction', 'Probability of being a Point Source',
                'Probability of being a Point Source + Other',
                'Probability of being a Cosmic Ray',
                'Probability of being an Electronic Glitch']


def classify_data_forest(X_classify, trained_forest):
    """Classify glitches using a trained random forest.

    Parameters
    ----------
    X_classify : numpy.ndarray
        2-D array of shape ``(n_samples, n_features)``.  Columns must be
        ordered to match ``trained_forest.feature_names_in_``.
    trained_forest : sklearn.ensemble.RandomForestClassifier
        Trained random forest.

    Returns
    -------
    preds : numpy.ndarray
        2-D array of shape ``(n_samples, 5)``.  Columns correspond to
        :data:`PRED_COLUMNS`.
    """
    y_pred_forest = trained_forest.predict(X_classify)

    y_pred_forest_probs = trained_forest.predict_proba(X_classify)

    preds = np.column_stack((y_pred_forest, y_pred_forest_probs))

    return preds


def classify_snippets(snippets, trained_forest):
    """
    From their summary statistics, classify the glitch snippets using a trained
    random forest.

    Parameters
    ----------
    snippets: list
        A list of glitch snippets, each of which is an AxisManager object
        containing the glitch data.
    trained_forest: RandomForestClassifier or str
        The trained random forest classifier. If a string is provided, it is
        assumed to be the filename of a pickled RandomForestClassifier object.

    Returns
    -------
    preds : numpy.ndarray
        2-D array of shape ``(n_snippets, 5)`` with predictions and
        probabilities.  Columns correspond to :data:`PRED_COLUMNS`.
        Rows with invalid stats are NaN.
    stats_array: numpy.ndarray
        2D array of shape (n_snippets, n_stats) with the computed statistics
        used for classification.
    col_names: dict
        Dictionary with keys ``'preds'`` and ``'stats'``, giving the column
        names for ``preds`` and ``stats_array``.
    """
    if isinstance(trained_forest, str):
        with open('{}.pkl'.format(trained_forest), 'rb') as f:
            trained_forest = pk.load(f)

    # Compute the summary statistics in the order the forest expects
    stat_names = list(trained_forest.feature_names_in_)
    stats_array = np.array([compute_summary_stats(s, stats=stat_names) for s in snippets])

    # Build a mask of valid rows (no inf or nan)
    valid_mask = np.all(np.isfinite(stats_array), axis=1)

    # Classify only valid rows
    preds = np.full((len(snippets), len(PRED_COLUMNS)), np.nan)
    if np.any(valid_mask):
        preds[valid_mask] = classify_data_forest(stats_array[valid_mask], trained_forest)

    col_names = {'preds': list(PRED_COLUMNS), 'stats': stat_names}

    return preds, stats_array, col_names


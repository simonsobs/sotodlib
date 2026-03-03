import numpy as np
from sotodlib.tod_ops import detrend_tod
import pickle as pk

from sklearn.ensemble import RandomForestClassifier

from sotodlib.tod_ops.glitch_stats import get_stat, get_all_stat_names


def get_default_stats():
    """Return the default set of summary statistics for glitch classification.

    Returns all stat names from the
    :mod:`~sotodlib.tod_ops.glitch_stats` registry, in registration order.

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
        :func:`sotodlib.tod_ops.glitch.extract_snippets`.
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
        :data:`PRED_COLUMNS`:

        0. Glitch Prediction (int label 0--3 for the classes below)
        1. Probability of being a Point Source
        2. Probability of being a Point Source + Other
        3. Probability of being a Cosmic Ray
        4. Probability of being an Electronic Glitch
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
    preds: numpy.ndarray
        2D array of shape (n_snippets, 5) with predictions and probabilities.
        Columns correspond to PRED_COLUMNS. Rows with invalid stats are NaN.
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

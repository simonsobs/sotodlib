import numpy as np
from sotodlib.tod_ops import detrend_tod
import os, pickle as pk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sotodlib.tod_ops import filter_stats_functions as func


stat_dict = {
    'Number of Detectors': func.num_of_det,
    'Y and X Extent Ratio': func.x_and_y_histogram_extent_ratio,
    'Mean abs(Correlation)': func.mean_correlation,
    'Mean abs(Time Lag)': func.mean_time_lags,
    'Y Hist Max and Adjacent/Number of Detectors': func.max_and_adjacent_y_pos_ratio,
    'Within 0.1 of Y Hist Max/Number of Detectors': func.max_and_near_y_pos_ratio,
    'Number of Peaks': func.compute_num_peaks,
}

def get_default_stats():
    """Return the default set of summary statistics for glitch classification.

    Returns
    -------
    list
        A list of the default summary statistics for glitch classification.
    """
    return list(stat_dict.keys())

def compute_summary_stats(snippet, stats=None):
    '''
    Compute all of the summary statistics for glitch classification

    Input: snippet: snippet object/axis manager computed with
    sotodlib.tod_ops.glitch.extract_snippets
    Optional: stats: list of summary statistics to compute
    (default: all in stat_dict)
    Output: stats_arr: numpy.ndarray with all of the summary statistics
    required for glitch classification
    '''
    if stats is None:
        stats = get_default_stats()

    signal = detrend_tod(snippet, method = 'median')

    roll_corr = -np.mean(snippet.boresight.roll)  # roll correction
    xi, eta = snippet.focal_plane.xi, snippet.focal_plane.eta
    x_wnans = np.rad2deg(xi * np.cos(roll_corr) - eta * np.sin(roll_corr))
    y_wnans = np.rad2deg(eta * np.cos(roll_corr) + xi * np.sin(roll_corr))

    x_t, y_t = x_wnans[np.logical_not(np.isnan(x_wnans))], y_wnans[np.logical_not(np.isnan(y_wnans))]

    stats_arr = np.array([stat_dict[stat](signal, x_t, y_t) for stat in stats]).T

    return stats_arr


def split_into_training_and_test(data, labels, train_size=None, random_state=None):

    '''
    Split data and labels into training and test sets.

    Input: data: numpy.ndarray of shape (n_samples, n_features),
    labels: numpy.ndarray of shape (n_samples,),
    Optional: train_size: fraction of the data for training set in decimal form
    random_state: random state for reproducibility
    Output: data_train, labels_train: training data and labels,
    data_test, labels_test: test data and labels
    '''

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, train_size=train_size, random_state=random_state)

    return data_train, labels_train, data_test, labels_test

def train_forest(X_train, y_train, stats=None, n_trees=50, max_depth=15):

    '''
    Train random forest.

    Input: X_train: numpy.ndarray of shape (n_samples, n_features) with training data,
    y_train: numpy.ndarray of shape (n_samples,) with training labels
    Optional: stats: list of feature names corresponding to columns of X_train
    (default = all in stat_dict),
    n_trees: number of trees in forest (default = 50),
    max_depth: maximum number of splits for each tree (default = 15)
    Output: forest: trained random forest (set with feature_names_in_)
    '''
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

    """Classify glitches using a random forest.

    Parameters:
        X_classify (numpy.ndarray):
            2D array of shape (n_samples, n_features). Assumes columns are
            ordered to match trained_forest.feature_names_in_.
        trained_forest (RandomForestClassifier):
            Trained random forest.

    Returns:
        numpy.ndarray:
            2D array of shape (n_samples, 5). Columns correspond to
            PRED_COLUMNS:
            0: Glitch Prediction (int label 0-3 for the classes below)
            1: Probability of being a Point Source
            2: Probability of being a Point Source + Other
            3: Probability of being a Cosmic Ray
            4: Probability of being an Electronic Glitch
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
        Dictionary with keys 'preds' and 'stats', giving the column names
        for ``preds`` and ``stats_array``.
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

def plot_confusion_matrix(pred_labs, true_labs, colours=['purple', 'coral', '#40A0A0', '#FFE660'],
                         save=False, save_file_name='forest_confusion_matrix',
                         outdir=os.getcwd(), show=True):

    '''
    Plot confusion matrix.

    Input: pred_labs: numpy.ndarray of predicted labels,
    true_labs: numpy.ndarray of true labels
    Optional: colours: list colours for plotting, save: True or False for if you want to save the figure,
    save_file_name: file name for plots, outdir: output directory, show: True or False for if you want to show the plot
    Output: No output, plot will show and/or save
    '''

    acc = accuracy_score(true_labs, pred_labs)

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['font.size'] = 18

    leg_labs = ['Point Sources', 'Point Sources + Other', 'Cosmic Rays', 'Other']


    cm = confusion_matrix(true_labs, pred_labs)

    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=leg_labs, yticklabels=leg_labs)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Overall Accuracy: {}%'.format(np.round(acc*100, 1)))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()


    if save:
        plt.savefig('{}/{}.png'.format(outdir, save_file_name))
        plt.savefig('{}/{}.pdf'.format(outdir, save_file_name))

    if show:
        plt.show()
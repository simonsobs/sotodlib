import numpy as np
from sotodlib.tod_ops import detrend_tod
import os, pickle as pk

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    return stat_dict.keys()

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


def split_into_training_and_test(df, train_per):

    '''
    Split pandas dataframe into training and test set.

    Input: df: pandas data frame, train_per: percent of the data for training set in decimal form
    Output: df_train: pandas dataframe of the training data, df_test: pandas dataframe of the test data,
    inds: array of the indicies of the training data in orginal dataframe
    '''

    train_inds = np.random.choice(a = df.shape[0], size = int(df.shape[0]*train_per), replace = False)

    df_train = df.iloc[train_inds]

    df_test = df.drop(df.index[train_inds])

    return df_train, df_test, train_inds

def training_forest(df_train, stats = None, n_trees = 50, max_depth = 15):

    '''
    Train random forest.

    Input: df_train: pandas data frame of training data
    Optional: stats: list of columns of stats to train with (default = all in stat_dict),
    n_trees: number of trees in forest (default = 50),
    max_depth: maximum number of splits for each tree (default = 15)
    Output: forest: trained random forest
    '''
    if stats is None:
        stats = get_default_stats()

    X, Y = df_train[stats], df_train['Train_Lab']

    forest = RandomForestClassifier(criterion='entropy', n_estimators = n_trees, random_state=1, n_jobs=2, max_depth = max_depth)

    forest.fit(X, Y)

    return forest


def classify_data_forest(df_classify, trained_forest):

    """Classify glitches using a random forest.

    Parameters:
        df_classify (pandas.DataFrame):
            DataFrame of data to classify
        trained_forest (RandomForestClassifier):
            Trained random forest

    Returns:
        pandas.DataFrame:
            DataFrame with columns for the predicted labels for the glitches
            and the probabilities of them belonging to each category.
            Labels: int from 0-3 corresponding to
            0: Point Sources
            1: Point Sources + Other
            2: Cosmic Rays
            3: Electronic Glitches
    """

    col_preds = ['Glitch Prediction', 'Probability of being a Point Source',
                'Probability of being a Point Source + Other', 'Probability of being a Cosmic Ray',
                'Probability of being an Electronic Glitch']

    X_classify = df_classify[trained_forest.feature_names_in_]

    y_pred_forest = trained_forest.predict(X_classify)

    y_pred_forest_probs = trained_forest.predict_proba(X_classify)

    preds = pd.DataFrame(np.column_stack((y_pred_forest, y_pred_forest_probs)),
                    columns = col_preds, index = df_classify.index)

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
    df_preds: pandas.DataFrame
        The predictions for the glitches and the probabilities for each category.
    df_stats: pandas.DataFrame
        The computed statistics used for classifying the glitches.
    """
    if isinstance(trained_forest, str):
        trained_forest = pk.load(open('{}.pkl'.format(trained_forest), 'rb'))

    # Compute the summary statistics used in the provided trained_forest
    stats = trained_forest.feature_names_in_
    df_stats = pd.DataFrame([compute_summary_stats(s, stats=stats) for s in snippets], columns = stats)

    # Classify, after removing rows with invalid values
    df_preds = classify_data_forest(df_stats.replace([np.inf, -np.inf], np.nan).dropna(), trained_forest)

    # Insert rows of NaNs for missing indices (assumed to be due to removed rows)
    missing_idxs = df_stats.index.difference(df_preds.index)
    for i in missing_idxs:
        df_preds.loc[i] = np.full(df_preds.shape[1], np.nan)
    df_preds.sort_index(inplace=True)

    return df_preds, df_stats

def plot_confusion_matrix(pred_labs, df, colours = ['purple', 'coral', '#40A0A0', '#FFE660'], \
 save = False, save_file_name = 'forest_confusion_matrix', outdir = os.getcwd(), show = True):

    '''
    Plot confusion matrix.

    Input: pred_labs: predicted labels,
    df: pandas data frame of data ensure Glitch column is present for labels, title: plot title (str)
    Optional: colours: list colours for plotting, save: True or False for if you want to save the figure,
    save_file_name: file name for plots, outdir: output directory, show: True or False for if you want to show the plot
    Output: No output, plot will show and/or save
    '''

    acc = accuracy_score(df['Train_Lab'], pred_labs)

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['font.size'] = 18

    leg_labs = ['Point Sources', 'Point Sources + Other', 'Cosmic Rays', 'Other']


    cm = confusion_matrix(df['Train_Lab'], pred_labs)

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
import so3g
import numpy as np
from sotodlib.tod_ops import detrend_tod
from sotodlib.core import AxisManager
import os, pickle as pk

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sotodlib.tod_ops import filter_stats_functions as func


cols = ['Number of Detectors', 'Y and X Extent Ratio', 'Mean abs(Correlation)',
        'Mean abs(Time Lag)', 'Y Hist Max and Adjacent/Number of Detectors',
        'Within 0.1 of Y Hist Max/Number of Detectors', 'Dip Test for X Hist',
        'P Value for Dip Test for X Hist','Dip Test for Y Hist',
        'P Value for Dip Test for Y Hist', 'KS Test for X',
        'Obs ID', 'Snippet', 'Start timestamp', 'Stop timestamp']


def compute_summary_stats(snippet):

    '''
    Compute all of the summary statistics for glitch classification

    Input: snippet: snippet object/axis manager computed with
    sotodlib.tod_ops.glitch.extract_snippets
    Output: stats: numpy.ndarray with all of the summary statistics
    required for glitch classification
    '''

    data = detrend_tod(snippet, method = 'median')

    tstart, tstop = snippet.timestamps[0], snippet.timestamps[-1]

    x_wnans, y_wnans = snippet.det_info.wafer.x, snippet.det_info.wafer.y

    x_t, y_t = x_wnans[np.logical_not(np.isnan(x_wnans))], y_wnans[np.logical_not(np.isnan(y_wnans))]

    det_num = func.num_of_det(x_t)

    hist_ratio = func.x_and_y_histogram_extent_ratio(x_t, y_t)

    time_lag = func.mean_time_lags(data)

    corr = func.mean_correlation(data)

    near = func.max_and_near_y_pos_ratio(y_t)

    adjacent = func.max_and_adjacent_y_pos_ratio(y_t)

    ks = func.KS_test(x_t)

    # Enter the summary statistics into an array
    # Indices correspond those in `cols`
    stats = np.full(len(cols), np.nan)
    stats[0] = det_num
    stats[1] = hist_ratio
    stats[2] = corr
    stats[3] = time_lag
    stats[4] = adjacent
    stats[5] = near
    stats[10] = ks[0]
    # stats[11]  # obs_id
    # stats[12]  # snippet number
    stats[13] = tstart
    stats[14] = tstop

    if det_num > 3:
        dip_x, pval_x, dip_y, pval_y = func.dip_test(x_t, y_t)

        stats[6] = dip_x
        stats[7] = pval_x
        stats[8] = dip_y
        stats[9] = pval_y

    return stats

def build_dataframe_for_classification(snippets):
    """
    Compute the summary statistics to be used for classification for a list of
    glitch snippets and return them as a pandas DataFrame.

    Parameters
    ----------
    snippets: list
        A list of glitch snippets, each of which is an AxisManager object
        containing the glitch data.

    Returns
    -------
    df: pandas.DataFrame
        A DataFrame containing the summary statistics of each glitch snippet.
    """

    df = pd.DataFrame([compute_summary_stats(s) for s in snippets], columns = cols)

    df['Obs ID'] = snippets[0].obs_info.obs_id
    df['Snippet'] = range(1, len(snippets) + 1)

    return df


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

def training_forest(df_train, cols, n_trees = 50, max_depth = 15):

    '''
    Train random forest.

    Input: df_train: pandas data frame of training data, cols: list of columns of stats to train with
    Optional: n_trees: number of trees in forest (default = 50),
    max_depth: maximum number of splits for each tree (default = 15)
    Output: forest: trained random forest
    '''

    X, Y = df_train[cols], df_train['Train_Lab']

    forest = RandomForestClassifier(criterion='entropy', n_estimators = n_trees, random_state=1, n_jobs=2, max_depth = max_depth)
    
    forest.fit(X, Y)

    return forest


def classify_data_forest(df_classify, trained_forest):

    '''
    Classify glitches using a random forest.

    Input: df_classify: pandas data frame of data to classify,
    trained_forest: trained random forest
    Output: df_w_labs_and_stats: returns the dataframe with a column for the predicted labels - int from
    0 - 3 corresponding to 0: Point Sources, 1: Point Sources + Other 2: Cosmic Rays, 3: Other also columns
    with the probability that the glitch is each of the categories
    '''

    col_predictions = ['Glitch Prediction', 'Probability of being a Point Source', 'Probability of being a Point Source + Other', 'Probability of being a Cosmic Ray', 'Probability of being an Other']

    X_classify = df_classify[trained_forest.feature_names_in_]

    y_pred_forest = trained_forest.predict(X_classify)

    y_pred_forest_probs = trained_forest.predict_proba(X_classify)

    predictions = np.zeros((X_classify.shape[0], 5))

    predictions[:, 0] = y_pred_forest
    predictions[:, 1] = y_pred_forest_probs[:, 0]
    predictions[:, 2] = y_pred_forest_probs[:, 1]
    predictions[:, 3] = y_pred_forest_probs[:, 2]
    predictions[:, 4] = y_pred_forest_probs[:, 3]

    lab_df = pd.DataFrame(predictions, columns = col_predictions)

    df_w_labs_and_stats = pd.concat([df_classify, lab_df], axis = 1)

    return df_w_labs_and_stats

def classify_glitch_stats(stats, trained_forest):
    """
    From their summary statistics, classify the glitch snippets using a trained
    random forest.

    Parameters
    ----------
    stats: pandas.DataFrame or numpy.ndarray
        The summary statistics for the glitches. If numpy.ndarray, the columns
        are assumed to match the feature names of the trained random forest.
    trained_forest: RandomForestClassifier or str
        The trained random forest classifier. If a string is provided, it is
        assumed to be the filename of a pickled RandomForestClassifier object.

    Returns
    -------
    numpy.ndarray
        The predictions for the glitches and the probabilities for each category.
    """
    if isinstance(trained_forest, str):
        trained_forest = pk.load(open('{}.pkl'.format(trained_forest), 'rb'))

    df_stats = pd.DataFrame(stats, columns = trained_forest.feature_names_in_).dropna()

    df_w_predictions = classify_data_forest(df_stats, trained_forest)

    return df_w_predictions.to_numpy()[:,-5:]

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
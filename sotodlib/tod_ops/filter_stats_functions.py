import numpy as np
import scipy.stats as ss
import diptest

def KS_test(x_pos):

    '''
    Compute the Kolmogorov–Smirnov test of the x position hisotgram
    compared to a uniform distribution.

    Input: x_pos: x positions across the focal plane
    Output: KS test statistic, p-value
    ''' 

    
    #compute a uniform distribution with the same x range as the x positions
    dist_uni_guess = np.random.uniform(low = np.min(x_pos), high = np.max(x_pos), size = len(x_pos))

    #calculate CDF values
    dist_cdf = 1. * np.arange(len(x_pos)) / (len(x_pos) - 1)
    
    dist_uni_guess_cdf = 1. * np.arange(len(dist_uni_guess)) / (len(dist_uni_guess) - 1)
    

    ks = ss.kstest(x_pos, dist_uni_guess_cdf)
    
    return ks


def dip_test(x_pos, y_pos):

    '''
    Compute the dip test of the x position and y position 
    hisotgrams.

    Input: x_pos: x positions and y_pos: y positions across the focal plane
    Output: x dip test statistic, p-value for the x dip test statistic,
    y dip test statistic, p-value for the y dip test statistic
    ''' 

    # both the dip statistic and p-value
    dip_x, pval_x = diptest.diptest(x_pos)
    dip_y, pval_y = diptest.diptest(y_pos)

    return dip_x, pval_x, dip_y, pval_y

def num_of_det(x_pos):

    '''
    Compute the number of detectors affected by the glitch

    Input: x_pos: x positions across the focal plane 
    Output: number of detectirs affected
    ''' 

    return len(x_pos)


def x_and_y_histogram_extent_ratio(x_pos, y_pos):

    '''
    Compute the ratio of the extents of the x position and y position 
    hisotgrams.

    Input: x_pos: x positions and y_pos: y positions across the focal plane
    Output: ratio of the extents of the x and y histograms
    ''' 

    hist_ratio = (np.max(y_pos) - np.min(y_pos))/(np.max(x_pos) - np.min(x_pos))

    return hist_ratio

def mean_time_lags(data):

    '''
    Compute the mean of the absolute value of the time lags between detectors.

    Input: data: detector TOD signals (snippets[s].signal
    where s is the given snippet number, this will work better with detrened 
    TODs, e.g. using sotodlib.tod_ops.detrend_tod(snippets[s], method = 'median'))
    Output: mean of the absolute value of the time lags
    ''' 

    lags = np.full((len(data), len(data)), np.nan)


    for i in range(len(data)):
            if len(data[i]) >= 2:
                for j in range(len(data)):
                    if j > i:

                        #compute the time delays between detector pair
                        time_delay_pos = np.fft.ifft(np.fft.fft(data[i])*np.conjugate(np.fft.fft(data[j])))

                        #find the maixmum time delay which correspond to the time shift required to acheive the maximum correlation 
                        max_time_delay_pos = np.max(time_delay_pos)

                        time_delay_ind_t = np.where(time_delay_pos == max_time_delay_pos)[0][0]

                        #determine the difference between the required time shift and the length of the TOD
                        shift_t = time_delay_ind_t - len(data[i])

                        #Take the small between the required time shift and shift defined above. This allows for shifted backwards instead of looping around the TOD.
                        if np.abs(shift_t) < time_delay_ind_t:
                            lag_t = shift_t
                        else:
                            lag_t = time_delay_ind_t

                        lags[i, j] = lag_t

    time_lag = np.abs(np.nanmean(np.abs(lags)))

    return time_lag


def mean_correlation(data):

    '''
    Compute the mean of the absolute value of the Pearson correlation coefficient between detectors.

    Input: data: detector TOD signals (snippets[s].signal
    where s is the given snippet number, this will work better with detrened 
    TODs, e.g. using sotodlib.tod_ops.detrend_tod(snippets[s], method = 'median'))
    Output: mean of the absolute value of the correlations
    ''' 

    corr_coeff = np.full((len(data), len(data)), np.nan)

    for i in range(len(data)):
            if len(data[i]) >= 2:
                for j in range(len(data)):
                    if j >= i:

                        #compute the Pearson correlation coefficient between detector pair
                        corr_t = ss.pearsonr(data[i], data[j])[0]

                        corr_coeff[i, j] = corr_t
                        # corr_coeff[j, i] = corr_t

    mean_corr = np.nanmean(np.abs(corr_coeff))

    return mean_corr


def max_and_near_y_pos_ratio(y_pos):

    '''
    Compute the ratio of the maximum of the y hisotgram bin and positions within 0.1 on either side of the focal plane
    compared to the total number of detectors.

    Input: y_pos: y positions across the focal plane
    Output: sum of maximum and 0.1 to either side of the y hisotgram bins divided by the total number of detectors
    '''

    #determine the peak of the y histogram and its index
    y_max = np.max(np.histogram(y_pos)[0])

    ind_y_max = np.where(np.histogram(y_pos)[0] == y_max)[0][0]

    #find all bins within 0.1 of either side of the maximum bin
    ind_close_to_max = np.where(np.abs(np.histogram(y_pos)[1] - np.histogram(y_pos)[1][ind_y_max]) <= 0.1)[0]

    sum_close = np.sum(np.histogram(y_pos)[0][ind_close_to_max[:-1]])

    det_num = len(y_pos)

    return sum_close/det_num



def max_and_adjacent_y_pos_ratio(y_pos):

    '''
    Compute the ratio of the maximum and adjacent y hisotgram bins compared to the total number of detectors.

    Input: y_pos: y positions across the focal plane 
    Output: sum of maximum and adjacent y hisotgram bins divided by the total number of detectors
    '''

    #determine the peak of the y histogram and its index
    y_max = np.max(np.histogram(y_pos)[0])

    ind_y_max = np.where(np.histogram(y_pos)[0] == y_max)[0][0]

    #check if there are adjacent bins on either side of the maximun bin
    if ind_y_max + 1 <= len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 >= 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max - 1] + np.histogram(y_pos)[0][ind_y_max + 1]

    elif ind_y_max + 1 > len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 >= 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max - 1]

    elif ind_y_max + 1 <= len(np.histogram(y_pos)[0]) - 1 and ind_y_max - 1 < 0:
        sum_near = np.histogram(y_pos)[0][ind_y_max] + np.histogram(y_pos)[0][ind_y_max + 1]   
    
    
    det_num = len(y_pos)

    return sum_near/det_num


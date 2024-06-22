from sotodlib import core
import numpy as np

# Note to future developers with a need for speed: there are two
# obvious places where OpenMP acceleration in a C++ routine would be
# helpful.  np.cov is not threaded, for some reason.  And in our mode
# removal we loop over detectors in python.  In both cases the
# algorithms are simple and parallelize trivially.  Thankfully
# np.linalg.eig is already threaded.


def get_pca_model(tod=None, pca=None, n_modes=None, signal=None,
                  wrap=None):
    """Convert a PCA decomposition into the signal basis, i.e. into
    time-dependent modes that one might use for cleaning or
    calibrating.

    The generalization of "common mode" computation is to convert the
    eigen-decomposition of the covariance matrix into a limited set of
    modes that explain a high fraction of the power in the input
    signals.

    Here we select the strongest eigenmodes from the PCA and compute
    the time-dependent modes and the projection of each detector
    signal onto each mode.  An approximation for the input signal may
    then be computed as

       signal_model = weights . modes

    where weights has shape (dets, eigen), modes has shape (eigen,
    samps), and . indicates matrix multipliaction.  The size of the
    eigen axis can be as large as the number of dets, but is typically
    smaller and in this routine is set by n_modes.

    Arguments:
        tod: AxisManager with dets and samps axes.
        pca: AxisManager holding a PCA decomposition, as returned by
            get_pca.  If not specified, it will be computed from the
            data.
        n_modes: integer specifying the number of modes to compute;
            the strongest modes are taken and this sets the size of
            the "eigen" axis in the output.  Defaults to len(dets),
            but beware that the resulting data object will be the same
            size as the input signal.
        signal: array of shape (dets, samps) that is used to construct
            the requested eigen modes.  If pca is not passed in, this
            signal is also used to compute the covariance for PCA.
        wrap: string; if specified then the returned result is also
            stored in tod under that name.

    Returns:
        An AxisManager with (dets, eigen, samps) axes.  The field
        'weights' has shape (dets, eigen) and the field 'modes' has
        shape (eigen, samps).

    """
    if pca is None:
        pca = get_pca(tod=tod, signal=signal)
    if n_modes is None:
        n_modes = pca.eigen.count

    mode_axis = core.IndexAxis('eigen', n_modes)
    output = core.AxisManager(tod.dets, mode_axis, tod.samps)
    if signal is None:
        signal = tod.signal

    R = pca.R[:, :n_modes]
    output.wrap('weights', R, [(0, 'dets'), (1, 'eigen')])
    output.wrap('modes', np.dot(R.transpose(), signal),
                [(0, 'eigen'), (1, 'samps')])
    if not(wrap is None):
        tod.wrap(wrap, output)
    return output


def get_pca(tod=None, cov=None, signal=None, wrap=None):
    """Compute a PCA decomposition of the kind useful for signal analysis.
    A symmetric non-negative matrix cov of shape(n_dets, n_dets) can
    be decomposed into matrix R (same shape) and vector E (length
    n_dets) such that

        cov = R . diag(E) . R^T

    with . denoting matrix multiplication and T denoting matrix
    transposition.

    Arguments:
        tod: AxisManager with dets and samps axes.
        cov: covariance matrix to decompose; if None then cov is
            computed from tod.signal (or signal).
        signal: array of shape (dets, samps).  If cov is not provided,
            it will be computed from this matrix.  Defaults to
            tod.signal.
        wrap: string; if set then the returned result is also stored
            in tod under this name.

    Returns:
        AxisManager with axes 'dets' and 'eigen' (of the same length),
        containing fields 'R' of shape (dets, eigen) and 'E' of shape
        (eigen).  The eigenmodes are sorted from strongest to weakest.

    """
    if cov is None:
        # Compute it from signal
        if signal is None:
            signal = tod.signal
        cov = np.cov(signal)
    dets = tod.dets

    mode_axis = core.IndexAxis('eigen', dets.count)
    output = core.AxisManager(dets, mode_axis)
    output.wrap('cov', cov, [(0, dets.name), (1, dets.name)])

    # Note eig will sometimes return complex eigenvalues.
    E, R = np.linalg.eig(cov)  # eigh nans sometimes...
    E[np.isnan(E)] = 0.
    E, R = E.real, R.real

    idx = np.argsort(-E)
    output.wrap('E', E[idx], [(0, mode_axis.name)])
    output.wrap('R', R[:, idx], [(0, dets.name), (1, mode_axis.name)])
    if not(wrap is None):
        tod.wrap(wrap, output)
    return output


def add_model(tod, model, scale=1., signal=None, modes=None, weights=None):
    """Adds modeled modes, multiplied by some scale factor, into signal.

    Given a matrix of weights (dets, eigen) and modes (eigen, samps),
    this computes model signal:

        model_signal = weights . modes

    and adds the result to signal, scaled by scale:

        signal += scale * (weights . modes)

    The intended use is for PCA or common mode removal, for which user
    should pass scale = -1.

    Arguments:
        tod: AxisManager with dets and samps axes.
        model: AxisManager with shape (dets, eigen, dets) containing
            'modes' and 'weights' arrays.  This is the type of object
            returned by get_pca_model, for example.
        scale: Factor by which to scale model_signal before
            accumulating into signal.
        signal: array into which to accumulate model_signal; defaults
            to tod.signal.
        modes: array of modes with shape (eigen, samps), defaults to
            model.modes.
        weights: array of weights with shape (dets, eigen), defaults
            to model.weights.

    Returns:
        The signal array into which modes were accumulated.

    Notes:
        If you only want to operate on a subset of detectors, the most
        efficient solution is to pass in weights explicitly, forced to
        0 for the dets you want to omit.  There's an optimization to
        skip the computation for a detector if all the mode coupling
        weights are zero.

    """
    if signal is None:
        signal = tod.signal
    if modes is None:
        modes = model.modes
    if weights is None:
        weights = model.weights
    for i in range(model.dets.count):
        if np.all(weights[i] == 0):
            continue
        signal[i] += np.dot(weights[i, :] * scale, modes)
    return signal


def get_trends(tod, remove=False, size=1, signal=None):
    """Computes trends for each detector signal that remove the slope
    connecting first and last points, as well as the mean of the
    signal.  The returned object can be treated like PCA model (e.g.,
    it can be passed as the model input to add_model).

    Arguments:
        tod: AxisManager with dets and samps axes.
        remove: boolean, if True then the computed trends (and means)
            are removed from the signal.
        size: the number of samples on each end of the signal to use
            for trend level computation.  Defaults to 1.
        signal: array of shape (dets, samps) to compute trends on.
            Defaults to tod.signal.

    Returns:
        An AxisManager with (dets, eigen, samps) axes.  The field
        'weights' has shape (dets, eigen) and the field 'modes' has
        shape (eigen, samps).  There are two modes, which always have
        the same form: index 0 is all ones, and index1 is a smooth
        line from -0.5 to +0.5.

    """
    if signal is None:
        signal = tod.signal
    trends = core.AxisManager(tod.dets, core.IndexAxis('eigen', 2), tod.samps)
    modes = np.ones((trends.eigen.count, trends.samps.count))
    modes[1] = np.linspace(-0.5, 0.5, modes.shape[1])
    weights = np.empty((trends.dets.count, trends.eigen.count))
    weights[:, 0] = signal.mean(axis=1)
    size = max(1, min(size, signal.shape[1] // 2))
    weights[:, 1] = (signal[:, -size:].mean(axis=1) -
                     signal[:, :size].mean(axis=1))
    trends.wrap('modes', modes)
    trends.wrap('weights', weights)
    if remove:
        add_model(tod, trends, scale=-1, signal=signal)
    return trends


def calc_pcabounds(aman, pca_aman, xfac=2, yfac=1.5):
    """Finds the bounds of the pca box using IQR 
    statistics

    Parameters
    ----------
    aman : AxisManager
        observation axismanagers
    pca_aman : AxisManager
        output pca axismanager
    signal : array
        is the low pass filter signal array that's passed 
        through
    xfac : int
        multiplicative factor for the width of the pca box.
        Default is 2.
    yfac : int
        multiplicative factor for the height of the box.
        Default is 1.5. 

    TODO: needs more args used in preprocess config file setup

    Returns
    -------
    aman
        aman that's wrapped with the x and y bounds and the good and bad dets
        
    """
    # TODO: I don't use signal; check preprocess code
    x = aman.det_cal.s_i
    y = np.abs(pca_aman.weights[:, 0])

    # remove positive Si values
    filt = np.where(x < 0)[0]
    xfilt = x[filt]
    yfilt = y[filt]

    # normalize weights
    ynorm = yfilt / np.median(yfilt)
    median_ynorm = np.median(ynorm)
    medianx = np.median(xfilt)

    # IQR of normalized weights
    q20 = np.percentile(ynorm, 20)
    q80 = np.percentile(ynorm, 80)
    iqry_norm = q80 - q20

    # IQR of Si's
    q20x = np.percentile(xfilt, 20)
    q80x = np.percentile(xfilt, 80)
    iqrx = q80x - q20x

    # Find box height using norm'd weights
    ylb_norm = median_ynorm - yfac * iqry_norm
    yub_norm = median_ynorm + yfac * iqry_norm

    # Convert y bounds back to the scale of the raw weights
    ylb = ylb_norm * np.median(yfilt)
    yub = yub_norm * np.median(yfilt)

    # Calculate box width
    mad = np.median(np.abs(xfilt - medianx))
    xlb = medianx - xfac * mad
    xub = medianx + xfac * mad

    xbounds = [xlb, xub]
    ybounds = [ylb, yub]

    # Get indices of the values in the box (indices are wrt `x` array)
    box_xfilt_inds = np.where((xfilt >= xlb) & (
        xfilt <= xub) & (yfilt >= ylb) & (yfilt <= yub))[0]
    box = filt[box_xfilt_inds] 
    notbox = np.setdiff1d(np.arange(len(x)), box)

    goodids = aman.det_info.det_id[box]
    badids = aman.det_info.det_id[notbox]
    
    bands = np.unique(aman.det_info.wafer.bandpass)
    bands = bands[bands != 'NC']
    medianw = np.median(pca_aman.weights[:,0]) # it will just be for one bandpass at a time
    relcal = medianw/pca_aman.weights[:,0]

    mask = np.isin(aman.det_info.det_id, badids)
    relcal = core.AxisManager(aman.dets, aman.samps,
                              core.LabelAxis(name='bandpass', vals=bands))
    relcal.wrap('pca_det_mask', mask, [(0, 'dets')])
    relcal.wrap('xbounds', np.array(xbounds))
    relcal.wrap('ybounds', np.array(ybounds))
    relcal.wrap('pca_mode0', pca_aman.modes[0], [(0, 'samps')])
    relcal.wrap('pca_weight0', pca_aman.weights[:, 0], [(0, 'dets')])
    relcal.wrap('relcal', relcal, [(0, 'dets')])
    relcal.wrap('medians', np.asarray([medianw]), [(0, 'bandpass')])

    # make an Si mask to also wrap which will tell us which Si's correspond to bad dets etc

    return pca_aman


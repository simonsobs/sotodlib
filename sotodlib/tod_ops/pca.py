from sotodlib import core
import numpy as np
import logging

logger = logging.getLogger(__name__)

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


def get_pca(tod=None, cov=None, signal=None, wrap=None, mask=None):
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
        mask: If specifed, a boolean array to select which dets are to
            be considered in the PCA decomp. This is achieved by
            modifying the cov to make the non-considered dets
            independent and low significance.

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

    if mask is not None:
        var_min = min(np.diag(cov)[mask])
        for i in (~mask).nonzero()[0]:
            cov[i,:] = 0
            cov[:,i] = 0
            cov[i,i] = var_min * 1e-2

    dets = tod.dets

    mode_axis = core.IndexAxis('eigen', dets.count)
    output = core.AxisManager(dets, mode_axis)
    output.wrap('cov', cov, [(0, dets.name), (1, dets.name)])

    E, R = np.linalg.eigh(cov)
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


def pca_cuts_and_cal(tod, pca_aman, xfac=2, yfac=1.5, calc_good_medianw=False):
    """Finds the bounds of the pca box using IQR 
    statistics

    Parameters
    ----------
    tod : AxisManager
        observation axismanagers
    pca_aman : AxisManager
        output pca axismanager from get_pca_model
    xfac : int
        multiplicative factor for the width of the pca box.
        Default is 2.
    yfac : int
        multiplicative factor for the height of the box.
        Default is 1.5. 
    calc_good_medianw : bool
        If true, the resulting median weight is calculated 
        excluding bad dets. Default is false.

    Returns
    -------
    pca_relcal : AxisManager
        AxisManager pca and relcal information.
        
    """
    x = tod.det_cal.s_i
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
    iqry_norm = np.percentile(ynorm, 80) - np.percentile(ynorm, 20)

    # IQR of Si's
    iqrx = np.percentile(xfilt, 80) - np.percentile(xfilt, 20)

    # Find box heights using norm'd weights
    # Convert y bounds back to the scale of the raw weights
    ylb = (median_ynorm - yfac * iqry_norm) * np.median(yfilt)
    yub = (median_ynorm + yfac * iqry_norm) * np.median(yfilt)

    # Calculate box width
    xlb = medianx - xfac * iqrx
    xub = medianx + xfac * iqrx
    if xub > 0:
        mad = np.median(np.abs(xfilt - medianx))
        xub = medianx + xfac * mad

    xbounds = (xlb, xub)
    ybounds = (ylb, yub)

    # Get indices of the values in the box (indices are wrt `x` array)
    ranges = [x >= xlb,
              x <= xub,
              y >= ylb,
              y <= yub]
    m = ~(np.all(ranges, axis=0))

    if calc_good_medianw:
        medianw = np.median(pca_aman.weights[:,0][~m])
    else:
        medianw = np.median(pca_aman.weights[:,0])
    relcal_val = medianw/pca_aman.weights[:,0]

    pca_relcal = core.AxisManager(tod.dets, tod.samps)
    pca_relcal.wrap('pca_det_mask', m, [(0, 'dets')])
    pca_relcal.wrap('xbounds', np.array(xbounds))
    pca_relcal.wrap('ybounds', np.array(ybounds))
    pca_relcal.wrap('pca_mode0', pca_aman.modes[0], [(0, 'samps')])
    pca_relcal.wrap('pca_weight0', pca_aman.weights[:, 0], [(0, 'dets')])
    pca_relcal.wrap('relcal', relcal_val, [(0, 'dets')])
    pca_relcal.wrap('median', medianw)

    return pca_relcal


def get_common_mode(
    tod,
    signal='signal',
    method='median',
    wrap=None,
    weights=None,
):
    """Returns common mode timestream between detectors.
    This uses method 'median' or 'average' across detectors as opposed to a principle
    component analysis to get the common mode.

    Arguments
    ---------
        tod: axis manager
        signal: str, optional
            The name of the signal to estimate common mode or ndarray with shape of
            (n_dets x n_samps). Defaults to 'signal'.
        method: str
            method of common mode estimation. 'median' or 'average'.
        wrap: str or None.
            If not None, wrap the common mode into tod with this name.
        weights: array with dets axis
            If not None, estimate common mode by taking average with this weights.

    Returns
    -------
        common mode timestream

    """
    if isinstance(signal, str):
        signal = tod[signal]
    elif isinstance(signal, np.ndarray):
        if np.shape(signal) != (tod.dets.count, tod.samps.count):
            raise ValueError("When passing signal as ndarray shape must match (n_dets x n_samps).")
    else:
        raise TypeError("signal must be str, or ndarray")

    if method == 'median':
        if weights is not None:
            logger.warning('weights will be ignored because median method is chosen')
        common_mode = np.median(signal, axis=0)
    elif method == 'average':
        common_mode = np.average(signal, axis=0, weights=weights)
    else:
        raise ValueError("method flag must be median or average")
    if wrap is not None:
        tod.wrap(wrap, common_mode, [(0, 'samps')])
    return common_mode
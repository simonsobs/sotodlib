"""Module for deprojecting median Q/U from the data"""
import numpy as np
from sotodlib import core

def get_qu_common_mode_coeffs(aman, Q_signal=None, U_signal=None, merge=False):
    """
    Gets the median signal (template) and coefficients for the coupling to that 
    signal for each detector for both the Q and U signals. Returns an
    AxisManager with the template and coefficients wrapped.

    Arguments:
    ----------
    aman: AxisManager
        Contains the signal to operate on.
    Q_signal: ndarray or str
        array or string with field in aman containing the demodulated Q signal.
    U_signal: ndarray or str
        array or string with field in aman containing the demodulated U signal.
    merge: bool
        If True wrap the returned AxisManager into aman.

    Returns:
    --------
    output_aman: AxisManager
        Contains the template signals for Q/U and coefficients coupling
        each detector to the templates.
    """
    if Q_signal is None:
        Q_signal = aman['demodQ']
    if isinstance(Q_signal, str):
        Q_signal = aman[Q_signal]
    if not isinstance(Q_signal, np.ndarray):
        raise TypeError("Signal is not an array")

    if U_signal is None:
        U_signal = aman['demodU']
    if isinstance(U_signal, str):
        U_signal = aman[U_signal]
    if not isinstance(U_signal, np.ndarray):
        raise TypeError("Signal is not an array")

    output_aman = core.AxisManager(aman.dets, aman.samps)
    for sig, name in zip([Q_signal, U_signal], ['Q','U']):
        coeffs, med = _get_qu_template(aman, sig, False)
        output_aman.wrap(f'coeffs_{name}', coeffs[:,0], [(0, 'dets')])
        output_aman.wrap(f'med_{name}', med, [(0, 'samps')])
    if merge:
        aman.wrap('qu_common_mode_coeffs', output_aman)
    return output_aman

def subtract_qu_common_mode(aman, Q_signal=None, U_signal=None, coeff_aman=None,
                            merge=False, subtract=True):
    """
    Subtracts the median signal (template) from each detector scaled by the a 
    coupling coefficient per detector.

    Arguments:
    ----------
    aman: AxisManager
        Contains the signal to operate on.
    Q_signal: ndarray or str
        array or string with field in aman containing the demodulated Q signal.
    U_signal: ndarray or str
        array or string with field in aman containing the demodulated U signal.
    coeff_aman: AxisManager
        contains the coefficients and templates to use for subtraction.
        See ``get_qu_common_mode_coeffs``.
    merge: bool
        If True wrap the returned AxisManager into aman.
    """
    if Q_signal is None:
        Q_signal = aman['demodQ']
        Q_signal_name = 'demodQ'
    if isinstance(Q_signal, str):
        Q_signal_name = Q_signal
        Q_signal = aman[Q_signal]
    if not isinstance(Q_signal, np.ndarray):
        raise TypeError("Signal is not an array")

    if U_signal is None:
        U_signal = aman['demodU']
        U_signal_name = 'demodU'
    if isinstance(U_signal, str):
        U_signal_name = U_signal
        U_signal = aman[U_signal]
    if not isinstance(U_signal, np.ndarray):
        raise TypeError("Signal is not an array")

    if coeff_aman is None:
        if 'QU_common_mode_coeffs' in aman:
            coeff_aman = aman['QU_common_mode_coeffs']
        else:
            coeff_aman = get_qu_common_mode_coeffs(aman, Q_signal, U_signal, merge)

    aman[Q_signal_name] -= np.atleast_2d(coeff_aman['coeffs_Q']).T*coeff_aman['med_Q']
    aman[U_signal_name] -= np.atleast_2d(coeff_aman['coeffs_U']).T*coeff_aman['med_U']


def _get_qu_template(aman, signal, correct):
    """
    Calculates coefficients and median for the given demodulated Q or U data
    used for the deprojection.

    Parameters:
    -----------
    aman : AxisManager
        An AxisManager containing the demodulated Q and U components.
    signal : str
        The AxisManager field to access the specific signal in the aman object.

    Returns:
    --------
    tuple: A tuple containing:
        - coeffs (numpy.ndarray): The deprojected coefficients.
        - med (numpy.ndarray): The median values of the input data along the first axis.
    """
    med = np.median(signal, axis=0) 
    vects = np.atleast_2d(med)
    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(signal, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    return coeffs, med

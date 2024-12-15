"""Module for deprojecting median Q/U from the data"""
import numpy as np

def deprojection(aman, signal, correct):
    """
    Calculates coefficients and median for the given demodulated Q or U data
    used for the deprojection. Optionally corrects the data by removing the 
    median deprojection coefficients.

    Parameters:
    aman (object): An object containing the demodulated Q and U components
                    (tod.demodQ and tod.demodU).
    signal (str): The key to access the specific signal in the aman object.
    correct (bool): If True, the function will correct the data by removing 
                    the median deprojection coefficients.

    Returns:
    tuple: A tuple containing:
        - coeffs (numpy.ndarray): The deprojected coefficients.
        - med (numpy.ndarray): The median values of the input data along the first axis.
    """
    demodQU = aman[signal]
    med = np.median(demodQU, axis=0) 
    vects = np.atleast_2d(med)
    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(demodQU, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    if correct:
        aman[signal] = aman[signal] - coeffs * med
    return coeffs, med
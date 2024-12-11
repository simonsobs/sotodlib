"""Module for deprojecting median Q/U from the data"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def deprojection(aman, signal):
    """
    Calculates coefficients and median for the given demodulated Q or U data
    used for the deprojection.

    Parameters:
    aman (object): An object containing the demodulated Q and U components
                    (tod.demodQ and tod.demodU).
    signal (str): The key to access the specific signal in the aman object.

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
    return coeffs, med
    
def medQU_correct(aman, signal, QUcoeffs, QUmed):
    """
    Corrects the Q and U components of the given TOD by 
    removing the median deprojection coefficients.

    Parameters:
    aman (object): An object containing the demodulated Q and U components 
                   (tod.demodQ and tod.demodU).
    QUcoeffs (numpy.ndarray): The deprojected coefficients.
    QUmed (numpy.ndarray): The median values of the input data along the first axis.
    signal (str): The key to access the specific signal in the aman object.

    Returns:
    numpy.ndarray: The corrected demodulated Q and U components.
    """
    aman[signal] = aman[signal] - QUcoeffs * QUmed
    return 

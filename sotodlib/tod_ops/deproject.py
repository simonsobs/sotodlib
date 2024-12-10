"""Module for deprojecting median Q/U from the data"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def deprojection(demodQU):
    med = np.median(demodQU, axis=0) 
    vects = np.atleast_2d(med)
    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(demodQU, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    return coeffs, med
    
def medQU_correct(tod):
    Qcoeffs, Qmed = deprojection(tod.demodQ)
    demodQ = tod.demodQ - Qcoeffs * Qmed
    Ucoeffs, Umed = deprojection(tod.demodU)
    demodU = tod.demodU - Ucoeffs * Umed
    return demodQ, demodU

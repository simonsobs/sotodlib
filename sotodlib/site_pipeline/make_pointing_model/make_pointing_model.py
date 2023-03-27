import numpy as np
from scipy.optimize import curve_fit
import toml

def open_inst_model(inst_model_path):
    """
    Load the instrument model from a TOML file.

    Inputs
    ------
    inst_model_path (str): full path to the instrument model

    Returns
    -------
    inst_model (dict): the instrument model in dictionary form
    """
    with open(inst_model_path, "r") as f:
        dstr = f.read()
        inst_model = toml.loads(dstr)
    return inst_model

def get_det_quats(inst_model):
    """
    Output a dictionary of the xi, eta form quaternions from the instrument model.
    The positions are relative to the center of the focal plane, which is defined
    as (0,0,0,0).
    """
    det_quats = {}
    for det in inst_model['detectors']:
        w, x, y, z = inst_model['detectors'][det]['quat']
        det_quats[det] = np.array([float(w), float(x), float(y), float(z)])
    return det_quats

def Gaussian(x, A, m, s):
    """
    Define a Gaussian function, to be used for fitting later.

    Inputs
    ------
    x (float): a generic independent variable object.
    A (float): amplitude of the curve
    m (float): the expected value (mean) of the curve (aka mu)
    s (float): the standard deviation of the curve (aka sigma)

    Returns
    -------
    y (float): a generic dependent variable
    """
    y = A * np.exp((-1/2)*((x-m)/s)**2)
    return y

def gaussian_fit(time_data, amp_data):
    """
    Make a Gaussian fit of some normal-ish data. This data needs to be limited
    to one point source observation (don't feed the fitter all of the data
    at once!)

    Inputs
    ------
    time_data (list or array): the timestamps of the observation
    amp_data (list or array): the amplitude of measurement from the detectors

    Returns
    -------
    parameters (dict): the Gaussian parameters A, m, and s; covariance; and FWHM
    
    """
    xdata = np.asarray(time_data)
    ydata = np.asarray(amp_data)

    params, cov = curve_fit(Gaussian, xdata, ydata)

    fit_A = params[0]
    fit_m = params[1]
    fit_s = params[2]

    fwhm = 2 * np.sqrt(2 * np.log(2)) * fit_s

    parameters = {'A': fit_A,
                  'mu': fit_m,
                  'sigma': fit_s,
                  'FWHM': fwhm,
                  'cov': cov}

    return parameters

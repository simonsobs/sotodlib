import numpy as np
from scipy.optimize import least_squares#, minimize

def get_offsets(filename=None):
    # placeholder for retrieving the offsets from output of
    # make-focal-plane
    if filename:
        return
    else:
        return

def models(az_enc, el_enc, theta):
    # all inputs in radians!
    if isinstance(theta, dict):
        theta = (theta['ia'], theta['ie'], theta['an'], theta['aw'], theta['npae'], theta['ca'])
    ia, ie, an, aw, npae, ca = theta
    model_az = az_enc + ia - an * np.tan(el_enc) * np.sin(az_enc) - aw * np.tan(el_enc) * np.cos(az_enc) - npae * np.tan(el_enc) - ca / np.cos(el_enc)
    model_el = el_enc + ie - an * np.cos(az_enc) + aw * np.sin(az_enc)
    
    return model_az, model_el

def objective_function(params, az_enc, el_enc, az_obs, el_obs):
    model_az, model_el = models(az_enc, el_enc, params)

    # Calculate the sum of squared residuals
    residuals_az = model_az - az_obs
    residuals_el = model_el - el_obs
    sum_squared_residuals = np.sum(residuals_az**2) + np.sum(residuals_el**2)

    return sum_squared_residuals

def compute_residuals(params, az_enc, el_enc, az_obs, el_obs):
    # Your model implementation using the given parameters
    model_az, model_el = models(az_enc, el_enc, params)
    residuals_az = az_obs - model_az
    residuals_el = el_obs - model_el
    return np.concatenate((residuals_az, residuals_el))

def optimize_params(initial_guess, az_enc, el_enc, az_obs, el_obs):
    result = minimize(objective_function, initial_guess, args=(az_enc, el_enc, all_obs_az, all_obs_el))
    return result

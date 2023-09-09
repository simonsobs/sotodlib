import numpy as np
from scipy.optimize import least_squares
import h5py, yaml

# TODO: load in delta_az from Saianeesh's output
# az_enc = np.radians(az_enc_data)
# el_enc = np.radians(el_enc_data)
# br_enc = np.radians(br_enc_data)
# delta_az = np.radians(delta_az_data)
# delta_el = np.radians(delta_el_data)
# delta_br = np.radians(delta_br_data)
# return az_enc, el_enc, br_enc, delta_az, delta_el, delta_br

# I think data is read in in degrees, so for now assume that is true.
# for the sake of future unit conversions, leave some conversion to radians here
DEG = np.pi / 180
ARCMIN = DEG / 60
ARCSEC = ARCMIN / 60

def az_fn(params, az_enc, el_enc):
    # TODO: what happens as el -> 90 deg?
    ia, an, aw, npae, ca = params
    az = np.radians(az_enc)
    el = np.radians(el_enc)
    # the independent parameters
    new_az = az + ia - ca * np.tan(el) - npae / np.cos(el)
    # the az/el shared parameters
    new_az += - an * np.tan(el) * np.sin(az) - aw * np.tan(el) * np.cos(az)
    return new_az # in radians

def el_fn(params, az_enc, el_enc):
    ie, an, aw = params
    # the independent params
    new_el = el_enc + ie
    # the shared params
    new_el += - an * np.cos(az_enc) + aw * np.sin(az_enc)
    return new_el # in radians

def br_fn(params, br_enc, el_enc):
    # terms borrowed from CLASS; is this useful/effective?
    c11, c12, c13, c21, c22, c23, c31, c32, c33 = params
    br = np.radians(br_enc)
    el = np.radians(el_enc)
    delta_br = 0
    # the parameters
    delta_br += c11
    delta_br += c12 * br
    delta_br += c13 * br ** 2
    delta_br += c21 * (el - np.pi/2)
    delta_br += c22 * (el - np.pi/2) * br
    delta_br += c23 * (el - np.pi/2) * br ** 2
    delta_br += c31 * (el - np.pi/2) ** 2
    delta_br += c32 * (el - np.pi/2) ** 2 * br
    delta_br += c33 * (el - np.pi/2) ** 2 + br ** 2

    new_br = br + delta_br
    return new_br # in radians

def objective_fn_az(params, az_enc, el_enc, delta_az):
    az_pred = az_fn(params, az_enc, el_enc)
    az_data = az_enc - delta_az
    return (az_data - az_pred) ** 2

def objective_fn_el(params, az_enc, el_enc, delta_el):
    el_pred = el_fn(params, az_enc, el_enc)
    el_data = el_enc - delta_el
    return (el_data - el_pred) ** 2

def objective_fn_br(params, br_enc, el_enc, delta_br):
    br_pred = br_fn(params, br_enc, el_enc)
    br_data = br_enc - delta_br
    return (br_data - br_pred) ** 2

def find_params_az(initial_guess_az):
#    initial_guess_az = [ia_guess, an_guess, aw_guess, npae_guess, ca_guess]
    az_result = least_squares(objective_fn_az, initial_guess_az)
    ia_opt, an_opt, aw_opt, npae_opt, ca_opt = az_result.x
    return ia_opt, an_opt, aw_opt, npae_opt, ca_opt

def find_params_el(initial_guess_el):
#    initial_guess_el = [ie_guess, an_guess, aw_guess]
    el_result = least_squares(objective_fn_el, initial_guess_el)
    ie_opt, an_opt, aw_opt = el_result.x
    return ie_opt, an_opt, aw_opt

def find_params_br(initial_guess_br):
#    initial_guess_br = [ib_guess]
    br_result = least_squares(objective_fn_br, initial_guess_br)
    ib_opt = br_result.x
    return c11, c12, c13, c21, c22, c23, c31, c32, c33

if __name__ == "__main__":
    # TODO: get the data from Saianeesh's file

    # TODO: get initial guess, either from star camera or previous PM
    guesses = {'ia': 1,
               'ie': 1,
               'ib': 1,
               'an': 1,
               'aw': 1,
               'npae': 1,
               'ca': 1,
               }
                   
    # TODO: get PM version from the previous PM file if one exists
    pm_version  = 0

    az_init_guess = [guesses['ia'], guesses['an'], guesses['aw'], guesses['npae'], guesses['ca']]
    az_opt_results = find_params_az(az_init_guess)

    el_init_guess = [guesses['ie'], az_opt_results[1], az_opt_results[2]]
    el_opt_results = find_params_el(el_init_guess)

    br_init_guess = [guesses['ib']]
    br_opt_results = find_params_br(br_init_guess)

    if el_opt_results[1] != az_opt_results[1]:
        an_opt = (el_opt_results[1] + az_opt_results[1]) / 2
    else:
        an_opt = az_opt_results[1]

    if el_opt_results[2] != az_opt_results[2]:
        aw_opt = (el_opt_results[2] + az_opt_results[2]) / 2
    else:
        aw_opt = az_opt_results[2]

    opt_results = {'version': version,
                   'params': {
                       'ia': az_opt_results[0],
                       'ie': el_opt_results[0],
                       'ib': br_opt_results[0],
                       'an': an_opt,
                       'aw': aw_opt,
                       'npae': az_opt_results[3],
                       'ca': az_opt_results[4],
                       },
                   }

    # TODO: clean up this save; should it be different file type?
    new_filename = 'pointing_version_%s.json' % version
    with open(new_filename, 'w') as f:
        yaml.dump(opt_results, f)

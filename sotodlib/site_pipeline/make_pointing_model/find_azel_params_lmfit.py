import numpy as np
import lmfit
import yaml

def azel_pmodels(azel, ia, ie, an, aw, npae, ca):
    """
    A generic model for a perturbative pointing model. All values
    in radians.

    Inputs
    ------
    azel (np.array): array of azimuth and elevation encoder
        values, structured as np.array([list_of_az, list_of_el])
    ia (float): zero-point azimuth encoder error value (TPOINT
        param IA)
    ie (float): zero-point elevation encoder error value (TPOINT
        param IE)
    an (float): north/south tilt value (TPOINT param AN)
    aw (float): east/west tilt value (TPOINT param AW)
    npae (float): (TPOINT param NPAE)
    ca (float): (TPOINT param CA)

    Outputs
    -------
    model_az (np.array): array of perturbed azimuth values
    model_el (np.array): array of perturbed elevation values
    """

    az_enc = azel[0]
    el_enc = azel[1]
    model_az = az_enc + ia - an * np.tan(el_enc) * np.sin(az_enc) - aw * np.tan(el_enc) * np.cos(az_enc) - npae * np.tan(el_enc) - ca / np.cos(el_enc)
    model_el = el_enc + ie - an * np.cos(az_enc) + aw * np.sin(az_enc)

    return model_az, model_el

def run_fit(enc_data, obs_data, initial_guess):
    """
    Uses lmfit package to fit parameters from data. All values
    in radians.

    Inputs
    ------
    enc_data (np.array): array of az and el encoder values. Same
        as azel input of azel_pmodels
    obs_data (np.array): array of observed (likely offset) data.
        Same format as azel input of azel_pmodels
    initial_guess (dict): initial guesses for the pointing params
        in a dictionary with keys 'ia', 'ie', 'an', 'aw', 'npae',
        and 'ca'

    Outputs
    -------
    fit_result (lmfit.model.ModelResult): fitting result, with
        values and related statistics
    """

    pmodel = lmfit.Model(azel_pmodels)
    ia_init = initial_guess['ia']
    ie_init = initial_guess['ie']
    an_init = initial_guess['an']
    aw_init = initial_guess['aw']
    npae_init = initial_guess['npae']
    ca_init = initial_guess['ca']

    params = pmodel.make_params(ia=ia_init, ie=ie_init, an=an_init, aw=aw_init, npae=npae_init, ca=ca_init)

    fit_result = pmodel.fit(obs_data, params, azel=enc_data)

    return fit_result

def write_results_file(fit_result, old_version, version_step):
    param_fits = fit_result.params
    pm_version = old_version + version_step
    info = {'pm_version': pm_version,
            'fit_stats': {'fit_method': fit_result.method,
                          'chisquare': fit_result.chisqr,
                          'bic': fit_result.bic,
                          'aic': fit_result.aic,
                          'rsquared': fit_result.rsquared,
                          },
            'fit_values': {'ia': param_fits['ia'].value,
                           'ie': param_fits['ie'].value,
                           'an': param_fits['an'].value,
                           'aw': param_fits['aw'].value,
                           'npae': param_fits['npae'].value,
                           'ca': param_fits['ca'].value,
                           },
            'fit_stderrs': {'ia': param_fits['ia'].stderr,
                            'ie': param_fits['ie'].stderr,
                            'an': param_fits['an'].stderr,
                            'aw': param_fits['aw'].stderr,
                            'npae': param_fits['npae'].stderr,
                            'ca': param_fits['ca'].stderr,
                            },
            'cov': {fit_result.covar},
            }
    with open('pointing_model_version_%s.yaml'%pm_version, 'w') as f:
        yaml.dump(info, f)
    f.close()

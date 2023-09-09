import numpy as np
import lmfit

def br_model(enc_vals, c11, c12, c13, c21, c22, c23):
    br_enc = enc_vals[0]
    el_enc = enc_vals[1]
    model_br = br_enc + c11 + c12 * br_enc + c13 * br_enc**2
    model_br += c21 * (el_enc - np.pi/2) + c22 * (el_enc - np.pi/2) * br_enc + c23 * (el_enc - np.pi/2) ** 2 * br_enc ** 2
#    model_br += c31 * (el_enc - np.pi/2) ** 2 + c32 * (el_enc - np.pi/2) ** 2 * br_enc + c33 * (el_enc - np.pi/2) ** 2 * br_enc ** 2
    return model_br

def make_model_fit(init_guess, obs_vals, enc_vals):
    br_obs = obs_vals[0]
    el_obs = obs_vals[1]
    br_enc = enc_vals[0]
    el_enc = enc_vals[1]
    bmodel = lmfit.Model(br_model)
    c11_init = init_guess['c11']
    c12_init = init_guess['c12']
    c13_init = init_guess['c13']
    c21_init = init_guess['c21']
    c22_init = init_guess['c22']
    c23_init = init_guess['c23']
    params = bmodel.make_params(c11=c11_init, c12=c12_init, c13=c13_init, c21=c21_init, c22=c22_init, c23=c23_init)
    
    fit_result = bmodel.fit(br_obs, params, br_enc)
    return fit_result

def write_results_to_file(fit_result, old_version, version_step):
    param_fits = fit_result.params
    pm_version = old_version + version_step
    info = {'pm_version': pm_version,
            'fit_stats': {'fit_method': fit_result.method,
                          'chisquare': fit_result.chisqr,
                          'bic': fit_result.bic,
                          'aic': fit_result.aic,
                          'rsquared': fit_result.rsquared,
                          },
            'fit_values': {'c11': param_fits['c11'].value,
                           'c12': param_fits['c11'].value,
                           'c13': param_fits['c11'].value,
                           'c21': param_fits['c11'].value,
                           'c22': param_fits['c11'].value,
                           'c23': param_fits['c11'].value,
                          },
            'fit_stderrs': {'c11': param_fits['c11'].stderr,
                            'c12': param_fits['c11'].stderr,
                            'c13': param_fits['c11'].stderr,
                            'c21': param_fits['c11'].stderr,
                            'c22': param_fits['c11'].stderr,
                            'c23': param_fits['c11'].stderr,
                           },
            'cov': {fit_result.covar},
            }
    with open('pointing_br_model_version_%s.yaml'%pm_version, 'w') as f:
        yaml.dump(info, f)
    f.close()

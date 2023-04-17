import numpy as np
from scipy.optimize import least_squares

def residuals(params, az_enc, el_enc, delta_az, delta_el):
    ia, ca, npae, an, aw, hece, ie = params

    res_az = delta_az - ia - (ca / np.cos(enc_el)) - npae * np.tan(enc_el) - an * np.tan(enc_el) * np.sin(enc_az) - aw * np.tan(enc_el) * np.cos(enc_az)
    res_el = delta_el + ie + hece * np.cos(enc_el) - an * np.cos(enc_az) + aw * np.sin(enc_az)

    res_az = np.atleast_1d(res_az)
    res_el = np.atleast_1d(res_el)

    return np.concatenate((res_x, res_y))

def interpret_residuals(result):
    ia, ca, npae, an, aw, hece, ie = result.x
    tpoint_params = {'IA': ia,
                     'IE': ie,
                     'HECE': hece,
                     'CA': ca,
                     'NPAE': npae,
                     'AN': an,
                     'AW': aw,
                     }
    return tpoint_params

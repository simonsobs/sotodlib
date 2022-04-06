from astropy import constants
from astropy import units as u
import numpy as np

def tb2s(tb, nu):
    """ Convert blackbody temperature to spectral
    radiance s_nu at frequency nu

    Args:
        tb: float or array
            blackbody temperature, unit: Kelvin
        nu: float or array (with same dimension as tb)
            frequency where the spectral radiance is evaluated, unit: Hz

    Return:
        s_nu: same dimension as tb
            spectral radiance s_nu, unit: W*sr−1*m−2*Hz−1
    """
    h = constants.h
    c = constants.c
    k_b = constants.k_B

    if isinstance(nu, u.Quantity):
        nu = nu.to(u.Hz)
    else:
        nu = nu * u.Hz

    if not tb.unit == u.K:
        tb = tb * u.K

    x = h * nu / (k_b * tb)

    return 2 * h * nu ** 3 / c ** 2 / (np.exp(x) - 1)


def s2tcmb(s_nu, nu):
    """ Convert spectral radiance s_nu at frequency nu to t_cmb,
    t_cmb is defined in the CMB community as the offset from the
    mean CMB temperature assuming a linear relation between t_cmb
    and s_nu, the t_cmb/s_nu slope is evalutated at the mean CMB
    temperature.

    Args:
        s_nu: float or array
            spectral radiance s_nu, unit: W*sr−1*m−2*Hz−1
        nu: float or array (with same dimension as s_nu)
            frequency where the evaluation is perfomed, unit: Hz

    Return:
        t_cmb: same dimension as s_nu
            t_cmb, unit: Kelvin_cmb
    """
    T_cmb = 2.72548 * u.K  # K from Fixsen, 2009, ApJ 707 (2): 916–920
    h = constants.h
    c = constants.c
    k_b = constants.k_B

    if isinstance(nu, u.Quantity):
        nu = nu.to(u.Hz)
    else:
        nu = nu * u.Hz

    x = h * nu / (k_b * T_cmb)

    slope = 2 * k_b * nu ** 2 / c ** 2 * ((x / 2) / np.sinh(x / 2)) ** 2

    return s_nu / slope


def tb2tcmb(tb, nu):
    """Convert blackbody temperature to t_cmb
    as defined above

    Args:
        tb: float or array
            blackbody temperature, unit: Kelvin
        nu: float or array (with same dimension as tb)
            frequency where the spectral radiance is evaluated, unit: Hz

    Return
        t_cmb: same dimension as tb
            t_cmb, unit: Kelvin_cmb
    """
    s_nu = tb2s(tb, nu)
    return s2tcmb(s_nu, nu)
# Copyright (c) 2018-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

from astropy import constants
from astropy import units as u
import numpy as np

def tb2s(tb, nu):
    """Convert blackbody temperature to spectral radiance s_nu at frequency nu.

    Args:
        tb (Quantity): Blackbody temperature
        nu (Quantity): frequency where the spectral radiance is evaluated

    Returns:
        (Quantity) s_nu with same dimensions as tb
            nominal unit: W*sr−1*m−2*Hz−1
    """
    h = constants.h
    c = constants.c
    k_b = constants.k_B
    nu = nu.to(u.Hz)
    tb = tb.to(u.K)
    x = h * nu / (k_b * tb)
    return 2 * h * nu**3 / c**2 / (np.exp(x) - 1)


def s2tcmb(s_nu, nu):
    """Convert spectral radiance s_nu at frequency nu to t_cmb.

    t_cmb is defined in the CMB community as the offset from the
    mean CMB temperature assuming a linear relation between t_cmb
    and s_nu, the t_cmb/s_nu slope is evalutated at the mean CMB
    temperature.

    Args:
        s_nu (Quantity):  spectral radiance s_nu (nominal unit: W*sr−1*m−2*Hz−1)
        nu (Quantity): frequency where the evaluation is perfomed

    Returns:
        (Quantity) t_cmb with same dimensions as tb.

    """
    T_cmb = 2.72548 * u.K  # K from Fixsen, 2009, ApJ 707 (2): 916–920
    h = constants.h
    c = constants.c
    k_b = constants.k_B
    nu = nu.to(u.Hz)
    x = h * nu / (k_b * T_cmb)

    slope = 2 * k_b * nu**2 / c**2 * ((x.value / 2) / np.sinh(x.value / 2)) ** 2
    return s_nu / slope


def tb2tcmb(tb, nu):
    """Convert blackbody temperature to t_cmb as defined above.

    Args:
        tb (Quantity): Blackbody temperature
        nu (Quantity): frequency where the spectral radiance is evaluated

    Returns:
        (Quantity) t_cmb with same dimensions as tb

    """
    s_nu = tb2s(tb, nu)
    result = s2tcmb(s_nu, nu)
    return result.decompose()

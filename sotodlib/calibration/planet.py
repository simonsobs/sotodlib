from sotodlib import coords
from astropy.time import Time
import os
import numpy as np
from numpy import sin, cos, pi
from numpy import deg2rad as d2r
from numpy import rad2deg as r2d

from astropy import units
au = units.au.si.to(units.m)


# fiducial models of planets
# R_eq, R_pole, Omega_ref, and d_ref are comes from Table2 of Weiland et al. (2010)
# SEVEN-YEAR WMAP OBSERVATIONS: PLANETS AND CELESTIAL CALIBRATION SOURCES
mars_fiducial = {
    'R_eq': 3396e3,
    'R_pole': 3376e3,
    'Omega_ref': 7.153e-10,
    'd_ref': 1.5 * au,
    'Trj': {'f090': None,
            'f150': None}
}
jupiter_fiducial = {
    'R_eq': 71492e3,
    'R_pole': 66854e3,
    'd_ref': 5.2 * au,
    'Omega_ref': 2.481e-8,
    'Trj': {'f090': 166.73,  # from ESA1 * SO bandpass
            'f150': 166.81}
}
saturn_fiducial = {
    'R_eq': 60268e3,
    'R_pole': 54364e3,
    'd_ref': 9.5 * au,
    'Omega_ref': 5.096e-9,
    'Trj': {'f090': None,
            'f150': None}
}
uranus_fiducial = {
    'R_eq': 25559e3,
    'R_pole': 24973e3,
    'Omega_ref': 2.482e-10,
    'd_ref': 19. * au,
    'Trj': {'f090': 130.76,  # from ESA4 * SO bandpass
            'f150': 104.10}
}
neptune_fiducial = {
    'R_eq': 24764e3,
    'R_pole': 24341e3,
    'Omega_ref': 1.006e-10,
    'd_ref': 29. * au,
    'Trj': {'f090': 121.77,  # from ESA4 * SO bandpass
            'f150': 108.10}
}

fiducial_models = {
    'mars': mars_fiducial,
    'jupiter': jupiter_fiducial,
    'saturn': saturn_fiducial,
    'uranus': uranus_fiducial,
    'neptune': neptune_fiducial
}


def calc_model_temperature(bandpass_name, bandpass_suffix,
                           planet_name, model_name):
    """
    Args:
        bandpass_name (str): Name of SO's bandpass (e.g. 'f090', 'f150')
        bandpass_suffix (str): Suffix of the bandpass file (e.g., 'design', 'measured').
        planet_name (str): Name of the planet (e.g. 'uranus', 'neptune')
        model_name (str): Name of the planet model file without the extension.
    Returns:
    """
    bandpass_dir = os.path.join(
        os.path.dirname(
            sotodlib.__file__),
        'calibration',
        'bandpass')
    bandpass_file = os.path.join(bandpass_dir,
                                 f'{bandpass_name}_{bandpass_suffix}.txt')
    bandpass_data = np.loadtxt(bandpass_file, comments='#')
    freq_GHz_bandpass = bandpass_data[:, 0]
    trans_bandpass = bandpass_data[:, 1]

    planet_model_dir = os.path.join(
        os.path.dirname(
            sotodlib.__file__),
        'calibration',
        'planet_models')
    planet_model_file = os.path.join(
        planet_model_dir,
        planet_name,
        f'{planet_name}_{model_name}.txt')
    planet_model_data = np.loadtxt(planet_model_file, comments='#')
    freq_GHz_planet = planet_model_data[:, 0]
    Trj_planet = planet_model_data[:, 1]

    Trj_planet_interp = np.interp(
        freq_GHz_bandpass,
        freq_GHz_planet,
        Trj_planet)

    integral_numer = np.trapz(
        trans_bandpass *
        Trj_planet_interp,
        freq_GHz_bandpass)
    integral_denom = np.trapz(trans_bandpass, freq_GHz_bandpass)
    band_averaged_Trj = integral_numer / integral_denom
    return band_averaged_Trj


def get_expected_T_Omega(planet, bandpass_name, timestamp):
    T_planet = fiducial_models[planet]['Trj'][bandpass_name]
    Omega_planet_ref = fiducial_models[planet]['Omega_ref']
    f_A = get_distance_correction_factor(planet, timestamp)
    De = get_sub_earth_latitude(planet, timestamp)
    f_d = get_disk_oblatenes_correction_factor(planet, De)
    expected_T_Omega = T_planet * Omega_planet_ref * f_A / f_d
    return expected_T_Omega


def get_distance_correction_factor(planet, timestamp):
    ra, dec, distance = coords.planets.get_source_pos(planet, timestamp)
    distance *= au
    disatnce_fiducial = fiducial_models[planet]['d_ref']
    f_d = (distance / disatnce_fiducial)**2
    return f_d


def get_disk_oblatenes_correction_factor(planet, De):
    R_eq = fiducial_models[planet]['R_eq']
    R_pole = fiducial_models[planet]['R_pole']
    A_ref = np.pi * R_pole * R_eq

    R_proj_pole = R_pole * \
        np.sqrt(1 - np.sin(De)**2 * (1 - (R_eq / R_pole)**2))
    A_proj_disk = np.pi * R_proj_pole * R_eq
    f_A = A_proj_disk / A_ref
    return f_A


def get_sub_earth_latitude(planet, timestamp):
    """
    Calculate sub-earth latitude, De, from "Report of the IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015"
    alpha0, delta0:
        Are ICRF equatorial coordinates at epoch J2000.0
        Approximate coordinates of the north pole of the invariable plane are alpha0 = 273◦.85, delta0 = 66◦.99
    T:
        Interval in Julian centuries (36,525 days) from the standard epoch
    d:
        Interval in days from the standard epoch
    where the standard epoch is JD 2451545.0

    Sun:
        alpha0 = 286◦.13
        delta0 = 63◦.87
        W = 84◦.176 + 14◦.1844000d(a)

    Mercury:
        alpha0 = 281.0103 − 0.0328 T
        delta0 = 61.4155 − 0.0049 T
        W = 329.5988 ± 0.0037 + 6.1385108d
            + 0◦.01067257 sin M1
            − 0◦.00112309 sin M2
            − 0◦.00011040 sin M3
            − 0◦.00002539 sin M4
            − 0◦.00000571 sin M5
            where
            M1 = 174◦.7910857 + 4◦.092335d
            M2 = 349◦.5821714 + 8◦.184670d
            M3 = 164◦.3732571 + 12◦.277005d
            M4 = 339◦.1643429 + 16◦.369340d
            M5 = 153◦.9554286 + 20◦.461675d

    Venus:
        alpha0 = 272.76
        delta0 = 67.16
        W = 160.20 − 1.4813688d

    Mars:
        alpha0 = 317.269202 − 0.10927547T
            + 0.000068 sin(198.991226 + 19139.4819985T )
            + 0.000238 sin(226.292679 + 38280.8511281T )
            + 0.000052 sin(249.663391 + 57420.7251593T )
            + 0.000009 sin(266.183510 + 76560.6367950T )
            + 0.419057 sin(79.398797 + 0.5042615T )
        delta0 = 54.432516 − 0.05827105T
            + 0.000051 cos(122.433576 + 19139.9407476T )
            + 0.000141 cos(43.058401 + 38280.8753272T )
            + 0.000031 cos(57.663379 + 57420.7517205T )
            + 0.000005 cos(79.476401 + 76560.6495004T )
            + 1.591274 cos(166.325722 + 0.5042615T )
        W = 176.049863 + 350.891982443297d
            + 0.000145 sin(129.071773 + 19140.0328244T )
            + 0.000157 sin(36.352167 + 38281.0473591T )
            + 0.000040 sin(56.668646 + 57420.9295360T )
            + 0.000001 sin(67.364003 + 76560.2552215T )
            + 0.000001 sin(104.792680 + 95700.4387578T )
            + 0.584542 sin(95.391654 + 0.5042615T )

    Jupiter:
        alpha0 = 268.056595 − 0.006499T + 0◦.000117 sin Ja + 0◦.000938 sin Jb
            + 0.001432 sin Jc + 0.000030 sin Jd + 0.002150 sin Je
        delta0 = 64.495303 + 0.002413T + 0.000050 cos Ja + 0.000404 cos Jb
            + 0.000617 cos Jc − 0.000013 cos Jd + 0.000926 cos Je
        W = 284.95 + 870.5360000d
        where
        Ja = 99◦.360714 + 4850◦.4046T, Jb = 175◦.895369 + 1191◦.9605T,
        Jc = 300◦.323162 + 262◦.5475T, Jd = 114◦.012305 + 6070◦.2476T,
        Je = 49◦.511251 + 64◦.3000T

    Saturn::
        alpha0 = 40.589 − 0.036T
        delta0 = 83.537 − 0.004T
        W = 38.90 + 810.7939024d

    Uranus:
        alpha0 = 257.311
        delta0 = −15.175
        W = 203.81 − 501.1600928d

    Neptune:
        alpha0 = 299.36 + 0.70 sin N
        delta0 = 43.46 − 0.51 cos N
        W = 249.978 + 541.1397757d − 0.48 sin N
        N = 357.85 + 52.316T

    Input:
        planet: name of the planet
        timestamp: timestamp of the observation in UTC
    Returns:
        De: sub-earth latitude = tilt angle of polar axis of planet towards the Earth
    """
    # convert UTC to JD
    time = Time(timestamp, format='unix')
    d = time.jd - 2451545.0
    T = d / 36525.0

    if planet == 'mercury':
        alpha = 281.0103 - 0.0328 * T
        delta = 61.4155 - 0.0049 * T
        M1 = 174.7910857 + 4.092335 * d
        M2 = 349.5821714 + 8.184670 * d
        M3 = 164.3732571 + 12.277005 * d
        M4 = 339.1643429 + 16.369340 * d
        M5 = 153.9554286 + 20.461675 * d
        W = 329.5469 + 6.1385025 * d + 0.00993822 * sin(d2r(M1)) + 0.00104581 * sin(d2r(
            M2)) + 0.00009431 * sin(d2r(M3)) + 0.00000790 * sin(d2r(M4)) + 0.00000071 * sin(d2r(M5))

    if planet == 'venus':
        alpha = 272.76
        delta = 67.16
        W = 160.20 - 1.4813688 * d

    if planet == 'mars':
        alpha = 317.269202 - 0.10927547 * T + 0.000068 * sin(d2r(198.991226 + 19139.4819985 * T)) + 0.000238 * sin(d2r(226.292679 + 38280.8511281 * T)) + 0.000052 * sin(
            d2r(249.663391 + 57420.7251593 * T)) + 0.000009 * sin(d2r(266.183510 + 76560.6367950 * T)) + 0.419057 * sin(d2r(79.398797 + 0.5042615 * T))
        delta = 54.432516 - 0.05827105 * T + 0.000051 * cos(d2r(122.433576 + 19139.9407476 * T)) + 0.000141 * cos(d2r(43.058401 + 38280.8753272 * T)) + 0.000031 * cos(
            d2r(57.663379 + 57420.7517205 * T)) + 0.000005 * cos(d2r(79.476401 + 76560.6495004 * T)) + 1.591274 * cos(d2r(166.325722 + 0.5042615 * T))
        W = 176.049863 + 350.891982443297 * d + 0.000145 * sin(d2r(129.071773 + 19140.0328244 * T)) + 0.000157 * sin(d2r(36.352167 + 38281.0473591 * T)) + 0.000040 * sin(d2r(
            56.668646 + 57420.9295360 * T)) + 0.000001 * sin(d2r(67.364003 + 76560.2552215 * T)) + 0.000001 * sin(d2r(104.792680 + 95700.4387578 * T)) + 0.584542 * sin(d2r(95.391654 + 0.5042615 * T))

    if planet == 'jupiter':
        Ja = 99.360714 + 4850.4046 * T
        Jb = 175.895369 + 1191.9605 * T
        Jc = 300.323162 + 262.5475 * T
        Jd = 114.012305 + 6070.2476 * T
        Je = 49.511251 + 64.3000 * T
        alpha = 268.056595 - 0.006499 * T + 0.000117 * sin(d2r(Ja)) + 0.000938 * sin(d2r(
            Jb)) + 0.001432 * sin(d2r(Jc)) + 0.000030 * sin(d2r(Jd)) + 0.002150 * sin(d2r(Je))
        delta = 64.495303 + 0.002413 * T + 0.000050 * cos(d2r(Ja)) + 0.000404 * cos(d2r(
            Jb)) + 0.000617 * cos(d2r(Jc)) - 0.000013 * cos(d2r(Jd)) + 0.000926 * cos(d2r(Je))
        W = 284.95 + 870.5360000 * d

    if planet == 'saturn':
        alpha = 40.589 - 0.036 * T
        delta = 83.537 - 0.004 * T
        W = 38.90 + 810.7939024 * d

    if planet == 'uranus':
        alpha = 257.311
        delta = -15.175
        W = 203.81 - 501.1600928 * d

    if planet == 'neptune':
        N = 357.85 + 52.316 * T
        alpha = 299.36 + 0.70 * sin(d2r(N))
        delta = 43.46 - 0.51 * cos(d2r(N))
        W = 249.978 + 541.1397757 * d - 0.48 * sin(d2r(N))

    alpha = d2r(alpha)
    delta = d2r(delta)
    W = d2r(W)

    # position of the planet
    ra, dec, distance_au = coords.planets.get_source_pos(planet, timestamp)

    # convert radec to iso
    theta1 = pi / 2 - dec
    phi1 = ra
    theta2 = pi / 2 - delta
    phi2 = alpha

    # calculate the latitude of the planet coordinate system relative to the
    # Earth
    n_sight = np.array(
        [sin(theta1) * cos(phi1), sin(theta1) * sin(phi1), cos(theta1)])
    n_pole = np.array([sin(theta2) *
                       cos(phi2), sin(theta2) *
                       sin(phi2), cos(theta2)])
    theta3 = np.arccos(np.dot(n_sight, n_pole))
    De = pi / 2 - theta3
    return De

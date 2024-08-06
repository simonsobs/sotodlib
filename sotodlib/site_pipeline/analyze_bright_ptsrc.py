#!/usr/bin/env python3
import matplotlib.pyplot as plt
import so3g
from so3g.proj import coords, quat
import sotodlib.coords.planets as planets
from sotodlib import core
from sotodlib.toast.ops import sim_sso
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset, read_dataset
from sotodlib.tod_ops.filters import high_pass_sine2, low_pass_sine2, fourier_filter
import pickle
from astropy import units as u
from copy import deepcopy
import numpy as np
import scipy.signal
import time
import ephem
import os
from tqdm.auto import tqdm
from scipy.optimize import curve_fit, minimize
from datetime import datetime
import argparse as ap
import yaml
from sotodlib.site_pipeline import util
from dataclasses import dataclass
import matplotlib
matplotlib.use('agg')

opj = os.path.join

logger = util.init_logger(__name__)


def find_source(
        tod,
        sso_name,
        beamsize=30,
        height=0.1,
        distance=10000,
        npeaks=3):
    """Find the detectors that see the source

    Args
    ------
    height: minimum required amplitude of a peak
    max_amp: over this amplitude it can only be a spike
    distance: distance between adjacent peaks -- depends on scan speed
    npeaks: number of required peaks to be found of certain amplitude
    beamsize: cluster of peaks with lower width than this is considered a spike

    Return
    ------
    good_dets : array of indices of the detectors that see the source
    """
    az = tod.boresight.az
    el = tod.boresight.el
    ctime = tod.timestamps

    good_dets = []

    for det_idx in range(len(tod.dets.vals)):

        data = tod.signal[det_idx, :]
        data -= np.mean(data)
        # find npeaks over some amplitude and sort them
        r, _ = scipy.signal.find_peaks(data, height=height, distance=distance)
        sorted_r = np.argsort(data[r])[::-1]
        # to avoid high amplitude spikes
        if len(r) < npeaks or np.nanmax(data) > 100 * height:
            continue

        peak_idxs = r[sorted_r[:npeaks]]
        peak_idxs_sorted = np.sort(peak_idxs)

        try:
            argmax = np.argmax(data[peak_idxs_sorted])
        except BaseException:
            continue

        # Separate between peaks left and right from the highest peak
        left_d = [data[i] for i in peak_idxs_sorted[:argmax + 1]]
        right_d = [data[i] for i in peak_idxs_sorted[argmax:]]
        # Assert the monotony changes at the highest peak to make sure
        # the source center is included
        if len(left_d) >= 2 and len(right_d) >= 2:
            if np.all(
                left_d == np.sort(left_d)) and np.all(
                right_d == np.sort(right_d)[
                    ::-1]):
                # Turn to source-centred coordinates
                all_coords = get_xieta_src_centered(
                    ctime, az, el, data, sso_name=sso_name, threshold=height, distance=distance)
                [xi, eta], [xi_det, eta_det] = all_coords
                # Attempt an estimation of the width to determine if we
                # are seeing a point source crossing or random spikes
                max_peak = np.nanmax(data) / 2
                sorted_idx = np.argsort(np.abs(data - max_peak))[:1000]

                xi_min = np.nanmin(xi[sorted_idx])
                xi_max = np.nanmax(xi[sorted_idx])
                eta_min = np.nanmin(eta[sorted_idx])
                eta_max = np.nanmax(eta[sorted_idx])

                fwhm_xi = np.degrees(xi_max - xi_min) * 60
                fwhm_eta = np.degrees(eta_max - eta_min) * 60

                # If it is smaller than 3/4 * beam size is a spike
                # we can't do narrower than the diffraction limit
                # + some effect from the filtering
                if fwhm_xi < 3 / 4 * beamsize and fwhm_eta < 3 / 4 * beamsize:
                    continue

                xi_det_center = xi - xi_det
                eta_det_center = eta - eta_det
                radius_main = np.radians(2.5)
                radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)
                idx_out = np.where(radius > radius_main)[0]
                snr = np.nanmax(data) / np.nanstd(data[idx_out])

                # Something extra we can do but not necessary - targetted to Moon data
                # Get the first sorted 20 peaks and assert the difference in amplitude
                # between the first and last is big (otherwise we caught some periodical
                # spikes and were unlucky enough to agree with the previous criteria)
                # r, _ = scipy.signal.find_peaks(data, height=height/3, distance=10000)
                # sorted_r = np.argsort(data[r])[::-1]

                # if len(r)<npeaks:
                #     continue

                # peak_idxs = r[sorted_r[:20]]
                # ratio = data[peak_idxs[-1]]/data[peak_idxs[0]]

                # As a subsequent sanity check look at the SNR
                if snr > 25:
                    good_dets.append(det_idx)

    return good_dets


def get_xieta_src_centered(
        ctime,
        az,
        el,
        data,
        sso_name,
        threshold,
        distance,
        prior_pos=None):
    """Create a planet centered coordinate system for single detector data

    Args
    ------
    ctime, az, el: unix time and boresight position coordinates
    data: single detector data
    sso_name: celestial source to insert to pyephem
    threshold: minimal peak amplitude required

    Return
    ------
    xi_total, eta_total : source's centre projected on the focal plane coords
    xi0, eta0 : source's centre projected on the detector coords
    """

    csl = so3g.proj.CelestialSightLine.az_el(ctime, az, el, weather="typical")
    q_bore = csl.Q

    # Signal strength criterion - a chance to flag detectors
    peaks, _ = scipy.signal.find_peaks(
        data, height=threshold, distance=distance)
    if not list(peaks):
        return

    src_idx = np.where(data == np.max(data[peaks]))[0][0]

    # planet position
    d1_unix = ctime[src_idx]
    planet = planets.SlowSource.for_named_source(sso_name, d1_unix * 1.)
    ra0, dec0 = planet.pos(d1_unix)
    q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)

    # find max point in term of the source's coord system if no prior
    # positions arrer given
    if prior_pos is not None:
        xi0, eta0 = prior_pos
    else:
        q_det = coords.quat.quat(*np.array(~q_bore * q_obj)[src_idx])
        xi0, eta0, gamma0 = quat.decompose_xieta(q_det)

    q_total = ~q_bore * q_obj
    xi, eta, gamma = quat.decompose_xieta(q_total)

    return np.array([xi, eta]), np.array([xi0, eta0])


def define_fit_params(config_file_path, band, tele, tube, sso_name):
    """
    Define initial position and allowed offsets for detector centroids,
    beam parameters, and time constant as read from configuration file
    located at 'config_file_path'.

    Args
    ------
    tele, tube : telescope name and tube name
    sso_name: source name

    Return
    ------
    initial_guess, bounds: arguments for the least square fits
                                    and beam size in radians.
    tau: time constant value set from the configuration file.
    """
    with open(config_file_path, "r") as stream:
        config_file = yaml.safe_load(stream)
    tube_idx = config_file['telescopes'][tele]['tube'].index(tube)

    # Pointing - initial guess at zero
    offset_bound = config_file['telescopes'][tele]['pointing_error'][tube_idx]
    # Beam
    beamsize = np.radians(
        config_file['telescopes'][tele]['beamsize'][tube_idx] / 60)
    fwhm_bound_frac = config_file['telescopes'][tele]['beam_error'][tube_idx]
    # Amplitude
    amp = config_file['telescopes'][tele]['detector_response'][sso_name][tube_idx]
    amp_error = config_file['telescopes'][tele]['amp_error'][tube_idx]
    # Beam angle
    theta_min, theta_init, theta_max = np.radians(
        np.array(config_file['telescopes'][tele]['theta']))
    # Time constant
    tau = config_file['telescopes'][tele]['tau']

    initial_guess = [amp, 0, 0, beamsize, beamsize, theta_init]
    errors = [amp_error, offset_bound, offset_bound, fwhm_bound_frac,
              fwhm_bound_frac, theta_min, theta_max]

    return initial_guess, errors, tau


def set_bounds(initial_guess, errors):
    """Synthesize the bounds from the initial guess
       and allowed errors"""
    amp, xi0, eta0, fwhm_xi, fwhm_eta, theta_init = initial_guess
    amp_error, offset_bound, offset_bound, fwhm_bound_frac, fwhm_bound_frac, theta_min, theta_max = errors

    fwhm_min_xi = fwhm_xi - (fwhm_bound_frac * fwhm_xi)
    fwhm_max_xi = fwhm_xi + (fwhm_bound_frac * fwhm_xi)
    fwhm_min_eta = fwhm_eta - (fwhm_bound_frac * fwhm_eta)
    fwhm_max_eta = fwhm_eta + (fwhm_bound_frac * fwhm_eta)

    bounds_min = (amp - (amp_error * amp), -offset_bound, -
                  offset_bound, fwhm_min_xi, fwhm_min_eta, theta_min)
    bounds_max = (amp + (amp_error * amp), offset_bound,
                  offset_bound, fwhm_max_xi, fwhm_max_eta, theta_max)
    bounds = np.array((bounds_min, bounds_max,))

    return bounds


def gaussian2d(xi, eta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi):
    """Simulate a time stream with an Gaussian beam model
    Args
    ------
    xi, eta: cordinates in the detector's system
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated)
        and the rotation angle (in radians)

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data


def tconst_convolve(ctime, data, tau):
    """Convolve the time stream with a given time constant
    Args
    ------
    ctime: 1d array of float
        containting the ctime of each data point
    data: 1d array of float
        time stream data for a given detector
    tau: float
        time constant to be added to the time stream (in seconds)

    Return
    ------
    data_tau: 1d array of float
        time-constant filtered time stream
    """
    k = int(np.ceil(np.log(len(data)) / np.log(2)))
    n = 2 ** k
    delta_time = np.mean(np.diff(ctime))
    freqs = np.fft.fftfreq(n, delta_time)
    data_pad = np.zeros(n)
    data_pad[: len(data)] = data
    fft_data = np.fft.fft(data_pad)
    fft_data *= 1.0 / (1.0 + 2.0j * np.pi * tau * freqs)
    data_tau = np.fft.ifft(fft_data).real[: len(data)]
    return data_tau


def tod_sim(xietat, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, idx_main, tau):
    """ Simulate the TOD of a single detector
    Args
    ----
    xietat: stacked array of xi, eta, ctime
    a, xi0, eta0, fwhm_xi, fwhm_eta, phi: all floats
        parameters for the beam model, as elaborated in the gaussian2d function
    idx_main: range of integers serving as data indices to select a region around
              the peak of radius = 2*FWHM (reduce data volume - speed the fitting).
    tau: float
        time constant
    """
    xi, eta, ctime = xietat
    data_sim = gaussian2d(
        xi, eta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi
    )

    if tau == 0.0:
        return data_sim[idx_main]
    data_sim_tau = tconst_convolve(ctime, data_sim, tau)
    return data_sim_tau[idx_main]


def fit_params(
        readout_id,
        data,
        ctime,
        az,
        el,
        band,
        sso_name,
        threshold_src,
        distance,
        init_params,
        fit_pointing=False,
        fit_beam=False,
        prior_pointing=None,
        representative_dets='no_detectors',
        **kwargs):
    """Function that fits individual time-streams and returns the readout id,
       peak amplitude, detector centroids, beam size in both directions and
       corresponding angle along with the estimated signal-to-noise ratio as
       an array

    Args
    ----
    readout_id: str
        Detector id string.
    data: array of float
        Vector of time-ordered data for this detector.
    ctime: array of float
        timestamps
    az: array of float
        azimuth of boresight
    el: array of float
        elevation of boresight
    sso_name: string
        Name of the celestial source
    init_params: arrays of floats; the output of 'define_fit_params'
    representative_dets: readout ids of detectors whose raw and fitted
                         timestreams should be plotted to evaluate
                         the fitting quality.
    """
    failed_params = np.array([readout_id, *np.full(7, np.nan)])

    data -= np.mean(data)

    if prior_pointing is not None and readout_id in prior_pointing.keys():
        prior_pos = [
            prior_pointing[readout_id]['xi0'],
            prior_pointing[readout_id]['eta0']]
    else:
        prior_pos = None

    coord_transforms = get_xieta_src_centered(
        ctime, az, el, data, sso_name, threshold_src, distance, prior_pos=prior_pos)

    if coord_transforms is None:
        return failed_params

    [xi_det_center, eta_det_center], [xi0, eta0] = coord_transforms
    # centre around detector - no angle information needed
    xi_det_center -= xi0
    eta_det_center -= eta0

    p0, errors, tau = init_params

    if sso_name == 'Moon':
        max_peak = np.nanmax(data) / 2
        sorted_idx = np.argsort(np.abs(data - max_peak))[:1000]

        xi_min = np.nanmin(xi_det_center[sorted_idx])
        xi_max = np.nanmax(xi_det_center[sorted_idx])

        eta_min = np.nanmin(eta_det_center[sorted_idx])
        eta_max = np.nanmax(eta_det_center[sorted_idx])

        fwhm_xi = xi_max - xi_min
        fwhm_eta = eta_max - eta_min

        p0[3:5] = fwhm_xi, fwhm_eta

        # Set the pointing error (optionally) as the difference
        # between two adjacent peaks / detector hits on the source
        # peak_idxs, _ = scipy.signal.find_peaks(data, distance=100)
        # peak_idxs_sorted = np.argsort(data[peak_idxs])[::-1][:6]
        # offset_bound_xi = np.abs(np.mean(np.diff(xi_det_center[peak_idxs_sorted])))
        # offset_bound_eta = np.abs(np.mean(np.diff(eta_det_center[peak_idxs_sorted])))

    # Trim the data around the source
    beamsize = np.nanmax([p0[3], p0[4]])
    radius_main = 2 * beamsize
    radius_cut = 2.5 * radius_main

    radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)
    # take a small piece of the data around the peak
    idx_main_in = np.where(radius < radius_main)[0]
    idx_main_out = np.where(radius > radius_main)[0]
    xi_main = xi_det_center[np.array(idx_main_in)]
    eta_main = eta_det_center[idx_main_in]
    data_main = data[idx_main_in]

    xi_eta_ctime = np.vstack((xi_det_center, eta_det_center, ctime))

    # append amplitude of max peak - safer for now
    # until we have insanely strict criteria on the source's amplitude/
    # efficiency
    p0[0] = np.nanmax(data_main)
    bounds = set_bounds(p0, errors)

    if fit_pointing is True:
        #  Fit the pointing
        f = lambda xyt, *pointing: tod_sim(
            xyt, pointing[0], pointing[1], pointing[2], p0[3], p0[4], p0[5], idx_main_in, tau)
        try:
            popt_pointing, _ = curve_fit(
                f, xi_eta_ctime, data_main, p0=p0[:3], bounds=bounds[:, :3])
            # logger.warning(f"fitted pointing {} {}".format(p0,bounds))
        except BaseException:
            return failed_params

        # Re-centering the data after fitting for the pointing
        p0[0] = popt_pointing[0]
        xi_det_center += popt_pointing[1]
        eta_det_center += popt_pointing[2]
        xi_eta_ctime = np.vstack((xi_det_center, eta_det_center, ctime))

        radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)
        idx_main_in = np.where(radius < radius_main)[0]
        idx_main_out = np.where(radius > radius_main)[0]
        xi_main = xi_det_center[idx_main_in]
        eta_main = eta_det_center[idx_main_in]
        data_main = data[idx_main_in]

        amp = popt_pointing[0]
        # The detector centroids are the initial positions we determined from the
        # peak amplitude plus the fitted offsets
        xi0_fitted = xi0 + popt_pointing[1]
        eta0_fitted = eta0 + popt_pointing[2]

    if fit_beam is True:
        # Fit the beam parameters
        f = lambda xyt, * \
            beam: tod_sim(xyt, p0[0], p0[1], p0[2], beam[0], beam[1], beam[2],
                          idx_main_in, tau)
        try:
            popt_beam, _ = curve_fit(
                f, xi_eta_ctime, data_main, p0=p0[3:], bounds=bounds[:, 3:])
            # logger.warning(f"fitted beam {} {}".format(p0,bounds))
        except BaseException:
            return failed_params

        fwhm_xi, fwhm_eta, phi = popt_beam

    # Estimate the noise as the sigma of the data outside a region of
    # radius>radius_cut
    noise = np.nanstd(data[np.where(radius > radius_cut)[0]])
    snr = amp / noise

    # If pointing / beam were not fitted return the initial guess
    # for this parameters
    if fit_pointing is False:
        amp, xi0_fitted, eta0_fitted = p0[0], xi0, eta0

    if fit_beam is False:
        fwhm_xi, fwhm_eta, phi = p0[3:6]

    # Plot the raw vs fitted tod for some representative detectors
    if readout_id in representative_dets:
        xi_det_center += popt_pointing[1]
        eta_det_center += popt_pointing[2]
        radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)
        idx_main_in = np.where(radius < radius_main)[0]
        xi_main = xi_det_center[idx_main_in]
        eta_main = eta_det_center[idx_main_in]
        data_main = data[idx_main_in]

        data_sim_main = gaussian2d(
            xi_main,
            eta_main,
            amp,
            0,
            0,
            fwhm_xi,
            fwhm_eta,
            phi)

        plot_tods(
            readout_id,
            xi_main,
            eta_main,
            data_main,
            data_sim_main,
            **kwargs)

    return np.array([readout_id, amp, xi0_fitted,
                    eta0_fitted, fwhm_xi, fwhm_eta, phi, snr])


def get_hw_positions(tele, band, tube=None):
    """ Get the hardware xi, eta, gamma positions and detector names
    that belong to a specific tube."""

    from sotodlib.sim_hardware import sim_nominal, sim_detectors_toast

    # Tube can be defined here for the SAT but should be provided for LAT
    tube_sat = {
        'f030': 'SAT4',
        'f040': 'SAT4',
        'f090': 'SAT1',
        'f150': 'SAT1',
        'f230': 'SAT3',
        'f290': 'SAT3',
    }

    if tube is None:
        tube = tube_sat['f' + str(band).zfill(3)]

    hw = sim_nominal()
    sim_detectors_toast(hw, tube)

    qdr, det_names_hw = [], []
    for names in hw.data['detectors'].keys():
        if band in names:
            det_names_hw.append(names)
            qdr.append([hw.data['detectors'][names]['quat'][3]] +
                       list(hw.data['detectors'][names]['quat'][:3]))

    quat_det = so3g.proj.quat.G3VectorQuat(np.array(qdr))
    xi_hw, eta_hw, gamma_hw = so3g.proj.quat.decompose_xieta(quat_det)

    return xi_hw, eta_hw, gamma_hw, np.array(det_names_hw)


def plot_planet_footprints(tod, sso_name, tele, tube, band, hw_pos, **kwargs):
    """ Plot the source's footprint on the focal plane """

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    xi_hw, eta_hw, _, dets_hw = hw_pos
    ax.plot(np.degrees(xi_hw), np.degrees(eta_hw), '.')

    csl = so3g.proj.CelestialSightLine.az_el(
        tod.timestamps, tod.boresight.az, tod.boresight.el, weather="typical")
    q_bore = csl.Q
    planet = planets.SlowSource.for_named_source(
        sso_name, tod.timestamps[0] * 1.)
    ra0, dec0 = planet.pos(tod.timestamps[0])
    planet_q = so3g.proj.quat.rotation_lonlat(ra0, dec0)
    q_total = ~q_bore * planet_q
    xi_total, eta_total, _ = quat.decompose_xieta(q_total)

    ax.plot(np.degrees(xi_total), np.degrees(
        eta_total), color='gray', alpha=.7)
    ax.set_xlabel('xi [deg]')
    ax.set_ylabel('eta [deg]')

    plt.savefig(opj(kwargs['img_dir'], 'source_footprint_' +
                kwargs['obs_id'] + '.png'), bbox_inches='tight')
    plt.close()


def make_fpu_plots(df, hw_pos,
                   input_param=None, fitted_par='beam', **kwargs):
    """ Make focal plane plots where the detectors are color-coded according
        to their (fitted pointing bias or) fitted beam size bias with respect
        to the input"""

    xi_hw, eta_hw, _, dets_hw = hw_pos
    df_det_idxs = [np.where(dets_hw == df['dets:readout_id'][i])[0][0]
                   for i in range(len(df['dets:readout_id']))]
    delta_xis = np.full(len(xi_hw), np.nan)
    delta_etas = np.full(len(eta_hw), np.nan)

    # if fitted_par == 'pointing':
    #     par1, par2 = 'xi0', 'eta0'
    #     x_hw_fitted, y_hw_fitted = xi_hw[df_det_idxs], eta_hw[df_det_idxs]

    # elif fitted_par == 'beam':
    par1, par2 = 'fwhm_xi', 'fwhm_eta'
    beamsize = input_param
    x_hw_fitted, y_hw_fitted = beamsize, beamsize

    x_df_fitted = df[par1].to_numpy().astype(float)
    y_df_fitted = df[par2].to_numpy().astype(float)

    delta_xis[df_det_idxs] = x_df_fitted - x_hw_fitted
    delta_etas[df_det_idxs] = y_df_fitted - y_hw_fitted

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
    # We can make plotting options user specified
    v1_lim = np.max(np.abs(np.array(
        [np.nanmin(np.degrees(delta_xis)), np.nanmax(np.degrees(delta_xis))])))
    v2_lim = np.max(np.abs(np.array(
        [np.nanmin(np.degrees(delta_etas)), np.nanmax(np.degrees(delta_etas))])))

    im1 = ax1.scatter(np.degrees(xi_hw), np.degrees(eta_hw), c=np.degrees(
        delta_xis), vmin=-v1_lim, vmax=v1_lim, cmap='seismic')
    cbar = plt.colorbar(im1)
    cbar.set_label('Bias in ' + par1 + '[degrees]',
                   labelpad=25, rotation=270, size=13)
    ax1.set_xlabel('ξ[degrees]', size=13)
    ax1.set_ylabel('η[degrees]', size=13)

    im2 = ax2.scatter(np.degrees(xi_hw), np.degrees(eta_hw), c=np.degrees(
        delta_etas), vmin=-v2_lim, vmax=v2_lim, cmap='seismic')
    cbar = plt.colorbar(im2)
    cbar.set_label('Bias in ' + par2 + '[degrees]',
                   labelpad=25, rotation=270, size=13)
    ax2.set_xlabel('ξ[degrees]', size=13)
    ax2.set_ylabel('η[degrees]', size=13)

    fig.subplots_adjust(wspace=0.5)
    plt.savefig(opj(kwargs['img_dir'], fitted_par + '_bias_' +
                kwargs['obs_id'] + '.png'), bbox_inches='tight')
    plt.close()


def plot_tods(readout_id, xi_main, eta_main, data_main, data_sim_main,
              **kwargs):
    """
    Plot raw vs fitted timestream in as data=f(xi,eta) and
    show the detector's position on the fpu provided a set
    of detector names"""

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4), dpi=300)
    ax1.plot(xi_main, data_main, lw=2, label='Raw data')
    ax1.plot(xi_main, data_sim_main, '--', lw=1)
    ax1.set_xlabel(r"$\xi$[deg]")
    ax1.set_ylabel('T(K)')
    ax1.legend(loc='upper left', frameon=False)
    ax2.plot(eta_main, data_main, lw=2)
    ax2.plot(eta_main, data_sim_main, '--', lw=1, label='Fitted data')
    ax2.set_xlabel(r"$\eta$[deg]")
    ax2.set_ylabel('T(K)')
    ax2.legend(loc='upper left', frameon=False)
    # ax3.scatter(xi_hw, eta_hw, c='gray')
    # ax3.scatter(xi_hw[det_idx], eta_hw[det_idx], c='r')
    # ax3.set_xlabel(r"$\xi$[rad]")
    # ax3.set_ylabel(r"$\eta$[rad]")
    fig.subplots_adjust(wspace=0.31)
    plt.savefig(opj(kwargs['img_dir'], 'tod_det_' +
                    readout_id +
                    '_obsid_' +
                    kwargs['obs_id'] +
                    '.png'), bbox_inches='tight')
    plt.close()


@dataclass
class QuickFitResult:
    """
    results for pointing_quickfit function

    Args
    ----
    xi: float
        fitted detector xi [rad] relative to boresight center
    eta: float
        fitted detector eta [rad] relative to boresight center
    amp: float
        Amplitude used in the fit. This will be in the same units as
        ``am.signal``, which is generally detector phase.
    fwhm: float
        gaussian FHWM [rad] used in the fit.
    fit_aman: AxisManager
        AxisManager containing xi and eta of source relative to boresight center,
        fitted signal, and model.
    """
    xi: float
    eta: float
    amp: float
    fwhm: float
    fit_aman: core.AxisManager

def pointing_quickfit(
        am_full, chan_idx, downsample_factor=20, bandpass_range=(0.01, 1.8),
        fwhm=np.deg2rad(0.5), max_rad=None
    ):
    """
    Fits pointing for a bright source such as the moon for a single detector.

    Args
    -----
    am_full: AxisManager
        AxisManager containing unfiltered detector signal. This won't be modified.
    chan_idx: int
        Channel index in the axis manager to fit
    downsample_factor: int
        If set, the signal and model will be downsampled by this factor when fitting.
    bandpass_range: tuple[float, float]
        Low and high cutoff frequencies [Hz, Hz] for bandpass. If either are None, will not apply
        the corresponding highpass/lowpass.
    fwhm: float
        Full width half max [radians] of the beam to use for the model. This should be roughly the beam
        size of the telescope.
    max_rad: float
        Only data where the angular distance [radians] between the src and the estimated det location
        is less than ``max_rad`` will be used in the fit. If None, will use 5 *
        fwhm.

    Returns
    --------
        result: QuickFitResult
            Result object containing fitted xi and eta, along with amp and fwhm
            used in the fit.
    """

    if max_rad is None:
        max_rad = 5 * fwhm

    am = am_full.restrict('dets', [am_full.dets.vals[chan_idx]], in_place=False)

    def filter_tod(am, signal_name='signal'):
        filt_kw = dict(
            detrend='linear', resize='zero_pad', axis_name='samps',
            signal_name=signal_name, time_name='timestamps'
        )
        if bandpass_range[0] is not None:
            highpass = high_pass_sine2(cutoff=bandpass_range[0])
            am[signal_name] = fourier_filter(am, highpass, **filt_kw)
        if bandpass_range[1] is not None:
            lowpass = low_pass_sine2(cutoff=bandpass_range[1])
            am[signal_name] = fourier_filter(am, lowpass, **filt_kw)
        return am

    filter_tod(am)
    sl = slice(None, None, downsample_factor)

    ts = am.timestamps[sl]
    az = am.boresight.az[sl]
    el = am.boresight.el[sl]
    sig = am.signal[0, sl]

    xieta_src_rel_fp, xieta_det_rel_fp_est = get_xieta_src_centered(ts, az, el, sig, 'moon', 5)
    xieta_src_rel_det = xieta_src_rel_fp - xieta_det_rel_fp_est[:, None]

    m = np.sqrt(xieta_src_rel_det[0] ** 2 + xieta_src_rel_det[1] ** 2) < max_rad
    amp = np.ptp(sig[m]) * 3

    model_tod = gaussian2d(
        xieta_src_rel_det[0, m], xieta_src_rel_det[1, m], amp, 0, 0, fwhm, fwhm, 0
    )
    fit_am = core.AxisManager().wrap('timestamps', ts[m], [(0, core.IndexAxis('samps'))])
    fit_am.wrap('signal', np.array([sig[m]]), [(0, deepcopy(am.dets)), (1, 'samps')])
    fit_am.wrap('model', np.array([model_tod]), [(0, 'dets'), (1, 'samps')])

    def fit_func(xi0, eta0):
        fit_am.model[0] = gaussian2d(
            xieta_src_rel_det[0, m], xieta_src_rel_det[1, m], amp, xi0, eta0, fwhm, fwhm, 0
        )
        filter_tod(fit_am, signal_name='model')
        return np.sum((fit_am.signal[0] - fit_am.model[0])**2)

    res = minimize(lambda x: fit_func(*x), [0, 0])

    xi_det_rel_fp = xieta_det_rel_fp_est[0] + res.x[0]
    eta_det_rel_fp = xieta_det_rel_fp_est[1] + res.x[1]

    fit_am.model[0] = gaussian2d(
        xieta_src_rel_det[0, m], xieta_src_rel_det[1, m], amp, res.x[0], res.x[1], fwhm, fwhm, 0
    )
    fit_am.wrap('xi', xieta_src_rel_fp[0, m])
    fit_am.wrap('eta', xieta_src_rel_fp[1, m])

    result = QuickFitResult(
        xi=xi_det_rel_fp,
        eta=eta_det_rel_fp,
        amp=amp,
        fwhm=fwhm,
        fit_aman=fit_am
    )
    return result

def plot_quickfit_res(result: QuickFitResult, eta_scale=50):
    """
    Makes cool pseudo-3d source plot
    """
    fig, ax = plt.subplots()
    fit = result.fit_aman
    ax.plot(fit.xi, fit.signal[0] + fit.eta* eta_scale, label='Signal')
    ax.plot(fit.xi, fit.model[0] + fit.eta * eta_scale, label='Model', alpha=0.8)
    ax.legend()
    ax.set_xlabel("Xi (rad, relative to boresight)")
    ax.set_ylabel("Amplitude + scale * eta")
    return fig, ax


def quickfit_and_save_pointing(output_path, am, chan_idxs, show_pb=False, h5_address='fit_results'):
    """
    Run pointing_quickfit for a list of dets and save results to a ResultSet

    Args
    -----
    output_path: str
        Path to h5 file to save results
    am: AxisManager
        Axis manager with unfiltered TODs
    chan_idxs: list
        List of channel indexes of the axis manager to fit pointing
    show_pb: bool
        If True, enable tqdm progress bar
    h5_address: str
        Address in the h5 to save result set to.
    """
    dtype = [
        ('readout_id', '<U40'),
        ('xi', 'f8'),
        ('eta', 'f8'),
    ]
    res_arr = np.array([], dtype=dtype)
    rset= core.metadata.ResultSet.from_friend(res_arr)
    for i in tqdm(chan_idxs, disable=(not show_pb)):
        try:
            result = pointing_quickfit(am, i)
        except Exception:
            continue
        rset.append(
            {'readout_id': am.det_info.readout_id[i], 'xi': result.xi, 'eta': result.eta}
        )
        write_dataset(rset, output_path, h5_address, overwrite=True)
    return rset


def main(
    ctx_file,
    obs_id,
    sso_name,
    band,
    tele,
    tube,
    ufm,
    max_samps,
    scan_rate,
    config_file_path,
    outdir,
    highpass,
    cutoff_high,
    lowpass,
    cutoff_low,
    threshold_src,
    distance,
    do_abs_cal,
    fit_pointing,
    fit_beam,
    pointing_dict=None,
    representative_dets='no_detectors',
    test_mode=False,
    plot_results=False,
):
    """
    Initiate an MPI environment and fit a simulation in the time domain
    """
    # Two local imports, to avoid docs depenency.
    import pandas as pd

    kwargs = dict()
    kwargs['obs_id'] = obs_id

    img_dir = opj(outdir, 'plots')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    kwargs['img_dir'] = img_dir

    t1 = time.time()

    # Get the initial parameters
    init_params = define_fit_params(
        config_file_path, band, tele, tube, sso_name)

    # Load prior on pointing if provided
    if pointing_dict is not None:
        prior_pointing = pickle.load(open(pointing_dict, 'rb'))
    else:
        prior_pointing = None

    # Set context, ufm, chunk size
    ctx = core.Context(ctx_file)
    obs = ctx.obsdb.query()

    stream_id = f'ufm_{ufm}'
    meta = ctx.get_meta(obs_id)

    if 'stream_id' in meta.det_info:
        meta.restrict(
            'dets',
            meta.dets.vals[meta.det_info.stream_id == stream_id]
        )

    if 'n_samps' in meta.obs_info:
        n_samps = meta.obs_info.n_samples
    else:
        d1 = datetime.fromtimestamp(meta.obs_info.start_time)
        d2 = datetime.fromtimestamp(meta.obs_info.stop_time)
        # If we are looking at simulation we need info on the scan
        # rate to estimate total number of samples from metadata
        n_samps = (d2 - d1).seconds * scan_rate

    n_chunks = n_samps // max_samps

    for i in range(n_chunks - 1):
        start = max_samps * i
        stop = max_samps * (i + 1)
        if i == max_samps - 1:
            stop = n_samps
        tod = ctx.get_obs(meta, samples=[start, stop], no_signal=False)

        logger.warning(f'Loading {obs_id} of {max_samps} samples for {ufm}')

        ctime, az, el = tod.timestamps, tod.boresight.az, tod.boresight.el

        # Filter
        if highpass and cutoff_high is not None:
            tod.signal = fourier_filter(
                tod,
                filt_function=high_pass_sine2(
                    cutoff=cutoff_high),
                detrend='linear',
                resize='zero_pad',
                axis_name='samps',
                signal_name='signal',
                time_name='timestamps')

        if lowpass and cutoff_low is not None:
            tod.signal = fourier_filter(
                tod,
                filt_function=low_pass_sine2(
                    cutoff=cutoff_low),
                detrend='linear',
                resize='zero_pad',
                axis_name='samps',
                signal_name='signal',
                time_name='timestamps')

        tod.signal[tod.signal < 0] = 0

        # Choose one band and calibrate to pW
        if 'det_match' in tod and 'det_cal' in tod:
            tod.restrict('dets',
                         tod.dets.vals[tod.det_match.det_bandpass == band])
            tod.signal = np.multiply(tod.signal.T, tod.det_cal.phase_to_pW).T

        # Find detectors with source
        dets_w_src = find_source(
            tod,
            sso_name=sso_name,
            height=threshold_src,
            distance=distance)
        rd_ids = tod.dets.vals[dets_w_src]
        logger.warning(f'got good dets')

        if len(dets_w_src) == 0:
            continue

        # Initiate a dataframe
        full_df = pd.DataFrame(
            columns=[
                "dets:readout_id",
                "amp",
                "xi0",
                "eta0",
                "fwhm_xi",
                "fwhm_eta",
                "phi",
                "snr",
            ], index=rd_ids,)

        count = 0
        for _i, rd_idx in enumerate(dets_w_src):
            logger.warning(f'Starting to fit the detectors')
            rd_id = rd_ids[_i]
            params = fit_params(rd_id, tod.signal[rd_idx, :],
                                ctime, az, el, band,
                                sso_name, threshold_src, distance, init_params,
                                fit_pointing, fit_beam, prior_pointing,
                                representative_dets, **kwargs)
            snr = float(params[-1])
            logger.info(f'Solved {rd_idx:<5d} "{rd_id}" with S/N={snr:.2f}')
            full_df.loc[_i, :] = np.array(params)
            count += 1
            if test_mode and count >= 2:
                break

        new_dtypes = {
            "dets:readout_id": str,
            "amp": np.float64,
            "xi0": np.float64,
            "eta0": np.float64,
            "fwhm_xi": np.float64,
            "fwhm_eta": np.float64,
            "phi": np.float64,
            "snr": np.float64,
        }

        full_df = full_df.astype(new_dtypes)

        # Store dataframe
        full_df.to_hdf(opj(outdir, 'parameter_fits_' + obs_id + '.h5'),
                       key='full_df', mode='w')

        # calculating relative and absolute calibration
        amp = full_df.amp.values
        full_df["rel_cal"] = amp / np.mean(amp)

        if do_abs_cal:
            beam_file = "/mnt/so1/shared/site-pipeline/bcp/%s_%s_beam.h5" % (
                tele, band)
            sso_obj = sim_sso.SimSSO(beam_file=beam_file, sso_name=sso_name)
            freq_arr_GHz, temp_arr = sso_obj._get_sso_temperature(sso_name)
            dt_obs = datetime.fromtimestamp(np.average(tod.timestamps))
            sso_ephem = getattr(ephem, sso_name)()
            sso_ephem.compute(dt_obs.strftime('%Y-%m-%d'))
            ttemp = np.interp(float(band[1:]) * u.GHz, freq_arr_GHz, temp_arr)
            beam, _ = sso_obj._get_beam_map(
                None, sso_ephem.size * u.arcsec, ttemp)
            amp_ref = beam(0, 0)[0][0]
            full_df["abs_cal"] = amp / amp_ref

        # Assert tables dependency is properly satisfied
        out_folder = opj(outdir, tele)
        if not os.path.exists(out_folder):
            # Create a new directory because it does not exist
            os.makedirs(out_folder)
        result_arr = full_df.to_records(index=False)
        result_rs = metadata.ResultSet.from_friend(result_arr)
        print(opj(out_folder, 'cal_obs_%s.h5' % tele))
        write_dataset(result_rs, opj(out_folder, 'cal_obs_%s.h5' %
                      tele), obs_id, overwrite=True)

        t2 = time.time()
        print("Time to run fittings for %s is %.2f seconds." %
              (obs_id, t2 - t1))

        # Plot planet footprint, bias across focal plane
        if plot_results:
            hw_pos = get_hw_positions(tele, band)
            plot_planet_footprints(
                tod, sso_name, tele, tube, band, hw_pos, **kwargs)
            # make_fpu_plots(full_df, tele, band,
            #                input_param=None, fitted_par='pointing', **kwargs)
            make_fpu_plots(
                full_df,
                hw_pos=hw_pos,
                input_param=init_params[2],
                fitted_par='beam',
                **kwargs)


def get_parser(parser=None):
    if parser is None:
        parser = ap.ArgumentParser(
            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--ctx_file",
        action="store",
        dest="ctx_file",
        help="The location of the context file.",
        type=str,
    )

    parser.add_argument(
        "--obs_id",
        action="store",
        dest="obs_id",
        required=True,
        help="Observation id in the context file.",
        type=str,
    )

    parser.add_argument(
        "--sso_name",
        action="store",
        dest="sso_name",
        required=True,
        help="Source name.",
        type=str,
    )

    parser.add_argument(
        "--band",
        action="store",
        dest="band",
        required=True,
        help="Frequency band.",
        type=str,
    )

    parser.add_argument(
        "--tele",
        action="store",
        dest="tele",
        required=True,
        help="Telescope name (SAT/LAT).",
        type=str,
    )

    parser.add_argument(
        "--tube",
        action="store",
        dest="tube",
        required=True,
        help="Telescope tube.",
        type=str,
    )

    parser.add_argument(
        "--ufm",
        action="store",
        dest="ufm",
        required=True,
        help="Telescope ufm.",
        type=str,
    )

    parser.add_argument(
        "--max_samps",
        action="store",
        dest="max_samps",
        required=True,
        help="Number of maximum samples to load",
        type=int,
    )

    parser.add_argument(
        "--scan_rate",
        action="store",
        dest="scan_rate",
        required=False,
        default=200,
        help="Scan rate, required for simulations",
        type=int,
    )

    parser.add_argument(
        "--config_file_path",
        action="store",
        dest="config_file_path",
        required=True,
        help="Location of configuration file that contains beam size \
              and fitting parameters.",
        type=str,
    )

    parser.add_argument(
        "--outdir",
        action="store",
        dest="outdir",
        help="The location for the .h5 output files to be stored.",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run analysis on a subset of detectors, to quickly check \
                for problems.")

    parser.add_argument(
        "--highpass",
        action="store_true",
        dest="highpass",
        help="If True, use highpass sine filter.",
    )

    parser.add_argument(
        "--cutoff_high",
        action="store",
        dest="cutoff_high",
        default=None,
        help="The cutoff frequency to be used in the filtering.",
        type=float,
    )

    parser.add_argument(
        "--lowpass",
        action="store_true",
        dest="lowpass",
        help="If True, use lowpass sine filter.",
    )

    parser.add_argument(
        "--cutoff_low",
        action="store",
        dest="cutoff_low",
        default=None,
        help="The cutoff frequency to be used in the filtering.",
        type=float,
    )

    parser.add_argument(
        "--threshold_src",
        action="store",
        dest="threshold_src",
        default=10,
        help="The max amplitude required for the peak finding",
        type=float,
    )

    parser.add_argument(
        "--distance",
        action="store",
        dest="distance",
        default=50000,
        help="The max amplitude required for the peak finding",
        type=float,
    )

    parser.add_argument(
        "--do_abs_cal",
        action="store_true",
        dest="do_abs_cal",
        default=False,
        help="Do absolute calibration fit.",
    )

    parser.add_argument(
        "--fit_pointing",
        action="store_true",
        dest="fit_pointing",
        default=False,
        help="Fit the pointing parameters.",
    )

    parser.add_argument(
        "--fit_beam",
        action="store_true",
        dest="fit_beam",
        default=False,
        help="Fit the beam parameters.",
    )

    parser.add_argument(
        "--pointing_dict",
        action="store",
        dest="pointing_dict",
        required=False,
        type=str,
        help="Path to a pickle file of a pointing dictionary.",
    )

    parser.add_argument(
        "--representative_dets",
        action="store",
        dest="representative_dets",
        default='no_detectors',
        nargs='+',
        help="Representative detectors across the focal plane whose \
              raw and fitted data should be plotted, given as a list of \
              readout ids.",
    )

    parser.add_argument(
        "--plot_results",
        action="store_true",
        dest="plot_results",
        default=False,
        help="Make plots of planet footprint and fitted results.",
    )

    return parser


if __name__ == "__main__":
    util.main_launcher(main, get_parser)

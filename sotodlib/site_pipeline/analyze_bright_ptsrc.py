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

from astropy import units as u
import numpy as np
import scipy.signal
import time
import ephem
import os
from scipy.optimize import curve_fit
from datetime import datetime
import argparse as ap
import yaml
from sotodlib.site_pipeline import util
import matplotlib
matplotlib.use('agg')

opj = os.path.join

logger = util.init_logger(__name__)


def get_xieta_src_centered(ctime, az, el, data, sso_name, threshold):
    """Create a planet centered coordinate system for single detector data

    Args
    ------
    ctime, az, el: unix time and boresight position coordinates
    data: single detector data
    sso_name: celestial source to insert to pyephem
    threshold: minimal peak amplitude required to assume the
               timestream includes the source of interest.
               The amplitude is given as number of times the
               standard deviation of the data.

    Return
    ------
    xi_total, eta_total : source's centre projected on the telescope
    xi0, eta0 : source's centre projected on the detector
    """

    csl = so3g.proj.CelestialSightLine.az_el(ctime, az, el, weather="typical")
    q_bore = csl.Q

    # Signal strength criterion - a chance to flag detectors
    peaks, _ = scipy.signal.find_peaks(data, height=threshold * np.std(data))
    if not list(peaks):
        return

    src_idx = np.where(data == np.max(data[peaks]))[0][0]

    # planet position
    d1_unix = ctime[src_idx]
    planet = planets.SlowSource.for_named_source(sso_name, d1_unix * 1.)
    ra0, dec0 = planet.pos(d1_unix)
    q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)

    # find peak amplitude from the detector timestream
    q_det = coords.quat.quat(*np.array(~q_bore * q_obj)[src_idx])
    q_total = ~q_bore * q_obj

    xi, eta, _ = quat.decompose_xieta(q_total)
    xi0, eta0, _ = quat.decompose_xieta(q_det)

    return np.array([xi, eta]), np.array([xi0, eta0])


def define_fit_params(config_file_path, tele, tube, sso_name):
    """
    Define initial position and allowed offsets for detector centroids,
    beam parameters and time constant as read from configuration file
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
    beamsize = config_file['telescopes'][tele]['beamsize'][tube_idx]
    # Provided in terms of the beam size
    offset_bound = config_file['telescopes'][tele]['pointing_error'][tube_idx]
    fwhm_bound_frac = config_file['telescopes'][tele]['beam_error'][tube_idx]
    # Planet expected temperature measured by the detector
    amp = config_file['telescopes'][tele]['detector_response'][sso_name][tube_idx]
    amp_error = config_file['telescopes'][tele]['amp_error'][tube_idx]

    theta_min, theta_init, theta_max = np.radians(
        np.array(config_file['telescopes'][tele]['theta']))
    tau = config_file['telescopes'][tele]['tau']
    beamsize = np.radians(beamsize / 60)
    offset_bound *= beamsize

    fwhm_bound = fwhm_bound_frac * beamsize
    fwhm_min = beamsize - fwhm_bound
    fwhm_max = beamsize + fwhm_bound

    initial_guess = [amp, 0, 0, beamsize, beamsize, theta_init]
    bounds_min = (amp - (amp_error * amp), -offset_bound, -
                  offset_bound, fwhm_min, fwhm_min, theta_min)
    bounds_max = (amp + (amp_error * amp), offset_bound, offset_bound,
                  fwhm_max, fwhm_max, theta_max)
    bounds = np.array((bounds_min, bounds_max,))

    return initial_guess, bounds, beamsize, tau


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
              the peak of radius = 2*FWHM.
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
        sso_name,
        threshold_src,
        init_params,
        representative_dets,
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
    coord_transforms = get_xieta_src_centered(
        ctime, az, el, data, sso_name, threshold_src)

    if coord_transforms is None:
        return failed_params

    [xi_det_center, eta_det_center], [xi0, eta0] = coord_transforms
    # centre around detector - no angle information needed
    xi_det_center -= xi0
    eta_det_center -= eta0

    p0, bounds, beamsize, tau = init_params
    radius_main = 2 * beamsize
    radius_cut = 2 * radius_main

    radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)
    # take a small piece of the data around the peak
    idx_main_in = np.where(radius < radius_main)[0]
    idx_main_out = np.where(radius > radius_main)[0]
    xi_main = xi_det_center[np.array(idx_main_in)]
    eta_main = eta_det_center[idx_main_in]
    data_main = data[idx_main_in]

    xi_eta_ctime = np.vstack((xi_det_center, eta_det_center, ctime))

    f = lambda xyt, *pointing: tod_sim(
        xyt, pointing[0], pointing[1], pointing[2], p0[3], p0[4], p0[5], idx_main_in, tau)
    try:
        popt_pointing, _ = curve_fit(
            f, xi_eta_ctime, data_main, p0=p0[:3], bounds=bounds[:, :3])
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

    # Fitting fwhm_xi, fwhm_eta, phi
    f = lambda xyt, * \
        beam: tod_sim(xyt, p0[0], p0[1], p0[2], beam[0], beam[1], beam[2],
                      idx_main_in, tau)
    try:
        popt_beam, _ = curve_fit(
            f, xi_eta_ctime, data_main, p0=p0[3:], bounds=bounds[:, 3:])
    except BaseException:
        return failed_params

    # Updating the initial guess after fitting for beam parameters
    p0[3:] = popt_beam[:]

    # Fitting all six parameters simultaneously
    f = lambda xyt, *all_params: tod_sim(xyt, all_params[0], all_params[1],
                                         all_params[2], all_params[3],
                                         all_params[4], all_params[5],
                                         idx_main_in, tau)
    try:
        popt, _ = curve_fit(f, xi_eta_ctime, data_main, p0=p0, bounds=bounds)
    except BaseException:
        return failed_params

    # Define as noise the sigma of the data outside a region of radius 4*FWHM
    noise = np.nanstd(data[np.where(radius > radius_cut)[0]])
    snr = popt[0] / noise

    # Plot the raw vs fitted tod for some representative detectors
    if readout_id in representative_dets:
        xi_det_center += popt[1]
        eta_det_center += popt[2]
        radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)
        idx_main_in = np.where(radius < radius_main)[0]
        xi_main = xi_det_center[idx_main_in]
        eta_main = eta_det_center[idx_main_in]
        data_main = data[idx_main_in]
        data_sim_main = gaussian2d(
            xi_main,
            eta_main,
            popt[0],
            0,
            0,
            popt[3],
            popt[4],
            popt[5])

        plot_tods(
            readout_id,
            xi_main,
            eta_main,
            data_main,
            data_sim_main,
            **kwargs)

    # The detector centroids are the initial positions we determined from the
    # peak amplitude plus fitted offsets
    popt[1] = xi0 + popt_pointing[1] + popt[1]
    popt[2] = eta0 + popt_pointing[2] + popt[2]

    return np.array([readout_id, *popt, snr])


def get_hw_positions(tele):
    """ Get the hardware xi, eta, gamma positions and detector names
    that belong to a specific tube."""

    from sotodlib.sim_hardware import sim_nominal, sim_detectors_toast

    freq_band = {
        'SAT4': 'f030',
        'SAT1': 'f090',
        'SAT2': 'f090',
        'SAT3': 'f230',
        'LAT_o6': 'f030',
        'LAT_i1': 'f090',
        'LAT_i3': 'f090',
        'LAT_i4': 'f090',
        'LAT_i6': 'f090',
        'LAT_i5': 'f230',
        'LAT_c1': 'f230'}
    hw = sim_nominal()
    sim_detectors_toast(hw, tele)

    qdr, det_names_hw = [], []
    for names in hw.data['detectors'].keys():
        if freq_band[tele] in names:
            det_names_hw.append(names)
            qdr.append([hw.data['detectors'][names]['quat'][3]] +
                       list(hw.data['detectors'][names]['quat'][:3]))

    quat_det = so3g.proj.quat.G3VectorQuat(np.array(qdr))
    xi_hw, eta_hw, gamma_hw = so3g.proj.quat.decompose_xieta(quat_det)

    return xi_hw, eta_hw, gamma_hw, np.array(det_names_hw)


def plot_planet_footprints(tod, sso, tele, tube, **kwargs):
    """ Plot the source's footprint on the focal plane """

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    if 'xi_hw' in kwargs.keys() and 'eta_hw' in kwargs.keys():
        xi_hw, eta_hw = kwargs['xi_hw'], kwargs['eta_hw']
    else:
        xi_hw, eta_hw, _, _ = get_hw_positions(tele)
    ax.plot(np.degrees(xi_hw), np.degrees(eta_hw), '.')

    csl = so3g.proj.CelestialSightLine.az_el(
        tod.timestamps, tod.boresight.az, tod.boresight.el, weather="typical")
    q_bore = csl.Q
    planet = planets.SlowSource.for_named_source(sso, tod.timestamps[0] * 1.)
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


def make_fpu_plots(df, tele, band,
                   input_param=None, fitted_par='pointing', **kwargs):
    """ Make focal plane plots where the detectors are color-coded according
        to their fitted pointing bias or fitted beam size bias with respect
        to the input"""

    if 'xi_hw' in kwargs.keys() and 'eta_hw' in kwargs.keys():
       if 'dets_hw' in kwargs.keys():
           xi_hw, eta_hw, dets_hw = kwargs['xi_hw'], kwargs['eta_hw'], kwargs['dets_hw']
       else:
           xi_hw, eta_hw, _, dets_hw = get_hw_positions(tele)
    df_det_idxs = [np.where(dets_hw == df['dets:readout_id'][i])[0][0]
                   for i in range(len(df['dets:readout_id']))]
    delta_xis = np.full(len(xi_hw), np.nan)
    delta_etas = np.full(len(eta_hw), np.nan)

    if fitted_par == 'pointing':
        par1, par2 = 'xi0', 'eta0'
        x_hw_fitted, y_hw_fitted = xi_hw[df_det_idxs], eta_hw[df_det_idxs]

    elif fitted_par == 'beam':
        par1, par2 = 'fwhm_xi', 'fwhm_eta'
        beamsize = input_param
        x_hw_fitted, y_hw_fitted = beamsize, beamsize

    else:
        logger.warning('No plotting quantity is specified')

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

    xi_hw, eta_hw, dets_hw = kwargs['xi_hw'], kwargs['eta_hw'], kwargs['dets_hw']
    det_idx = np.where(dets_hw == readout_id)[0][0]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(10, 4), width_ratios=[1, 1, 2], dpi=300)
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
    ax3.scatter(xi_hw, eta_hw, c='gray')
    ax3.scatter(xi_hw[det_idx], eta_hw[det_idx], c='r')
    ax3.set_xlabel(r"$\xi$[rad]")
    ax3.set_ylabel(r"$\eta$[rad]")
    fig.subplots_adjust(wspace=0.31)
    plt.savefig(opj(kwargs['img_dir'], 'tod_det_' +
                    readout_id +
                    '_obsid_' +
                    kwargs['obs_id'] +
                    '.png'), bbox_inches='tight')
    plt.close()


def main(
    ctx_file,
    obs_id,
    config_file_path,
    outdir,
    highpass,
    cutoff_high,
    lowpass,
    cutoff_low,
    threshold_src,
    do_abs_cal,
    representative_dets,
    test_mode=False,
    plot_results=False,
):
    """
    Initiate an MPI environment and fit a simulation in the time domain
    """
    # Two local imports, to avoid docs depenency.
    from mpi4py import MPI
    import pandas as pd

    kwargs = dict()
    kwargs['obs_id'] = obs_id

    img_dir = opj(outdir, 'plots')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    kwargs['img_dir'] = img_dir

    t1 = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ntask = comm.Get_size()

    ctx = core.Context(ctx_file)
    obs = ctx.obsdb.query()

    logger.warning(f'Loading {obs_id} in task {rank}')

    obs_idx = np.where(obs['obs_id'] == obs_id)[0][0]
    sso_name = obs[obs_idx]['target']
    tele = obs[obs_idx]['telescope']
    tube = obs[obs_idx]['tube_slot']
    tod = ctx.get_obs(obs_id, no_signal=False)
    rd_ids = tod.dets.vals

    # Get the initial parameters
    init_params = define_fit_params(config_file_path, tele[:3], tube, sso_name)
    ctime, az, el = tod.timestamps, tod.boresight.az, tod.boresight.el

    nrd = len(rd_ids)
    nsample_task = nrd // ntask + 1
    rd_idx_rng = np.arange(rank * nsample_task,
                           min((rank + 1) * nsample_task, nrd))

    df = pd.DataFrame(
        columns=[
            "dets:readout_id",
            "amp",
            "xi0",
            "eta0",
            "fwhm_xi",
            "fwhm_eta",
            "phi",
            "snr",
        ], index=rd_idx_rng,)

    # Load signal for only my dets.
    tod = ctx.get_obs(obs_id, dets=np.array(rd_ids)[rd_idx_rng])

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

    if representative_dets != 'no detectors':
        xi_hw, eta_hw, _, dets_hw = get_hw_positions(tele)
        kwargs['xi_hw'], kwargs['eta_hw'], kwargs['dets_hw'] = xi_hw, eta_hw, dets_hw

    count = 0
    for _i, rd_idx in enumerate(rd_idx_rng):
        rd_id = rd_ids[rd_idx]
        params = fit_params(rd_id, tod.signal[_i, :],
                            ctime, az, el,
                            sso_name, threshold_src, init_params,
                            representative_dets, **kwargs)
        snr = float(params[-1])
        logger.info(f'Solved {rd_idx:<5d} "{rd_id}" with S/N={snr:.2f}')
        df.loc[rd_idx, :] = np.array(params)
        count += 1
        if test_mode and count >= 2:
            break
    all_dfs = comm.gather(df, root=0)

    if rank == 0:
        full_df = pd.concat(all_dfs)
        full_df = full_df.dropna(subset=['snr'])
        full_df = full_df.set_index(full_df["dets:readout_id"])

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
        full_df.to_hdf(opj(outdir, 'parameter_fits.h5'),
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

        if plot_results:
            plot_planet_footprints(tod, sso_name, tele, tube, **kwargs)
            make_fpu_plots(full_df, tele, tube,
                           input_param=None, fitted_par='pointing', **kwargs)
            make_fpu_plots(
                full_df,
                tele,
                tube,
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
        help="The max amplitude required for the peak finding, \
              given as times the standard deviation of the data.",
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
        "--representative_dets",
        action="store",
        dest="representative_dets",
        default='no detectors',
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

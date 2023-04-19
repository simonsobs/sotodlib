#!/usr/bin/env python3

from scipy.interpolate import UnivariateSpline, interp1d
from mpi4py import MPI
from scipy.special import jv
import scipy
import healpy as hp
import warnings
from astropy.table import QTable, vstack
import astropy
import yaml
import pickle
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters
from pixell import enmap
import os
import matplotlib
matplotlib.use('agg')

opj = os.path.join


def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def read_enmaps(map_path, rescale, smooth, pk_normalise=True, **kwargs):
    """Read enmap and rescale/smooth if requested upon calling.
    Peak-normalise by default"""

    map_names = os.listdir(map_path)
    enmaps = []
    for map_name in map_names:
        map_file = opj(map_path, map_name)
        if not os.path.exists(map_file):
            warnings.warn("Map file was not found")
        try:
            pmap = enmap.read_fits(map_file)
        except BaseException:
            raise TypeError('Invalid enmap object')
        array_map, array_ra, array_dec = enmap_to_array(pmap)
        if smooth:
            array_map = np.asarray(smooth_map(pmap, kwargs['sigma']))
        if rescale:
            array_map, array_ra, array_dec = rescale_coords([array_map,
                                                             array_ra,
                                                             array_dec],
                                                            kwargs['scale'])
            pmap.posmap()[0], pmap.posmap()[1] = array_ra, array_dec
        if pk_normalise:
            array_map /= np.nanmax(array_map)
        pmap[0, :, :] = array_map
        enmaps.append(pmap)
    return enmaps


def enmap_to_array(pmap):
    """Convert enmap to np.array object for convenience"""

    pmap_arr = np.asarray(pmap)
    ra, dec = np.asarray(pmap.posmap()[0]), np.asarray(pmap.posmap()[1])
    return pmap_arr, ra, dec


def rescale_coords(p_array, scale):
    """Rescale the map data and map coordinates

    Args
    ----
    pmap_array: array [ndmap, coordinates]
    """

    rescaled_array = [x * scale for x in p_array]
    return rescaled_array


def smooth_map(pmap, sigma=0.5):
    """Smooth ndmap object with a Gaussian kernel of sigma=0.5"""

    return enmap.smooth_gauss(pmap, sigma=sigma)


def trim_single_map(pmap, pixbox=None, skybox=None, **kwargs):
    """Function that trims the input maps to some user-specified
        limits

    Args
    ----
    maps: array of ndmap objects
    pixbox: Tuple of ra and dec limits in terms of number of pixels
            e.g. pixbox=(np.asarray([[ra_pix_min, dec_pix_min],
                                     [ra_pix_max, dec_pix_max]]))
    skybox: Tuple of ra and dec physical limits
            e.g. skybox=np.degrees(np.asarray([[ra0, dec0], [ra1, dec1]]))

    Returns
    -------
    maps_trimmed: The trimmed version of maps

    """

    if 'pixbox' in kwargs:
        pixbox = kwargs['pixbox']
    elif 'skybox' in kwargs:
        skybox = kwargs['skybox']

    if pixbox is not None and skybox is None:
        skybox = enmap.pixbox2skybox(pmap.shape, pmap.wcs, pixbox)

    if skybox is not None:
        submap = enmap.submap(pmap, skybox, mode=None, wrap="auto", iwcs=None)
        return submap

    if skybox is None and pixbox is None:
        raise ValueError('map limits should be given either in physical \
                          or pixel coordinates')


def trim_multiple_maps(pmaps, pixbox=None, skybox=None,
                       make_equal=False, **kwargs):
    """Function that trims the input maps to some user-specified
        limits or adjusts them to have the same size """

    if 'pixbox' in kwargs:
        pixbox = kwargs['pixbox']
    elif 'skybox' in kwargs:
        skybox = kwargs['skybox']
    if 'make_equal' in kwargs:
        make_equal = kwargs['make_equal']

    if pixbox is not None or skybox is not None:
        enmaps_trimmed = [trim_single_map(map_i, pixbox=pixbox, skybox=skybox)
                          for map_i in pmaps]

    if make_equal:
        if np.any([map_i.shape == pmaps[0].shape for map_i in pmaps]) == False:
            min_grid = np.min([np.shape(map_i for map_i in pmaps)])
            lim_ra_pix, lim_dec_pix = int(
                min_grid[0] / 2), int(min_grid[1] / 2)
            pixbox = np.asarray([[-lim_ra_pix, -lim_dec_pix],
                                 [lim_ra_pix, lim_dec_pix]])

            enmaps_trimmed = []
            for map_i in pmaps:
                skybox = enmap.pixbox2skybox(map_i.shape, map_i.wcs, pixbox)
                enmaps_trimmed.append(enmap.submap(map_i, skybox))
    return enmaps_trimmed


def coadd_maps(pmaps):
    """Coadd an array of maps to increase the flux

    Args
    ----
    maps: array of ndmap objects

    Returns
    -------
    The coadded map as an ndmap object
    """

    coadd_map = np.zeros_like((pmaps[0]))
    coadd_ra, coadd_dec = [np.zeros_like((pmaps)) for i in range(2)]

    for t, map_i in enumerate(pmaps):
        coadd_map += map_i
        coords = map_i.posmap()
        coadd_ra[t], coadd_dec[t] = coords[0], coords[1]

    coadded_enmap = enmap.zeros(map_i.shape, map_i.wcs, dtype=np.float64)
    coadded_enmap[:] = coadd_map
    coadded_enmap.posmap()[0], coadded_enmap.posmap()[1] = np.nanmean(
        coadd_ra, axis=0), np.nanmean(coadd_dec, axis=0)
    return coadded_enmap


def correct_background(bins, prof, nsamps):
    """Subtract the average value of the data sample with
       the minimum gradient from the profile

    Args
    ----
    bins, prof: radial bins and beam profile
    nsamps: number of samples from the full data

    Returns
    -------
    The background-subtracted profile
    """

    chunk_idxs = np.linspace(0, len(prof), 10)

    grad = []
    for i in range(len(chunk_idxs[:-1])):
        grad.append(
            np.mean(np.gradient(prof[int(chunk_idxs[i]):
                                     int(chunk_idxs[i + 1])])))
    min_samp = find_nearest(grad, 0)
    offset = np.mean(prof[int(chunk_idxs[min_samp]):
                          int(chunk_idxs[min_samp + 1])])
    return prof - offset


def calculate_snr_single(pmap, r_noise, coords=None):
    """Calculate the the Signal-to-Noise Ratio of a single
       map. The noise is estimated outside a radius r_noise"""

    if coords is None:
        ra, dec = pmap.posmap()[0], pmap.posmap()[1]
    else:
        ra, dec = coords
    r = np.sqrt(ra**2 + dec**2)
    r = r.reshape(r.shape[-2], r.shape[-1])
    map_temp = pmap.reshape(pmap.shape[-2], pmap.shape[-1])
    neighbours = np.array(map_temp[np.where(r > r_noise)])
    rms = get_rms(neighbours)
    peak_idx = np.unravel_index(
        np.nanargmax(
            map_temp.flatten()),
        np.shape(map_temp))
    snr = map_temp[peak_idx] / rms

    if np.isnan(snr):
        warnings.warn("SNR is nan, try reducing the the noise radius")
    return snr


def calculate_snr(pmaps, r_noise, n_obs=1, n_max_sets=-1):
    """Calculate the Signal-to-Noise Ratio of an array of maps from a
       number of n_max_sets subsets for n_obs stacked observations

    Args
    ----
    maps: ndmap object
    r_noise [radians]: float. The radius outside of which to calculate the RMS
    n_obs: int. The number of maps to use (stack)
    n_max_sets: int. The number of possible combinations of n_obs maps
    Returns
    -------
    [mean(snr), std(snr)]: The mean and standard deviation of the SNR
    """

    import itertools

    if n_obs == len(pmaps):
        cmap, cra, cdec = coadd_maps(pmaps)
        return calculate_snr_single(cmap, r_noise=r_noise, coords=[cra, cdec])

    else:
        map_idxs = list(
            itertools.combinations(
                np.arange(
                    0,
                    len(pmaps) - 1),
                n_obs))

        snr_subset = []
        for t, subset_idxs in enumerate(map_idxs[:n_max_sets]):
            map_subset = [pmaps[idx] for idx in subset_idxs]
            coadd_map, coadd_ra, coadd_dec = coadd_maps(map_subset)
            snr_i = calculate_snr_single(coadd_map, r_noise=r_noise)
            snr_subset.append(snr_i)
        snr_subset = np.array(snr_subset)
    return np.nanmean(snr_subset), np.nanstd(snr_subset)


def gaussian_2d_rot(x, y, x0, y0, sigx, sigy, theta, **kwargs):
    """2D Gaussian function including rotation. Inputs given in radians

    Args
    ----
    x0, y0: float centers of the Gaussian
    sigx, sigy: sigma of the distribution in x,y directions
    theta: orientation angle
    """

    try:
        amp = kwargs['amp']
    except BaseException:
        amp = 1
    sigx2 = sigx**2
    sigy2 = sigy**2
    x_rot = np.cos(theta)**2 / (2 * sigx2) + np.sin(theta)**2 / (2 * sigy2)
    y_rot = np.sin(theta)**2 / (2 * sigx2) + np.cos(theta)**2 / (2 * sigy2)
    xy_rot = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

    d2_rot_gauss = -x_rot * (x - x0)**2 - y_rot * \
        (y - y0)**2 - 2 * xy_rot * (x - x0) * (y - y0)
    return amp * np.exp(d2_rot_gauss)


def get_rms(x):
    """Estimate rms levels after gap-filling in x,y directions"""
    return np.sqrt(np.nanmean(x**2))


def airy_disc(x, y, amp, x0, y0, R, theta):
    """Airy pattern"""

    Rz = 1.2196698912665045
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    z = np.pi * r / (R / Rz)
    j1 = scipy.special.jv(1, z)
    airy = (2 * j1 / z)**2
    return amp * airy


def get_input_params(INITIAL_PARA_FILE, tele, band):
    """Get the initial parameters configuration from a .yaml
    file for telescope 'tele' and frequency band 'band'"""

    with open(INITIAL_PARA_FILE, "r") as stream:
        try:
            config_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    band_idx = config_file['telescopes'][tele]['bands'].index(band)
    beamsize = config_file['telescopes'][tele]['beamsize'][band_idx] 
    band_c = config_file['telescopes'][tele]['band_c'][band_idx]
    d = config_file['telescopes'][tele]['aperture']

    wlength = 3 / (band_c)
    wlength *= 1e-01
    R = 1.22 * wlength / d
    init_params = {'amp': 1,
                   'x0': 1e-05,
                   'y0': 1e-05,
                   'fwhm_x': np.radians(beamsize / 60),
                   'fwhm_y': np.radians(beamsize / 60),
                   'theta': 1e-06,
                   'R': R}
    return init_params


def mask_source(pmap, r_mask):
    """Mask a region around the source or radius r_mask and gap-fill"""

    mask = enmap.zeros(pmap.shape, pmap.wcs, dtype=np.float64)
    d = enmap.distance_from(mask.shape, mask.wcs, [[0], [0]])
    mask += 1. * (d < r_mask)

    return mask * pmap


def get_widx(prof, wing_cutoff):
    """Get the index correspondint to beam core/wing transition"""

    mask_idxs = np.where(10 * np.log10(np.abs(prof)) < wing_cutoff)
    widx = mask_idxs[0][0]

    return widx


def get_res(ra, dec):
    """Get map resolution"""

    c_ra = (ra.max() - ra.min()) / ra.shape[0]
    c_dec = (dec.max() - dec.min()) / dec.shape[1]
    res = np.mean([c_ra, c_dec])
    return res


def make_model_params(dependent_params, b_acc):
    """Make the lmfit parameter object for the fitting function"""

    params = Parameters()
    for idx_key, key in enumerate(dependent_params.keys()):
        key_value = dependent_params[key]
        params.add(key, value=key_value,
                   min=-b_acc * key_value + key_value,
                   max=b_acc * key_value + key_value)
    return params


def get_ref_beam(beamfile, theta_trunc=None, idx_trunc=-1, lmax=None):
    """Get an estimation for the solid angle of the input beam"""

    bdict = pickle.load(open(beamfile, 'rb'), encoding='latin1')
    prof_ref, bins_ref, res = bdict['profiles'], np.radians(
        bdict['ru']), np.radians(bdict['size'][0][1])

    if theta_trunc:
        idx_trunc = find_nearest(bins_ref, theta_trunc)

#     from pixell import curvedsky
#     bl = curvedsky.profile2harm(br, r, lmax=lmax)

    bl_ref = hp.beam2bl(prof_ref, bins_ref, lmax=lmax)

    return bins_ref[:idx_trunc], prof_ref[:idx_trunc], prof_ref[:idx_trunc].sum(
    ) * res, bl_ref


def prof2map(prof, bins, init_map):
    """Create a map from a radially averaged profile

    Args
    ----
    bins, prof: radial bins and beam profile values
    init_map: the ndmap object before the binning
              to obtain the x,y positions on the sky

    Returns
    -------
    An ndmap object
    """

    out_map = enmap.zeros(init_map.shape, init_map.wcs, dtype=np.float64)
    ra, dec = out_map.posmap()
    r = np.sqrt(ra**2 + dec**2)

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):

            idx_nearest = find_nearest(bins, r[i, j])
            out_map[0, i, j] = prof[idx_nearest]

    return out_map


def plot_maps(data=None, fit=None, img_file=None):
    """Plot data, fit and residual maps"""

    fig, axs = plt.subplots(1, 3, dpi=300)

    log_data = 10 * np.log10(np.abs(data[0]))
    log_fit = 10 * np.log10(np.abs(fit[0]))

    lim = int(np.degrees(data.posmap()[0].max()))
    im0 = axs[0].imshow(log_data, vmin=-50, vmax=0)
    axs[0].set_title('Data')

    axs[0].set_xticks(np.linspace(0, log_data.shape[0], 5))
    axs[0].set_xticklabels([-lim, -lim / 2, 0, lim / 2, lim], size=9)
    axs[0].set_yticks(np.linspace(0, log_data.shape[1], 5))
    axs[0].set_yticklabels([-lim, -lim / 2, 0, lim / 2, lim], size=9)
    axs[0].set_xlabel('Azimuth (degrees)', size=10)
    axs[0].set_ylabel('Elevation (degrees)', size=10)

    axs[1].imshow(log_fit, vmin=-50, vmax=0)
    axs[1].set_title('Fit')

    axs[1].set_xticks([])
    axs[1].set_xticklabels([])
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])

    axs[2].imshow(
        10 *
        np.log10(
            np.abs(
                data[0] -
                fit[0])),
        vmin=-
        50,
        vmax=0)
    axs[2].set_title('Residuals')

    axs[2].set_xticks([])
    axs[2].set_xticklabels([])
    axs[2].set_yticks([])
    axs[2].set_yticklabels([])

    cbar_ax = fig.add_axes([0.125, 0.2, 0.775, 0.03])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Amplitude [dB]', size=12)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def plot_profile(bins=None, data_mean=None, data_std=None,
                 ref=None, img_file=None, fsize=12):
    """Plot average profile and 1σ-band"""

    fig = plt.figure(dpi=300)
    axs = fig.add_subplot(111)

    axs.plot(np.degrees(bins), data_mean, label='Mean data')
    axs.fill_between(
        np.degrees(bins),
        data_mean - data_std,
        data_mean + data_std,
        alpha=0.7,
        label='$1\\sigma error$')

    if ref is not None:
        axs.semilogy(np.degrees(bins), ref, 'k--', label='Reference')

    axs.legend()
    axs.set_yscale('log')
    axs.set_xlabel('Radius [degrees]', size=fsize)
    axs.set_ylabel('Beam Power [dB]', size=11)
    axs.set_xlim(0, np.degrees(bins[-1]))
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def plot_bls(data_mean=None, data_std=None,
             ref=None, img_file=None, fontsize=12):
    """Plot average beam transfer function and 1σ-band"""

    fig = plt.figure(dpi=300)
    axs = fig.add_subplot(111)

    ells = np.arange(0, len(data_mean))

    axs.plot(ells, data_mean, label='Mean data', lw=2)
    axs.fill_between(
        ells,
        data_mean -
        data_std,
        data_mean +
        data_std,
        alpha=0.7,
        label='$1\\sigma error$')

    if ref is not None:
        ref /= np.nanmax(ref)
        axs.plot(ref, 'k--', label='Reference')

    axs.legend()
    axs.set_ylabel('$B_{\\ell}$', size=fontsize)
    axs.set_xlim(0, ells[-1])
    axs.set_xlabel('Multipole, $\\ell$', size=fontsize)
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def fit_main_lobe(pmap, res, init_params, n_iter, acc):
    """Fit an Airy pattern to find the beam's first minimum,
       mask the sidelobes and fit a 2D Gaussian

    Args
    ----
    pmap: the ndmap object to be fitted
    res: the size of the radial bins
    init_params: initial guess for the fitted parameters
    n_iter: number of iterations
    acc: desired accuracy of the fit

    Returns
    -------
    Dictionary of the fitted parameters
    """

    prof_pmap, bins_pmap = enmap.rbin(pmap, bsize=res)
    bins_pmap_pos = bins_pmap[np.where(prof_pmap[0] > 0)]
    prof_pmap_pos = prof_pmap[0][np.where(prof_pmap[0] > 0)]
    # Fit for the first dark ring
    array_map, ra, dec = enmap_to_array(pmap)

    amp, x0, y0, R, theta = [init_params[key] for key in ['amp',
                                                          'x0',
                                                          'y0',
                                                          'R',
                                                          'theta']]
    error = np.sqrt(array_map.ravel() + 1)

    # Roughly the dark ring is FWHM/0.8
    fmodel = Model(airy_disc, independent_vars=('x', 'y'))
    params = make_model_params(dependent_params={'amp': amp,
                                                 'x0': x0,
                                                 'y0': y0,
                                                 'R': R,
                                                 'theta': theta},
                               b_acc=2 * acc)
    result_a = fmodel.fit(data=array_map.ravel(),
                          x=ra.ravel(),
                          y=dec.ravel(),
                          params=params,
                          weights=1 / error,
                          max_nfev=n_iter)

    amp, x0, y0, R, theta = result_a.best_values.values()

    # Define a 10% region around the first dark ring
    idx_min, idx_max = find_nearest(
        bins_pmap, R - 0.1 * R), find_nearest(bins_pmap, R + 0.1 * R)
    trunc_prof = prof_pmap[0][idx_min:idx_max]

    # find the minimum of positive values in -- sometimes filtering creates
    # two negative bumps around the peak
    first_zero_touch = idx_min + \
        np.where(
            np.logical_and(
                trunc_prof > 0,
                trunc_prof == np.min(trunc_prof)))[0][0]

    # interpolate over negative points
    sp1d = interp1d(bins_pmap_pos, prof_pmap_pos)
    prof_pmap[0][:idx_max] = sp1d(bins_pmap[:idx_max])

    # Mask after the first minimum to isolate main lobe
    rmask = bins_pmap[first_zero_touch]
    pmap_masked = mask_source(pmap, rmask)
    prof_pmap_masked, bins_pmap_masked = enmap.rbin(pmap_masked, bsize=res)

    # Fit a Gaussian to the masked map
    array_map, ra, dec = enmap_to_array(pmap_masked)

    fact = np.sqrt(8 * np.log(2))
    sigx, sigy = init_params['fwhm_x'] / fact, init_params['fwhm_y'] / fact

    fmodel = Model(gaussian_2d_rot, independent_vars=('x', 'y'))
    params = make_model_params(dependent_params={'x0': x0,
                                                 'y0': y0,
                                                 'sigx': sigx,
                                                 'sigy': sigy,
                                                 'theta': theta},
                               b_acc=acc)

    result_g = fmodel.fit(array_map.ravel(), x=ra.ravel(), y=dec.ravel(),
                          params=params, weights=1 / error, max_nfev=n_iter)
    fitted_values_g = result_g.best_values

    # Store the best-fit values
    fwhm_x, fwhm_y = fitted_values_g['sigx'] * \
        fact, fitted_values_g['sigy'] * fact
    fwhm = np.sqrt(
        8 *
        np.log(2) *
        fitted_values_g['sigx'] *
        fitted_values_g['sigy'])
    ell = np.abs(fwhm_x - fwhm_y) / (fwhm_x + fwhm_y)

    fitted_values_cp = {'fwhm_x': fwhm_x,
                        'fwhm_y': fwhm_y,
                        'fwhm': fwhm,
                        'theta': theta,
                        'amp': amp,
                        'ell': ell,
                        'R': R,
                        }

    return fitted_values_cp, result_a.chisqr + result_g.chisqr


def full_beam_fit(bins, prof, widx, wedge, downsample_f, minimize=False):
    """
    Fit the beam core sidelobes with splines and the wing with a
    1/angle^3 function

    Args
    ----
    bins, prof: radial bins and beam profile values
    widx, wedge: index of the core/wing transition and maximum angle
    downsample_f: downsampling factor * the full length of the data
                  chosen for the interpolation reference points
    minimize: If True return only the chi-square value, the code is
              optimizing the wing scale.
              If False return the best-fit beam profile and radial bins
    """

    if widx > wedge:
        warnings.warn(
            "Wing scale is larger then theta max, returning fill value \
            for chi2")
        return 10**4

    # Do this process shifting bins and taking inverse fit_res averaged result
    # Downsample
    # Scipy Univariate spline does not work well with numbers between 0,1 so
    # multiply with some constant
    spline1d = UnivariateSpline(
        bins[:widx][::downsample_f], prof[:widx][::downsample_f] * 10**4)

    # Increase the resolution again
    bins_near_lobes = bins[:widx]
    prof_near_lobes = spline1d(bins_near_lobes)
    prof_near_lobes /= 10**4

    # Set the last of the downsampled bins as a starting point for the wing
    # fit and theta_max as the end point of the wing fit
    w0 = find_nearest(bins, bins_near_lobes[::downsample_f][-1])

    # Fit a 3rd degree polynomial
    poly_wing = np.poly1d(np.polyfit(bins[w0:wedge + 1][::downsample_f],
                                     prof[w0:wedge + 1][::downsample_f], 3))
    bins_wing = bins[widx:wedge + 1]
    prof_wing = poly_wing(bins_wing)

    # slinear interpolation for the 'stitching' of core and wing between
    # stitch_sample
    bins_between = [
        bins[widx - 1 - int(downsample_f / 2)], bins[widx + int(downsample_f / 2)]]
    prof_between = [
        prof_near_lobes[-int(downsample_f / 2)], prof_wing[int(downsample_f / 2)]]
    lin1d = scipy.interpolate.interp1d(
        bins_between, prof_between, kind='slinear')

    new_bins_between = bins[widx - 1 -
                            int(downsample_f / 2):widx + int(downsample_f / 2)]
    new_prof_between = lin1d(new_bins_between)

    full_bins = np.concatenate((bins[:widx - 1 - int(downsample_f / 2)],
                                new_bins_between,
                                bins[widx + int(downsample_f / 2):wedge + 1]))
    full_prof = np.concatenate((prof_near_lobes[:widx - 1 - int(downsample_f / 2)],
                                new_prof_between,
                                prof_wing[int(downsample_f / 2):wedge + 1 - widx]))

    fit_res = ((np.linspace(0, 1, len(full_prof))[
               ::-1]) * ((full_prof - prof[:wedge + 1])**2)).sum()
    if minimize:
        return fit_res
    else:
        return [full_bins, full_prof]


def fit_single_map(pmap, theta_max, acc, trim_factor, n_iter, wing_cutoff,
                   res, lmax, downsample_f, init_params, test_init_cond,
                   init_dependence, **kwargs):
    """Correct for the background, do map operations, optimize the wing scale
       and fit a single map

    Args
    ----
    pmap: ndmap object
    theta_max: maximum angle to fit for
    trim_factor: create a submap of trim_factor * theta_max
    wing_cutoff: Beam power in dB where the wing scale should be placed
                 Should be provided as a touple of min, max, nsamps to
                 consider for optimizing the wing scale.
    test_init_cond: If True, perturb the initial parameter values
                    by acc/2 and see if the change of the fitted values exceeds
                    the 'init_dependence' fraction.

    Returns
    -------
    The fitted values, beam profile and transfer function truncated to lmax
    """

    # Trim the map to avoid edge mask effects and get the profile
    if trim_factor != 1:
        trim_size = trim_factor * (theta_max / np.sqrt(2))
        submap = enmap.submap(pmap,
                              [[-trim_size, -trim_size],
                               [trim_size, trim_size]],
                              iwcs=pmap.wcs)
        pmap = submap

    # Ignore negative values that might still remain
    profall, binsall = enmap.rbin(pmap, bsize=res)
    bins_all_pos = binsall[np.where(profall[0] >= 0)]
    prof_all_pos = profall[0][np.where(profall[0] >= 0)]

    # Find index of maximum specified angle for the positive and full data
    wedge = find_nearest(binsall, theta_max)
    wedge_pos = find_nearest(bins_all_pos, theta_max)

    # If there many negative data points raise a warnign about the fidelity of
    # the fitting
    if len(prof_all_pos[:wedge_pos]) / \
            len(profall[0][:wedge]) < kwargs['data_vol']:
        t = len(prof_all_pos[:wedge_pos]) / len(profall[0][:wedge])
        warnings.warn('Over {} % of the data points of the profile were \
            negative with t={}'.format(int((1 - kwargs['data_vol']) * 100), t))

    # Correct for the background
    prof_c = correct_background(
        binsall[:wedge], profall[0][:wedge], kwargs['nsamps'])
    profall[0][:wedge] = prof_c

    bins_all_pos = binsall[np.where(profall[0] >= 0)]
    prof_all_pos = profall[0][np.where(profall[0] >= 0)]

    fitted_values_cp, fit_res_main = fit_main_lobe(
        pmap, res, init_params, n_iter, acc)

    # Test robustness of the fitting against fluctuating initial conditions
    if test_init_cond:
        fluctuation = np.random.uniform(-acc / 2,
                                        acc / 2, len(init_params.values()))
        params_perturbed = dict()
        for i, key in enumerate(init_params.keys()):
            params_perturbed[key] = init_params[key] + \
                fluctuation[i] * init_params[key]

        fitted_values_cp_perturbed, fit_resmain_p = fit_main_lobe(
            pmap, res, params_perturbed, n_iter, acc)

        for key in fitted_values_cp.keys():
            if np.abs(fitted_values_cp_perturbed[key] / fitted_values_cp[key] -
                      1) < init_dependence:
                print('Parameter passed')
                pass
            else:
                warnings.warn('Dependency on initial conditions over {} %'
                              .format(init_dependence * 100))

    # Define the wing scale within a range -- fit everything before with
    # splines
    if wing_cutoff is None:
        widx_min, widx_max, widx_samples = -40, -25, 5
    else:
        widx_min, widx_max, widx_samples = wing_cutoff

    widxs = [
        get_widx(
            prof_all_pos,
            wing_cutoff) for wing_cutoff in np.linspace(
            wing_cutoff[0],
            wing_cutoff[1],
            wing_cutoff[2])]
    wedge_pos = find_nearest(bins_all_pos, theta_max)

    fit_ress = []
    for widx in widxs:

        fit_res = full_beam_fit(bins_all_pos, prof_all_pos, widx, wedge_pos,
                                downsample_f, minimize=True)
        fit_ress.append(fit_res)

    # Use the wing scale with the lowest fit residuals
    full_bins, full_prof = full_beam_fit(bins_all_pos,
                                         prof_all_pos,
                                         widxs[np.argmin(fit_ress)],
                                         wedge_pos,
                                         downsample_f)

    fitted_values_cp['fit_res'] = np.min(fit_ress)
    fitted_values_cp['snr'] = calculate_snr_single(pmap, kwargs['r_noise'])

    # Compute solid angle
    omega = full_prof.sum(axis=0) * \
        ((full_bins[-1] - full_bins[0]) / len(full_bins))
    fitted_values_cp['omega'] = omega

    # Compute harmonic transform
    bl_fit = hp.beam2bl(full_prof, full_bins, lmax=lmax)

    # Interpolate profile to the initial resolution
    line1d = interp1d(full_bins, full_prof, fill_value='extrapoate')

    return fitted_values_cp, [binsall[:wedge], line1d(binsall[:wedge])], bl_fit


def run_fit(tele, band, map_path, rescale, smooth, pk_normalise, init_params,
            test_init_cond, init_dependence, theta_max, acc, trim_factor,
            n_iter, wing_cutoff, res, lmax, downsample_f, make_plots,
            save_stats, write_beam, outdir, prefix, **kwargs):
    """Read the maps, perform the fits, gather all fitted parameters,
       beam profiles and transfer functions and store/plot the results

    Args
    ----
    tele, band: telescope and frequency band
    map_path: path to the ndmap objects
    outdir, prefix: path to save the results under assigned prefix
    """

    # Decide on parallelization scheme
    enmaps = read_enmaps(map_path,
                         rescale,
                         smooth,
                         pk_normalise,
                         **kwargs)

    if init_params is None:
        try:
            init_params = get_input_params(
                kwargs['initial_parameters_file'], tele, band)
        except BaseException:
            raise ValueError('Missing arguments')

    map_names = os.listdir(map_path)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = len(enmaps)
    rank_batchsize = int(np.floor(N / size))
    quotient, remainder = divmod(N, size)

    if rank <= (remainder - 1):
        rank_batchsize += 1

    batch = np.arange(0, rank_batchsize, dtype=np.int32)
    batch += rank * (rank_batchsize)

    if rank > remainder:
        batch += int(remainder)

    df_keys = ['fwhm',
               'ell',
               'fwhm_x',
               'fwhm_y',
               'theta',
               'omega',
               'fit_res',
               'snr']

    qt = QTable(np.zeros((N, len(df_keys))), names=df_keys)
    qt.add_column(map_names, name='map_name')
    qt.add_index('map_name')

    if rank == 0:

        bins = np.zeros((len(enmaps)))
        profiles = np.zeros((len(enmaps)))
        harmonic_t = np.zeros((len(enmaps), lmax + 1))
        weights = np.ones((len(enmaps)))
    else:
        profiles, harmonic_t = None, None

    if rank == 0:
        profiles = comm.scatter(profiles, 0)
        bins = comm.scatter(bins, 0)
        harmonic_t = comm.scatter(harmonic_t, 0)

    for map_idx in batch:
        fitted_params, [rb, prof], bl = fit_single_map(pmap=enmaps[map_idx],
                                                       theta_max=theta_max,
                                                       acc=acc,
                                                       trim_factor=trim_factor,
                                                       init_params=init_params,
                                                       n_iter=n_iter,
                                                       test_init_cond=test_init_cond,
                                                       init_dependence=init_dependence,
                                                       wing_cutoff=wing_cutoff,
                                                       res=res,
                                                       lmax=lmax,
                                                       downsample_f=downsample_f,
                                                       **kwargs)

        for key in df_keys:
            qt.iloc[map_idx][key] = fitted_params[key]

        profiles = prof
        bins = rb
        bls = bl / np.nanmax(bl)

    all_profiles = comm.gather(profiles, root=0)
    all_bins = comm.gather(bins, root=0)
    all_bl = comm.gather(bls, root=0)
    all_qts = comm.gather(qt, root=0)

    if rank == 0:
        stats = astropy.table.vstack(all_qts)
        rows = np.where(stats['snr'] == 0)
        stats.remove_rows(rows)

    if save_stats and rank == 0:
        astropy.io.misc.hdf5.write_table_hdf5(stats, opj(
            outdir, prefix + '_stats' + '.h5'), overwrite=True)

    if write_beam and rank == 0:

        # choose a weighting scheme
        # weights = 1/stats['snr']

        all_bins = np.array(all_bins)
        all_profiles = np.array(all_profiles)

        avg_bins = np.nanmean(all_bins * weights[:, np.newaxis], axis=0)
        avg_prof = np.nanmean(all_profiles * weights[:, np.newaxis], axis=0)
        std_prof = np.nanstd(all_profiles * weights[:, np.newaxis], axis=0)

        avg_bl = np.nanmean(all_bl * weights[:, np.newaxis], axis=0)
        std_bl = np.nanstd(all_bl * weights[:, np.newaxis], axis=0)

        np.savetxt(opj(outdir, prefix + '_prof.txt'),
                   [avg_bins, avg_prof, std_prof])
        np.savetxt(opj(outdir, prefix + '_bl.txt'), [avg_bl, std_bl])

        plot_profile(
            avg_bins,
            avg_prof,
            std_prof,
            img_file=opj(
                outdir,
                prefix +
                '_prof.png'))
        plot_bls(avg_bl, std_bl, img_file=opj(outdir, prefix + '_bl.png'))

    return


def get_parser():
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # Those shall mostly be replaced with configuration files
    parser.add_argument('--tele',
                        action='store',
                        dest='tele',
                        default='SAT',
                        type=str,
                        help='Telescope [SAT,LAT]')
    parser.add_argument('--band',
                        action='store',
                        dest='band',
                        default='f090',
                        type=str,
                        help='Frequency band')
    parser.add_argument('--map_path',
                        action='store',
                        dest='map_path',
                        default=None,
                        type=str,
                        help='Location of the maps')
    parser.add_argument('--rescale',
                        action='store_true',
                        dest='rescale',
                        default=False,
                        help='If True, rescale the maps with a scale defined as \
                        scale in kwargs')
    parser.add_argument('--smooth',
                        action='store_true',
                        dest='smooth',
                        default=False,
                        help='If True, smooth the maps by some factor defined \
                        as sigma in kwargs')
    parser.add_argument('--pk_normalise',
                        action='store_true',
                        dest='pk_normalise',
                        default=True,
                        help='If True, peak-normalise the maps')
    parser.add_argument('--init_params',
                        action='store',
                        dest='init_params',
                        default=None,
                        help='Initial parameters to be used for the fitting')
    parser.add_argument('--test_init_cond',
                        action='store_true',
                        dest='test_init_cond',
                        default=False,
                        help='Test fitting result dependency on initial \
                        parameters')
    parser.add_argument('--init_dependence',
                        action='store',
                        type=float,
                        dest='init_dependence',
                        default=.01,
                        help='Allowed maximum dependence on initial conditions')
    parser.add_argument('--theta_max',
                        action='store',
                        dest='theta_max',
                        default=None,
                        type=float,
                        help='Maximum angle of the map')
    parser.add_argument('--acc',
                        action='store',
                        dest='acc',
                        default=.05,
                        help='Accuracy requirement')
    parser.add_argument('--trim_factor',
                        action='store',
                        dest='trim_factor',
                        default=1,
                        type=float,
                        help='Avoid edge effects by trimming the map to \
                        trim_factor*theta_max')
    parser.add_argument('--n_iter',
                        action='store',
                        dest='n_iter',
                        type=int,
                        default=None,
                        help='Number of iterations for fitting')
    parser.add_argument('--wing_cutoff',
                        action='store',
                        dest='wing_cutoff',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Beam Power in dB lower than which we define the \
                        beam wing')
    parser.add_argument('--wing_fit_max',
                        action='store',
                        dest='wing_fit_max',
                        default=1,
                        help='Times*theta_max to extrapolate a 3rd degree \
                        polynomial fit')
    parser.add_argument('--res',
                        action='store',
                        dest='res',
                        default=None,
                        type=float,
                        help='Map resolution')
    parser.add_argument('--downsample_f',
                        action='store',
                        dest='downsample_f',
                        default=10,
                        type=int,
                        help='Downsampling factor if the full beam data')
    parser.add_argument('--lmax',
                        action='store',
                        dest='lmax',
                        default=400,
                        help='Maximum multipole number to take into account \
                        for the harmonic transform')
    parser.add_argument('--make_plots',
                        action='store_true',
                        dest='make_plots',
                        default=False,
                        help='Plot maps, beam profiles and harmonic transforms')
    parser.add_argument('--save_stats',
                        action='store_true',
                        dest='save_stats',
                        default=False,
                        help='Save fitted parameters table as .h5 file')
    parser.add_argument('--write_beam',
                        action='store_true',
                        dest='write_beam',
                        default=False,
                        help='Write beam profile and harmonic transform')
    parser.add_argument('--prefix',
                        action='store',
                        dest='prefix',
                        default='test_maps',
                        help='prefix to add at the name of saved parameters \
                        and plots')
    parser.add_argument('--outdir',
                        action='store',
                        dest='outdir',
                        default=None,
                        help='Output directory')
    # kwargs
    parser.add_argument('--beamfile',
                        action='store',
                        dest='beamfile',
                        default=None,
                        type=str,
                        help='File containing the beam')
    parser.add_argument('--initial_parameters_file',
                        action='store',
                        dest='initial_parameters_file',
                        default='/mnt/so1/users/konstad/'\
                                'pwg-scripts_sp/pwg-bcp/'\
                                'sotodlib_staging/data/'\
                                'initial_parameters.yaml',
                        type=str,
                        help='Location of the initial parameters file')
    parser.add_argument('--r_noise',
                        action='store',
                        dest='r_noise',
                        type=float,
                        default=None,
                        help='Radius after which beam is negligible--used for \
                        SNR estimation')
    parser.add_argument('--sigma',
                        action='store',
                        dest='sigma',
                        default=None,
                        help='Smoothing scale')
    parser.add_argument('--scale',
                        action='store',
                        dest='scale',
                        default=None,
                        help='Scaing factor if rescaling of the maps is called \
                        for')
    parser.add_argument('--nsamps',
                        action='store',
                        dest='nsamps',
                        default=10,
                        help='Number of chunks to evaluate the minim gradient \
                        on for corrrecting the background')
    parser.add_argument('--data_vol',
                        action='store',
                        dest='data_vol',
                        default=0.7,
                        help='Acceptable fraction of negative to full data \
                        for a map to be fitted')

    args = parser.parse_args()

    return args


def main():

    args = get_parser()

    all_kwargs = {'initial_parameters_file': args.initial_parameters_file,
                  'beamfile': args.beamfile,
                  'r_noise': args.r_noise,
                  'sigma': args.sigma,
                  'scale': args.scale,
                  'nsamps': args.nsamps,
                  'data_vol': args.data_vol,
                  }

    run_fit(tele=args.tele, band=args.band,
            map_path=args.map_path,
            rescale=args.rescale,
            smooth=args.smooth,
            pk_normalise=args.pk_normalise,
            init_params=args.init_params,
            test_init_cond=args.test_init_cond,
            init_dependence=args.init_dependence,
            theta_max=args.theta_max,
            acc=args.acc,
            trim_factor=args.trim_factor,
            n_iter=args.n_iter,
            wing_cutoff=args.wing_cutoff,
            res=args.res,
            lmax=args.lmax,
            downsample_f=args.downsample_f,
            make_plots=args.make_plots,
            save_stats=args.save_stats,
            write_beam=args.write_beam,
            outdir=args.outdir,
            prefix=args.prefix,
            **all_kwargs)


if __name__ == "__main__":
    main()

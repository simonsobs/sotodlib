#!/usr/bin/env python3

from scipy.interpolate import UnivariateSpline, interp1d
from mpi4py import MPI
from scipy.special import jv
import scipy
import healpy as hp
import logging
from astropy.table import QTable, vstack
import astropy
import yaml
import pickle
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters
from pixell import enmap, curvedsky
import os
import h5py
import itertools
import sys
import matplotlib
matplotlib.use('agg')

opj = os.path.join

logger = logging.getLogger(__name__)


def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_observation_names(ctx_file_name, query_dict):
    """ Returns a list of obs_id of all observations 
        corresponding to the query parameters, parsed as a dictionary,
        given the context file path."""

    from sotodlib import core
    ctx = core.Context(ctx_file_name)

    if not bool(query_keys):
        logger.debug("There are no specified query parameters")
        return

    query_string = str()

    for key in list(query_dict.keys())[:-1]:
        query_string += '{}=="{}" and '.format(key, query_dict[key])
    query_string += '{}=="{}"'.format(list(query_dict.keys())
                                      [-1], a[list(query_dict.keys())[-1]])
    obs_set = ctx.obsdb.query(query_string)\

    map_names = []
    for obs_idx in range(len(obs_set)):
        map_names.append(obs_set[obs_idx]['obs_id'])
    return map_names


def read_enmaps(
        map_names,
        rescale,
        smooth,
        ctx=None,
        pk_normalize=True,
        pol=False,
        **kwargs):
    """Read map files and rescale/smooth if requested.
    The rescaling/smoothing factor should be parsed as kwargs if
    `rescale'/ `smooth' is set to be True.
    The function peak-normalizes the maps by default.

    Args
    ----
    map_names: List of fits files or observation ids.
    ctx: Context object to be parsed if the map_names are given as
         observation ids.
    """

    fits_files = ['.fits' in map_name for map_name in map_names]
    if np.all(fits_files) == True:
        reading_func = enmap.read_fits
    else:
        if np.any(fits_files) == True:
            logger.debug(
                "Exiting function because mixed file types were provided.")
            sys.exit()
        else:
            if ctx is not None:
                reading_func = ctx.get_obs
            else:
                logger.debug(
                    "Exiting function because no context file was provided.")
                sys.exit()

    enmaps = []
    for map_name in map_names:
        if not os.path.exists(map_name):
            warnings.warn("Map file was not found")
        try:
            pmap = reading_func(map_name)
        except BaseException:
            raise TypeError('Invalid enmap object')
        array_map, array_ra, array_dec = enmap_to_array(pmap)
        if smooth:
            array_map = np.asarray(smooth_map(pmap, kwargs['sigma']))
        if rescale:
            array_map = [array_map[i, :] * kwargs['scale']
                         for i in range(array_map.shape[0])]
        if pk_normalize:
            array_map /= np.nanmax(array_map)
        pmap[:, :, :] = array_map
        enmaps.append(pmap)
    if pol is False:
        enmaps = [enmaps[i][np.newaxis, 0, :, :] for i in range(len(enmaps))]
    return enmaps


def enmap_to_array(pmap):
    """Split an enmap into (pure_array, dec, ra) arrays."""

    pmap_arr = np.asarray(pmap)
    dec, ra = pmap.posmap()
    return pmap_arr, np.asarray(dec), np.asarray(ra)


def dummy_source_detection(map_names, pmaps, r_noise, threshold=100):
    """Function that assigns a source included to the map if
       SNR > threshold."""

    pmaps_source, pmaps_source_names = [], []
    for pmap, pmap_name in zip(pmaps, map_names):

        snr = calculate_snr_single(pmap, r_noise, coords=None)

        if snr > threshold:
            pmaps_source.append(pmap)
            pmaps_source_names.append(pmap_name)
        else:
            logger.debug("The map SNR={}, estimated from data on the outer {} % of the map\
               does not exceed the threshold value set as {}".format(snr, (1-r_noise)*100, threshold))

    return pmaps_source_names, pmaps_source


def trim_multiple_maps(pmaps, res_precision=.000001, skybox=None):
    """Function that trims the input maps to some user-specified
        limits or adjusts them to have the same size, provided the
        input maps are source-centred and have the same resolution.

    Args
    ----
    pmaps: array of ndmap objects
    res_precision: difference in resolution tolerance
                   between different maps
    skybox: Tuple of ra and dec physical limits
            e.g. skybox=np.asarray([[dec0, ra0], [dec1, ra1]])

    Returns
    -------
    maps_trimmed: The trimmed version of maps

    """

    if skybox is not None:
        skybox = [[skybox[0], skybox[1]], [skybox[2], skybox[3]]]
    else:
        same_dimension = all(map_i.shape == pmaps[0].shape for map_i in pmaps)
        same_resolution = np.logical_and(
            all(
                map_i.wcs.wcs.cdelt[0] -
                pmaps[0].wcs.wcs.cdelt[0] < res_precision for map_i in pmaps),
            all(
                map_i.wcs.wcs.cdelt[1] -
                pmaps[0].wcs.wcs.cdelt[1] < res_precision for map_i in pmaps))

        same_refpix = np.logical_and(
            all(
                map_i.wcs.wcs.crpix[0] == pmaps[0].wcs.wcs.crpix[0] for map_i in pmaps),
            all(
                map_i.wcs.wcs.crpix[1] == pmaps[0].wcs.wcs.crpix[1] for map_i in pmaps))

        if not same_resolution:
            raise ValueError("The maps have different resolution")

        # The map geometries are identical so no need for extra operations
        if same_refpix and same_dimension:
            logger.info("The maps have the same size, resolution and reference \
                         pixel, and no skybox is parsed so no need to trim them")
            return pmaps

        # The source-scan coordinate system places the source at (0,0) but not necessarily at the center of the map
        # We iterate though the maps and find the minimum pixel number from the reference point to the edge of the map
        # (in ra and dec). This way we create a pixbox which is converted to a skybox setting the coordinate limits
        # for every map which is about to be trimmed.

        if same_refpix == False:

            min_ra_pixels_around_refpix = [np.min([map_i.shape[2]-int(map_i.wcs.wcs.crpix[0]), int(map_i.wcs.wcs.crpix[0])])
                                           for map_i in pmaps]
            min_dec_pixels_around_refpix = [np.min([map_i.shape[1]-int(map_i.wcs.wcs.crpix[1]), int(map_i.wcs.wcs.crpix[1])])
                                            for map_i in pmaps]
            min_ra_pix_idx = np.min(min_ra_pixels_around_refpix)
            min_dec_pix_idx = np.min(min_dec_pixels_around_refpix)

            pmaps_trimmed = []
            for map_i in pmaps:
                refpix = map_i.wcs.wcs.crpix
                pixbox = (np.asarray([[refpix[1]-min_ra_pix_idx, refpix[0]-min_dec_pix_idx], [
                          refpix[1]+min_ra_pix_idx, refpix[0]+min_dec_pix_idx]]))
                skybox = enmap.pixbox2skybox(map_i.shape, map_i.wcs, pixbox)
                pmaps_trimmed.append(enmap.submap(map_i, box=skybox))

        # If the maps have the same reference pixel then it is enough to resize
        # them to match the size of the smaller map.
        elif same_dimension == False:

            min_ra_idx = np.argmin([map_i.shape[1] for map_i in pmaps])
            min_dec_idx = np.argmin([map_i.shape[2] for map_i in pmaps])
            ra_min_range = pmaps[min_ra_idx].posmap()[0]
            dec_min_range = pmaps[min_dec_idx].posmap()[1]
            ra_min, ra_max = ra_min_range.min(), ra_min_range.max()
            dec_min, dec_max = dec_min_range.min(), dec_min_range.max()
            skybox = [[dec_min, ra_min], [dec_max, ra_max]]

    if not pmaps_trimmed:
        pmaps_trimmed = [enmap.submap(pmap, box=skybox) for map_i in pmaps]

    return pmaps_trimmed


def coadd_maps(pmaps, skybox):
    """Coadd source-centred maps of same
       size and similar resolution.

    Args
    ----
    pmaps: array of ndmap objects

    Returns
    -------
    The coadded map as an ndmap object
    """

    pmaps_t = trim_multiple_maps(pmaps, skybox=skybox)

    # The maps have now the same number of pixels and are centred on the source.
    # However the pixel coordinates might slightly differ.
    # Projecting the maps to the geometry of the first map before co-adding.
    projected_maps = []
    for map_i in pmaps_t:
        projected_maps.append(enmap.project(
            map_i, pmaps_t[0].shape, pmaps_t[0].wcs))

    coadded_enmap = enmap.zeros(
        pmaps_t[0].shape, pmaps_t[0].wcs, dtype=np.float64)
    coadded_enmap[0, :, :] = sum(np.array(projected_maps)[:, 0, :, :])

    return coadded_enmap


def correct_background(prof, nsamps):
    """Split the radial profile, prof, to 'nsamps' chunks and subtract
       the average value of the chunck with the minimum gradient from 
       the full profile. The function returns the background corrected 
       profile."""

    chunk_idxs = np.linspace(0, len(prof), nsamps)

    grad = []
    for i in range(len(chunk_idxs[:-1])):
        try:
            grad.append(
                np.mean(np.gradient(prof[int(chunk_idxs[i]):
                                         int(chunk_idxs[i + 1])])))
        except BaseException:
            raise ValueError("The requested number of samples is large \
                              compared to the profile resolution")
    min_samp = find_nearest(grad, 0)
    offset = np.mean(prof[int(chunk_idxs[min_samp]):
                          int(chunk_idxs[min_samp + 1])])
    return prof - offset


def calculate_snr_single(pmap, r_noise, npoints=10, coords=None):
    """Calculate the Signal-to-Noise Ratio of a single
       map. The noise is estimated as the standard deviation
       outside a radius, `r_noise', where we assume the beam power 
       has fallen below 1%. The peak amplitude value is calculated from
       the `npoints' brightest pixels.

    Args
    ----
    coords: (dec,ra) values. Those will be estimated from posmap(), 
            if not parsed.
    """

    if coords is None:
        dec, ra = pmap.posmap()
    else:
        dec, ra = coords
    r = np.sqrt(ra**2 + dec**2)
    r = r.reshape(r.shape[-2], r.shape[-1])
    map_temp = pmap.reshape(pmap.shape[-2], pmap.shape[-1])
    neighbours = np.array(map_temp[np.where(r > r_noise)])
    rms = get_rms(neighbours)

    pmap_1d = pmap[0].reshape(pmap.shape[1] * pmap.shape[2])
    av_max = np.mean(np.sort(pmap_1d)[::-1][:npoints])

    snr = av_max / rms

    if np.isnan(snr):
        logger.debug("SNR is nan, try reducing the the noise radius")
    return snr


def calculate_snr(pmaps, r_noise, n_obs=1, n_max_sets=-1):
    """Calculate the Signal-to-Noise Ratio of an array of pmaps from a
       number of `n_max_sets' randomly chosen subsets, each containing `n_obs' maps.
       The function returns the mean and standard deviation of the SNR across the 
       `n_max_sets' subsets."""

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


def gaussian_2d_rot(x, y, x0, y0, amp, sigx, sigy, theta, **kwargs):
    """2D Gaussian function including rotation `theta' around point (x0,y0).
       The sigma of the distribution in x,y directions, `sigx' and `sigy', and 
       rotation should be given in radians.

    Args
    ----
    amp: will be included in kwargs if the map is not peak-normalized
    """

    sigx2 = sigx**2
    sigy2 = sigy**2
    x_rot = np.cos(theta)**2 / (2 * sigx2) + np.sin(theta)**2 / (2 * sigy2)
    y_rot = np.sin(theta)**2 / (2 * sigx2) + np.cos(theta)**2 / (2 * sigy2)
    xy_rot = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

    d2_rot_gauss = -x_rot * (x - x0)**2 - y_rot * \
        (y - y0)**2 - 2 * xy_rot * (x - x0) * (y - y0)
    return amp * np.exp(d2_rot_gauss)


def get_rms(x):
    return np.sqrt(np.nanmean(x**2))


def airy_disc(x, y, amp, x0, y0, R, theta):
    """Airy pattern function estimated around point (x0,y0)"""

    Rz = 1.2196698912665045
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    r = np.ma.masked_equal(r, 0)
    z = np.pi * r / (R / Rz)
    j1 = scipy.special.jv(1, z)
    airy = (2 * j1 / z)**2
    return amp * airy


def get_input_params(INITIAL_PARA_FILE, tele, band):
    """Get the initial parameters configuration from a .yaml
    file for telescope 'tele' and frequency band 'band'"""

    with open(INITIAL_PARA_FILE, "r") as stream:
        config_file = yaml.safe_load(stream)

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

    mask = enmap.zeros(pmap.shape, pmap.wcs, dtype=np.float64)
    d = enmap.distance_from(mask.shape, mask.wcs, [[0], [0]])
    mask += 1. * (d < r_mask)

    return mask * pmap


def get_widx(prof, wing_cutoff):
    """Get the index corresponding to the `wing_cutoff' beam power value 
       provided in dB. The angle of this index corresponds to the beam 
       core/wing transition scale."""

    mask_idxs = np.where(10 * np.log10(np.abs(prof)) < wing_cutoff)
    widx = mask_idxs[0][0]

    return widx


def make_model_params(dependent_params, b_acc):
    """Initialize mean values and ranges of the dependent
       parameters. The fitting range is based on the desired 
       accuracy."""

    params = Parameters()
    for idx_key, key in enumerate(dependent_params.keys()):
        key_value = dependent_params[key]
        params.add(key, value=key_value,
                   min=-b_acc * key_value + key_value,
                   max=b_acc * key_value + key_value)
    return params


def get_ref_beam(beamfile, theta_trunc=None, lmax=None):
    """Returns the radial profile (and corresponding bins), solid angle,
       and harmonic transform of a reference beam, loaded from `beamfile'. 
       The reference beam profile is truncated to `theta_trunc' and the 
       harmonic transform assumes a maximum multipole number `lmax'."""

    bdict = pickle.load(open(beamfile, 'rb'), encoding='latin1')
    prof_ref, bins_ref, res = bdict['profiles'], np.radians(
        bdict['ru']), np.radians(bdict['size'][0][1])

    if theta_trunc:
        idx_trunc = find_nearest(bins_ref, theta_trunc)

    bl = curvedsky.profile2harm(br, r, lmax=lmax)

    return bins_ref[:idx_trunc], prof_ref[:idx_trunc], prof_ref[:idx_trunc].sum(
    ) * res, bl_ref


def prof2map(prof, bins, output_shape, output_wcs):
    """Create a 2D enmap object from a radial profile given
       radial bins and output 2D enmap shape and wcs"""

    out_map = enmap.zeros(output_shape, output_wcs, dtype=np.float64)
    dec, ra = out_map.posmap()
    r = np.sqrt(ra**2 + dec**2)

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):

            idx_nearest = find_nearest(bins, r[i, j])
            out_map[0, i, j] = prof[idx_nearest]

    return out_map


def plot_maps(data=None, fit=None, img_file=None):
    """Plot data, fit and residual maps and store
       under `img_file' name."""

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
    """Plot average profile and 1σ-band and store under 
       `img_file' name.

    Args
    ----
    ref: add reference profile to the plot
    fsize: label size
    """

    fig = plt.figure(dpi=300)
    axs = fig.add_subplot(111)

    label_d = 'Coadded data'

    if data_std is not None:
        axs.fill_between(
            np.degrees(bins),
            data_mean - data_std,
            data_mean + data_std,
            alpha=0.7,
            label='$1\\sigma error$')

        label_d = 'Mean data'

    if ref is not None:
        axs.semilogy(np.degrees(bins), ref, 'k--', label='Reference')

    axs.plot(np.degrees(bins), data_mean, label=label_d)
    axs.legend()
    axs.set_yscale('log')
    axs.set_xlabel('Radius [degrees]', size=fsize)
    axs.set_ylabel('Beam Power [dB]', size=11)
    axs.set_xlim(0, np.degrees(bins[-1]))
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def plot_bls(data_mean=None, data_std=None,
             ref=None, img_file=None, fontsize=12):
    """Plot average beam transfer function and 1σ-band
       under `img_file' name.

    Args
    ----
    ref: add reference harmonic transform to the plot
    fsize: label size
    """

    fig = plt.figure(dpi=300)
    axs = fig.add_subplot(111)

    ells = np.arange(0, len(data_mean))
    label_d = 'Coadded data'

    if data_std is not None:
        axs.fill_between(
            ells,
            data_mean -
            data_std,
            data_mean +
            data_std,
            alpha=0.7,
            label='$1\\sigma error$')

        label_d = 'Mean data'

    if ref is not None:
        ref /= np.nanmax(ref)
        axs.plot(ref, 'k--', label='Reference')

    axs.plot(ells, data_mean, label=label_d, lw=2)
    axs.legend()
    axs.set_ylabel('$B_{\\ell}$', size=fontsize)
    axs.set_xlim(0, ells[-1])
    axs.set_xlabel('Multipole, $\\ell$', size=fontsize)
    plt.savefig(img_file, bbox_inches='tight')
    plt.close()


def fit_main_lobe(pmap, res, init_params, n_iter, acc):
    """Fit an Airy pattern to find the beam's first minimum,
       mask the sidelobes and fit a 2D Gaussian.

    Args
    ----
    pmap: the ndmap object to be fitted
    res: the size of the radial bins. if None, assigned
         from enmap.rbin()
    init_params: initial guess for the fitted parameters
    n_iter: number of iterations
    acc: desired accuracy of the fit

    Returns
    -------
    Dictionary of the fitted parameters.
    Dict.keys() = 'fwhm_x', 'fwhm_y', 'fwhm', 'theta',
                  'amp', 'ell', 'R'.
    """

    prof_pmap, bins_pmap = enmap.rbin(pmap, bsize=res)
    bins_pmap_pos = bins_pmap[np.where(prof_pmap[0] > 0)]
    prof_pmap_pos = prof_pmap[0][np.where(prof_pmap[0] > 0)]
    # Fit for the first dark ring
    array_map, dec, ra = enmap_to_array(pmap)

    amp, x0, y0, R, theta = [init_params[key] for key in ['amp',
                                                          'x0',
                                                          'y0',
                                                          'R',
                                                          'theta']]
    error = np.sqrt(array_map.ravel() + 1)

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
    array_map, dec, ra = enmap_to_array(pmap_masked)

    fact = np.sqrt(8 * np.log(2))
    sigx, sigy = init_params['fwhm_x'] / fact, init_params['fwhm_y'] / fact

    fmodel = Model(gaussian_2d_rot, independent_vars=('x', 'y'))
    params = make_model_params(dependent_params={'x0': x0,
                                                 'y0': y0,
                                                 'amp': amp,
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


def full_beam_fit(bins, prof, widx, wedge, downsample_f, stitch_f, minimize=False):
    """
    Fit the beam's near sidelobes with splines and the wing with a
    1/angle^3 function

    Args
    ----
    bins, prof: radial bins and beam profile values
    widx, wedge: index of the core/wing transition and maximum angle
                 we fit for
    downsample_f: downsampling factor * the full length of the data
                  chosen for the interpolation reference points
    minimize: If True return only the chi-square value, the code is
              optimizing for the wing scale.
              If False return the best-fit beam profile and radial bins
    """

    if widx > wedge:
        logger.debug(
            "Wing scale is larger than theta max, returning fill value \
            for chi2")
        return 10**4

    # Downsample
    # Scipy Univariate spline does not work well with numbers between 0,1 so
    # multiply with some constant which will be divided out after
    try:
        spline1d = UnivariateSpline(
            bins[:widx][::downsample_f], prof[:widx][::downsample_f] * 10**4)
    except BaseException:
        logger.debug("The downsampling factor is too large")
        # try increasing the number of sampling points
        try:
            if downsample_f > 2:
                downsample_f -= 2
            spline1d = UnivariateSpline(
                bins[:widx][::downsample_f], prof[:widx][::downsample_f] * 10**4)
        except BaseException:
            return

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

    # linear interpolation for the 'stitching' of core and wing between
    # stitch_sample
    bins_between = [
        bins[widx - 1 - stitch_f], bins[widx + stitch_f]]
    prof_between = [
        prof_near_lobes[-stitch_f], prof_wing[stitch_f]]
    lin1d = scipy.interpolate.interp1d(
        bins_between, prof_between, kind='slinear')

    new_bins_between = bins[widx - 1 -
                            stitch_f:widx + stitch_f]
    new_prof_between = lin1d(new_bins_between)

    full_bins = np.concatenate((bins[:widx - 1 - stitch_f],
                                new_bins_between,
                                bins[widx + stitch_f:wedge + 1]))
    full_prof = np.concatenate((prof_near_lobes[:widx -
                                                1 -
                                                stitch_f], new_prof_between, prof_wing[stitch_f:wedge +
                                                                                       1 -
                                                                                       widx]))

    fit_res = ((np.linspace(0, 1, len(full_prof))[
               ::-1]) * ((full_prof - prof[:wedge + 1])**2)).sum()
    if minimize:
        return fit_res
    else:
        return [full_bins, full_prof]


def fit_single_map(pmap, theta_max, acc, trim_factor, n_iter, wing_cutoff,
                   res, lmax, downsample_f, stitch_f, init_params,
                   test_init_cond, init_dependence, dB_thresh=None, **kwargs):
    """The function performs the following operations:
       - corrects for the map's background level
       - trims the map if needed
       - estimates the Signal-to-Noise ratio
       - estimates the beam parameters, 
       - optimizes the wing scale
       - fits the radial profile of the map
       - computes the solid angle and harmonic transform from the
         fitted profile truncated to some beam power value, `dB_thresh'. 

    Args
    ----
    pmap: ndmap object
    theta_max: maximum angle to fit for
    trim_factor: create a submap of trim_factor * theta_max, if trim_factor!=1.
    wing_cutoff: Beam power in dB where the wing scale should be placed
                 Should be provided as a touple of min, max, nsamps to
                 consider for optimizing the wing scale.
    test_init_cond: If True, perturb the initial parameter values
                    by acc/2 and see if the change of the fitted values exceeds
                    the 'init_dependence' fraction.

    Returns
    -------
    fitted values dictionary, [radial bins, fitted profile],  harmonic transform
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
        logger.warning("Over {} % of the data points of the profile were \
            negative with t={}".format(int((1 - kwargs['data_vol']) * 100), t))

    # Correct for the background
    prof_c = correct_background(
        profall[0][:wedge], kwargs['nsamps'])
    profall[0][:wedge] = prof_c

    bins_all_pos = binsall[np.where(profall[0] >= 0)]
    prof_all_pos = profall[0][np.where(profall[0] >= 0)]

    # Fit main lobe
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
                logger.info("Parameter passed")
                pass
            else:
                logger.warning("Dependency on initial conditions over {} %"
                               .format(init_dependence * 100))

    # Define the wing scale within a range -- fit everything before with
    # splines and after with a 1/angle^3 function
    if wing_cutoff is None:
        widx_min, widx_max, widx_samples = -50, -35, 10
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
                                downsample_f, stitch_f, minimize=True)
        fit_ress.append(fit_res)

    # Use the wing scale with the lowest fit residuals
    full_bins, full_prof = full_beam_fit(bins_all_pos,
                                         prof_all_pos,
                                         widxs[np.argmin(fit_ress)],
                                         wedge_pos,
                                         downsample_f,
                                         stitch_f)

    fitted_values_cp['fit_res'] = np.min(fit_ress)
    fitted_values_cp['snr'] = calculate_snr_single(pmap, kwargs['r_noise'])

    # Compute harmonic transform
    if dB_thresh is not None:
        w_db_t = find_nearest(10 * np.log10(np.abs(full_prof)), dB_thresh)
    else:
        w_db_t = -1

    # Compute solid angle
    omega = full_prof[:w_db_t].sum(axis=0) * \
        ((full_bins[:w_db_t][-1] - full_bins[:w_db_t][0]) /
         len(full_bins[:w_db_t]))
    fitted_values_cp['omega'] = omega

    bl_fit = curvedsky.profile2harm(
        full_prof[:w_db_t], full_bins[:w_db_t], lmax=lmax)

    # Interpolate profile to the initial resolution
    line1d = interp1d(full_bins, full_prof, fill_value='extrapoate')

    return fitted_values_cp, [binsall[:wedge], line1d(binsall[:wedge])], bl_fit


def run_fit(
        tele,
        band,
        map_files,
        map_path,
        query_param_file,
        rescale,
        smooth,
        pk_normalize,
        init_params,
        threshold,
        coadd,
        skybox,
        pol,
        test_init_cond,
        init_dependence,
        theta_max,
        acc,
        trim_factor,
        n_iter,
        wing_cutoff,
        res,
        lmax,
        downsample_f,
        stitch_f,
        make_plots,
        save_stats,
        write_beam,
        outdir,
        prefix,
        dB_thresh,
        **kwargs):
    """ This function performs the following operations:
        - Reads the maps from an input path or finds relevant observations based
         on input query params. The latter are included in a provided .yaml file.
         A list of map names can be parsed, alternatively.
        - Reads the input fitting parameters given telescope, `tele', and `band'.
        - Co-adds the maps if `coadd' is True.
        - Gathers all fitted beam parameters profiles and harmonic transforms.
        - Stores them to beam parameters to .h5 files where the data are contained
          in astropy tables, if `save_stats' is True.
        - Stores the (average) beam profile and harmonic transform as text files,
          if `write_beam' is True at given `outdir' under given `prefix'.
        - Plots the beam profile and harmonic transform if `make_plots' is True
          at given `outdir' under given `prefix'.
    """

    ctx = None
    if map_files is None:
        if map_path is not None:
            map_files = [opj(map_path, map_file)
                         for map_file in os.listdir(map_path)]
        elif query_param_file is not None:

            with open(query_param_file, "r") as stream:
                query_params = yaml.safe_load(stream)
            query_dict = {key: query_params['params'][key]
                          for key in query_params['params'].keys()}
            map_files = get_observation_names(
                query_params['ctx_file_name'], query_dict)
        else:
            logger.debug("Exiting function because no map files were found. \
                Provide a list of comma-separated files or a path.")
            sys.exit()

    # Decide on parallelization scheme
    enmaps = read_enmaps(map_files,
                         rescale,
                         smooth,
                         ctx,
                         pk_normalize,
                         pol,
                         **kwargs)

    map_names, enmaps = dummy_source_detection(map_files,
                                               enmaps,
                                               kwargs['r_noise'],
                                               threshold=threshold)

    if init_params is None:
        try:
            init_params = get_input_params(
                kwargs['initial_parameters_file'], tele, band)
        except BaseException:
            raise ValueError('Missing arguments')

    if coadd:
        coadded_map = coadd_maps(enmaps, skybox)

        if pk_normalize:
            coadded_map /= np.nanmax(coadded_map)

        fitted_params, [rb, prof], bl = fit_single_map(pmap=coadded_map,
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
                                                       stitch_f=stitch_f,
                                                       dB_thresh=dB_thresh,
                                                       **kwargs)

        bl /= np.nanmax(bl)

        h = h5py.File(opj(outdir, prefix + '_stats' + '_coadded.h5'), 'w')
        for k, v in fitted_params.items():
            h.create_dataset(k, data=np.array(v))
        h.close()

        np.savetxt(opj(outdir, prefix + '_prof_coadded.txt'), [rb, prof])
        np.savetxt(opj(outdir, prefix + '_bl_coadded.txt'), bl)

        plot_profile(rb,
                     prof,
                     data_std=None,
                     img_file=opj(
                         outdir,
                         prefix + '_prof_coadded.png'))
        plot_bls(bl,
                 data_std=None,
                 img_file=opj(outdir,
                              prefix +
                              '_bl_coadded.png'))
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = len(enmaps)
    batch = np.arange(rank * N // size, (rank + 1) * N // size)

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
        profiles, bins, harmonic_t = None, None, None

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
                                                       stitch_f=stitch_f,
                                                       dB_thresh=dB_thresh,
                                                       **kwargs)

        for key in df_keys:
            qt.iloc[map_idx][key] = fitted_params[key]

        profiles = prof
        bins = rb
        harmonic_t = bl / np.nanmax(bl)

    all_profiles = comm.gather(profiles, root=0)
    all_bins = comm.gather(bins, root=0)
    all_bl = comm.gather(harmonic_t, root=0)
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

        if make_plots:

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
    parser.add_argument('--map_files',
                        action='store',
                        dest='map_files',
                        default=None,
                        type=str,
                        nargs='+',
                        help='List of map files to be fitted')
    parser.add_argument('--map_path',
                        action='store',
                        dest='map_path',
                        default=None,
                        type=str,
                        help='Location of the maps')
    parser.add_argument('--query_param_file',
                        action='store',
                        dest='query_param_file',
                        default=None,
                        type=str,
                        help='Location of query .yaml file')
    parser.add_argument(
        '--rescale',
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
    parser.add_argument('--pk_normalize',
                        action='store_true',
                        dest='pk_normalize',
                        default=True,
                        help='If True, peak-normalize the maps')
    parser.add_argument('--threshold',
                        type=float,
                        action='store',
                        dest='threshold',
                        default=100,
                        help='Number of times the sigma of the map\
                              than which the max amplitude must be larger\
                              to assume a source is included.')
    parser.add_argument(
        '--coadd',
        action='store_true',
        dest='coadd',
        default=False,
        help='Co-add all maps of input direrctory. No MPI needed in\
                              in this case.')
    parser.add_argument('--pol',
                        action='store_true',
                        dest='pol',
                        default=False,
                        help='If False, only fit for temperature')
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
    parser.add_argument(
        '--init_dependence',
        action='store',
        type=float,
        dest='init_dependence',
        default=.01,
        help='Allowed maximum dependence on initial conditions')
    parser.add_argument('--theta_max',
                        action='store',
                        dest='theta_max',
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
                        type=int,
                        help='Downsampling factor if the full beam data')
    parser.add_argument('--stitch_f',
                        action='store',
                        dest='stitch_f',
                        type=int,
                        help='Number of data points to be used on each side of the wing \
                              scale as reference points for stitching core and wing fit')
    parser.add_argument('--lmax',
                        action='store',
                        dest='lmax',
                        type=int,
                        help='Maximum multipole number to take into account \
                        for the harmonic transform')
    parser.add_argument(
        '--make_plots',
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
    parser.add_argument('--dB_thresh',
                        action='store',
                        dest='dB_thresh',
                        required=False,
                        type=float,
                        help='Do not include profile data under dB_thresh\
                              for estimating the harmonic transform')
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
                        default='/mnt/so1/users/konstad/pwg-scripts_sp/pwg-bcp/sotodlib_staging/data/initial_parameters.yaml',
                        type=str,
                        help='Location of the initial parameters file')
    parser.add_argument('--r_noise',
                        action='store',
                        dest='r_noise',
                        type=float,
                        help='Percentage of the map max radius after which the beam is considered negligible--used for SNR estimation')
    parser.add_argument('--sigma',
                        action='store',
                        dest='sigma',
                        default=None,
                        help='Smoothing scale')
    parser.add_argument(
        '--scale',
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
    parser.add_argument('--skybox',
                        action='store',
                        dest='skybox',
                        type=float,
                        nargs='+',
                        default=None,
                        help='Ra/dec grid for resizing the map')

    args = parser.parse_args()

    return args


def main():

    args = get_parser()

    if not args.theta_max or not args.lmax or not args.downsample_f:
        tele, band = args.tele, args.band
        with open(args.initial_parameters_file, "r") as stream:
            config_file = yaml.safe_load(stream)
        band_idx = config_file['telescopes'][tele]['bands'].index(band)
        beamsize = config_file['telescopes'][tele]['beamsize'][band_idx]

        if not args.theta_max:
            args.theta_max = 8 * np.radians(beamsize / 60)
        if not args.lmax:
            args.lmax = int(180 / (beamsize / 60))
        if not args.downsample_f:
            args.downsample_f = config_file['telescopes'][tele]['downsample_f']

    if not args.r_noise:
        args.r_noise = 0.8

    r_noise = args.r_noise * args.theta_max

    all_kwargs = {'initial_parameters_file': args.initial_parameters_file,
                  'beamfile': args.beamfile,
                  'r_noise': r_noise,
                  'sigma': args.sigma,
                  'scale': args.scale,
                  'nsamps': args.nsamps,
                  'data_vol': args.data_vol,
                  }

    run_fit(tele=args.tele, band=args.band,
            map_files=args.map_files,
            map_path=args.map_path,
            query_param_file=args.query_param_file,
            rescale=args.rescale,
            smooth=args.smooth,
            pk_normalize=args.pk_normalize,
            init_params=args.init_params,
            coadd=args.coadd,
            skybox=args.skybox,
            threshold=args.threshold,
            pol=args.pol,
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
            stitch_f=args.stitch_f,
            make_plots=args.make_plots,
            save_stats=args.save_stats,
            write_beam=args.write_beam,
            outdir=args.outdir,
            prefix=args.prefix,
            dB_thresh=args.dB_thresh,
            **all_kwargs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import numpy as np
import lmfit
from lmfit import Model, Parameters
from pixell import enmap, enplot, wcsutils
import os
import matplotlib
import argparse as ap
matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt
import warnings
import healpy as hp
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.integrate import dblquad
from scipy.special import jv
import h5py 
from mpi4py import MPI

opj = os.path.join
INITIAL_PARA_FILE = '/mnt/so1/users/konstad/initial_parameters.hdf5'
beamdir = '/mnt/so1/users/konstad'

def read_enmaps(map_path, map_names, rescale, smooth, pk_normalise, **kwargs):
    """Read enmap and rescale/smooth if requested upon calling. Peak-normalise by 
    default"""

    enmaps = []
    for map_name in map_names:
        try:
            pmap = enmap.read_fits(opj(map_path,map_name))
        except:
            raise NameError('File not found')
        if len(pmap.shape)!=3:
            raise ValueError('enmap object should have len()==3')
        array_map, array_ra, array_dec = enmap_to_array(pmap) 
        if smooth:
            array_map = np.asarray(smooth_map(pmap, kwargs['sigma']))
        if rescale:
            array_map, array_ra, array_dec = rescale_coords([array_map, ra, dec], 
                                                 kwargs['scale'])
            pmap.posmap()[0], pmap.posmap()[1] = ra, dec
        if pk_normalise:
            array_map /= np.nanmax(array_map)
        pmap[0,:,:] = array_map
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

    rescaled_array = [x*scale for x in p_array]
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
        submap=enmap.submap(pmap, skybox, mode=None, wrap="auto", iwcs=None)
        return submap
    
    if skybox is None and pixbox is None:
        raise ValueError('map limits should be given either in physical or pixel\
                         coordinates')

def trim_multiple_maps(pmaps, pixbox=None, skybox=None, make_equal=False, **kwargs):
    """Function that trims the input maps to some user-specified
        limits or adjusts them to have the same size """
    
    ### to be updated to process different size maps after co-centering    
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
        if np.any([map_i.shape==pmaps[0].shape for map_i in pmaps])==False:
            min_grid = np.min([np.shape(map_i for map_i in pmaps)])
            lim_ra_pix, lim_dec_pix = int(min_grid[0]/2), int(min_grid[1]/2)
            pixbox = np.asarray([[-lim_ra_pix, -lim_dec_pix], 
                                 [lim_ra_pix, lim_dec_pix]])
             
            enmaps_trimmed = []
            for map_i in pmaps:
                skybox = pixbox2skybox(map_i.shape, map_i.wcs, pixbox)
                enmaps_trimmed.append(enmap.submap(pmap, skybox))
    return enmaps_trimmed

def coadd_maps(pmaps, trim_pmaps=False, **kwargs):

    """Coadd an array of maps to increase the flux

    Args
    ----
    maps: array of ndmap objects
    trim_pmaps: bool

    Returns
    -------
    The coadded map and averaged coordinate values
    """

    if trim_pmaps:
        trim_multiple_maps(pmaps, **kwargs)
    coadd_map = np.zeros_like((pmaps[0]))
    coadd_ra, coadd_dec = [np.zeros_like((pmaps)) for i in range(2)]  

    for t,map_i in enumerate(pmaps): 

        coadd_map += map_i

        coords = map_i.posmap()
        coadd_ra[t], coadd_dec[t] = coords[0], coords[1]
    return coadd_map, np.nanmean(coadd_ra, axis=0), np.nanmean(coadd_dec, axis=0)

def calculate_snr_single(pmap, r_noise):
    """Calculate the the Signal-to-Noise Ratio of a single
       map. The noise is estimated outside a radius r_noise"""
    
    ra, dec = pmap.posmap()[0], pmap.posmap()[1]
    r = np.sqrt(ra**2+dec**2)
    r = r.reshape(r.shape[-2], r.shape[-1])
    map_temp = pmap.reshape(pmap.shape[-2], pmap.shape[-1])
    neighbours = np.array(map_temp[np.where(r>r_noise)])
    rms = get_rms(neighbours)       
    peak_idx = np.unravel_index(np.nanargmax(map_temp.flatten()), np.shape(map_temp))    
    return map_temp[peak_idx]/rms 
    
def calculate_snr(pmaps, r_noise, n_obs=-1, n_max_sets=-1):
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

    if n_obs==len(maps):
        map_subsets = pmaps
    else:       
        map_idxs = list(itertools.combinations(np.arange(0, len(pmaps)-1), n_obs))
        
        snr_subset=[]
        for t,subset_idxs in enumerate(map_idxs[:n_max_sets]):
            map_subset = [maps[idx] for idx in subset_idxs]                  
            coadd_map, coadd_ra, coadd_dec = coadd_maps(map_subset) 
            r = np.sqrt(coadd_ra**2+coadd_dec**2)
            snr_subset.append(calculate_snr_single(coadd_map, r, r_noise=r_noise))
    snr_subset = np.array(snr_subset)
    return np.nanmean(snr_subset), np.nanstd(snr_subset)

def gaussian_1d(x, x0, sigx, **kwargs):
    """1D Gaussian function
    """
    
    if 'amp' in kwargs:
        amp = kwargs['amp']
    sigx2 = sigx**2

    return np.exp(-(x-x0)**2/(2*sigx2))

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
    except:
        amp = 1 
    sigx2 = sigx**2
    sigy2 = sigy**2
    x_rot = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
    y_rot = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
    xy_rot = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
    
    d2_rot_gauss = -x_rot*(x-x0)**2 - y_rot*(y-y0)**2 - 2*xy_rot*(x-x0)*(y-y0) 
    return amp*np.exp(d2_rot_gauss) 

def get_rms(x):
    """Estimate rms levels after gap-filling in x,y directions"""
    return np.sqrt(np.nanmean(x**2))

def airy_disc(x, y, amp, x0, y0, R, theta):
    """Airy pattern"""
    
    Rz = 1.2196698912665045 
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    z = np.pi*r/(R/Rz)
    j1 = scipy.special.jv(1, z)
    airy = (2 * j1/z)**2
    return amp*airy

def get_input_params(INITIAL_PARA_FILE, tele, band):
    """Get the initial parameters configuration for
    telescope 'tele' and frequency band 'band' 
    """
    
    hf = h5py.File(INITIAL_PARA_FILE, "r")

    if tele != "LAT":
        import re

        tele = re.split("(\d+)", tele)[0]
    try:
        idx = list(hf.get(tele)["frequency-band name"]).index(band)
    except:
        raise KeyError("Telescope name must be one of LAT,SAT")
    beamsize = (hf.get(tele)["beam size"])[idx]
    ## Band-centers to be extracted from passbands --fix in the future
    band_c = {'f030':27, 'f040':39, 'f090':93, 'f150':145, 'f230':225, 'f290':280}
    
    wlength = 3 / (band_c[band])
    wlength *= 1e-01
    d = {'SAT':0.42, 'LAT':6.0}
    R = 1.22*wlength/d[tele]
    init_params={'amp':1, 
                 'x0':1e-05, 
                 'y0':1e-05, 
                 'fwhm_x':np.radians(beamsize/60), 
                 'fwhm_y':np.radians(beamsize/60), 
                 'theta':1e-06,
                 'R': R}
    return init_params

def get_init_solid_angle(tele, beamdir, band):
    """Get an estimation for the solid angle of the input beam"""

    bdict = pickle.load(open(opj(beamdir, tele+'_'+ band+'_beam.pkl'),
                        'rb'), encoding='latin1')
    size = bdict['size']
    data_in = bdict['data']
    data_in /= np.nanmax(data_in)
    omega = data_in.sum()*np.radians(size[0][1])**2
    return omega

def get_input_br_wf(tele, beamdir, band, lmax):
    """Get an estimation for the input beam profile and 
       harmonic tramsform"""

    bdict = pickle.load(open(opj(beamdir, tele+'_'+ band+'_beam.pkl'),
                        'rb'), encoding='latin1')
    ru_in = np.radians(bdict['ru'])
    profile_in = bdict['profiles']
    wf = hp.beam2bl(profile_in/np.nanmax(profile_in), ru_in, lmax=lmax)
    return ru_in, bdict['profiles'], wf

def mask_source(pmap, r_mask):
    """Mask a region around the source or radius r_mask and gap-fill"""
    mask = np.ones((np.shape(pmap)))
    mask_idxs = np.where(np.sqrt(pmap.posmap()[0]**2+pmap.posmap()[1]**2)<r_mask)[0]
    mask[0][~mask_idxs]=0    
    return mask*pmap

def make_model_params(dependent_params, b_acc):
    """Make the lmfit parameter object for the fitting function"""
    
    params=Parameters()
    for idx_key, key in enumerate(dependent_params.keys()):
        key_value = dependent_params[key]
        params.add(key, value=key_value, 
                        min=-b_acc*key_value+key_value, 
                        max=b_acc*key_value+key_value)
    return params

def get_res(ra, dec):
    """Get map resolution"""
                                   
    c_ra = (ra.max()-ra.min())/ra.shape[0]
    c_dec = (dec.max()-dec.min())/dec.shape[1]
    res=np.mean([c_ra, c_dec])   
    return res

def compute_omega_1d(pmap, br, r, **arg_vals):
    """Integrate the raw/fitted beam profiles br over bins 
       r. **arg_vals should be parsed to the fitted function"""

    res = (r.max()-r.min())/len(r)
    if pmap is not None and arg_vals:
        Omega = gaussian_1d(r, np.sqrt(arg_vals['x0']**2+arg_vals['y0']**2), 
                        np.sqrt(arg_vals['sigx']*arg_vals['sigy'])).sum()*res**2
    else:
        Omega = br.sum()*res**2
    return Omega

def compute_omega_2d(pmap=None, **arg_vals):
    """Integrate the raw/fitted beam maps br over pmap.posmap() 
       coords. **arg_vals should be parsed to the fitted function""" 
    
    ra, dec = pmap.posmap()[0], pmap.posmap()[1]
    res = get_res(ra, dec)
    
    x0, y0, sigx, sigy, theta = [arg_vals[key] for key in ['x0',
                                                           'y0',
                                                           'sigx',
                                                           'sigy',
                                                           'theta']]
    if arg_vals:
        Omega = gaussian_2d_rot(x=np.asarray(ra).ravel(), 
                                y=np.asarray(dec).ravel(),
                                x0=x0,
                                y0=y0,
                                sigx=sigx,
                                sigy=sigy,
                                theta=theta).sum()*res**2
    else:
        Omega = np.asarray(pmap).sum()*res**2
    return Omega

def int_function(pars, lim_ra, lim_dec):
    """Integral of 2D gaussian (assumes already centred beam)"""

    sigx, sigy, theta, map_omega = pars
    ## To be computed with analytical expression in the future to avoid 
    #non-continuity error
    intgrl, abserr = dblquad(lambda x,y: np.exp(-(np.cos(theta)**2/(2*sigx**2) + 
                                         np.sin(theta)**2/(2*sigy**2))*(x**2) - 
                                        (np.sin(theta)**2/(2*sigx**2) + 
                                         np.cos(theta)**2/(2*sigy**2))*(y**2) - 
                                         2*(np.sin(2*theta)/(4*sigx**2) - 
                                         np.sin(2*theta)/(4*sigy**2))*x*y) - 
                                         map_omega, 
                                         -lim_ra, lim_ra, -lim_dec, lim_dec)  
    return abs(intgrl)

def fit_w_omega_constrain(pmap, init_params, acc):
    """Try to fit simultaneously beam parameters and solid angle
       allowing for acc (%) error"""
    
    pmap_arr, ra, dec = enmap_to_array(pmap) 
    lim_ra, lim_dec = np.abs(np.min([ra.min(),ra.max()])), \
                             np.abs(np.min([dec.min(),dec.max()]))
    
    fact = np.sqrt(8*np.log(2))
    fwhm_xin, fwhm_yin, input_theta, input_omega = [init_params[key] 
                                                    for key in ['fwhm_x',
                                                                'fwhm_y',
                                                                'theta',
                                                                'omega']]

    bound_xmin, bound_xmax = fwhm_xin-fwhm_xin*acc, \
                             fwhm_xin+fwhm_xin*acc
    bound_ymin, bound_ymax = fwhm_yin-fwhm_yin*acc, \
                             fwhm_yin+fwhm_yin*acc
    bound_omin, bound_omax = input_omega-input_omega*acc, \
                             input_omega+input_omega*acc
    bound_thetamin, bound_thetamax = input_theta-input_theta*acc, \
                                     input_theta+input_theta*acc
    bounds = ((bound_xmin/fact, bound_xmax/fact), 
              (bound_ymin/fact, bound_ymax/fact), 
              (bound_thetamin, bound_thetamax), 
              (bound_omin, bound_omax))

    # make discrete instead of continuous integration (error depends on resolution)
    res = minimize(int_function, 
                   x0=[fwhm_xin/fact, fwhm_yin/fact, input_theta, input_omega], 
                   args=(lim_ra, lim_dec), 
                   bounds=bounds, 
                   method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': True})

    sigx, sigy, theta, omega = res.x
    fitted_params={'sigx':sigx,
                   'sigy':sigy,
                   'theta':theta,
                   'omega':omega,
                   'fwhm_x':fact*sigx,
                   'fwhm_y':fact*sigy,
                   'fwhm':fact*np.sqrt(sigx*sigy)}
    return fitted_params

def radial_profile(pmap, bin_size=0.001, **arg_vals):
    """Make a radial profile of pmap with bins of 
    size bin_size"""
    
    if arg_vals:
        ra, dec = pmap.posmap()[0], pmap.posmap()[1]
        x0, y0, amp, R, theta = [arg_vals[i] for i in ['x0',
                                                       'y0',
                                                       'amp',
                                                       'R',
                                                       'theta']]
        airy_sim = airy_disc(ra.ravel(), 
                             dec.ravel(), 
                             amp, 
                             x0, 
                             y0, 
                             R, 
                             theta)        
        wcs = wcsutils.tan([0,0], res=pmap.wcs.wcs.cdelt, shape=pmap[0].shape)
        sim_enmap = enmap.zeros(pmap.shape, wcs, dtype=np.float64)
        sim_enmap[0,:,:] = airy_sim.reshape(np.shape(pmap)[-1], np.shape(pmap)[-2])
        return enmap.rbin(sim_enmap, bsize=bin_size)
    else:
        return enmap.rbin(pmap, bsize=bin_size)

def harm_transform(br, r, lmax=1000):
    """Calculate harmonic transform of beam profile
       br evaluated on r bins. Output is of shape lmax+1."""

    from pixell import curvedsky
    return hp.beam2bl(br, r, lmax=lmax)
# uncomment for simons1
#     return curvedsky.profile2harm(br, r, lmax=lmax)

def plot_profile(r_in, br_in, r, br, errors_r=None, errors_b=None, out_dir=None, 
                 full_file_name=None):
    """Plot the beam profile and compare with input"""
    
    plt.figure()
    br /= np.nanmax(br)
    plt.plot(r, br, label='Fitted')
    plt.plot(r_in, np.abs(br_in), label='Input', ls='--')

    if errors_b is not None:
        plt.errorbar(r[::10], br[::10], yerr=errors_b[::10]/2, fmt='.', 
                     ls='none', capsize=4, label='errors')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Beam Power (dB)')
    plt.autoscale(tight=True)
    plt.xlim(0,np.min([r.max(), r_in.max()]))
    plt.legend()
    plt.savefig(opj(out_dir, full_file_name+'_profile.png'))
    plt.close()
    return
    
def plot_wf(wf_in, wf, errors_wf=None, calibrate=True, out_dir=None, 
            full_file_name=None, lmax=None, crange=None):
    """ Plot window function after calibrating on input"""
    
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[2,1]})                            
    ell_in, ell = np.arange(0,len(wf_in)), np.arange(0, len(wf))
    wf_in /= np.nanmax(wf_in)
    wf /= np.nanmax(wf)
    
    if calibrate:
        cmin, cmax = crange
        cal_fact = np.average(wf_in[cmin:cmax]) / np.average(wf[cmin:cmax])
        wf *= cal_fact
        
    ax1.plot(ell_in, wf_in, label='input', ls='--', color='k')
    ax1.plot(ell, wf, label='fitted')
    if errors_wf is not None:
        ax1.errorbar(ell[::50], wf[::50], yerr=errors_wf[::50]/2, fmt='.', 
                    ls='none', capsize=4, label='errors')
    ax1.set_xlabel('$\ell$')
    ax1.set_ylabel('$B_{\ell}$')
    ax1.legend()
    ax1.autoscale(tight=True)

    ax2.plot(ell, np.abs(wf/wf_in)-1, label='error')
    ax2.hlines(xmin=ell.min(), xmax=ell.max(), y=0, colors='k', linestyles='dashed')
    
    if errors_wf is not None:
        ax2.errorbar(ell[::50], np.abs(wf[::50]/wf_in[::50])-1, 
                     yerr=np.abs((errors_wf[::50]/2)/wf_in[::50]), 
                     fmt='.', ls='none', capsize=4, label='errors')
        
    ax2.set_xlim(0, lmax/2) 
    ax2.set_ylim(-.05, .05)
    ax2.set_xlabel('$\ell$')
    ax2.set_ylabel('$B_{\ell}/B_{\ell}^{in} - 1$')
    fig.tight_layout()
    plt.savefig(opj(out_dir, full_file_name+'_wf.png'))
    plt.close()
    return

def plot_residuals(pmap, fitted_params, out_dir=None, full_file_name=None):
    """Plot residuals between true and fitted map using enplot"""
    
    ## Update to parse plotting options as kwargs
    x0, y0, amp, R, theta = [fitted_params[i] for i in ['x0',
                                                        'y0',
                                                        'amp',
                                                        'R',
                                                        'theta']]
    airy_sim = airy_disc(pmap.posmap()[0].ravel(), 
                         pmap.posmap()[1].ravel(), 
                         amp, 
                         x0, 
                         y0, 
                         R, 
                         theta)
    wcs = wcsutils.tan([0,0], res=pmap.wcs.wcs.cdelt, shape=pmap[0].shape)
    sim_enmap = enmap.zeros(pmap.shape, wcs, dtype=np.float64)
    sim_enmap[0,:,:] = airy_sim.reshape(np.shape(pmap)[-1], np.shape(pmap)[-2])
    plot = enplot.plot(pmap-sim_enmap, grid=False, colorbar=True)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    enplot.write(opj(out_dir, full_file_name+'.png'), plot)
    
    return

def fit_beam_params(pmap=None, init_params=None, Nmax=None, test_init_cond=False, 
                    acc=.05, init_dependence=.01, **kwargs):
    """Fit for the beam parameters (and optionally centre position) of a map 
       using scipy. First we fit an airy pattern to find the first dark ring.
       Then we mask around it and find beamsize in x,y by fitting a Gaussian.

    Args
    ----
    pmap: ndmap object
    init_params [radians]: initial values for the fit [could be also drawn from 
                           a brute force search]
    Nmax: maximum number of allowed iterations of least squares minimization
    test_init_cond: Perturb initial fit values and check the impact on the fit
                    as a user specified percentage as 'init_dependence'
    acc: the accuracy the fitting should achieve (default 5%)
    
    Returns
    -------
    beam parameters and weight, if requested
    """
    
    array_map, ra, dec = enmap_to_array(pmap)          

    amp, x0, y0, R, theta = [init_params[key] for key in ['amp',
                                                          'x0',
                                                          'y0',
                                                          'R',
                                                          'theta']]                         
    error = np.sqrt(array_map.ravel()+1)
    
    fmodel = Model(airy_disc, independent_vars=('x','y'))
    params = make_model_params(dependent_params={'amp':amp, 
                                                 'x0':x0, 
                                                 'y0':y0, 
                                                 'R':R, 
                                                 'theta':theta}, 
                                                 b_acc=2*acc)
    
    result_a = fmodel.fit(data=array_map.ravel(), 
                          x=ra.ravel(), 
                          y=dec.ravel(), 
                          params=params, 
                          weights=1/error)
    fitted_values_a = result_a.best_values
    amp, x0, y0, R, theta = fitted_values_a.values()
    
    m_pmap = mask_source(pmap, R)
    array_map, ra, dec = enmap_to_array(pmap)
    fact = np.sqrt(8*np.log(2))
    sigx, sigy = init_params['fwhm_x']/fact, init_params['fwhm_y']/fact 

    fmodel = Model(gaussian_2d_rot, independent_vars=('x','y'))
    params = make_model_params(dependent_params={'x0':x0, 
                                                 'y0':y0, 
                                                 'sigx':sigx, 
                                                 'sigy':sigy, 
                                                 'theta':theta}, 
                               b_acc=acc)
    result_g = fmodel.fit(array_map.ravel(), x=ra.ravel(), y=dec.ravel(), 
                        params=params, weights=1/error, max_nfev=Nmax)    
    fitted_values_g = result_g.best_values  
  
    fwhm_x, fwhm_y = fitted_values_g['sigx']*fact, fitted_values_g['sigy']*fact
    fwhm = np.sqrt(8*np.log(2)*fitted_values_g['sigx']*fitted_values_g['sigy'])
    ell = np.abs(fwhm_x-fwhm_y)/(fwhm_x+fwhm_y)
    fitted_values_cp = dict()
    fitted_values_cp['fwhm_x'] = fwhm_x
    fitted_values_cp['fwhm_y'] = fwhm_y
    fitted_values_cp['fwhm'] = fwhm
    fitted_values_cp['R'] = R
    fitted_values_cp['amp'] = amp
    fitted_values_cp['ellipticity'] = ell
    
    stat_w = 1/(result_a.chisqr + result_g.chisqr)
    fitted_values_cp['stat_w'] = stat_w
        
    fitted_values_g.update(fitted_values_cp)
    
    if test_init_cond:
        fluctuation = np.random.uniform(-2*acc,2*acc,len(init_params.values()))
        params_perturbed = dict()
        for i,key in enumerate(init_params.keys()):
            params_perturbed[key] = init_params[key]+fluctuation[i]*init_params[key]

        fwhm_perturbed = fit_beam_params(pmap, 
                                         init_params=params_perturbed, 
                                         Nmax=None,
                                         test_init_cond=False, 
                                         acc=acc, 
                                         init_dependence=init_dependence,
                                         **kwargs)['fwhm']

        if np.abs(fwhm_perturbed/fwhm-1) < init_dependence:
            pass
        else:
            warnings.warn('Dependency on initial conditions over {} %'
                          .format(init_dependence*100))
    
    return fitted_values_g
                             
def fit_single_map(tele, pmap, init_params, r_noise, Nmax, test_init_cond, acc,
                   init_dependence, plot_res, plot_beam, lmax, **kwargs):
    """Fit beam parameters and solid angle for a map, calculate its beam profile 
    and window function."""

    snr = calculate_snr_single(pmap, r_noise)

    fitted_params=fit_beam_params(pmap,
                                  init_params=init_params, 
                                  Nmax=Nmax, 
                                  test_init_cond=test_init_cond, 
                                  acc=acc, 
                                  init_dependence=init_dependence,
                                  **kwargs)
    if 'stat_w' in fitted_params:
        fitted_params.update({'stat_w':fitted_params['stat_w']+1/snr})
    if plot_res:
        plot_residuals(pmap, fitted_params, kwargs['out_dir'], kwargs['full_file_name'])
        
    solid_angle = compute_omega_2d(pmap, **fitted_params)
    fitted_params.update({'omega':solid_angle})
    
    omega_init = get_init_solid_angle(tele, beamdir, kwargs['band'])

    # Allowing for a slightly larger error -- to be discussed more
    if np.abs(solid_angle/omega_init)-1>2*acc:
        warnings.warn('Failed to compute omega within {}% accuracy'.format(2*acc*100))
        try:
            init_params_c = fitted_params.copy()
            init_params_c.update({'omega':omega_init})
            c_fitted_params = fit_w_omega_constrain(pmap, init_params_c, acc=.01)
            fitted_params.update(c_fitted_params)
        except:
            pass
        
    br, r = radial_profile(pmap, bin_size=kwargs['bsize'], **fitted_params)
    br = br.reshape(br.shape[-1])
    r = r[np.where(np.isfinite(br))]
    br = br[np.where(np.isfinite(br))]
    bl = harm_transform(br, r, lmax=lmax)
    
    return fitted_params, [br,r], [bl, np.arange(0,len(bl)+1)] 

def run_fit(tele, map_path, map_names, rescale, smooth, pk_normalise, trim_pmaps, 
            r_noise, Nmax, init_params, test_init_cond, acc, init_dependence,
            plot_res, plot_beam, save_stats, **kwargs): 
    """Read one more maps, fit the beam parameters and solid angle, plot the 
       results distributions, create beam profiles and window functions
       and store the results in .h5 format"""
    
    try:
        import lmfit
        from lmfit import Model, Parameters
    except:
        raise ImportError('module lmfit is not imported')
    ##Decide on parallelization scheme
    enmaps = read_enmaps(map_path, 
                         map_names, 
                         rescale, 
                         smooth, 
                         pk_normalise, 
                         **kwargs)

    if init_params is None:
        try:
            init_params = get_input_params(INITIAL_PARA_FILE, tele, kwargs['band'])
        except:
            raise ValueError('Missing arguments')
            
    beamsize = np.sqrt(init_params['fwhm_x']*init_params['fwhm_y'])
    
    if trim_pmaps:
        # will make this user-specified
        ## trim maps a bit -- 4*beamsize -- change this so it's not hardcoded
        map_lim = 4*beamsize
        enmaps = trim_multiple_maps(enmaps, skybox=[[-map_lim,-map_lim], 
                                                    [map_lim,map_lim]])
    # get from beam resolution
    lmax = int(2*np.pi/beamsize)

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
               'ellipticity',
               'fwhm_x', 
               'fwhm_y', 
               'theta', 
               'omega',
               'stat_w']
   
    df = pd.DataFrame(columns=df_keys)

    if rank==0:
        r = enmaps[0].modrmap(ref=[0,0])
        n = int(np.max(r/kwargs['bsize']))

        profiles = np.zeros((len(enmaps), 2, n-1))
        harmonic_t = np.zeros((len(enmaps), 2, lmax+1))
    else:
        profiles, harmonic_t = None, None

    profiles = comm.scatter(profiles,0)
    harmonic_t = comm.scatter(harmonic_t, 0)

    for map_idx in batch:
        fitted_params, [br,r], [bl, ell] = fit_single_map(tele=tele,
                                                          pmap=enmaps[map_idx], 
                                                          init_params=init_params,
                                                          r_noise=r_noise, 
                                                          Nmax=Nmax, 
                                                          test_init_cond=test_init_cond, 
                                                          acc=acc, 
                                                          init_dependence=init_dependence, 
                                                          plot_res=plot_res, 
                                                          plot_beam=plot_beam, 
                                                          lmax=lmax, 
                                                          **kwargs)                                                   

        df.loc[map_idx, :] = np.array([fitted_params[key] for key in df_keys])
        df.loc[map_idx, 'map_name'] = map_names[map_idx]

        profiles = [r, br]
        harmonic_t = bl
        
    all_profiles = comm.gather(profiles, root=0)
    all_harmonic_t = comm.gather(harmonic_t, root=0)

    all_dfs = comm.gather(df, root=0)

    if rank == 0:
        stats = pd.concat(all_dfs)
        stats = stats.set_index(stats['map_name']) 
        stats = stats.drop(columns=['map_name'])       

        if plot_beam:
            r_in, br_in, bl_in = get_input_br_wf(tele, 
                                                 beamdir, 
                                                 kwargs['band'],
                                                 lmax)        
            weights = stats['stat_w'].values

            average_profile = np.average(all_profiles, weights=weights, axis=0)
            profile_errors = np.average((all_profiles-average_profile)**2, 
                                         weights=weights, axis=0)
            plot_profile(r_in=r_in, 
                         br_in=br_in, 
                         r=average_profile[0], 
                         br=average_profile[1],
                         errors_b = profile_errors[1],
                         out_dir = kwargs['out_dir'],
                         full_file_name=kwargs['full_file_name']
                    )

            average_htransform = np.average(all_harmonic_t, weights=weights, axis=0)
            htransform_errors = np.average((all_harmonic_t-average_htransform)**2, 
                                            weights=weights, axis=0)
            plot_wf(wf_in=bl_in, 
                    wf=average_htransform, 
                    errors_wf=htransform_errors,
                    out_dir = kwargs['out_dir'],
                    full_file_name=kwargs['full_file_name'],
                    lmax=lmax,
                    crange=kwargs['crange'])

    if save_stats and rank==0:
        stats.to_hdf(opj(kwargs['out_dir'], kwargs['h5_name'] + ".h5"), key="beam_parameters", mode="w")
       
    return 

def main():

    parser = ap.ArgumentParser(formatter_class=\
    ap.ArgumentDefaultsHelpFormatter)

    ## Those shall mostly be replaced with configuration files 
    parser.add_argument('--tele', 
                        action='store', 
                        dest='tele',
                        default='SAT', 
                        type=str, 
                        help='Telescope [SAT,LAT]')
    parser.add_argument('--map_path', 
                        action='store', 
                        dest='map_path',
                        default=None, 
                        type=str, 
                        help='Location of the maps')
    parser.add_argument('--map_names', 
                        action='store', 
                        dest='map_names',
                        default=None, 
                        type=str, 
                        nargs='+', 
                        help='Location of the maps')
    parser.add_argument('--rescale', 
                        action='store_true', 
                        dest='rescale',
                        default=False, 
                        help='If True, rescale the maps')
    parser.add_argument('--smooth', 
                         action='store_true', 
                        dest='smooth',
                        default=False, 
                        help='If True, smooth the maps')
    parser.add_argument('--pk_normalise', 
                        action='store_true', 
                        dest='pk_normalise',
                        default=True, 
                        help='If True, peak-normalise the maps')
    parser.add_argument('--trim_pmaps', 
                        action='store_true', 
                        dest='trim_pmaps',
                        default=True, 
                        help='If True, trim the maps to some indicated size')
    parser.add_argument('--r_noise', 
                        action='store', 
                        dest='r_noise',
                        default=np.radians(4.2), 
                        type=float,
                        help='Radius outside of which to evaluate noise')
    parser.add_argument('--Nmax', 
                        action='store', 
                        dest='Nmax',
                        default=None, 
                        help='Maximum allowed number of iterations in the fitting')
    parser.add_argument('--init_params', 
                        action='store', 
                        dest='init_params',
                        default=None, 
                        help='Initial parameters to be used for the fitting')
    parser.add_argument('--test_init_cond', 
                        action='store_true', 
                        dest='test_init_cond',
                        default=False, 
                        help='Test fitting result dependency on initial parameters')
    parser.add_argument('--acc', 
                        action='store', 
                        dest='acc',
                        default=.05, 
                        help='Accuracy requirement')
    parser.add_argument('--init_dependence', 
                        action='store', 
                        dest='init_dependence',
                        default=.01,
                        help='Allowed maximum dependence on initial conditions')
    parser.add_argument('--plot_res', 
                        action='store_true', 
                        dest='plot_res',
                        default=False, 
                        help='Plot residual maps between data and simulations')
    parser.add_argument('--plot_beam', 
                        action='store_true', 
                        dest='plot_beam',
                        default=False, 
                        help='Plot beam profile and harmonic transform')
    parser.add_argument('--save_stats', 
                        action='store_true', 
                        dest='save_stats',
                        default=False, 
                        help='Save fitted parameters table as .h5 file')
    parser.add_argument('--all_kwargs', 
                        action='store', 
                        dest='all_kwargs',
                        default={'band':'f090', 
                                 'full_file_name':'map_0', 
                                 'out_dir':'./beam_analysis_results', 
                                 'bsize':0.0001, 
                                 'crange':[50,200], 
                                 'h5_name':'analysis_stats'},
                                 help='Args, to be parsed as a dictionary')
    args = parser.parse_args()


    run_fit(tele=args.tele,
            map_path=args.map_path, 
            map_names=args.map_names, 
            rescale=args.rescale, 
            smooth=args.smooth, 
            pk_normalise=args.pk_normalise, 
            trim_pmaps=args.trim_pmaps, 
            r_noise=args.r_noise, 
            Nmax=args.Nmax, 
            init_params=args.init_params, 
            test_init_cond=args.test_init_cond, 
            acc=args.acc, 
            init_dependence=args.init_dependence, 
            plot_res=args.plot_res, 
            plot_beam=args.plot_beam, 
            save_stats=args.save_stats, 
            **args.all_kwargs)

if __name__ == "__main__":
    main()

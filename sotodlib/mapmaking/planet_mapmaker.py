__all__ = ['planet_mapmake_eachobs']
import os, time, gc, yaml, datetime, re, json, copy, pickle, math
import numpy as np
from sqlalchemy import create_engine, exc
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker
from typing import Optional
import requests
from io import StringIO
from scipy.optimize import curve_fit
from functools import partial
from scipy.special import eval_hermite
from scipy.optimize import curve_fit

import so3g
from pixell import enmap, reproject

from .. import coords, core, preprocess, io, tod_ops
from ..site_pipeline.utils import logging
#from sotodlib.site_pipeline.utils import logging

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret


def FWHM2sigma(FWHM):
    fac = np.sqrt(8*np.log(2))
    return FWHM/fac

def expand(obj, vars):
    """
    Function to expand the variables in the configuration file.
    Args:
        obj: Object to expand.
        vars: Variables to expand.
    Returns:
        obj: Expanded object.
    """
    if isinstance(obj, str):
        try:
            return obj.format(**vars)
        except (KeyError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {k: expand(v, vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v, vars) for v in obj]
    return obj

def read_configs(config_path):
    """Function to read the configuration file with expanding variables.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        cfg (dict): Config file after expanding.
    """
    configs = yaml.safe_load(open(config_path, "r"))
    cfg = configs.copy()
    for _ in range(3):
        cfg = expand(cfg, cfg)
    return cfg

def get_inv_var(configs, aman, full, logger, uniform=False):
    if not uniform:
        ipipe = configs['mapmaking']['inv_var']
        if ipipe['signal'] == 'demodQU':
            logger.info('Using demodQU signal for inverse variance calculation')
            wnq = np.array([full.noiseQ_nofit.white_noise[idet == full.dets.vals][0] for idet in aman.dets.vals])        
            wnu = np.array([full.noiseU_nofit.white_noise[idet == full.dets.vals][0] for idet in aman.dets.vals])
            wn = np.sqrt(wnq**2 + wnu**2)
        elif ipipe['signal'] == 'demodQ':
            logger.info('Using only demodQ signal for inverse variance calculation')
            wn = np.array([full.noiseQ_nofit.white_noise[idet == full.dets.vals][0] for idet in aman.dets.vals])
        elif ipipe['signal'] == 'demodU':
            logger.info('Using only demodU signal for inverse variance calculation')
            wn = np.array([full.noiseU_nofit.white_noise[idet == full.dets.vals][0] for idet in aman.dets.vals])
        else:
            raise ValueError(f"Unknown signal type: {ipipe['signal']}")
        aman.wrap('inv_var', wn**(-2), [(0, 'dets')])
    else:       
        aman.wrap('inv_var', np.ones(aman.dets.count), [(0, 'dets')])
    return 

def planet_mapmake_eachobs(config_path, obs_id, wafer_info, verbosity = 3, debug = False):
    """
    Function to process planet_mapmaking for each observation.
    Args:
        config_path (str): Path to the configuration file.
        obs_is (str): Observation ID.
        wafer_info (dict): Wafer information including wafer slot and bandpass. e.g., {'wafer_slot': wafer, 'wafer.bandpass': band}
    Returns:
        dbinfo: Database information object.
    """
    logger = logging.init_logger("planet_mapmaking", verbosity=verbosity)
    logger.info(f"[PID {os.getpid()}] Running, {obs_id}, {wafer_info}, starttime = {time.perf_counter()}")

    configs = read_configs(config_path)
    context = core.Context(configs["context_file"])

    logger.info("Loading meta data")
    meta = context.get_meta(obs_id, dets = wafer_info)

    # To avoid to run observation of bad date
    #if bool_bad_date(meta.obs_info.start_time, meta.obs_info.stop_time, meta.obs_info.telescope):
    #    logger.warning(f"Bad date: {obs_id}")
    #    return 

    if debug:
        meta.restrict('dets', meta.dets.vals[200:400]) # for debug
    else:
        meta.restrict('dets', meta.dets.vals)
    logger.info(f"Loading preprocess pipeline data, {obs_id}, {wafer_info}")
    pipe = preprocess.pcore.Pipeline(configs["process_pipe"], logger=logger)
    logger.info("Run the pipeline")
    aman = context.get_obs(meta)
    full, _ = pipe.run(aman)
    logger.info("Finish the pipeline")

    # get inverce variance
    get_inv_var(configs, aman, full, logger=logger)
    
    # Fit TOD
    if configs['mapmaking']['fittod']['process']:
        logger.debug('Execute to fit TOD')
        if isinstance(configs['mapmaking']['fittod']['r_use'], str):
            configs['mapmaking']['fittod']['r_use'] = eval(configs['mapmaking']['fittod']['r_use'])
        if isinstance(configs['mapmaking']['fittod']['r_use'], str):
            configs['mapmaking']['fittod']['r_fit'] = eval(configs['mapmaking']['fittod']['r_fit'])
        if isinstance(configs['mapmaking']['fittod']['mask_res'], str):
            configs['mapmaking']['fittod']['mask_res'] = eval(configs['mapmaking']['fittod']['mask_res'])
        execute_todfit(aman, dbpath=configs['mapmaking']['fittod'].get('dbpath'), center_on=configs['mapmaking']['map'].get('source'), 
                       Tsignal=configs['mapmaking']['fittod'].get('Tsignal', 'dsT'), flags=configs['mapmaking']['fittod'].get('flags', None), centroid_position=(0,0),
                       r_use=configs['mapmaking']['fittod']['r_use'], r_fit=configs['mapmaking']['fittod']['r_fit'], res=configs['mapmaking']['fittod']['mask_res'],
                       max_pix=4000000, detrend=configs['mapmaking']['fittod'].get('detrend_eachscan', True), npoly=configs['mapmaking']['fittod'].get('npoly', 0),
                       defl_model=configs['mapmaking']['fittod'].get('deflection_model', True), imask_use=None, imask_fit=None, logger=logger)
    
    # apply todfit based cut
    if configs['mapmaking']['fittod']['fitselection']:
        logger.info('Apply todfit based selection')
        numdet_b_fitsel = apply_todfit_selection(aman,toddbpath=configs['mapmaking']['fittod']['dbpath'],
                                                 thoreshold_path=configs['mapmaking']['fittod']['threshold_path'],
                                                 logger=logger)
    else:
        numdet_b_fitsel = 0

    # if mapping the PCA modes, re-assign the aman.signal, aman.dsT, aman.demodQ, aman.demodU here
    #has to be after calculating the inverse variance, so that weighting in maps is the same as in the actual maps
    if configs['mapmaking'].get('map_PCA',False): #default is False
        logger.info('Replacing aman.signal, aman.dsT, aman.demodQ, aman.demodU with the calculated PCA modes')
        for signame in ['signal', 'dsT', 'demodQ', 'demodU']:
            modes,weights = aman[f'pca_model_{signame}'].modes, aman[f'pca_model_{signame}'].weights
            aman[signame] = weights @ modes
            aman[signame] = aman[signame].astype(np.float32)
        logger.info('Finished replacing aman.signal, aman.dsT, aman.demodQ, aman.demodU with the calculated PCA mode signal')

    # Make maps
    logger.debug(configs['mapmaking']['map']['coordinate'])
    if configs['mapmaking']['map']['coordinate'] in ['detector_center', 'boresight_center']:
        logger.debug('instrument centered coordinate')
        dets_used, yc, xc, map_vars = make_instrument_center(aman, configs['mapmaking'], logger=logger)
    elif configs['mapmaking']['map']['coordinate'] in ['planet_horizon', 'planet_equatorial']:
        logger.debug('planet centered coordinate')
        dets_used, yc, xc, map_vars = make_planet_center(aman, configs['mapmaking'], logger=logger)
    else:
        raise ValueError("Unknown coordinate system: {}. Must be one of ['detector_center', 'boresight_center', 'planet_horizon', 'planet_equatorial']".format(configs['mapmaking']['map']['coordinate']))
    
    # Make database
    keys = []
    valids = []
    for ikey in full.keys():
        try:
            ivalid = np.all(full[ikey].valid.mask(), axis = 1)
            valids.append(ivalid)
            keys.append(ikey)
        except:
            pass
    dbinfo = make_info(aman)
    dbinfo.detnum_before_fitselection = numdet_b_fitsel
    dbinfo.total_detnum = len(dets_used)
    dbinfo.detid = list(dets_used)
    dbinfo.recenter = configs['mapmaking']['map'].get('recenter', False)
    dbinfo.yc = yc
    dbinfo.xc = xc
    dbinfo.proc = keys
    dbinfo.detnum = [len(np.where(idetid)[0]) for idetid in valids]
    dbinfo.Tmap_variance = map_vars[0]
    dbinfo.Qmap_variance = map_vars[1]
    dbinfo.Umap_variance = map_vars[2]
    logger.info(f"Finish the map making, {obs_id}, {wafer_info}")
    logger.info(f"[PID {os.getpid()}] Finished {obs_id}, {wafer_info} endtime = {time.perf_counter()}")
    del aman, full
    gc.collect()
    return dbinfo

def make_planet_center(aman, config, logger, rot_q=None, debug = False, fits_name=None):
    '''
    Function to make Q/U maps of a slow moving source (i.e. not fixed on the celestial sphere). NOTE: demodulation must have been done to use this function. Unlike the above mapping function, this horizon version creates a projection matrix with 
    all detectors in horizon coordinates, so that planets and other structures attached to the instrument do not rotate with the celestial sphere.

    Args:
        aman: axis manager
        config: "mapmaking" part of config file.
        logger: logger object
        rot_q: rotation quaternion to apply sight.Q. This is mainly for simlating planet signal
        debug: debug mode. Returns aman and result if True
    Returns:
        detids: list of detector ids
        yc, xc: peak position of Tmap
        map_vars: variance of maps calculated within from var_minr to var_maxr.
    '''
    obsinfo = get_obsinfo(aman)
    iobsid = obsinfo['obsid']
    itele = obsinfo['telescope']
    iband = obsinfo['band']
    iws = obsinfo['ws']
    isite = obsinfo['site']

    if isinstance(config['map']['res'], str):
        config['map']['res'] = eval(config['map']['res'])
    if isinstance(config['map']['size'], str):
        config['map']['size'] = eval(config['map']['size'])
    box = np.array([[-1, -1], [1, 1]]) * config['map']['size'] * coords.DEG
    geom = enmap.geometry(pos=box, res=config['map']['res'] * coords.DEG, proj = config['map']['proj'])

    # calculate planet in horizon coordinates (az/el)
    if config['map']['coordinate'] == 'planet_horizon':
        paz, pel = coords.planets.calc_planet_azel_approx(aman.timestamps, source = config['map']['source'], site = isite)
        pq = so3g.proj.quat.rotation_lonlat(-paz, pel)
        sight = so3g.proj.CelestialSightLine.for_horizon(aman.timestamps, aman.boresight.az, aman.boresight.el, roll=aman.boresight.roll)
    elif config['map']['coordinate'] == 'planet_equatorial':
        pra, pdec = coords.planets.calc_planet_radec_approx(aman.timestamps, source = config['map']['source'])
        pq = so3g.proj.quat.rotation_lonlat(pra, pdec)
        sight = so3g.proj.CelestialSightLine.az_el(aman.timestamps, aman.boresight.az, aman.boresight.el, roll=aman.boresight.roll,
                                                    weather='typical', site=isite)
    else:
        raise ValueError(f"Unknown coordinate system: {config['map']['coordinate']}")

    if rot_q is not None:
        #rotate boresight to aubitary frame. This is mainly for trasfer function simulation
        sight.Q = sight.Q * ~rot_q

    if config['deflection_correction']['process'] & config['deflection_correction']['wafer_base']:
        logger.debug('Deflection correction is done each wafer')
        if hasattr(aman, "wobble_params") and aman.wobble_params:
            logger.debug("Use wobble params stored in metadata")
            params = (aman.wobble_params['amp'][0], aman.wobble_params['phase'][0])
            deflq = get_defl_quat(aman, params=params)
        else:
            logger.debug('No wobble params found in metadata. Using default deflection parameters.')
            deflq = get_defl_quat(aman, iband, iws, telescope=itele)
        sight.Q = sight.Q * ~deflq
        
    # select detectors based on fitted values or fit False
    if config['fittod']['process']:
        defl_params = []
        bls = []
        for i, idetid in enumerate(aman.det_info.det_id):
            try:
                if config['xieta_correction']['process']:
                    fitinfo = get_db_planettod(config['fittod']['dbpath'], obs_id = iobsid, freq_channel = iband, wafer = iws, detid = idetid)[0]
                    aman.focal_plane.xi[i] += fitinfo.xo
                    aman.focal_plane.eta[i] += fitinfo.yo

                if config['deflection_correction']['process'] and not config['deflection_correction']['wafer_base']:
                    fitinfo = get_db_planettod(config['fittod']['dbpath'], obs_id = iobsid, freq_channel = iband, wafer = iws, detid = idetid)[0]
                    params = (fitinfo.defla, fitinfo.deflp)
                    if np.isnan(fitinfo.defla) or np.isnan(fitinfo.deflp):
                        bls.append(False)
                        defl_params.append((np.nan, np.nan))
                    else:
                        bls.append(True)
                        defl_params.append(params)
                else:
                    bls.append(True)
            except Exception as e:
                bls.append(False)
                defl_params.append((np.nan, np.nan))
        logger.info(f'Number of detectors is reduced based on todfit: {aman.dets.count} --> {len(np.where(bls)[0])}')
        logger.info(f'len(defl_params) = {len(defl_params)}')
        aman = aman.restrict('dets', np.array(bls), in_place=True)
        defl_params = np.array(defl_params)[np.array(bls)]
        logger.info(f'len(defl_params) = {len(defl_params)}')

    rot = so3g.proj.quat.rotation_lonlat(0, 0)
    # reciver coordinates sysetm or not for planet_horizon
    if (config['map']['coordinate'] == 'planet_horizon'):
        if config['map'].get('recv_coords'):
            logger.debug('Using recv_coords on horizon coordinates')
            rot = so3g.proj.quat.rotation_lonlat(0, 0) * so3g.proj.quat.euler(2, -aman.boresight.roll)

    if config['deflection_correction']['process'] and not config['deflection_correction']['wafer_base']:
        # apply deflection correction per detector
        results = []
        detids = []
        for i, idetid in enumerate(aman.det_info.det_id):
            iaman = aman.restrict('dets', aman.det_info.det_id == idetid, in_place=False)
            isight = copy.copy(sight) # need to copy sight for each detector because it is modified in map_make_detcen_horizon_each
            deflq = get_defl_quat(aman, params = defl_params[i])
            isight.Q = rot * ~pq * isight.Q * ~deflq
            P = coords.P.for_tod(iaman, sight=isight, geom=geom, hwp=True, comps='TQU', cuts=iaman.flags[config['map']['flags']])
            iresult = coords.demod.make_map(iaman, P=P, dsT=iaman[config['map']['Tsignal']], det_weights_demod=iaman.inv_var)
            iresult['hit'] = P.to_map(tod=iaman, signal=np.ones(iaman[config['map']['Tsignal']].shape, dtype = np.float32), det_weights=None, comps='T')[0]
            results.append(iresult)
            detids.append(idetid)
            if config['map']['single_save']:
                # save individual detector map
                direeach = os.path.join(config['map']['save_dire'], f'each_det/{iobsid}/{idetid}/')
                save_results(iresult, iobsid, iband, iws, direeach, fits_name=fits_name)
        result = coadd_maps(results)
    else:
        # not apply deflection correction
        sight.Q = rot * ~pq * sight.Q
        if config['map']['single_save']:
            results = []
            detids = []
            for i, idetid in enumerate(aman.det_info.det_id):
                iaman = aman.restrict('dets', aman.det_info.det_id == idetid, in_place=False)
                isight = copy.copy(sight) # might not need to copy because we use the same isight
                P = coords.P.for_tod(iaman, sight=isight, geom=geom, hwp=True, comps='TQU', cuts=iaman.flags[config['map']['flags']])
                iresult = coords.demod.make_map(iaman, P=P, dsT=iaman[config['map']['Tsignal']], det_weights_demod=iaman.inv_var)
                iresult['hit'] = P.to_map(tod=iaman, signal=np.ones(iaman[config['map']['Tsignal']].shape, dtype = np.float32), det_weights=None, comps='T')[0]
                results.append(iresult)
                detids.append(idetid)
                direeach = os.path.join(config['map']['save_dire'], f'each_det/{iobsid}/{idetid}/')
                # save individual detector map
                save_results(iresult, iobsid, iband, iws, direeach, fits_name=fits_name)
            result = coadd_maps(results)
        else:
            P = coords.P.for_tod(aman, sight=sight, geom=geom, hwp=True, comps='TQU', cuts=aman.flags[config['map']['flags']])
            result = coords.demod.make_map(aman, P=P, dsT=aman[config['map']['Tsignal']], det_weights_demod=aman.inv_var)
            result['hit'] = P.to_map(tod=aman, signal=np.ones(aman[config['map']['Tsignal']].shape, dtype = np.float32), det_weights=None, comps='T')[0]
            if config['map'].get('recenter', False):
                yc, xc = get_map_center(result['map'][0])
                result = recenter_map(result, yc, xc, r=config['map']['size']*coords.DEG,
                                    res=config['map']['res']*coords.DEG, proj=config['map']['proj'])
            detids = aman.det_info.det_id

    # save maps
    if config['map'].get('recenter', False):
        logger.debug('Recentering coadded maps')
        yc, xc = get_map_center(result['map'][0])
        result = recenter_map(result, yc, xc, r=config['map']['size']*coords.DEG,
                            res=config['map']['res']*coords.DEG, proj=config['map']['proj'])
    else:
        yc, xc = np.nan, np.nan
    save_results(result, iobsid, iband, iws, config['map']['save_dire'], fits_name=fits_name)
    map_vars = get_map_variances(result['map'], minr=config['map']['var_minr'], maxr=config['map']['var_maxr'])
    if debug:
        return aman, result
    else:
        return detids, yc, xc, map_vars

def make_instrument_center(aman, config, logger, rot_q=None, debug=False, fits_name=None):
    """ Make instruemnt center maps.
    Args:
        aman: axis manager
        config: "mapmaking" part of config file.
        logger: logger object
        rot_q: rotation quaternion to apply sight.Q. This is mainly for simlating planet signal
        debug: debug mode. Returns aman and result if True
    Returns:
        detids: list of detector ids
        yc, xc: peak position of Tmap
        map_vars: variance of maps calculated within from var_minr to var_maxr.
    """
    obsinfo = get_obsinfo(aman)
    iobsid = obsinfo['obsid']
    itele = obsinfo['telescope']
    iband = obsinfo['band']
    iws = obsinfo['ws']
    isite = obsinfo['site']

    if isinstance(config['map']['res'], str):
        config['map']['res'] = eval(config['map']['res'])
    if isinstance(config['map']['size'], str):
        config['map']['size'] = eval(config['map']['size'])

    sight = so3g.proj.CelestialSightLine.for_horizon(aman.timestamps, aman.boresight.az, aman.boresight.el, roll=aman.boresight.roll)

    if rot_q is not None:
        #rotate boresight to aubitary frame. This is mainly for trasfer function simulation
        sight.Q = sight.Q * ~rot_q
        
    if config['deflection_correction']['process'] & config['deflection_correction']['wafer_base']:
        logger.debug('Deflection correction is done each wafer')
        if hasattr(aman, "wobble_params") and aman.wobble_params:
            logger.debug("Use wobble params stored in metadata")
            params = (aman.wobble_params['amp'][0], aman.wobble_params['phase'][0])
            deflq = get_defl_quat(aman, params=params)
        else:
            logger.debug('No wobble params found in metadata. Using default deflection parameters.')
            deflq = get_defl_quat(aman, iband, iws, telescope=itele)
        sight.Q = sight.Q * ~deflq

    # calculate planet in horizon coordinates (az/el)
    logger.debug('Calculating planet position in horizon coordinates')
    azpl, elpl = coords.planets.calc_planet_azel_approx(aman.timestamps, source = config['map']['source'], site = isite)

    results = []
    dets_used = []
    logger.info(f'Length of aman = {aman.dets.count}')
    for idetid in aman.det_info.det_id:
        iaman = aman.restrict('dets', aman.det_info.det_id == idetid, in_place=False)
        isight = copy.copy(sight) # need to copy sight for each detector because it is modified in map_make_detcen_horizon_each
        try:
            iresult = make_instrument_center_each(iaman, iobsid, iband, iws, idetid,
                                                azpl = azpl, elpl = elpl,
                                                sight = isight, config = config,
                                                logger = logger)
            results.append(iresult)
            dets_used.append(idetid)
            del iaman


            if config['map']['single_save']:
                direeach = os.path.join(config['map']['save_dire'], f'each_det/{iobsid}/{idetid}/')
                save_results(iresult, iobsid, iband, iws, direeach, fits_name=fits_name)
        except Exception as e:
            logger.debug(f'{idetid}, {e}')

    logger.debug(f'Length of results = {len(results)}')
    logger.debug(f'Length of dets_used = {len(dets_used)}')
    logger.debug('Coadded all maps')
    result = coadd_maps(results)

    # run bootstrap and save it.
    if config.get('bootstrap', {}).get('process', False):
        logger.info('Running bootstrap')
        os.makedirs(config['bootstrap']['save_dire'], exist_ok=True)
        N_bootstrap = config['bootstrap']['N_bootstrap']
        Nlen = len(results)
        for i in range(N_bootstrap):
            ind = np.random.randint(0, Nlen, Nlen)
            iweights = []
            iweighted_maps = []
            for iind in ind:
                iweights.append(results[iind]['weight'])
                iweighted_maps.append(results[iind]['weighted_map'])
            coadd_map, _, _ = add_maps(iweights, iweighted_maps)
            fits_name_boot = f'map_{iobsid}_{iband}_{iws}_boot_{i}.fits'
            if config['bootstrap'].get('save_map', True):
                enmap.write_fits(os.path.join(config['bootstrap']['save_dire'], fits_name_boot), coadd_map)
                np.save(os.path.join(config['bootstrap']['save_dire'], f'index_{iobsid}_{iband}_{iws}_boot_{i}.npy'), ind)
            if config['bootstrap'].get('fit_map', False):
                ifitr = fit_map(coadd_map, fitthre = config['bootstrap'].get('fitthre', 1.0), sig_ran = config['bootstrap'].get('sig_ran', 0.8))
                save_pkl(ifitr, os.path.join(config['bootstrap']['save_dire'], f'fit_{iobsid}_{iband}_{iws}_boot_{i}.pkl'))

    # save maps
    if config['map'].get('recenter', False):
        logger.debug('Recentering coadded maps')
        yc, xc = get_map_center(result['map'][0])
        result = recenter_map(result, yc, xc, r=config['map']['size']*coords.DEG,
                            res=config['map']['res']*coords.DEG, proj=config['map']['proj'])
    else:
        yc, xc = np.nan, np.nan
    save_results(result, iobsid, iband, iws, config['map']['save_dire'], fits_name=fits_name)
    map_vars = get_map_variances(result['map'], minr=config['map']['var_minr'], maxr=config['map']['var_maxr'])
    if debug:
        return aman, result, results
    else:
        del results, result, azpl, elpl, sight
        gc.collect()
        return dets_used, yc, xc, map_vars


def make_instrument_center_each(aman, obsid, band, ws, idetid, azpl, elpl, sight, config, logger):
    """Make a map in instrument-centered coordinates for each detector.
    Args:
        aman: axis manager that include one detector.
        obsid: observation id
        band: frequency band
        ws: wafer slot
        idetid: detector id
        azpl: planet azimuth, in radians.
        elpl: planet elevation, in radians.
        sight: CelestialSightLine object in horizon coordinates.
        config: configuration dictionary
        logger: logger object
    Returns:
        result: Dictionary that include 'map', 'weighted_map', 'weight', 'hit'
    """
    # select detectors based on fitted values or fit False
    if config['xieta_correction']['process'] and config['fittod']['process']:
        fitinfo = get_db_planettod(config['fittod']['dbpath'], obs_id = obsid, freq_channel = band, wafer = ws, detid = idetid)[0]
        aman.focal_plane.xi[0] = fitinfo.xo
        aman.focal_plane.eta[0] = fitinfo.yo


    if config['deflection_correction']['process'] and not config['deflection_correction']['wafer_base']:
        logger.debug('Deflection correction is done each detector')
        fitinfo = get_db_planettod(config['fittod']['dbpath'], obs_id = obsid, freq_channel = band, wafer = ws, detid = idetid)[0]
        params = (fitinfo.defla, fitinfo.deflp)
        deflq = get_defl_quat(aman, params = params)
        sight.Q = sight.Q * ~deflq

    if config['map']['coordinate'] == 'boresight_center':
        boresight_center = True
    elif config['map']['coordinate'] == 'detector_center':
        boresight_center = False
    else:
        raise ValueError(f"Unknown coordinate system: {config['map']['coordinate']}")
    P = coords.planets.get_each_instrument_P(aman, azpl, elpl, sight = sight, size=config['map']['size'] * coords.DEG,
                            res=config['map']['res'] * coords.DEG,
                            flags=aman.flags[config['map']['flags']],
                            boresight_centered = boresight_center)
    result = coords.demod.make_map(aman, P=P, dsT = aman[config['map']['Tsignal']], det_weights_demod=aman.inv_var)
    result['hit'] = P.to_map(tod=aman, signal=np.ones(aman[config['map']['Tsignal']].shape, dtype = np.float32), det_weights=None, comps='T')[0]
    return result

def get_map_center(tmap):
    """Recenter maps with reproject.thumbnails
    Args:   
        tmap: enmap of Temapreture map
    returns:
        y(dec), x(ra): center position of map. NOTE that their order.
    """
    y, x = tmap.posmap()
    p0 = [tmap.max(),0,0,0.003,0.003,0.1,0]
    twoD_Gaussian = partial(twoD_Gaussian_normalized, normalize=False)
    popt, _ = curve_fit(twoD_Gaussian, [x.ravel(), y.ravel()], tmap.ravel(),p0)
    return np.asarray([popt[2], popt[1]])

def recenter_map(result, yc, xc, **kwargs):
    """Recenter maps with reproject.thumbnails.
    Args:
        result: dictionary of enmap of maps: Usually contains 'map', 'weighted_map', 'weight', and 'hit'
        y, x: center position of map
        See detail of other kwargs in reproject.thumbnails.
    returns:
        dictionary of recentered maps
    """
    new_result = {}
    for ikey in result.keys():
        new_result[ikey] = reproject.thumbnails(result[ikey], coords=np.asarray([yc,xc]), **kwargs)
    return new_result

def coadd_maps(results):
    """Coadd maps from list of results maps
    Args:
      results: a list of dictionary of maps
    Returns:
      ret: dictionary of coadded maps
    """
    carwmaps = []
    carweights = []
    carhits = []
    for iresult in results:
        carwmaps.append(iresult['weighted_map'])
        carweights.append(iresult['weight'])
        carhits.append(iresult['hit'])

    deshit, dessummaps, desws, deswmaps = add_maps_all(carhits, carweights, carwmaps)
    ret = {'hit': deshit,
           'map': dessummaps,
           'weighted_map': deswmaps,
           'weight': desws}
    return ret

def add_maps_all(hits, weight, weightmaps):
    """Coadd maps from list of maps
    Args:
      hits: a list of maps
      weight: a list of weight maps
      weightmaps: a list of weighted signal maps
    Returns:
      deshit: coadded hit map
      dessummaps: coadded map
      desws: coadded weight map
      deswmaps: coadded weighted signal map
    """
    sumhit = np.sum(hits, axis = 0)
    sumws = np.sum(weight, axis = 0)
    sumwmaps = np.sum(weightmaps, axis = 0)
    summaps = exe_remove_weights(sumwmaps, sumws)
    deshit = enmap.full(hits[0].shape, hits[0].wcs, sumhit)
    desws = enmap.full(weight[0].shape, weight[0].wcs, sumws)
    deswmaps = enmap.full(weightmaps[0].shape, weightmaps[0].wcs, sumwmaps)
    dessummaps = enmap.full(weightmaps[0].shape, weightmaps[0].wcs, summaps)
    return deshit, dessummaps, desws, deswmaps

def add_maps(weight, weightmaps):
    sumws = np.sum(weight, axis = 0)
    sumwmaps = np.sum(weightmaps, axis = 0)
    summaps = exe_remove_weights(sumwmaps, sumws)
    desws = enmap.full(weight[0].shape, weight[0].wcs, sumws)
    deswmaps = enmap.full(weightmaps[0].shape, weightmaps[0].wcs, sumwmaps)
    dessummaps = enmap.full(weightmaps[0].shape, weightmaps[0].wcs, summaps)
    return dessummaps, desws, deswmaps

def exe_remove_weights(signal_map, weights_map, dest=None, eigentol=1e-4):
    """Remove weights from signal map
    Args:
      signal_map: weighted signal map
      weights_map: weight map
      dest: output signal map
    Returns:
      dest: output map
    """
    inverse_weights_map = cal_inverse_weights(weights_map, dest = None, eigentol=eigentol)
    if dest is None:
        dest = np.zeros_like(signal_map)
    dest[:] = coords.helpers._apply_inverse_weights_map(inverse_weights_map, signal_map)
    return dest

def cal_inverse_weights(weights_map, dest = None, eigentol=1e-4):
    """Calculate inverse weights map
    Args:
      weights_map: weight map
      dest: output inverse weight map
    Returns:
      dest: inverse weight map
    """
    if dest is None:
        dest = np.zeros_like(weights_map)
    dest[:] = coords.helpers._invert_weights_map(weights_map, eigentol=eigentol, UPLO='U')
    return dest

def save_results(res, obsid, band, ws, dire = './output/map', fits_name=None):
    """ Save results
    Args:
      res: dictionary of maps that is supposed to have 'hit', 'map', 'weighted_map', and 'weight'
      obsid: observation id
      band: band
      ws: wafer slot
      dire: save directory
    """
    h = res['hit']
    m = res['map']
    weighted_map = res['weighted_map']
    weight = res['weight']

    map_root_dir = dire
    hit_dir = os.path.join(map_root_dir, band, 'hit')
    map_dir = os.path.join(map_root_dir, band, 'map')
    weighted_map_dir = os.path.join(map_root_dir, band, 'weighted_map')
    weight_dir = os.path.join(map_root_dir, band, 'weight')
    plots_dir = os.path.join(map_root_dir, band, 'plots')

    for _dir in [map_dir, weighted_map_dir, weight_dir, plots_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # fits
    if fits_name is None:
        fits_name = f'{obsid}_{ws}.fits'
    enmap.write_fits(os.path.join(hit_dir, fits_name), h)
    enmap.write_fits(os.path.join(map_dir, fits_name), m)
    enmap.write_fits(os.path.join(weighted_map_dir, fits_name), weighted_map)
    enmap.write_fits(os.path.join(weight_dir, fits_name), weight)
    return 

def get_map_variances(maps, minr=50, maxr=72):
    """Calculate map standard deviation (pixell space STD) within the radius from minr to maxr.
    minr and maxr is valided for MF-SATs.
    Args:
        maps: list of enmaps to calculate standard deviation. Usually, these are T, Q, and U maps.
        minr: minimum radius in arcmin to calculate standard deviation. Default is 50 arcmin.
        maxr: maximum radius in arcmin to calculate standard deviation. Default is 72 arcmin.
    returns:
        list of standard deviations
    """
    minr = minr/60*coords.DEG # convert to radians
    maxr = maxr/60*coords.DEG # convert to radians
    ix, iy = maps.posmap() # in radian
    irs = np.sqrt(ix**2 + iy**2)
    ifl = (irs > minr) & (irs < maxr)
    map_vars = []
    for imap in maps:
        map_vars.append(np.std(imap[ifl]))
    return map_vars

#### TOD fit functions
def execute_todfit(aman, dbpath, center_on, Tsignal='dsT', flags=None, centroid_position=(0,0), r_use=None, r_fit=None, res=2, max_pix=4000000, detrend=True, npoly=0, defl_model=True, imask_use=None, imask_fit=None, logger=None):
    """Execute TOD fit for one observation axismanager.
    Assuming aman contains only one observation per band per wafer.

    Args:
        aman: axismanager
        dbpath: path to todfit database
        flags: str, flags to use
        center_on: source name to center on
        dsT: signal name. Default is 'dsT'.
        centroid_position: centroid model in (xi, eta) in degrees
        r_use: radius to make source flag for use, in degrees
        r_fit: radius to make source flag for fit, in degrees
        res: map resolution for making source flag in arcmin
        max_pix: max number of pixels to use for making source flag
        detrend: whether to detrend the data
        npoly: number of polynomial terms to fit for detrending (np.polyfit)
        defl_model: whether to use deflection model for fit
        imask_use: mask for tods to use.
        imask_fit: mask for tods to fit.
    """
    if logger is None:
        logger = logging.init_logger("todfit", verbosity=2)
    if r_use is None:
        if aman.det_info.wafer.bandpass[0] == 'f090':
            r_use = 1
        elif aman.det_info.wafer.bandpass[0] == 'f150':
            r_use = 0.8
    if r_fit is None:
        if aman.det_info.wafer.bandpass[0] == 'f090':
            r_fit = 0.8
        elif aman.det_info.wafer.bandpass[0] == 'f150':
            r_fit = 0.6
    # get planet xi/eta position. Not corrected for individual detector position.
    isite = 'so_lat'
    azpl, elpl = coords.planets.calc_planet_azel_approx(aman.timestamps, source = center_on, site = isite)
    sight = so3g.proj.CelestialSightLine.for_horizon(aman.timestamps, aman.boresight.az, aman.boresight.el, roll=aman.boresight.roll)
    q_pla = so3g.proj.quat.rotation_lonlat(-azpl, elpl)
    q_tot = ~sight.Q*q_pla
    ixi, ieta, _ = so3g.proj.quat.decompose_xieta(q_tot)

    # make source flags
    source_flags_name_use = 'source_flags_jupiter_use'
    source_flags_name_fit = 'source_flags_jupiter_fit'
    if imask_use is None:
        imask_use = {'shape': 'circle', 'xyr': [centroid_position[0], centroid_position[1], r_use]}
    if imask_fit is None:
        imask_fit = {'shape': 'circle', 'xyr': [centroid_position[0], centroid_position[1], r_fit]}
    source_flag_use = tod_ops.flags.get_source_flags(aman, merge=False, overwrite=True, source_flags_name=source_flags_name_use, mask=imask_use, center_on=center_on, res=res, max_pix=max_pix)
    source_flag_fit = tod_ops.flags.get_source_flags(aman, merge=False, overwrite=True, source_flags_name=source_flags_name_fit, mask=imask_fit, center_on=center_on, res=res, max_pix=max_pix)

    # comnined flag for making mask of background region
    from so3g.proj import RangesMatrix
    source_flag_bgest = RangesMatrix.zeros(source_flag_use.shape)
    source_flag_bgest += source_flag_use
    source_flag_bgest *= ~source_flag_fit
    if flags is not None:
        logger.info(f'Using flags: {flags} for tod fit')
        imask_flag = ~aman.flags[flags] # because flags is regions we do not want to use
        source_flag_use *= imask_flag
        source_flag_fit *= imask_flag
        source_flag_bgest *= imask_flag
        
    signal = aman[Tsignal]
    signal = signal.copy()
    time = aman.timestamps
    
    # fit for each detector
    for i in range(aman.dets.count):
        isignal = signal[i]
        imaskbg = source_flag_bgest.mask()[i]
        imaskfit = source_flag_fit.mask()[i]
        if np.any(imaskfit):
            try:
                if detrend:
                    # apply detrending
                    fitrange = source_flag_fit[i].ranges()
                    bgrange = source_flag_bgest[i].ranges()

                    for j in range(len(fitrange)):
                        ifitrange = fitrange[j]
                        ibgidx1 = np.where(bgrange.T[1] == ifitrange[0])[0][0]
                        ibgidx2 = np.where(bgrange.T[0] == ifitrange[1])[0][0]
                        ibgrange1 = bgrange[ibgidx1]
                        ibgrange2 = bgrange[ibgidx2]
                        
                        islicefit = slice(ifitrange[0], ifitrange[1])
                        islicebg1 = slice(ibgrange1[0], ibgrange1[1])
                        islicebg2 = slice(ibgrange2[0], ibgrange2[1])
                        ix = np.concatenate([time[islicebg1], time[islicebg2]])
                        iy = np.concatenate([isignal[islicebg1], isignal[islicebg2]])
                        icoeff = np.polyfit(ix, iy, npoly)
                        ibaseline = np.polyval(icoeff, time[islicefit])
                        isignal[islicefit] -= ibaseline

                itod4fit = isignal[imaskfit]
                ierr4fit = np.std(isignal[imaskbg], ddof=1)
                ixietatheta4fit = (ixi[imaskfit], ieta[imaskfit], aman.hwp_angle[imaskfit])
                ixioff = aman.focal_plane.xi[i]
                ietaoff = aman.focal_plane.eta[i]
                if defl_model:    
                    p0 = [np.max(itod4fit)*4, ixioff, ietaoff, FWHM2sigma(20/60*coords.DEG), FWHM2sigma(20/60*coords.DEG), 0, -0.4/60*coords.DEG, 60*coords.DEG] # from Feb 25 2025
                    popt, pcov = curve_fit(defl_fit_model, ixietatheta4fit, itod4fit, p0, sigma = ierr4fit)
                    ifit = defl_fit_model(ixietatheta4fit, *popt)
                    errs, chisq, dof = calc_err_redchi(itod4fit, ifit, ierr4fit, popt, pcov)
                else:
                    p0 = [np.max(itod4fit)*4, ixioff, ietaoff, FWHM2sigma(20/60*coords.DEG), FWHM2sigma(20/60*coords.DEG), 0] # from Feb 25 2025
                    ixieta4fit = (ixietatheta4fit[0], ixietatheta4fit[1])
                    popt, pcov = curve_fit(twoD_gaussian_fit_model, ixieta4fit, itod4fit, p0, sigma = ierr4fit)
                    ifit = twoD_gaussian_fit_model(ixieta4fit, *popt)
                    errs, chisq, dof = calc_err_redchi(itod4fit, ifit, ierr4fit, popt, pcov)
                if np.all(np.isfinite(errs)):
                    # save fit result
                    idetid = aman.det_info.det_id[i]
                    info = make_info_planettod(aman, popt, errs, chisq, dof, idetid)
                    save_info(info, dbpath = dbpath)
            except Exception as e:
                pass
    return 

def get_defl_quat(aman, params=None):
    """Get deflection quaternion. If params is None, use default deflection parameters stored in axis manager from wobble metadata.
    Assuming axismanager is consist of single wafer/band/telescope because wobble params are currently per-wafer/band/telescope.
    Args:
        aman: axis manager
        params: (amplitude [arcmin], phase [radian]) of wobble model.
    Returns:
        deflection quaternions
    """
    
    if params is None:
        assert hasattr(aman, "wobble_params"), "wobble metadata is not found in axis manager. Please provide params."
        ph_def = aman.wobble_params['amp'][0]/60*coords.DEG # convert to radian
        ph_hwp = aman.wobble_params['phase'][0]
    else:
        ph_def, ph_hwp = params
        ph_def = ph_def/60*coords.DEG # convert to radian
        
    dxi = ph_def*np.cos(aman.hwp_angle-ph_hwp) #params calculated in a planet centered coordinate system
    deta = -ph_def*np.sin(aman.hwp_angle-ph_hwp)
    deflq = so3g.proj.quat.rotation_xieta(xi = dxi, eta = deta)
    return deflq

def defl_fit_model(xieta_theta, amplitude, xo, yo, sigma_x, sigma_y, theta, ph_def, ph_hwp):
    """ Fit model that accounts for pointing deflection. Offset is set to be 0.
    Args:
        xieta_theta: (xi, eta, theta) in radian
        amplitude: amplitude of Gaussian
        xo, yo: center of Gaussian in radian
        sigma_x, sigma_y: sigma of Gaussian in radian
        theta: rotation angle of Gaussian in radian
        ph_def: amplitude of deflection in radian
        ph_hwp: phase od deflection in radian
    Returns:
        normalized 2D elliptical Gaussian with deflection
    """
    offset = 0
    xi, eta, hwptheta = xieta_theta
    xietaq = so3g.proj.quat.rotation_xieta(xi = xi, eta = eta)
    dxi = ph_def*np.cos(hwptheta-ph_hwp)
    deta = -ph_def*np.sin(hwptheta-ph_hwp)
    deflq = so3g.proj.quat.rotation_xieta(xi = dxi, eta = deta)
    q_tot_defl = deflq*xietaq
    nxi, neta, _ = so3g.proj.quat.decompose_xieta(q_tot_defl)
    xieta = (nxi, neta)
    beam = twoD_Gaussian_normalized(xieta, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    return beam

def twoD_gaussian_fit_model(xieta, amplitude, xo, yo, sigma_x, sigma_y, theta):
    """ Fit model that accounts for pointing deflection. Offset is set to 0.
    Args:
        xieta: (xi, eta) in radian
        amplitude: amplitude of Gaussian
        xo, yo: center of Gaussian in radian
        sigma_x, sigma_y: sigma of Gaussian in radian
        theta: rotation angle of Gaussian in radian
    Returns:
        normalized 2D elliptical Gaussian
    """
    offset = 0
    xi, eta = xieta
    xietaq = so3g.proj.quat.rotation_xieta(xi = xi, eta = eta)
    q_tot_defl = xietaq
    nxi, neta, _ = so3g.proj.quat.decompose_xieta(q_tot_defl)
    xieta = (nxi, neta)
    beam = twoD_Gaussian_normalized(xieta, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    return beam

def twoD_Gaussian_normalized(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset=0, normalize=True):
    """Normalized 2D elliptical Gaussian
    Args:
        xy: (x, y) in radian
        amplitude: amplitude of Gaussian
        xo, yo: center of Gaussian in radian
        sigma_x, sigma_y: sigma of Gaussian in radian
        theta: rotation angle of Gaussian in radian
        offset: offset of Gaussian
        normalize: normalize amplitude or not so integral become unity
    Returns:
        normalized 2D elliptical Gaussian
    """
    xori, yori = xy
    xo = float(xo)
    yo = float(yo)
    x_dif = xori - xo
    y_dif = yori - yo
    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    sigma = np.sqrt(sigma_x*sigma_y)
    coeff = 2*np.pi*sigma**2
    if normalize:
        g = offset + coeff**(-1) * amplitude *np.exp( - (a*(x_dif**2) + 2*b*(x_dif)*(y_dif) + c*((y_dif)**2)))
    else:
        g = offset + amplitude *np.exp( - (a*(x_dif**2) + 2*b*(x_dif)*(y_dif) + c*((y_dif)**2)))
    return g

def calc_err_redchi(data, fit, err, popt, pcov):
    """Calculate errors on parameters and chi-squared/degree of freedom from scipy.optimize.curve_fit.
    Args:
        data: signal
        fit: fitted signal based on popt
        err: signal error
        popt: optimized parameters from curve_fit
        pcov: covariance matrix from curve_fit
    Returns:
        errs: 1 sigma error for each parameter
        chisq: chi-squared
        dof: degree of freedom
    """
    chisq = np.sum(((data - fit)/err)**2)
    dof = (len(data) - len(popt))
    redchi = chisq/(len(data) - len(popt))
    errs = [np.sqrt(pcov[i][i]/redchi) for i in range(len(popt))]
    return errs, chisq, dof

def apply_todfit_selection(aman, toddbpath, thoreshold_path, logger, apply_selection = ['amplitude', 'xo', 'yo', 'sigmax', 'sigmay', 'defla', 'deflp']):
    """Apply selection based on todfit results. This needs todfit database.
    Args:
        aman: axis manager
        toddbpath: path to todfit database
        thoreshold_path: path to selection threshold for todfit results
        logger: logger for logging info
        apply_selection: list of parameters to apply selection on. Possible parameters include 'amplitude', 'xo', 'yo', 'sigmax', 'sigmay', 'defla', 'deflp'. For 'defla', it will apply selection on the absolute value of deflection amplitude. For 'deflp', it will apply selection on deflection phase modulo pi.
    Returns:
        num_det_before: number of dets before applying selection
    """
    logger.info(f'Apply todfit based selection with paramters: {apply_selection}')
    iobsid, itele, iband, iws, _ = get_obsinfo(aman)
    logger.info(f'Read threshold')
    thresholds = load_select_threshold(thoreshold_path, itele, aman.obs_info.start_time, iband, iws)
    logger.info(f'get db')
    fit_database = get_db_planettod(toddbpath, obs_id=iobsid, freq_channel=iband, wafer=iws)
    detids= np.array([ifitinfo.detid for ifitinfo in fit_database])
    logger.info(f'afer detitds')
    bls = np.full(aman.dets.count, False)
    for i, idetid in enumerate(aman.det_info.det_id):
        ifl = detids == idetid
        if np.any(ifl):
            ifitresult = get_fitresult(fit_database, detids, idetid)
            ibl = fit_selection_each(thresholds, ifitresult, apply_selection)
            bls[i] = ibl
    logger.info(f'Finish calculate')
    num_det_before = aman.dets.count
    aman.restrict('dets', bls, in_place=True)
    return num_det_before

def load_select_threshold(path, sat, ts, band, ws, satp1_dt_thre =datetime.datetime(2025,2,27, tzinfo=datetime.timezone.utc)):
    dt = datetime.datetime.fromtimestamp(ts, datetime.UTC)
    if sat == 'satp1':
        if dt < satp1_dt_thre:
            icool = 'scr4'
        else:
            icool = 'scr5'
    elif sat == 'satp3':
        icool = 'run13'
    thres = read_pkl(path)
    return thres[f'{sat}_{icool}'][band][ws]

def get_fitresult(fit_database, detids, idetid):
    ifl = detids == idetid
    ifitinfo = np.array(fit_database)[ifl][0]
    
    ifitresult = {}
    ifitresult['amplitude'] = ifitinfo.amplitude
    ifitresult['xo'] = ifitinfo.xo
    ifitresult['yo'] = ifitinfo.yo
    ifitresult['sigmax'] = ifitinfo.sigmax
    ifitresult['sigmay'] = ifitinfo.sigmay
    #ifitresult['theta'] = ifitinfo.theta
    ifitresult['defla'] = ifitinfo.defla
    ifitresult['deflp'] = ifitinfo.deflp
    return ifitresult
    
def fit_selection_each(thoreshold, fit_result, apply_selection = ['amplitude', 'xo', 'yo', 'sigmax', 'sigmay', 'defla', 'deflp']):
    bls = []
    for i, ikey in enumerate(apply_selection):
        low_thre = thoreshold[ikey][0]
        high_thre = thoreshold[ikey][1]
        if ikey == 'defla':
            ival = np.abs(fit_result[ikey])
        elif ikey == 'deflp':
            ival = fit_result[ikey]%np.pi
        else:
            ival = fit_result[ikey]
        ibl = (low_thre < ival) & (high_thre > ival)
        bls.append(ibl)
    bls = np.array(bls)
    #print(bls)
    return np.all(bls)

#### database function

def save_info(info, dbpath = '/home/ys5857/workspace/jupyter/2025/05/dbtest/db.sqlite'):
    """Save Database at a given path.
    Args:
        info: instance of database class.
        dbpath: path to sqlite database.
    """
    dir_path = os.path.dirname(dbpath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    engine = create_engine("sqlite:///%s" % dbpath, echo=False)
    Base.metadata.create_all(bind=engine)

    Session = sessionmaker(bind=engine)

    with Session() as session:
        session.add(info)
        try:
            session.commit()
        except exc.IntegrityError:
            session.rollback()

########## Planet observation database #############
Base = declarative_base()
class PlanetInfo(Base):
    """Planet observation database class.
    """
    __tablename__ = "planet"

    obs_id: Mapped[str] = mapped_column(primary_key=True)
    telescope: Mapped[str] = mapped_column(primary_key=True)
    freq_channel: Mapped[str] = mapped_column(primary_key=True)
    wafer: Mapped[str] = mapped_column(primary_key=True)
    ctime: Mapped[int] = mapped_column(primary_key=True)
    duration: Mapped[Optional[float]]
    elevation: Mapped[Optional[float]]
    azimuth: Mapped[Optional[float]]
    pwv: Mapped[Optional[float]]
    pwv_std: Mapped[Optional[float]]
    pwv_p2p: Mapped[Optional[float]]
    pwv_apex: Mapped[Optional[float]]
    pwv_apex_std: Mapped[Optional[float]]
    pwv_apex_p2p: Mapped[Optional[float]]
    f_hwp: Mapped[Optional[float]]
    roll_angle: Mapped[Optional[float]]
    scan_speed: Mapped[Optional[float]]
    accl_speed: Mapped[Optional[float]]
    detnum_before_fitselection: Mapped[Optional[int]]
    total_detnum: Mapped[Optional[int]]
    recenter: Mapped[Optional[str]]
    yc: Mapped[Optional[float]]
    xc: Mapped[Optional[float]]
    Tmap_variance: Mapped[Optional[float]]
    Qmap_variance: Mapped[Optional[float]]
    Umap_variance: Mapped[Optional[float]]
    _proc: Mapped[Optional[str]]
    _detnum: Mapped[Optional[str]]
    _detid: Mapped[Optional[str]]

    def __init__(self, obs_id, telescope, freq_channel, wafer, ctime):
        self.obs_id = obs_id
        self.telescope = telescope
        self.freq_channel = freq_channel
        self.wafer = wafer
        self.ctime = ctime

    @property
    def proc(self):
        return json.loads(self._proc)

    @proc.setter
    def proc(self, value):
        self._proc = json.dumps(value)

    @property
    def detnum(self):
        return json.loads(self._detnum)

    @detnum.setter
    def detnum(self, value):
        self._detnum = json.dumps(value)

    @property
    def detid(self):
        return json.loads(self._detid)

    @detid.setter
    def detid(self, value):
        self._detid = json.dumps(value)

    def __repr__(self):
        return f"({self.obs_id},{self.telescope},{self.freq_channel},{self.wafer},{self.ctime})"

def make_info(aman):
    """Make PlanetTodFitInfo instance.
    Args:
        aman: axismanager for one detector.
    """
    dbinfo = PlanetInfo(obs_id=aman.obs_info.obs_id,
                    telescope=aman.obs_info.telescope,
                    freq_channel=aman.det_info.wafer.bandpass[0],
                    wafer=aman.det_info.wafer_slot[0],
                    ctime=aman.obs_info.timestamp
                    )
    dbinfo.duration = aman.obs_info.stop_time - aman.obs_info.start_time
    dbinfo.elevation = aman.obs_info.el_center
    dbinfo.azimuth = aman.obs_info.az_center
    try:
        pwvs = get_pwv_sync(aman)
        bl = (pwvs < 3) & (pwvs >0)
        dbinfo.pwv = np.mean(pwvs[bl])
        dbinfo.pwv_std = np.std(pwvs[bl])
        dbinfo.pwv_p2p = np.max(pwvs[bl]) - np.min(pwvs[bl])
    except:
        dbinfo.pwv = -999
        dbinfo.pwv_std = -999
        dbinfo.pwv_p2p = -999
    try:
        _, pwvs_apex = load_apex_pwv_range(aman.timestamps[0], aman.timestamps[-1])
        if np.all(np.isnan(pwvs_apex)):
            dbinfo.pwv_apex = -999
            dbinfo.pwv_apex_std = -999
            dbinfo.pwv_apex_p2p = -999
        else:
            dbinfo.pwv_apex = np.nanmean(pwvs_apex) * 0.84 + 0.03
            dbinfo.pwv_apex_std = np.nanstd(pwvs_apex) * 0.84 + 0.03
            dbinfo.pwv_apex_p2p = (np.nanmax(pwvs_apex) - np.nanmin(pwvs_apex)) * 0.84 + 0.03
    except:
        dbinfo.pwv_apex = -999
        dbinfo.pwv_apex_std = -999
        dbinfo.pwv_apex_p2p = -999

    if dbinfo.pwv_apex == -999:        
        try:
            pwvs_apex = get_pwv_apex_sync(aman) 
            dbinfo.pwv_apex = np.nanmean(pwvs_apex) * 0.84 + 0.03 # Convert from APEX to corresponding PWV at SO site (CLASS radiometer).
            dbinfo.pwv_apex_std = np.nanstd(pwvs_apex) * 0.84 + 0.03
            dbinfo.pwv_apex_p2p = (np.nanmax(pwvs_apex) - np.nanmin(pwvs_apex)) * 0.84 + 0.03
        except:
            dbinfo.pwv_apex = -999
            dbinfo.pwv_apex_std = -999
            dbinfo.pwv_apex_p2p = -999
        
    dbinfo.f_hwp = float((np.sum(np.diff(np.unwrap(aman.hwp_angle)))) / (aman.timestamps[-1] - aman.timestamps[0]) / (2 * np.pi))
    dbinfo.roll_angle = aman.obs_info.roll_center
    dbinfo.scan_speed = float(np.rad2deg(1) * np.median(np.abs(np.diff(aman.boresight.az)) / np.diff(aman.timestamps)))
    return dbinfo

def get_db(dbpath, obs_id = None, telescope=None, wafer=None, freq_channel=None, echo = False):
    """Get PlanetTodFitInfo from database.
    Args:
        dbpath: path to sqlite database.
        obs_id: observation id. (e.g., 'obs_1736469509_satp3_1111111')
        telescope: telescope name. (e.g., "satp1")
        freq_channel: frequency channel. (e.g. "f090")
        wafer: wafer name. (e.g., "ws0")
        echo: if True, echo SQL statements.
    """
    engine = create_engine("sqlite:///%s" % dbpath, echo=echo)
    Session = sessionmaker(bind=engine)
    session = Session()

    filters = []
    if obs_id is not None:
        filters.append(PlanetInfo.obs_id == obs_id)
    if telescope is not None:
        filters.append(PlanetInfo.telescope == telescope)
    if wafer is not None:
        filters.append(PlanetInfo.wafer == wafer)
    if freq_channel is not None:
        filters.append(PlanetInfo.freq_channel == freq_channel)

    results = session.query(PlanetInfo).filter(*filters).all()
    return results

def get_pwv_sync(aman):
    """Get pwv of class radiometer from hk.
    Args:
        aman: axismanager
    """
    hkaman = io.hk_utils.get_hkaman(start=float(aman.timestamps[0]), stop=float(aman.timestamps[-1]), config=None, alias=None, fields = ['site.env-radiometer-class.feeds.pwvs.pwv',], data_dir = '/scratch/gpfs/SIMONSOBS/so/tracked/data/site/hk')

    x = hkaman['env-radiometer-class'].timestamps
    y = hkaman['env-radiometer-class']['env-radiometer-class'][0]
    ifl = (x > aman.timestamps[0]) & (x < aman.timestamps[-1])
    pwvs = y[ifl]

    del hkaman
    return pwvs


def get_apex_data(start_date=datetime.datetime(2024,5,19),
                  end_date=datetime.datetime(2024,5,21)):
    """Get APEX weather data from the ESO archive.
    This function cannot be run from cluster, like Tiger3. Plese use this fron login node.
    Otherwise, apex data has to be pre-downloaded and loaded.
    Args:
        start_date : Start date for the data.
        end_date : End date for the data.
    
    Returns
        outdata : Dictionary with keys 'timestamps' and 'pwv', which are lists of
                unix ctimestamps and precipitable water vapor values, respectively.
    """
    APEX_DATA_URL = 'http://archive.eso.org/wdb/wdb/eso/meteo_apex/query'

    request = requests.post(APEX_DATA_URL, data={
            'wdbo': 'csv/download',
            'max_rows_returned': 1000000,
            'start_date': start_date.strftime('%Y-%m-%dT%H:%M:%S') + '..' \
                + end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'tab_pwv': 'on',
            'shutter': 'SHUTTER_OPEN',
            #'tab_shutter': 'on',
        })

    def date_converter(d):
        return datetime.datetime.fromisoformat(d).replace(tzinfo=datetime.UTC)

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=',', skip_header=2,
        converters={0: date_converter},
        dtype=[('dates', datetime.datetime), ('pwv', float)],
    )
    
    outdata = {'timestamps':[d.timestamp() for d in data['dates']],
               'pwv':data['pwv']}
    return outdata

def save_apex_data(sy, sm, ey, em, savepath='/scratch/gpfs/SIMONSOBS/users/ys5857/share/pwv_apex/apex_data.pkl'):
    """Save apex data
    Args:
        sy, sm, ey, em: start year, start month, end year, end month
        savepath: save path
    """
    start_date = datetime.datetime(sy,sm,1, tzinfo=datetime.UTC)
    end_date   = datetime.datetime(ey,em,1, tzinfo=datetime.UTC)
    data = get_apex_data(start_date=start_date, end_date=end_date)
    np.savez(savepath, timestamp=data['timestamps'], pwv=data['pwv'])

def load_apex_pwv_range(start_ts, end_ts, data_dir = "/scratch/gpfs/SIMONSOBS/users/ys5857/share/pwv_apex/"):
    """Load apex data that pre-saved in data_dir
    Args:
        start_ts, end_ts: start and end unix timestamp
        data_dir: data directory
    Returns:
        timestamps, pwvs: timestaps and pwv within the range
    """

    all_t = []
    all_p = []

    for fname in os.listdir(data_dir):
        m = re.match(r"apex_pwv_(\d+)_(\d+)_(\d+)_(\d+)\.npz", fname)
        if m is None:
            continue
        y1, m1, y2, m2 = map(int, m.groups())
        t1 = datetime.datetime(y1, m1, 1, tzinfo=datetime.UTC).timestamp()
        if m2 == 12:
            y3, m3 = y2 + 1, 1
        else:
            y3, m3 = y2, m2 + 1
        t2 = datetime.datetime(y3, m3, 1, tzinfo=datetime.UTC).timestamp()
        # overlap check
        if t2 < start_ts or t1 > end_ts:
            continue
        path = os.path.join(data_dir, fname)
        d = np.load(path)
        all_t.append(d["timestamp"])
        all_p.append(d["pwv"])

    if len(all_t) == 0:
        return np.nan, np.nan

    timestamps = np.concatenate(all_t)
    pwv = np.concatenate(all_p)
    print('5')
    m = (timestamps >= start_ts) & (timestamps <= end_ts)

    return timestamps[m], pwv[m]

def get_pwv_apex_sync(aman):
    """
    Get APEX weather data from the ESO archive.

    Parameters
    ----------
    start_date : datetime.datetime
        Start date for the data.
    end_date : datetime.datetime
        End date for the data.
    
    Returns
    -------
    outdata : dict
        Dictionary with keys 'timestamps' and 'pwv', which are lists of
        unix ctimestamps and precipitable water vapor values, respectively.
    """
    APEX_DATA_URL = 'http://archive.eso.org/wdb/wdb/eso/meteo_apex/query'

    start_date = datetime.datetime.fromtimestamp(aman.timestamps[0], tz=datetime.UTC)
    end_date = datetime.datetime.fromtimestamp(aman.timestamps[-1], tz=datetime.UTC)
    request = requests.post(APEX_DATA_URL, data={
            'wdbo': 'csv/download',
            'max_rows_returned': 1000000,
            'start_date': start_date.strftime('%Y-%m-%dT%H:%M:%S') + '..' \
                + end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'tab_pwv': 'on',
            'shutter': 'SHUTTER_OPEN',
            #'tab_shutter': 'on',
        })

    def date_converter(d):
        return datetime.datetime.fromisoformat(d).replace(tzinfo=datetime.UTC)

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=',', skip_header=2,
        converters={0: date_converter},
        dtype=[('dates', datetime.datetime), ('pwv', float)],
    )

    timestamps = np.array([d.timestamp() for d in data['dates']])
    ifl = (timestamps > aman.timestamps[0]) & (timestamps < aman.timestamps[-1])
    pwv = data['pwv'][ifl]
    return pwv

############ TODFIT database #############
class PlanetTodFitInfo(Base):
    """Planet TOD fit database class.
    """
    __tablename__ = "planet_todfit"

    obs_id: Mapped[str] = mapped_column(primary_key=True)
    telescope: Mapped[str] = mapped_column(primary_key=True)
    freq_channel: Mapped[str] = mapped_column(primary_key=True)
    wafer: Mapped[str] = mapped_column(primary_key=True)
    detid: Mapped[str] = mapped_column(primary_key=True)
    amplitude: Mapped[Optional[float]]
    xo: Mapped[Optional[float]]
    yo: Mapped[Optional[float]]
    sigmax: Mapped[Optional[float]]
    sigmay: Mapped[Optional[float]]
    theta: Mapped[Optional[float]]
    defla: Mapped[Optional[float]]
    deflp: Mapped[Optional[float]]
    amplitude_error: Mapped[Optional[float]]
    xo_error: Mapped[Optional[float]]
    yo_error: Mapped[Optional[float]]
    sigmax_error: Mapped[Optional[float]]
    sigmay_error: Mapped[Optional[float]]
    theta_error: Mapped[Optional[float]]
    defla_error: Mapped[Optional[float]]
    deflp_error: Mapped[Optional[float]]
    chisq: Mapped[Optional[float]]
    dof: Mapped[Optional[float]]

    def __init__(self, obs_id, telescope, freq_channel, wafer, detid):
        self.obs_id = obs_id
        self.telescope = telescope
        self.freq_channel = freq_channel
        self.wafer = wafer
        self.detid = detid

    def __repr__(self):
        return f"({self.obs_id},{self.telescope},{self.freq_channel},{self.wafer},{self.detid})"

def make_info_planettod(aman, popt, errs, chisq, dof, detid=None):
    """Make PlanetTodFitInfo instance.
    Args:
        aman: axismanager for one detector.
    """
    if detid is None:
        detid = aman.det_info.det_id[0]
    dbinfo = PlanetTodFitInfo(obs_id=aman.obs_info.obs_id,
                    telescope=aman.obs_info.telescope,
                    freq_channel=aman.det_info.wafer.bandpass[0],
                    wafer=aman.det_info.wafer_slot[0],
                    detid=detid
                    )
    if len(popt) == 8:
        dbinfo.amplitude = popt[0]
        dbinfo.xo = popt[1]
        dbinfo.yo = popt[2]
        dbinfo.sigmax = popt[3]
        dbinfo.sigmay = popt[4]
        dbinfo.theta = popt[5]
        dbinfo.defla = popt[6]
        dbinfo.deflp = popt[7]
        dbinfo.amplitude_error = errs[0]
        dbinfo.xo_error = errs[1]
        dbinfo.yo_error = errs[2]
        dbinfo.sigmax_error = errs[3]
        dbinfo.sigmay_error = errs[4]
        dbinfo.theta_error = errs[5]
        dbinfo.defla_error = errs[6]
        dbinfo.deflp_error = errs[7]
    elif len(popt) == 6:
        dbinfo.amplitude = popt[0]
        dbinfo.xo = popt[1]
        dbinfo.yo = popt[2]
        dbinfo.sigmax = popt[3]
        dbinfo.sigmay = popt[4]
        dbinfo.theta = popt[5]
        dbinfo.amplitude_error = errs[0]
        dbinfo.xo_error = errs[1]
        dbinfo.yo_error = errs[2]
        dbinfo.sigmax_error = errs[3]
        dbinfo.sigmay_error = errs[4]
        dbinfo.theta_error = errs[5]
    else:
        raise ValueError(f'len(popt) should be 6 or 8, but {len(popt)} is given.')
    dbinfo.chisq = chisq
    dbinfo.dof = dof
    return dbinfo

def get_db_planettod(dbpath, obs_id = None, telescope=None, freq_channel=None, wafer=None, detid=None, echo = False):
    """Get PlanetTodFitInfo from database.
    Args:
        dbpath: path to sqlite database.
        obs_id: observation id. (e.g., 'obs_1736469509_satp3_1111111')
        telescope: telescope name. (e.g., "satp1")
        freq_channel: frequency channel. (e.g. "f090")
        wafer: wafer name. (e.g., "ws0")
        detid: detector id. (e.g., "Mv17_f090_Cr07c00B")
        echo: if True, echo SQL statements.
    """
    engine = create_engine("sqlite:///%s" % dbpath, echo=echo)
    Session = sessionmaker(bind=engine)
    session = Session()

    filters = []
    if obs_id is not None:
        filters.append(PlanetTodFitInfo.obs_id == obs_id)
    if telescope is not None:
        filters.append(PlanetTodFitInfo.telescope == telescope)
    if wafer is not None:
        filters.append(PlanetTodFitInfo.wafer == wafer)
    if freq_channel is not None:
        filters.append(PlanetTodFitInfo.freq_channel == freq_channel)
    if detid is not None:
        filters.append(PlanetTodFitInfo.detid == detid)

    results = session.query(PlanetTodFitInfo).filter(*filters).all()
    return results

#### fit functions in map-space

def make_sub_box(ramin, ramax, decmin, decmax):
    """Make a sub-box based on ramin, ramax, decmin, decmax.
    NOTE: All unit are in degrees for usability.
    Args:
        ramin(float): right Ascension minimum value in degrees.
        ramax(float): right Ascension maximum value in degrees.
        decmin(float): Declination minimum value in degrees.
        decmax(float): Declination maximum value in degrees.
    Returns:
        sub_box(numpy.array): A sub-box of the given box in radians.
    """
    sub_box = np.deg2rad([[decmin,ramin],[decmax,ramax]])
    return sub_box

def get_ext(imap):
    return np.array([imap.posmap()[1][0][0], imap.posmap()[1][0][-1],  imap.posmap()[0][-1][0], imap.posmap()[0][0][0]])/DEG

def get_sub_map(imap, ramin, ramax, decmin, decmax):
    """Get a sub-map from the given map based on ramin, ramax, decmin, decmax.
    NOTE: All unit are in degrees for usability.
    Args:
        imap: pixell map for TQU.
        ramin(float): right Ascension minimum value in degrees.
        ramax(float): right Ascension maximum value in degrees.
        decmin(float): Declination minimum value in degrees.
        decmax(float): Declination maximum value in degrees.
    Returns:
        A dictionary containing sub-maps and their metadata.
    """
    sub_box = make_sub_box(ramin, ramax, decmin, decmax)
    subcar = imap.submap(sub_box)
    if len(subcar.shape) == 2: # assume hit map or only T map
        submaps = get_car_sig(subcar)
        shape = subcar.shape
        ira = submaps[1]
        idec = submaps[2] 
        ext = [ira[0], ira[-1], idec[-1], idec[0]]
        return {'T': submaps[0], 'ra': ira, 'dec': idec, 'shape': shape, 'ext': ext}
    else:
        submaps = get_car_sigs(subcar)
        shape = subcar[0].shape
        ira = submaps[0][1]
        idec = submaps[0][2] 
        ext = [ira[0], ira[-1], idec[-1], idec[0]]
        return {'T': submaps[0][0], 'Q': submaps[1][0], 'U': submaps[2][0], 'ra': ira, 'dec': idec, 'shape': shape, 'ext': ext}

def get_car_sig(maps):
    """Get arrays of signal, ra, dec from a pixell map.
    ra, dec depends on maps(wcs).
    Args:
        maps: pixell map for each Stokes
    Returns:
        sig: 1D array of sig values
        ra: 1D array of right ascension values
        dec: 1D array of declination values
    """
    radec = np.rad2deg(enmap.posmap(maps.shape, maps.wcs))
    dec = radec[0].flatten()
    ra = radec[1].flatten()
    sig = np.array([imap for imap in maps]).flatten()
    return (sig, ra, dec)

def get_car_sigs(maps):
    """Get signal, ra, dec arrays from multiple pixell maps."""
    return np.array([get_car_sig(imap) for imap in maps])


def twoD_Gaussian_normalized(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset = 0):
    """ Normalized 2D Gaussian function
    Args:
        xy: (x, y) position of 2D map
        amplitude: Gaussian amplitude
        xo, yo: beam center
        sigma_x, sigma_y: beam sigma
        theta: beam orientation
    Return:
        gp: modelled gaussian signal
    """
    
    xori, yori = xy
    xo = float(xo)
    yo = float(yo)
    x_dif = xori - xo
    y_dif = yori - yo
    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    sigma = np.sqrt(sigma_x*sigma_y)
    coeff = 2*np.pi*sigma**2
    g = offset + coeff**(-1) * amplitude *np.exp( - (a*(x_dif**2) + 2*b*(x_dif)*(y_dif) + c*((y_dif)**2)))
    return g

def hermite_2d_model(xy, nx, ny, xo, yo, sigma):
    """Hermite 2D model
    Args:
        xy: (x, y) position of 2D map
        nx, ny: order of hermite polynomial
        xo, yo: beam center
        sigma: beam size, same unit as xy
    Return:
        gp: modelled leakage signal for each order pair
    """
    xori, yori = xy

    xo = float(xo)
    yo = float(yo) 
    xdif = xori - xo
    ydif = yori - yo
    
    exp = np.exp(- (xdif**2 + ydif**2) / (2*sigma**2) )
    herm = eval_hermite(nx, xdif/sigma)*eval_hermite(ny, ydif/sigma)
    coeff = 2**(nx+ny)*math.factorial(nx)*math.factorial(ny)*np.pi*sigma**2
    
    return (coeff)**(-1/2) *exp * herm

def T2P_leak_model_hermite_md(xy, am0, ad1, ad2, xo, yo, sigma, offset = 0):
    """Monopole and dipole model
    Args:
        xy: (x, y) position of 2D map
        am0: monopole amplitude
        ad1, ad2: dipole amplitude
        xo, yo: beam center
        sigma: beam size, Same unit as xy
    Return:
        gp: modelled leakage signal
    """
    # this is for modeling monopole/dipole
    
    b00 = hermite_2d_model(xy, 0, 0, xo, yo, sigma = sigma)
    b01 = hermite_2d_model(xy, 0, 1, xo, yo, sigma = sigma)
    b10 = hermite_2d_model(xy, 1, 0, xo, yo, sigma = sigma)
    m0 = b00
    d1 = b10
    d2 = b01
    gp = am0*m0 + ad1*d1 + ad2*d2 + offset
    return gp

def fit_map(imapcar, fitthre = 1, sig_ran = 0.9, xflip=False, yflip=False):
    """ This function fit T as main beam Q/U as monopole and dipole leakage.
    """
    # make maps used for fitting
    ramin, ramax, decmin, decmax = -2,2,-2,2
    box = make_sub_box(ramin, ramax, decmin, decmax)
    imaps = get_sub_map(imapcar, ramin, ramax, decmin, decmax)
    
    # calculate get background mean and standard deviation
    ix = imaps['ra']
    iy = imaps['dec']
    if xflip:
        ix = -imaps['ra']
    if yflip:
        iy = -imaps['dec']
    ixy = (ix, iy)
    ir = np.sqrt(ixy[0]**2 + ixy[1]**2)
    ifl_sig = (ir > sig_ran) & (ir < fitthre)
    bgmeant = np.mean(imaps['T'][ifl_sig])
    bgmeanq = np.mean(imaps['Q'][ifl_sig])
    bgmeanu = np.mean(imaps['U'][ifl_sig])

    bgstdt = np.std(imaps['T'][ifl_sig])
    bgstdq = np.std(imaps['Q'][ifl_sig])
    bgstdu = np.std(imaps['U'][ifl_sig])

    # To fit T map after subtracting back ground
    ifl = np.sqrt(ixy[0]**2 + ixy[1]**2) < fitthre
    ixy = (ix[ifl], iy[ifl])
    iTmap = imaps['T'][ifl] - bgmeant

    def twoD_Gaussian_offset_fixed(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
        offset = 0 # offset is set as 0 because we already subtracted it.
        return twoD_Gaussian_normalized(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    
    p0 = [np.max(iTmap), 0, 0, 20/60, 20/60, 0]
    popt, pcov = curve_fit(twoD_Gaussian_offset_fixed, ixy, iTmap, p0, sigma = bgstdt)
    ifit = twoD_Gaussian_offset_fixed(ixy, *popt)
    errt, chisqt, doft = calc_err_redchi(iTmap, ifit, bgstdt, popt, pcov)
    redchit = chisqt/doft
    # fit function of Q,U map
    def T2P_leak_model_hermite_fixed(xy, am0, ad1, ad2, sigma):
        xo, yo = popt[1], popt[2] # beam center from main beam
        offset = 0 # offset is set as 0 because we already subtracted it.
        return T2P_leak_model_hermite_md(xy, am0, ad1, ad2, xo, yo, sigma, offset)
    
    # fit in Q
    iQmap = imaps['Q'][ifl] - bgmeanq
    p0 = [1,1,1,0.16]
    poptq, pcovq = curve_fit(T2P_leak_model_hermite_fixed, ixy, iQmap, p0, sigma=bgstdq)
    ifitq = T2P_leak_model_hermite_fixed(ixy, *poptq)
    errq, chisq, dofq = calc_err_redchi(iQmap, ifitq, bgstdq, poptq, pcovq)
    redchiq = chisq/dofq

    # fit in U
    iUmap = imaps['U'][ifl] - bgmeanu
    p0 = [1,1,1,0.16]
    poptu, pcovu = curve_fit(T2P_leak_model_hermite_fixed, ixy, iUmap, p0, sigma=bgstdu)
    ifitu = T2P_leak_model_hermite_fixed(ixy, *poptu)
    erru, chisqu, dofu = calc_err_redchi(iUmap, ifitu, bgstdu, poptu, pcovu)
    redchiu = chisqu/dofu

    ret = {}
    ret['popt'] = popt
    ret['poptq'] = poptq
    ret['poptu'] = poptu
    ret['errt'] = errt
    ret['errq'] = errq
    ret['erru'] = erru
    ret['redchit'] = redchit
    ret['redchiq'] = redchiq
    ret['redchiu'] = redchiu
    ret['pcov'] = pcov
    ret['pcovq'] = pcovq
    ret['pcovu'] = pcovu
    return ret


#### Util functions
def get_obsinfo(aman):
    iobsid = aman.obs_info.obs_id
    itele = aman.obs_info.telescope
    iband = aman.det_info.wafer.bandpass[0]
    iws = aman.det_info.wafer_slot[0]
    its = np.median(aman.timestamps)
    isite = 'so_lat' # because we usually use this.
    if itele == 'satp3':
        icool = 'run13'
    elif itele == 'satp1':
        if its < datetime.datetime(2025,2,1).timestamp():
            icool = 'scr4'
        else:
            icool = 'scr5'
    info = {'obsid': iobsid, 'telescope': itele, 'band': iband, 'ws': iws, 'site': isite, 'cool': icool, 'ts': its}
    return info

def test1():
    print('test1')
    config_path = '/home/ys5857/workspace/script/pwg-scripts/flp/planet_mapmaker/example_satp3_detcen2.yaml'
    obs_id = 'obs_1723111505_satp3_1011111' # this is jupiter for ws0
    ws = 'ws6'
    band = 'f150'
    dets={'wafer_slot': ws, 'wafer.bandpass': band, }
    planet_mapmake_eachobs(config_path, obs_id, dets, verbosity = 3, debug=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./examplet.yaml', help='Path to the configuration file')
    parser.add_argument('--obslist_path', type=str, default=None, help='Path to the obslist file')
    parser.add_argument('--verbosity', type=int, default=2, help='Number for logger verbosity')
    parser.add_argument('--test', action='store_true', help='perform test function or not')
    args = parser.parse_args()
    if args.test:
        test1()
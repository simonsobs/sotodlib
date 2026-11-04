import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import so3g
from sotodlib import coords

logger = logging.getLogger("Leakage_Sim")

def read_pkl(path):
    """read pickle file"""
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def get_planck_map(savedir, freq='100', nside=512, field=np.arange(18), sat=None, cool=None, ws=None, qu=None):
    """ Get the corresponding Planck map for the given frequency, nside, telescope, cooling run, wafer slot, and Q/U component.
    Q/U component is divided here because Q and U leakage has different beam width.
    If argument is not provided property, then fiducial one is used.
    Args:
        savedir (str): Path to the directory containing the Planck maps.
        freq (str): Frequency of the Planck map (e.g., '100', '143')
        nside (int): Nside of the Planck map (e.g., 512, 1024).
        sat (str): Satellite identifier (e.g., 'SAT1', 'SAT3').
        cool (str): Cooling run identifier (e.g., 'scr5', 'run13')
        ws (str): Wafer slot identifier (e.g., 'ws1').
        qu (str): Q/U component identifier (e.g., 'q', 'u').
    Returns:
        planck_hp_map (array): Healpix map of the Planck data.
    """
    import healpy as hp
    p = os.path.join(savedir, f'maps_{freq}_{nside}_{sat}_{cool}_{ws}_{qu}.fits')
    if os.path.exists(p):
        logger.info(f'Planck map path to read: {p}')
        planck_hp_map = hp.read_map(p, field=field)
    else:
        p = os.path.join(savedir, f'maps_{freq}_{nside}.fits')
        if not os.path.exists(p):
            raise FileNotFoundError(f'No leakage derivative map for {sat} {cool} {ws} {qu} and no fiducial {p}')
        logger.warning(f'No ws-specific map for {sat} {cool} {ws} {qu}; using fiducial {p}')
        planck_hp_map = hp.read_map(p, field=field, dtype=np.float32)
    return planck_hp_map

def get_planck_qumap(savedir, freq='100', nside=512, field=np.arange(18), sat=None, cool=None, ws=None):
    """ wrapper function to get the corresponding Planck Q and U maps for the given frequency, nside, telescope, cooling run, and wafer slot.
    """
    planck_hp_qmap = get_planck_map(savedir, freq=freq, nside=nside, field=field, sat=sat, cool=cool, ws=ws, qu='q')
    planck_hp_umap = get_planck_map(savedir, freq=freq, nside=nside, field=field, sat=sat, cool=cool, ws=ws, qu='u')
    return planck_hp_qmap, planck_hp_umap

def construct_beam(isaved = None):
    """ Helper function to construct beam parameters from the saved pickle file.
    params in isaved should follow the format of beam_model.
    """
    ibeam_param = read_pkl(isaved)

    fitparamq_md = ibeam_param['fitrqu'][:ibeam_param['num_md']]
    fitparamu_md = ibeam_param['fitrqu'][ibeam_param['num_md']:ibeam_param['num_md']*2]
    fitparamq_md = np.concatenate([fitparamq_md[:3], ibeam_param['fitrt'][1:3], [fitparamq_md[3]]])
    fitparamu_md = np.concatenate([fitparamu_md[:3], ibeam_param['fitrt'][1:3], [fitparamu_md[3]]])
    fitparamq_quad = ibeam_param['fitrqu'][ibeam_param['num_md']*2:ibeam_param['nmode'] + ibeam_param['num_md']*2]
    fitparamu_quad = ibeam_param['fitrqu'][ibeam_param['nmode'] + ibeam_param['num_md']*2:len(ibeam_param['fitrqu'])]

    ixn_iyn = []
    for isum in ibeam_param['resi_sums']:
        for i in range(isum + 1):
            ixn_iyn.append((i, isum - i))

    ixn_iyn_xo_yo = []
    for iixy in ixn_iyn:
        ixn, iyn = iixy[0], iixy[1]
        ixn_iyn_xo_yo.append((int(ixn), int(iyn), ibeam_param['fitrt'][1], ibeam_param['fitrt'][2]))
    
    return ibeam_param, fitparamq_md, fitparamu_md, fitparamq_quad, fitparamu_quad, ixn_iyn_xo_yo

def get_corresponds_differential_pointing(beamd, sat, cool, band, ws):
    """ Get the corresponding differential pointing parameters of dipole leakage
    for the given telescope, cooling run,band, and wafer slot.
    Args:
        beamd (str): Path to the directory containing the beam parameters.
        band (str): Band of the telescope (e.g., 'f090', 'f150', 'f220').
        sat (str): Satellite identifier (e.g., 'SAT1', 'SAT3').
        cool (str): Cooling run identifier (e.g., 'scr5', 'run13')
        ws (str): Wafer slot identifier (e.g., 'ws1').
    Returns:
        rhoqx (float): Differential pointing parameter for Q in x direction.
        rhoqy (float): Differential pointing parameter for Q in y direction.
        rhoux (float): Differential pointing parameter for U in x direction.
        rhouy (float): Differential pointing parameter for U in y direction.
    """
    ibeamp = os.path.join(beamd, f"{sat}_{cool}_{band}_{ws}.pkl")
    if not os.path.exists(ibeamp):
        logger.info(f"There is no beam params for {sat}_{cool}_{band}_{ws}, so load ws1 instead (suppose for ws4).")
        ibeamp = os.path.join(beamd, f"{sat}_{cool}_{band}_ws1.pkl")
    logger.info(f'Beam path to read: {ibeamp}')
    ibeam_param, fitparamq_md, fitparamu_md, fitparamq_quad, fitparamu_quad, ixn_iyn_xo_yo = construct_beam(isaved = ibeamp)
    ifitt = ibeam_param['fitrt']
    ipeakt = ifitt[0]/(2*np.pi*ifitt[3]*ifitt[4])
    sigmaq = fitparamq_md[-1]
    sigmau = fitparamu_md[-1]
    sigmat = np.sqrt(ifitt[3]*ifitt[4])
    # sqrt(2) is different than ABS's definition sqrt(8), because we will multiply this coeffs with derivative maps.        
    rhoqx, rhoqy = fitparamq_md[1:3]/(np.sqrt(np.pi)*sigmaq) / ipeakt * np.sqrt(2) * sigmat*coords.DEG
    rhoux, rhouy = fitparamu_md[1:3]/(np.sqrt(np.pi)*sigmau) / ipeakt * np.sqrt(2) * sigmat*coords.DEG

    return rhoqx, rhoqy, rhoux, rhouy

def sim_first_derivative_tod_from_map(maps, ra, dec):
    """ Calulate first derivative of TOD based on its first and second derivative of T maps.
    Parameters
    ----------
    maps : array
        Healpix maps including first and second derivatives of T map. RING ordering is assumed.
    ra : array
        Right ascension of the pointing in radian.
    dec : array
        Declination of the pointing in radian.
    return tods_dx, tods_dy : array
        Simulated TOD of Q and U
    """
    import healpy as hp
    nside = hp.get_nside(maps[0])
    pixs = hp.ang2pix(nside, np.pi/2 - dec, ra) # input need to be theta/phi
    angs = np.array(hp.pix2ang(nside, pixs)).T
    # angs is theta/phi, and maps is based on theta/phi
    # dx means difference btw pixel and real pointing in Theta/(90 - Dec)
    # dy means difference btw pixel and real pointing in Phi/RA
    # NOTE first/second derivative map of T w.r.t. phi is already divided by sin(theta) = sin(90 - dec)
    # https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.alm2map_der1.html
    dx = (np.pi/2 - dec) - angs.T[0]
    dy = (ra - angs.T[1] + np.pi) % (2*np.pi) - np.pi
    sin = np.sin(np.pi/2 - dec)
    coeffs = np.array([np.ones(len(dx)), dx, dy*sin])
    inds_dx = [3,9,12]
    inds_dy = [6,12,15]
    tods_dx = np.sum(maps[inds_dx][:,pixs]*coeffs, axis = 0)
    tods_dy = np.sum(maps[inds_dy][:,pixs]*coeffs, axis = 0)
    # d/dtheta of (dT/dphi / sin(theta)) also has a -cot(theta) * (dT/dphi / sin(theta)) term
    tods_dy -= dx * np.cos(np.pi/2 - dec) / sin * maps[6][pixs]
    return tods_dx, tods_dy

#### dipole leakge ####
def sim_dipole_leakage_tod_from_map(qmaps, umaps, ra, dec, psi, rhoqx=1, rhoqy=1, rhoux=1, rhouy=1, second_der=False):
    """ Calculate the dipole leakage TOD based on first derivative of T map.
    Parameters
    ----------
    qmaps : array
        Healpix Q maps including first derivative of T map. RING ordering is assumed.
    umaps : array
        Healpix U maps including first derivative of T map. RING ordering is assumed.
    ra : array
        Right ascension of the pointing in radian.
    dec : array
        Declination of the pointing in radian.
    psi : array
        Angle of detector with respect to sky in radian.
    rhoqx : float
        Corresponding pointing differential of dipole coefficient along x direction for Q
    rhoqy : float
        Corresponding pointing differential of dipole coefficient along y direction for Q
    rhoux : float
        Corresponding pointing differential of dipole coefficient along x direction for U
    rhouy : float
        Corresponding pointing differential of dipole coefficient along y direction for U

    return Qdp, Udp : array
        Dipole leakage TOD for Q and U at sky. For detector timestream, we should consider detector pol angle.
    """
    import healpy as hp
    nside = hp.get_nside(qmaps[0])
    pixs = hp.ang2pix(nside, np.pi/2 - dec, ra)
    inds_dT = [0, 3, 6] # T, dxT, dyT/sinx
    iqmap_dT = qmaps[inds_dT]
    iumap_dT = umaps[inds_dT]
    if second_der:
        # use secound derivative to simulate first derivative of TOD.
        # This is more accurate than just using first derivative of T map.
        #print('Use second derivative of T map to simulate first derivative of TOD')
        qtods_dx, qtods_dy = sim_first_derivative_tod_from_map(qmaps, ra, dec)
        utods_dx, utods_dy = sim_first_derivative_tod_from_map(umaps, ra, dec)
        qtod_dT = np.array([iqmap_dT[0][pixs], qtods_dx, qtods_dy])
        utod_dT = np.array([iumap_dT[0][pixs], utods_dx, utods_dy])
    else:
        #print('Use nearest pixel of first derivative of T map to simulate first derivative of TOD')
        qtod_dT = iqmap_dT[:,pixs]
        utod_dT = iumap_dT[:,pixs]

    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    cos2psi = np.cos(2*psi)
    sin2psi = np.sin(2*psi)

    # healpy derivative convention
    dTdx_qsky =  qtod_dT[2]   # +RA / East
    dTdy_qsky =  qtod_dT[1]   # +Dec / North
    dTdx_usky =  utod_dT[2]   # +RA / East
    dTdy_usky =  utod_dT[1]   # +Dec / North

    # Again: tod_dT[1] is delT/del_theta, tod_dT[2] is delT/del_phi/sin(theta)
    xnabtq =  cospsi*dTdx_qsky - sinpsi*dTdy_qsky
    ynabtq = -sinpsi*dTdx_qsky - cospsi*dTdy_qsky
    xnabtu =  cospsi*dTdx_usky - sinpsi*dTdy_usky
    ynabtu = -sinpsi*dTdx_usky - cospsi*dTdy_usky

    Qdp = rhoqx*xnabtq + rhoqy*ynabtq
    Udp = rhoux*xnabtu + rhouy*ynabtu

    Qtod_dp = cos2psi*Qdp - sin2psi*Udp
    Utod_dp = sin2psi*Qdp + cos2psi*Udp
    Ttod = qtod_dT[0]
    return Ttod, Qtod_dp, Utod_dp

def replace_dipole_leakage_simtod(aman, qmaps, umaps, rhoqx=1, rhoqy=1, rhoux=1, rhouy=1,
                                  second_der=False, zeroTmap=True, flip_gamma=True, modulated=True,
                                  wrap=True, sight=None, site='so', weather='typical'):
    """ Simlate dipole leakage TOD based on first derivative of T map.
    Parameters
    ----------
    aman : axis manager
        Axis manager containing the TOD information.
    qmaps : array
        Healpix maps including first derivative of T map for TtoQ leakage. RING ordering is assumed.
    umaps : array
        Healpix maps including first derivative of T map for TtoU leakage. RING ordering is assumed.
    rhoqx : float
        Corresponding pointing differential of dipole coefficient along x direction for Q
    rhoqy : float
        Corresponding pointing differential of dipole coefficient along y direction for Q
    rhoux : float
        Corresponding pointing differential of dipole coefficient along x direction for U
    rhouy : float
        Corresponding pointing differential of dipole coefficient along y direction for U
    second_der : bool
        If True, use second derivative of T map to simulate first derivative of TOD. This is more accurate than just using first derivative of T map.
    zeroTmap : bool
        If True, set the T map to zero. This is useful for simulating only Q and U.
    flip_gamma : bool
        If True, flip the sign of the detector angle. This is needed due to HWP. HWP is almost parallel to FP, so need to reflect focal_plane_gamma (2d condition) rather than an angle in some other condition.
    modulated : bool
        If True, return modulated TOD. If False, return demodulated TOD.
    wrap : bool
        If True, wrap the TOD into the axis manager. If False, return the TOD as a numpy array.
    sight : so3g.proj.CelestialSightLine or None
        The sight line for the simulation. If None, it will be computed from the boresight information.
    """
    # maps 9 or 18 in healpix.
    # 9 corresponds tp ['T', 'Q', 'U', 'dxT', 'dxQ', 'dxU', 'dyT/sinx', 'dyQ/sinx', 'dyU/sinx']
    # 18 corresponds tp ['T', 'Q', 'U', 'dxT', 'dxQ', 'dxU', 'dyT/sinx', 'dyQ/sinx', 'dyU/sinx',
    # 'dxxT', 'dxxQ', 'dxxU', 'dxyT/sinx', 'dxyQ/sinx', 'dxyU/sinx', 'dyyT/sinx^2', 'dyyQ/sinx^2', 'dyyU/sinx^2]
    # where x = theta, y = phi, and sinx = sin(theta) = sin(90 - dec)
    # https://healpix.sourceforge.io/html/sub_alm2map_der.htm
    
    if sight is None:
        sight = so3g.proj.CelestialSightLine.az_el(aman.timestamps, aman.boresight.az, aman.boresight.el, roll = aman.boresight.roll, site=site, weather=weather)
    dsTs = []
    demodQs = []
    demodUs = []
    for i in tqdm(range(aman.dets.count), disable=True):
        iq_xieta = so3g.proj.quat.rotation_xieta(aman.focal_plane.xi[i], aman.focal_plane.eta[i])
        iqp = sight.Q*iq_xieta
        ira, idec, ipsi = so3g.proj.quat.decompose_lonlat(iqp) # This psi is the same as the one derived from so3g.proj.quat.decompo
        iT, iskyQ, iskyU = sim_dipole_leakage_tod_from_map(qmaps, umaps, ira, idec, ipsi, rhoqx=rhoqx, rhoqy=rhoqy, rhoux=rhoux, rhouy=rhouy, second_der=second_der)
        if zeroTmap:
            iT = np.zeros(len(iskyQ))
        if flip_gamma:
            theta_dets = ipsi - aman.focal_plane.gamma[i] # this need due to HWP. HWP is almost parallel to FP, so need to reflect focal_plane_gamma (2d condition) rather than an angle in some other condition.
        else:
            theta_dets = ipsi + aman.focal_plane.gamma[i] 
        theta_dets = - theta_dets # this need due to defintion of theta_dets (from North to West) vs COSMO  (from North to East)
        QUrot = (iskyQ + 1j*iskyU)*np.exp(1j*2*theta_dets)
        demodQ = QUrot.real
        demodU = QUrot.imag
        dsTs.append(iT)
        demodQs.append(demodQ)
        demodUs.append(demodU)
    dsTs = np.array(dsTs)
    demodQs = np.array(demodQs)
    demodUs = np.array(demodUs)
    if modulated:
        tod = dsTs + demodQs*np.cos(4*aman.hwp_angle) + demodUs*np.sin(4*aman.hwp_angle)
        tod = tod.astype(np.float32)
        if wrap:
            if 'signal' in aman:
                aman.move('signal', None)
            aman.wrap('signal', tod, [(0, 'dets'), (1, 'samps')])
        return tod

    else:
        dsTs = dsTs.astype(np.float32)
        demodQs = demodQs.astype(np.float32)
        demodUs = demodUs.astype(np.float32)
        if wrap:
            try:
                aman.move('dsT', None)
                aman.move('demodQ', None)
                aman.move('demodU', None)
            except:
                logger.warning('There was no dsT/demodQ/demodU')
                pass
            aman.wrap('dsT', dsTs, [(0, 'dets'), (1, 'samps')])
            aman.wrap('demodQ', demodQs, [(0, 'dets'), (1, 'samps')])
            aman.wrap('demodU', demodUs, [(0, 'dets'), (1, 'samps')])
        return dsTs, demodQs, demodUs

def from_map(aman, leakage_config, ws=None, band=None, wrap=True, modulated=True, sight=None):
    """ Simulate leakage TOD based on leakage_config and replace the TOD in the AxisManager.
    This function assumes all detectors belong to the same wafer slot.
    Args:
        aman (AxisManager): AxisManager containing the TOD information.
        leakage_config (dict): LeakageCfg. Details in bb-awg-scripts/pipeline/configs.py.
        ws (str, optional): Wafer slot. If None, it will be inferred from the AxisManager.
        band (str, optional): Band of the telescope. If None, it will be inferred from the AxisManager.
        wrap (bool, optional): If True, wrap the simulated TOD into the AxisManager. Default is True.
        modulated (bool, optional): If True, return modulated TOD. If False, return demodulated TOD. Default is True.
        sight (so3g.proj.CelestialSightLine, optional): The sight line
            for the simulation. If None, it will be computed from the boresight information.
    """
    if ws is None:
        ws = np.unique(aman.det_info.wafer_slot)
        if len(ws) != 1:
            raise ValueError(f'leakage sim needs a single wafer slot, got {ws}')
        ws = ws[0]
    if band is None:
        band = np.unique(aman.det_info.wafer.bandpass)
        if len(band) != 1:
            raise ValueError(f'leakage sim needs a single band, got {band}')
        band = band[0]
    # get beam model
    if leakage_config.leakage_type == 'dipole':
        field_max = 18 if leakage_config.second_der else 7
        tmapq, tmqpu = get_planck_qumap(savedir=leakage_config.sim_dir,
                                        freq=leakage_config.planck_map_freq,
                                        nside=leakage_config.nside,
                                        field=np.arange(field_max),
                                        sat=leakage_config.sat,
                                        cool=leakage_config.cool,
                                        ws=ws,
                                        )
                
        rhoqx, rhoqy, rhoux, rhouy = get_corresponds_differential_pointing(beamd=leakage_config.leak_beam_dir, 
                                                                        sat=leakage_config.sat,
                                                                        cool=leakage_config.cool,
                                                                        band=band,
                                                                        ws=ws)
            
        _ = replace_dipole_leakage_simtod(aman, tmapq, tmqpu, rhoqx, rhoqy, rhoux, rhouy,
                                                second_der=leakage_config.second_der,
                                                zeroTmap=leakage_config.zeroTmap,
                                                wrap=wrap, modulated=modulated, sight=sight)
    return
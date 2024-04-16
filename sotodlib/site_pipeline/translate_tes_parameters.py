"""
Module copied from sodetlib to use in site_pipeline.
Author: Remy Gerras, Satoru Takakura
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm.auto import trange
from copy import deepcopy

def translate_tes_params(iva, bsa):
    """
    Driver function that recalculates IV TES parameters using the
    thevenin equivalent voltage. Then computes the RP curve following
    Satoru's methods. Finally, load in and recalculate bias step analysis
    TES parameters using a shifted RP curve determined by the change in 
    optical power between operations. 
    
    Args:
    -----
    iv_file : str
        filepath to iv_analysis.npy file to re-analyze.
        
    bs_file : str
        filepath to bs_analysis.npy file to re-analyze
    
    save_iv : bool
        Boolean that determines if the re-analyzed iva is saved
        
    save_bs : bool
        Boolean that determines if the re-analyzed bsa is saved
    """
    iva2 = reanalyze_iv(iva)
    bsa2 = rerun_analysis(bsa, iva2)
    
    return iva2, bsa2


###############################################################################
# IVA translation functions below:
# 1. Calculate TES parameters using thevenin equivalent voltage.
# 2. Compute RP curve and storing parameters that determine RP curve.
# 3. Save updated IVAnalysis object as a numpy file.
###############################################################################

def model_logalpha(r, logp0=0., p1=7., p2=1., logp3=0.):
    """
    Fits Alpha' parameter. P0 has units of [uW]^-1.
    """
    return logp0 + p1 * (1-r) + p2 * r ** np.exp(logp3) * np.log(1-r)


def calculate_rpcurve(iva, minrfrac=0.2, maxrfrac=0.9):
    """
    Fit RP curve from IVA data. minrfrac and maxrfrac define the range
    in which the RP curve is fit over.
    
    Args:
    ----
    iva : IVAnalysis
    
    
    Returns:
    --------
    popts : np.array of shape (ndets x 6)
        Fitted parameters for logp0, p1, p2, logp3, P_opt, and drn
        P0 is optical power
    """
    nchans = iva['nchans']
    popt_size = 6
    popts = np.full((nchans, popt_size), np.nan)
    
    r1 = np.linspace(min(minrfrac, 0.0001), max(maxrfrac, 0.9999), 9999)
    dr = r1[1] - r1[0]

    def model_Pj(r, logp0=0., p1=7., p2=1., logp3=1., P_opt=0., drn=0.):
        loga1 = model_logalpha(r1, logp0=logp0, p1=p1, p2=p2, logp3=logp3)
        P1 = np.cumsum(dr / (np.exp(loga1) * r1))
        P1 -= np.interp(0.5, r1, P1)
        return np.interp(r / (1. + drn), r1, P1) + P_opt

    for ch in range(nchans):
        try:
            bg = iva['bgmap'][ch]
            if bg == -1:
                continue
                
            isc = iva['idxs'][ch][0]
            R = iva['R'][ch][isc:] 
            Rn = iva['R_n'][ch]
            r = R / Rn
            P = iva['p_tes'][ch][isc:] * 1e12
            ok = (r > minrfrac) * (r < maxrfrac)
            p0_guess = [0,3,1,1,0,0]        
            RP_popts, pcov = curve_fit(model_Pj, r[ok], P[ok], p0=p0_guess)
            
        except:
            continue
    
        popts[ch,:] = RP_popts
    
    iva['RP_popts'] = popts
    return iva['RP_popts']


def recompute_psats(iva2, psat_level=0.9):
    """
    Re-computes Psat for an IVAnalysis object. Will save results to iva.p_sat.
    This assumes i_tes, v_tes, and r_tes have already been calculated.

    Args
    ----
    iva2 : Dictionary
        Dictionary built from original IV Analysis .npy file
    psat_level : float
        R_frac level for which Psat is defined. If 0.9, Psat will be the
        power on the TES when R_frac = 0.9.

    Returns
    -------
    p_sat : np.ndarray
        Array of length (nchan) with the p-sat computed for each channel (W)
    """
    # calculates P_sat as P_TES when Rfrac = psat_level
    # if the TES is at 90% R_n more than once, just take the first crossing
    iva2['p_sat'] = np.full(iva2['p_sat'].shape, np.nan)
    
    for i in range(iva2['nchans']):
        if np.isnan(iva2['R_n'][i]):
            continue

        level = psat_level
        R = iva2['R'][i]
        R_n = iva2['R_n'][i]
        p_tes = iva2['p_tes'][i]
        cross_idx = np.where(R/R_n > level)[0]

        if len(cross_idx) == 0:
            iva2['p_sat'][i] = np.nan
            continue

        # Takes cross-index to be the first time Rfrac crosses psat_level
        cross_idx = cross_idx[0]
        if cross_idx == 0:
            iva2['p_sat'][i] = np.nan
            continue

        iva2['idxs'][i, 2] = cross_idx
        try:
            iva2['p_sat'][i] = interp1d(
                R[cross_idx-1:cross_idx+1]/R_n,
                p_tes[cross_idx-1:cross_idx+1]
            )(level)
        except ValueError:
            iva2['p_sat'][i] = np.nan

    return iva2['p_sat']


def recompute_si(iva2):
    """
    Recalculates responsivity using the thevenin equivalent voltage.
    
        Args
    ----
    iva2 : Dictionary
        Dictionary built from original IVAnalysis object (un-analyzed)

    Returns
    -------
    si : np.ndarray
        Array of length (nchan, nbiases) with  the responsivity as a fn of 
        thevenin equivalent voltage for each channel (V^-1).

    """
    iva2['si'] = np.full(iva2['si'].shape, np.nan)
    smooth_dist = 5
    w_len = 2 * smooth_dist + 1
    w = (1./float(w_len))*np.ones(w_len)  # window
    
    v_thev_smooth = np.convolve(iva2['v_thevenin'], w, mode='same')
    dv_thev = np.diff(v_thev_smooth)
    
    R_bl = iva2['meta']['bias_line_resistance']
    R_sh = iva2['meta']['R_sh']    
    
    for i in range(iva2['nchans']):
        sc_idx = iva2['idxs'][i, 0]

        if np.isnan(iva2['R_n'][i]) or sc_idx == -1: 
            # iva2['si'][i] = np.nan
            continue

        # Running average
        i_tes_smooth = np.convolve(iva2['i_tes'][i], w, mode='same')
        v_tes_smooth = np.convolve(iva2['v_tes'][i], w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth
    
        R_L = iva2['R_L'][i]

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth-1)) * R_L
        R_L_smooth[:sc_idx] = dv_tes[:sc_idx]/di_tes[:sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        beta = 0.

        # artificially setting rL to 0 for now,
        # to avoid issues in the SC branch
        # don't expect a large change, given the
        # relative size of rL to the other terms
        rL = 0

        # Responsivity estimate, derivation done here by MSF
        # https://www.overleaf.com/project/613978cb38d9d22e8550d45c
        si = -(1./(i0*r0*(2+beta)))*(1-((r0*(1+beta)+rL)/(dv_thev/di_tes)))
        si[:sc_idx] = np.nan
        iva2['si'][i, :-1] = si
    
    return iva2['si']
                   

def reanalyze_iv(iva, psat_level=0.9, save=False, update_cfg=False, show_pb=False):
    """
    Recalculates i_bias, v_bias, v_tes, i_tes, p_tes, R, R_n, R_L, 
    Si, psat, thevenin equivalent voltage, and calculates RP curve.
    
    Saves:
    v_thevenin:
        Array of shape (nbiases) containing thevenin equivalent voltage.
        Units same as v_tes
    i_bias:
        Array of shape (nbiases) containing corrected i_bias.
        Same units as original IVA i_bias
    v_tes:
        Array of shape (nchans x nbiases) containing corrected v_tes 
        of each detector. Same units as original IVA v_tes
    i_tes:
        Array of shape (nchans x nbiases) containing corrected i_tes 
        of each detector. Same units as original IVA i_tes
    p_tes:
        Array of shape (nchans x nbiases) containing corrected p_tes
        of each detector. Same units as original IVA p_tes.
    R:
        Array of shape (nchans x nbiases) containing corrected R
        of each detector. Same units as original IVA R.
    R_n:
        Array of shape (nchans) containing corrected R_n
        of each detector. Same units as original IVA R_n.
    R_L:
        Array of shape (nchans) containing corrected R_L
        of each detector. Same units as original IVA R_L.
    si:
        Array of shape (nchans x nbiases) containing corrected si
        of each detector. Same units as original IVA si.
    p_sat:
        Array of shape (nchans) containing corrected p_sat
        of each detector. Same units as original IVA p_sat.
    RP_popts:
        Array of shape (nchans x 6) containing fitted parameter
        values for the RP curve.
    """
    
    iva2 = deepcopy(iva)

    R_sh = iva2['meta']['R_sh']
    R_bl = iva2['meta']['bias_line_resistance']

    iva2['i_bias'] = iva2['v_bias'] / R_bl
    iva2['v_tes'] = np.full(iva2['v_tes'].shape, np.nan)
    iva2['i_tes'] = np.full(iva2['i_tes'].shape, np.nan)
    iva2['p_tes'] = np.full(iva2['p_tes'].shape, np.nan)
    iva2['R'] = np.full(iva2['R'].shape, np.nan)
    iva2['R_n'] = np.full(iva2['R_n'].shape, np.nan)
    iva2['R_L'] = np.full(iva2['R_L'].shape, np.nan)

    for i in trange(iva2['nchans'], disable=(not show_pb)):
        sc_idx = iva2['idxs'][i, 0]
        nb_idx = iva2['idxs'][i, 1]
        
        R = R_sh * (iva2['i_bias']/(iva2['resp'][i]) - 1)
        R_par = np.nanmean(R[1:sc_idx])
        R_n = np.nanmean(R[nb_idx:]) - R_par
        # R_n = np.nanmean(R[nb_idx:])
        R_L = R_sh + R_par
        R_tes = R - R_par

        iva2['v_thevenin'] = iva2['i_bias'] * R_sh
        iva2['v_tes'][i] = iva2['v_thevenin'] * (R_tes / (R_tes + R_L))
        iva2['i_tes'][i] = iva2['v_tes'][i] / R_tes
        iva2['p_tes'][i] = iva2['v_tes'][i]**2 / R_tes
        
        iva2['R'][i] = R_tes
        iva2['R_n'][i] = R_n
        iva2['R_L'][i] = R_L

    calculate_rpcurve(iva2, minrfrac=0.1, maxrfrac=0.999)
    recompute_psats(iva2, psat_level)
    recompute_si(iva2)
    
    return iva2

###############################################################################
# BSA translation functions below:
# 1. Fit for delta_Popt by fitting the associated IVA RP_curve to 
#     the measured dI/dIb from BSA.
# 2. Recompute R, Si, Pj, and I0 based on RP_cuve shifted by delta_Popt
# 3. Save updated BSAnalysis object as a numpy file.
###############################################################################
def rerun_analysis(bsa, iva2=None, R0_thresh=30e-3):
    """
    Runs the bias step analysis. Re-calculates TES parameters from RP curve
    created from the associated IVAnalysis object. 
    
    **NOTE: User must run reanalyze_iv(IVA),
    and pass the new IVA into this function.**

    Parameters:
        bs_file (string):
            Path to bias step analysis numpy file to load in and re-analyze
        iva2 (dictionary):
            Corrected IVAnalysis object loaded from .npy file
        R0_thresh (float):
            Any channel with resistance greater than R0_thresh will be
            unassigned from its bias group under the assumption that it's
            crosstalk
    Saves:
        dP_opts:
            Array of shape (nchans) containing change in optical power
            of each detector. Units of [uW]-1
        I0:
            Array of shape (nchans) containing I_tes of each detector.
            Same units as original BSA I0
        R0:
            Array of shape (nchans) containing R_tes of each detector.
            Same units as original BSA R0
        Pj:
            Array of shape (nchans) containing P_tes of each detector.
            Same units as original BSA R0
        Si:
            Array of shape (nchans) containing Si of each detector.
            Same units as original BSA Si.
        Rfrac:
            Array of shape (nchans) containing Rfrac of each detector.
        R_n_IV:
            Array of shape (nchans) containing R_n of each detector.

    """
    bsa2 = deepcopy(bsa)
    assert iva2 is not None
    
    bias_line_resistance = bsa2['meta']['bias_line_resistance']
    R_sh = bsa2['meta']['R_sh']
    bsa2['Vthevenin'] = bsa2['Ibias'] * R_sh
    fit_dP_opts(bsa2, iva2)
    recompute_dc_params(bsa2, iva2)
    recompute_bsa_rfrac(bsa2, iva2)
    
    return bsa2


def fit_dP_opts(bsa, iva, rmin=0.1, rmax=0.99):
    """
    Fits change in optical power between IVA and BSA
    operations.
    """
    dP_opts = np.full(iva['nchans'], np.nan)
    Rsh = bsa['meta']['R_sh']
    
    n = int(np.ceil(max([1. / rmin, 1. / (1. - rmax), 1000])))
    n = min([n, 100000])
    r = np.linspace(rmin, rmax, n+1)
    dr = (rmax - rmin) / n
    r0 = 0.5
    
    alpha = np.nan
    Rn = np.nan
    def func_bs_model(i_bias, dP_opt=0.):
        P = np.cumsum(dr / (r * alpha))
        P += - np.interp(r0, r, P) + P_opt + dP_opt
        R = r * Rn
        ok = (P > 0.) * (R > Rsh) 
        P = P[ok]
        R = R[ok]
        L0 = alpha[ok] * P
        i_tes = np.sqrt(P / R)
        i_bias1 = i_tes * (R + Rsh) / Rsh * 1e-3
        irat = (1. - L0) / ((1. - L0) + (1. + L0) * R / Rsh)
        result = np.interp(i_bias, i_bias1, irat)
        return result
    
    for ch in range(iva['nchans']):
        try:
            bg = iva['bgmap'][ch]
            if bg == -1:
                continue
            Ib_bs = bsa['Ibias'][bg] * 1e3
            Irat_bs = bsa['dItes'][ch] / bsa['dIbias'][bg]
            Rn = iva['R_n'][ch]
            RP_params = iva['RP_popts'][ch]
            logp0 = RP_params[0]
            p1 = RP_params[1]
            p2 = RP_params[2]
            logp3 = RP_params[3]
            P_opt = RP_params[4]

            logalpha = model_logalpha(
                r, logp0=logp0, p1=p1, p2=p2, logp3=logp3)
            alpha = np.exp(logalpha)
            popt, pcov = curve_fit(func_bs_model, np.atleast_1d(Ib_bs), np.atleast_1d(Irat_bs), p0 = P_opt*1e-2)
            dP_opts[ch] = popt

        except:
            continue
  
    bsa['dP_opts'] = dP_opts
    return bsa['dP_opts']
    
def recompute_TES_params_fromcurve(bsa2, iva, rmin=0.1, rmax=0.99):
    """
    Computes I0, R0, Pj, Si from RP fitted RP curve.
    """
    n = int(np.ceil(max([1. / rmin, 1. / (1. - rmax), 1000])))
    n = min([n, 100000])
    r1 = np.linspace(rmin, rmax, n+1)
    dr = (rmax - rmin) / n
    r0 = 0.5
    Rsh = bsa2['meta']['R_sh']
    
    I0s = np.full(iva['nchans'], np.nan)
    R0s = np.full(iva['nchans'], np.nan)
    Pjs = np.full(iva['nchans'], np.nan)
    Sis = np.full(iva['nchans'], np.nan)
    
    for ch in range(iva['nchans']):
        try:
            bg = iva['bgmap'][ch]
            if bg == -1:
                continue

            RP_params = iva['RP_popts'][ch]
            Rn = iva['R_n'][ch]
            logp0 = RP_params[0]
            p1 = RP_params[1]
            p2 = RP_params[2]
            logp3 = RP_params[3]
            P_opt = RP_params[4]
            dP_opt = bsa2['dP_opts'][ch]
            Ib_bs = bsa2['Ibias'][bg]

            logalpha1 = model_logalpha(r1, logp0, p1, p2, logp3)
            alpha1 = np.exp(logalpha1)

            P1 = np.cumsum(dr / (r1 * alpha1))
            P1 += - np.interp(r0, r1, P1) + P_opt + dP_opt
            R1 = r1 * Rn
            ok = (P1 > 0.) * (R1 > Rsh)
            P1 = P1[ok]
            R1 = R1[ok]
            L01 = alpha1[ok] * P1
            L1 = (R1 - Rsh) / (R1 + Rsh) * L01 
            I1 = np.sqrt(P1 / R1)
            Ib1 = I1 * (R1 + Rsh) / Rsh
            Si1 = -1 / (I1*(R1 - Rsh)) * (L1 / (L1 + 1))

            I0 = np.interp(Ib_bs*1e3, Ib1*1e-3, I1) 
            R0 = np.interp(Ib_bs*1e3, Ib1*1e-3, R1)
            Pj = np.interp(Ib_bs*1e3, Ib1*1e-3, P1)
            Si = np.interp(Ib_bs*1e3, Ib1*1e-3, Si1) 
            I0s[ch] = I0*1e-6
            R0s[ch] = R0
            Pjs[ch] = Pj*1e-12
            Sis[ch] = Si*1e6

        except:
            continue
            
    
    return I0s, R0s, Pjs, Sis    
   

def recompute_dc_params(bsa2, iva2=None, R0_thresh=30e-3):
    """
    Calculates v_thevenin from the bias steps numpy file, and then
    runs the DC param calc to re-estimate R0, I0, Pj, etc.
    Args:
        bsa2: (dict)
            Re-analyzed BSA using RP curve
        iva2: (dict)
            Re-analyzed IVA using RP curve
        R0_thresh: (float)
            Any channel with resistance greater than R0_thresh will be
            unassigned from its bias group under the assumption that it's
            crosstalk

    Returns:
        I0:
            Array of shape (nchans) containing I_tes of each detector.
            Same units as original BSA I0
        R0:
            Array of shape (nchans) containing R_tes of each detector.
            Same units as original BSA R0
        Pj:
            Array of shape (nchans) containing P_tes of each detector.
            Same units as original BSA R0
        Si:
            Array of shape (nchans) containing Si of each detector.
            Same units as original BSA Si.
    """

    I0, R0, Pj, Si = recompute_TES_params_fromcurve(bsa2, iva2)
    
    # If resistance is too high, most likely crosstalk so just reset
    # bg mapping and det params
    if R0_thresh is not None:
        m = np.abs(R0) > R0_thresh
        bsa2['bgmap'][m] = -1
        for arr in [R0, I0, Pj, Si]:
            arr[m] = np.nan

    bsa2['I0'] = I0
    bsa2['R0'] = R0
    bsa2['Pj'] = Pj
    bsa2['Si'] = Si

    return I0, R0, Pj, Si


def recompute_bsa_rfrac(bsa2, iva2):
    """
    Recomputes Rfrac using *corrected* R, Rn from IVAnalysis
    and *corrected* R0 from BSAnalysis.
    
    Parameters:
        bsa2 (dictionary):
            Corrected bias step analysis object (loaded from .npy file)
        iva2 (dictionary):
            Corrected iv curve analysis object (loaded from .npy file)
    
    Returns:
    R_n_IV : array of (nchans)
        Normal resistance from corrected IVAanalysis Object
    Rfrac : array of (nchans)
        R0 / R_n_IV
    """
    bsa2['R_n_IV'] = iva2['R_n']
    bsa2['Rfrac'] = bsa2['R0'] / bsa2['R_n_IV'] 
import numpy as np
import scipy
import so3g
from sotodlib import coords, tod_ops
from pixell import enmap
from sotodlib.coords.helpers import get_deflected_sightline


###########################Noise modeling###################################
def noise_model(f, params):
    alpha,f_knee,Nwhite = params
    return Nwhite*(1 + (f/f_knee)**alpha)

def neglnlike(params, f, pxx, bin_size=1):
    model = noise_model(f, params)
    output = np.nansum((np.log(model) + pxx / model)*bin_size)
    if not np.isfinite(output):
        return 1.0e30
    return output

def fit_noise_model(f, pxx, fmin = 5e-3, fmax = 1.95, alpha_est=-3.8,fknee_est=0.1,wn_est=2e-9,):
    '''Default values are reasonable for PSD taken in K_cmb units'''
    import sys
    fbin,bin_size = tod_ops.fft_ops.log_binning(f, unbinned_mode=3, base=1.05, mask=None,
                    return_bin_size=True, drop_nan=False)
    pxxbin = tod_ops.fft_ops.log_binning(pxx, unbinned_mode=3, base=1.05, mask=None,
                return_bin_size=False,drop_nan=False)

    fmsk = (fbin>=fmin) & (fbin<=fmax)
    #don't fit to freqs near the scan frequency and the first harmonic of the HWPSS
    bin_msk1 = (fbin > (1/80)*0.9) & (fbin < (1/80)*1.1) #tuned for satp3
    bin_msk2 = (fbin > (1/160)*0.8) & (fbin < (1/160)*1.2)
    bin_msk3 = (fbin > (2)*0.8) & (fbin < (2)*1.2)
    bin_msk = ~(bin_msk1 | bin_msk2 | bin_msk3)
    fmsk = fmsk & bin_msk

    initial_params = np.array([alpha_est,fknee_est,wn_est])
    bounds= ((-10., 10.),(sys.float_info.min, None), (sys.float_info.min, None))
    p0 = initial_params.T
    fit_func = lambda params: neglnlike(params,fbin[fmsk], pxxbin[fmsk],bin_size[fmsk])
    res = scipy.optimize.minimize(fit_func,
                           p0, bounds=bounds, method="Nelder-Mead")
    return res

##############Sim injection##############################
def tod_from_pixell_map(tod, signal_map, site = 'so', weather='toco', cuts=None, flip_gamma=True, wrap=False, 
                        modulated=False, apply_wobble=False, add_to_tod=False, apply_abscal=False, apply_relcal=False, angledelay=None, reverse_hwp=False):
    """
    Generate simulated TOD with HWP from a given signal map.

    Args:
        tod : an axisManager object
        signal_map: pixell.enmap.ndmap containing (Tmap, Qmap, Umap) representing the signal.
        cuts (RangesMatrix, optional): Cuts to apply to the data. Default is None.
        flip_gamma (bool, optional): Whether to flip detector coordinate. If you use the HWP, keep it `True`. Default is True.
        wrap (bool, optional): Whether to wrap the simulated data. Default is False.
        modulated (bool, optional): If True, return modulated signal. If False, return the demodulated signal 
        (`dsT`, `demodQ`, and `demodU`). Default is False. 
        apply_wobble (bool, optional): If True, will apply wobble correction. Default is False.
        angledelay (float, optional): Angle delay for the HWP. Default is None.
        reverse_hwp (bool, optional): If True, will reverse the HWP angle. Default is False.
        add_to_tod (bool, optional): If True, will add the simulated signal to existing TOD signal instead of overwriting it. Default is False.

    Returns:
        `modulate==False`: A tuple containing the TOD (np.array) of dsT, demodQ and demodU.
        `modulate==True` : The modulated TOD (np.array)
        
    """
    Tmap, Qmap, Umap = signal_map

    if apply_wobble:
        wobble_meta = tod.wobble_params
        sight = get_deflected_sightline(tod, wobble_meta, site=site, weather=weather)
    else:
        sight = so3g.proj.CelestialSightLine.az_el(tod.timestamps, tod.boresight.az, tod.boresight.el, roll=tod.boresight.roll, weather=weather, site=site)

    if reverse_hwp:
        tod.hwp_angle = np.flip(tod.hwp_angle)
    
    P = coords.P.for_tod(tod, sight=sight, geom=signal_map.geometry, hwp=flip_gamma, comps='QU', cuts=cuts)
    dsT_sim = P.from_map(Tmap, comps='T')
    demodQ_sim = P.from_map(enmap.enmap([Qmap, Umap]), comps='QU')
    demodU_sim = P.from_map(enmap.enmap([Umap, -Qmap]), comps='QU')

    if modulated is False:
        if wrap:
            tod.wrap('dsT_sim', dsT_sim, [(0, 'dets'), (1, 'samps')])
            tod.wrap('demodQ_sim', demodQ_sim, [(0, 'dets'), (1, 'samps')])
            tod.wrap('demodU_sim', demodU_sim, [(0, 'dets'), (1, 'samps')])
                    
        return dsT_sim, demodQ_sim, demodU_sim
    else:
        assert 'hwp_angle' in tod._assignments,'hwp angle not found in metadata'
        if angledelay is None:
            angledelay = np.zeros_like(tod.hwp_angle)
        signal_sim = dsT_sim + demodQ_sim*np.cos(4*(tod.hwp_angle+angledelay)) + demodU_sim*np.sin(4*(tod.hwp_angle+angledelay))
        signal_sim *= 1e-6 #convert to K_cmb units, assumes input map in uK_cmb units
        if add_to_tod:
            if apply_abscal:
                assert 'abscal' in tod._assignments, 'abscal metadata not found'
                signal_sim = np.multiply(signal_sim.T,1/tod.abscal.abscal_cmb).T #convert to pW, requires that the abscal metadata be included in the contextfile
            if apply_relcal:
                assert 'relcal' in tod._assignments, 'relcal metadata not found'
                signal_sim = np.divide(signal_sim.T,tod.relcal.rel_factor).T #apply relative calibration if included in the metadata
                
            tod['signal'] += signal_sim
            
        if wrap:
            tod.wrap('signal_sim', signal_sim, [(0, 'dets'), (1, 'samps')])
        return signal_sim

def inject_demod_1_over_f(obs ,common_alpha=[-4.8,-3.8],common_knee=[85*1e-3,115*1e-3], common_wn = [2e-3,2e-3],  only_common = True, 
                          det_wn=None, det_alpha=None, det_knee=None):
    
    n_samp = obs.samps.count
    common = np.random.normal(size=n_samp)
    common_fft = np.fft.rfft(common)
    dt = np.median(np.diff(obs.timestamps))
    fs = 1 / dt
    freqs = np.fft.rfftfreq(n_samp, d=dt)
    commonU_fft = common_fft.copy()
    commonQ_fft = common_fft.copy()
    commonU_fft[0] = 0
    commonQ_fft[0] = 0
    commonU_fft[1:] *=  np.sqrt(noise_model(freqs[1:], (common_alpha[1], common_knee[1], common_wn[1]))) 
    commonQ_fft[1:] *= np.sqrt(noise_model(freqs[1:], (common_alpha[0], common_knee[0], common_wn[0]))) 
    commonU = np.fft.irfft(commonU_fft, n=n_samp)
    commonQ = np.fft.irfft(commonQ_fft, n=n_samp)
    del common, common_fft, commonU_fft, commonQ_fft

    if det_wn is None:
        det_wn = [obs.preprocess['noiseQ_mapmaking']['white_noise']**2,obs.preprocess['noiseU_mapmaking']['white_noise']**2]
    
    #calculate scaling factor for injected noise spectrum based on white noise level of detectors in the observation
    sigmaQ = np.sqrt(det_wn[0] * fs / 2)[:,np.newaxis] # per-sample RMS
    sigmaU = np.sqrt(det_wn[1] * fs / 2)[:,np.newaxis]  # per-sample RMS

    if only_common:
        obs.demodQ = commonQ * sigmaQ 
        obs.demodU = commonU * sigmaU
        del commonQ, commonU
        return

    else:
        if not det_alpha is not None and det_knee is not None:
            raise ValueError("Both det_alpha and det_knee must be specified lists to add per detector uncorrelated noise")
        for idet in np.arange(obs.dets.count):
            # Uncorrelated Q noise
            detQ = np.random.normal(size=n_samp)
            detQ_fft = np.fft.rfft(detQ)
            detQ_fft[0] = 0
            detQ_fft[1:] *= np.sqrt(noise_model(freqs[1:], det_alpha[0][idet], det_knee[0][idet], 0, 1))
            detQ = np.fft.irfft(detQ_fft, n=n_samp)
            # Uncorrelated U noise
            detU = np.random.normal(size=n_samp)
            detU_fft = np.fft.rfft(detU)
            detU_fft[0] = 0
            detU_fft[1:] *= np.sqrt(noise_model(freqs[1:], det_alpha[1][idet], det_knee[1][idet], 0, 1))
            detU = np.fft.irfft(detU_fft, n=n_samp)
            
            # Co-add with common mode to get total noise
            obs.demodQ[idet] = (commonQ + detQ) * sigmaQ
            obs.demodU[idet] = (commonU + detU) * sigmaU
        del commonQ, commonU, detQ, detU
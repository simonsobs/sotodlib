import numpy as np
import matplotlib.pyplot as plt
import lmfit
from pathlib import Path
import ruptures as rpt

from sotodlib import tod_ops, core
from sotodlib.io import hkdb
from sotodlib.stimulator.plot_stimulator import plot_hkdata, plot_tod
from sotodlib.stimulator.utils_stimulator import func_sines, func_response_amplitude, func_response_phase_with_dt  

CHOPPING_FREQS = {'f1': 6, 'f2': 15, 'f3': 33,'f4': 63, 'f5': 93, 'f6': 123, 'f7': 147}


def get_hk(hkdb_cfg, aman=None, t_start=None, t_end=None):
    """
    Get housekeeping data for one axis manager.
    
    Args:
        hkdb_cfg: HK database config.
        aman: Axis manager of detector data.
        t_start: Start time of HK data. If None, use aman's start time.
        t_end: End time of HK data. If None, use aman's end time.

    Return:
        Housekeeping data.
    """
    feed_list = ['stimulator-blh.motor.*',
                 'stimulator-ds378.relay.*',
                 'stimulator-enc.stim_enc.*',
                 'stimulator-enc.stim_enc_downsampled.*',
                 'stimulator-pcr500ma.heater_source.*',
                 'stimulator-thermo.temperatures.*'] 

    if t_start is None:
        t_start = aman.timestamps[0]
        
    if t_end is None:
        t_end = aman.timestamps[-1]


    # Load stimulator's data
    lspec = hkdb.LoadSpec(
        hkdb_cfg, 
        fields=feed_list,
        start=t_start, 
        end=t_end,
    )
    result = hkdb.load_hk(lspec) 

    return result
    

def get_downsample_factor(aman, ctx):
    """
    Get downsample factor of SMuRF readout for the axis manager data.
    
    Args:
        aman: Axis manager of detector data
        ctx: context file
    """
    downsample_factor_tag = get_downsample_factor_tags(ctx)

    obs_list = ctx.obsdb.query(
        f"obs.obs_id == '{aman.obs_info.obs_id}'",
        tags=downsample_factor_tag
        )

    obs = obs_list[0]
    for tag in downsample_factor_tag:
        if obs[tag] == 1:
            downsample_factor = int(tag.split('_')[-1])

    aman.obs_info.wrap('downsample_factor', downsample_factor, overwrite=True)
    aman.obs_info.wrap('sampling_rate', 4000/downsample_factor, overwrite=True)


def get_downsample_factor_tags(ctx):
    """
    Get downsample factor of SMuRF readout.
    
    Args:
        ctx: context file

    Return:
        List of downsample factor tags in the database.
    """
    cursor = ctx.obsdb.conn.execute("SELECT DISTINCT tag FROM tags")
    all_tags = np.array([row[0] for row in cursor.fetchall()])
    mask = np.char.find(all_tags,'downsample') != -1
    
    return np.array(all_tags)[mask].tolist()
    

def calc_gain(aman, hkdata, idxs=None, bool_plot=False, bool_save=False, bool_preprocess=True, output_dir=None):
    """
    Calculate the gain of the detectors.
    
    Args:
        aman: Axis manager of TOD data, including timestamps and raw signal.
        hkdata: HK data for aman.
        idxs: List of detector indices for calculation. If None, all detectors are calculated.
        bool_plot: If true, make plots.
        bool_save: If true, save plots.
        bool_preprocess: If True, perform preprocessing before calculation.
        output_dir: Directory to save plots.

    Return:
        bool: True if data is valid and calculation is performed, False otherwise.
    """
    det_mask = np.full(aman.dets.count, False)
    if idxs is None:
        det_mask[:] = True
    else:
        det_mask[idxs] = True

    valid_data = True
    if bool_preprocess:
        valid_data = get_encoder_timing(aman, hkdata)  # Get timing against encoder t0

        get_chopping_status(aman)

        get_timing_cut(aman)

        get_signal_temp(aman,hkdata)

    if bool_plot:
        fig_hk, axes_hk = plot_hkdata(aman,hkdata,cal_type='gain')
        if not bool_plot:
            plt.close(fig_hk)
   
    if not valid_data:
        return valid_data


    chopping_freqs = {}
    if round(aman.stm_ana.chopping_freqs[0]) != CHOPPING_FREQS['f1']: 
        chopping_freqs['f1_gain'] = round(aman.stm_ana.chopping_freqs[0])
    else:
        chopping_freqs['f1_gain'] = CHOPPING_FREQS['f1']


    filtering_params = filtering(aman,chopping_freqs,'gain')


    model, params_base = get_fit_params(cal_type='gain')

    # Make and get co-added data
    coadd_data, fit_result = get_dicts('gain')
    n_bins = 40
    get_coadd_data(aman,coadd_data,n_bins,det_mask)

    for i_det,m in enumerate(det_mask):
        if not m:
            fill_none(fit_result=fit_result) 
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            fill_none(fit_result=fit_result) 
            continue
            
        
        # Fitting
        for filt_key in fit_result['fit_coadd'].keys():

            params = params_base.copy()

            x = coadd_data[filt_key]['f1_gain']['x'][i_det]
            y = coadd_data[filt_key]['f1_gain']['y'][i_det]
            yerr = coadd_data[filt_key]['f1_gain']['yerr'][i_det]

            if y is None:
                fit_result['fit_coadd'][filt_key]['f1_gain'].append(None)
                continue

            if filt_key=='lpf':
                params['a1'].set(value=0,vary=False)
                params['a2'].set(value=0,vary=False)
                params['a3'].set(value=0,vary=False)
                params['a4'].set(value=0,vary=False)
                params['a5'].set(value=0,vary=False)
                params['a6'].set(value=0,vary=False)
                params['t1'].set(value=0,vary=False)
                params['t2'].set(value=0,vary=False)
                params['t3'].set(value=0,vary=False)
                params['t4'].set(value=0,vary=False)
                params['t5'].set(value=0,vary=False)
                params['t6'].set(value=0,vary=False)
                result = model.fit(y, params, t=x, weights=1/np.array(yerr)) 
            else:
                result = model.fit(y, params, t=x, weights=1/np.array(yerr)) 

            fit_result['fit_coadd'][filt_key]['f1_gain'].append(result)

        if not bool_plot and not bool_save:
            pass
        else: 
            fig_tod, axes_tod = plot_tod(aman,i_det,coadd_data,fit_result,filtering_params,cal_type='gain')
            
            if bool_save:
                obs_id = aman.obs_info.obs_id
                if output_dir is not None:
                    ufm = aman.det_info.stream_id[i_det][4:]
                    ufm = ufm[0].upper() + ufm[1:]
                    output_dir_ = Path(f'{output_dir}/{ufm}_{obs_id}')
                    output_dir_.mkdir(parents=True, exist_ok=True)
                    plt.savefig(f'{output_dir_}/Gain_det{i_det:04d}.png')
            if not bool_plot:
                plt.close(fig_tod)
            
    fill_data(aman,coadd_data,fit_result,n_bins,cal_type='gain')    
    
    return valid_data


def calc_timeconstant(aman, hkdata, idxs=None, bool_plot=False, bool_save=False, bool_preprocess=True, output_dir=None):
    """
    Calculate the time constant of the detectors.
    
    Args:
        aman: Axis manager of TOD data, including timestamps and raw signal.
        hkdata: HK data for aman.
        idxs: List of detector indices for calculation. If None, all detectors are calculated.
        bool_plot: If true, make plots.
        bool_save: If true, save plots.
        bool_preprocess: If True, perform preprocessing before calculation.
        output_dir: Directory to save plots.

    Return:
        bool: True if data is valid and calculation is performed, False otherwise.
    """
    det_mask = np.full(aman.dets.count, False)
    if idxs is None:
        det_mask[:] = True
    else:
        det_mask[idxs] = True

    valid_data = True
    if bool_preprocess:
        valid_data = get_encoder_timing(aman, hkdata)  # Get timing against encoder t0

        get_chopping_status(aman)

        get_timing_cut(aman)
        if aman.stm_ana.chopping_freqs.shape[0] <= 2:
            valid_data = False

        get_signal_temp(aman,hkdata)

    if bool_plot:
        fig_hk, axes_hk = plot_hkdata(aman,hkdata,cal_type='timeconstant')
        if not bool_plot:
            plt.close(fig_hk)
    if not valid_data:
        return valid_data
    
    chopping_freqs = {}
    for i,(key,f) in enumerate(CHOPPING_FREQS.items()):
        if round(aman.stm_ana.chopping_freqs[0]) != f: 
            chopping_freqs[key] = round(aman.stm_ana.chopping_freqs[i])
        else:
            chopping_freqs[key] = f

    filtering_params = filtering(aman,chopping_freqs,'timeconstant')

    models, params_bases = get_fit_params(cal_type='timeconstant')
    
    coadd_data, fit_result = get_dicts('timeconstant')
    n_bins = 40
    get_coadd_data(aman,coadd_data,n_bins,det_mask)


    for i_det,m in enumerate(det_mask):
        if not m:
            fill_none(fit_result=fit_result) 
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            fill_none(fit_result=fit_result) 
            continue
            
        # Fitting
        for filt_key in fit_result['fit_coadd'].keys():

            for f_key in chopping_freqs.keys():
                params = params_bases['fit_coadd'].copy()
                
                x = coadd_data[filt_key][f_key]['x'][i_det]
                y = coadd_data[filt_key][f_key]['y'][i_det]
                yerr = coadd_data[filt_key][f_key]['yerr'][i_det]

                if y is None:
                    fit_result['fit_coadd'][filt_key][f_key].append(None)
                    continue

                if filt_key=='lpf':
                    params['a1'].set(value=0,vary=False)
                    params['a2'].set(value=0,vary=False)
                    params['a3'].set(value=0,vary=False)
                    params['a4'].set(value=0,vary=False)
                    params['a5'].set(value=0,vary=False)
                    params['a6'].set(value=0,vary=False)
                    params['t1'].set(value=0,vary=False)
                    params['t2'].set(value=0,vary=False)
                    params['t3'].set(value=0,vary=False)
                    params['t4'].set(value=0,vary=False)
                    params['t5'].set(value=0,vary=False)
                    params['t6'].set(value=0,vary=False)
                    result = models['fit_coadd'].fit(y, params, t=x, weights=1/np.array(yerr)) 
                else:
                    result = models['fit_coadd'].fit(y, params, t=x, weights=1/np.array(yerr)) 
    
                fit_result['fit_coadd'][filt_key][f_key].append(result)

            a0s = [fit_result['fit_coadd'][filt_key][f_key][-1].best_values['a0'] if fit_result['fit_coadd'][filt_key][f_key][-1] is not None else np.nan for f_key in fit_result['fit_coadd'][filt_key].keys()]
            t0s = [fit_result['fit_coadd'][filt_key][f_key][-1].best_values['t0'] if fit_result['fit_coadd'][filt_key][f_key][-1] is not None else np.nan for f_key in fit_result['fit_coadd'][filt_key].keys()]

            # Edit a0 and t0 result
            for i in range(len(t0s)):
                if a0s[i] < 0:
                    t0s[i] = t0s[i]-0.5
                if i > 0:
                    if t0s[i] < t0s[i-1]:
                        t0s[i] = t0s[i] + 1
            a0s = np.abs(a0s)
            t0s = np.array(t0s)

            f = [chopping_freqs[f_key] for f_key in chopping_freqs.keys()]
            for fit_key in ['fit_amp', 'fit_phase__fix_tau', 'fit_phase__free']:
                params = params_bases[fit_key].copy()
                if fit_key == 'fit_amp':
                    if np.isnan(a0s).all():
                        fit_result[fit_key][filt_key].append(None)
                        continue

                    weights = [fit_result['fit_coadd'][filt_key][f_key][-1].params['a0'].stderr if fit_result['fit_coadd'][filt_key][f_key][-1] is not None and fit_result['fit_coadd'][filt_key][f_key][-1].params['a0'].stderr is not None else np.nan for f_key in fit_result['fit_coadd'][filt_key].keys()]
                    weights = np.array(weights)
                    result = models[fit_key].fit(a0s,params,f=f, method='least_squares')
                else:
                    if np.isnan(t0s).all():
                        fit_result[fit_key][filt_key].append(None)
                        continue

                    weights = [fit_result['fit_coadd'][filt_key][f_key][-1].params['t0'].stderr if fit_result['fit_coadd'][filt_key][f_key][-1] is not None and fit_result['fit_coadd'][filt_key][f_key][-1].params['t0'].stderr is not None else np.nan for f_key in fit_result['fit_coadd'][filt_key].keys()]
                    weights = np.array(weights) *360
                    if fit_key == 'fit_phase__fix_tau':
                        if fit_result['fit_amp'][filt_key][-1] is not None:
                            params['tau'].set(value=fit_result['fit_amp'][filt_key][-1].best_values['tau'],vary=False)
                            result = models[fit_key].fit(-np.array(t0s)*360,params,f=f, method='least_squares')
                        else:
                            result = None
                    else:
                        result = models[fit_key].fit(-np.array(t0s)*360,params,f=f, method='least_squares')
                fit_result[fit_key][filt_key].append(result)
        
        
        if not bool_plot and not bool_save:
            pass
        else:
            fig_tod, axes_tod = plot_tod(aman,i_det,coadd_data,fit_result,filtering_params,cal_type='timeconstant')

            obs_id = aman['obs_info']['obs_id']
            if bool_save:
                ufm = aman.det_info.stream_id[i_det][4:]
                ufm = ufm[0].upper() + ufm[1:]
                output_dir_ = Path(f'{output_dir}/{ufm}_{obs_id}')
                output_dir_.mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{output_dir_}/Tau_det{i_det:04d}.png')
            if not bool_plot:
                plt.close(fig_tod)

    
    fill_data(aman,coadd_data,fit_result,n_bins,cal_type='timeconstant')    
        
    return valid_data


def get_encoder_timing(aman, hkdata):
    """
    Get timing against encoder start time t0.
    
    Args:
        aman: Axis manager with the detector data.
        hkdata: The housekeeping data.
    
    Returns:
        bool: True if timing data is valid, False otherwise.
    """
    state = np.array(hkdata.data['stimulator-enc.stim_enc.state'][1])
    t_enc = np.array(hkdata.data['stimulator-enc.stim_enc.timestamps_tai'][1])[state == 0]-37
    t_hk = np.array(hkdata.data['stimulator-enc.stim_enc.timestamps_tai'][0])[state == 0]

    # Cut Encoder data not to contain next run data
    # Take 30s buffer from last TOD data
    mask = t_hk < aman.timestamps[-1]+30
    t_enc = t_enc[mask]
    t_hk = t_hk[mask]

    # Get timing against encoder t0 
    i_enc=0
    frac_timing = []
    for t in aman.timestamps:
        if i_enc >= len(t_enc)-2:
            continue
        t1_enc_tmp = t_enc[i_enc]
        t2_enc_tmp = t_enc[i_enc+1]
        
        while t2_enc_tmp < t:
            i_enc += 1
            t1_enc_tmp = t_enc[i_enc]
            t2_enc_tmp = t_enc[i_enc+1]

        frac_timing.append((t - t1_enc_tmp)/(t2_enc_tmp-t1_enc_tmp))
    

    if 'stm_samps' not in aman._axes.keys():
        stm_samps = core.IndexAxis('stm_samps',t_enc.size)
        stm_ana = core.AxisManager(stm_samps)
        aman.wrap('stm_ana',stm_ana,overwrite=True)
    aman.stm_ana.wrap('t_enc', t_enc, [(0,'stm_samps')], overwrite=True)
    aman.stm_ana.wrap('t_hk', t_hk, [(0,'stm_samps')], overwrite=True)

    valid_data = True
    if t_enc[-1] < aman.timestamps[-1]:
        print(f'Invalid HK data. last t_enc: {t_enc[-1]}, last TOD timestamp: {aman.timestamps[-1]}')
        valid_data = False
    else:
    # Add timing against encoder to axis manager
        aman.wrap('frac_timing', np.array(frac_timing), [(0,'samps')], overwrite=True)

    return valid_data

    
def get_chopping_status(aman):
    """
    Get chopping frequency and timing when the chopping speed changes.
    
    Args:
        aman: axis manager with aman.stm_ana field
    """
    xx0 = aman.stm_ana.t_enc[:-1]
    xx1 = aman.stm_ana.t_enc[1:]
    x = (xx0+xx1)/2 - aman.timestamps[0]
    y = 1/(xx1-xx0)

    x_min = int(x.min())
    x_max = int(min(x.max(),aman.timestamps[-1]-aman.timestamps[0]))
    if x_min < 0:
        x_min = 0

    # Get 1s average data
    n_bins = x_max-x_min # 1s bin
    data = [[] for _ in range(n_bins)]
    for i_bin in range(n_bins):
        cut = (i_bin <= x) & (x < i_bin+1)
        data[i_bin] = y[cut]

    x_ave = np.linspace(x_min,x_max-1,n_bins)+0.5
    y_ave = np.array([np.nanmean(d) for d in data])

    # Get breakout points where the chopping speed changes
    model1 = rpt.Pelt(model='l2',min_size=10).fit(y_ave)
    bkps = model1.predict(pen=100) # index +1

    # Re-calculate the breakout points
    refined_bkps = []
    window = 5
    for b in bkps:
        left = max(0, b - window)
        right = min(len(y_ave), b + window)

        local_diff = np.abs(np.diff(y_ave[left:right]))

        # Cut small fluctuation
        if np.max(local_diff) < 1:
            continue
        idx = np.argmax(local_diff)

        refined_bkps.append(left + idx +1)

    # Add last point
    refined_bkps.append(x_ave.size) # index +1

    # Get average of chopping frequency 
    wait_t = 5
    delta_t = 5 # 5s average
    n_freq = len(refined_bkps)
    chopping_freqs = []
    for i_freq in range(n_freq):
        if i_freq==0:
            i_min = 0
        else:
            i_min = refined_bkps[i_freq-1]-1+wait_t

        chopping_freqs.append(np.average(y_ave[i_min:i_min+delta_t]))

    t_chopping_change = x_ave[np.array(refined_bkps)-1]
    t_chopping_change = np.concatenate([[0],t_chopping_change],0)

    aman.stm_ana.wrap('t_chopping_change',t_chopping_change,overwrite=True)
    aman.stm_ana.wrap('chopping_freqs',np.array(chopping_freqs),overwrite=True)


def get_timing_cut(aman, dt_gain=60, dt_timeconstant=10, dt_wait=9):
    """
    Get timing range for analysis.

    Args:
        aman: axis manager with aman.stm_ana field
        dt_gain: Time duration for gain analysis.
        dt_timeconstant: Time duration for time constant analysis from the time when chopping speed changes and dt_wait passed.
        dt_wait: Waiting time after chopping speed changes to avoid chopper's acceleration time.
    """
    if len(aman.stm_ana.t_chopping_change)==2:
        cal_type = 'gain'
    elif len(aman.stm_ana.t_chopping_change)>2:
        cal_type = 'gain_timeconstant'
    else:
        ValueError('Chopping information is incorrect.')


    t_cuts = []

    t_start = 2+aman.stm_ana.t_chopping_change[0]  # 2 seconds to avoid ringing by high-pass filter
    t_end = min(t_start+dt_gain, aman.stm_ana.t_chopping_change[1]) 
    t_cuts.append((t_start,t_end))

    if cal_type == 'gain_timeconstant':
        for i in range(len(aman.stm_ana.t_chopping_change)-1):
            t_start = aman.stm_ana.t_chopping_change[i] + dt_wait
            if i==0:
                t_start = 2+aman.stm_ana.t_chopping_change[i]  # 2 seconds to avoid ringing by high-pass filter
            t_end = min(t_start+dt_timeconstant, aman.stm_ana.t_chopping_change[i+1])
            t_cuts.append((t_start,t_end))

    if cal_type == 'gain':
        label_freqs = core.LabelAxis('freqs',['f1_gain'])
    else:
        keys = ['f1_gain']
        for i in range(len(aman.stm_ana.t_chopping_change)-1):
            keys.append(f'f{int(i+1)}')
        label_freqs = core.LabelAxis('freqs',keys)
    aman.stm_ana.add_axis(label_freqs)

    aman.stm_ana.wrap('t_cuts',np.array(t_cuts),[(0,'freqs')],overwrite=True)


def get_signal_temp(aman, hkdata):
    """
    Get signal temperature for chopping frequency.

    Args:
        aman: axis manager with aman.stm_ana field
        hkdata: Housekeeping data including temperature data.
    """

    position_keys = ['heater','chopper_rear','chopper_front','air']
    keys = ['stimulator-thermo.temperatures.Channel_0_T',
            'stimulator-thermo.temperatures.Channel_4_T',
            'stimulator-thermo.temperatures.Channel_6_T',
            'stimulator-thermo.temperatures.Channel_5_T']
    temps = {}
    for position_key in position_keys:
        temps[position_key] = []

    for key,position_key in zip(keys,position_keys):
        x = hkdata.data[key][0] - aman.timestamps[0]
        y = hkdata.data[key][1]+273.15

        for t_min,t_max in aman.stm_ana.t_cuts:
            arr = y[(t_min <= x) & (x < t_max)]
            y_mean = np.mean(arr) if arr.size > 0 else np.nan
            temps[position_key].append(y_mean)

        temps[position_key] = np.array(temps[position_key])


    temps['env'] = (temps['chopper_rear']+temps['chopper_front']+temps['air'])/3
    position_keys.append('env')

    label_positions = core.LabelAxis('positions',position_keys)
    aman.stm_ana.add_axis(label_positions)
    arr = [x for x in temps.values()]
    arr = np.array(arr)

    aman.stm_ana.wrap('temps',arr,[(0,'positions'),(1,'freqs')],overwrite=True)


def filtering(aman, chopping_freqs, cal_type):
    """
    Filtering signal data.
    
    Args:
        aman: axis manager
        cal_type: 'gain' or 'timeconstant'. type of calibration
        chopping_freqs: dictionary of chopping frequencies
    """

    #Invert IIR filter
    iirc_filter = tod_ops.filters.iir_filter(aman, invert=True)
    signal_new = tod_ops.fourier_filter(aman, iirc_filter, signal_name='signal')
    aman.wrap('signal_iirc', signal_new, [(0,'dets'),(1,'samps')], overwrite=True)

    # Make HPFed data
    hpf_cutoff = 1
    hpf = tod_ops.filters.high_pass_sine2(hpf_cutoff)
    filtering__without_delay_filter(aman, hpf, signal_name_pre='signal_iirc', signal_name_new='signal_hpf') 

    # Make LPFed data
    cutoff_factor = 1.5
    width_fraction = 1/5

    if cal_type == 'gain':
        filter_cutoff = cutoff_factor*chopping_freqs['f1_gain']
        lpf = tod_ops.filters.low_pass_sine2(filter_cutoff,filter_cutoff*width_fraction)
        filtering__without_delay_filter(aman, lpf, signal_name_pre='signal_hpf', signal_name_new='signal_lpf') 

    elif cal_type == 'timeconstant':
        for key, chopping_freq in chopping_freqs.items():
            filter_cutoff = cutoff_factor*chopping_freq
            lpf = tod_ops.filters.low_pass_sine2(filter_cutoff,filter_cutoff*width_fraction)
            filtering__without_delay_filter(aman, lpf, signal_name_pre='signal_hpf', signal_name_new=f'signal_lpf_{key}') 

    else:
        raise ValueError(f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'.")

    filtering_params = {'hpf_cutoff': hpf_cutoff, 'lpf_cutoff_factor': cutoff_factor, 'lpf_width_fraction': width_fraction, 'chopping_freqs': chopping_freqs}

    return filtering_params


def filtering__with_delay_filter(aman, filter, signal_name_pre, signal_name_new):
    """
    Filtering signal data using the filter which has timing delay.
    
    Args:
        aman: axis manager
        filter: filter of tod_ops.filters
        signal_name_pre: signal name which will be filtered
        signal_name_new: New name for filtered signal
    """
    signal_new = tod_ops.fourier_filter(aman, filter, signal_name=signal_name_pre)
    signal_new = np.fliplr(signal_new)
    aman.wrap(signal_name_new, signal_new, [(0,'dets'),(1,'samps')], overwrite=True)
    
    signal_new = tod_ops.fourier_filter(aman, filter, signal_name=signal_name_new)
    signal_new = np.fliplr(signal_new)
    aman.wrap(signal_name_new, signal_new, [(0,'dets'),(1,'samps')], overwrite=True)


def filtering__without_delay_filter(aman, filter, signal_name_pre, signal_name_new):
    """
    Filtering signal data using the filter which doesn't have timing delay.
    
    Args:
        aman: axis manager
        filter: filter of tod_ops.filters
        signal_name_pre: signal name which will be filtered
        signal_name_new: New name for filtered signal
    """
    signal_new = tod_ops.fourier_filter(aman, filter, signal_name=signal_name_pre)
    aman.wrap(signal_name_new, signal_new, [(0,'dets'),(1,'samps')], overwrite=True)


def get_dicts(cal_type):
    """
    Get dictionaries for co-added data and fit result.
    
    Args:
        cal_type: 'gain' or 'timeconstant'. type of calibration

    Return:
        coadd_data: Dictionary for co-added data
        fit_result: Dictionary for fit result
    """
    filt_keys = ['iirc','hpf','lpf']

    coadd_data = {}
    fit_result = {}

    if cal_type == 'gain':
        freq_keys = ['f1_gain']
        fit_keys = ['fit_coadd']
    elif cal_type == 'timeconstant':
        freq_keys = CHOPPING_FREQS.keys()
        fit_keys = ['fit_coadd','fit_amp','fit_phase__fix_tau','fit_phase__free']
    else:
        raise ValueError(f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'.")

    # Initialize coadd_data dictionary
    for filt_key in filt_keys:
        coadd_data[filt_key] = {}

        for freq_key in freq_keys:
            coadd_data[filt_key][freq_key] = {}
            for key in ['x','y','yerr']:
                coadd_data[filt_key][freq_key][key] = []


    # Initialize fit_result dictionary
    for fit_key in fit_keys:
        fit_result[fit_key] = {}
        for filt_key in filt_keys:
            if filt_key != 'iirc':
                if fit_key == 'fit_coadd':
                    fit_result[fit_key][filt_key] = {}
                    for freq_key in freq_keys:
                        fit_result[fit_key][filt_key][freq_key] = []
                else: 
                    fit_result[fit_key][filt_key] = []


    return coadd_data, fit_result    


def fill_none(coadd_data=None, fit_result=None):
    """
    Fill None for co-added data and fit result when there is no valid data for calculation.
    
    Args:
        coadd_data: Dictionary for co-added data
        fit_result: List for fit result
    """
    if fit_result is not None:
        for fit_key in fit_result.keys():
            for filt_key in fit_result[fit_key].keys():
                if isinstance(fit_result[fit_key][filt_key], list):
                    fit_result[fit_key][filt_key].append(None)
                else:
                    for freq_key in fit_result[fit_key][filt_key].keys():
                        fit_result[fit_key][filt_key][freq_key].append(None)

    if coadd_data is not None:
        for filt_key in coadd_data.keys():
            for freq_key in coadd_data[filt_key].keys():
                for key in coadd_data[filt_key][freq_key].keys():
                    coadd_data[filt_key][freq_key][key].append(None)


def get_coadd_data(aman, coadd_data, n_bins, det_mask):
    """
    Making co-added data.
    
    Args:
        aman: axis manager of tod data, including timestamps and tod signal
        coadd_data: Dictionary for co-added data
        n_bins: # of bins for co-added data
        t_min: Minimum time for the data analysis
        t_max: Maximum time for the data analysis
    """
    t0 = aman.timestamps[0]
    bins = np.linspace(0,1-1/n_bins,n_bins)  
    x = bins + 1/n_bins/2

    # Get cuts for co-addition
    cuts = {}
    for freq_key in coadd_data['hpf'].keys():
        idx = np.where(aman.stm_ana.freqs.vals == freq_key)[0][0]
        t_min,t_max = aman.stm_ana.t_cuts[idx]

        cut1 = (t_min <= aman.timestamps-t0) & (aman.timestamps-t0 < t_max)

        cuts[freq_key] = []
        for i_bin in range(n_bins):
            cut2 = (i_bin/n_bins <= aman.frac_timing) & (aman.frac_timing < (i_bin+1)/n_bins)
            cut = cut1 & cut2
            cuts[freq_key].append(cut)
    
    for i_det,m in enumerate(det_mask):
        if not m:
            fill_none(coadd_data=coadd_data)
            continue
        if not np.isfinite(aman.signal[i_det]).all():
            fill_none(coadd_data=coadd_data)
            continue

        for filt_key in coadd_data.keys():
            for freq_key in coadd_data['hpf'].keys():
                if filt_key == 'hpf' or filt_key == 'iirc':
                    key = f'signal_{filt_key}'
                elif filt_key == 'lpf':
                    if freq_key == 'f1_gain':
                        key = 'signal_lpf'
                    else:
                        key = f'signal_lpf_{freq_key}'

                data = [[] for _ in range(n_bins)]
                for i_bin in range(n_bins):
                    data[int(i_bin)] = aman[key][i_det][cuts[freq_key][i_bin]]
                    

                y = np.array([np.nanmean(d) if not np.isnan(d).all() else np.nan for d in data])
                yerr = np.array([np.array(d).std(ddof=1)/np.sqrt(len(d)) if len(d) > 0 and not np.isnan(d).all() else np.nan for d in data])

                mask = np.isfinite(y)
                xx = x[mask]
                y = y[mask]
                yerr = yerr[mask]

                if y.size==0:
                    xx,y,yerr = (None, None, None)

                coadd_data[filt_key][freq_key]['x'].append(xx)
                coadd_data[filt_key][freq_key]['y'].append(y)
                coadd_data[filt_key][freq_key]['yerr'].append(yerr)


def get_fit_params(cal_type):
    """
    Get fit parameters for fitting.
    
    Args:
        cal_type: 'gain' or 'timeconstant'. type of calibration

    Return:
        model: lmfit model for fitting
        params_base: lmfit parameters for fitting
    """
    if cal_type == 'gain':
        model = lmfit.Model(func_sines)
       
        params_base = lmfit.Parameters()
        params_base.add('a0',value=1e-3)
        params_base.add('a1',value=2e-4)
        params_base.add('a2',value=2e-4)
        params_base.add('a3',value=1e-4)
        params_base.add('a4',value=1e-4)
        params_base.add('a5',value=1e-5)
        params_base.add('a6',value=1e-5)
        params_base.add('t0',value=0.5,min=0,max=1)
        params_base.add('t1',value=0.5,min=0,max=1)
        params_base.add('t2',value=0.5,min=0,max=1)
        params_base.add('t3',value=0.5,min=0,max=1)
        params_base.add('t4',value=0.5,min=0,max=1)
        params_base.add('t5',value=0.5,min=0,max=1)
        params_base.add('t6',value=0.5,min=0,max=1)

    elif cal_type == 'timeconstant':
        model = {}
        model['fit_coadd']          = lmfit.Model(func_sines)
        model['fit_amp']            = lmfit.Model(func_response_amplitude)
        model['fit_phase__fix_tau'] = lmfit.Model(func_response_phase_with_dt, independent_vars=['f'])
        model['fit_phase__free']    = lmfit.Model(func_response_phase_with_dt, independent_vars=['f'])

        params_base = {}
        params_base['fit_coadd'] = lmfit.Parameters()
        params_base['fit_amp'] = lmfit.Parameters()
        params_base['fit_phase__fix_tau'] = lmfit.Parameters()
        params_base['fit_phase__free'] = lmfit.Parameters()
        params_base['fit_coadd'].add('a0',value=1e-3)
        params_base['fit_coadd'].add('a1',value=2e-4)
        params_base['fit_coadd'].add('a2',value=2e-4)
        params_base['fit_coadd'].add('a3',value=1e-4)
        params_base['fit_coadd'].add('a4',value=1e-4)
        params_base['fit_coadd'].add('a5',value=1e-5)
        params_base['fit_coadd'].add('a6',value=1e-5)
        params_base['fit_coadd'].add('t0',value=0.5,min=0,max=1)
        params_base['fit_coadd'].add('t1',value=0.5,min=0,max=1)
        params_base['fit_coadd'].add('t2',value=0.5,min=0,max=1)
        params_base['fit_coadd'].add('t3',value=0.5,min=0,max=1)
        params_base['fit_coadd'].add('t4',value=0.5,min=0,max=1)
        params_base['fit_coadd'].add('t5',value=0.5,min=0,max=1)
        params_base['fit_coadd'].add('t6',value=0.5,min=0,max=1)
        params_base['fit_amp'].add('a',value=1e-3,min=0,max=1)
        params_base['fit_amp'].add('tau',value=1e-3,min=0,max=1)
        params_base['fit_phase__fix_tau'].add('theta_geo',value=0,min=-90,max=90)
        params_base['fit_phase__free']   .add('theta_geo',value=0,min=-90,max=90)
        params_base['fit_phase__fix_tau'].add('tau')
        params_base['fit_phase__free']   .add('tau',value=1e-3,min=0,max=0.1)
        params_base['fit_phase__fix_tau'].add('dt',value=0.125*1e-3,min=-3e-3,max=3e-3)
        params_base['fit_phase__free']   .add('dt',value=0.125*1e-3,min=-3e-3,max=3e-3)

    return model,params_base


def fill_data(aman, coadd_data, fit_result, n_bins, cal_type):
    """
    Fill co-added data and fit result to axis manager.
    
    Args:
        aman: axis manager
        coadd_data: co-added data
        fit_result: fit result
        n_bins: number of bins
        cal_type: 'gain' or 'timeconstant'. type of calibration
    """
    if 'coadd_data' not in aman.stm_ana.keys():
        bins = core.IndexAxis('bins',n_bins)
        aman.stm_ana.wrap('coadd_data',core.AxisManager(aman._axes['dets'],bins))
    
    # Fill co-added data to axis manager
    for filt_key in coadd_data.keys():
        if filt_key not in aman.stm_ana.coadd_data.keys():
            aman.stm_ana.coadd_data.wrap(filt_key,core.AxisManager())

        for freq_key in coadd_data[filt_key].keys():
            if freq_key not in aman.stm_ana.coadd_data[filt_key].keys():
                aman.stm_ana.coadd_data[filt_key].wrap(freq_key, core.AxisManager(aman._axes['dets'],aman.stm_ana.coadd_data._axes['bins']))

            for key in coadd_data[filt_key][freq_key].keys():
                arr = np.array([x if x is not None else np.full(n_bins, np.nan) for x in coadd_data[filt_key][freq_key][key]]) 
                aman.stm_ana.coadd_data[filt_key][freq_key].wrap(f'{key}', arr, [(0,'dets'), (1,'bins')], overwrite=True)

    # Fill fit result to axis manager
    for fit_key in fit_result.keys():
        if fit_key not in aman.stm_ana.keys():
            aman.stm_ana.wrap(fit_key, core.AxisManager(aman._axes['dets']))

        for filt_key in fit_result[fit_key].keys():
            if fit_key == 'fit_coadd':
                if filt_key not in aman.stm_ana[fit_key].keys():
                    aman.stm_ana[fit_key].wrap(filt_key, core.AxisManager(aman._axes['dets']))
                for freq_key in fit_result[fit_key][filt_key].keys():
                    aman.stm_ana[fit_key][filt_key].wrap(freq_key, core.AxisManager(aman._axes['dets'],aman.stm_ana.coadd_data._axes['bins']),overwrite=True)
            else:
                aman.stm_ana[fit_key].wrap(filt_key, core.AxisManager(aman._axes['dets'],core.IndexAxis('ndata_taufit',len(coadd_data[filt_key].keys()))),overwrite=True)


        for filt_key in fit_result[fit_key].keys():
            def fill_field(field, result, weight_axis):
                valid_result = [x for x in result if x is not None]
                for key in valid_result[0].params.keys():
                    arr = np.array([float(x.params[key].value) if x is not None else np.nan for x in result])
                    field.wrap(key, arr, [(0,'dets')], overwrite=True)
                    arr = np.array([x.params[key].stderr if x is not None else np.nan for x in result])
                    arr = np.array([float(x.params[key].stderr) if x is not None and x.params[key].stderr is not None else np.nan for x in result])
                    field.wrap(f'{key}_stderr', arr, [(0,'dets')], overwrite=True)
                    arr = np.array([float(x.chisqr) if x is not None else np.nan for x in result])
                    field.wrap('chisqr', arr, [(0,'dets')], overwrite=True)
                    arr = np.array([float(x.redchi) if x is not None else np.nan for x in result])
                    field.wrap('redchi', arr, [(0,'dets')], overwrite=True)
                    if weight_axis == 'bins':  # Because we don't use weight for time-constant fit for the first step analysis
                        arr = np.array([x.weights if x is not None else np.full(field._axes[weight_axis].count, np.nan) for x in result])
                        arr = arr.astype(float)
                        field.wrap('weights', arr, [(0,'dets'), (1,weight_axis)], overwrite=True)

            if fit_key == 'fit_coadd':
                for freq_key in fit_result[fit_key][filt_key].keys():
                    field = aman.stm_ana.fit_coadd[filt_key][freq_key]
                    result = fit_result[fit_key][filt_key][freq_key]
                    weight_axis = 'bins'

                    fill_field(field, result, weight_axis)
            else:
                field = aman.stm_ana[fit_key][filt_key]
                result = fit_result[fit_key][filt_key]
                weight_axis = 'ndata_taufit'
                fill_field(field, result, weight_axis)


    # Fill result to aman.det_cal
    if 'det_cal' not in aman.keys():
        aman.wrap('det_cal', core.AxisManager(aman._axes['dets']))

    if cal_type == 'gain':
        idx = np.where(aman.stm_ana.positions.vals == 'heater')[0][0]
        heater_temp = aman.stm_ana.temps[idx][0]
        idx = np.where(aman.stm_ana.positions.vals == 'env')[0][0]
        env_temp = aman.stm_ana.temps[idx][0]

        arr = np.array([float(x.best_values['a0']) if x is not None else np.nan for x in fit_result['fit_coadd']['lpf']['f1_gain']])
        arr = abs(arr) * (750/(heater_temp-env_temp))
        aman.det_cal.wrap('stm_gain', arr, [(0,'dets')], overwrite=True)
    if cal_type == 'timeconstant':
        arr = np.array([float(x.best_values['tau']) if x is not None else np.nan for x in fit_result['fit_amp']['lpf']])
        aman.det_cal.wrap('stm_tau', arr, [(0,'dets')], overwrite=True)
        arr = np.array([float(x.best_values['dt']) if x is not None else np.nan for x in fit_result['fit_phase__fix_tau']['lpf']])
        aman.det_cal.wrap('readout_delay', arr, [(0,'dets')], overwrite=True)
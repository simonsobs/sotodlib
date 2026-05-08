import numpy as np
import matplotlib.pyplot as plt
import lmfit
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from pathlib import Path
import ruptures as rpt

from sotodlib import tod_ops, core
from sotodlib.io import hkdb

CHOPPING_FREQS = {'f1':6, 'f2':15,'f3':33,'f4':63,'f5':93,'f6':123,'f7':147}


def get_ufm_list(type):
    # Getting UFM names in LATR
    #
    # type: 'SO' or 'ASO', which means before or after ASO deployment on 2026 Jan, respectively.

    if type == 'SO':
        list_ufm = ['Uv38','Uv39','Uv46',
                    'Uv31','Uv42','Uv47',
                    'Mv21','Mv24','Mv28',
                    'Mv13','Mv20','Mv34',
                    'Mv14','Mv32','Mv49',
                    'Mv25','Mv26','Mv11']
    elif type == 'ASO':
        list_ufm = ['Uv54','Uv58','Uv60',
                    'Uv57','Uv59','Uv62',
                    'Mv29','Mv68','Mv73',
                    'Mv65','Mv67','Mv75',
                    'Mv15','Mv64','Mv70',
                    'Mv63','Mv76','Mv77',
                    'Ln2','Ln3','Ln4']
    else:
        raise ValueError("Invalid type. Please specify either 'SO' or 'ASO'.")

    return list_ufm


def get_ot(ufm):
    # Getting Optics tube name
    # ufm: ufm name

    ufm = ufm.lower()
    
    if ufm in ['uv38','uv39','uv46']:
        ot = 'c1'
    elif ufm in ['mv21','mv24','mv28']:
        ot = 'i1'
    elif ufm in ['uv54','uv58','uv60']:
        ot = 'i2'
    elif ufm in ['mv13','mv20','mv34']:
        ot = 'i3'
    elif ufm in ['mv14','mv32','mv49']:
        ot = 'i4'
    elif ufm in ['uv31','uv42','uv47']:
        ot = 'i5'
    elif ufm in ['mv25','mv26','mv11']:
        ot = 'i6'
    elif ufm in ['uv57','uv59','uv62']:
        ot = 'o1'
    elif ufm in ['mv29','mv68','mv73']:
        ot = 'o2'
    elif ufm in ['mv65','mv67','mv75']:
        ot = 'o3'
    elif ufm in ['mv15','mv64','mv70']:
        ot = 'o4'
    elif ufm in ['mv63','mv76','mv77']:
        ot = 'o5'
    elif ufm in ['ln2','ln3','ln4']:
        ot = 'o6'
    else:
        raise ValueError('No OT')
    
    return ot


def get_downsample_factor_tag(ctx):
    # Getting downsample factor of SMuRF readout
    # ctx: context file

    cursor = ctx.obsdb.conn.execute("SELECT DISTINCT tag FROM tags")
    all_tags = np.array([row[0] for row in cursor.fetchall()])
    mask = np.char.find(all_tags,'downsample') != -1
    
    return np.array(all_tags)[mask].tolist()


def get_downsample_factor(aman,ctx):
    # Getting downsample factor of SMuRF readout for the axis manager data
    # aman: Axis manager of detector data
    # ctx: context file

    downsample_factor_tag = get_downsample_factor_tag(ctx)

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


def get_obs_list(ufm,ctx,timestamp_min,timestamp_max):
    # Getting obs_list for stimulator during specified time range
    # 
    # ufm: ufm name
    # ctx: context file
    # timestamp_min: Minimum timestamp
    # timestamp_max: Maximum timestamp

    ufm = ufm.lower()
    ot = get_ot(ufm)
    downsample_factor_tag = get_downsample_factor_tag(ctx)
    
    obs_list = ctx.obsdb.query(
        f"tube_slot == '{ot}' and stimulator and timestamp >= {timestamp_min} and timestamp <= {timestamp_max}",
        tags=['stimulator', 'gain', 'time_constant', 'gain_and_timeconstant']+downsample_factor_tag
        )
    
    if len(obs_list) == 0:
        raise ValueError("Error: No stimulator data in the specified time range")

    return obs_list


def get_meta(ufm,ctx,obs_id,bool_restrict=True):
    # Getting meta data for specified UFM and obs_id
    #
    # ufm: ufm name
    # ctx: context file
    # obs_id: observation ID
    # bool_restrict: If True, enable custom restriction.
    #                UFM restriction will be applyied regardress this specification.

    ufm = ufm.lower()
    
    #meta = ctx.get_meta(obs_id)
    meta = ctx.get_meta(obs_id, ignore_missing=True)
    meta.restrict("dets", meta.det_info.stream_id == f"ufm_{ufm}")
    
    # remove saturated / latched detectors
    if bool_restrict == True:
        meta.restrict("dets", (meta.det_cal.r_frac>0.2)*(meta.det_cal.r_frac<0.8))

    return meta


def get_hk(hkdb_cfg, aman=None, t_start=None, t_end=None):
    # Getting hk data for one axis manager data
    #
    # hkdb_cfg: HK data base config
    # aman: Axis manager of detector data
    # t_start: Start time of hk data. If None, use aman's start time.
    # t_end: End time of hk data. If None, use aman's end time.

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


def func_sines(t,a0,a1,a2,a3,a4,a5,a6,t0,t1,t2,t3,t4,t5,t6):
    # Define fitting function
    #
    # t: time or timing_frac of stimulator signal.
   
    y = (a0*np.sin(1*(t-t0)*2*np.pi)
         + a1*np.sin(2*(t-t1)*2*np.pi)
         + a2*np.sin(3*(t-t2)*2*np.pi)
         + a3*np.sin(4*(t-t3)*2*np.pi)
         + a4*np.sin(5*(t-t4)*2*np.pi)
         + a5*np.sin(6*(t-t5)*2*np.pi)
         + a6*np.sin(7*(t-t6)*2*np.pi))
    
    return y


def func_response_amplitude(f, tau, a):
    # Detector response function of amplitude
    #
    # f: Chopping frequency
    # tau: time constant of a detector in [s]
    # a: Amplitude of sin function for '0' frequency signal

    y = a /np.sqrt(1+(2*np.pi*f*tau)**2)
    return y


def func_response_phase(f, tau, theta_geo):
    # Detector response function of phase
    #
    # f: Chopping frequency [Hz]
    # tau: time constant of a detector in [s]
    # theta_geo: Offset of phase delay [deg]
    # theta: Phase delay of stimulator signal [deg]

    theta = np.arctan(-2*np.pi*f*tau)*(180/np.pi) + theta_geo
    return theta


def func_response_phase_with_dt(f, tau, theta_geo, dt):
    # Detector response function of phase
    #
    # f: Chopping frequency [Hz]
    # tau: time constant of a detector in [s]
    # theta: Phase delay of stimulator signal [deg]
    # theta_geo: Offset of phase delay due to hardware effect, geo = geometry [deg]
    # theta_dt: Offset of phase delay due to readout issue [deg], theta_dt*(pi/180) =  -delta_t*2pi*f
    # dt: Time difference due to wrong time stamps

    theta_dt = -dt*2*np.pi*f *(180/np.pi)
    theta = np.arctan(-2*np.pi*f*tau)*(180/np.pi) + theta_geo + theta_dt
    return theta
    
    
def s_open(theta):
    #theta: 0 to pi/8
    r1 = 70.5e-3 #Radius for chopper. chopper center to optical pipe center
    r2 = 43e-3/2 #Radius of optical pipe
    theta_1 = np.arcsin(r2/r1) #Angle at the border
    
    if 0<= theta and theta < theta_1:
        phi1 = np.arcsin(-1/r2 * np.sqrt(r2**2-r1**2*np.sin(theta)**2))
        phi2 = np.arcsin( 1/r2 * np.sqrt(r2**2-r1**2*np.sin(theta)**2))
        return np.pi*r2**2 + 2*r1 * np.sin(theta) * np.sqrt(r2**2 - r1**2 * np.sin(theta)**2) - r2**2/2 * (1/2*np.sin(2*phi2)+phi2) + r2**2/2 * (1/2*np.sin(2*phi1)+phi1)
    elif theta < np.pi/8:
        return np.pi*r2**2
    else:
        return 'error'


def stm_signal_raytrace(temp_heater,temp_blackbody,theta):
    #theta: 0 to pi/4, signal of half cycle
    r2 = 43e-3/2 #Radius of optical pipe
    
    if theta < np.pi/8:
        theta_tmp = theta
    else:
        theta_tmp = np.pi/8 - (theta-np.pi/8)

    return s_open(theta_tmp)/(np.pi*r2**2) * (temp_heater - temp_blackbody) + temp_blackbody


def func_opening(frac_timing,a,c,timing0):
    #frac_timing: list,0 to 1
    #timing0: 0 to 1
    #a: amplitude
    #c: offset

    theta_list = (frac_timing-timing0) * np.pi/2#1 cycle = pi/2
    
    y = [] 

    for theta in theta_list:
        if theta < 0:
            theta = theta + np.pi/2

        if theta < np.pi/4:
            y.append(stm_signal_raytrace(1,0,theta)*a + c)
        else:
            theta = theta - np.pi/4
            y.append(stm_signal_raytrace(0,1,theta)*a + c)
    
    return np.array(y)


def calc_gain(aman,hkdata,idxs=None,bool_plot=False,bool_save=False,bool_preprocess=True,output_dir=None):
    # aman: axis manager of tod data, including timestamps and raw signal
    # hkdata: hkdata for aman
    # idxs: list of detector index for calculation. If None, all detectors are calculated.
    # bool_plot: If true, makes plot.
    # bool_save: If true, save plot

    mask = np.full(aman.dets.count, False)
    mask[idxs] = True

    if bool_preprocess:
        get_encoder_timing(aman, hkdata)# Get timing against encoder t0

        get_chopping_status(aman)

        get_timing_cut(aman)

        get_signal_temp(aman,hkdata)

    if bool_plot==True:
        plot_hkdata(aman,hkdata,cal_type='gain')

    chopping_freqs = {}
    if round(aman.stm_ana.chopping_freqs[0]) != CHOPPING_FREQS['f1']: 
        chopping_freqs['f1_gain'] = round(aman.stm_ana.chopping_freqs[0])
    else:
        chopping_freqs['f1_gain'] = CHOPPING_FREQS['f1']


    filtering_params = filtering(aman,chopping_freqs,'gain')

    coadd_data, fit_result = get_dicts('gain')
    n_bins = 40

    model, params_base = get_fit_params(cal_type='gain')

    for i_det,m in enumerate(mask):
        if not m:
            fill_none(coadd_data, fit_result) 
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            fill_none(coadd_data, fit_result) 
            continue
            
        # Make and get co-added data
        idx = np.where(aman.stm_ana.freqs.vals == 'f1_gain')[0][0]
        get_coadd_data(aman,coadd_data,i_det,n_bins,t_min=aman.stm_ana.t_cuts[idx][0],t_max=aman.stm_ana.t_cuts[idx][1],freq_key='f1_gain')
        
        # Fitting
        for filt_key in fit_result['fit_coadd'].keys():

            params = params_base.copy()

            x = coadd_data[filt_key]['f1_gain']['x'][-1]
            y = coadd_data[filt_key]['f1_gain']['y'][-1]
            yerr = coadd_data[filt_key]['f1_gain']['yerr'][-1]

            if y is None:
                fit_result['fit_coadd'][filt_key]['f1_gain'].append(None)
                continue

            result = model.fit(y, params, t=x, weights=1/np.array(yerr)) 

            fit_result['fit_coadd'][filt_key]['f1_gain'].append(result)

        if bool_plot==False and bool_save==False:
            pass
        else: 
            plot(aman,i_det,coadd_data,fit_result,filtering_params,cal_type='gain')
            
            if bool_save == True:
                obs_id = aman.obs_info.obs_id
                if output_dir is not None:
                    output_dir_ = Path(f'{output_dir}/{ufm}_{obs_id}')
                    output_dir_.mkdir(parents=True, exist_ok=True)
                    plt.savefig(f'{output_dir_}/Gain_det{i_det:04d}.png')
            if bool_plot==False:
                plt.close(fig)
            
    fill_data(aman,coadd_data,fit_result,n_bins,cal_type='gain')    

    return fit_result


def calc_timeconstant(aman,hkdata,idxs=None,bool_plot=False,bool_save=False,bool_preprocess=True,output_dir=None):
    # aman: axis manager of tod data, including timestamps and raw signal
    # hkdata: hkdata for aman
    # idxs: list of detector index for calculation. If None, all detectors are calculated.
    # bool_plot: if true, makes plot.
    # bool_save: If true, save plot


    mask = np.full(aman.dets.count, False)
    mask[idxs] = True

    if bool_preprocess:
        get_encoder_timing(aman, hkdata)# Get timing against encoder t0

        get_chopping_status(aman)

        get_timing_cut(aman)

        get_signal_temp(aman,hkdata)

    if bool_plot==True:
        plot_hkdata(aman,hkdata,cal_type='timeconstant')
    
    chopping_freqs = {}
    for i,(key,f) in enumerate(CHOPPING_FREQS.items()):
        if round(aman.stm_ana.chopping_freqs[0]) != f: 
            chopping_freqs[key] = round(aman.stm_ana.chopping_freqs[i])
        else:
            chopping_freqs[key] = f

    filtering_params = filtering(aman,chopping_freqs,'timeconstant')

    coadd_data, fit_result = get_dicts('timeconstant')
    n_bins = 40

    models, params_bases = get_fit_params(cal_type='timeconstant')
    
    
    for i_det,m in enumerate(mask):
        if not m:
            fill_none(coadd_data, fit_result) 
            continue

        # Strange data check
        if not np.isfinite(aman.signal[i_det]).all():
            fill_none(coadd_data, fit_result) 
            continue
            
        # Make and get co-added data
        for f_key in chopping_freqs.keys():
            idx = np.where(aman.stm_ana.freqs.vals == f_key)[0][0]
            get_coadd_data(aman,coadd_data,i_det,n_bins,t_min=aman.stm_ana.t_cuts[idx][0],t_max=aman.stm_ana.t_cuts[idx][1],freq_key=f_key)
    

        # Fitting
        for filt_key in fit_result['fit_coadd'].keys():
            for f_key in chopping_freqs.keys():
                params = params_bases['fit_coadd'].copy()
                
                x = coadd_data[filt_key][f_key]['x'][-1]
                y = coadd_data[filt_key][f_key]['y'][-1]
                yerr = coadd_data[filt_key][f_key]['yerr'][-1]

                if y is None:
                    fit_result['fit_coadd'][filt_key][f_key].append(None)
                    continue

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

            f = [chopping_freqs[f_key] for f_key in chopping_freqs.keys()]
            for fit_key in ['fit_amp', 'fit_phase__no_dt', 'fit_phase__fix_tau', 'fit_phase__free']:
                params = params_bases[fit_key].copy()
                if fit_key == 'fit_amp':
                    weights = [fit_result['fit_coadd'][filt_key][f_key][-1].params['a0'].stderr if fit_result['fit_coadd'][filt_key][f_key][-1] is not None else np.nan for f_key in fit_result['fit_coadd'][filt_key].keys()]
                    weights = np.array(weights)
                    #result = models[fit_key].fit(a0s,params,f=f,weights=weights, method='least_squares')
                    result = models[fit_key].fit(a0s,params,f=f, method='least_squares')# No weight for first step analysis
                else:
                    weights = [fit_result['fit_coadd'][filt_key][f_key][-1].params['t0'].stderr if fit_result['fit_coadd'][filt_key][f_key][-1] is not None else np.nan for f_key in fit_result['fit_coadd'][filt_key].keys()]
                    weights = np.array(weights) *360
                    if fit_key == 'fit_phase__fix_tau':
                        params['tau'].set(value=fit_result['fit_amp'][filt_key][-1].best_values['tau'],vary=False)
                    #result = models[fit_key].fit(-np.array(t0s)*360,params,f=f, weights=weights, method='least_squares')
                    result = models[fit_key].fit(-np.array(t0s)*360,params,f=f, method='least_squares')# No weight for first step analysis
                fit_result[fit_key][filt_key].append(result)
        
        
        if bool_plot==False and bool_save==False:
            pass
        else:
            plot(aman,i_det,coadd_data,fit_result,filtering_params,cal_type='timeconstant')

            obs_id = aman['obs_info']['obs_id']
            if bool_save==True:
                output_dir_ = Path(f'{output_dir}/{ufm}_{obs_id}')
                output_dir_.mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{output_dir_}/Tau_det{i_det:04d}.png')
            if bool_plot==False:
                plt.close(fig)

    
    fill_data(aman,coadd_data,fit_result,n_bins,cal_type='timeconstant')    


def get_encoder_timing(aman,hkdata):
    # Get timing against encoder t0 
    
    # Get encoder t0 list
    state = np.array(hkdata.data['stimulator-enc.stim_enc.state'][1])
    t_enc = np.array(hkdata.data['stimulator-enc.stim_enc.timestamps_tai'][1])[state == 0]-37
    t_hk = np.array(hkdata.data['stimulator-enc.stim_enc.timestamps_tai'][0])[state == 0]

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
    
    # Add timing against encoder to axis manager
    aman.wrap('frac_timing', np.array(frac_timing), [(0,'samps')], overwrite=True)

    if not 'stm_samps' in aman._axes.keys():
        stm_samps = core.IndexAxis('stm_samps',t_enc.size)
        stm_ana = core.AxisManager(stm_samps)
        aman.wrap('stm_ana',stm_ana,overwrite=True)
    aman.stm_ana.wrap('t_enc', t_enc, [(0,'stm_samps')], overwrite=True)
    aman.stm_ana.wrap('t_hk', t_hk, [(0,'stm_samps')], overwrite=True)

def get_chopping_status(aman):
    # Get chopping frequency and timing
    # aman: axis manager with aman.stm_ana field

    # Get chopping frequency vs time
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


def get_timing_cut(aman,dt_gain=60,dt_timeconstant=10,dt_wait=9):
    # Get timing range for analysis

    if len(aman.stm_ana.t_chopping_change)==2:
        cal_type = 'gain'
    elif len(aman.stm_ana.t_chopping_change)>2:
        cal_type = 'gain_timeconstant'
    else:
        ValueError('Chopping information is incorrect.')


    t_cuts = []

    t_start = 2+aman.stm_ana.t_chopping_change[0]# 2 seconds to avoid ringing by high-pass filter
    t_end = min(t_start+dt_gain, aman.stm_ana.t_chopping_change[1]) 
    t_cuts.append((t_start,t_end))

    if cal_type == 'gain_timeconstant':
        for i in range(len(aman.stm_ana.t_chopping_change)-1):
            t_start = aman.stm_ana.t_chopping_change[i] + dt_wait
            if i==0:
                t_start = 2+aman.stm_ana.t_chopping_change[i]# 2 seconds to avoid ringing by high-pass filter
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


def get_signal_temp(aman,hkdata):
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
            y_mean = y[(t_min <= x) & (x < t_max)].mean()
            temps[position_key].append(y_mean)

        temps[position_key] = np.array(temps[position_key])


    temps['env'] = (temps['chopper_rear']+temps['chopper_front']+temps['air'])/3
    position_keys.append('env')

    label_positions = core.LabelAxis('positions',position_keys)
    aman.stm_ana.add_axis(label_positions)
    arr = [x for x in temps.values()]
    arr = np.array(arr)

    aman.stm_ana.wrap('temps',arr,[(0,'positions'),(1,'freqs')],overwrite=True)


def filtering__with_delay_filter(aman,filter,signal_name_pre,signal_name_new):
    # filtering signal data using the filter which has timing delay
    # aman: axis manager
    # filter: filter of tod_ops.filters
    # signal_name_pre: signal name which will be filtered 
    # signal_name_new: New name for filtered signal

    signal_new = tod_ops.fourier_filter(aman, filter, signal_name=signal_name_pre)
    signal_new = np.fliplr(signal_new)
    aman.wrap(signal_name_new, signal_new, [(0,'dets'),(1,'samps')], overwrite=True)
    
    signal_new = tod_ops.fourier_filter(aman, filter, signal_name=signal_name_new)
    signal_new = np.fliplr(signal_new)
    aman.wrap(signal_name_new, signal_new, [(0,'dets'),(1,'samps')], overwrite=True)


def filtering__without_delay_filter(aman,filter,signal_name_pre,signal_name_new):
    # filtering signal data using the filter which doesn't have timing delay
    # aman: axis manager
    # filter: filter of tod_ops.filters
    # signal_name_pre: signal name which will be filtered 
    # signal_name_new: New name for filtered signal

    signal_new = tod_ops.fourier_filter(aman, filter, signal_name=signal_name_pre)
    aman.wrap(signal_name_new, signal_new, [(0,'dets'),(1,'samps')], overwrite=True)


def filtering(aman, chopping_freqs, cal_type):
    # filtering signal data
    # aman: axis manager
    # cal_type: 'gain' or 'timeconstant'. type of calibration
    # chopping_freqs: dictionary of chopping frequencies

    # Invert IIR filter 
    iirc_filter = tod_ops.filters.iir_filter(aman, invert=True)
    signal_new = tod_ops.fourier_filter(aman, iirc_filter, signal_name='signal')
    aman.wrap(f'signal_iirc', signal_new, [(0,'dets'),(1,'samps')], overwrite=True)

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


def get_dicts(cal_type):
    # Get dictionaries for co-added data and fit result
    # cal_type: 'gain' or 'timeconstant'. type of calibration

    filt_keys = ['iirc','hpf','lpf']

    coadd_data = {}
    fit_result = {}

    if cal_type == 'gain':
        freq_keys = ['f1_gain']
        fit_keys = ['fit_coadd']
    elif cal_type == 'timeconstant':
        freq_keys = CHOPPING_FREQS.keys()
        fit_keys = ['fit_coadd','fit_amp','fit_phase__fix_tau','fit_phase__free','fit_phase__no_dt']
    else:
        ValueError(f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'.")
    
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


def fill_none(coadd_data, fit_result):
    # Fill None for co-added data and fit result when there is no valid data for calculation
    # coadd_data: Dictionary for co-added data 
    # fit_result: List for fit result

    for fit_key in fit_result.keys():
        for filt_key in fit_result[fit_key].keys():
            if type(fit_result[fit_key][filt_key]) == list:
                fit_result[fit_key][filt_key].append(None)
            else:
                for freq_key in fit_result[fit_key][filt_key].keys():
                    fit_result[fit_key][filt_key][freq_key].append(None)

    for filt_key in coadd_data.keys():
        for freq_key in coadd_data[filt_key].keys():
            for key in coadd_data[filt_key][freq_key].keys():
                coadd_data[filt_key][freq_key][key].append(None)


def get_coadd_data(aman,coadd_data,i_det,n_bins,t_min,t_max,freq_key):
    # Making co-added data
    # aman: axis manager of tod data, including timestamps and tod signal
    # coadd_data: Dictionary for co-added data 
    # n_bins: # of bins for co-added data
    # t_min: Minimum time for the data analysis
    # t_max: Maximum time for the data analysis


    t0 = aman.timestamps[0]
    bins = np.linspace(0,1-1/n_bins,n_bins)  

    for filt_key in coadd_data.keys():
        data = [[] for _ in range(n_bins)]

        for i_bin in range(n_bins):
            cut1 = (i_bin/n_bins <= aman.frac_timing) & (aman.frac_timing < (i_bin+1)/n_bins)
            cut2 = (t_min <= aman.timestamps-t0) & (aman.timestamps-t0 < t_max)
            cut = cut1 & cut2

            if filt_key == 'hpf' or filt_key == 'iirc':
                key = f'signal_{filt_key}'
            elif filt_key == 'lpf':
                if freq_key == 'f1_gain':
                    key = 'signal_lpf'
                else:
                    key = f'signal_lpf_{freq_key}'

            data[int(i_bin)] = aman[key][i_det][cut]

        x = bins + 1/n_bins/2
        y = np.array([np.nanmean(d) for d in data])
        yerr = np.array([np.array(d).std(ddof=1)/np.sqrt(len(d)) for d in data])

        mask = np.isfinite(y)
        x = x[mask]
        y = y[mask]
        yerr = yerr[mask]

        if y.size==0:
            x,y,yerr = (None, None, None)

        coadd_data[filt_key][freq_key]['x'].append(x)
        coadd_data[filt_key][freq_key]['y'].append(y)
        coadd_data[filt_key][freq_key]['yerr'].append(yerr)


def get_fit_params(cal_type):
    # Get fit parameters for fitting
    # cal_type: 'gain' or 'timeconstant'. type of calibration
    

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
        model['fit_phase__no_dt']   = lmfit.Model(func_response_phase_with_dt, independent_vars=['f'])

        params_base = {}
        params_base['fit_coadd'] = lmfit.Parameters()
        params_base['fit_amp'] = lmfit.Parameters()
        params_base['fit_phase__no_dt'] = lmfit.Parameters()
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
        params_base['fit_phase__no_dt']  .add('theta_geo',value=0,min=-90,max=90)
        params_base['fit_phase__free']   .add('theta_geo',value=0,min=-90,max=90)
        params_base['fit_phase__fix_tau'].add('tau')
        params_base['fit_phase__no_dt']  .add('tau',value=1e-3,min=0,max=0.1)
        params_base['fit_phase__free']   .add('tau',value=1e-3,min=0,max=0.1)
        params_base['fit_phase__fix_tau'].add('dt',value=0.125*1e-3,min=-3e-3,max=3e-3)
        params_base['fit_phase__no_dt']  .add('dt',value=0,vary=False)
        params_base['fit_phase__free']   .add('dt',value=0.125*1e-3,min=-3e-3,max=3e-3)

    return model,params_base


def plot_hkdata(aman,hkdata,cal_type):

    fig, axes = plt.subplot_mosaic([['A','A'],['B','C'],['D','E']],figsize=(10,8))

    t0 = aman.timestamps[0]
    
    # Chopping freq with t_cuts
    y = 1/(aman.stm_ana.t_enc[1:] - aman.stm_ana.t_enc[:-1])
    axes['A'].plot(aman.stm_ana.t_enc[:-1]-t0,y,label='Chopping_freq')
    axes['A'].set_xlabel('Time [s]')
    axes['A'].set_ylabel('Chopping frequency [Hz]')
    axes['A'].vlines(aman.timestamps[0] -t0,ymin=min(y),ymax=max(y),linestyle='--', color='black', alpha=0.5,label='TOD start')
    axes['A'].vlines(aman.timestamps[-1]-t0,ymin=min(y),ymax=max(y),linestyle='-',  color='black', alpha=0.5,label='TOD end')


    # Environmental temperature
    data_temp = {}
    for key,key_hk in [('heater',       'stimulator-thermo.temperatures.Channel_0_T'),
                       ('chopper_rear', 'stimulator-thermo.temperatures.Channel_4_T'),
                       ('chopper_front','stimulator-thermo.temperatures.Channel_6_T'),
                       ('air',          'stimulator-thermo.temperatures.Channel_5_T')]:
        data_temp[key] = {}
        data_temp[key]['t']    = hkdata.data[key_hk][0]-t0
        data_temp[key]['temp'] = hkdata.data[key_hk][1]+273.15
    
    for key in data_temp.keys():
        if key != 'heater':
            axes['B'].plot(data_temp[key]['t'],data_temp[key]['temp'],'-',label=key)

    idx = np.where(aman.stm_ana.positions.vals == 'env')[0][0]
    env_temps = aman.stm_ana.temps[idx]

    axes['B'].set_ylim(env_temps[0] - 5, env_temps[0] + 5) 
    axes['B'].set_ylabel('Temperature [K]')
    axes['B'].set_xlabel('Time [s]')

    for key_freq, env_temp, (t_min, t_max) in zip(aman.stm_ana.freqs.vals,env_temps,aman.stm_ana.t_cuts):
        if cal_type=='gain':
            if key_freq=='f1_gain':
                axes['B'].hlines(env_temp,t_min,t_max,color='red',label='environment')
        else:
            if key_freq!='f1_gain':
                if key_freq == 'f1':
                    axes['B'].hlines(env_temp,t_min,t_max,color='red',label='environment')
                else:
                    axes['B'].hlines(env_temp,t_min,t_max,color='red')


    # Heater temperature
    axes['D'].plot(data_temp['heater']['t'],data_temp['heater']['temp'],'-',label='heater')
    idx = np.where(aman.stm_ana.positions.vals == 'heater')[0][0]
    heater_temp = aman.stm_ana.temps[idx][0]
    axes['D'].set_ylim(heater_temp - 5, heater_temp + 5) 
    axes['D'].set_ylabel('Temperature [K]')
    axes['D'].set_xlabel('Time [s]')


    # Encoder timing and stream timing
    axes['C'].plot(aman.stm_ana.t_enc-t0,aman.stm_ana.t_enc-aman.stm_ana.t_hk,'.')
    axes['C'].set_title(f'PTP_time - hk_time')
    axes['C'].set_xlabel('t_enc - t0_stream [s]')
    axes['C'].set_ylabel('t_enc - t_hk [s] (should be -0.1<t<0)')


    # Plot timing cut area
    for key_ax in ['A','B','D']:
        for key_freq, (t_min, t_max) in zip(aman.stm_ana.freqs.vals,aman.stm_ana.t_cuts):
            if cal_type=='gain':
                if key_freq == 'f1_gain':
                    axes[key_ax].axvspan(t_min, t_max, alpha=0.3, label='used data')
            else:
                if key_freq != 'f1_gain':
                    if key_freq == 'f1':
                        axes[key_ax].axvspan(t_min, t_max, alpha=0.3, label='used data')
                    else:
                        axes[key_ax].axvspan(t_min, t_max, alpha=0.3)


    # Misc
    for key_ax in ['A','B','C','D']:
        axes[key_ax].grid()
    axes['A'].legend()
    axes['B'].legend(loc='upper left',bbox_to_anchor=(1,1))
    axes['D'].legend(loc='upper left',bbox_to_anchor=(1,1))

    plt.tight_layout()


def plot(aman,i_det,coadd_data,fit_result,filtering_params,cal_type):
    # Make plot for one detector
    # aman: axis manager
    # data: co-added data for one detector
    # result: fit result for one detector
    # cal_type: 'gain' or 'timeconstant'. type of calibration

    t0 = aman.timestamps[0]
    ufm = aman.det_info.stream_id[i_det][4:]
    ufm = ufm[0].upper() + ufm[1:]


    if cal_type == 'gain':
        fig, axes = plt.subplots(3,2,figsize=(10,8))
        fig.suptitle(f'Stimulator data, {ufm}, i_det: {i_det}, det_id: {aman.det_info.det_id[i_det]}')

        i_y=0;i_x=0
        y = aman.signal[i_det]-np.mean(aman.signal[i_det])
        axes[i_y,i_x].plot(aman.timestamps-t0,y,label='Raw_data - mean')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data')
        axes[i_y,i_x].set_ylim(min(y)-(max(y)-min(y))*0.1,max(y)+(max(y)-min(y))*0.1)
        axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}')
        axes[i_y,i_x].set_xlabel('time [s]')
        axes[i_y,i_x].set_ylabel('TOD [pW]')
        axes[i_y,i_x].legend()

    
        i_y=1;i_x=0
        x_min = 30
        x_max = 30.5
        i_x_min=int(aman.obs_info.sampling_rate*x_min)
        i_x_max=int(aman.obs_info.sampling_rate*x_max)
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det]- np.mean(aman.signal[i_det][i_x_min:i_x_max]),label='Raw data - mean')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data',color='C1')
        axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}')
        axes[i_y,i_x].set_xlabel('time [s]')
        axes[i_y,i_x].set_ylabel('TOD [pW]')
        axes[i_y,i_x].legend()
        axes[i_y,i_x].set_ylim(min(aman.signal_hpf[i_det][100:-100]),max(aman.signal_hpf[i_det][100:-100]))
        if ufm[0] == 'M':
            axes[i_y,i_x].set_ylim(-0.005,0.005)
        elif ufm[0] == 'U':
            axes[i_y,i_x].set_ylim(-0.02,0.02)
        axes[i_y,i_x].set_xlim(x_min,x_max)
        i_y=2;i_x=0
        f = np.arange(0,10,0.01)
        axes[i_y,i_x].set_title(f'High Pass Filter')
        axes[i_y,i_x].set_xlabel('Frequency [Hz]')
        axes[i_y,i_x].set_ylabel('HPF')
   
        i_y=0;i_x=1
        x = coadd_data['iirc']['f1_gain']['x'][-1]
        y = coadd_data['iirc']['f1_gain']['y'][-1]
        yerr = coadd_data['iirc']['f1_gain']['yerr'][-1]
        axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=5)
        axes[i_y,i_x].set_title('Co-added signal: Raw data')
        axes[i_y,i_x].set_xlabel('Timing (1 cycle)')
        axes[i_y,i_x].set_ylabel('TOD ave [pW]')
    
        i_y=1;i_x=1
        x = coadd_data['hpf']['f1_gain']['x'][-1]
        y = coadd_data['hpf']['f1_gain']['y'][-1]
        yerr = coadd_data['hpf']['f1_gain']['yerr'][-1]
        axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=5, color='C1',zorder=0,label='HPFed')
        axes[i_y,i_x].set_title(f'Co-added signal: Filtered data, {ufm}')
        
        x = coadd_data['lpf']['f1_gain']['x'][-1]
        y = coadd_data['lpf']['f1_gain']['y'][-1]
        yerr = coadd_data['lpf']['f1_gain']['yerr'][-1]
        axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=5, color='C2',zorder=0,label='(HPF+LPF)ed')
        axes[i_y,i_x].set_xlabel('Timing (1 cycle)')
        axes[i_y,i_x].set_ylabel('TOD ave [pW]')
        axes[i_y,i_x].plot(x, fit_result['fit_coadd']['hpf']['f1_gain'][-1].best_fit, '-', color='red',zorder=1)
        axes[i_y,i_x].plot(x, fit_result['fit_coadd']['lpf']['f1_gain'][-1].best_fit, '-', color='green',zorder=1)
        y = fit_result['fit_coadd']['hpf']['f1_gain'][-1].best_values['a0']*np.sin((x-fit_result['fit_coadd']['hpf']['f1_gain'][-1].best_values['t0'])*2*np.pi)
        axes[i_y,i_x].plot(x, y, linestyle=(0,(2,8)), color='red',zorder=1,label=fr'sin$\theta$ for HPF fit')
        axes[i_y,i_x].legend()
   
    
        i_y=2;i_x=1
        axes[i_y,i_x].plot(aman.stm_ana.t_enc[:-1]-t0,1/(aman.stm_ana.t_enc[1:] - aman.stm_ana.t_enc[:-1]),label='y: chopping freq, t: encoder')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det],label='TOD')
        for key_freq, (t_min, t_max) in zip(aman.stm_ana.freqs.vals,aman.stm_ana.t_cuts):
            if key_freq == 'f1_gain':
                axes[i_y,i_x].axvspan(t_min, t_max, alpha=0.3, label='used data')
        axes[i_y,i_x].set_title('Overplot')
        axes[i_y,i_x].set_ylabel('Chopping frequency[Hz]')
        axes[i_y,i_x].set_xlabel('TOD time [s]')
        axes[i_y,i_x].legend()

        i_y=2;i_x=0
        hpf = tod_ops.filters.high_pass_sine2(filtering_params['hpf_cutoff'])
        filter_cutoff = filtering_params['lpf_cutoff_factor']*filtering_params['chopping_freqs']['f1_gain']
        lpf = tod_ops.filters.low_pass_sine2(filter_cutoff, filter_cutoff*filtering_params['lpf_width_fraction'])

        x = np.arange(0,filter_cutoff*1.2,0.1)
        y = np.full(x.shape[0],1)
        axes[i_y,i_x].plot(x, hpf(x,y),label='HPF', color='C1')
        axes[i_y,i_x].plot(x, lpf(x,y),label='LPF', color='C2')
        axes[i_y,i_x].legend()


    
    elif cal_type == 'timeconstant':
        # Plot basic data 
        fig, axes = plt.subplots(9,2,figsize=(10,18))
        fig.suptitle(f'Stimulator data, i_det: {i_det}, det_id: {aman.det_info.det_id[i_det]}')

        i_y=0;i_x=0
        y = aman.signal[i_det]-np.mean(aman.signal[i_det])
        axes[i_y,i_x].plot(aman.timestamps-t0,y,label='Raw_data - mean')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data')
        axes[i_y,i_x].set_ylim(min(y)-(max(y)-min(y))*0.1,max(y)+(max(y)-min(y))*0.1)
        axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}')
        axes[i_y,i_x].set_xlabel('time [s]')
        axes[i_y,i_x].set_ylabel('TOD [pW]')
        axes[i_y,i_x].legend()

        i_y=0;i_x=1
        axes[i_y,i_x].plot(aman.stm_ana.t_enc[:-1]-t0,1/(aman.stm_ana.t_enc[1:] - aman.stm_ana.t_enc[:-1]),label='y: chopping freq, t: encoder')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det],label='TOD')
        for key_freq, (t_min, t_max) in zip(aman.stm_ana.freqs.vals,aman.stm_ana.t_cuts):
            if key_freq != 'f1_gain':
                if key_freq == 'f1':
                    axes[i_y,i_x].axvspan(t_min, t_max, alpha=0.3, label='used data')
                else:
                    axes[i_y,i_x].axvspan(t_min, t_max, alpha=0.3)
        axes[i_y,i_x].set_title('Overplot')
        axes[i_y,i_x].set_ylabel('Chopping frequency[Hz]')
        axes[i_y,i_x].set_xlabel('TOD time [s]')
        axes[i_y,i_x].legend()

        i_y=1; i_x=0
        f = fit_result['fit_amp']['lpf'][-1].userkws['f']
        #axes[i_y,i_x].errorbar(f,fit_result['fit_amp']['lpf'][-1].data,fit_result['fit_amp']['lpf'][-1].weights,fmt='o')
        axes[i_y,i_x].plot(f,fit_result['fit_amp']['lpf'][-1].data,'o')# No errorbar for first step analysis
        axes[i_y,i_x].plot(f,fit_result['fit_amp']['lpf'][-1].best_fit, '-', color='red',zorder=3,label=fr'$\tau$= {fit_result['fit_amp']['lpf'][-1].best_values['tau']*1e3:.2f}ms')
        axes[i_y,i_x].set_xlabel('Chopping freq [Hz]')
        axes[i_y,i_x].set_ylabel('sin_theta amplitude [pW]')
        axes[i_y,i_x].set_title('Amplitude fit')
        axes[i_y,i_x].legend()
        
        i_y=1; i_x=1
        result = fit_result['fit_phase__free']['lpf'][-1]
        f = result.userkws['f']
        #axes[i_y,i_x].errorbar(f,result.data,result.weights,fmt='o')
        axes[i_y,i_x].plot(f,result.data,'o')# No errorbar for first step analysis

        result = fit_result['fit_phase__no_dt']['lpf'][-1]
        axes[i_y,i_x].plot(f,result.best_fit,'-', color='red',  zorder=3,label=fr'$\tau$={result.best_values['tau']*1e3:.2f}ms, $\theta_\text{{geo}}$={result.best_values['theta_geo']:.0f}deg')
        result = fit_result['fit_phase__free']['lpf'][-1]
        axes[i_y,i_x].plot(f,result.best_fit,'-', color='blue', zorder=3,label=fr'$\tau$={result.best_values['tau']*1e3:.2f}ms, $\theta_\text{{geo}}$={result.best_values['theta_geo']:.0f}deg, $\Delta t$={result.best_values['dt']*1e3:.2f}ms')
        result = fit_result['fit_phase__fix_tau']['lpf'][-1]
        axes[i_y,i_x].plot(f,result.best_fit,'-', color='green',zorder=3,label=fr'$\tau$={result.best_values['tau']*1e3:.2f}ms(fix), $\theta_\text{{geo}}$={result.best_values['theta_geo']:.0f}deg , $\Delta t$={result.best_values['dt']*1e3:.2f}ms')
        
        axes[i_y,i_x].set_xlabel('Chopping freq [Hz]')
        axes[i_y,i_x].set_ylabel('Phase delay [deg]')
        axes[i_y,i_x].set_title('Phase fit')
        axes[i_y,i_x].legend()

        i_y = 1
        for f_key in filtering_params['chopping_freqs'].keys():
            i_y += 1
        
            i_x = 0
            idx = np.where(aman.stm_ana.freqs.vals == f_key)[0][0]
            x_min = aman.stm_ana.t_cuts[idx][0]
            x_max = x_min+0.2 
            i_x_min=int(aman.obs_info.sampling_rate*x_min)
            i_x_max=int(aman.obs_info.sampling_rate*x_max)
            axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det]- np.mean(aman.signal[i_det][i_x_min:i_x_max]),label='Raw data - mean')
            axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data',color='C1')
            axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}, f={filtering_params["chopping_freqs"][f_key]}Hz')
            axes[i_y,i_x].set_xlabel('time [s]')
            axes[i_y,i_x].set_ylabel('TOD [pW]')
            axes[i_y,i_x].legend()
            axes[i_y,i_x].set_ylim(min(aman.signal_hpf[i_det][100:-100]),max(aman.signal_hpf[i_det][100:-100]))
            axes[i_y,i_x].set_xlim(x_min,x_max)
            if ufm[0] == 'M':
                axes[i_y,i_x].set_ylim(-0.005,0.005)
            elif ufm[0] == 'U':
                axes[i_y,i_x].set_ylim(-0.02,0.02)
            elif ufm[0] == 'L':
                axes[i_y,i_x].set_ylim(-0.005,0.005)


            i_x=1
            x = coadd_data['iirc'][f_key]['x'][-1]
            y = coadd_data['iirc'][f_key]['y'][-1]
            yerr = coadd_data['iirc'][f_key]['yerr'][-1]
            axes[i_y,i_x].errorbar(x,y-np.mean(y),yerr, fmt='o', capsize=5, label='IIRCed data - mean')
            axes[i_y,i_x].set_title(f'Co-added signal, f={filtering_params["chopping_freqs"][f_key]}Hz')
            axes[i_y,i_x].set_xlabel('Timing (1 cycle)')
            axes[i_y,i_x].set_ylabel('TOD [pW]')
            
            x = coadd_data['hpf'][f_key]['x'][-1]
            y = coadd_data['hpf'][f_key]['y'][-1]
            yerr = coadd_data['hpf'][f_key]['yerr'][-1]
            axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=3, color='C1', label='(IIRC+HPF)ed data')
            
            x = coadd_data['lpf'][f_key]['x'][-1]
            y = coadd_data['lpf'][f_key]['y'][-1]
            yerr = coadd_data['lpf'][f_key]['yerr'][-1]
            axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=3, color='C2', label='(IIRC+HPF+LPF)ed data')
            
            axes[i_y,i_x].plot(x, fit_result['fit_coadd']['hpf'][f_key][-1].best_fit, '-', color='red',zorder=5)
            axes[i_y,i_x].plot(x, fit_result['fit_coadd']['lpf'][f_key][-1].best_fit, '-', color='green',zorder=5)
            y = fit_result['fit_coadd']['hpf'][f_key][-1].best_values['a0']*np.sin((x-fit_result['fit_coadd']['hpf'][f_key][-1].best_values['t0'])*2*np.pi)
            axes[i_y,i_x].plot(x, y, linestyle=(0,(2,8)), color='red',zorder=1,label=fr'sin$\theta$ for HPF fit')
            axes[i_y,i_x].legend()
        
    else:
        raise ValueError(f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'.")

    for i_y in range(len(axes)):
        for i_x in range(len(axes[0])):
            axes[i_y,i_x].grid() 
    plt.tight_layout()


def fill_data(aman,coadd_data,fit_result,n_bins,cal_type):
    # Fill co-added data and fit result to axis manager
    # aman: axis manager
    # coadd_data: co-added data 
    # fit_result: fit result
    # n_bins: number of bins
    # cal_type: 'gain' or 'timeconstant'. type of calibration

    if not 'coadd_data' in aman.stm_ana.keys():
        bins = core.IndexAxis('bins',n_bins)
        aman.stm_ana.wrap('coadd_data',core.AxisManager(aman._axes['dets'],bins))
    
    # Fill co-added data to axis manager
    for filt_key in coadd_data.keys():
        if not filt_key in aman.stm_ana.coadd_data.keys():
            aman.stm_ana.coadd_data.wrap(filt_key,core.AxisManager())

        for freq_key in coadd_data[filt_key].keys():
            if not freq_key in aman.stm_ana.coadd_data[filt_key].keys():
                aman.stm_ana.coadd_data[filt_key].wrap(freq_key, core.AxisManager(aman._axes['dets'],aman.stm_ana.coadd_data._axes['bins']))

            for key in coadd_data[filt_key][freq_key].keys():
                arr = np.array([x if x is not None else np.full(n_bins, np.nan) for x in coadd_data[filt_key][freq_key][key]]) 
                aman.stm_ana.coadd_data[filt_key][freq_key].wrap(f'{key}', arr, [(0,'dets'), (1,'bins')], overwrite=True)

    # Fill fit result to axis manager
    for fit_key in fit_result.keys():
        if not fit_key in aman.stm_ana.keys():
            aman.stm_ana.wrap(fit_key, core.AxisManager(aman._axes['dets']))

        for filt_key in fit_result[fit_key].keys():
            if fit_key == 'fit_coadd':
                if not filt_key in aman.stm_ana[fit_key].keys():
                    aman.stm_ana[fit_key].wrap(filt_key, core.AxisManager(aman._axes['dets']))
                for freq_key in fit_result[fit_key][filt_key].keys():
                    aman.stm_ana[fit_key][filt_key].wrap(freq_key, core.AxisManager(aman._axes['dets'],aman.stm_ana.coadd_data._axes['bins']),overwrite=True)
            else:
                aman.stm_ana[fit_key].wrap(filt_key, core.AxisManager(aman._axes['dets'],core.IndexAxis('ndata_taufit',len(coadd_data[filt_key].keys()))),overwrite=True)


        for filt_key in fit_result[fit_key].keys():
            if fit_key == 'fit_coadd':
                for freq_key in fit_result[fit_key][filt_key].keys():
                    field = aman.stm_ana.fit_coadd[filt_key][freq_key]
                    result = fit_result[fit_key][filt_key][freq_key]
                    weight_axis = 'bins'
            else:
                field = aman.stm_ana[fit_key][filt_key]
                result = fit_result[fit_key][filt_key]
                weight_axis = 'ndata_taufit'

            vailed_result = [x for x in result if x is not None]
            for key in vailed_result[0].params.keys():
                arr = np.array([float(x.params[key].value) if x is not None else np.nan for x in result])
                field.wrap(key, arr, [(0,'dets')], overwrite=True)
                arr = np.array([float(x.params[key].stderr) if x is not None else np.nan for x in result])
                field.wrap(f'{key}_stderr', arr, [(0,'dets')], overwrite=True)
                arr = np.array([float(x.chisqr) if x is not None else np.nan for x in result])
                field.wrap(f'chisqr', arr, [(0,'dets')], overwrite=True)
                arr = np.array([float(x.redchi) if x is not None else np.nan for x in result])
                field.wrap(f'redchi', arr, [(0,'dets')], overwrite=True)
                if weight_axis == 'bins':# Because we don't use weight for time-constant fit for the first step analysis
                    arr = np.array([x.weights if x is not None else np.full(field._axes[weight_axis].count, np.nan) for x in result])
                    arr = arr.astype(float)
                    field.wrap(f'weights', arr, [(0,'dets'), (1,weight_axis)], overwrite=True)

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
        aman.det_cal.wrap(f'stm_gain', arr, [(0,'dets')], overwrite=True)
    if cal_type == 'timeconstant':
        arr = np.array([float(x.best_values['tau']) if x is not None else np.nan for x in fit_result['fit_amp']['lpf']])
        aman.det_cal.wrap(f'stm_tau', arr, [(0,'dets')], overwrite=True)
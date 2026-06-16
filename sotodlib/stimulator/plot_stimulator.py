import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from pathlib import Path

from sotodlib.stimulator.utils_stimulator import func_response_amplitude, func_response_phase_with_dt, func_sines
from sotodlib import tod_ops


def plot_hkdata(aman, hkdata, cal_type, show=True, output_dir=None):
    """
    Plot housekeeping data and timing information.

    Args:
        aman: axis manager with aman.stm_cal field
        hkdata: Housekeeping data including temperature data and encoder timing data.
        cal_type: Type of calibration to be performed.
        show: whether to display the plot
        output_dir: directory to save the plot. If None, the plot will not be saved.

    Return:
        fig: figure object
        axes: axes object
    """

    fig, axes = plt.subplot_mosaic([['A','A'],['B','C'],['D','E']],figsize=(10,8))

    t0 = aman.timestamps[0]

    # Chopping freq with t_cuts
    y = 1/(aman.stm_cal.t_enc[1:] - aman.stm_cal.t_enc[:-1])
    axes['A'].plot(aman.stm_cal.t_enc[:-1]-t0,y,label='Chopping_freq')
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

    idx = np.where(aman.stm_cal.positions.vals == 'env')[0][0]
    env_temps = aman.stm_cal.temps[idx]

    axes['B'].set_ylim(env_temps[0] - 5, env_temps[0] + 5)
    axes['B'].set_ylabel('Temperature [K]')
    axes['B'].set_xlabel('Time [s]')

    for key_freq, env_temp, (t_min, t_max) in zip(aman.stm_cal.freqs.vals,env_temps,aman.stm_cal.t_cuts):
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
    idx = np.where(aman.stm_cal.positions.vals == 'heater')[0][0]
    heater_temp = aman.stm_cal.temps[idx][0]
    axes['D'].set_ylim(heater_temp - 5, heater_temp + 5)
    axes['D'].set_ylabel('Temperature [K]')
    axes['D'].set_xlabel('Time [s]')


    # Encoder timing and stream timing
    axes['C'].plot(aman.stm_cal.t_enc-t0,aman.stm_cal.t_enc-aman.stm_cal.t_hk,'.')
    axes['C'].set_title('PTP_time - hk_time')
    axes['C'].set_xlabel('t_enc - t0_stream [s]')
    axes['C'].set_ylabel('t_enc - t_hk [s] (should be -0.1<t<0)')


    # Plot timing cut area
    for key_ax in ['A','B','D']:
        for key_freq, (t_min, t_max) in zip(aman.stm_cal.freqs.vals,aman.stm_cal.t_cuts):
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

    if output_dir is not None:
        i_det = 0
        obs_id = aman.obs_info.obs_id
        ufm = aman.det_info.stream_id[i_det][4:]
        ufm = ufm[0].upper() + ufm[1:]

        output_dir_ = Path(f'{output_dir}/{ufm}_{obs_id}')
        output_dir_.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir_}/{cal_type}_hk.png')

    if not show:
        plt.close(fig)
    return fig, axes


def plot_tod(aman, i_det, cal_type, show=True, output_dir=None):
    """
    Make plot for one detector.

    Args:
        aman: axis manager
        i_det: detector index
        cal_type: 'gain' or 'timeconstant'. type of calibration
        show: whether to display the plot
        output_dir: directory to save the plot. If None, the plot will not be saved.

    Return:
        fig: figure object
        axes: axes object
    """
    t0 = aman.timestamps[0]
    ufm = aman.det_info.stream_id[i_det][4:]
    ufm = ufm[0].upper() + ufm[1:]


    if cal_type == 'gain':
        fig, axes = plt.subplots(3,2,figsize=(10,8))
        fig.suptitle(f'Stimulator data, {ufm}, i_det: {i_det}, det_id: {aman.det_info.det_id[i_det]}')

        i_y = 0
        i_x = 0
        y = aman.signal[i_det]-np.mean(aman.signal[i_det])
        axes[i_y,i_x].plot(aman.timestamps-t0,y,label='Raw_data - mean')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data')
        axes[i_y,i_x].set_ylim(min(y)-(max(y)-min(y))*0.1,max(y)+(max(y)-min(y))*0.1)
        axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}')
        axes[i_y,i_x].set_xlabel('time [s]')
        axes[i_y,i_x].set_ylabel('TOD [pW]')
        axes[i_y,i_x].legend()


        i_y = 1
        i_x = 0
        x_min = 30
        x_max = 30.5
        i_x_min=int(aman.stm_cal.sampling_rate*x_min)
        i_x_max=int(aman.stm_cal.sampling_rate*x_max)
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
        i_y = 2
        i_x = 0
        f = np.arange(0,10,0.01)
        axes[i_y,i_x].set_title('High Pass Filter')
        axes[i_y,i_x].set_xlabel('Frequency [Hz]')
        axes[i_y,i_x].set_ylabel('HPF')

        i_y = 0
        i_x = 1
        x = aman.stm_cal.coadd_data['iirc']['f1_gain']['x'][i_det]
        y = aman.stm_cal.coadd_data['iirc']['f1_gain']['y'][i_det]
        yerr = aman.stm_cal.coadd_data['iirc']['f1_gain']['yerr'][i_det]
        axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=5)
        axes[i_y,i_x].set_title('Co-added signal: Raw data')
        axes[i_y,i_x].set_xlabel('Timing (1 cycle)')
        axes[i_y,i_x].set_ylabel('TOD ave [pW]')

        i_y = 1
        i_x = 1
        x = aman.stm_cal.coadd_data['hpf']['f1_gain']['x'][i_det]
        y = aman.stm_cal.coadd_data['hpf']['f1_gain']['y'][i_det]
        yerr = aman.stm_cal.coadd_data['hpf']['f1_gain']['yerr'][i_det]
        axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=5, color='C1',zorder=0,label='HPFed')
        axes[i_y,i_x].set_title(f'Co-added signal: Filtered data, {ufm}')

        x = aman.stm_cal.coadd_data['lpf']['f1_gain']['x'][i_det]
        y = aman.stm_cal.coadd_data['lpf']['f1_gain']['y'][i_det]
        yerr = aman.stm_cal.coadd_data['lpf']['f1_gain']['yerr'][i_det]
        axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=5, color='C2',zorder=0,label='(HPF+LPF)ed')
        axes[i_y,i_x].set_xlabel('Timing (1 cycle)')
        axes[i_y,i_x].set_ylabel('TOD ave [pW]')

        a0s = [aman.stm_cal['fit_coadd']['hpf']['f1_gain'][f'a{i_freq}'][i_det] for i_freq in range(7)]
        t0s = [aman.stm_cal['fit_coadd']['hpf']['f1_gain'][f't{i_freq}'][i_det] for i_freq in range(7)]
        axes[i_y,i_x].plot(x, func_sines(x, *a0s, *t0s), '-', color='red',zorder=1)
        a0s = [aman.stm_cal['fit_coadd']['lpf']['f1_gain'][f'a{i_freq}'][i_det] for i_freq in range(7)]
        t0s = [aman.stm_cal['fit_coadd']['lpf']['f1_gain'][f't{i_freq}'][i_det] for i_freq in range(7)]
        axes[i_y,i_x].plot(x, func_sines(x, *a0s, *t0s), '-', color='green',zorder=1)
        y = aman.stm_cal['fit_coadd']['hpf']['f1_gain']['a0'][i_det]*np.sin((x-aman.stm_cal['fit_coadd']['hpf']['f1_gain']['t0'][i_det])*2*np.pi)
        axes[i_y,i_x].plot(x, y, linestyle=(0,(2,8)), color='red',zorder=1,label=r'sin$\theta$ for HPF fit')
        axes[i_y,i_x].legend()


        i_y = 2
        i_x = 1
        axes[i_y,i_x].plot(aman.stm_cal.t_enc[:-1]-t0,1/(aman.stm_cal.t_enc[1:] - aman.stm_cal.t_enc[:-1]),label='y: chopping freq, t: encoder')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det],label='TOD')
        for key_freq, (t_min, t_max) in zip(aman.stm_cal.freqs.vals,aman.stm_cal.t_cuts):
            if key_freq == 'f1_gain':
                axes[i_y,i_x].axvspan(t_min, t_max, alpha=0.3, label='used data')
        axes[i_y,i_x].set_title('Overplot')
        axes[i_y,i_x].set_ylabel('Chopping frequency[Hz]')
        axes[i_y,i_x].set_xlabel('TOD time [s]')
        axes[i_y,i_x].legend()

        i_y = 2
        i_x = 0
        hpf = tod_ops.filters.high_pass_sine2(aman.stm_cal.filtering_params['hpf_cutoff'])
        filter_cutoff = aman.stm_cal.filtering_params['lpf_cutoff_factor']*aman.stm_cal.filtering_params['chopping_freqs_gain'][0]
        lpf = tod_ops.filters.low_pass_sine2(filter_cutoff, filter_cutoff*aman.stm_cal.filtering_params['lpf_width_fraction'])

        x = np.arange(0,filter_cutoff*1.2,0.1)
        y = np.full(x.shape[0],1)
        axes[i_y,i_x].plot(x, hpf(x,y),label='HPF', color='C1')
        axes[i_y,i_x].plot(x, lpf(x,y),label='LPF', color='C2')
        axes[i_y,i_x].legend()


    elif cal_type == 'timeconstant':
        # Plot basic data
        fig, axes = plt.subplots(9,2,figsize=(10,18))
        fig.suptitle(f'Stimulator data, i_det: {i_det}, det_id: {aman.det_info.det_id[i_det]}')

        i_y = 0
        i_x = 0
        y = aman.signal[i_det]-np.mean(aman.signal[i_det])
        axes[i_y,i_x].plot(aman.timestamps-t0,y,label='Raw_data - mean')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data')
        axes[i_y,i_x].set_ylim(min(y)-(max(y)-min(y))*0.1,max(y)+(max(y)-min(y))*0.1)
        axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}')
        axes[i_y,i_x].set_xlabel('time [s]')
        axes[i_y,i_x].set_ylabel('TOD [pW]')
        axes[i_y,i_x].legend()

        i_y = 0
        i_x = 1
        axes[i_y,i_x].plot(aman.stm_cal.t_enc[:-1]-t0,1/(aman.stm_cal.t_enc[1:] - aman.stm_cal.t_enc[:-1]),label='y: chopping freq, t: encoder')
        axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det],label='TOD')
        for key_freq, (t_min, t_max) in zip(aman.stm_cal.freqs.vals,aman.stm_cal.t_cuts):
            if key_freq != 'f1_gain':
                if key_freq == 'f1':
                    axes[i_y,i_x].axvspan(t_min, t_max, alpha=0.3, label='used data')
                else:
                    axes[i_y,i_x].axvspan(t_min, t_max, alpha=0.3)
        axes[i_y,i_x].set_title('Overplot')
        axes[i_y,i_x].set_ylabel('Chopping frequency[Hz]')
        axes[i_y,i_x].set_xlabel('TOD time [s]')
        axes[i_y,i_x].legend()

        i_y = 1
        i_x = 0
        f = aman.stm_cal['fit_amp']['lpf']['f'][i_det]
        axes[i_y,i_x].plot(f, aman.stm_cal['fit_amp']['lpf']['data'][i_det],'o')
        axes[i_y,i_x].plot(f, func_response_amplitude(f, tau=aman.stm_cal['fit_amp']['lpf']['tau'][i_det], a=aman.stm_cal['fit_amp']['lpf']['a'][i_det]), '-', color='red',zorder=3,label=fr'$\tau$= {aman.stm_cal["fit_amp"]["lpf"]["tau"][i_det]*1e3:.2f}ms')
        axes[i_y,i_x].plot(f, func_response_amplitude(f, tau=aman.stm_cal['fit_amp']['lpf']['tau'][i_det]*1.1, a=aman.stm_cal['fit_amp']['lpf']['a'][i_det]) , '--', color='orange',zorder=3,label=r'$\pm 10\%$ change of $\tau$')
        axes[i_y,i_x].plot(f, func_response_amplitude(f, tau=aman.stm_cal['fit_amp']['lpf']['tau'][i_det]*0.9, a=aman.stm_cal['fit_amp']['lpf']['a'][i_det]) , '--', color='orange',zorder=3)
        axes[i_y,i_x].set_xlabel('Chopping freq [Hz]')
        axes[i_y,i_x].set_ylabel('sin_theta amplitude [pW]')
        axes[i_y,i_x].set_title('Amplitude fit')
        axes[i_y,i_x].legend()

        i_y = 1
        i_x = 1
        f = aman.stm_cal['fit_phase__free']['lpf']['f'][i_det]
        data = aman.stm_cal['fit_phase__free']['lpf']['data'][i_det]
        axes[i_y,i_x].plot(f, data, 'o')

        tau = aman.stm_cal['fit_phase__free']['lpf']['tau'][i_det]
        theta_geo = aman.stm_cal['fit_phase__free']['lpf']['theta_geo'][i_det]
        dt = aman.stm_cal['fit_phase__free']['lpf']['dt'][i_det]
        axes[i_y,i_x].plot(f, func_response_phase_with_dt(f, tau, theta_geo, dt),
                            '-', color='blue', zorder=3,
                            label=fr'$\tau$={tau*1e3:.2f}ms, $\theta_\text{{geo}}$={theta_geo:.0f}deg, $\Delta t$={dt*1e3:.2f}ms')

        tau = aman.stm_cal['fit_phase__fix_tau']['lpf']['tau'][i_det]
        theta_geo = aman.stm_cal['fit_phase__fix_tau']['lpf']['theta_geo'][i_det]
        dt = aman.stm_cal['fit_phase__fix_tau']['lpf']['dt'][i_det]
        axes[i_y,i_x].plot(f, func_response_phase_with_dt(f, tau, theta_geo, dt),
                            '-', color='green', zorder=3,
                            label=fr'$\tau$={tau*1e3:.2f}ms(fix), $\theta_\text{{geo}}$={theta_geo:.0f}deg , $\Delta t$={dt*1e3:.2f}ms')

        axes[i_y,i_x].set_xlabel('Chopping freq [Hz]')
        axes[i_y,i_x].set_ylabel('Phase delay [deg]')
        axes[i_y,i_x].set_title('Phase fit')
        axes[i_y,i_x].legend()

        i_y = 1
        for i_freq, f_key in enumerate(['f1','f2','f3','f4','f5','f6','f7']):
            i_y += 1

            i_x = 0
            idx = np.where(aman.stm_cal.freqs.vals == f_key)[0][0]
            x_min = aman.stm_cal.t_cuts[idx][0]
            x_max = x_min+0.2
            i_x_min=int(aman.stm_cal.sampling_rate*x_min)
            i_x_max=int(aman.stm_cal.sampling_rate*x_max)
            axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal[i_det]- np.mean(aman.signal[i_det][i_x_min:i_x_max]),label='Raw data - mean')
            axes[i_y,i_x].plot(aman.timestamps-t0,aman.signal_hpf[i_det],label='HPFed data',color='C1')
            axes[i_y,i_x].set_title(f'TOD data, i_det={i_det}, f={aman.stm_cal.filtering_params.chopping_freqs_tau[i_freq]:.0f}Hz')
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
            x = aman.stm_cal.coadd_data['iirc'][f_key]['x'][i_det]
            y = aman.stm_cal.coadd_data['iirc'][f_key]['y'][i_det]
            yerr = aman.stm_cal.coadd_data['iirc'][f_key]['yerr'][i_det]
            axes[i_y,i_x].errorbar(x,y-np.mean(y),yerr, fmt='o', capsize=5, label='IIRCed data - mean')
            axes[i_y,i_x].set_title(f'Co-added signal, f={aman.stm_cal.filtering_params.chopping_freqs_tau[i_freq]:.0f}Hz')
            axes[i_y,i_x].set_xlabel('Timing (1 cycle)')
            axes[i_y,i_x].set_ylabel('TOD [pW]')

            x = aman.stm_cal.coadd_data['hpf'][f_key]['x'][i_det]
            y = aman.stm_cal.coadd_data['hpf'][f_key]['y'][i_det]
            yerr = aman.stm_cal.coadd_data['hpf'][f_key]['yerr'][i_det]
            axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=3, color='C1', label='(IIRC+HPF)ed data')

            x = aman.stm_cal.coadd_data['lpf'][f_key]['x'][i_det]
            y = aman.stm_cal.coadd_data['lpf'][f_key]['y'][i_det]
            yerr = aman.stm_cal.coadd_data['lpf'][f_key]['yerr'][i_det]
            axes[i_y,i_x].errorbar(x,y,yerr, fmt='o', capsize=3, color='C2', label='(IIRC+HPF+LPF)ed data')


            a0s = [aman.stm_cal['fit_coadd']['hpf'][f_key][f'a{i_freq}'][i_det] for i_freq in range(7)]
            t0s = [aman.stm_cal['fit_coadd']['hpf'][f_key][f't{i_freq}'][i_det] for i_freq in range(7)]
            axes[i_y,i_x].plot(x, func_sines(x, *a0s, *t0s), '-', color='red', zorder=5)
            a0s = [aman.stm_cal['fit_coadd']['lpf'][f_key][f'a{i_freq}'][i_det] for i_freq in range(7)]
            t0s = [aman.stm_cal['fit_coadd']['lpf'][f_key][f't{i_freq}'][i_det] for i_freq in range(7)]
            axes[i_y,i_x].plot(x, func_sines(x, *a0s, *t0s), '-', color='green', zorder=5)
            y = aman.stm_cal['fit_coadd']['hpf'][f_key]['a0'][i_det]*np.sin((x-aman.stm_cal['fit_coadd']['hpf'][f_key]['t0'][i_det])*2*np.pi)
            axes[i_y,i_x].plot(x, y, linestyle=(0,(2,8)), color='red',zorder=1,label=r'sin$\theta$ for HPF fit')
            axes[i_y,i_x].legend()

    else:
        raise ValueError(f"'{cal_type}' is a wrong type. Please specify 'gain' or 'timeconstant'.")

    for i_y in range(len(axes)):
        for i_x in range(len(axes[0])):
            axes[i_y,i_x].grid()
    plt.tight_layout()


    if output_dir is not None:
        obs_id = aman.obs_info.obs_id
        ufm = aman.det_info.stream_id[i_det][4:]
        ufm = ufm[0].upper() + ufm[1:]

        output_dir_ = Path(f'{output_dir}/{ufm}_{obs_id}')
        output_dir_.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir_}/{cal_type}_det{i_det:04d}.png')

    if not show:
        plt.close(fig)


    return fig, axes

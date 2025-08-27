from sotodlib import core
from sotodlib.io import hkdb
import sotodlib.site_pipeline.util as sp_util

import argparse
import time
import yaml
import h5py
import os
import traceback
from tqdm import tqdm
from typing import Optional
import datetime as dt
import requests
from io import StringIO
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import find_peaks, peak_widths
from so3g import RangesInt32


def get_acu_db(h5_path):
    """
    Get precomputed scan properites data azimuth including speeds and
    accelerations from the ACU hkdb.

    Parameters
    ----------
    h5_path : str
        Path to the h5 file

    Returns
    -------
    scan_props : ndarray
        Array containing scan properties
    """
    with h5py.File(h5_path, "r") as f:
        scan_props = f['scan_props'][:]
    return scan_props


def get_apex_data(start_date=dt.datetime(2024, 5, 19, tzinfo=dt.timezone.utc),
                  end_date=dt.datetime(2024, 6, 19, tzinfo=dt.timezone.utc)):
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

    request = requests.post(APEX_DATA_URL, data={
            'wdbo': 'csv/download',
            'max_rows_returned': 79400,
            'start_date': start_date.strftime('%Y-%m-%dT%H:%M:%S') + '..' \
                + end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'tab_pwv': 'on',
            'shutter': 'SHUTTER_OPEN',
            #'tab_shutter': 'on',
        })

    def date_converter(d):
        naive_dt = dt.datetime.fromisoformat(d)
        return naive_dt.replace(tzinfo=dt.timezone.utc)

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=',', skip_header=2,
        converters={0: date_converter},
        dtype=[('dates', dt.datetime), ('pwv', float)],
    )

    outdata = {
        'timestamps': [d.timestamp() for d in data['dates']],
        'pwv': data['pwv']
    }
    return outdata


def rolling_mean(arr, window_size):
    view = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    return np.mean(view, axis=1)


def smooth_mask(mask, clean, extend):
    r = RangesInt32.from_mask(mask)
    return (~(~r).buffer(clean)).buffer(extend)


def denumpy(output):
    if isinstance(output, dict):
        return {k: denumpy(v)
                for k, v in output.items()}
    if isinstance(output, list):
        return [denumpy(x) for x in output]
    if isinstance(output, tuple):
        return tuple((denumpy(x) for x in output))
    if isinstance(output, np.generic):
        return output.item()
    return output


def get_vel_accel(data, time_range):
    result = data
    t00 = time_range[0]

    # Bulk classification from udp stream.
    az = np.convolve(result['acu.acu_udp_stream.Corrected_Azimuth'][1], np.ones(21) / 21, mode='valid')
    t = result['acu.acu_udp_stream.Corrected_Azimuth'][0][10:-10]

    v = np.gradient(az) / np.gradient(t)

    pos = smooth_mask(v > .05, 5, 10)
    neg = smooth_mask(v < -.05, 5, 10)

    # Classify each sweep.
    csweeps = []
    for rr in [pos, neg]:
        for r in rr.ranges():
            sl = slice(*r)
            az_mid = az[sl].mean()
            vtyp = np.median(v[sl])
            stable_mask = abs(v[sl] - vtyp) < .02
            stable = stable_mask.sum() / (r[1] - r[0])
            # Fit line.
            const_vel = stable > 0.8
            if const_vel:
                _t0 = t[sl][stable_mask].mean()
                p = np.polyfit(t[sl][stable_mask] - _t0,
                               az[sl][stable_mask], 1)
            else:
                _t0, p = None, None
            csweeps.append(denumpy((r.tolist(), const_vel, vtyp, az_mid, _t0, p)))

    csweeps.sort()

    # Under the assumption of *constant* turn-around time, you can figure out if there's an az_drift.
    # Just check the intersection point of two adjacent sweeps; see if those corners evolve in time.
    points = []
    for sw0, sw1 in zip(csweeps[:-1], csweeps[1:]):
        if not (sw0[1] and sw1[1]):
            continue
        assert(sw0[2] * sw1[2] < 0)
        # At what time and az do the two trajectories intersect?
        #     y0 = m0 (t - t0) + b0
        #     y1 = m1 (t - t1) + b1
        # ->
        #     m0 (t - t0) + b0 = m1 (t - t0 + (t0 - t1)) + b1
        #     (m0 - m1) (t - t0) = m1(t0 - t1) + b1 - b0
        #     (t - t0) = (m1 (t0 - t1) + (b1 - b0)) /  (m0 - m1)
        t0, (m0, b0) = sw0[4:6]
        t1, (m1, b1) = sw1[4:6]
        t_t0 = (m1 * (t0 - t1) + (b1 - b0)) / (m0 - m1)
        y0 = m0 * t_t0 + b0
        points.append(denumpy((np.sign(sw0[2]), t_t0 + t0, y0, m0, m1)))

    # Measure az_drift.
    sgn, _t, _y, _, _ = np.transpose(points)
    drifts = []
    for _sgn in [-1, 1]:
        p = np.polyfit(_t[sgn==_sgn] - t00, _y[sgn==_sgn], 1)
        drifts.append(p)

    az_drift, az_center = np.mean(drifts, axis=0)

    # Loop over turn-arounds again and get adjusted values for az_cmd.
    sweep_time = np.median(np.diff(_t))
    window = sweep_time * .2
    cmd_t, cmd_az = result['acu.acu_status.Azimuth_commanded_position']
    cmd_lims = []
    for tw in _t:
        s = (abs(cmd_t - tw) < window)
        if not s.any():
            continue
        mn, mx = cmd_az[s].min(), cmd_az[s].max()
        mn -= az_drift * (tw - t00)
        mx -= az_drift * (tw - t00)
        cmd_lims.extend(denumpy([mn, mx]))

    xaz_min, xaz_max = np.min(cmd_lims).item(), np.max(cmd_lims).item()

    # Now use those to get the turn-around time of each sweep.
    props = []
    for p in points:
        sgn, tw, y0, m0, m1 = p
        y0 = y0 - az_drift * (tw - t00)
        az_turn = xaz_min if m0<0 else xaz_max
        tt0 = (y0 - az_turn) / m0
        tt1 = (az_turn - y0) / m1
        t_turnaround = tt0 + tt1
        a_turnaround = -sgn * (m1 - m0) / t_turnaround
        props.append((sgn*m0, a_turnaround))

    vel, accel = np.mean(props, axis=0).round(4)
    return vel, accel


def get_hk_and_pwv_data(obs, apex_data, cfg_site, cfg_plat,
                        frame_offsets_site, frame_offsets_plat,
                        nextline_db, acu_scan_props):
    """
    Get housekeeping and PWV data from the site and platform HK databases.

    Parameters
    ----------
    obs : OrderedDict
        Returned from obsdb.
    apex_data : dict
        Dictionary with keys 'timestamps' and 'pwv', which are lists of
        unix ctimestamps and precipitable water vapor values, respectively.
        Returned by ``get_apex_data``.
    hkdb_site : hkdb.HkDB
        hkdb object for the site.
    hkdb_plat : hkdb.HkDB
        hkdb object for the platform.
    platform : str
        Platform name i.e. satp1, satp2, lat

    Returns
    -------
    hk_aman : core.AxisManager
        AxisManager object with the following fields:
        - pwv : float
            Median PWV value.
        - dpwv : float
            PWV peak-peak range.
        - ambient_temp : float
            Median ambient temperature.
        - uv : float
            Median UV index.
        - hwp_rate : float
            Median HWP rate.
        - dhwp_rate : float
            HWP rate peak-peak range.
        - hwp_direction : float
            HWP direction +/- 1.
    """
    try:
        logger = sp_util.init_logger("split_info")
        oid = obs['obs_id']
        t0 = obs['start_time']
        t1 = obs['start_time']+obs['duration']

        cfg_site = hkdb.HkConfig.from_yaml(cfg_site)
        lspec_site = hkdb.LoadSpec(start=t0, end=t1, cfg=cfg_site,
                                fields=['pwv', 'ambient_temp', 'uv', 'wind_spd', 'wind_dir'],
                                frame_offsets=frame_offsets_site)
        result_site = hkdb.load_hk(lspec_site, show_pb=False)

        cfg_plat = hkdb.HkConfig.from_yaml(cfg_plat)
        lspec_plat = hkdb.LoadSpec(start=t0, end=t1, cfg=cfg_plat,
                                   fields=['hwp_rate1', 'hwp_rate2', 'hwp_direction', 'az_pos', 'az'],
                                   frame_offsets=frame_offsets_plat)
        result_plat = hkdb.load_hk(lspec_plat, show_pb=False)

        try:
            if 'env-radiometer-class.pwvs.pwv' in result_site.data.keys():
                ts = result_site.data['env-radiometer-class.pwvs.pwv'][0]
                apex_interp = np.interp(ts, apex_data['timestamps'], apex_data['pwv'])
                pwvs_apex = 0.03 + 0.84*apex_interp
                pwvs_class = result_site.data['env-radiometer-class.pwvs.pwv'][1]
                # only apex
                if ((np.all(np.isnan(pwvs_class))) and (not np.all(np.isnan(pwvs_apex)))):
                    pwv = np.nanmedian(pwvs_apex)
                    pwv_start = pwvs_apex[0]
                    pwv_end = pwvs_apex[-1]
                    dpwv = np.ptp(pwvs_apex[~np.isnan(pwvs_apex)])
                # class and apex
                elif ((not np.all(np.isnan(pwvs_class))) and (not np.all(np.isnan(pwvs_apex)))):
                    pwvs_corr = np.asarray([pwva if ((pwvc < 0.3) | (pwvc > 3)) else pwvc for pwvc, pwva in zip(pwvs_class, pwvs_apex)])
                    pwv = np.nanmedian(pwvs_corr)

                    if len(pwvs_corr) > 1:
                        pwv_start = pwvs_corr[0]
                        pwv_end = pwvs_corr[-1]
                    else:
                        pwv_start = pwvs_corr[0]
                        pwv_end = pwvs_corr[0]
                    dpwv = np.ptp(pwvs_corr[~np.isnan(pwvs_corr)])
                # only class
                elif ((not np.all(np.isnan(pwvs_class))) and (np.all(np.isnan(pwvs_apex)))):
                    pwvs_corr  = np.asarray([pwvc if pwvc > 0.3 and pwvc < 3 else np.nan for pwvc in pwvs_class])
                    pwv = np.nanmedian(pwvs_corr)
                    if len(pwvs_corr) > 1:
                        pwv_start = pwvs_corr[0]
                        pwv_end = pwvs_corr[-1]
                    else:
                        pwv_start = pwvs_corr[0]
                        pwv_end = pwvs_corr[0]
                    dpwv = np.ptp(pwvs_corr[~np.isnan(pwvs_corr)])
                # none
                else:
                    pwv = 5500
                    pwv_start = 5500
                    pwv_end = 5500
                    dpwv = 5500
            # only apex found
            else:
                ts = np.array([t0, t1])
                apex_interp = np.interp(ts, apex_data['timestamps'], apex_data['pwv'])
                pwvs_apex = 0.03 + 0.84*apex_interp
                # if valid apex values
                if (not np.all(np.isnan(pwvs_apex))):
                    pwv = np.nanmedian(pwvs_apex)
                    if len(pwvs_apex) > 1:
                        pwv_start = pwvs_apex[0]
                        pwv_end = pwvs_apex[-1]
                    else:
                        pwv_start = pwvs_apex[0]
                        pwv_end = pwvs_apex[0]
                    dpwv = np.ptp(pwvs_apex[~np.isnan(pwvs_apex)])
                else:
                    pwv = 5500
                    pwv_start = 5500
                    pwv_end = 5500
                    dpwv = 5500
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"ERROR getting pwv:\n{errmsg}\n{tb}")
            pwv = 5500
            pwv_start = 5500
            pwv_end = 5500
            dpwv = 5500

        try:
            ambient_temp = np.nanmedian(result_site.data['env-vantage.weather_data.temp_outside'][1])
            uv = np.nanmedian(result_site.data['env-vantage.weather_data.UV'][1])/10
            wind_spd = np.nanmedian(result_site.data['env-vantage.weather_data.wind_speed'][1]*1.609)
            wind_dir = np.nanmedian(result_site.data['env-vantage.weather_data.wind_dir'][1])
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"ERROR getting site hkdb data:\n{errmsg}\n{tb}")
            ambient_temp = 5500
            uv = 5500
            wind_spd = 5500
            wind_dir = 5500

        try:
            _hwp_rate = np.array([])
            for k in result_plat.data.keys():
                if 'hwp-bbb' in k:
                    _hwp_rate = np.concatenate((_hwp_rate, result_plat.data[k][1]))
            hwp_rate = np.nanmedian(_hwp_rate)
            dhwp_rate = np.ptp(_hwp_rate[~np.isnan(_hwp_rate)])
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"ERROR getting site hwp rate data:\n{errmsg}\n{tb}")
            hwp_rate = 5500
            dhwp_rate = 5500

        try:
            hwp_direction = 2*int(np.nanmedian(result_plat.data['hwp-pid.hwppid.direction'][1])) - 1
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"ERROR getting hwp direction:\n{errmsg}\n{tb}")
            hwp_direction = 5500

        try:
            # use ACU db
            if acu_scan_props is not None:
                index = np.where(acu_scan_props['obs_id'] == np.bytes_(oid))[0]
                if acu_scan_props['data_found'][index][0]:
                    avg_vel = acu_scan_props['vel'][index][0]
                    avg_acc = acu_scan_props['accel'][index][0]
                else:
                    avg_vel = 5500
                    avg_acc = 5500
            # use nextline db
            elif nextline_db is not None:
                start = dt.datetime.utcfromtimestamp(t0).replace(tzinfo=dt.timezone.utc)
                stop = dt.datetime.utcfromtimestamp(t1).replace(tzinfo=dt.timezone.utc)

                mask = (nextline_db['time'] >= start) & (nextline_db['time'] <= stop)
                if len(nextline_db['speed'][mask]) > 0:
                    avg_vel = np.nanmedian(nextline_db['speed'][mask])
                else:
                    avg_vel = 5500
                if len(nextline_db['accel'][mask]) > 0:
                    avg_acc = np.nanmedian(nextline_db['accel'][mask])
                else:
                    avg_acc = 5500

                if np.isnan(avg_vel):
                    avg_vel = 5500
                if np.isnan(avg_acc):
                    avg_acc = 5500
            # calculate from ACU
            else:
                az = result_plat.data['acu.acu_status.Azimuth_commanded_position'][1]
                ts = result_plat.data['acu.acu_status.Azimuth_commanded_position'][0]
                # vel = np.gradient(az, ts - ts[0])
                # # acc = np.gradient(vel, ts - ts[0])
                # avg_vel = np.round(np.nanmedian(np.abs(vel)),2)

                # ts = ts - ts[0]
                # # acc = np.abs(acc)

                # # acc = acc[100:-100]
                # ts = ts[100:-100]
                # vel = vel[100:-100]

                # # peaks, _ = find_peaks(acc, height=0.1*np.max(acc), distance=10)
                # # results_half = peak_widths(acc, peaks, rel_height=0.9)

                # peaks, _ = find_peaks(-np.abs(vel), height=0.5*np.min(-np.abs(vel)), distance=15)
                # results_half = peak_widths(-np.abs(vel), peaks, rel_height=0.9)

                # left = results_half[2]
                # right = results_half[3]
                # left = np.ceil(left).astype(int)
                # right = np.floor(right).astype(int)

                # ts_diff = np.zeros(len(left))
                # for i, (l, r) in enumerate(zip(left, right)):
                #     ts_diff[i] = (ts[r] - ts[l])
                # avg_acc = np.round(2*avg_vel/np.median(ts_diff), 3)

                # avg_acc = np.round(np.nanmedian(avg_acc), 2)

                # if np.isnan(avg_vel):
                #     avg_vel = 5500
                # if np.isnan(avg_acc):
                #     avg_acc = 5500
                avg_vel, avg_acc = get_vel_accel(result_plat.data, [t0, t1])

        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"ERROR getting site az scan params:\n{errmsg}\n{tb}")
            avg_acc = 5500
            avg_vel = 5500

        entry_dict = {'obs:obs_id': oid, 'pwv': pwv, 'dpwv': dpwv, 'pwv_start': pwv_start,
                      'pwv_end': pwv_end, 'wind_spd': wind_spd, 'wind_dir': wind_dir,
                      'ambient_temp': ambient_temp, 'uv': uv, 'hwp_rate': hwp_rate,
                      'dhwp_rate': dhwp_rate, 'az_acc': avg_acc, 'az_vel': avg_vel,
                      'hwp_direction': hwp_direction}
        return None, entry_dict
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.info(f"ERROR:\n{errmsg}\n{tb}")
        return 'Fail', [errmsg, tb]

def get_man_db(configs):
    """Get or create a ManifestDb found for a given
    config.

    Arguments
    ----------
    configs : dict
        The configuration dictionary.

    Returns
    -------
    db : ManifestDb
        ManifestDb object
    """
    if os.path.exists(configs['mandb_fpath']):
        db = core.metadata.ManifestDb(configs['mandb_fpath'])
    else:
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('pwv')
        scheme.add_data_field('dpwv')
        scheme.add_data_field('ambient_temp')
        scheme.add_data_field('uv')
        scheme.add_data_field('wind_spd')
        scheme.add_data_field('wind_dir')
        scheme.add_data_field('hwp_rate')
        scheme.add_data_field('pwv_start')
        scheme.add_data_field('pwv_end')
        scheme.add_data_field('dhwp_rate')
        scheme.add_data_field('hwp_direction')
        scheme.add_data_field('az_acc')
        scheme.add_data_field('az_vel')
        db = core.metadata.ManifestDb(configs['mandb_fpath'],
                                      scheme=scheme)
    return db

def gather_frame_offsets(cfg, obs_list, logger=None):
    if logger is None:
        logger = sp_util.init_logger("split_info")

    logger.info('Frame offset file missing, gathering frame offsets...')
    hkcfg_site = hkdb.HkConfig.from_yaml(cfg['site_hkdb_cfg'])
    hkcfg_plat = hkdb.HkConfig.from_yaml(cfg['plat_hkdb_cfg'])

    db_plat = hkdb.HkDb(hkcfg_plat)
    db_site = hkdb.HkDb(hkcfg_site)

    site_fields = ['pwv', 'ambient_temp', 'uv', 'wind_spd', 'wind_dir']
    plat_fields = ['hwp_rate1', 'hwp_rate2', 'hwp_direction', 'az_pos', 'az']
    site_offsets = {}
    plat_offsets = {}

    for obs in tqdm(obs_list):
        start, stop = obs['start_time'], obs['stop_time']
        site_offsets[obs['obs_id']] = hkdb.get_frame_offsets(
            hkcfg_site,
            start, stop,
            site_fields,
            db_site
        )
        plat_offsets[obs['obs_id']] = hkdb.get_frame_offsets(
            hkcfg_plat,
            start, stop,
            plat_fields,
            db_plat
        )

    return site_offsets, plat_offsets

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Config File")
    parser.add_argument(
        '--query',
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",
        type=str
    )
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )
    return parser


def main(config: str,
         query: Optional[str] = None,
         nproc: Optional[int] = 4):
    logger = sp_util.init_logger("split_info", verbosity=3)
    cfg = yaml.safe_load(open(config, 'r'))
    ctx = core.Context(cfg['context_file'])
    logger.info('Constructing obslist and getting database...')
    obslist = ctx.obsdb.query(query)
    man_db = get_man_db(cfg)
    entries = man_db.inspect()
    if len(entries) != 0:
        db_obs = [entry['obs:obs_id'] for entry in entries]
        all_obs = np.array([o['obs_id'] for o in obslist])
        excluded_obs = np.setxor1d(db_obs, all_obs)
        obslist = [o for o in obslist if o['obs_id'] in excluded_obs]

    logger.info('Collecting frame offsets...')
    if os.path.exists(cfg['frame_offset_path']):
        logger.info('Frame offset file exists, loading from file...')
        frame_offsets = yaml.safe_load(open(cfg['frame_offset_path'], 'r'))
        frame_offsets_site = frame_offsets['site']
        frame_offsets_plat = frame_offsets['plat']
    else:
        frame_offsets_site, frame_offsets_plat = gather_frame_offsets(cfg, obslist)
        yaml.dump({'site': frame_offsets_site, 'plat': frame_offsets_plat},
                  open(cfg['frame_offset_path'], 'w'))

    logger.info('Finished collecting offsets...')

    apex_data = np.load(cfg['apex_arxiv'], allow_pickle=True).item()

    errlog = os.path.join(os.path.dirname(cfg['mandb_fpath']),
                          'errlog.txt')

    if cfg.get('acu_db_path', None):
        acu_scan_props = get_acu_db(cfg['acu_db_path'])
    else:
        acu_scan_props = None

    if cfg.get('nextline_db_fpath', None):
        nextline_data = np.load(cfg['nextline_db_fpath'], allow_pickle=True)
        nextline_db = {key: nextline_data[key] for key in nextline_data.files}
    else:
        nextline_db = None

    multiprocessing.set_start_method('spawn')

    i = 0
    tot_num = len(obslist)
    logger.info('Launching multiproc pool...')
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_hk_and_pwv_data, obs=o, apex_data=apex_data,
                              cfg_site=cfg['site_hkdb_cfg'],
                              cfg_plat=cfg['plat_hkdb_cfg'],
                              frame_offsets_site=frame_offsets_site[o['obs_id']],
                              frame_offsets_plat=frame_offsets_plat[o['obs_id']],
                              nextline_db=nextline_db,
                              acu_scan_props=acu_scan_props) for o in obslist]
        for future in as_completed(futures):
            logger.info(f'New future as_completed result {i}/{tot_num}')
            try:
                err, entry_dict = future.result()
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                logger.info(f"ERROR: future.result()\n{errmsg}\n{tb}")
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            i+=1

            if err:
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, processing error\n{entry_dict[0]}\n{entry_dict[1]}\n')
                f.close()
                continue
            else:
                man_db.add_entry(entry_dict)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)

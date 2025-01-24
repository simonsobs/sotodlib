from sotodlib import core
from sotodlib.io import hkdb
import sotodlib.site_pipeline.util as sp_util

import argparse
import time
import h5py
import yaml
import os
import traceback
from typing import Optional
import datetime as dt
import requests
from io import StringIO
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_apex_data(start_date=dt.datetime(2024,5,19),
                  end_date=dt.datetime(2024,6,19)):
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
        return dt.datetime.fromisoformat(d.decode("utf-8"))

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=',', skip_header=2,
        converters={0: date_converter},
        dtype=[('dates', dt.datetime), ('pwv', float)],
    )
    
    outdata = {'timestamps':[d.timestamp() for d in data['dates']],
               'pwv':data['pwv']}
    return outdata

def rolling_mean(arr, window_size):
    view = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    return np.mean(view, axis=1)

def get_hk_and_pwv_data(obs, apex_data, platform, cfg_site,
                        cfg_plat, axiv_path, ctx_fpath, frame_offsets_site, frame_offsets_plat):
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
                                fields=['pwv', 'ambient_temp', 'uv'],
                                frame_offsets=frame_offsets_site)
        result_site = hkdb.load_hk(lspec_site, show_pb=False)
        
        cfg_plat = hkdb.HkConfig.from_yaml(cfg_plat)
        lspec_plat = hkdb.LoadSpec(start=t0, end=t1, cfg=cfg_plat,
                                   fields=['hwp_rate2', 'hwp_rate2', 'hwp_direction', 'az_pos'],
                                   frame_offsets=frame_offsets_plat)
        result_plat = hkdb.load_hk(lspec_plat, show_pb=False)

        try:
            ts = result_site.data['env-radiometer-class.pwvs.pwv'][0]
            apex_interp = np.interp(ts, apex_data['timestamps'], apex_data['pwv'])
            pwvs_apex = [0.03+0.84*apex_interp[np.argmin(np.abs(result_site.data['env-radiometer-class.pwvs.pwv'][0]-oid))] for oid in ts]
            pwvs_class = result_site.data['env-radiometer-class.pwvs.pwv'][1]
            if ((np.all(np.isnan(pwvs_class))) & (not np.all(np.isnan(pwvs_apex)))):
                pwv = np.nanmedian(pwvs_apex)
                dpwv = np.ptp(pwvs_apex[~np.isnan(pwvs_apex)])
            elif ((not np.all(np.isnan(pwvs_class))) & (not np.all(np.isnan(pwvs_apex)))):
                pwvs_corr = np.asarray([pwva if (pwvc < 0.3) | (pwvc > 3) else pwvc for pwvc, pwva in zip(pwvs_class, pwvs_apex)])
                pwv = np.nanmedian(pwvs_corr)
                dpwv = np.ptp(pwvs_corr[~np.isnan(pwvs_corr)])
            else:
                pwv = 5500
                dpwv = 5500
        except:
            pwv = 5500
            dpwv = 5500

        try:
            ambient_temp = np.nanmedian(result_site.data['env-vantage.weather_data.temp_outside'][1])
            uv = np.nanmedian(result_site.data['env-vantage.weather_data.UV'][1])
        except Exception as e:
            print(e)
            ambient_temp = 5500
            uv = 5500
            
        try:
            _hwp_rate = np.array([])
            for k in result_plat.data.keys():
                if 'hwp-bbb' in k:
                    _hwp_rate = np.concatenate((_hwp_rate, result_plat.data[k][1]))
            hwp_rate = np.nanmedian(_hwp_rate)
            dhwp_rate = np.ptp(_hwp_rate[~np.isnan(_hwp_rate)])
        except Exception as e:
            print(e)
            hwp_rate = 5500
            dhwp_rate = 5500

        try:
            hwp_direction = 2*int(np.nanmedian(result_plat.data['hwp-pid.hwppid.direction'][1])) - 1
        except Exception as e:
            print(e)
            hwp_direction = 5500

        try:
            n_inspect = len(result_plat.data['acu.acu_udp_stream.Corrected_Azimuth'][0])//4
            az = rolling_mean(result_plat.data['acu.acu_udp_stream.Corrected_Azimuth'][1][n_inspect:-n_inspect], 200)
            ts = rolling_mean(result_plat.data['acu.acu_udp_stream.Corrected_Azimuth'][0][n_inspect:-n_inspect], 200)
            vel = np.gradient(az)/np.gradient(ts)
            acc = np.gradient(vel)/np.gradient(ts)
            avg_acc = np.round(np.mean(np.sort(np.abs(acc))[-1:-100:-1])/1.88,2)
            avg_vel = np.round(np.median(np.abs(vel)),2)
        except Exception as e:
            print(e)
            avg_acc = 5500
            avg_vel = 5500

        entry_dict = {'obs:obs_id': oid, 'pwv': pwv, 'dpwv': dpwv, 'ambient_temp': ambient_temp,
                      'uv': uv, 'hwp_rate': hwp_rate, 'dhwp_rate': dhwp_rate, 'az_acc': avg_acc,
                      'az_vel': avg_vel, 'hwp_direction': hwp_direction}
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
        scheme.add_data_field('hwp_rate')
        scheme.add_data_field('dhwp_rate')
        scheme.add_data_field('hwp_direction')
        scheme.add_data_field('az_acc')
        scheme.add_data_field('az_vel')
        db = core.metadata.ManifestDb(configs['mandb_fpath'],
                                      scheme=scheme)
    return db

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
    logger = sp_util.init_logger("split_info")
    cfg = yaml.safe_load(open(config, 'r'))
    ctx = core.Context(cfg['context_file'])
    obslist = ctx.obsdb.query(query)
    db = get_man_db(cfg)
    entries = db.inspect()
    if len(entries) != 0:
        db_obs = [entry['obs:obs_id'] for entry in entries]
        all_obs = np.array([o['obs_id'] for o in obslist])
        excluded_obs = np.setxor1d(db_obs, all_obs)
        obslist = [o for o in obslist if o['obs_id'] in excluded_obs]
    
    apex_data = np.load(cfg['apex_arxiv'], allow_pickle=True).item()
    basedir = os.path.dirname(cfg['mandb_fpath'])

    errlog = os.path.join(os.path.dirname(cfg['mandb_fpath']),
                          'errlog.txt')

    multiprocessing.set_start_method('spawn')

    man_db = get_man_db(cfg)
    i = 0
    tot_num = len(obslist)
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_hk_and_pwv_data, obs=o, apex_data=apex_data,
                              cfg_site=cfg['site_hkdb_cfg'], cfg_plat=cfg['plat_hkdb_cfg'],
                              platform=cfg['platform'], ctx_fpath=cfg['context_file'],
                              axiv_path=basedir) for o in obslist]
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


import numpy as np
import datetime
import argparse
import time
import sys
import matplotlib.pyplot as plt

from sotodlib.core import *

def find_fractions(obs_collection, weight_bool=False, debug=False):
    i = 0

    planet_time = 0
    cmb_time = 0
    non_planet_cal_time = 0
    oper_time = 0
    det_est_time = 0

    weights = []

    while True:
        one_obs = obs_collection.fetchone()
    
        if one_obs is None:
            break
    
        if one_obs["type"] == 'obs':
            if weight_bool:
                iweight = np.sum(np.array(list(one_obs["obs_id"].split('_')[-1])).astype(int)) / 7.
                if debug:
                    print(iweight)
                weights.append(iweight)
            else:
                iweight = 1.
                
            if one_obs["subtype"] == 'cal':
                #print(one_obs["tag"])
                if one_obs["tag"] in ['jupiter', 'saturn', 'moon', 'taua']:
                    planet_time += (one_obs["stop_time"] - one_obs["start_time"]) * iweight
                    det_est_time += 15.*60.
                    if debug:
                        print(one_obs["obs_id"], one_obs["tag"])
                elif np.any([k in one_obs["tag"] for k in ['el_nod', 'wg']]):
                    non_planet_cal_time += (one_obs["stop_time"] - one_obs["start_time"]) * iweight
                else:
                    pass
            elif one_obs["subtype"] == 'cmb':
                cmb_time += (one_obs["stop_time"] - one_obs["start_time"]) * iweight
                i += 1
            
                if i%5:
                    det_est_time += 15.*60.
                
        elif one_obs["type"] == 'oper' and one_obs["tag"] == 'take_noise':
            oper_time += (one_obs["stop_time"] - one_obs["start_time"])

    return (planet_time, cmb_time, non_planet_cal_time, oper_time, det_est_time), weights

def get_obs_results(platform_name, start_ctime, end_ctime):

    context_file = f'/so/metadata/{platform_name}/contexts/use_this.yaml'
    ctx = Context(filename=context_file)

    returned_obs = ctx.obsdb.conn.execute("select distinct obs_id, type, subtype, tag, start_time, stop_time from obs natural join tags where start_time > {} and stop_time < {}".format(start_ctime, end_ctime))

    return returned_obs


def print_fractions(raw_times, span=3600.*24.):
    time_org = ['planet', 'cmb', 'other_cal', 'oper', 'tune_estimated']
    for i,tt in enumerate(raw_times):
        print(f'Fraction spent in obs type {time_org[i]}: {tt/span}')

def plot_obs_eff(xax, fields, labels=[], weight_bool=False, weights=[]):
    idx_delta = (10 if len(xax) > 30. else 2)
    print(idx_delta)
    
    fig = plt.figure()
    init_vec = np.zeros(len(xax))
    for i,f in enumerate(fields):
        l = plt.plot(xax, f+init_vec, '.-', label=labels[i])
        c = l[0].get_color()
        plt.fill_between(xax, init_vec, f+init_vec, color=c, alpha=0.5)
        init_vec += f

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Efficiency (x / calendar time)')

    if len(xax) > 10:
        xt = plt.xticks()
        plt.xticks(xt[0][::idx_delta], xt[1][::idx_delta])

    plt.xticks(rotation=20.)

    if weight_bool:
        ax = plt.twinx()
        weights = np.array(weights)
        weights[np.isnan(weights)] = 0.
        ax.plot(xax, weights, 'k.-')

        ax.set_ylabel('(Avg # UFM)/7')

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('platform')
    parser.add_argument('-r', '--range', default='', help='Range over which to estimate obs efficiency. Options are d for day, w for week, or m for month')
    parser.add_argument('-d', '--daily', default=False, action='store_true', help='Group data into daily data points for plotting. Should be applied if -p arg is passed.')
    parser.add_argument('-w', '--weight', action='store_true', default=False, help='If passed, will calculate outputs weighted by UFM yield (whether for full time range or over days.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='If passed, will plot results. Should be used with -d for now.')
    parser.add_argument('-o', '--plot-output-path', default='test.png', help='Path to save plotted figure to.')
    parser.add_argument('-t', '--time-range', nargs=2, default=(0.,0.), type=float, help='Ctimes written as "-t x y" over which to estimate obs eff')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()

    print(args.weight)

    if args.time_range[0] == 0.:
        # Start with end of last day
        end_datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        end_datetime = end_datetime.replace(day=end_datetime.day - 1)
        end_datetime = end_datetime.replace(hour=23)
        end_datetime = end_datetime.replace(minute=59)
        end_datetime = end_datetime.replace(second=59)

        if args.range == 'm':
            start_datetime = end_datetime - datetime.timedelta(days=30)
        elif args.range == 'w':
            start_datetime = end_datetime - datetime.timedelta(days=7)
        elif args.range == 'd':
            start_datetime = end_datetime - datetime.timedelta(hours=24)
        
        start_ctime = start_datetime.timestamp()
        end_ctime = end_datetime.timestamp()
    else:
        start_ctime, end_ctime = args.time_range

    print(start_ctime, end_ctime)

    if not args.daily:
        obs_out = get_obs_results(args.platform, start_ctime, end_ctime)
        times, _ = find_fractions(obs_out, weight_bool=args.weight, debug=args.debug)
        print_fractions(times, span=end_ctime-start_ctime)
        
    else:
        start_dt = datetime.datetime.fromtimestamp(start_ctime)
        end_dt = datetime.datetime.fromtimestamp(end_ctime)
        datediff = end_dt - start_dt
        numdays = datediff.days

        outputs = []
        starts = []
        weights = []
        
        for d in range(numdays+1):
            istart = start_dt + datetime.timedelta(days=d)
            st_label = istart.date() + datetime.timedelta(hours=1)
            starts.append(st_label.isoformat())
            
            iend = start_dt + datetime.timedelta(days=d+1)

            #print(d, istart.timestamp(), iend.timestamp())
            
            obs_out = get_obs_results(args.platform, istart.timestamp(), iend.timestamp())
            times, w = find_fractions(obs_out, weight_bool=args.weight, debug=args.debug)
            outputs.append(times)
            weights.append(np.mean(w))

        #print(starts)
        outputs = np.array(outputs)
        #print(outputs)
        outputs *= 1./(3600.*24.)

        total_cmb_obs_frac = outputs[:,1]
        total_planet_obs_frac = outputs[:,0]
        total_sky_obs_frac = np.sum(outputs[:,:2], axis=1)
        total_obs_frac = np.sum(outputs[:,:3], axis=1)

        if not args.plot:
            for i in range(total_cmb_obs_frac.size):
                print(f'For day {i}:')
                print_fractions([total_planet_obs_frac[i], total_cmb_obs_frac[i]], span=1.)
                print('\n')

        if not args.weight:
            if args.plot:
                figout = plot_obs_eff(starts, [total_cmb_obs_frac, total_planet_obs_frac], labels=['CMB', 'planet'], weight_bool=False)

        else:
            if args.plot:
                figout = plot_obs_eff(starts, [total_cmb_obs_frac, total_planet_obs_frac], labels=['CMB\nUFM weighted', 'planet\nUFM weighted'], weight_bool=True, weights=weights)

        if args.plot:
            plt.savefig(args.plot_output_path)
            plt.close()

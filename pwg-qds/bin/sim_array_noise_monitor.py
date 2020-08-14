# Largely based on 20200514_FCT_Software_Example.ipynb from the pwg-fct
import sys
import numpy as np

from so3g.proj import Ranges, RangesMatrix

import scipy as sp
from scipy import stats

from sotodlib import core
import sotodlib.io.load as so_data_load
from sotodlib.core import FlagManager

import sotodlib.flags as flags
import sotodlib.sim_flags as sim_flags

import sotodlib.tod_ops.filters as filters
from sotodlib.tod_ops import fourier_filter, rfft

import qds

monitor = qds.Monitor('localhost', 56777, 'qds')

context = core.Context('pipe_s0001_v2.yaml')
observations = context.obsfiledb.get_obs()
print('Found {} Observations'.format(len(observations)))
o_list = range(len(observations)) # all observations
#o_list = [99] #np.random.randint(len(observations))

#for o in range(100,140):
for o in o_list:
    obs_id = observations[o]
    print('Looking at observation #{} named {}'.format(o,obs_id))
    
    c = context.obsfiledb.conn.execute('select distinct DS.name, DS.det from detsets DS '
                            'join files on DS.name=files.detset '
                            'where obs_id=?', (obs_id,))
    dets_in_obs = [tuple(r) for r in c.fetchall()]
    wafers = np.unique([x[0] for x in dets_in_obs])
    
    print('There are {} detectors on {} wafers in this observation'.format(len(dets_in_obs), len(wafers)))
    ## if you want to load a random wafer
    # w_idx = np.random.randint(len(wafers))
    # wafer_name = wafers[w_idx]
    
    for wafer in wafers:
        # Check calculation completed for this wafer
        check_tags = {'wafer': wafer} 
        if monitor.check('white_noise_level', obs_id, check_tags):
            continue

        # Process Obs+Wafer
        # Build detector list for this wafer
        det_list = []
        for det in dets_in_obs:
            if det[0] == wafer:
                det_list.append(det[1])
        print('{} detectors on this wafer'.format(len(det_list)))

        tod = so_data_load.load_observation(context.obsfiledb, obs_id, dets=det_list )

        print('This observation is {} minutes long. Has {} detectors and {} samples'.format(round((tod.timestamps[-1]-tod.timestamps[0])/60.,2),
                                                                              tod.dets.count, tod.samps.count))

        print('This TOD AxisManager has Axes: ')
        for k in tod._axes:
            print('\t{} with {} entries'.format(tod[k].name, tod[k].count ) )

        print('This TOD  AxisManager has fields : [axes]')
        for k in tod._fields:
            print('\t{} : {}'.format(k, tod._assignments[k]) )
            if type(tod._fields[k]) is core.AxisManager:
                for kk in tod[k]._fields:
                    print('\t\t {} : {}'.format(kk, tod[k]._assignments[kk] ))

        # Compute the FFT and detector white noise levels
        ffts, freqs = rfft(tod)
        ## This is using these default options:
        # detrend='linear', resize='zero_pad', window=np.hanning,
        # axis_name='samps', signal_name='signal'

        tsamp = np.median(np.diff(tod.timestamps))
        norm_fact = (1.0/tsamp)*np.sum(np.abs(np.hanning(tod.samps.count))**2)

        fmsk = freqs > 10
        det_white_noise = 1e6*np.median(np.sqrt(np.abs(ffts[:,fmsk])**2/norm_fact), axis=1)

        # Publish to monitor
        timestamps = np.ones(len(det_white_noise))*tod.timestamps[0]
        base_tags = {'telescope': 'LAT', 'wafer': wafer}
        tag_list = []
        for det in det_list:
            det_tag = dict(base_tags)
            det_tag['detector'] = det
            tag_list.append(det_tag)
        log_tags = {'observation': obs_id, 'wafer': wafer}
        monitor.record('white_noise_level', det_white_noise, timestamps, tag_list, 'detector_stats', log_tags=log_tags)
        monitor.write()


## push all sims to db!

# notes
    # we want to push det_white_noise to a monitor
    # I want to get to info that will form our tags in influx, i.e. 150, 90 GHz,
    # etc., but I don't know how to access that information
    
    # Alright, so I have the time stamp for pushing all the det_white_noise numbers in tod.timestamps[0]
    # let's just insert with a single tag for telescope=LAT, and one for detector=an index of the fft array

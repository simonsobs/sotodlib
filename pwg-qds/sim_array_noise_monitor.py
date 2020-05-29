# Largely based on 20200514_FCT_Software_Example.ipynb from the pwg-fct
import numpy as np
#import matplotlib.pyplot as plt

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

context = core.Context('pipe_s0001_v2.yaml')
observations = context.obsfiledb.get_obs()
print('Found {} Observations'.format(len(observations)))
o = 138 #np.random.randint(len(observations))
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
wafer_name = 'MFF2_wafer_17'

det_list = []
for det in dets_in_obs:
    if det[0] == wafer_name:
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

## looking at data
## fig = plt.figure(figsize=(8,3.5))
## plt.subplot(1,2,1)
## plt.plot( tod.timestamps-tod.timestamps[0], tod.boresight.az)
## plt.xlabel('Time (s)')
## plt.ylabel('Azimuth (rad)');
## plt.xlim(0, tod.timestamps[-1]-tod.timestamps[0])
## plt.subplot(1,2,2)
## for i in range(10):
##     plt.plot( tod.timestamps-tod.timestamps[0], tod.signal[i] )
##     
## plt.xlabel('Time (s)')
## plt.xlim(0, tod.timestamps[-1]-tod.timestamps[0])
## plt.ylabel('Signal (Kelvin)');
## fig.tight_layout()

## # scan rate?
## flags.get_turnaround_flags(tod, merge=True, name='turnarounds');
## print( tod.flags.turnarounds )
## 
## tmsk = tod.flags.turnarounds.mask()
## scan_rate = np.median( np.abs(np.diff(tod.boresight.az[~tmsk]))) / np.median(np.diff(tod.timestamps))
## print( 'The scan rate is {} deg / s'.format(round(np.degrees(scan_rate),3) ))
## 
## turn = np.where( np.diff(tod.timestamps[tmsk]) > 0.005 )[0]
## turn_time = np.median( np.diff(tod.timestamps[tmsk][turn]))
## print('We turn around every {} s'.format(round(turn_time,3)))

#fig=plt.figure(figsize=(4,3.5))
#nslice = 6000
#plt.plot( tod.timestamps[:nslice], tod.boresight.az[:nslice])
#plt.plot( tod.timestamps[:nslice][tmsk[:nslice]], tod.boresight.az[:nslice][tmsk[:nslice]], 'o')
#plt.xlabel('Time (s)');
#plt.ylabel('Azimuth (rad)')
#fig.tight_layout()

# FFT, the part we care about here
ffts, freqs = rfft(tod)
## This is using these default options:
# detrend='linear', resize='zero_pad', window=np.hanning, 
# axis_name='samps', signal_name='signal'

tsamp = np.median(np.diff(tod.timestamps))
norm_fact = (1.0/tsamp)*np.sum(np.abs(np.hanning(tod.samps.count))**2)

fmsk = freqs > 10
det_white_noise = 1e6*np.median(np.sqrt(np.abs(ffts[:,fmsk])**2/norm_fact), axis=1)

# we want to push det_white_noise to a monitor
# I want to get to info that will form our tags in influx, i.e. 150, 90 GHz,
# etc., but I don't know how to access that information

# Alright, so I have the time stamp for pushing all the det_white_noise numbers in tod.timestamps[0]
# let's just insert with a single tag for telescope=LAT, and one for detector=an index of the fft array
from influxdb import InfluxDBClient

client = InfluxDBClient(host='localhost', port=56777)
db_list = client.get_list_database()
db_names = [x['name'] for x in db_list]
if 'qds' not in db_names:
    print("qds DB doesn't exist, creating DB")
    client.create_database('qds')
client.switch_database('qds')

#time = tod.timetamps
#wafer_name = 'MFF2_wafer_17'

time_ns = int(tod.timestamps[0]*1e9)
payload = []
for index, noise_level in enumerate(det_white_noise):
    influxdata = f'detector_stats,wafer={wafer_name},telescope=LAT,detector={index} white_noise_level={noise_level} {time_ns}'
    payload.append(influxdata)

client.write_points(payload, protocol='line')

# TODO
## Generalize so that we can compute and upload this for more observations/wafers/files
## track locally what we've computed and uploaded
## push all sims to db!

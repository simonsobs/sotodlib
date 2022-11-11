#!/usr/bin/env python3

# Generates a G3 file containing simulated telescope pointing data
# in the "SO HK" format.

import numpy as np
from spt3g import core
from so3g import hk

# Define some telescope scanning patterns

def scanwave(x0, v_scan, dt, turntime, halfscan, start=False, end=False):
    """
    Generate a single scan waveform
    """
    hscantime = halfscan / v_scan  # scan time in each direction

    v = v_scan * np.ones(int(hscantime/dt))
    v = np.append(v, np.linspace(v_scan, -v_scan, int(turntime/dt)))
    v = np.append(v, -v_scan * np.ones(2*int(hscantime/dt)))
    v = np.append(v, np.linspace(-v_scan, v_scan, int(turntime/dt)))
    v = np.append(v, v_scan * np.ones(int(hscantime/dt)))

    # Accelerate from 0 velocity
    if start:
        v = np.append(np.linspace(0, v_scan, int(turntime/2/dt)), v[int(turntime/2/dt/2):])
    # Decelerate to 0 velocity
    if end:
        v = np.append(v[:-int(turntime/2/dt/2)], np.linspace(v_scan, 0, int(turntime/2/dt)))

    x = x0 + np.cumsum(v) * dt
    return x

def slew(v_slew, dt, start_pos, end_pos):
    """
    Generate a slew movement
    """
    v_slew *= np.sign(end_pos - start_pos)   # need to account for direction
    ntotal = np.abs(end_pos - start_pos)/dt  # total samples during slew

    v = v_slew * np.ones(int(ntotal))
    x = start_pos + np.cumsum(v) * dt
    return x

########################################
# Generate the simulated scan pattern
########################################

# Initial telescope location
az0 = 0.
el0 = 0.

v_az = 1.0  # degrees/second
dt = 0.005  # sampling cadence
turntime = 0.5     # turnaround time
halfscan = 9.75    # degrees (excl turnaround)

# Part 1: Do 10 scans
az = scanwave(az0, v_az, dt, turntime, halfscan, start=True)
az = np.append(az, np.concatenate([scanwave(az0, v_az, dt, turntime, halfscan) for i in range(8)]))
az = np.append(az, scanwave(az0, v_az, dt, turntime, halfscan, end=True))

# Part 2: Stop for 2 minutes
az = np.append(az, np.zeros(int(120./dt)))

# Part 3: Do another 10 scans
az = np.append(az, scanwave(az0, v_az, dt, turntime, halfscan, start=True))
az = np.append(az, np.concatenate([scanwave(az0, v_az, dt, turntime, halfscan) for i in range(8)]))
az = np.append(az, scanwave(az0, v_az, dt, turntime, halfscan, end=True))

# Part 4: Stop scanning and slew to new az/el
az1 = -10. # new az
el1 = 5.   # new el

az = np.append(az, np.zeros(int(60./dt)))
el = el0 * np.ones(len(az))

az = np.append(az, slew(v_az, dt, az0, az1))
el = np.append(el, slew(v_az, dt, el0, el1))

az = np.append(az, az1*np.ones(int(60./dt)))

# Part 5: Do another 10 scans at new az/el
az = np.append(az, scanwave(az1, v_az, dt, turntime, halfscan, start=True))
az = np.append(az, np.concatenate([scanwave(az1, v_az, dt, turntime, halfscan) for i in range(8)]))
az = np.append(az, scanwave(az1, v_az, dt, turntime, halfscan, end=True))

el = np.append(el, el1 * np.ones(len(az)-len(el)))

########################################
# Organize scan data into frames and
# write to G3 file.
########################################

# Start a "Session" to help generate template frames.
session = hk.HKSessionHelper(hkagg_version=2)

# Create an output file and write the initial "session" frame.
writer = core.G3Writer('simset1_hkdata.g3')
writer.Process(session.session_frame())

# Create a new data "provider".
prov_id = session.add_provider('observatory.acu1.feeds.acu_udp_stream')

# Whenever there is a change in the active "providers", write a
# "status" frame.
writer.Process(session.status_frame())

# Parameters
frame_time = 1630468800  # 2021-09-01-00:00:00
n = 10000  # length of each frame (samples)
start = 0

# Main loop
for i in range(len(az)//n + 1):
    end = min(start+n, len(az))
    sli = slice(start, end)
    # Vector of unix timestamps
    t = frame_time + dt * np.arange(min(n, end-start))

    # Construct a "block"
    block = core.G3TimesampleMap()
    block.times = core.G3VectorTime([core.G3Time(_t * core.G3Units.s) for _t in t])
    block['Azimuth_Corrected'] = core.G3VectorDouble(az[sli])
    block['Elevation_Corrected'] = core.G3VectorDouble(el[sli])

    # Create an output data frame template associated with this provider.
    frame = session.data_frame(prov_id)

    # Add the block and block name to the frame, and write it.
    frame['block_names'].append('ACU_position')
    frame['blocks'].append(block)
    writer.Process(frame)

    # For next iteration.
    start += n
    frame_time += n * dt

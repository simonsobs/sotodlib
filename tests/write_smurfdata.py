#!/usr/bin/env python3

import numpy as np
import so3g
from spt3g import core
from sosmurf.SessionManager import SessionManager

time = 1630468800  # 2021-09-01-00:00:00

# Start a "Session" to help generate template frames
session = SessionManager(stream_id="crate1slot3")
session.session_id = int(time)

# Create an output file and write the initial frame
writer = core.G3Writer('simset1_smurfdata01.g3')
writer.Process(session.start_session())

# Create a status frame
writer.Process(session.status_frame())

# Sampling parameters
frame_time = time
dt = 0.005  # seconds
n = 1000    # number of samples
Nchan = 10  # number of channels

# Placeholder data: a linear sweep from 0 to 1
sweep = np.linspace(0., 1., n)

# Create a scan frame
for i in range(10):
    # Vector of unix timestamps
    t = frame_time + dt * np.arange(n)

    # Construct the G3SuperTimestream containing the data
    data = so3g.G3SuperTimestream()
    data.times = core.G3VectorTime([core.G3Time(_t * core.G3Units.s) for _t in t])
    data.names = [f'r{c:04d}' for c in range(Nchan)]
    data.quanta = np.ones(Nchan, dtype=np.double)
    data.data = np.vstack([sweep for i in range(Nchan)])

    # Create an output data frame and insert the data
    frame = core.G3Frame(core.G3FrameType.Scan)
    frame['data'] = data

    # Write the frame
    frame['timing_paradigm'] = "Low Precision"
    writer.Process(session(frame)[0])

    # For next iteration
    frame_time += n * dt

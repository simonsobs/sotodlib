#!/usr/bin/env python3

from spt3g import core
import numpy as np
import so3g

def generate_hk_frame(t):
    session = so3g.hk.HKSessionHelper(hkagg_version=2)
    prov_id = session.add_provider("test")

    block = core.G3TimesampleMap()
    block.times = core.G3VectorTime([core.G3Time(_t * core.G3Units.s) for _t in t])
    block['Azimuth_Corrected'] = core.G3VectorDouble(np.zeros(len(t)))
    block['Elevation_Corrected'] = core.G3VectorDouble(np.zeros(len(t)))
    frame = session.data_frame(prov_id)
    frame['block_names'].append('ACU_position')
    frame['blocks'].append(block)
    return frame

def generate_smurf_frame(t):
    frame = core.G3Frame(core.G3FrameType.Scan)
    data = so3g.G3SuperTimestream()
    data.times = core.G3VectorTime([core.G3Time(_t * core.G3Units.s) for _t in t])
    data.names = ['r0000']
    data.quanta = np.ones(len(data.names), dtype=np.double)
    data.data = np.ones((len(data.names), len(data.times)))
    frame['data'] = data
    return frame

def test_hk_gaps():
    import bookbinder.bookbinder as bb

    # Create 3 HK frames, with gaps in time in between them
    start_time = 1630470250.75000000
    n = 10
    dt = 0.005
    t0 = start_time + dt * np.arange(n)
    t1 = t0 + 2*n*dt
    t2 = t1 + 1.5*n*dt

    acu_frames = []
    for t in [t0, t1, t2]:
        acu_frames.append(generate_hk_frame(t))

    B = bb.Bookbinder(smurf_files=None, book_id='test')
    smurf_frames     = [generate_smurf_frame(start_time + 0.00437812 + np.arange(50)*dt),
                        generate_smurf_frame(start_time + 0.05437812 + np.arange(25)*dt)]
    expected_outputs = [core.G3VectorDouble(np.concatenate((np.zeros(9), np.full(11, B.frameproc.FLAGGED_SAMPLE_VALUE),
                                                            np.zeros(9), np.full(6, B.frameproc.FLAGGED_SAMPLE_VALUE),
                                                            np.zeros(9), np.full(6, np.nan)))),
                        core.G3VectorDouble(np.concatenate((np.full(10, B.frameproc.FLAGGED_SAMPLE_VALUE),
                                                            np.zeros(9), np.full(6, B.frameproc.FLAGGED_SAMPLE_VALUE))))]

    ##################################################
    # Run test TWICE to test two cases
    # Case 1 : Single SMuRF frame covers both gaps
    # Case 2 : SMuRF frame begins in 1st gap and ends
    #          ends in 2nd gap
    ##################################################
    for smurf_frame, expected_output in zip(smurf_frames, expected_outputs):
        B = bb.Bookbinder(smurf_files=None, book_id='test')
        B._hk_files = []
        B.hk_iter = acu_frames
        B.process_HK_files()

        assert len(B.frameproc._hk_gaps) == 2
        assert B.frameproc._hk_gaps[0] == (163047025079500000, 163047025085000000)
        assert B.frameproc._hk_gaps[1] == (163047025089500000, 163047025092500000)

        B.frameproc(smurf_frame)
        B.frameproc.flush_time = core.G3Time(1e18)
        B.frameproc.hkbundle.data['Azimuth_Velocity'] = np.append(B.frameproc.hkbundle.data['Azimuth_Velocity'], 0)
        B.frameproc.hkbundle.data['Elevation_Velocity'] = np.append(B.frameproc.hkbundle.data['Elevation_Velocity'], 0)
        output = B.frameproc.flush()

        assert len(output) == 1
        assert output[0]['state'] == 1
        np.testing.assert_array_equal(output[0]['ancil']['az_enc'], expected_output, verbose=True)

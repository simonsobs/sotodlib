#!/usr/bin/env python3

import numpy as np
import so3g  # noqa: F401
from spt3g import core


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
    if not isinstance(t, core.G3VectorTime):
        t = core.G3VectorTime([core.G3Time(_t * core.G3Units.s) for _t in t])

    frame = core.G3Frame(core.G3FrameType.Scan)
    frame['data'] = so3g.G3SuperTimestream(['r0000', 'r0001'], t, np.ones((2, len(t)), dtype=np.int32))
    return frame

def test_replace_times_and_trim_frame():
    import sotodlib.io.bookbinder as bb

    F = bb.FrameProcessor()

    ##################################################
    # Test 1a: When the primary timestream is not
    #          available, it should default to the
    #          times recorded in the .times field
    ##################################################
    t = np.full(10, 167330757642916600) + np.array([0, 499600, 1001200, 1499500, 2000600,
                                            2501300, 3001400, 3503100, 4001100, 4499000])
    frame = generate_smurf_frame(core.G3VectorTime(t))
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, t)

    ##################################################
    # Test 1b: When the timing system is on but the
    #          timing counters are absent, an
    #          exception should be raised
    ##################################################
    F.timing_system = True
    np.testing.assert_raises(bb.TimingSystemError, F.replace_times_and_trim_frame, frame)
    c0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c1 = c0
    c2 = c0
    frame['primary'] = so3g.G3SuperTimestream(["Counter0", "Counter1", "Counter2"],
                                core.G3VectorTime(t), np.vstack((c0, c1, c2)))
    np.testing.assert_raises(bb.TimingSystemError, F.replace_times_and_trim_frame, frame)

    ##################################################
    # Test 1c: When the timing system is on and the
    #          timing counters are present, calculate
    #          the true timestamps from the counters
    ##################################################
    c0 = [205909, 208309, 210709, 213109, 215509, 217909, 220309, 222709, 225109, 227509]
    c1 = [4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
          4294967295, 4294967295, 4294967295]
    c2 = [4476024116700374110, 4476024116705374055, 4476024116710373999, 4476024116715373944,
          4476024116720373889, 4476024116725373834, 4476024116730373779, 4476024116735373723,
          4476024116740373668, 4476024116745373613]
    true_timestamps = [167330757642897696, 167330757643397728, 167330757643897696, 167330757644397696,
          167330757644897696, 167330757645397696, 167330757645897696, 167330757646397696,
          167330757646897696, 167330757647397696]
    frame['primary'].data = np.vstack((c0, c1, c2))
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, true_timestamps)

    ##################################################
    # Test 2: Trim the samples outside start/end times
    ##################################################
    # Start/end times in middle of frame
    F.BOOK_START_TIME = core.G3Time(true_timestamps[1] + 1)
    F.BOOK_END_TIME   = None
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, true_timestamps[2:])
    F.BOOK_END_TIME   = core.G3Time(true_timestamps[-1] - 1)
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, true_timestamps[2:-1])

    # Start time after end of frame
    F.BOOK_START_TIME = core.G3Time(true_timestamps[-1] + 1)
    F.BOOK_END_TIME   = None
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, [])
    F.BOOK_END_TIME   = core.G3Time(true_timestamps[-1] + 2)
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, [])

    # End time before start of frame
    F.BOOK_START_TIME = None
    F.BOOK_END_TIME   = core.G3Time(true_timestamps[0] - 1)
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, [])
    F.BOOK_START_TIME = core.G3Time(true_timestamps[0] - 2)
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, [])

    # Frame contained well within start/end times
    F.BOOK_START_TIME = core.G3Time(true_timestamps[0] - 1)
    F.BOOK_END_TIME   = core.G3Time(true_timestamps[-1] + 1)
    np.testing.assert_array_equal(F.replace_times_and_trim_frame(frame)['data'].times, true_timestamps)

def test_hk_gaps():
    import sotodlib.io.bookbinder as bb

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

def test_smurf_gaps():
    import sotodlib.io.bookbinder as bb

    # Create 3 HK frames, with NO gaps in time in between them
    start_time = 1630470250.75000000
    n = 10
    dt = 0.005
    t0 = start_time + dt * np.arange(n)
    t1 = t0 + n*dt
    t2 = t1 + n*dt

    acu_frames = []
    for t in [t0, t1, t2]:
        acu_frames.append(generate_hk_frame(t))

    # Create 3 SMuRF frames, with gaps in time between them
    Nsmurf = 20 # Number of SMuRF samples
    ts0 = start_time + 0.00437812 + np.arange(Nsmurf)*dt
    ts1 = ts0 + Nsmurf*dt
    ts2 = ts1 + Nsmurf*dt
    timestamps = np.unique([ts0, ts1, ts2])
    smurf_frames = []
    for ts in [ts0, ts1[4:], ts2[5:]]:
        smurf_frames.append(generate_smurf_frame(ts))

    ##################################################
    # Run the test TWICE to test the two cases
    # Case 1 : (No timestamps available) estimate
    #          missing samples using approx sample
    #          interval
    # Case 2 : (Timestamps given) find missing samples
    #          using list of timestamps
    ##################################################
    for tlist in [None, (timestamps * core.G3Units.s).astype(int)]:
        B = bb.Bookbinder(smurf_files=None, book_id='test')
        B._hk_files = []
        B.hk_iter = acu_frames
        B.process_HK_files()

        B.frameproc._smurf_timestamps = tlist

        for s in smurf_frames:
            B.frameproc(s)
        B.frameproc.flush_time = core.G3Time(1e18)
        B.frameproc.hkbundle.data['Azimuth_Velocity'] = np.append(B.frameproc.hkbundle.data['Azimuth_Velocity'], 0)
        B.frameproc.hkbundle.data['Elevation_Velocity'] = np.append(B.frameproc.hkbundle.data['Elevation_Velocity'], 0)
        output = B.frameproc.flush()

        assert len(output) == 1
        assert output[0]['state'] == 1

        expected_output = core.G3VectorDouble(np.concatenate((np.ones(20),
                                                              np.full(4, B.frameproc.FLAGGED_SAMPLE_VALUE),
                                                              np.ones(16),
                                                              np.full(5, B.frameproc.FLAGGED_SAMPLE_VALUE),
                                                              np.ones(15))))
        np.testing.assert_array_equal(output[0]['signal'].data[0], expected_output, verbose=True)

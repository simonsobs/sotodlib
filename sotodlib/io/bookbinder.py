#!/usr/bin/env python3

import os.path as op, os
import so3g  # noqa: F401
from spt3g import core
import numpy as np
import itertools
import yaml
from scipy.interpolate import interp1d
from .load_smurf import split_ts_bits


class _HKBundle():
    """
    Buffer for Housekeeping data. Use add() to add data and rebundle() to output
    data up to but not including flush_time.
    """
    def __init__(self):
        self.times = []
        self.data = None

        self.azimuth_events = []
        self.azimuth_starts = []
        self.azimuth_stops = []
        self.elevation_events = []
        self.elevation_starts = []
        self.elevation_stops = []

    def ready(self):
        """
        Returns True if a zero-crossing event exists
        """
        return len(self.azimuth_events) > 0

    def add(self, b):
        """
        Cache the data from Housekeeping G3TimesampleMap b
        """
        self.times.extend(b.times)
        if self.data is None:
            self.data = {c: [] for c in b.keys()}
        for c in b.keys():
            self.data[c].extend(b[c])

    def rebundle(self, flush_time):
        """
        Return the buffered Housekeeping G3TimesampleMap with all samples timestamped up to but
        not including flush_time. Keep the samples from flush_time onward in buffer.

        If there are no samples to bundle, None is returned.

        Parameters
        ----------
        flush_time : G3Time
            Buffered data up to (but not including) this time will be output, the rest kept

        Returns
        -------
        output : G3SuperTimestream or None
            Output signal timestream
        """
        if len(self.times) == 0:
            return None

        output = core.G3TimesampleMap()
        output.times = core.G3VectorTime([t for t in self.times if t < flush_time])
        self.times = [t for t in self.times if t >= flush_time]

        for c in self.data.keys():
            output[c] = core.G3VectorDouble(np.array(self.data[c][:len(output.times)]))
            if len(output.times) < len(self.data[c]):
                self.data[c] = self.data[c][len(output.times):]
            else:
                self.data[c] = np.array([])

        return output

class _SmurfBundle():
    """
    Buffer for SMuRF data. Use add() to add data and rebundle() to output data
    up to but not including flush_time.
    """
    def __init__(self):
        self.times = []
        self.signal = None
        self.biases = None
        self.primary = None
        self.dtypes = {'data': np.int32, 'tes_biases': np.int32, 'primary': np.int64}

    def ready(self, flush_time):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1] >= flush_time

    def add(self, f):
        """
        Cache the data from input SMuRF frame f

        Parameters
        ----------
        f : G3Frame
            Input SMuRF frame to be buffered
        """
        self.times.extend(f['data'].times)

        if self.signal is None:
            self.signal = {'names': f['data'].names, 'times': [], 'data': []}
        assert (np.asarray(f['data'].names) == np.asarray(self.signal['names'])).all() # there's a bug where G3VectorString(x)==G3VectorString(x) returns False
        assert f['data'].data.dtype == self.dtypes['data']
        self.signal['times'].append(f['data'].times)
        self.signal['data'].append(f['data'].data)

        if 'tes_biases' in f.keys():
            if self.biases is None:
                self.biases = {'names': f['tes_biases'].names, 'times': [], 'data': []}
            assert (np.asarray(f['tes_biases'].names) == np.asarray(self.biases['names'])).all()
            assert f['tes_biases'].data.dtype == self.dtypes['tes_biases']
            self.biases['times'].append(f['tes_biases'].times)
            self.biases['data'].append(f['tes_biases'].data)


        if 'primary' in f.keys():
            if self.primary is None:
                self.primary = {'names': f['primary'].names, 'times': [], 'data': []}
            assert (np.asarray(f['primary'].names) == np.asarray(self.primary['names'])).all()
            assert f['primary'].data.dtype == self.dtypes['primary']
            self.primary['times'].append(f['primary'].times)
            self.primary['data'].append(f['primary'].data)

    def rebundle(self, flush_time):
        """
        Return the buffered SMuRF G3SuperTimestreams with all samples timestamped up to but
        not including flush_time. Keep the samples from flush_time onward in buffer.

        If there are no samples to bundle, None is returned.

        Parameters
        ----------
        flush_time : G3Time
            Buffered data up to (but not including) this time will be output, the rest kept

        Returns
        -------
        signalout : G3SuperTimestream
            Output signal timestream
        biasout : G3SuperTimestream or None
            Output TES biases timestream
        primout : G3SuperTimestream or None
            Output primary timestream
        """

        # Timestamps to be output and kept in buffer, respectively
        # (assumes signal, bias, and primary have same timestamps, which is a reasonable assumption)
        tout       = [t for t in self.times if t < flush_time]
        self.times = [t for t in self.times if t >= flush_time]

        # Since G3SuperTimestreams cannot be shortened, the data must be copied to
        # a new instance for rebundling of the buffer
        signalstack = np.hstack(self.signal['data'])
        signalout = so3g.G3SuperTimestream(self.signal['names'], tout, signalstack[:,:len(tout)])
        self.signal = {'names': self.signal['names'], 'times': [self.times], 'data': [signalstack[:,len(tout):]]}

        if self.biases is not None:
            biasstack = np.hstack(self.biases['data'])
            biasout = so3g.G3SuperTimestream(self.biases['names'], tout, biasstack[:,:len(tout)])
            self.biases = {'names': self.biases['names'], 'times': [self.times], 'data': [biasstack[:,len(tout):]]}
        else:
            biasout = None

        if self.primary is not None:
            primstack = np.hstack(self.primary['data'])
            primout = so3g.G3SuperTimestream(self.primary['names'], tout, primstack[:,:len(tout)])
            self.primout = {'names': self.primary['names'], 'times': [self.times], 'data': [primstack[:,len(tout):]]}
        else:
            primout = None

        return signalout, biasout, primout


class FrameProcessor(object):
    """
    Module to process HK and SMuRF frames, adding to and rebundling from their buffers to
    produce frames for the output Book.

    Parameters
    ----------
    start_time : G3Time, optional
        Start time for the output Book
    end_time : G3Time, optional
        End time for the output Book
    smurf_timestamps : list, optional
        Externally provided list of timestamps, used to check for missing samples in
        current timestream
    timing_system : bool
        Whether the timing system was on during observation. If true, will attempt to use
        timing counters to calculate correct timestamps; if false, will fall back to using
        smurf-streamer recorded timestamps in the timestreams
    flush_time : G3Time
        Buffered data up to (but not including) this time will be output, the rest kept
    maxlength : int
        Maximum allowed length (in samples) of an output frame
    """
    def __init__(self, start_time=None, end_time=None, smurf_timestamps=None,
                 timing_system=False, **config):
        self.hkbundle = None
        self.smbundle = None
        self.flush_time = None
        self.maxlength = config.get("maxlength", 10000)
        self.FLAGGED_SAMPLE_VALUE = config.get("flagged_sample_value", -1)
        self.current_state = 0  # default to scan state
        self.GAP_THRESHOLD = config.get("gap_threshold", 0.05)
        self.BOOK_START_TIME = start_time
        self.BOOK_END_TIME = end_time
        self.timing_system = timing_system
        self._frame_splits = []
        self._hk_gaps = []
        self._prev_smurf_sample = None
        self._smurf_timestamps = smurf_timestamps
        self._readout_ids = config.get("readout_ids", None)

        # Ensure start and end times have the correct type
        if self.BOOK_START_TIME is not None:
            self.BOOK_START_TIME = core.G3Time(self.BOOK_START_TIME)
        if self.BOOK_END_TIME is not None:
            self.BOOK_END_TIME = core.G3Time(self.BOOK_END_TIME)

    def ready(self):
        """
        Check if criterion passed in HK (for now, sign change in Az scan velocity data)
        """
        return self.hkbundle.ready() if (self.hkbundle is not None) else False

    def locate_crossing_events(self, t, dy=0.001, min_gap=200):
        """
        Locate places where a noisy vector t changes sign: +ve, -ve, and 0. To handle noise,
        "zero" is defined to be a threshold region from -dy to +dy. Thus t is said to have a
        zero-crossing if it fully crosses the threshold region, with the actual crossing
        approximated as halfway between entering and exiting the region. Entrances and exits
        without a crossing are also identified (as "stops" and "starts" respectively).

        Parameters
        ----------
        dy : float, optional
            Amplitude of threshold region
        min_gap : int, optional
            Length of gap (in samples) longer than which events are considered separate

        Returns
        -------
        events : list
            Full list containing all zero-crossings, starts, and stops
        starts : list
            List of places where t exits the threshold region (subset of events)
        stops : list
            List of places where t enters the threshold region (subset of events)
        """
        tmin = np.min(t)
        tmax = np.max(t)

        if len(t) == 0:
            return [], [], []

        if np.sign(tmax) == np.sign(tmin):
            return [], [], []

        # If the data does not entirely cross the threshold region,
        # do not consider it a sign change
        if tmin > -dy and tmax < dy:
            return [], [], []

        # Find where the data crosses the lower and upper boundaries of the threshold region
        c_lower = (np.where(np.sign(t[:-1]+dy) != np.sign(t[1:]+dy))[0] + 1)
        c_upper = (np.where(np.sign(t[:-1]-dy) != np.sign(t[1:]-dy))[0] + 1)

        # Classify each crossing as entering or exiting the threshold region
        c_lower_enter = []; c_lower_exit = []
        c_upper_enter = []; c_upper_exit = []

        # Noise handling:
        # If there are multiple crossings of the same boundary (upper or lower) in quick
        # succession (i.e., less than min_gap), it is mostly likely due to noise. In this case,
        # take the average of each group of crossings.
        if len(c_lower) > 0:
            spl = np.array_split(c_lower, np.where(np.ediff1d(c_lower) > min_gap)[0] + 1)
            c_lower = np.array([int(np.ceil(np.mean(s))) for s in spl])
            # Determine if this crossing is entering or exiting threshold region
            for i, s in enumerate(spl):
                if t[s[0]-1] < t[s[-1]+1]:
                    c_lower_enter.append(c_lower[i])
                else:
                    c_lower_exit.append(c_lower[i])
        if len(c_upper) > 0:
            spu = np.array_split(c_upper, np.where(np.ediff1d(c_upper) > min_gap)[0] + 1)
            c_upper = np.array([int(np.ceil(np.mean(s))) for s in spu])
            # Determine if this crossing is entering or exiting threshold region
            for i, s in enumerate(spu):
                if t[s[0]-1] < t[s[-1]+1]:
                    c_upper_exit.append(c_upper[i])
                else:
                    c_upper_enter.append(c_upper[i])

        starts = np.sort(np.concatenate((c_lower_exit, c_upper_exit)))
        stops = np.sort(np.concatenate((c_lower_enter, c_upper_enter)))
        events = np.sort(np.concatenate((starts, stops)))

        # Look for zero-crossings
        zc = []
        while len(c_lower) > 0 and len(c_upper) > 0:
            # Crossing from -ve to +ve
            if c_lower[0] < c_upper[0]:
                b = c_lower[c_lower < c_upper[0]]
                zc.append(int( np.ceil(np.mean([b[-1], c_upper[0]])) ))
                c_lower = c_lower[len(b):]
            # Crossing from +ve to -ve
            elif c_upper[0] < c_lower[0]:
                b = c_upper[c_upper < c_lower[0]]
                zc.append(int( np.ceil(np.mean([b[-1], c_lower[0]])) ))
                c_upper = c_upper[len(b):]

        # Replace all upper and lower crossings that contain a zero-crossing in between with the
        # zero-crossing itself, but ONLY if those three events happen in quick succession (i.e.,
        # shorter than min_gap). Otherwise, they are separate events -- there is likely a stop
        # state in between; in this case, do NOT perform the replacement.
        for z in zc:
            before_z = events[events < z]
            after_z  = events[events > z]
            if (after_z[0] - before_z[-1]) < min_gap:
                events = np.concatenate((before_z[:-1], [z], after_z[1:]))

        # If the last event is close to the end, a crossing of the upper or lower threshold
        # boundaries may or may not be a significant event -- there is not enough remaining data
        # to determine whether it is stopping or not. This will be clarified in the next iteration
        # of the loop, when more HK data will have been added. (Or if there is no more HK data to
        # follow, the loss in information by removing the last crossing is negligible.)
        # On the other hand, if the last event is a zero-crossing, then that is unambiguous and
        # should not be removed.
        if (len(t) - events[-1] < min_gap) and events[-1] not in zc:
            events = events[:-1]

        # A similar problem occurs when the first boundary-crossing event is close to the
        # beginning -- it could either be following a zero-crossing OR be entering/exiting a
        # stopped state. In this case, we use previous data to disambiguate the two cases: if the
        # telescope is currently in the scan state, a zero-crossing has just occurred so discard
        # the event; otherwise, the threshold crossing indicates a change of state and should NOT
        # be ignored.
        if events[0] < min_gap and self.current_state == 0:
            events = events[1:]

        return events, starts, stops

    def determine_state(self, v_az, v_el, dy=0.001):
        """
        Determine in which mode the telescope is operating:
        - Mode 0: Scanning
        - Mode 1: Stopped
        - Mode 2: Slewing

        Parameters
        ----------
        v_az : numpy.ndarray
            Azimuth velocity
        v_el : numpy.ndarray
            Elevation velocity
        dy : float, optional
            Amplitude of the threshold region
        """
        if np.min(v_el) < -dy or np.max(v_el) > dy:
            # If the el velocity exceeds the threshold region, the telescope is slewing
            state = 2
        elif np.min(v_az) < -dy or np.max(v_az) > dy:
            # If the az velocity (but not el velocity) exceeds the threshold region, the telescope
            # is scanning
            state = 0
        else:
            # If the velocities lie entirely in the threshold region, the telescope is stopped
            state = 1

        self.current_state = state

    def replace_times_and_trim_frame(self, f):
        """
        Replace the data timestamps with the most accurate available, then trim any samples
        occurring before the Book start time and after the Book end time.

        Parameters
        ----------
        f : G3Frame
            Input frame to be processed

        Returns
        -------
        trimmed_frame : G3Frame
            Processed frame
        """
        if self.BOOK_START_TIME is not None and self.BOOK_END_TIME is not None:
            assert self.BOOK_START_TIME <= self.BOOK_END_TIME

        # Calculate the timestamps for the current frame, if available, or default to the
        # recorded times in the 'data' field
        t = get_timestamps(f, self.timing_system)

        # Trim starting samples (if needed)
        if self.BOOK_START_TIME is not None and t[0] < self.BOOK_START_TIME:
            trimmed_frame_start = core.G3Frame(f.type)
            t = core.G3VectorTime([_t for _t in t if _t >= self.BOOK_START_TIME])
            for k in f.keys():
                if k in ['data', 'tes_biases', 'primary']:
                    trimmed_frame_start[k] = so3g.G3SuperTimestream(f[k].names, t, f[k].data[:,(len(f[k].times)-len(t)):])
                else:
                    trimmed_frame_start[k] = f[k]
            trimmed_frame = trimmed_frame_start
        else:
            for k in f.keys():
                if k in ['data', 'tes_biases', 'primary']:
                    f[k].times = t
            trimmed_frame = f

        if len(trimmed_frame['data'].times) == 0:
            return trimmed_frame

        # Trim ending samples (if needed)
        if self.BOOK_END_TIME is not None and t[-1] > self.BOOK_END_TIME:
            trimmed_frame_end = core.G3Frame(trimmed_frame.type)
            t = core.G3VectorTime([_t for _t in t if _t <= self.BOOK_END_TIME])
            for k in f.keys():
                if k in ['data', 'tes_biases', 'primary']:
                    trimmed_frame_end[k] = so3g.G3SuperTimestream(f[k].names, t, f[k].data[:,:len(t)])
                else:
                    trimmed_frame_end[k] = trimmed_frame[k]
            trimmed_frame = trimmed_frame_end

        return trimmed_frame

    def fill_in_missing_samples(self, data, flush_time, return_flags=False):
        """
        Locate and patch missing samples in data.

        If a list of timestamps is provided, use it to fill in the missing samples (note: the
        timestamps present in the input data must exactly match their counterparts in the list).
        If not, determine the missing samples using an estimate of the sample interval.

        Parameters
        ----------
        data : G3SuperTimestream
            Input data to be screened for missing samples
        flush_time : G3Time
            Upper time limit (exclusive) for data to be returned
        return_flags : bool, optional
            Whether to return the list of flags indicating filled-in samples

        Return
        ------
        dataout : G3SuperTimestream
            Data G3SuperTimestream with filled-in samples wherever gaps are detected
        flag_gap : G3VectorBool, optional
            Array of boolean flags indicating filled-in samples that should not be used for
            analysis. Set return_flags = True to return.
        """
        # There is no _prev_smurf_sample at the beginning of the book
        if self._prev_smurf_sample is None:
            prev_frame_end = self.BOOK_START_TIME.time - 1
        else:
            prev_frame_end = self._prev_smurf_sample.time
        # Define a strict upper limit (exclusive) up until which to look/interpolate for missing samples
        if self.BOOK_END_TIME is not None:
            end_time = min(flush_time.time, self.BOOK_END_TIME.time)
        elif self._smurf_timestamps is not None:
            end_time = min(flush_time.time, self._smurf_timestamps[-1].time + 1)
        else:
            end_time = min(flush_time.time, data.times[-1].time + 1)

        # Obtain list of filled-in timestamps, whether from provided list or estimate with fill_time_gaps
        if self._smurf_timestamps is not None:
            # trim the list of timestamps to the current frame
            m_ = np.logical_and(self._smurf_timestamps < end_time, self._smurf_timestamps > prev_frame_end)
            ts = self._smurf_timestamps[m_]
        else:
            # fallback when no timestamp is provided. We switch to estimate sample interval based on data
            assert data.times[0].time >= prev_frame_end
            # Use fill_time_gaps to fill in any missing samples in the middle, as well as
            # at the beginning (between previous frame and current frame)
            dt = np.median(np.diff(data.times))  # estimate sample interval
            if self._prev_smurf_sample is None:
                # Use estimated sample interval to fill backward to prev_frame_end = book start time - 1
                # since book start time does not usually line up with sample interval
                ts = np.append(np.flip(np.arange(data.times[0].time-dt, prev_frame_end, -dt)), data.times)
                ts = fill_time_gaps(ts)
            elif (data.times[0].time - prev_frame_end)/dt > 0.5:
                ts = fill_time_gaps(np.insert(data.times, 0, prev_frame_end))[1:]
            else:
                ts = fill_time_gaps(data.times)
            # Use estimated sample interval to fill until end_time. Not using fill_time_gaps because
            # end_time is externally determined and does not usually line up with sample interval
            ts = np.append(ts, np.arange(data.times[-1].time+dt, end_time, dt))
            # for consistency with smurf_timestamps
            ts /= core.G3Units.s

        # Create new G3SuperTimestream with filled-in samples
        if len(data.times) < len(ts):
            dt = np.median(np.diff(ts))
            i_missing, _ = find_missing_samples(ts, np.array(data.times)/core.G3Units.s, atol=dt/2)
            m = np.ones(len(ts), dtype=bool)
            m[i_missing] = False
            assert np.sum(m) == len(data.times)
            new_data = np.full((data.data.shape[0], len(ts)), self.FLAGGED_SAMPLE_VALUE)
            new_data[:,m] = data.data
            dataout = so3g.G3SuperTimestream(data.names, core.G3VectorTime(ts), new_data)
            flag_gap = core.G3VectorBool(~m)
            print("{} missing samples detected".format(np.sum(~m)))
        elif len(data.times) > len(ts):
            raise ValueError("There are more samples in the timestream than in list provided")
        else:
            dataout = data
            flag_gap = core.G3VectorBool(np.zeros(len(ts)))

        if return_flags:
            return dataout, flag_gap
        else:
            return dataout

    def flush(self, flush_time=None):
        """
        Produce frames for the output Book, up to but not including flush_time

        Parameters
        ----------
        flush_time : G3Time, optional
            Buffered data up to (but not including) this time will be output, the rest kept

        Returns
        -------
        output : list
            List containing output frames
        """
        if flush_time == None:
            flush_time = self.flush_time
        output = []

        smurf_data, smurf_bias, smurf_primary = self.smbundle.rebundle(flush_time)

        # Fill in missing samples and write data to frame
        f = core.G3Frame(core.G3FrameType.Scan)
        f['signal'], flag_smurfgap = self.fill_in_missing_samples(smurf_data, flush_time, return_flags=True)
        if smurf_bias is not None:
            f['tes_biases'] = self.fill_in_missing_samples(smurf_bias, flush_time)
        if smurf_primary is not None:
            f['primary'] = self.fill_in_missing_samples(smurf_primary, flush_time)
        # Update last sample
        self._prev_smurf_sample = smurf_data.times[-1]

        try:
            hk_data = self.hkbundle.rebundle(flush_time)
        except AttributeError:
            # No az/el encoder data found in HK files
            f['state'] = 3
        else:
            # Co-sampled (interpolated) az/el encoder data
            cosamp_az = np.interp(smurf_data.times, hk_data.times, hk_data['Azimuth_Corrected'], left=np.nan, right=np.nan)
            cosamp_el = np.interp(smurf_data.times, hk_data.times, hk_data['Elevation_Corrected'], left=np.nan, right=np.nan)

            # Flag any samples falling within a gap in HK data
            t = np.array([int(_t) for _t in smurf_data.times])
            flag_hkgap = np.zeros(len(smurf_data.times))
            for gap_start, gap_end in self._hk_gaps:
                flag_hkgap = np.logical_or(flag_hkgap, np.logical_and(t > gap_start, t < gap_end))
            cosamp_az = np.array([self.FLAGGED_SAMPLE_VALUE if flag_hkgap[i] else cosamp_az[i] for i in range(len(cosamp_az))])
            cosamp_el = np.array([self.FLAGGED_SAMPLE_VALUE if flag_hkgap[i] else cosamp_el[i] for i in range(len(cosamp_el))])
            f['flag_hkgap'] = core.G3VectorBool(flag_hkgap)

            # Gaps in SMuRF data
            f['flag_smurfgap'] = core.G3VectorBool(flag_smurfgap)

            # Ancillary data (co-sampled HK encoder data)
            anc_data = core.G3TimesampleMap()
            anc_data.times = smurf_data.times
            anc_data['az_enc'] = core.G3VectorDouble(cosamp_az)
            anc_data['el_enc'] = core.G3VectorDouble(cosamp_el)

            # Write ancillary data to frame
            f['ancil'] = anc_data

            self.determine_state(hk_data['Azimuth_Velocity'], hk_data['Elevation_Velocity'])
            f['state'] = self.current_state
        finally:
            output += [f]

        return output

    def __call__(self, f):
        """
        Process a frame. Only Housekeeping and SMuRF frames will be manipulated;
        others will be passed through untouched.

        Parameters
        ----------
        f : G3Frame
            Input frame to be processed

        Returns
        -------
        output : list
            List containing frames to be entered into output Book
        """

        if f.type != core.G3FrameType.Housekeeping and f.type != core.G3FrameType.Scan:
            return [f]

        if f.type == core.G3FrameType.Housekeeping:
            if self.hkbundle is None:
                self.hkbundle = _HKBundle()

            try:
                i = list(f['block_names']).index('ACU_position')
            except ValueError:
                print("'ACU_position' field not found in block names.")
                return
            self.hkbundle.add(f['blocks'][i])

            self.hkbundle.data['Azimuth_Velocity'] = pos2vel(self.hkbundle.data['Azimuth_Corrected'])
            self.hkbundle.data['Elevation_Velocity'] = pos2vel(self.hkbundle.data['Elevation_Corrected'])

            azimuth_events = self.locate_crossing_events(self.hkbundle.data['Azimuth_Velocity'])
            self.hkbundle.azimuth_events = [self.hkbundle.times[i] for i in azimuth_events[0]]
            self.hkbundle.azimuth_starts = [self.hkbundle.times[i] for i in azimuth_events[1]]
            self.hkbundle.azimuth_stops  = [self.hkbundle.times[i] for i in azimuth_events[2]]

            elevation_events = self.locate_crossing_events(self.hkbundle.data['Elevation_Velocity'])
            self.hkbundle.elevation_events = [self.hkbundle.times[i] for i in elevation_events[0]]
            self.hkbundle.elevation_starts = [self.hkbundle.times[i] for i in elevation_events[1]]
            self.hkbundle.elevation_stops  = [self.hkbundle.times[i] for i in elevation_events[2]]

        if f.type == core.G3FrameType.Scan:
            if self.smbundle is None:
                self.smbundle = _SmurfBundle()

            output = []

            # Replace the times in 'data', 'tes_biases', and 'primary' with the correct timestamps
            # and trim any samples occuring outside the specified start/end times
            f = self.replace_times_and_trim_frame(f)
            # Replace the channel names with readout IDs (if available)
            f['data'].names = get_channel_names(f['data'], rids=self._readout_ids)

            t = f['data'].times
            # If all the samples have been trimmed, we can ignore this frame
            if len(t) == 0:
                return []

            # Add current frame
            self.smbundle.add(f)

            # If the existing data exceeds the specified maximum length
            while len(self.smbundle.times) >= self.maxlength:
                split_time = self.smbundle.times[self.maxlength-1]
                self._frame_splits.append(split_time)
                output += self.flush(split_time + 1)
                if self.flush_time is not None and split_time >= self.flush_time:
                    return output
            # If a frame split event has been reached
            if self.flush_time is not None and self.smbundle.ready(self.flush_time):
                output += self.flush()

            return output

class Bookbinder(object):
    """
    Bookbinder module. Co-samples and co-frames the provided Housekeeping (HK) and SMuRF data based
    on telescope scan patterns to create the final, properly formatted archival Book.

    If az/el encoder data exists in the HK file, Bookbinder identifies suitable frame split locations,
    using this information. When no az/el encoder data is found in HK timestream, Bookbinder will run
    in default mode, where frames are split once they reach the maximum length (Frameprocessor.MAXLENGTH).
    HK data is co-sampled with SMuRF data before output.
    """
    def __init__(self, smurf_files, hk_files=None, out_root='.', book_id=None,
                 session_id=None, stream_id=None, start_time=None, end_time=None,
                 smurf_timestamps=None, timing_system=False, verbose=True, **config):
        self._smurf_files = smurf_files
        self._hk_files = hk_files
        self._book_id = book_id
        self._out_root = out_root
        self._session_id = session_id
        self._stream_id = stream_id
        self._start_time = start_time
        self._end_time = end_time
        self._verbose = verbose
        self._frame_splits_file = op.join(out_root, book_id, 'frame_splits.txt')
        self._frame_splits = []
        self._meta_file = op.join(out_root, book_id, f'M_index.yaml')
        self.metadata = []
        self.frame_num = 0
        self.sample_num = 0
        self.ofile_num = 0
        self.default_mode = True  # True: time-based split; False: scan-based split
        self.MAX_SAMPLES_TOTAL = int(config.get("max_samples_total", 1e9))
        self.max_nchannels = int(config.get("max_nchannels", 1e3))
        self.MAX_SAMPLES_PER_CHANNEL = self.MAX_SAMPLES_TOTAL // self.max_nchannels
        self.DEFAULT_TIME = core.G3Time(1e18)  # 1e18 = 2286-11-20T17:46:40.000000000 (in the distant future)
        self.OVERWRITE_ANCIL_FILE = config.get("overwrite_afile", False)

        if isinstance(self._hk_files, str):
            self._hk_files = [self._hk_files]

        # Verify start and end times
        if self._start_time is not None:
            self._start_time = core.G3Time(self._start_time)
        if self._end_time is not None:
            self._end_time = core.G3Time(self._end_time)

        if self._start_time is not None and self._end_time is not None:
            if int(self._start_time) >= int(self._end_time):
                raise ValueError("Start time should be before end time." +
                                 "\nStart time: " + str(self._start_time) +
                                 "\nEnd time:   " + str(self._end_time))

        # Set up the FrameProcessor
        self.frameproc = FrameProcessor(start_time=self._start_time, end_time=self._end_time,
                                        smurf_timestamps=smurf_timestamps,
                                        timing_system=timing_system,
                                        **config.get("frameproc_config", {}))

        # Set up file I/O
        if isinstance(self._smurf_files, list):
            ifile = self._smurf_files.pop(0)
            if self._verbose: print(f"Bookbinding {ifile}")
            self.smurf_iter = core.G3File(ifile)

            self.create_file_writers()

    def create_file_writers(self):
        """
        Create the G3Writer instances for the output D-file and, if it doesn't already
        exist, the A-file. Optionally, overwrite the existing A-file setting the
        OVERWRITE_ANCIL_FILE attribute to True.
        """
        out_bdir = op.join(self._out_root, self._book_id)
        if not op.exists(out_bdir): os.makedirs(out_bdir)
        ofile = op.join(out_bdir, f'D_{self._stream_id}_{self.ofile_num:03d}.g3')
        if self._verbose: print(f"Writing {ofile}")
        self.writer = core.G3Writer(ofile)

        afile = op.join(out_bdir, f'A_ancil_{self.ofile_num:03d}.g3')
        if not op.exists(afile) or self.OVERWRITE_ANCIL_FILE:
            self.ancil_writer = core.G3Writer(afile)

    def add_misc_data(self, f):
        """
        Insert miscellaneous metadata into frame

        Parameters
        ----------
        f : G3Frame
            Frame to be processed

        Returns
        -------
        oframe : G3Frame
            Output frame with metadata inserted
        """
        oframe = core.G3Frame(f.type)
        oframe['book_id'] = self._book_id
        for k in f.keys():
            if k not in ['frame_num', 'session_id', 'stream_id', 'sostream_id', 'time']:
                oframe[k] = f[k]
        oframe['frame_num'] = core.G3Int(self.frame_num)
        if self._session_id is not None:
            oframe['session_id'] = core.G3Int(self._session_id)
        if self._stream_id is not None:
            oframe['stream_id'] = core.G3String(self._stream_id)
        if oframe.type == core.G3FrameType.Scan:
            oframe['sample_range'] = core.G3VectorInt([self.sample_num-len(f['signal'].times), self.sample_num-1])
        return oframe

    def write_frames(self, frames_list):
        """
        Write frames to file

        Parameters
        ----------
        frames_list : list
            List of frames to be written out
        """
        if not isinstance(frames_list, list):
            frames_list = list(frames_list)

        if len(frames_list) == 0: return

        for f in frames_list:
            # If the number of samples (per channel) exceeds the max allowed, create a new output file
            if f.type == core.G3FrameType.Scan:
                nsamples = len(f['signal'].times)   # number of samples in current frame
                self.sample_num += nsamples         # cumulative number of samples in current output file
                if self.sample_num > self.MAX_SAMPLES_PER_CHANNEL:
                    self.ofile_num += 1
                    self.create_file_writers()
                    self.sample_num = nsamples      # reset sample number to length of current frame
                    self.frame_num = 0              # reset frame number to 0
            # add misc metadata to output frame and write it out
            oframe = self.add_misc_data(f)
            self.writer.Process(oframe)
            # if ancil_writer exists, write out the ancil frame
            if hasattr(self, 'ancil_writer'):
                aframe = core.G3Frame(oframe.type)
                if 'ancil' in oframe.keys():
                    aframe['ancil'] = f['ancil']
                if 'sample_range' in oframe.keys():
                    aframe['sample_range'] = oframe['sample_range']
                self.ancil_writer(aframe)
            # update cumulative frame number
            self.frame_num += 1

    def find_frame_splits(self):
        """
        Determine the frame boundaries of the output Book

        Returns
        -------
        frame_splits : list
            List of timestamps where the frames should be split in the output Book
        """
        frame_splits = []

        if self.default_mode:
            frame_splits += [self.DEFAULT_TIME]  # no split, leave frameprocessor to decide based on maxlength
            return frame_splits

        # Assign a value to each event based on what occurs at that time: 0 for a zero-crossing,
        # +/-1 for an azimuth start/stop, and +/-2 for an elevation start/stop. The cumulative
        # sum of the resulting vector marks the state of the telescope at each event.
        event_times = np.unique(np.concatenate((self.frameproc.hkbundle.azimuth_events, self.frameproc.hkbundle.elevation_events)))
        values = []
        for e in event_times:
            # Note that multiple events can occur at the same event time, e.g., both az and el
            # can start moving at the same time
            v = 0
            if e in self.frameproc.hkbundle.azimuth_starts:
                v += 1
            if e in self.frameproc.hkbundle.azimuth_stops:
                v -= 1
            if e in self.frameproc.hkbundle.elevation_starts:
                v += 2
            if e in self.frameproc.hkbundle.elevation_stops:
                v -= 2
            values.append(v)
        seq = np.cumsum(values)

        # If the housekeeping file starts while the telescope is moving, a stop event will occur
        # prior to a start event, resulting in negative values in the seq vector.
        # In that case, shift every value up by the lowest value.
        starts_while_moving = False
        m = np.min(seq)
        if m < 0:
            starts_while_moving = True
            seq -= m

        # Each time the seq returns to zero, the telescope is coming to a complete stop (in both
        # az and el). Split the seq vector at these places.
        sp = np.where(seq == 0)[0] + 1
        sps = np.array_split(seq, sp)
        spi = np.array_split(range(len(seq)), sp)

        # Determine the times where the frame should be split
        for i, s in enumerate(sps):
            if len(s) == 0:
                continue
            if np.max(s) >= 2:
                # The telescope is slewing; only record the start (if it exists) and
                # the stop of the slew
                if i == 0 and starts_while_moving:
                    slew_stop = event_times[spi[i][-1]]
                    frame_splits += [slew_stop]
                else:
                    slew_start = event_times[spi[i][0]]
                    slew_stop  = event_times[spi[i][-1]]
                    frame_splits += [slew_start, slew_stop]
            else:
                # All the events are frame splits
                frame_splits += list(event_times[spi[i]])

        return frame_splits

    def process_HK_files(self):
        """
        Subroutine to process any provided Housekeeping (HK) files.

        If no HK files provided, Bookbinder will proceed in default mode.
        """
        if self._hk_files is not None:
            if not isinstance(self._hk_files, list):
                raise TypeError("Please provide HK files in a list.")

            # Chain multiple hkfile iteratables together so that
            # by using `next` we iterate across all hk files
            if not hasattr(self, 'hk_iter'):
                self.hk_iter = []
                for hkfile in self._hk_files:
                    self.hk_iter = itertools.chain(self.hk_iter, core.G3File(hkfile))

            # Initialize variables for gap detection
            prev_frame_last_sample     = None
            prev_frame_sample_interval = None

            # Loop over HK frames to look for:
            # 1. Az/El encoder data; run in default mode otherwise
            # 2. Gaps in time between frames, which need to be flagged
            for h in self.hk_iter:
                if h['hkagg_type'] != 2 or ('ACU_position' not in h['block_names']):
                    continue

                # Check if a time gap exists since previous frame containing ACU position data
                acu_pos_index = list(h['block_names']).index('ACU_position')
                t = h['blocks'][acu_pos_index].times
                if prev_frame_last_sample is not None and prev_frame_sample_interval is not None:
                    this_frame_first_sample = t[0].time
                    time_since_prev_frame   = this_frame_first_sample - prev_frame_last_sample
                    if time_since_prev_frame/prev_frame_sample_interval - 1 > self.frameproc.GAP_THRESHOLD:
                        # Add gap to internal list
                        self.frameproc._hk_gaps.append((prev_frame_last_sample, this_frame_first_sample))
                # update values
                prev_frame_last_sample = t[-1].time
                prev_frame_sample_interval = (t[-1].time - t[0].time)/(len(t)-1)

                self.default_mode = False
                self.frameproc(h)

    def compile_mfile_dict(self):
        """
        Compile a dictionary of values to be entered into metadata file (M-file)

        Returns
        -------
        d : dict
            Dictionary of values
        """
        d = {'book_id':    self._book_id,
             'session_id': self._session_id,
             'start_time': self._start_time.time/core.G3Units.s,
             'end_time':   self._end_time.time/core.G3Units.s,
             'n_frames':   self.frame_num,
             'n_samples':  self.sample_num}
        return d

    def __call__(self):
        """
        Process Housekeeping (if provided) and SMuRF data.

        Write out frame_splits file and metadata file before exiting.
        """
        # Determine if HK files contain az/el encoder data and find any gaps between HK frames
        self.process_HK_files()

        # If frame splits file exists, load it; if not, determine appropriate split locations
        if op.isfile(self._frame_splits_file):
            self._frame_splits = [core.G3Time(t) for t in np.loadtxt(self._frame_splits_file, dtype='int', ndmin=1)]
        else:
            # note that find_frame_splits depends on process_HK_files having run
            self._frame_splits = self.find_frame_splits()

        for event_time in self._frame_splits:
            self.frameproc.flush_time = event_time
            output = []

            if self.frameproc.smbundle is not None and self.frameproc.smbundle.ready(self.frameproc.flush_time):
                output += self.frameproc.flush()
                output = [o for o in output if len(o['signal'].times) > 0]  # Remove 0-length frames
                if len(output) > 0:
                    output += self.metadata
                    self.metadata = []
                self.write_frames(output)
                continue

            while self.frameproc.smbundle is None or not self.frameproc.smbundle.ready(self.frameproc.flush_time):
                try:
                    f = next(self.smurf_iter)
                except StopIteration:
                    # If there are remaining files, update the
                    # SMuRF source iterator and G3 file writer
                    if len(self._smurf_files) > 0:
                        ifile = self._smurf_files.pop(0)
                        if self._verbose: print(f"Bookbinding {ifile}")
                        self.smurf_iter = core.G3File(ifile)
                    else:
                        # If there are no more SMuRF files, output remaining SMuRF data
                        self.frameproc.flush_time = self.DEFAULT_TIME
                        if self.frameproc.hkbundle is not None:
                            # As long as pos2vel() uses np.ediff1d(), the velocity vector will be 1 sample shorter than position
                            # Add an extra element to the end of velocity vectors to compensate
                            self.frameproc.hkbundle.data['Azimuth_Velocity'] = np.append(self.frameproc.hkbundle.data['Azimuth_Velocity'], 0)
                            self.frameproc.hkbundle.data['Elevation_Velocity'] = np.append(self.frameproc.hkbundle.data['Elevation_Velocity'], 0)
                        output += self.frameproc.flush()
                        # note that `signal` field only in the product of the FrameProcessor, not the input data
                        output = [o for o in output if len(o['signal'].times) > 0]  # Remove 0-length frames
                        self.write_frames(output + self.metadata)
                        output = []
                        self.metadata = []
                        break
                else:
                    if f.type != core.G3FrameType.Scan:
                        if self.frameproc.smbundle is None or len(self.frameproc.smbundle.times) == 0:
                            output += [f]
                        else:
                            self.metadata += [f]
                    else:
                        output += self.frameproc(f)  # FrameProcessor returns a list of frames (can be empty)
                        output = [o for o in output if len(o['signal'].times) > 0]  # Remove 0-length frames
                        # Write out metadata frames only when FrameProcessor outputs one or more (scan) frames
                        if len(output) > 0:
                            output += self.metadata
                            self.metadata = []
                    self.write_frames(output)

                    # Clear output buffer after writing
                    output = []

        self._frame_splits += self.frameproc._frame_splits
        # Write frame splits file if it doesn't already exist
        if not op.isfile(self._frame_splits_file):
            np.savetxt(self._frame_splits_file, np.unique([int(t) for t in self._frame_splits]), fmt='%i')

        # Write metadata file ('M-file')
        with open(self._meta_file, 'w') as mfile:
            yaml.dump(self.compile_mfile_dict(), mfile, sort_keys=False)


##############
# exceptions #
##############

class TimingSystemError(Exception):
    """Exception raised when the timing system is on but timing counters are not found"""
    pass


#####################
# Utility functions #
#####################

def pos2vel(p):
    """From a given position vector, compute the velocity vector

    Parameters
    ----------
    p : np.ndarray
        Position vector

    Returns
    -------
    np.ndarray
        Velocity vector

    """
    return np.ediff1d(p)

def get_channel_names(s, rids=None):
    """
    Retrieve the channel names in a G3SuperTimestream. If provided, matches readout IDs to the
    associated channels and returns them

    Parameters
    ----------
    s : G3SuperTimestream
        Timestream to query
    rids : list
        List of readout IDs

    Returns
    -------
    list
        List of channel names
    """
    if rids is not None:
        # check compatability
        assert len(s.names) == len(rids)
        assert list(s.names) == sorted(s.names)  # name sure things are in our assumed order
        # maybe more checks are needed here...
        return rids
    else:
        return s.names

def get_channel_data_from_name(s, channel_name):
    """
    From the channel name of a G3SuperTimestream (as listed in .names), retrieve
    the associated data vector

    Parameters
    ----------
    s : G3SuperTimestream
        Input timestream
    channel_name : str
        The name of the channel, exactly as listed in the .names field

    Returns
    -------
    np.ndarray
        Data vector associated with that channel name
    """
    if channel_name not in s.names:
        raise KeyError(f"{channel_name} not found in this SuperTimestream")
    idx = list(s.names).index(channel_name)
    return s.data[idx]

def counters_to_timestamps(c0, c2):
    s, ns = split_ts_bits(c2)

    # Add 20 years in seconds (accounting for leap years) to handle
    # offset between EPICS time referenced to 1990 relative to UNIX time.
    c2 = s + ns*1e-9 + 5*(4*365 + 1)*24*60*60
    ts = np.round(c2 - (c0 / 480000) ) + c0 / 480000
    return ts

def get_timestamps(f, use_counters):
    """
    Calculate the timestamp field for loaded data

    Copied from load_smurf.py (Jan 23, 2023)

    Parameters
    ----------
    f : G3Frame
        Input SMuRF frame containing data in G3SuperTimestream format
    use_counters : bool
        Whether to calcuate the timestamps from the timing counters.
        If false, returns the times recorded in the .times field.

    Returns
    -------
    G3VectorTime
        The array of computed timestamps
    """
    if use_counters and 'primary' in f.keys():
        counter0 = get_channel_data_from_name(f['primary'], 'Counter0')
        if np.any(counter0):
            counter2 = get_channel_data_from_name(f['primary'], 'Counter2')
            timestamps = counters_to_timestamps(counter0, counter2)
            timestamps *= core.G3Units.s
        else:
            raise TimingSystemError("No timing counters found")
    elif use_counters and 'primary' not in f.keys():
        raise TimingSystemError("'primary' field not found")
    else:
        timestamps = f['data'].times
    return core.G3VectorTime(timestamps)

def fill_time_gaps(ts):
    """
    Fills gaps in an array of timestamps.

    Parameters
    -------------
    ts : np.ndarray
        List of timestamps of length `n`, potentially with gaps

    Returns
    --------
    new_ts : np.ndarray
        New list of timestamps of length >= n, with gaps filled.
    """
    # Find indices where gaps occur and how long each gap is
    dts = np.diff(ts)
    dt = np.median(dts)
    missing = np.round(dts/dt - 1).astype(int)
    total_missing = int(np.sum(missing))

    # Create new  array with the correct number of samples
    new_ts = np.full(len(ts) + total_missing, np.nan)

    # Insert old timestamps into new array with offsets that account for gaps
    offsets = np.concatenate([[0], np.cumsum(missing)])
    i0s = np.arange(len(ts))
    new_ts[i0s + offsets] = ts

    # Use existing data to interpolate and fill holes
    m = np.isnan(new_ts)
    xs = np.arange(len(new_ts))
    interp = interp1d(xs[~m], new_ts[~m])
    new_ts[m] = interp(xs[m])

    return new_ts

def find_missing_samples(refs, vs, atol=0.5):
    """
    Find missing samples in a list of timestamps (vs) given a list of
    reference timestamps (refs). The reference timestamps are assumed
    to be the "true" timestamps, and the list of timestamps (vs) are
    assumed to be missing some of the samples. The function returns
    the missing samples in the list of timestamps (vs).

    Parameters
    ----------
    refs : array_like
        List of reference timestamps
    vs : array_like
        List of timestamps
    atol : float
        Absolute tolerance for the difference between the reference

    Returns
    -------
    i_missing : array_like
        indices (in refs) of the missing samples in vs
    t_missing : array_like
        values (in refs) of the missing samples in vs
    """
    # Find the indices of the samples in the list of timestamps (vs)
    # that are closest to the reference timestamps
    idx = np.searchsorted(vs, refs, side='left')
    idx = np.clip(idx, 1, len(vs)-1)
    # shift indices to the closest sample
    left = vs[idx-1]
    right = vs[idx]
    idx -= refs - left < right - refs
    missing = np.where(np.abs(vs[idx] - refs) > atol)[0]
    return missing, refs[missing]
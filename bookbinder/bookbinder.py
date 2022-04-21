#!/usr/bin/env python3

from spt3g import core
import numpy as np
import so3g

def pos2vel(p):
    return np.ediff1d(p)

def smurf_reader(filename):
    reader = so3g.G3IndexedReader(filename)
    while True:
        frames = reader.Process(None)
        assert len(frames) <= 1
        if len(frames) == 0:
            break
        yield frames[0]

class _HKBundle():
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
        return len(self.azimuth_events) > 0

    def add(self, b):
        self.times.extend(b.times)
        if self.data is None:
            self.data = {c: [] for c in b.keys()}
        for c in b.keys():
            self.data[c].extend(b[c])

    def rebundle(self, flush_time):
        if len(self.times) == 0:
            return None

        output = core.G3TimesampleMap()
        output.times = core.G3VectorTime([t for t in self.times if t < flush_time])
        self.times = [t for t in self.times if t >= flush_time]

        for c in self.data.keys():
            output[c] = core.G3Timestream(np.array(self.data[c][:len(output.times)]))

        self.data = {c: self.data[c][len(output.times):] for c in self.data.keys()}

        return output

class _SmurfBundle():
    def __init__(self):
        self.times = []
        self.signal = None
        self.bias = None
        self.primary = None

    def ready(self, flush_time):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1] >= flush_time

    def createNewSuperTimestream(self, f):
        """
        From the input SuperTimestream, create a new (empty) SuperTimestream
        with the same channels and channel names
        """
        dtype = np.int32 if (f.data.dtype != np.int64) else np.int64

        sts = so3g.G3SuperTimestream()
        sts.names = f.names
        sts.times = core.G3VectorTime([])
        sts.data = np.empty( (len(f.names), 0) , dtype=dtype)

        return sts

    def add(self, f):
        self.times.extend(f['data'].times)

        if self.signal is None:
            self.signal = self.createNewSuperTimestream(f['data'])
        self.signal.times.extend(f['data'].times)
        self.signal.data = np.hstack((self.signal.data, f['data'].data)).astype(np.int32)

        if 'tes_biases' in f.keys():
            if self.bias is None:
                self.bias = self.createNewSuperTimestream(f['tes_biases'])
            self.bias.times.extend(f['tes_biases'].times)
            self.bias.data = np.hstack((self.bias.data, f['tes_biases'].data)).astype(np.int32)

        if 'primary' in f.keys():
            if self.primary is None:
                self.primary = self.createNewSuperTimestream(f['primary'])
            self.primary.times.extend(f['primary'].times)
            self.primary.data = np.hstack((self.primary.data, f['primary'].data)).astype(np.int64)

    def rebundle(self, flush_time):
        """
        Output the buffered data before flush_time, keep the rest in buffer
        """
        def rebundle_sts(sts, flush_time):
            if sts is None:
                return None, sts

            # Output SuperTimestream (to be written to frame)
            stsout = so3g.G3SuperTimestream()
            stsout.names = sts.names
            stsout.times = core.G3VectorTime([t for t in sts.times if t < flush_time])
            stsout.data = sts.data[:,:len(stsout.times)]
            # New buffer SuperTimestream
            newsts = so3g.G3SuperTimestream()
            newsts.names = sts.names
            newsts.times = core.G3VectorTime([t for t in sts.times if t >= flush_time])
            newsts.data = sts.data[:,len(stsout.times):]

            return stsout, newsts

        signalout, self.signal = rebundle_sts(self.signal, flush_time)
        biasout, self.bias = rebundle_sts(self.bias, flush_time)
        primout, self.primary = rebundle_sts(self.primary, flush_time)

        self.times = [t for t in self.times if t >= flush_time]

        return signalout, biasout, primout

class FrameProcessor(object):
    def __init__(self):
        self.hkbundle = None
        self.smbundle = None
        self.flush_time = None
        self.maxlength = 10000
        self.current_state = 0  # default to scan state

    def ready(self):
        """
        Check if criterion passed in HK (for now, sign change in Az scan velocity data)
        """
        return self.hkbundle.ready() if (self.hkbundle is not None) else False

    def locate_crossing_events(self, t, dy=0.001, min_gap=200):
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

        starts = np.sort(np.concatenate((c_lower_exit, c_upper_exit))).astype(np.int64)
        stops = np.sort(np.concatenate((c_lower_enter, c_upper_enter))).astype(np.int64)
        events = np.sort(np.concatenate((starts, stops))).astype(np.int64)

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

    def flush(self, flush_time=None):
        if flush_time == None:
            flush_time = self.flush_time
        output = []

        smurf_data, smurf_bias, smurf_primary = self.smbundle.rebundle(flush_time)

        # Create SuperTimestream containing SMuRF data
        sts = so3g.G3SuperTimestream()
        sts.times = smurf_data.times
        sts.names = [k for k in smurf_data.names]
        sts.data = smurf_data.data

        # Write signal data to frame
        f = core.G3Frame(core.G3FrameType.Scan)
        f['signal'] = sts
        if smurf_bias is not None:
            f['tes_bias'] = smurf_bias
        if smurf_primary is not None:
            f['primary'] = smurf_primary

        try:
            hk_data = self.hkbundle.rebundle(flush_time)
        except:
            # No az/el encoder data found in HK files
            f['state'] = 3
        else:
            # Co-sampled (interpolated) az/el encoder data
            cosamp_az = np.interp(smurf_data.times, hk_data.times, hk_data['Azimuth_Corrected'], left=np.nan, right=np.nan)
            cosamp_el = np.interp(smurf_data.times, hk_data.times, hk_data['Elevation_Corrected'], left=np.nan, right=np.nan)

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
        Process a frame
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

            self.smbundle.add(f)

            # If the existing data exceeds the specified maximum length
            while len(self.smbundle.times) >= self.maxlength:
                output += self.flush(self.smbundle.times[self.maxlength-1] + 1)
            # If a frame split event has been reached
            if self.flush_time is not None and self.smbundle.ready(self.flush_time):
                output += self.flush()

            return output

class Bookbinder(object):
    """
    Bookbinder
    """
    def __init__(self, hk_files, smurf_files, out_files, verbose=True):
        self._hk_files = hk_files
        self._smurf_files = smurf_files
        self._out_files = out_files
        self._verbose = verbose
        self.metadata = []

        self.frameproc = FrameProcessor()

        ifile = self._smurf_files.pop(0)
        if self._verbose: print(f"Bookbinding {ifile}")
        self.smurf_iter = smurf_reader(ifile)

        ofile = self._out_files.pop(0)
        if self._verbose: print(f"Writing {ofile}")
        self.writer = core.G3Writer(ofile)

    def write_frames(self, frames_list):
        """
        Write frames to file
        """
        if not isinstance(frames_list, list):
            frames_list = list(frames_list)

        if len(frames_list) == 0: return
        if self._verbose: print(f"=> Writing {len(frames_list)} frames")

        for f in frames_list:
            self.writer.Process(f)

    def find_frame_splits(self):
        """
        Determine the frame boundaries of the output Book
        """
        frame_splits = []

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

    def __call__(self):
        N_hkfr_with_enc = 0    # Number of HK frames that contain encoder data
        for hkfile in self._hk_files:
            for h in core.G3File(hkfile):
                if h['hkagg_type'] != 2 or ('ACU_position' not in h['block_names']):
                    continue
                self.frameproc(h)
                N_hkfr_with_enc += 1

        if N_hkfr_with_enc == 0:
            print("No encoder data found in HK file(s); continuing with default mode.")

            # Loop over SMuRF frames/files
            while len(self._smurf_files) >= 0:
                output = []
                for f in self.smurf_iter:
                    if f.type != core.G3FrameType.Scan:
                        if self.frameproc.smbundle is None or len(self.frameproc.smbundle.times) == 0:
                            output += [f]
                        else:
                            self.metadata += [f]
                    else:
                        o = self.frameproc(f)
                        # Write out metadata frames only when FrameProcessor outputs one or more (scan) frames
                        if len(o) > 0:
                            output += o + self.metadata
                            self.metadata = []

                    # To avoid keeping too many frames in memory, write out buffer if too long
                    if len(output) >= 10:
                        if self._verbose:
                            print("Writing out buffered frames to avoid storing too many in memory.")
                        self.write_frames(output)
                        output = []

                # If there are no more SMuRF frames, output remaining SMuRF data
                if len(self.frameproc.smbundle.times) > 0:
                    output += self.frameproc.flush(self.frameproc.smbundle.times[-1] + 1)
                self.write_frames(output + self.metadata)
                self.metadata = []

                # If there are remaining files, update the
                # SMuRF source iterator and G3 file writer
                if len(self._smurf_files) > 0:
                    ifile = self._smurf_files.pop(0)
                    if self._verbose: print(f"Bookbinding {ifile}")
                    self.smurf_iter = smurf_reader(ifile)
                if len(self._out_files) > 0:
                    ofile = self._out_files.pop(0)
                    if self._verbose: print(f"Writing {ofile}")
                    self.writer = core.G3Writer(ofile)
                else:
                    break

            return

        for event_time in self.find_frame_splits():
            self.frameproc.flush_time = event_time
            output = []

            if self.frameproc.smbundle is not None and self.frameproc.smbundle.ready(self.frameproc.flush_time):
                output += self.frameproc.flush()
                self.write_frames(output)
                continue

            while self.frameproc.smbundle is None or not self.frameproc.smbundle.ready(self.frameproc.flush_time):
                try:
                    f = next(self.smurf_iter)
                except StopIteration:
                    # If there are no more SMuRF frames, output remaining SMuRF data
                    if len(self.frameproc.smbundle.times) > 0:
                        self.frameproc.flush_time = self.frameproc.smbundle.times[-1] + 1  # +1 to ensure last sample gets included (= 1e-8 sec << sampling cadence)
                        output += self.frameproc.flush()
                    self.write_frames(output)

                    # If there are remaining files, update the
                    # SMuRF source iterator and G3 file writer
                    if len(self._smurf_files) > 0:
                        ifile = self._smurf_files.pop(0)
                        if self._verbose: print(f"Bookbinding {ifile}")
                        self.smurf_iter = smurf_reader(ifile)
                    if len(self._out_files) > 0:
                        ofile = self._out_files.pop(0)
                        if self._verbose: print(f"Writing {ofile}")
                        self.writer = core.G3Writer(ofile)
                    else:
                        break

                    # Reset the state of the loop
                    self.frameproc.flush_time = event_time
                    output = []
                else:
                    if f.type != core.G3FrameType.Scan:
                        if self.frameproc.smbundle is None or len(self.frameproc.smbundle.times) == 0:
                            output += [f]
                        else:
                            self.metadata += [f]
                    else:
                        output += self.frameproc(f)  # FrameProcessor returns a list of frames (can be empty)
                        # Write out metadata frames only when FrameProcessor outputs one or more (scan) frames
                        if len(output) > 0:
                            output += self.metadata
                    self.write_frames(output)

                    # Clear buffers after writing
                    output = []
                    self.metadata = []

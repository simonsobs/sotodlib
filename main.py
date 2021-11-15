#!/usr/bin/env python3

import so3g
from spt3g import core
import numpy as np

def pos2vel(p):
    return np.ediff1d(p)

def locate_sign_changes(t):
    return np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0] + 1

class _DataBundle():
    def __init__(self):
        self.times = []
        self.data = None

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

class _HKBundle(_DataBundle):
    def __init__(self):
        super().__init__()
        self.turnaround_times = []

    def set_azimuth_velocity(self):
        self.data['Azimuth_Velocity'] = pos2vel(self.data['Azimuth_Corrected'])

    def set_turnaround_times(self):
        if 'Azimuth_Velocity' not in self.data.keys():
            self.set_azimuth_velocity()
        self.turnaround_times = [self.times[i] for i in
                                 locate_sign_changes(self.data['Azimuth_Velocity'])]

    def ready(self):
        return len(self.turnaround_times) > 0

class _SmurfBundle(_DataBundle):
    def ready(self, flush_time):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1] >= flush_time

class FrameProcessor(object):
    def __init__(self):
        self.hkbundle = None
        self.smbundle = None
        self.flush_time = None
        self.maxlength = 10000

    def ready(self):
        """
        Check if criterion passed in HK (for now, sign change in Az scan velocity data)
        """
        return self.hkbundle.ready() if (self.hkbundle is not None) else False

    def split_frame(self, f, maxlength=10000):
        output = []

        smb = _SmurfBundle()
        smb.add(f['data'])

        hkb = _HKBundle()
        hkb.add(f['hk'])

        while len(smb.times) > maxlength:
            t = smb.times[maxlength]

            g = core.G3Frame(core.G3FrameType.Scan)
            g['data'] = smb.rebundle(t)
            g['hk'] = hkb.rebundle(t)

            output += [g]

        g = core.G3Frame(core.G3FrameType.Scan)
        g['data'] = smb.rebundle(smb.times[-1] + 1)
        g['hk'] = hkb.rebundle(hkb.times[-1] + 1)

        output += [g]

        return output

    def flush(self):
        output = []

        f = core.G3Frame(core.G3FrameType.Scan)
        f['data'] = self.smbundle.rebundle(self.flush_time)
        f['hk'] = self.hkbundle.rebundle(self.flush_time)

        # Co-sampled (interpolated) azimuth encoder data
        f['data']['Azimuth'] = core.G3Timestream(np.interp(f['data'].times, f['hk'].times, f['hk']['Azimuth_Corrected'], left=np.nan, right=np.nan))

        if len(f['data'].times) > self.maxlength:
            output += self.split_frame(f, maxlength=self.maxlength)
        else:
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

            self.hkbundle.add(f['blocks'][0])   # 0th block for now
            self.hkbundle.set_azimuth_velocity()
            self.hkbundle.set_turnaround_times()

        if f.type == core.G3FrameType.Scan:
            if self.smbundle is None:
                self.smbundle = _SmurfBundle()

            output = []

            self.smbundle.add(f['data'])

            if self.smbundle.ready(self.flush_time):
                output += self.flush()

            return output

class Bookbinder(object):
    """
    Bookbinder
    """
    def __init__(self, smurf_files, out_files):
        self._smurf_files = smurf_files
        self._out_files = out_files

        self.frameproc = FrameProcessor()
        self.smurf_iter = core.G3File(self._smurf_files.pop(0))
        self.writer = core.G3Writer(self._out_files.pop(0))

    def write_frames(self, frames_list):
        """
        Write frames to file
        """
        if not isinstance(frames_list, list):
            frames_list = list(frames_list)

        for f in frames_list:
            self.writer.Process(f)

    def __call__(self, fr):
        """
        Main loop

        Strategy:
        1. Add HK frames until sign change detected - get list of (timestamps of) sign changes
        2. While len(sign_changes) > 0, pop the first sign change, add SMuRF frames until that time;
        emit new frame; truncate HK and SMuRF frames
        3. Repeat Step 2 until len(sign_changes) == 0, then go back to Step 1
        """
        if fr.type != core.G3FrameType.Housekeeping:
            return

        if fr['hkagg_type'] != 2:
            return

        self.frameproc(fr)
        if not self.frameproc.ready():
            return

        tt = self.frameproc.hkbundle.turnaround_times
        output = []
        while len(tt) > 0:
            self.frameproc.flush_time = tt.pop(0)

            if self.frameproc.smbundle is not None and self.frameproc.smbundle.ready(self.frameproc.flush_time):
                output += self.frameproc.flush()

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
                        self.smurf_iter = core.G3File(self._smurf_files.pop(0))
                    if len(self._out_files) > 0:
                        self.writer = core.G3Writer(self._out_files.pop(0))

                    return

                if f.type != core.G3FrameType.Scan:
                    continue
                output += self.frameproc(f)  # FrameProcessor returns a list of frames (can be empty)

        self.write_frames(output)
        return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--g3', dest='g3file', nargs='+', type=str, required=True,
                        help='full path to G3 file')
    parser.add_argument('--hk', dest='hkfile', nargs='+', type=str, required=True,
                        help='full path to HK file')
    args = parser.parse_args()

    B = Bookbinder(args.g3file, ['out{:03d}.g3'.format(i) for i in range(len(args.g3file))])
    for h in core.G3File(args.hkfile):
        B(h)

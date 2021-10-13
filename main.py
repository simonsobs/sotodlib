#!/usr/bin/env python3

import so3g
from spt3g import core
import numpy as np

def locate_sign_changes(t):
    return np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0] + 1

class _HKBlockBundle(object):
    def __init__(self):
        self.times = []
        self.data = None

    def add(self, b):
        self.times.extend(b.times)
        if self.data is None:
            self.data = {c: [] for c in b.keys()}
        for c in b.keys():
            self.data[c].extend(b[c])

    def ready(self):
        return len(locate_sign_changes(np.ediff1d(self.data['Azimuth_Corrected']))) > 0

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

class _ScanDataBundle(object):
    def __init__(self):
        self.times = []
        self.data = None

    def add(self, b):
        self.times.extend(b.times)
        if self.data is None:
            self.data = {c: [] for c in b.keys()}
        for c in b.keys():
            self.data[c].extend(b[c])

    def ready(self, flush_time):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1] >= flush_time

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

class Bookbinder(object):
    def __init__(self):
        self.hkbundle = None
        self.sdbundle = None
        self.flush_time = None
        self.maxlength = 10000

    def ready(self):
        """
        Check if criterion passed in HK (for now, sign change in Az scan velocity data)
        """
        return self.hkbundle.ready() if (self.hkbundle is not None) else False

    def split_frame(self, f, maxlength=10000):
        output = []

        sdb = _ScanDataBundle()
        sdb.add(f['data'])

        hkb = _HKBlockBundle()
        hkb.add(f['hk'])

        while len(sdb.times) > maxlength:
            t = sdb.times[maxlength]

            g = core.G3Frame(core.G3FrameType.Scan)
            g['data'] = sdb.rebundle(t)
            g['hk'] = hkb.rebundle(t)

            output += [g]

        g = core.G3Frame(core.G3FrameType.Scan)
        g['data'] = sdb.rebundle(sdb.times[-1] + 1)
        g['hk'] = hkb.rebundle(hkb.times[-1] + 1)

        output += [g]

        return output

    def flush(self):
        output = []

        f = core.G3Frame(core.G3FrameType.Scan)
        f['data'] = self.sdbundle.rebundle(self.flush_time)
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
        if f.type == core.G3FrameType.EndProcessing:
            return self.sdbundle.rebundle(self.flush_time) + [f]

        if f.type != core.G3FrameType.Housekeeping and f.type != core.G3FrameType.Scan:
            return f

        if f.type == core.G3FrameType.Housekeeping:
            if self.hkbundle is None:
                self.hkbundle = _HKBlockBundle()

            self.hkbundle.add(f['blocks'][0])   # 0th block for now

        if f.type == core.G3FrameType.Scan:
            if self.sdbundle is None:
                self.sdbundle = _ScanDataBundle()

            output = []

            self.sdbundle.add(f['data'])

            if self.sdbundle.ready(self.flush_time):
                output += self.flush()

            return output

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--g3', dest='g3file', type=str, required=True, help='full path to G3 file')
    parser.add_argument('--hk', dest='hkfile', type=str, required=True, help='full path to HK file')
    args = parser.parse_args()

    B = Bookbinder()

    smurfiter = core.G3File(args.g3file)

    """
    Main loop

    Strategy:
    1. Add HK frames until sign change detected - get list of (timestamps of) sign changes
    2. While len(sign_changes) > 0, pop the first sign change, add SMuRF frames until that time;
       emit new frame; truncate HK and SMuRF frames
    3. Repeat Step 2 until len(sign_changes) == 0, then go back to Step 1
    """
    def framesource(fr):
        if fr.type == core.G3FrameType.EndProcessing:
            return None

        if fr.type != core.G3FrameType.Housekeeping:
            return []

        if fr['hkagg_type'] != 2:
            return []

        B(fr)
        if not B.ready():
            return []

        sc = locate_sign_changes(np.ediff1d(B.hkbundle.data['Azimuth_Corrected']))
        tc = [B.hkbundle.times[i] for i in sc]
        output = []
        while len(tc) > 0:
            B.flush_time = tc.pop(0)

            while B.sdbundle is None or not B.sdbundle.ready(B.flush_time):
                try:
                    f = next(smurfiter)
                except StopIteration:
                    # If there are no more SMuRF frames, output remaining SMuRF data
                    # and terminate program by ending loop
                    if len(B.sdbundle.times):
                        B.flush_time = B.sdbundle.times[-1] + 1  # +1 to ensure last sample gets included (= 1e-8 sec << sampling cadence)
                        output += B.flush()
                    return output

                if f.type != core.G3FrameType.Scan:
                    continue
                bookframes = B(f)  # returns a list of frames
                if len(bookframes) > 0:
                    output += bookframes
        return output

    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=args.hkfile)
    pipe.Add(framesource)
    pipe.Add(core.G3Writer, filename='out.g3')
    pipe.Run()

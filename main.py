#!/usr/bin/env python3

import so3g
from spt3g import core
import numpy as np

def locate_sign_changes(t):
    return np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0] + 1

class _HKBlockBundle(object):
    def __init__(self):
        self.times = []
        self.data = []

    def add(self, b):
        self.times.extend([int(t) for t in b.times])
        self.data.extend(b['Azimuth_current_velocity'])

    def ready(self):
        return len(locate_sign_changes(self.data)) > 0

    def rebundle(self, flush_time):
        if len(self.times) == 0:
            return None

        self.times = [t for t in self.times if t >= flush_time]
        self.data = self.data[-len(self.times):]

class _ScanDataBundle(object):
    def __init__(self):
        self.times = []

    def add(self, b):
        self.times.extend(b.times())

    def ready(self, flush_time):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1].time >= flush_time

    def rebundle(self, flush_time):
        if len(self.times) == 0:
            return None
        # For now, just return the times as output
        output = [t for t in self.times if t.time < flush_time]
        self.times = [t for t in self.times if t.time >= flush_time]

        return output

class Bookbinder(object):
    def __init__(self):
        self.hkbundle = None
        self.sdbundle = None
        self.flush_time = None

    def ready(self):
        """
        Check if criterion passed in HK (for now, sign change in Az scan velocity data)
        """
        return self.hkbundle.ready() if (self.hkbundle is not None) else False

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
                self.hkbundle.rebundle(self.flush_time)
                output = self.sdbundle.rebundle(self.flush_time)

            return output

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--g3', dest='g3file', type=str, required=True, help='full path to G3 file')
    parser.add_argument('--hk', dest='hkfile', type=str, required=True, help='full path to HK file')
    args = parser.parse_args()

    scanframes = [f for f in core.G3File(args.g3file) if f.type == core.G3FrameType.Scan]

    B = Bookbinder()

    hkframes = [h for h in core.G3File(args.hkfile)]
    acu_summary = [h for h in hkframes if h['hkagg_type'] == 2 and h['block_names'][0] == 'ACU_summary_output']

    """
    Main loop

    Strategy:
    1. Add HK frames until sign change detected - get list of (timestamps of) sign changes
    2. While len(sign_changes) > 0, pop the first sign change, add SMuRF frames until that time;
       emit new frame; truncate HK and SMuRF frames
    3. Repeat Step 2 until len(sign_changes) == 0, then go back to Step 1
    """
    while len(acu_summary) > 0:
        while not B.ready():
            # If there are no more HK frames, terminate program by ending loop
            if len(acu_summary) == 0:
                break
            B(acu_summary.pop(0))

        sc = locate_sign_changes(B.hkbundle.data)
        tc = [B.hkbundle.times[i] for i in sc]
        while len(tc) > 0:
            B.flush_time = tc.pop(0)

            while B.sdbundle is None or not B.sdbundle.ready(B.flush_time):
                # If there are no more SMuRF frames, output remaining SMuRF data
                # and terminate program by ending loop
                if len(scanframes) == 0:
                    output = B.sdbundle.rebundle(B.flush_time)
                    tc = []
                    acu_summary = []
                    break
                output = B(scanframes.pop(0))

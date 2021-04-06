#!/usr/bin/env python3

import so3g
from spt3g import core
import numpy as np

def locate_sign_changes(t):
    return np.where(np.sign(t[:-1]) != np.sign(t[1:]))[0] + 1

class _ScanDataBundle(object):
    def __init__(self, flush_time):
        self.times = []
        self.flush_time = flush_time

    def add(self, b):
        self.times.extend(b.times())

    def ready(self):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1].time >= self.flush_time

    def rebundle(self):
        if len(self.times) == 0:
            return None
        # For now, just return the times as output
        output = [t for t in self.times if t.time < self.flush_time]
        self.times = [t for t in self.times if t.time >= self.flush_time]

        return output

class DataReframer(object):
    def __init__(self):
        self.bundle = None
        self.flush_time = None

    def ready(self):
        return self.bundle.ready() if (self.bundle is not None) else False

    def __call__(self, f):
        """
        Process a frame
        """
        if f.type == core.G3FrameType.EndProcessing:
            return self.bundle.rebundle() + [f]

        if f.type != core.G3FrameType.Scan:
            return f

        if self.bundle is None:
            self.bundle = _ScanDataBundle(self.flush_time)

        output = []

        self.bundle.add(f['data'])

        if self.bundle.ready():
            output = self.bundle.rebundle()

        return output

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--g3', dest='g3file', type=str, required=True, help='full path to G3 file')
    parser.add_argument('--hk', dest='hkfile', type=str, required=True, help='full path to HK file')
    args = parser.parse_args()

    scanframes = [f for f in core.G3File(args.g3file) if f.type == core.G3FrameType.Scan]

    rf = DataReframer()

    hkframes = [h for h in core.G3File(args.hkfile)]
    acu_summary = [h for h in hkframes if h['hkagg_type'] == 2 and h['block_names'][0] == 'ACU_summary_output']

    for s in acu_summary:
        sc = locate_sign_changes(s['blocks'][0]['Azimuth_current_velocity'])
        tc = [int(s['blocks'][0]['ctime'][int(i)]*1e8) for i in sc]

        for t in tc:
            out = []
            rf.flush_time = t

            if rf.bundle is not None:
                rf.bundle.flush_time = t
            while len(scanframes) > 0 and len(out) == 0:
                out = rf.bundle.rebundle() if rf.ready() else rf(scanframes.pop(0))

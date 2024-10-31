import unittest


import numpy as np
from sotodlib import core
from sotodlib.tod_ops import flags


class TestFlags(unittest.TestCase):
    def test_trending(self):
        nsamps = 1000
        timestamps = np.arange(nsamps)
        signal = np.random.normal(size=(2, nsamps))
        signal[1] += timestamps
        dets = ["det0", "det1"]
        aman = core.AxisManager(
            core.LabelAxis("dets", dets), core.IndexAxis("samps", len(timestamps))
        )
        aman.wrap("timestamps", timestamps, [(0, "samps")])
        aman.wrap("signal", signal, [(0, "dets"), (1, "samps")])

        for max_size in (nsamps // 2, nsamps, nsamps * 2):
            cut = flags.get_trending_flags(aman, max_samples=max_size, max_trend=0.5)
            self.assertTupleEqual(cut.shape, (2, 1000))
            self.assertTrue(np.array_equal(cut.ranges[1].ranges(), [[0, 1000]]))
            self.assertEqual(len(cut.ranges[0].ranges()), 0)

        cut = flags.get_trending_flags(aman, max_trend=0.5, t_piece=333)
        self.assertTupleEqual(cut.shape, (2, 1000))
        self.assertTrue(np.array_equal(cut.ranges[1].ranges(), [[0, 1000]]))
        self.assertEqual(len(cut.ranges[0].ranges()), 0)


if __name__ == "__main__":
    unittest.main()

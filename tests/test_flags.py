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

    def test_get_glitch_flags(self):
        nsamps = 100000
        glitch_index = 50000
        timestamps = np.arange(nsamps)
        signal = np.random.normal(0, 1, size=(2, nsamps))
        signal[1][glitch_index] = 100
        dets = ["det0", "det1"]
        aman = core.AxisManager(
            core.LabelAxis("dets", dets), core.IndexAxis("samps", len(timestamps))
        )
        aman.wrap("timestamps", timestamps, [(0, "samps")])
        aman.wrap("signal", signal, [(0, "dets"), (1, "samps")])
        aman.wrap("flags", core.AxisManager(aman.dets, aman.samps))

        # test on normal signal
        flag = flags.get_glitch_flags(aman)
        self.assertTupleEqual(flag.shape, (2, nsamps))
        self.assertEqual(len(flag.ranges[1].ranges()), 1)
        self.assertEqual(flag[1].mask()[glitch_index], True)
        self.assertEqual(len(flag.ranges[0].ranges()), 0)

        # test on 1d array such as common mode
        aman.wrap("common_mode", signal[1], [(0, "samps")])
        flag = flags.get_glitch_flags(aman, signal_name='common_mode', name='common_glitches')
        self.assertTupleEqual(flag.shape, (nsamps,))
        self.assertEqual(len(flag.ranges()), 1)
        self.assertEqual(flag.mask()[glitch_index], True)


if __name__ == "__main__":
    unittest.main()

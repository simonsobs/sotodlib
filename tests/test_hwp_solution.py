""" Test the HWP angle solver
This tests part of site_pipeline.make_hwp_solutions.py
"""


import unittest
import numpy as np

from sotodlib import core
from sotodlib.hwp.g3thwp import G3tHWP


class HwpSolutionTest(unittest.TestCase):
    def test_empty_input(self):
        t = np.arange(0, 3600, 1/200)
        tod = core.AxisManager(core.OffsetAxis('samps', count=len(t)))
        tod.wrap('timestamps', t, [(0, 'samps')])

        solver = G3tHWP()  # default config
        solver.data = {}  # input empty data
        solution = solver.make_solution(tod)
        self.assertTrue(solution.logger_1 == 'No HWP data')
        self.assertTrue(solution.logger_2 == 'No HWP data')
        self.assertTrue(np.all(solution.hwp_angle == 0))

    def test_fake_input(self):
        period: int = 3600  # second
        bbbclk: float = 2e8  # Hz
        fhwp: float = 2.  # Hz
        num_edges: int = 1140  # number of edge per rotation

        t = np.arange(0, period, 1/200)
        tod = core.AxisManager(core.OffsetAxis('samps', count=len(t)))
        tod.wrap('timestamps', t, [(0, 'samps')])

        counter = np.cumsum(
            [bbbclk * 0.95 / num_edges / fhwp,
             bbbclk * 1.05 / num_edges / fhwp] * period * num_edges)
        index = np.arange(period * num_edges * fhwp)
        # reference slot
        counter = counter[(index % num_edges != 0) & (index % num_edges != 1)]
        time = counter / bbbclk * num_edges * fhwp
        slow_t = np.arange(period)

        data = {
            'counter_1': np.array([time, counter]),
            'counter_2': np.array([time, counter]),
            'counter_index_1': np.array([time, np.arange(len(counter))]),
            'counter_index_2': np.array([time, np.arange(len(counter))]),
            'irig_time_1': np.array([slow_t, slow_t]),
            'irig_time_2': np.array([slow_t, slow_t]),
            'rising_edge_count_1': np.array([slow_t, slow_t * bbbclk]),
            'rising_edge_count_2': np.array([slow_t, slow_t * bbbclk]),
            'quad_1': np.array([slow_t, [1] * period]),
            'quad_2': np.array([slow_t, [1] * period]),
            'approx_hwp_freq_1': np.array([slow_t, [fhwp] * period]),
            'approx_hwp_freq_2': np.array([slow_t, [fhwp] * period]),
            'pid_direction': np.array([slow_t, [1] * period]),
        }

        solver = G3tHWP()  # default config
        solver.data = data
        solution = solver.make_solution(tod)
        self.assertTrue(solution.logger_1 == 'Angle calculation succeeded')
        self.assertTrue(solution.logger_2 == 'Angle calculation succeeded')
        self.assertTrue(solution.version == 3)
        self.assertTrue(np.all(np.isfinite(solution.hwp_angle)))


if __name__ == '__main__':
    unittest.main()

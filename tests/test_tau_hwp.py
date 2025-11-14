import unittest
import numpy as np

from sotodlib import core
from sotodlib.hwp.hwp import get_tau_hwp


def get_hwp_spin_up_down(tod, spin_up, sign):
    """ Make spin up down hwp_angle """
    if spin_up:
        f0, f1 = 0, 2
    else:
        f0, f1 = 2, 0
    hwp_rate = np.linspace(f0, f1, tod.samps.count)
    hwp_angle = np.cumsum(np.ones(tod.samps.count) / 200
                          * sign * hwp_rate * 2 * np.pi)
    tod.wrap('hwp_angle', np.mod(hwp_angle, 2 * np.pi),
             axis_map=[(0, 'samps')])


def make_hwp_spin_up_down_tod(
    taus=np.array([1e-3, 2e-3]),
    A=3e-2,
    wn=5e-5,
    spin_up=True,
    sign=1,
):
    """
    Make an axis manager of hwp spin up down observation
    """
    t = np.arange(200 * 600)/200.  # 10 mins

    dets = [f'det{i}' for i in range(len(taus))]

    tod = core.AxisManager(core.LabelAxis('dets', vals=dets),
                           core.OffsetAxis('samps', count=len(t)))
    tod.wrap('timestamps', t, axis_map=[(0, 'samps')])
    get_hwp_spin_up_down(tod, spin_up, sign)

    hwp_rate = np.gradient(np.unwrap(tod.hwp_angle)) * 200
    signal = A * np.cos(4 * (tod.hwp_angle - hwp_rate * taus[:, None]))
    signal = np.float32(signal)
    np.random.seed(0)
    signal += np.random.normal(0, wn * np.sqrt(200),
                               (tod.dets.count, tod.samps.count))
    tod.wrap('signal', signal, axis_map=[(0, 'dets'), (1, 'samps')])
    return tod


class TauHWPTest(unittest.TestCase):
    """ Test the tau_hwp fitting functions """
    def test_fit(self):
        taus = np.array([1e-3, 2e-3])

        tod = make_hwp_spin_up_down_tod(taus=taus, spin_up=True, sign=1)
        result = get_tau_hwp(tod, full_output=True)
        self.assertAlmostEqual(result.tau_hwp[0], taus[0], delta=1e-4)
        self.assertAlmostEqual(result.tau_hwp[1], taus[1], delta=1e-4)

        tod = make_hwp_spin_up_down_tod(taus=taus, spin_up=False, sign=1)
        result = get_tau_hwp(tod)
        self.assertAlmostEqual(result.tau_hwp[0], taus[0], delta=1e-4)
        self.assertAlmostEqual(result.tau_hwp[1], taus[1], delta=1e-4)

        tod = make_hwp_spin_up_down_tod(taus=taus, spin_up=True, sign=-1)
        result = get_tau_hwp(tod)
        self.assertAlmostEqual(result.tau_hwp[0], taus[0], delta=1e-4)
        self.assertAlmostEqual(result.tau_hwp[1], taus[1], delta=1e-4)

        tod = make_hwp_spin_up_down_tod(taus=taus, spin_up=False, sign=-1)
        result = get_tau_hwp(tod)
        self.assertAlmostEqual(result.tau_hwp[0], taus[0], delta=1e-4)
        self.assertAlmostEqual(result.tau_hwp[1], taus[1], delta=1e-4)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from sotodlib import core, tod_ops
from sotodlib.tod_ops import t2pleakage
from sotodlib.hwp import hwp

def quick_dets_axis(n_det):
    return core.LabelAxis('dets', ['det%04i' % i for i in range(n_det)])

def gen_1_over_f(n_samp, dt, fknee):
    tmp = np.random.random(size=n_samp) * 2 - 1
    S = np.fft.rfft(tmp)

    fknee = 1
    freqs = np.fft.rfftfreq(n_samp, dt)
    freqs[0] = freqs[1]
    fil = 1 + (freqs / fknee) ** -2
    S = S * fil
    s = np.fft.irfft(S)
    s /= np.max(np.abs(s))
    return s

def quick_tod(n_det, n_samp, dt=.005, fknee=0.5, fhwp=2.0, t2q=1e-3, t2u=4e-3):
    dets = quick_dets_axis(n_det)
    tod = core.AxisManager(
        quick_dets_axis(n_det),
        core.OffsetAxis('samps', n_samp))
    timestamps = np.arange(n_samp) * dt
    
    tod.wrap_new('timestamps', shape=('samps', ))[:] = 1800000000. + timestamps
    tod.wrap_new('signal', shape=('dets', 'samps'), dtype='float32')
    
    _t = tod.timestamps - tod.timestamps[0]
    tod.wrap_new('hwp_angle',  shape=('samps', ))[:] = (2*np.pi*fhwp*_t + 1.32 + 1e-5*np.random.randn(len(_t))) % (np.pi*2)
    
    for di, det in enumerate(tod.dets.vals):
        atm = gen_1_over_f(n_samp,dt, fknee)
        atm += np.ptp(atm) * 5
        tod.signal[di] = atm * (1 + t2q*np.cos(4*tod.hwp_angle) + t2u*np.sin(4*tod.hwp_angle))
    
    return tod

class LeakageSubtractionTest(unittest.TestCase):
    "Test the temperature-to-polarization leakage subtaction"
    def test_leakage_subtraction(self):
        #prepare tod
        n_det = 10
        n_samp = 200*600
        dt=.005
        fknee=0.5
        fhwp=2.0
        t2q=1e-3
        t2u=4e-3
        
        tod = quick_tod(n_det=n_det, n_samp=n_samp, 
                        dt=dt, fknee=fknee, fhwp=fhwp,
                        t2q=t2q, t2u=t2u)
        
        hwp.demod_tod(tod)
        oman = t2pleakage.get_t2p_coeffs(tod)
        
        self.assertTrue(np.all(np.isclose(oman.coeffsQ, t2q, atol=oman.errorsQ*5, rtol=0)))
        self.assertTrue(np.all(np.isclose(oman.coeffsU, t2u, atol=oman.errorsU*5, rtol=0)))
        
if __name__ == '__main__':
    unittest.main()
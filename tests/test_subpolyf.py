import unittest
import numpy as np
import matplotlib.pyplot as plt
from sotodlib import core, tod_ops, sim_flags

def quick_dets_axis(n_det):
    return core.LabelAxis('dets', ['det%04i' % i for i in range(n_det)])

def quick_focal_plane(dets, scale=0.5):
    n_det = dets.count
    nrow = int(np.ceil((n_det/2)**2))
    entries = []
    gamma = 0.
    for r in range(nrow):
        y = r * scale
        for c in range(nrow):
            x = c * scale
            entries.extend([(x, y, gamma),
                            (x, y, gamma + 90)])
            gamma += 15.
    entries = entries[(n_det - len(entries)) // 2:]
    entries = entries[:n_det]
    fp = core.AxisManager(dets)
    for k, v in zip(['xi', 'eta', 'gamma'], np.transpose(entries)):
        v = (v - v.mean()) * np.pi/180
        fp.wrap(k, v, [(0, 'dets')])
    return fp

def quick_scan(tod, n_samp):
    az_min = 100.
    az_max = 120.
    v_az = 2.
    
    dt = tod.timestamps[1] - tod.timestamps[0]
    samps_1scan = (az_max - az_min) / (v_az * dt) # one scan duration
    num_scans = int(n_samp / samps_1scan)
    turnaround_idxes = np.linspace(0, n_samp, num_scans, dtype=int)
    turnaround_template = np.zeros(turnaround_idxes.shape)
    turnaround_template[::2] = az_min
    turnaround_template[1::2] = az_max
    az = np.interp(np.arange(0, n_samp), turnaround_idxes, turnaround_template)
    bs = core.AxisManager(tod.samps)
    bs.wrap_new('az'  , shape=('samps', ))[:] = az * np.pi/180
    bs.wrap_new('el'  , shape=('samps', ))[:] = 50 * np.pi/180
    bs.wrap_new('roll', shape=('samps', ))[:] = 0. * az
    return bs

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

def quick_tod(n_det, n_samp, dt=.005):
    dets = quick_dets_axis(n_det)
    tod = core.AxisManager(
        quick_dets_axis(n_det),
        core.OffsetAxis('samps', n_samp))
    tod.wrap('focal_plane', quick_focal_plane(tod.dets))
    timestamps = np.arange(n_samp) * dt
    
    tod.wrap_new('timestamps', shape=('samps', ))[:] = 1800000000. + timestamps
    tod.wrap('boresight', quick_scan(tod, n_samp=n_samp))
    tod.wrap_new('signal', shape=('dets', 'samps'), dtype='float32')
    
    tod.wrap('flags', core.FlagManager(tod.dets, tod.samps))
    fknee = 0.5
    for di, det in enumerate(tod.dets.vals):
        tod.signal[di] = gen_1_over_f(n_samp,dt, fknee)
    params = {'n_glitches': 10, 'sig_n_glitch': 10, 'h_glitch': 10,
              'sig_h_glitch': 2}
    sim_flags.add_random_glitches(tod, params=params, signal='signal',
                                  flag='glitches', overwrite=False)
    return tod

class SubpolyfTest(unittest.TestCase):
    "Test the subscan polyfilter functions"
    def test_subpolyf(self):
        # prepare tod
        n_det = 10
        n_samp = 200 * 60 * 3
        tod = quick_tod(n_det, n_samp)
        tod.boresight.az[:200*10] = tod.boresight.az[200*10] #non scanning part at the start of obs
        _ = tod_ops.flags.get_turnaround_flags(tod, truncate=False)
        tod_ori = tod.copy()
        
        # 
        for method in ['polyfit', 'legendre']:
            tod = tod_ori.copy()
            _ = tod_ops.sub_polyf.subscan_polyfilter(tod, degree=5, signal_name="signal", exclude_turnarounds=False,
                       mask='glitches', exclusive=True, method=method, in_place=True)
            
            masked_signal = np.ma.masked_array(tod.signal, tod.flags.glitches.mask())
            masked_signal_ori = np.ma.masked_array(tod_ori.signal, tod_ori.flags.glitches.mask())
            std_suppression = np.mean(np.std(masked_signal, axis=1) / np.std(masked_signal_ori, axis=1))
            print(f'({method}) std is suppressed by factor of {std_suppression:.2e}')
            self.assertTrue(std_suppression < 0.1)
            
if __name__ == '__main__':
    unittest.main()
    
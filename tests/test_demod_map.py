from sotodlib import core, hwp, coords
import so3g
import numpy as np

import unittest

# Support functions for a simple obs-like AxisManager...

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
        v = (v - v.mean()) * coords.DEG
        fp.wrap(k, v, [(0, 'dets')])
    return fp

def quick_scan(tod):
    az_min = 100.
    az_max = 120.
    v_az = 2.
    dt = tod.timestamps - tod.timestamps[0]
    az = az_min + (v_az * dt) % (az_max - az_min)
    bs = core.AxisManager(tod.samps)
    bs.wrap_new('az'  , shape=('samps', ))[:] = az * coords.DEG
    bs.wrap_new('el'  , shape=('samps', ))[:] = 50 * coords.DEG
    bs.wrap_new('roll', shape=('samps', ))[:] = 0. * az
    return bs

def quick_tod(n_det, n_samp):
    dets = quick_dets_axis(n_det)
    tod = core.AxisManager(
        quick_dets_axis(n_det),
        core.OffsetAxis('samps', n_samp))
    tod.wrap('focal_plane', quick_focal_plane(tod.dets))
    DT = .1
    dt = np.arange(n_samp) * DT
    tod.wrap_new('timestamps', shape=('samps', ))[:] = 1800000000. + dt
    f_hwp = 2.
    n_hwp = np.ceil(DT * n_samp * f_hwp)
    f_hwp = n_hwp / (DT * n_samp)
    v = DT * n_samp
    tod.wrap_new('hwp_angle',  shape=('samps', ))[:] = (dt * f_hwp + 1.32) % (np.pi*2)
    tod.wrap('boresight', quick_scan(tod))
    tod.wrap_new('signal', shape=('dets', 'samps'), dtype='float32')
    return tod


class DemodMapmakingTest(unittest.TestCase):
    def test_00_direct(self):
        """Test the coords.demod.make_map function on directly generated
        demodQ and demodU timestreams.

        """
        # Constant sky components.
        Q_stream = 1.0
        U_stream = 0.25
        TOL = 0.0001

        tod = quick_tod(10, 10000)
        fp = so3g.proj.FocalPlane.from_xieta(
            tod.dets.vals, tod.focal_plane.xi, tod.focal_plane.eta, tod.focal_plane.gamma)
        csl = so3g.proj.CelestialSightLine.az_el(
            tod.timestamps, tod.boresight.az, tod.boresight.el, roll=tod.boresight.roll,
            site='so_sat1', weather='toco')

        for k in ['dsT', 'demodQ', 'demodU']:
            tod.wrap_new(k, shape=('dets', 'samps'), dtype='float32')

        for i, det in enumerate(tod.dets.vals):
            q_total = csl.Q * fp[det]
            ra, dec, alpha = so3g.proj.quat.decompose_lonlat(q_total)
            GAMMA = alpha - 2 * tod.focal_plane.gamma[i]
            tod.demodQ[i] = Q_stream * np.cos(2 * GAMMA) + U_stream * np.sin(2 * GAMMA)
            tod.demodU[i] = U_stream * np.cos(2 * GAMMA) - Q_stream * np.sin(2 * GAMMA)

        results = coords.demod.make_map(tod)
        m0 = results['map']
        s = m0[1] != 0
        means = [m[s].mean() for m in m0]
        print(means)
        assert(abs(means[1] - Q_stream) < TOL)
        assert(abs(means[2] - U_stream) < TOL)

    def test_10_mod_demod(self):
        """Test the coords.demod.make_map function on timestreams starting
        from the HWP-modulated signal.

        """
        # Constant sky components.
        Q_stream = 1.0
        U_stream = 0.25
        TOL = .01

        tod = quick_tod(10, 10000)
        fp = so3g.proj.FocalPlane.from_xieta(
            tod.dets.vals, tod.focal_plane.xi, tod.focal_plane.eta, tod.focal_plane.gamma)
        csl = so3g.proj.CelestialSightLine.az_el(
            tod.timestamps, tod.boresight.az, tod.boresight.el, roll=tod.boresight.roll,
            site='so_sat1', weather='toco')

        c_4chi, s_4chi = np.cos(tod.hwp_angle * 4), np.sin(tod.hwp_angle * 4)
        for i, det in enumerate(tod.dets.vals):
            q_total = csl.Q * fp[det]
            ra, dec, alpha = so3g.proj.quat.decompose_lonlat(q_total)
            GAMMA = alpha - 2 * tod.focal_plane.gamma[i]
            c, s = np.cos(2*GAMMA), np.sin(2*GAMMA)
            tod.signal[i] = (Q_stream * (c*c_4chi - s*s_4chi) +
                             U_stream * (s*c_4chi + c*s_4chi))

        hwp.demod_tod(tod)
        results = coords.demod.make_map(tod)
        m0 = results['map']
        s = m0[1] != 0
        means = [m[s].mean() for m in m0]
        print(means)
        assert(abs(means[1] - Q_stream) < TOL)
        assert(abs(means[2] - U_stream) < TOL)
        
    def test_from_map_demodulated(self):
        """Test the coords.demod.from_map function of demodulated signal.

        """
        tod = quick_tod(10, 10000)
        TOL = 0.0001
        
        shape, wcs = enmap.fullsky_geometry(res=0.5*coords.DEG)
        signal_map = enmap.zeros((3, *shape), wcs)
        T_stream, Q_stream, U_stream = 1., 0.25, 0.01
        signal_map[0] += T_stream
        signal_map[1] += Q_stream
        signal_map[2] += U_stream
        _ = coords.demod.from_map(tod, signal_map, modulated=False, wrap=True)
        
        results = coords.demod.make_map(tod)
        m0 = results['map']
        s = m0[1] != 0
        means = [m[s].mean() for m in m0]
        assert(abs(means[1] - Q_stream) < TOL)
        assert(abs(means[2] - U_stream) < TOL)
        
    def test_from_map_modulated(self):
        """Test the coords.demod.from_map function of modulated signal.

        """
        tod = quick_tod(10, 10000)
        tod.move('signal', None)
        TOL = .01

        shape, wcs = enmap.fullsky_geometry(res=0.5*coords.DEG)
        signal_map = enmap.zeros((3, *shape), wcs)
        
        T_stream, Q_stream, U_stream = 1., 0.25, 0.01
        signal_map[0] += T_stream
        signal_map[1] += Q_stream
        signal_map[2] += U_stream

        _ = coords.demod.from_map(tod, signal_map, modulated=True, wrap=True)
        hwp.demod_tod(tod)
        results = coords.demod.make_map(tod)

        m0 = results['map']
        s = m0[1] != 0.
        means = [m[s].mean() for m in m0]
        assert(abs(means[1] - Q_stream) < TOL)
        assert(abs(means[2] - U_stream) < TOL)
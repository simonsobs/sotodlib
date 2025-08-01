from sotodlib import core, hwp, coords
import so3g
import numpy as np
from pixell import enmap
from ._helpers import quick_tod

import unittest

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
        fp  = coords.helpers.get_fplane(tod)
        csl = so3g.proj.CelestialSightLine.az_el(
            tod.timestamps, tod.boresight.az, tod.boresight.el, roll=tod.boresight.roll,
            site='so_sat1', weather='toco')

        for k in ['dsT', 'demodQ', 'demodU']:
            tod.wrap_new(k, shape=('dets', 'samps'), dtype='float32')

        for i, qdet in enumerate(tod.dets.vals):
            q_total = csl.Q * fp.quats[i]
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
        fp  = coords.helpers.get_fplane(tod)
        csl = so3g.proj.CelestialSightLine.az_el(
            tod.timestamps, tod.boresight.az, tod.boresight.el, roll=tod.boresight.roll,
            site='so_sat1', weather='toco')

        c_4chi, s_4chi = np.cos(tod.hwp_angle * 4), np.sin(tod.hwp_angle * 4)
        for i, det in enumerate(tod.dets.vals):
            q_total = csl.Q * fp.quats[i]
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

        # from_map expects and outputs float64
        _ = coords.demod.from_map(tod, signal_map, modulated=True, wrap=True)
        hwp.demod_tod(tod)
        # make_map expects float32
        for signal in ['signal', 'dsT', 'demodQ', 'demodU']:
           tod[signal] = tod[signal].astype('float32')
        results = coords.demod.make_map(tod)

        m0 = results['map']
        s = m0[1] != 0.
        means = [m[s].mean() for m in m0]
        assert(abs(means[1] - Q_stream) < TOL)
        assert(abs(means[2] - U_stream) < TOL)

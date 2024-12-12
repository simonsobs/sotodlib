# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of some coordinate routines.

"""

import itertools
import unittest
import numpy as np

from sotodlib import coords, core
from sotodlib.coords import optics as co
import so3g
from pixell import enmap

DEG = np.pi/180
CRVAL = [34.0*DEG, -20.9*DEG]
TOL_RAD = .00001 * DEG

def get_sightline():
    ra0, dec0, gamma0 = CRVAL[0], CRVAL[1], 0
    ra, dec, gamma = [np.array([_x]) for _x in [ra0, dec0, gamma0]]
    sight = so3g.proj.CelestialSightLine()
    sight.Q = (
        so3g.proj.quat.euler(2, ra) *
        so3g.proj.quat.euler(1, np.pi/2 - dec) *
        so3g.proj.quat.euler(2, gamma)
    )
    return sight


class FP:
    def __init__(self, xi, eta):
        self.xi = np.array(xi)
        self.eta = np.array(eta)


class FootprintTest(unittest.TestCase):
    def test_10_kernels(self):
        """Test creation of sensible WCS kernels for basic cases."""
        ra0, dec0 = CRVAL
        res = 0.01 * DEG

        # Test zenithal -- (ra0, dec0) is the reference point.
        for proj in ['TAN', 'ZEA']:
            wcsk = coords.get_wcs_kernel(proj, ra0, dec0, res)
            msg = f'Check crpix for {proj}'
            self.assertAlmostEqual(wcsk.wcs.crpix[0], 1, delta=TOL_RAD, msg=msg)
            self.assertAlmostEqual(wcsk.wcs.crpix[1], 1, delta=TOL_RAD, msg=msg)

        # Test cylindrical -- pixell puts the crval[1] on the equator
        # and dec0 is used for the conformal latitude.
        for proj in ['CAR', 'CEA']:
            wcsk = coords.get_wcs_kernel(proj, ra0, dec0, res)
            msg = f'Check crpix for {proj}'
            self.assertAlmostEqual(wcsk.wcs.crpix[0], 1, delta=TOL_RAD, msg=msg)
            self.assertNotAlmostEqual(wcsk.wcs.crpix[1], 1, delta=TOL_RAD, msg=msg)

        # This is going to break.
        fp = FP(xi =[0., -0.01*DEG],
                eta=[0., -0.01*DEG])
        sight = get_sightline()
        tod = core.AxisManager(core.LabelAxis('dets', ['a']))
        fp = coords.get_footprint(tod, wcs_kernel=wcsk, focal_plane=fp, sight=sight)

    def test_20_supergeom_simple(self):
        """Check that coords.get_supergeom does sensible things in simple cases."""
        for proj in ['TAN', 'CEA']:
            ra0, dec0 = CRVAL
            res = 0.01 * DEG
            wcs = coords.get_wcs_kernel(proj, ra0, dec0, res)

            wcs.wcs.crpix = (60, 70)
            map0 = enmap.zeros((100,200), wcs=wcs)
            map0[2, 3] = 10.
            map0[90, 192] = 11.

            # Extracts.
            m1 = map0[:10,:10]
            m2 = map0[-10:,-10:]
            
            # Reconstruct.
            sg = coords.get_supergeom((m1.shape, m1.wcs), (m2.shape, m2.wcs))
            mapx = enmap.zeros(*sg)
            mapx.insert(m1)
            mapx.insert(m2)
            self.assertTupleEqual(map0.shape, mapx.shape)
            self.assertTrue(np.all(mapx==map0))

    def test_30_supergeom_translate(self):
        """Check that coords.get_supergeom does sensible thing for maps in
        cylindrical projections with compatible but not identical
        crval.

        """
        proj = 'CAR'
        ra0, dec0 = CRVAL
        res = 0.01 * DEG
        wcs = coords.get_wcs_kernel(proj, ra0, dec0, res)

        wcs.wcs.crpix = (60, 70)
        map0 = enmap.zeros((100,200), wcs=wcs)
        map0[2, 3] = 10.
        map0[90, 192] = 11.

        # Extracts.
        m1 = map0[:10,:10]
        m2 = map0[-10:,-10:]

        # In simple cylindrical projections, there's a degeneracy
        # between crval and crpix in the longitude component -- crval
        # can be anywhere on the equator.  It is useful to be able to
        # join maps even if they have different crval[0], provided the
        # pixel centers line up.  (The same is not true of crval[1],
        # which tips the native equator relative to the celestial
        # equator.)

        for axis, should_work in [(0, True), (1, False)]:
            dpix = 10.5
            m2 = map0[-10:,-10:]
            m2.wcs.wcs.crpix[axis] += dpix
            m2.wcs.wcs.crval[axis] += dpix * m2.wcs.wcs.cdelt[axis]

            if should_work:
                sg = coords.get_supergeom((m1.shape, m1.wcs), (m2.shape, m2.wcs))
                mapx = enmap.zeros(*sg)
                mapx.insert(m1)
                mapx.insert(m2)
                self.assertTupleEqual(map0.shape, mapx.shape,
                                      msg="Reconstructed map shape.")
                self.assertTrue(np.all(mapx==map0),
                                msg="Reconstructed map data.")

            else:
                msg = "Translating crval in dec should cause "\
                    "coord consistency check failure."
                with self.assertRaises(ValueError, msg=msg):
                    sg = coords.get_supergeom((m1.shape, m1.wcs), (m2.shape, m2.wcs))

class CoordsUtilsTest(unittest.TestCase):
    def test_valid_arg(self):
        from sotodlib.coords.helpers import _valid_arg
        tod = core.AxisManager()
        tod.wrap('a', np.array([1,2,3]))
        self.assertIs(_valid_arg(None, 'a', src=tod), tod.a)
        self.assertIs(_valid_arg('a', None, src=tod), tod.a)
        self.assertIs(_valid_arg(None, tod.a), tod.a)
        self.assertIs(_valid_arg(tod.get('b')), None)
        self.assertIs(_valid_arg(tod.get('b'), 'a', src=tod), tod.a)

    def test_scalar_last_quat(self):
        test_array = np.array([[2,3,4,1],[90,100,23,14]])

        # Convert one quat
        qa = coords.ScalarLastQuat(test_array[0])
        self.assertIsInstance(qa, np.ndarray)
        q3 = qa.to_g3()
        self.assertIsInstance(q3, so3g.proj.quat.quat)
        self.assertEqual(q3.a, 1)
        qb = coords.ScalarLastQuat(q3)
        np.testing.assert_array_equal(qa, qb)

        # Convert a vector of quats
        qa = coords.ScalarLastQuat(test_array)
        v3 = qa.to_g3()
        self.assertIsInstance(v3, so3g.proj.quat.G3VectorQuat)
        self.assertEqual(v3[0].a, 1)
        self.assertEqual(v3[1].a, 14)
        qb = coords.ScalarLastQuat(v3)
        np.testing.assert_array_equal(qa, qb)

    def test_cover(self):
        x0, y0 = 1*DEG, 4*DEG
        R = 5*DEG
        x = np.linspace(-R, R, 50)
        y = np.linspace(-R, R, 45)
        xy = np.transpose(list(itertools.product(x, y)))
        s = xy[0]**2 + xy[1]**2 < R**2
        
        xy = xy[:,s] + np.array([x0, y0])[:,None]
        (xi0, eta0), R0, (xi, eta) = \
            coords.helpers.get_focal_plane_cover(count=16, xieta=xy)
        np.testing.assert_allclose([xi0, eta0, R], [x0, y0, R0],
                                   atol=R*0.05)
        self.assertEqual(len(xi), 16)

        # Works with nans?
        xy[0,0] = np.nan
        coords.helpers.get_focal_plane_cover(xieta=xy)

        # Exclude dets using det_weights?
        det_weights = np.ones(xy.shape[1])
        det_weights[3:34] = 0.
        for dtype in ['float', 'int', 'bool']:
            coords.helpers.get_focal_plane_cover(
                xieta=xy, det_weights=det_weights.astype(dtype))

        # Works for only a single det?
        det_weights[2:] = 0.
        (xi0, eta0), R0, _ = \
            coords.helpers.get_focal_plane_cover(xieta=xy, det_weights=det_weights)

        # Fails if all dets excluded somehow?
        det_weights[1] = 0.
        with self.assertRaises(ValueError):
            coords.helpers.get_focal_plane_cover(xieta=xy, det_weights=det_weights)

        # Fails with all nans?
        xy[1,1:] = np.nan
        with self.assertRaises(ValueError):
            coords.helpers.get_focal_plane_cover(xieta=xy)
            

class OpticsTest(unittest.TestCase):
    def test_sat_fp(self):
        x = np.array([-100, 0, 100]) 
        y = x.copy()
        pol = x.copy()

        xi, eta, gamma = co.get_focal_plane(None, x, y, pol, 0, "SAT", "ws1", ufm_to_fp_pars={'theta': 60.0, 'dx': 0.0, 'dy': 128.5})
        self.assertTrue(np.all(np.isclose(xi, np.array([-6.4406e-02,  0,  5.58489e-02]))))
        self.assertTrue(np.all(np.isclose(eta, np.array([0.01425728, -0.2207397, -0.404499]))))
        self.assertTrue(np.all(np.isclose(gamma, np.array([5.409, 3.6846, 1.8156]))))

if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of some coordinate routines.

"""

import unittest
import numpy as np

from sotodlib import coords, core
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


if __name__ == '__main__':
    unittest.main()

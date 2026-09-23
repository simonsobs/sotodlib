import unittest
from sotodlib import coords
from pixell import enmap, tilemap
from sotodlib.coords import healpix_utils as hp_utils
from ._helpers import quick_tod
import numpy as np
import so3g
try:
    import healpy as hp
except:
    hp = False

class PmatTest(unittest.TestCase):
    """ Test coords.pmat.P.

    Check that for_tod, to_map, to_weights, to_inverse_weights, remove_weights and from_map
    all run without crashing, that maps computed with different settings match, and that
    from_map recovers a constant input signal.
    """

    def test_pmat_rectpix(self):
        obs = quick_tod(10, 10000)
        obs.signal[:,:] = 1
        shape, wcs = enmap.fullsky_geometry(res=0.5*coords.DEG)
        comps = 'T'
        out = run_test(obs, (shape, wcs), comps, None, False, False) # Basic
        _ = run_test(obs, None, comps, wcs, False, False) # Use wcs_kernel
        out2 = run_test(obs, tilemap.geometry(shape, wcs, tile_shape=(100, 100)), comps, None, False, True) # Tiled
        assert np.array_equal(out, out2)

    def test_pmat_healpix(self):
        obs = quick_tod(10, 10000)
        obs.signal[:,:] = 1
        nside = 64
        comps = 'T'
        out = run_test(obs, hp_utils.get_geometry(nside), comps, None, True, False) # Basic un-tiled
        out2 = run_test(obs, hp_utils.get_geometry(nside, 4), comps, None, True, True) # Explicit tiling
        assert np.array_equal(out, out2)
        out2 = run_test(obs, hp_utils.get_geometry(nside, 'auto'), comps, None, True, True) # Auto tiling
        assert np.array_equal(out, out2)
        out2 = run_test(obs, None, comps, f'nside={nside}', True, False) # Basic un-tiled with wcs_kernel
        assert np.array_equal(out, out2)
        out2 = run_test(obs, hp_utils.get_geometry(nside, None, 'RING'), comps, None, True, False) # Un-tiled ring
        if hp:
            out2 = hp.reorder(out2, r2n=True)
            assert np.array_equal(out, out2)

    def test_pmat_instrument_centered(self):
        obs = quick_tod(10, 2000)
        # Put dets 0 and 1 at the boresight. Dets 2 and 3 share a
        # position but have different gamma.
        obs.focal_plane.xi[:2] = 0.
        obs.focal_plane.eta[:2] = 0.
        assert obs.focal_plane.xi[2] == obs.focal_plane.xi[3]
        assert obs.focal_plane.eta[2] == obs.focal_plane.eta[3]
        assert obs.focal_plane.gamma[2] != obs.focal_plane.gamma[3]
        geom = enmap.fullsky_geometry(res=0.5*coords.DEG)
        source = 'moon'

        with self.assertRaises(ValueError):
            coords.pmat.P.for_tod(obs, geom=geom, comps='TQU',
                                  instrument_centered=source)

        pmat = coords.pmat.P.for_tod(obs, geom=geom, comps='T', weather='vacuum',
                                     instrument_centered=source)
        assert pmat.det_left
        ic = np.array(pmat._get_proj().get_coords(pmat._get_asm()))
        lon, lat = ic[..., 0], ic[..., 1]

        # Positions don't depend on the polarization angle.
        np.testing.assert_allclose(ic[2, :, :2], ic[3, :, :2], atol=1e-10)

        # The distance from the origin must match the angular distance
        # between each detector and the source on the sky.
        pmat0 = coords.pmat.P.for_tod(obs, geom=geom, comps='T', weather='vacuum')
        radec = np.array(pmat0._get_proj().get_coords(pmat0._get_asm()))
        ra, dec = radec[..., 0], radec[..., 1]
        q_src = coords.planets.get_source_quat(source, obs.timestamps, site='so')
        ra_s, dec_s, _ = so3g.proj.quat.decompose_lonlat(q_src)
        sep_sky = np.arccos(np.clip(np.sin(dec) * np.sin(dec_s) + np.cos(dec)
                                    * np.cos(dec_s) * np.cos(ra - ra_s), -1, 1))
        sep_ic = np.arccos(np.clip(np.cos(lat) * np.cos(lon), -1, 1))
        np.testing.assert_allclose(sep_ic, sep_sky, atol=1e-5)

        # Orientation, for the boresight detector (roll = 0): lat
        # increases towards higher elevation, lon towards lower azimuth.
        def vec(az, el):
            return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        for i in [0, 500, 1000, 1999]:
            az_b, el_b = obs.boresight.az[i], obs.boresight.el[i]
            az_s, el_s, _ = coords.planets.get_source_azel(
                source, obs.timestamps[i], site='so')
            m = vec(az_s, el_s)
            u_el = np.array([-np.sin(el_b)*np.cos(az_b), -np.sin(el_b)*np.sin(az_b), np.cos(el_b)])
            u_az = np.array([-np.sin(az_b), np.cos(az_b), 0.])
            TOL = 1e-3
            assert abs(np.sin(lat[0, i]) - m @ u_el) < TOL
            assert abs(np.cos(lat[0, i]) * np.sin(lon[0, i]) + m @ u_az) < TOL

        # Projection operations run.
        mask = enmap.zeros((1,) + tuple(geom[0]), geom[1])
        mask[:] = 1.
        tod = pmat.from_map(mask)
        np.testing.assert_allclose(tod, 1.)
        m = pmat.to_map(obs, signal=tod)
        assert np.sum(m) == tod.size

def run_test(obs, geom, comps, wcs_kernel, is_healpix, is_tiled):
    pmat = coords.pmat.P.for_tod(obs, comps=comps, geom=geom, wcs_kernel=wcs_kernel)
    assert pmat.tiled == is_tiled # Check tiled flag works

    imap = pmat.to_map(obs)
    ncomp = len(comps)
    zeros = pmat.zeros((ncomp,ncomp))
    weights = pmat.to_weights(obs, dest=zeros)
    iweights = pmat.to_inverse_weights(weights)
    remove_weights = pmat.remove_weights(tod=obs)
    tod = pmat.from_map(remove_weights)
    TOL = 1e-9
    assert np.all(np.abs(tod-obs.signal) < TOL)

    # Confirm we can do map-space ops without a pointing op first
    pmat = coords.pmat.P.for_tod(obs, comps=comps, geom=geom, wcs_kernel=wcs_kernel)
    _ = pmat.to_inverse_weights(weights)
    pmat = coords.pmat.P.for_tod(obs, comps=comps, geom=geom, wcs_kernel=wcs_kernel)
    _ = pmat.remove_weights(weights)

    # Confirm from_map works on uninitialized pmat
    pmat = coords.pmat.P.for_tod(obs, comps=comps, geom=geom, wcs_kernel=wcs_kernel)
    tod2 = pmat.from_map(remove_weights)
    assert np.all(np.abs(tod - tod2) < TOL)

    # And also zeros.
    pmat = coords.pmat.P.for_tod(obs, comps=comps, geom=geom, wcs_kernel=wcs_kernel)
    pmat.zeros()

    if is_tiled:
        if is_healpix:
            remove_weights = hp_utils.tiled_to_full(remove_weights)
        else:
            remove_weights = tilemap.to_enmap(remove_weights)
    return remove_weights

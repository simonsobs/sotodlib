import unittest
from sotodlib import coords
from pixell import enmap, tilemap
from sotodlib.coords import healpix_utils as hp_utils
from ._helpers import quick_tod
import numpy as np
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

    if is_tiled:
        if is_healpix:
            remove_weights = hp_utils.tiled_to_full(remove_weights)
        else:
            remove_weights = tilemap.to_enmap(remove_weights)
    return remove_weights

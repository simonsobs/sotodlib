import numpy as np
import so3g
import unittest
from sotodlib import coords, core
from pixell import enmap
from sotodlib.mapmaking import DemodSignalMap, DemodMapmaker, NmatWhite
from ._helpers import quick_tod

class DemodMapmakerTest(unittest.TestCase):
    """ Test mapmaking.demod_mapmaker.

    Check that we can make a weighted map and weights map, and that de-weighting
    recovers a constant input signal.
    """
    def test_make_map_healpix(self):
        Q_stream = 1.0
        U_stream = 0.25
        obs = get_tod(Q_stream, U_stream)

        nside = 128
        for nside_tile in [16, None, 'auto']:
            imap = make_map(obs, nside=nside, nside_tile=nside_tile)
            check_map(imap, Q_stream, U_stream, TOL=1e-9)

    def test_make_map_rectpix(self):
        Q_stream = 1.0
        U_stream = 0.25
        obs = get_tod(Q_stream, U_stream)

        shape, wcs = enmap.fullsky_geometry(res=0.5*coords.DEG)

        imap = make_map(obs, shape=shape, wcs=wcs)
        check_map(imap, Q_stream, U_stream, TOL=1e-9)

        # Try calling get_footprint
        imap = make_map(obs, shape=None, wcs=wcs)
        check_map(imap, Q_stream, U_stream, TOL=1e-9)


def make_map(obs, nside=None, nside_tile=None, shape=None, wcs=None, comps='TQU'):
    if nside is not None:
        signal_map = DemodSignalMap.for_healpix(nside, nside_tile, comps=comps)
    else:
        signal_map = DemodSignalMap.for_rectpix(shape, wcs, None, comps=comps)

    signals    = [signal_map]
    mapmaker   = DemodMapmaker(signals, noise_model=NmatWhite(), comps=comps)
    mapmaker.add_obs('obs0', obs)
    imap = unweight_map(signal_map.rhs[0], signal_map.div[0])
    return imap

def check_map(imap, Q_stream, U_stream, TOL):
    assert not np.any(imap[0])
    imap = imap[1:]
    imap[imap == 0] = np.nan
    axis = tuple([ii for ii in range(1, imap.ndim)])
    means = np.nanmean(imap, axis=axis)
    means[np.isnan(means)] = 0
    assert np.all(np.abs(means - np.array([Q_stream, U_stream])) < TOL)

def unweight_map(rhs, div):
    div_diag = np.moveaxis(div.diagonal(), -1, 0)
    idiv = np.zeros_like(div_diag)
    pos = np.where(div_diag > 0)
    idiv[pos] = 1/div_diag[pos]
    return rhs * idiv

def get_tod(Q_stream, U_stream):
        tod = quick_tod(10, 10000)
        tod.wrap('flags', core.FlagManager.for_tod(tod))
        tod.wrap('weather', np.full(1, 'toco'))
        tod.wrap('site', np.full(1, 'so_sat1'))
        tod.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(tod.shape), [(0, 'dets'), (1, 'samps')])

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
        return tod

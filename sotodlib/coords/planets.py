import datetime
import logging
import time

import numpy as np
from scipy.optimize import fmin

from astropy import units
from astropy.utils import data as au_data
from skyfield.api import load, wgs84, utc

import so3g
from pixell import enmap

from .. import tod_ops, coords

logger = logging.getLogger(__name__)

# Files that we might want to download and cache using
# astropy.utils.data
RESOURCE_URLS = {
    'de421.bsp': 'ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp',
}


class SlowSource:
    def __init__(self, timestamp, ra, dec, v_ra=0., v_dec=0.,
                 precision=.0001 * coords.DEG):
        self.timestamp = timestamp
        self.ra = ra
        self.dec = dec
        self.v_ra = v_ra
        self.v_dec = v_dec

    @classmethod
    def for_named_source(cls, name, timestamp):
        dt = 3600
        ra0, dec0, distance = get_source_pos(name, timestamp)
        ra1, dec1, distance = get_source_pos(name, timestamp + dt)
        return cls(timestamp, ra0, dec0, ((ra1-ra0+180) % 360 - 180)/dt, (dec1-dec0)/dt)

    def pos(self, times):
        if not isinstance(times, np.ndarray):
            ra, dec = self.pos(np.array([times]))
            return ra[0], dec[0]
        dt = times - self.timestamp
        return self.ra + self.v_ra * dt, self.dec + self.v_dec * dt

def model(t, az, el, planet):
    csl = so3g.proj.CelestialSightLine.az_el(t, az, el, weather='typical', site='so')
    ra0, dec0 = planet.pos(t)
    return csl.Q, ~so3g.proj.quat.rotation_lonlat(ra0, dec0) * csl.Q

def get_scan_q(tod, planet, refq=None):
    """Identify the point (in time and azimuth space) at which the
    specified planet crosses the boresight elevation for the
    observation in tod.  The rotation taking boresight coordinates to
    celestial coordinates at that moment in time (and pointing) is
    computed, and its conjugate is returned.

    The returned rotation is useful because it represents a fixed way
    to rotate the celestial such that the target source ends up at
    (xi, eta) = 0, with the telescope scan direction parallel to xi
    and elevation parallel to eta; this is a useful system for
    measuring beam and pointing parameters.

    """
    if isinstance(planet, str):
        planet = SlowSource.for_named_sources(planet)
    if refq is None:
        refq = so3g.proj.quat.rotation_xieta(0, 0)
    # Get reference elevation...
    el = np.median(tod.boresight.el[::10])
    az = np.median(tod.boresight.az[::10])
    t = (tod.timestamps[0] + tod.timestamps[-1]) / 2
    def distance(p):
        dt, daz = p
        q, qnet = model(t + dt, az + daz, el, planet)
        lon, lat, phi = so3g.proj.quat.decompose_lonlat(qnet)
        return 90 * coords.DEG - lat
    p0 = np.array([0, 0])
    X = fmin(distance, p0, disp=0, full_output=1)
    p, fopt, n_iter, n_calls, warnflag = X
    if warnflag != 0:
        print('Warnflag!')
    q, qnet = model(t+p[0], az+p[1], el, planet)
    psi = so3g.proj.quat.decompose_xieta(qnet)[2][0]
    ra, dec = planet.pos(t+p[0])
    rot = ~so3g.proj.quat.rotation_lonlat(ra, dec, psi)
    return {'rot': rot,
            'timestamp': t+p[0],
            'az': az+p[1],
            'el': el,
            'ra': ra,
            'dec': dec,
            'psi': psi,
            'planet': planet}

def get_scan_P(tod, planet, refq=None, res=0.01*coords.DEG, comps='TQU'):
    """Get a standard Projection Matrix targeting a planet (or some
    interesting fixed position), in source-scan coordinates.

    Returns a Projection Matrix and the output from get_scan_q.

    """
    X = get_scan_q(tod, planet)
    rot = so3g.proj.quat.rotation_lonlat(0, 0) * X['rot']
    wcs_kernel = coords.get_wcs_kernel('tan', 0., 0., res=res)
    P = coords.P.for_tod(tod, comps=comps, rot=rot, wcs_kernel=wcs_kernel)
    return P, X

def filter_for_sources(tod=None, signal=None, source_flags=None,
                       wrap=None):
    """Mask and gap-fill the signal at samples flagged by source_flags.
    Then PCA the resulting time ordered data.  Restore the flagged
    signal, remove the strongest modes from PCA.

    The operations are done in place, so if you want to preserve
    tod.signal, pass in signal=tod.signal.copy().

    Args:
      tod: AxisManager from which defaults will be drawn.
      signal: Time-ordered data to operate on.  Defaults to
        tod.signal.
      source_flags: RangesMatrix to use for source flagging.  Defaults
        to tod.source_flags.
      wrap (str): If specified, the result will be stored at
        tod[wrap].

    Returns:
      The filtered signal.

    """
    if source_flags is None:
        source_flags = tod.source_flags
    if signal is None:
        signal = tod.signal

    # Get a reasonable gap fill.
    gaps = tod_ops.get_gap_fill(tod, signal=tod.signal, flags=source_flags)
    gaps.swap(tod, signal=signal)

    # Measure TOD means (after gap fill).
    levels = signal.mean(axis=1)

    # Clean the means, get PCA model, restore means.
    tod.signal -= levels[:,None]
    pca = tod_ops.pca.get_pca_model(tod, signal=signal, n_modes=10)
    tod.signal += levels[:,None]

    # Swap original data back into the TOD, remove means.
    gaps.swap(tod, signal=signal)
    tod.signal -= levels[:,None]

    # Remove the PCA model.
    tod_ops.pca.add_model(tod, pca, -1, signal=signal)

    if wrap:
        tod.wrap(wrap, signal, [(0, 'dets'), (1, 'samps')])
    return signal

def get_source_pos(src, timestamp):
    # Get the ephemeris -- this will trigger a 16M download on first use.
    de_url = RESOURCE_URLS['de421.bsp']
    de_filename = au_data.download_file(de_url, cache=True)

    planets = load(de_filename)
    try:
        target = planets[src]
    except KeyError:
        target = planets[src + ' barycenter']

    earth = planets['earth']
    site = so3g.proj.SITES['so']
    so = earth + wgs84.latlon(site.lat, site.lon, site.elev)

    ts = load.timescale()
    dt = datetime.datetime.fromtimestamp(timestamp, tz=utc)
    t0 = ts.from_datetime(dt)
    amet0 = so.at(t0).observe(target)
    ra, dec, distance = amet0.radec()
    return ra.to(units.rad).value, dec.to(units.rad).value, distance.to(units.au).value

import datetime
import logging
import time

import numpy as np
from scipy.optimize import fmin

from astropy import units
from astropy.utils import data as au_data
from skyfield import jpllib, api as skyfield_api

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
    if refq is None:
        refq = so3g.proj.quat.rotation_xieta(0, 0)
    # Get reference elevation...
    el = np.median(tod.boresight.el[::10])
    az = np.median(tod.boresight.az[::10])
    t = (tod.timestamps[0] + tod.timestamps[-1]) / 2
    if isinstance(planet, str):
        planet = SlowSource.for_named_source(planet, t)
    def distance(p):
        dt, daz = p
        q, qnet = model(t + dt, az + daz, el, planet)
        lon, lat, phi = so3g.proj.quat.decompose_lonlat(qnet)
        return 90 * coords.DEG - lat
    p0 = np.array([0, 0])
    X = fmin(distance, p0, disp=0, full_output=1)
    p, fopt, n_iter, n_calls, warnflag = X
    if warnflag != 0:
        logger.warning('Source-scan solver failed to converge or otherwise '
                       f'complained!  warnflag={warnflag}')
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

def get_scan_P(tod, planet, refq=None, res=0.01*coords.DEG, **kw):
    """Get a standard Projection Matrix targeting a planet (or some
    interesting fixed position), in source-scan coordinates.

    Returns a Projection Matrix and the output from get_scan_q.

    """
    X = get_scan_q(tod, planet)
    rot = so3g.proj.quat.rotation_lonlat(0, 0) * X['rot']
    wcs_kernel = coords.get_wcs_kernel('tan', 0., 0., res=res)
    P = coords.P.for_tod(tod, rot=rot, wcs_kernel=wcs_kernel, **kw)
    return P, X

def filter_for_sources(tod=None, signal=None, source_flags=None,
                       wrap=None):
    """Mask and gap-fill the signal at samples flagged by source_flags.
    Then PCA the resulting time ordered data.  Restore the flagged
    signal, remove the strongest modes from PCA.

    If signal is not passed in tod.signal will be modified directly.
    To leave tod.signal intact, pass in signal=tod.signal.copy().

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
    signal -= levels[:,None]
    pca = tod_ops.pca.get_pca_model(tod, signal=signal, n_modes=10)
    signal += levels[:,None]

    # Swap original data back into the TOD, remove means.
    gaps.swap(tod, signal=signal)
    signal -= levels[:,None]

    # Remove the PCA model.
    tod_ops.pca.add_model(tod, pca, -1, signal=signal)

    if wrap:
        tod.wrap(wrap, signal, [(0, 'dets'), (1, 'samps')])
    return signal

def get_source_pos(src, timestamp, site='_default'):
    # Get the ephemeris -- this will trigger a 16M download on first use.
    de_url = RESOURCE_URLS['de421.bsp']
    de_filename = au_data.download_file(de_url, cache=True)

    planets = jpllib.SpiceKernel(de_filename)
    try:
        target = planets[src]
    except KeyError:
        target = planets[src + ' barycenter']

    if isinstance(site, str):
        site = so3g.proj.SITES[site]

    observatory = site.skyfield_site(planets)

    timescale = skyfield_api.load.timescale()
    sf_timestamp = timescale.from_datetime(
        datetime.datetime.fromtimestamp(timestamp, tz=skyfield_api.utc))
    amet0 = observatory.at(sf_timestamp).observe(target)
    ra, dec, distance = amet0.radec()
    return ra.to(units.rad).value, dec.to(units.rad).value, distance.to(units.au).value


def make_map(tod, center_on=None, scan_coords=True, thread_algo=False,
             res=0.01*coords.DEG, wcs_kernel=None, comps='TQU',
             filename=None, source_flags=None, cuts=None,
             eigentol=1e-3, inplace=False, info={}):
    """Make a compact source map from the TOD.  Specify filename to write
    things to disk; this should be a format string, for example
    '{obs_id}_{map}.fits', where 'map' will be given values of
    ['binned', 'solved', 'weights'] and any other keys (obs_id in this
    case) must be passed through info.

    """
    assert(center_on is not None)  # Pass in the source name, e.g. 'uranus'

    # Test the map format string...
    if filename is not None:
        try:
            filename.format(map='binned', **info)
        except:
            raise ValueError('Failed to process filename format "%s" with info=%s' %
                             (filename, info))

    class MmTimer(coords.Timer):
        PRINT_FUNC = logger.info
        FMT = 'make_map: {msg}: {elapsed:.3f} seconds'

    with MmTimer('filter for sources'):
        if inplace:
            signal = tod.signal
        else:
            signal = tod.signal.copy()
        filter_for_sources(tod, signal=signal, source_flags=source_flags)

    if thread_algo == 'none':
        thread_algo = False

    with MmTimer('setup thread_algo=%s' % thread_algo):
        if scan_coords:
            P, X = coords.planets.get_scan_P(tod, center_on, res=res,
                                             comps=comps, threads=thread_algo)
        else:
            planet = coords.planets.SlowSource.for_named_source(center_on, tod.timestamps[0])
            ra0, dec0 = planet.pos(tod.timestamps.mean())
            wcsk = coords.get_wcs_kernel('tan', ra0, dec0, res=res)
            P = coords.P.for_tod(tod, comps=comps,
                                 threads=thread_algo,
                                 wcs_kernel=wcsk)
        P._get_proj_threads()

    with MmTimer('project signal and weight maps'):
        map1 = P.to_map(tod, signal=signal)
        wmap1 = P.to_weights()

    with MmTimer('compute and apply inverse weights map'):
        iwmap1 = P.to_inverse_weights(weights_map=wmap1, eigentol=eigentol)
        map1b = P.remove_weights(map1, inverse_weights_map=iwmap1)

    if filename is not None:
        with MmTimer('Write out'):
            map1b.write(filename.format(map='solved', **info))
            if filename.format(map='x', **info) != filename.format(map='y', **info):
                map1.write(filename.format(map='binned', **info))
                wmap1.write(filename.format(map='weights', **info))

    return {'binned': map1,
            'weights': wmap1,
            'solved': map1b}

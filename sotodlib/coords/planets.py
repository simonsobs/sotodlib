import re
import datetime
import logging

import numpy as np

from scipy.optimize import fmin

from astropy import units
from astropy.utils import data as au_data
from skyfield import jpllib, api as skyfield_api

import so3g
from pixell import enmap

from .. import core, tod_ops, coords

logger = logging.getLogger(__name__)


# Default source list
SOURCE_LIST = ['mercury',
               'venus',
               'moon',
               'mars',
               'jupiter',
               'saturn',
               'uranus',
               'neptune',
               ('tauA', 83.6272579, 22.02159891),
               ('rcw38', 134.7805107, -47.50911231),
               ('iras16183-4958', 245.5154, -50.09168292),
               ('iras19078+0901', 287.5575891, 9.107188994),
               ('rcw122', 260.0339538, -38.95673421),
               ('cenA', 201.3625336, -43.00797508),
               ('3c279', 194.0409868, -5.79174024),
               ('3c273', 187.2775626, 2.053532671),
               ('G025.36-00.14', 279.5264042, -6.793169326),
               ('QSO_J2253+1608', 343.4952422, 16.14301323),
               ('galactic_center', -93.5833, -29.0078)]


class SlowSource:
    """Class to track the time-dependent position of a slow-moving source,
    such as a Solar System planet in equatorial coordinates.
    "Slow-moving" is in relation to the time range of interest.  A
    linear approximation is good enough for Solar System, arcsecond
    accuracy, on time scales of a few hours.

    Args:
      timestamp (float): reference time, as a unix timestamp.
      ra (float): Right Ascension at reference time, in radians.
      dec (float): Declination at reference time, in radians.
      v_ra (float): rate of change of RA, in radians.
      v_dec (float): rate of change of dec, in radians.
      precision: not implemented, don't worry about it.

    """

    def __init__(self, timestamp, ra, dec, v_ra=0., v_dec=0.,
                 precision=.0001 * coords.DEG):
        self.timestamp = timestamp
        self.ra = ra
        self.dec = dec
        self.v_ra = v_ra
        self.v_dec = v_dec

    @classmethod
    def for_named_source(cls, name, timestamp):
        """Returns a SlowSource for planet ``name``, with position and
        peculiar velocity measured at time timestamp (float, unix).

        """
        dt = 3600
        ra0, dec0, distance = get_source_pos(name, timestamp)
        ra1, dec1, distance = get_source_pos(name, timestamp + dt)
        return cls(timestamp, ra0, dec0, ((ra1-ra0+180) % 360 - 180)/dt, (dec1-dec0)/dt)

    def pos(self, timestamps):
        """Get the (approximate) source position at the times given by the
        array of unix timestamps.

        Returns two arrays (RA, dec), the same shape as timestamps,
        both in radians.  The RA are not corrected into any particular
        branch (the mapping from timestamp to RA is continuous).

        """
        if not isinstance(timestamps, np.ndarray):
            ra, dec = self.pos(np.array([timestamps]))
            return ra[0], dec[0]
        dt = timestamps - self.timestamp
        return self.ra + self.v_ra * dt, self.dec + self.v_dec * dt


def get_scan_q(tod, planet, boresight_offset=None, refq=None):
    """Identify the point (in time and azimuth space) at which the
    specified planet crosses the boresight elevation for the
    observation in tod.  The rotation taking boresight coordinates to
    celestial coordinates at that moment in time (and pointing) is
    computed, and its conjugate is returned.

    Boresight_offset is for taking into account off-centered wafer 
    that have position offset in the focal plane, which is either None
    or a (xi, eta) tuple in radian.

    The returned rotation is useful because it represents a fixed way
    to rotate the celestial such that the target source ends up at
    (xi, eta) = 0, with the telescope scan direction parallel to xi
    and elevation parallel to eta; this is a useful system for
    measuring beam and pointing parameters.

    """
    if boresight_offset is None:
        boresight_offset = (0, 0)
    q_xieta = so3g.proj.quat.rotation_xieta(boresight_offset[0], boresight_offset[1])
    if refq is None:
        refq = so3g.proj.quat.rotation_xieta(0, 0)
    # Get reference elevation...
    el = np.median(tod.boresight.el[::10])
    az = np.median(tod.boresight.az[::10])
    t = (tod.timestamps[0] + tod.timestamps[-1]) / 2
    if isinstance(planet, str):
        planet = SlowSource.for_named_source(planet, t)

    def scan_q_model(t, az, el, planet):
        csl = so3g.proj.CelestialSightLine.az_el(
            t, az, el, weather='typical', site='so')
        ra0, dec0 = planet.pos(t)
        return csl.Q, ~so3g.proj.quat.rotation_lonlat(ra0, dec0) * csl.Q * q_xieta

    def distance(p):
        dt, daz = p
        q, qnet = scan_q_model(t + dt, az + daz, el, planet)
        lon, lat, phi = so3g.proj.quat.decompose_lonlat(qnet)
        return 90 * coords.DEG - lat

    p0 = np.array([0, 0])
    X = fmin(distance, p0, disp=0, full_output=1)
    p, fopt, n_iter, n_calls, warnflag = X
    if warnflag != 0:
        logger.warning('Source-scan solver failed to converge or otherwise '
                       f'complained!  warnflag={warnflag}')
    q, qnet = scan_q_model(t+p[0], az+p[1], el, planet)
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


def get_scan_P(tod, planet, boresight_offset=None, refq=None, res=None, size=None, **kw):
    """Get a standard Projection Matrix targeting a planet (or some
    interesting fixed position), in source-scan coordinates.

    Returns a Projection Matrix and the output from get_scan_q.

    """
    logger.debug(f'get_scan_P: init for {planet}')

    if res is None:
        res = 0.01 * coords.DEG
    X = get_scan_q(tod, planet, boresight_offset=boresight_offset, refq=refq)
    rot = so3g.proj.quat.rotation_lonlat(0, 0) * X['rot']
    wcs_kernel = coords.get_wcs_kernel('tan', 0., 0., res=res)

    logger.debug(f'get_scan_P: getting projection matrix for {wcs_kernel}.')
    P = coords.P.for_tod(tod, rot=rot, wcs_kernel=wcs_kernel, **kw)
    if size is not None:
        # Trim to a local square
        mz = P.zeros(comps='T').submap([[-size/2, size/2], [size/2, -size/2]])
        P.geom = enmap.Geometry(shape=mz.shape, wcs=mz.wcs)
    return P, X


def get_horizon_P(tod, az, el, **kw):
    """Get a standard Projection Matrix targeting arbitrary source (for example
    drone) in horizon coordinates.

    Args:
      tod: AxisManager of the observation
      az: azimuth of the target source (rad)
      el: elevation of the target source (rad)

    Return:
      a Projection Matrix

    """
    sight = so3g.proj.CelestialSightLine.for_horizon(
        tod.timestamps,
        tod.boresight.az,
        tod.boresight.el,
        tod.boresight.roll
    )
    pq = so3g.proj.quat.rotation_lonlat(-az, el)
    sight.Q = so3g.proj.quat.rotation_lonlat(0, 0) * ~pq * sight.Q
    P = coords.P.for_tod(tod, sight, **kw)
    return P


def filter_for_sources(tod=None, signal=None, source_flags=None,
                       n_modes=10, low_pass=None,
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
      low_pass: Frequency, in Hz, at which to apply low pass filter to
        signal.  If None, no filtering is done.  You can pass in a
        filter from tod_ops.filters if you want.
      n_modes (int): Number of eigenmodes to remove... interface
        subject to change.
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
    signal_pca = signal.copy()
    gaps = tod_ops.get_gap_fill(tod, signal=signal_pca, flags=source_flags)
    gaps.swap(tod, signal=signal_pca)

    # Low pass filter?
    if low_pass is not None:
        if isinstance(low_pass, tod_ops.filters._chainable):
            filt = low_pass
        else:
            filt = tod_ops.filters.low_pass_butter4(low_pass)

        n_det, n = signal.shape
        a, b, t_1, t_2 = tod_ops.fft_ops.build_rfft_object(n_det, n, 'BOTH')
        a[:] = signal_pca
        t_1()
        times = tod.timestamps
        delta_t = (times[-1]-times[0])/(tod.samps.count - 1)
        freqs = np.fft.rfftfreq(n, delta_t)
        filt.apply(freqs, tod, target=b)
        signal_pca = t_2()
        del a, b

    # Measure TOD means (after gap fill, low pass, etc).
    if isinstance(n_modes, str) and n_modes == 'all':
        # Don't overthink that.
        signal -= signal_pca

    else:
        # Make sure PCA decomposition targets (signal_pca) are mean 0.
        # It's convenient to remove the same levels from signal now,
        # too.
        levels = signal_pca.mean(axis=1)
        signal_pca -= levels[:, None]
        signal -= levels[:, None]

        # Get PCA model and discard the source vectors.
        pca = tod_ops.pca.get_pca_model(
            tod, signal=signal_pca, n_modes=n_modes)
        del signal_pca

        # Remove the PCA model.
        tod_ops.pca.add_model(tod, pca, -1, signal=signal)

    if wrap:
        tod.wrap(wrap, signal, [(0, 'dets'), (1, 'samps')])
    return signal

def _get_astrometric(source_name, timestamp, site="_default"):
    """
    Derive skyfield's Astrometric object of a celestial source at a
    specific timestamp and observing site, which is used to derive
    radec/azel in get_source_pos/get_source_azel.

    Note that it will download a 16M ephemeris file on first use.

    Args:
      source_name: Planet name; in capitalized format, e.g. "Jupiter",
        or fixed source specification.
      timestamp: unix timestamp.
      site (str or so3g.proj.EarthlySite): if this is a string, the
        site will be looked up in so3g.proj.SITES dict.

    Returns:
      astrometric: skyfield's astrometric object
    """
    # Get the ephemeris
    de_filename = core.get_local_file("de421.bsp")

    planets = jpllib.SpiceKernel(de_filename)
    for k in [
        source_name,
        source_name + " barycenter",
    ]:
        try:
            target = planets[k]
            break
        except (ValueError, KeyError):
            pass
    else:
        options = list(planets.names().values())
        raise ValueError(
            f'Failed to find a match for "{source_name}" in ephemeris: {options}'
        )

    if isinstance(site, str):
        site = so3g.proj.SITES[site]

    observatory = site.skyfield_site(planets)

    timescale = skyfield_api.load.timescale()
    sf_timestamp = timescale.from_datetime(
        datetime.datetime.fromtimestamp(timestamp, tz=skyfield_api.utc)
    )
    astrometric = observatory.at(sf_timestamp).observe(target)
    return astrometric


def get_source_pos(source_name, timestamp, site='_default'):
    """Get the equatorial coordinates of a planet (or fixed-position
    source, see note) at some time.  Returns the apparent position,
    accounting for geographical position on earth, but assuming no
    atmospheric refraction.
    
    Args:
      source_name: Planet name; in capitalized format, e.g. "Jupiter",
        or fixed source specification.
      timestamp: unix timestamp.
      site (str or so3g.proj.EarthlySite): if this is a string, the
        site will be looked up in so3g.proj.SITES dict.

    Returns:
      ra (float): in radians.
      dec (float): in radians.
      distance (float): in AU.

    Note:

      Before checking in the ephemeris, the source_name will be
      matched against a regular expression and if it has the format
      'Jxxx[+-]yyy', where xxx and yyy are decimal numbers, then a
      fixed-position source at RA,Dec = xxx,yyy in degrees will be
      processed.  In that case, the distance is returned as Inf.

    """
    # Check against fixed-position template...
    m = re.match(
        r'J(?P<ra_deg>\d+(\.\d*)?)(?P<dec_deg>[+-]\d+(\.\d*)?)', source_name)
    if m:
        ra, dec = float(m['ra_deg']) * \
            coords.DEG, float(m['dec_deg']) * coords.DEG
        return ra, dec, float('inf')
    
    # Derive from skyfield astrometric object
    amet0 = _get_astrometric(source_name, timestamp, site)
    ra, dec, distance = amet0.radec()
    return ra.to(units.rad).value, dec.to(units.rad).value, distance.to(units.au).value


def get_source_azel(source_name, timestamp, site='_default'):
    """
    Get the apparent azimuth and elevation of a celestial source at a 
    specific timestamp and observing site. Returns the apparent position,
    accounting for geographical position on earth, but assuming no
    atmospheric refraction.

    Args:
        source_name: Planet name; in capitalized format, e.g. "Jupiter"
        timestamp (float): The Unix timestamp representing the time for 
          which to calculate azimuth and elevation.
        site (str or so3g.proj.EarthlySite): if this is a string, the
        site will be looked up in so3g.proj.SITES dict.

    Returns:
      az (float): in radians.
      el (float): in radians.
      distance (float): in AU.
    """
    amet0 = _get_astrometric(source_name, timestamp, site)
    el, az, distance = amet0.apparent().altaz()
    return az.to(units.rad).value, el.to(units.rad).value, distance.to(units.au).value


def get_nearby_sources(tod=None, source_list=None, distance=1.):
    """Identify solar system objects (especially "planets") that might be
    within a TOD's scan footprint.

    Arguments:
      tod (AxisManager): The data to check.  Needs to have
        focal_plane, boresight, timestamps.
      source_list (list or None): A list of source names or None to
        use a default list.  Use simple planet names in lower case
        (e.g. ['uranus']), or tuples with source name, RA, and dec in
        degrees (e.g. [('tau_a', 83.63, 22.01)]).
      distance (float): Maximum distance from the source center, in
        degrees, to consider as "within the footprint".  (This should
        be at least the sum of the beam radius and the planet radius
        ... though there's usually no harm in going a bit larger than
        that.)

    Returns:
      List of tuples (source_name, SlowSource) that satisfy the
      "nearby" condition.

    """
    # Make a full sky map with not very many pixels.
    shape, wcs = enmap.fullsky_geometry(res=2 * coords.DEG, proj='car')

    # Sight line
    sight = so3g.proj.CelestialSightLine.az_el(
        tod.timestamps, tod.boresight.az, tod.boresight.el,
        site='so', weather='typical')

    # One central detector
    xieta0, R, _ = coords.helpers.get_focal_plane_cover(tod, 0)
    fp = so3g.proj.FocalPlane.from_xieta(xieta0[0], xieta0[1])

    asm = so3g.proj.Assembly.attach(sight, fp)
    p = so3g.proj.Projectionist.for_geom(shape, wcs)
    w = p.to_map(np.zeros((1, len(tod.timestamps)),
                 'float32')+1, asm, comps='T')
    w = enmap.enmap(w, wcs=wcs)

    if source_list is None:
        source_list = SOURCE_LIST

    positions = []
    for source_name in source_list:
        t = tod.timestamps[0]
        if isinstance(source_name, (list, tuple)):
            source_name, ra, dec = source_name
            sl = coords.planets.SlowSource(t, float(ra) * coords.DEG,
                                           float(dec) * coords.DEG)
        else:
            sl = coords.planets.SlowSource.for_named_source(source_name, t)
        x = w.distance_from([[sl.dec], [sl.ra]])
        md = x[w[0] != 0].min()
        logger.debug(('Source {:12} is at ({:8.4f},{:8.4f}); '
                      'that is {:5.2f} degrees off footprint.').format(
                          source_name, sl.ra / coords.DEG,
                          sl.dec / coords.DEG, md/coords.DEG))
        if md < (R * 1.1 + distance * coords.DEG):
            positions.append((source_name, sl))
    return positions


def compute_source_flags(tod=None, P=None, mask=None, wrap=None,
                         center_on=None, res=None, max_pix=4e6):
    """Process masking instructions and create RangesMatrix that flags
    samples in the TOD that are within the masked region.  This
    masking makes use of a map with the footprint encoded in P, so
    flagging boundaries will correspond to pixel edges.

    The interface for "mask" is subject to change -- use of a simple
    text file is interim.

    Args:
      tod (AxisManager): the observation.
      P (Projection Matrix): if passed in, must include a map geom
        (shape and WCS).  If None, will be created from get_scan_P
        using center_on and res parameters.
      mask: source masking instructions (see note).
      wrap: key in tod at which to store the result.
      res: If P is None, sets the target mask map resolution
        (radians).
      max_pix: If P is None, this sets the maximum acceptable number
        of pixels for the mask map.  This is to catch cases where an
        incorrect source has been passed in, for example, leading to a
        weird map footprint

    Returns:
      RangesMatrix marking the samples inside the masked region.

    Notes:
      The mask can be a dict or a list of dicts.  Each dict must be of
      the form::

        {'shape': 'circle',
         'xyr': (XI, ETA, R)}

      where R is the radius of the circular mask, and XI and ETA are
      the center of the circle, all in degrees.

    """
    if P is None:
        logger.info('Getting Projection Matrix ...')
        P, X = get_scan_P(tod, center_on, res=res, comps='T')
        shape, wcs = tuple(P.geom)
        if shape[0] * shape[1] > max_pix:
            raise ValueError(f'Mask map too large: {shape}')

    if isinstance(mask, str):
        # Assume it's a filename, and file is simple columns of (x, y,
        # radius) in deg.  (Deprecated!)
        mask = [{'xyr': list(map(float, line.split()))}
                for line in open(mask)]

    mask_map = P.zeros()
    _add_to_mask(mask, mask_map)
    a = P.from_map(mask_map)
    source_flags = so3g.proj.RangesMatrix(
        [so3g.proj.Ranges.from_mask(r != 0) for r in a])

    if wrap:
        assert tod is not None, "Pass in a tod to 'wrap' the output."
        tod.flags.wrap(wrap, source_flags, [(0, 'dets'), (1, 'samps')])
    return source_flags


def _add_to_mask(req, mask_map):
    # Helper for compute_source_flags.
    if req is None:
        raise ValueError(f'Requested mask is None.  For no mask, pass [].')
    if isinstance(req, (list, tuple)):
        for _r in req:
            # Also, maybe test this somehow?
            _add_to_mask(_r, mask_map)
    elif isinstance(req, dict):
        shape = req.get('shape', 'circle')
        if shape == 'circle':
            x, y, r = req['xyr']
            d = enmap.distance_from(mask_map.shape, mask_map.wcs,
                                    [[y * coords.DEG], [x * coords.DEG]])
            mask_map += 1. * (d < r * coords.DEG)
        else:
            raise ValueError(f'Unknown shape="{shape}" in mask request.')
    else:
        raise ValueError(f'Weird mask request: {req}')


def load_detector_splits(tod=None, filename=None, dataset=None,
                         source=None, wrap=None):
    """Convert a partition of detectors into a dict of disjoint
    RangesMatrix objects; such an object can be passed to make_map()
    to efficiently make detector-split maps.

    The "detector split" data can be read from an HDF5 dataset, or
    passed in directly as an AxisManager.

    Args:
      tod (AxisManager): This is required, to get the list of dets and
        the samps count.
      filename (str): The HDF filename, or filename:dataset.
      dataset (str): The HDF dataset (if not passed in with filename).
      source (array, ResultSet or AxisManager): If not None, then
        filename and dataset are ignored and this object is processed
        (as though it had just been loaded from HDF).
      wrap (str): If not None, the address in tod where to store the
        loaded split data.

    The format of detector splits, in an HDF5 dataset, is as one would
    write from a ResultSet with columns ['dets:name', 'group'] (both
    str).  All detectors sharing a value in the group column will
    grouped together and the group label will be that value.

    If passing in "source" directly as a ResultSet, it should have
    columns 'dets:name' and 'group'; if as an AxisManager then it
    should have a 'dets' axis and a vector 'group' with shape
    ('dets',) providing the group name for each detector.  If it is a
    numpy array, it is assumed to correspond one-to-one with the .dets
    axis of TOD and the array gives the group name for each detector.

    Returns:
      data_splits (dict of RangesMatrix): Each entry of the dict is a
        RangesMatrix that can be interpreted as cuts to apply during
        mapmaking.  In this case the RangesMatrix will simply mark
        each detector as either fully cut (flagged) or fully uncut.

    """
    from sotodlib.io import metadata

    if source is None:
        if dataset is None:
            filename, dataset = filename.split(':')
        source = metadata.read_dataset(filename, dataset)
    if isinstance(source, np.ndarray):
        source, _s = core.AxisManager(tod.dets), source
        source.wrap('group', _s)
    elif isinstance(source, metadata.ResultSet):
        di = core.metadata.loader.unconvert_det_info(tod.det_info)
        source = core.metadata.loader.broadcast_resultset(
            source, di, axis_key='name')
    else:
        source = source.copy()  # is this an AxisManager?
    source.restrict_axes([tod.dets])
    if wrap:
        tod.wrap(wrap, source)
    yes = so3g.proj.RangesMatrix.zeros(tod.signal.shape[1])
    flags = {}
    for group in source['group']:
        if group in flags:
            continue
        flags[group] = so3g.proj.RangesMatrix.ones((tod.signal.shape))
        for i in (source['group'] == group).nonzero()[0]:
            flags[group].ranges[i] = yes
    return flags


def get_det_weights(tod, signal=None, wrap=None,
                    outlier_clip=None):
    """Compute detector weights, based on variance of signal.  If
    outlier_clip is set, it is used to trim outliers.  See code for
    details, but think of it as the number of stdev away from the mean
    weight that will be kept, and try 2.5.

    Returns:
      A det_weights array compatible with projection matrix functions.

    """
    if signal is None:
        signal = tod.signal
    det_weights = np.zeros(signal.shape[:-1])
    sigmas = signal.std(axis=-1)
    ok = sigmas != 0
    det_weights[ok] = sigmas[ok]**-2
    if outlier_clip:
        qs = np.quantile(det_weights[ok], [.16, .84])  # "1 sigma" lims
        q0, dq = qs.mean(), np.diff(qs)[0]/2
        ok *= ((q0 - outlier_clip * dq < det_weights) *
               (det_weights < q0 + outlier_clip * dq))
    return det_weights * ok


def write_det_weights(tod, filename, dataset, det_weights=None):
    """Save detector weights to an HDF file dataset.

    Args:
      tod (AxisManager): provides the dets axis, and det_weights if
        not passed explicitly.
      filename (str or h5py.File): destination filename (or open File)
      dataset (str): address in the HDF5 file to put the result (it
        will be overwritten if exists already).
      det_weights (array): the weights to write; must be in
        correspondence with tod.dets.  Defaults to tod.det_weights.

    Returns:
      ResultSet with the info that was written.

    """
    from sotodlib.io import metadata
    if det_weights is None:
        det_weights = tod.det_weights
    rs = core.metadata.ResultSet(['dets:name', 'det_weights'])
    rs.rows.extend(list(zip(tod.dets.vals, det_weights)))
    metadata.write_dataset(rs, filename, dataset, overwrite=True)
    return rs


def make_map(tod, center_on=None, scan_coords=True, thread_algo=False,
             res=0.01*coords.DEG, size=None, wcs_kernel=None, comps='TQU',
             signal=None,
             det_weights=None,
             filename=None, source_flags=None, cuts=None,
             data_splits=None,
             low_pass=None, n_modes=10,
             eigentol=1e-3, info={}):
    """Make a compact source map from the TOD.  Specify filename to write
    things to disk; this should be a format string, for example
    '{obs_id}_{map}.fits', where 'map' will be given values of
    ['binned', 'solved', 'weights'] and any other keys (obs_id in this
    case) must be passed through info.

    """
    assert (center_on is not None)  # Pass in the source name, e.g. 'uranus'

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

    if signal is None:
        signal = tod.signal

    if thread_algo == 'none':
        thread_algo = False

    with MmTimer('setup thread_algo=%s' % thread_algo):
        if scan_coords:
            P, X = coords.planets.get_scan_P(tod, center_on, res=res, size=size,
                                             comps=comps, cuts=cuts,
                                             threads=thread_algo)
        else:
            planet = coords.planets.SlowSource.for_named_source(
                center_on, tod.timestamps[0])
            ra0, dec0 = planet.pos(tod.timestamps.mean())
            wcsk = coords.get_wcs_kernel('tan', ra0, dec0, res=res)
            P = coords.P.for_tod(tod, comps=comps,
                                 cuts=cuts,
                                 threads=thread_algo,
                                 wcs_kernel=wcsk)
    with MmTimer('get_proj_threads'):
        P._get_proj_threads()

    with MmTimer('filter for sources'):
        filter_for_sources(tod, signal=signal, source_flags=source_flags,
                           low_pass=low_pass, n_modes=n_modes)

    if det_weights is None:
        det_weights = get_det_weights(tod, signal=signal, outlier_clip=2.)

    if data_splits is not None:
        # Clear P's internal cuts as we'll be modifying those to pass
        # in directly.
        base_cuts, P.cuts = P.cuts, None
        output = {
            'P': P,
            'det_weights': det_weights,
            'splits': {}
        }

        # Write out _map and _weights for each group.
        for group_label, group_cuts in data_splits.items():
            logger.info(f'Mapping split "{group_label}"')
            if base_cuts is not None:
                group_cuts = group_cuts + base_cuts
            with MmTimer('getting weights'):
                w = P.to_weights(cuts=group_cuts, det_weights=det_weights)
            with MmTimer('getting map and applying inverse weights'):
                m = P.remove_weights(
                    tod=tod, signal=signal, weights_map=w, cuts=group_cuts,
                    det_weights=det_weights, eigentol=eigentol)
            output['splits'][group_label] = {
                'binned': None,
                'weights': w.astype('float32'),
                'solved': m.astype('float32'),
            }
            if filename is not None:
                m.astype('float32').write(
                    filename.format(map=f'{group_label}_map', **info))
                w.astype('float32').write(
                    filename.format(map=f'{group_label}_weights', **info))
        return output

    with MmTimer('project signal and weight maps'):
        map1 = P.to_map(tod, signal=signal, det_weights=det_weights)
        wmap1 = P.to_weights(det_weights=det_weights)

    with MmTimer('compute and apply inverse weights map'):
        iwmap1 = P.to_inverse_weights(weights_map=wmap1, eigentol=eigentol)
        map1b = P.remove_weights(map1, inverse_weights_map=iwmap1)

    if filename is not None:
        with MmTimer('Write out'):
            map1b.write(filename.format(map='solved', **info))
            if filename.format(map='x', **info) != filename.format(map='y', **info):
                map1.write(filename.format(map='binned', **info))
                wmap1.write(filename.format(map='weights', **info))
                write_det_weights(tod, filename.format(map='detweights', **info).
                                  replace('.fits', '.h5'), 'detweights',
                                  det_weights=det_weights)

    return {'binned': map1,
            'weights': wmap1,
            'solved': map1b,
            'P': P,
            'det_weights': det_weights,
            }

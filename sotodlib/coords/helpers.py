import so3g.proj
import numpy as np
from pixell import enmap, wcsutils, utils

import time
import re

DEG = np.pi/180


def _get_csl(sight):
    """Return the CelestialSightLine equivalent of sight.  If sight is
    already of that class, it is returned.  If it's a G3VectorQuat or
    an array [n,4], those coordinates are wrapped and returned (note a
    reference may be taken rather than a copy made).

    """
    if isinstance(sight, so3g.proj.CelestialSightLine):
        return sight
    if isinstance(sight, so3g.proj.quat.G3VectorQuat):
        c = so3g.proj.CelestialSightLine()
        c.Q = sight
        return c
    qa = np.asarray(sight, dtype='float')
    c = so3g.proj.CelestialSightLine()
    c.Q = so3g.proj.quat.G3VectorQuat(qa)
    return c

def _valid_arg(*args, src=None):
    """Return some data, possibly extracted from an AxisManager (or
    dict...), based on the arguments args.  This is to help with
    processing function arguments that override a default behavior
    which is to look up a thing in axisman.  For example::

      signal = _valid_arg(signal, 'signal', src=axisman)

    is equivalent to::

      if signal is None:
          signal = 'signal'
      if isinstance(signal, str):
          signal = axisman[signal]

    Similarly::

        sight = _valid_arg(sight, src=axisman)

    is a shorthand for::

        if sight is not None:
           if isinstance(sight, string):
               sight = axisman[sight]

    Each element of args should be either a data vector (non string),
    a string, or None.  The arguments are processed in order.  The
    first argument that is not None will cause the function to return;
    if that argument k is a string, then axisman[k] is returned;
    otherwise k is returned directly.  If all arguments are None, then
    None is returned.

    """
    for a in args:
        if a is not None:
            if isinstance(a, str):
                try:
                    a = src[a]
                except TypeError:
                    raise TypeError(f"Tried to look up '{a}' in axisman={axisman}")
            return a
    return None

def _not_both(a, b, name='{item}', dtype=None):
    if a is not None:
        if b is not None:
            raise ValueError('self.%s and kwarg %s both not None!' % (name, name))
        b = a
    if b is not None and dtype is not None:
        b = b.astype(dtype)
    return b

class Timer:
    """Context manager that prints elapsed time to terminal or a log or
    whatever.  For example::

        with Timer('tod-to-map projection', logger.info):
            P.to_map(tod)

    Should log a message of the form::

        INFO:tod-to-map projection:            3.212 seconds

    Use the fmt keyword argument to enter your own format string,
    which can include references to {msg}, {start_time}, {end_time},
    {elapsed}.

    To avoid repeating the same arguments, subclass this and set
    FORMAT and PRINT_FUNC for your use cases::

       class MyTimer(Timer):
           FORMAT = 'mapmaker7: {msg} took {elapsed:10.3f} seconds'
           PRINT_FUNC = lambda x: (logger.info(x), print(x))

    """
    FMT = '{msg:40}: {elapsed:10.3f} seconds'
    PRINT_FUNC = print

    def __init__(self, msg=None, print_func=None, fmt=None):
        if msg is None:
            msg = 'timed operation'
        if fmt is None:
            fmt = self.FMT
        if print_func is None:
            print_func = self.PRINT_FUNC
        self.msg = msg
        self.fmt = fmt
        self.print_func = print_func
        self.start_time = time.time()
    def __enter__(self):
        return self
    def __exit__(self, *args, **kw):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.text = self.fmt.format(
            msg=self.msg, start_time=self.start_time, end_time=self.end_time,
            elapsed=self.elapsed)
        self.print_func(self.text)


def get_radec(tod, wrap=False, dets=None, timestamps=None, focal_plane=None,
              boresight=None, sight=None):
    """Get the celestial coordinates of all detectors at all times.

    Args:
      wrap (bool): If True, the output is stored into tod['radec'].
        This can also be a string, in which case the result is stored
        in tod[wrap].
      dets (list of str): If set, then coordinates are only computed
        for the requested detectors.  Note you probably can't wrap the
        result, in this case, as the dets axis is non-concordant.

    Returns:
      The returned array has shape (n_det, n_samp, 4).  The four
      components in the last dimension correspond to (lon, lat,
      cos(psi), sin(psi)).  The lon and lat are in radians and
      correspond to RA and dec of equatorial coordinates.  Psi is is
      the parallactic rotation, measured from North towards West
      (opposite the direction of standard Position Angle).

    """
    dets = _valid_arg(dets, tod.dets.vals, src=tod)
    fp = _valid_arg(focal_plane, 'focal_plane', src=tod)
    if sight is None:
        timestamps = _valid_arg(timestamps, 'timestamps', src=tod)
        boresight = _valid_arg(boresight, 'boresight', src=tod)
        sight = so3g.proj.CelestialSightLine.az_el(
            timestamps, boresight.az, boresight.el, roll=boresight.roll,
            site='so', weather='typical')
    else:
        sight = _get_csl(_valid_arg(sight, 'sight', src=tod))
    fp = so3g.proj.FocalPlane.from_xieta(dets, fp.xi, fp.eta, fp.gamma)
    asm = so3g.proj.Assembly.attach(sight, fp)
    output = np.zeros((len(dets), len(sight.Q), 4))
    proj = so3g.proj.Projectionist()
    proj.get_coords(asm, output=output)
    if wrap is True:
        wrap = 'radec'
    if wrap:
        tod.wrap(wrap, output, [(0, 'dets'), (1, 'samps')])
    return output

def get_horiz(tod, wrap=False, dets=None, timestamps=None, focal_plane=None,
              boresight=None):
    """Get the horizon coordinates of all detectors at all times.

    Args:
      wrap (bool): If True, the output is stored into tod['horiz'].
        This can also be a string, in which case the result is stored
        in tod[wrap].
      dets (list of str): If set, then coordinates are only computed
        for the requested detectors.  Note you probably can't wrap the
        result, in this case, as the dets axis is non-concordant.

    Returns:
      The returned array has shape (n_det, n_samp, 4).  The four
      components in the last dimension correspond to (-lon, lat,
      cos(psi), sin(psi)).  The -lon and lat are in radians and
      correspond to horizon azimuth and elevation.  Psi is is
      the parallactic rotation, measured from North towards West
      (opposite the direction of standard Position Angle).

    """
    dets = _valid_arg(dets, tod.dets.vals, src=tod)
    timestamps = _valid_arg(timestamps, 'timestamps', src=tod)
    boresight = _valid_arg(boresight, 'boresight', src=tod)
    fp = _valid_arg(focal_plane, 'focal_plane', src=tod)

    sight = so3g.proj.CelestialSightLine.for_horizon(
        timestamps, boresight.az, boresight.el, roll=boresight.roll)

    fp = so3g.proj.FocalPlane.from_xieta(dets, fp.xi, fp.eta, fp.gamma)
    asm = so3g.proj.Assembly.attach(sight, fp)
    output = np.zeros((len(dets), len(timestamps), 4))
    proj = so3g.proj.Projectionist()
    proj.get_coords(asm, output=output)
    # The lonlat pair is (-az, el), so restore the az sign.
    output[:,:,0] *= -1
    if wrap is True:
        wrap = 'horiz'
    if wrap:
        tod.wrap(wrap, output, [(0, 'dets'), (1, 'samps')])
    return output


def get_wcs_kernel(proj, ra=None, dec=None, res=None):
    """Construct a WCS.  This fixes the projection type (e.g. CAR, TAN),
    centered with respect to a reference point (at ra,dec = (0,0)), and
    resolution of a pixelization, without specifying a particular grid
    of pixels.

    This interface is subject to change.

    Args:
      proj (str): Either the name of a FITS projection to use (e.g. 'car',
        'cea', 'tan'), in which case "res" must also be specified, or a
        string containing both the projection and the resolution, in this
        format:

                      proj-res

        where proj is a projection type (e.g. 'car', 'tan, 'gnom') and res
        is the resolution, in appropriate units (deg, arcmin or arcsec).
        Examples of acceptable args:

                      'car-0.5deg'
                      'tan-3arcmin'
                      'gnom-25arcsec'

      ra: Right Ascension (longitude) of the reference position, in
        radians.
      dec: Declination (latitude) of the reference position, in
        radians.
      res: Resolution, in radians.

    Returns a WCS object that captures the requested pixelization.

    """
    if len(proj) > 4:
        m = re.match('(?P<proj>car|tan|cea|gnom)-(?P<res>[0-9]*.?[0-9]*)(?P<unit>arcsec|arcmin|deg)',
                     proj)
        if m is None:
            raise ValueError("Input projection string is incorrectly formatted.")
        proj = m['proj']
        res = float(m['res'])
        unit = m['unit']

        # Convert to radians
        res = res * np.pi/180.
        if unit == 'arcmin':
            res /= 60.
        if unit == 'arcsec':
            res /= 3600.

        _, wcs = enmap.geometry(np.array((0, 0)), shape=(1, 1), proj=proj,
                                res=(res, -res))

    else:
        assert np.isscalar(res)  # This ain't enlib.
        _, wcs = enmap.geometry(np.array((dec, ra)), shape=(1, 1), proj=proj,
                                res=(res, -res))

    if proj == 'car':
        wcs.wcs.crpix = [1.0, 0.5]

        # Check if the resolution is sensible, for SHTs
        res = res * 180./np.pi  # Easier to do this check in degs
        div = 90./res - np.round(90./res)
        if abs(div) > 1e-8:
            raise ValueError(f"The requested resolution {res} deg does not divide 90 deg evenly.")

    return wcs


def get_footprint(tod, wcs_kernel, dets=None, timestamps=None, boresight=None,
                  focal_plane=None, sight=None, rot=None):
    """Find a geometry (in the sense of enmap) based on wcs_kernel that is
    big enough to contain all data from tod.  Returns (shape, wcs).

    """
    if isinstance(wcs_kernel, str):
        wcs_kernel = get_wcs_kernel(wcs_kernel)

    dets = _valid_arg(dets, tod.dets.vals, src=tod)
    fp0 = _valid_arg(focal_plane, 'focal_plane', src=tod)
    if sight is None and 'sight' in tod:
        sight = tod.sight
    sight = _valid_arg(sight, tod.get('sight'), src=tod)
    if sight is None:
        # Let's try to either require a sightline or boresight info.
        timestamps = _valid_arg(timestamps, 'timestamps', src=tod)
        boresight = _valid_arg(boresight, 'boresight', src=tod)
        sight = so3g.proj.CelestialSightLine.az_el(
            timestamps, boresight.az, boresight.el, roll=boresight.roll,
            site='so', weather='typical').Q
    sight = _get_csl(sight)
    n_samp = len(sight.Q)

    # Get a convex hull focal plane.
    (xi0, eta0), R, xieta1 = get_focal_plane_cover(
        focal_plane=fp0, count=16)
    fake_dets = ['hull%i' % i for i in range(xieta1.shape[1])]
    fp1 = so3g.proj.FocalPlane.from_xieta(fake_dets, xieta1[0], xieta1[1])

    asm = so3g.proj.Assembly.attach(sight, fp1)
    output = np.zeros((len(fake_dets), n_samp, 4))
    proj = so3g.proj.Projectionist.for_geom((1,1), wcs_kernel)
    if rot:
        # Works whether rot is a quat or a vector of them.
        asm.Q = rot * asm.Q
    proj.get_planar(asm, output=output)

    output2 = output*0
    proj.get_coords(asm, output=output2)

    # Get the pixel extrema in the form [{xmin,ymin},{xmax,ymax}]
    delts  = wcs_kernel.wcs.cdelt * DEG
    planar = output[:,:,:2]
    ranges = utils.minmax(planar/delts,(0,1))
    # These are in units of pixel *offsets* from crval. crval
    # might not correspond to a pixel center, though. So the
    # thing that should be integer-valued to preserve pixel compatibility
    # is crpix + ranges, not just ranges. Let's add crpix to transform this
    # into offsets from the bottom-left pixel to make it easier to reason
    # about integers
    ranges += wcs_kernel.wcs.crpix
    del output

    # Start a new WCS and set the lower left corner.
    w = wcs_kernel.deepcopy()
    corners      = utils.nint(ranges)
    w.wcs.crpix -= corners[0]
    shape        = tuple(corners[1]-corners[0]+1)[::-1]
    return (shape, w)

def get_focal_plane_cover(tod=None, count=0, focal_plane=None,
                          xieta=None):
    """Process a bunch of detector positions into a center and radius such
    that a circle with that center and radius contains all the
    detectors.  Also return detector positions, arranged approximately
    in a circle, whose convex hull contains all input detectors.

    Args:
      tod (AxisManager): source of focal_plane, if it is not passed.
      count (int): number of points on the enclosing circle to return.
      focal_plane (AxisManager): source for xi and eta, if xieta is
        not passed in
      xieta (array): (2, n) array (or similar) of xi and eta
        detector positions.

    Returns:
      xieta0: array[2] with array center, (xi0, eta0).
      radius: radius of the enclosing circle.
      xietas: array with shape (2, count) carrying the xi and eta
        coords of the circular convex hull.

    Notes:
      If count=0, an empty list is returned for xietas.  Otherwise,
      count must be at least 3 so that the shape is not degenerate.

    """
    if xieta is None:
        if focal_plane is None:
            focal_plane = tod.focal_plane
        xi = focal_plane.xi
        eta = focal_plane.eta
    else:
        xi, eta = xieta[:2]
    qs = so3g.proj.quat.rotation_xieta(xi, eta)

    # Starting guess for center
    xi0, eta0 = xi.mean(), eta.mean()
    for i in range(10):
        q = so3g.proj.quat.rotation_xieta(xi0, eta0)
        d_xi, d_eta, _ = so3g.proj.quat.decompose_xieta(~q * qs)
        if abs(d_xi.mean()) + abs(d_eta.mean()) < 1e-5*DEG:
            break
        xi0 += d_xi.mean()*.8
        eta0 += d_eta.mean()*.8
    else:
        raise ValueError('No convergence?')
    R = (d_xi**2 + d_eta**2).max()**.5
    if count == 0:
        return ((xi0, eta0), R, [])
    if count < 3:
        raise ValueError('count must be 0 or >=3.')

    dphi = 2*np.pi/count
    phi = np.arange(count) * dphi
    # cos(dphi/2) is the largest underestimate in radius one can make when
    # replacing a circle with an n_circ-sided polygon, as we do here.
    L = 1.01 * R / np.cos(dphi/2)
    xi, eta = L * np.cos(phi), L * np.sin(phi)

    # Rotate those into place.
    xi, eta, _ = so3g.proj.quat.decompose_xieta(
        q * so3g.proj.quat.rotation_xieta(xi, eta))
    return (np.array([xi0, eta0]), R, np.array([xi, eta]))

def get_supergeom(*geoms, tol=1e-3):
    """Given a set of compatible geometries [(shape0, wcs0), (shape1,
    wcs1), ...], return a geometry (shape, wcs) that includes all of
    them as a subset.

    Each argument can be a geometry (shape, wcs) or an ndmap.

    """
    def as_geom(item):
        if isinstance(item, enmap.ndmap):
            shape, wcs = item.geometry
        else:
            shape, wcs = item
        return (shape[-2:], wcs)

    s0, w0 = as_geom(geoms[0])
    w0 = w0.deepcopy()

    for item in geoms[1:]:
        s, w = as_geom(item)
        # is_compatible is necessary but not sufficient.
        if not wcsutils.is_compatible(w0, w):
            raise ValueError('Incompatible wcs: %s <- %s' % (w0, w))

        # Depending on the projection, it may be possible to translate
        # crval and crpix along each dimension and maintain exact
        # pixel center correspondence.
        translate = (False, False)
        if wcsutils.is_plain(w0):
            translate = (True, True)
        elif wcsutils.is_cyl(w0) and w0.wcs.crval[1] == 0.:
            translate = (True, False)

        cdelt = w0.wcs.cdelt
        if np.any(abs(w.wcs.cdelt - cdelt) / cdelt > tol):
            raise ValueError("CDELT not the same.")

        # Determine what shift in w.crpix would make the crval the same.
        d_crpix = w.wcs.crpix - w0.wcs.crpix
        d_crval = w.wcs.crval - w0.wcs.crval
        tweak_crpix = [0, 0]
        for axis in [0, 1]:
            if d_crval[axis] != 0 and not translate[axis]:
                raise ValueError(f"Incompatible CRVAL in axis {axis}")
            d = d_crval[axis] / cdelt[axis] - d_crpix[axis]
            if abs((d + 0.5) % 1 - 0.5) > tol:
                raise ValueError(f"CRVAL not separated by integer pix in axis {axis}.")
            tweak_crpix[axis] = int(np.round(d))

        d = np.array(tweak_crpix[::-1])
        # Position of s in w0?
        corner_a = d + [0, 0]
        corner_b = d + s
        # Super footprint, in w0.
        corner_a = np.min([corner_a, [0, 0]], axis=0)
        corner_b = np.max([corner_b, s0], axis=0)
        # Boost the WCS
        w0.wcs.crpix -= corner_a[::-1]
        s0 = corner_b - corner_a
    return tuple(map(int, s0)), w0

def _invert_weights_map(weights, eigentol=1e-6, kill_partials=True,
                        UPLO='U'):
    """Compute an inverse weights matrix, using eigendecomposition methods
    that are safe against singular matrices.  This is similar to
    scipy.linalg.pinvh, but applied to each pixel in a map in an
    efficient way.

    Args:
      weights (array): an array (or ndarray) with at least 2
        dimensions, where the leading two dimensions represent
        submatrices that are to be inverted; the other dimensions
        index "the pixel".  Valid shapes would be, for example, (3, 3,
        200, 100) or (3, 3, 10231230).
      eigentol (float): sets the threshold for keeping eigenvectors.
        In each sub-matrix inversion, any eigenvectors whose absolute
        values are less than eigentol times the largest absolute
        eigenvalue are set to zero (and thus excluded from inclusion
        in the inverse).
      kill_partials (bool): if True, then pixels where any
        eigenvectors are zero (or have been forced to zero) will have
        all their eigenvectors forced to zero.  Stated another way,
        all pixels with singular or nearly singular weights
        sub-matrices will be treated as having no weight at all.
      UPLO (str): this can be 'U' or 'L', signifying that only the
        upper diagonal or lower diagonal (respectively) elements of
        each weights sub-matrix should be considered.  (This argument
        is passed through to np.linalg.eigh.)

    Returns:
      A matrix with the same shape as weights, but where the submatrix
      carried in the first two dimensions is the inverse of the
      corresponding submatrix of weights; or possibly a pseudo-inverse
      or the zero matrix depending on arguments described above.

    """
    # Quick short circuit in trivial case...
    if weights.shape[:2] == (1, 1):
        iw = np.zeros_like(weights)  # yes, this preserves wcs
        iw[weights!=0] = 1./weights[weights!=0]
        return iw

    # Collapse and reindex weights map so it is (npix, n, n).
    w = weights.reshape(weights.shape[:2] + (-1,)).transpose(2, 0, 1)

    # Get eigendecomposition of each (n, n) sub-matrix
    v, U = np.linalg.eigh(w, UPLO)

    # Identify acceptable eigenvalues -- reject ones that are non-positive or too
    # small relative to max eigenvalue in their pixel.
    eig_ok = (v > 0) * (v > v[:,-1:] * eigentol)

    # Does one bad eig spoil the basket?
    if kill_partials:
        eig_ok *= np.all(eig_ok, axis=1)[:,None]

    # Force each unacceptable eigenmode to 0.
    U *= eig_ok[:,None,:]

    # Set bad eigenvalues to 1, to avoid the divide-by-zero.
    v[~eig_ok] = 1.

    # Compute the effective inverse, U (1/diag(v)) U.T.
    A = (U / v[:,None,:])
    B = U.transpose(0,2,1)
    iw = np.matmul(A, B)

    # Reshape the output to match what was passed in.
    return iw.transpose(1,2,0).reshape(weights.shape)

def _apply_inverse_weights_map(inverse_weights, target):
    """Apply a map of matrices to a map of vectors.

    Assumes inverse_weights.shape = (a, b, ...) and target.shape =
    (b, ...); the result has shape (a, ...).

    """
    # master had:
    #iw = inverse_weights.transpose((2,3,0,1))
    #m = target.transpose((1,2,0)).reshape(
    #    target.shape[1], target.shape[2], target.shape[0], 1)
    #m1 = np.matmul(iw, m)
    #return m1.transpose(2,3,0,1).reshape(target.shape)
    iw = np.moveaxis(inverse_weights, (0,1), (-2,-1))
    t  = np.moveaxis(target[:,None],  (0,1), (-2,-1))
    m  = np.matmul(iw, t)
    return np.moveaxis(m, (-2,-1), (0,1))[:,0]

class ScalarLastQuat(np.ndarray):
    """Wrapper class for numpy arrays carrying quaternions with the ijk1
    signature.

    The practice in many codes, including TOAST, quaternionarray, and
    scipy is to store quaternions in numpy arrays with shape (..., 4),
    with the real part of the quaternion at index [..., 3].  In
    contrast, spt3g_software inherits the 1ijk from boost.

    This class serves two main purposes:

    - It can be used to annotate numpy arrays as carrying quaternions
      in the ijk1 signature (without disturbing their behavior as
      numpy arrays).
    - It provides conversion to and from G3 quaternion containers,
      including signature correction.

    When instantiating an object of this class with a numpy array as
    an argument, the object will wrap a view of the array (not a
    copy).

    When instantiating an object of this class with a G3 quat as an
    argument, the object will store a copy of the G3 quat, translated
    to ijk1 signature::

      q_g3 = spt3g.core.quat(1.,2.,3.,4.)
      q_tq = ScalarLastQuat(q_g3)
      q_g3_again = q_tq.to_g3()

      print(q_g3, q_tq, q_g3_again)
      =>  (1,2,3,4) [2. 3. 4. 1.] (1,2,3,4)

    """
    def __new__(cls, arr):
        if isinstance(arr, so3g.proj.quat.G3VectorQuat):
            arr = np.asarray(arr)
            obj = np.empty(arr.shape)
            obj[:,:3] = arr[:,1:]
            obj[:,3] = arr[:,0]
        elif isinstance(arr, so3g.proj.quat.quat):
            obj = np.array((arr.b, arr.c, arr.d, arr.a))
        else:
            obj = np.asarray(arr)
        return obj.view(cls)

    def to_g3(self):
        """Return the ijk1-signature equivalent of the enclosed quaternion
        array, as a spt3g.core.quat (if 1-d) or a G3VectorQuat (if
        2-d).

        """
        if self.shape[-1] != 4:
            raise ValueError("Last axis must have 4 elements.")
        if self.ndim == 1:
            b, c, d, a = self[:].astype(float)
            return so3g.proj.quat.quat(a, b, c, d)
        if self.ndim == 2:
            temp = np.zeros(self.shape, float)
            temp[..., 0] = self[..., 3]
            temp[..., 1:] = self[..., :3]
            return so3g.proj.quat.G3VectorQuat(temp)
        raise ValueError("Can only convert 1- or 2-d arrays to G3.")

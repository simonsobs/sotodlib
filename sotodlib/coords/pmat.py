import so3g.proj
import numpy as np
import scipy
from pixell import enmap

from .helpers import _get_csl, _valid_arg, _not_both
from . import helpers

import logging
logger = logging.getLogger(__name__)


class P:
    """Projection Matrix.

    This class provides functions to apply a Projection Matrix (or its
    transpose).  The Projection Matrix, P, also sometimes called the
    Pointing Matrix, describes how a vector of time-ordered
    measurements d are determined by a vector of map pixels m::

      d = P m

    We are working in aspace, of course, where d is not a vector, but
    rather an array of vectors, and m is not a vector, it's a
    multi-dimensional array with two of the dimensions corresponding
    to a rectangular pixelization of the sky.

    If you are making filter+bin maps, you will want to use these functions:

    - to_map
    - to_inverse_weights
    - remove_weights

    If you are solving for a map iteratively you will probably just need:

    - to_map
    - from_map

    Important keyword arguments that are used in many functions:

    - tod: The AxisManager from which signal and pointing information
      should be taken.
    - signal: The array to use for time-ordered signal.  If a string,
      it will be looked up in tod.  Defaults to 'signal'.
    - det_weights: A vector of floats representing the inverse
      variance of each detector, under the assumption of white noise.
    - cuts: A RangesMatrix that identifies samples that should be
      excluded from projection operations.
    - comps: the component code (e.g. 'T', 'TQU', 'QU', ...) that
      specifies the spin-components being modeled in the map.
    - dest: the appropriately shaped array in which to place the
      computed result (as an alternative to allocating a new object to
      store the result).

    Note that in the case of det_weights and cuts, the Projection
    Matrix may also have cached values for those.  It is an error to
    pass either of these as a keyword argument to a projection routine
    if a value has been cached for it.

    Objects of this class cache certain pre-computed information (such
    as the rotation taking boresight coordinates to celestial
    coordinates) and certain context-dependent settings (such as a map
    shape, WCS, and spin-component configuration).  You may want to
    inspect or borrow these results, perhaps to reuse them when
    constructing new instances with slight modifications.  The cached
    attributes are:

    - sight: A CelestialSightLine, representing the boresight pointing
      in celestial coordinates. [samps]
    - fp: G3VectorQuat representing the focal plane offsets of each
      detector. [dets]
    - geom: The target map geometry (shape, wcs).
    - comps: String indicating the spin-components to include in maps.
      E.g., 'T', 'QU', 'TQU'.
    - rot: quat giving an additional fixed rotation to apply to get
      from boresight to celestial coordinates.  Not for long...
    - cuts (optional): RangesMatrix indicating what samples to exclude
      from projection operations (the indicated samples have
      projection matrix element 0 in all components). [dets, samps]
    - threads (optional): RangesMatrix that assigns ranges of samples
      to specific threads.  This is necessary for TOD-to-map
      operations that use OpenMP. [dets, samps]
    - det_weights (optional): weights (one per detector) to apply to
      time-ordered data when binning a map (and also when binning a
      weights matrix).  [dets]

    These things can be updated freely, with the following caveats:

    - If the number of "samples" or "detectors" is changed in one
      attribute, it will need to be changed in the others to match.
    - The threads attribute, if in use, needs to be recomputed if
      anything about the pointing changes (this includes map geometry
      but does not include map components).

    Setting the "threads" argument to certain special values will
    activate different threading computation algorithms:

    - False: do not use threading; to_map projections will be
      single-threaded.
    - 'simple' (or None): compute self.threads using simple map-stripe
      algorithm.
    - 'domdir': compute self.threads using dominant-direction
      algorithm.

    """
    def __init__(self, sight=None, fp=None, geom=None, comps='T',
                 cuts=None, threads=None, det_weights=None):
        self.sight = sight
        self.fp = fp
        self.geom = geom
        self.comps = comps
        self.cuts = cuts
        self.threads = threads
        self.det_weights = det_weights

    @classmethod
    def for_tod(cls, tod, sight=None, fp=None, geom=None, comps='T',
                rot=None, cuts=None, threads=None, det_weights=None,
                timestamps=None, focal_plane=None, boresight=None,
                boresight_equ=None, wcs_kernel=None):
        """Set up a Projection Matrix for a TOD.  This will ultimately call
        the main P constructor, but some missing arguments will be
        extracted from tod and computed along the way.

        To determine the boresight pointing in celestial coordinates
        (ultimately passed to constructor as sight=), the first
        non-None item in the following list is used:

        - the sight= keyword argument.
        - the boresight_equ= keyword argument.
        - the boresight= keyword argument
        - tod.get('boresight_equ')
        - tod.get('boresight')

        If the map geometry geom is not specified, but the wcs_kernel
        is provided, then get_footprint will be called to determine
        the geom.

        """

        if sight is None:
            if boresight_equ is None:
                if boresight is None:
                    boresight_equ = tod.get('boresight_equ')
            if boresight_equ is not None:
                sight = so3g.proj.CelestialSightLine.for_lonlat(
                    boresight_equ.ra, boresight_equ.dec, boresight_equ.get('psi'))
            else:
                timestamps = _valid_arg(timestamps, 'timestamps',  src=tod)
                boresight = _valid_arg(boresight, 'boresight', src=tod)
                assert(boresight is not None)
                sight = so3g.proj.CelestialSightLine.az_el(
                    timestamps, boresight.az, boresight.el, roll=boresight.roll,
                    site='so', weather='typical')
        else:
            sight = _get_csl(sight)

        # Apply a rotation from equatorial to map WCS coordinates.
        if rot is not None:
            sight.Q = rot * sight.Q

        # Set up the detectors in the focalplane
        fp = _valid_arg(focal_plane, 'focal_plane', src=tod)
        fp = so3g.proj.quat.rotation_xieta(fp.xi, fp.eta, fp.get('gamma'))

        if geom is None and wcs_kernel is not None:
            geom = helpers.get_footprint(tod, wcs_kernel, sight=sight)

        return cls(sight=sight, fp=fp, geom=geom, comps=comps,
                   cuts=cuts, threads=threads, det_weights=det_weights)

    @classmethod
    def for_geom(cls, tod, geom, comps='TQU', timestamps=None,
                 focal_plane=None, boresight=None, rot=None, cuts=None):
        """Deprecated, use .for_tod."""
        return cls.for_tod(tod, geom=geom, comps=comps,
                           timestamps=timestamps, focal_plane=focal_plane,
                           boresight=boresight, rot=rot, cuts=cuts)

    def zeros(self, super_shape=None, comps=None):
        """Returns an enmap concordant with this object's configured geometry
        and component count.

        Args:
          super_shape (tuple): The leading dimensions of the array.
            If None, self._comp_count(comps) is used.
          comps: The component list, to override self.comps.

        Returns:
          An enmap with shape super_shape + self.geom[0].

        """
        if super_shape is None:
            super_shape = (self._comp_count(comps), )
        proj = self._get_proj()
        return self._enmapify(proj.zeros(super_shape))

    def to_map(self, tod=None, dest=None, comps=None, signal=None,
               det_weights=None, cuts=None, eigentol=None):
        """Project time-ordered signal into a map.  This performs the operation

            m += P d

        and returns m.

        Args:
          tod: AxisManager; possible source for 'signal', 'det_weights'.
          dest (enmap): the map or array into which the data should be
            accumulated.  (If None, a new enmap is created and
            initialized to zero.)
          signal: The time-ordered data, d.  If None, tod.signal is used.
          det_weights: The per-detector weight vector.  If None,
            self.det_weights will be used; if that is not set then
            uniform weights of 1 are applied.
          cuts: Sample cuts to exclude from processing.  If None,
            self.cuts is used.
          eigentol: This is ignored.

        """
        signal = _valid_arg(signal, 'signal', src=tod)
        det_weights = _not_both(det_weights, self.det_weights,
                                'det_weights', dtype='float32')
        cuts = _not_both(cuts, self.cuts, 'cuts')

        if comps is None: comps = self.comps
        if dest is None:  dest  = self.zeros(comps=comps)

        proj, threads = self._get_proj_threads(cuts=cuts)
        proj.to_map(signal, self._get_asm(), output=dest, det_weights=det_weights, comps=comps, threads=threads)
        return dest

    def to_weights(self, tod=None, dest=None, comps=None, signal=None,
                   det_weights=None, cuts=None):
        """Computes the weights matrix for the uncorrelated noise model and
        returns it.  I.e.:

            W += P N^-1 P^T

        and returns W.  Here the inverse noise covariance has shape
        (n_dets), and carries a single weight (1/var) value for each
        detector.

        Args:
          tod (AxisManager): possible source for det_weights.
          dest (enmap): the map or array into which the weights should
            be accumulated.  (If None, a new enmap is created and
            initialized to zero.)
          det_weights: The per-detector weight vector.  If None,
            tod.det_weights will be used; if that is not set then
            uniform weights of 1 are applied.
          cuts: Sample cuts to exclude from processing.  If None,
            self.cuts is used.

        """
        det_weights = _not_both(det_weights, self.det_weights,
                                'det_weights', dtype='float32')
        cuts = _not_both(cuts, self.cuts, 'cuts')

        if comps is None:
            comps = self.comps
        if dest is None:
            _n   = self._comp_count(comps)
            dest = self.zeros((_n, _n))

        proj, threads = self._get_proj_threads(cuts=cuts)
        proj.to_weights(self._get_asm(), output=dest, det_weights=det_weights, comps=comps, threads=threads)
        return dest

    def to_inverse_weights(self, weights_map=None, tod=None, dest=None,
                           comps=None, signal=None, det_weights=None, cuts=None,
                           eigentol=1e-4,
                           ):
        """Compute an inverse weights map, W^-1, from a weights map.  If no
        weights_map is passed in, it will be computed by calling
        to_weights, passing through all other arguments.

        """
        if weights_map is None:
            logger.info('to_inverse_weights: calling .to_weights')
            weights_map = self.to_weights(
                tod=tod, comps=comps, signal=signal, det_weights=det_weights, cuts=cuts)

        if dest is None:
            dest = self._enmapify(np.zeros_like(weights_map))

        logger.info('to_inverse_weights: calling _invert_weights_map')
        dest[:] = helpers._invert_weights_map(
            weights_map, eigentol=eigentol, UPLO='U')

        return dest

    def remove_weights(self, signal_map=None, weights_map=None, inverse_weights_map=None,
                       dest=None, **kwargs):
        """Apply the inverse weights matrix to a signal map.

              m' = W^-1 m

        If W or m are not fully specified, they will be computed by
        calling other routines inline, with relevant arguments passed
        through.

        Args:
          signal_map: The map m to filter.
          inverse_weights_map: the matrix W^-1 to apply to the map.
            Shape should be (n_comp, n_comp, n_row, n_col), but only
            the upper diagonal in the first two dimensions needs to be
            populated.  If this is None, then "weights_map" is taken
            as W, and it will be inverted and applied.
          weights_map: the matrix W.  Shape should be (n_comp, n_comp,
            n_row, n_col), but only the upper diagonal in the first
            two dimensions needs to be populated.  If this is None,
            then W will be computed by a call to

        """
        if inverse_weights_map is None:
            inverse_weights_map = self.to_inverse_weights(weights_map=weights_map, **kwargs)
        if signal_map is None:
            signal_map = self.to_map(**kwargs)
        if dest is None:
            dest = self._enmapify(np.empty(signal_map.shape, signal_map.dtype))

        dest[:] = helpers._apply_inverse_weights_map(
            inverse_weights_map, signal_map)
        return dest

    def from_map(self, signal_map, dest=None, comps=None, wrap=None,
                 cuts=None, tod=None):
        """Project from a map into the time-domain.

            d += P m

        Args:
          signal_map: The map m.  This can probably be just about
            anything supported by so3g.proj; it doesn't have to match
            the internally configured geometry.
          dest: Time-ordered data array, shape (dets, samps).  If
            None, a new array will be created to hold the result.
            Otherwise, data are *accumulated* into d, so clear it
            manually if you are trying to do d = P m.
          comps (str): Projection components, if you want to override.
          cuts: RangesMatrix, shape (dets, samps) flagging samples
            that should not be populated.  Defaults to empty.
          wrap (str): If specified, wraps the result as tod[wrap]
            (after removing whatever was in there).

        Returns:
            The dest array.

        Notes:
            Since this is a set of one-to-many operation, OpenMP can
            be used without carefully assigning samples to threads.

        """
        assert cuts is None  # whoops, not implemented.

        proj = self._get_proj()
        if comps is None:
            comps = self.comps
        tod_shape = (len(self.fp), len(self.sight.Q))
        if dest is None:
            dest = np.zeros(tod_shape, np.float32)
        assert(dest.shape == tod_shape)  # P.fp/P.sight and dest argument disagree
        proj.from_map(signal_map, self._get_asm(), signal=dest, comps=comps)

        if wrap is not None:
            if wrap in tod:
                del tod[wrap]
            tod.wrap(wrap, dest, [(0, 'dets'), (1, 'samps')])

        return dest

    def _comp_count(self, comps=None):
        """Returns the number of spin components for component code comps.

        """
        if comps is None:
            comps = self.comps
        return len(comps)

    def _get_proj(self):
        if self.geom is None:
            raise ValueError("Can't project without a geometry!")
        return so3g.proj.Projectionist.for_geom(*self.geom)

    def _get_proj_threads(self, cuts=None):
        """Return the Projectionist and thread assignment for the present
        geometry.  If self.threads has not yet been computed, it is
        done now.

        """
        proj = self._get_proj()
        if cuts is None:
            cuts = self.cuts
        if self.threads is False:
            return proj, ~cuts
        if self.threads is None:
            self.threads = 'simple'
        if isinstance(self.threads, str):
            if self.threads == 'simple':
                self.threads = proj.assign_threads(self._get_asm())
            elif self.threads == 'domdir':
                asm = self._get_asm()
                self.threads = so3g.proj.mapthreads.get_threads_domdir(
                    asm, asm.dets, shape=self.geom[0], wcs=self.geom[1],
                    offs_rep=asm.dets[::100])
            else:
                raise ValueError('Request for unknown algo threads="%s"' % self.threads)
        if cuts:
            return proj, self.threads * ~cuts
        return proj, self.threads

    def _get_asm(self):
        """Bundles self.fp and self.sight into an "Assembly" for calling
        so3g.proj routines."""
        so3g_fp = so3g.proj.FocalPlane()
        for i, q in enumerate(self.fp):
            so3g_fp[f'a{i}'] = q
        return so3g.proj.Assembly.attach(self.sight, so3g_fp)

    def _enmapify(self, data):
        """Promote a numpy.ndarray to an enmap.ndmap by attaching
        wcs=self.geom[1].  In sensible cases (e.g. data is an ndarray
        or ndmap) this will not cause a copy of the underlying data
        array.

        """
        return enmap.ndmap(data, wcs=self.geom[1])

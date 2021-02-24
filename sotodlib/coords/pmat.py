import so3g.proj
import numpy as np
import scipy

from .helpers import _find_field


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
    - comps: the component code (e.g. 'T', 'TQU', 'QU', ...) that
      specifies the spin-components being modeled in the map.
    - dest: the appropriately shaped array in which to place the
      computed result (as an alternative to allocating a new object to
      store the result).

    Objects of this class cache certain pre-computed information (such
    as the rotation taking boresight coordinates to celestial
    coordinates) and certain context-dependent settings (such as a map
    shape, WCS, and spin-component configuration).  You may want to
    inspect or borrow these results, perhaps to reuse them when
    constructing new instances with slight modifications.  Some of
    those attributes are:

    - sight: A CelestialSightLine, representing the boresight pointing
      in celestial coordinates.
    - focal_plane: A FocalPlane, representing positions of detectors
      in the focal plane.
    - geom: The target map geometry (shape, wcs).
    - comps: The spin-components being mapped, e.g. "T" or "TQU'.
      This sets the leading dimension(s) of output maps.
    - cuts: RangesMatrix indicating what samples to exclude from
      projection operations (the indicated samples have projection
      matrix element 0 in all components).
    - threads: RangesMatrix that assigns ranges of samples to specific
      threads.  This is necessary for TOD-to-map operations that use
      OpenMP.

    """
    def __init__(self, tod, geom=None, comps='TQU', dets=None,
                 timestamps=None, focal_plane=None, boresight=None,
                 sight=None, rot=None, cuts=None):
        if sight is None:
            timestamps = _find_field(tod, 'timestamps',  timestamps)
            boresight  = _find_field(tod, 'boresight',   boresight)
            sight = so3g.proj.CelestialSightLine.az_el(
                timestamps, boresight.az, boresight.el, roll=boresight.roll,
                site='so', weather='typical')
        else:
            sight = so3g.proj.quat.G3VectorQuat(sight)
        # Set up the detectors in the focalplane
        dets       = _find_field(tod, tod.dets.vals, dets)
        fp         = _find_field(tod, 'focal_plane', focal_plane)
        fp         = so3g.proj.FocalPlane.from_xieta(dets, fp.xi, fp.eta, fp.gamma)
        self.asm   = so3g.proj.Assembly.attach(sight, fp)
        self.rot   = rot
        self.cuts  = cuts
        self.set_geom(geom)
        self.default_comps = comps

    @classmethod
    def for_geom(cls, tod, geom, comps='TQU', dets=None, timestamps=None,
                 focal_plane=None, boresight=None, rot=None, cuts=None):
        """This just passes all arguments to __init__, so perhaps we don't
        need it.

        """
        return cls(tod, geom=geom, comps=comps, dets=dets,
                   timestamps=timestamps, focal_plane=focal_plane,
                   boresight=boresight, rot=rot, cuts=cuts)

    def set_geom(self, geom, lazy=True):
        """Set the geometry, self.geom.  If not lazy, then re-estimation of
        proj threads is performed immediately (potentially expensive).

        """
        self.geom    = geom
        self.proj    = None
        self.threads = None
        if not lazy:
            self._get_proj_threads()

    def set_cuts(self, cuts, lazy=True):
        """Updates self.cuts.  If not lazy, then re-estimation of proj threads
        is performed immediately (potentially expensive).

        """
        self.cuts    = cuts
        self.proj    = None
        self.threads = None
        if not lazy:
            self._get_proj_threads()

    def comp_count(self, comps=None):
        """Returns the number of spin components for component code comps.

        """
        if comps is None:
            comps = self.default_comps
        return len(comps)

    def zeros(self, super_shape=None, comps=None):
        """Returns an enmap concordant with this object's configured geometry
        and component count.

        Args:
          super_shape (tuple): The leading dimensions of the array.
            If None, self.comp_count(comps) is used.
          comps: The component list, to override self.default_comps.

        Returns:
          An enmap with shape super_shape + self.geom[0].

        """
        if super_shape is None:
            super_shape = (self.comp_count(comps), )
        proj, _ = self._get_proj_threads()
        return proj.zeros(super_shape)

    def to_map(self, tod=None, dest=None, comps=None, signal=None, det_weights=None):
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
            tod.det_weights will be used; if that is not set then
            uniform weights of 1 are applied.

        """
        signal      = _find_field(tod, 'signal', signal)
        det_weights = _find_field(tod, None, det_weights)  # note defaults to None

        if comps is None: comps = self.default_comps
        if dest is None:  dest  = self.zeros(comps=comps)

        proj, threads = self._get_proj_threads()
        proj.to_map(signal, self.asm, output=dest, det_weights=det_weights, comps=comps, threads=threads)
        return dest

    def to_weights(self, tod=None, dest=None, comps=None, signal=None, det_weights=None):
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

        """
        det_weights = _find_field(tod, None, det_weights)  # note defaults to None

        if comps is None:
            comps = self.default_comps
        if dest is None:
            _n   = self.comp_count(comps)
            dest = self.zeros((_n, _n))

        proj, threads = self._get_proj_threads()
        proj.to_weights(self.asm, output=dest, det_weights=det_weights, comps=comps, threads=threads)
        return dest

    def to_inverse_weights(self, weights_map=None, tod=None, dest=None,
                           comps=None, signal=None, det_weights=None):
        """Compute an inverse weights map, W^-1, from a weights map.  If no
        weights_map is passed in, it will be computed by calling
        to_weights, passing through all other arguments.

        """
        if weights_map is None:
            weights_map = self.to_weights(tod=tod, comps=comps, signal=signal, det_weights=det_weights)
        # Invert in each pixel.
        if dest is None:
            dest = weights_map * 0
        if weights_map.shape[0] == 1:
            s = weights_map[0,0] != 0
            dest[0,0][s] = 1./weights_map[0,0][s]
        else:
            temp_shape = weights_map.shape[:2] + (-1,)
            w_reshaped = weights_map.reshape(temp_shape)
            dest.shape = temp_shape
            # This is a strong candidate for C-ification.
            for i in range(dest.shape[2]):
                dest[:,:,i] = scipy.linalg.pinvh(w_reshaped[:,:,i], lower=False)
            dest.shape = weights_map.shape
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
            dest = np.empty(signal_map.shape, signal_map.dtype)
        # Is there really no way to avoid looping over pixels?
        iw = inverse_weights_map.reshape(inverse_weights_map.shape[:2] + (-1,))
        sm = signal_map.reshape(signal_map.shape[:1] + (-1,))
        dest.shape = sm.shape
        for i in range(sm.shape[1]):
            dest[:,i] = np.dot(iw[:,:,i], sm[:,i])
        dest.shape = signal_map.shape
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

        proj, _ = self._get_proj_threads()
        if comps is None:
            comps = self.default_comps
        tod_shape = (len(self.asm.dets), len(self.asm.Q))
        if dest is None:
            dest = np.zeros(tod_shape, np.float32)
        assert(dest.shape == tod_shape)  # P.asm and dest argument disagree
        proj.from_map(signal_map, self.asm, signal=dest, comps=comps)

        if wrap is not None:
            if wrap in tod:
                del tod[wrap]
            tod.wrap(wrap, dest, [(0, 'dets'), (1, 'samps')])

        return dest

    def _get_proj_threads(self):
        """Return the Projectionist and thread assignment for the present
        geometry.  If these have not yet been computed and cached, it
        is done now.

        """
        if self.proj is None:
            if self.geom is None:
                raise ValueError("Can't project without a geometry!")
            else:
                self.proj    = so3g.proj.Projectionist.for_geom(*self.geom)
                if self.rot:
                    self.proj.q_celestial_to_native *= self.rot
                self.threads = self.proj.assign_threads(self.asm)
                if self.cuts:
                    self.threads *= self.cuts
        return self.proj, self.threads

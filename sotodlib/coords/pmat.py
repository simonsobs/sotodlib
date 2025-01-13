import so3g.proj
import numpy as np
import scipy
from pixell import enmap, tilemap

from .helpers import _get_csl, _valid_arg, _not_both, _confirm_wcs
from . import helpers
from . import healpix_utils as hp_utils

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
      detector. When constructing with a tod argument, fp can be
      automatically populated from tod.focal_plane; note that if hwp
      is passed as True then the gamma (polarization) angles will be
      reflected (gamma' = -gamma). [dets]
    - geom: The target map geometry. This is a pixell.enmap.Geometry
      object, with attributes .shape and .wcs (for non-tiled rectpix);
      a pixell.tilemap.TileGeometry (if tiled); or a Healpix geometry as
      defined in coords.healpix_utils.
    - comps: String indicating the spin-components to include in maps.
      E.g., 'T', 'QU', 'TQU'.
    - rot: quat giving an additional fixed rotation to apply to get
      from boresight to celestial coordinates.  Not for long...
    - cuts (optional): RangesMatrix indicating what samples to exclude
      from projection operations (the indicated samples have
      projection matrix element 0 in all components). [dets, samps]
    - threads (optional): list of RangesMatrix objects, to control the
      use of threads in TOD-to-map operations using OpenMP.  See so3g
      documentation. [[threads, dets, samps], ...]
    - det_weights (optional): weights (one per detector) to apply to
      time-ordered data when binning a map (and also when binning a
      weights matrix).  [dets]
    - interpol (optional): How to interpolate the values for samples
      between pixel centers. Forwarded to Projectionist. Valid
      options are:

      - None, 'nn' or 'nearest': Standard nearest neighbor mapmaking.
      - 'lin' or 'bilinear': Linearly interpolate between the four
        closest pixels.

      Default: None

    These things can be updated freely, with the following caveats:

    - If the number of "samples" or "detectors" is changed in one
      attribute, it will need to be changed in the others to match.
    - The threads attribute, if in use, needs to be recomputed if
      anything about the pointing changes (this includes map geometry
      but does not include map components).

    Setting the "threads" argument to certain special values will
    activate different thread assignment algorithms:

    - False: do not use threading; to_map projections will be
      single-threaded.
    - True: use the default algorithm, 'domdir'.
    - None: same as True.
    - 'simple': compute self.threads using simple map-stripe
      algorithm.
    - 'domdir': compute self.threads using dominant-direction
      algorithm (recommended).
    - 'tiles': for tiled geometries, design self.threads such that
      each tile is assigned to a single thread (each thread may be in
      charge of multiple tiles).

    """
    def __init__(self, sight=None, fp=None, geom=None, comps='T',
                 cuts=None, threads=None, det_weights=None, interpol=None):
        self.sight = sight
        self.fp = fp
        self.geom = wrap_geom(geom)
        self.comps = comps
        self.cuts = cuts
        self.threads = threads
        self.active_tiles = None
        self.det_weights = det_weights
        self.interpol = interpol
        self.pix_scheme = _infer_pix_scheme(self.geom)

        if self.pix_scheme == "healpix" and (self.geom.nside_tile is not None):
            if isinstance(self.geom.nside_tile, int):
                self.geom.nside_tile = min(self.geom.nside, self.geom.nside_tile)
            self.geom.ntile = True # Not used for healpix but needed for tiled()

    @classmethod
    def for_tod(cls, tod, sight=None, geom=None, comps='T',
                rot=None, cuts=None, threads=None, det_weights=None,
                timestamps=None, focal_plane=None, boresight=None,
                boresight_equ=None, wcs_kernel=None, weather='typical',
                site='so', interpol=None, hwp=False, qp_kwargs={}):
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

        qp_kwargs are passed through to qpoint to compute sight
        if sight and boresight_equ args are None.

        If the map geometry geom is not specified, but the wcs_kernel
        is provided, then get_footprint will be called to determine
        the geom.

        The per-detector positions and polarization directions are
        extracted from focal_plane (xi, eta, gamma), or else
        tod.focal_plane.  If hwp=True, then the rotation angles
        (gamma) are reflected (gamma' = -gamma) before storing in
        self.fp.

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
                    site=site, weather=weather, **qp_kwargs)
        else:
            sight = _get_csl(sight)

        # Apply a rotation from equatorial to map WCS coordinates.
        if rot is not None:
            sight.Q = rot * sight.Q

        # Set up the detectors in the focalplane
        fp = _valid_arg(focal_plane, 'focal_plane', src=tod)
        xi, eta, gamma  = fp.xi, fp.eta, fp.get('gamma')
        if np.any(np.isnan(np.array([xi, eta]))):
            raise ValueError('np.nan is included in xi or eta')
        if gamma is not None and np.any(np.isnan(gamma)):
            raise ValueError('np.nan is included in gamma')
        assert (gamma is not None or not hwp)
        if hwp:
            gamma = -gamma
        fp = so3g.proj.quat.rotation_xieta(xi, eta, gamma)

        if geom is None and wcs_kernel is not None:
            geom = helpers.get_footprint(tod, wcs_kernel, sight=sight)

        return cls(sight=sight, fp=fp, geom=geom, comps=comps,
                   cuts=cuts, threads=threads, det_weights=det_weights,
                   interpol=interpol)

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
          An enmap with shape super_shape + self.geom.shape.

        """
        if super_shape is None:
            super_shape = (self._comp_count(comps), )
        proj, _ = self._get_proj_tiles()
        if self.pix_scheme == 'healpix':
            return proj.zeros(super_shape)
        elif self.pix_scheme == 'rectpix':
            if self.tiled:
                return tilemap.from_tiles(proj.zeros(super_shape), self.geom)
            else:
                return enmap.ndmap(proj.zeros(super_shape), wcs=self.geom.wcs)

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
        proj.to_map(signal, self._get_asm(), output=self._prepare_map(dest),
                det_weights=det_weights, comps=comps, threads=unwrap_ranges(threads))
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
        proj.to_weights(self._get_asm(), output=self._prepare_map(dest),
                det_weights=det_weights, comps=comps, threads=unwrap_ranges(threads))
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

        weights_map, uf_info = self._flatten_map(weights_map)

        if dest is not None:
            dest, uf_info = self._flatten_map(dest, uf_info)

        _dest = helpers._invert_weights_map(
                weights_map, eigentol=eigentol, UPLO='U')
        if dest is not None:
            dest[:] = _dest
        else:
            dest = _dest
        del _dest

        dest = self._unflatten_map(dest, uf_info)
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
            then W will be computed and inverted via
            ``self.to_inverse_weights``.

        """
        if inverse_weights_map is None:
            inverse_weights_map = self.to_inverse_weights(weights_map=weights_map, **kwargs)
        if signal_map is None:
            signal_map = self.to_map(**kwargs)

        # Get flat numpy-compatible forms for the maps.
        signal_map, uf_info = self._flatten_map(signal_map)
        inverse_weights_map, uf_info = self._flatten_map(inverse_weights_map, uf_info)

        if dest is not None:
            dest, uf_info = self._flatten_map(dest, uf_info)

        dest = helpers._apply_inverse_weights_map(inverse_weights_map, signal_map, out=dest)
        dest = self._unflatten_map(dest, uf_info)
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

        # This is not free but it is pretty fast, doesn't do thread
        # assignments.
        proj, _ = self._get_proj_tiles()

        if comps is None:
            comps = self.comps
        tod_shape = (len(self.fp), len(self.sight.Q))
        if dest is None:
            dest = np.zeros(tod_shape, np.float32)
        assert(dest.shape == tod_shape)  # P.fp/P.sight and dest argument disagree

        if self.tiled and self.pix_scheme == 'rectpix':
            # so3g <= 0.1.15 has a dims check on signal_map that fails on the tiled map format.
            so3g.proj.wcs._ProjectionistBase.from_map(
                proj, self._prepare_map(signal_map), self._get_asm(), signal=dest, comps=comps)
        else:
            proj.from_map(self._prepare_map(signal_map), self._get_asm(), signal=dest, comps=comps)

        if wrap is not None:
            if wrap in tod:
                del tod[wrap]
            tod.wrap(wrap, dest, [(0, 'dets'), (1, 'samps')])

        return dest

    @property
    def tiled(self):
        """Duck-typing to see if we're tiled or not. Reload-safe, unlike isinstance"""
        try:
            self.geom.ntile
            return True
        except AttributeError:
            return False

    def _comp_count(self, comps=None):
        """Returns the number of spin components for component code comps.

        """
        if comps is None:
            comps = self.comps
        return len(comps)

    def _get_proj(self):
        # Backwards compatibility for old so3g
        interpol_kw = _get_interpol_args(self.interpol)
        if self.geom is None:
            raise ValueError("Can't project without a geometry!")
        if self.pix_scheme == "healpix":
            return so3g.proj.ProjectionistHealpix.for_healpix(
                self.geom.nside, self.geom.nside_tile, self.active_tiles,
                self.geom.ordering, **interpol_kw)
        elif self.pix_scheme == "rectpix":
            if self.tiled:
                return so3g.proj.Projectionist.for_tiled(
                    self.geom.shape, self.geom.wcs, self.geom.tile_shape,
                    active_tiles=self.active_tiles, **interpol_kw)
            else:
                return so3g.proj.Projectionist.for_geom(self.geom.shape,
                    self.geom.wcs, **interpol_kw)

    def _get_proj_tiles(self, assign=False):
        # Get Projectionist and compute self.active_tiles if it's not
        # already known.  Return Projectionist with active_tiles set,
        # which is suitable for from_map and zeros (though not for
        # threaded to_map etc).
        proj = self._get_proj()
        if not self.tiled or (self.active_tiles is not None and not assign):
            return proj, {}
        tile_info = proj.get_active_tiles(self._get_asm(), assign=assign)
        self.active_tiles = tile_info['active_tiles']

        if self.pix_scheme == "healpix":
            self.geom.nside_tile = proj.nside_tile # Update nside_tile if it was 'auto'
            self.geom.ntile = 12*proj.nside_tile**2
        elif self.pix_scheme == 'rectpix':
            # Promote geometry to one with the active tiles marked.
            self.geom = tilemap.geometry(
                self.geom.shape, self.geom.wcs, self.geom.tile_shape,
                active=self.active_tiles)

        return self._get_proj(), tile_info

    def _get_proj_threads(self, cuts=None):
        """Return the Projectionist and sample-thread assignment for the
        present geometry.  If the thread assignment has not been
        determined yet, it is done now and cached in self.threads.  In
        tiled geometries, if self.active_tiles has not been
        determined, that is done now and cached.

        The returned sample-thread assignment is modified by "cuts",
        which defaults to self.cuts if passed as None.  I.e. after
        computing or looking up the full self.threads, the code
        returns (proj, self.threads*~cuts).

        Returns:
          Tuple (proj, threads*cuts).

        """
        proj = self._get_proj()
        if cuts is None:
            cuts = self.cuts

        if self.threads is None:
            if self.pix_scheme == "rectpix":
                self.threads = 'domdir'
            elif self.pix_scheme == "healpix":
                self.threads = 'tiles' if (self.geom.nside_tile is not None) else 'simple'

        need_tiles = (self.active_tiles is None)
        need_assign = (self.threads in ['tiles'])
        if need_tiles or need_assign:
            proj, tile_info = self._get_proj_tiles(need_assign)
        if need_assign:
            _tile_threads = wrap_ranges(tile_info['group_ranges'])

        if self.threads is False:
            return proj, ~cuts

        if isinstance(self.threads, str):
            if self.threads in ['simple', 'domdir']:
                logger.info(f'_get_proj_threads: assigning using "{self.threads}"')
                self.threads = wrap_ranges(proj.assign_threads(
                    self._get_asm(), method=self.threads))
            elif self.threads == 'tiles':
                # Computed above unless logic failed us...
                self.threads = _tile_threads
            else:
                raise ValueError('Request for unknown algo threads="%s"' % self.threads)
        if cuts:
            threads = [_t * ~cuts for _t in self.threads]
        else:
            threads = self.threads
        return proj, threads

    def _get_asm(self):
        """Bundles self.fp and self.sight into an "Assembly" for calling
        so3g.proj routines."""
        so3g_fp = so3g.proj.FocalPlane()
        for i, q in enumerate(self.fp):
            so3g_fp[f'a{i}'] = q
        return so3g.proj.Assembly.attach(self.sight, so3g_fp)

    def _prepare_map(self, map):
        """Gently reformat a map in order to send it to so3g."""
        if self.tiled and self.pix_scheme == "rectpix":
            return list(map.tiles)
        else:
            return map

    def _flatten_map(self, map, uf_base=None):
        """Get a version of the map that is a numpy array, for passing
        to per-pixel math operations.  Relies on (self.pix_scheme,
        self.tiled) to interpret map.

        This also tries to extract wcs info (if rectpix) from the map,
        for inline consistency checking (e.g. so we're not happily
        projecting into a map we loaded from disk that has the same
        shape but is off by a few pixels from what pmat thinks is the
        right footprint).  It also looks at active_tiles / tile_list,
        and stores that for downstream compatibility checking with
        other flattened maps.

        If uf_base is passed in, it should be an unflatten_info dict
        (likely from a previous call to _flatten_map).  The analysis
        here will be checked against it for compatibility and any
        missing values (ahem wcs) will be used to augment the
        unflatten_info that is returned.

        Returns:
          array: The map, reformatted as an array (could simply be the
            input arg map, or a view of that, or a copy if necessary).
          unflatten_info: dict with misc compatibility info.

        """
        ufinfo = {'pix_scheme': self.pix_scheme,
                  'tiled': self.tiled}
        wcs = None
        crit_dims = 1
        if self.pix_scheme == 'healpix':
            if self.tiled:
                ufinfo['tile_list'] = [_m is not None for _m in map]
                map = hp_utils.tiled_to_compressed(map, -1)
                crit_dims = 2
            else:
                pass
        elif self.pix_scheme == 'rectpix':
            if self.tiled:
                if isinstance(map, tilemap.TileMap):
                    wcs = map.geometry.wcs
                    ufinfo['active_tiles'] = list(map.active)
                    ufinfo['tile_geom'] = map.geometry.copy(pre=())
            else:
                if isinstance(map, enmap.ndmap):
                    wcs = map.wcs
                crit_dims = 2
        ufinfo.update({'wcs': wcs,
                       'crit_dims': crit_dims,
                       'shape': map.shape})
        if uf_base is not None:
            ufinfo['wcs'] = _check_compat(uf_base, ufinfo)
        return map, ufinfo

    def _unflatten_map(self, map, uf_info):
        """Restore a map to full format, assuming it's currently an
        ndarray.  Intended as the inverse op to _flatten_map.
        Minimize the use of cached self.* here ... rely instead on
        uf_info.

        """
        if uf_info['pix_scheme'] == 'healpix':
            if uf_info['tiled']:
                map = hp_utils.compressed_to_tiled(map, uf_info['tile_list'], -1)
            else:
                pass
        elif uf_info['pix_scheme'] == 'rectpix':
            if uf_info['tiled']:
                if not isinstance(map, tilemap.TileMap):
                    g = uf_info['tile_geom']
                    g = tilemap.geometry(map.shape[:-1] + g.shape, g.wcs, g.tile_shape,
                                         active=uf_info['active_tiles'])
                    map = tilemap.TileMap(map, g)
            else:
                if not isinstance(map, enmap.ndmap) and uf_info['wcs']:
                    map = enmap.ndmap(map, uf_info['wcs'])
        return map


class P_PrecompDebug:
    def __init__(self, geom, pixels, phases):
        self.geom   = wrap_geom(geom).nopre
        self.pixels = pixels
        self.phases = phases
    def zeros(self, super_shape=None):
        if super_shape is None: super_shape = (self.phases.shape[2],)
        return enmap.zeros(super_shape + self.geom.shape, self.geom.wcs)
    def to_map(self, dest=None, signal=None, comps=None):
        if dest is None: dest  = self.zeros()
        proj = so3g.ProjEng_Precomp_NonTiled()
        proj.to_map(dest, self.pixels, self.phases, signal, None, None)
        return dest
    def from_map(self, signal_map, dest=None, comps=None):
        if dest is None: dest = np.zeros(self.pixels.shape[:2], np.float32)
        proj  = so3g.ProjEng_Precomp_NonTiled()
        proj.from_map(signal_map, self.pixels, self.phases, dest)
        return dest

def wrap_geom(geom):
    if isinstance(geom, tuple) or isinstance(geom, list):
        return enmap.Geometry(*geom)
    else:
        return geom

# Helpers for backwards compatibility with old so3g. Consider removing
# these once transition is done.  The idea is for sotodlib to use the
# more recent interface ("threads" is a list of RangesMatrix objects)
# while supporting use with older so3g (where "threads" is a single
# 3-d RangesMatrix).

def wrap_ranges(ranges):
    # Run this on the "threads" result returned from so3g thread
    # assignement routines, before storing the result internally.
    if _so3g_ivals_format() == 1:
        return [ranges]
    else:
        return ranges

def unwrap_ranges(ranges):
    # Run this on the internally stored threads object before passing
    # it to so3g projection routines.
    if _so3g_ivals_format() == 1:
        assert len(ranges) == 1, "Old so3g only supports simple (1-bunch) thread ranges, but got thread ranges with shape %s" % (str(ranges.shape))
        return ranges[0]
    else:
        return ranges

def _so3g_ivals_format():
    projclass = so3g.proj.Projectionist
    if not hasattr(projclass, '_ivals_format'):
        return 1
    else:
        return projclass._ivals_format

def _get_interpol_args(interpol):
    if _so3g_ivals_format() >= 2:
        return {'interpol': interpol}
    assert interpol in [None, "nn", "nearest"], "Old so3g does not support interpolated mapmaking"
    return {}

def _infer_pix_scheme(geom):
    fail_err = ValueError(f"Cannot determine pix_scheme from geom {geom}")

    # Healpix: geom contains 'nside' attribute
    if hasattr(geom, "nside"):
        pix_scheme = "healpix"

    # Rectpix: geom is None, an enmap Geometry or TileGeometry or a tuple/list that can be converted to one
    elif isinstance(geom, enmap.Geometry) or isinstance(geom, tilemap.TileGeometry):
        pix_scheme = "rectpix"
    elif isinstance(geom, tuple) or isinstance(geom, list):
        try:
            enmap.Geometry(*geom)
            pix_scheme = "rectpix"
        except:
            raise fail_err
    elif geom is None:
        pix_scheme = "rectpix"
    else:
        raise fail_err
    return pix_scheme

def _check_compat(*uf_infos):
    # Given one or more "uf_info" dicts, as returned by _flatten_map,
    # check that the flattened arrays are pixel-correspondent,
    # including (for rectpix cases) the wcs.
    #
    # Raises an error if any of that doesn't pan out.  On success,
    # returns the agreed-upon wcs (which could be None).
    ref_uf = uf_infos[0]
    for uf in uf_infos[1:]:
        for k in ['pix_scheme', 'tiled', 'active_tiles']:
            if ref_uf.get(k) != uf.get(k):
                raise ValueError(f"Inconsistent map structures: {uf_infos}")
        # Not sure how to handle broadcasting of lefter dims, so focus
        # on the pixel dim(s).
        dims_to_check = ref_uf['crit_dims']
        if ref_uf['shape'][-dims_to_check:] != uf['shape'][-dims_to_check:]:
            raise ValueError(f"Non-broadcastable map shapes: {uf_infos}")

    # And the wcs.
    wcss = [uf.get('wcs') for uf in uf_infos]
    wcs_to_use = _confirm_wcs(*wcss)

    return wcs_to_use

import so3g.proj
import numpy as np
import scipy

from .helpers import _find_field


class P:
    """Pointing Matrix.  This class caches certain pre-computed
    information (such as rotation taking boresight coordinates to
    celestial coordinates) and certain context-dependent settings
    (such as a map shape, WCS, and spin-component configuration), and
    provides easy access to projection, deprojection, and coordinate
    routines.

    """
    @classmethod
    def for_geom(cls, tod, geom, comps='TQU',
                 dets=None, timestamps=None, focal_plane=None, boresight=None):
        dets = _find_field(tod, tod.dets.vals, dets)
        timestamps = _find_field(tod, 'timestamps', timestamps)
        boresight = _find_field(tod, 'boresight', boresight)
        fp = _find_field(tod, 'focal_plane', focal_plane)
        sight = so3g.proj.CelestialSightLine.az_el(
            timestamps, boresight.az, boresight.el, roll=boresight.roll,
            site='so', weather='typical')
        fp = so3g.proj.FocalPlane.from_xieta(dets, fp.xi, fp.eta, fp.gamma)
        asm = so3g.proj.Assembly.attach(sight, fp)

        self = cls()
        self.sight = sight
        self.asm = asm
        self.default_comps = comps
        self.geom = geom
        return self

    def comp_count(self, comps=None):
        if comps is None:
            comps = self.default_comps
        return len(comps)

    def zeros(self, super_shape=None, comps=None):
        if super_shape is None:
            super_shape = (self.comp_count(comps), )
        proj = so3g.proj.Projectionist.for_geom(*self.geom)
        return proj.zeros(super_shape)

    def to_map(self, tod=None, dest=None, comps=None, signal=None, det_weights=None):
        signal = _find_field(tod, 'signal', signal)
        det_weights = _find_field(tod, None, det_weights)  # note defaults to None

        if comps is None:
            comps = self.default_comps
        if dest is None:
            dest = self.zeros(comps=comps)

        proj = so3g.proj.Projectionist.for_geom(self.geom[0], self.geom[1])

        proj.to_map(signal, self.asm, output=dest, det_weights=det_weights, comps=comps)
        return dest

    def to_weights(self, tod=None, dest=None, comps=None, signal=None, det_weights=None):
        det_weights = _find_field(tod, None, det_weights)  # note defaults to None

        if comps is None:
            comps = self.default_comps
        if dest is None:
            _n = self.comp_count(comps)
            dest = self.zeros((_n, _n))

        proj = so3g.proj.Projectionist.for_geom(*self.geom)

        proj.to_weights(self.asm, output=dest, det_weights=det_weights, comps=comps)
        return dest

    def to_inverse_weights(self, weights_map=None,
                           tod=None, dest=None, comps=None, signal=None, det_weights=None):
        if weights_map is None:
            weights_map = self.to_weights(tod=tod, comps=comps, signal=signal,
                                          det_weights=det_weights)
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

    def remove_weights(self, signal_map=None, weights_map=None,
                       inverse_weights_map=None, dest=None,
                       **kwargs):
        if inverse_weights_map is None:
            inverse_weights_map = self.to_inverse_weights(
                weights_map=weights_map, **kwargs)
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

    def from_map(self, signal_map, tod=None, dest=None, comps=None,
                 wrap=None):
        proj = so3g.proj.Projectionist.for_geom(*self.geom)
        if comps is None:
            comps = self.default_comps
        tod_shape = (len(self.asm.dets), len(self.asm.Q))
        if dest is None:
            dest = np.zeros(tod_shape, 'float32')
        assert(dest.shape == tod_shape)  # P.asm and dest argument disagree
        proj.from_map(signal_map, self.asm, signal=dest, comps=comps)
        return dest

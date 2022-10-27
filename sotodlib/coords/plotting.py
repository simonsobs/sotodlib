from matplotlib.axes import Axes
from matplotlib.ticker import Formatter, MultipleLocator
from matplotlib.projections import register_projection

import numpy as np
from pixell import enmap

__all__ = ['BeamMapAxes']


class _RescaleFormatter(Formatter):
    """Tick formatter that divides the tick positions by some unit, and
    rounds the result to some number of decimal places for display.

    """
    def __init__(self, unit=1., precision=0):
        self.unit = unit
        self.precision = precision
    def __call__(self, x, pos=None):
        label = x / self.unit
        if self.precision == 0:
            return str(int(np.round(label)))
        return str(np.round(label, self.precision))


class BeamMapAxes(Axes):
    """Axes sub-class adapted for displaying beam maps.  This extends
    standard Axes in two ways:

    - Arrays passed to imshow for rendering must in fact be
      enmap.ndarrays; the wcs will be inspected to set the image
      extent (in degrees).
    - The X and Y scales and axis labels will be automatically
      rescaled, depending on the current viewport, to show
      degrees/arcminutes/arcseconds as appropriate.

    Args:
      unit (str): Force the axis & label units to one of 'deg',
        'arcmin', 'arcsec'.  If this is not set, the units will adapt
        to the viewport.
      n_major (int or (int, int)): The maximum number of major ticks
        to permit on the x and y axes.

    Notes:

      This class isn't used directly but rather is registered in
      matplotlib's projection system under the name "so-beammap",
      which can then be used as a projection argument to pyplot.axes
      or pyplot.subplot.  E.g.::

        ax = plt.subplot(221, projection='so-beammap')

      You can also use the .factor classmethod to achieve the same thing::

        ax = plt.subplot(221, projection=BeamMapAxes.factory())

      This latter form allows for defaults to be overridden::

        ax = plt.subplot(221, projection=BeamMapAxes.factory(units='arcmin', max_n=12))

      You might also find it useful to set the same projection on a
      set of subplots; that's done like this:

          fig, axs = plt.subplots(1, 3, subplot_kwargs={'projection': BeamMapAxes()})

    """
    name = 'so-beammap'

    rules = [
        # min_extent, unitstr
        (    2,  'deg'  ),
        ( 2/60, 'arcmin'),
        (    0, 'arcsec')
    ]

    def __init__(self, fig, rect, unit=None, n_major=None, **kw):
        self._force_unit = unit
        try:
            n_major = tuple(n_major)
        except TypeError:
            n_major = (n_major, n_major)
        self.n_major = n_major
        super().__init__(fig, rect, **kw)

    def _set_lim_helper(self, axis, ex, n):
        if n is None:
            n = 8

        # Extent and upper bound tick spacing
        extent = abs(ex[1] - ex[0])
        step = extent / n

        label = self._force_unit
        if label is None:
            for min_size, label in self.rules:
                if extent > min_size:
                    break
        unit = {'deg': 1., 'arcmin': 1/60, 'arcsec': 1/3600}[label]

        if step < unit:
            # If ticks will be smaller than 1 unit, start at some
            # 10**-n.
            base = 10
            zooms = []
        else:
            # But for ticks that look to be integer multiples of a
            # unit; start at 60**n, and go up in nice factors of 60.
            base = 60
            zooms = [2, 2.5, 2, 1.5, 2, 2]

        step = unit * base**np.floor(np.log(step/unit) / np.log(base))
        while round(abs(extent) / step) > n:
            if not zooms:
                zooms = [2, 2.5, 2]  # 10s.
            step *= zooms.pop(0)

        precision = max(int(np.ceil(np.log10(unit / step))),0)
        axis.set_major_locator(MultipleLocator(step))
        axis.set_major_formatter(_RescaleFormatter(unit, precision))

        return label

    def set_xlim(self, *args, **kw):
        super().set_xlim(*args, **kw)
        label = self._set_lim_helper(self.xaxis, self.get_xlim(), self.n_major[0])
        self.set_xlabel(f'$\\xi$ ({label})')

    def set_ylim(self, *args, **kw):
        super().set_ylim(*args, **kw)
        label = self._set_lim_helper(self.yaxis, self.get_ylim(), self.n_major[1])
        self.set_ylabel(f'$\\eta$ ({label})')

    def imshow(self, *args, **kwargs):
        """Wrapper for Axes.imshow with the following adjustments:

        - The image you pass in must be a 2d enmap.ndarray, i.e. it
          should have a ``wcs`` attribute.
        - kwarg 'origin' must be 'lower' (or omitted)
        - kwarg 'extent' must be None (or omitted); the boundary of
          the map, in degrees, will be used to set the "extent".  (As
          a result, means that any viewport adjustments should be
          passed in degrees, rather than pixel index coords or
          whatever units are being shown on the axis labels.)
        - kwarg 'aspect' will be set to match the pixel aspect ratio,
          unless it is explicitly passed in.

        """
        assert(kwargs.pop('origin', 'lower') == 'lower')
        assert(kwargs.pop('extent', None) is None)

        # just make extent be degrees!
        img = args[0]
        assert(isinstance(img, enmap.ndmap))
        wcs = img.wcs.wcs
        if kwargs.get('aspect') is None:
            # H / W of the pixel
            kwargs['aspect'] = abs(wcs.cdelt[1] / wcs.cdelt[0])
        extent = []
        for n, ax, sgn in [(img.shape[1], 0, -1), (img.shape[0], 1, 1)]:
            lohi = ((np.array([0, n]) - 0.5) + 1 - wcs.crpix[ax]) * wcs.cdelt[ax] + wcs.crval[ax]
            extent.extend(list(lohi * sgn))
        return super().imshow(*args, origin='lower', extent=extent, **kwargs)

    @classmethod
    def factory(cls, **kw):
        class _AxesFactory:
            def __init__(self, **kw):
                self.kw = kw
            def _as_mpl_axes(self):
                return cls, self.kw
        return _AxesFactory(**kw)


# Register 'so-beammap' with matplotlib
register_projection(BeamMapAxes)

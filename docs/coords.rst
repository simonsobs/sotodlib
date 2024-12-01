.. py:module:: sotodlib.coords

================
coords submodule
================

This submodule includes functions for computing celestial coordinates
associated with time-ordered data, and projecting signals from maps to
time domain and back again.

The main categories of functionality provided by this module are: the
Projection Matrix, for quick projections between map and time-ordered
domain; utilities for building useful map geometries.

The maps produced by this module are objects of class pixell.enmap;
see `pixell`_ documentation.

Coordinate systems referred to here are defined in more detail in
the ``coord_sys`` document, in the `tod2maps_docs`_ repository.

.. _`pixell`: https://pixell.readthedocs.io/
.. _`tod2maps_docs`: https://github.com/simonsobs/tod2maps_docs/


Conventions
===========

Functions in this module that work with time-ordered data will work
most easily if the data and pointing information are stored in an
AxisManager, which we will denote ``tod``, according to the following
conventions:

- Axes called "dets" and "samps", with dets a LabelAxis.
- ``tod['signal']``: 2-d array with shape (dets, samps) that carries
  the detector measurements, often called "the data".
- ``tod['timestamps']``: timestamps (as unix timestamps).
- ``tod['boresight']``: child AxisManager with boresight pointing in
  horizon coordinates.
- ``tod['focal_plane']``: child AxisManager, with shape (dets) and
  fields ``xi``, ``eta``, ``gamma`` representing detector position and
  orientation in the focal plane.  See ``coord_sys`` documentation for
  exact details, but following fields:

  - ``'xi'``: position of the detector, measured "to the right" of the
    boresight, i.e. parallel to increasing azimuth.
  - ``'eta'``: position of the detector, measured "up" from the
    boresight, i.e. parallel to increasing elevation.
  - ``'gamma'``: orientation of the detector, measured clockwise from
    North, i.e. 0 degrees is parall to the eta axis, and 90 degrees is
    parallel to the xi axis.

The following item is supported, as an alternative to
``tod['boresight']`` and ``tod['timestamps']``:

- ``tod['boresight_equ']``: child AxisManager giving the boresight
  pointing in equatorial coordinates (``'ra'``, ``'dec'``, ``'psi'``,
  forming a lonlat triple, with ``'psi'`` optional but recommended).

Many functions require an AxisManager ``tod`` as the main
argument, and extract the above fields from that object by default.
To override that, pass the field as a keyword argument.


Coordinate Conversions
======================

To compute the RA and dec of all detectors at all times, use
:func:`get_radec`.  For example::

  # Get coords for all detectors
  radec = sotodlib.coords.get_radec(tod)

  # Plot RA and dec for detector at index 10.
  det_idx = 10
  ra, dec = radec[det_idx,:,0], radec[det_idx,:,1]
  plt.plot(ra, dec)

If want the horizon coordinates, use :func:`get_horizon`.


Map Geometry Assistance
=======================

In the terminology of `pixell`_, a "geometry" is the combination of a
WCS describing a rectangular pixelization, and the number of pixels in
each of the celestial axes.  Supposing `em` is an `enmap` object, then
the geometry is just::

   geom = (em.shape[-2:], em.wcs)

So the geometry is a way to define a specific rectangular grid of
pixels on the celestial sky, independent of a specific enmap.

Here are some common tasks related to geometries that sotodlib.coords
helps with.

**1. Get a map big enough to contain all samples from a TOD.**

First call :func:`get_wcs_kernel` to define the pixelization you want
(without specifying the size of the map).  Then call
:func:`get_footprint` to return a geometry.  For example::

    wcsk = coords.get_wcs_kernel('tan', 0., 0., 0.01*coords.DEG)
    wcs = coords.get_footprint(tod, wcsk)

**2. Given a list of geometries that are mutually pixel-compatible get
a geometry that contains all of them.**

Use :func:`get_supergeom`.

For example, you might use :func:`get_wcs_kernel` to define a
pixelization on the sky, then run :func:`get_footprint` on several
TODs, with that same kernel, to get their individual geometries.  Then
call :func:`get_supergeom` on the set of footprints.


Projection Matrix
=================

Usage
-----

Assuming that you have an AxisManager "tod" with "signal",
"timestamps", "boresight", and "focal_plane" elements, then this
should work::

  from sotodlib import coords

  wcsk = coords.get_wcs_kernel('car', 0., 0., 0.01*coords.DEG)
  P = coords.P.for_tod(tod, wcs_kernel=wcsk)

  map1 = P.remove_weights(tod=tod)
  map1.write('output_car.fits')

It is also possible to make maps in Healpix pixelization::

  import healpy as hp

  hp_geom = coords.healpix_utils.get_geometry(nside=512, nside_tile=8)
  P = coords.P.for_tod(tod, geom=hp_geom)

  map1 = P.remove_weights(tod=tod)
  map1 = coords.healpix_utils.tiled_to_full(map1)
  hp.write_map('output_healpix.fits.gz', map1, nest=True)

For more advice, see :class:`P` in the module reference.  See also the
`pwg-tutorials`_ repository (this may be private to SO members),
especially so_pipeline_pt2_20200623.ipynb.


Background
----------

Mapmaking is based on the model

.. math::

   d = P m + n

In this representation, :math:`d` is a vector of all time-ordered
measurements, :math:`n` is the vector of time-ordered noise, :math:`m`
is a vector of all map "pixels" (including all relevant
spin-components, i.e. T, Q, U), and P is the (sparse) "Projection
Matrix" (also called the "Pointing Matrix") that transfers some
combination of map elements into each sample.

In practice we want to solve the mapmaker's equation:

.. math::

   (P^T N^{-1} P) m = P^T N^{-1} d

It's assumed that we have measurements :math:`d` and projection matrix
:math:`P`, and we somehow know the noise covariance matrix :math:`N =
\langle nn^T \rangle`; the solution of this equation is the maximum
likelihood estimate of the map, :math:`m`.

Note that in the absence of fancy math formatting, that equation would
be written as::

   (P^T N^-1 P) m = P^T N^-1 d

This notation may be used in docstrings and comments.

Anyway, if the matrix acting on :math:`m` in the LHS of the mapmaker's
equation, i.e. P^T N^-1 P, cannot be computed and inverted explicity,
then the equation can be solved through iterative methods such as PCG.

One case in which the inversion of the matrix is tractable is if
the noise matrix is perfectly diagonal (i.e. if the noise in each
detector is white, and if the detectors are not correlated with
each other).  In that case, the pixel-pixel covariance matrix is
nearly diagonal -- the only off-diagonal components are ones that
couple the T,Q,U elements of each sky pixel with each other.
While P^T N^1 P in general has shape (n_pix*n_comp, n_pix*n_comp),
it can be factored into n_pix separate matrices of shape (n_comp,
n_comp).  This is a much smaller structure, whose inverse is
obtained by inverting each (n_comp, n_comp) sub-matrix.

The use of the present class for mapmaking is that it provides the
operation P.  In the case of uncorrelated white noise, it is also
able to assist with the application of N^-1 and the inversion of
P^T N^-1 P.

In "filter and bin" mapmaking, the signal d is filtered to produce
d' = F d, and the above equations are used to model d' instead of
d.  The point of the filter is to suppress bright and/or
correlated noise, so that the noise covariance matrix of the
timestream d' is more plausibly diagonal.  The map m~ made from
data d' will be biased; but since the mapmaking equation can be
solved directly, it is relatively cheap to make maps and the
characterization of the bias involves similarly cheap operations
on (noise-free) simulated signal.

Connecting to the methods of this class, for iterative map-making
with a complicated noise covariance matrix:

1) Compute the RHS, P^T N^-1 d:

   - Apply your inverse noise matrix to your data, to get d' =
     N^-1 d.
   - Call .to_map() with signal set to d'.  The result is P^T N^-1
     d.

2) During iteration, apply (P^T N^-1 P) to a map m:

   - Call from_map on map m, returning a timestream y.
   - Compute inverse-noise filtered timestream  y' = N^-1 y
   - Call to_map with signal y'.  The result is the LHS.

In the case of simple uncorrelated noise:

0) Filter your timestream data somehow; call the result d.

1) Compute (P^T N^-1 d), by calling to_map().  If your noise
   matrix includes per-detector weights, pass them through
   det_weights.

2) Compute (P^T N-1 P)^-1, by calling .to_inverse_weights().  This
   doesn't depend on the data, but only on the pointing
   information.  If your noise matrix includes per-detector
   weights, pass them through det_weights.

3) Apply the inverse pixel-pixel cov matrix to your binned map,
   using .remove_weights.

In fact, .remove_weights can perform steps (1), (2), and (3).  But it
is worth mentioning the full series, because although step (2) is the
most expensive it does not depend on the input data and so its output
map can be re-used for different filters or input signals.

.. _`pwg-tutorials`: https://github.com/simonsobs/pwg-tutorials/


Reference
=========

Class and function references should be auto-generated here.

Projection Matrix
-----------------

.. autoclass:: P
   :members:

Coordinate Conversions
----------------------

.. autofunction:: get_radec
.. autofunction:: get_horiz
.. autoclass:: ScalarLastQuat
   :members:

Map and Geometry helpers
------------------------

.. autofunction:: get_footprint
.. autofunction:: get_wcs_kernel
.. autofunction:: get_supergeom
.. autofunction:: sotodlib.coords.healpix_utils.get_geometry
.. autofunction:: sotodlib.coords.healpix_utils.tiled_to_full

Planet Mapmaking support
------------------------

.. automodule:: sotodlib.coords.planets
   :members:

Mapmaking for HWP data
----------------------

.. automodule:: sotodlib.coords.demod
   :members:

Local coordinate systems
------------------------

.. automodule:: sotodlib.coords.local
   :members:

Focal Plane from Physical Optics
--------------------------------

.. automodule:: sotodlib.coords.optics
   :members:

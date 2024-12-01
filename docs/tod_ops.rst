.. py:module:: sotodlib.tod_ops

=================
tod_ops submodule
=================

This submodule includes functions for processing time-ordered signals,
including time and Fourier domain filtering, PCA, gap-filling, etc.

.. _fourier-filters:

Fourier space filters
=====================

Applying Fourier space filters is facilitated by the
``fourier_filter`` function:

.. autofunction:: fourier_filter


Here's an example, assuming you have time-ordered data in an
AxisManager ``tod``.  Note how we combine two Fourier space filters
using the multiplication operator::

  # Filter with a smooth-edged band, passing 1 Hz to 10 Hz.
  filt = tod_ops.filters.high_pass_sine2(1.) * tod_ops.filters.low_pass_sine2(10.)
  tod_ops.filters.fourier_filter(tod, filt)

Several filters are described in :mod:`sotodlib.tod_ops.filters`.  See the
source code for conventions on defining and wrapping new filters to be
compatible with this system.


tod_ops.filters
===============

.. py:module:: sotodlib.tod_ops.filters

The composable filters that can be passed into fourier_filter should
be auto-documented here.

.. It seems sphinx does handle automodule with decorated functions very gracefully.

.. autofunction:: gain
.. autofunction:: low_pass_butter4
.. autofunction:: high_pass_butter4
.. autofunction:: timeconst_filter
.. autofunction:: timeconst_filter_single
.. autofunction:: tau_filter
.. autofunction:: gaussian_filter
.. autofunction:: low_pass_sine2
.. autofunction:: high_pass_sine2
.. autofunction:: iir_filter

Some types of low, high, and band pass filters can be derived by wrapper functions below.

.. autofunction:: get_lpf
.. autofunction:: get_hpf
.. autofunction:: get_bpf

tod_ops.gapfill
===============

Tutorial
--------

The gapfill submodule includes functions and classes for patching
short segments of a TOD signal.  For example, after identifying
glitches in a timestream, one might want to eliminate those
pathological data points by simply interpolating between good data
segments at each end of the bad sample range.

The ``get_gap_fill`` function will fit low-order polynomial models to
flagged data segments, and return vectors with modeled data that can
be used to fill those gaps.  For example, suppose we have an
AxisManager ``tod`` pre-populated with signal and flags of some kind::

  >>> tod.signal.shape
  (619, 165016)
  >>> tod.source_flags
  RangesMatrix(619,165016)
  >>> gaps = tod_ops.get_gap_fill(tod, flags=tod.source_flags)
  >>> gaps
  ExtractMatrix(619,165016@2.0%)
  >>> gaps.swap(tod)

Note that by default the ``get_gap_fill`` function does not patch the
signal data, and that is why we called ``gaps.swap(tod)``.  (See the
``swap`` argument, however.)  The object returned by
``get_gap_fill`` is an ``ExtractMatrix`` (see reference below).  This
contains only the gap-filled data -- i.e. new values for each sample
marked in tod.source_flags.  The repr string tells us that the
ExtractMatrix is compatible with a signal array of shape (619,
165016), and that it is only actually storing data for 2.0% of the
full 2d matrix.  The ExtractMatrix and Extract classes have several
methods to help move data back and forth between compressed and
expanded representation; see the full class reference below.

Superior gap-filling can sometimes be achieved with a mode-based
model.  For example, when there are large gaps in a single detector
timestream, the signal common mode might be a better model for filling
the gap than a polynomial interpolation based on only the single
detector's good amples.  So if one has a model, such
as the kind returned by the ``tod_ops.pca.get_pca_model`` function,
the function ``get_gap_model`` can be used to lookup the values of the
model at some set of flagged samples::

  >>> model
  AxisManager(weights[dets,eigen], modes[eigen,samps],
      dets:LabelAxis(619), eigen:IndexAxis(10),
      samps:OffsetAxis(165016))
  >>> gaps = tod_ops.get_gap_model(tod, model, flags=tod.source_flags)
  >>> gaps
  ExtractMatrix(619,165016@2.0%)
  >>> gaps.swap(tod)

Note that ``get_gap_model`` returns the same sort of object as
``get_gap_fill``, and again in this example we have swapped the model
into the tod.


Reference
---------

Class and function references should be auto-generated here.

.. autofunction:: sotodlib.tod_ops.get_gap_fill

.. autofunction:: sotodlib.tod_ops.get_gap_fill_single

.. autofunction:: sotodlib.tod_ops.get_gap_model

.. autofunction:: sotodlib.tod_ops.get_gap_model_single

.. autoclass:: sotodlib.tod_ops.gapfill.ExtractMatrix
   :special-members: __init__
   :members:

.. autoclass:: sotodlib.tod_ops.gapfill.Extract
   :special-members: __init__
   :members:

.. autofunction:: sotodlib.tod_ops.gapfill.get_contaminated_ranges

.. _pca-background:

tod_ops.pca
===========

Support for principal component analysis, useful in the
signal-processing context, is provided in this submodule.

.. automodule:: sotodlib.tod_ops.pca
   :members:


tod_ops.detrend
===============

Here are some detrending functions.  Note detrending is applied
automatically by :func:`sotodlib.tod_ops.fourier_filter`, by default,
using this submodule.

.. automodule:: sotodlib.tod_ops.detrend
   :members:


tod_ops.jumps
=============

Functions to find and fix jumps in data.
There are currently three jump finders included here each targeting a different use case:

* :func:`sotodlib.tod_ops.twopi_jumps`: For jumps that are a multiple of 2pi, these are a byproduct of the readout.
* :func:`sotodlib.tod_ops.slow_jumps`: For things that look like jumps when zoomed out, but happen on longer time scales (ie: a short unlock, timing issues, etc.). 
* :func:`sotodlib.tod_ops.find_jumps`: For all other jumps, note here that we expect these to be "true" jumps and happen in a small number of samples. 


When running multiple jump finders in a row it is advisible for performance reasons to cache the noise level used to compute the thresholds
(ie: the output of :func:`sotodlib.tod_ops.std_est`) to avoid recomputing it. This can then be used to pass in variables like `min_size` and `abs_thresh`. 
Another way to increase performance is the set the `NUM_FUTURES` system variable to control how many futures are used to parallelize jump finding and fixing
(currently only used by :func:`sotodlib.tod_ops.find_jumps` and :func:`sotodlib.tod_ops.jumpfix_subtract_heights`).
Remember that at a certain point the overhead from increased parallelization will be greater than the speedup, so use with caution.

Historically the jump finder included options to do multiple passes with recursion and/or iteration.
This was because in very jumpy data smaller jumps can be hard enough to find smaller jumps near large jumps.
This was removed for the practical reason that the detectors that this helps with usually have low enough data quality that they should be cut anyways.
If you would like to use iterative jump finding you can do so with a for loop or similar but remember to jumpfix and gapfill between each itteration.

.. automodule:: sotodlib.tod_ops.jumps
   :members:

tod_ops.azss
=============

Function for binning signal by azimuth and fitting it with legendre polynomials in 
az-vs-signal space. This has been used in ABS and other experiments to get 
an Azimuth synchronous signal (AzSS) template largely due to polarized ground signal 
to remove from the data.

.. autofunction:: sotodlib.tod_ops.azss.get_azss

.. autofunction:: sotodlib.tod_ops.azss.subtract_azss

tod_ops.binning
===============

Function for binning signal along specified axis (i.e. azimuth, time, hwp angle).

.. autofunction:: sotodlib.tod_ops.binning.bin_signal

tod_ops.sub_polyf
=================

Function for remove low order polynominal component in each subscan.

.. automodule:: sotodlib.tod_ops.sub_polyf
   :members:

tod_ops.flags
=============

Module containing functions for generating flags for cuts and tod masking.

.. automodule:: sotodlib.tod_ops.flags
   :members:

tod_ops.fft_ops
===============

Module containing functions for FFTs and related operations.

.. automodule:: sotodlib.tod_ops.fft_ops
   :members:

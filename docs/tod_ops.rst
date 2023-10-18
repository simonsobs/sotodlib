.. py:module:: sotodlib.tod_ops

=================
tod_ops submodule
=================

This submodule includes functions for processing time-ordered signals,
including time and Fourier domain filtering, PCA, gap-filling, etc.


Fourier space filters
=====================

Applying Fourier space filters is facilitated by the
``fourier_filter`` function:

.. autofunction:: fourier_filter


Here's an example, assuming you have time-ordered data in an
AxisManager ``tod``.  Note how we combine two Fourier space filters
using the multiplication operator::

  # Filter with a smooth-edged band, passing 1 Hz to 10 Hz.
  filt = tod_ops.filters.high_pass_sine(1.) * tod_ops.filters.low_pass_sine2(10.)
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

Functions to find jumps in data. The jumps are found by taking the cumulative
sum of mean subtracted data (which may be filtered) and looking for peaks in the
output. Taking the cumulative sum is functionally the same as convolving with
an unit step, so it is acting as a matched filter.

.. automodule:: sotodlib.tod_ops.jumps
   :members:


tod_ops.binning
Function to binning signal by a time-ordered data, such as HWP angle and Azimuth
of boresight.

.. automodule:: sotodlib.tod_ops.binning.bin_signal

tod_ops.sss
=============

Function for fitting legendre polynomials in az-vs-signal space. This has been used in ABS and other experiments to get a scan synchronous signal (SSS) template largely due to polarized ground signal to remove from the data.

.. autofunction:: sotodlib.tod_ops.sss.get_sss

.. autofunction:: sotodlib.tod_ops.sss.subtract_sss
	  

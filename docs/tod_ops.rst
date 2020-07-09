.. py:module:: sotodlib.tod_ops

=================
tod_ops submodule
=================

This submodule includes functions for processing time-ordered signals,
including time and Fourier domain filtering, PCA.


Fourier space filters
=====================

Applying Fourier space filters is facilitated by the ``fourier_filter`` function:

.. autofunction:: fourier_filter


tod_ops.filters
===============

.. py:module:: sotodlib.tod_ops.filters

The composable filters that can be passed into fourier_filter should
be auto-documented here.

.. It seems sphinx does handle automodule with decorated functions very gracefully.

.. autofunction:: low_pass_butter4
.. autofunction:: high_pass_butter4
.. autofunction:: tau_filter
.. autofunction:: gaussian_filter
.. autofunction:: low_pass_sine2
.. autofunction:: high_pass_sine2


tod_ops.pca
===========

Support for principal component analysis, useful in the
signal-processing context, is provided in this submodule.

.. automodule:: sotodlib.tod_ops.pca
   :members:


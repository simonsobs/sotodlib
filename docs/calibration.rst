.. py:module:: sotodlib.site_pipeline.calibration

===========
calibration
===========

General description about the calibration pipeline scripts here.


Command line interface
======================

Usage
-----

To get calibrated angles for detectors, users have to load an axismanger,
which corresponds to the target run of the wire grid operation.
And the calibration script of the wire grid requires the HWP preprocess,
that is, ``hwp.demod_tod``.

The main functions of the wire grid consists of 4 functions::
  - correct_wg_angle
  - wrap_qu_cal
  - fit_with_circle
  - get_cal_gamma

.. code-block:: python

  correct_wg_angle(aman)
  wrap_qu_cal(aman)
  fit_with_circle(aman)
  get_cal_gamma(aman, wrap_aman=True)

AxisManager will have a ``gamma_cal`` field that has::
  - gamma_raw
  - gamma_raw_err
  - gamma
  - gamma_err
  - background_pol_rad
  - background_pol_relative_amp
  - theta_det_instr

.. automodule:: sotodlib.site_pipeline.calibration.wiregrid
    :members:
    :undoc-members:

.. py:module:: sotodlib.site_pipeline.calibration

===========
calibration
===========

General description about the calibration pipeline scripts here.


Command line interface
======================

Usage
-----

To get calibrated detector polarization angle, ``'gamma'``,
pipeline below will be applied to an axismanger,
which corresponds to the target wire grid operation.
And the calibration script of the wire grid requires the HWP preprocess,
that is, ``apply_hwp_angle_model`` and ``hwp.demod_tod``.

``'gamma'`` here is the same definition as in the coords module:
    orientation of the detector, measured clockwise from
    North, i.e. 0 degrees is parall to the eta axis, and 90 degrees is
    parallel to the xi axis.


The main functions of the wire grid consists of 8 functions:

 - load_data
 - wrap_wg_hk
 - correct_wg_angle
 - find_operation_range
 - wrap_wiregrid_data
 - calc_calibration_data_set
 - fit_with_circle
 - get_cal_gamma

One can get calibration results by calling these functions.
``tod`` here stands for an AxisManager. For example::

  # Apply HWP angle model, IIR Filtering, and demodulation before this pipeline

  # Load house-keeping data of the wire grid, L3Config as an exmaple
  raw_data_dict = load_data(L3Config(
      path='./satp1_wg_hkdb.yaml',
      start_time=tod.timestamps[0],
      stop_time=tod.timestamps[-1],))
  tod = wrap_wg_hk(tod, raw_data_dict, is_merge=True)

  # Correct the hardware specific constants of the wire grid
  correct_wg_angle(tod)

  # Find the wire grid operation range in the tod and wrap the calibration data set
  idxi, idxf = find_operation_range(
    tod, stopped_time=10, is_restrict=True, remove_trembling=True)
  wrap_wiregrid_data(tod, idxi, idxf)

  # Analyze the calibration data set and fit them with circles
  calc_calibration_data_set(tod)
  _ = fit_with_circle(tod)

  # Get gamma and wrap it into the tod
  get_cal_gamma(tod, wrap_aman=True, remove_cal_data=True)

Finally, the AxisManager has the field of ``gamma_cal`` that has:

 - ``'gamma'``: gamma, theta_det by the wire grid calibration,
 - ``'gamma_err'``: statistical errors on gamma,
 - ``'wires_relative_power'``: input power of the wire grid itself, that is, input QU minus the center offset,
 - ``'background_pol_relative_power'``: center offset in the QU plane, pseudo or background polarization,
 - ``'background_pol_rad'``: the direction of the center offset,
 - ``'theta_det_instr'``: estimated instrumental polarization response direction

Background
----------

Wire grid calibration is based on the model

.. math::

    \mathrm{d} = \mathrm{I}_{\mathrm{in}} + \left[A_{\mathrm{wire}}\ e^{2i\theta_\mathrm{wire}} + A_{\mathrm{background}}\ e^{2i\theta_\mathrm{bg}} +\mathcal{O}(\varepsilon) \left(\mathrm{CMB}\right)\right]\exp i\left[-4\theta_{\mathrm{HWP}} + 2\theta_{\mathrm{det}}\right] + c.c.

In this representation, :math:`\mathrm{d}` is a raw time-ordered data which consists of
the intensity term and the polrization term.
The intensity of the input signal is represented as :math:`\mathrm{I}_\mathrm{in}`.
The polarization terms includes wires power :math:`A_\mathrm{wire}`, static background :math:`A_\mathrm{background}`, and tiny amount of the CMB.

The wire grid signal and the static background polarization have dependencies of
:math:`2\theta_\mathrm{wire}` and :math:`2\theta_\mathrm{background}`, respectively.

Demodulation of the HWP provides two independent polarization term:

.. math::

    \mathcal{F}_{\mathrm{LP}}\left[\mathcal{F}_{\mathrm{BP}}\left[\mathrm{d}\right] \times \exp(4i\theta_{\mathrm{HWP}})\right] & \simeq A_{\mathrm{background}}\ e^{2i\theta_{\mathrm{bg}}+2i\theta_\mathrm{det}} + A_{\mathrm{wire}}\ e^{2i\theta_{\mathrm{wire}}+2i\theta_\mathrm{det}} \\
    & = (Q_\mathrm{offset} + iU_\mathrm{offset}) + (Q_\mathrm{wire} + iU_\mathrm{wire})

We call the static background polarization the offset term. The calibrated polarization response directions, ``'gamma'``, can be obtained by removing the direction of wires from the input polarization

.. math::

    \Phi & \equiv \arctan\frac{U_{\mathrm{wire}} - U_\mathrm{offset}}{Q_{\mathrm{wire}} - Q_\mathrm{offset}} = 2\theta_{\mathrm{det}}+2\theta_{\mathrm{wire}} \\
    \theta_{\mathrm{det}} & = \frac{1}{2}\left[\Phi-2\theta_{\mathrm{wire}}\right]

This module gives the result of calibration as fields like:

 - ``'gamma_raw'``: the calibrated angle at the j-th measurement step of the wire grid, :math:`\theta^{(j)}_\mathrm{det}`
 - ``'gamma_raw_err'``: the set of the standard deviation of the calibrated angle for each measurement step,
 - ``'gamma'``: the calibrated angle by the wire grid, :math:`\theta_\mathrm{det}`
 - ``'gamma_err'``: the statistical error of the calibration, :math:`\sigma (\theta_\mathrm{det})`
 - ``'wires_relative_power'``: radius of the circle used for the fitting (arbitrary unit),:math:`\arctan([(U_{\mathrm{wire}} - U_\mathrm{offset}) / (Q_{\mathrm{wire}} - Q_\mathrm{offset})])`
 - ``'background_pol_rad'``: direction of the background polarization in radian, :math:`\arctan(U_\mathrm{offset} / Q_\mathrm{offset})`
 - ``'background_pol_relative_power'``: deviation to the origin of the background polarization, :math:`\sqrt{Q_\mathrm{offset}^2 + U_\mathrm{offset}^2}`
 - ``'theta_det_instr'``: polarization angle for the instrumental definition, :math:`0.5\pi - \theta_\mathrm{det}`

.. automodule:: sotodlib.site_pipeline.calibration.wiregrid
    :members:
    :undoc-members:

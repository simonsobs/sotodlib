.. py:module:: sotodlib.site_pipeline.calibration

===========
calibration
===========

General description about the calibration pipeline scripts here.


Command line interface
======================

Usage
-----

To get calibrated polarization angles for detectors, ``'gamma'``,
users have to load an axismanger, which corresponds to the target run
of the wire grid operation. And this calibration script requires the HWP preprocess,
``hwp.demod_tod``, before applying wire grid methods.

``'gamma'`` here is the same definition as in the coords module:
    orientation of the detector, measured clockwise from
    North, i.e. 0 degrees is parall to the eta axis, and 90 degrees is
    parallel to the xi axis. Currently, the results are adjusted to
    the longitudinal direction of the SAT cryostat.


The main functions of the wire grid consists of 4 functions:

 - initialize_wire_grid
 - wrap_qu_cal
 - fit_with_circle
 - get_cal_gamma

One can get calibration results by calling these functions.
``tod`` here stands for an AxisManager. For example::

  # Include the wire grid operation into your AxisManager
  # with hardware correction. If you want to use any other
  # step size data, 5 sec or 20 sec stopped time, please
  # specify stopped_time argument here.
  initialize_wire_grid(tod)

  # Wrap Q and U signal related to the steps of the wire grid
  wrap_qu_cal(tod)

  # Fit the stepped QU signal by a circle
  fit_with_circle(tod)

  # Get calibrated polarization response direction, gamma
  get_cal_gamma(tod, wrap_aman=True, remove_cal_data=True)

Finally, the AxisManager has the field of ``gamma_cal`` that has:

 - ``'gamma_raw'`` : raw estimated value corresponding to each step
 - ``'gamma_raw_err'`` : error of the raw values
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

    \mathrm{d} = \mathrm{I}_{\mathrm{in}} + \left[A_{\mathrm{wire}}\ e^{2i\theta_\mathrm{wire}} + A_{\mathrm{background}}\ e^{2i\theta_\mathrm{bg}} +\mathcal{O}(\varepsilon) \left(\mathrm{CMB, sky}\right)\right]\exp i\left[-4\theta_{\mathrm{HWP}} + 2\theta_{\mathrm{det}}\right] + c.c.

In this representation, :math:`\mathrm{d` is a time-ordered measurements consists of
the Intensity of the input power, :math:`\mathrm{I}_\mathrm{in}` and the polarization terms of
some static background :math:`A_\mathrm{background}`, wires power :math:`A_\mathrm{wire}`,
sky signal, and tiny amount of CMB.

The static background polarization and the wire signal have polarization angle dependencies,
:math:`2\theta_\mathrm{wire}`, and :math:`2\theta_\mathrm{background}`, respectively.
The demodulation by the HWP and the projection by the detector polarization response
directions are multiplied as the overall factor in the polarization term.

Unwrapping the charateristic of HWP from the TOD (Demodulation) gives the static
background(=offset) polarization and the polarized signal by wires independently

.. math::

    \mathcal{F}_{\mathrm{BP}}\left[\mathrm{d}\right] \times \exp(4i\theta_{\mathrm{HWP}}) & \simeq A_{\mathrm{background}}\ e^{2i\theta_{\mathrm{bg}}+2i\theta_\mathrm{det}} + A_{\mathrm{wire}}\ e^{2i\theta_{\mathrm{wire}}+2i\theta_\mathrm{det}} \\
    & = (Q_\mathrm{offset} + iU_\mathrm{offset}) + (Q_\mathrm{wire} + iU_\mathrm{wire})

The calibrated polarization response directions, ``'gamma'``, can be obtained by removing
the direction of wires from the input polarization

.. math::

    \Phi & \equiv \arctan\frac{U_{\mathrm{wire}} - U_\mathrm{offset}}{Q_{\mathrm{wire}} - Q_\mathrm{offset}} = 2\theta_{\mathrm{det}}+2\theta_{\mathrm{wire}} \\
    \theta_{\mathrm{det}} & = \frac{1}{2}\left[\Phi-2\theta_{\mathrm{wire}}\right]

This module gives the result of calibration as fields like:

 - ``'gamma'``: :math:`\theta_\mathrm{det}` the calibrated angle by the wire grid,
 - ``'gamma_err'``: :math:`\sigma (\theta_\mathrm{det})` the statistical error of the calibration,
 - ``'wires_relative_power'``: :math:`\arctan([(U_{\mathrm{wire}} - U_\mathrm{offset}) / (Q_{\mathrm{wire}} - Q_\mathrm{offset})])`,
 - ``'background_pol_relative_power'``: :math:`\sqrt{Q_\mathrm{offset}^2 + U_\mathrm{offset}^2}`
 - ``'background_pol_rad'``: :math:`\arctan(U_\mathrm{offset} / Q_\mathrm{offset})` in radian,
 - ``'theta_det_instr'``: :math:`0.5\pi - \theta_\mathrm{det}`

.. automodule:: sotodlib.site_pipeline.calibration.wiregrid
    :members:
    :undoc-members:

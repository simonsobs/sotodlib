.. py:module:: sotodlib.site_pipeline.calibration

===========
calibration
===========

General description about the calibration pipeline scripts here.


Command line interface
======================

Usage
-----

To get calibrated angles for detectors, ``'gamma'``, users have to load an axismanger,
which corresponds to the target run of the wire grid operation in this pipeline.
And the calibration script of the wire grid requires the HWP preprocess,
that is, ``hwp.demod_tod``.

``'gamma'`` here is the same definition as in the coords module:
    orientation of the detector, measured clockwise from
    North, i.e. 0 degrees is parall to the eta axis, and 90 degrees is
    parallel to the xi axis.


The main functions of the wire grid consists of 4 functions:

 - correct_wg_angle
 - wrap_qu_cal
 - fit_with_circle
 - get_cal_gamma

One can get calibration results by calling these functions.
``tod`` here stands for an AxisManager. For example::

  # Get wires' direction in a single operation with hardware correction
  _, idx_wg_inside = correct_wg_angle(tod)

  # May have to restrict the AxisManger into the opration rangle
  tod.restrict('samps', (idx_wg_inside[0], idx_wg_inside[-1]), in_place=True)

  # Wrap Q and U signal related to the steps of the wire grid
  # stopped_time can be changed for each calibration
  wrap_qu_cal(tod, stopped_time=10)

  # Fit the stepped QU signal by a circle
  fit_with_circle(tod)

  # Get calibrated polarization response direction, gamma
  get_cal_gamma(tod, wrap_aman=True, remove_cal_data=True)

Finally, the AxisManager has the field of ``gamma_cal`` that has:

 - ``'gamma'``: gamma, theta_det by the wire grid calibration
 - ``'gamma_err'``: statistical errors on gamma
 - ``'wires_relative_power'``: input power of the wire grid itself,
    that is, input QU minus the center offset
 - ``'background_pol_relative_power'``: center offset in the QU plane
 - ``'background_pol_rad'``: the direction of the center offset
 - ``'theta_det_instr'``: estimated instrumental polarization response direction

Background
----------

Wire grid calibration is based on the model

.. math::

    \mathrm{d} = \mathrm{I}_{\mathrm{in}} + \left[A_{\mathrm{wire}}\ e^{2i\theta_\mathrm{wire}} + A_{\mathrm{background}}\ e^{2i\theta_\mathrm{bg}} +\mathcal{O}(\varepsilon) \left(\mathrm{CMB, sky}\right)\right]\exp i\left[-4\theta_{\mathrm{HWP}} + 2\theta_{\mathrm{det}}\right] + c.c.

In this representation, :math:`d` is a time-ordered measurements consists of
the Intensity of the input power,:math:`I_\mathrm{in}` and the polarization terms of
some static background :math:`A_\mathrm{background}`, wires power :math:`A_\mathrm{wire}`,
sky signal, and tiny amount of CMB.

The static background polarization and the wire signal have polarization angle dependencies respectively,
:math:`2\theta_\mathrm{wire}`, and :math:`2\theta_\mathrm{background}`.
The demodulation by the HWP and the projection of the polarization response direction of each detector are
multiplied as the overall factor.

Unwrapping the charateristic of HWP from the TOD (demodulation) gives static background polarization and
signal polarized in wires' direction independently::

    \mathcal{F}_{\mathrm{BP}}\left[\mathrm{d}\right] \times \exp(4i\theta_{\mathrm{HWP}}) & \simeq A_{\mathrm{background}}\ e^{2i\theta_{\mathrm{bg}}+2i\theta_\mathrm{det}} + A_{\mathrm{wire}}\ e^{2i\theta_{\mathrm{wire}}+2i\theta_\mathrm{det}} \\
    & = (Q_\mathrm{offset} + iU_\mathrm{offset}) + (Q_\mathrm{wire} + iU_\mathrm{wire})

.. automodule:: sotodlib.site_pipeline.calibration.wiregrid
    :members:
    :undoc-members:

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


The main functions of the wire grid consists of 7 functions:
 - load_data
 - wrap_wg_hk
 - find_operation_range
 - calc_calibration_data_set
 - fit_with_circle
 - get_cal_gamma


One can get calibration results by calling these functions.
``tod`` in the scripts stands for an AxisManager. For example::

  # Apply HWP angle model, IIR Filtering, and demodulation before this pipeline

  # Load house-keeping data of the wire grid
  wg_cfg = wg_config(**wiregrid_config)
  raw_data_dict = load_data(wg_cfg, tod.timestamps[0],tod.timestamps[-1])
  tod = wrap_wg_hk(tod, raw_data_dict, merge=True)

  # Find the wire grid operation range in the tod and wrap the calibration data set
  idxi, idxf = find_operation_range(
    tod, is_restrict=True, remove_trembling=True)

  # Analyze the calibration data set and fit them with circles
  calc_calibration_data_set(tod, idxi, idxf)
  _ = fit_with_circle(tod)

  # Get gamma and wrap it into the tod
  get_cal_gamma(tod, merge=True, remove_cal_data=True)

Here, we load the wire grid data using the ``wiregrid_config``.
The example for SATp1 at NERSC is as follows::

  hk_root: '/global/cfs/cdirs/sobs/data/satp1/hk'
  db_file: '/global/cfs/cdirs/sobs/users/mhasse/work/250404/hkdb-satp1.db'
  site: False  

  aliases:
    enc_count: 'wg-encoder.wgencoder_full.reference_count'
    LSL1: 'wg-actuator.wgactuator.limitswitch_LSL1'
    LSL2: 'wg-actuator.wgactuator.limitswitch_LSL2'
    LSR1: 'wg-actuator.wgactuator.limitswitch_LSR1'
    LSR2: 'wg-actuator.wgactuator.limitswitch_LSR2'
    angleX: 'wg-tilt-sensor.wgtiltsensor.angleX'
    angleY: 'wg-tilt-sensor.wgtiltsensor.angleY'
    tempX: 'wg-tilt-sensor.wgtiltsensor.temperatureX'
    tempY: 'wg-tilt-sensor.wgtiltsensor.temperatureY'
    temp_rotator: 'wg-labjack.sensors_downsampled.AIN0C'
    temp_rotation_motor: 'wg-labjack.sensors_downsampled.AIN1C'
    temp_elec_plate: 'wg-labjack.sensors_downsampled.AIN2C'  
  # hardware specific constants
  wg_count: 52000 # SATP1~SATP3, JSAT(SATP4?) 
  wg_offset: 12.13 # SAT1, MF1
  telescope: 'satp1'

Finally, the AxisManager has the field of ``wg.gamma_cal`` that has:

 - ``'gamma_raw'``: the calibrated angle at the j-th measurement step of the wire grid,
 - ``'gamma_raw_err'``: the set of the standard deviation of the calibrated angle for each measurement step,
 - ``'gamma'``: the calibrated angle by the wire grid,
 - ``'gamma_err'``: the statistical error of the calibration,
 - ``'wires_relative_power'``: radius of the circle used for the fitting (arbitrary unit),
 - ``'background_pol_rad'``: direction of the background polarization in radian,
 - ``'background_pol_relative_power'``: deviation to the origin of the background polarization,
 - ``'theta_det_instr'``: polarization angle for the instrumental definition,

Background
----------

Wire grid calibration is based on the model

.. math::

    \mathrm{d} = \mathrm{I}_{\mathrm{in}} + \left[A_{\mathrm{wire}}\ e^{2i\theta^{(j)}_\mathrm{wire}} + A_{\mathrm{background}}\ e^{2i\theta_\mathrm{bg}} +\mathcal{O}(\varepsilon) \left(\mathrm{CMB}\right)\right]\exp i\left[-4\theta_{\mathrm{HWP}} + 2\theta_{\mathrm{det}}\right] + c.c.

In this representation, :math:`\mathrm{d}` is a raw time-ordered data which consists of
the intensity term and the polrization term.
The intensity of the input signal is represented as :math:`\mathrm{I}_\mathrm{in}`.
The polarization terms includes wires power :math:`A_\mathrm{wire}`, static background :math:`A_\mathrm{background}`, and tiny amount of the CMB.
:math:`\theta^{(j)}_\mathrm{wire}` is the j-th direction of wires. :math:`\theta_\mathrm{bg}` is the direction of the static background polarization. :math:`\theta_\mathrm{HWP}` and :math:`\theta_\mathrm{det}` are the fast axis direction of the HWP and the detector polarization angle, ``gamma``.
In this case, :math:`\theta_\mathrm{det} = \gamma`.

Demodulation of the HWP provides two independent polarization term:

.. math::

    \mathcal{F}_{\mathrm{LP}}\left[\mathcal{F}_{\mathrm{BP}}\left[\mathrm{d}\right] \times \exp(4i\theta_{\mathrm{HWP}})\right] & \simeq A_{\mathrm{background}}\ e^{2i\theta_{\mathrm{bg}}+2i\theta_\mathrm{det}} + A_{\mathrm{wire}}\ e^{2i\theta^{(j)}_{\mathrm{wire}}+2i\theta_\mathrm{det}} \\
    & = (Q_\mathrm{offset} + iU_\mathrm{offset}) + (Q^{(j)}_\mathrm{wire} + iU^{(j)}_\mathrm{wire})

We call the static background polarization the offset term. The calibrated polarization response directions, ``'gamma'`` or :math:`\theta_\mathrm{det}`, can be obtained by removing the direction of wires from the input polarization

.. math::

    \Phi(j) & \equiv \arctan\frac{U^{(j)}_{\mathrm{wire}} - U_\mathrm{offset}}{Q^{(j)}_{\mathrm{wire}} - Q_\mathrm{offset}} = 2\theta_{\mathrm{det}}+2\theta^{(j)}_{\mathrm{wire}} \\
    \theta^{(j)}_{\mathrm{det}} & = \frac{1}{2}\left[\Phi(j)-2\theta^{(j)}_{\mathrm{wire}}\right]

This module provides the averaged :math:`\theta_{det}` as the best estimation of ``gamma``.
Other return fields are listed as follows:

 - ``'gamma_raw'``: :math:`\theta^{(j)}_\mathrm{det}`
 - ``'gamma_raw_err'``: :math:`\sigma (\theta^{(j)}_\mathrm{det})`
 - ``'gamma'``: :math:`\theta_\mathrm{det}`
 - ``'gamma_err'``: :math:`\sigma (\theta_\mathrm{det})`
 - ``'wires_relative_power'``: :math:`\arctan([(U^{(j)}_{\mathrm{wire}} - U_\mathrm{offset})/(Q^{(j)}_{\mathrm{wire}} - Q_\mathrm{offset})])`
 - ``'background_pol_rad'``: :math:`\arctan(U_\mathrm{offset} / Q_\mathrm{offset})`
 - ``'background_pol_relative_power'``: :math:`\sqrt{Q_\mathrm{offset}^2 + U_\mathrm{offset}^2}`
 - ``'theta_det_instr'``: :math:`0.5\pi - \theta_\mathrm{det}`

.. automodule:: sotodlib.site_pipeline.calibration.wiregrid
    :members:
    :undoc-members:

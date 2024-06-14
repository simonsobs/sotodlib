.. py:module:: sotodlib.preprocess.pcore

.. _preprocess-module:

==========
Preprocess
==========


The preprocess module defines a standardized interface for TOD processing
operations so that they can be easily implemented in automatic data analysis
scripts. The core of the system is in two parts, the ``_Preprocess`` modules
and the ``Pipeline`` object. The ``_Preprocess`` modules each define how a TOD
operation is run on an AxisManager TOD and the ``Pipeline`` object is used to
define the order of the operations and then run them. The
``site-pipeline.preprocess_tod`` script is used to run and save Pipelines on
lists of observations, grouped by detset. The ``site-pipeline.preprocess_obs``
script is used for observation-level preprocessing. This module is similar to
``site-pipeline.preprocess_tod`` but removes grouping by detset so that the
entire observation is loaded, without signal. For example, pipeline steps such
as ``DetBiasFlags`` requires tod-level data including signal, whereas
``SSOFootprint`` does not and uses observation-level data.



Preprocessing Pipelines
------------------------

A preprocessing pipeline is series of modules, each inheriting from
``_Preprocess``, that are defined through a configuration file and intended to be
run successively on an AxisManager containing time ordered data.

.. autoclass:: sotodlib.preprocess.pcore._Preprocess
    :members:

The preprocessing pipeline is defined in the ``Pipeline`` class. This class
inherits from list so that you can easily find and interact with the various
pipeline elements. Note that splicing a pipeline will return a list of process
modules that can be used to make a new pipeline.

.. autoclass:: sotodlib.preprocess.pcore.Pipeline
    :members:



Processing Scripts
------------------
These scripts are designed to be the ones that interact with specific
configuration files and specific manifest databases.


.. autofunction:: sotodlib.site_pipeline.preprocess_tod.preprocess_tod

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.load_preprocess_det_select

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.load_preprocess_tod

.. autofunction:: sotodlib.site_pipeline.preprocess_obs.preprocess_obs

.. autofunction:: sotodlib.site_pipeline.preprocess_obs.load_preprocess_obs

Example TOD Pipeline Configuration File
---------------------------------------

Suppose we want to run a simple pipeline that runs the glitch calculator and
estimates the white noise levels of the data. A configuration file for the
processing pipeline would look like::

    # Context for the data
    context_file: 'context.yaml'

    # Plot directory prefix
    plot_dir: './plots'

    # How to subdivide observations
    subobs:
        use: detset
        label: detset

    # Metadata index & archive filenaming
    archive:
        index: 'preprocess_archive.sqlite'
        policy:
            type: 'simple'
            filename: 'preprocess_archive.h5'

    process_pipe:
        - name : "fft_trim"
          process:
            axis: 'samps'
            prefer: 'right'

        - name: "trends"
          calc:
            max_trend: 30
            n_pieces: 5
          save: True
          select:
            kind: "any"

        - name: "glitches"
          calc:
            t_glitch: 0.002
            hp_fc: 0.5
            n_sig: 10
            buffer: 20
          save: True
          select:
            max_n_glitch: 20
            sig_glitch: 30

        - name: "detrend"
          process:
            method: "linear"
            count: 10
        
        - name: "calibrate"
          process:
            kind: "single_value"
            ## phase_to_pA: 9e6/(2*np.pi)
            val: 1432394.4878270582
        
        - name: "psd"
          process:
            detrend: False
            window: "hann"

        - name: "noise"
          calc:
            low_f: 5
            high_f: 10
          save: True
          select:
            max_noise: 2000

This pipeline can be run through the functions saved in ``site_pipeline``. Each
entry in "process_pipe" key will be used to generate a Preprocess module based
on the name it is registered to. These entries will then be run in order through
the processing pipe. The ``process`` function is always run before the
``calc_and_save`` function for each module. The ``plot`` function can be run after
``calc_and_save`` when ``plot: True`` for a module that supports it.

Example Planet TOD Pipeline Configuration File
----------------------------------------------
Similar to a regular TOD pipeline, if we want to run one for planet observations,
we must first flag sources in the signal and gapfill them. An example configuration
file should be equivalent to non-planet data processing after a few extra first
steps::

    # Context for the data
    context_file: 'context.yaml'

    # Plot directory prefix
    plot_dir: './plots'

    # How to subdivide observations
    subobs:
        use: wafer_slot
        label: wafer_slot

    # Metadata index & archive filenaming
    archive:
        index: 'preprocess_archive.sqlite'
        policy:
            type: 'simple'
            filename: 'preprocess_archive.h5'

    process_pipe:
        - name : "dark_dets"
          calc: True
          save: True
          select: True

        - name: "source_flags"
          calc:
            mask: {'shape': 'circle',
                  'xyr': [0, 0, 1.]}
            center_on: 'jupiter' # set to 'planet' for variable according to planet tag of each obs (must use --planet-obs argument of site-pipeline script)
            res: 20 # np.radians(20/60)
            max_pix: 4.0e+6
          save: True

        - name: "glitchfill"
          flag_aman: "sources"
          flag: "source_flags"
          process:
            nbuf: 10
            use_pca: True
            modes: 3

Example Obs Pipeline Configuration File
---------------------------------------

Suppose we want to run an observation-level pipeline that creates a SSO footprint.
A configuration file for the processing pipeline would look like::

    # Context for the data
    context_file: 'context.yaml'

    # Plot directory prefix
    plot_dir: './plots'

    # Metadata index & archive filenaming
    archive:
        index: 'preprocess_archive.sqlite'
        policy:
            type: 'simple'
            filename: 'preprocess_archive.h5'

    process_pipe:
        - name: "sso_footprint"
          calc:
            # If you want to search for nearby sources, exclude source_list
            source_list: ['jupiter']
            distance: 20
            nstep: 100
          save: True
          plot:
            wafer_offsets: {'ws0': [-2.5, -0.5],
                            'ws1': [-2.5, -13],
                            'ws2': [-13, -7],
                            'ws3': [-13, 5],
                            'ws4': [-2.5, 11.5],
                            'ws5': [8.5, 5],
                            'ws6': [8.5, -7]}
            focal_plane: 'focal_plane_positions.npz'

Processing Modules
------------------


TOD Operations
::::::::::::::
.. autoclass:: sotodlib.preprocess.processes.FFTTrim
.. autoclass:: sotodlib.preprocess.processes.Detrend
.. autoclass:: sotodlib.preprocess.processes.PSDCalc
.. autoclass:: sotodlib.preprocess.processes.Apodize
.. autoclass:: sotodlib.preprocess.processes.SubPolyf
.. autoclass:: sotodlib.preprocess.processes.Jumps
.. autoclass:: sotodlib.preprocess.processes.FixJumps
.. autoclass:: sotodlib.preprocess.processes.FourierFilter

Calibration
:::::::::::
.. autoclass:: sotodlib.preprocess.processes.Calibrate
.. autoclass:: sotodlib.preprocess.processes.PCARelCal

Flagging and Products
:::::::::::::::::::::
.. autoclass:: sotodlib.preprocess.processes.Trends
.. autoclass:: sotodlib.preprocess.processes.GlitchDetection
.. autoclass:: sotodlib.preprocess.processes.GlitchFill
.. autoclass:: sotodlib.preprocess.processes.Noise
.. autoclass:: sotodlib.preprocess.processes.FlagTurnarounds
.. autoclass:: sotodlib.preprocess.processes.DarkDets
.. autoclass:: sotodlib.preprocess.processes.SourceFlags

HWP Related
:::::::::::
.. autoclass:: sotodlib.preprocess.processes.EstimateHWPSS
.. autoclass:: sotodlib.preprocess.processes.SubtractHWPSS
.. autoclass:: sotodlib.preprocess.processes.Demodulate

Obs Operations
::::::::::::::
.. autoclass:: sotodlib.preprocess.processes.SSOFootprint

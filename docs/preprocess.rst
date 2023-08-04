.. py:module:: sotodlib.preprocess.core

.. _preprocess-module:

==========
Preprocess
==========


The preprocess module defines a standardized interface for TOD processing
operations so that they can be easily implemented in automatic data analysis
scripts. These are used to build sotodlib / AxisManager based pipelines that can
be run through the preprocess_tod related site-pipeline scripts. 


Preprocessing Pipeline
----------------------

A preprocessing pipeline is series of modules, each inheriting from
``_Preprocess``, that are defined through a configuration file and intended to be
run successively on an AxisManager containing time ordered data.

.. autoclass:: sotodlib.preprocess.core._Preprocess
    :members:


Example Pipeline
----------------

Suppose we want to run a simple pipeline that runs the glitch calculator and
estimates the white noise levels of the data. A configuration file for the
processing pipeline would look like::

    # Context for the data
    context_file: 'context.yaml'

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
``calc_and_save`` function for each module.

Processing Functions
--------------------

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.preprocess_tod

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.run_preprocess

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.load_preprocess_det_select

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.load_preprocess_tod

Processing Modules
------------------


TOD Operations
::::::::::::::
.. autoclass:: sotodlib.preprocess.processes.FFTTrim
.. autoclass:: sotodlib.preprocess.processes.Detrend
.. autoclass:: sotodlib.preprocess.processes.PSDCalc
.. autoclass:: sotodlib.preprocess.processes.Calibrate
.. autoclass:: sotodlib.preprocess.processes.Apodize

Flagging and Products
:::::::::::::::::::::
.. autoclass:: sotodlib.preprocess.processes.Trends
.. autoclass:: sotodlib.preprocess.processes.GlitchDetection
.. autoclass:: sotodlib.preprocess.processes.Noise

HWP Related
:::::::::::
.. autoclass:: sotodlib.preprocess.processes.EstimateHWPSS
.. autoclass:: sotodlib.preprocess.processes.SubtractHWPSS
.. autoclass:: sotodlib.preprocess.processes.Demodulate


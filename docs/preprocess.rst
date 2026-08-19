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

Single-layer vs. two-layer ("multilayer") pipelines
:::::::::::::::::::::::::::::::::::::::::::::::::::

A single ``process_pipe`` config file, run through ``preprocess_tod``, is a
**single-layer** pipeline. For some platforms (in particular the SATs) the
full processing recipe is instead split into **two layers**, run through
``multilayer_preprocess_tod`` with two config files: an **init** (first)
layer and a **proc** (second) layer that depends on it. The proc layer
records which init layer it was built from, so if you try to run a proc
config against a different init archive than the one it was created with, it
will fail rather than silently mixing incompatible layers. Layers are
chainable, so a proc layer's output can itself become the init input for a
further layer, though doing so currently means running that further layer
interactively because there is no ``three_layer_preprocess_tod`` script.

This split was introduced to separate the pre-demodulation and
post-demodulation steps for the SATs: the first (init) layer runs pointing,
flagging, HWP-synchronous-signal removal, calibration, PCA relcal, and
demodulation, producing calibrated demodulated Q/U timestreams; the second
(proc) layer runs the filters that operate on that demodulated data (e.g.
``azss``, ``estimate_t2p``/``subtract_t2p``, ``sub_polyf``). Splitting the
pipeline this way means the second layer's filtering can be iterated on and
re-run without repeating the expensive first-layer processing. LAT configs in
production are currently single-layer only (no init/proc split), reflecting
that LAT does not run the HWP-demodulation/ground-pickup/T-to-P-leakage
processing chain the SATs do in this pipeline.

For real, in-production examples of both designs (as opposed to the
synthetic examples on this page), see the platform config directories in the
`site-pipeline-configs <https://github.com/simonsobs/site-pipeline-configs>`_
repository, e.g. ``satp1/preprocess_config_init.yaml`` +
``satp1/preprocess_config_proc.yaml`` (two-layer) and
``lat/preprocess_config_cmb_mf.yaml`` / ``lat/preprocess_config_cmb_uhf.yaml``
(single-layer).

`[TOD Processing] Infrastructure Introduction <https://simonsobs.atlassian.net/wiki/spaces/DM/pages/305627401/TOD+Processing+Infrastructure+Introduction>`_ has a fuller walkthrough (jobdb usage, extensive interactive/Python examples for loading, running, and saving preprocessing archives) that complements this reference page -- including a deeper look at first-run-vs-rerun behavior than the summary below.

Running for the first time vs. re-running against an existing archive
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

The ``archive`` config key names two things together: ``archive.index``, a
ManifestDb sqlite file keyed on ``obs_id`` plus the ``subobs.use`` grouping
fields (e.g. detset, or ``["wafer_slot", "wafer.bandpass"]``), and
``archive.policy.filename``, the HDF5 file the actual ``proc_aman`` results
are written into. Together these are "the archive" for a given config.

- **First run**: ``archive.index`` doesn't exist yet, so
  :func:`sotodlib.preprocess.preprocess_util.get_preprocess_db` creates a new,
  empty ManifestDb there. Every obs_id/group the query returns is processed
  from scratch -- each pipeline step's ``calc_and_save`` runs, and the results
  are written into the archive as they complete.
- **Re-running**: if ``archive.index`` already exists, ``preprocess_tod``
  inspects it first and drops any obs_id/group already present in the archive
  from the run list, so a second invocation over the same (or an overlapping)
  observation list only processes what's new -- already-archived
  observations are left alone rather than recomputed. Pass ``--overwrite`` to
  force everything in the query to be reprocessed and the existing entries
  replaced instead.
- This is also what makes the archive reusable for **simulations**: when
  :meth:`sotodlib.preprocess.pcore.Pipeline.run` is called with an existing
  ``proc_aman`` (as loaded from the archive), every step's ``calc_and_save``
  is skipped and only ``process()`` re-runs on top of the already-computed
  products -- e.g. reapplying a filter fit on real data to a signal-only sim
  without redoing the fit.

Preprocessing job tracking (``jobdb``)
::::::::::::::::::::::::::::::::::::::

Preprocessing configs can optionally include a top-level ``jobdb`` key giving
the path to a SQLite database that tracks the state (``open``/``done``/
``failed``) of each preprocessing job (an obs_id + group combination).
Passing ``--run-from-jobdb`` to ``preprocess_tod``/``multilayer_preprocess_tod``
resumes a batch run from an existing jobdb's run list instead of rebuilding
one — useful for continuing a large run that was interrupted (e.g. a Slurm
job hitting its walltime) without redoing already-completed or
already-known-to-fail work. It also starts immediately, since the wafer/band
groups don't need to be recomputed. Jobs can be inspected programmatically via
``sotodlib.site_pipeline.jobdb.JobManager``.

Preprocessing Pipelines
:::::::::::::::::::::::

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
::::::::::::::::::
These scripts are designed to be the ones that interact with specific
configuration files and specific manifest databases.


.. autofunction:: sotodlib.site_pipeline.preprocess_tod.preprocess_tod

.. autofunction:: sotodlib.site_pipeline.preprocess_tod.load_preprocess_tod_sim

.. autofunction:: sotodlib.site_pipeline.multilayer_preprocess_tod.multilayer_preprocess_tod

.. autofunction:: sotodlib.site_pipeline.preprocess_obs.preprocess_obs

.. autofunction:: sotodlib.preprocess.preprocess_util.load_and_preprocess

.. autofunction:: sotodlib.preprocess.preprocess_util.multilayer_load_and_preprocess

.. autofunction:: sotodlib.preprocess.preprocess_util.multilayer_load_and_preprocess_sim

.. autofunction:: sotodlib.preprocess.preprocess_util.preproc_or_load_group


Processing Util Functions
:::::::::::::::::::::::::
These functions support and are used within the driver processing scripts
above and are useful for saving, loading, and verifying preprocessing archives
and databases.

.. autoclass:: sotodlib.preprocess.preprocess_util.PreprocessErrors
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: sotodlib.preprocess.preprocess_util.filter_preproc_runlist_by_jobdb

.. autofunction:: sotodlib.preprocess.preprocess_util.init_logger

.. autofunction:: sotodlib.preprocess.preprocess_util.get_preprocess_context

.. autofunction:: sotodlib.preprocess.preprocess_util.get_groups

.. autofunction:: sotodlib.preprocess.preprocess_util.get_preprocess_db

.. autofunction:: sotodlib.preprocess.preprocess_util.swap_archive

.. autofunction:: sotodlib.preprocess.preprocess_util.load_preprocess_det_select

.. autofunction:: sotodlib.preprocess.preprocess_util.find_db

.. autofunction:: sotodlib.preprocess.preprocess_util.get_preproc_group_out_dict

.. autofunction:: sotodlib.preprocess.preprocess_util.save_group_and_cleanup

.. autofunction:: sotodlib.preprocess.preprocess_util.cleanup_obs

.. autofunction:: sotodlib.preprocess.preprocess_util.cleanup_mandb

.. autofunction:: sotodlib.preprocess.preprocess_util.get_pcfg_check_aman

.. autofunction:: sotodlib.preprocess.preprocess_util.check_cfg_match


Example TOD Pipeline Configuration File
:::::::::::::::::::::::::::::::::::::::

Suppose we want to run a simple pipeline that runs the glitch calculator and
estimates the white noise levels of the data. A configuration file for the
processing pipeline would look like::

    # Context for the data
    context_file: 'context.yaml'

    # Plot directory prefix
    plot_dir: './plots'

    # How to subdivide observations
    subobs:
        use: ["wafer_slot", "wafer.bandpass"]
        label: "wafer_slot"

    # Metadata index & archive filenaming
    archive:
        index: 'preprocess_archive.sqlite'
        policy:
            type: 'simple'
            filename: 'preprocess_archive.h5'
        batch_size: 50

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
            kind: "array"
            cal_array: "det_cal.phase_to_pW"

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
::::::::::::::::::::::::::::::::::::::::::::::
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
:::::::::::::::::::::::::::::::::::::::

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

Process Step Glossary
:::::::::::::::::::::

Quick reference for every registered ``process_pipe`` step name (the
``name:`` string you put in a config file), grouped by what it's for.
The "What it does" column is the first line of each class's docstring
-- see that class's full entry below (including ``calc``/``save``/
``select``/``plot`` config options and an example config block) in the
same subsection. This table is generated from
``sotodlib.preprocess.processes``; if you add a new registered process,
add a row to the relevant table below and a matching ``.. autoclass::``
entry in the same subsection --
``tests/test_preprocess_docs.py`` will fail CI if a class is registered
but has no ``.. autoclass::`` entry anywhere on this page.

General / Utility
-----------------

================== ================ ==================================================================================================
``name:``          Class            What it does
================== ================ ==================================================================================================
``fft_trim``       ``FFTTrim``      Trim the AxisManager to optimize for faster FFTs later in the pipeline.
``detrend``        ``Detrend``      Remove mean, median or linear trend from the data.
``move``           ``Move``         Rename or remove a data field (used to replace gamma angles with those from wiregrid for example).
``trim_flag_edge`` ``TrimFlagEdge`` Trim edge until given flags of all detectors are False.
================== ================ ==================================================================================================

.. autoclass:: sotodlib.preprocess.processes.FFTTrim
.. autoclass:: sotodlib.preprocess.processes.Detrend
.. autoclass:: sotodlib.preprocess.processes.Move
.. autoclass:: sotodlib.preprocess.processes.TrimFlagEdge

Detector Cuts and Flags
-----------------------

====================== ====================== ==========================================================================================================================
``name:``              Class                  What it does
====================== ====================== ==========================================================================================================================
``det_bias_flags``     ``DetBiasFlags``       Derive poorly biased detectors from IV and Bias Step data.
``trends``             ``Trends``             Check for large linear ramping in the data to look for unlocked detectors.
``ptp_flags``          ``PTPFlags``           Find (and cut) detectors with anomalous peak-to-peak signal.
``inv_var_flags``      ``InvVarFlags``        Find (and cut) detectors with too high inverse variance.
``cut_bad_dist``       ``CutBadDistribution`` Detector cuts to keep a statistic (i.e white noise, peak-peak, fknee, etc.) within some bounds of a gaussian distribution.
``detcal_nan_cuts``    ``DetcalNanCuts``      Remove detectors with NaN values in the specified det_cal metadata fields.
``fp_flags``           ``FocalplaneNanFlags`` Cut detectors which have nans in their pointing information.
``dark_dets``          ``DarkDets``           Cut dark detectors in the data.
``acu_drop_flags``     ``AcuDropFlags``       Expands ACU drop (bad ACU/platform pointing data) flag fields in aman to all detectors.
``smurfgaps_flags``    ``SmurfGapsFlags``     Expand smurfgaps (bad smurf data) flag of each stream_id to all detectors.
``load_premade_flags`` ``LoadPremadeFlags``   Load premade flags from aman.
``tod_stats``          ``GetStats``           Get basic statistics from a TOD or its power spectrum to use for flags and cuts.
====================== ====================== ==========================================================================================================================

.. autoclass:: sotodlib.preprocess.processes.DetBiasFlags
.. autoclass:: sotodlib.preprocess.processes.Trends
.. autoclass:: sotodlib.preprocess.processes.PTPFlags
.. autoclass:: sotodlib.preprocess.processes.InvVarFlags
.. autoclass:: sotodlib.preprocess.processes.CutBadDistribution
.. autoclass:: sotodlib.preprocess.processes.DetcalNanCuts
.. autoclass:: sotodlib.preprocess.processes.FocalplaneNanFlags
.. autoclass:: sotodlib.preprocess.processes.DarkDets
.. autoclass:: sotodlib.preprocess.processes.AcuDropFlags
.. autoclass:: sotodlib.preprocess.processes.SmurfGapsFlags
.. autoclass:: sotodlib.preprocess.processes.LoadPremadeFlags
.. autoclass:: sotodlib.preprocess.processes.GetStats

Glitches & Jumps
----------------

============== =================== ===============================================================
``name:``      Class               What it does
============== =================== ===============================================================
``glitches``   ``GlitchDetection`` Run glitch detection algorithm to find glitches.
``glitchfill`` ``GlitchFill``      Fill glitches.
``jumps``      ``Jumps``           Run generic jump finding and fixing algorithm.
``fix_jumps``  ``FixJumps``        Repairs the jump heights given a set of jump flags and heights.
============== =================== ===============================================================

.. autoclass:: sotodlib.preprocess.processes.GlitchDetection
.. autoclass:: sotodlib.preprocess.processes.GlitchFill
.. autoclass:: sotodlib.preprocess.processes.Jumps
.. autoclass:: sotodlib.preprocess.processes.FixJumps

Noise & PSD
-----------

=============== ============== ========================================================================================================================
``name:``       Class          What it does
=============== ============== ========================================================================================================================
``psd``         ``PSDCalc``    Calculate the PSD of the data and add it to the AxisManager.
``noise_ratio`` ``NoiseRatio`` Compute ratios of "signal band" to white noise in PSDs (simple fit-independent way to check for excessive 1/f channels).
``noise``       ``Noise``      Estimate the white noise levels in the data.
=============== ============== ========================================================================================================================

.. autoclass:: sotodlib.preprocess.processes.PSDCalc
.. autoclass:: sotodlib.preprocess.processes.NoiseRatio
.. autoclass:: sotodlib.preprocess.processes.Noise

Calibration
-----------

====================== ==================== ================================================================================================================
``name:``              Class                What it does
====================== ==================== ================================================================================================================
``calibrate``          ``Calibrate``        Calibrate the timestreams based on some provided information (Abscal, relcal, bias step)--just a multiplication.
``pca_relcal``         ``PCARelCal``        Estimate the relcal factor from the atmosphere using PCA.
``correct_iir_params`` ``CorrectIIRParams`` Correct missing iir_params (readout downsampling filter) by default values.
====================== ==================== ================================================================================================================

.. autoclass:: sotodlib.preprocess.processes.Calibrate
.. autoclass:: sotodlib.preprocess.processes.PCARelCal
.. autoclass:: sotodlib.preprocess.processes.CorrectIIRParams

HWP & Demodulation
------------------

=================== ================= ====================================================================================
``name:``           Class             What it does
=================== ================= ====================================================================================
``hwp_angle_model`` ``HWPAngleModel`` Apply hwp angle model (from metadata) to the TOD.
``estimate_hwpss``  ``EstimateHWPSS`` Builds a HWPSS (HWP-synchronous-signal) template.
``subtract_hwpss``  ``SubtractHWPSS`` Subtracts a HWPSS template from signal.
``demodulate``      ``Demodulate``    Demodulate the TOD.
``a2_stats``        ``A2Stats``       Calculate statistical metrics for A2, the 2f-demodulated Q and U signals.
``get_tau_hwp``     ``GetTauHWP``     Analyze observation with hwp spinning up or spinning down to estimate time constant.
=================== ================= ====================================================================================

.. autoclass:: sotodlib.preprocess.processes.HWPAngleModel
.. autoclass:: sotodlib.preprocess.processes.EstimateHWPSS
.. autoclass:: sotodlib.preprocess.processes.SubtractHWPSS
.. autoclass:: sotodlib.preprocess.processes.Demodulate
.. autoclass:: sotodlib.preprocess.processes.A2Stats
.. autoclass:: sotodlib.preprocess.processes.GetTauHWP

Ground Pickup (AzSS)
--------------------

========================== ======================== ====================================================================================================
``name:``                  Class                    What it does
========================== ======================== ====================================================================================================
``azss``                   ``AzSS``                 Estimates Azimuth Synchronous Signal (AzSS) by binning signal by azimuth of boresight  and subtract.
``subtract_azss_template`` ``SubtractAzSSTemplate`` Subtract Azimuth Synchronous Signal (AzSS) common template.
========================== ======================== ====================================================================================================

.. autoclass:: sotodlib.preprocess.processes.AzSS
.. autoclass:: sotodlib.preprocess.processes.SubtractAzSSTemplate

Filtering
---------

=================== ================= =======================================================================
``name:``           Class             What it does
=================== ================= =======================================================================
``fourier_filter``  ``FourierFilter`` Applies a chain of Fourier filters (defined in fft_ops) to the data.
``sub_polyf``       ``SubPolyf``      Fit TOD in each subscan with polynomial of given order and subtract it.
``apodize``         ``Apodize``       Apodize the edges of a signal.
``scan_freq_cut``   ``ScanFreqCut``   Apply high-pass cut at the scan frequency.
``pca_filter``      ``PCAFilter``     Applies a pca filter to the data.
``get_common_mode`` ``GetCommonMode`` Calculate common mode (average over detectors not PCA filtered).
=================== ================= =======================================================================

.. autoclass:: sotodlib.preprocess.processes.FourierFilter
.. autoclass:: sotodlib.preprocess.processes.SubPolyf
.. autoclass:: sotodlib.preprocess.processes.Apodize
.. autoclass:: sotodlib.preprocess.processes.ScanFreqCut
.. autoclass:: sotodlib.preprocess.processes.PCAFilter
.. autoclass:: sotodlib.preprocess.processes.GetCommonMode

Pointing & Focal-Plane Geometry
-------------------------------

=========================== ======================== ==================================================================
``name:``                   Class                    What it does
=========================== ======================== ==================================================================
``pointing_model``          ``PointingModel``        Apply pointing model to the TOD.
``rotate_focal_plane``      ``RotateFocalPlane``     Interpret the boresight rotation effect as a focal plane rotation.
``rotate_qu``               ``RotateQU``             Rotate Q and U components to/from telescope coordinates.
``subtract_qu_common_mode`` ``SubtractQUCommonMode`` Subtract Q and U common mode.
=========================== ======================== ==================================================================

.. autoclass:: sotodlib.preprocess.processes.PointingModel
.. autoclass:: sotodlib.preprocess.processes.RotateFocalPlane
.. autoclass:: sotodlib.preprocess.processes.RotateQU
.. autoclass:: sotodlib.preprocess.processes.SubtractQUCommonMode

Sources & Planets
-----------------

====================== ==================== ===========================================================================
``name:``              Class                What it does
====================== ==================== ===========================================================================
``source_flags``       ``SourceFlags``      Calculate the source flags in the data.
``filter_for_sources`` ``FilterForSources`` Mask and gap-fill the signal at samples flagged by source_flags.
``sso_footprint``      ``SSOFootprint``     Find nearby sources within a given distance and get SSO footprint and plot.
====================== ==================== ===========================================================================

.. autoclass:: sotodlib.preprocess.processes.SourceFlags
.. autoclass:: sotodlib.preprocess.processes.FilterForSources
.. autoclass:: sotodlib.preprocess.processes.SSOFootprint

T-to-P Leakage
--------------

================ =============== =====================================
``name:``        Class           What it does
================ =============== =====================================
``estimate_t2p`` ``EstimateT2P`` Estimate T to P leakage coefficients.
``subtract_t2p`` ``SubtractT2P`` Subtract T to P leakage.
================ =============== =====================================

.. autoclass:: sotodlib.preprocess.processes.EstimateT2P
.. autoclass:: sotodlib.preprocess.processes.SubtractT2P

Scan / Turnaround Flags
-----------------------

======================= =================== ================================================================================================
``name:``               Class               What it does
======================= =================== ================================================================================================
``flag_turnarounds``    ``FlagTurnarounds`` From the Azimuth encoder data, flag turnarounds, left-going, and right-going.
``noisy_subscan_flags`` ``BadSubscanFlags`` Identifies and flags bad subscans (statistics of the subscan non-gaussian, e.g., high kurtosis).
======================= =================== ================================================================================================

.. autoclass:: sotodlib.preprocess.processes.FlagTurnarounds
.. autoclass:: sotodlib.preprocess.processes.BadSubscanFlags

Flag Combination (Mapmaking / Splits)
-------------------------------------

================= ================ =====================================================================================
``name:``         Class            What it does
================= ================ =====================================================================================
``union_flags``   ``UnionFlags``   Deprecated -- use ``combine_flags`` instead. Kept only to load old process archives.
``combine_flags`` ``CombineFlags`` Do the combination of relevant flags for mapping (generalizes ``union_flags``).
``split_flags``   ``SplitFlags``   Get flags used for map splitting/bundling.
================= ================ =====================================================================================

.. autoclass:: sotodlib.preprocess.processes.UnionFlags
.. autoclass:: sotodlib.preprocess.processes.CombineFlags
.. autoclass:: sotodlib.preprocess.processes.SplitFlags

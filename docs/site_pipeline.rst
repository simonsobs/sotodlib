.. py:module:: sotodlib.site_pipeline

=============
site_pipeline
=============

The site_pipeline submodule contains programs supporting
quick-turnaround data processing at the observatory.

.. note::

  Documentation of interfaces is currently held separately.  The focus
  here will be on operation; command line parameters and config file
  syntax.


Command line interface
======================


Usage
-----

To execute a pipeline element from the command line, use the
``so-site-pipeline`` command.  For example, ``make-source-flags`` can
be invoked as::

  so-site-pipeline make-source-flags [options]

To configure tab-completion of element names, in bash, run::

  eval `so-site-pipeline --bash-completion`


Wrapping a pipeline script
--------------------------

.. automodule:: sotodlib.site_pipeline.cli


Pipeline Elements
=================

update-g3tsmurf-db
-------------------

This script set up to create and maintain g3tsmurf databases. :ref:`See details
here<g3tsmurf-update-section>`.

update-book-plan
----------------

This script is designed to help with the bookbinding. It will search a given
Level 2 G3tSmurf database for observations that overlap in time. The different
optional arguments will let us pass information from something like the sorunlib
database to further filter the observations. 

check-book
----------

For a description and documentation of the config file format, see
:mod:`sotodlib.site_pipeline.check_book` module autodocumentation below.

Command line arguments
``````````````````````

.. argparse::
   :module: sotodlib.site_pipeline.check_book
   :func: get_parser
   :prog: check-book

Module documentation
````````````````````

.. automodule:: sotodlib.site_pipeline.check_book
   :members:
   :undoc-members:

update-obsdb
------------

For a description and documentation of the config file format, see
:mod:`sotodlib.site_pipeline.update_obsdb` module autodocumentation below.

Command line arguments
``````````````````````

.. argparse::
   :module: sotodlib.site_pipeline.update_obsdb
   :func: get_parser
   :prog: update-obsdb

Module documentation
````````````````````

.. automodule:: sotodlib.site_pipeline.update_obsdb
   :members:
   :undoc-members:

update-smurf-caldbs
-----------------------
This update script is used to add detset and calibration metadata to manifest
dbs

Module Docs
`````````````````````````
.. automodule:: sotodlib.site_pipeline.update_smurf_caldbs
   :no-members:
  
The calibration info described below is used to populate the calibration db.
For more information on how calibration info is computed in sodetlib, checkout
the following docs and source code:

- `Bias step docs <https://sodetlib.readthedocs.io/en/latest/operations/bias_steps.html>`_
- `IV docs <https://sodetlib.readthedocs.io/en/latest/operations/iv.html>`_
- `sodetlib source code <https://github.com/simonsobs/sodetlib>`_

.. autoclass:: sotodlib.site_pipeline.update_smurf_caldbs.CalInfo
   :no-members:

Command line arguments
`````````````````````````
.. argparse::
   :module: sotodlib.site_pipeline.update_smurf_caldbs
   :func: get_parser
   :prog: update_smurf_caldbs.py

Detector and Readout ID Mapping
-------------------------------

These processes are interrelated and use a combination of the DetMap software
package and sotodlib. The two scripts below are designed to use the same config
files for simplicity and can be run with the level 2 G3tSmurf setup. The
resulting ManifestDbs should work for both level 2 and level 3 SMuRf data. 

make_det_info_wafer
```````````````````
This script uses based array construction inputs to build detector IDs for a set of
UFMs and save them in a ManifestDb / HDF5 file. The formatting of the ResultSet 
saved in HDF5 file will map all this information into ``det_info.wafer`` when used 
with a correctly formatted context file and a readout to detector id mapping.
The detector info mapping created by this script will be stable as long as the
same UFMs are used in the same optics tube positions, meaning it only needs to
be re-made if the physical hardware setup changes. 

Although the full config presented for ``make_read_det_match`` will
work, here's a more basic example that will work::

  det_db: "./det_info_wafer.db"
  det_info: "./det_info_wafer.h5"
  array_info_dir: "/home/so/git/site-pipeline-configs/shared/detmapping/design/"

  arrays:
    - name: mv7
      stream_id: ufm_mv7
    - name: mv9
      stream_id: ufm_mv9


make_read_det_match
```````````````````
This script generates the readout ID to detector ID mapping required to
translate between the detector hardware information (ex: pixel position) and the
readout IDs of the resonators used to index the SMuRF data. The script uses the 
G3tSmurf database to generate a list of TuneSets (tune files) for a set of
arrays / stream ids and runs the DetMap mapping software to generate a mapping
between detectors and resonators. The saved metadata is formatted so with the
correctly formatted Context file the detector ids can be automatically loaded in
the ``det_info`` AxisManager.

Config file format
``````````````````
Here's an example configuration file. Many of these values depend on hardware
setup and readout software setup. Making the detector ID info only requires a
subset of these parameters but the processes are linked so it is probably worth
always having the same configuration file. Tested mapping strategies include
``assignment`` and ``map_by_freq``.

.. code-block:: yaml

    data_prefix : "/path/to/level2/data/"
    g3tsmurf_db: "/path/to/g3tsmurf.db"
    read_db: "/path/to/readout_2_detector_manifest.db"
    read_info: "/path/to/readout_2_detector_hdf5.h5"
    det_db : "/path/to/det_info/wafer/det_info_manifest.db"
    det_info : "/path/to/det_info/wafer/det_info_hdf5.h5"

    arrays:
      # name must match DetMap array names
      - name: "Cv4"
        stream_id: "ufm_cv4"
        # Based on hardware config
        north_is_highband: False
        dark_bias_lines: []
        # how we want to call DetMap
        mapping :
          version : 0
          strategy: "assignment"
          # parameters for mapping strategy
          params: {
            "output_parent_dir":"/writable/path/",
            "do_csv_output": False,
            "verbose": False,
            "save_layout_plot": False,
            "show_layout_plot": False,
            }

Context file format
```````````````````
To load these metadata with context, these entries must be part of the context
file. Since the detector hardware information loads off of the ``det_id`` field,
which is loaded from the readout to detector mapping, the order of the metadata 
entries mater.

.. code-block:: yaml
    
    imports:
      - sotodlib.io.load_smurf
      - sotodlib.io.metadata

    obs_loader_type: 'g3tsmurf'

    metadata:
        - db: "/path/to/readout_2_detector_manifest.db"
          det_info: true
        - db: "/path/to/det_info/wafer/det_info_manifest.db"
          det_info: true
          multi: true

update_det_match
------------------

The ``update_det_match`` script will run the ``det_match`` module on any new
detsets with available calibration metadata. It loads smurf and resonator
information from the AxisManager metadata, and matches resonators against a
solution file in the site-pipeline-configs.

To run, this script requires a config file described below. If run without the
``--all`` flag, it will only run one detset at a time.  If run with the
``--all`` flag, will continue running until all detsets have been mantched.

.. argparse::
   :module: sotodlib.site_pipeline.update_det_match
   :func: make_parser

Generated results
```````````````````
This generates the following data in the specified ``results`` directory:

 - A match file, with the path ``<results_path>/matches/<detset>.h5`` is written
   for every detset.
 - The file ``<results_path>/assignment.sqlite`` is a manifestdb, that contains
   the mapping from readout-id to detector-id. This is compatible with
   the ``det_info_wafer`` and ``focal_plane`` metadata.
 - The ``<results_path>/det_match.sqlite`` file, that contains the
   ``det_match.Resonator`` data from the match for each resonator.

Configuration
`````````````````
This script takes in a config yaml file, which corresponds directly to the
``UpdateDetMatchesConfig`` class (see docs below).

For example, this can run simply with the config file:

.. code-block:: yaml

  results_path: /path/to/results
  context_path: /path/to/context.yaml

Note that by default, this will run a scan of frequency offsets between the
solution and the resonator metadata to find the freq-offset with the best
match. To disable this, you can run a config file like the following:

.. code-block:: yaml

  results_path: /path/to/results
  context_path: /path/to/context.yaml
  freq_offset_range_args: None

Below is a more complex config used for SATp1 matching:

.. code-block:: yaml

  results_path: /so/metadata/satp1/manifests/det_match/satp1_det_match_240220m
  context_path: /so/metadata/satp1/contexts/smurf_detcal.yaml
  show_pb: False
  freq_offset_range_args: NULL
  apply_solution_pointing: False
  solution_type: resonator_set
  resonator_set_dir: /so/metadata/satp1/ancillary/detmatch_solutions/satp1_detmatch_solutions_240219r1
  match_pars:
    freq_width: 0.2

Below is the full docs of the configuration class.

.. autoclass:: sotodlib.site_pipeline.update_det_match.UpdateDetMatchesConfig

update-hkdb
----------------

The update_hkdb site-pipeline script is used to scan through housekeeping files,
and update the index database. Configuration for this script must be passed in
through a config file, with fields that map to the HkConfig dataclass, described
below:

.. autoclass:: sotodlib.io.hkdb.HkConfig
  :no-index:

analyze-bright-ptsrc
--------------------

This script analyzes an observation of a bright point source (such as
a planet) and performs per-detector fitting of beam parameters
including amplitude and FWHM.

.. argparse::
   :module: sotodlib.site_pipeline.analyze_bright_ptsrc
   :func: get_parser

finalize-focal-plane
--------------------

This element produces a finalized focal plane for a given array.
It consumes the output of pointing fits (ie from ``analyze-bright-ptsrc``)
with a detector map to combine results across multiple tuning epochs.
It works by averaging the provided ``analyze-bright-ptsrc`` results using weights,
determined by how well each fit matches the nominal template, to produce a final focal plane.
An affine transformation that lines up the template focal plane computed with physical optics
is then computed to create a "noise-free" focal plane.

This element also computes the receiver and optics tube "common mode" transformation.
The optics tube common mode is how all of the arrays in one optics tube move together,
and the receiver common mode is how all of the optics tubes move together.
In the case of the SATs where there is only one tube, the optics tube common mode
is always taken to be the identity.
Given the smaller number of data points, these common modes are simple rigid transforms
(shift and rotation) rather than a full affine transform.

``finalize_focal_plane`` can optionally be run in a "per obs" mode where no averaging is done,
in this case the output database is indexed by ``obs_id``.


.. automodule:: sotodlib.site_pipeline.finalize_focal_plane
   :members:
   :undoc-members:


Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml

  # There are two options to get the data in
  # One is to pass in ResultSets like so:
  resultsets:
    obs_1: # obs_id associated with this data
      # There are 3 possible ResultSets you can pass
      # pointing is mandatory
      pointing:
        - "PATH/TO/FITS.h5" # The path to the ResultSet
        - "focalplane" # The name of the ResultSet in the h5 file
      # polarization and detmap are optional
      polarization :
        - "PATH/TO/FITS.h5"
        - "polarization"
      detmap:
        - "PATH/TO/DETMAP.h5"
        - "merged"
    obs_2: ...
  # When using results sets you also need to pass in additional metadata like
  stream_id: "ufm_mv29"
  wafer_slot: "ws0"
  telescope_flavor: "SAT"
  tube_slot: "st1"
  # Note that in the ResultSets case only single wafer fits are supported

  # You can also load the data in with context like so
  context:
    path: PATH/TO/CONTEXT
    # There are two pointing fields in case we have both a tod and map fit for one obs_id
    # This may change down the line
    map_pointing: "map_pointing" # The name of the map based pointing metadata field
    tod_pointing: "tod_pointing" # The name of the TOD based pointing metadata field
    polarization: "polarization" # The name of the polarization metadata field (optional)
    # There are two ways to specitfy the observation, obs_id and query
    # Both can be provided 
    obs_id: [obs_1, obs_2] # Pass in the obs_id directly
    query: QUERY # Pass in a query
    # You can pass in detector restrictions here as well
    dets: {} # Should be a dict you would pass to the dets areg of ctx.get_meta
  
  per_obs: False # Set to true if you want to run in per obs mode
  weight_factor: 1000 # Weights are computed with sigma=template_spacing/weight_factor.
                      # This is an advanced feature and should be used with caution.

  # There are a few ways to pass in a template as well
  template: "PATH/TO/TEMPLATE.h5" # As a h5 file with a ResultSet named the same as the UFM
  gen_template: False # Or by setting this true to generate the template on the fly

  # You also will need to provide some information for using the optics code
  pipeline_config_dir : "PATH/TO/PIPELINE/CONFIGS" # If not provided the sysvar $PIPELINE_CONFIG_DIR is used
  zemax_path: "PATH/TO/ID9_checked_trace_data.npz" # Only needed for the LAT
  
  # Plotting info
  plot: True # Set to output plot
  plot_dir: "./plots" # Where to save plots
  
  # Output info
  outdir: "."
  append: "test" # Will have a "_" before it.
  

Output file format
``````````````````

The results of ``finalize_focal_plane`` are stored in an HDF5 file containing
multiple datasets. The datasets are made using the ``ResultSet`` class and can be
loaded back as such but metadata stored as attributes require ``h5py``.

The datasets and attributes are organized by tube and array as seem below:

.. code-block:: text

   focal_plane.h5
   - (attr) center # The nominal center of the receive on sky
   - (attr) center_transformed # The center with the common mode transform applied
   - (group) transform # The receiver common mode
   - (group) tube1 # The first tube (ie st1, oti1, etc.)
     - (attr) center # The nominal center of the tube on sky
     - (attr) center_transformed # The center with the common mode transform applied
     - (group) transform # The tube common mode
     - (group) ufm_1 # The first ufm for thi tube (ie ufm_mv29) 
       - (attr) template_centers # The nominal center for this array
       - (attr) fit_centers # The fit center for this array
       - (group) transform # The transform for the ufm, includes parameters with and without the common mode
       - (dataset) focal_plane # The focal_plane with just fit positions
         - (attr) measured_gamma # If gamma was actually measured
       - (dataset) focal_plane_full # Also includes avg positions, weights, and counts
     - (group) ufm_2
       ...
     ...


The ``focal_plane`` dataset contains four columns:

- ``dets:det_id``: The detector id
- ``xi``: The transformed template xi in radians
- ``eta``: The transformed template eta in radians
- ``gamma``: The transformed template gamma in radians.

If no polarization angles are provided them ``gamma`` will be populated
with the nominal values from physical optics.
There is an attribute called ``measured_gamma`` that will be ``False`` in this case.

The ``focal_plane_full`` dataset contains nine columns: 

- ``dets:det_id``: The detector id
- ``xi_t``: The transformed template xi in radians
- ``eta_t``: The transformed template eta in radians
- ``gamma_t``: The transformed template gamma in radians.
- ``xi_m``: The measured xi in radians
- ``eta_m``: The measured eta in radians
- ``gamma_m``: The measured gamma in radians.
- ``weights``: The average weights of the measurements for this det.
- ``n_point``: The number of pointing fits used for the det.
- ``n_gamma``: The number of gamma fits used for this det.

All the attributes having to do with the centers of things are ``(1,3)`` arrays
in the form ``((xi), (eta), (gamma))`` in radians.

This transformation for ``xi`` and ``eta`` is an affine transformation defined as
:math:`m = An + t`, where:

- ``m`` is the measured ``xi-eta`` pointing
- ``n`` is the nominal ``xi-eta`` pointing
- ``A`` is the 2x2 affine matrix
- ``t`` is the final translation

``A`` is then decomposed into a rotation of the ``xi-eta`` plane, a shear parameter,
and a scale along each axis.
This decomposition is done assuming the order as ``A = rotation*shear*scale``.

For gamma the transformation is also technically affine, but since it is in just
one dimension it can be described by a single shift and scale.

All of these results are stored as attributes in the ``transform`` groups.
These nominally are:

- ``affine``: The full affine matrix
- ``shift``: The shift in ``(xi, eta, gamma)`` in radians
- ``scale``: The scale along ``(xi, eta, gamma)`` in radians
- ``rot``: The rotation of the ``xi-eta`` plane
- ``shear``: The shear of the ``xi-eta`` plane

The ``transform`` group for the arrays also include these attributes with
whe common mode removed, the names have ``_nocm`` appended (ie ``rot_nocm``).

Since the common mode transformations are fit as affine transforms ``scale`` will
always be ``(1, 1, 1)`` and ``shear`` will be ``0``.


``finalize_focal_plane`` will also output a ``ManifestDb`` as a file called ``db.sqlite``
in the output directory.
By default this will be indexed by ``stream_id`` and will point to the ``focal_plane`` dataset.
If you are running in ``per_obs`` mode then it will also be indexed by ``obs_id`` and will point
to results associated with data observation.
Be warned that in this case there will only be entries for observations with pointing fits,
so design your context accordingly.


preprocess-tod
--------------
This script is set up to run a preprocessing pipeline using the preprocess
module. See details in :ref:`See details here<preprocess-module>` for how to
build a preprocessing pipeline. 

This module includes the functions designed to be run as part of a batch script
for automated analysis as well as options for loading AxisManagers that have all
the preprocessing steps applied to them.

.. argparse::
   :module: sotodlib.site_pipeline.preprocess_tod
   :func: get_parser

preprocess-obs
--------------
This script is set up to run a preprocessing pipeline using the preprocess
module. See details in :ref:`See details here<preprocess-module>` for how to
build an obs preprocessing pipeline.

This module is similar to ``preprocess_tod`` but removes grouping by detset so
that the entire observation is loaded, without signal.

.. argparse::
   :module: sotodlib.site_pipeline.preprocess_obs
   :func: get_parser


make-source-flags
-----------------

Command line arguments
``````````````````````

.. argparse::
   :module: sotodlib.site_pipeline.make_source_flags
   :func: get_parser
   :prog: make-source-flags


Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml
  
  # Context for <whatever>
  context_file: ./context4_b.yaml

  # How to subdivide observations (by detset, but call it "wafer_slot")
  subobs:
    use: detset
    label: wafer_slot

  # Metadata index & archive filenaming
  archive:
    index: 'archive.sqlite'
    policy:
      type: 'simple'
      filename: 'archive.h5'

  # Mask parameters
  mask_params:
    mask_res: [2, 'arcmin']
    default: {'xyr': [0., 0., 0.1]}


make-uncal-beam-map
-------------------

.. automodule:: sotodlib.site_pipeline.make_uncal_beam_map
   :members:
   :undoc-members:


Command line arguments
``````````````````````

.. argparse::
   :module: sotodlib.site_pipeline.make_uncal_beam_map
   :func: get_parser
   :prog: make-uncal-beam-map

Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml

  # Data source
  context_file: ./act_uranus/context.yaml

  # Sub-observation data grouping
  subobs:
    use: detset
    label: wafer_slot

  # Database of results
  archive:
    index: 'archive.sqlite'
    policy:
      type: 'directory'
      root_dir: './'
      pattern: 'maps/{product_id}'

  # Output selection and naming
  output:
    map_codes: ['solved', 'weights']
    pattern: '{product_id}_{split}_{map_code}.fits'

  # Plot generation
  plotting:
    zoom:
      f090: [10, arcmin]
      f150: [10, arcmin]

  # Preprocessing
  preprocessing:
    cal_keys: ['abscal', 'relcal']
    pointing_keys: ['boresight_offset']

  # mapmaking parameters
  mapmaking:
    force_source: Uranus
    res:
      f090: [15, arcsec]
      f150: [15, arcsec]


Inputs
``````

The Context should cause the TOD to be loaded with all supporting
metadata loaded into the AxisManager.  Here are key members that will
be processed:

- Deconvolution step:

  - ``'timeconst'``
  - ``'iir_params'``

- Calibration:

  - Whatever is listed in preprocessing.cal_keys

- Pointing correction:

  - ``'boresight_offset'``

- Demodulation and downsampling:

  - not implemented

- Planet mapmaking:

  - ``'source_flags'``
  - ``'glitch_flags'`` - optional


update-hwp-angle
----------------

Script for running updates on (or creating) a hwp angle g3 file.
This script will run periodically even when hwp is not spinning.
Meaning is designed to work from something like a cronjob.
The output hwp angle should be synchronized to SMuRF timing outside this script. 
:ref:`See details here<g3thwp-section>`.

Command line arguments
``````````````````````
.. argparse::
   :module: sotodlib.site_pipeline.update_hwp_angle
   :func: get_parser
   :prog: update_hwp_angle


make-hwp-solutions
------------------

This element generates HWP angle-related metadata, 
which contains the calibrated HWP angle and flags.
The HWP angle is synchronized with the input SMuRF timestamp.
:ref:`See details here<g3thwp-section>`.

Command line arguments
``````````````````````
.. argparse::
    :module: sotodlib.site_pipeline.make_hwp_solutions
    :func: get_parser
    :prog: make_hwp_solutions


make-cosamp-hk
------------------

This element generates house-keeping data with timestamps co-sampled
with detector timestamps. 

Command line arguments
``````````````````````
.. argparse::
    :module: sotodlib.site_pipeline.make_cosamp_hk
    :func: get_parser
    :prog: make-cosamp-hk
    
An example config for wiregrid is shown below.
With the config file, ``wiregrid.sqlite`` and ``wiregrid_XXXX.h5``, 
where XXXX is substituted with the first four digits of timestamps, 
are generated on ``/path/to/manifests/wiregrid``. ``context_file``, 
``input_dir``, ``output_dir``, ``fields``, ``aliases``, and ``output_prefix``
are required::

    context_file: '/path/to/context.yaml'
    query_text: 'type == "obs"'
    min_ctime: 1700000000
    max_ctime: null
    query_tags: ['wiregrid=1']
    input_dir: '/path/to/level2/hk'
    output_dir: '/path/to/manifests/wiregrid'
    fields: 
            ['satpX.wg-encoder.feeds.wgencoder_full.reference_count',
             'satpX.wg-actuator.feeds.wgactuator.limitswitch_LSL1',
             'satpX.wg-actuator.feeds.wgactuator.limitswitch_LSL2',
             'satpX.wg-actuator.feeds.wgactuator.limitswitch_LSR1',
             'satpX.wg-actuator.feeds.wgactuator.limitswitch_LSR2',]
    aliases: 
        ['encoder',
         'LS1',
         'LS2',
         'LSR1',
         'LSR2']
    output_prefix: 'wiregrid'

If you specifiy some of ``save_mean``, ``save_median``, ``save_rms``, ``save_ptp`` boolean
in config file, those values are calculated for the first parameter of ``fields`` and 
stored to columns of sqlite with name of like "{output_prefix}_mean". If you specify
``min_valid_value``, ``max_valid_value``, ``max_valid_dvalue_dt`` in config file, values
out of the range are set to ``np.nan`` and ``{output_prefix}_nan_fraction`` is added to
columns of sqlite. A config below is an example for PWV data with its valid range is 
0.3 < pwv < 3.0 mm and its valid time derivative is 0.01 mm/s::

    context_file: '/path/to/context.yaml'
    query_text: null
    min_ctime: null 
    max_ctime: null
    update_delay: 1
    query_tags: null
    input_dir: '/path/to/level2/hk'
    output_dir: '/path/to/manifests/pwv_clas'
    fields: 
        ['site.env-radiometer-class.feeds.pwvs.pwv',]
    aliases: ['pwv_class',]
    output_prefix: 'pwv_class'

    save_mean: True
    save_median: True
    save_rms: True
    save_ptp: True
    min_valid_value: 0.3
    max_valid_value: 3.0
    max_valid_dvalue_dt: 0.01

    
make-ml-map
-----------

This submodule can be used to call the maximum likelihood mapmaker.
The mapmaker will produce ``bin``, ``div`` and ``sky`` maps. The mapmaker
has several different flags (see the example config file below) that can
be passed via the CLI or a ``config.yaml`` file. If an argument is not 
specified, a value is selected from a set of defaults.

The arguments ``freq``,  ``area`` and ``context`` are required; they should
either be supplied through the CLI or the ``config.yaml``.

Command line arguments
``````````````````````
.. argparse::
   :module: sotodlib.site_pipeline.make_ml_map
   :func: get_parser



Default Mapmaker Values
```````````````````````
The following code block contains the hard-coded default values for non-
essential mapmaker arguments. The can be overidden in the CLI or in the
``config.yaml``.
   
.. code-block:: python 

        defaults = {"query": "1",
                    "comps": "T",
                    "ntod": None,
                    "tods": None,
                    "nset": None,
                    "site": 'so_lat',
                    "nmat": "corr",
                    "max_dets": None,
                    "verbose": 0,
                    "quiet": 0,
                    "center_at": None,
                    "window": 0.0,
                    "inject": None,
                    "nocal": True,
                    "nmat_dir": "/nmats",
                    "nmat_mode": "build",
                    "downsample": 1,
                    "maxiter": 500,
                    "tiled": 1,
                    "wafer": None,
                   }



Config file format
``````````````````

Example of a config file:

.. code-block:: yaml
       
         # Query
        query: "1"

        # Context file containing TODs
        context: 'context.yaml'

        # Telescope info
        freq: 'f150'
        site: 'so_lat'

        # Mapping area footprint
        area: 'geometry.fits'

        # Output Directory and file name prefix
        odir: './output/'
        prefix: 'my_maps'

        # Detectors info. null by default
        tods: [::100] # Restrict TOD selections by index
        ntod: 3 # Special case of `tods` above. Implemented as follows: [:ntod]
        nset: 10 # Number of detsets kept
        max-dets: 200 # Maximum dets kept
        wafer: 'w17' # Restrict which wafers are mapped. Can do multiple wafers

        # Mapmaking meta
        comps: 'T' # TQU
        inject: null 
        nocal: True # No relcal or abscal
        downsample: 1 # Downsample TOD by this factor
        tiled: 0 # Tiling boolean (0 or 1)
        nmat-dir: './nmats/' # Dir to save or load nmat
        nmat: 'corr' # 'corr' or 'uncorr' 
        maxiter: 500 # Max number of iterative steps
        nmat_mode: 'build' # 'cache', 'build', 'load' or 'save'
        center_at: null 
        window: 0.0
        inject: null

        # Scripting tools
        verbose: True
        quiet: False


QDS Monitor
===========
The QDS Monitor is meant to be a simple to use class that allows users to
publish the results of their calculations to a live monitor. The live monitor
backend is an Influx Database, which is used with the SO Data Acquisition
system, known as the Observatory Control System. This allows us to use the same
live monitoring interface, Grafana.

Overview
--------
The ``Monitor`` class wraps the InfluxDB interface, and provide a few simple
methods -- ``check``, ``record``, and ``write`` -- detailed in the
:ref:`API section <monitor_api>`.

``check`` is meant to be used to check if the calculation already has been
performed for the given observation/tag set. This can be used to ensure
expensive calculations are not repeated when running batch jobs. ``record``
takes your calculations, timestamps, and a set of identifying tags, and queues
them for batch writing to the InfluxDB. Finally, ``write`` will write your
recorded results to the InfluxDB, clearing the queue.

This perhaps is best demonstrated with some examples, shown in the next section.

Examples
--------
Simple Pseudocode
`````````````````
The general outline we're aiming for is as follows::

    from sotodlib.site_pipeline.monitor import Monitor
    
    # Initialize DB Connection
    monitor = Monitor('localhost', 8086, 'qdsDB')
    
    # Load observation
    tod = so_data_load.load_observation(context,                       
              observation_id, detectors_list)
    
    # Compute statistic
    result = interesting_calculation(tod)
    
    # Tag and write to DB
    tags = {'telescope': 'LAT', 'wafer': wafer_name}
    monitor.record('white_noise_level', result, timestamp, tags)
    monitor.write()

Real World Example
``````````````````

The following is a real world example of the ``Monitor`` in action. We'll walk
through the important parts, omitting some descriptive print statements. The
full script is included below.

To start, we will import the module and create our ``Monitor`` object.
You will need to know the address and port for your InfluxDB, as well as the
name of the database within InfluxDB that you want to write to.::

    from sotodlib.site_pipeline.monitor import Monitor

    monitor = Monitor('localhost', 8086, 'qds')

.. note::
    Secure connection to an external InfluxDB is supported. To connect use to
    https://example.com/influxdb/ use::

        monitor = Monitor(host='example.com',
                          port=443,
                          username=u'username',
                          password=u'ENTER PASSWORD HERE',
                          path='influxdb',
                          ssl=True)

Let's say we want to load some of the sims, we'll create our Context and get
the observations with::

    context = core.Context('pipe_s0001_v2.yaml')
    observations = context.obsfiledb.get_obs()

Then we can, for example, loop over all observations, determining the detectors
and wafers in each observation::

    for obs_id in observations:
        c = context.obsfiledb.conn.execute('select distinct DS.name, DS.det from detsets DS '
                        'join files on DS.name=files.detset '
                        'where obs_id=?', (obs_id,))
        dets_in_obs = [tuple(r) for r in c.fetchall()]
        wafers = np.unique([x[0] for x in dets_in_obs])

We'll run our calculation for each wafer, so let's loop over those now,
building a detector list for the wafer, and loading the TOD for just those
detectors and computing their FFTs::

    for wafer in wafers:
        det_list = build_det_list(dets_in_obs, wafer)
        tod = so_data_load.load_observation(context.obsfiledb, obs_id, dets=det_list)

        # Compute ffts
        ffts, freqs = rfft(tod)
        det_white_noise = calculate_noise(tod, ffts, freqs)

Now we want to save our results to the monitor. To do this, we'll need two
other lists, one for the timestamps associated with each noise value (in this
case, these are all the same, and use the first timestamp in the TOD), and one
for the tags for each noise value (in this example we tag each detector
individually with their detector ID, along with the wafer it is on and what
telescope we're working with -- this probably is in the context somewhere, but
I'm just writing in SAT1)::

        timestamps = np.ones(len(det_white_noise))*tod.timestamps[0]
        base_tags = {'telescope': 'SAT1', 'wafer': wafer}
        tag_list = []
        for det in det_list:
            det_tag = dict(base_tags)
            det_tag['detector'] = det
            tag_list.append(det_tag)
        log_tags = {'observation': obs_id, 'wafer': wafer}
        monitor.record('white_noise_level', det_white_noise, timestamps, tag_list, 'detector_stats', log_tags=log_tags)
        monitor.write()

We also include a set of log tags, these are to record that we've completed
this calculation for this observation and wafer. Lastly we record the
measurement, giving it the name "white_noise_level", passing our three lists of
equal length (``det_white_noise``, ``timestamps``, ``tag_list``), and recording
the measurement as completed in the "detector_stats" log with the observation ID
and wafer log tags.

Where these log tags could come in handy is if we need to stop and restart our
calculation and want to skip recomputing the results. Since we saved the wafer
along with the observation ID it would make sense to check at the wafer level
loop::

    for wafer in wafers:
        # Check calculation completed for this wafer
        check_tags = {'wafer': wafer} 
        if monitor.check('white_noise_level', obs_id, check_tags):
            continue

Add this to the top of our wafer loop would skip already recorded wafers for
this observation id.

The example script in its entirety is shown here:

.. code-block:: python

    # Largely based on 20200514_FCT_Software_Example.ipynb from the pwg-fct
    import numpy as np
    
    from sotodlib import core
    import sotodlib.io.load as so_data_load
    
    from sotodlib.tod_ops import rfft
    
    import qds
    
    monitor = qds.Monitor('localhost', 56777, 'qds')
    
    context = core.Context('pipe_s0001_v2.yaml')
    observations = context.obsfiledb.get_obs()
    print('Found {} Observations'.format(len(observations)))
    o_list = range(len(observations)) # all observations
    
    for o in o_list:
        obs_id = observations[o]
        print('Looking at observation #{} named {}'.format(o,obs_id))
        
        c = context.obsfiledb.conn.execute('select distinct DS.name, DS.det from detsets DS '
                                'join files on DS.name=files.detset '
                                'where obs_id=?', (obs_id,))
        dets_in_obs = [tuple(r) for r in c.fetchall()]
        wafers = np.unique([x[0] for x in dets_in_obs])
        
        print('There are {} detectors on {} wafers in this observation'.format(len(dets_in_obs), len(wafers)))
        
        for wafer in wafers:
            # Check calculation completed for this wafer
            check_tags = {'wafer': wafer}
            if monitor.check('white_noise_level', obs_id, check_tags):
                continue
    
            # Process Obs+Wafer
            # Build detector list for this wafer
            det_list = []
            for det in dets_in_obs:
                if det[0] == wafer:
                    det_list.append(det[1])
            print('{} detectors on this wafer'.format(len(det_list)))
    
            tod = so_data_load.load_observation(context.obsfiledb, obs_id, dets=det_list )
    
            print('This observation is {} minutes long. Has {} detectors and {} samples'.format(round((tod.timestamps[-1]-tod.timestamps[0])/60.,2),
                                                                                  tod.dets.count, tod.samps.count))
    
            print('This TOD AxisManager has Axes: ')
            for k in tod._axes:
                print('\t{} with {} entries'.format(tod[k].name, tod[k].count ) )
    
            print('This TOD  AxisManager has fields : [axes]')
            for k in tod._fields:
                print('\t{} : {}'.format(k, tod._assignments[k]) )
                if type(tod._fields[k]) is core.AxisManager:
                    for kk in tod[k]._fields:
                        print('\t\t {} : {}'.format(kk, tod[k]._assignments[kk] ))
    
            # Compute the FFT and detector white noise levels
            ffts, freqs = rfft(tod)
    
            tsamp = np.median(np.diff(tod.timestamps))
            norm_fact = (1.0/tsamp)*np.sum(np.abs(np.hanning(tod.samps.count))**2)
    
            fmsk = freqs > 10
            det_white_noise = 1e6*np.median(np.sqrt(np.abs(ffts[:,fmsk])**2/norm_fact), axis=1)
    
            # Publish to monitor
            timestamps = np.ones(len(det_white_noise))*tod.timestamps[0]
            base_tags = {'telescope': 'LAT', 'wafer': wafer}
            tag_list = []
            for det in det_list:
                det_tag = dict(base_tags)
                det_tag['detector'] = det
                tag_list.append(det_tag)
            log_tags = {'observation': obs_id, 'wafer': wafer}
            monitor.record('white_noise_level', det_white_noise, timestamps, tag_list, 'detector_stats', log_tags=log_tags)
            monitor.write()
    
.. _monitor_api:

API
---
.. autoclass:: sotodlib.site_pipeline.monitor.Monitor
    :members:


Support
=======

.. automodule:: sotodlib.site_pipeline.util
   :members:
   :undoc-members:

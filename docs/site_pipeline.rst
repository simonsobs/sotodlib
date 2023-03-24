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


Pipeline Elements
=================

update-g3tsmurf-db
-------------------

This script set up to create and maintain g3tsmurf databases. :ref:`See details
here<g3tsmurf-update-section>`.

imprinter
----------

This script is designed to help with the bookbinding. It will search a given
Level 2 G3tSmurf database for observations that overlap in time. The different
optional arguments will let us pass information from something like the sorunlib
database to further filter the observations. 

Currently outputs a list of tuples where each tuple is one or more observation
ids. Each tuple has at least some overlap.

.. argparse::
   :module: sotodlib.site_pipeline.imprinter
   :func: get_parser


Detector and Readout ID Mapping
-------------------------------

These processes are interrelated and use a combination of the DetMap software
package and sotodlib. The two scripts below are designed to use the same config
files for simplicity and can be run with the level 2 G3tSmurf setup. The
resulting ManifestDbs should work for both level 2 and level 3 SMuRf data. 

make_det_info_wafer
```````````````````
This script uses the DetMap software package to build detector IDs for a set of
UFMs and save them in a ManifestDb / HDF5 file. The formatting of the ResultSet 
saved in HDF5 file will map all this information into ``det_info.wafer`` when used 
with a correctly formatted context file and a readout to detector id mapping.
The detector info mapping created by this script will be stable as long as the
same UFMs are used in the same optics tube positions, meaning it only needs to
be re-made if the physical hardware setup changes. 

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
always having the same configuration file.

.. code-block:: yaml

    data_prefix : "/path/to/level2/data/"
    g3tsmurf_db: "/path/to/g3tsmurf.db"
    read_db: "/path/to/readout_2_detector_manifest.db"
    read_info: "/path/to/readout_2_detector_hdf5.h5"
    det_db : "/path/to/det_info/wafer/manifest.db"
    det_info : "/path/to/det_info/wafer/hdf5.h5"

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
          strategy: "map_by_freq"
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
          det_key: "dets:readout_id"
        - db: "/path/to/det_info/wafer/manifest.db"
          det_info: true
          det_key: "dets:det_id"


make-source-flags
-----------------

Command line arguments
``````````````````````

.. argparse::
   :module: sotodlib.site_pipeline.make_source_flags
   :func: get_parser

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
   :func: _get_parser
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

  # Mapmaking parameters
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


make-position-match
-------------------

This element builds a `readout_id` to `detector_id` mapping by matching measured pointing and polarization angles against a template.
It is capable of taking multiple measurements from the same tuning epoch and combining them to produce a single mapping and an average focal plane.

.. automodule:: sotodlib.site_pipeline.make_position_match
   :members:
   :undoc-members:


Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml

  # Data sources
  pointing_data:
    - "pointing_res_1.h5"
    - "pointing_res_2.h5"
  # Note here that the number of pointing results match the number of polang results
  # It is currently assumed that this is true, and that the i'th file for each are associated
  # If the numbers do not match files can be repeated or "" can be given to skip polangs for that pointing
  # Polangs can also be fully omitted in which case do not include the "polangs" key in the file
  polangs:
    - "polang_res_1.h5"
    - "polang_res_2.h5"
  detmap: "detmap_results.csv"
  bias_map: "bias_map_results.npz"
  
  g3tsmurf:
    obs_id: "obs_ufm_uv8_1677778438"
    paths:
        archive_path: "/mnt/so1/data/chicago-latrt/timestreams/"
        meta_path: "/mnt/so1/data/chicago-latrt/smurf/"
        db_path: "/mnt/so1/shared/site-pipeline/data_pkg/chicago-latrt/g3tsmurf.db"
  
  # Configuration options
  ufm: "Uv8"
  gen_template: False # Generate template using InstModel
  radial_thresh: 2.0 # Threshoud at which to dut detectors based on their pointing
  dm_transform: True # Apply an initial transformation based on the detmap

  # Settings to generate priors from detmap
  # Do not include if you dont want priors
  priors:
    val: 1.5
    method: "gaussian"
    width: 1
    basis: "fit_fr_mhz"
  # Value that liklihoods are normalized to when making priors from them.
  prior_normalization: .2

  # Settings for the matching processes
  matching:
    out_thresh: .4 # Liklihood below which things will be considered outliers
    reverse: False # Reverse the match direction
    vis: False # Play an animation of match iterations
    cpd_args: # Args to pass pycpd
      max_iterations: 1000

  outdir: "."
  # Optional string to be appended to output filename.
  # Will have a "_" before it.
  append: "test" 

Ouput file format
`````````````````

The results of `make_position_match` are stored in an HDF5 file containing two datasets.
The datasets are made using the `ResultSet` class and can be loaded back as such.

The first dataset is called `input_data_paths` and has two columns: `type` and `path`.
It contains the list of input files used to produce the output, with `type` being the type of input
and `path` being the actual path.

The current types of paths are:

- config
- tunefile
- bgmap
- pointing
- polang

The second dataset is called `focal_plane` and has columns:

- `dets:det_id`, the detector id as matched by this element.
- `dets:readout_id`, the readout id.
- `band`, the SMuRF band.
- `channel`, the SMuRF channel.
- `avg_xi`, average xi for each detector.
- `avg_eta`, average eta for each detector.
- `avg_polang`, average polarization angle for each detector.
- `meas_x`, the measured x position of each detector on the array.
- `meas_y`, the measured y position of each detector on the array.
- `meas_pol`, the measured polarization angle of each detector on the array.
- `likelihood`, the liklihood of the match for each detector.
- `outliers`, flag that shows which detectors look like outliers.


Support
=======

.. automodule:: sotodlib.site_pipeline.util
   :members:
   :undoc-members:

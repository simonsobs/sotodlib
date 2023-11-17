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
It consumes the output of ``make-position-match`` and is designed to
combine results across multiple tuning epochs.
It works by averaging the provided ``make-position-match`` results without
using points that are flagged as outliers to make the final focal plane.
Then a transformation from a nominal focal plane made with physical optics
to this final focal plane is computed. This transformation is affine so it
can describe shifts, scales, shear, and rotation but not any non-linear mapping.

This focal plane is indexed by ``detector_id`` so it requires a mapping between
``readout_id`` and ``detector_id`` for each provided result.
Nominally this mapping is provided by ``DetMap`` but when the ``use_matched``
setting in the config file is set to ``True`` it will use the mapping
generated by ``make-position-match`` instead.

To get full use of this element all of the measurements that go into it should be
taken at similar ``az-el-bs``. Faillure to do so will cause the element to only
produce offsets in ``xi-eta-gamma`` and not provide a reference ``az-el-gamma``.
This can still be useful for debugging and health check purposes but not for the pointing model.

.. automodule:: sotodlib.site_pipeline.finalize_focal_plane
   :members:
   :undoc-members:


Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml

  # Data sources
  context:
      path: "context.yaml"
      position_match: "position_match"
      obs_ids:
          - "obs_ufm_uv8_1678148785"
      query: "query_string"

  # For multi observation results just specify the path
  multi_obs:
    - "path_to_h5"
  # These should be one per provided result
  # Observations loaded from context can have this information included already
  # In that case just set the detmap for that observation to null
  # Order is the ones from obs_id first, then query, then multi_obs
  detmaps:
      - null
      - "path_to_detmap"
  
  # Configuration options
  use_matched: False # Set to True to use the mapping from make_position_match
  # Show a plot of the nominal, measured and transformed coords
  # If set to True the plot will be shown
  # If set to a path the plot will be saved there
  plot: True
  coord_transform:
      telescope: "LAT"
      tube: "c1"
      slot: 0
      rot: 0
      config_path: "/home/saianeesh/code/site-pipeline-configs/ufm_to_fp.yaml"
      zemax_path: "/home/saianeesh/data/ID9_checked_trace_data.npz"
  
  # Output info
  ufm: "Uv8"
  outdir: "."
  append: "test" # Will have a "_" before it.
  

Output file format
``````````````````

The results of ``finalize_focal_plane`` are stored in an HDF5 file containing
three datasets. The datasets are made using the ``ResultSet`` class and can be
loaded back as such but metadata stored as attributes require ``h5py``.

The first dataset is called ``focal_plane`` and contains three columns:

- ``dets:det_id``: The detector id
- ``xi``: xi in radians
- ``eta``: eta in radians
- ``gamma``: gamma in radians

If a given detector has no good pointing information provided then the three
pointing columns will be ``nan`` for it. If no polarization angles are provided
them ``gamma`` will be populated with the nominal values from physical optics.
There is an attribute called ``measured_gamma`` that will be ``False`` in this case.

The second dataset is called ``offsets`` and contains the information to transform from
the nominal pointing to the measured pointing.

This transformation for ``xi`` and ``eta`` is an affine transformation defined as
:math:`m = An + t`, where:

- ``m`` is the measured ``xi-eta`` pointing
- ``n`` is the nominal ``xi-eta`` pointing
- ``A`` is the 2x2 affine matrix
- ``t`` is the final translation

``A`` is then decomposed into a rotation of the ``xi-eta`` plane, a shear parameter,
and a scale along each axis.

For gamma the transformation is also technically affine, but since it is in just
one dimension it can be described by a single shift and scale.

All of these parameters are stored in a ``ResultSet`` with two rows.
The first row is in the ``xi-eta-gamma`` basis and the second in ``az-el-bs``.
Its columns are:

- ``d_x``: The shift along the measured ``xi``/``az`` axis.
- ``d_y``: The shift along the measured ``eta``/``el`` axis.
- ``d_z``: The shift along the measured ``gamma``/``bs`` axis.
- ``s_x``: The scale along the measured ``xi``/``az`` axis.
- ``s_y``: The scale along the measured ``eta``/``el`` axis.
- ``s_z``: The scale along the measured ``gamma``/``bs`` axis.
- ``shear``: The shear parameter of the ``xi-eta``/``az-el`` plane.
- ``rot``: The rotation of the ``xi-eta``/``az-el`` plane in radians.

This dataset also has the attributes ``affine_xieta`` and ``affine_horiz`` which contains the
affine transformation matrices which are decomposed to produce some of values in the ``ResultSet``.
These matrices are ``A`` in the equation :math: `m = An + t` that is described above.

The third dataset is called ``reference`` and contains useful reference points. 

This dataset is stored as a ``ResultSet`` with two rows,
The first row is the nominal ``xi-eta-gamma`` coordinates of the center of the array mapped onto the sky,
the origin of this coordinate system is the telescope boresite.
The second row is the ``az-el-bs`` of the telescope's nominal boresite for the
measurements that feed into this code.
Its columns are:

- ``x``: The nominal ``xi`` of the center of the array or the ``az`` of the telescope.
- ``y``: The nominal ``eta`` of the center of the array or the ``el`` of the telescope.
- ``z``: The nominal ``gamma`` of the center of the array (computed with a polarization angle of 0)
  or the nominal ``bs`` of the telescope.

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

make-position-match
-------------------

This element builds a ``readout_id`` to ``detector_id`` mapping by matching
measured pointing and polarization angles against a template.
It is capable of taking multiple measurements from the same tuning epoch and
combining them to produce a single mapping and an average focal plane.

.. automodule:: sotodlib.site_pipeline.make_position_match
   :members:
   :undoc-members:


Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml

  # Data sources
  # Both of these are technically optional but are strongly reccomended
  detmap: "detmap_results.csv"
  bias_map: "bias_map_results.npz"
  
  context:
    path: "context.yaml"
    # What the metadata is called in the context file
    pointing: "beammap"
    polarization: null # Set to null to not use
    # List of observation ids to run on
    # You need at least one of these, but can also provide both
    obs_ids:
        - "obs_ufm_uv31_1677980982"
    query: "sqlite query string"
    # Optional det selection dict to be passed to context loader
    dets:
        wafer: "w03"
  
  # Configuration options
  ufm: "Uv31"
  gen_template: False # Generate template using InstModel
  radial_thresh: 2.0 # Threshold at which to cut detectors based on their pointing
  dm_transform: True # Apply an initial transformation based on the detmap

  # Settings to generate priors from detmap
  # Do not include if you dont want priors
  priors:
    val: 1.5
    method: "gaussian"
    width: 1
    basis: "fit_fr_mhz"
  # Value that likelihoods are normalized to when making priors from them.
  prior_normalization: .2

  # Settings for the matching processes
  matching:
    out_thresh: .4 # Liklihood below which things will be considered outliers
    reverse: False # Reverse the match direction
    # To save the animation, set vis to the directory where you want to save it
    # If the directory doesn't exist it will be made
    # If you instead want to watch the animation in real time set to True
    vis: False # Play an animation of match iterations
    cpd_args: # Args to pass pycpd
      max_iterations: 1000

  outdir: "."
  # Optional string to be appended to output filename.
  # Will have a "_" before it.
  append: "test" 
  # ManifestDb to store things in 
  manifest_db: "file.sqlite"

Ouput file format
`````````````````

The results of ``make_position_match`` are stored in an HDF5 file containing
three datasets. The datasets are made using the ``ResultSet`` class and can be
loaded back as such.

The first dataset is called ``input_data_paths`` and has two columns:
``type`` and ``path``. It contains the list of input files used to produce
the output, with ``type`` being the type of input and ``path`` being the
actual path.

The current types of paths are:

- config
- tunefile
- context
- bgmap
- detmap
- obs_id (not actually a path, but useful to track)

The second dataset is called ``focal_plane`` and has columns:

- ``dets:readout_id``, the readout id.
- ``matched_det_id``, the detector id as matched by this element.
- ``band``, the SMuRF band.
- ``channel``, the SMuRF channel.
- ``xi``, average xi for each detector. Nominally in radians.
- ``eta``, average eta for each detector. Nominally in radians.
- ``polang``, average polarization angle for each detector.
  Nominally in radians.
- ``meas_x``, the measured x position of each detector on the array.
  Nominally in mm.
- ``meas_y``, the measured y position of each detector on the array.
  Nominally in mm.
- ``meas_pol``, the measured polarization angle of each detector on the array.
  Nominally in deg.
- ``likelihood``, the likelihood of the match for each detector.
- ``outliers``, boolean flag that shows which detectors look like outliers.

The third dataset is called ``encoders`` and stores the nominal encoder values for
each input observation. It cah columns:

- ``obs_id``, the observation id thathe encoder values are associated with.
- ``az``, the nominal azimuth for this observation.
- ``el``, the nominal elevation for this observation.
- ``bs``, the nominal boresite angle for this observation.

All of the angles are reported in radians.

If the input contained only a single observation the results can be found
using the ManifestDb specified in the config file, the ManifestDb is indexed by `obs_id`.

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

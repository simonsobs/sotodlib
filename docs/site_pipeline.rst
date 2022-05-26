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
    default: {'xyr': [0., 0., 0.1]}


Support
=======

.. automodule:: sotodlib.site_pipeline.util
   :members:
   :undoc-members:

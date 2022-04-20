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

This script is designed to help with the bookbinding. It will search a give
level 2 G3tSmurf database for observations that overlap in time. The different
optional arguments will let us pass information from something like the sorunlib
database to further filter the observations. 

Currently outputs a list of tuples where each tuple is one or more observation
ids. Each tuple has at least some overlap.

.. argparse::
   :module: sotodlib.site_pipeline.imprinter
   :func: get_parser

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

The formal inputs should all be made available through the Context.
They are:

- Obs Book
- HWP Angle
- Chanmap
- Basecal - store in 'cal_base'
- RelCal model - store in 'cal_flat'
- Timeconst model
- Focal plane
- Glitch flags - store in 'glitch_flags'
- Noise cuts - store in 'noise_cuts'
- HWP Spinsync


Support
=======

.. automodule:: sotodlib.site_pipeline.util
   :members:
   :undoc-members:

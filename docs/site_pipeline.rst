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


make-glitch-flags
-----------------

The glitch flags script saves two types of data in the glitch AxisManager,
`glitch_flags` is a RangesMatrix containing the masks for the default parameters
and `glitch_detection` which contains the significance of the glitch detection
(for all points over the `n_sig` threshold). The glitch detection information
ignores the buffer parameter.

Command line arguments
``````````````````````

.. argparse::
   :module: sotodlib.site_pipeline.make_glitch_flags
   :func: get_parser

Config file format
``````````````````

Here's an annotated example:

.. code-block:: yaml
  
  # Context for <whatever>
  context_file: ./context4_b.yaml

  # How to subdivide observations (by detset)
  subobs:
    use: detset
    label: detset

  # Metadata index & archive filenaming
  archive:
    index: 'archive.sqlite'
    policy:
      type: 'simple'
      filename: 'archive.h5'

  # Flag parameters
  flag_params:
    default: {
      't_glitch': 0.002,
      'hp_fc': 0.5,
      'n_sig': 10,
      'buffer': 200,
    }

Support
=======

.. automodule:: sotodlib.site_pipeline.util
   :members:
   :undoc-members:

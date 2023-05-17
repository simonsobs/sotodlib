.. py:module:: sotodlib.hwp.g3thwp

.. _g3thwp-section:

=============
HWP submodule
=============

G3tHWP
======
G3tHWP is class to analyze HWP parameter in level 2 housekeeping g3 file 

.. autoclass:: G3tHWP

Initialize
----------
G3tHWP class has be instantiated with a config yaml file.
The config file contains the HWP hardware information.
We can include a time range + data path or a full-path file list of input g3 data.
::

  hwp_angle_tool = hwp.G3tHWP('hwp_config.yaml')

Here's an annotated example of yaml config file:

.. code-block:: yaml

  # path to L2 HK directory
  data_dir: '/data/prefix/hk/'

  # path to L2.5 HK directory
  output_dir: '/path/to/save/files/'

  # prefix of field name 
  # for 1st encoder
  field_instance: 'observatory.HBA.feeds.HWPEncoder'

  # for 2nd encoder 
  field_instance_sub: 'observatory.HBA2.feeds.HWPEncoder'

  # name list of HWP_encoder field variable
  field_list:
  - 'rising_edge_count'
  - 'irig_time'
  - 'counter'
  - 'counter_index'
  - 'irig_synch_pulse_clock_time'
  - 'irig_synch_pulse_clock_counts'
  - 'quad'
   
  # HWP OCS packet size
  packet_size: 120

  # IRIG type
  # 0: 1Hz IRIG (default), 1: 10Hz IRIG
  irig_type: 0
  
  # number of slots * 2
  num_edges: 1140

  # Number of slots representing the width of the reference slot
  ref_edges: 2

  # Threshoild for outlier data to calculate nominal slit width
  slit_width_lim: 0.1
  
  # Search range of reference slot
  ref_range: 0.1

  # force to quad value
  # 0: use measured quad value (default)
  # 1: positive rotation direction, -1: negative rotation direction
  force_quad: 0
 

Data Loading 
------------

There are two functions to load L2 HK g3 file.

1) Load data for a given time range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
HK data can be loaded by `load_data` function with time range and data file path.
::
   
   data = hwp_angle_tool.load_data(start, end, data_dir)

We can also specify start, end and data_dir by config yaml file
::
   
   # start and end time for `update_hwp_angle.load_data`
   start: timestamp
   end: timestamp

   # HK data directory for `update_hwp_angle.load_data` 
   data_dir: '/path/to/L2/HK/data/'

If you input both, config file setting will be overwritten. 

As optional, HWP field name instance can be overwritten from config file.
::
   
   data = hwp_angle_tool.load_data(start, end, data_dir, instance)

2) Load data with given file name list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::
   
   data = hwp_angle_tool.load_data(file_list)

We can also specify file_list by config yaml file
::
   
   # input file name for `update_hwp_angle.load_file` 
   file_list: '/mnt/so1/data/ucsd-sat1/hk/16183/1618372860.g3'

If you input both, config file will be overwritten. 

As optional, HWP field name instance can be overwritten from config file.
::
   
   data = hwp_angle_tool.load_file(file_list, instance)


Analyze loaded data
-------------------

We can simply get HWP angle and related info. by
::

   solved = hwp_angle_tool.analyze(data)

where the return value is python dict of
::

   {fast_time, angle, slow_time, stable, locked, hwp_rate}

See :ref:`refelence<g3thwp-ref-section>` section for the definition of each variable.

Write HWP g3 file
-----------------

There is a fucntion to output analyzed data with HWP g3 format
::

   hwp_angle_tool.write_solution(solved,'output.g3')

We can also specify output file path+name in config file.
::

   # output g3 file path + name
   output: './output.g3'

If you specify both, config file setting will be overwritten.

.. _g3thwp-ref-section:

Reference
---------

Class and function references should be auto-generated here.

.. autoclass:: G3tHWP
    :members: load_data
    :noindex:

.. autoclass:: G3tHWP
    :members: load_file
    :noindex:

.. autoclass:: G3tHWP
    :members: analyze
    :noindex:

.. autoclass:: G3tHWP
    :members: write_solution
    :noindex:


HWPSS
=====
.. autofunction:: sotodlib.hwp.hwp.get_hwpss
	  
.. py:module:: sotodlib.hwp.sim_hwp

.. autofunction:: sotodlib.hwp.sim_hwp.sim_hwpss

.. autofunction:: sotodlib.hwp.sim_hwp.sim_hwpss_2f4f

.. autofunction:: sotodlib.hwp.sim_hwp.I_to_P_param

Demodulation
============

.. autofunction:: sotodlib.hwp.hwp.demod_tod

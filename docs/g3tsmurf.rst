.. py:module:: sotodlib.io.load_smurf

=========
G3tSMuRF 
=========


G3tSmurf currently consists of two parts.

1. A databasing setup for tracking available SMuRF data saved using the SO
software stack. 

2. A mechanism for loading that data.

Eventually the databasing part will be separated out and migrated to work within
the Context system and the loading data functions will be altered to load data
through a Context object. 


.. autofunction:: load_file
.. autofunction:: get_channel_mask
.. autofunction:: get_channel_info


.. autoclass:: G3tSmurf
   :special-members: __init__, load_data
   :members:

.. autoclass:: SmurfStatus
   :special-members: from_file, from_time

Database Tables
---------------
.. autoclass:: Observations
   :members:
.. autoclass:: Detsets
   :members:
.. autoclass:: ChanAssignments
   :members:
.. autoclass:: Bands
   :members:

.. autoclass:: Files
   :members:
.. autoclass:: Frames
   :members:

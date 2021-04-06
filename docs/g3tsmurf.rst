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

.. autoclass:: sotodlib.io.load_smurf.G3tSmurf
   :special-members: __init__
   :members:

Database Tables
---------------
.. autoclass:: sotodlib.io.load_smurf.Observations
   :members:
.. autoclass:: sotodlib.io.load_smurf.Detsets
   :members:
.. autoclass:: sotodlib.io.load_smurf.ChanAssignments
   :members:
.. autoclass:: sotodlib.io.load_smurf.Bands
   :members:

.. autoclass:: sotodlib.io.load_smurf.Files
   :members:
.. autoclass:: sotodlib.io.load_smurf.Frames
   :members:

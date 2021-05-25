.. py:module:: sotodlib.io.load_smurf

=========
G3tSMuRF 
=========


G3tSmurf currently consists of two parts.

1. A database setup for tracking available SMuRF data saved using the SO
software stack. 

2. Mechanisms for loading that data.

The G3tSmurf database can be used to load data through `load_data` or
`load_file`. The only real difference between these (currently) is that
`load_data` accepts a start and stop time while `load_file` accepts a list of
paths.

A tutorial with examples of data loading is avaliable
`here
<https://github.com/simonsobs/pwg-tutorials/blob/master/sotodlib_G3tSMuRF_Tutorial_20210415.ipynb>`_.
The description of the databases are partially out of date.

Database Updating
-----------------
Here is an example database update script::
    
    from sotodlib.io.load_smurf import (G3tSmurf, Observations)

    SMURF = G3tSmurf( archive_path='/data/timestreams/',
                      meta_path='/data/smurf/', 
                      db_path='/path/to/things/g3t_smurf_db.db')
    SMURF.index_archive()
    SMURF.index_metadata()

    ## Check recent observations for updates
    ## Accounts for partial data transfers
    session = SMURF.Session()
    new_obs = session.query(Observations)
    new_obs = new_obs.filter( Observations.start >= dt.datetime.now()-dt.timedelta(days=1))
    new_obs = new_obs.all()
    for obs in new_obs:
        SMURF.update_observation_files(obs, session)
    session.close()

Usage with Context
------------------
The G3tSmurf database can now be used with the larger `sotodlib` Context system.
In this setup, the main G3tSmurf database is both the ObsFileDb and the ObsDb.
The DetDb needs to be created using the `dump_DetDb(SMURF, detdb_file)`. This
function call can be added to the standard database update script to keep it up
to date. 

Example context yaml file::

    tags:
    g3tsmurf_dir: '/path/to/things/smurf_context'

    obsfiledb: '{g3tsmurf_dir}/g3t_smurf_db.db'
    obsdb: '{g3tsmurf_dir}/g3t_smurf_db.db'
    detdb: '{g3tsmurf_dir}/detdb.db'

    imports:
    - sotodlib.io.load_smurf

    obs_loader_type: 'g3tsmurf'

    #metadata:
  

Data Loading Functions
----------------------
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

.. image:: images/G3tSmurf_table_relations.png
   :width: 600

.. autoclass:: Observations
   :members:
.. autoclass:: Tunes
   :members
.. autoclass:: TuneSets
   :members:
.. autoclass:: ChanAssignments
   :members:
.. autoclass:: Channels
   :members:

.. autoclass:: Files
   :members:
.. autoclass:: Frames
   :members:


==============================
Loading L3 Houskeeping Data
==============================

The ``sotodlib.io.hkdb`` module provides methods for indexing, and quickly
loading housekeeping data from an archive.
The databases implemented store the following information:

- List of all housekeeping files, with start and stop times
- List of every frame in the file, with the start and stop times of each frame,
  agent instance-id, and the byte-offset of the frame in the file.

To index and load housekeeping information, you will need to create a
``HkConfig`` object, documented in the API section below. For example, the
following yaml file can be loaded in directly with ``HkConfig.from_yaml``:

.. code-block:: yaml

    hk_root: /so/data/satp1/hk
    db_file: satp1_hk.db
    aliases:
        fp_temp: cryo-ls372-lsa21yc.temperatures.Channel_02_T

In the example above, the ``db_file`` is the path to the sqlite index database.
It is possible to use a general database engine instead of sqlite by using the
``db_url`` parameter instead of ``db_file``, which should be a valid sqlalchemy
database URL as is described `here
<https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls>`_. For
example, if the index is saved in a postgres database, you can use the following
config:

.. code-block:: yaml

    hk_root: /so/data/satp1/hk
    db_url: postgresql://<user>:${psql_password}@storage1/hkdb_satp1
    aliases:
        fp_temp: cryo-ls372-lsa21yc.temperatures.Channel_02_T

where ``psql_password`` will be expanded with the value of the environment
variable. Similarly, the ``db_url`` can also be set as a dictionary with fields
matching the URL fields from the sqlalchemy docs, for example:

.. code-block:: yaml

    hk_root: /so/data/satp1/hk
    db_url:
        drivername: postgresql
        username: <user>
        password: ${psql_password}
        host: storage1
        database: hkdb_satp1
    aliases:
        fp_temp: cryo-ls372-lsa21yc.temperatures.Channel_02_T


In addition to the hk-root and database paths, an "alias" dictionary can
be defined here, to provide a mapping from a human-readable name to the
field-id of the housekeeping data, where field-ids are always
of the form ``<agent-instance>.<feed>.<field>``.  These aliases are only used
for data loading, do not effect how the index data is stored in the database.
This means aliases can also be added or modified, either in the config file or
programmatically to the HkConfig object, to assist with data loading after that
hk data has already been indexed.

Loading Data
---------------
To load data from the hk archive, you can pass a ``LoadSpec`` object to the
``load_hk`` function, which defines the time range and fields to load.

For example, the code below will load the focal-plane temp for the last week:

.. code-block:: python

    from sotodlib.io import hkdb
    cfg = hkdb.HkConfig.from_yaml('satp1_hk.yaml')
    t1 = time.time()
    t0 = t1 - 24 * 3600 * 7
    lspec = hkdb.LoadSpec(
        cfg=cfg, start=t0, end=t1,
        fields=['fp_temp'],
    )
    result = hkdb.load_hk(lspec, show_pb=True)

The result object will be an HkResult object, where ``result.data`` contains the
mapping from field-id to a numpy array of shape (2, nsamp), where the first
entry are the timestamps, and the second entry are the data values.  If any of
the requested fields have an alias defined in the ``cfg`` object, that alias
will be set as an attribute in the HkResult object:

.. code-block:: python

    >> print(result.data['cryo-ls372-lsa21yc.temperatures.Channel_02_T'].shape)
    (2, 4160195)
    >> print(result.fp_temp)  # same exact thing as above
    (2, 4160195)
    >> plt.plot(*result.fp_temp)  # plots focal-plane temp

The ``fields`` argument to ``LoadSpec`` is a list of field-specifications, which
can either be:

- An alias in the HkConfig
- A field-id of the form ``<agent-instance>.<feed>.<field>``

Field-ids specifications will also allow wildcards as the feed or field, to
allow one to easily load all the fields of an agent or a feed.
For example, you can run:

.. code-block:: python

    from sotodlib.io import hkdb
    cfg = hkdb.HkConfig.from_yaml('satp1_hk.yaml')
    t1 = time.time()
    t0 = t1 - 24 * 3600 * 7
    lspec = hkdb.LoadSpec(
        cfg=cfg, start=t0, end=t1,
        fields=['cryo-ls372-lsa21yc.*.*'],
    )
    result = hkdb.load_hk(lspec, show_pb=True)
    print(result.data.keys())

    >> dict_keys(['cryo-ls372-lsa21yc.temperatures.Channel_02_R',
                  'cryo-ls372-lsa21yc.temperatures.Channel_02_T',
                  'cryo-ls372-lsa21yc.temperatures.sample_heater_out'])

    # Note that the fp_temp alias will still be found, and assigned to the
    # correct field.
    print(result.fp_temp.shape)
    >> (2, 4157762)


API
-------
.. automodule:: sotodlib.io.hkdb
   :members: HkConfig, HkDb, LoadSpec, load_hk, HkResult
   :undoc-members:
   :show-inheritance:
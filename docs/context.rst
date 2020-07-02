.. _context:

===========================
Context and Metadata system
===========================

This documentation is incomplete -- some key classes are not
documented at all.


.. contents:: Jump to:
   :local:

---------------
Context [empty]
---------------

--------------------------------------
 DetDb: Detector Information Database
--------------------------------------

The purpose of the ``DetDb`` class is to give analysts access to
quasi-static detector metadata.  Some good example of quasi-static
detector metadata are: the name of telescope the detector lives in,
the approximate central frequency of its passband, the approximate
position of the detector in the focal plane.  This database is not
intended to carry precision results needed for mapping, such as
calibration information or precise pointing and polarization angle
data.

Certain properties may change with time, for example if wiring or
optics tube arrangements are adjusted from one season of observations
to the next.  The ``DetDb`` is intended to support values that change
with time.  **The timestamp support is still under development.**


Using a DetDb (Tutorial)
========================

Loading a database into memory
------------------------------

To load an existing :py:obj:`DetDb<sotodlib.core.metadata.DetDb>` into memory, use
the :py:obj:`DetDb.from_file<sotodlib.core.metadata.DetDb.from_file>` class method::

  >>> from sotodlib.core import metadata
  >>> my_db = metadata.DetDb.from_file('path/to/database.sqlite')

.. _DetDb : :py:obj:`blech<sotodlib.core.metadata.DetDb.from_file>`

This function understands a few different formats; see the method
documentation.

If you want an example database to play with, run this::

  >>> from sotodlib.core import metadata
  >>> my_db = metadata.get_example('DetDb')
  Creating table base
  Creating table geometry
  Creating LF-type arrays...
  Creating MF-type arrays...
  Creating HF-type arrays...
  Committing 17094 detectors...
  Checking the work...
  >>> my_db
  <sotodlib.core.metadata.detdb.DetDb object at 0x7f691ccb4080>

The usage examples below are based on this example database.


Detectors and Properties
------------------------

The typical use of DetDb involves alternating use of the ``dets`` and
``props`` functions.  The ``dets`` function returns a list of
detectors with certain indicated properties; the ``props`` function
returns the properties of certain indicated detectors.

We can start by getting a list of *all* detectors in the database::

  >>> det_list = my_db.dets()
  >>> det_list
  ResultSet<[name], 17094 rows>

The :py:obj:`ResultSet<sotodlib.core.metadata.ResultSet>` is a simple container for
tabular data.  Follow the link to the class documentation for the
detailed interface.  Here we have a single column, giving the detector
name::

  >>> det_list['name']
  array(['LF1_00000', 'LF1_00001', 'LF1_00002', ..., 'HF2_06501',
         'HF2_06502', 'HF2_06503'], dtype='<U9')

Similarly, we can retrieve all of the properties for all of the
detectors in the database::

  >>> props = my_db.props()
  >>> props
  ResultSet<[base.instrument,base.camera,base.array_code,
  base.array_class,base.wafer_code,base.freq_code,base.det_type,
  geometry.wafer_x,geometry.wafer_y,geometry.wafer_pol], 17094 rows>

The output of ``props()`` is also a ``ResultSet``; but it has many
columns.  The property values for the first detector are:

  >>> props[0]
  {'base.instrument': 'simonsobs', 'base.camera': 'latr',
   'base.array_code': 'LF1', 'base.array_class': 'LF',
   'base.wafer_code': 'W1', 'base.freq_code': 'f027',
   'base.det_type': 'bolo', 'geometry.wafer_x': 0.0,
   'geometry.wafer_y': 0.0, 'geometry.wafer_pol': 0.0}

We can also inspect the data by column, e.g. ``props['base.camera']``.
Note that `name` isn't a column here... each row corresponds to a
single detector, in the order returned by my_db.dets().


Querying detectors based on properties
--------------------------------------

Suppose we want to get the names of the detectors in the (nominal) 93
GHz band.  These are signified, in this example, by having the value
``'f093'`` for the ``base.freq_code`` property.  We call ``dets()``
with this specfied::

  >>> f093_dets = my_db.dets(props={'base.freq_code': 'f093'})
  >>> f093_dets
  ResultSet<[name], 5184 rows>

The argument passed to the ``props=`` keyword, here, is a dictionary
containing certain values that must be matched in order for a detector
to be included in the output ResultSet.  One can also pass a *list* of
such dictionaries (in which case a detector is included if it fully
matches any of the dicts in the list).  One can, to similar effect,
pass a ResultSet, which results in detectors being checked against
each row of the ResultSet.

Similarly, we can request the properties of some sub-set of the
detectors; let's use the ``f093_dets`` list to confirm that these
detectors are all in ``MF`` arrays::

  >>> f093_props = my_db.props(f093_dets, props=['base.array_class'])
  >>> list(f093_props.distinct())
  [{'base.array_class': 'MF'}]

Note we've used the
:py:obj:`ResultSet.distinct()<sotodlib.core.metadata.ResultSet.distinct>` method to
eliminate duplicate entries in the output from ``props()``.  If you
prefer to work with unkeyed data, you can work with ``.rows`` instead
of converting to a list::

  >>> f093_props.distinct().rows
  [('MF',)]


Grouping detectors by property
------------------------------

Suppose we want to loop over all detectors, but with them grouped by
array name and frequency band.  There are many ways to do this, but a
very general approach is to generate a list of tuples representing the
distinct combinations of these properties.  We then loop over that
list, pulling out the names of the matching detectors for each tuple
of property values.

Here's an example, which simply counts the results::

  # Get the two properties, one row per detector.
  >>> props = my_db.props(props=[
  ...   'base.array_code', 'base.freq_code'])
  # Reduce to the distinct combinations (only 14 rows remain).
  >>> combos = props.distinct()
  # Loop over all 14 combos:
  >>> for combo in combos:
  ...   these_dets = my_db.dets(props=combo)
  ...   print('Combo {} includes {} dets.'.format(combo, len(these_dets)))
  ...
  Combo {'base.array_code': 'HF1', 'base.freq_code': 'f225'} includes 1626 dets.
  Combo {'base.array_code': 'HF1', 'base.freq_code': 'f278'} includes 1626 dets.
  Combo {'base.array_code': 'HF2', 'base.freq_code': 'f225'} includes 1626 dets.
  # ...
  Combo {'base.array_code': 'MF4', 'base.freq_code': 'f145'} includes 1296 dets.


Extracting useful detector properties
-------------------------------------

There are a couple of standard recipes for getting data out
efficiently.  SUppose you want to extract two verbosely-named
numerical columns `geometry.wafer_x` and `geometry.wafer_y`.  We want
to be sure to only type those key names out once::

  # Find all 'LF' detectors.
  >>> LF_dets = my_db.dets(props={'base.array_class': 'LF'})
  >>> LF_dets
  ResultSet<[name], 222 rows>
  # Get positions for those detectors.
  >>> positions = my_db.props(LF_dets, props=['geometry.wafer_x',
  ... 'geometry.wafer_y'])
  >>> x, y = numpy.transpose(positions.rows)
  >>> y
  array([0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.18, 0.2 ,
         0.22, 0.24, 0.26, 0.28, 0.3 , 0.32, 0.34, 0.36, 0.38, ...])
  # Now go plot stuff using x and y...
  # ...


Note in the last line we've used numpy to transform the tabular data
(in `ResultSet.rows`) into a simple (n,2) float array, which is then
transposed to a (2,n) array, and broadcast to variables x and y.  It
is import to include the `.rows` there -- a direct array conversion on
`positions` will not give you what you want.

Inspecting a database
---------------------

If you want to see a list of the properties defined in the database,
just call ``props`` with an empty list of detectors.  Then access the
``keys`` data member, if you want programmatic access to the list of
properties::

  >>> props = my_db.props([])
  >>> props
  ResultSet<[base.instrument,base.camera,base.array_code,
  base.array_class,base.wafer_code,base.freq_code,base.det_type,
  geometry.wafer_x,geometry.wafer_y,geometry.wafer_pol], 0 rows>
  >>> props.keys
  ['base.instrument', 'base.camera', 'base.array_code',
  'base.array_class', 'base.wafer_code', 'base.freq_code',
  'base.det_type', 'geometry.wafer_x', 'geometry.wafer_y',
  'geometry.wafer_pol']


Creating a DetDb [empty]
========================


Database organization
=====================

**The dets Table**

The database has a primary table, called ``dets``.  The ``dets`` table
has only the following columns:

``name``

    The string name of each detector.

``id``

    The index, used internally, to enumerate detectors.  Do not assume
    that the correspondence between index and name will be static --
    it may change if the underlying inputs are changed, or if a
    database subset is generated.


**The Property Tables**

Every other table in the database is a property table.  Each row of
the property table associates one or more (key,value) pairs with a
detector for a particular range of time.  A property table contains at
least the following 3 columns:

``det_id``

    The detector's internal id, a reference to ``dets.id``.

``time0``

    The timestamp indicating the start of validity of the properties
    in the row.

``time1``

    The timestamp indicating the end of validity of the properties in
    the row.  The pair of timestamps ``time0`` and ``time1`` define a
    semi-open interval, including all times ``t`` such that ``time0 <=
    t < time1``.

All other columns in the property table provide detector metadata.
For a property table to contain valid data, the following criteria
must be satisfied:

1. The range of validity must be non-negative, i.e. ``time0 <=
   time1``.  Note that if ``time0 == time1``, then the interval is
   empty.

2. In any one property table, the time intervals associated with a
   single ``det_id`` must not overlap.  Otherwise, there would be
   ambiguity about the value of a given property.

Internally, query code will assume that above two conditions are
satisfied.  Functions exist, however, to verify compliance of property
tables.


DetDb auto-documentation
========================

Auto-generated documentation should appear here.

.. autoclass:: sotodlib.core.metadata.DetDb
   :members:


------------------------------------
Observation Database (ObsDb) [empty]
------------------------------------

---------
ResultSet
---------

Auto-generated documentation should appear here.

.. autoclass:: sotodlib.core.metadata.ResultSet
   :members:


-------------------------------------
Observation File Database (ObsFileDb)
-------------------------------------

The purpose of ObsFileDb is to provide a map into a large set of TOD
files, giving the names of the files and a compressed expression of
what time indices and detectors are present in each file.

Organization of Files
=====================

The ObsFileDb is represented on disk by an sqlite database file.  The
sqlite database contains information about data files, and the
*partial paths* of the data files.  By *partial paths*, we mean that
only the part of the filenames relative to some root node of the data
set should be stored in the database.  In order for the code to find
the data files easily, it is most natural to place the
obsfiledb.sqlite file in that same root node of the data set.
Consider the following file listing as an example::

  /
   data/
        planet_obs/
                   obsfiledb.sqlite     # the database
                   observation0/        # directory for an obs
                                data0_0 # one data file
                                data0_1
                   observation1/
                                data1_0
                                data1_1
                   observation2/
                   ...

On this particular file system, our "root node of the data set" is
located at ``/data/planet_obs``.  All data files are located in
subdirectories of that one root node directory.  The ObsFileDb
database file is also located in that directory, and called
``obsfiledb.sqlite``.  The filenames in obsfiledb.sqlite are written
relative to that root node directory; for example
``observation0/data0_1``.  This means that we can copy or move the
contents of the ``planet_obs`` directory to some other path and the
ObsFileDb will not need to be updated.

There are functions in ObsFileDb that return the full path to the
files, rather than the partial path.  This is achieved by combining
the partial file names in the database with the ObsFileDb instance's
"prefix".  By default, "prefix" is set equal to the directory where
the source sqlite datafile is located.  But it can be overridden, if
needed, when the ObsFileDb is instantiated (or afterwards).


Data Model
==========

We assume the following organization of the data:

- The data are divided into contiguous segments of time called
  "Observations".  An observation is identified by an ``obs_id``,
  which is a string.
- An Observation involves a certain set of co-sampled detectors.  The
  files associated with the Observation must contain data for all the
  Observation's detectors at all times covered by the Observation.
- The detectors involved in a particular Observation are divided into
  groups called detsets.  The purpose of detset grouping is to map
  cleanly onto files, thus each file in the Observation should contain
  the data for exactly one detset.

Here's some ascii art showing an example of how the data in an
observation must be split between files::

     sample index
   X-------------------------------------------------->
 d |
 e |   +-------------------------+------------------+
 t |   | obs0_waf0_00000         | obs0_waf0_01000  |
 e |   +-------------------------+------------------+
 c |   | obs0_waf1_00000         | obs0_waf1_01000  |
 t |   +-------------------------+------------------+
 o |   | obs0_waf2_00000         | obs0_waf2_01000  |
 r |   +-------------------------+------------------+
   V


In this example the data for the observation has been distributed into
6 files.  There are three detsets, probably called ``waf0``, ``waf1``,
and ``waf2``.  In the sample index (or time) direction, each detset is
associated with two files; apparently the observation has been split
at sample index 1000.

Notes:

- Normally detsets will be coherent across a large set of observations
  -- i.e. because we will probably always group the detectors into
  files in the same way.  But this is not required.
- In the case of non-cosampled arrays that are observing at the same
  time on the same telescope: these qualify as different observations
  and should be given different obs_ids.
- It is currently assumed that in a single observation the files for
  each detset will be divided at the same sample index.  The database
  structure doesn't have this baked in, but some internal verification
  code assumes this behavior.  So this requirement can likely be
  loosened, if need be.


The database consists of two main tables.  The first is called
``detsets`` and associates detectors (string ``detsets.det``) with a
particular detset (string ``detset.name``).  The second is called
``files`` and associates files (``files.name`` to each Observation
(string ``files.obs_id``), detset (string ``files.detset``), and
sample range (integers ``sample_start`` and ``sample_stop``).

The ObsFileDb is intended to be portable with the TOD data.  It should
thus be placed near the data (such as in the base directory of the
data), and use relative filenames.

Constructing the ObsFileDb involves building the detsets and files
tables, using functions ``add_detset`` and ``add_obsfile``.  Using the
ObsFileDb is accomplished through the functions ``get_dets``,
``get_detsets``, ``get_obs``, and through custom SQL queries on
``conn``.


Example Usage
=============

Suppose we have a coherent archive of TOD data files living at
``/mnt/so1/shared/todsims/pipe-s0001/v2/``.  And suppose there's a
database file, ``obsfiledb.sqlite``, in that directory.  We can load
the observation database like this::

  import sotoddb
  db = sotoddb.ObsFileDb.from_file('/mnt/so1/shared/todsims/pip-s0001/v2/')

Note we've given it a directory, not a filename... in such cases the
code will read ``obsfiledb.sqlite`` in the stated directory.

Now we get the list of all observations, and choose one::

  all_obs = db.get_obs()
  print(all_obs[0])   # -> 'CES-Atacama-LAT-Tier1DEC-035..-045_RA+040..+050-0-0_LF'

We can list the detsets present in this observation; then get all the
file info (paths and sample indices) for one of the detsets::

  all_detsets = db.get_detsets(all_obs[0])
  print(all_detsets)  # -> ['LF1_tube_LT6', 'LF2_tube_LT6']

  files = db.get_files(all_obs[0], detsets=[all_detsets[0]])
  print(files['LF1_tube_LT6'])
                      # -> [('/mnt/so1/shared/todsims/pipe-s0001/v2/datadump_LAT_LF1/CES-Atacama-LAT-Tier1DEC-035..-045_RA+040..+050-0-0/LF1_tube_LT6_00000000.g3', 0, None)]


Class Documentation
===================

*The class documentation of ObsFileDb should appear below.*

.. autoclass:: sotodlib.core.metadata.ObsFileDb
   :members:

   .. automethod:: __init__


---------------------------------------
Metadata Manifest Database (ManifestDb)
---------------------------------------

.. py:module:: sotodlib.core.metadata

*This documentation is incomplete.  We will need to explain the role
of ManifestDb and its relation to ObsDb and DetDb.*


Class auto-documentation
========================

*The class documentation of ManifestScheme should appear below.*

.. autoclass:: ManifestScheme
   :members:

*The class documentation of ManifestDb should appear below.*

.. autoclass:: ManifestDb
   :members:


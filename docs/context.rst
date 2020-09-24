.. _context:

===========================
Context and Metadata system
===========================

.. contents:: Jump to:
   :local:

--------
Overview
--------

The Context and Metadata system is intended to provide easy
programmatic access to combinations of time-ordered detector data and
supporting data products ("metadata").  The sectionon on
:ref:`context-section` deals with the configuration files that
describe a dataset, and the use of the Context object to load data and
metadata into sotodlib containers; the section on
:ref:`metadata-section` describes how to store and index metadata
information for use with this system.  Information on TOD indexing and
loading can be found in the section on :ref:`ObsFileDb
<obsfiledb-section>`.


.. _context-section:

-------
Context
-------

.. py:module:: sotodlib.core

Assuming someone has set up a Context for a particular dataset, you
would instantiate it like this::

  from sotodlib.core import Context
  ctx = Context('path/to/the/context.yaml')

This will cause the specified ``context.yaml`` file to be parsed, as
well as user and site configuration files, if those have been set up.
Once the Context is loaded, you will probably have access to the
various databases (``detdb``, ``obsdb``, ``obsfiledb``) and you'll be
able to load TOD and metadata using ``get_obs(...)``.


Dataset, User, and Site Context files
=====================================

The three configuration files are:

Dataset Context File
  This will often be called ``context.yaml`` and will contain a
  description of how to access a particular TOD data set (for example
  a set of observations output from a particular simulation run) and
  supporting metadata (including databases listing the observations,
  detector properties, and intermediate analysis products such as cuts
  or pointing offsets).  When instantiating a Context object, the path
  to this file will normally be the only argument.

Site Context File
  This file contains settings that are common to a particular
  computing site but are likely to differ from one site to the next.
  It is expected that a single file will be made available to all
  sotodlib users on a system.  The main purpose of this is to describe
  file locations, so that the Dataset Context File can be written in a
  way that is portable between computing sites.

User Context File
  This is a yaml file contains user-specific settings, and is loaded
  after the Site Context File but before the Dataset Context File.
  This plays the same role as Site Context but allows for per-user
  tweaking of parameters.

The User and Site Context Files will be looked for in certain places
if not specified explicitly; see :class:`Context` for details.


Annotated Example
=================

The Dataset Context File is yaml, that decodes to a dict.  Here is an
annotated example.

.. code-block:: yaml

  # Define some "tags".  These are string variables that are eligible
  # for substitution into other settings.  The most common use is to
  # establish base paths for data and metadata products.
  tags:
    actpol_shared:   /mnt/so1/shared/data/actpol/depots/scratch
    depot_scratch:  /mnt/so1/shared/data/actpol/depots/scratch
    metadata_lib:   /mnt/so1/shared/data/actpol/depots/scratch/uranus_200327/metadata

  # List of modules to import.  Importing modules such as
  # moby2.analysis.socompat or sotodlib.io.metadata causes certain IO
  # functions to be registered for the metadata system to use.
  imports:
    - moby2.analysis.socompat

  # The basic databases.  The sotodlib TOD loading system uses an
  # ObsFileDb to determine what files to read.  The metadata association
  # and loading system uses a DetDb and an ObsDb to find and read
  # different kinds of metadata.
  obsfiledb: '{metadata_lib}/obsfiledb_200407.sqlite'
  detdb:     '{metadata_lib}/detdb_200109.sqlite'
  obsdb:     '{metadata_lib}/obsdb_200618.sqlite'

  # Additional settings related to TOD loading.
  obs_colon_tags: ['band']
  obs_loader_type: actpol_moby2

  # A list of metadata products that should be loaded along with the
  # TOD.  Each entry in the list points to a metadata database (an
  # sqlite file) and specifies the name under which that information
  # should be associated in the loaded data structure.  In some
  # cases a "loader" name is also given, but usually this is
  # specified in the database.
  metadata:
    - db: "{metadata_lib}/cuts_s17_c11_200327_cuts.sqlite"
      name: "glitch_flags&flags"
    - db: "{metadata_lib}/cuts_s17_c11_200327_planet_cuts.sqlite"
      name: "source_flags&flags"
    - db: "{metadata_lib}/cal_s17_c11_200327.sqlite"
      name: "relcal&cal"
    - db: "{metadata_lib}/timeconst_200327.sqlite"
      name: "timeconst&"
      loader: "PerDetectorHdf5"
    - db: "{metadata_lib}/abscal_190126.sqlite"
      name: "abscal&cal"
    - db: "{metadata_lib}/detofs_200218.sqlite"
      name: "focal_plane"
    - db: "{metadata_lib}/pointofs_200218.sqlite"
      name: "pointofs"


With a context like the one above, a user can load a TOD and its
supporting data very simply::

  from sotodlib.core import Context
  from moby2.analysis import socompat   # For special ACT loader functions

  context = Context('context.yaml')

  # Get a random obs_id from the ObsDb we loaded:
  context.obsdb.get()[10]
  # output is: OrderedDict([('obs_id', '1500022312.1500087647.ar6'), ('timestamp', 1500022313.0)])

  # Load the TOD and metadata for that obs_id:
  tod = context.get_obs('1500022312.1500087647.ar6')

  # The result is an AxisManager with members [axes]:
  # - signal ['dets', 'samps']
  # - timestamps ['samps']
  # - flags ['samps']
  # - boresight ['samps']
  # - array_data ['dets']
  # - glitch_flags ['dets', 'samps']
  # - source_flags ['dets', 'samps']
  # - relcal ['dets']
  # - timeconst ['dets']
  # - abscal ['dets']
  # - focal_plane ['dets']
  # - pointofs ['dets']


Context Schema
==============

Here are all of the top-level entries with special meanings in the
Context system:


``tags``
    A map from string to string.  This entry is treated in a special
    way when the Site, User, and Dataset context files are evalauted
    in series; see below.

``imports``
    A list of modules that should be imported prior to attempting any
    metadata operations.  The purpose of this is to allow IO functions
    to register themselves for use by the Metadata system.  This list
    will usually need to include at least ``sotodlib.io.metadata``.

``obsfiledb``, ``obsdb``, ``detdb``
    Each of these should provide a string.  The string represents the
    path to files carrying an ObsFileDb, ObsDb, and DetDb,
    respectively.  These are all technically optional but it will be
    difficult to load TOD data with the ObsFileDb and it will be
    difficult to load metadata without the ObsDb and DetDb.

``obs_colon_tags``
    A list of strings.  The strings in this list must refer to columns
    from the DetDb.  When a string appears in this list, then the
    values appearing in that column of the DetDb may be used to modify
    an obs_id when requestion TOD data.  For example, suppose DetDb
    has a column 'band' with values ['f027', 'f039', ...].  Suppose
    that ``'obs201230'`` is an observation ID for an array that has
    detectors in bands 'f027' and 'f039'.  Then passing
    ``'obs201230:f027'`` to Context.get_obs will read and return only
    the timestream and metadata for the 'f027' detectors.

``obs_loader_type``
    A string, giving the name of a loader function that should be used
    to load the TOD.  The functions are registered in the module
    variable ``sotodlib.io.load.OBSLOADER_REGISTRY``.


``metadata``
    A list of metadata specs.  Each metadata spec has the following entries:

    ``db``
        The path to the ManifestDb for this metadata spec.

    ``name``
        A string giving instructions for what the loaded thing should
        be called, in the AxisManager.  There is a bit of a
        complicated syntax here, to support renaming things on load
        and stuff like that.  Find more details in the `Metadata`
        section, if you're lucky.

    ``loader``
        The name of the metadata loader function to use; these
        functions must be registered in module variable
        sotodlib.core.metadata.REGISTRY.  This is optional; if absent
        it will default to 'HdfPerDetector' or (more likely) to
        whatever is specified in the ManifestDb.


Context object API
==================

The Context object should be documented below.

.. autoclass:: Context
   :special-members: __init__
   :members:


.. py:module:: sotodlib.core.metadata
.. _metadata-section:

--------
Metadata
--------

The purpose of the "metadata system" in sotodlib is to help identify,
load, and correctly label supporting ancillary data for a particular
observation and set of detectors.

The "metadata" we are talking about here consists of things like
detector pointing offsets, calibration factors, detector time
constants, and so on.  It may also include more complicated or
voluminous data, such as per-sample flags, or signal modes.

When the Context system processes a metadata entry, it will make use
of the DetDb and ObsDb in its interactions with the loaded ManifestDb.

This process has 5 stages.

1. Promote Request.

   The metadata request is likely to include information about what
   observation and what detectors are of interest.  But the ManifestDb
   may desire a slightly different form for this information.  For
   example, an ``obs_id`` might be provided in the request, but the
   ManifestDb might index its results using timestamps
   (``obs:timestamp``).  In the *Promote Request* step, the ManifestDb
   is interrogated for what ``obs`` and ``dets`` fields it requires as
   Index Data.  If those fields are not already in the request, then
   the request is enrichened to include them; this typically requires
   interaction with the ObsDb and/or DetDb.  The enriched request is
   the result fo the Promotion Step.

2. Get Endpoints.

   Once the enriched request is computed, the ManifestDb can be
   queried to see what endpoints match the request.  The ManifestDb
   will return a list of Endpoint results.  Each Endpoint result
   describes the location of some metadata (i.e. a filename and
   possibly some address within that file), as well as any limitations
   on the applicability of that data (e.g. it may specify that
   although the results include values for 90 GHz and 150 GHz
   detectors, only the results for 150 GHz detectors should be kept).
   The metadata results are not yet loaded.

3. Read Metadata.

4. Combine Metadata.

5. Wrap Metadata.


Metadata Archives
=================

A "Metadata Archive" is a set of files that hold metadata to be
accessed by the Context/Metadata system.  The metadata system is
designed to be flexible with respect to how such archives are
structured, at least in terms of the numbers and formats of the files.

One possible form for an archive is a set of HDF5 files, where simple
tabular data are stored in datasets.  This type of archive is used for
the reference implementation of the Metadata system interface,
discussed in the next section.

ResultSet over HDF5
-------------------

This is enabled by sotodlib.metadata.io.  In particular:

- The function ``write_dataset`` knows how to write a ResultSet to an
  HDF5 dataset.
- The ResultSetHdfLoader class is used by the Context/Metadata system
  to load populate ResultSet objects from HDF5 datasets.

Here's an example that creates a compatible dataset, and writes it to
an HDF5 file::

  import h5py
  import numpy as np

  obs_id = 'obs123456'
  timeconst = np.array([('obs123456', 0.001)], dtype=[('obs:obs_id', 'S20'),
                                                      ('timeconst', float)])
  with h5py.File('test.h5', 'w') as fout:
      fout.create_dataset('timeconst_for_obs123456', data=timeconst)

Here's the nearly equivalent operation, using ResultSet and
write_dataset::

  import h5py
  from sotodlib.core import metadata
  from sotodlib.io.metadata import write_dataset

  timeconst = ResultSet(keys=['obs:obs_id', 'timeconst'])
  timeconst.rows.append(('obs123456', 0.001))

  write_dataset(timeconst, 'test2.h5', 'timeconst_for_obs123456', overwrite=True)


------------------------------------
DetDb: Detector Information Database
------------------------------------

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

.. autoclass:: DetDb
   :special-members: __init__
   :members:


---------------------------
ObsDb: Observation Database
---------------------------

Overview
===========

The purpose of the ``ObsDb`` is to help a user select observations
based on high level criteria (such as the target field, the speed or
elevation of the scan, whether the observation was done during the day
or night, etc.).  The ObsDb is also used, by the Context system, to
select the appropriate metadata results from a metadata archive (for
example, the timestamp associated with an observation could be used to
find an appropriate pointing offset).

The ObsDb is constructed from two tables.  The "obs table" contains
one row per observation and is appropriate for storing basic
descriptive data about the observation.  The "tags table" associates
observations to particular labels (called tags) in a many-to-many
relationship.

The ObsDb code does not really require that any particular information
be present; it does not insist that there is a "timestamp" field, for
example.  Instead, the ObsDb can contain information that is needed in
a specific context.  However, recommended field names for some typical
information types are given in `Standardized ObsDb field names`_.

.. note::

   The difference between ObsDb and ObsFileDb is that ObsDb contains
   arbitrary high-level descriptions of Observations, while ObsFileDb
   only contains information about how the Observation data is
   organized into files on disk.  ObsFileDb is consulted when it is
   time to load data for an observation into memory.  The ObsDb is a
   repository of information about Observations, independent of where
   on disk the data lives or how the data files are organized.

Creating an ObsDb
=================

To create an ObsDb, one must define columns for the obs table, and
then add data for one or more observations, then write the results to
a file.

Here is a short program that creates an ObsDb with entries for two
observations::

  from sotodlib.core import metadata

  # Create a new Db and add two columns.
  obsdb = metadata.ObsDb()
  obsdb.add_obs_columns(['timestamp float', 'hwp_speed float', 'drift string'])

  # Add two rows.
  obsdb.update_obs('myobs0', {'timestamp': 1900000000.,
                              'hwp_speed': 2.0,
                              'drift': 'rising'})
  obsdb.update_obs('myobs1', {'timestamp': 1900010000.,
                              'hwp_speed': 1.5,
                              'drift': 'setting'})

  # Apply some tags (this could have been done in the calls above
  obsdb.update_obs('myobs0', tags=['hwp_fast', 'cryo_problem'])
  obsdb.update_obs('myobs1', tags=['hwp_slow'])

  # Save (in gzip format).
  obsdb.to_file('obsdb.gz')


The column definitions must be specified in a format compatible with
sqlite; see :py:meth:`ObsDb.add_obs_columns`.  When the user adds data
using :py:meth:`ObsDb.update_obs`, the first argument is the
``obs_id``.  This is the primary key in the ObsDb and is also used to
identify observations in the ObsFileDb.  When we write the database
using :py:meth:`ObsDb.to_file`, using a .gz extension selects gzip
output by default.


Using an ObsDb
==============

The :py:meth:`ObsDb.query` function is used to get a list of
observations with particular properties.  The user may pass in an
sqlite-compatible expression that refers to columns in the obs table,
or to the names of tags.

Basic queries
-------------

Using our example database from the preceding section, we can try a
few queries::

  >>> obsdb.query()
  ResultSet<[obs_id,timestamp,hwp_speed], 2 rows>

  >>> obsdb.query('hwp_speed >= 2.')
  ResultSet<[obs_id,timestamp,hwp_speed], 1 rows>

  >>> obsdb.query('hwp_speed > 1. and drift=="rising"')
  ResultSet<[obs_id,timestamp,hwp_speed], 1 rows>

The object returned by obsdb.query is a :py:obj:`ResultSet`, from
which individual columns or rows can easily be extracted:

  >>> rs = obsdb.query()
  >>> print(rs['obs_id'])
  ['myobs0' 'myobs1']
  >>> print(rs[0])
  OrderedDict([('obs_id', 'myobs0'), ('timestamp', 1900000000.0),
    ('hwp_speed', 2.0), ('drift', 'rising')])


Queries involving tags
----------------------

Information from the tags table will only show up in the output if
explicitly requested.  For example, we can ask for the ``'hwp_fast'``
and ``'hwp_slow'`` fields to be included::

  >>> obsdb.query(tags=['hwp_fast', 'hwp_slow'])
  ResultSet<[obs_id,timestamp,hwp_speed,drift,hwp_fast,hwp_slow], 2 rows>

Tag columns will have value 1 if the tag has been applied to that
observation, and 0 otherwise.  A query can be filtered based on tags;
there are two ways to do this.  One is to append '=0' or '=1' to the
end of some of the tag strings::

  >>> obsdb.query(tags=['hwp_fast=1'])
  ResultSet<[obs_id,timestamp,hwp_speed,drift,hwp_fast], 1 rows>

Alternately, the values of tags can be used in query strings::

  >>> obsdb.query('(hwp_fast==1 and drift=="rising") or (hwp_fast==0 and drift="setting")',
    tags=['hwp_fast'])
  ResultSet<[obs_id,timestamp,hwp_speed,drift,hwp_fast], 2 rows>
    

Getting a description of a single observation
---------------------------------------------

If you just want the basic information for an observation of known
obs_id, use the get function :py:meth:`ObsDb.get` function::

  >>> obsdb.get('myobs0')
  OrderedDict([('obs_id', 'myobs0'), ('timestamp', 1900000000.0),
    ('hwp_speed', 2.0), ('drift', 'rising')])

If you want a list of all tags for an observation, call get with
tags=True::

  >>> obsdb.get('myobs0', tags=True)
  OrderedDict([('obs_id', 'myobs0'), ('timestamp', 1900000000.0),
    ('hwp_speed', 2.0), ('drift', 'rising'),
    ('tags', ['hwp_fast', 'cryo_problem'])])

So here we see that the observation is associated with tags
``'hwp_fast'`` and ``'cryo_problem'``.

  
Standardized ObsDb field names
==============================

Other than obs_id, specific field names are not enforced in code.
However, there are certain typical bits of information for which it
makes sense to strongly encourage standardized field names.  These are
defined below.

``timestamp``
  A specific moment (as a Unix timestamp) that should be used to
  represent the observation.  Best practice is to have this be fairly
  close to the start time of the observation.

``duration``
  The approximate length of the observation in seconds.


Class auto-documentation
========================

*The class documentation of ObsDb should appear below.*

.. autoclass:: ObsDb
   :special-members: __init__
   :members:


.. _obsfiledb-section:

------------------------------------
ObsFileDb: Observation File Database
------------------------------------

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

.. autoclass:: ObsFileDb
   :special-members: __init__
   :members:


--------------------------------------
ManifestDb: Metadata Manifest Database
--------------------------------------

*This documentation is incomplete.  We will need to explain the role
of ManifestDb and its relation to ObsDb and DetDb.*


Class auto-documentation
========================

*The class documentation of ManifestScheme should appear below.*

.. autoclass:: ManifestScheme
   :special-members: __init__
   :members:

*The class documentation of ManifestDb should appear below.*

.. autoclass:: ManifestDb
   :special-members: __init__
   :members:


---------
ResultSet
---------

Auto-generated documentation should appear here.

.. autoclass:: ResultSet
   :special-members: __init__
   :members:


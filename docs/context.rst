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
supporting data products ("metadata").  The section on
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
    - sotodlib.io.metadata
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
    variable ``sotodlib.core.OBSLOADER_REGISTRY``; also see
    :func:`sotodlib.core.context.obsloader_template`.

``metadata``

    A list of metadata specs.  For more detailed description, see the
    docstring for
    :func:`SuperLoader.load_raw<metadata.SuperLoader.load_raw>`.  But
    briefly each metadata spec is a dictionary with the following
    entries:

    ``db``
        The path to a ManifestDb.

    ``name``

        A string describing what field to extract from the metadata,
        and where to store it in the output AxisManager.  For details
        on the syntax, see
        :func:`Unpacker.decode<metadata.Unpacker.decode>`.

    ``loader``
        The name of the metadata loader function to use; these
        functions must be registered in module variable
        sotodlib.core.metadata.REGISTRY.  This is optional; if absent
        it will default to 'HdfPerDetector' or (more likely) to
        whatever is specified in the ManifestDb.

``context_hooks``

    A string that identifies a set of functions to hook into
    particular parts of the Context system.  See uses of _call_hook()
    in the Context code, for any implemented hook function names and
    their signatures.  To use this feature, an imported module must
    register the hook set in Context.hook_sets (a dict),
    e.g. ``Context.hook_sets["my_sim_hooks"] = {"on-context-ready":
    ...}``, and then also set ``context_hooks: "my_sim_hooks"`` in the
    context.yaml.


Context system APIs
===================

Context
-------

.. autoclass:: Context
   :special-members: __init__
   :members:

obsloader
---------

Data formats are abstracted in the Context system, and "obsloader"
functions provide the implementations to load data for a particular
storage format.  The API is documented in the ``obsloader_template``
function:

.. autofunction:: sotodlib.core.context.obsloader_template

SuperLoader
-----------

.. autoclass:: sotodlib.core.metadata.SuperLoader
   :special-members: __init__
   :members:

Unpacker
--------

.. autoclass:: sotodlib.core.metadata.Unpacker
   :special-members: __init__
   :members:


.. py:module:: sotodlib.core.metadata
.. _metadata-section:

--------
Metadata
--------

Overview and Examples
=====================

The "metadata" we are talking about here consists of things like
detector pointing offsets, calibration factors, detector time
constants, and so on.  It may also include more complicated or
voluminous data, such as per-sample flags, or signal modes.  The
purpose of the "metadata system" in sotodlib is to help identify,
load, and correctly label such supporting ancillary data for a
particular observation and set of detectors.

Storing metadata information on disk requires both a **Metadata
Archive**, which is a set of files containing the actual data, and a
**Metadata Index**, represented by the ManifestDb class, which encodes
instructions for associating metadata information from the archive
with particular observations and detector sets.

When the Context system processes a metadata entry, it will make use
of the DetDb and ObsDb in its interactions with the ManifestDb.
Ultimately the system will load numbers and associate with them with
detectors in a particular observation.  But a Metadata Archive does
not need to have separate records of each number, for each detector
and each observation.  When appropriate, the data could be stored so
that there is a single value for each detector, applicable for an
entire season.  Or there could be different calibration numbers for
each wafer in an array, that change depending on the observation ID.
As long as the ObsDb and DetDb know about the parameters being used to
index things in the Metadata Archive (e.g., the timestamp of
observations and the wafer for each detector), the system can support
resolving the metadata request, loading the results, and broadcasting
them to their intended targets.


Example 1
---------

Let's say we want to build an HDF5 database with a number ``thing``
per detector per observation::

    from sotodlib.core import Context, metadata
    import sotodlib.io.metadata as io_meta

    context = Context('context_file_no_thing.yaml')
    obs_rs = context.obsdb.query()
    h5_file = 'thing.h5'

    for i in range(len(obs_rs)):
        aman = context.get_obs(obs_rs[i]['obs_id'])
        things = calculate_thing(aman)
        thing_rs = metadata.ResultSet(keys=['dets:readout_id', 'thing'])
        for d, det in enumerate(aman.dets.vals):
            thing_rs.rows.append((det, things[d]))
        io_meta.write_dataset(thing_rs, h5_file, f'thing_for_{obs_id}')

Once we've built the lower level HDF5 file we need to add it to a
metadata index::

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match('obs:obs_id')
    scheme.add_data_field('dataset')

    db = metadata.ManifestDb(scheme=scheme)
    for i in range(len(obs_rs)):
        obs_id = obs_rs[i]['obs_id']
        db.add_entry({'obs:obs_id': obs_id,
                      'dataset': f'thing_for_{obs_id}',},
                       filename=h5_file)

    db.to_file('thing_db.gz')

Then have a new context file that includes::

    metadata:
        - db : 'thing_db.gz'
          name : 'thing'

Using that context file::

    context = Context('context_file_with_thing.yaml')
    aman = context.get_obs(your_favorite_obs_id)

will return an AxisManager that includes ``aman.thing`` for that
specific observation.

Example 2
---------

In this example, we loop over observations found in an ObsDb and
create an AxisManager for each one that contains new, interesting
supporting data.  The AxisManager is written to HDF5, and the
ManifestDb is updated with indexing information so the metadata system
can find the right dataset automatically.

Here is the script to generate the HDF5 and ManifestDb on disk::

  from sotodlib import core

  # We will create an entry for every obs found in this context.
  ctx = core.Context('context/context-basic.yaml')

  # Set up our scheme -- one result per obs_id, to be looked up in an
  # archive of HDF5 files at address stored in dataset.
  scheme = core.metadata.ManifestScheme()
  scheme.add_exact_match('obs:obs_id')
  scheme.add_data_field('dataset')
  man_db = core.metadata.ManifestDb(scheme=scheme)

  # Path for the ManifestDb
  man_db_filename = 'my_new_info.sqlite'

  # Use a single HDF5 file for now.
  output_filename = 'my_new_info.h5'

  # Loop over obs.
  obs = ctx.obsdb.get()
  for obs_id in obs['obs_id']:
      print(obs_id)

      # Load the observation, without signal, so we can see the samps
      # and dets axes, timestamps, etc.
      tod = ctx.get_obs(obs_id, no_signal=True)

      # Create an AxisManager using tod's axes.  (Note that the dets
      # axis is required for compatibility with the metadata system,
      # even if you're not going to use it.)
      new_info = core.AxisManager(tod.dets, tod.samps)

      # Add some stuff to it
      new_info.wrap_new('my_special_vector', shape=('samps', ))

      # Save the info to HDF5
      h5_address = obs_id
      new_info.save(output_filename, h5_address, overwrite=True)

      # Add an entry to the database
      man_db.add_entry({'obs:obs_id': obs_id, 'dataset': h5_address},
                        filename=output_filename)

  # Commit the ManifestDb to file.
  man_db.to_file(man_db_filename)


The new context.yaml file, that includes this new metadata, would have
a metadata list that includes::

    metadata:
      - db: 'my_new_info.sqlite'
        name: 'new_info'

If you want to load the single vector ``my_special_vector`` into the
top level of the AxisManager, under name ``special``, use this
syntax::

    metadata:
      - db: 'my_new_info.sqlite'
        name: 'special&my_special_vector'


Tips
----

If this is example is almost, but not quite, what you need, consider the
following:

- You can use multiple HDF5 files in your Metadata Archive -- the
  filename is a parameter to ``db.add_entry``.  That helps to keep
  your HDF5 files a manageable size, and is good practice in cases
  where there are regular (e.g. daily or hourly) updates to an
  archive.
- You can use an archive format other than HDF5 (if you must), see the
  example in :ref:`metadata-archives-custom`.
- The entries in the Index do not have to be per-``obs_id``.  You can
  associate results to ranges of time or to other fields in the ObsDb.
  See examples in :ref:`metadata-indexes`.
- The entries in the Archive do not need to be per-detector.  You can
  specify results for a whole group of detectors, if that group is
  enumerated in the DetDb (or added to det_info using other metadata).
  For example, if DetDb contains a column ``passband``, the dataset
  could contain columns ``dets:passband`` and ``cal`` and simply
  report one calibration number for each frequency band.  (On load,
  the Context Metadata system will automatically broadcast the ``cal``
  number so that it has shape ``(n_dets,)`` in the fully populated
  AxisManager.)
- You can store all the results (i.e., results for multiple
  ``obs_id``) in a single HDF5 dataset.  This is not usually a good
  idea if your results are per-detector, per-observation... the
  dataset will be huge, and not easy to update incrementally.  But for
  smaller things (one or two numbers per observation, as in the
  ``dets:passband`` example above) it can be convenient.  Doing this
  requires including ``obs:obs_id`` (or some other ObsDb column) in
  the dataset.


.. _metadata-archives:

Metadata Archives
=================

A **Metadata Archive** is a set of files that hold metadata to be
accessed by the Context/Metadata system.  The metadata system is
designed to be flexible with respect to how such archives are
structured, at least in terms of the numbers and formats of the files.

A **Metadata Instance** is a set of numerical and string data, stored
at a particular address within a Metadata Archive.  The Instance will
typically have information organized in a tabular form, equivalent to
a grid where the columns have names and each row contains a set of
coherent data.  Some of the columns describe the detectors or
observations to which the row applies.  The other columns provide
numerical or string information that is to be assocaited with those
detectors or observations.

For example, consider a table with columns called `dets:id` and `cal`.
The entry for `dets:id` in each row specifies what detector ID the
number in the `cal` column of that row should be applied to.  The
indexing information in `dets:id` is called *instrinsic indexing
information*, because it is carried within the Metadata Instance.  In
contrast, *extrinsic indexing information* is specified elsewhere.
Usualy, extrinsic indexing information is provided through the
Metadata Index (ManifestDb).

One possible form for an archive is a set of HDF5 files, where simple
tabular data are stored in datasets.  This type of archive is used for
the reference implementation of the Metadata system interface, so we
will describe support for that first and then later go into using
other formats.

ResultSet over HDF5
-------------------

The ResultSet is a container for tabular data.  Functions are
available in sotodlib.io.metadata for reading and writing ResultSet
data to HDF5 datasets.

Here's an example using h5py that creates a compatible dataset, and
writes it to an HDF5 file::

  import h5py
  import numpy as np

  obs_id = 'obs123456'
  timeconst = np.array([('obs123456', 0.001)], dtype=[('obs:obs_id', 'S20'),
                                                      ('timeconst', float)])
  with h5py.File('test.h5', 'w') as fout:
      fout.create_dataset('timeconst_for_obs123456', data=timeconst)

Here's the equivalent operation, accomplished using :class:`ResultSet`
and :func:`write_dataset<sotodlib.io.metadata.write_dataset>`::

  from sotodlib.core import metadata
  from sotodlib.io.metadata import write_dataset

  timeconst = metadata.ResultSet(keys=['obs:obs_id', 'timeconst'])
  timeconst.rows.append(('obs123456', 0.001))

  write_dataset(timeconst, 'test2.h5', 'timeconst_for_obs123456', overwrite=True)

The advantages of using write_dataset instead of h5py primitives are:

- You can pass it a ResultSet directly and it will handle creating the
  right types.
- Passing overwrite=True will handle the removal of any existing
  entries at the target path.

To inspect datasets in a HDF5 file, you can load them using h5py
primitives, or with
:func:`read_dataset<sotodlib.io.metadata.read_dataset>`::

  from sotodlib.io.metadata import read_dataset

  timeconst = read_dataset('test2.h5', 'timeconst_for_obs123456')

The returned object looks like this::

  >>> print(timeconst)
  ResultSet<[obs:obs_id,timeconst], 1 rows>


The metadata handling code does not use read_dataset.  Instead it uses
ResultSetHdfLoader, which has some optimizations for loading batches
of metadata from HDF5 files and datasets, and will forcefully
reconcile any columns prefixed by ``obs:`` or ``dets:`` against the
provided request (using detdb and obsdb, potentially).  Loading the
time constants for ``obs123456`` is done like this::

  from sotodlib.io.metadata import ResultSetHdfLoader

  loader = ResultSetHdfLoader()

  request = {'filename': 'test2.h5',
             'dataset': 'timeconst_for_obs123456',
             'obs:obs_id': 'obs123456'}

  timeconst = loader.from_loadspec(request)

The resulting object looks like this::

  >>> print(timeconst)
  ResultSet<[timeconst], 1 rows>

Note the ``obs:obs_id`` column is gone -- it was taken as index
information, and matched against the ``obs:obs_id`` in the
``request``.

.. _metadata-archives-custom:

Custom Archive Formats
----------------------

HDF5 is cool but sometimes you need or want to use a different storage
system.  Setting up a custom loader function involves the following:

- A loader class that can read the metadata from that storage
  system, respecting the request API.
- A module, containg the loader class, and also the ability to
  register the loader class with sotodlib, under a particular laoder
  name.
- A ManifestDb data field called ``loader``, with the value set to the
  loader name.

Here's a sketchy example.  We start by defining a loader class, that
will read a number from a text file::

  from sotodlib.io import metadata
  from sotodlib.core.metadata import ResultSet, SuperLoader, LoaderInterface

  class TextLoader(LoaderInterface):
      def from_loadspec(self, load_params):
          with open(load_params['filename']) as fin:
              the_answer = float(fin.read())
          rs = ResultSet(keys=['answer'], [(the_answer, )])

  SuperLoader.register_metadata('text_loader', TextLoader)

Let's suppose that code (including the SuperLoader business) is in a
module called ``othertel.textloader``.  To get this code to run
whenever we're working with a certain dataset, add it to the
``imports`` list in the context.yaml:

.. code-block:: yaml

  # Standard i/o import, and TextLoader for othertel.
  imports:
    - sotodlib.io.metadata
    - othertel.textloader

Now for the ManifestDb::

  scheme = metadata.ManifestScheme()
  scheme.add_exact_match('obs:obs_id')
  scheme.add_data_field('loader')

  db = metadata.ManifestDb(scheme=scheme)
  db.add_entry({'obs:obs_id: 'obs12345',
                'loader': 'text_loader'},
               filename='obs12345_timeconst.txt')
  db.add_entry({'obs:obs_id: 'obs12600',
                'loader': 'text_loader'},
               filename='obs12600_timeconst.txt')

Now if a metadata request is made for ``obs12345``, for example, a
single number will be loaded from ``obs12345_timeconst.txt``.

Note the thing returned by ``TextLoader.from_loadspec`` is a
ResultSet.  Presently the only types you can return from a loader
class function are ResultSet and AxisManager.


.. _metadata-indexes:

Metadata Indexes
================

In the last example above, a ``request`` dictionary was passed to
ResultSetLoaderHdf, and provided instructions for locating a
particular result.  Such request dictionaries will normally be
generated by a ManifestDb object, which is connected to an sqlite3
database that provides a means for converting high-level requests for
metadata into specific request dictionaries.

The database behind a ManifestDb has 2 main tables.  One of them is a
table with columns for Index Criteria and Endpoint Data.  The Index
Criteria columns are intended to be matched against observation
metadata, such as the ``obs_id`` or the timestamp of the observation.
Endpoint Data contain a filename and other instructions required to
locate and load the data, as well as additional restrictions to put on
the result.

Please see the class documentation for :class:`ManifestDb` and
:class:`ManifestScheme`.  The remainder of this section demonstrates
some basic usage patterns.


Examples
--------

Example 1: Observation ID
`````````````````````````

The simplest Metadata index will translate an ``obs_id`` to a
particular dataset in an HDF file.  The ManifestScheme for this case
is constructed as follows::

  from sotodlib.core import metadata
  scheme = metadata.ManifestScheme()
  scheme.add_exact_match('obs:obs_id')
  scheme.add_data_field('dataset')

Then we can instantiate a ManifestDb using this scheme, add some data
rows, and write the database (including the scheme) to disk::

  db = metadata.ManifestDb(scheme=scheme)
  db.add_entry({'obs:obs_id': 'obs123456', 'dataset': 'timeconst_for_obs123456'},
               filename='test2.h5')
  db.add_entry({'obs:obs_id': 'obs123500', 'dataset': 'timeconst_for_obs123500'},
               filename='test2.h5')
  db.add_entry({'obs:obs_id': 'obs123611', 'dataset': 'timeconst_for_obs123611'},
               filename='test2.h5')
  db.add_entry({'obs:obs_id': 'obs123787', 'dataset': 'timeconst_for_obs123787'},
               filename='test2.h5')
  db.to_file('timeconst.gz')


Example 2: Inspecting and modifying the index
`````````````````````````````````````````````

Starting from the previous example, suppose we were updating the Index
in a cronjob and needed to first check whether we had already entered
some entry.  We can use :func:`ManifestDb.inspect` to retrieve
records, matching on *any* fields (they don't have to be index
fields).  Here's a quick set of examples::

  # Do we have any items in the file called "test2.h5"?
  entries = db.inspect({'filename': 'test2.h5'})
  if len(entries) > 0:
    # yes, we do ...

  # Have we already added the item for obs_id='obs123787'?
  entries = db.inspect({'obs:obs_id': 'obs123787'})
  if len(entries) == 0:
    # no, so add it ...

Entries retrieved using inspect are dicts and contain an '_id' element
that allows you to modify or delte those records from the Index, using
:func:`ManifestDb.update_entry` and :func:`ManifestDb.remove_entry`.
For example::

  # Delete all entries that refer to test2.h5.
  for entry in db.inspect({'filename': 'test2.h5'}):
    db.remove_entry(entry)

  # Change the spelling of 'timeconst' in the 'dataset' field of all records.
  for entry in db.inspect({}):
    entry['dataset'] = entry['dataset'].replace('timeconst', 'TimECOnSt')
    # currently it's not possible to change the filename, so don't mention it...
    del entry['filename']
    db.update_entry(entry)


Example 3: Timestamp
````````````````````

Another common use case is to map to a result based on an
observation's timestamp instead of obs_id.  The standardized key for
timestamp is ``obs:timestamp``, and we include it in the scheme with
:func:`add_range_match<ManifestScheme.add_range_match>` instead of
:func:`add_exact_match<ManifestScheme.add_exact_match>`::

  scheme = metadata.ManifestScheme()
  scheme.add_range_match('obs:obs_timestamp')
  scheme.add_data_field('dataset')

  db = metadata.ManifestDb(scheme=scheme)
  db.add_entry({'obs:timestamp': (123400, 123600),
                'dataset': 'timeconst_for_early_times'},
                filename='test2.h5')
  db.add_entry({'obs:timestamp': (123600, 123800),
                'dataset': 'timeconst_for_late_times'},
                filename='test2.h5')
  db.to_file('timeconst_by_timestamp.gz')

In the this case, when we add entries to the ManifestDb, we pass a
tuple of timestamps (lower inclusive limit, higher non-inclusive
limit) for the key ``obs:timestamp``.

Example 4: Other observation selectors
``````````````````````````````````````

Other fields from ObsDb can be used to build the Metadata Index.
While timestamp or obs_id are quite general, a more compact and direct
association can be made if ObsDb contains a field that is more
directly connected to the metadata.

For example, suppose there was an intermittent problem with a subset
of the detectors that requires us to discard those data from analysis.
The problem occurred randomly, but it could be identified and each
observation could be classified as either having that problem or not.
We decide to eliminate those bad detectors by applying a calibration
factor of 0 to the data.

We create an HDF5 file called ``bad_det_issue.h5`` with two datasets:

- ``cal_all_ok``: has columns ``dets:name`` (listing all detectors)
  and ``cal``, where ``cal`` is all 1s.
- ``cal_mask_bad``: same but with ``cal=0`` for the bad detectors.

We update the ObsDb we are using to include a column
``bad_det_issue``, and for each observation we set it to value 0 (if
the problem is not seen in that observation) or 1 (if it is).

We build the Metadata Index to select the right dataset from
``bad_det_issue.h5``, depending on the value of ``bad_det_issue`` in
the ObsDb::

  scheme = metadata.ManifestScheme()
  scheme.add_exact_match('obs:bad_det_issue')
  scheme.add_data_field('dataset')

  db = metadata.ManifestDb(scheme=scheme)
  db.add_entry({'obs:bad_det_issue': 0,
                'dataset': 'cal_all_ok'},
                filename='bad_det_issue.h5')
  db.add_entry({'obs:bad_det_issue': 1,
                'dataset': 'cal_mask_bad'},
                filename='bad_det_issue.h5')
  db.to_file('cal_bad_det_issue.gz')

The context.yaml metadata entry would probably look like this::

  metadata:
    ...
    - db: '{metadata_lib}/cal_bad_det_issue.gz'
      name: 'cal_remove_bad&cal'
    ...



ManifestDb reference
--------------------

*The class documentation of ManifestDb should appear below.*

.. autoclass:: ManifestDb
   :special-members: __init__
   :members:

ManifestScheme reference
------------------------

*The class documentation of ManifestScheme should appear below.*

.. autoclass:: ManifestScheme
   :special-members: __init__
   :members:


Metadata Request Processing
===========================

Metadata loading is triggered automatically when
:func:`Context.get_obs()<sotodlib.core.Context.get_obs>` (or
``get_meta``) is called.  The parameters to ``get_obs`` define an
observation of interest, through an ``obs_id``, as well as
(potentially) a limited set of detectors of interest.  Processing the
metadata request may require the code to refer to the ObsDb for more
information about the specified ``obs_id``, and to the DetDb or
``det_info`` dataset for more information about the detectors.

Steps in metadata request processing
------------------------------------

For each item in the context.yaml ``metadata`` entry, a series of
steps are performed:

1. Read ManifestDb.

   The ``db`` file specified in the ``metadata`` entry is loaded into
   memory.

2. Promote Request.

   The metadata request is likely to include information about what
   observation and what detectors are of interest.  But the ManifestDb
   may desire a slightly different form for this information.  For
   example, an ``obs_id`` might be provided in the request, but the
   ManifestDb might index its results using timestamps
   (``obs:timestamp``).  In the *Promote Request* step, the ManifestDb
   is interrogated for what ``obs`` and ``dets`` fields it requires as
   Index Data.  If those fields are not already in the request, then
   the request is augmented to include them; this typically requires
   interaction with the ObsDb and/or DetDb and det_info.  The
   augmented request is the result of the Promotion Step.

3. Get Endpoints.

   Once the augmented request is computed, the ManifestDb can be
   queried to see what endpoints match the request.  The ManifestDb
   will return a list of Endpoint results.  Each Endpoint result
   describes the location of some metadata (i.e. a filename and
   possibly some address within that file), as well as any limitations
   on the applicability of that data (e.g. it may specify that
   although the results include values for 90 GHz and 150 GHz
   detectors, only the results for 150 GHz detectors should be kept).
   The metadata results are not yet loaded.

4. Read Metadata.

   Each Endpoint result is processed and the relevant files accessed
   to load the specified data products.  The data within are trimmed
   to only include items that were actually requested by the Index
   data (for example, although results for 90 GHz and 150 GHz
   detectors are included in an HDF5 dataset, the data may be trimmed
   to only include the 150 GHz detector results).  This will yield one
   metadata result per Endpoint item.

5. Combine Metadata.

   The individual results from each Endpoint are combined into a
   single object, of the same type.

6. Wrap Metadata.

   The metadata object is converted to an AxisManager, and wrapped as
   specified by the user (this could include storing the entire object
   as a field; or it could mean extracting and renaming a single
   field from the result, for example).


Rules for augmenting and using det_info
---------------------------------------

The metadata loader associates metadata results to individual
detectors using fields from the observation's ``det_info``.  For any
single observation, the ``det_info`` is first initialized from:

- The ObsFileDb; this provides the unique ``readout_id`` for each
  channel, as well as the ``detset`` to which each belongs.
- If a DetDb has been specified, everything in there is copied into
  ``det_info``.

From that starting point, additional fields can be loaded into the
``det_info``; these fields can then be used to load metadata indexed
in a variety of ways.  For the mu-mux readout used in SO, the
following additional steps will usually be performed to augment the
``det_info``:

- A "channel map" (a.k.a. "det map", "det match", ...) result will be
  loaded, to associate a ``det_id`` with each of the loaded
  ``readout_id`` values (or most of them, hopefully).  While the
  ``readout_id`` describes a specific channel of the readout hardware,
  the ``det_id`` corresponds to a particular optical device (e.g. a
  detector with known position, orientation, and passband).
- Some "wafer info" will be loaded, containing various properties of
  the detectors, as designed (for example, their approximate
  locations; passbands; wiring information; etc.).  This is a single
  table of data for each physical wafer, indexed naturally by
  ``det_id``, but the rows here can not be associated with each
  ``readout_id`` until we have loaded the "channel map".

Special metadata entries in context.yaml are used to augment the
``det_info``.  table; these are marked with ``det_info: true`` and do
not have a ``name: ...`` field.  For example::

  metadata:
  - ...
  - db: 'more_det_info.sqlite'
    det_info: true
  - ...

It is expected that the database (``more_det_info.sqlite``) is a
standard ManifestDb, which will be queried in the usual way except
that when we get to the "Wrap Metadata" step, instead the following is
performed:

- All the loaded fields are inspected, and any fields that are already
  found in the current ``det_info`` are used as Index fields.
- The columns from the new metadata are merged into the active
  ``det_info``, ensuring that the index field values correspond.
- Only the rows for which the index field has the same value in the
  two objects are kept.

Here are a few more tips about det_info:

- *All* fields in the det_info metdata should be prefixed with
  ``dets:``, to signify that they are associated with the dets axis
  (this is similar to fields used as index fields in standard
  metadata.
- The field called ``dets:readout_id`` is assumed, throughout the
  Context/Metdata code, to correspond to the values in the ``.dets``
  axis of the TOD AxisManager.
- By convention, ``dets:det_id`` is used for the physical detector (or
  pseudo-detector device) identifier.  The special value "NO_MATCH" is
  used for cases where the ``det_id`` could not be determined.

When new det_info are merged in, any fields found in both the existing
and new det_info will be used to form the association.  Many-to-many
matching is fully supported, meaning that a unique index
(e.g. ``readout_id``) does not need to be used in the new det_info.
However, it is expected that in most cases either ``readout_id`` or
``det_id`` will be used to label det_info contributions.


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


.. note::

   DetDb is not planned for use in SO, because of the complexity of
   the ``readout_id`` to ``det_id`` mapping problem.  The ``det_info``
   system (described in the :ref:`metadata-section` section) is used
   for making detector information available in the loaded TOD object.
   DetDb is still used for simulations and for wrapping of data from
   other readout systems.


Using a DetDb (Tutorial)
========================

Loading a database into memory
------------------------------

To load an existing :py:obj:`DetDb<sotodlib.core.metadata.DetDb>` into memory, use
the :py:obj:`DetDb.from_file<sotodlib.core.metadata.DetDb.from_file>` class method::

  >>> from sotodlib.core import metadata
  >>> my_db = metadata.DetDb.from_file('path/to/database.sqlite')

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


Creating a DetDb
================

For an example, see source code of :func:`get_example<detdb.get_example>`.

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

.. autofunction:: sotodlib.core.metadata.detdb.get_example


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

.. _obsdb-names-section:
 
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

The ObsFileDb is an sqlite database.  It carries some information
about each "Observation" and the "detectors"; but is complementary to
the ObsDb and DetDb.


Data Model
==========

We assume the following organization of the time-ordered data:

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
   +-------------------------------------------------->
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

Constructing the ObsFileDb involves building the detsets and files
tables, using functions ``add_detset`` and ``add_obsfile``.  Using the
ObsFileDb is accomplished through the functions ``get_dets``,
``get_detsets``, ``get_obs``, and through custom SQL queries on
``conn``.


Relative vs. Absolute Paths
===========================

The filenames stored in ObsFileDb may be specified with relative or
absolute paths.  (Absolute paths are assumed if the filename starts
with a /.)  Relative paths are taken as being relative to the
directory where the ObsFileDb sqlite file lives; this can be
overridden by setting the ``prefix`` attribute.  Consider the
following file listing as an example::

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

Note the ``obsfiledb.sqlite`` file, located in ``/data/planet_obs``.
The filenames in obsfiledb.sqlite might be specified in one of two
ways:

  1. Using paths relative to the directory where ``obsfiledb.sqlite``
     is located.  For example, ``observation0/data0_1``.  Relative paths
     permit one to move the tree of data to other locations without
     needing to alter the ``obsfiledb.sqlite`` (as long as the
     relative locations of the data and sqlite file remain fixed).
  2. Using absolute paths on this file system; for example
     ``/data/planet_obs/observation0/data0_1``.  This is not portable,
     but it is a better choice if the ObsFileDb ``.sqlite`` file
     isn't kept near the TOD data files.

A database may contain a mixture of relative and absolute paths.


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


---------
ResultSet
---------

Auto-generated documentation should appear here.

.. autoclass:: ResultSet
   :special-members: __init__
   :members:

--------------------
sotodlib.io.metadata
--------------------

This module contains functions for working with ResultSet in HDF5.

Here's the docstring for ``write_dataset``:

.. autofunction:: sotodlib.io.metadata.write_dataset

Here's the docstring for ``read_dataset``:

.. autofunction:: sotodlib.io.metadata.read_dataset

Here's the class documentation for ResultSetHdfLoader:

.. autoclass:: sotodlib.io.metadata.ResultSetHdfLoader
   :inherited-members: __init__
   :members:

-----------------
Command Line Tool
-----------------


The `so-metadata` script is a tool that can be used to inspect and
alter the contents of ObsFileDb, ObsDb, and ManifestDb ("metadata")
sqlite3 databases.  In the case of ObsFileDb and ManifestDb, it can
also be used to perform batch filename updates.

To summarize a database, pass the db type and then the path to the db
file.  It might be convenient to start by printing a summary of the
context.yaml file, as this will give full paths to the various
databases, that can be copied and pasted::

  so-metadata context /path/to/context.yaml


Analyzing individual databases::

  so-metadata obsdb /path/to/context/obsdb.
  so-metadata obsfiledb /path/to/context/obsfiledb.sqlite
  so-metadata metadata /path/to/context/some_metadata.sqlite


Usage
=====

.. argparse::
   :module: sotodlib.core.metadata.cli
   :func: get_parser
   :prog: so-metadata



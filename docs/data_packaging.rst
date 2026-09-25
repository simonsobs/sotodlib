.. _data-packaging:

==============
Data Packaging
==============

"Data packaging" is the set of pipeline elements that turn the raw output of
the SMuRF systems and the OCS housekeeping aggregator (**Level 2** data) into
**Books** (**Level 3** data), ship those Books offsite via the Librarian, and
then delete the Level 2 copies once they are safely archived.

Six scripts do the work.  None of them run themselves on a timer: in
production each one is launched by `Prefect <https://docs.prefect.io>`_ — the
workflow automation service that runs SO's automated pipelines — as a
recurring *Prefect deployment*, one set per DAQ node (see
:ref:`dpkg-deployment`).  This page describes what each script does, the
databases and configuration they share, and what to do when one of them
fails.

.. contents::
   :local:
   :depth: 2


Overview
========

The pipeline is a conveyor belt.  Each script advances data one step and
records that step in a database, so the next run picks up where the last one
left off.  Nothing is ever deleted until a later stage has confirmed the data
survived the earlier ones.

.. code-block:: none

    Level 2 on disk                     Databases                  Scripts
    ---------------                     ---------                  -------

    <data_prefix>/hk/<tc>/       ---->  G3tHk (hk db)        <---- update_g3thk_database
      *.g3 from OCS aggregator

    <data_prefix>/smurf/<tc>/    ---->  G3tSmurf (lvl2 db)   <---- update_g3tsmurf_db
    <data_prefix>/timestreams/          files, observations,
      *.g3 from SMuRF                   timecodes, finalization

                                        Imprinter (book db)  <---- update_book_plan
                                        one row per Book,          (registers Books)
                                        status = UNBOUND

    output_root/...                <--- Imprinter            <---- make_book
      bound Books on staging            status = BOUND             (binds Books)

    Librarian (offsite)            <--- Imprinter            <---- update_librarian
                                        status = UPLOADED

    Level 2 deleted                <--- Imprinter            <---- cleanup_level2
    staging deleted                     status = DONE
                                        lvl2_deleted = True

The scripts, in the order they must run:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Script
     - Typical cadence
     - What it does
   * - :ref:`update_g3thk_database <dpkg-update-g3thk-database>`
     - ~hourly
     - Index new housekeeping ``.g3`` files into the G3tHk database.
   * - :ref:`update_g3tsmurf_db <dpkg-update-g3tsmurf-db>`
     - ~hourly, after the HK indexer
     - Index new SMuRF files, observations and timecodes into G3tSmurf, and
       advance the data-transfer finalization time.
   * - :ref:`update_book_plan <dpkg-update-book-plan>`
     - ~hourly, after the SMuRF indexer
     - Decide which Level 2 observations belong in which Book and register
       those Books (``UNBOUND``) in the Imprinter database.
   * - :ref:`make_book <dpkg-make-book>`
     - a few times a day
     - Bind registered Books: write the actual Level 3 files to the staging
       area (``BOUND``).
   * - :ref:`update_librarian <dpkg-update-librarian>`
     - a few times a day
     - Upload bound Books to the Librarian (``UPLOADED``).
   * - :ref:`cleanup_level2 <dpkg-cleanup-level2>`
     - daily
     - Verify a timecode is completely packaged, then delete the staged copy
       and the Level 2 originals (``DONE``, ``lvl2_deleted``).

The ordering matters, but it is enforced by status checks rather than by when
each script happens to be launched: ``make_book`` only touches ``UNBOUND``
Books, ``update_librarian`` only touches ``BOUND`` Books, and
``cleanup_level2`` refuses to delete anything that is not at least
``UPLOADED``.  Running them out of order is safe; it just means nothing
happens on that pass.  The Prefect deployments do put each element on a cron
schedule that follows the one it feeds, so that in normal running a given
hour's data reaches the Imprinter in that hour, but correctness depends on the
Book statuses rather than on the clock.


Concepts
========

.. _dpkg-timecodes:

Timecodes
---------

A **timecode** is the first five digits of a ctime, i.e. ``int(ctime // 1e5)``.
One timecode covers :math:`10^5` seconds, about 27.8 hours.  Level 2 data is
laid out in timecode directories under the ``data_prefix`` of the node::

    <data_prefix>/
        hk/<timecode>/
        smurf/<timecode>/<stream_id>/<action folders>
        timestreams/<timecode>/<stream_id>/*.g3

Timecodes are the unit of bookkeeping for everything that is not an
observation: ``hk``, ``smurf`` and ``stray`` Books are one-per-timecode, and
``cleanup_level2`` iterates over timecodes rather than over Books.


Books
-----

A **Book** is the Level 3 packaging unit: a directory (or a zip file, for
newer ``smurf`` Books) containing a self-consistent set of data plus its
index files.  Five types are produced:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Type
     - Contents
   * - ``obs``
     - Science observation.  Detector timestreams from every wafer that was
       streaming simultaneously, resampled onto a common set of samples, plus
       the co-sampled ancillary (pointing/HWP) data.
   * - ``oper``
     - A SMuRF operation (IV curve, bias steps, bias group map, noise
       measurement, ...).  Same file structure as ``obs``, plus a ``Z_smurf/``
       directory holding the raw sodetlib output files.
   * - ``hk``
     - All housekeeping ``.g3`` files for one timecode, copied verbatim.
   * - ``smurf``
     - All SMuRF metadata (the ``smurf/<timecode>`` action tree) for one
       timecode.  Files matching ``SMURF_EXCLUDE_PATTERNS`` (``*.dat``,
       ``*_mask.txt``, ``*_freq.txt``) are dropped.
   * - ``stray``
     - Any Level 2 ``.g3`` timestream file in the timecode that did **not**
       end up in an ``obs`` or ``oper`` Book.  This is the safety net that
       guarantees no raw file is deleted without having been archived
       somewhere.

``VALID_OBSTYPES`` also lists ``misc``, but nothing in the current pipeline
registers Books of that type.

Book IDs
````````

============  ==========================================================
Type          Book ID
============  ==========================================================
``obs``       ``obs_<timestamp>_<tel_tube>_<slot_flags>``
``oper``      ``oper_<timestamp>_<tel_tube>_<slot_flags>``
``hk``        ``hk_<timecode>_<daq_node>``
``smurf``     ``smurf_<timecode>_<daq_node>``
``stray``     ``stray_<timecode>_<daq_node>``
============  ==========================================================

For ``obs``/``oper``, ``<timestamp>`` is the ctime taken from the *first*
Level 2 ``obs_id`` in the set, and ``<slot_flags>`` is a bit string with one
character per ``wafer_slot`` configured for that tel tube — ``1`` if that
wafer contributed data to this Book, ``0`` otherwise.  So
``obs_1700000000_satp1_1110111`` is a seven-slot SAT observation missing
``ws3``.  The ID is built by :meth:`sotodlib.io.imprinter.ObsSet.get_id`.

Book layout on disk
```````````````````

Book paths are stored relative to ``output_root`` in the ``path`` column of
the Imprinter database, and are generated by
:meth:`sotodlib.io.imprinter.Imprinter.get_book_path`.  ``output_root`` is a
staging area: Books live there only until the Librarian has them and
``cleanup_level2`` removes the staged copy::

    output_root/
        <tel_tube>/
            obs/<first5>/obs_<timestamp>_<tel_tube>_<slot_flags>/
            oper/<first5>/oper_<timestamp>_<tel_tube>_<slot_flags>/
        <daq_node>/
            hk/hk_<timecode>_<daq_node>/
            smurf/smurf_<timecode>_<daq_node>[.zip]
            stray/stray_<timecode>_<daq_node>/

``<first5>`` is the timecode of the Book start.  ``smurf`` Books written with
schema version 1 or higher are single zip files rather than directories.

Inside an ``obs`` or ``oper`` Book:

.. code-block:: none

    D_<stream_id>_000.g3    detector data, one series of files per wafer
    A_ancil_000.g3          co-sampled ancillary data (az/el/boresight/HWP)
    M_index.yaml            per-book index: timing, sample ranges, detsets
    M_book.yaml             provenance: book type, schema version, sotodlib version
    Z_bookbinder_log.txt    binding log
    Z_smurf/                (oper books only) raw sodetlib outputs

The ``hk_fields`` entry in the Imprinter config determines which HK feeds are
pulled into ``A_ancil``; it is required and binding fails without it.

.. _dpkg-book-status:

Book status
-----------

Each Book row carries an integer ``status``.  These constants live in
:mod:`sotodlib.io.imprinter`:

.. list-table::
   :header-rows: 1
   :widths: 14 8 78

   * - Constant
     - Value
     - Meaning
   * - ``WONT_BIND``
     - ``-2``
     - Set by hand (or by autofix).  This Book will never be bound; its
       Level 2 ``.g3`` files fall through into the timecode's ``stray`` Book.
   * - ``FAILED``
     - ``-1``
     - Binding raised; the traceback is stored in the ``message`` column.
   * - ``UNBOUND``
     - ``0``
     - Registered, waiting for ``make_book``.
   * - ``REBIND``
     - ``1``
     - Defined but effectively unused —
       :func:`~sotodlib.io.imprinter_utils.set_book_rebind` resets Books to
       ``UNBOUND``, not to ``REBIND``.
   * - ``BOUND``
     - ``2``
     - Level 3 files written to the staging area and validated by
       :class:`~sotodlib.site_pipeline.check_book.BookScanner`.
   * - ``UPLOADED``
     - ``3``
     - Handed to the Librarian.
   * - ``DONE``
     - ``4``
     - Staged copy deleted.  (Level 2 deletion is tracked separately, by the
       ``lvl2_deleted`` boolean column.)

The normal path is ``UNBOUND -> BOUND -> UPLOADED -> DONE``.  ``FAILED`` is
the only state that requires a human or the autofixer; see
:ref:`dpkg-troubleshooting`.

.. _dpkg-finalization:

Finalization time
-----------------

The single most important safety mechanism in the pipeline is the
**finalization time**: the ctime before which we are confident that *all*
Level 2 data has arrived on the DAQ node and been indexed.  Books are never
planned past it.

It is computed by :meth:`sotodlib.io.load_smurf.G3tSmurf.get_final_time` as
the minimum over:

* ``G3tSmurf.last_update`` — how far ``update_g3tsmurf_db`` has indexed;
* the ``finalized_until`` HK field published by each relevant
  ``smurf-suprsync`` and ``timestream-suprsync`` agent, i.e. how much data
  each SMuRF server has successfully transferred;
* the last update time of the HK database itself.

With ``check_control=True`` the ``pysmurf-monitor`` HK feeds are used to work
out which servers were actually in control of which ``stream_id`` over the
window, so that a server that was switched off does not hold back the
finalization time for wafers it was not driving.

This is why ``update_g3thk_database`` must run before ``update_g3tsmurf_db``,
and both must run before ``update_book_plan``: the transfer status is itself
housekeeping data.

Related timecode bookkeeping lives in the G3tSmurf ``TimeCodes`` table, which
records a row when a suprsync agent finishes transferring a timecode's
``FILES`` or ``META``.  ``smurf`` Books are registered once all ``META``
entries exist; ``stray`` Books once all ``FILES`` entries exist *and* every
``obs``/``oper`` Book in the timecode is bound.


Databases
=========

Four independent stores are involved.  All of them are SQLite by default and
all of them are indexes over data that lives on a filesystem — none of them
hold bulk data.

G3tSmurf
    Index of Level 2 SMuRF data: files, frames, observations, tunes, channel
    assignments, timecodes and finalization state.  Written by
    ``update_g3tsmurf_db``.  See :doc:`g3tsmurf`.

G3tHk
    Index of Level 2 housekeeping files, the agents in them and their fields.
    Written by ``update_g3thk_database``.  G3tSmurf reads it to compute the
    finalization time.  (This is distinct from :mod:`sotodlib.io.hkdb`, which
    indexes *Level 3* housekeeping for analysis — see :doc:`hkdb`.)

Imprinter ("book db")
    One row per Book, plus a join table mapping Level 2 ``obs_id`` to Book.
    Written by ``update_book_plan``, ``make_book``, ``update_librarian`` and
    ``cleanup_level2``.  Schema:
    :class:`~sotodlib.io.imprinter.Books` and
    :class:`~sotodlib.io.imprinter.Observations`.

Librarian
    External service (``hera_librarian``) that holds the offsite copies.  The
    Imprinter talks to it through
    :meth:`~sotodlib.io.imprinter.Imprinter.upload_book_to_librarian` and
    :meth:`~sotodlib.io.imprinter.Imprinter.check_book_in_librarian`.

There is exactly one Imprinter instance and one G3tSmurf instance per DAQ
node, in one-to-one correspondence.  A housekeeping-only node — a site-wide HK
aggregator, say — is the exception: it sets ``build_det: False`` and its
G3tSmurf config file defines only ``data_prefix`` and ``g3thk_db``, so it has
a G3tHk database, no G3tSmurf database, and no reason to run
``update_g3tsmurf_db``.


.. _dpkg-configuration:

Configuration
=============

The environment file
--------------------

All data packaging config files are loaded through
:func:`sotodlib.io.datapkg_utils.load_configs`, which does ``{tag}``
substitution using a YAML "environment" file pointed at by the
``DATAPKG_ENV`` environment variable.  This is what lets one set of config
files work in several places — inside a pipeline worker, on the DAQ node
itself, on an analysis cluster, on a laptop — with only the environment file
changing.

.. code-block:: yaml

    # $DATAPKG_ENV
    lvl2_data:  /path/to/level2        # {lvl2_data}/<platform>/{hk,smurf,timestreams}
    lvl2_smurf: /path/to/databases     # g3tsmurf.db and imprinter.db
    lvl2_hkdb:  /path/to/databases     # g3hk.db
    configs:    /path/to/site-pipeline-configs
    staged:     /path/to/staging       # output_root for bound Books

A config file then refers to those tags::

    g3tsmurf_db: "{lvl2_smurf}/satp1/g3tsmurf.db"

The tag names are arbitrary — the ones above are the names the SO configs
happen to use.  Any tag a config uses simply has to be defined in the
environment file, and if it is not, ``load_configs`` raises a ``ValueError``
naming the missing tag.  Note that the paths in an environment file must make
sense *in the environment where the element runs*: if the pipeline runs in a
container, they are the paths as seen inside that container, and a copy of the
configs used from a shell outside it needs its own environment file.

One tag is special: ``DATAPKG_ENV`` must define ``configs`` for
:meth:`Imprinter.for_platform <sotodlib.io.imprinter.Imprinter.for_platform>`
(and therefore for ``cleanup_level2``, ``imprinter-cli``, and ``make_book
--n-proc`` > 1) to work, because
:func:`~sotodlib.io.datapkg_utils.get_imprinter_config` resolves the config
path as ``<configs>/<platform>/imprinter.yaml``.  This is why every platform's
config directory has to use exactly that filename.

G3tSmurf configuration
----------------------

Shared by ``update_g3tsmurf_db``, ``update_g3thk_database`` and (indirectly,
via the ``g3tsmurf`` key of the Imprinter config) everything else:

.. code-block:: yaml

    data_prefix: "{lvl2_data}/<platform>"     # contains hk/, smurf/, timestreams/
    g3tsmurf_db: "{lvl2_smurf}/<platform>/g3tsmurf.db"
    g3thk_db:    "{lvl2_hkdb}/<platform>/g3hk.db"

    finalization:
      servers:
      - pysmurf-monitor: <monitor instance-id>       # OCS instance-ids
        smurf-suprsync: <smurf sync instance-id>
        timestream-suprsync: <timestream sync instance-id>
      - pysmurf-monitor: ...

One ``servers`` entry per SMuRF server feeding the node, so a couple of
entries for a SAT and several for the LAT.  The ``finalization`` block and
``g3thk_db`` are what make :ref:`finalization <dpkg-finalization>` work;
without them the pipeline cannot tell whether Level 2 data transfer is
complete.  ``G3tHk`` derives its archive path as ``<data_prefix>/hk`` and
takes its list of instance-ids from the same ``finalization`` block.

A housekeeping-only node omits both the SMuRF database and the finalization
block, leaving only ``data_prefix`` and ``g3thk_db``.

Imprinter configuration
-----------------------

.. code-block:: yaml

    db_path: "{lvl2_smurf}/<platform>/imprinter.db"   # the book database
    daq_node: satp1                          # names hk/smurf/stray books & dirs
    g3tsmurf: "{configs}/<platform>/g3tsmurf_config.yaml"
    output_root: "{staged}"                  # where bound books are written
    librarian_conn: <connection name>        # key in the hera_librarian client settings
    build_hk: True                           # register/bind hk books?
    build_det: True                          # register/bind obs/oper/smurf/stray books?
    require_hwp: True                        # default for bind_book

    tel_tubes:
      satp1:
        tube_slot: st1
        tube_flavor: mf
        wafer_slots:
          - wafer_slot: ws0
            stream_id: ufm_mv6
            wafer_flavor: mf
          - wafer_slot: ws1
            stream_id: ufm_mv9
            wafer_flavor: mf
          - wafer_slot: ws2
            stream_id: None      # empty slot

    # HK feeds co-sampled into A_ancil; required for binding obs/oper books
    hk_fields:
      az: acu.acu_udp_stream.Corrected_Azimuth
      el: acu.acu_udp_stream.Corrected_Elevation
      boresight: acu.acu_udp_stream.Corrected_Boresight
      az_mode: acu.acu_status.Azimuth_mode
      hwp_freq: hwp-bbb-e1.HWPEncoder.approx_hwp_freq

    # optional: InfluxDB reporting from update_book_plan --use-monitor
    monitor:
      connect_configs: "{configs}/shared/monitor_configs.yaml"
      telescope: "<platform>"
      measurement: "book_counts"

Notes on ``tel_tubes``:

* Each tel tube becomes its own set of ``obs``/``oper`` Books and its own
  output directory.  SATs have one entry; the LAT has one per optics tube.
* ``wafer_slots`` must be listed in order (``ws0``, ``ws1``, ...); the
  Imprinter asserts this at startup, because slot position determines the
  ``slot_flags`` field of every Book ID.
* ``stream_id: None`` marks a physically empty slot.  It still occupies a
  position in ``slot_flags``.
* Every ``stream_id`` listed across all tubes is what the pipeline considers
  "ours".  Data from any other ``stream_id`` will never be put into an
  ``obs``/``oper`` Book and will end up in ``stray``.
* Duplicate ``stream_id`` entries require an extra ``wafer`` key to
  disambiguate, and are only allowed for LF tubes.
* ``tube_flavor`` and the per-slot ``wafer_flavor`` are not used for planning
  or binding decisions; they are copied through into the Book's
  ``M_index.yaml`` (``tube_flavor``, ``wafer_slots``) and are read downstream,
  e.g. by ``update_obsdb`` and the mapmaker's observation grouping.

``require_hwp`` defaults to ``True``, and is a per-node choice: a node whose
telescope has no HWP, or that regularly observes with the HWP stopped or
spinning down, sets it ``False`` so those observations still bind.  Setting it
``False`` does not discard HWP data when there is any; it only stops
``NoHWPData`` from being a binding failure.

``build_hk`` and ``build_det`` let one node handle only housekeeping or only
detector data.  A housekeeping-only node sets ``build_det: False`` and needs
no ``tel_tubes``, ``hk_fields`` or ``require_hwp``, since it only ever
produces ``hk`` Books.

``librarian_conn`` is not a hostname: it names an entry in
``client_settings.connections`` from the ``hera_librarian`` client
configuration (``~/.hl_client.cfg``), which must be readable by the account
running the pipeline.


.. _dpkg-deployment:

Deployment
==========

None of these elements run themselves on a timer, and none of them are meant
to be run by hand in production.  At the SO site they are driven by
`Prefect <https://docs.prefect.io>`_: each element is registered as a Prefect
*deployment* with a cron schedule, a Prefect *server* holds those schedules
and the run history, and a Prefect *worker* picks up the due runs and executes
them.  The worker image, the flow wrappers, and the per-site deployment
definitions live outside sotodlib, in the ``so-workflow`` and
``site-pipeline-configs`` repositories.

.. note::

    "Schedule" on this page always means a Prefect cron schedule for a
    pipeline element.  It has nothing to do with the observing schedules
    produced by the SO scheduler; data packaging never reads those.

.. note::

    Site-specific Prefect details — hosts, accounts, filesystem layout, cron
    schedules and the parameter values in use — are documented in the
    ``so-workflow`` repository (``docs/data-packaging.md``), which is where
    people who operate the pipeline should look.  This page documents the
    behaviour of the elements themselves, which stays true however they are
    launched.

The interface sotodlib exposes to Prefect
-----------------------------------------

The Prefect wrappers do not shell out to ``so-site-pipeline``.  They import
each module and wrap its ``main()`` function as a Prefect flow, roughly:

.. code-block:: python

    mod = importlib.import_module(f"sotodlib.site_pipeline.{modname}")
    flow(name=modname, log_prints=True)(mod.main)

so **the** ``main()`` **signatures on this page are the Prefect deployment
API**, and several things follow from that:

* Prefect deployment parameters are keyword arguments to ``main()``, not
  command line flags: ``update_delay=2``, with an underscore, and a real
  Python value.
* Argparse defaults do not apply.  Where a module's ``main()`` and its
  ``get_parser()`` disagree — see the note under
  :ref:`cleanup_level2 <dpkg-cleanup-level2>` — the ``main()`` default is what
  a Prefect deployment gets.  Changing either one without the other is a way to
  silently change production behaviour.
* Errors propagate.  Every element here is written to raise rather than exit
  quietly, so that a failure becomes a ``Failed`` Prefect flow run rather than
  a silent no-op; ``make_book`` additionally posts to the webhooks given in
  its ``alert_webhook`` argument.
* ``DATAPKG_ENV`` must be set in the environment the Prefect flow runs in,
  since
  ``cleanup_level2``, ``imprinter-cli`` and parallel ``make_book`` workers
  build their ``Imprinter`` with
  :meth:`~sotodlib.io.imprinter.Imprinter.for_platform`.

Constraints on the Prefect schedules
------------------------------------

However the cron schedules are set, they have to respect the dependencies
described in the :ref:`overview <data-packaging>`:

* ``update_g3thk_database`` before ``update_g3tsmurf_db`` before
  ``update_book_plan``, since the finalization time is computed from
  housekeeping data.
* ``update_delay`` (and ``update_delay_timecodes``) should comfortably exceed
  the interval between Prefect runs, so that a missed run is repaired by the
  next one rather than leaving a gap.
* ``make_book`` is the element that does real I/O; it is usually run less
  often than the indexers.
* ``cleanup_level2`` is long-running and deletes data.  Give it
  ``max_runtime``, keep its completion range wider than its deletion ranges
  (the script enforces this), and expect one instance at a time per node.

Pipeline elements
=================

.. _dpkg-update-g3thk-database:

update_g3thk_database
---------------------

Indexes Level 2 housekeeping ``.g3`` files into the G3tHk database: one row
per file, plus rows for each OCS agent found in the file and each field that
agent published, with start/stop times.

By default the script picks up where it left off, starting 10 seconds before
the last file already in the database.  ``--from-scratch`` (or an empty
database) restarts from ctime ``1.6e9``, which takes a while.

This must run *before* ``update_g3tsmurf_db``, because the suprsync
``finalized_until`` fields it indexes are what set the G3tSmurf finalization
time.

It reads the same G3tSmurf config file as ``update_g3tsmurf_db``, of which
only the ``data_prefix`` and ``g3thk_db`` keys are used, and its Prefect
deployment normally fires a little ahead of that one.

.. code-block:: bash

    python -m sotodlib.site_pipeline.update_g3thk_database \
        /path/to/g3tsmurf_config.yaml

.. argparse::
    :module: sotodlib.site_pipeline.update_g3thk_database
    :func: get_parser
    :prog: update_g3thk_database

.. _dpkg-update-g3tsmurf-db:

update_g3tsmurf_db
------------------

Indexes Level 2 SMuRF data and advances the finalization time.  In order, it:

#. indexes SMuRF metadata (action folders, tunes, channel assignments,
   bias group maps) — ``index_metadata``;
#. indexes ``.g3`` timestream files and the frames in them, grouping them
   into Level 2 observations — ``index_archive``;
#. optionally reconstructs observations from action folders
   (``--index-via-actions``) — needed for data older than October 2022, but
   it races against automatic Level 2 deletion, so it is off at the site;
#. indexes suprsync timecode completion rows — ``index_timecodes``;
#. recomputes the finalization time — ``update_finalization``;
#. revisits recent observations that have no stop time or no tuneset and
   tries to complete them.

Finally it raises if it finds completed observations with bad timing or with
no tuneset (i.e. no readout IDs), since those cannot be bound into Books.
Observations that have been inspected by a human and deliberately accepted
can be listed one-per-line in a file passed as ``--checked-file`` to suppress
the error.

The time range searched defaults to the last ``--update-delay`` days.
``--min_ctime`` must not be later than the current finalization time, or the
script refuses to run — otherwise a gap would be silently left in the index.

Prefect deployments normally set ``update_delay`` to a couple of days, so
that each run re-walks recent data and a single missed run costs nothing, and
point ``checked_file`` at a persistent file so that manually accepted
observations stay accepted.  ``index_via_actions`` is left off wherever automatic Level 2
deletion is running.

This is the one data-packaging element registered with the
``so-site-pipeline`` wrapper:

.. code-block:: bash

    so-site-pipeline update-g3tsmurf-db /path/to/g3tsmurf_config.yaml

The user running it needs read, write and execute permission on the database
file.

.. argparse::
    :module: sotodlib.site_pipeline.update_g3tsmurf_db
    :func: get_parser
    :prog: update-g3tsmurf-db

.. _dpkg-update-book-plan:

update_book_plan
----------------

Reads the G3tSmurf database and decides which Level 2 observations belong
together in which Book, then registers those Books as ``UNBOUND``.  It writes
no data.

Three registration passes run per invocation:

``obs``/``oper`` Books
    :meth:`~sotodlib.io.imprinter.Imprinter.update_bookdb_from_g3tsmurf`, once
    per tel tube.  For each Level 2 observation in the window it looks for
    observations on *other* configured ``stream_id``\ s that overlap it by at
    least ``min_overlap`` (30 s).  Mutually overlapping observations are
    grouped into one multi-wafer Book.  Observations tagged as operations
    always get their own ``oper`` Book.  Observations with low-precision
    timing are split back out into single-wafer Books, since they cannot be
    put on a common sample grid with anything else.

``hk`` Books
    :meth:`~sotodlib.io.imprinter.Imprinter.register_hk_books`, one per
    timecode directory under ``<data_prefix>/hk``, excluding the most recent
    one (which may still be growing).

``smurf`` and ``stray`` Books
    :meth:`~sotodlib.io.imprinter.Imprinter.register_timecode_books`, driven
    by the suprsync ``TimeCodes`` rows as described in
    :ref:`dpkg-finalization`.

The two time windows are controlled separately: ``--update-delay`` (default
1 day) for ``obs``/``oper``, and ``--update-delay-timecodes`` (default 7
days) for the timecode Books, because a timecode can only be closed out well
after the fact.

Staleness and stuck-stream handling is governed by ``--delay_warning``
(default 3 h) and ``--delay_error`` (default 6 h).  If the finalization time
falls behind "now" by more than these, the script warns and then raises — a
stale G3tSmurf or HK database is treated as a hard error, because continuing
would plan Books over incomplete data.  The same thresholds apply to
individual observations with no stop time: after ``delay_warning`` the script
checks whether the same slot has started streaming again and, if so, forces
the earlier observation closed; after ``delay_error`` it raises.

Errors are collected per tel tube (see the ``loop_over_tubes`` decorator) so
that one broken tube does not stop the others, and re-raised together at the
end.

Its Prefect deployment is cron'd to follow that node's
``update_g3tsmurf_db``, with an ``update_delay`` larger than the argparse
default so a missed run is repaired by the next one, and with
``--use-monitor`` where InfluxDB reporting is wanted.  A node whose Level 2 transfers arrive in bursts may need
``delay_error`` raised above the 6 h default to avoid spurious failures.

.. code-block:: bash

    python -m sotodlib.site_pipeline.update_book_plan \
        --config /path/to/imprinter.yaml

.. argparse::
    :module: sotodlib.site_pipeline.update_book_plan
    :func: get_parser
    :prog: update_book_plan

.. note::

    ``update_book_plan`` takes ``--config`` as an *option*, while
    ``make_book`` and ``update_g3tsmurf_db`` take the config as a
    *positional* argument and ``cleanup_level2`` takes a platform name
    instead.  This inconsistency is historical, and it only affects command
    line use: the Prefect deployments call ``main()`` directly, so every
    element takes a ``config:`` parameter (or ``platform:``) in the same
    way.

.. _dpkg-make-book:

make_book
---------

The only element that writes Level 3 data.  It fetches all ``UNBOUND`` Books
and binds them, then makes one retry pass over ``FAILED`` Books.

Binding is delegated to :mod:`sotodlib.io.bookbinder`:

* ``obs``/``oper`` Books go through
  :class:`~sotodlib.io.bookbinder.BookBinder`, which reads the Level 2 frames,
  puts every wafer on a common sample grid, fills small gaps, co-samples the
  HK fields listed in ``hk_fields``, and writes ``D_*.g3`` / ``A_ancil_*.g3``.
  For ``oper`` Books it also copies the sodetlib output files into
  ``Z_smurf/``.
* ``hk``, ``smurf`` and ``stray`` Books go through
  :class:`~sotodlib.io.bookbinder.TimeCodeBinder`, which copies files
  verbatim (optionally into a zip, for ``smurf`` schema >= 1).

After binding, ``M_book.yaml`` and ``M_index.yaml`` are written, and
``obs``/``oper`` Books are re-read by
:class:`~sotodlib.site_pipeline.check_book.BookScanner` to confirm they are
internally consistent.  Only then is the status set to ``BOUND``.

Pre-binding sanity checks that raise rather than produce a bad Book:

* ``obs`` Books shorter than 60 s (``ObsBookTooShort``) — these are set
  ``WONT_BIND`` by the autofixer so their files land in ``stray``;
* individual Level 2 files larger than 10 GB for ``obs`` or 5 GB for ``oper``
  (``FileTooLargeError``), which indicates a runaway full-rate stream;
* missing readout IDs (``MissingReadoutIDError``).

The retry pass is deliberately shallow: a Book that was already ``FAILED``
when the script started is not retried, so nothing fails twice in one run.
Books that fail their retry trigger an alert to ``--alert-webhook``.

``--n-proc`` > 1 binds Books in a ``ProcessPoolExecutor``; each worker builds
its own ``Imprinter`` via
:meth:`~sotodlib.io.imprinter.Imprinter.for_platform`, so ``DATAPKG_ENV``
must be set for parallel runs.

This is the element that does the heavy I/O, so its Prefect deployment
normally runs on a longer cadence than the indexers rather than following
every ``update_book_plan``.  ``--alert-webhook`` takes one or more Slack webhooks;
deployments generally give it both a shared operations channel and a
per-platform one.

.. code-block:: bash

    python -m sotodlib.site_pipeline.make_book /path/to/imprinter.yaml

.. argparse::
    :module: sotodlib.site_pipeline.make_book
    :func: get_parser
    :prog: make_book

.. _dpkg-update-librarian:

update_librarian
----------------

Uploads every ``BOUND`` Book to the Librarian and sets it ``UPLOADED``.  The
Librarian connection is named by ``librarian_conn`` in the Imprinter config
and resolved against the ``hera_librarian`` client settings.

The loop stops after 5 failed uploads so a Librarian outage does not produce
thousands of error lines, and then re-raises the first failure, which exits
non-zero and marks the Prefect flow run ``Failed``.  ``config`` is its only
argument.

.. code-block:: bash

    python -m sotodlib.site_pipeline.update_librarian \
        --config /path/to/imprinter.yaml

.. argparse::
    :module: sotodlib.site_pipeline.update_librarian
    :func: get_parser
    :prog: update_librarian

.. _dpkg-cleanup-level2:

cleanup_level2
--------------

The deletion stage, and the only element addressed by *platform* rather than
by config path — it uses ``DATAPKG_ENV`` to find
``<configs>/<platform>/imprinter.yaml``.

It walks timecodes in order and, for each one, runs up to three operations
built on :class:`sotodlib.io.datapkg_completion.DataPackaging`:

1. **Completion check** (always).
   :meth:`~sotodlib.io.datapkg_completion.DataPackaging.make_timecode_complete`
   is an aggressive self-repair pass: re-index any files on disk missing from
   G3tSmurf, force-complete dangling observations, register any Level 2
   observations that never made it into a Book (falling back to single-wafer
   registration), bind anything still ``UNBOUND``, run the autofixer over
   anything ``FAILED``, force the timecode final if a server was off, and
   register the ``hk``/``smurf``/``stray`` Books.  Then
   :meth:`~sotodlib.io.datapkg_completion.DataPackaging.verify_timecode_deletable`
   compares the list of files on disk against the list of files the Book
   database claims to have archived, and fails if anything on disk is
   unaccounted for.

2. **Staged deletion** (``--delete-staged``).  Removes the bound copy from
   ``output_root``; the Book is now only in the Librarian.  Sets ``DONE``.

3. **Level 2 deletion** (``--delete-lvl2``).  Removes the raw Level 2 files,
   after asking the Librarian to confirm two independent copies exist
   (``check_book_in_librarian(n_copies=2)``).  Sets ``lvl2_deleted``.

Each range is set either by a lag in days before "now"
(``--completion-lag``, default 14; ``--staged-deletion-lag``, default 28;
``--lvl2-deletion-lag``, default 28) or by explicit
``--min-*``/``--max-*`` timecode overrides.  The completion-check
range must fully contain both deletion ranges; the script validates this at
startup and refuses to run otherwise, so it is impossible to delete a
timecode that was never checked.

``--dry-run`` logs the plan without touching anything.  ``--max-runtime``
(minutes) stops cleanly at the next timecode boundary, which matters because
a from-scratch completion pass over years of timecodes can run for a very
long time; accumulated failures are still raised.

The three lags are what set the site's data retention policy: how long after
acquisition a timecode is required to be fully packaged, when its staged copy
goes, and when the raw Level 2 data goes.  Deployments set all three
explicitly.  When working through a backlog, the ``--min-complete-timecode`` /
``--min-staged-delete-timecode`` / ``--min-lvl2-delete-timecode`` overrides
pin the bottom of each range so a run does not spend its whole
``--max-runtime`` re-checking old, known-good timecodes.

.. note::

    ``main()`` defaults ``staged_deletion_lag`` to 14 while the command line
    defaults ``--staged-deletion-lag`` to 28, so calling ``main()`` directly
    from Python — as the Prefect flow wrapper does; see
    :ref:`dpkg-deployment` — is not equivalent to running the script with no
    arguments.  Pass every lag
    explicitly and the ambiguity goes away.

.. code-block:: bash

    # see what would happen
    python -m sotodlib.site_pipeline.cleanup_level2 satp1 --dry-run

    # the real daily run
    python -m sotodlib.site_pipeline.cleanup_level2 satp1 \
        --delete-staged --delete-lvl2 --max-runtime 120

.. argparse::
    :module: sotodlib.site_pipeline.cleanup_level2
    :func: get_parser
    :prog: cleanup_level2

.. warning::

    ``cleanup_level2`` deletes raw data.  It is heavily guarded — status
    checks, file-list reconciliation, and a Librarian two-copy check — but
    always run with ``--dry-run`` first when changing the timecode ranges by
    hand.


.. _dpkg-troubleshooting:

Operations and troubleshooting
==============================

Before you run anything by hand
-------------------------------

Two things have to be true before any of the commands below will do what you
expect:

* ``DATAPKG_ENV`` points at an environment file whose tags resolve correctly
  *from where you are running*.  If the pipeline itself runs in a container,
  its environment file describes container paths and will not work from a
  shell outside it.
* You are running as the account that owns the databases, ``output_root`` and
  the Level 2 files — the same account the Prefect worker runs as.  Running as
  anyone else is the usual cause of locked or read-only database errors, and
  can leave behind files the pipeline is then unable to clean up.

Re-running a whole element is usually better done from the Prefect UI, by
triggering its deployment, than by hand; see :ref:`dpkg-deployment`.

Inspecting the book database
----------------------------

.. code-block:: python

    from sotodlib.io.imprinter import Imprinter

    imprint = Imprinter.for_platform('satp1')   # needs DATAPKG_ENV

    imprint.get_unbound_books()
    imprint.get_failed_books()
    imprint.get_bound_books()
    imprint.get_uploaded_books()

    book = imprint.get_book('obs_1700000000_satp1_1111111')
    print(book.status, book.message, book.path)

imprinter-cli
-------------

:mod:`sotodlib.io.imprinter_cli` is the interactive tool for dealing with
``FAILED`` Books.  It needs ``DATAPKG_ENV`` set and an account with write
access to both the database and ``output_root``::

    python -m sotodlib.io.imprinter_cli <platform> report
    python -m sotodlib.io.imprinter_cli <platform> autofix
    python -m sotodlib.io.imprinter_cli <platform> failed

``report``
    Classify every failed Book against the known error types and print a
    one-line summary each.  Read-only.

``autofix``
    Same classification, but apply the known fix.  A Book whose fix itself
    fails is marked ``SECOND-FAIL`` in its message and skipped on subsequent
    runs, so autofix converges rather than looping.  This is also called
    automatically from ``cleanup_level2``.

``failed``
    Walk failed Books one at a time and prompt for an action: retry, retry
    with Level 2 updates, rebind with specific flags, permanently skip
    (``WONT_BIND``), or skip and delete Level 2.

Known failure modes
-------------------

The error classes registered in ``AUTOFIX_ERRORS``, in the order they are
matched:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Error in ``book.message``
     - Automated response
   * - ``SECOND-FAIL``
     - None.  Autofix already tried and failed; needs a human.
   * - ``BookDirHasFiles``
     - Stale files in the staging directory from an interrupted run.  Clear
       and rebind.
   * - ``MissingReadoutIDError``
     - SMuRF was not set up for science observations, or G3tSmurf indexing is
       wrong.  Drop the offending Level 2 observation from the Book and
       re-register; if every observation is affected, ``WONT_BIND``.
   * - ``ObsBookTooShort``
     - ``obs`` Book under 60 s.  ``WONT_BIND``; the files go to ``stray``.
   * - ``NoScanFrames``
     - No detector data.  ``WONT_BIND`` for ``oper`` or short Books,
       otherwise drop the empty observations.
   * - ``NoHWPData``
     - Rebind with ``require_hwp=False``.
   * - ``DuplicateAncillaryData``
     - An OCS aggregator glitch.  Rebind with ``ancil_drop_duplicates=True``.
   * - ``NoMountData``
     - ACU not reading out.  ``oper`` Books rebind with ``require_acu=False``;
       ``obs`` Books cannot be autofixed.
   * - ``DroppedMountData``
     - ACU dropouts.  Rebound with ``require_acu=False`` if the total dropout
       is under 300 s.
   * - ``TimingSystemOff``
     - Low-precision timing: rebind with ``allow_bad_timing=True``.  If the
       message says the timing counters are not incrementing, the Level 2
       metadata is wrong — ``oper`` Books are re-indexed and rebound, ``obs``
       Books need manual intervention.
   * - ``FileTooLargeError``
     - Runaway full-rate stream.  Delete the Level 2 observation and the
       Book.
   * - ``BadTimeSamples``
     - Dropped samples.  Rebind allowing bad timing if under 10 000 samples
       per observation; otherwise re-index Level 2 and, failing that, drop
       the bad observations or ``WONT_BIND``.
   * - ``NonMonotonicAncillaryTimes``
     - Rebind with ``require_monotonic_times=False``, but only for
       whitelisted fields and small sample counts.
   * - ``TimingCounterError``
     - Bad counter statistics.  Re-index Level 2, split the Book into
       single-wafer Books, rebind allowing bad timing.

Two errors come from ``update_book_plan`` rather than binding, and are not
autofixable:

``OverlapObsError``
    A Level 2 observation could legitimately belong to more than one Book, so
    the planner refuses to guess.  Use
    :func:`~sotodlib.io.imprinter_utils.find_overlaps` to list the candidate
    ``ObsSet``\ s and register the right one by hand::

        import sotodlib.io.imprinter_utils as utils
        rsets = utils.find_overlaps(imprint, 'obs_ufm_mv9_1714406208',
                                    min_ctime, max_ctime)
        imprint.register_book(rsets[0], commit=True)

"G3tSMURF + HK databases are stale"
    Not a Book problem: ``update_g3thk_database`` or ``update_g3tsmurf_db``
    has stopped running, or suprsync has stopped transferring.  Fix the
    upstream job; ``update_book_plan`` will catch up on its own.

Manual interventions
--------------------

:mod:`sotodlib.io.imprinter_utils` holds the operations that are deliberately
*not* part of the automated pipeline.  All of them expect an ``Imprinter``
instance:

.. code-block:: python

    import sotodlib.io.imprinter_utils as utils

    # clear staged files and set a book back to UNBOUND
    utils.set_book_rebind(imprint, book, update_level2=False)

    # never bind this book; its .g3 files will go to stray instead
    utils.set_book_wont_bind(imprint, book, message="why")

    # split a multi-wafer book into single-wafer books
    utils.split_book_by_obs(imprint, book)

    # drop one bad level 2 observation and re-register the rest
    utils.remove_level2_obs_from_book(imprint, book, bad_obs_id)

    # force a timecode final when a suprsync agent never reported
    utils.set_timecode_final(imprint, timecode)

There are also block operations that sweep every failed Book with a given
error (:func:`~sotodlib.io.imprinter_utils.block_set_rebind`,
:func:`~sotodlib.io.imprinter_utils.block_fix_bad_timing`,
:func:`~sotodlib.io.imprinter_utils.block_fix_duplicate_timestamps`) and two
diagnostic reports that print where in an observation the timestamps or the
ACU data went wrong
(:func:`~sotodlib.io.imprinter_utils.report_smurf_timestamp_error`,
:func:`~sotodlib.io.imprinter_utils.report_acu_timestamp_error`).

.. warning::

    ``set_book_wont_bind`` and ``set_book_rebind`` delete the Book's staged
    directory.  ``delete_level2_obs_and_book`` deletes raw data.  None of
    these ask for confirmation.

Checking a timecode by hand
---------------------------

To investigate one timecode without running the full cleanup script:

.. code-block:: python

    from sotodlib.io.datapkg_completion import DataPackaging

    dpk = DataPackaging('satp1')

    ok, msg = dpk.make_timecode_complete(170000, try_binding_books=False)
    print(ok, msg)

    ok, msg = dpk.verify_timecode_deletable(170000, verify_with_librarian=False)
    print(ok, msg)

    for book in dpk.books_in_timecode(170000):
        print(book.bid, book.status)

Both check methods return a ``(bool, str)`` pair: whether the timecode passed
and a human-readable accumulation of everything that was wrong.  Passing
``try_binding_books=False`` makes ``make_timecode_complete`` report without
writing Books.


API reference
=============

Imprinter
---------

.. automodule:: sotodlib.io.imprinter
   :no-index:

.. autoclass:: sotodlib.io.imprinter.Imprinter
   :members:

.. autoclass:: sotodlib.io.imprinter.ObsSet
   :members:

Book database schema
````````````````````

.. autoclass:: sotodlib.io.imprinter.Books
.. autoclass:: sotodlib.io.imprinter.Observations

Imprinter utilities
-------------------

.. automodule:: sotodlib.io.imprinter_utils
   :members:

Data packaging completion
-------------------------

.. autoclass:: sotodlib.io.datapkg_completion.DataPackaging
   :members:

Configuration helpers
---------------------

.. automodule:: sotodlib.io.datapkg_utils
   :members:

Command-line cleanup tool
-------------------------

.. automodule:: sotodlib.io.imprinter_cli

.. argparse::
    :module: sotodlib.io.imprinter_cli
    :func: get_parser
    :prog: imprinter-cli

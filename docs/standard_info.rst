==============================
Standard obs_info and det_info
==============================

Here we list the ``obs_info`` and ``det_info`` fields used in the
site-pipeline and processing of Level 3 data.


SO det_info
===========

The context metadata system populates the ``det_info`` entry with
per-detector information.  Depending on what supporting metadata
products are available, certain ``det_info`` items may or may not be
present.

.. list-table:: det_info (root level)
   :widths: 5 5 20
   :header-rows: 1
   :class: definition-table

   * - Name
     - Requirements
     - Description
   * - ``readout_id``
     - ObsFileDb
     - The name of the readout channel, within the context of a
       particular smurf tuning.  This field is unique across all
       channels in an observation.  Values of ``readout_id`` are
       associated with exactly one smurf tuning.
   * - ``detset``
     - ObsFileDb
     - The ``detset`` is a string tied to how the data are stored (see
       ObsFileDb).  In SO observations, time ordered data are
       organized together by wafer, so the ``detset`` string is likely
       to correspond to either the array name (as in ``wafer.array``),
       or the special stream identifier string ``wafer.stream_id``.
       (These might all have the same value.)
   * - ``det_id``
     - Channel map
     - The name of the physical detector.  Famously, this is not known
       unambiguously at the time of data acquisition, and is instead
       populated as a metadata product.  The ``det_id`` is unique,
       except for the special value "NO_MATCH" which indicates a
       readout channel for which a physical detector was not assigned
       by the matching algorithm.
   * - ``wafer*``
     - Channel map + Wafer design
     - See ``det_info.wafer`` detail.  Note the contents of ``wafer``
       are entirely as *designed* rather than as measured.
   * - ``tube*``
     - Channel map + Tube design
     - See ``det_info.tube`` detail.  Note the contents of ``tube``
       are entirely as *designed* rather than as measured.


Here is some detail on what those "Requirements" entail:

- "ObsFileDb" -- these fields are always available, as they are
  present in the ObsFileDb.
- "Channel map" -- these fields cannot be populated until a channel
  map (``readout_id`` to ``det_id`` association) is created.
- "Wafer design" -- these fields require a "wafer design" product, as
  output by ``make-det-info-wafer``.
- "Tube design" -- these fields require a "tune design" product, as
  output by ... ``make-det-info-tube`` (??).




.. csv-table:: ``det_info.wafer`` -- based on make-det-info output.
   :file: det_info_wafer.csv
   :widths: 20, 60
   :header-rows: 1
   :class: definition-table

.. csv-table:: ``det_info.tube`` --proposal
   :file: det_info_tube.csv
   :widths: 20, 60
   :header-rows: 1
   :class: definition-table


SO obs_info
===========

The fields in ``obs_info`` mirror the entries in the ObsDb, which are
mostly populated from information put into the Book's M_index.yaml file
by the bookbinder.

.. csv-table:: ``obs_info`` -- based on update-obsdb / bookbinder
   :file: obs_info.csv
   :widths: 20, 60
   :header-rows: 1
   :class: definition-table


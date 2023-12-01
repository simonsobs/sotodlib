"""check_book.py

This module an entry point to io.check_book, for checking obs/oper
Books for internal consistency & proper schema.  It may also be used
to create/update an ObsFileDb for such Books.

A configuration file can be used to set the ObsFileDb filename and
root path for ObsFileDb entries.

The config file can also be used to enable work-arounds and bypass
certain exceptions (which should not be necessary on compliant books.)

At the of this writing a minimal config file might be simply:

.. code-block:: yaml

    # Database setup (this is the default).
    obsfiledb: './obsfiledb.sqlite'

    # For obsdb filenames, path relative to which those names should
    # be specified.  (/ is the default.)
    root_path: '/'

    # Work-arounds
    extra_extra_files: ['frame_splits.txt']


But here is a more complete example, with lots of work-arounds
enabled:

.. code-block:: yaml

    # Database setup (this is the default).
    obsfiledb: './obsfiledb.sqlite'

    # For obsdb filenames, path relative to which those names should
    # be specified.  (/ is the default.)
    root_path: '/'

    # Work-arounds
    stream_file_pattern: 'D_obs_{stream_id}_{index:03d}.g3'
    extra_extra_files: ['frame_splits.txt']
    sample_range_inclusive_hack: True
    tolerate_missing_ancil: True
    tolerate_missing_ancil_timestamps: True
    tolerate_timestamps_value_discrepancy: False

    # Tolerate arbitrary extra files, except explicitly named ones
    tolerate_stray_files: True
    banned_files: ['frame_splits.txt']

    # If stream_ids are not provided in metadata, list them here.
    stream_ids:
      ufm_mv14
      ufm_mv18
      ufm_mv19
      ufm_mv22
      ufm_mv6 
      ufm_mv7 
      ufm_mv9 

    # If detset names are not provided in metadata, provide a map from
    # stream_id to detset name here.
    detset_map:
      ufm_mv14: sch_mv14
      ufm_mv18: sch_mv18
      ufm_mv19: sch_mv19
      ufm_mv22: sch_mv22
      ufm_mv6:  sch_mv6
      ufm_mv7:  sch_mv7
      ufm_mv9:  sch_mv9

"""

import sotodlib
from sotodlib.core import metadata
from sotodlib.io import check_book

import argparse
import sys
import yaml

from . import util


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.description = """

        Scan an "obs" or "oper" book and check for schema compliance;
        optionally update an obsfiledb.

    """
    parser.add_argument('book_dir',
                        help="Path to the Book.")
    parser.add_argument('--config', '-c',
                        help="Path to config file with work-arounds and ObsFileDb config.")
    parser.add_argument('--add', action='store_true',
                        help="After inspecting the book, add it to the ObsFileDb.")
    parser.add_argument('--overwrite', action='store_true',
                        help="If adding to ObsFileDb, remove existing references "
                        "to this obs first (prevents 'UNIQUE constraint' error).")
    return parser


def main(book_dir, config=None, add=None, overwrite=None, logger=None):
    if logger is None:
        logger = util.init_logger(__name__, 'check_book: ')

    logger.info(f'Examining {book_dir}')

    if config is not None:
        logger.debug(f'Loading config from {config}')
        config = yaml.safe_load(open(config, 'rb'))
    else:
        config = {}
    bs = check_book.BookScanner(book_dir, config)

    bs.go()
    bs.report()

    if len(bs.results['errors']):
        logger.error('Cannot register this obs due to errors.')
        sys.exit(1)

    if not add:
        sys.exit(0)

    detset_rows, file_rows = bs.prep_obsfiledb(config.get('root_path', '/'))

    # Write to obsfiledb
    obsfiledb_file = config.get('obsfiledb', 'obsfiledb.sqlite')
    logger.debug('Updating %s ...' % obsfiledb_file)
    db = metadata.ObsFileDb(obsfiledb_file)

    if overwrite:
        # Note this only drops the obs ... if detsets need to be
        # rewritten, you'd better start over entirely.
        logger.debug(' -- removing any existing references.')
        db.drop_obs(file_rows[0]['obs_id'])

    logger.debug(
        ' -- adding %i detsets and %i file refs' % (len(detset_rows), 
        len(file_rows))
    )
    for name, dets in detset_rows:
        if len(db.get_dets(name)) == 0:
            db.add_detset(name, dets)
    for row in file_rows:
        db.add_obsfile(**row)


if __name__ == '__main__':
    util.main_launcher(main, get_parser)

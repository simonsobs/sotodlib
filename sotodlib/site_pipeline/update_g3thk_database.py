"""
Script for running updates on (or creating) a G3tHk database, the index of
level 2 housekeeping (HK) data. This setup is specifically designed to work
when the data is dynamically coming in, meaning it is designed to work from
something like a cronjob.

For each ``.g3`` file under ``<data_prefix>/hk`` the database records the
file, the OCS agents that appear in it, and the fields each agent published,
along with their start and stop times.

This is the first stage of the data packaging pipeline and must run before
``update_g3tsmurf_db``: the ``finalized_until`` fields published by the
suprsync agents, which G3tSmurf uses to work out how much level 2 data has
actually been transferred, are themselves housekeeping data.

By default the update resumes from 10 seconds before the last file already in
the database. Use ``--from-scratch`` (implied if the database is empty) to
rebuild from ctime 1.6e9.

It takes the same configuration file as ``update_g3tsmurf_db``, and requires
at minimum the ``data_prefix`` and ``g3thk_db`` keys::

    data_prefix : "/path/to/daq-node/"
    g3thk_db: "/path/to/g3hk.db"

See the Data Packaging page of the sotodlib documentation for the full
pipeline description.
"""
from typing import Optional
import argparse
import logging
from sqlalchemy import desc

from sotodlib.site_pipeline.utils.profiler import profile, add_profile_args

from sotodlib.io.g3thk_db import G3tHk, HKFiles, logger


@profile("update_g3thk_database")
def main(config: Optional[str]=None, from_scratch: bool=False, verbosity: int=2):
    """Index new level 2 housekeeping files into the G3tHk database.

    Arguments
    ---------
    config: str
        path to the G3tSmurf/G3tHk configuration file. Requires the
        ``data_prefix`` and ``g3thk_db`` keys; the ``finalization`` block, if
        present, supplies the list of OCS instance-ids to index.
    from_scratch: bool
        if True, index from ctime 1.6e9 (all SO time) instead of resuming from
        the last file in the database. Also forced when the database is empty.
    verbosity: int
        0-3, higher numbers = more printouts. 0:Error, 1:Warning, 2:Info,
        3:Debug. A progress bar is shown for verbosity > 1.
    """

    show_pb = True if verbosity > 1 else False

    if verbosity == 0:
        logger.setLevel(logging.ERROR)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 3:
        logger.setLevel(logging.DEBUG)

    HK = G3tHk.from_configs(config)

    if from_scratch or HK.session.query(HKFiles).count()==0:
        logger.info("Building Database from Scratch, May take awhile")
        min_time = int(1.6e9)
    else:
        ## start at the last file in the database
        last_file = HK.session.query(HKFiles)
        last_file = last_file.order_by(desc(HKFiles.global_start_time)).first()

        logger.info(f"Starting from last file in database: {last_file.filename}")
        min_time = last_file.global_start_time - 10
        logger.debug(f"Setting minium time to {min_time}")

    HK.add_hkfiles(min_ctime=min_time, show_pb=show_pb)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Index level 2 housekeeping .g3 files into the G3tHk "
            "database. Resumes from the last indexed file unless "
            "--from-scratch is passed."
        )

    parser.add_argument('config',
                        help="G3tSmurf/G3tHk configuration file. Requires the "
                        "'data_prefix' and 'g3thk_db' keys.")
    parser.add_argument('--from-scratch',
                        help="Index from ctime 1.6e9 instead of resuming from "
                        "the last file in the database",
                        action="store_true")
    parser.add_argument("--verbosity", help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
                       default=2, type=int)
    
    add_profile_args(parser)
    
    return parser


if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))

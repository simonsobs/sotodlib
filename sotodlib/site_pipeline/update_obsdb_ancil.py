"""update-obsdb-ancil

This script uses io.ancil modules to update archives of reduced
ancillary data and to update an obsdb with quantities computed based
on those ancillary data.

The entry point is :func:`main` but the different command functions
can be called directly.

"""

import argparse
import itertools
import logging
import time
from typing import Optional
import yaml

from sotodlib import core
from sotodlib.site_pipeline.utils import logging
from sotodlib.io import ancil


logger = logging.init_logger('update-obsdb-ancil', 'update-obsdb-ancil: ')

DAY = 86400
DEFAULT_CONFIG = {
    'lookback_time':  7 * DAY,
    'datasets': {},
}
DEFAULT_CONFIG_FILENAME = 'ancil.yaml'


def _engines_iter(cfg, args_datasets):
    if args_datasets is None or len(args_datasets) == 0:
        args_datasets = cfg['datasets'].keys()
    for k in args_datasets:
        yield (k, ancil.utils.get_engine(k, cfg['datasets'][k]))


def _get_config(config_file):
    cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))
    datasets = cfg.get('datasets')
    if datasets is None:
        return cfg
    for broadcast_key in ['data_prefix']:
        if (val := cfg.get(broadcast_key)) is None:
            continue
        for ds in datasets.values():
            if ds.get(broadcast_key) is None:
                ds[broadcast_key] = val
    return cfg


def update_base_data(config_file, time_range=None, datasets=None, full_scan=False):
    cfg = _get_config(config_file)

    if full_scan:
        logger.info(f'Performing full update of base data.')
    elif time_range is None:
        now = time.time()
        time_range = (now - cfg['lookback_time'], now)
        logger.info(f'Updating base data over time range {time_range}.')

    for dataset, engine in _engines_iter(cfg, datasets):
        logger.info(f'Updating base data for {dataset}...')
        engine.update_base(time_range)

    logger.info(f'Finished updating base data.')


def update_obsdb(config_file, time_range=None, datasets=None, redo=False):
    cfg = _get_config(config_file)

    logger.info(f'Updating obsdb')
    for dataset, engine in _engines_iter(cfg, datasets):
        logger.info(f'Processing {dataset}...')
        engine.register_friends(cfg['datasets'])

        # Ensure target columns are present in db.
        obsdb = core.metadata.ObsDb(cfg['target_obsdb'])
        engine.obsdb_check(obsdb, create_cols=True)

        # Find records that need to be updated.
        q = engine.obsdb_query(time_range=time_range, redo=redo)

        logger.info(f'Query for records to update is: "{q}"')
        recs = obsdb.query(q)
        del obsdb

        logger.info(f' ... identified {len(recs)} items to update.')
        if len(recs) == 0:
            continue

        recs_it = iter(recs)
        while rec_bunch := list(itertools.islice(recs_it, 200)):
            results = engine.collect(rec_bunch, for_obsdb=True)
            logger.info(f' ... updating obsdb.')
            obsdb = core.metadata.ObsDb(cfg['target_obsdb'])
            for rec, result in zip(rec_bunch, results):
                logger.debug(f"{rec['obs_id']} : {result}")
                obsdb.update_obs(rec['obs_id'], result, commit=False)
            obsdb.conn.commit()
            del obsdb


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    def add_args(p, *args):
        if '--config-file' in args:
            p.add_argument('--config-file', '-c', default=DEFAULT_CONFIG_FILENAME,
                           help="Path to config file.")
        if '--dataset' in args:
            p.add_argument(
                '--dataset', '-d', action='append',
                help="""
                Limit processing to only the specified dataset.
                If passed multiple times, the chosen datasets are
                processed in the corresponding order.""")
        if '--time-range' in args:
            p.add_argument(
                '--time-range', nargs=2, type=float,
                help="""
                Restrict processing to items falling between the
                two provided unix timestamps.""")
            p.add_argument(
                '--lookback-days', type=float,
                help="""
                As an alternative to --time-range, restrict
                processing to items falling between the present time
                and the specified number of days in the past.""")

    parser.add_argument('--verbose', '-v', action='count', help=
                        "Pass one or more times to increase logging verbosity.")
    sps = parser.add_subparsers(title='Commands', dest='command')

    p = sps.add_parser('update-base-data', help="""Update the base
    data for an ancillary data archive. This will typically trigger
    loading of the data from the source, and then reduction and
    storage in the local archive.""")

    add_args(p, '--config-file', '--dataset', '--time-range')
    p.add_argument('--full-scan', action='store_true')

    p = sps.add_parser('update-obsdb', help="""
    Update obsdb records based on data already in the archives. This
    is typically run after refreshing the base data with
    update-base-data.""")

    add_args(p, '--config-file', '--dataset', '--time-range')
    p.add_argument('--redo', action='store_true')

    p = sps.add_parser('check', help="""Read the config file and test
    datasets by attempting to instantiate them. This should not
    normally touch the underlying data or attempt to load new data
    from base sources.""")

    add_args(p, '--config-file', '--dataset', '--time-range')

    p = sps.add_parser('test', help="""Perform the obsdb data
    reduction on a specified obs_id. Results are not saved to obsdb.""")

    add_args(p, '--config-file', '--dataset')
    p.add_argument('--query', '-q', help=
                   """
                   Use this obsdb query to select records for
                   computation, instead of passing specific obs_id.""")
    p.add_argument('--compare', action='store_true', help=
                   """
                   If passed, the computed values will be compared to
                   values already present in the obsdb.""" )
    p.add_argument('obs_id', nargs='*', help=
                   "obs_id (pass multiple) to compute results for.")

    p = sps.add_parser('run-job', help="""Run a job defined in the config file.""")
    add_args(p, '--config-file')
    p.add_argument('job_name', help=
                  " Name of job to run (matched to entry in config['job_defs']).")

    return parser


def main(
        command : Optional[str]=None,
        verbose : Optional[bool]=None,
        config_file : Optional[str]=None,
        time_range : Optional[list]=None,
        dataset : Optional[list]=None,
        lookback_days : Optional[float]=None,
        query : Optional[str]=None,
        obs_id : Optional[list]=None,
        full_scan : Optional[bool]=None,
        redo : Optional[bool]=None,
        compare : Optional[bool]=None,
        job_name : Optional[str]=None,
):
    """Entry point for CLI or job runner.  Some parameters are only
    used by some commands.

    """
    if verbose:
        ancil.logger.setLevel(logging.DEBUG)
        logger.handlers[0].setLevel(logging.DEBUG)

    if time_range is None:
        if lookback_days is not None:
            now = time.time()
            time_range = (now - lookback_days * DAY, now)
    elif lookback_days is not None:
        raise RuntimeError("Do not pass both time_range and lookback_days.")

    if command == 'update-base-data':
        update_base_data(config_file, datasets=dataset,
                         time_range=time_range, full_scan=full_scan)

    elif command == 'update-obsdb':
        update_obsdb(config_file, datasets=dataset,
                     time_range=time_range, redo=redo)

    elif command == 'check':
        # Just trial load the config, processors.
        print(f'Loading config file {config_file} ...')
        cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))
        for dataset, engine in _engines_iter(cfg, dataset):
            print(f'  dataset={dataset}')
            for k, v in engine.check_base().items():
                print(f'    {k}: {v}')


    elif command == 'test':
        # Run on some obs_ids.
        cfg = _get_config(config_file)

        obsdb = core.metadata.ObsDb(cfg['target_obsdb'])
        if query:
            assert len(obs_id) == 0, "User passed --query and obs_id"
            items = list(iter(obsdb.query(query)))
        else:
            items = []
            for o in obs_id:
                if (oi := obsdb.get(o)) is None:
                    print(f'No obsdb record for "{o}", skipping test.')
                else:
                    items.append(oi)

        results = [{} for r in items]
        for dataset, engine in _engines_iter(cfg, dataset):
            print(dataset)
            engine.register_friends(cfg['datasets'])
            ei = iter(engine.getter(items, results))
            for o, r in zip(items, results):
                _r = next(ei)
                r.update(_r)
                print(o['obs_id'], _r)
                if compare:
                    ex = obsdb.get(o['obs_id'])
                    for k1, k2 in engine._obsdb_map().items():
                        print(f'  {k2:<20s}', ex.get(k2), r[k1])
            print()
        print()

    elif command == 'run-job':
        cfg = _get_config(config_file)
        for job_def in cfg.get('job_defs', []):
            if job_def['name'] == job_name:
                break
        else:
            raise RuntimeError(f"Job '{job_name}' not found in {config_file} job_defs entry.")

        for step in job_def['steps']:
            print('Step ...')
            step_cfg = {'config_file': config_file} | step
            main(**step_cfg)

    else:
        raise RuntimeError(f"Invalid command: {command}")

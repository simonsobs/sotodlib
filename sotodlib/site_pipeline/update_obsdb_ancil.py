import argparse
import itertools
import logging
import time
import yaml

from sotodlib import core
from sotodlib.site_pipeline import util
from sotodlib.io import ancil


logger = util.init_logger('update-obsdb-ancil', 'update-obsdb-ancil: ')

DAY = 86400
DEFAULT_CONFIG = {
    'lookback_time':  7 * DAY,
    'datasets': {},
}


def _get_engine(key, cfg):
    if cfg.get('class') is not None:
        key = cfg['class']
    cls = ancil.ANCIL_ENGINES[key]
    return cls(cfg)


def _engines_iter(cfg, args_datasets):
    if args_datasets is None or len(args_datasets) == 0:
        args_datasets = cfg['datasets'].keys()
    for k in args_datasets:
        yield (k, _get_engine(k, cfg['datasets'][k]))


def update_base_data(config_file, time_range=None, datasets=None, full_scan=False):
    cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))

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
    cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))

    tquery = None
    if time_range is not None:
        t0, t1 = time_range
        if t0 is None:
            tquery = f'timestamp < {t1}'
        elif t1 is None:
            tquery = f'timestamp >= {t0}'
        else:
            tquery = f'(timestamp >= {t0}) and (timestamp < {t1})'

    logger.info(f'Updating obsdb')
    for dataset, engine in _engines_iter(cfg, datasets):
        logger.info(f'Processing {dataset}...')
        engine.register_friends(cfg['datasets'])

        # Ensure target columns are present in db.
        obsdb = core.metadata.ObsDb(cfg['target_obsdb'])
        for k in engine.obsdb_fields:
            obsdb.add_obs_columns([k + ' float'])

        # Find records that need to be updated.
        if redo:
            q = '1'
        else:
            q = engine._get_obsdb_query()

        if tquery:
            q = f'{tquery} and {q}'

        logger.debug(f'Query for records to update is: "{q}"')
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

    parser.add_argument('--verbose', '-v', action='store_true')
    sps = parser.add_subparsers(title='commands', dest='command')

    p = sps.add_parser('update-base-data')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')
    p.add_argument('--lookback-days', type=float)
    p.add_argument('--full-scan', action='store_true')

    p = sps.add_parser('update-obsdb')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')
    p.add_argument('--lookback-days', type=float)
    p.add_argument('--redo', action='store_true')

    p = sps.add_parser('check')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')
    p.add_argument('--lookback-days', type=float)

    p = sps.add_parser('test')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')
    p.add_argument('--query', '-q')
    p.add_argument('obs_id', nargs='+')

    return parser

def main(command=None, verbose=None, lookback_days=None,
         obs_id=None, config_file=None, dataset=None, query=None,
         full_scan=None, redo=None):
    if verbose:
        ancil.logger.setLevel(logging.DEBUG)

    time_range = None
    if lookback_days is not None:
        now = time.time()
        time_range = (now - lookback_days * DAY, now)

    if command == 'update-base-data':
        update_base_data(config_file, datasets=dataset,
                         time_range=time_range, full_scan=full_scan)

    elif command == 'update-obsdb':
        update_obsdb(config_file, datasets=dataset,
                     time_range=time_range, redo=redo)

    elif command == 'check':

        # Just trial load the config, processors.
        cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))
        for dataset, engine in _engines_iter(cfg, dataset):
            print(dataset, engine)

    elif command == 'test':

        # Run on some obs_ids.
        cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))

        obsdb = core.metadata.ObsDb(cfg['target_obsdb'])
        if query:
            items = list(iter(obsdb.query(query)))
        else:
            items = [obsdb.get(o) for o in obs_id]

        results = [{} for r in items]
        for dataset, engine in _engines_iter(cfg, dataset):
            print(dataset)
            engine.register_friends(cfg['datasets'])
            ei = iter(engine.getter(items, results))
            for o, r in zip(items, results):
                _r = next(ei)
                r.update(_r)
                print(o['obs_id'], _r)
            print()
        print()

    else:
        print('Pass -h for uage.')
        return False

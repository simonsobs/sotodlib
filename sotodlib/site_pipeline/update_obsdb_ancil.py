import argparse
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
    return cls(**cfg)

def _engines_iter(cfg, args_datasets):
    if args_datasets is None or len(args_datasets) == 0:
        args_datasets = cfg['datasets'].keys()
    for k in args_datasets:
        yield (k, _get_engine(k, cfg['datasets'][k]))
    

def update_base_data(config_file, time_range=None, datasets=None):
    cfg = DEFAULT_CONFIG | yaml.safe_load(open(config_file, 'rb'))

    if time_range is None:
        now = time.time()
        time_range = (now - cfg['lookback_time'], now)
    logger.info(f'Updating base data over time range {time_range}.')

    for dataset, engine in _engines_iter(cfg, datasets):
        logger.info(f'Updating base data for {dataset}...')
        engine.update_base(time_range)
    logger.info(f'Finished updating base data.')


def update_obsdb(config_file):
    # Base dataset updates.
    pass


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', '-v', action='store_true')
    sps = parser.add_subparsers(title='commands', dest='command')

    p = sps.add_parser('update-base-data')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')
    p.add_argument('--lookback-days', type=float)

    p = sps.add_parser('update-obsdb')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')

    p = sps.add_parser('check')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')

    p = sps.add_parser('test')
    p.add_argument('--config-file', '-c', default='cli.yaml')
    p.add_argument('--dataset', '-d', action='append')
    p.add_argument('obs_id', nargs='+')

    return parser

def main(command=None, verbose=None, lookback_days=None,
         obs_id=None, config_file=None, dataset=None):
    if verbose:
        ancil.logger.setLevel(logging.DEBUG)

    if command == 'update-base-data':
        time_range = None
        if lookback_days is not None:
            now = time.time()
            time_range = (now - lookback_days * DAY, now)
        update_base_data(config_file, datasets=dataset,
                         time_range=time_range)

    elif command == 'update-obsdb':

        cfgfile = 'cli.yaml'
        cfg = DEFAULT_CONFIG | yaml.safe_load(open(cfgfile, 'rb'))

        # Records
        obsdb = core.metadata.ObsDb(cfg['source_obsdb'])

        obs = obsdb.query("type='obs'")
        sl = slice(6500, 6600)
        recs = obs[sl]

        trs = [(_o['start_time'], _o['start_time'] + _o['duration'])
               for _o in recs]
        obs_ids = recs['obs_id']

        engines = []
        gens = []
        results = [{}] * len(recs)
        for dataset, engine in _engines_iter(cfg, dataset):
            print(dataset, engine)
            engines.append(engine)
            gens.append(engine.getter(targets=recs, results=results))

        for obs_id, info in zip(obs_ids, results):
            for g in gens:
                info.update(next(g))
            print(obs_id, info)


    elif command == 'check':
        # Just trial load the config, processors.

        cfgfile = 'cli.yaml'
        cfg = DEFAULT_CONFIG | yaml.safe_load(open(cfgfile, 'rb'))

        for dataset, engine in _engines_iter(cfg, dataset):
            print(dataset, engine)

    elif command == 'test':
        # Run on some obs_ids.

        cfgfile = 'cli.yaml'
        cfg = DEFAULT_CONFIG | yaml.safe_load(open(cfgfile, 'rb'))

        obsdb = core.metadata.ObsDb(cfg['source_obsdb'])
        #rec = obsdb.get('obs_1746817691_satp1_1111111')
        items = [obsdb.get(o) for o in obs_id]
        results = [{} for r in items]
        for dataset, engine in _engines_iter(cfg, dataset):
            print(dataset)
            ei = iter(engine.getter(items, results))
            for o, r in zip(items, results):
                _r = next(ei)
                r.update(_r)
                print(o['obs_id'], _r)
            print()
        print()

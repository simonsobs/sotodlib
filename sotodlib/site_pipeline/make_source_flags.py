from argparse import ArgumentParser
import logging
import numpy as np
import os
import sys
import yaml

import sotodlib
from sotodlib import core, coords, site_pipeline
from . import util

logger = logging.getLogger(__name__)

def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('obs_id',help=
                        "Observation for which to generate flags.")
    parser.add_argument('-c', '--config-file', help=
                        "Configuration file.")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help="Pass multiple times to increase.")
    return parser

def get_config(config_file, obs_id=None, verbose=None):
    cfg = yaml.safe_load(open(config_file, 'r'))
    for k in ['obs_id', 'verbose']:
        cfg[k] = getattr(args, k)
    return cfg

def main(config_file=None, verbose=0, obs_id=None):
    config = get_config(config_file)

    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    ctx = core.Context(config['context_file'])

    # This will process data by detset, but will insist that this maps
    # one-to-one with some detdb field.
    group_by = config['subobs'].get('group_by', 'dets:detset')
    if group_by.startswith('dets:'):
        group_by = group_by.split(':',1)[1]
    else:
        raise ValueError("Can't group by '{group_by}'")

    if os.path.exists(config['archive']['index']):
        logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
        db = core.metadata.ManifestDb(config['archive']['index'])
    else:
        logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dets:' + group_by)
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(config['archive']['index'], scheme=scheme)

    # We will loop over detsets here, but insist on a simple
    # correspondence to the output index values.
    index_map = {}

    detsets = ctx.obsfiledb.get_detsets(obs_id)
    for detset in detsets:
        logger.info(f'Loading {config["obs_id"]}:detset={detset}')
        # Load pointing and dets axis; we don't need signal though.
        tod = ctx.get_obs(obs_id, detsets=[detset],
                          no_signal=True)

        # Figure out the indexing info for this.
        this_index = tod.det_info[group_by]
        assert([x == this_index[0] for x in this_index])
        this_index = this_index[0]
        if this_index in index_map:
            raise RuntimeError(f"Some detset already mapped to {this_index}: "
                               f"{index_map[this_index]}")
        index_map[this_index] = detset
        logger.info(f'detset "{detset}" corresponds to {group_by}="{this_index}"')

        # Load / compute mask parameters.
        mask_params = config['mask_params']['default']
        mask_res = util.parse_quantity(
            util.lookup_conditional(
                config['mask_params'], 'res', default=[1, 'arcmin']),
            'deg').value

        sources = coords.planets.get_nearby_sources(tod)
        flags = None
        for source_name, eph_object in sources:
            logger.info(f'Flagging for {source_name} ...')
            _flags = coords.planets.compute_source_flags(
                tod, center_on=source_name, res=mask_res*coords.DEG,
                mask=mask_params)
            weight = np.mean(_flags.get_stats()['samples']) / _flags.shape[1]
            logger.info(f' ... weight for {source_name} was {weight*100:.2}%.')
            if flags is None:
                flags = _flags
            else:
                flags += _flags

        if flags is None:
            import so3g  # This is here to dodge sphinxarg mocking failure.
            flags = so3g.proj.RangesMatrix.zeros(
                shape=(tod.dets.count, tod.samps.count))

        # Compute fraction of samples
        weight = np.mean(flags.get_stats()['samples']) / flags.shape[1]
        logger.info(f'Total mask weight is {weight*100:.2}%.')

        # Wrap result into AxisManager for HDF5 off-load.
        aman = core.AxisManager(tod.dets, tod.samps)
        aman.wrap('source_flags', flags, [(0, 'dets'), (1, 'samps')])

        # Get file + dataset from policy.
        policy = util.ArchivePolicy.from_params(config['archive']['policy'])
        dest_file, dest_dataset = policy.get_dest(obs_id)
        aman.save(dest_file, dest_dataset, overwrite=True)

        # Update the index.
        db_data = {'obs:obs_id': obs_id,
                   'dataset': dest_dataset,
                   f'dets:{group_by}': this_index}
        db.add_entry(db_data, dest_file, replace=True)

    # Return something?
    return tod, aman

if __name__ == '__main__':
    util.main_launcher(main, get_parser)

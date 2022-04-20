"""This module produces maps for a single observation of a bright
point source.  The observation is identifed by an obs_id.  The data
for the observation may be divided into different detector groups; and
each 'group' will be loaded and mapped independently (this will
normally be associated with a "detset").  The data for each
observation in each group may be further subdivided into 'data
splits'; this normally corresponds to frequency "band".

"""

from argparse import ArgumentParser
import logging
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt

import sotodlib
import so3g
from sotodlib import core, coords, site_pipeline, tod_ops

from . import util

logger = logging.getLogger(__name__)

def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', help=
                        "Configuration file.")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help="Pass multiple times to increase.")
    parser.add_argument('obs_id',help=
                        "Observation for which to make source map.")
    parser.add_argument('--test', action='store_true', help=
                        "Reduce detector count for quick tests.")

    return parser

def _get_config(args):
    cfg = yaml.safe_load(open(args.config_file, 'r'))
    for k in ['obs_id', 'verbose']:
        cfg[k] = getattr(args, k)
    cfg['_args'] = args
    return cfg

def _plot_map(bundle, filename=None, **kwargs):
    # To-do for the plot:
    # - T map and weights map (full footprint)
    # - T, Q, and U zoomed in on source
    # - labeled array diagram showing participating detectors and
    #  their weights
    src_map = bundle['solved']
    fig, axs = plt.subplots(3, 1, subplot_kw={'projection': src_map.wcs},
                            figsize=(5, 8))
    mask = bundle['weights'][0,0] != 0
    for _m, ax in zip(src_map, axs):
        _m1 = np.ma.masked_where(~mask, _m)
        ax.imshow(_m1, origin='lower')
    if filename is not None:
        fig.savefig(filename)
    return fig

def _adjust_focal_plane(tod, focal_plane=None, boresight_offset=None):
    """Apply pointing correction to focal plane.  These are assumed to be
    boresight corrections, even though there is one per detector.

    """
    if focal_plane is None:
        focal_plane = tod.focal_plane
    if boresight_offset is None:
        boresight_offset = tod.boresight_offset

    # Get detector the boresight pointing quaternions
    fp = focal_plane
    fp_q = so3g.proj.quat.rotation_xieta(fp.xi, fp.eta, fp.gamma)

    # Get the adjustments
    bc = boresight_offset
    fp_adjust = so3g.proj.quat.rotation_xieta(bc.dx, bc.dy, bc.gamma)

    # Apply focal plane adjust.
    fp_new = fp_adjust * fp_q

    # Get modified xi, eta, gamma.
    xi, eta, gamma = so3g.proj.quat.decompose_xieta(fp_new)
    fp.xi[:] = xi
    fp.eta[:] = eta
    fp.gamma[:] = gamma


def main(args=None):
    """Entry point."""
    if args is None:
        args = sys.argv[1:]
    parser = _get_parser()
    config = _get_config(parser.parse_args(args))

    if config['verbose'] >= 1:
        logger.setLevel('INFO')
    if config['verbose'] >= 2:
        sotodlib.logger.setLevel('INFO')
    if config['verbose'] >= 3:
        sotodlib.logger.setLevel('DEBUG')

    ctx = core.Context(config['context_file'])

    group_by = config['subobs'].get('use', 'detset')
    if group_by == 'detset':
        groups = ctx.obsfiledb.get_detsets(config['obs_id'])
    elif group_by.startswith('dets:'):
        group_by = group_by.split(':',1)[1]
        groups = ctx.detdb.props(props=group_by).distinct()[group_by]
    else:
        raise ValueError("Can't group by '{group_by}'")

    if len(groups) == 0:
        logger.warning(f'No map groups found for obs_id={config["obs_id"]}')

    if os.path.exists(config['archive']['index']):
        logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
        db = core.metadata.ManifestDb(config['archive']['index'])
    else:
        logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('group')
        scheme.add_data_field('split')
        db = core.metadata.ManifestDb(config['archive']['index'], scheme=scheme)

    for group in groups:
        logger.info(f'Loading {config["obs_id"]}:{group_by}={group}')
        map_info = {'obs_id': config['obs_id'],
                    'group': group,
                    group_by: group,
        }
        map_info['product_id'] = '{obs_id}-{group}'.format(**map_info)

        if group_by == 'detset':
            tod = ctx.get_obs(config['obs_id'], detsets=[group])
        else:
            tod = ctx.get_obs(config['obs_id'], dets={group_by: group})

        # Modify dets axis for testing
        if config['_args'].test:
            dets = tod.dets.vals[::10]
            tod.restrict('dets', dets)

        # Modify samps axis for FFTs.
        tod_ops.fft_trim(tod)

        # Determine what source to map.
        ## To-do: have a mode where it maps all sources it finds.
        source_name = config['mapmaking'].get('force_source')
        if source_name is None:
            sources = coords.planets.get_nearby_sources(tod)
            if len(sources) == 1:
                source_name = sources[0][0]
            elif len(sources) == 0:
                logger.error("No mappable sources in footprint.")
                sys.exit(1)
            else:
                logger.error("Multiple sources in footprint: %s" %
                             ([n for n, s in sources],))
                sys.exit(1)
        map_info['source_name'] = source_name
        
        # Plan to split on frequency band
        band_splits = core.metadata.ResultSet(['dets:name', 'group'])
        band_splits.rows = list(zip(tod.dets.vals, tod.array_data['fcode']))
        band_splits = coords.planets.load_detector_splits(tod, source=band_splits)

        # Deconvolve readout filter and detector time constants.
        tod_ops.fft_trim(tod)
        tod_ops.detrend_tod(tod)
        filt = (tod_ops.filters.iir_filter(invert=True)
                * tod_ops.filters.timeconst_filter(invert=True))
        tod.signal[:] = tod_ops.fourier_filter(tod, filt, resize=None, detrend=None)

        # Demodulation & downsampling.
        # ...

        # Fix pointing.
        logger.info('Applying pointing corrections.')
        if 'boresight_offset' in tod:
            logger.info(' ... applying "boresight_offset".')
            _adjust_focal_plane(tod)

        # Apply calibration
        cal = None
        for k in config['preprocessing']['cal_keys']:
            if k in tod:
                if cal is None:
                    cal = 1.
                cal *= tod[k]
        if cal is not None:
            tod.signal *= cal[:,None]

        # Figure out the resolution
        reses = [
            util.lookup_conditional(config['mapmaking'], 'res', tags=[b])
            for b in band_splits.keys()]
        assert(all([r == reses[0] for r in reses]))  # Resolution conflict
        res_deg = util.parse_angle(reses[0])
        
        # Where to put things.
        policy = util.ArchivePolicy.from_params(
            config['archive']['policy'])
        
        dest_dir = policy.get_dest(**map_info)
        if os.path.exists(dest_dir):
            logger.info(f' ... destination already exists ({dest_dir})')
        else:
            logger.info(f' ... creating destination dir ({dest_dir})')
            os.makedirs(dest_dir)

        # Make the maps.
        logger.info(f'Calling mapmaker.')
        output = coords.planets.make_map(tod, center_on=source_name,
                                         res=res_deg*coords.DEG,
                                         data_splits=band_splits,
                                         info=map_info)

        # Save and plot
        ocfg = config['output']
        pcfg = config['plotting']
        opattern = ocfg['pattern']
        ppattern = pcfg.get('pattern', opattern + '.png')

        used_names = []  # watch for accidental self-overwrites
        db_code = ocfg['map_codes'][0]
        
        for key, bundle in output['splits'].items():
            # Save
            for code in ocfg['map_codes']:
                filename = os.path.join(
                    dest_dir, opattern.format(map_code=code, split=key, **map_info))
                bundle[code].write(filename)
                if filename in used_names:
                    logger.warning(f'Wrote this file more than once: {filename}; '
                                   f'using {key}, {code}')
                used_names.append(filename)
                if code == db_code:
                    # Index
                    db_data = {'obs:obs_id': config['obs_id'],
                               'group': group,
                               'split': key}
                    db.add_entry(db_data, filename, replace=True)

            # Plot
            plot_filename = os.path.join(
                dest_dir, ppattern.format(map_code=code, split=key, **map_info))
            _plot_map(bundle, plot_filename)


    # Return something?
    return True

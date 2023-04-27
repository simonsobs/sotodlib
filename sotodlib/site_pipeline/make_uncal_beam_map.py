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

import sotodlib
import so3g
from sotodlib import core, coords, site_pipeline, tod_ops, hwp

from . import util

logger = None


def get_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', help=
                        "Configuration file.")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help="Pass multiple times to increase.")
    parser.add_argument('obs_id', nargs='?', help=
                        "Observation for which to make source map.")
    parser.add_argument('--test', action='store_true', help=
                        "Reduce detector count for quick tests.")
    parser.add_argument('--force-source', help=
                        "Specify which source to target (when ambiguous).")

    return parser

def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def _clip_map(m, mask=None, edges=[0.05, 0.95]):
    if mask is None:
        mask = ~np.isnan(m)
    lims = np.quantile(m[mask], edges)
    lims = lims[0] + (lims[1] - lims[0]) * \
           (np.array([0., 1.]) - edges[0]) / (edges[1] - edges[0])
    out = np.clip(m, *lims)
    out[~mask] = np.nan
    return out

def _renorm(im):
    s = ~np.isnan(im)
    mag = abs(im[s]).max()
    if mag <= 0:
        alpha = 0
    else:
        alpha = int(np.floor(np.log10(mag)))
    rescale = 10**-alpha
    label = '$10^{%d}$' % alpha
    imr = im.copy()
    imr[s] *= rescale
    return imr, label, rescale

def _squarify(ax, lims=None):
    if lims is None:
        lims = []
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        for z in [xl, yl]:
            z0, dz = (z[0] + z[1])/2, (z[0] - z[1])/2
            lims.append([z0, np.sign(dz), abs(dz)])
        dz = max([l[2] for l in lims])
        lims = [(l[0] + l[1]*dz, l[0] - l[1]*dz) for l in lims]
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])

def _plot_one_map(fig, ax, m):
    mr, units, _ = _renorm(m)
    im = ax.imshow(mr)
    fig.colorbar(im, ax=ax, label=f'Units ({units})',
                 orientation='horizontal')
    return im

def plot_map(bundle, filename=None, tod=None, obs_info=None, det_info=None,
             focal_plane=None, det_mask=None, group=None, subset=None,
             zoom_size=None, title=None, **kwargs):
    import matplotlib.pyplot as plt

    if tod is not None:
        if obs_info is None:
            obs_info = tod.get('obs_info')
        if det_info is None:
            det_info = tod.get('det_info')
        if focal_plane is None:
            focal_plane = tod.get('focal_plane')

    src_map = bundle['solved']
    if src_map.shape[0] == 3:
        fig, axs = plt.subplots(2, 3, figsize=(9, 8),
                                subplot_kw={'projection': 'so-beammap'})
        axs = {
            'W':  axs[0, 0],
            'T':  axs[0, 1],
            'fp': axs[0, 2],
            'Tz': axs[1, 0],
            'Qz': axs[1, 1],
            'Uz': axs[1, 2],
        }

    else:
        fig, axs = plt.subplots(2, 2, figsize=(7, 8),
                                subplot_kw={'projection': 'so-beammap'})
        axs = {
            'W':  axs[0, 0],
            'T':  axs[0, 1],
            'Tz': axs[1, 0],
            'fp': axs[1, 1],
        }

    mask = bundle['weights'][0,0] != 0
    _plot_one_map(fig, axs['W'],
                  _clip_map(bundle['weights'][0, 0], mask))
    _plot_one_map(fig, axs['T'],
                  _clip_map(src_map[0], mask))
    _squarify(axs['W'])
    _squarify(axs['T'])

    # zoomed in ...
    if zoom_size is None:
        zoom_size = 0.5  # deg
    Z = zoom_size * coords.DEG
    box = np.array([[-Z, Z], [Z, -Z]])
    zoomed = src_map.submap(box)

    # Plot whichever of T/Q/U are appearing here.
    for i, k in enumerate(['Tz', 'Qz', 'Uz']):
        if k in axs:
            _plot_one_map(fig, axs[k], zoomed[i])

    if focal_plane:
        x, y = focal_plane.xi / coords.DEG, focal_plane.eta / coords.DEG
        if det_mask is not None:
            x, y = x[det_mask], y[det_mask]
        axs['fp'].scatter(x, y, s=2, alpha=.25)
        axs['fp'].set_aspect('equal')

    if title is None:
        try:
            title = '{obs_info.obs_id} - {group} - {subset}'.format(
                obs_info=obs_info, group=group, subset=subset)
        except:
            title = ''
    plt.suptitle(title)

    for k, label, c in [
            ('W', 'Weight', 'k'),
            ('T', 'Signal', 'k'),
            ('Tz', 'T', 'w'),
            ('Qz', 'Q', 'w'),
            ('Uz', 'U', 'w')]:
        if k in axs:
            axs[k].text(0.95, 0.95, label, color=c,
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=axs[k].transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=.9)

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

def main(config_file=None, obs_id=None, verbose=0, test=False,
         force_source=None):
    """Entry point."""
    config = _get_config(config_file)

    logger = util.init_logger(__name__, 'make_uncal_beam_map: ')
    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    ctx = core.Context(config['context_file'])

    group_by = config['subobs'].get('use', 'detset')
    if group_by.startswith('dets:'):
        group_by = group_by.split(':',1)[1]

    if group_by == 'detset':
        groups = ctx.obsfiledb.get_detsets(obs_id)
    else:
        det_info = ctx.get_det_info(obs_id)
        groups = det_info.subset(keys=[group_by]).distinct()[group_by]

    if len(groups) == 0:
        logger.error(f'Invalid obs_id, "{obs_id}"')
        sys.exit(1)

    # Ignore some values?
    groups = [g for g in groups if g not in config['subobs'].get('ignore_vals', [])]

    if len(groups) == 0:
        logger.warning(f'No map groups found for obs_id={obs_id}')

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
        logger.info(f'Loading {obs_id}:{group_by}={group}')
        map_info = {'obs_id': obs_id,
                    'group': group,
                    group_by: group,
        }
        map_info['product_id'] = '{obs_id}-{group}'.format(**map_info)

        # Read data.
        tod = ctx.get_obs(obs_id, dets={group_by: group})
        if len(tod.signal) == 0:
            logger.warning(f'No detectors loaded for obs_id={obs_id}, '
                           f'group {group_by}={group}; skipping.')
            continue

        # Modify dets axis for testing
        if test:
            logger.warning(f'Decimating focal plane (--test).')
            dets = tod.dets.vals[::10]
            tod.restrict('dets', dets)

        # Modify samps axis for FFTs.
        logger.info(f' -- Before trimming, TOD shape is: {tod.shape}.')
        tod_ops.fft_trim(tod)
        logger.info(f' -- After trimming, TOD shape is:  {tod.shape}.')

        # Determine what source to map.
        ## To-do: have a mode where it maps all sources it finds.
        source_name = config['mapmaking'].get('force_source')
        if force_source is not None:
            source_name = force_source
        if source_name is None:
            sources = coords.planets.get_nearby_sources(tod)
            logger.info(f"Mappable sources in this region: {sources}")
            if len(sources) == 1:
                source_name = sources[0][0]
            elif len(sources) == 0:
                logger.error("No mappable sources in footprint.")
                sys.exit(1)
            else:
                logger.error("Multiple sources in footprint: %s" %
                             ([n for n, s in sources],))
                sys.exit(1)

        logger.info(f"Map will center on source: {source_name}")
        map_info['source_name'] = source_name

        # Plan to split on frequency band
        band_splits = coords.planets.load_detector_splits(tod, source=tod.det_info['band'])

        # Deconvolve readout filter and detector time constants.
        tod_ops.fft_trim(tod)
        tod_ops.detrend_tod(tod)

        filt = tod_ops.filters.identity_filter()
        if 'iir_params' in tod:
            filt *= tod_ops.filters.iir_filter(invert=True)
        if 'timeconst' in tod:
            filt *= tod_ops.filters.timeconst_filter(invert=True)

        tod.signal[:] = tod_ops.fourier_filter(tod, filt, resize=None, detrend=None)

        # Demodulation & downsampling.
        demodQU = None
        unmirror_det_angles = None
        comps = None
        if 'hwp_angle' in tod:
            logger.info('Demodulating HWP...')
            hwp.demod_tod(tod, signal_name='signal')
            demodQU = [tod['demodQ'], tod['demodU']]
            comps = 'T'

            # You should set this True only if your sim, or whatever,
            # did not account for the reflection effect of the HWP.
            unmirror_det_angles = config['preprocessing'].get('unmirror_det_angles', False)
            if unmirror_det_angles:
                logger.warning('Unmirroring detector angles, by request '
                               '(probably for a broken sim).')

        # Fix pointing.
        logger.info('Applying pointing corrections.')
        if 'boresight_offset' in tod:
            logger.info(' -- applying "boresight_offset".')
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
        res_deg = util.parse_quantity(reses[0], 'deg').value

        # Map size can be specified
        if 'map_size' in config['mapmaking']: # not quite right ...
            sizes = [
                util.lookup_conditional(config['mapmaking'], 'map_size', tags=[b])
                for b in band_splits.keys()]
            size_rad = util.parse_quantity(sizes[0], 'rad').value
        else:
            size_rad = None

        # Where to put things.
        policy = util.ArchivePolicy.from_params(
            config['archive']['policy'])

        dest_dir = policy.get_dest(**map_info)
        if os.path.exists(dest_dir):
            logger.info(f' -- destination already exists ({dest_dir})')
        else:
            logger.info(f' -- creating destination dir ({dest_dir})')
            os.makedirs(dest_dir)

        # Make the maps.
        logger.info(f'Calling mapmaker.')
        cuts = tod.get('glitch_flags', None)
        output = coords.planets.make_map(tod, center_on=source_name,
                                         res=res_deg*coords.DEG,
                                         size=size_rad,
                                         data_splits=band_splits,
                                         cuts=cuts,
                                         comps=comps,
                                         demodQU=demodQU,
                                         unmirror_det_angles=unmirror_det_angles,
                                         info=map_info,
                                         thread_algo='domdir')

        # Save and plot
        ocfg = config['output']
        pcfg = config['plotting']
        opattern = ocfg['pattern']
        ppattern = pcfg.get('pattern', opattern + '.png')

        used_names = []  # watch for accidental self-overwrites
        db_code = ocfg['map_codes'][0]

        for key, bundle in output['splits'].items():
            logger.info(f'Processing outputs for group={group}, band={key}...')
            # Save
            for code in ocfg['map_codes']:
                filename = os.path.join(
                    dest_dir, opattern.format(map_code=code, split=key, **map_info))
                logger.info(f' -- writing map to {filename}')
                bundle[code].write(filename)
                if filename in used_names:
                    logger.warning(f'Wrote this file more than once: {filename}; '
                                   f'using {key}, {code}')
                used_names.append(filename)
                if code == db_code:
                    # Index
                    db_data = {'obs:obs_id': obs_id,
                               'group': group,
                               'split': key}
                    db.add_entry(db_data, filename, replace=True)

            # Plot
            plot_filename = os.path.join(
                dest_dir, ppattern.format(map_code='solved', split=key, **map_info))
            logger.info(f' -- writing plots to {plot_filename}...')

            band_mask = (tod.det_info['band'] == key)
            zoom_size = util.parse_quantity(
                util.lookup_conditional(pcfg, 'zoom', tags=[key]),
                'deg').value

            plot_map(bundle, plot_filename, zoom_size=zoom_size,
                     tod=tod, det_mask=band_mask, group=group, subset=key)


    # Return something?
    return True


if __name__ == '__main__':
    util.main_launcher(main, get_parser)

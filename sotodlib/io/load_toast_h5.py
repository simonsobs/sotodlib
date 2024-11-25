"""TOD loader functions, within context system, for HDF5 format output
from TOAST (especially TOAST 3 circa 2023).

"""

import so3g
import numpy as np

from .. import core, coords


def load_toast_h5_obs(db, obs_id, dets=None, samples=None, prefix=None,
                      no_signal=None,
                      **kwargs):
    """Obsloader function for TOAST HDF5 output TODs.  Based on output
    from toast_so_sim.py, SSO sims, toast 3.0.0a9.

    See API template, `sotodlib.core.context.obsloader_template`, for
    details.

    """
    if prefix is None:
        prefix = db.prefix
        if prefix is None:
            prefix = './'

    # Get a ResultSet<[dets:detset,dets:readout_id], N rows>
    props = db.get_det_table(obs_id)
    if dets is None:
        detsets = props.subset(keys=['dets:detset']).distinct()['dets:detset']
    elif len(dets) == 0:
        props = props.subset(rows=[0])
        detsets = props['dets:detset']
    else:
        # Cross-match.
        vals, i0, i1 = core.util.get_coindices(props['dets:readout_id'], dets)
        if len(i1) != len(dets):
            raise ValueError('Some dets not found.')
        # Implicated detsets?
        detsets = props.subset(rows=i0, keys=['dets:detset']).distinct()['dets:detset']

    files_by_detset = db.get_files(obs_id, detsets=detsets)

    components = []
    for detset, files in files_by_detset.items():
        assert(len(files) == 1)
        dets = props.subset(rows=(props['dets:detset'] == detset))['dets:readout_id']
        components.append(
            load_toast_h5_file(
                files[0][0], dets=dets, no_signal=no_signal, samples=samples))
    if len(components) > 1:
        msg = "This loader only supports loading detectors from the same wafer"
        raise RuntimeError(msg)
    return components[0]

def load_toast_h5_dichroic_hack(db, obs_id, dets=None, samples=None, prefix=None,
                                no_signal=None,
                                **kwargs):
    """Obsloader function for TOAST HDF5 output TODs, in the specific case
    that each detset contains detectors for two bands and those
    detectors are distributed into two separate HDF5 files.

    See API template, `sotodlib.core.context.obsloader_template`, for
    details.

    """
    if prefix is None:
        prefix = db.prefix
        if prefix is None:
            prefix = './'

    # Get a ResultSet<[dets:detset,dets:readout_id], N rows>
    props = db.get_det_table(obs_id)
    if dets is None:
        detsets = props.subset(keys=['dets:detset']).distinct()['dets:detset']
    elif len(dets) == 0:
        props = props.subset(rows=[0])
        detsets = props['dets:detset']
    else:
        # Cross-match.
        vals, i0, i1 = core.util.get_coindices(props['dets:readout_id'], dets)
        if len(i1) != len(dets):
            raise ValueError('Some dets not found.')
        # Implicated detsets?
        detsets = props.subset(rows=i0, keys=['dets:detset']).distinct()['dets:detset']

    # Filter dets by band ...
    band_dets = {b: [] for b in ['f090', 'f150']}
    for d in props['dets:readout_id']:
        for b, dest in band_dets.items():
            if b in d:
                dest.append(d)
                break

    bands_here = sorted([b for b, d in band_dets.items() if len(d)])
    files_by_detset = db.get_files(obs_id, detsets=detsets)
    base_band = [b for b in bands_here if b in list(files_by_detset.items())[0][1][0][0]][0]

    components = []
    for detset, files in files_by_detset.items():
        assert(len(files) == 1)
        detset_dets = props.subset(rows=(props['dets:detset'] == detset))['dets:readout_id']
        for band in bands_here:
            band_file = files[0][0].replace(base_band, band)
            _dets = [d for d in detset_dets if d in band_dets[band]]
            components.append(
                load_toast_h5_file(
                    band_file, dets=_dets, no_signal=no_signal, samples=samples))

    if len(components) > 1:
        return components[0].concatenate(components, axis='dets', other_fields='first')

    return components[0]

def load_toast_h5_file(filename, dets=None, samples=None, no_signal=False,
                       enhance=False):
    """Reads data from a single HDF5 file.  Returns an AxisManager with
    standard SO field names.  Supports load_toast_h5_obs.

    If enhance=True, then certain metadata from the file are also
    processed and included: "focal_plane".

    """
    import h5py

    with h5py.File(filename, 'r') as f:
        dets_in_file = [v['name'].decode('ascii') for v in f['instrument']['focalplane']]

        if dets is None:
            dets = dets_in_file
            dets_mode = 'all'
        elif len(dets) == 0:
            dets_mode = 'none'
        else:
            #  Allow that dets might be identical to dets_in_file...
            vals, i0, i1 = core.util.get_coindices(dets_in_file, dets)
            if len(i0) != len(dets) or np.any(i1 != i0):
                raise ValueError('dets should be None or [].')
            dets = dets_in_file
            dets_mode = 'all'

        count = f['detdata']['signal'].shape[1]

        aman = core.AxisManager(
            core.LabelAxis('dets', dets),
            core.OffsetAxis('samps', count, 0),
        )

        aman.wrap_new('timestamps', ('samps', ), dtype='float64')

        bman = core.AxisManager(aman.samps.copy())
        bman.wrap('az', f['shared']['azimuth'], [(0, 'samps')])
        bman.wrap('el', f['shared']['elevation'], [(0, 'samps')])
        bman.wrap('roll', 0*bman.az, [(0, 'samps')])
        aman.wrap('boresight', bman)

        aman.timestamps[:] = f['shared']['times']
        if dets_mode == 'all':
            if not no_signal:
                aman.wrap_new('signal', ('dets', 'samps'), dtype='float32')
                aman.signal[:] = f['detdata']['signal']
        else:
            assert(len(dets) == 0)

        if 'hwp_angle' in f['shared']:
            aman.wrap('hwp_angle', f['shared']['hwp_angle'][()], [(0, 'samps')])

        # For the LAT corotator angle / SAT boresight rotation --
        # store each one, but also process them into a boresight
        # "roll".

        if 'corotator_angle' in f['shared']:
            a = f['shared']['corotator_angle'][()]
            aman.wrap('corotator_angle', a, [(0, 'samps')])

            # When uncorrected by the corotator, increasing elevation
            # causes footprint on sky to rotate clockwise.
            aman.boresight.roll[:] = (aman.boresight.el - np.radians(60) - a)

        if 'boresight_angle' in f['shared']:
            a = f['shared']['boresight_angle'][()]
            aman.wrap('boresight_angle', a, [(0, 'samps')])

            # Positive boresight angle corresponds to
            # counter-clockwise rotation of projection on sky, which
            # is a negative roll angle.
            aman.boresight.roll[:] = -a

        # Save the full toast fp info -- lots of interesting stuff in there.
        aman.wrap('toast_focalplane', f['instrument']['focalplane'], [(0, 'dets')])

    if enhance:
        # Write out equivalent sotodlib det pos.
        q = coords.ScalarLastQuat(aman.toast_focalplane['quat']).to_g3()
        xi, eta, gamma = so3g.proj.quat.decompose_xieta(q)
        fp = core.AxisManager(aman.dets)
        for key, value in zip(['xi', 'eta', 'gamma'], [xi, eta, gamma]):
            fp.wrap_new(key, shape=('dets', ))[:] = value
        aman.wrap('focal_plane', fp)

    if samples is not None:
        aman.restrict('samps', samples)

    return aman


core.OBSLOADER_REGISTRY['toast3-hdf-dichroic-hack'] = load_toast_h5_obs

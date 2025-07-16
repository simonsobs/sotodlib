import logging
import math
import numpy as np

import so3g

from . import utils
from . import configcls as cc


logger = logging.getLogger(__name__)


def _smooth_mask(mask, clean, extend):
    r = so3g.RangesInt32.from_mask(mask)
    return (~(~r).buffer(clean)).buffer(extend)


def get_scan_params(enc_az, cmd_az, fail_on_return=False):
    """Compute scan parameters based on fast-sampled az encoder data,
    and slowly sampled "commanded az" info.

    Args:
      enc_az: tuple
        The fast-sampled position readings. Tuple contains vector of
        timestamps, and vector of az readings in deg.
      cmd_az: tuple
        The slow-sampled commanded azimuth values.
      fail_on_return: bool
        If True, raise an exception on solution failure (instead of
        logging an error and returning the data structure).

    Returns:
      dict
        Fields are:
        - ``vel``: The scan speed during constant velocity portions,
          deg/s.
        - ``accel``: The mean acceleration of the turn-arounds,
          deg/s/s.
        - ``az_drift``: The drift velocity (deg/s), for scans where
          the turn-around positions are changing with time.
        - ``solved``: True if the scan properties were extracted
          sensibly, false otherwise.
        - ``data_found``: True if the data were non-empty, so solution
          could be attempted; False indicates that one of the data
          fields was empty.

    """
    if fail_on_return:
        def fail_func(msg):
            raise RuntimeError(msg)
    else:
        def fail_func(msg):
            logger.info(msg)

    no_data = {
        'vel': math.nan,
        'accel': math.nan,
        'az_drift': math.nan,
        'type': -1,
        'turntype': -1,
        'solved': False,
        'data_found': False,
    }
    unsolved = no_data | {'data_found': True}
    solved = unsolved | {'solved': True}

    if len(enc_az[0]) == 0:
        fail_func('Scan analysis aborted due to zero encoder points.')
        return no_data
    if len(cmd_az) == 0:
        fail_func('Scan analysis aborted due to zero commanded points.')
        return no_data

    # Bulk classification from udp stream.
    n_smooth = 10
    _n = 2 * n_smooth + 1
    az = np.convolve(enc_az[1], np.ones(_n) / _n, mode='valid')
    t = enc_az[0][n_smooth:-n_smooth]
    t00 = t[0]

    v = np.gradient(az) / np.gradient(t)

    still = _smooth_mask(abs(v) < .01, 5, 10)
    if still.mask().sum() / still.count > .99:
        fail_func('Scan analysis determines platform mostly not moving.')
        return solved | {
            'vel': 0.,
            'accel': 0.,
            'az_drift': 0.,
        }

    pos = _smooth_mask(v > .05, 5, 10)
    neg = _smooth_mask(v < -.05, 5, 10)

    # Classify each sweep.
    csweeps0 = []
    for rr in [pos, neg]:
        for r in rr.ranges():
            sl = slice(*r)
            if np.dot(t[sl][[0, -1]], [-1, 1]) < 2:
                continue # short scans hard to analyze
            az_mid = az[sl].mean()
            vtyp = np.median(v[sl])
            stable_mask = abs(v[sl] - vtyp) < .05
            stable = stable_mask.sum() / (r[1] - r[0])
            # Fit line.
            const_vel = stable > 0.5
            if const_vel:
                #_t0 = t[sl][stable_mask].mean()
                _ta, _tb = t[sl][stable_mask][[0, -1]]
                _t0, _dt = (_ta + _tb) / 2, (_tb - _ta)
                p = np.polyfit(t[sl][stable_mask] - _t0,
                               az[sl][stable_mask], 1)
            else:
                _t0, _dt, p = None, None, None
            csweeps0.append(utils.denumpy((r.tolist(), const_vel, vtyp, az_mid, _t0, _dt, p)))

    # Screen the sweeps on whether vel is typical -- helpful for
    # rejecting a first or last leg with odd properties.
    vtyps = [abs(sw[2]) for sw in csweeps0 if sw[1]]
    vtyp = np.median(vtyps)
    accept = max(vtyp * .05, .1)

    csweeps = [sw for sw in csweeps0 if sw[1] and abs(abs(sw[2]) - vtyp) < accept]
    csweeps.sort()

    # Characterize the typical leg separation in time, so we can more
    # easily recognize adjacent legs.
    if len(csweeps) > 3:
        leg_sep_time = [sw[4] for sw in csweeps]
        leg_sep_time = np.median(np.diff(leg_sep_time))
    else:
        leg_sep_time = None

    # Under the assumption of *constant* turn-around time, you can
    # figure out if there's an az_drift.  Just check the intersection
    # point of two adjacent sweeps; see if those corners evolve in
    # time.
    points = []
    for sw0, sw1 in zip(csweeps[:-1], csweeps[1:]):
        if not (sw0[1] and sw1[1]):
            continue
        if leg_sep_time and abs((sw1[4] - sw0[4]) / leg_sep_time - 1) > .2:
            continue
        if (sw0[2] * sw1[2] > 0):
            continue
        # At what time and az do the two trajectories intersect?
        #     y0 = m0 (t - t0) + b0
        #     y1 = m1 (t - t1) + b1
        # ->
        #     m0 (t - t0) + b0 = m1 (t - t0 + (t0 - t1)) + b1
        #     (m0 - m1) (t - t0) = m1(t0 - t1) + b1 - b0
        #     (t - t0) = (m1 (t0 - t1) + (b1 - b0)) /  (m0 - m1)
        t0, _, (m0, b0) = sw0[4:7]
        t1, _, (m1, b1) = sw1[4:7]
        t_t0 = (m1 * (t0 - t1) + (b1 - b0)) / (m0 - m1)
        y0 = m0 * t_t0 + b0
        points.append(utils.denumpy((np.sign(sw0[2]), t_t0 + t0, y0, m0, m1)))

    if len(points) == 0:
        fail_func('Scan analysis aborted due to no turn-around points.')
        return unsolved

    # Measure az_drift.
    sgn, _t, _y, _, _ = np.transpose(points)
    drifts = []
    for _sgn in [-1, 1]:
        s = (sgn == _sgn)
        if s.sum() == 0:
            continue
        if s.sum() == 1:
            drifts.append((0., _y[sgn==_sgn][0]))
        else:
            p = np.polyfit(_t[sgn==_sgn] - t00, _y[sgn==_sgn], 1)
            drifts.append(p)

    az_drift, az_center = np.mean(drifts, axis=0)

    # Loop over turn-arounds again and get adjusted values for az_cmd.
    if len(_t) > 1:
        sweep_time = np.median(np.diff(_t))
    else:
        sweep_time = csweeps[0][-2]

    window = sweep_time * .2
    cmd_t, cmd_az = cmd_az
    cmd_lims = []
    for tw in _t:
        s = (abs(cmd_t - tw) < window)
        if not s.any():
            continue
        mn, mx = cmd_az[s].min(), cmd_az[s].max()
        mn -= az_drift * (tw - t00)
        mx -= az_drift * (tw - t00)
        cmd_lims.extend(utils.denumpy([mn, mx]))

    if len(cmd_lims) == 0:
        fail_func('Scan analysis could not figure out commanded limits.')
        return unsolved

    xaz_min, xaz_max = np.min(cmd_lims).item(), np.max(cmd_lims).item()

    # Now use those to get the turn-around time of each sweep.
    props = []
    for p in points:
        sgn, tw, y0, m0, m1 = p
        y0 = y0 - az_drift * (tw - t00)
        az_turn = xaz_min if m0<0 else xaz_max
        tt0 = (y0 - az_turn) / m0
        tt1 = (az_turn - y0) / m1
        t_turnaround = tt0 + tt1
        a_turnaround = -sgn * (m1 - m0) / t_turnaround
        props.append((sgn*m0, a_turnaround))

    vel, accel = np.mean(props, axis=0).round(4)

    return utils.denumpy(solved | {
        'vel': vel,
        'accel': accel,
        'az_drift': az_drift.round(6),
        'type': 0,
        'turntype': 0,
    })


@cc.register_engine('scan-props', cc.ScanPropsConfig)
class ScanProps(utils.HkExtract):
    _fields = [
        ('vel', 'float'),
        ('accel', 'float'),
        ('az_drift', 'float'),
        ('type', 'int'),
        ('turntype', 'int'),
        ('solved', 'bool'),
        ('data_found', 'bool'),
    ]

    def getter(self, targets=None, results=None, **kwargs):
        time_ranges = self._target_time_ranges(targets)

        for time_range in time_ranges:
            data = self._load(time_range)
            yield get_scan_params(data['udp_az'],
                                  data['cmd_az'])

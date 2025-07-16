import logging
import math
import numpy as np

import so3g

from . import utils


logger = logging.getLogger(__name__)


def _smooth_mask(mask, clean, extend):
    r = so3g.RangesInt32.from_mask(mask)
    return (~(~r).buffer(clean)).buffer(extend)


def get_scan_props(udp_az, cmd_az):
    """
    Yup, get them.
    """
    # Bulk classification from udp stream.
    n_smooth = 10
    _n = 2 * n_smooth + 1
    az = np.convolve(udp_az[1], np.ones(_n) / _n, mode='valid')
    t = udp_az[0][n_smooth:-n_smooth]
    t00 = t[0]

    v = np.gradient(az) / np.gradient(t)

    still = _smooth_mask(abs(v) < .01, 5, 10)
    if still.mask().sum() / still.count > .99:
        return {
            'vel': 0.,
            'accel': math.nan,
            'az_drift': 0.,
        }

    pos = _smooth_mask(v > .05, 5, 10)
    neg = _smooth_mask(v < -.05, 5, 10)

    # Classify each sweep.

    csweeps = []
    for rr in [pos, neg]:
        for r in rr.ranges():
            sl = slice(*r)
            az_mid = az[sl].mean()
            vtyp = np.median(v[sl])
            stable_mask = abs(v[sl] - vtyp) < .02
            stable = stable_mask.sum() / (r[1] - r[0])
            #print(vtyp, stable, az_mid)
            # Fit line.
            const_vel = stable > 0.8
            if const_vel:
                #print('  ', az[sl][0], az[sl][-1])
                _t0 = t[sl][stable_mask].mean()
                p = np.polyfit(t[sl][stable_mask] - _t0,
                               az[sl][stable_mask], 1)
            else:
                _t0, p = None, None
            csweeps.append(utils.denumpy((r.tolist(), const_vel, vtyp, az_mid, _t0, p)))

    csweeps.sort()

    # Under the assumption of *constant* turn-around time, you can figure out if there's an az_drift.  Just check the intersection point of two adjacent sweeps; see if those corners evolve in time.
    points = []
    for sw0, sw1 in zip(csweeps[:-1], csweeps[1:]):
        if not (sw0[1] and sw1[1]):
            continue
        assert(sw0[2] * sw1[2] < 0)
        # At what time and az do the two trajectories intersect?
        #     y0 = m0 (t - t0) + b0
        #     y1 = m1 (t - t1) + b1
        # ->
        #     m0 (t - t0) + b0 = m1 (t - t0 + (t0 - t1)) + b1
        #     (m0 - m1) (t - t0) = m1(t0 - t1) + b1 - b0
        #     (t - t0) = (m1 (t0 - t1) + (b1 - b0)) /  (m0 - m1)
        t0, (m0, b0) = sw0[4:6]
        t1, (m1, b1) = sw1[4:6]
        t_t0 = (m1 * (t0 - t1) + (b1 - b0)) / (m0 - m1)
        y0 = m0 * t_t0 + b0
        points.append(utils.denumpy((np.sign(sw0[2]), t_t0 + t0, y0, m0, m1)))

    if len(points) == 0:
        logger.info('Scan analysis aborted due to no turn-around points.')
        return {
            'vel': math.nan,
            'accel': math.nan,
            'az_drift': math.nan,
        }

    # Measure az_drift.
    sgn, _t, _y, _, _ = np.transpose(points)
    drifts = []
    for _sgn in [-1, 1]:
        p = np.polyfit(_t[sgn==_sgn] - t00, _y[sgn==_sgn], 1)
        drifts.append(p)

    az_drift, az_center = np.mean(drifts, axis=0)

    # Loop over turn-arounds again and get adjusted values for az_cmd.
    sweep_time = np.median(np.diff(_t))
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
        #print(t_turnaround, a_turnaround)
        props.append((sgn*m0, a_turnaround))

    vel, accel = np.mean(props, axis=0).round(4)

    return utils.denumpy({
        'vel': vel,
        'accel': accel,
        'az_drift': az_drift.round(6),
    })


class ScanProps(utils.HkExtract):
    DEFAULTS = {
        'aliases': {
            'cmd_az': 'acu.acu_status.Azimuth_commanded_position',
            'cur_az': 'acu.acu_status.Azimuth_current_position',
            'udp_az': 'acu.acu_udp_stream.Corrected_Azimuth',
        },
    }

    def getter(self, targets=None, results=None, **kwargs):
        time_ranges = self._target_time_ranges(targets)

        for time_range in time_ranges:
            data = self._load(time_range)
            yield get_scan_props(data['udp_az'],
                                 data['cmd_az'])

# Copyright (c) 2024-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of coords.pointing_model.

"""

import itertools
import unittest
import numpy as np

from sotodlib import coords, core
from so3g.proj import quat
from pixell import enmap

pm = coords.pointing_model

DEG = np.pi/180
ARCMIN = DEG / 60

def to_rad(*arg):
    return [a * DEG for a in arg]

def to_deg(*arg):
    return [a / DEG for a in arg]

def full_vectors(az=0., el=0., roll=0.):
    z = 0 * az + 0*el + 0*roll
    return z + az, z + el, z + roll

def center_branch(x, period=360., center=0):
    # Shift values in x by multiples of period, so they're within
    # period/w of center.
    branch = center - period / 2
    return (x - branch) % period + branch

def quick_focal_plane(size, count):
    # Returns focal_plane AxisManager with count^2 detectors within a
    # size x size (deg) footprint.
    _x = np.linspace(-size/2, size/2, count) * DEG
    xi = _x[:,None] * 0 +_x
    eta = xi.T
    xi, eta = xi.ravel(), eta.ravel()
    det_count = len(xi)
    fp = core.AxisManager(core.LabelAxis('dets', ['d%02i' % i for i in range(det_count)]))
    pm._update_focal_plane(fp, xi, eta, xi*0, in_place=True)
    return fp

def bigness(*args):
    return sum([(a**2).mean() for a in args])**.5

def update_pmodel(tod, **kw):
    if 'pointing_model' not in tod:
        tod.wrap('pointing_model', core.AxisManager())
    for k, v in kw.items():
        tod['pointing_model'].wrap(k, v)


class CoordsUtilsTest(unittest.TestCase):
    def test_populate(self):
        # Test that model populates tod.
        tod0 = core.AxisManager(core.OffsetAxis('samps', 1000))
        ancil = core.AxisManager(tod0.samps)
        for k, v in zip(['az', 'el', 'boresight'],
                        full_vectors(np.linspace(0, 360, tod0.samps.count), 60, 0)):
            ancil.wrap_new(f'{k}_enc', shape=('samps',))[:] = v
        tod0.wrap('ancil', ancil)
        tod0.wrap('obs_info', core.AxisManager())
        tod0['obs_info'].wrap('telescope', 'satp1')

        results = []
        # TOD without 'pointing_model' -- should work, with warning
        tod = tod0.copy()
        with self.assertLogs(pm.__name__, level='WARN'):
            pm.apply_pointing_model(tod)
        results.append(tod.boresight)
        del tod['boresight']

        update_pmodel(tod, version='sat_v1')
        pm.apply_pointing_model(tod, wrap_boresight=False)
        assert 'boresight' not in tod

        pm.apply_pointing_model(tod)
        assert 'boresight' in tod
        assert 'az' in tod.boresight
        results.append(tod.boresight)

        pm.apply_pointing_model(tod)

        # Check consistency...
        for r in results[1:]:
            for k in ['az', 'el', 'roll']:
                d = center_branch(r[k] - results[0][k], 2*np.pi)
                np.testing.assert_array_almost_equal(d, 0*d)

        # Check that focal_plane / focal_plane_template are treated
        # as desired.
        fp0 = quick_focal_plane(3., 5)
        for fp_val, to_put, to_check in [
                (None, 'focal_plane_template', 'focal_plane'),
                (fp0, 'focal_plane_template', 'focal_plane'),
                (None, 'focal_plane', 'focal_plane_template'),
                (fp0, 'focal_plane', 'focal_plane_template'),
        ]:
            tod = tod0.copy()
            update_pmodel(tod, version='sat_v1')
            tod.wrap(to_put, fp_val)
            pm.apply_pointing_model(tod)
            if fp_val is None:
                assert tod[to_check] is None
            else:
                for k in ['xi', 'eta', 'gamma']:
                    np.testing.assert_array_equal(tod[to_put][k], tod[to_check][k])

        # Check specifically that stale focal_plane is discarded in
        # favor of template.
        fp1 = fp0.copy()
        fp1.xi += .01
        tod = tod0.copy()
        update_pmodel(tod, version='sat_v1')
        tod.wrap('focal_plane_template', fp0)
        tod.wrap('focal_plane', fp1)
        assert abs(fp0.xi - fp1.xi).mean() > .005

        pm.apply_pointing_model(tod)
        np.testing.assert_array_equal(tod.focal_plane_template.xi,
                                      tod.focal_plane.xi)

    def test_populate_distortions(self):
        # Test that model populates tod.
        tod0 = core.AxisManager(core.OffsetAxis('samps', 1000))
        ancil = core.AxisManager(tod0.samps)
        for k, v in zip(['az', 'el', 'corotator'],
                        full_vectors(np.linspace(0, 360, tod0.samps.count), 60, 0)):
            ancil.wrap_new(f'{k}_enc', shape=('samps',))[:] = v
        tod0.wrap('ancil', ancil)
        tod0.wrap('obs_info', core.AxisManager())
        tod0['obs_info'].wrap('telescope', 'satp1')

        # Test that no-dist models do not require focal_plane*
        tod = tod0.copy()
        update_pmodel(tod, version='lat_v2')
        pm.apply_pointing_model(tod)

        # Test that dist models do require focal_plane{,_template}
        tod = tod0.copy()
        update_pmodel(tod,
                      version='lat_v2',
                      roll_dist_model=2)
        with self.assertRaises(Exception):
            pm.apply_pointing_model(tod)

        # This should cause focal_plane_template to be created from
        # focal_plane.
        tod = tod0.copy()
        update_pmodel(tod,
                      version='lat_v2',
                      roll_dist_model=2)
        tod.wrap('focal_plane', quick_focal_plane(3., 20))
        pm.apply_pointing_model(tod)
        assert 'focal_plane_template' in tod

        # This should create a copy of focal_plane_template in focal_plane
        tod = tod0.copy()
        update_pmodel(tod,
                      version='lat_v2',
                      roll_dist_model=2)
        tod.wrap('focal_plane_template', quick_focal_plane(3., 20))
        pm.apply_pointing_model(tod)
        assert 'focal_plane' in tod
        fp1 = tod.focal_plane.copy()

        # But non-trivial...
        assert np.any(fp1.xi != tod.focal_plane_template.xi)

        # But idempotent...
        pm.apply_pointing_model(tod)
        np.testing.assert_array_equal(fp1.xi, tod.focal_plane.xi)

    def test_sat_v1(self):
        # Test model_sat_v1 general behaviors.
        az, el, roll = full_vectors(az=np.linspace(-90, 90, 100),
                                    el=60)
        params = core.AxisManager()
        params.wrap('version', 'sat_v1')
        az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll))[0])
        params0 = {'version': 'sat_v1'}
        az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll))[0])

        params = dict(params0)
        params['enc_offset_az'] = 1. * DEG
        az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll))[0])
        assert np.all(center_branch(az1 - az) > 0)

        def tpoint_base_tilt(an, aw, az, el):
            c, s = np.cos(az * DEG), np.sin(az * DEG)
            delta_az = (-aw * c - an * s) * np.tan(el * DEG)
            delta_el = aw * s - an * c
            return az + delta_az / DEG, el + delta_el / DEG

        # Trial a few different base tilt combinations.
        az, el, roll = full_vectors(az=np.arange(0, 360), el=30.)
        tpoint_err_max = 0.01 # deg
        for (bt_c, bt_s, high_az) in [
                (1 * DEG, 0 * DEG, 0.),
                (0 * DEG, 1 * DEG, 270.),
                (2**-0.5 * DEG, 2**-0.5 * DEG, 315.),
        ]:
            params = dict(params0)
            params.update({
                'base_tilt_cos': bt_c,
                'base_tilt_sin': bt_s,  # West up.
                })
            az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll))[0])

            # Check that delta el, near the expected high_az point, is postive.
            d_el = el1 - el
            s = abs(center_branch(az - high_az)) < 10.
            assert (all(d_el[s]) > 0)

            # Check consistency with tpoint linear approximation --
            # bt_c and bt_s correspond to AN and AW terms,
            # respectively.
            az2, el2 = tpoint_base_tilt(bt_c, bt_s, az, el)
            d_az = abs(center_branch(az2 - az1))
            d_el = abs(center_branch(el2 - el1))
            assert all(d_az < tpoint_err_max)
            assert all(d_el < tpoint_err_max)

        # Boresight center.
        params = dict(params0)
        params.update({
            'fp_rot_xi0': 0.4 * DEG,
            'fp_rot_eta0': 0.1 * DEG,
            #'fp_offset_xi0': 0.05 * DEG,
            #'fp_offset_eta0': 0.01 * DEG,
            })
        az, el, roll = full_vectors(roll=np.linspace(0., 360., 20), el=45.)
        for xi, eta, sig in [
                (0.4, 0.1, 0.),
                (0.5, 0.2, 0.1*2**.5),
        ]:
            az1, el1, roll1 = pm.model_sat_v1(params, *to_rad(az, el, roll))[0]
            # Measure az, el on sky of a detector (xi, eta):
            q_hs1 = (quat.rotation_lonlat(-az1, el1, roll1) *
                     quat.rotation_xieta(xi * DEG, eta * DEG))
            neg_az2, el2, roll2 = quat.decompose_lonlat(q_hs1)
            d_el = el2 - el * DEG
            d_azc = center_branch(-neg_az2 - az*DEG, 2*np.pi) * np.cos(el*DEG)
            sig_meas = (d_el.std()**2 + d_azc.std()**2)**.5
            assert(abs(sig_meas - sig * DEG) < .001*DEG)


    def test_lat_v1(self):
        az, el, roll = full_vectors(az=np.linspace(-90, 90, 100),
                                    el=60)
        params = {'version': 'lat_v1'}
        az1, el1, roll1 = to_deg(*pm.model_lat_v1(params, *to_rad(az, el, roll))[0])

        # Backwards compatibility for not providing el sag or base tilt
        params['enc_offset_az'] = 1. * DEG
        az1, el1, roll1 = to_deg(*pm.model_lat_v1(params, *to_rad(az, el, roll))[0])
        assert np.all(center_branch(az1 - az) > 0)

        # Try an el sag
        params['el_sag_lin'] = 1
        az2, el2, roll2 = to_deg(*pm.model_lat_v1(params, *to_rad(az, el, roll))[0])
        assert np.all(np.isclose(el/el2, 2))


    def test_lat_dist_model(self):
        """Targeted tests of just the focal plane distortion function."""
        fp0 = quick_focal_plane(6, 20)
        az, el, rollz = [np.zeros(1000) + _x * DEG for _x in
                        [45, 60, 0]]

        # These all mean "no correction".
        for params in [
                {},
                {'roll_dist_model': 0},
                {'roll_dist_model': 1,
                 'arc_amp':        0.,
                 'arc_r0':         3.1913387e-02,
                 'arc_roll0':      90 * DEG,
                 },
                {'roll_dist_model': 1,
                 'arc_amp':        0.0001,
                 'arc_r0':         3.1913387e-02,
                 'arc_roll0':      0,
                 },
        ]:
            fp1 = pm.apply_lat_distortion_model(params, az, el, rollz, fp0, in_place=False)
            dx, dy = fp1.xi - fp0.xi, fp1.eta - fp0.eta
            assert bigness(dx, dy) < .01 * ARCMIN

        # Model 1 - empirical correction for non-linear secondary effects.
        params = {
            'roll_dist_model': 1,
            'arc_amp':        1.0083065e-04,
            'arc_r0':         3.1913387e-02,
            'arc_roll0':      0., # 3.4087353e-02,
            }
        for roll, roll0, big_x, big_y in [
                (0., 0., 0, 0),
                (90., 0., 1, 1),
                (180., 0., 1, 1),
                (0., 90., 1, 1),
        ]:
            params['arc_roll0'] = roll0 * DEG
            fp1 = pm.apply_lat_distortion_model(params, az, el, rollz + roll,
                                                fp0, in_place=False)
            dx, dy = fp1.xi - fp0.xi, fp1.eta - fp0.eta
            assert (not big_x) ^ (bigness(dx) >= .1 * ARCMIN)
            assert (not big_y) ^ (bigness(dy) >= .1 * ARCMIN)

        # Model 2 - secondary distortion model from ray tracing. No
        # params other than to set roll_dist_model.
        params = {
            'roll_dist_model': 2,
        }
        fp1 = pm.apply_lat_distortion_model(params, az, el, rollz, fp0, in_place=False)

        dx, dy = fp1.xi - fp0.xi, fp1.eta - fp0.eta
        assert bigness(dx, dy) > .1 * ARCMIN

        # Make sure model actually seems to change with roll...
        results = [fp0]
        for roll in [0., 30 * DEG, 180 * DEG]:
            fp1 = pm.apply_lat_distortion_model(params, az, el, rollz + roll,
                                                fp0, in_place=False)
            # Must differ from all previous results...
            for fp in results:
                assert bigness(fp1.xi - fp.xi, fp1.eta - fp.eta) > .1 * ARCMIN
            results.append(fp1)


if __name__ == '__main__':
    unittest.main()

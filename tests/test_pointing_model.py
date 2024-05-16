# Copyright (c) 2024 Simons Observatory.
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

def to_rad(*arg):
    return [a * DEG for a in arg]

def to_deg(*arg):
    return [a / DEG for a in arg]

def full_vectors(az=0., el=0., roll=0.):
    z = 0 * az + 0*el + 0*roll
    return z + az, z + el, z + roll


class CoordsUtilsTest(unittest.TestCase):
    def test_populate(self):
        # Test that model populates tod.
        tod = core.AxisManager(core.OffsetAxis('samps', 1000))
        ancil = core.AxisManager(tod.samps)
        for k, v in zip(['az', 'el', 'boresight'],
                        full_vectors(np.linspace(0, 360, tod.samps.count), 60, 0)):
            ancil.wrap_new(f'{k}_enc', shape=('samps',))[:] = v
        tod.wrap('ancil', ancil)
        tod.wrap('obs_info', core.AxisManager())
        tod['obs_info'].wrap('telescope', 'satp1')

        results = []
        # TOD without 'pointing_model' -- should work, with warning
        with self.assertLogs(pm.__name__, level='WARN'):
            pm.apply_pointing_model(tod)
        results.append(tod.boresight)
        del tod['boresight']

        tod.wrap('pointing_model', core.AxisManager())
        tod['pointing_model'].wrap('version', 'sat_v1')
        pm.apply_pointing_model(tod, wrap=False)
        assert 'boresight' not in tod

        pm.apply_pointing_model(tod)
        assert 'boresight' in tod
        assert 'az' in tod.boresight
        results.append(tod.boresight)

        pm.apply_pointing_model(tod)

        # Check consistency...
        for r in results[1:]:
            for k in ['az', 'el', 'roll']:
                d = (r[k] - results[0][k] + np.pi) % (2*np.pi) - np.pi
                np.testing.assert_array_almost_equal(d, 0*d)

    def test_sat_v1(self):
        # Test model_sat_v1 general behaviors.
        az, el, roll = full_vectors(az=np.linspace(-90, 90, 100),
                                    el=60)
        params = core.AxisManager()
        params.wrap('version', 'sat_v1')
        az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll)))
        params0 = {'version': 'sat_v1'}
        az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll)))

        params = dict(params0)
        params['enc_offset_az'] = 1.
        az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll)))
        assert np.all((az1 - az + 180) % 360 - 180 > 0)

        params = dict(params0)
        params.update({
            'base_tilt_cos': 0 * DEG,
            'base_tilt_sin': 1 * DEG,  # West up.
            })

        for az0, ex in [
                (0.,  0),
                (90., -1),
                (180., 0),
                (270., +1),
        ]:
            az, el, roll = full_vectors(az=np.linspace(-1, 1, 5) + az0, el=30.)
            az1, el1, roll1 = to_deg(*pm.model_sat_v1(params, *to_rad(az, el, roll)))
            d_el = el1 - el
            if ex == 0:
                assert any(d_el > 0) and any(d_el < 0)
            else:
                assert all(d_el * ex > 0)

        # Boresight center.
        params = dict(params0)
        params.update({
            'bs_xi0': 0.4 * DEG,
            'bs_eta0': 0.1 * DEG,
        })
        az, el, roll = full_vectors(roll=np.linspace(0., 360., 20), el=45.)
        for xi, eta, sig in [
                (0.4, 0.1, 0.),
                (0.5, 0.2, 0.1*2**.5),
        ]:
            az1, el1, roll1 = pm.model_sat_v1(params, *to_rad(az, el, roll))
            # Measure az, el on sky of a detector (xi, eta):
            q_hs1 = (quat.rotation_lonlat(-az1, el1, roll1) *
                     quat.rotation_xieta(xi * DEG, eta * DEG))
            neg_az2, el2, roll2 = quat.decompose_lonlat(q_hs1)
            d_el = el2 - el * DEG
            d_azc = ((-neg_az2 - az*DEG - np.pi) % (2 * np.pi) + np.pi) * np.cos(el*DEG)
            sig_meas = (d_el.std()**2 + d_azc.std()**2)**.5
            assert(abs(sig_meas - sig * DEG) < .001*DEG)


if __name__ == '__main__':
    unittest.main()

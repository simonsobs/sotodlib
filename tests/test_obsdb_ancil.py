import unittest
from unittest.mock import patch
from contextlib import contextmanager
from collections import Counter

from sotodlib.core import metadata
from sotodlib.io import ancil
from sotodlib.site_pipeline import update_obsdb_ancil as uoa
import so3g

import numpy as np
import os
import tempfile
import yaml

from ._helpers import skip_if_mpi


@ancil.configcls.dataclass
class _ExampleEngineConfig(ancil.configcls.LowResTableConfig):
    dataset_name: str = 'example-dataset'
    obsdb_query: str = '{pwv} is null'


@ancil.configcls.register_engine('example-dataset', _ExampleEngineConfig)
class _ExampleEngine(ancil.utils.LowResTable):
    _fields = [('pwv', 'float')]

    def __init__(self, cfg):
        super().__init__(cfg)
        self._data_ranges = so3g.IntervalsDouble()
    def update_base(self, time_range, reset=False):
        self._data_ranges.add_interval(*time_range)
    def _load(self, time_range):
        rs = metadata.ResultSet(keys=['timestamp', 'pwv'])
        data_ranges = self._data_ranges * so3g.IntervalsDouble().add_interval(*time_range)
        for t0, t1 in data_ranges.array():
            tt = np.arange(t0, t1, 60.)
            pwv = np.random.uniform(0, 3, size=len(tt))
            for row in zip(tt, pwv):
                rs.rows.append(tuple(row))
        return rs
    def getter(self, targets=None, **kwargs):
        time_ranges = self._target_time_ranges(targets)
        for time_range in time_ranges:
            rs = self._load(time_range)
            yield ancil.utils.denumpy({
                'pwv': np.mean(rs['pwv'])})


_t0 = 1700000000.
time_ranges = {
    k: (_t0 + start, _t0 + end)
    for k, start, end in [
            ('universe',        0, 1000000),
            ('first-up',        0,  100000),
            ('second-up',  100000,  200000),
            ('first-obs',   10000,   11000),
            ('second-obs', 210000,  211000),
    ]
}


def get_example(stuff_missing=False):
    # Create a new Db and add two columns.
    obsdb = metadata.ObsDb()
    obsdb.add_obs_columns(['timestamp float', 'start_time float', 'stop_time float',
                           'pwv float', 'other_pwv float'])

    for k in ['first-obs', 'second-obs']:
        tr = time_ranges[k]
        obsdb.update_obs(k, {'timestamp': tr[0],
                             'start_time': tr[0],
                             'stop_time': tr[1],
                             })
    return obsdb

@contextmanager
def _get_test_env():
    with tempfile.TemporaryDirectory() as tempd:
        cfile = os.path.join(tempd, 'ancil.yaml')
        with open(cfile, 'w') as fout:
            fout.write(yaml.dump({
                'target_obsdb': os.path.join(tempd, 'obsdb0.sqlite'),
                'datasets': {
                    'example-dataset': {},
                    'other-dataset': {
                        'dataset_name': 'other_dataset',
                        'class': 'example-dataset',
                        'obsdb_format': 'other_{field}',
                    },
                },
            }))
        yield cfile


@skip_if_mpi
class TestObsDbAncil(unittest.TestCase):
    def test_00_basic(self):
        # Check config
        c = _ExampleEngineConfig()
        d = _ExampleEngine({})
        assert d.cfg.dataset_name == 'example-dataset'

        d = ancil.utils.get_engine('example-dataset', {})
        assert d.cfg.dataset_name == 'example-dataset'

        d = ancil.utils.get_engine('other-dataset', {'class': 'example-dataset',
                                                     'dataset_name': 'other-dataset'})
        assert d.cfg.dataset_name == 'other-dataset'

    def test_10_api(self):
        with _get_test_env() as cfile:
            cfg = yaml.safe_load(open(cfile, 'r'))

            # Engine instantiation
            engines = {k: ancil.utils.get_engine(k, v)
                       for k, v in cfg['datasets'].items()}

            # Base data update
            for k, eng in engines.items():
                eng.update_base(time_ranges['first-up'])

            # Compute reduced results
            recs = []
            for k in ['first-obs', 'second-obs']:
                tr = time_ranges[k]
                recs.append({
                    'obs_id': k,
                    'start_time': tr[0],
                    'stop_time': tr[1],
                })

            for k, eng in engines.items():
                results = eng.collect(recs)
                assert np.isfinite(results[0]['pwv'])
                assert not np.isfinite(results[1]['pwv'])

            # Update an obsdb
            obsdb0 = get_example()
            for outstanding in [2, 1]:
                for k, eng in engines.items():
                    q = eng.obsdb_query()
                    recs = obsdb0.query(q)
                    assert len(recs) == outstanding
                    results = eng.collect(recs, for_obsdb=True)
                    for rec, result in zip(recs, results):
                        obsdb0.update_obs(rec['obs_id'], result)

@contextmanager
def _get_module_test_env():
    with tempfile.TemporaryDirectory() as tempd:
        cfile = os.path.join(tempd, 'ancil.yaml')
        with open(cfile, 'w') as fout:
            fout.write(yaml.dump({
                'target_obsdb': os.path.join(tempd, 'obsdb0.sqlite'),
                'data_prefix': tempd,
                'datasets': {
                    # PWV, default settings...
                    'apex-pwv': {},
                    'toco-pwv': {},
                    'pwv-combo': {},
                    # Test output directory combos
                    'apex1': {
                        'class': 'apex-pwv',
                        'data_prefix': tempd + '/prefix',
                    },
                    'apex2': {
                        'class': 'apex-pwv',
                        'data_dir': 'apex_qwv',
                    },
                    'apex3': {
                        'class': 'apex-pwv',
                        'data_prefix': tempd + '/prefix',
                        'data_dir': 'apex_qwv',
                    },
                },
                'job_defs': [
                    {'name': 'basic',
                     'steps': [
                         {'command': 'update-base-data',
                          'time_range': list(time_ranges['universe'])},
                         {'command': 'update-obsdb',
                          'time_range': list(time_ranges['universe'])},
                     ],
                     },
                ],
            }))
        yield cfile


@skip_if_mpi
class TestAncilModules(unittest.TestCase):

    # Set expectations for data validity.
    update_range = time_ranges['universe']
    data_max_time = time_ranges['first-up'][1]

    def _run_module_test(self, dataset_name, friends=[]):
        output = {}
        with _get_module_test_env() as cfile:
            cfg = uoa._get_config(cfile)
            engines = {k: ancil.utils.get_engine(k, v)
                       for k, v in cfg['datasets'].items()}
            assert dataset_name in engines

            # Base data update
            for k, eng in engines.items():
                eng.register_friends(cfg['datasets'])
                if k in [dataset_name] + friends:
                    eng.update_base(self.update_range)

            # Update the db.
            obsdb0 = get_example()
            for k, eng in engines.items():
                if k not in [dataset_name] + friends:
                    continue
                eng.obsdb_check(obsdb0, create_cols=True)
                q = eng.obsdb_query()
                recs = obsdb0.query(q)
                results = eng.collect(recs, for_obsdb=True)
                for rec, result in zip(recs, results):
                    obsdb0.update_obs(rec['obs_id'], result)

                if k == dataset_name:
                    output[k] = list(zip(recs, results))
        return output

    def _check_vals(self, results, expectations, key):
        for (rec, result), expect in zip(results, expectations, strict=True):
            if expect == 'finite':
                assert np.isfinite(result[key])
            elif expect == 'not finite':
                assert not np.isfinite(result[key])
            elif expect == '':
                pass
            else:
                raise ValueError(f"Malformed expectation: '{expect}'")

    def test_PwvApex(self):
        data_func = ancil.apex.ApexDataMocker(t_max=self.data_max_time).get_raw
        with patch('sotodlib.io.ancil.apex.ApexPwv._get_raw', data_func):
            results = self._run_module_test('apex-pwv')
            self._check_vals(results['apex-pwv'], ['finite', 'not finite'], 'apex_pwv_mean')

    def test_PwvToco(self):
        data_func = ancil.so_hk.TocoPwvMocker(t_max=self.data_max_time).get_raw
        with patch('sotodlib.io.ancil.so_hk.TocoPwv._get_raw', data_func):
            results = self._run_module_test('toco-pwv')
            self._check_vals(results['toco-pwv'], ['finite', 'not finite'], 'toco_pwv_mean')

    def test_PwvCombo(self):
        data_func1 = ancil.apex.ApexDataMocker(t_max=self.data_max_time).get_raw
        data_func2 = ancil.so_hk.TocoPwvMocker(t_max=self.data_max_time).get_raw
        with \
             patch('sotodlib.io.ancil.apex.ApexPwv._get_raw', data_func1), \
             patch('sotodlib.io.ancil.so_hk.TocoPwv._get_raw', data_func2):
            results = self._run_module_test('pwv-combo', friends=['apex-pwv', 'toco-pwv'])
            self._check_vals(results['pwv-combo'], ['finite', 'not finite'], 'pwv_mean')

    def test_cli(self):
        """Test for update_obsdb_ancil."""
        data_func1 = ancil.apex.ApexDataMocker(t_max=self.data_max_time).get_raw
        data_func2 = ancil.so_hk.TocoPwvMocker(t_max=self.data_max_time).get_raw
        with \
             patch('sotodlib.io.ancil.apex.ApexPwv._get_raw', data_func1), \
             patch('sotodlib.io.ancil.so_hk.TocoPwv._get_raw', data_func2):
            with _get_module_test_env() as cfile:
                cfg = uoa._get_config(cfile)
                obsdb0 = get_example()
                obsdb0.to_file(cfg['target_obsdb'])
                uoa.main('update-base-data', config_file=cfile,
                         time_range=self.update_range)
                uoa.main('update-obsdb', config_file=cfile,
                         time_range=self.update_range)
                # At the end of it all, one row should be fully
                # populated and one row should remain nan.
                obsdb1 = metadata.ObsDb.from_file(cfg['target_obsdb'])
                rows = obsdb1.query()
                for f in ['apex_pwv_mean', 'toco_pwv_mean', 'pwv_mean']:
                    c = Counter(np.isfinite(rows[f]))
                    assert c[True] == 1
                    assert c[False] == 1
                # The following files should all exist...
                d0 = os.path.split(cfile)[0]
                for subloc in ['apex_pwv', 'prefix/apex_pwv', 'apex_qwv', 'prefix/apex_qwv']:
                    assert os.path.exists(
                        os.path.join(d0, subloc, 'apex_pwv_1700000000.h5'))



    def test_cli_jobs(self):
        """Test for update_obsdb_ancil."""
        data_func1 = ancil.apex.ApexDataMocker(t_max=self.data_max_time).get_raw
        data_func2 = ancil.so_hk.TocoPwvMocker(t_max=self.data_max_time).get_raw
        with \
             patch('sotodlib.io.ancil.apex.ApexPwv._get_raw', data_func1), \
             patch('sotodlib.io.ancil.so_hk.TocoPwv._get_raw', data_func2):
            with _get_module_test_env() as cfile:
                cfg = uoa._get_config(cfile)
                obsdb0 = get_example()
                obsdb0.to_file(cfg['target_obsdb'])

                for job in cfg['job_defs']:
                    print(job['name'])
                    uoa.main('run-job', config_file=cfile, job_name=job['name'])

                # At the end of it all, one row should be fully
                # populated and one row should remain nan.
                obsdb1 = metadata.ObsDb.from_file(cfg['target_obsdb'])
                rows = obsdb1.query()
                for f in ['apex_pwv_mean', 'toco_pwv_mean', 'pwv_mean']:
                    c = Counter(np.isfinite(rows[f]))
                    assert c[True] == 1
                    assert c[False] == 1

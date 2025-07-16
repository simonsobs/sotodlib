import unittest
from contextlib import contextmanager

from sotodlib.core import metadata
from sotodlib.io import ancil
from sotodlib.site_pipeline import update_obsdb_ancil as uoa
import so3g

import numpy as np
import os
import tempfile
import yaml

from ._helpers import mpi_multi


@ancil.configcls.dataclass
class _ExampleEngineConfig(ancil.configcls.LowResTableConfig):
    dataset_name: str = 'example-dataset'
    obsdb_query: str = '{pwv} is null'


@ancil.configcls.register_engine('example-dataset', _ExampleEngineConfig)
class _ExampleEngine(ancil.utils.LowResTable):
    result_fields = ['pwv']

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
                'ancil_base': tempd,
                'source_obsdb': os.path.join(tempd, 'obsdb0.sqlite'),
                'dest_obsdb': os.path.join(tempd, 'obsdb1.sqlite'),
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


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestObsDbAncil(unittest.TestCase):
    def test_00_basic(self):
        # Check config
        c = _ExampleEngineConfig()
        d = _ExampleEngine({})
        assert d.cfg.dataset_name == 'example-dataset'

        d = ancil.utils.get_engine('example-dataset', {})
        assert d.cfg.dataset_name == 'example-dataset'

        d = ancil.utils.get_engine('other-dataset', {'class': 'example-dataset'})
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
                    q = eng._get_obsdb_query()
                    recs = obsdb0.query(q)
                    assert len(recs) == outstanding
                    results = eng.collect(recs, for_obsdb=True)
                    for rec, result in zip(recs, results):
                        obsdb0.update_obs(rec['obs_id'], result)

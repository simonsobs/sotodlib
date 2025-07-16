import logging
import os
import time

import h5py
import numpy as np
from tqdm import tqdm

import flacarray as fa
FA = False

from sotodlib.core.metadata import ResultSet
from sotodlib.io import metadata, hkdb


logger = logging.getLogger(__name__)


def denumpy(output):
    if isinstance(output, dict):
        return {k: denumpy(v)
                for k, v in output.items()}
    if isinstance(output, list):
        return [denumpy(x) for x in output]
    if isinstance(output, tuple):
        return tuple((denumpy(x) for x in output))
    if isinstance(output, np.generic):
        return output.item()
    return output


def intersect_ranges(r1, r2, empty_as_none=False):
    if r2[0] < r1[0]:
        r1, r2 = r2, r1
    if r1[1] < r2[0]:
        out = (r1[0], r1[0])
    else:
        out = (max(r1[0], r2[0]), min(r1[1], r2[1]))
    if out[0] == out[1] and empty_as_none:
        return None
    return out


def merge_rs(rs0, rs1):
    """Merge rows from rs1 into rs0 (both ResultSets), sorting by field
    'timestamp'.  Where timestamps are equal r1 values are ignored.

    """
    i0, i1 = 0, 0
    G0 = [(t, -1, i) for i, t in enumerate(rs0['timestamp'])]
    G1 = [(t, i, -1) for i, t in enumerate(rs1['timestamp'])]
    G = sorted(G0 + G1)

    # De-duplicate
    rows_out = []
    last_t = None,
    for t, i1, i0 in G:
        if t == last_t:
            continue
        if i0 >= 0:
            rows_out.append(rs0.rows[i0])
        else:
            rows_out.append(rs1.rows[i1])
        last_t = rows_out[-1][0]
    rs0.rows = rows_out



def get_engine(key, cfg):
    from . import ANCIL_ENGINES
    if cfg.get('class') is not None:
        key = cfg['class']
    cls = ANCIL_ENGINES[key]
    return cls(**cfg)


class AncilEngine:
    DEFAULTS = {}

    def __init__(self, **cfg):
        self.cfg = self.DEFAULTS | cfg
        if self.cfg.get('friends'):
            self.friends = {k: None for k in cfg['friends']}

    def register_friends(self, friend_configs):
        for k in self.friends.keys():
            self.friends[k] = friend_configs[k]

    def _get_friend(self, friend, as_engine=True):
        if not self.friends.get(friend):
            raise RuntimeError(f'Friend not configured: "{friend}"')
        if as_engine:
            return get_engine(friend, self.friends[friend])
        return self.friends[friend]

    def update_base(self, time_range=None, reset=False):
        """Update the base dataset, for the indicated time_range.

        If reset is True, then new download / computation replaces all
        data in the time_range.

        """
        pass

    def getter(self, targets=None, results=None):
        raise NotImplementedError()

    def collect(self, targets=None, results=None, show_pbar=True):
        return list(self.getter(targets=tqdm(targets, disable=not show_pbar),
                                results=results))

    def _target_obs_ids(self, targets):
        for t in targets:
            yield t['obs_id']

    def _target_time_ranges(self, targets):
        for t in targets:
            yield (t['start_time'], t['stop_time'])


class LowResTable(AncilEngine):
    """Helper class for archiving fairly low resolution data into a set of
    HDF5 files, as the "base data".

    """
    _DEFAULTS = {
        'archive_block_seconds': 2000000,
        'big_bang': 1704000000,
        'now': None,
        'data_dir': None,  # Set this to identify the dataset.
        'gap_size': 300,
        'filename_pattern': '{dataset_name}_{timestamp}.h5',
        'dtypes': None,
    }

    def __init__(self, **cfg):
        self.DEFAULTS = self._DEFAULTS | self.DEFAULTS
        super().__init__(**cfg)

        if self.cfg['now'] is None:
            self.cfg['now'] = time.time()
        if self.cfg['data_dir'] is None:
            self.cfg['data_dir'] = self.cfg['dataset_name'] + '/'
        if self.cfg['data_prefix'] is not None:
            self.cfg['data_dir'] = os.path.join(self.cfg['data_prefix'],
                                                self.cfg['data_dir'])

    def _get_raw(self, time_range):
        raise NotImplemented

    def update_base(self, time_range=None, reset=False):
        """Attempt to patch any gaps in the dataset, for time_range, by
        grabbing data from the source.

        If reset is True, then the grabbed data replaces any archived
        data in that time_range.
        
        See _get_raw for the specific activity of the subclass.

        """
        if time_range is None:
            time_range = self.cfg['big_bang'], self.cfg['now']

        dataset = self.cfg['dataset_name']

        for t0, t1, filename in self._get_filenames(time_range):
            if not os.path.exists(filename) or reset:
                rs0 = ResultSet(keys=['timestamp', 'pwv'])
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            else:
                rs0 = metadata.read_dataset(filename, dataset) 

            if reset:
                gap_ranges = [time_range]
            else:
                # Are there gaps that cross the target range?
                super_t = np.hstack((t0, rs0['timestamp'], t1))
                gap_t = np.diff(super_t)

                gaps = (gap_t >= self.cfg['gap_size']).nonzero()[0]
                gap_ranges = [(super_t[i], super_t[i+1]) for i in gaps]
                gap_ranges = list(filter(lambda x: x[1] > x[0],
                              [intersect_ranges(time_range, g)
                               for g in gap_ranges]))

            if len(gap_ranges) == 0:
                continue
                
            # One query for this whole thing ...
            query_range = (gap_ranges[0][0], gap_ranges[-1][1])
            logger.info(f'Pulling {dataset} data for %s (%.1f, %1.f) ...' % (
                filename, query_range[0], query_range[1]))
            rs = self._get_raw((query_range[0], query_range[1]))

            # Filter down to only data that's inside a gap range.
            t = rs['timestamp']
            mask = np.zeros(len(t), bool)
            for t0, t1 in gap_ranges:
                mask += (t0 <= t) * (t < t1)
            rs = rs.subset(rows=mask)

            # Merge into rs0.
            before = len(rs0)
            merge_rs(rs0, rs)
            after = len(rs0)
            logger.info(' ... row count %i -> %i' % (before, after))
            if after > before:
                logger.info(' ... writing new %s' % filename)
                rs0_a = rs0.asarray(dtypes=self.cfg['dtypes'])
                metadata.write_dataset(rs0_a, filename, dataset, overwrite=True)

    def _get_filenames(self, time_range):
        ns = int(self.cfg['archive_block_seconds'])
        t0 = int((time_range[0] // ns) * ns)
        t1 = time_range[1]
        rows = []
        while t1 > t0:
            fn = self.cfg['filename_pattern'].format(
                timestamp=t0,
                dataset_name=self.cfg['dataset_name'])
            fn = f'{self.cfg["data_dir"]}/{fn}'
            rows.append((max(t0, time_range[0]), min(t1, t0 + ns), fn))
            t0 += ns
        return rows

    def _load(self, time_range):
        rs = ResultSet(keys=['timestamp', 'pwv'])
        for t0, t1, filename in self._get_filenames(time_range):
            if os.path.exists(filename):
                _rs = metadata.read_dataset(filename, self.cfg['dataset_name'])
                rs.rows.extend(_rs.rows)
        t = rs['timestamp']
        s = (time_range[0] <= t) * (t < time_range[1])
        if not np.all(s):
            rs = rs.subset(rows=s)
        return rs

    def getter(self, targets=None, **kwargs):
        raise NotImplemented


class HkExtract(AncilEngine):
    """Helper class for storing extracts from HK data.

    """
    _DEFAULTS = {
        'archive_block_seconds': 2000000,
        'dataset_block_seconds': 100000,
        'big_bang': 1704000000,
        'now': None,
        'data_dir': None,  # Set this to identify the dataset.
        'filename_pattern': '{dataset_name}_{timestamp}.h5',
        'default_dtype': 'float32',
        'dtypes': {},
        'aliases': None,
    }

    def __init__(self, **cfg):
        self.DEFAULTS = self._DEFAULTS | self.DEFAULTS
        super().__init__(**cfg)

        if self.cfg['now'] is None:
            self.cfg['now'] = time.time()
        if self.cfg['data_dir'] is None:
            self.cfg['data_dir'] = self.cfg['dataset_name'] + '/'
        if self.cfg['data_prefix'] is not None:
            self.cfg['data_dir'] = os.path.join(self.cfg['data_prefix'],
                                                self.cfg['data_dir'])

    def _clear_dataset(self, filename, dataset):
        with h5py.File(filename, 'a') as hin:
            if dataset in hin:
                del hin[dataset]

    def _get_datasets(self, time_range):
        """Return a list of all (file, dataset) pairs that overlap with
        time_range, along with the overlap.

        """
        ns = int(self.cfg['archive_block_seconds'])
        nb = int(self.cfg['dataset_block_seconds'])
        assert (ns % nb == 0)
        t0 = int((time_range[0] // ns) * ns)
        t1 = time_range[1]
        rows = []
        while t0 < t1:
            fn = self.cfg['filename_pattern'].format(
                timestamp=t0,
                dataset_name=self.cfg['dataset_name'])
            fn = f'{self.cfg["data_dir"]}/{fn}'
            sub_t1 = min(t0 + ns, t1)
            while t0 < sub_t1:
                ds = f't{t0}'
                if t0 + nb > time_range[0]:
                    rows.append((#max(t0, time_range[0]), min(t1, t0 + nb), fn, ds))
                        t0, t0 + nb, fn, ds))
                t0 += nb
        return rows

    def update_base(self, time_range=None, reset=False):
        """Update data by grabbing for HK archive.  Only data in time_range
        will be loaded, and even then only for times not confidently
        covered by the archive.  If reset=True then active data in the
        archive for this time_range is discarded, and fully replaced with
        newly loaded results.

        Each file has groups containing data for a specific time
        interval (probably ~1 day).  That group also specifies the
        complete_until time.  If complete_until is lower than the end
        time for the group, it is eligible to be updated.  To
        invalidate existing data for time_range, and definitely
        replace it, pass reset=True.

        You have to assume that "coverage" is a contiguous
        super-interval.  Once you get data at t, the coverage is
        extend to t.  If there's a gap, you can fill it using *reset*,
        which should not pay attention to coverage.

        Coverage interval is then a ~global archive value; or
        per-file?  If you do per-dataset, that's perhaps inefficient
        but fine.

        So -- for each dataset, check coverage time and read data
        starting from the end of that.

        """
        if time_range is None:
            time_range = self.cfg['big_bang'], self.cfg['now']
        else:
            time_range = (max(self.cfg['big_bang'], time_range[0]),
                          min(self.cfg['now'], time_range[1]))

        field_list = list(self.cfg['aliases'].keys())
        tbuf = 120.

        cfg = hkdb.HkConfig.from_yaml(self.cfg['hkdb_platform'])
        cfg.aliases = self.cfg['aliases']
        db = hkdb.HkDb(cfg)

        def insert_vector(origd, newd, time_range):
            if newd is not None:
                s = (time_range[0] <= newd[0]) * (newd[0] < time_range[1])
                if origd[0] is None:
                    return (newd[0][s], newd[1][s])
            else:
                if origd[0] is None:
                    return (None, None)
                newd = [[], []]
                s = slice(0, 0)
            left = (origd[0] < time_range[0])
            right = (origd[0] >= time_range[1])
            return tuple(np.hstack((o[left], n[s], o[right]))
                         for o, n in zip(origd, newd))

        for dst0, dst1, filename, dataset in self._get_datasets(time_range):
            # If reset, then we will load and replace any data in t0,
            # t1.  coverage may be modified if it's <= t1.

            # If not reset, then (t0, t1) is further restricted to
            # start at coverage; and that sets the new data insertion
            # window.  coverage may only be increased in this case.
            logger.info(f'{dst0} {dst1} {filename} {dataset}')

            vects = {k: (None, None) for k in field_list}
            complete_to = dst0
            if os.path.exists(filename):
                with h5py.File(filename, 'r') as hin:
                    if dataset in hin:
                        dset = hin[dataset]
                        if 'complete_to' in dset.attrs:
                            complete_to = dset.attrs['complete_to']
                        for k in field_list:
                            if k not in dset:
                                continue
                            try:
                                if not FA:
                                    vects[k] = (dset[k]['timestamp'][()],
                                              dset[k]['value'][()])
                                else:
                                    vects[k] = (fa.hdf5.read_array(dset[k]['timestamp']),
                                                fa.hdf5.read_array(dset[k]['value']))
                            except Exception as e:
                                logger.error(f'Invalid data in {filename} : {dataset}.')
                                raise e

            t0 = max(time_range[0], dst0, complete_to)
            t1 = min(time_range[1], dst1)
            if t0 >= t1:
                logger.info('No updates; archive complete for requested range.')
                continue

            # Load.
            lspec = hkdb.LoadSpec(
                cfg=cfg, start=t0 - tbuf, end=t1 + tbuf,
                fields=field_list,
                hkdb=db)

            logger.info(f' ... loading hk {t0}-{t1}')
            result = hkdb.load_hk(lspec)
            # Merge the data.
            output = {}
            new_complete_to = []
            for k in field_list:
                newd = getattr(result, k, None)
                if newd is not None:
                    new_complete_to.append(max(newd[0]))
                else:
                    new_complete_to.append(min(t0, complete_to))
                output[k] = insert_vector(vects[k], newd, (t0, t1))
                if output[k][0] is not None:
                    dtype = self.cfg['dtypes'].get(k)
                    if dtype is None:
                        dtype = self.cfg['default_dtype']
                    if dtype is not None:
                        output[k] = (output[k][0].astype('float64'),
                                     output[k][1].astype(dtype))
            new_complete_to = min(min(new_complete_to), t1)

            if t0 > complete_to:
                new_complete_to = complete_to

            # Write it out.
            logger.info(' ... writing %s : %s' % (filename, dataset))
            with h5py.File(filename, 'a') as hin:
                if dataset in hin:
                    del hin[dataset]
                g = hin.create_group(dataset)
                try:
                    g.attrs['complete_to'] = float(new_complete_to)
                    for k in field_list:
                        if output[k][0] is not None:
                            g1 = g.create_group(k)
                            if not FA:
                                g1.create_dataset('timestamp', data=output[k][0])
                                g1.create_dataset('value', data=output[k][1])
                            else:
                                fa.hdf5.write_array(output[k][0],
                                                g1.create_group('timestamp'),
                                                quanta=.005)
                                fa.hdf5.write_array(output[k][1],
                                                g1.create_group('value'),
                                                quanta=.001)
                                                
                            
                except Exception as e:
                    logger.error(f'Problem storing {dataset} in {filename}, removing')
                    raise e

    def _load(self, time_range):
        field_list = list(self.cfg['aliases'].keys())
        output = {k: [[], []] for k in field_list}
        
        for _, _, filename, dataset in self._get_datasets(time_range):
            if not os.path.exists(filename):
                continue
            with h5py.File(filename, 'r') as hin:
                if dataset not in hin:
                    continue
                ds = hin[dataset]
                for k in field_list:
                    if not FA:
                        t, v = ds[k]['timestamp'][()], ds[k]['value'][()]
                    else:
                        t, v = (fa.hdf5.read_array(ds[k]['timestamp']),
                                fa.hdf5.read_array(ds[k]['value']))
                    s = (time_range[0] <= t) * (t < time_range[1])
                    if s.any():
                        output[k][0].append(t[s])
                        output[k][1].append(v[s])

        for k in output.keys():
            if len(output[k][0]):
                output[k] = [np.hstack(x) for x in output[k]]
            else:
                output[k] = [np.zeros(0, dtype='float64'),
                             np.zeros(0, dtype='float32')]

        return output
                    
    def getter(self, targets=None, **kwargs):
        raise NotImplemented



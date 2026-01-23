import functools
import logging
import glob
import os
import string
import time

import h5py
import numpy as np
from tqdm import tqdm

from sotodlib.core.metadata import ResultSet, ObsDb
from sotodlib.io import metadata, hkdb


logger = logging.getLogger(__name__)


def denumpy(output):
    """Traverse a container, recursively, and convert any numpy
    scalars to generic python scalars. Note ndarrays are not converted
    -- the denumpification is only applied to scalars nested in
    standard python containers (dict/list/tuple).

    """
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
    """Return the intersection of semi-open intervals r1 = [a, b) and r2 =
    [c, d).  If empty_as_none, then empty intervals are returned as
    None instead of [e, e).

    """
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
    """Merge rows from rs1 into rs0 (both ResultSets), sorting rows by
    field 'timestamp'.  Because the primary application is to update
    some dataset based on expanding source data, rows that repeat some
    'timestamp' value are dropped, with precedence given to data from
    rs0.

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
    _cfg = {k: v for k, v in cfg.items()
            if k not in ['class']}
    if 'class' in cfg:
        class_name = cfg['class']
    else:
        class_name = key
    cls = ANCIL_ENGINES[class_name]
    return cls(_cfg)


def _get_time_range(*time_ranges, now=None):
    if now is None:
        now = time.time()
    for time_range in time_ranges:
        if time_range is None:
            continue
        return tuple(now if t is None else t for t in time_range)
    return (now, now)


class AncilEngine:
    config_class = None
    _fields = []

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = self.config_class(**cfg)
        self.cfg = cfg
        if self.cfg.friends:
            self.friends = {k: None for k in self.cfg.friends}
        else:
            self.friends = {}

    def register_friends(self, friend_configs):
        for k in self.friends.keys():
            self.friends[k] = friend_configs[k]

    def _get_friend(self, friend):
        if friend not in self.friends:
            raise RuntimeError(f'Friend not configured: "{friend}"')
        if isinstance(self.friends[friend], dict):
            return get_engine(friend, self.friends[friend])
        return self.friends[friend]

    @property
    def obsdb_fields(self):
        if self.cfg.obsdb_format:
            # transform (bare_field, ...) into (obsdb_field, ...)
            return [
                (self.cfg.obsdb_format.format(
                    dataset=self.cfg.dataset_name, field=item[0]),) + item[1:]
                for item in self._fields]
        return self._fields

    def _obsdb_map(self, target=None):
        if target:
            return {v[0]: target[k[0]]
                    for k, v in zip(self._fields, self.obsdb_fields)}
        return {k[0]: v[0] for k, v in zip(self._fields, self.obsdb_fields)}

    def obsdb_query(self, time_range=None, redo=False):
        """Get obsdb query string to identify records that need recomputation.
        If time_range is not None, then screening on timestamp will be
        included.  If redo is True, then the engine-specific value
        testing will be skipped and all records in the time_range will
        be queried.

        """
        if redo:
            vquery = '1'
        else:
            vquery = self.cfg.obsdb_query
            assert vquery is not None

        if time_range is not None:
            t0, t1 = time_range
            if t0 is None:
                tquery = f'timestamp < {t1}'
            elif t1 is None:
                tquery = f'timestamp >= {t0}'
            else:
                tquery = f'(timestamp >= {t0}) and (timestamp < {t1})'

            vquery = f'{tquery} and {vquery}'

        return vquery.format(**self._obsdb_map())

    def obsdb_check(self, obsdb, create_cols=False):
        ok = False
        try:
            obsdb.conn.execute('select %s from obs limit 1' % (
                ','.join(['`%s`' % f[0] for f in self.obsdb_fields])))
            ok = True
        except:
            pass
        if not ok and create_cols:
            for field_row in self.obsdb_fields:
                k, t = field_row[:2]
                obsdb.add_obs_columns([f'{k} {t}'])
            return True
        return ok

    def check_base(self):
        return {}

    def update_base(self, time_range=None, reset=False):
        """Update the base dataset, for the indicated time_range.

        If reset is True, then new download / computation replaces all
        data in the time_range.

        """
        pass

    def getter(self, targets=None, results=None):
        raise NotImplementedError()

    def collect(self, targets=None, results=None, show_pbar=True, for_obsdb=False):
        output = list(self.getter(targets=tqdm(targets, disable=not show_pbar),
                                  results=results))
        remap = self._obsdb_map()
        if for_obsdb and self.cfg.obsdb_format:
            for i in range(len(output)):
                output[i] = {remap[k]: v for k, v in output[i].items()}
        return output

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
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.data_dir is None:
            self.cfg.data_dir = self.cfg.dataset_name + '/'
        if self.cfg.data_prefix is not None:
            self.cfg.data_dir = os.path.join(self.cfg.data_prefix,
                                                self.cfg.data_dir)

    def _get_raw(self, time_range):
        raise NotImplementedError()

    def update_base(self, time_range=None, reset=False):
        """Attempt to patch any gaps in the dataset, for time_range, by
        grabbing data from the source.

        If reset is True, then the grabbed data replaces any archived
        data in that time_range.

        See _get_raw for the specific activity of the subclass.

        """
        time_range = _get_time_range(time_range, self.cfg.dataset_time_range)

        dataset = self.cfg.dataset_name

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

                gaps = (gap_t >= self.cfg.gap_size).nonzero()[0]
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
                rs0_a = rs0.asarray(dtypes=self.cfg.dtypes)
                metadata.write_dataset(rs0_a, filename, dataset, overwrite=True)

    def check_base(self):
        """Check the base data -- determine output directory, count files
        therein, take note of extra files, etc.

        """
        info = {
            'output_dir': self.cfg.data_dir,
            'files_found': 0,
        }
        time_range = _get_time_range(self.cfg.dataset_time_range)
        for t0, t1, filename in self._get_filenames(time_range):
            if os.path.exists(filename):
                info['files_found'] += 1
        return info

    def _get_filenames(self, time_range):
        ns = int(self.cfg.archive_block_seconds)
        t0 = int((time_range[0] // ns) * ns)
        t1 = time_range[1]
        rows = []
        while t1 > t0:
            fn = self.cfg.filename_pattern.format(
                timestamp=t0,
                dataset_name=self.cfg.dataset_name)
            fn = f'{self.cfg.data_dir}/{fn}'
            rows.append((max(t0, time_range[0]), min(t1, t0 + ns), fn))
            t0 += ns
        return rows

    def _load(self, time_range):
        # If you don't cache, the dataset reads can be very
        # inefficient, especially when looping over a bunch of obs
        # from the same time period.
        @functools.lru_cache(maxsize=4)
        def _get_dataset(filename, dataset):
            if not os.path.exists(filename):
                return None
            return metadata.read_dataset(filename, dataset)

        rs = None
        for t0, t1, filename in self._get_filenames(time_range):
            _rs = _get_dataset(filename, self.cfg.dataset_name)
            if _rs is not None:
                if rs is None:
                    rs = _rs
                else:
                    rs.rows.extend(_rs.rows)
        if rs is None:
            rs = ResultSet(keys=['timestamp', 'pwv'])
        t = rs['timestamp']
        s = (time_range[0] <= t) * (t < time_range[1])
        if not np.all(s):
            rs = rs.subset(rows=s)
        return rs

    def getter(self, targets=None, **kwargs):
        raise NotImplementedError()


class HkExtract(AncilEngine):
    """Helper class for storing extracts from HK data.

    """
    def __init__(self, cfg):
        super().__init__(cfg)

        if self.cfg.data_dir is None:
            self.cfg.data_dir = self.cfg.dataset_name + '/'
        if self.cfg.data_prefix is not None:
            self.cfg.data_dir = os.path.join(self.cfg.data_prefix,
                                                self.cfg.data_dir)

    def _clear_dataset(self, filename, dataset):
        with h5py.File(filename, 'a') as hin:
            if dataset in hin:
                del hin[dataset]

    def _get_datasets(self, time_range):
        """Return a list of all (file, dataset) pairs that overlap with
        time_range, along with the overlap.

        """
        ns = int(self.cfg.archive_block_seconds)
        nb = int(self.cfg.dataset_block_seconds)
        assert (ns % nb == 0)
        t0 = int((time_range[0] // ns) * ns)
        t1 = time_range[1]
        rows = []
        while t0 < t1:
            fn = self.cfg.filename_pattern.format(
                timestamp=t0,
                dataset_name=self.cfg.dataset_name)
            fn = f'{self.cfg.data_dir}/{fn}'
            sub_t1 = min(t0 + ns, t1)
            while t0 < sub_t1:
                ds = f't{t0}'
                if t0 + nb > time_range[0]:
                    rows.append((#max(t0, time_range[0]), min(t1, t0 + nb), fn, ds))
                        t0, t0 + nb, fn, ds))
                t0 += nb
        return rows

    def check_base(self):
        """Check the base data -- determine output directory, count files
        therein, take note of extra files, etc.

        """
        info = {
            'output_dir': self.cfg.data_dir,
            'files_found': 0,
        }
        time_range = _get_time_range(self.cfg.dataset_time_range)
        ds_count = {}
        for t0, t1, filename, _ in self._get_datasets(time_range):
            if filename in ds_count:
                if ds_count[filename] == -1:
                    continue
            else:
                if os.path.exists(filename):
                    ds_count[filename] = 0
                else:
                    ds_count[filename] = -1
        info['files_found'] = len([v for v in ds_count.values() if v == 0])
        return info

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
        # Validate archive time range...
        now = time.time()
        _tr = self.cfg.dataset_time_range
        _tr = ((0  if _tr[0] is None else _tr[0]),
               (now if _tr[1] is None else _tr[1]))
        time_range = _get_time_range(time_range, _tr)
        time_range = intersect_ranges(time_range, _tr)

        field_list = list(self.cfg.aliases.keys())
        tbuf = 120.

        cfg = hkdb.HkConfig.from_yaml(self.cfg.hkdb_config)
        cfg.aliases = self.cfg.aliases
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
                                vects[k] = (dset[k]['timestamp'][()],
                                          dset[k]['value'][()])
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
                    dtype = self.cfg.dtypes.get(k)
                    if dtype is None:
                        dtype = self.cfg.default_dtype
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
                            g1.create_dataset('timestamp', data=output[k][0])
                            g1.create_dataset('value', data=output[k][1])

                except Exception as e:
                    logger.error(f'Problem storing {dataset} in {filename}, removing')
                    raise e

    def _load(self, time_range):
        field_list = list(self.cfg.aliases.keys())
        output = {k: [[], []] for k in field_list}

        for _, _, filename, dataset in self._get_datasets(time_range):
            if not os.path.exists(filename):
                continue
            with h5py.File(filename, 'r') as hin:
                if dataset not in hin:
                    continue
                ds = hin[dataset]
                for k in field_list:
                    if k not in ds:
                        continue
                    t, v = ds[k]['timestamp'][()], ds[k]['value'][()]
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
        raise NotImplementedError()


def get_example_obsdb(start, end=None, step=3600, filename=None):
    obsdb = ObsDb()
    obsdb.add_obs_columns(['timestamp float', 'start_time float', 'stop_time float'])

    if end is None:
        end = start + step * 10
    for t in np.arange(start, end, step):
        obs_id = f'obs_{t:.0f}'
        obsdb.update_obs(obs_id, denumpy({'timestamp': t,
                                          'start_time': t,
                                          'stop_time': t + step,
                                          }))

    if filename is not None:
        obsdb.to_file(filename)
    return obsdb

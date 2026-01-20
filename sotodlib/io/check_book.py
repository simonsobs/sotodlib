"""check_book.py

This module contains a class for checking obs/oper Books (for internal
consistency, proper schema, etc).  The results can also be used to
help update an ObsFileDb.

"""

import so3g
from spt3g import core

import os
import fnmatch
import glob
import logging
import yaml

import numpy as np


logger = logging.getLogger(__name__)

# The default config for BookScanner.

_DEFAULT_CONFIG = {
    # File patterns ...
    'stream_file_pattern': 'D_{stream_id}_{index:03d}.g3',
    'ancil_file_pattern': 'A_ancil_{index:03d}.g3',

    # Schema ...
    'extra_files': ['M_index.yaml', 'M_book.yaml'],
    'extra_extra_files': [],
    'banned_files': [],

    # Work-arounds ...
    'sample_range_inclusive_hack': False,
    'require_ordered_detsets': True,
    'tolerate_sample_range_resetting': False,
    'tolerate_missing_ancil': False,
    'tolerate_missing_ancil_timestamps': False,
    'tolerate_timestamps_value_discrepancy': False,
    'tolerate_stray_files': False,
    'tolerate_missing_extra_files': False,
    'ignore_fixed_tones': False,
}
    

class _compact_list(list):
    # list with abbreviated repr.
    def __repr__(self):
        others = len(self) - 5
        if len(self) > 10:
            return '[' + ', '.join([repr(x) for x in self[:5]]) \
                + f', and {others} other items ...]'
        else:
            return super.__repr__(self)


class BookScanner:
    """The BookScanner helps to catalog the contents of an obs/oper book,
    validate that the contents look right, and produce entries for
    ObsFileDb.

    It is used like this::

      # Instantiate with path to a obs/oper Book.
      bs = BookScanner(book_dir)

      # Analyze book contents.
      bs.go()

      # Optional: print summary of warnings / errors to terminal.
      bs.report()

      # Check for errors.
      if len(bs.results['errors']):
          raise RuntimeError("This book contains errors.")

      # Get stuff for the obsfiledb (paths relative to ...)
      detset_rows, file_rows = bs.get_obsfiledb_info('/')

    """
    logger = logger  # cache module-level logger

    def __init__(self, book_dir, config={}, logger=None):
        self.config = dict(_DEFAULT_CONFIG, **config)
        self.book_dir = book_dir
        self.results = {
            'errors': [],
            'warnings': [],
            'det_lists': {},
            'sample_ranges': None,
            'ready': False,
            'metadata': None,
        }
        if logger is not None:
            self.logger = logger

    def go(self):
        """Scan a book for completeness and consistency; record misc. metadata
        to help with ObsFileDb registration.

        """
        # 1. Check metadata completeness.
        try:
            self.check_meta()
            self.gapfill_meta()
        except Exception as e:
            self.logger.error('Failed in section 1!')
            self.report()
            raise e

        # 2. Verify file presence
        try:
            self.check_file_presence()
        except Exception as e:
            self.logger.error('Failed in section 2!')
            self.report()
            raise e

        try:
            # 3. Check sample range, timestamp alignment, etc.
            self.check_frames()
        except Exception as e:
            self.logger.error('Failed in section 3!')
            self.report()
            raise e

    def _err(self, filename, msg, warn=False):
        if warn:
            self.results['warnings'].append((filename, msg))
        else:
            self.results['errors'].append((filename, msg))

    def _get_filename(self, basename, **kw):
        basename = basename.format(**kw)
        return os.path.join(self.book_dir, basename)

    def check_meta(self):
        """Extract essential metadata from the M_index.yaml.

        """
        err, warn = self.results['errors'], self.results['warnings']
        meta = yaml.safe_load(open(self._get_filename('M_index.yaml'), 'rb'))
        self.results['metadata'] = meta

        keys = ['book_id', 'type', 'telescope', 'stream_ids', 'sample_ranges']
        for k in keys:
            if k not in meta:
                self._err('M_index.yaml', f'Missing entry for "{k}".', warn=True)

    def gapfill_meta(self):
        """If some metadata are missing, try to guess them from the filenames
        and stuff.

        """
        meta = self.results['metadata']

        if 'book_id' not in meta:
            meta['book_id'] = os.path.split(self.book_dir)[1]
        tokens = meta['book_id'].split('_')  # type, time, tel, slot-flags
        if 'type' not in meta:
            meta['type'] = tokens[0]
        if 'telescope' not in meta:
            meta['telescope'] = tokens[2]
        if 'stream_ids' not in meta:
            assert 'stream_ids' in self.config  # needed to patch missing stream_id info
            meta['stream_ids'] = []
            for sid, f in zip(self.config['stream_ids'], tokens[3]):
                if f == '0':
                    continue
                elif f == '1':
                    meta['stream_ids'].append(sid)
                else:
                    self._err(
                        None, f'Invalid stream_flags in book name: {tokens[3]}.')

        if 'detsets' not in meta:
            assert 'detset_map' in self.config
            meta['detsets'] = [self.config['detset_map'][k]
                               for k in meta['stream_ids']]

        # Check that detsets and stream_ids are same length.
        assert len(meta['detsets']) == len(meta['stream_ids'])

        # Don't check sample_ranges here, do it later when we're
        # reading the files for per-frame sample ranges.

    def check_file_presence(self):
        """See if all expected files are there; compare number of files in
        each file group.

        """
        strays = glob.glob(self._get_filename('*'))
        for basename in self.config['extra_files'] + self.config['extra_extra_files']:
            fn = self._get_filename(basename)
            try:
                strays.remove(fn)
            except ValueError:
                self._err(basename, 'Expected "extra file" and did not find it.',
                          warn=self.config['tolerate_missing_extra_files'])

        meta = self.results['metadata']
        file_counts = []

        for stream_id in meta['stream_ids'] + ['ancil']:
            if stream_id == 'ancil':
                pattern = self.config['ancil_file_pattern']
            else:
                pattern = self.config['stream_file_pattern']
            index = 0
            while True:
                fn = self._get_filename(pattern, stream_id=stream_id, index=index)
                if not os.path.exists(fn):
                    break
                strays.remove(fn)
                index += 1
            file_counts.append(index)

        file_count = file_counts[0]
        if any([f != file_count for f in file_counts]):
            self._err(None, f'Inconsistent file counts: {file_counts}')

        _file_count = file_count if 'sample_ranges' not in meta \
                      else len(meta['sample_ranges'])
        if _file_count != file_count:
            self._err(None, f'File counts {file_counts} do not agree with '
                      'sample_ranges entry {_file_count}')
        meta['file_count'] = file_count
        if len(strays):
            self._err(None, f'Extra files found in book dir: {strays}',
                      warn=self.config['tolerate_stray_files'])
            for bf in self.config['banned_files']:
                # interpret as a pattern
                for banned in fnmatch.filter(strays, self._get_filename(bf)):
                    self._err(banned, f'File is banned from book by pattern: {bf}')

    def check_frames(self):
        """Read the "Scan" frames from all files and check structure /
        consistency.

        """
        meta = self.results['metadata']
        to_check = ['ancil'] + meta['stream_ids']

        timestamps_master = None
        sample_ranges = []
        for stream_id in to_check:
            if stream_id == 'ancil':
                pattern = self.config['ancil_file_pattern']
            else:
                pattern = self.config['stream_file_pattern']

            timestamps = []
            start, end = 0, None
            hack_offset = 0
            dets = None

            for index in range(meta['file_count']):
                filename = self._get_filename(pattern, stream_id=stream_id, index=index)
                basename = os.path.split(filename)[1]
                for frame in core.G3File(filename):
                    if frame.type == core.G3FrameType.Scan:
                        a, b = list(frame['sample_range'])
                        if self.config['sample_range_inclusive_hack']:
                            # Change (0, 999) into (0, 1000).
                            b = b + 1  # schema hack

                        if end is None:
                            # This is first frame in new file and
                            # "start" is expected first sample index.
                            if a == 0 and start != 0:
                                self._err(basename, 'sample_range is resetting on file boundaries.',
                                          warn=self.config['tolerate_sample_range_resetting'])
                                hack_offset = start
                            if a != start - hack_offset:
                                self._err(basename, f'sample_range entries are not abutting: '
                                          f'[..., {start - hack_offset}] -> [{a}, {b}].')
                                raise RuntimeError()
                        else:
                            # This is not the first frame, "end" from
                            # last frame should match start of this one.
                            if a != end - hack_offset:
                                self._err(basename, f'sample_range entries are not abutting: '
                                          f'[..., {end - hack_offset}] -> [{a}, {b}].')
                                raise RuntimeError()
                        end = b + hack_offset

                        if 'ancil' in frame:
                            timestamps.append(np.asarray(frame['ancil'].times))
                        else:
                            self._err(basename, f'No "ancil" entry',
                                      warn=self.config['tolerate_missing_ancil'])
                            if 'primary' in frame:
                                timestamps.append(np.asarray(frame['primary'].times))

                        if 'signal' in frame and dets is None:
                            dets = _compact_list(frame['signal'].names)

                # Wrap up this file
                if index >= len(sample_ranges):
                    sample_ranges.append((start, end))
                else:
                    if sample_ranges[index] != (start, end):
                        self._err(basename, f'sample_range discrepancy, {(start, end)} '
                                  f'instead of {sample_ranges[index]}')

                # For next file, requeue expectations.
                start, end = end, None

            # Check / record stuff for the stream_id.
            self.results['det_lists'][stream_id] = dets

            if len(timestamps) == 0:
                self._err(None, f'No timestamps in fileset.',
                          warn=(stream_id == 'ancil' and self.config['tolerate_missing_ancil_timestamps']))
            else:
                timestamps = np.hstack(timestamps)
                if timestamps_master is None:
                    timestamps_master = timestamps
                else:
                    if len(timestamps) != len(timestamps_master):
                        self._err(None, f'Timestamps length discrepancy.')
                    elif not (timestamps == timestamps_master).all():
                        dmax_us = np.max(abs(timestamps - timestamps_master)) \
                                  / core.G3Units.microseconds
                        self._err(None, f'Timestamps value discrepancy (up to {dmax_us:.3} us).',
                                  warn=self.config['tolerate_timestamps_value_discrepancy'])

        # If metadata included sample_ranges, check that.
        _sample_ranges = meta.get('sample_ranges')
        if _sample_ranges is not None:
            A, B = np.array(_sample_ranges), np.array(sample_ranges)
            if A.shape != B.shape or np.any(A != B):
                self._err(None, f'sample_ranges from metadata is not as found in files.')
        self.results['sample_ranges'] = sample_ranges

    def get_obsfiledb_info(self, filebase_root):
        """Get entries ready for obsfiledb addition.

        Args:
          filebase_root (str): Path relative to which the file paths should
            be specified.  If this is '/' then an absolute path will
            be recorded.

        Returns:
          detset_rows: list of implicated detsets, with each entry a
            tuple (detset_name, detset_dets).
          file_rows: list of file table entries; each entry is a dict
            that can be passed as kwargs to ObsFileDb.add_file.

        """
        meta, det_lists, sample_ranges = [
            self.results[k] for k in ['metadata', 'det_lists', 'sample_ranges']]
        detset_rows = []
        file_rows = []

        for stream_id, detset in zip(meta['stream_ids'], meta['detsets']):
            if self.config['require_ordered_detsets']:
                assert stream_id in detset
            elif stream_id not in detset:
                # if detsets don't need to be in the correct order just find the
                # right one.
                detset = [x for x in meta['detsets'] if stream_id in x][0]

            # Screen out "fixed tones".  We don't want these in the
            # obsfiledb detsets.
            dets = [d for d in det_lists[stream_id] if "NONE" not in d]
            n_fixed_tone = len(det_lists[stream_id]) - len(dets)
            if not self.config['ignore_fixed_tones']:
                # Yes, "ignore" just means don't even talk about it.
                self.logger.warn(f'Suppressing {n_fixed_tone} fixed tones '
                                 f'in {stream_id}')
            detset_rows.append((detset, dets))

            for index in range(meta['file_count']):
                pattern = self.config['stream_file_pattern']
                path = self._get_filename(
                    pattern, stream_id=stream_id, index=index)
                if filebase_root == '/':
                    relpath = os.path.abspath(path)
                else:
                    relpath = os.path.relpath(path, filebase_root)
                file_rows.append(
                    {'filename': relpath,
                     'obs_id': meta['book_id'],
                     'detset': detset,
                     'sample_start': sample_ranges[index][0],
                     'sample_stop': sample_ranges[index][1],
                     })

        return detset_rows, file_rows

    def report(self):
        """Print a summary of warning and error messages."""
        self.logger.info('Warnings:')
        for err in self.results['warnings']:
            self.logger.info(err)
        self.logger.info('Errors:')
        for err in self.results['errors']:
            self.logger.info(err)


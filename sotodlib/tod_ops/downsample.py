"""Downsampling of the axis manager


Absolute-grid convention
------------------------
Kept samples are those whose *absolute* index (OffsetAxis offset + local
index) is a multiple of ``factor``.  The output axis is
``OffsetAxis(samps, n_kept, ceil(offset / factor), origin_tag)``, i.e. the
downsampled data lives on a global grid independent of how the observation
was trimmed.  This is what makes the saved proc archive reproducible: the
same observation downsampled after different sample cuts lands on the same
grid, and the load path reproduces exactly the sampling stored in the
archive.  The convention (and the factor) is recorded in the proc archive
as ``downsample_cfg`` and validated at load time.

Config
------
Top-level key in the proc-layer config::

    downsample:
      factor: 50        # required, int >= 2
      method: slice     # optional: 'slice' (default) or 'mean'

'slice' takes every factor-th sample.  It does not anti-alias; it is
intended for post-demodulation (low-pass-filtered) timestreams.  'mean'
block-averages float arrays (including timestamps, keeping them aligned).
Flag types (Ranges / RangesMatrix / bool arrays) are always pooled with
"any sample in the block is flagged", so cuts never get lost, regardless
of method.

Sparse arrays (csr, samps on the last axis) are treated as flag-like,
which is how they are used in proc_aman: each nonzero column is remapped
to the output block containing it (searchsorted), and nonzeros that merge
into the same output column are reduced with *max*.  For a boolean csr
this is exactly the 'any within block' rule used for Ranges; for a
numeric csr the largest value in the block is kept.  Either way, a flag
sitting on a dropped sample never gets lost.  'mean' is never applied to
sparse data.
"""
import logging

import numpy as np
from scipy.sparse import issparse
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from so3g.proj import Ranges, RangesMatrix

from .. import core

logger = logging.getLogger(__name__)


def _downsample_indices(count, offset, factor):
    """Local indices of kept samples on the absolute grid, i.e. samples
    whose absolute index (offset + local) is a multiple of factor."""
    start = (-offset) % factor
    return np.arange(start, count, factor)


def _pool_bool(arr, idx, axis):
    """'Any within block' pooling of a boolean array along ``axis``.
    Block k spans local indices [idx[k], idx[k+1]) (last block runs to the
    end); samples before idx[0] are outside the grid and dropped, matching
    the 'slice' behavior for data arrays."""
    pooled = np.maximum.reduceat(arr.astype(np.uint8), idx, axis=axis)
    return pooled.astype(bool)


def _block_mean(arr, idx, axis, count):
    """Block average of a float array along ``axis`` (same blocks as
    _pool_bool)."""
    sums = np.add.reduceat(arr, idx, axis=axis)
    lens = np.diff(np.append(idx, count))
    shape = [1] * arr.ndim
    shape[axis] = len(lens)
    return sums / lens.reshape(shape)


def _pool_csr(v, idx):
    """'Max within block' pooling of a csr array (samps = columns).  Each
    nonzero column is remapped to the output block that contains it;
    nonzeros merging into the same output column are reduced with max.
    For boolean data this is 'any within block'.  Dtype is preserved."""
    coo = v.tocoo()
    keep = coo.col >= idx[0]          # columns before the grid are dropped
    rows = coo.row[keep]
    cols = np.searchsorted(idx, coo.col[keep], side='right') - 1
    data = coo.data[keep]
    if len(data):
        # group duplicates (same row, same output column), reduce with max
        order = np.lexsort((cols, rows))
        rows, cols, data = rows[order], cols[order], data[order]
        first = np.r_[True, (np.diff(rows) != 0) | (np.diff(cols) != 0)]
        data = np.maximum.reduceat(data, np.flatnonzero(first))
        rows, cols = rows[first], cols[first]
    return csr_array((data, (rows, cols)), shape=(v.shape[0], len(idx)))


def _downsample_flags(flags, idx):
    """Downsample a Ranges, or a (possibly nested) RangesMatrix, by 'any
    within block' pooling of its mask.  The samps axis must be the last
    dimension."""
    if isinstance(flags, RangesMatrix):
        return RangesMatrix([_downsample_flags(row, idx)
                             for row in flags.ranges])
    return Ranges.from_mask(_pool_bool(flags.mask(), idx, 0))


def down_sample_aman(aman, factor, method='slice', axis='samps'):
    """Return a new AxisManager with ``axis`` downsampled by ``factor`` on
    the absolute grid (see module docstring).  All fields assigned to
    ``axis`` are downsampled; wrapped AxisManagers are processed
    recursively; everything else is copied through.

    Arguments
    ---------
    aman : AxisManager
        Input; not modified.
    factor : int
        Downsampling factor, >= 2.
    method : str
        'slice' (default) or 'mean'.  Applies to float ndarrays only;
        flags are always pooled with 'any', non-float arrays are sliced.
    axis : str
        Name of the axis to downsample.  Must be an OffsetAxis or
        IndexAxis if present; if absent, a copy of ``aman`` is returned.
    """
    methods = ('slice', 'mean')
    if method not in methods:
        raise ValueError(f"method must be one of {methods}, got {method!r}")
    old_ax = aman._axes.get(axis)
    if old_ax is None:
        return aman.copy()
    if isinstance(old_ax, core.OffsetAxis):
        offset = old_ax.offset
    elif isinstance(old_ax, core.IndexAxis):
        offset = 0
    else:
        raise ValueError(f"Cannot downsample axis of type {type(old_ax)}")

    idx = _downsample_indices(old_ax.count, offset, factor)

    new_axes = []
    for k, v in aman._axes.items():
        if k != axis:
            new_axes.append(v)
        elif isinstance(old_ax, core.OffsetAxis):
            # absolute position of idx[0] is offset + (-offset) % factor,
            # which is exactly -(-offset // factor) * factor.
            new_axes.append(core.OffsetAxis(
                axis, len(idx), -(-offset // factor), old_ax.origin_tag))
        else:
            new_axes.append(core.IndexAxis(axis, len(idx)))

    if isinstance(aman, core.FlagManager):
        dest = core.FlagManager(*new_axes)
    else:
        dest = core.AxisManager(*new_axes)

    for k, assign in aman._assignments.items():
        v = aman._fields[k]
        axis_map = [(i, a) for i, a in enumerate(assign) if a is not None]
        if isinstance(v, core.AxisManager):
            dest.wrap(k, down_sample_aman(v, factor, method=method, axis=axis))
        elif axis not in assign:
            if np.isscalar(v) or v is None:
                dest.wrap(k, v)
            else:
                dest.wrap(k, v.copy(), axis_map)
        elif isinstance(v, (Ranges, RangesMatrix)):
            if assign[-1] != axis:
                raise ValueError(f"Field '{k}': flags with '{axis}' not on "
                                 "the last dimension are unsupported")
            dest.wrap(k, _downsample_flags(v, idx), axis_map)
        elif issparse(v):
            if assign[-1] != axis:
                raise ValueError(f"Field '{k}': sparse array with '{axis}' "
                                 "not on the last dimension is unsupported")
            # flag-like: max-pool within blocks (any, for bool), like Ranges
            dest.wrap(k, _pool_csr(v, idx), axis_map)
        elif isinstance(v, np.ndarray):
            dim = assign.index(axis)
            if v.dtype == bool:
                new_v = _pool_bool(v, idx, dim)
            elif method == 'mean' and np.issubdtype(v.dtype, np.floating):
                new_v = _block_mean(v, idx, dim, old_ax.count)
            else:
                new_v = np.take(v, idx, axis=dim)
            dest.wrap(k, new_v, axis_map)
        else:
            raise ValueError(f"Field '{k}': type {type(v)} assigned to "
                             f"'{axis}' is not supported by downsample")
    return dest

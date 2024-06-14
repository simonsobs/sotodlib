import so3g
import numpy as np
import json
import h5py

## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array
import astropy.units as u

from .axisman import *
from .flagman import FlagManager

# Flatten / expand RangesMatrix

def flatten_RangesMatrix(rm):
    """Given a RangesMatrix, flatten into a small number of arrays that
    can be written to HDF or npy and then reconstructed later.

    Returns:
      Dict with arrays called 'shape', 'intervals', and 'ends'.
    """
    shape = rm.shape
    if len(shape) == 2:
        intervals = [r.ranges().reshape(-1) for r in rm.ranges]
        ends = np.cumsum([len(i) for i in intervals])
        intervals = np.hstack(intervals)
        return {
            'shape': np.array(shape),
            'intervals': intervals,
            'ends': ends,
        }
    last_end = 0
    ends, intervals = [], []
    for r in rm.ranges:
        subrm = flatten_RangesMatrix(r)
        ends.append(subrm['ends'] + last_end)
        intervals.append(subrm['intervals'])
        last_end += subrm['ends'][-1]
    return {
        'shape': np.array(shape),
        'intervals': np.hstack(intervals),
        'ends': np.hstack(ends),
    }

def expand_RangesMatrix(flat_rm):
    """Reconstruct a RangesMatrix given a dict like the one returned by
    flatten_RangesMatrix."""
    shape, intervals, ends = [
        flat_rm[k] for k in ['shape', 'intervals', 'ends']]
    if len(shape) == 1:
        r = intervals.reshape((-1, 2))
        return so3g.proj.Ranges.from_array(r, shape[0])
    ranges = []
    if shape[0] == 0:
        return so3g.proj.RangesMatrix([], child_shape=shape[1:])
    # Otherwise non-trivial
    count = np.product(shape[:-1])
    start, stride = 0, count // shape[0]
    for i in range(0, len(ends), stride):
        _e = ends[i:i+stride] - start
        _i = intervals[start:ends[i+stride-1]]
        ranges.append(expand_RangesMatrix(
            {'shape': shape[1:], 'intervals': _i, 'ends': _e}))
        start = ends[i+stride-1]
    return so3g.proj.RangesMatrix(ranges, child_shape=shape[1:])

## Flatten and Expand sparse arrays
def flatten_csr_array(arr):
    """Extract information from scipy.sparse.csr_array for saving in
    hdf5 files"""
    return {
        'data': arr.data,
        'indices': arr.indices,
        'indptr': arr.indptr,
        'shape': arr.shape,
    }


def expand_csr_array(flat_arr):
    """Reconstruct csr_array from flattened versions
    """
    return csr_array( (flat_arr['data'], flat_arr['indices'],
                       flat_arr['indptr']),
                     shape=flat_arr['shape'])

# Helper functions for numpy arrays containing unicode strings; must
# be written to HDF5 as "S" type arrays.

def _retype_for_write(data):
    """Convert arrays of unicode strings to "S" arrays; pass non-string
    arrays back without doing anything..

    """
    if data.dtype.kind == 'U':
        return data.astype('S')
    return data

def _retype_for_read(data):
    """Convert arrays of type "S" to arrays of unicode strings; pass
    non-string arrays back without doing anything.

    """
    if data.dtype.kind == 'S':
        return data.astype('U')
    return data


# save/load AxisManager to HDF5.

def _safe_scalars(x):
    # Kill any np.integer, np.floating, or np.str_ vals... they don't serialize.
    if isinstance(x, list):
        return [_safe_scalars(_x) for _x in x]
    if isinstance(x, tuple):
        return tuple([_safe_scalars(_x) for _x in x])
    if isinstance(x, dict):
        return {k: _safe_scalars(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.floating, np.str_, np.bool_)):
        return x.item()
    # Must be fine then!
    return x

def _save_axisman(axisman, dest, group=None, overwrite=False, compression=None):
    """
    See AxisManager.save.
    """
    # Scheme it out...
    schema = []
    for k, assign in axisman._assignments.items():
        item = {'name': k,
                'axes': assign,
                'encoding': 'unknown',
                }
        v = axisman[k]
        if v is None or np.isscalar(v):
            item['encoding'] = 'scalar'
        elif isinstance(v, u.quantity.Quantity):
            if v.shape == ():
                item['encoding'] = 'scalar_quantity'
            else:
                item['encoding'] = 'quantity'
        elif isinstance(v, np.ndarray):
            item['encoding'] = 'ndarray'
        elif isinstance(v, AxisInterface):
            item['encoding'] = 'axis'
        elif isinstance(v, AxisManager):
            item['encoding'] = 'axisman'
            if v.__class__ is AxisManager:
                item['subclass'] = 'AxisManager'
            elif v.__class__ is FlagManager:
                item['subclass'] = 'FlagManager'
                item['special_axes'] = v._dets_name, v._samps_name
            else:
                raise ValueError(f"No encoder system for {k}={v.__class__}")
        elif isinstance(v, so3g.proj.RangesMatrix):
            item['encoding'] = 'rangesmatrix'
        elif isinstance(v, csr_array):
            item['encoding'] = 'csrarray'
        else:
            print(v.__class__)
        schema.append(item)

    for k, v in axisman._axes.items():
        if isinstance(v, LabelAxis):
            schema.append({'name': k,
                           'encoding': 'axis',
                           'type': 'label',
                           'args': (v.name, list(v.vals))})
        elif isinstance(v, OffsetAxis):
            schema.append({'name': k,
                           'encoding': 'axis',
                           'type': 'offset',
                           'args': (v.name, v.count, v.offset, v.origin_tag)})
        elif isinstance(v, IndexAxis):
            schema.append({'name': k,
                           'encoding': 'axis',
                           'type': 'index',
                           'args': (v.name, v.count)})
        else:
            raise ValueError(f"No encoder for axis class: {v.__class__}")

    # Sanitize ...
    schema = _safe_scalars(schema)

    # Resolve the destination group.
    file_to_close = None
    if isinstance(dest, str):
        file_to_close = h5py.File(dest, 'a')
        dest = file_to_close['/']
    assert isinstance(dest, h5py.Group)  # filename or Group expected
    if group is not None:
        if group in dest:
            dest = dest[group]
        else:
            dest = dest.create_group(group)

    # Needs emptying?  This might be slower than just del dest[group],
    # but it also works for '/' or Groups passed in by reference only.
    for target in [dest, dest.attrs]:
        for k in list(target.keys()):
            if overwrite:
                del target[k]
            else:
                raise RuntimeError(
                    f'Destination group "{dest.name}" is not empty; '
                    f'pass overwite=True to clobber.')

    dest.attrs['_axisman'] = json.dumps({
        'version': 0,
        'schema': schema,
        })
    scalars = {}
    units = {}

    for item in schema:
        data = axisman[item['name']]
        if item['encoding'] == 'scalar':
            scalars[item['name']] = data
        elif item['encoding'] == 'ndarray':
            dest.create_dataset(item['name'], data=_retype_for_write(data), compression=compression)
        elif item['encoding'] == 'quantity':
            dest.create_dataset(item['name'], data=_retype_for_write(data), compression=compression)
            units[item['name']] = data.unit.to_string()
        elif item['encoding'] == 'scalar_quantity':
            scalars[item['name']] = data.value
            units[item['name']] = data.unit.to_string()
        elif item['encoding'] == 'rangesmatrix':
            g = dest.create_group(item['name'])
            for k, v in flatten_RangesMatrix(data).items():
                g.create_dataset(k, data=v, compression=compression)
        elif item['encoding'] == 'csrarray':
            g = dest.create_group(item['name'])
            for k, v in flatten_csr_array(data).items():
                g.create_dataset(k, data=v, compression=compression)
        elif item['encoding'] == 'axisman':
            g = dest.create_group(item['name'])
            _save_axisman(data, g, compression=compression)
        elif item['encoding'] == 'axis':
            pass #
        else:
            print(f'Unhandled {item["name"]}->{item["encoding"]}')

    if len(scalars):
        dest.attrs['_scalars'] = json.dumps(scalars)

    if len(units):
        dest.attrs['_units'] = json.dumps(units)

    if file_to_close:
        file_to_close.close()

def _get_subfields(fields, prefix):
    if fields is None:
        return None
    subfields = []
    for f in fields:
        if f.startswith(prefix + '.'):
            subfields.append(f[len(prefix)+1:])
    if len(subfields) == 0:
        return None
    return subfields

def _load_axisman(src, group=None, cls=None, fields=None):
    """
    See AxisManager.load.
    """
    if cls is None:
        cls = AxisManager

    if isinstance(src, str):
        f = h5py.File(src, 'r')
        if group is None:
            src = f
        else:
            src = f[group]
    else:
        f = None

    info = json.loads(src.attrs['_axisman'])
    assert(info['version'] == 0)
    schema = info['schema']

    scalars = {}
    if '_scalars' in src.attrs:
        scalars = json.loads(src.attrs['_scalars'])

    units = {}
    if '_units' in src.attrs:
        units = json.loads(src.attrs['_units'])

    # Reconstruct axes.
    axes = []
    for item in schema:
        if item['encoding'] == 'axis':
            if item['type'] == 'label':
                axes.append((LabelAxis(*item['args'])))
            elif item['type'] == 'offset':
                axes.append((OffsetAxis(*item['args'])))
            elif item['type'] == 'index':
                axes.append((IndexAxis(*item['args'])))
    axisman = cls(*axes)
    for item in schema:
        subfields = _get_subfields(fields, item['name'])
        if (fields is not None) and (item['name'] not in fields) and subfields is None:
            continue
        assign = [(i, a) for i, a in enumerate(item.get('axes', [])) if a is not None]
        if item['encoding'] == 'axis':
            pass
        elif item['encoding'] == 'scalar':
            axisman.wrap(item['name'], scalars[item['name']])
        elif item['encoding'] == 'ndarray':
            axisman.wrap(item['name'], _retype_for_read(src[item['name']][:]), assign)
        elif item['encoding'] == 'quantity':
            axisman.wrap(item['name'], _retype_for_read(src[item['name']][:]) << u.Unit(units[item['name']]), assign)
        elif item['encoding'] == 'scalar_quantity':
            axisman.wrap(item['name'], scalars[item['name']] << u.Unit(units[item['name']]), assign)
        elif item['encoding'] == 'axisman':
            x = _load_axisman(src[item['name']], fields=subfields)
            if item['subclass'] == 'FlagManager':
                x = FlagManager.promote(x, *item['special_axes'])
            axisman.wrap(item['name'], x)
        elif item['encoding'] == 'rangesmatrix':
            x = src[item['name']]
            rm_flat = {k: x[k][:] for k in ['shape', 'intervals', 'ends']}
            axisman.wrap(item['name'], expand_RangesMatrix(rm_flat), assign)
        elif item['encoding'] == 'csrarray':
            x = src[item['name']]
            csr_flat = {k: x[k][:] for k in ['shape', 'data', 'indices', 'indptr']}
            axisman.wrap(item['name'], expand_csr_array(csr_flat), assign)
        else:
            print('No decoder for:', item)

    return axisman

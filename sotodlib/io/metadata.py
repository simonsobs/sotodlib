"""Support for reading and writing simple metadata types to HDF5.

"Simple" metadata, at this point, means tabular data with columns that
mix extrinsic indices, intrinsic indices, and metadata fields.

String data is awkward in HDF5 / numpy / Python 3.  The approach
adopted here is to maintain data structures in unicode-compatible data
types (numpy 'U'), because this permits simple string comparisons
(such as mask = (results['band_name'] == 'rem')).  As a result, we
must convert 'U' fields to numpy 'S' type when writing to HDF5, and
then back to 'U' type on load from HDF5.  See
http://docs.h5py.org/en/stable/strings.html for a little more info.

"""

import numpy as np
import h5py

from sotodlib.core import AxisManager
from sotodlib.core.metadata import ResultSet, SuperLoader, LoaderInterface
import warnings

def write_dataset(data, filename, address, overwrite=False, mode='a'):
    """Write a metadata object to an HDF5 file as a single dataset.

    Args:
      data: The metadata object.  Currently only ResultSet and numpy
        structured arrays are supported.
      filename: The path to the HDF5 file, or an open h5py.File.
      address: The path within the HDF5 file at which to create the
        dataset.
      overwrite: If True, remove any existing group or dataset at the
        specified address.  If False, raise a RuntimeError if the
        write address is already occupied.
      mode: The mode specification used for opening the file
        (ignored if filename is an open file).

    """
    if isinstance(data, ResultSet):
        data = data.asarray(hdf_compat=True)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError("I do not know how to write type %s" % data.__class__)

    if isinstance(filename, str):
        context = h5py.File(filename, mode)
    else:
        # Wrap in a nullcontext so that the block below doesn't
        # close the File on exit.
        fout = filename
        filename = fout.filename
        context = _nullcontext(fout)

    with context as fout:
        if address in fout:
            if overwrite:
                del fout[address]
            else:
                raise RuntimeError(
                    f'Address {address} already exists in {filename}; '
                    f'pass overwrite=True to clobber it.')
        fout.create_dataset(address, data=data)


def read_dataset(fin, dataset):
    """Read a dataset from an HDF5 file and return it as a ResultSet.

    Args:
      fin: Filename or h5py.File open for reading.
      dataset: Dataset path.

    Returns:
      ResultSet populated from the dataset.  Note this is passed
      through _decode_array, so byte strings are converted to unicode.

    """
    if isinstance(fin, str):
        fin = h5py.File(fin, 'r')
    data = fin[dataset][()]
    data = _decode_array(data)
    rs = ResultSet(keys=list(data.dtype.names))
    for row in data:
        rs.rows.append(tuple(row))
    return rs


class DefaultHdfLoader(LoaderInterface):
    """Determine the type of H5 saved data and pass off the loading to the
    correct class.
    """
    def from_loadspec(self, load_params, **kwargs):
        with h5py.File(load_params['filename'], mode='r') as fin:
            # look for AxisManager save signature
            if '_axisman' in fin[ load_params['dataset'] ].attrs.keys():
                newload = AxisManagerHdfLoader()
                return newload.from_loadspec(load_params, **kwargs)
            else:
                newload = ResultSetHdfLoader()
                return newload.from_loadspec(load_params, **kwargs)

class AxisManagerHdfLoader(LoaderInterface):
    def from_loadspec(self, load_params, **kwargs):
        """ Generate an AxisManager from the load_params dictionary.
        """
        _kwargs = {k1: kwargs[k2] for k1, k2 in [('fields', 'load_fields')]
                   if k2 in kwargs}
        aman = AxisManager.load(load_params['filename'],
                                load_params['dataset'],
                                **_kwargs)
        return aman


class ResultSetHdfLoader(LoaderInterface):
    def _prefilter_data(self, data_in, key_map={}):
        """When a dataset is loaded and converted to a structured numpy
        array, this function is called before the data are returned to
        the user.  The key_map can be used to rename fields, on load.

        This function may be extended in subclasses, but you will
        likely want to call the super() handler before doing
        additional processing.  The loading functions do not pass in
        key_map -- this is for the exclusive use of subclasses.

        """
        return _decode_array(data_in, key_map=key_map)

    def _populate(self, data, keys=None, row_order=None):
        """Process the structured numpy array "data" and return a ResultSet.
        keys should be a list of field names to load from the data
        (default is None, which will load all fields).  row_order
        should be a list of indices into the desired rows of data
        (default is None, which will load all rows, in order).

        (This function can be overridden in subclasses, without
        calling the super.)

        """
        if keys is None:
            keys = [k for k in data.dtype.names]
        if row_order is None:
            row_order = range(len(data))
        rs = ResultSet(keys=keys)
        for i in row_order:
            rs.append({k: data[k][i] for k in rs.keys})
        return rs

    def from_loadspec(self, load_params, **kwargs):
        """Retrieve a metadata result from an HDF5 file.

        Arguments:
          load_params: an index dictionary (see below).

        Returns a ResultSet (or, for subclasses, whatever sort of
        thing is returned by self._populate).

        The "index dictionary", for the present case, may contain
        extrinsic and intrinsic selectors (for the 'obs' and 'dets'
        axes); it must also contain:

        - 'filename': full path to an HDF5 file.
        - 'dataset':  name of the dataset within the file.

        Note that this just calls batch_from_loadspec.

        """
        return self.batch_from_loadspec([load_params], **kwargs)[0]

    def batch_from_loadspec(self, load_params, **kwargs):
        """Retrieves a batch of metadata results.  load_params should be a
        list of valid index data specifications.  Returns a list of
        objects, corresponding to the elements of load_params.

        This function is relatively efficient in the case that many
        requests are made for data from a single file.

        """
        # Gather all relevant HDF5 files.
        file_map = {}
        for idx, load_par in enumerate(load_params):
            fn = load_par['filename']
            if fn not in file_map:
                file_map[fn] = []
            file_map[fn].append(idx)
        # Open each one and pull out the result.
        results = [None] * len(load_params)
        for filename, indices in file_map.items():
            with h5py.File(filename, mode='r') as fin:
                # Don't reread dataset unless it changes.
                last_dataset = None
                for idx in indices:
                    dataset = load_params[idx]['dataset']
                    if dataset is not last_dataset:
                        data = fin[dataset][()]
                        data = self._prefilter_data(data)
                        last_dataset = dataset

                    # Dereference the extrinsic axis request.  Every
                    # extrinsic axis key in the dataset must have a
                    # value specified in load_params.
                    ex_keys = []
                    mask = np.ones(len(data), bool)
                    for k in data.dtype.names:
                        if k.startswith('obs:'):
                            ex_keys.append(k)
                            mask *= (data[k] == load_params[idx][k])

                    # Has user made an intrinsic request as well?
                    for k in data.dtype.names:
                        if k.startswith('dets:') and k in load_params[idx]:
                            mask *= (data[k] == load_params[idx][k])

                    # TODO: handle non-concordant extrinsic /
                    # intrinsic requests.

                    # Output.
                    keys_out = [k for k in data.dtype.names
                                if k not in ex_keys]
                    results[idx] = self._populate(data, keys=keys_out,
                                                  row_order=mask.nonzero()[0])
        return results


def _decode_array(data_in, key_map={}):
    """Converts a structured numpy array to a structured numpy array,
    rewriting any 'S'-type string fields as 'U'-type string fields.

    Args:
      data_in: A structure numpy array (i.e. an ndarray with a dtype
        consisting of multiple named fields).
      key_map: A dict specifying how to rename fields.  Any key=>value
        pair here will cause data_in[key] to be written to
        data_out[value].  If value is None, the specified field will
        not be included in data_out.

    Returns:
        A new structured array, unless no changes are needed, in which
        case data_in is returned unmodified.

    """
    changes = False
    new_dtype = []
    columns = []
    for i, k in enumerate(data_in.dtype.names):
        key_out = key_map.get(k, k)
        changes = changes or (key_out != k)
        if key_out is None:
            continue
        if data_in.dtype[k].char == 'S':
            # Convert to unicode.
            columns.append(np.array([v.decode('ascii') for v in data_in[k]]))
            changes = True
        else:
            columns.append(data_in[k])
        if len(data_in[k].shape) == 1:
            new_dtype.append((key_out, columns[-1].dtype))
        else:
            new_dtype.append((key_out, columns[-1].dtype, data_in[k].shape[1:]))
    new_dtype = np.dtype(new_dtype)
    output = np.empty(data_in.shape, dtype=new_dtype)
    for k, c in zip(new_dtype.names, columns):
        output[k] = c
    return output


# Starting in Python 3.7, this can be had from contextlib.
class _nullcontext:
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


SuperLoader.register_metadata('DefaultHdf', DefaultHdfLoader)
SuperLoader.register_metadata('AxisManagerHdf', AxisManagerHdfLoader)
SuperLoader.register_metadata('ResultSetHdf', ResultSetHdfLoader)

# The old name... remove some day.
SuperLoader.register_metadata('PerDetectorHdf5', ResultSetHdfLoader)

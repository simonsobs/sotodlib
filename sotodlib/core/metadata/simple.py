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
import contextlib

from .resultset import ResultSet


class _Hdf5Writer:
    """This class is a light extension for ResultSet that can write the
    data to an HDF5 file as a dataset.

    """
    def write_dataset(self, filename, address, overwrite=False, mode='a'):
        """Write self (a ResultSet or similar) to an HDF5 file as a single
        dataset.

        Arguments:
          filename: The path to the HDF5 file.  Alternately, pass in an
            open h5py.File.
          address: The path within the HDF5 file at which to create the
            dataset.
          overwrite: If true, remove any existing group or dataset at
            the specified address.
          mode: The mode specification used for opening the file
            (ignored if filename is an open file).

        """
        if isinstance(filename, str):
            context = h5py.File(filename, mode)
        else:
            # Wrap in a nullcontext so that the block below doesn't
            # close the File on exit.
            fout = filename
            filename = fout.filename
            context = contextlib.nullcontext(fout)

        with context as fout:
            if address in fout:
                if overwrite:
                    del fout[address]
                else:
                    raise RuntimeError(
                        f'Address {address} already exists in {filename}; '
                        f'pass overwrite=True to clobber it.')
            data = self.asarray()
            # If there are "unicode" strings in there, convert to a
            # numpy fixed-length 'S' type.  This will choke if the
            # strings are not ascii-compatible.
            new_dtype = []
            for k in data.dtype.names:
                if data.dtype[k].char == 'U':
                    max_len = max(map(len, data[k]))
                    new_dtype.append((k, 'S%i' % max_len))
                else:
                    new_dtype.append((k, data.dtype[k]))
            data = data.astype(np.dtype(new_dtype))
            fout.create_dataset(address, data=data)


class PerDetectorHdf5(ResultSet, _Hdf5Writer):
    """This class is designed to read and write metadata to HDF5.  It
    treats 'obs' as an extrinsic axis, and 'dets' as an intrinsic
    axis.  However, data for multiple 'obs' can be stored in a single
    dataset (rather than in distinct datasets) if desired.

    """
    intrinsic_axes = ['dets']

    @classmethod
    def _prefilter_data(cls, data_in, key_map={}):
        """This function is called by the data loader, after loading a dataset
        but before matching the indices.  It should always be called,
        to convert numpy 'S'-type to numpy 'U'-type strings.

        Only the positional arguments are passed in by the data
        loader.  The keyword arguments can be used by subclasses
        (through super()._prefilter_data(...)).

        If key_map is passed in, it will be used to rename columns
        (from key to value).

        """
        new_dtype = []
        columns = []
        for i, k in enumerate(data_in.dtype.names):
            if data_in.dtype[k].char == 'S':
                # Convert to unicode.
                columns.append(np.array([v.decode('ascii') for v in data_in[k]]))
            else:
                columns.append(data_in[k])
            if len(data_in[k].shape) == 1:
                new_dtype.append((key_map.get(k, k), columns[-1].dtype))
            else:
                new_dtype.append((key_map.get(k, k), columns[-1].dtype, data_in[k].shape[1:]))
        new_dtype = np.dtype(new_dtype)
        output = np.empty(data_in.shape, dtype=new_dtype)
        for k, c in zip(new_dtype.names, columns):
            output[k] = c
        return output

    @classmethod
    def from_loadspec(cls, load_params,
                      detdb=None,
                      obsdb=None):

        """Retrieve a metadata result.

        Arguments:
          load_params: an index dictionary (see below).
          detdb: a DetDB which may be used to resolve 'dets' indices.
          obsdb: an ObsDB which may be used to resolve 'obs' indices.

        Returns an object of the present class.

        The "index dictionary", for the present case, may contain
        extrinsic and intrinsic selectors (for the 'obs' and 'dets'
        axes); it must also contain:

        - 'filename': full path to an HDF5 file.
        - 'dataset':  name of the dataset within the file.

        Note that this just calls batch_from_loadspec.

        """
        return cls.batch_from_loadspec(
            [load_params], detdb=detdb, obsdb=obsdb)[0]

    @classmethod
    def batch_from_loadspec(cls, load_params, detdb=None, obsdb=None):
        """Retrieve a batch of metadata results.  The arguments here are the
        same as for from_loadspec, expect that load_params must be a
        /list/ of index dictionaries.  This function returns a list of
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
                        data = cls._prefilter_data(data)
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
                    self = cls(keys=keys_out)
                    for i in mask.nonzero()[0]:
                        self.append({k: data[k][i] for k in self.keys})
                    results[idx] = self
        return results

    @classmethod
    def from_proddb(cls, proddb, req):
        proddb.match(req)

    @classmethod
    def loader_class(cls):
        class _Loader:
            """(Temporary?) Loader class for PerDetectorHdf5..."""
            def __init__(self, detdb=None, obsdb=None):
                self.detdb = detdb
                self.obsdb = obsdb
            def from_loadspec(self, request):
                return cls.from_loadspec(
                    request, detdb=self.detdb, obsdb=self.obsdb)
        return _Loader

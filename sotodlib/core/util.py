import numpy as np
import h5py
import time


class H5LockTimeoutError(Exception):
    """Raised when an HDF5 file cannot be opened due to lock timeouts."""
    pass


class H5ContextManager:
    """
    Class to open an HDF5 file with retry logic on file locking.  This class
    can be used either as a context manager or as a regular function.

    Examples
    --------
    As a context manager
        with H5ContextManager("data.h5", "r") as f:
            print(list(f.keys()))

    Direct function call:
        f = H5ContextManager("data.h5", "r").open()
        print(list(f.keys()))
        f.close()

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    *args :
        Additional positional arguments passed to `h5py.File`.
    max_attempts : int, optional
        Number of times to retry opening the file if it is locked.
    delay : int or float, optional
        Delay in seconds between retries.
    **kwargs :
        Additional keyword arguments passed to `h5py.File`.
    """
    def __init__(self, filename, *args, max_attempts=3, delay=5, **kwargs):
        self.filename = filename
        self.args = args
        self.kwargs = kwargs
        self.max_attempts = max_attempts
        self.delay = delay
        self.f = None

        assert self.max_attempts > 0
        assert self.delay >= 0

    def open(self):
        for attempt in range(self.max_attempts):
            try:
                self.f = h5py.File(self.filename, *self.args, **self.kwargs)
                return self.f
            except BlockingIOError as e:
                # If the file is locked, retry opening it after a delay
                if attempt + 1 < self.max_attempts:
                    time.sleep(self.delay)
                else:
                    raise H5LockTimeoutError(
                        f"Failed to open {self.filename} after "
                        f"{self.max_attempts} attempts"
                    ) from e
            except Exception as e:
                # Other errors should fail immediately
                raise e

    def __enter__(self):
        if self.f is None:
            self.open()
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f:
            self.f.close()
            self.f = None


def tag_substr(dest, tags, max_recursion=20):
    """ Do string substitution of all our tags into dest (in-place
    if dest is a dict). Used for context and data packaging tag replacements
    """
    assert(max_recursion > 0)  # Too deep this dictionary.
    if isinstance(dest, str):
        # Keep subbing until it doesn't change any more...
        new = dest.format(**tags)
        while dest != new:
            dest = new
            new = dest.format(**tags)
        return dest
    if isinstance(dest, list):
        return [tag_substr(x,tags) for x in dest]
    if isinstance(dest, tuple):
        return (tag_substr(x,tags) for x in dest)
    if isinstance(dest, dict):
        for k, v in dest.items():
            dest[k] = tag_substr(v,tags, max_recursion-1)
        return dest
    return dest

def get_coindices(v0, v1, check_unique=False):
    """Given vectors v0 and v1, each of which contains no duplicate
    values, determine the elements that are found in both vectors.
    Returns (vals, i0, i1), i.e. the vector of common elements and
    the vectors of indices into v0 and v1 where those elements are
    found.

    This routine will use np.intersect1d if it can.  The ordering of
    the results is different from intersect1d -- vals is not sorted,
    but rather the elements will appear in the same order that they
    were found in v0 (so that i0 is strictly increasing).

    The behavior is undefined if either v0 or v1 contain duplicates.
    Pass check_unique=True to assert that condition.

    """
    if check_unique:
        assert(len(set(v0)) == len(v0))
        assert(len(set(v1)) == len(v1))

    try:
        vals, i0, i1 = np.intersect1d(v0, v1, return_indices=True)
        order = np.argsort(i0)
        return vals[order], i0[order], i1[order]
    except TypeError:  # return_indices not implemented in numpy < 1.15
        pass

    # The old fashioned way
    v0 = np.asarray(v0)
    w0 = sorted([(j, i) for i, j in enumerate(v0)])
    w1 = sorted([(j, i) for i, j in enumerate(v1)])
    i0, i1 = 0, 0
    pairs = []
    while i0 < len(w0) and i1 < len(w1):
        if w0[i0][0] == w1[i1][0]:
            pairs.append((w0[i0][1], w1[i1][1]))
            i0 += 1
            i1 += 1
        elif w0[i0][0] < w1[i1][0]:
            i0 += 1
        else:
            i1 += 1
    if len(pairs) == 0:
        return (np.zeros(0, v0.dtype), np.zeros(0, int), np.zeros(0, int))
    pairs.sort()
    i0, i1 = np.transpose(pairs)
    return v0[i0], i0, i1


def get_multi_index(short_list, long_list):
    """For each item in long_list, determine the index at which it occurs
    in short_list.  Returns the equivalent of::

       np.array([short_list.index(x) if x in short_list else -1
                 for x in long_list])

    """
    w0 = sorted([(j, i) for i, j in enumerate(short_list)])
    w1 = sorted([(j, i) for i, j in enumerate(long_list)])
    i0, i1 = 0, 0
    indices = []
    while i0 < len(w0) and i1 < len(w1):
        if w0[i0][0] == w1[i1][0]:
            indices.append((w1[i1][1], w0[i0][1]))
            i1 += 1
        elif w0[i0][0] < w1[i1][0]:
            i0 += 1
        else:
            indices.append((w1[i1][1], -1))
            i1 += 1
    while i1 < len(w1):
        indices.append((w1[i1][1], -1))
        i1 += 1
    if len(indices) == 0:
        return np.zeros(0, int)
    indices.sort()
    return np.array([i0 for i1, i0 in indices])

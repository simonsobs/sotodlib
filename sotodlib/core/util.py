import numpy as np

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

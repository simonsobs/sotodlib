import numpy as np
from collections import OrderedDict as odict


class AxisInterface:
    """Abstract base class for axes managed by AxisManager."""

    count = None
    name = None

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        raise NotImplementedError

    def _minirepr_(self):
        return self.__repr__()

    def copy(self):
        raise NotImplementedError

    def resolve(self, src, axis_index=None):
        """Perform a check or promote-and-check of this Axis against a data
        object.

        The promotion step only applies to "unset" Axes, i.e. those
        here count is None.  Not all Axis types will be able to
        support this.  Promotion involves inspection of src and
        axis_index to fix free parameters in the Axis.  If promotion
        is successful, a new ("set") Axis is returned.  If promotion
        is attempted and fails then a ValueError is raised.X

        The check step involes confirming that the data object
        described by src (and axis_index) is compatible with the
        current axis (or with the axis resulting from axis Promotion).
        Typically that simply involves confirming that
        src.shape[axis_index] == self.count.  If the check fails, a
        ValueError is raised.

        Arguments:
          src: The data object to be wrapped (e.g. a numpy array)
          axis_index: The index of the data object to test for
            compatibility.

        Returns:
          axis: Either self, or the result of promotion.

        """
        # The default implementation performs the check step.
        # Subclasses may attempt promotion, then call this.
        ref_count = src.shape[axis_index]
        if self.count != ref_count:
            raise ValueError(
                "Dimension %i of data is incompatible with axis %s" %
                (axis_index, repr(self)))
        return self

    def restriction(self, selector):
        """Apply `selector` to the elements of this axis, returning a new Axis
        of the same type and an array indexing object (a slice or an
        array of integer indices) that may be used to extract the
        corresponding elements from a vector.

        See class header for acceptable selector objects.

        Returns (new_axis, ar_index).

        """
        raise NotImplementedError

    def intersection(self, friend, return_slices=False):
        """Find the intersection of this Axis and the friend Axis, returning a
        new Axis of the same type.  Optionally, also return array
        indexing objects that select the common elements from array
        dimensions corresponding to self and friend, respectively.

        See class header for acceptable selector objects.

        Returns (new_axis), or (new_axis, ar_index_self,
        ar_index_friend) if return_slices is True.

        """
        raise NotImplementedError


class IndexAxis(AxisInterface):
    """This class manages a simple integer-indexed axis.  When
    intersecting data, the longer one will be simply truncated to
    match the shorter.  Selectors must be tuples compatible with
    slice(), e.g. (2,8,2).

    """
    def __init__(self, name, count=None):
        super().__init__(name)
        self.count = count

    def copy(self):
        return IndexAxis(self.name, self.count)

    def __repr__(self):
        return 'IndexAxis(%s)' % self.count

    def resolve(self, src, axis_index=None):
        if self.count is None:
            return IndexAxis(self.name, src.shape[axis_index])
        return super().resolve(src, axis_index)

    def restriction(self, selector):
        if not isinstance(selector, slice):
            sl = slice(*selector)
        else:
            sl = selector
        start, stop, stride = sl.indices(self.count)
        assert stride == 1
        assert stop <= self.count
        return IndexAxis(self.name, stop - start), sl

    def intersection(self, friend, return_slices=False):
        count_out = min(self.count, friend.count)
        ax = IndexAxis(self.name, count_out)
        if return_slices:
            return ax, slice(count_out), slice(count_out)
        else:
            return ax


class OffsetAxis(AxisInterface):
    """This class manages an integer-indexed axis, with an accounting for
    an integer offset of any single vector relative to some absolute
    reference point.  For example, one vector could could have 100
    elements at offset 50, and a second vector could have 100 elements
    at offset -20.  On intersection, the result would have 30 elements
    at offset 50.

    The property `origin_tag` may be used to identify the absolute
    reference point.  It could be a TOD name ('obs_2020-12-01') or a
    timestamp or whatever.

    """

    origin_tag = None
    offset = 0

    def __init__(self, name, count=None, offset=0, origin_tag=None):
        super().__init__(name)
        self.count = count
        self.offset = offset
        self.origin_tag = origin_tag

    def copy(self):
        return OffsetAxis(self.name, self.count, self.offset, self.origin_tag)

    def __repr__(self):
        return 'OffsetAxis(%s:%s%+i)' % (
            self.count, self.origin_tag, self.offset)

    def _minirepr_(self):
        return 'OffsetAxis(%s)' % (self.count)

    def resolve(self, src, axis_index=None):
        if self.count is None:
            return OffsetAxis(self.name, src.shape[axis_index])
        return super().resolve(src, axis_index)

    def restriction(self, selector):
        sl = slice(*selector)
        start, stop, stride = sl.indices(self.count + self.offset)
        assert stride == 1
        assert start >= self.offset
        assert stop <= self.offset + self.count
        return (OffsetAxis(self.name, stop - start, start, self.origin_tag),
                slice(start - self.offset, stop - self.offset, stride))

    def intersection(self, friend, return_slices=False):
        offset = max(self.offset, friend.offset)
        count = min(self.count + self.offset,
                    friend.count + friend.offset) - offset
        count = max(count, 0)
        ax = OffsetAxis(self.name, count, offset, self.origin_tag)
        if return_slices:
            return ax, \
                slice(offset - self.offset, count + offset - self.offset), \
                slice(offset - friend.offset, count + offset - friend.offset)
        else:
            return ax


class LabelAxis(AxisInterface):
    """This class manages a string-labeled axis, i.e., an axis where each
    element has been given a unique name.  The vector of names can be
    found in self.vals.

    On intersection of two vectors, only elements whose names appear
    in both axes will be preserved.

    Selectors should be lists (or arrays) of label strings.

    """

    def __init__(self, name, vals=None):
        super().__init__(name)
        if vals is not None:
            vals = np.array(vals)
        self.vals = vals

    @property
    def count(self):
        if self.vals is None:
            return None
        return len(self.vals)

    def __repr__(self):
        if self.vals is None:
            items = ['?']
        elif len(self.vals) > 20:
            items = ([repr(v) for v in self.vals[:3]] + ['...'] +
                     [repr(v) for v in self.vals[-4:]])
        else:
            items = [repr(v) for v in self.vals]
        return 'LabelAxis(%s:' % self.count + ','.join(items) + ')'

    def _minirepr_(self):
        return 'LabelAxis(%s)' % (self.count)

    def copy(self):
        return LabelAxis(self.name, self.vals)

    def resolve(self, src, axis_index=None):
        if self.count is None:
            raise RuntimeError(
                'LabelAxis cannot be naively promoted from data.')
        return super().resolve(src, axis_index)

    def restriction(self, selector):
        # Selector should be list of vals.  Returns new axis and the
        # indices into self.vals that project out the elements.
        _vals, i0, i1 = get_coindices(selector, self.vals)
        assert len(i0) == len(selector)  # not a strict subset!
        return LabelAxis(self.name, selector), i1

    def intersection(self, friend, return_slices=False):
        _vals, i0, i1 = get_coindices(self.vals, friend.vals)
        ax = LabelAxis(self.name, _vals)
        if return_slices:
            return ax, i0, i1
        else:
            return ax


class AxisManager:
    """A container for numpy arrays and other multi-dimensional
    data-carrying objects (including other AxisManagers).  This object
    keeps track of which dimensions of each object are concordant, and
    allows one to slice all hosted data simultaneously.

    """

    def __init__(self, *args):
        self._axes = odict()
        self._assignments = {}  # data_name -> [ax0_name, ax1_name, ...]
        self._fields = odict()
        for a in args:
            if isinstance(a, AxisManager):
                # merge in the axes and copy in the values.
                self.merge(a)
            elif isinstance(a, AxisInterface):
                self._axes[a.name] = a
            else:
                raise ValueError("Cannot handle type %s in constructor." % a)

    @property
    def shape(self):
        return tuple([a.count for a in self._axes.values()])

    def copy(self, axes_only=False):
        out = AxisManager()
        for k, v in self._axes.items():
            out._axes[k] = v
        if axes_only:
            return out
        for k, v in self._fields.items():
            out._fields[k] = v.copy()
        for k, v in self._assignments.items():
            out._assignments[k] = v.copy()
        return out

    def _managed_ids(self):
        ids = [id(self)]
        for v in self._fields.values():
            if isinstance(v, AxisManager):
                ids.extend(v.managed_ids)
        return ids

    def __delitem__(self, name):
        if name in self._fields:
            del self._fields[name]
            del self._assignments[name]
        elif name in self._axes:
            del self._axes[name]
            for v in self._assignments.values():
                for i, n in enumerate(v):
                    if n == name:
                        v[i] = None
        else:
            raise KeyError(name)

    def move(self, name, new_name):
        """Rename or remove a data field.  To delete the field, pass
        new_name=None.

        """
        if new_name is None:
            del self._fields[name]
            del self._assignments[name]
        else:
            self._fields[new_name] = self._fields.pop(name)
            self._assignments[new_name] = self._assignments.pop(name)
        return self

    def __contains__(self, name):
        return name in self._fields or name in self._axes

    def __getitem__(self, name):
        if name in self._fields:
            return self._fields[name]
        if name in self._axes:
            return self._axes[name]
        raise KeyError(name)

    def __setitem__(self, name, val):
        if name in self._fields:
            self._fields[name] = val
        else:
            raise KeyError(name)

    def __setattr__(self, name, value):
        # Assignment to members update those members
        if "_fields" in self.__dict__ and name in self._fields.keys():
            self._fields[name] = value
        else:
            # Other assignments update this object
            self.__dict__[name] = value

    def __getattr__(self, name):
        # Prevent members from override special class members.
        if name.startswith("__"): raise AttributeError(name)
        return self[name]

    def __dir__(self):
        return sorted(tuple(self.__dict__.keys()) + tuple(self.keys()))

    def keys(self):
        return list(self._fields.keys()) + list(self._axes.keys())

    def shape_str(self, name):
        s = []
        for n, ax in zip(self._fields[name].shape, self._assignments[name]):
            if ax is None:
                s.append('%i' % n)
            else:
                s.append('%s' % ax)
        return ','.join(s)

    def __repr__(self):
        def branch_marker(name):
            return '*' if isinstance(self._fields[name], AxisManager) else ''
        stuff = (['%s%s[%s]' % (k, branch_marker(k), self.shape_str(k))
                  for k in self._fields.keys()]
                 + ['%s:%s' % (k, v._minirepr_())
                    for k, v in self._axes.items()])
        return ("{}(".format(type(self).__name__) + ', '.join(stuff) + ")")

    # constructors...
    @classmethod
    def from_resultset(cls, rset, detdb,
                       axis_name='dets',
                       prefix='dets:'):
        # Determine the dets axis columns
        dets_cols = {}
        for k in rset.keys:
            if k.startswith(prefix):
                dets_cols[k] = k[len(prefix):]
        # Get the detector names for entry.
        if prefix + 'name' in dets_cols:
            dets = rset[prefix + 'name']
            self = cls(LabelAxis(axis_name, dets))
            for k in rset.keys:
                if not k.startswith(prefix):
                    self.wrap(k, rset[k], [(0, axis_name)])
        else:
            # Generate the expansion map...
            if detdb is None:
                raise RuntimeError(
                    'Expansion to dets axes requires detdb '
                    'but None was not passed in.')
            dets = []
            indices = []
            for row_i, row in enumerate(rset.subset(keys=dets_cols.keys())):
                props = {v: row[k] for k, v in dets_cols.items()}
                dets_i = list(detdb.dets(props=props)['name'])
                assert all(d not in dets for d in dets_i)  # Not disjoint!
                dets.extend(dets_i)
                indices.extend([row_i] * len(dets_i))
            indices = np.array(indices)
            self = cls(LabelAxis(axis_name, dets))
            for k in rset.keys:
                if not k.startswith(prefix):
                    self.wrap(k, rset[k][indices], [(0, axis_name)])
        return self

    def restrict_dets(self, restriction, detdb=None):
        # Just convert the restriction to a list of dets, compare to
        # what we have, and return the reduced result.
        props = {k[len('dets:'):]: v for k, v in restriction.items()
                 if k.startswith('dets:')}
        if len(props) == 0:
            return self
        restricted_dets = detdb.dets(props=props)
        ax2 = LabelAxis('new_dets', restricted_dets['name'])
        new_ax, i0, i1 = self._axes['dets'].intersection(ax2, True)
        return self.restrict('dets', new_ax.vals)

    @staticmethod
    def concatenate(items, axis=0):
        """Concatenate multiple AxisManagers along the specified axis, which
        can be an integer (corresponding to the order in
        items[0]._axes) or the string name of the axis.

        This operation is difficult to sanity check so it's best to
        use it only in simple, controlled cases!  The first item is
        given significant privilege in defining what fields are
        relevant.  Only fields actually referencing the target axis
        will be propagated in this operation.

        """
        if not isinstance(axis, str):
            axis = list(items[0]._axes.keys())[axis]
        fields = []
        for name in items[0]._fields.keys():
            ax_dim = None
            for i, ax in enumerate(items[0]._assignments[name]):
                if ax == axis:
                    if ax_dim is not None:
                        raise ValueError('Entry %s has axis %s on more than '
                                         '1 dimension.' % (name, axis))
                    ax_dim = i
            if ax_dim is not None:
                fields.append((name, ax_dim))
        # Design the new axis.
        vals = np.hstack([item._axes[axis].vals for item in items])
        new_ax = LabelAxis(axis, vals)
        # Concatenate each entry.
        new_data = {}
        for name, ax_dim in fields:
            shape0 = None
            keepers = []
            for item in items:
                shape1 = list(item._fields[name].shape)
                if 0 in shape1:
                    continue
                shape1[ax_dim] = -1  # This dim doesn't have to match.
                if shape0 is None:
                    shape0 = shape1
                elif shape0 != shape1:
                    raise ValueError('Field %s has incompatible shapes: '
                                     % name
                                     + '%s and %s' % (shape0, shape1))
                keepers.append(item._fields[name])
            if len(keepers) == 0:
                # Well we tried.
                keepers = [items[0]]
            # Call class-specific concatenation.
            if isinstance(keepers[0], np.ndarray):
                new_data[name] = np.concatenate(keepers, axis=ax_dim)
            else:
                # The general compatible object should have a static
                # method called concatenate.
                new_data[name] = keepers[0].concatenate(keepers, axis=ax_dim)
        # Construct.
        new_axes = []
        for ax_name, ax_def in items[0]._axes.items():
            if ax_name == axis:
                ax_def = new_ax
            new_axes.append(ax_def)
        output = AxisManager(*new_axes)
        for k, v in items[0]._assignments.items():
            axis_map = [(i, n) for i, n in enumerate(v) if n is not None]
            output.wrap(k, new_data[k], axis_map)
        return output

    # Add and remove data while maintaining internal consistency.

    def wrap(self, name, data, axis_map=None,
             accept_new=True, accept_merge=True):
        """Add data into the AxisManager.

        Arguments:

          name (str): name of the new data.

          data: The data to register.  This must be of an acceptable
            type, i.e. a numpy array or another AxisManager.

          axis_map: A list that assigns dimensions of data to
            particular Axes.  Each entry in the list must be a tuple
            with the form (dim, name) or (dim, ax), where dim is the
            index of the dimension being described, name is a string
            giving the name of an axis already described in the
            present object, and ax is an AxisInterface object.

        """
        # Don't permit AxisManager reference loops!
        if isinstance(data, AxisManager):
            assert(id(self) not in data._managed_ids())
            assert(axis_map is None)
            axis_map = [(i, v) for i, v in enumerate(data._axes.values())]
        # Promote input data to a full AxisManager, so we can call up
        # to self.merge.
        helper = AxisManager()
        assign = [None for s in data.shape]
        # Resolve each axis declaration into an axis object, and check
        # for conflict.  If an axis is passed by name only, the
        # dimensions must agree with self.  If a full axis definition
        # is included, then intersection will be performed, later.
        if axis_map is not None:
            for index, axis in axis_map:
                if not isinstance(axis, AxisInterface):
                    # So it better be a string label... that we've heard of.
                    if axis not in self._axes:
                        raise ValueError("Axis assignment refers to unknown "
                                         "axis '%s'." % axis)
                    axis = self._axes[axis]
                axis = axis.resolve(data, index)
                helper._axes[axis.name] = axis
                assign[index] = axis.name
        helper._fields[name] = data
        helper._assignments[name] = assign
        return self.merge(helper)

    def restrict_axes(self, axes, in_place=True):
        if in_place:
            dest = self
        else:
            dest = AxisManager(*self._axes.values())
            dest._assignments.update(self._assignments)
        sels = {}
        for ax in axes.values():
            if ax.name not in self._axes:
                continue
            if self._axes[ax.name].count is None:
                self._axes[ax.name] = ax
                continue
            _, sel0, sel1 = ax.intersection(self._axes[ax.name], True)
            sels[ax.name] = sel1
            self._axes[ax.name] = ax
        for k, v in self._fields.items():
            if isinstance(v, AxisManager):
                dest._fields[k] = v.restrict_axes(axes)
            else:
                # I.e. an ndarray.
                sslice = [sels.get(ax, slice(None))
                          for ax in self._assignments[k]]
                sslice = self._broadcast_selector(sslice)
                sslice = simplify_slice(tuple(sslice), v.shape)
                dest._fields[k] = v[sslice]
        return dest

    @staticmethod
    def _broadcast_selector(sslice):
        """sslice is a list of selectors, which will typically be slice(), or
        an array of indexes.  Returns a similar list of selectors, but
        with any indexing selectors promoted to a higher
        dimensionality so that the output object will be broadcast to
        the desired shape.

        For example if the input is

           (array([0,1]), slice(0,100,2), array([12,13,14]))

        then the output will be

           (array([[0],[1]]), slice(0,100,2), array([12,13,14]))

        and the result can then be used to index an array and produce
        a view with shape (2,50,3).

        """
        ex_dim = 0
        output = [s for s in sslice]
        for i in range(len(sslice) - 1, -1, -1):
            if isinstance(sslice[i], np.ndarray):
                output[i] = sslice[i].reshape(sslice[i].shape + (1,)*ex_dim)
                ex_dim += 1
        return tuple(output)

    def restrict(self, axis_name, selector, in_place=True):
        new_ax, sl = self._axes[axis_name].restriction(selector)
        for k, v in self._fields.items():
            if isinstance(v, AxisManager):
                if axis_name in v._axes:
                    self._fields[k] = v.copy().restrict(axis_name, selector)
            else:
                sslice = [sl if n == axis_name else slice(None)
                          for n in self._assignments[k]]
                sslice = self._broadcast_selector(sslice)
                self._fields[k] = v[sslice]
        self._axes[axis_name] = new_ax
        return self

    @staticmethod
    def intersection_info(*items):
        # Get the strictest intersection of each axis.
        axes_out = odict()
        for aman in items:
            for ax in aman._axes.values():
                if ax.count is None:
                    continue
                if ax.name not in axes_out:
                    axes_out[ax.name] = ax.copy()
                else:
                    axes_out[ax.name] = axes_out[ax.name].intersection(
                        ax, False)
        return axes_out

    def merge(self, *amans):
        # Get the intersected axis descriptions.
        axes_out = self.intersection_info(self, *amans)
        # Reduce the data in self, update our axes.
        self.restrict_axes(axes_out)
        # Import the other ones.
        for aman in amans:
            aman = aman.restrict_axes(axes_out, in_place=False)
            for k, v in aman._axes.items():
                if k not in self._axes:
                    self._axes[k] = v
            for k, v in aman._fields.items():
                assert(k not in self._fields)
                self._fields[k] = v
            self._assignments.update(aman._assignments)
        return self


def get_coindices(v0, v1):
    """Given vectors v0 and v1, each of which contains no duplicate
    values, determine the elements that are found in both vectors.
    Returns (vals, i0, i1), i.e. the vector of common elements and
    the vectors of indices into v0 and v1 where those elements are
    found.

    This routine will use np.intersect1d if it can.  The ordering of
    the results is different from intersect1d -- vals is not sorted,
    but rather will the elements will appear in the same order that
    they were found in v0 (so i0 is strictly increasing).

    """
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

def simplify_slice(sslice, shape):
    """Given a tuple of slices, such as what __getitem__ might produce, and the
    shape of the array it would be applied to, return a new tuple of slices that
    accomplices the same thing, but while avoiding costly general slices if possible."""
    res = []
    for n, s in zip(shape, sslice):
        os = slice(None)
        # Normal slices are OK, since they don't cause copies
        if not isinstance(s, slice) and not np.all(s == np.arange(n)):
            os = s
        res.append(os)
    return tuple(res)

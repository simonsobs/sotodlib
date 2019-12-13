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

    def copy(self):
        return LabelAxis(self.name, self.vals)

    def resolve(self, src, axis_index=None):
        if self.count is None:
            raise RuntimeError(
                'LabelAxis cannot be naively promoted from data.')
        return super().resolve(src, axis_index)

    def restriction(self, selector):
        # Selector should be list of vals.
        _vals, i0, i1 = np.intersect1d(self.vals, selector,
                                       return_indices=True)
        assert len(i1) == len(selector)  # not a strict subset!
        # Un-sort
        i0 = i0[i1]
        return LabelAxis(self.name, self.vals[i0]), i0

    def intersection(self, friend, return_slices=False):
        _vals, i0, i1 = np.intersect1d(self.vals, friend.vals,
                                       return_indices=True)
        # Maintain the order of self.
        i0, i1 = [np.array(_i) for _i in zip(*sorted(zip(i0, i1)))]
        ax = LabelAxis(self.name, self.vals[i0])
        if return_slices:
            return ax, i0, i1
        else:
            return ax


class AxisManager:
    """A container for numpy arrays (and other AxisManager objects) that
    knows about.

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
        self._fields[new_name] = self._fields.pop(name)
        self._assignments[new_name] = self._assignments.pop(name)
        return self

    def __getitem__(self, name):
        if name in self._fields:
            return self._fields[name]
        if name in self._axes:
            return self._axes[name]
        raise KeyError(name)

    def __getattr__(self, name):
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
                 + ['%s:%s' % (k, repr(v)) for k, v in self._axes.items()])
        return ("AxisManager(" + ', '.join(stuff) + ")")

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
                sslice = [sels.get(ax, slice(None))
                          for ax in self._assignments[k]]
                sslice = self._broadcast_selector(sslice)
                dest._fields[k] = v[tuple(sslice)]
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

import numpy as np
from collections import OrderedDict as odict

import scipy.sparse as sparse
## "temporary" fix to deal with scipy>1.8 changing the sparse setup
try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

from .util import get_coindices


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
    
    def rename(self, name):
        self.name = name

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
    match the shorter.  Selectors must be slice objects (with stride
    1!) or tuples to be passed into slice(), e.g. (0, 1000) or (0,
    None, 1)..

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

    Selectors must be slice objects (with stride 1!) or tuples to be
    passed into slice(), e.g. (0, 1000) or (0, None, 1).

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
        if not isinstance(selector, slice):
            sl = slice(*selector)
        else:
            sl = selector
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

    Instantiation with labels that are not strings will raise a TypeError.

    On intersection of two vectors, only elements whose names appear
    in both axes will be preserved.

    Selectors should be lists (or arrays) of label strings.

    """

    def __init__(self, name, vals=None):
        super().__init__(name)
        if vals is not None:
            if len(vals):
                vals = np.array(vals)
            else:
                vals = np.array([], dtype=np.str_)
            if vals.dtype.type is not np.str_:
                raise TypeError(
                        'LabelAxis labels must be strings not %s' % vals.dtype)
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
                self._axes[a.name] = a.copy()
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
            if np.isscalar(v) or v is None:
                out._fields[k] = v
            else:
                out._fields[k] = v.copy()
        for k, v in self._assignments.items():
            out._assignments[k] = v.copy()
        return out

    def _managed_ids(self):
        ids = [id(self)]
        for v in self._fields.values():
            if isinstance(v, AxisManager):
                ids.extend(v._managed_ids())
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
    
    def add_axis(self, a):
        assert isinstance( a, AxisInterface)
        self._axes[a.name] = a.copy()

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

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def shape_str(self, name):
        if np.isscalar(self._fields[name]) or self._fields[name] is None:
            return ''
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
        return ("{}(".format(type(self).__name__)
                + ', '.join(stuff).replace('[]', '') + ")")

    @staticmethod
    def concatenate(items, axis=0, other_fields='exact'):
        """Concatenate multiple AxisManagers along the specified axis, which
        can be an integer (corresponding to the order in
        items[0]._axes) or the string name of the axis.

        This operation is difficult to sanity check so it's best to
        use it only in simple, controlled cases!  The first item is
        given significant privilege in defining what fields are
        relevant.  Fields that appear in the first item, but do not
        share the target axis, will be treated as follows depending on
        the value of other_fields:

        - If other_fields='exact' will compare entries in all items
          and if they're identical will add it. Otherwise will fail with
          a ValueError.
        - If other_fields='fail', the function will fail with a ValueError.
        - If other_fields='first', the values from the first element
          of items will be copied into the output.
        - If other_fields='drop', the fields will simply be ignored
          (and the output will only contain fields that share the
          target axis).

        """
        assert other_fields in ['exact', 'fail', 'first', 'drop']
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
                keepers = [items[0]._fields[name]]
            # Call class-specific concatenation if needed.
            if isinstance(keepers[0], AxisManager):
                new_data[name] = AxisManager.concatenate(
                    keepers, axis=ax_dim, other_fields=other_fields)
            elif isinstance(keepers[0], np.ndarray):
                new_data[name] = np.concatenate(keepers, axis=ax_dim)
            elif isinstance(keepers[0], csr_array):
                # Note in scipy 1.11 the default format for vstack
                # and/or hstack seems to have change, as we started
                # seeing induced cso format here.  Force preservation
                # of incoming format.
                if ax_dim == 0:
                    new_data[name] = sparse.vstack(keepers, format=keepers[0].format)
                elif ax_dim == 1:
                    new_data[name] = sparse.hstack(keepers, format=keepers[0].format)
                else:
                    raise ValueError('sparse arrays cannot concatenate along '
                                     f'axes greater than 1, received {ax_dim}')
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
            if isinstance(items[0][k], AxisManager):
                axis_map = None  # wrap doesn't like this.
            if k in new_data:
                output.wrap(k, new_data[k], axis_map)
            else:
                if other_fields == "exact":
                    ## if every item named k is a scalar 
                    err_msg = (f"The field '{k}' does not share axis '{axis}'; " 
                              f"{k} is not identical across all items " 
                              f"pass other_fields='drop' or 'first' or else " 
                              f"remove this field from the targets.")
                                
                    if np.any([np.isscalar(i[k]) for i in items]):
                        if not np.all([np.isscalar(i[k]) for i in items]):
                            raise ValueError(err_msg)
                        if not np.all( [i[k]==items[0][k] for i in items]):
                            raise ValueError(err_msg)
                        output.wrap(k, items[0][k], axis_map)
                        continue
                        
                    elif not np.all([i[k].shape==items[0][k].shape for i in items]):
                        raise ValueError(err_msg)
                    elif not np.all([i[k]==items[0][k] for i in items]):
                        raise ValueError(err_msg)
                        
                    output.wrap(k, items[0][k].copy(), axis_map)
                    
                elif other_fields == 'fail':
                    raise ValueError(
                        f"The field '{k}' does not share axis '{axis}'; "
                        f"pass other_fields='drop' or 'first' or else "
                        f"remove this field from the targets.")
                elif other_fields == 'first':
                    # Just copy it.
                    if np.isscalar(items[0][k]):
                        output.wrap(k, items[0][k], axis_map)
                    else:
                        output.wrap(k, items[0][k].copy(), axis_map)
                elif other_fields == 'drop':
                    pass
        return output

    # Add and remove data while maintaining internal consistency.

    def wrap(self, name, data, axis_map=None,
             accept_new=True, accept_merge=True):
        """Add data into the AxisManager.

        Arguments:

          name (str): name of the new data.

          data: The data to register.  This must be of an acceptable
            type, i.e. a numpy array or another AxisManager.
            If scalar (or None) then data will be directly added to
            _fields with no associated axis.

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
        # Handle scalars
        if np.isscalar(data) or data is None:
            if name in self._fields:
                raise ValueError(f'Key: {name} already found in {self}')
            if np.iscomplex(data):
                # Complex values aren't supported by HDF scheme right now.
                raise ValueError(f'Cannot store complex value as scalar.')
            if isinstance(data, (np.integer, np.floating, np.str_, np.bool_)):
                # Convert sneaky numpy scalars to native python int/float/str
                data = data.item()
            self._fields[name] = data
            self._assignments[name] = []
            return self
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

    def wrap_new(self, name, shape=None, cls=None, **kwargs):
        """Create a new object and wrap it, with axes mapped.  The shape can
        include axis names instead of ints, and that will cause the
        new object to be dimensioned properly and its axes mapped.

        Args:

          name (str): name of the new data.

          shape (tuple of int and std): shape in the same sense as
            numpy, except that instead of int it is allowed to pass
            the name of a managed axis.

          cls (callable): Constructor that should be used to construct
            the object; it will be called with all kwargs passed to
            this function, and with the resolved shape as described
            here.  Defaults to numpy.ndarray.

        Examples:

            Construct a 2d array and assign it the name
            'boresight_quat', with its first axis mapped to the
            AxisManager tod's "samps" axis:

            >>> tod.wrap_new('boresight_quat', shape=('samps', 4), dtype='float64')

            Create a new empty RangesMatrix, carrying a per-det, per-samp flags:

            >>> tod.wrap_new('glitch_flags', shape=('dets', 'samps'),
                             cls=so3g.proj.RangesMatrix.zeros)

        """
        if cls is None:
            cls = np.zeros
        # Turn the shape into a tuple of ints and an axis map.
        shape_ints, axis_map = [], []
        for dim, s in enumerate(shape):
            if isinstance(s, int):
                shape_ints.append(s)
            elif isinstance(s, str):
                if s in self._axes:
                    shape_ints.append(self._axes[s].count)
                    axis_map.append((dim, self._axes[s]))
                else:
                    raise ValueError(f'shape includes axis "{s}" which is '
                                     f'not in _axes: {self._axes}')
            elif isinstance(s, AxisInterface):
                # Sure, why not.
                shape_ints.append(s.count)
                axis_map.append((dim, s.copy()))
        data = cls(shape=shape_ints, **kwargs)
        return self.wrap(name, data, axis_map=axis_map)[name]

    def restrict_axes(self, axes, in_place=True):
        """Restrict this AxisManager by intersecting it with a set of Axis
        definitions.

        Arguments:
          axes (list or dict of Axis):
          in_place (bool): If in_place == True, the intersection is
            applied to self.  Otherwise, a new object is returned,
            with data copied out.

        Returns:
          The restricted AxisManager.
        """
        if in_place:
            dest = self
        else:
            dest = self.copy(axes_only=True)
            dest._assignments.update(self._assignments)
        sels = {}
        # If simple list/tuple of Axes is passed in, convert to dict
        if not isinstance(axes, dict):
            axes = {ax.name: ax for ax in axes}
        for name, ax in axes.items():
            if name not in dest._axes:
                continue
            if dest._axes[name].count is None:
                dest._axes[name] = ax
                continue
            _, sel0, sel1 = ax.intersection(dest._axes[name], True)
            sels[name] = sel1
            dest._axes[ax.name] = ax
        for k, v in self._fields.items():
            if isinstance(v, AxisManager):
                dest._fields[k] = v.restrict_axes(axes, in_place=in_place)
            elif np.isscalar(v) or v is None:
                dest._fields[k] = v
            else:
                # I.e. an ndarray.
                sslice = [sels.get(ax, slice(None))
                          for ax in dest._assignments[k]]
                sslice = tuple(dest._broadcast_selector(sslice))
                sslice = simplify_slice(sslice, v.shape)
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
        """Restrict the AxisManager by selecting a subset of items in some
        Axis.  The Axis definition and all data fields mapped to that
        axis will be modified.
        
        Arguments:
          axis_name (str): The name of the Axis.
          selector (slice or special): Selector, in a form understood
            by the underlying Axis class (see the .restriction method
            for the Axis).
          in_place (bool): If True, modifications are made to this
            object.  Otherwise, a new object with the restriction
            applied is returned.
        
        Returns:
          The AxisManager with restrictions applied.
        
        """
        if in_place:
            dest = self
        else:
            dest = self.copy(axes_only=True)
            dest._assignments.update(self._assignments)
        new_ax, sl = dest._axes[axis_name].restriction(selector)
        for k, v in self._fields.items():
            if isinstance(v, AxisManager):
                dest._fields[k] = v.copy()
                if axis_name in v._axes:
                    dest._fields[k].restrict(
                        axis_name, 
                        selector, 
                        ## copies of axes made above
                        in_place=True 
                    )
            elif np.isscalar(v) or v is None:
                dest._fields[k] = v
            else:
                sslice = [sl if n == axis_name else slice(None)
                          for n in dest._assignments[k]]
                sslice = dest._broadcast_selector(sslice)
                if in_place:
                    dest._fields[k] = v[sslice]
                else:
                    dest._fields[k] = v[sslice].copy()
        dest._axes[axis_name] = new_ax
        return dest

    @staticmethod
    def intersection_info(*items):
        """Given a list of AxisManagers, scan the axes and combine (intersect)
        any common axes.  Returns a dict that maps axis name to
        restricted Axis object.

        """
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
        """Merge the data from other AxisMangers into this one.  Axes with the
        same name will be intersected.

        """
        # Before messing with anything, check for key interference.
        fields = set(self._fields.keys())
        for aman in amans:
            newf = set(aman._fields.keys())
            both = fields.intersection(newf)
            if len(both):
                raise ValueError(f'Key conflict: more than one merge target '
                                 f'shares keys: {both}')
            fields.update(newf)

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
                assert(k not in self._fields)  # Should have been caught in pre-check
                self._fields[k] = v
            self._assignments.update(aman._assignments)
        return self

    def save(self, dest, group=None, overwrite=False, compression=None):
        """Write this AxisManager data to an HDF5 group.  This is an
        experimental feature primarily intended to assist with
        debugging.  The schema is subject to change, and it's possible
        that not all objects supported by AxisManager can be
        serialized.

        Args:
          dest (str or h5py.Group): Place to save it (in combination
            with group).
          group (str or None): Group within the HDF5 file (relative to
            dest).
          overwrite (bool): If True, remove any existing thing at the
            specified address before writing there.
          compression (str or None): Compression filter to apply. E.g.
            'gzip'. This string is passed directly to HDF5 dataset
            routines.

        Notes:
          If dest is a string, it is taken to be an HDF5 filename and
          is opened in 'a' mode.  The group, in that case, is the
          full group name in the file where the data should be written.

          If dest is an h5py.Group, the group is the group name in the
          file relative to dest.

          The overwrite argument only matters if group is passed as a
          string.  A RuntimeError is raised if the group address
          already exists and overwrite==False.

          For example, these are equivalent::

            # Filename + group address:
            axisman.save('test.h5', 'x/y/z')

            # Open h5py.File + group address:
            with h5py.File('test.h5', 'a') as h:
              axisman.save(h, 'x/y/z')

            # Partial group address
            with h5py.File('test.h5', 'a') as h:
              g = h.create_group('x/y')
              axisman.save(g, 'z')

          When passing a filename, the code probably won't use a
          context manager... so if you want that protection, open your
          own h5py.File as in the 2nd and 3rd example.

        """
        from .axisman_io import _save_axisman
        return _save_axisman(self, dest, group=group, overwrite=overwrite, compression=compression)

    @classmethod
    def load(cls, src, group=None):
        """Load a saved AxisManager from an HDF5 file and return it.  See docs
        for save() function.

        The (src, group) args are combined in the same way as (dest,
        group) in the save function.  Examples::

          axisman = AxisManager.load('test.h5', 'x/y/z')

          with h5py.File('test.h5', 'r') as h:
            axisman = AxisManager.load(h, 'x/y/z')
        """
        from .axisman_io import _load_axisman
        return _load_axisman(src, group, cls)


def simplify_slice(sslice, shape):
    """Given a tuple of slices, such as what __getitem__ might produce, and the
    shape of the array it would be applied to, return a new tuple of slices that
    accomplices the same thing, but while avoiding costly general slices if possible."""
    res = []
    for n, s in zip(shape, sslice):
        # Numpy arrays slicing is expensive, and unnecessary if they just select
        # the same elemnts in the same order
        if isinstance(s, np.ndarray):
            # Is this a trivial numpy slice? If so, replace it
            if s.size == n and np.all(s == np.arange(n)):
                res.append(slice(None))
            # Otherwise bail, and keep the whole original
            else:
                return sslice
        # For anything else just pass it through. This includes normal slices
        else: res.append(s)
    return tuple(res)

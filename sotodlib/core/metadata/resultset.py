import numpy as np
from collections import OrderedDict


class ResultSet(object):
    """ResultSet is a special container for holding the results of
    database queries, i.e. columnar data.  The repr of a ResultSet
    states the name of its columns, and the number of rows::

      >>> print(rset)
      ResultSet<[array_code,freq_code], 17094 rows>

    You can access the column names in .keys::

      >>> print(rset.keys)
      ['array_code', 'freq_code']

    You can request a column by name, and a numpy array of values will
    be constructed for you:

      >>> rset['array_code']
      array(['LF1', 'LF1', 'LF1', ..., 'LF1', 'LF1', 'LF1'], dtype='<U3')

    You can request a row by number, and a dict will be constructed
    for you:

      >>> rset[10]
      {'base.array_code': 'LF1', 'base.freq_code': 'f027'}

    (You can also access the raw row data in .rows, which is a simple
    list of tuples.)

    You can get a structured numpy array using:

      >>> ret.asarray()
      array([('LF1', 'f027'), ('LF1', 'f027'), ('LF1', 'f027'), ...,
              ('LF1', 'f027'), ('LF1', 'f027'), ('LF1', 'f027')],
            dtype=[('array_code', '<U3'), ('freq_code', '<U4')])

    Slicing works along the row axis; and you can combine two results.
    So you could reorganize results like this, if you wanted:

      >>> rset[::2] + rset[1::2]
      ResultSet<[array_code,freq_code], 17094 rows>

    Finally, the .distinct() method returns a ResultSet containing the
    distinct elements:

      >>> rset.distinct()
      ResultSet<[array_code,freq_code], 14 rows>

    """

    #: Once instantiated, a list of the names of the ResultSet
    #: columns.
    keys = None

    #: Once instantiated, a list of the raw data tuples.
    rows = None

    def __init__(self, keys, src=None):
        self.keys = list(keys)
        if src is None:
            self.rows = []
        else:
            self.rows = [tuple(x) for x in src]

    @classmethod
    def from_friend(cls, source):
        if isinstance(source, np.ndarray):
            keys = source.dtype.names # structured array?
            return cls(keys, list(source))
        if isinstance(source, ResultSet):
            return cls(source.keys, source.rows)

    def copy(self):
        return self.__class__(self.keys, self.rows)

    def subset(self, keys=None, rows=None):
        """Returns a copy of the object, selecting only the keys and rows
        specified.

        Arguments:
          keys: a list of keys to keep.  None keeps all.

          rows: a list or array of the integers representing which
            rows to keep.  This can also be specified as an array of
            bools, of the same length as self.rows, to select row by
            row.  None keeps all.

        """
        if keys is None:
            keys = self.keys
            def key_sel_func(row):
                return row
        else:
            key_idx = [self.keys.index(k) for k in keys]
            def key_sel_func(row):
                return [row[i] for i in key_idx]
        if rows is None:
            new_rows = map(key_sel_func, self.rows)
        elif isinstance(rows, np.ndarray) and rows.dtype == bool:
            assert(len(rows) == len(self.rows))
            new_rows = [key_sel_func(r) for r, s in zip(self.rows, rows) if s]
        else:
            new_rows = [key_sel_func(self.rows[i]) for i in rows]
        return self.__class__(keys, new_rows)

    @classmethod
    def from_cursor(cls, cursor, keys=None):

        """Create a ResultSet using the results stored in cursor, an
        sqlite.Cursor object.  The cursor must have be configured so
        that .description is populated.

        """
        if keys is None:
            keys = [c[0] for c in cursor.description]
        self = cls(keys)
        self.rows = [tuple(r) for r in cursor]
        return self

    def asarray(self, simplify_keys=False):
        """
        Returns a numpy structured array containing a copy of this
        ResultSet.  The names of the fields are taken from self.keys.
        If simplify_keys=True, then the keys are stripped of any
        prefix; an error is thrown if this yields duplicate key names.
        """
        keys = [k for k in self.keys]
        if simplify_keys:  # remove prefixes
            keys = [k.split('.')[-1] for k in keys]
            assert(len(set(keys)) == len(keys))  # distinct.
        columns = tuple(map(np.array, zip(*self.rows)))
        dtype = [(k, c.dtype, c.shape[1:]) for k, c in zip(keys, columns)]
        output = np.ndarray(shape=len(columns[0]), dtype=dtype)
        for k, c in zip(keys, columns):
            output[k] = c
        return output

    def distinct(self):
        """
        Returns a ResultSet that is a copy of the present one, with
        duplicates removed.  The rows are sorted (according to python
        sort).
        """
        return self.__class__(self.keys, sorted(list(set(self.rows))))

    def strip(self, patterns=[]):
        """For any keys that start with a string in patterns, remove that
        string prefix from the key.  Operates in place.

        """
        for i, k in enumerate(self.keys):
            for p in patterns:
                if k.startswith(p):
                    self.keys[i] = k[len(p):]
                    break
        assert(len(self.keys) == len(set(self.keys)))

    # Todo: decide whether axisman conversion code lives here or axisman...
    def axismanager(self, detdb=None):
        from sotodlib import core
        return core.AxisManager.from_resultset(self, detdb=detdb)

    # Todo: move this to the right place, too.
    def restrict_dets(self, restriction, detdb=None):
        # There are 4 classes of keys:
        # - dets:* keys appearing only in restriction
        # - dets:* keys appearing only in self
        # - dets:* keys appearing in both
        # - other.
        new_keys = [k for k in restriction if k.startswith('dets:')]
        match_keys = []
        for k in self.keys:
            if k in new_keys:
                match_keys.append(k)
                new_keys.remove(k)
        other_keys = [k for k in self.keys if k not in match_keys]
        output_keys = new_keys + match_keys + other_keys # disjoint.
        output_rows = []
        for row in self:
            row = dict(row)  # copy
            for k in match_keys:
                if row[k] != restriction[k]:
                    break
            else:
                # You passed.
                row.update({k: restriction[k] for k in new_keys})
            output_rows.append([row[k] for k in output_keys])
        # That's all.
        return self.__class__(output_keys, output_rows)

    # Everything else is just implementing container-like behavior

    def __repr__(self):
        keystr = 'empty'
        if self.keys is not None:
            keystr = ','.join(self.keys)
        return ('{}<[{}], {} rows>'.format(self.__class__.__name__,
                                           keystr, len(self)))

    def __len__(self):
        return len(self.rows)

    def append(self, item):
        vals = []
        for k in self.keys:
            if k not in item.keys():
                raise ValueError(f"Item to append must include key '{k}'")
            vals.append(item[k])
        self.rows.append(tuple(vals))

    def extend(self, items):
        if not isinstance(items, ResultSet):
            raise TypeError("Extension only valid for two ResultSet objects.")
        if self.keys != items.keys:
            raise ValueError("Keys do not match: {} <- {}".format(
                self.keys, items.keys))
        self.rows.extend(items.rows)

    def __getitem__(self, item):
        # Simple row look-up... convert to dict.
        if isinstance(item, int):
            return OrderedDict([(k,v) for k, v in
                                zip(self.keys, self.rows[item])])
        # Look-up by column...
        if isinstance(item, str):
            index = self.keys.index(item)
            return np.array([x[index] for x in self.rows])
        # Slicing.
        output = self.__class__(self.keys, self.rows[item])
        return output

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __add__(self, other):
        output = self.copy()
        output += other
        return output

    @staticmethod
    def concatenate(items, axis=0):
        assert(axis == 0)
        output = items[0].copy()
        for item in items[1:]:
            output += item
        return output

    def merge(self, src):
        """Merge with src, which must have same number of rows as self.
        Duplicate columns are not allowed.

        """
        if len(self) != len(src):
            raise ValueError("self and src have different numbers of rows.")
        for k in src.keys:
            if k in self.keys:
                raise ValueError("Duplicate key: %s" % k)
        new_keys = self.keys + src.keys
        new_rows = [r0 + r1 for r0, r1 in zip(self.rows, src.rows)]
        self.keys, self.rows = new_keys, new_rows

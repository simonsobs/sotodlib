import sqlite3
import gzip
import os

from .resultset import ResultSet
from . import common


class SchemaError(Exception):
    """
    This is raised in cases where the code detects a schema violation,
    such as tables not having the required named columns.
    """
    pass


class IntervalError(Exception):
    """
    This is raised in cases where the code detects that time intervals
    in a property table are of negative size or overlap with other
    intervals for the same det_id.
    """
    pass


TABLE_DEFS = {
    'dets': [
        "`id` integer primary key autoincrement",
        "`name` varchar(256) unique",
        ],
}
SPECIAL_COLS = ['det_id', 'time0', 'time1']


class DetDb(object):
    """
    Detector database.  The database stores data about a set of
    detectors.

    The ``dets`` table lists all valid detectors, associating a
    (time-invariant) name to each ``id``.

    The other tables in the database are user configurable "property
    tables" that must obey certain rules:

    1. They have at least the following columns:

       - ``det_id`` integer
       - ``time0`` integer (unix timestamp)
       - ``time1`` integer (unix timestamp)

    2. The values time0 and time1 define an interval ``[time0,time1)``
       over which the data in the row is valid.  Every row shall
       respect the constraint that ``time0 <= time1``.

    3. No two rows in a property table shall have the same ``det_id``
       and overlapping time intervals.  Note that since the intervals
       are half-open, the intervals [t0, t1) and [t1, t2) do not
       overlap.
    """

    #: A time-range that is meant to signify "all reasonable times";
    #: in this case it spans from years 1970 - 2096.
    ALWAYS = (0., 4e9)

    #: Column definitions (a list of strings) that must appear in all
    #: Property Tables.
    TABLE_TEMPLATE = [
        "`det_id` integer",
        "`time0` integer",
        "`time1` integer",
    ]

    def __init__(self, map_file=None, init_db=True, readonly=False):
        """Instantiate a DetDb.

        If map_file is provided, the database will
        be connected to the indicated sqlite file on disk, and any
        changes made to this object be written back to the file.

        Args:
            map_file (string): sqlite database file to map.  Defaults to
                ':memory:'.
            init_db (bool): If True, attempt to create the database
                tables.
            readonly (bool): If True, the database file will be mapped
                in read-only mode.  Not valid on dbs held in :memory:.

        """
        if init_db and readonly:
            raise ValueError("Cannot initialize a read-only DB")
        self._readonly = readonly
        if isinstance(map_file, sqlite3.Connection):
            self.conn = map_file
        else:
            self.conn = common.sqlite_connect(
                filename=map_file,
                mode=("r" if readonly else "w"),
            )
        self.conn.row_factory = sqlite3.Row  # access columns by name

        if init_db:
            # Create dets table if not found.
            c = self.conn.cursor()
            c.execute("SELECT name FROM sqlite_master "
                      "WHERE type='table' and name not like 'sqlite_%';")
            tables = [r[0] for r in c]
            if 'dets' not in tables:
                self.create_table('dets', TABLE_DEFS['dets'], raw=True)

    def __len__(self):
        return self.conn.execute('select count(id) from dets').fetchone()[0]

    def _get_property_tables(self):
        """Return a list of all property tables."""
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE "
                  "type='table' and name not like 'sqlite_%' "
                  "and name != 'dets';")
        return [str(x[0]) for x in c]

    def _get_property_list(self, table_name):
        """Return a list of all property columns for the specified property
        table.

        """
        c = self.conn.cursor()
        c.execute(f'PRAGMA table_info({table_name})')
        return [r[1] for r in c if r[1] not in SPECIAL_COLS]

    def validate(self):
        """
        Checks that the database is following internal rules.
        Specifically we check that a ``dets`` table exists and has the
        necessary columns; then we check that all other tables do not
        have overlapping property intervals.  Raises SchemaError in
        the first case, IntervalError in the second.
        """
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' "
                  "and name not like 'sqlite_%';")
        tables = [r[0] for r in c]
        if 'dets' not in tables:
            raise SchemaError("Database does not contain a `dets` table.")
        tables.remove('dets')
        for t in tables:
            try:
                c.execute("SELECT det_id,time0,time1 from `%s` "
                          "order by det_id,time0" % t)
            except sqlite3.OperationalError as e:
                raise SchemaError("Key columns not found in table `%s`" % t)
            last_id, last_t1 = None, None
            for r in c:
                _id, _t0, _t1 = r
                if (_t1 < _t0):
                    raise IntervalError(
                        "Negative size time interval for table %s, "
                        "det_id=%i" % (t, _id))
                if _id == last_id:
                    if _t0 < last_t1:
                        raise IntervalError(
                            "Overlapping interval for table %s, "
                            "det_id=%i" % (t, _id))
                    last_t1 = _t1
                else:
                    last_id, last_t1 = _id, _t1

    def create_table(self, table_name, column_defs, raw=False, commit=True):
        """Add a property table to the database.

        Args:
          table_name (str): The name of the new table.
          column_defs (list): A list of sqlite column definition
            strings.
          raw (bool): See below.
          commit (bool): Whether to commit the changes to the db.

        The special columns `det`, `time0` and `time1` will be
        pre-pended unless raw=True.  An example of column_defs is::

          column_defs=[
            "`x_pos` float",
            "`y_pos` float",
          ]

        """
        if self._readonly:
            raise RuntimeError("Cannot use create_table() on a read-only DB")
        c = self.conn.cursor()
        pre_cols = self.TABLE_TEMPLATE
        if raw:
            pre_cols = []
        q = ('create table if not exists `%s` (' % table_name +
             ','.join(pre_cols + column_defs) + ')')
        c.execute(q)
        if commit:
            self.conn.commit()
        return self

    def copy(self, map_file=None, overwrite=False):
        """
        Duplicate the current database into a new database object, and
        return it.  If map_file is specified, the new database will be
        connected to that sqlite file on disk.  Note that a quick way
        of writing a Db to disk to call copy(map_file=...) and then
        simply discard the returned object.
        """
        if map_file is not None and os.path.exists(map_file):
            if overwrite:
                os.remove(map_file)
            else:
                raise RuntimeError("Output file %s exists (overwrite=True "
                                   "to overwrite)." % map_file)
        new_db = DetDb(map_file=map_file, init_db=False, readonly=False)
        script = ' '.join(self.conn.iterdump())
        new_db.conn.executescript(script)
        return new_db

    def to_file(self, filename, overwrite=True, fmt=None):
        """Write the present database to the indicated filename.

        Args:
          filename (str): the path to the output file.
          overwrite (bool): whether an existing file should be
            overwritten.
          fmt (str): 'sqlite', 'dump', or 'gz'.  Defaults to 'sqlite'
            unless the filename ends with '.gz', in which it is 'gz'.

        """
        if fmt is None:
            if filename.endswith('.gz'):
                fmt = 'gz'
            else:
                fmt = 'sqlite'
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f'File {filename} exists; remove or pass '
                               'overwrite=True.')
        if fmt == 'sqlite':
            _ = self.copy(map_file=filename, overwrite=overwrite)
        elif fmt == 'dump':
            with open(filename, 'w') as fout:
                for line in self.conn.iterdump():
                    fout.write(line)
        elif fmt == 'gz':
            with gzip.GzipFile(filename, 'wb') as fout:
                for line in self.conn.iterdump():
                    fout.write(line.encode('utf-8'))
        else:
            raise RuntimeError(f'Unknown format "{fmt}" requested.')

    @classmethod
    def from_file(cls, filename, fmt=None, force_new_db=True):
        """This method calls
            :func:`sotodlib.core.metadata.common.sqlite_from_file`
        """
        conn = common.sqlite_from_file(filename, fmt=fmt,
                                       force_new_db=force_new_db)
        return cls(conn, init_db=False)


    def reduce(self, dets=None, time0=None, time1=None,
               inplace=False):
        """Discard information from the database unless it is "relevant".

        Args:
          dets (list or ResultSet): A list of detectors names that
            are relevant.  If this is a ResultSet, the 'name' column
            is used.  If None, all dets are relevant.
          time0 (int): If time1 is None, then a property's time range
            must contain this time for it to be considered relevant.
            If time1 is not None, see below.
          time1 (int): Along with time0, forms a time range that have
            non-zero intersection with the property's time range for
            the entry to be considered relevant.
          inplace (bool): Whether to act on the present object, or to
            return a modified copy.

        Returns the reduced data (which is self, if inplace is True).

        """
        if inplace:
            if self._readonly:
                raise RuntimeError(
                    "Cannot do inplace reduce of a read-only DB."
                )
        else:
            return self.copy().reduce(dets, time0, time1, inplace=True)

        time_clause = '0'
        if time0 is not None:
            if time1 is None:
                time_clause = '(%i >= time0) or (time1 > %i)' % (time0, time0)
            else:
                assert(time1 >= time0)
                time_clause = '(%i <= time1) or (time0 < %i)' % (time0, time1)
        else:
            assert(time1 is None)

        c = self.conn.cursor()
        if dets is not None:
            # Convert to a list of names.
            if isinstance(dets, ResultSet):
                dets = dets['name']
            # Create a temporary table to list dets we're keeping.
            self.create_table('_keepers', ["`name` varchar"], raw=True)
            for d in dets:
                c.execute('insert into _keepers (name) values (?)', (d,))
            # Intersect dets against _keepers, discard _keepers.
            c.execute('delete from dets where not dets.name in '
                      '(select name from _keepers)')
            c.execute('drop table _keepers')
            det_clause = 'not det_id in (select id from dets)'
        else:
            det_clause = '0'

        # Remove orphaned rows from other tables.
        for t in self._get_property_tables():
            c.execute('delete from %s where %s or %s' %
                      (t, det_clause, time_clause))
        self.conn.commit()

        # Compact the db.
        c.execute('vacuum')
        self.conn.commit()

        return self

    # Construction.
    def get_id(self, name, commit=True, create=True):
        """Returns a detector's internal id.  If the detector isn't in the
        dets table yet, and create==True, then it is added.

        """
        c = self.conn.execute('select id from dets where name=?',
                              (name,))
        db_id = c.fetchone()
        if db_id is not None:
            return db_id[0]
        if not create:
            raise ValueError("Detector {} not in table and "
                             "create=False".format(name))
        c = self.conn.execute('insert into dets (name) values (?)',
                              (name,))
        db_id = c.lastrowid
        if commit:
            self.conn.commit()
        return db_id

    def add_props(self, table_, name_, time_range=None, commit=True, **kw):
        """Add property information for a detector.

        Args:
          table_ (str): The property table name.
          name_ (str): The detector name.
          time_range (pair of ints): The time range over which the
            property value is applicable.
          commit (bool): Whether or not to commit the db.

        All other keyword arguments are interpreted as data to write
        into the property table.

        """
        if self._readonly:
            raise RuntimeError("Cannot add_props() on a read-only DB")
        if time_range is None:
            time_range = self.ALWAYS
        row_id = self.get_id(name_, create=True, commit=False)
        keys, values = zip(*kw.items())
        key_string = ('det_id,time0,time1' +
                      (''.join([',`{}`'] * len(keys)).format(*keys)))
        val_string = '?,?,?' + ''.join([',?'] * len(keys))
        q = (f'insert into {table_} ({key_string}) '
             f'values ({val_string})')
        self.conn.execute(
            q, (row_id, time_range[0], time_range[1]) + tuple(values))
        if commit:
            self.conn.commit()

    # Forward lookup.
    def dets(self, timestamp=None, props={}):
        """
        Get a list of detectors matching the conditions listed in the
        "props" dict.  If timestamp is not provided, then time range
        restriction is not applied.

        Returns a list of detector names.
        """
        # Accumulate a query, and args.
        q = 'select dets.name as name from dets'
        args = []

        # Whatever we were given, convert it to a list of dicts.
        if isinstance(props, ResultSet):
            prop_sets = list(props.distinct())
        elif isinstance(props, dict):
            prop_sets = [props]
        else:
            prop_sets = props

        # Expand each match row into query and args.
        other_tables = []
        row_wheres = []
        for props in prop_sets:
            r = []
            for m, v in props.items():
                if '.' in m:
                    t, m = m.split('.', 1)
                else:
                    t, m = 'base', m
                if t not in other_tables:
                    other_tables.append(t)
                r.append((f'{t}.{m}=?', v))
            row_wheres.append(r)

        # Joins.
        for t in other_tables:
            q += ' join %s on %s.det_id=dets.id' % (t, t)
        # Accumulate restriction strings...
        restricts = []
        if timestamp is not None:
            time_clause = '%s.time0 <= ? and ? < %s.time1'
            for t in other_tables:
                restricts.append(time_clause % (t, t))
                args.extend([timestamp, timestamp])
        # Matching of each prop_set.
        prop_criteria = []
        for r in row_wheres:
            if len(r) == 0:
                prop_criteria.append('1')
                continue
            conds, vals = zip(*r)
            prop_criteria.append(' and '.join(conds))
            args.extend(vals)

        if len(prop_criteria) == 0:
            prop_criteria.append('0')
        restricts.append(' or '.join(['(' + pc + ')' for pc in prop_criteria]))

        # Apply restrictions...
        if (restricts):
            q += ' where ' + ' and '.join(restricts)
        q = q + ' group by id'
        c = self.conn.cursor()
        c.execute(q, tuple(args))
        return ResultSet.from_cursor(c)

    # Reverse lookup.
    def props(self, dets=None, timestamp=None, props=None,
              concise=False):
        """
        Get the value of the properties listed in props, for each detector
        identified in dets (a list of strings, or a ResultSet with a
        column called 'name').
        """
        # Create temporary table
        c = self.conn.cursor()
        c.execute('begin transaction')
        c.execute('drop table if exists _dets')
        c.execute('create temp table _dets (`name` varchar(32))')
        q = 'insert into _dets (name) values (?)'
        if dets is None:
            dets = self.dets()['name']
        if isinstance(dets, ResultSet):
            dets = dets['name']
        elif isinstance(dets, dict):
            # This is intended to handle a single row extracted from a
            # ResultSet.
            dets = [dets['name']]
        for a in dets:
            c.execute(q, (a,))
        c.execute('end transaction')

        # Expand props argument.
        if props is None:
            props = [t + '.' for t in self._get_property_tables()]

        props, props_ = [], props
        for p in props_:
            if p.endswith('.'):
                table_p = self._get_property_list(p[:-1])
                props.extend([p + _p for _p in table_p])
            else:
                props.append(p)

        # Now look stuff up in it.
        other_tables = []
        fields, keys = [], []
        for i, m in enumerate(props):
            if '.' in m:
                t, f = m.split('.', 1)
            else:
                t, f = 'base', m
            if t not in other_tables:
                other_tables.append(t)
            key = f'{t}.{f}'
            keys.append(key)
            fields.append(f'{key} as result{i}')
        q = ('select ' + ', '.join(fields) +
             ' from _dets join dets on _dets.name=dets.name ' +
             ' '.join(['join %s on %s.det_id=dets.id' % (m, m)
                       for m in other_tables]))
        c.execute(q)
        results = ResultSet.from_cursor(c, keys=keys)
        c.execute('drop table if exists _dets')
        results.strip(['base.'])
        return results

    def intersect(self, *specs, resolve=False):
        """Intersect the provided detector specs.  Each entry is either a list
        (or similar iterable) of detector names, or a dictionary
        specifying detector properties.

        If resolve=True, then the returned item is a list (rather
        than, possibly, a dict).

        """
        if len(specs) == 0:
            return []
        dicts = [s for s in specs if isinstance(s, dict)]
        others = [s for s in specs if not isinstance(s, dict)]
        # Reduce the dicts.
        req = {}
        for d in dicts:
            for k, v in d.items():
                if k in req:
                    if req[k] != v:
                        return []
                else:
                    req[k] = v
        if len(others) == 0:
            if resolve:
                return self.dets(props=req)['name']
            return req

        # Turn it into a list.
        req = self.dets(props=req)['name']
        keepers = set(req)
        for other in others:
            keepers.intersection_update(other)
        return [n for n in req if n in keepers]


def get_example():
    """Returns an example DetDb, mapped to RAM.  The two property tables
    are called "base" and "geometry".  This example is for
    demonstrating the code and interface and has no relation to any
    instrument's actual detector layout!

    """
    db = DetDb()

    TABLES = [
        ('base', [
            "`instrument` varchar(32)",
            "`camera` varchar(32)",
            "`array_code` varchar(16)",
            "`array_class` varchar(16)",
            "`wafer_code` varchar(32)",
            "`freq_code` varchar(16)",
            "`det_type` varchar(32)",
        ]),
        ('geometry', [
            "`wafer_x` float",
            "`wafer_y` float",
            "`wafer_pol` float",
        ]),
    ]

    for n, d in TABLES:
        print('Creating table %s' % n)
        db.create_table(n, d)

    tel_info = {'instrument': 'simonsobs',
                'camera': 'latr'}

    det_names = []
    for ar_type, bands, n_ar, n_wa, n_det in [
            ('LF', [27, 39], 1, 3, 37),
            ('MF', [93, 145], 4, 3, 432),
            ('HF', [225, 278], 2, 3, 542)
    ]:
        print('Creating %s-type arrays...' % ar_type)
        for fi, f in enumerate(bands):
            for ar in range(n_ar):
                for wa in range(n_wa):
                    iofs = (fi*n_ar*n_wa + n_wa*ar + wa)*n_det
                    info = {'freq_code': 'f%03i' % f,
                            'array_class': ar_type,
                            'array_code': '%s%i' % (ar_type, ar+1),
                            'wafer_code': 'W%i' % (wa+1)}
                    for i in range(iofs, iofs + n_det):
                        det_names.append('%s%i_%05i' % (ar_type, ar+1, i))
                        db.add_props('base', det_names[-1],
                                     det_type='bolo',
                                     **tel_info, **info, commit=False)

    print('Committing {} detectors...'.format(len(det_names)))
    db.conn.commit()

    # Organize these dets in a big square.  This is not the plan.
    n_row = int(len(det_names)**.5 + 1)
    for i in range(n_row):
        for j in range(n_row):
            d = i*n_row+j
            if d >= len(det_names):
                break
            x, y, ang = i * .02, j * .02, (i+j) % 12. * 15
            db.add_props('geometry', det_names[d], commit=False,
                         wafer_x=x, wafer_y=y, wafer_pol=ang)

    db.conn.commit()

    print('Checking the work...')
    db.validate()

    return db

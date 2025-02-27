"""
The purpose of ManifestDb is to provide a layer of abstraction that
assists code with identifying a file that contains some desired
information, based on a piece of index information.  For example, a
user might have a timestamp and an array name and want to know the
filename for the effective beam that should be used for that array at
that time.

**Definitions:**

Index Data
  A set of (key,value) pairs provided by the user.

Endpoint Data
  A set of (key,value) pairs returned by the Manifest.

The ManifestDb is primarily responsible for transforming Index Data
into Endpoint Data.  The implementation has the following features:

* The rules and data for mapping Index Data to Endpoint Data are
  stored in an sqlite database, for portability.

* It is assumed that a 'filename' will often be a piece of the
  Endpoint Data, so this is given a special status and table storage.

If a ManifestDb is used as part of the interface to a large metadata
archive, then minor updates to the metadata can be made by only
providing a the changed data (in a separate archive file), with an
updated ManifestDb that sends only the affected values of Index Data
to the new archive file.  When this approach is used, is suggests that
metadata archive files and the ManifestDb should be stored in slightly
different places, since there is a many-to-many relationship between
them.

When generating a ManifestDb, a ManifestScheme must first be defined.
The ManifestScheme includes:

* a description of what values may be present in the Index Data, and
  they should be used.
* a description of Endpoint Data to associate with valid Index Data.
"""

import sqlite3
import os
import sys
import json
import numpy as np
import argparse

from . import common, resultset

TABLE_DEFS = {
    'input_scheme': [
        "`id`      integer primary key autoincrement",
        "`name`    varchar(32) unique",
        "`purpose` varchar(32)",
        "`match`   varchar(32)",
        "`dtype`   varchar(32)",
        ],
    'map_template': [
        "`id` integer primary key autoincrement",
        "`file_id` integer",
    ],
    'files': [
        "`id` integer primary key autoincrement",
        "`name` varchar(256)",
    ],
}

# Because sqlite will let you store anything anywhere, the most
# sensible default column type is "numeric" -- this will automatically
# cast number-like strings to numbers (while accepting strings).
DEFAULT_DTYPE = 'numeric'


class ManifestScheme:
    def __init__(self):
        self.cols = []

    def new_db(self, **kwargs):
        """Use this scheme to instantiate a ManifestDb, and return it.  Aall
        kwargs are passed to ManifestDb constructor.

        """
        return ManifestDb(scheme=self, **kwargs)

    # Methods for constructing table.

    def add_exact_match(self, name, dtype=DEFAULT_DTYPE):
        """
        Add a field to the scheme, that must be matched exactly.
        """
        self.cols.append((name, 'in', 'exact', dtype))
        return self

    def add_range_match(self, name, purpose='in', dtype=DEFAULT_DTYPE):
        """
        Add a field to the scheme, that represents a range of values to be
        matched by a single input value.
        """
        self.cols.append((name, purpose, 'range', dtype))
        return self

    def add_data_field(self, name, dtype=DEFAULT_DTYPE):
        """
        Add a field to the scheme that is returned along with the matched
        filename.
        """
        self.cols.append((name, 'out', 'exact', dtype))
        return self

    def as_resultset(self):
        """Get the scheme structure as a ResultSet.  This is a safer
        alternative to inspecting .cols directly.

        """
        rs = resultset.ResultSet(['field', 'purpose', 'col_type', 'data_type'])
        rs.rows = list(self.cols)
        return rs

    def _get_scheme_rows(self):
        """
        Returns a list of tuples (name,purpose,match,dtype), suitable
        for populating the input_scheme table.
        """
        rows = []
        for c in self.cols:
            name, purpose, match, dtype = c
            if match == 'exact':
                rows.append((name, purpose, 'exact', dtype))
            elif match == 'range':
                rows.append((name, purpose, 'range', dtype))
            else:
                raise ValueError("Bad ctype '%s'" % match)
        return rows

    def _get_map_table_def(self):
        """
        Returns column definitions for the map table.
        """
        entries = [row for row in TABLE_DEFS['map_template']]
        uniques = []
        for c in self.cols:
            name, purpose, match, dtype = c
            if match == 'exact':
                entries.append('`%s` %s' % (name, dtype))
                uniques.append('`%s`' % name)
            elif match == 'range':
                entries.append('`%s__lo` %s' % (name, dtype))
                entries.append('`%s__hi` %s' % (name, dtype))
                uniques.append('`%s__lo`' % name)
                uniques.append('`%s__hi`' % name)
            else:
                raise ValueError("Bad ctype '%s'" % match)
        entries.append('UNIQUE(' + ','.join(uniques) + ')')
        return entries

    def _format_row(self, r):
        """Rewrite a dict of index and/or endpoint data from the database so
        that it is compatible with the format expected by
        ManifestDb.add_entry / update_entry.

        This modifies the provided object (and also returns it).

        """
        for c in self.cols:
            name, purpose, match, dtype = c
            if match == 'range':
                a, b = r.pop('%s__lo' % name), r.pop('%s__hi' % name)
                r[name] = (a, b)
        return r

    @classmethod
    def from_database(cls, conn, table_name='input_scheme'):
        """
        Decode a ManifestScheme from the provided sqlite database
        connection.
        """
        c = conn.cursor()
        c.execute('select name,purpose,match,dtype from %s' % table_name)
        self = cls()
        for r in c:
            (name, purpose, match, dtype) = r
            if match in ['exact', 'range']:
                self.cols.append((name, purpose, match, dtype))
            else:
                raise ValueError("Bad ctype '%s'" % match)
        return self

    def get_match_query(self, params, partial=False, strict=False):
        """Get sql query fragments for this ManifestDb.

        Arguments:
          params: a dict of values to match against.
          partial: if True, then operate in "inspection" mode (see notes).
          strict: if True, then reject any requests that include
            entries in params that are not known to the schema.

        Returns:
          (where_string, values_tuple, ret_cols)

        Notes:
          The normal mode (partial=False) requires that every "in"
          column in the scheme has a key=value pair in the params
          dict, and the ret_cols are the "out" columns.  In inspection
          mode (partial=True), then any column can be matched against
          the params, and the complete row data of all matching rows
          is returned.

        """
        qs = []
        vals = []
        ret_cols = []
        unassigned = dict(params)
        assert('filename' not in params)

        for col in self.cols:
            (name, purpose, match, dtype) = col
            purposes = [purpose]

            if partial:
                # in/out direction is entirely determined by whether
                # user passed a value.
                if name in params:
                    purposes = ['in', 'out']
                else:
                    purposes = ['out']
            else:
                if 'in' in purposes and name not in params:
                    raise ValueError('Parameter %s is not optional.' % name)

            if 'in' in purposes:
                if match == 'exact':
                    qs.append('`%s`=?' % name)
                    vals.append(params[name])
                elif match == 'range':
                    qs.append('(`%s__lo` <= ?) and (? < `%s__hi`)' % (name, name))
                    vals.extend([params[name], params[name]])
                else:
                    raise ValueError("Bad ctype '%s'" % match)
                unassigned.pop(name)
            if 'out' in purposes:
                # Include that column's value in result.
                if match == 'range':
                    ret_cols.extend(['%s__lo' % name, '%s__hi' % name])
                else:
                    ret_cols.append(name)
        if strict and len(unassigned):
            raise ValueError(f'Failed to match params: {unassigned}')
            assert(len(unassigned) == 0)
        return (' and '.join(qs), tuple(vals), ret_cols)

    def get_insertion_query(self, params):
        """Get sql query fragments for inserting a new entry with the provided
        params.

        Returns:
          (fields, values) where fields is a string with the field
          names (comma-delimited) and values is a tuple of values.

        """
        qs = []
        vals = []
        unassigned = list(params.keys())
        for col in self.cols:
            (name, purpose, match, dtype) = col
            if not name in params:
                raise ValueError('Parameter %s is not optional.' % name)
            unassigned.remove(name)
            if match == 'exact':
                qs.append('`%s`' % name)
                vals.append(params[name])
            elif match == 'range':
                qs.append('`%s__lo`,`%s__hi`' % (name,name))
                vals.extend(params[name])
            else:
                raise ValueError("Bad ctype '%s'" % match)
        if len(unassigned):
            raise ValueError('Attempting to add data for unknown fields: %s' % unassigned)
        return ','.join(qs), tuple(vals)

    def get_update_query(self, params):
        """Get sql query fragments for updating an entry.

        Returns:
          (setstring, values) where setstring is of the form
          "A=?,...,Z=?" and values is the corresponding tuple of values.

        """
        keys = []
        vals = []
        unassigned = [k for k in params.keys() if k not in '_id']
        for col in self.cols:
            (name, purpose, match, dtype) = col
            if not name in params:
                continue
            unassigned.remove(name)
            if match == 'exact':
                keys.append('`%s`' % name)
                vals.append(params[name])
            elif match == 'range':
                keys.extend(['`%s__lo`' % name, '`%s__hi`' % name])
                vals.extend(params[name])
            else:
                raise ValueError("Bad ctype '%s'" % match)
        if len(unassigned):
            raise ValueError('Attempting to update data for unknown fields: %s' % unassigned)
        return ','.join([f'{k}=?' for k in keys]), tuple(vals)

    def get_required_params(self):
        """
        Returns a list of parameter names that are required for matching.
        """
        return [c[0] for c in self.cols if c[1] == 'in']


class ManifestDb:
    """
    Expose a map from Index Data to Endpoint Data, including a
    filename.
    """

    def __init__(self, map_file=None, scheme=None):
        """Instantiate a database.  If map_file is provided, the
        database will be connected to the indicated sqlite file on
        disk, and any changes made to this object be written back to
        the file.

        If scheme is None, the scheme will be loaded from the
        database; pass scheme=False to prevent that and leave the db
        uninitialized.

        """
        if isinstance(map_file, sqlite3.Connection):
            self.conn = map_file
        else:
            if map_file is None:
                map_file = ':memory:'
            self.conn = sqlite3.connect(map_file)

        self.conn.row_factory = sqlite3.Row  # access columns by name

        if scheme is None:
            self.scheme = ManifestScheme.from_database(self.conn)
        elif scheme is False:
            pass
        else:
            self._create(scheme)

    def _create(self, manifest_scheme):
        """
        Create the database tables, incorporating the provided
        ManifestScheme.
        """
        # Create the tables:
        table_defs = [
            ('input_scheme', TABLE_DEFS['input_scheme']),
            ('files', TABLE_DEFS['files']),
            ('map', manifest_scheme._get_map_table_def())]
        c = self.conn.cursor()
        for table_name, column_defs in table_defs:
            q = ('create table if not exists `%s` (' % table_name  +
                 ','.join(column_defs) + ')')
            c.execute(q)
        self.conn.commit()
        # Commit the schema...
        for r in manifest_scheme._get_scheme_rows():
            c.execute('insert into input_scheme (name,purpose,match,dtype) '
                      'values (?,?,?,?)', tuple(r))
        self.conn.commit()

        self.scheme = ManifestScheme.from_database(self.conn)

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
        new_db = ManifestDb(map_file=map_file, scheme=False)
        script = ' '.join(self.conn.iterdump())
        new_db.conn.executescript(script)
        new_db.scheme = ManifestScheme.from_database(new_db.conn)
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
        return common.sqlite_to_file(self.conn, filename, overwrite=overwrite, fmt=fmt)

    @classmethod
    def from_file(cls, filename, fmt=None, force_new_db=True):
        """Instantiate an ManifestDb and return it, with the data copied in
        from the specified file.

        Args:
          filename (str): path to the file.
          fmt (str): format of the input; see to_file for details.
          force_new_db (bool): Used if connecting to an sqlite
            database. If True the database is copied into memory and
            if False returns a read-only connection to the database
            without reading it into memory

        Returns:
          ManifestDb with an sqlite3 connection that is mapped to memory.

        Notes:
          Note that if you want a `persistent` connection to the file,
          you should instead pass the filename to the ManifestDb
          constructor map_file argument.

        """
        conn = common.sqlite_from_file(filename, fmt=fmt, force_new_db=force_new_db)
        return cls(conn)

    @classmethod
    def readonly(cls, filename):
        """Instantiate an ManifestDb connected to an sqlite database on disk,
        and return it.  The database remains mapped to disk, in readonly mode.

        Args:
          filename (str): path to the file.

        Returns:
          ManifestDb.

        """
        conn = sqlite3.connect('file:%s?mode=ro' % filename, uri=True)
        return cls(conn)

    def _get_file_id(self, filename, create=False):
        """
        Lookup a file_id in the file table, or create it if `create` and not found.
        """
        c = self.conn.cursor()
        c.execute('select id from files where name=?', (filename,))
        row_id = c.fetchone()
        if row_id is None:
            if create:
                c.execute('insert into files (name) values (?)', (filename,))
                row_id = c.lastrowid
                return row_id
            return None
        return row_id[0]

    def match(self, params, multi=False, prefix=None):
        """Given Index Data, return Endpoint Data.

        Arguments:
          params (dict): Index Data.
          multi (bool): Whether more than one result may be returned.
          prefix (str or None): If set, it will be os.path.join-ed to
            the filename from the db.

        Returns:
          A dict of Endpoint Data, or None if no match was found.  If
          multi=True then a list is returned, which could have 0, 1,
          or more items.

        """
        q, p, rp = self.scheme.get_match_query(params)
        cols = ['files`.`name'] + list(rp)
        c = self.conn.cursor()
        where_str = ''
        if len(q):
            where_str = 'where %s' % q
        c.execute('select `%s` ' % ('`,`'.join(cols)) + 
                  'from map join files on map.file_id=files.id %s' % where_str, p)
        rows = c.fetchall()
        rp.insert(0, 'filename')
        rows = [dict(zip(rp, r)) for r in rows]
        if prefix is not None:
            for r in rows:
                r['filename'] = os.path.join(prefix, r['filename'])
        if multi:
            return rows
        if len(rows) == 0:
            return None
        if len(rows) > 1:
            raise ValueError('Matched multiple rows with index data: %s' % rows)
        return rows[0]

    def inspect(self, params={}, strict=True, prefix=None):
        """Given (partial) Index Data and Endpoint Data, find and return the
        complete matching records.

        Arguments:
          params (dict): any mix of Index Data and Endpoint Data.
          strict (bool): if True, a ValueError will be raised if
            params contains any keys that aren't recognized as Index
            or Endpoint data.
          prefix (str or None): As in .match().

        Returns:
          A list of results matching the query.  Each result in the
          list is a dict, containing complete entry data.  A special
          entry, '_id', is the database row id and can be used to
          update or remove specific entries.

        """
        params = dict(params)
        filename = params.pop('filename', None)

        q, p, rp = self.scheme.get_match_query(params, partial=True, strict=strict)
        cols = ['map`.`id', 'files`.`name'] + list(rp)
        rp = ['_id', 'filename'] + rp
        c = self.conn.cursor()
        where_str = ''

        if len(q):
            where_str = 'where %s' % q

        c.execute('select `%s` ' % ('`,`'.join(cols)) +
                  'from map join files on map.file_id=files.id %s' % where_str, p)
        rows = c.fetchall()
        rows = [self.scheme._format_row(dict(zip(rp, r))) for r in rows]
        if prefix is not None:
            for r in rows:
                r['filename'] = os.path.join(prefix, r['filename'])
        if filename:
            # manual filter...
            rows = [r for r in rows if r['filename'] == filename]
        return rows

    def add_entry(self, params, filename=None, create=True, commit=True,
                  replace=False):
        """Add an entry to the map table.

        Arguments:
          params: a dict of values for the Index Data columns.  In the
            case of 'range' columns, a pair of values must be
            provided.  Endpoint Data, other than the filename, should
            also be included here.
          filename: the filename to associate with matching Index Data.
          create: if False, do not create new entry in the file table
            (and fail if entry for filename does not already exist).
          commit: if False, do not commit changes to the database (for
            batch use).
          replace: if True, do not raise an error if the index data
            matches a row of the table already; instead just update
            the record.

        Notes:
          The uniqueness check in the database will only prevent (or
          replace) *identical* index entries.  Other inconsistent
          states, such as overlapping time ranges that would both
          match some single timestamp, are not caught here.

        """
        # Validate the input data.
        q, p = self.scheme.get_insertion_query(params)
        file_id = self._get_file_id(filename, create=create)
        assert(file_id is not None)
        c = self.conn.cursor()
        marks = ','.join('?' * len(p))
        c = self.conn.cursor()
        query = 'into map (%s,file_id) values (%s,?)' % (q, marks)
        if replace:
            query = 'insert or replace ' + query
        else:
            query = 'insert ' + query
        c.execute(query, p + (file_id,))
        if commit:
            self.conn.commit()

    def update_entry(self, params, filename=None, commit=True):
        """Update an existing entry.

        Arguments:
          params: Index data to change.  This must include key '_id',
            with the value corresponding to an existing row in the
            table.

        Notes:

          Only columns expressly included in params will be updated.
          The params can include 'filename', in which case a new value
          is set.

        """
        # Validate the input data.
        q, p = self.scheme.get_update_query(params)
        _id = params.get('_id')
        # Are we changing the filename?  Hope not ...
        assert(filename is None and 'filename' not in params)
        if filename is None:
            filename = params.get('filename')

        c = self.conn.cursor()
        marks = ','.join('?' * len(p))
        c = self.conn.cursor()
        query = 'update map set %s where id=?' % q
        c.execute(query, p + (_id,))# + (file_id,))
        if commit:
            self.conn.commit()

    def remove_entry(self, _id, commit=True):
        """Remove the entry identified by row id _id.

        If _id is a dict, _id['_id'] is used.  Entries returned by
        .inspect() should have _id populated in this way, and thus can
        be passed directly into this function.

        """
        if isinstance(_id, dict):
            _id = _id['_id']
        c = self.conn.cursor()
        file_id = c.execute('select file_id from map where id=?',
                             (_id, )).fetchall()
        if len(file_id) == 0:
            raise ValueError(f'Row with id={_id} does not exist!')
        file_id = file_id[0][0]
        c.execute('delete from map where id=?', (_id, ))
        if len(c.execute('select id from map where file_id=?', (file_id, )).fetchall()) == 0:
            c.execute('delete from files where id=?', (file_id, ))
        if commit:
            self.conn.commit()

    def get_entries(self, fields):

        """Return list of all entry names in database
        that are in the listed fields

        Arguments
        ---------
        fields: list of strings
            should correspond to columns in map table made through 
            ManifestScheme.add_data_field( field_name )

        Returns
        --------
        ResultSet with keys equal to field names
        """
        if not isinstance(fields, list):
            raise ValueError("fields must be a list")
        q = f"select distinct {','.join(fields)} from map"
        c = self.conn.execute(q)
        return resultset.ResultSet.from_cursor(c)
        
    def validate(self):
        """
        Checks that the database is following internal rules.
        Specifically...

        Raises SchemaError in the first case, IntervalError in the second.
        """
        return False


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            epilog="""For details of individual modes, pass a dummy database argument
            followed by the mode and -h, e.g.: "%(prog)s x files -h" """)

    parser.add_argument('filename', help="Path to a ManifestDb.",
                        metavar='my_db.sqlite')

    cmdsubp = parser.add_subparsers(
        dest='mode')

    # "summary"
    p = cmdsubp.add_parser(
        'summary', help=
        "Summarize database structure and number of entries.",
        usage="""

    %(prog)s

        This will print a summary of the index fields and
        endpoint fields.  (This mode is chosen by default.)
        """)

    # "entries"
    p = cmdsubp.add_parser(
        'entries', help=
        "Show all entries in the database.",
        usage="""Syntax:

    %(prog)s

        This will print every row of the metadata map table,
        including the filename, and with two header rows.
        """)


    # "files"
    p = cmdsubp.add_parser(
        'files', usage="""Syntax:

    %(prog)s
    %(prog)s --all
    %(prog)s --clean

        This will print out a list of archive files referenced
        by the database.  --all prints all the rows, even if
        there are a lot of them.  --clean is used to get a
        simple list, one file per line, for use with rsync
        or whatever.
        """,
        help="List the files referenced in the database.")

    p.add_argument('--clean', action='store_true',
                   help="Print a simple list of all files (for script digestion).")
    p.add_argument('--all', action='store_true',
                   help="Print all files, not an abbreviated list.")

    # "lookup"
    p = cmdsubp.add_parser(
        'lookup', help=
        "Query database for specific index data and display matched endpoint data.",
        usage="""Syntax:

    %(prog)s val1,val2,... [val1,val2,... ]

        Each command line argument is converted to a single query, and
        must consist of comma-delimited fields to be associated
        one-to-one with the index fields.

    Example 1: if the single index field is "obs:obs_id":

        %(prog)s obs_1840300000

    Example 2: do two queries (single index field)

        %(prog)s obs_1840300000 obs_185040012

    Example 3: single query, two index fields (perhaps timestamp and wafer):

        %(prog)s 1840300000,wafer29

        """)
    p.add_argument('index', nargs='*', help=
                   "Index information.  Comma-delimit your data, for example: "
                   "1456453245,wafer5")

    # "reroot"
    p = cmdsubp.add_parser(
        'reroot', help=
        "Batch change filenames (by prefix) in the database.",
        usage="""Syntax:

    %(prog)s old_prefix new_prefix [output options]

Examples:

    %(prog)s /path/on/system1 /path/on/system2 -o my_new_manifest.sqlite
    %(prog)s /path/on/system1 /new_path/on/system1 --overwrite
    %(prog)s ./result1/obs_12345.h5 ./result2/obs_12345.h5 --overwrite

        These operations will create a new ManifestDb, will all the
        entries from my_db.sqlite, but with the filenames
        (potentially) altered.  Any filename that starts with the
        first argument will be changed, in the output, to instead
        start with the second argument.  When you do this you must
        either say where to write the output (-o) or give the program
        permission to overwrite your input database file.  Note that
        the first argument need not match all entries in the database;
        you can use it to pick out a subset (even a single entry).
    """)
    p.add_argument('old_prefix', help=
                   "Prefix to match in current database.")
    p.add_argument('new_prefix', help=
                   "Prefix to replace it with.")
    p.add_argument('--overwrite', action='store_true', help=
                   "Store modified database in the same file.")
    p.add_argument('--output-db', '-o', help=
                   "Store modified database in this file.")
    p.add_argument('--dry-run', action='store_true', help=
                   "Run the conversion steps but do not write the results anywhere.")

    return parser

def main(args=None):
    """Entry point for the so-metadata tool."""
    if args is None:
        args = sys.argv[1:]
    if not isinstance(args, argparse.Namespace):
        parser = get_parser()
        args = parser.parse_args(args)

    if args.mode is None:
        args.mode = 'summary'

    db = ManifestDb.from_file(args.filename, force_new_db=False)

    if args.mode == 'summary':
        header = f'Summary for {args.filename}'
        print(header)
        print('-' * len(header))
        print()

        schema = db.scheme.as_resultset()
        row_count = db.conn.execute('select count(id) from map').fetchone()[0]
        print(f'Total number of index entries:  {row_count:>7}')
        print()

        print('Index fields:')
        fmt = '   {field:20}  {data_type:20} {col_type:>10}'
        hdr = fmt.format(field="Field", data_type="Type", col_type="Match")
        print(hdr)
        print('   ' + '-' * (len(hdr) - 3))
        for row in schema:
            if row['purpose'] == 'in':
                print(fmt.format(**row))
        print()

        print('Endpoint fields:')
        fmt = '   {field:20}  {data_type:20} {count:>10}'
        hdr = fmt.format(field="Field", data_type="Type", count="Entries")
        print(hdr)
        print('   ' + '-' * (len(hdr) - 3))
        for row in schema:
            if row['purpose'] == 'out':
                count = len(db.conn.execute('select distinct `%s` from map' % row['field']).fetchall())
                print(fmt.format(count=count, **row))
        file_count = db.conn.execute('select count(id) from files').fetchone()[0]
        print(fmt.format(
            field='filename', data_type='filename', count=file_count))

        print()

    elif args.mode == 'entries':
        # Print the table of entries.
        schema = db.scheme.as_resultset()
        keys = []
        for purp in ['in', 'out']:
            for s in schema:
                if s['purpose'] != purp:
                    continue
                if s['col_type'] == 'range':
                    keys.extend([s['field'] + '__lo',
                                 s['field'] + '__hi'])
                else:
                    keys.append(s['field'])
        keys.append('filename')
        print(keys)
        print(['-'] * len(keys))
        for row in db.conn.execute('select map.*, files.name as filename from '
                                   'map join files where map.file_id=files.id'):
            print([row[k] for k in keys])

    elif args.mode == 'files':
        # Get all files.
        rows = db.conn.execute(
            'select files.id, files.name as filename, count(map.id) '
            'from map join files on '
            'map.file_id==files.id group by filename').fetchall()

        if args.clean:
            for _id, filename, count in rows:
                print(filename)
        else:
            fmt = '  {_id:>7} {count:>7} {filename}'
            hdr = fmt.format(_id="file_id", count="Count", filename="Filename")
            print(hdr)
            print('-' * (len(hdr) + 20))
            n = len(rows)
            super_count = sum([r[2] for r in rows])
            if n > 20 and not args.all:
                rows = rows[:10]
            for _id, filename, count in rows:
                print(fmt.format(_id=_id, filename=filename, count=count))
            if len(rows) < n:
                other_count = super_count - sum([r[1] for r in rows])
                print(fmt.format(count=other_count, filename='...'))
                print('(Pass --all to show all results.)')
            print()

    elif args.mode == 'lookup':
        schema = db.scheme.as_resultset()

        index_fields = [r['field'] for r in schema if r['purpose'] == 'in']
        endpoint_fields = [r['field'] for r in schema if r['purpose'] == 'out']

        results = []
        for index in args.index:
            vals = index.split(',')
            if len(vals) != len(index_fields):
                print(f'Index data "{index}" decodes to "{len(vals)}" fields, ')
                print(f'but we expected "{len(index_fields)}".')

            query = {r: a for r, a in zip(index_fields, vals)}
            matches = db.match(query, multi=True)

            results.append({'query': query,
                            'matches': []})
            endpoint_fields.append('filename')
            for i, m in enumerate(matches):
                results[-1]['matches'].append({})
                for k in endpoint_fields:
                    results[-1]['matches'][-1][k] = matches[i][k]

        print(json.dumps(results, indent=4))

    elif args.mode == 'reroot':
        # Reconnect with write?
        if args.overwrite:
            if args.output_db:
                parser.error("Specify only one of --overwrite or --output-db.")
            db = ManifestDb.from_file(args.filename, force_new_db=True)
            args.output_db = args.filename
        else:
            if args.output_db is None:
                parser.error("Specify an output database name with --output-db, "
                             "or pass --overwrite to clobber.")
            db = ManifestDb.from_file(args.filename, force_new_db=True)

        # Get all files matching this prefix ...
        c = db.conn.execute("select id, name from files "
                            "where name like '%s%%'" % (args.old_prefix))
        rows = c.fetchall()
        print('Found %i records matching prefix ...'
               % len(rows))

        print('Converting to new prefix ...')
        n_examples = 1

        if not args.dry_run:
            c = db.conn.cursor()

        for (_id, name) in rows:
            new_name = args.new_prefix + name[len(args.old_prefix):]
            if n_examples > 0:
                print(f'  Example: converting filename\n'
                      f'      "{name}"\n'
                      f'    to\n'
                      f'      "{new_name}"')
                n_examples -= 1
            if not args.dry_run:
                c.execute('update files set name=? where id=?', (new_name, _id))

        print('Saving to %s' % args.output_db)
        if not args.dry_run:
            db.conn.commit()
            c.execute('vacuum')
            db.to_file(args.output_db)

    else:
        print(f'Sorry, {args.mode} not implemented.')

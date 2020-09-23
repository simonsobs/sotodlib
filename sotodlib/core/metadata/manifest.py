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
import numpy as np

from . import common

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


class ManifestScheme:
    def __init__(self):
        self.cols = []

    # Methods for constructing table.

    def add_exact_match(self, name, dtype='varchar(16)'):
        """
        Add a field to the scheme, that must be matched exactly.
        """
        self.cols.append((name, 'in', 'exact', dtype))
        return self

    def add_range_match(self, name, purpose='in', dtype='varchar(16)'):
        """
        Add a field to the scheme, that represents a range of values to be
        matched by a single input value.
        """
        self.cols.append((name, purpose, 'range', dtype))
        return self

    def add_data_field(self, name, dtype='varchar(16)'):
        """
        Add a field to the scheme that is returned along with the matched
        filename.
        """
        self.cols.append((name, 'out', 'exact', dtype))
        return self

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
        for c in self.cols:
            name, purpose, match, dtype = c
            if match == 'exact':
                entries.append('`%s` %s' % (name, dtype))
            elif match == 'range':
                entries.append('`%s__lo` %s' % (name, dtype))
                entries.append('`%s__hi` %s' % (name, dtype))
            else:
                raise ValueError("Bad ctype '%s'" % match)
        return entries

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

    def get_match_query(self, params):
        """
        Arguments:
          params: a dict of values to match against.

        Returns:
          (where_string, values_tuple, ret_cols)
        """
        qs = []
        vals = []
        ret_cols = []
        for col in self.cols:
            (name, purpose, match, dtype) = col
            if purpose == 'in':
                if not name in params:
                    raise ValueError('Parameter %s is not optional.' % name)
                if match == 'exact':
                    qs.append('`%s`=?' % name)
                    vals.append(params[name])
                elif match == 'range':
                    qs.append('(`%s__lo` <= ?) and (? < `%s__hi`)' % (name, name))
                    vals.extend([params[name], params[name]])
                else:
                    raise ValueError("Bad ctype '%s'" % match)
            if purpose == 'out':
                # Include that column's value in result.
                ret_cols.append(name)
        return (' and '.join(qs), tuple(vals), ret_cols)

    def get_insertion_query(self, params):
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
    def from_file(cls, filename, fmt=None):
        """Instantiate an ManifestDb and return it, with the data copied in
        from the specified file.

        Args:
          filename (str): path to the file.
          fmt (str): format of the input; see to_file for details.

        Returns:
          ManifestDb with an sqlite3 connection that is mapped to memory.

        Notes:
          Note that if you want a `persistent` connection to the file,
          you should instead pass the filename to the ManifestDb
          constructor map_file argument.

        """
        conn = common.sqlite_from_file(filename, fmt=fmt)
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

    def match(self, params, multi=False):
        """Given Index Data, return Endpoint Data.

        Arguments:

          params: a dict of Index Data.

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
        if multi:
            return rows
        if len(rows) == 0:
            return None
        if len(rows) > 1:
            raise ValueError('Matched multiple rows with index data: %s' % rows)
        return rows[0]

    def add_entry(self, params, filename=None, create=True, commit=True):
        """
        Add an entry to the map table.

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
        """
        # Validate the input data.
        q, p = self.scheme.get_insertion_query(params)
        file_id = self._get_file_id(filename, create=create)
        assert(file_id is not None)
        c = self.conn.cursor()
        marks = ','.join('?' * len(p))
        c = self.conn.cursor()
        c.execute('insert into map (%s,file_id) values (%s,?)' % (q, marks),
                  p + (file_id,))
        if commit:
            self.conn.commit()

    def validate(self):
        """
        Checks that the database is following internal rules.
        Specifically...

        Raises SchemaError in the first case, IntervalError in the second.
        """
        return False

import sqlite3
import gzip
import os

from .resultset import ResultSet
from . import common #import sqlite_to_file, sqlite_from_file

# Design of observation database... certainly the main table contains
# basic incontrovertible facts.  Like an obs_id and a timestamp.  But
# there should be some nice way of supporting tags too -- such as
# 'uranus', 'planet', 'daytime'.
#
# So much boilerplate...
#

TABLE_DEFS = {
    'obs': [
        "`obs_id` varchar(256) primary key",
        "`timestamp` float",
    ],
    'tags': [
        "`obs_id` varchar(256) primary key",
        "`tag` varchar(256)",
    ],
}


class ObsDb(object):
    """Observation database.

    The ObsDb helps to associate observations, indexed by an obs_id,
    with properties of the observation that might be useful for
    selecting data or for identifying metadata.

    The main ObsDb table is called 'obs', and contains at least the
    columns obs_id (string) and timestamp (float representing a unix
    timestamp).  Additional columns may be added to this table as
    needed.

    The other table is called 'tags', and facilitates the association
    of obs_id with tag names.

    """

    TABLE_TEMPLATE = [
        "`obs_id` varchar(256)",
    ]

    def __init__(self, map_file=None, init_db=True):
        """Instantiate an ObsDb.

        Args:
          map_file (str or sqlite3.Connection): If this is a string,
            it will be treated as the filename for the sqlite3
            database, and opened as an sqlite3.Connection.  If this is
            an sqlite3.Connection, it is cached and used.  If this
            argument is None (the default), then the
            sqlite3.Connection is opened on ':memory:'.
          init_db (bool): If True, then any ObsDb tables that do not
            already exist in the database will be created.

        Notes:
          If map_file is provided, the database will be connected to
          the indicated sqlite file on disk, and any changes made to
          this object be written back to the file.

        """
        if isinstance(map_file, sqlite3.Connection):
            self.conn = map_file
        else:
            if map_file is None:
                map_file = ':memory:'
            self.conn = sqlite3.connect(map_file)

        self.conn.row_factory = sqlite3.Row  # access columns by name
        if init_db:
            c = self.conn.cursor()
            c.execute("SELECT name FROM sqlite_master "
                      "WHERE type='table' and name not like 'sqlite_%';")
            tables = [r[0] for r in c]
            changes = False
            for k, v in TABLE_DEFS.items():
                if k not in tables:
                    self._create_table(k, v, raw=True)
                    q = ('create table if not exists `%s` (' % k +
                         ','.join(v) + ')')
                    c.execute(q)
                    changes = True
            if changes:
                self.conn.commit()

    def __len__(self):
        return self.conn.execute('select count(obs_id) from obs').fetchone()[0]

    def _create_table(self, table_name, column_defs, raw=False, commit=True):
        """Add a table to the database.

        Args:
          table_name (str): The name of the new table.
          column_defs (list): A list of sqlite column definition
            strings.
          raw (bool): See below.
          commit (bool): Whether to commit the changes to the db.

        An example of column_defs is::

          column_defs=[
            "`az` float",
            "`el` float",
          ]

        """
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


    def add_obs_columns(self, column_defs, commit=True, ignore_duplicates=True):
        """
        column_defs can be any of the following:
        """
        current_cols = self.conn.execute('pragma table_info("obs")').fetchall()
        current_cols = [r[1] for r in current_cols]
        if isinstance(column_defs, str):
            column_defs = column_defs.split(',')
        for column_def in column_defs:
            if isinstance(column_def, str):
                column_def = column_def.split()
            name, typestr = column_def
            if typestr is float:
                typestr = 'float'
            elif typestr is int:
                typestr = 'int'
            elif typestr is str:
                typestr = 'text'
            check_name = name
            if name.startswith('`'):
                check_name = name[1:-1]
            else:
                name = '`' + name + '`'
            if check_name in current_cols:
                if ignore_duplicates:
                    continue
                raise ValueError("Column %s already exists in table obs" % check_name)
            self.conn.execute('ALTER TABLE obs ADD COLUMN %s %s' % (name, typestr))
            current_cols.append(check_name)
        if commit:
            self.conn.commit()
        return self

    def update_obs(self, obs_id, data, commit=True):
        """
        Update an entry in the obs table.

        Arguments:
            obs_id (str): The id of the obs to update.
            data (dict): map from column_name to value.

        Returns:
            self.
        """
        c = self.conn.cursor()
        settors = [f'{k}=?' for k in data.keys()]
        c.execute('insert or ignore into obs (obs_id) values (?)',
                  (obs_id,))
        c.execute('update obs set ' + ','.join(settors) + ' '
                  'where obs_id=?',
                  tuple(data.values()) + (obs_id, ))
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
        new_db = ObsDb(map_file=map_file, init_db=False)
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
        return common.sqlite_to_file(self.conn, filename, overwrite=overwrite, fmt=fmt)

    @classmethod
    def from_file(cls, filename, fmt=None):
        """Instantiate an ObsDb and return it, with the data copied in from
        the specified file.

        Args:
          filename (str): path to the file.
          fmt (str): format of the input; see to_file for details.

        Note that if you want a `persistent` connection to the file,
        you should instead pass the filename to the ObsDb constructor
        map_file argument.

        """
        conn = common.sqlite_from_file(filename, fmt=fmt)
        return cls(conn, init_db=False)

    def get(self, obs_id=None, add_prefix=''):
        """Returns the entry for obs_id, as an ordered dict.  If obs_id is
        None, returns all entries, as a ResultSet.  Yup, those are the
        options.

        add_prefix is used to alter the names of fields; this is
        mostly for constructing metadata selectors ('obs:obs_id' and
        'obs:timestamp'), with add_prefix='obs:'.

        """
        if obs_id is None:
            return self.query('1', add_prefix=add_prefix)
        results = self.query(f'obs_id="{obs_id}"', add_prefix=add_prefix)
        if len(results) == 0:
            return None
        if len(results) > 1:
            raise ValueError('Too many rows...') # or integrity error...
        return results[0]

    def query(self, query_text='1', tags=None, keys=None, add_prefix='', sort=['obs_id']):
        """Queries the ObsDb using user-provided text.  Returns a ResultSet.

        """
        assert(keys is None)  # Not implemented, sry.
        assert(tags is None)  # Not implemented, sry.
        sort_text = ''
        if sort is not None and len(sort):
            sort_text = ' order by ' + ','.join(sort)

        c = self.conn.execute('select * from obs where %s %s' % (query_text, sort_text))
        results = ResultSet.from_cursor(c)
        if add_prefix is not None:
            results.keys = [add_prefix + k for k in results.keys]
        return results

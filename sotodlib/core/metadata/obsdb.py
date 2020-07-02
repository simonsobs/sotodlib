import sqlite3
import gzip
import os

from .resultset import ResultSet

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
}


class ObsDB(object):
    """Observation database.

    """

    #: Column definitions (a list of strings) that must appear in all
    #: Property Tables.
    TABLE_TEMPLATE = [
        "`obs_id` varchar(256)",
    ]

    def __init__(self, map_file=None, init_db=True):
        """Instantiate an ObsDB.  If map_file is provided, the database will
        be connected to the indicated sqlite file on disk, and any
        changes made to this object be written back to the file.

        """
        if map_file is None:
            map_file = ':memory:'
        self.conn = sqlite3.connect(map_file)
        self.conn.row_factory = sqlite3.Row  # access columns by name

        if init_db:
            # Create dets table if not found.
            c = self.conn.cursor()
            c.execute("SELECT name FROM sqlite_master "
                      "WHERE type='table' and name not like 'sqlite_%';")
            tables = [r[0] for r in c]
            if 'obs' not in tables:
                self.create_table('obs', TABLE_DEFS['obs'], raw=True)

    def __len__(self):
        return self.conn.execute('select count(obs_id) from obs').fetchone()[0]

    def create_table(self, table_name, column_defs, raw=False, commit=True):
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

    def copy(self, map_file=None, overwrite=False):
        """
        Duplicate the current database into a new database object, and
        return it.  If map_file is specified, the new database will be
        connected to that sqlite file on disk.  Note that a quick way
        of writing a DB to disk to call copy(map_file=...) and then
        simply discard the returned object.
        """
        if map_file is not None and os.path.exists(map_file):
            if overwrite:
                os.remove(map_file)
            else:
                raise RuntimeError("Output file %s exists (overwrite=True "
                                   "to overwrite)." % map_file)
        new_db = ObsDB(map_file=map_file, init_db=False)
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
            self.copy(map_file=filename, overwrite=overwrite)
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
    def from_file(cls, filename, fmt=None):
        """Instantiate an ObsDB and return it, with the data copied ain from the
        specified file.

        Args:
          filename (str): path to the file.
          fmt (str): format of the input; see to_file for details.

        Note that if you want a `persistent` connection to the file,
        you should instead pass the filename to the ObsDB constructor
        map_file argument.

        """
        if fmt is None:
            fmt = 'sqlite'
            if filename.endswith('.gz'):
                fmt = 'gz'
        if fmt == 'sqlite':
            db0 = cls(map_file=filename)
            return db0.copy(map_file=None)
        elif fmt == 'dump':
            with open(filename, 'r') as fin:
                data = fin.read()
        elif fmt == 'gz':
            with gzip.GzipFile(filename, 'r') as fin:
                data = fin.read().decode('utf-8')
        else:
            raise RuntimeError(f'Unknown format "{fmt}" requested.')
        new_db = cls(map_file=None, init_db=False)
        new_db.conn.executescript(data)
        return new_db

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

    def query(self, query_text='1', tags=None, keys=None, add_prefix=''):
        """Queries the ObsDb using user-provided text.  Returns a ResultSet.

        """
        assert(keys is None)  # Not implemented, sry.
        assert(tags is None)  # Not implemented, sry.
        c = self.conn.execute('select * from obs where %s' % query_text)
        results = ResultSet.from_cursor(c)
        if add_prefix is not None:
            results.keys = [add_prefix + k for k in results.keys]
        return results

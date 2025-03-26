import sqlite3
import os
import numpy as np

from .resultset import ResultSet
from . import common


TABLE_DEFS = {
    'obs': [
        "`obs_id` varchar(256) primary key",
        "`timestamp` float",
    ],
    'tags': [
        "`obs_id` varchar(256)",
        "`tag` varchar(256)",
        "CONSTRAINT one_tag UNIQUE (`obs_id`, `tag`)",
    ],
}

TABLE_DEFS_OWB = {
    'obs': [
        "`obs_id` varchar(256)",
        "`wafer_slot` varchar(256)",
        "`bandpass` varchar(256)",
        "`timestamp` float",
        "PRIMARY KEY (`obs_id`, `wafer_slot`, `bandpass`)"
    ],
    'tags': [
        "`obs_id` varchar(256)",
        "`tag` varchar(256)",
        "CONSTRAINT one_tag UNIQUE (`obs_id`, `tag`)",
    ],
}

class ObsDb(object):
    """Observation database.

    The ObsDb helps to associate observations, indexed by an obs_id,
    with properties of the observation that might be useful for
    selecting data or for identifying metadata.

    The main ObsDb table is called 'obs', and contains the columns
    obs_id (string), plus any others deemed important for this context
    (you will probably find timestamp (float representing a unix
    timestamp)).  Additional columns may be added to this table as
    needed.

    The second ObsDb table is called 'tags', and facilitates grouping
    observations together using string labels.

    """

    TABLE_TEMPLATE = [
        "`obs_id` varchar(256)",
    ]

    def __init__(self, map_file=None, init_db=True, obs_wafer_band=False):
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
        
        global TABLE_DEFS
        global TABLE_DEFS_OWB
        if obs_wafer_band:
            TABLE_DEFS = TABLE_DEFS_OWB

        self.conn.row_factory = sqlite3.Row  # access columns by name
        if init_db:
            c = self.conn.cursor()
            c.execute("SELECT name FROM sqlite_master "
                      "WHERE type='table' and name not like 'sqlite_%';")
            tables = [r[0] for r in c]
            changes = False
            for k, v in TABLE_DEFS.items():
                if k not in tables:
                    q = ('create table if not exists `%s` (' % k +
                         ','.join(v) + ')')
                    c.execute(q)
                    changes = True
            if changes:
                self.conn.commit()
        self.primary_keys = self.get_primary_keys()
        if ~np.isin(len(self.primary_keys), [1,3]):
            raise ValueError("Unsupported number of primary keys in the 'obs' table"
                             " only 1 or 3 are allowed.")

    def get_primary_keys(self):
        """Retrieve the primary keys of the specified table.
           This is used whether to index by obs_id or obs-wafer-band."""
        query = "PRAGMA table_info('obs')"
        c = self.conn.execute(query)
        primary_keys = [row['name'] for row in c.fetchall() if row['pk'] > 0]
        return primary_keys
    
    def warn_primary_keys(self, wafer_band):
        """Warn the user if the primary keys are not specified 
           and we're defaulting to using _all."""
        if (None in wafer_band):
            warn_str = 'WARNING: Primary key(s)'
            for i, wb in enumerate(wafer_band):
                if wb is None:
                    wafer_band[i] = '_all'
                    warn_str += f' wafer_band[{i}],'
            warn_str += """
            are not specified and ObsDb is indexed by obs_id, wafer_slot,
            and bandpass. These keys will be set to _all.
            """
            print(warn_str)
        return wafer_band
    
    def __len__(self):
        return self.conn.execute(f'SELECT COUNT({self.primary_keys[0]}) FROM obs').fetchone()[0]
    
    def add_obs_columns(self, column_defs, ignore_duplicates=True, commit=True):
        """Add columns to the obs table.

        Args:
          column_defs (list of pairs of str): Column descriptions, see
            notes.
          ignore_duplicates (bool): If true, requests for new columns
            will be ignored if the column name is already present in
            the table.

        Returns:
          self.

        Notes:
          The input format for column_defs is somewhat flexible.
          First of all, if a string is passed in, it will converted to
          a list by splitting on ",".  Second, if the items in the
          list are strings (rather than tuples), the string will be
          broken into 2 components by splitting on whitespace.
          Finally, each pair of items is interpreted as a (name, data
          type) pair.  The name can be a simple string, or a string
          inside backticks; so 'timestamp' and '`timestamp`' are
          equivalent.  The data type can be any valid sqlite type
          expression (e.g. 'float', 'varchar(256)', etc) or it can be
          one of the three basic python type objects: str, float, int.
          Here are some examples of valid column_defs arguments::

            [('timestamp', float), ('drift', str)]
            ['`timestamp` float', '`drift` varchar(32)']
            'timestamp float, drift str'

        """
        current_cols = self.conn.execute('pragma table_info(obs)').fetchall()
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

    def update_obs(self, obs_id, data={}, tags=[],
                   wafer_band=(None, None), commit=True):
        """Update an entry in the obs table.

        Arguments:
          obs_id (str): The id of the obs to update.
          wafer_band (tuple of str): The wafer_band[0] = wafer_slot and wafer_band[1] = bandpass
            to update.  The default value is ('_all', '_all').
          data (dict): map from column_name to value.
          tags (list of str): tags to apply to this observation (if a
            tag name is prefxed with '!', then the tag name will be
            un-applied, i.e. cleared from this observation.

        Returns:
            self.

        """
        if (len(self.primary_keys) == 3):
            wafer_band = self.warn_primary_keys(wafer_band)
            obs_id = {'obs_id': obs_id, 'wafer_slot': wafer_band[0],
                      'bandpass': wafer_band[1]}
        else:
            obs_id = {'obs_id': obs_id}

        c = self.conn.cursor()
        columns = ', '.join(obs_id.keys())
        placeholders = ', '.join(['?'] * len(obs_id))
        c.execute(f'INSERT OR IGNORE INTO obs ({columns}) VALUES ({placeholders})',
                  tuple(obs_id.values()))
            
        if len(data.keys()):
            settors = [f'{k}=?' for k in data.keys()]
            where_str = ' AND '.join([f'{k}=?' for k in obs_id.keys()])
            c.execute(f'UPDATE obs SET {", ".join(settors)} WHERE {where_str}',
                      tuple(data.values()) + tuple(obs_id.values()))
            
        for t in tags:
            if t[0] == '!':
                # Kill this tag.
                c.execute('DELETE FROM tags WHERE obs_id=? AND tag=?',
                          (obs_id["obs_id"], t[1:]))
            else:
                c.execute('INSERT OR REPLACE INTO tags (obs_id, tag) '
                          'VALUES (?,?)', (obs_id["obs_id"], t))
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
    def from_file(cls, filename, fmt=None, force_new_db=True):
        """This method calls
            :func:`sotodlib.core.metadata.common.sqlite_from_file`
        """
        conn = common.sqlite_from_file(filename, fmt=fmt, force_new_db=force_new_db)
        return cls(conn, init_db=False)

    def get(self, obs_id=None, wafer_band=(None, None), tags=None, add_prefix=''):
        """Returns the entry for obs_id, as an ordered dict.

        If obs_id is None, returns all entries, as a ResultSet.
        However, this usage is deprecated in favor of self.query().

        Args:
          obs_id (str): The observation id to get info for.
          wafer_band (tuple of str): The wafer_band[0] = wafer_slot and wafer_band[1] = bandpass
            to update.  The default value is ('_all', '_all').
          add_prefix (str): A string that will be prepended to each
            field name.  This is for the lazy metadata system, because
            obsdb selectors are prefixed with 'obs:'.

        Returns:
          An ordered dict with the obs table entries for this obs_id,
          or None if the obs_id is not found.  If tags have been
          requested, they will be stored in 'tags' as a list of strings.

        """
        if obs_id is None:
            return self.query('1', add_prefix=add_prefix)
        
        if len(self.primary_keys) == 1:
            query_condition = f"obs_id = ?"
            query_values = (obs_id,)

        elif len(self.primary_keys) == 3:
            wafer_band = self.warn_primary_keys(wafer_band)
            query_condition = " AND ".join([f"{key} = ?" for key in primary_keys])
            query_values = (obs_id, wafer_band[0], wafer_band[1])

        else:
            raise ValueError("Unsupported number of primary keys in the 'obs' table.")

        results = self.query(query_condition, add_prefix=add_prefix, query_values=query_values)
        if len(results) == 0:
            return None
        if len(results) > 1:
            raise ValueError('Too many rows...')  # or integrity error...
        output = results[0]
        if tags:
            # "distinct" should not be needed given uniqueness constraint.
            c = self.conn.execute('select distinct tag from tags where obs_id=?', (obs_id,))
            output['tags'] = [r[0] for r in c]
        return output

    def query(self, query_text='1', tags=None, sort=['obs_id'], add_prefix='',
              query_values=()):
        """Queries the ObsDb using user-provided text.  Returns a ResultSet.

        Args:
          query_text (str): The sqlite query string.  All fields
            should refer to the obs table, or to tags explicitly
            listed in the tags argument.
          query_values (tuple): Values to pass in tuples for obs-wafer-band indexing.
          tags (list of str): Tags to include in the output; if they
            are listed here then they can also be used in the query
            string.  Filtering on tag value can be done here by
            appending '=0' or '=1' to a tag name.

        Returns:
          A ResultSet with one row for each Observation matching the
          criteria.

        Notes:
          Tags are added to the output on request.  For example,
          passing tags=['planet','stare'] will cause the output to
          include columns 'planet' and 'stare' in addition to all the
          columns defined in the obs table.  The value of 'planet' and
          'stare' in each row will be 0 or 1 depending on whether that
          tag is set for that observation.  We can include expressions
          involving planet and stare in the query, for example::

            obsdb.query('planet=1 or stare=1', tags=['planet', 'stare'])

          For simple filtering on tags, pass '=1' or '=0', like this::

            obsdb.query(tags=['planet=1','hwp=1'])

          When filtering is activated in this way, the returned
          results must satisfy all the criteria (i.e. the individual
          constraints are AND-ed).

        """
        sort_text = ''
        if sort is not None and len(sort):
            sort_text = ' ORDER BY ' + ','.join(sort)
        joins = ''
        extra_fields = []
        if tags is not None and len(tags):
            for tagi, t in enumerate(tags):
                if '=' in t:
                    t, val = t.split('=')
                else:
                    val = None
                if val is None:
                    join_type = 'left join'
                    extra_fields.append(f"ifnull(tt{tagi}.obs_id,'') != '' as {t}")
                elif val == '0':
                    join_type = 'left join'
                    extra_fields.append(f"ifnull(tt{tagi}.obs_id,'') != '' as {t}")
                    query_text += f' and {t}==0'
                else:
                    join_type = 'join'
                    extra_fields.append(f'1 as {t}')
                joins += (f" {join_type} (select distinct obs_id from tags where tag='{t}') as tt{tagi} on "
                          f"obs.obs_id = tt{tagi}.obs_id")
        extra_fields = ''.join([','+f for f in extra_fields])
        q = 'select obs.* %s from obs %s where %s %s' % (extra_fields, joins, query_text, sort_text)
        c = self.conn.execute(q, query_values)
        results = ResultSet.from_cursor(c)
        if add_prefix is not None:
            results.keys = [add_prefix + k for k in results.keys]
        return results
    
    def query_linked_dbs(primary_db, secondary_db, query_condition,
                         query_values=(), add_prefix=''):
        """
        Query two ObsDb objects and link their results based on obs_id.

        Args:
            primary_db (ObsDb): The primary database to query.
            secondary_db (ObsDb): The secondary database to query for linked rows.
            query_condition (str): The query condition for the primary database.
            query_values (tuple): Values to parameterize the query for the primary database.
            add_prefix (str): A string to prepend to field names in the result.

        Returns:
            results (list of dict): A list containing dictionaries containing the
                query results from the primary db and the linked secondary db results.
        """
        # Query the primary database
        primary_results = primary_db.query(query_condition, query_values=query_values,
                                           add_prefix=add_prefix)
        if len(primary_results) == 0:
            return None

        results = []
        for pr in primary_results:
            secondary_result = secondary_db.query("obs_id = ?", query_values=(pr['obs_id'],))
            results.append({'primary_db': pr, 
                            'secondary_db': secondary_result})

        return results

    def info(self):
        """Return a dict summarizing the structure and contents of the obsdb;
        this is used by the CLI.

        """
        def _short_list(items, max_len=40):
            i, acc, keepers = 0, 0, []
            while (len(keepers) < 1 or acc < max_len) and i < len(items):
                keepers.append(str(items[i]))
                i += 1
                acc += len(keepers[-1]) + 2
            return  ('[' + ', '.join(map(str, keepers))
                     + (' ...' * (i < len(items))) + ']')

        # Summarize the fields ...
        rs = self.query()
        fields = {}
        for k in rs.keys:
            items = list(set(rs[k]))
            fields[k] = (len(items), _short_list(items))

        # Count occurances of each tag ...
        c = self.conn.execute('select tag, count(obs_id) from tags group by tag order by tag')
        tags = {r[0]: r[1] for r in c}

        return {
            'count': len(rs),
            'fields': fields,
            'tags': tags,
        }

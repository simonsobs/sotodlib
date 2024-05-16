import sqlite3
import os

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

def _generate_query_components_from_tags(query_text='1', tags=None):
    """
    Generate query components from given tags.

    This function creates SQL query components based on the provided tags.
    It generates join clauses, extra fields, and updates the query text to be included in the final query.

    Args:
        query_text (str, optional): The initial query condition. Defaults to '1'.
        tags (list of str, optional): Tags to include in the output; if they
            are listed here then they can also be used in the query string.
            Filtering on tag value can be done here by appending '=0' or '=1' to a tag name.

    Returns:
        tuple: A tuple containing:
            - extra_fields (str): Comma-separated string of extra fields for the SELECT clause.
            - joins (str): String of join clauses to be added to the query.
            - query_text (str): Updated query text including conditions for the tags.
    """
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
                extra_fields.append(f'ifnull(tt{tagi}.obs_id,"") != "" as {t}')
            elif val == '0':
                join_type = 'left join'
                extra_fields.append(f'ifnull(tt{tagi}.obs_id,"") != "" as {t}')
                query_text += f' and {t}==0'
            else:
                join_type = 'join'
                extra_fields.append(f'1 as {t}')
            joins += (f' {join_type} (select distinct obs_id from tags where tag="{t}") as tt{tagi} on '
                      f'obs.obs_id = tt{tagi}.obs_id')
    extra_fields = ''.join([','+f for f in extra_fields])
    return extra_fields, joins, query_text

def _generate_query_components_from_subdb(filepath,
                                          alias,
                                          query_list=None, 
                                          params_list=None,
                                          table_name=None,
                                          obs_id_name=None):
    """
    Generate query components from a sub-database.

    This function creates SQL query components based on the provided sub-database information.
    It generates join clauses, extra fields, and query conditions to be included in the final query.

    Args:
        filepath (str): The file path to the sub-database.
        alias (str): The alias to be used for the sub-database in the query.
        query_list (list of str, optional): A list of query conditions to be applied on the sub-database.
        params_list (list of str, optional): A list of parameters to be selected from the sub-database.
        table_name (str, optional): The name of the table in the sub-database. Defaults to 'map'.
        obs_id_name (str, optional): The name of the observation ID field in the sub-database. Defaults to 'obs:obs_id'.

    Returns:
        tuple: A tuple containing:
            - extra_fields (str): Comma-separated string of extra fields for the SELECT clause.
            - join (str): String of join clause to be added to the query.
            - query (str): String of query conditions to be added to the WHERE clause.
    """
    if table_name is None:
        table_name = 'map'
    if obs_id_name is None:
        obs_id_name = 'obs:obs_id'
    
    if params_list is not None and isinstance(params_list, list):
        extra_fields = []
        for _param in params_list:
            extra_fields.append(f'{alias}.{table_name}.{_param}')
        extra_fields = ''.join([','+f for f in extra_fields])
    elif params_list is None:
        extra_fields = ''
    else:
        raise InputError('Invalid input for params_list')
    
    join  = f' join {alias}.{table_name} on obs.obs_id =  {alias}.{table_name}."{obs_id_name}"'
    
    if query_list is not None and isinstance(query_list, list):
        query = []
        for _query_component in query_list:
            query.append(f'{alias}.{table_name}.{_query_component}')
        query = ''.join([' and '+_q for _q in query])
    elif query_list is None:
        query = ''
    else:
        raise InputError('Invalid input for query_list')
    
    return extra_fields, join, query

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
                    q = ('create table if not exists `%s` (' % k +
                         ','.join(v) + ')')
                    c.execute(q)
                    changes = True
            if changes:
                self.conn.commit()

    def __len__(self):
        return self.conn.execute('select count(obs_id) from obs').fetchone()[0]

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

    def update_obs(self, obs_id, data={}, tags=[], commit=True):
        """Update an entry in the obs table.

        Arguments:
          obs_id (str): The id of the obs to update.
          data (dict): map from column_name to value.
          tags (list of str): tags to apply to this observation (if a
            tag name is prefxed with '!', then the tag name will be
            un-applied, i.e. cleared from this observation.

        Returns:
            self.

        """
        c = self.conn.cursor()
        c.execute('INSERT OR IGNORE INTO obs (obs_id) VALUES (?)',
                  (obs_id,))
        if len(data.keys()):
            settors = [f'{k}=?' for k in data.keys()]
            c.execute('update obs set ' + ','.join(settors) + ' '
                      'where obs_id=?',
                      tuple(data.values()) + (obs_id, ))
        for t in tags:
            if t[0] == '!':
                # Kill this tag.
                c.execute('DELETE FROM tags WHERE obs_id=? AND tag=?',
                          (obs_id, t[1:]))
            else:
                c.execute('INSERT OR REPLACE INTO tags (obs_id, tag) '
                          'VALUES (?,?)', (obs_id, t))
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

    def get(self, obs_id=None, tags=None, add_prefix=''):
        """Returns the entry for obs_id, as an ordered dict.

        If obs_id is None, returns all entries, as a ResultSet.
        However, this usage is deprecated in favor of self.query().

        Args:
          obs_id (str): The observation id to get info for.
          tags (bool): Whether or not to load and return the tags.
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
        results = self.query(f'obs_id="{obs_id}"', add_prefix=add_prefix)
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

    def query(self, query_text='1', tags=None, sort=['obs_id'], add_prefix='', subdbs_info_list=None):
        """
        Generate and execute a query on the main database with optional sub-databases.

        This function constructs and executes an SQL query on the main database,
        incorporating conditions, joins, and fields from optional sub-databases and tags.

        Args:            
            query_text (str, optional): The initial query condition. Defaults to '1'.
            tags (list of str, optional): A list of tags to filter the observations.
                See _generate_query_components_from_tags for details.
            sort (list of str, optional): A list of fields to sort the results by. Defaults to ['obs_id'].
            add_prefix (str, optional): A prefix to add to the result keys. Defaults to ''.
            subdbs_info_list (list of dict, optional): A list of dictionaries containing sub-database information. 
                Each dictionary should contain, filepath (str), query_list (list of str, optional), params_list 
                (list of str, optional), table_name (str, optional), and obs_id_name (str, optional).
                See the Notes(2) for details. If not provided, only the query for the main obsdb is executed.
                
        Returns:
            ResultSet: The result set of the executed query.

        Notes:
            (1) Tags are added to the output on request.  For example,
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

            (2) Sub-databases can be attached and queried in the main query.
            For example, passing subdbs_info_list with appropriate
            parameters allows the function to include extra fields,
            joins, and query conditions from the sub-databases. Each
            sub-database is attached with a unique alias and is used
            to join the main obsdb. The query is then constructed 
            to include the necessary fields and conditions from both 
            the main and sub-databases. For instance::

            subdb_info = {
                'filepath': '/path/to/pwv_class.sqlite',
                'query_list': ['pwv_class_median<2.0', 'pwv_class_rms<0.1'],
                'params_list': ['pwv_class_median', 'pwv_class_rms'],
            }
            obsdb.query(query_text='start_time>1700000000 and planet=1',
                        tags=['planet'], 
                        subdbs_info_list=[subdb_info])

            This queries observations with a start_time greater than 1700000000, 
            a tag of planet, a median pwv smaller than 2.0 [mm], and an rms of pwv
            smaller than 0.1 [mm]. 

            If you do not know the parameters in the sub-database, you can view the params 
            as long as it is a ManifestDb like below::

            from sotolib.core import metadata
            subdb = metadata.ManifestDb('/path/to/pwv_class.sqlite')
            print(subdb.scheme._get_map_table_def())
        
        """
        cursor = self.conn.cursor()
        extra_fields_main, joins_main, query_text_main = _generate_query_components_from_tags(query_text=query_text, tags=tags)

        if subdbs_info_list is not None:
            assert isinstance(subdbs_info_list, list)
            extra_fields_sub = []
            joins_sub = []
            query_text_sub = []
            for i, subdb_info in enumerate(subdbs_info_list):
                assert isinstance(subdb_info, dict)
                if 'filepath' not in subdb_info.keys():
                    raise InputError(f'subdb_info does not have "filepath" in keys')
                filepath = subdb_info['filepath']
                alias = f'subdb{i}'
                attach = f"ATTACH DATABASE '{filepath}' AS '{alias}'"
                cursor = cursor.execute(attach)            
                _extra_fields_sub, _join_sub, _query_sub = _generate_query_components_from_subdb(filepath=filepath, 
                                                                                                alias=alias,
                                                                                                query_list=subdb_info.get('query_list', None),
                                                                                                params_list=subdb_info.get('params_list', None),
                                                                                                table_name=subdb_info.get('table_name', None),
                                                                                                obs_id_name=subdb_info.get('obs_id_name', None),
                                                                                                )
                extra_fields_sub.append(_extra_fields_sub)
                joins_sub.append(_join_sub)
                query_text_sub.append(_query_sub)
            extra_fields_sub = ''.join([''+f for f in extra_fields_sub])
            joins_sub = ''.join(' '+_j for _j in joins_sub)
            query_text_sub = ''.join(' '+q for q in query_text_sub)
            tot_query = f'SELECT obs.* {extra_fields_main} {extra_fields_sub} FROM obs {joins_main} {joins_sub} WHERE {query_text_main} {query_text_sub}'
        else:
            tot_query = f'SELECT obs.* {extra_fields_main} FROM obs {joins_main} WHERE {query_text_main}'

        cursor = cursor.execute(tot_query)
        results = ResultSet.from_cursor(cursor)
        if add_prefix is not None:
            results.keys = [add_prefix + k for k in results.keys]
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

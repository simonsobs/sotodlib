import os
import psycopg
from typing import Optional, Union

from .resultset import ResultSet
from . import common


TABLE_DEFS = {
    "obs": [
        "obs_id varchar(256) primary key",
        "timestamp real",
    ],
    "tags": [
        "obs_id varchar(256)",
        "tag varchar(256)",
        "CONSTRAINT one_tag UNIQUE (obs_id, tag)",
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
        "obs_id varchar(256)",
    ]

    def __init__(self, map_file: psycopg.Connection = None, init_db: bool = True):
        """Instantiate an ObsDb.

        Args:
          map_file (psycopg.Connection): This is
            an psycopg.Connection, it is cached and used.  If this
            argument is None (the default), then the
            psycopg.Connection is opened in 'localhost:5432'.
          init_db (bool): If True, then any ObsDb tables that do not
            already exist in the database will be created.

        Notes:
          If map_file is provided, the database will be connected to
          the indicated postgres server, and any changes made to
          this object be written back to the file.

        """
        if isinstance(map_file, psycopg.Connection):
            self.conn = map_file
        else:
            raise RuntimeError("map_file is not a postgres")

        # self.conn.row_factory = psycopg.Row  # access columns by name
        # Values are by default returned in tuples
        if init_db:
            # c is the cursor
            obsdb_cursor = self.conn.cursor()
            obsdb_cursor.execute(
                "select table_name"
                + " from information_schema.tables"
                + " where table_type='BASE TABLE'"
                + " and table_schema='public';"
            )
            tables = [r[0] for r in obsdb_cursor]
            changes = False
            for k, v in TABLE_DEFS.items():
                if k not in tables:
                    create_query = (
                        f"create table if not exists {k} (" + ",".join(v) + ")"
                    )
                    obsdb_cursor.execute(create_query)
                    changes = True
            if changes:
                self.conn.commit()

    def __len__(self):
        return self.conn.execute("select count(obs_id) from obs").fetchone()[0]

    def add_obs_columns(
        self,
        column_defs: Union[list[tuple[str, str]]],
        ignore_duplicates: Optional[bool] = True,
        commit: Optional[bool] = True,
    ) -> "ObsDb":
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
        current_cols = self.conn.execute(
            "select column_name, data_type, character_maximum_length from "
            + "information_schema.columns where table_name = 'obs';"
        ).fetchall()
        current_cols = [r[1] for r in current_cols]
        if isinstance(column_defs, str):
            column_defs = column_defs.split(',')
        for column_def in column_defs:
            if isinstance(column_def, str):
                column_def = column_def.split()
            name, typestr = column_def
            if typestr is float:
                typestr = "real"
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
                raise ValueError(f"Column {check_name} already exists in table obs")
            self.conn.execute(f"alter table obs add column {name}, {typestr}")
            current_cols.append(check_name)
        if commit:
            self.conn.commit()
        return self

    def update_obs(
        self,
        obs_id: str,
        data: dict = {},
        tags: Optional[list[str]] = [],
        commit: Optional[bool] = True,
    ):
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
        obsdb_cursor = self.conn.cursor()
        obsdb_cursor.execute(
            f"insert into obs (obs_id) values ({obs_id}) on conflict (obs_id) do nothing",
        )

        if len(data.keys()):
            settors = [f"{key} = %s" for key in data.keys()]
            obsdb_cursor.execute(
                "update obs set " + ", ".join(settors) + " where obs_id = %s",
                tuple(data.values()) + (obs_id,),
            )

        for t in tags:
            if t[0] == '!':
                # Kill this tag.
                obsdb_cursor.execute(
                    "delete from tags where obs_id = %s and tag = %s", (obs_id, t[1:])
                )
            else:
                obsdb_cursor.execute(
                    f"insert into tags (obs_id, tag) values ({obs_id}, {t}) "
                    "on conflict (obs_id, tag) do update set obs_id = excluded.obs_id, tag = excluded.tag",
                )
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
        script = common.dump_database(self.conn)
        for line in script:
            new_db.conn.execute(line.strip())
        new_db.conn.commit()
        return new_db

    def to_file(self, filename, overwrite=True, fmt=None):
        """Write the present database to the indicated filename.

        Args:
          filename (str): the path to the output file.
          overwrite (bool): whether an existing file should be
            overwritten.
          fmt (str): 'dump', or 'gz'.  Defaults to 'dump'
            unless the filename ends with '.gz', in which it is 'gz'.

        """
        return common.postgres_to_file(
            self.conn, filename, overwrite=overwrite, fmt=fmt
        )

    @classmethod
    def from_file(
        cls, filename: str, conn: psycopg.Connection, fmt=None, force_new_db=True
    ) -> "ObsDb":
        """This method calls
            :func:`sotodlib.core.metadata.common.sqlite_from_file`
        """
        conn = common.postgres_from_file(
            filename, conn, fmt=fmt, force_new_db=force_new_db
        )
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

    def query(self, query_text="", tags=None, sort=["obs_id"], add_prefix=""):
        """Queries the ObsDb using user-provided text.  Returns a ResultSet.

        Args:
          query_text (str): The sqlite query string.  All fields
            should refer to the obs table, or to tags explicitly
            listed in the tags argument.
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
            sort_text = " order by " + ",".join(sort)
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
                    extra_fields.append(f'coalesce(tt{tagi}.obs_id,"") != "" as {t}')
                elif val == '0':
                    join_type = 'left join'
                    extra_fields.append(f'coalesce(tt{tagi}.obs_id,"") != "" as {t}')
                    query_text += f' and {t}==0'
                else:
                    join_type = 'join'
                    extra_fields.append(f'1 as {t}')
                joins += (
                    f" {join_type} (select distinct obs_id from tags where tag='{t}') as tt{tagi} on "
                    f"obs.obs_id = tt{tagi}.obs_id"
                )
        extra_fields = ''.join([','+f for f in extra_fields])
        where_statement = ""
        if len(query_text):
            where_statement = f" where {query_text}"
        q = f"select obs.* {extra_fields} from obs {joins} {where_statement} {sort_text}"
        c = self.conn.execute(q)
        results = ResultSet.from_cursor(c)
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

import sqlite3
import os
import numpy as np
import warnings

from .resultset import ResultSet
from . import common
from .. import util


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
    '_indices': {
        'idx_obs': 'obs(obs_id)',
        'idx_tags': 'tags(obs_id)',
    },
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
            c.execute("SELECT type, name FROM sqlite_master "
                      "WHERE type in ('table', 'index') and name not like 'sqlite_%';")
            tables = [r[1] for r in c]
            changes = False
            for k, v in TABLE_DEFS.items():
                if k[0] != '_' and k not in tables:
                    q = ('create table if not exists `%s` (' % k +
                         ','.join(v) + ')')
                    c.execute(q)
                    changes = True
            for index, cols in TABLE_DEFS['_indices'].items():
                if index not in tables:
                    c.execute(f'CREATE INDEX IF NOT EXISTS {index} on {cols}')
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
        results = self.query(f"obs_id='{obs_id}'", add_prefix=add_prefix)
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

    def query(self, query_text='1', tags=None, sort=['obs_id'], add_prefix=''):
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
            sort_text = ' ORDER BY ' + ','.join(sort)
        if '"' in query_text:
            warnings.warn('obsdb.query text contains double quotes (") -- '
                          'replacing with single quotes (\').')
            query_text = query_text.replace('"', "'")

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


def diff_obsdbs(obsdb_left, obsdb_right, return_detail=False):
    """Examine all records in two obsdbs and construct a list of changes
    that could made to obsdb_left in order to make it match
    obsdb_right.

    Returns a dict with following entries:

    - ``different`` (bool): whether the two databases carry different
      information.
    - ``patchable`` (bool): whether the function was able to construct
      patching instructions.
    - ``unpatchable_reason`` (str): if not patchable, a string
      explaining why.
    - ``detail`` (various): if not patchable, and return_detail, then
      this will contain detail about the offending data (e.g. obs rows
      in the two dbs that contain discrepant data).
    - ``patch_data`` (dict): if patchable, the data needed to patch
      obsdb_left.  The fields are:

      - ``remove_obs`` (list of obs_id): entries to remove from obs
        table.
      - ``remove_tags`` (list of tuple): entries to remove from tags
        table.
      - ``new_obs`` (list of dict): rows of new data for obs table --
        each dict can be passed directly to obsdb.update_obs.
      - ``new_tags`` (list of tuple): rows of data for tags table
        (obs_id, tag).


    Notes:

    In the present implementation, only changes involving adding rows
    to obsdb_left (either whole obs rows or tag rows) will yield a
    patchable result.  Cases where some data has changed, or obs or
    tags have been deleted, will simply return as unpatchable.  This
    is probably pretty easy to extend, should the need arise.

    """
    if isinstance(obsdb_left, str):
        obsdb_left = ObsDb.from_file(obsdb_left, force_new_db=False)
    if isinstance(obsdb_right, str):
        obsdb_right = ObsDb.from_file(obsdb_right, force_new_db=False)

    def failure_declaration(reason, detail=None):
        if not return_detail:
            detail = None
        return {'different': True,
                'patchable': False,
                'unpatchable_reason': reason,
                'detail': detail}

    full = [db.query() for db in [obsdb_left, obsdb_right]]
    if full[0].keys != full[1].keys:
        return failure_declaration(
            'obsdb_left and obsdb_right have different column names.',
            detail=[full[0].keys, full[1].keys])

    # Convert to arrays.
    obs_ids = [set(f['obs_id']) for f in full]

    # Insist right is superset of left.
    left_not_right = sorted(list(obs_ids[0].difference(obs_ids[1])))
    if len(left_not_right):
        return failure_declaration(
            f'obsdb_left contains {len(left_not_right)} '
            'obs not found in obsdb_right.',
            detail=left_not_right)

    # Any obs in common?
    unmatched_right = np.ones(len(full[1]), bool)
    common = sorted(list(obs_ids[0].intersection(obs_ids[1])))
    if len(common):
        common, i0, i1 = util.get_coindices(*(f['obs_id'] for f in full))
        diffs = []
        for i, (_i0, _i1) in enumerate(zip(i0, i1)):
            if full[0][_i0] != full[1][_i1]:
                diffs.append((full[0][_i0], full[1][_i1]))
        if len(diffs):
            return failure_declaration(
                f'obsdb_left and obsdb_right have {len(diffs)} obs '
                'in common, with different data.',
                detail=diffs)
        unmatched_right[i1] = False

    # Ok finally
    pd = {
        'remove_obs': [],
        'remove_tags': [],
        'new_obs': [],
        'new_tags': [],
    }
    for idx in unmatched_right.nonzero()[0]:
        pd['new_obs'].append(full[1][idx])

    # Tag check.
    tags_tuples = [
        list(map(tuple, db.conn.execute(
            'select distinct obs_id, tag from tags  '
            'order by obs_id, tag').fetchall()))
        for db in [obsdb_left, obsdb_right]]

    # Collapse tags to single strings and eliminate duplicates.
    DELIM = ':::/:::'
    common, i0, i1 = util.get_coindices(*[[t[0] + DELIM + t[1] for t in tt]
                                          for tt in tags_tuples])
    if len(i0) != len(tags_tuples[0]):
        return failure_declaration(
            f'obsdb_left contains {len(tags_tuples[0]) - len(i0)} '
            'tags not found in obsdb_right',
            detail=list(set(tags_tuples[0]).difference(tags_tuples[1])))
    unmatched_right = np.ones(len(tags_tuples[1]), bool)
    unmatched_right[i1] = False
    for idx in unmatched_right.nonzero()[0]:
        pd['new_tags'].append(tags_tuples[1][idx])

    different = any([(len(v) != 0) for v in pd.values()])

    return {
        'different': different,
        'patchable': True,
        'patch_data': pd,
    }


def patch_obsdb(patch_data, target_db):
    """Update an ObsDb with a batch of changes.

    Args:
      target_db (ObsDb): the database where changes should be made.
      patch_data (dict): patch information, as returned by
        diff_obsdbs.

    """
    assert len(patch_data['remove_obs']) == 0
    assert len(patch_data['remove_tags']) == 0

    for obs_entry in patch_data['new_obs']:
        target_db.update_obs(obs_entry['obs_id'], obs_entry,
                             commit=False)

    # Group new tags by obs.
    tags_obsed = {}
    for k, v in patch_data['new_tags']:
        if k not in tags_obsed:
            tags_obsed[k] = [v]
        else:
            tags_obsed[k].append(v)
    for obs, tags in tags_obsed.items():
        target_db.update_obs(obs, {}, tags=tags, commit=False)

    target_db.conn.commit()

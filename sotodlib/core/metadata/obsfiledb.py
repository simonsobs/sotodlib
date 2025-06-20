import sqlite3
import os
import sys
import argparse
from collections import OrderedDict
import numpy as np

from . import common
from .. import util
from .resultset import ResultSet

_DB_VERSION = 3

TABLE_DEFS = {
    'detsets': [
        "`name`    varchar(16)",
        "`det`     varchar(32)",
        "CONSTRAINT name_det UNIQUE (name, det)",
    ],
    'files': [
        "`name`    varchar(256) unique",
        "`detset`  varchar(16)",
        "`obs_id`  varchar(256)",
        "`sample_start` int",
        "`sample_stop`  int",
    ],
    'frame_offsets': [
        "`file_name` varchar(256)",
        "`frame_index` int",
        "`byte_offset` int",
        "`frame_type` varchar(16)",
        "`sample_start` int",
        "`sample_stop` int",
    ],
    'meta': [
        "`param` varchar(32) UNIQUE",
        "`value` varchar",
    ],
    '_indices': [
        "CREATE INDEX IF NOT EXISTS idx_obs_id_column ON files(obs_id)",
        "CREATE INDEX IF NOT EXISTS idx_detset_column ON detsets(name)",
    ],
}


class ObsFileDb:
    """sqlite3-based database for managing large archives of files.

    The data model here is that each distinct "Observation" comprises
    co-sampled detector data for a large number of detectors.  Each
    detector belongs to a single "detset", and there is a set of files
    containing the data for each detset.  Finding the file that
    contains data for a particular detector is a matter of looking up
    what detset the detector is in, and looking up what file covers
    that detset.

    Note that many functions have a "commit" option, which simply
    affects whether the .commit is called on the database or not (it
    can be faster to suppress commit ops when a running a batch of
    updates, and commit manually at the end).

    """

    #: The sqlite3 database connection.
    conn = None

    #: Path relative to which filenames in the database should be
    #: interpreted.  This only applies to relative filenames (those not
    #: starting with /).
    prefix = ''

    def __init__(self, map_file=None, prefix=None, init_db=True, readonly=False):
        """Instantiate an ObsFileDb.

        Arguments:
          map_file (string): sqlite database file to map.  Defaults to
            ':memory:'.
          prefix (string): as described in class documentation.
          init_db (bool): If True, attempt to create the database
            tables.
          readonly (bool): If True, the database file will be mapped
            in read-only mode.  Not valid on dbs held in :memory:.

        """
        if isinstance(map_file, sqlite3.Connection):
            self.conn = map_file
        else:
            if map_file is None:
                map_file = ':memory:'
            self.conn = sqlite3.connect(map_file)
            uri = False
            if readonly:
                if map_file == ':memory:':
                    raise ValueError('Cannot honor request for readonly db '
                                     'mapped to :memory:.')
                map_file, uri = 'file:%s?mode=ro' % map_file, True

            self.conn = sqlite3.connect(map_file, uri=uri)

        self.conn.row_factory = sqlite3.Row  # access columns by name

        self.prefix = self._get_prefix(map_file, prefix)
        if init_db and not readonly:
            self._create()

    @staticmethod
    def _get_prefix(map_file, prefix):
        """Common logic for setting the file prefix based on map_file and
        prefix arguments.

        """
        if prefix is not None:
            return prefix
        if map_file == ':memory:':
            return ''
        return os.path.split(os.path.abspath(map_file))[0] + '/'

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
    def from_file(cls, filename, prefix=None, fmt=None, force_new_db=True):
        """This method calls
            :func:`sotodlib.core.metadata.common.sqlite_from_file`
        """
        conn = common.sqlite_from_file(filename, fmt=fmt,
                                       force_new_db=force_new_db)
        if prefix is None:
            prefix = os.path.split(filename)[0] + '/'
        return cls(conn, init_db=False, prefix=prefix, )

    @classmethod
    def for_dir(cls, path, filename='obsfiledb.sqlite', readonly=True):
        """Deprecated; use from_file()."""
        print('Use of ObsFileDb.for_dir() is deprecated... use from_file.')
        return cls.from_file(os.path.join(path, filename), prefix=path)

    def copy(self, map_file=None, overwrite=False):
        """
        Duplicate the current database into a new database object, and
        return it.  If map_file is specified, the new database will be
        connected to that sqlite file on disk.  Note that a quick way
        of writing a Db to disk to call copy(map_file=...).
        """
        if map_file is None:
            map_file = ':memory:'
        script = ' '.join(self.conn.iterdump())
        if map_file != ':memory:' and os.path.exists(map_file):
            if not overwrite:
                raise RuntimeError("Output database '%s' exists -- remove or "
                                   "pass overwrite=True to copy." % map_file)
            os.remove(map_file)
        new_db = ObsFileDb(map_file, init_db=False)
        new_db.conn.executescript(script)
        new_db.prefix = self.prefix
        return new_db

    def _get_version(self, conn=None):
        if conn is None:
            conn = self.conn
        rows = conn.execute("select value from meta where "
                            "param='obsfiledb_version'").fetchall()
        if len(rows) == 0:
            return None
        return int(rows[0][0])

    def _create(self):
        """
        Create the database tables if they do not already exist.
        """
        # Create the tables:
        c = self.conn.cursor()
        for table_name, column_defs in TABLE_DEFS.items():
            if table_name.startswith('_'):
                continue
            q = ('create table if not exists `%s` (' % table_name  +
                 ','.join(column_defs) + ')')
            c.execute(q)

        for index in TABLE_DEFS['_indices']:
            c.execute(index)

        if self._get_version(conn=c) is None:
            c.execute('insert or ignore into meta (param,value) values (?,?)',
                      ('obsfiledb_version', _DB_VERSION))

        self.conn.commit()

    def add_detset(self, detset_name, detector_names, commit=True):
        """Add a detset to the detsets table (by adding detectors with
        specific names to it).

        Arguments:
          detset_name (str): The name of the detset.
          detector_names (list of str): New detectors belonging to
            this detset.

        """
        for d in detector_names:
            q = 'insert into detsets (name,det) values (?,?)'
            self.conn.execute(q, (detset_name, d))
        if commit:
            self.conn.commit()

    def add_obsfile(self, filename, obs_id, detset, sample_start=None, sample_stop=None,
                    commit=True):
        """Add an observation file to the files table.

        Arguments:
          filename (str): The filename, relative to the data base
            directory and without a leading /.
          obs_id (str): The observation id.
          detset (str): The detset name.
          sample_start (int): The observation sample index at the
            start of this file.
          sample_stop (int): sample_start + n_samples.

        """
        self.conn.execute(
            'insert into files (name,detset,obs_id,sample_start,sample_stop) '
            'values (?,?,?,?,?)',
            (filename,detset,obs_id,sample_start,sample_stop))
        if commit:
            self.conn.commit()

    # Retrieval

    def get_obs(self):
        """Returns all a list of all obs_id present in this database.

        """
        c = self.conn.execute('select distinct obs_id from files')
        return [r[0] for r in c]
    
    def get_obs_with_detset(self, detset):
        """Returns a list of all obs_ids that include a specified detset"""
        c = self.conn.execute(
            f"select distinct obs_id from files where detset='{detset}'"
        )
        return [r[0] for r in c]

    def get_detsets(self, obs_id):
        """Returns a list of all detsets represented in the observation
        specified by obs_id.

        """
        c = self.conn.execute('select distinct detset from files '
                              'where obs_id=?', (obs_id,))
        return [r[0] for r in c]

    def get_dets(self, detset):
        """Returns a list of all detectors in the specified detset.

        """
        c = self.conn.execute('select det from detsets where name=?', (detset,))
        return [r[0] for r in c]

    def get_det_table(self, obs_id):
        """Get table of detectors and detsets suitable for use with Context
        det_info.  Returns Resultset with keys=['dets:detset','dets:readout_id'].

        """
        c = self.conn.execute(
            'select distinct detsets.name as `dets:detset`, det as `dets:readout_id`'
            'from detsets join files '
            'on files.detset=detsets.name where obs_id=?', (obs_id, ))
        return ResultSet.from_cursor(c)

    def get_files(self, obs_id, detsets=None, prefix=None):
        """Get the file names associated with a particular obs_id and detsets.

        Returns:

          OrderedDict where the key is the detset name and the value
          is a list of tuples of the form (full_filename,
          sample_start, sample_stop).

        """
        if prefix is None:
            prefix = self.prefix

        if detsets is None:
            c = self.conn.execute('select detset, name, sample_start, sample_stop '
                                  'from files where obs_id=? '
                                  'order by detset, sample_start',
                                  (obs_id,))
        else:
            c = self.conn.execute('select detset, name, sample_start, sample_stop '
                                  'from files where obs_id=? and detset in (%s) '
                                  'order by detset, sample_start' %
                                  ','.join(['?' for _ in detsets]),
                                  (obs_id,) + tuple(detsets))
        output = OrderedDict()
        for r in c:
            if not r[0] in output:
                output[r[0]] = []
            output[r[0]].append((os.path.join(prefix, r[1]), r[2], r[3]))
        return output

    def lookup_file(self, filename, resolve_paths=True, prefix=None, fail_ok=False):
        """Determine what, if any, obs_id (and detset and sample range) is
        associated with the specified data file.

        Args:
          filename (str): a string corresponding to a file that is
            covered by this db.  See note on how this is resolved.
          resolve_paths (bool): If True, then the incoming filename is
            treated as a path to a specific file on disk, and the
            database is queried for files that also resolve to that
            are equivalent to that same file on disk (accounting for
            prefix).  If False, then the incoming filename is taken as
            opaque text to match against the corresponding entries in
            the obsfiledb file "name" column (including whatever path
            information is in either of those strings).
          fail_ok (bool): If True, then None is returned if the
            filename is not found in the db (instead of raising
            RuntimeError).

        Returns:
          A dict with entries:
          - ``obs_id``: The obs_id
          - ``detsets``: A list containing the name of the single detset
            covered by this file.
          - ``sample_range``: A tuple with the start and stop sample
            indices for this file.

        """
        if prefix is None:
            prefix = self.prefix

        if resolve_paths:
            # Clarify our target ...
            filename = os.path.realpath(filename)
            basename = os.path.split(filename)[1]
            # Do a non-conclusive match against the basename ...
            c = self.conn.execute(
                'select name, obs_id, detset, sample_start, sample_stop '
                'from files where name like ?', ('%' + basename, ))
            rows = c.fetchall()
            # Keep only the rows that are definitely our target file.
            rows = [r for r in rows
                    if os.path.realpath(os.path.join(prefix, r[0])) == filename]
        else:
            # Do literal exact matching of filename to database.
            c = self.conn.execute(
                'select name, obs_id, detset, sample_start, sample_stop '
                'from files where name=?', (filename, ))
            rows = c.fetchall()

        if len(rows) == 0:
            if fail_ok:
                return None
            raise RuntimeError('No match found for "%s"' % filename)
        if len(rows) > 1:
            raise RuntimeError('Multiple matches found for "%s"' % filename)
        _, obs_id, detset, start, stop = tuple(rows[0])

        return {'obs_id': obs_id,
                'detsets': [detset],
                'sample_range': (start, stop)}

    def verify(self, prefix=None):

        """Check the filesystem for the presence of files described in the
        database.  Returns a dictionary containing this information in
        various forms; see code for details.

        This function is used internally by the drop_incomplete()
        function, and may also be useful for debugging file-finding
        problems.

        """
        if prefix is None:
            prefix = self.prefix

        # Check for the presence of each listed file.
        c = self.conn.execute('select name, obs_id, detset, sample_start '
                              'from files')
        rows = []
        for r in c:
            fp = os.path.join(prefix, r[0])
            rows.append((os.path.exists(fp), fp) + tuple(r))

        obs = OrderedDict()
        for r in rows:
            present, fullpath, name, obs_id, detset, sample_start = r
            if obs_id not in obs:
                obs[obs_id] = {'present': [],
                               'absent': []}
            if present:
                obs[obs_id]['present'].append((detset, sample_start))
            else:
                obs[obs_id]['absent'].append((detset, sample_start))

        # Make a detset, sample_start grid for each observation.
        grids = OrderedDict()
        for k, v in obs.items():
            items = v['present']
            detsets = list(set([a for a, b in items]))
            sample_starts = list(set([b for a, b in items]))
            grid = np.zeros((len(detsets), len(sample_starts)), bool)
            for a, b in items:
                grid[detsets.index(a), sample_starts.index(b)] = True
            grids[k] = {'detset': detsets,
                        'sample_start': sample_starts,
                        'grid': grid}

        return {'raw': rows,
                'obs_id': obs,
                'grids': grids}

    def drop_obs(self, obs_id):
        """Delete the specified obs_id from the database.  Returns a list of
        files that are no longer covered by the database (with prefix).

        """
        # What files does this affect?
        c = self.conn.execute('select name from files where obs_id=?',
                              (obs_id,))
        affected_files = [os.path.join(self.prefix, r[0]) for r in c]
        # Drop them.
        self.conn.execute('delete from frame_offsets where file_name in '
                          '(select name from files where obs_id=?)',
                          (obs_id,))
        self.conn.execute('delete from files where obs_id=?',
                          (obs_id,))
        self.conn.commit()
        return affected_files

    def drop_detset(self, detset):
        """Delete the specified detset from the database.  Returns a list of
        files that are no longer covered by the database (with
        prefix).

        """
        # What files does this affect?
        c = self.conn.execute('select name from files where detset=?',
                              (detset,))
        affected_files = [os.path.join(self.prefix, r[0]) for r in c]
        # Drop them.
        self.conn.execute('delete from frame_offsets where file_name in '
                          '(select name from files where detset=?)',
                          (detset,))
        self.conn.execute('delete from files where detset=?',
                          (detset,))
        self.conn.commit()
        return affected_files

    def drop_incomplete(self):
        """Compare the files actually present on the system to the ones listed
        in this database.  Drop detsets from each observation, as
        necessary, such that the database is consistent with the file
        system.

        Returns a list of files that are on the system but are no
        longer included in the database.

        """
        affected_files = []
        scan = self.verify()
        for obs_id, info in scan['grids'].items():
            # Drop any detset that does not have complete sample
            # coverage.
            detset_to_drop = np.any(~info['grid'], axis=1)
            for i in detset_to_drop.nonzero()[0]:
                affected_files.extend(
                    [r[0] for r in self.conn.execute(
                        'select name from files where obs_id=? and detset=?',
                        (obs_id, info['detset'][i]))])
                self.conn.execute(
                    'delete from files where obs_id=? and detset=?',
                    (obs_id, info['detset'][i]))
        # Drop any detset that no longer appear in any files.
        self.conn.execute('delete from detsets where name not in '
                            '(select distinct detset from files)')
        self.conn.commit()
        self.conn.execute('vacuum')

        # Return the full paths of only the existing files that have
        # been dropped from the Db.
        path_map = {r[2]: r[1] for r in scan['raw'] if r[0]}
        return [r[1] for r in scan['raw'] if r[2] in affected_files]

    def get_file_list(self, fout=None):
        """Returns a list of all files in the database, without the file
        prefix, sorted by observation / detset / sample_start.  This
        is the sort of list one might use with rsync --files-from.

        If you pass an open file or filename to fout, the names will
        be written there, too.

        """
        c = self.conn.execute('select name from files order by '
                              'obs_id, detset, sample_start')
        output = [r[0] for r in c]
        if fout is not None:
            if isinstance(fout, str):
                assert(not os.path.exists(fout))
                fout = open(fout, 'w')
            for line in output:
                fout.write(line+'\n')
        return output


def _reconcile_rs(rs0, rs1, sep='##'):
    keys0 = np.array(['##'.join(map(str, r)) for r in rs0.rows])
    keys1 = np.array(['##'.join(map(str, r)) for r in rs1.rows])
    common, i0, i1 = util.get_coindices(keys0, keys1)
    ex0 = np.ones(len(keys0), bool)
    ex1 = np.ones(len(keys1), bool)
    ex0[i0] = False
    ex1[i1] = False
    return (common, i0, i1), (ex0, ex1)


def diff_obsfiledbs(db_left, db_right, return_detail=False):
    """Examine all records in two obsfiledbs and construct a list of
    changes that could made to db_left in order to make it match
    db_right.

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
      db_left.  The fields are:

      - ``remove_files`` (list): entries to remove from obs table.
      - ``remove_dets`` (list): entries to remove from tags table.
      - ``new_files`` (ResultSet): new data for obs table -- iteration
        will yield dict that can be passed directly to db.update_obs.
      - ``new_dets`` (ResultSet): new data for dets table.


    Notes:

    In the present implementation, only changes involving adding rows
    to db_left (either whole obs rows or tag rows) will yield a
    patchable result.  Cases where some data has changed, or rows
    deleted, will simply return as unpatchable.

    """
    if isinstance(db_left, str):
        db_left = ObsFileDb.from_file(db_left, force_new_db=False)
    if isinstance(db_right, str):
        obsdb_right = ObsFileDb.from_file(db_right, force_new_db=False)

    def failure_declaration(reason, detail=None):
        if not return_detail:
            detail = None
        return {'different': True,
                'patchable': False,
                'unpatchable_reason': reason,
                'detail': detail}

    # Check meta.
    metas = [ResultSet.from_cursor(db.conn.execute(
        'select * from meta order by param')) for db in [db_left, db_right]]
    (common, i0, i1), (ex0, ex1) = _reconcile_rs(*metas)
    if not (len(metas[0]) == len(metas[1]) == len(common)):
        return failure_declaration(
            'Databases have different meta params.',
            detail=metas)

    # Check frame_offsets are not populated.
    foffs = [ResultSet.from_cursor(db.conn.execute(
        'select * from frame_offsets')) for db in [db_left, db_right]]
    if not (len(foffs[0]) == len(foffs[1]) == 0):
        return failure_declaration(
            'One or other database has non-empty frame_offsets table.',
            detail=metas)

    # The detsets table.
    detsets = [ResultSet.from_cursor(db.conn.execute(
        'select name, det from detsets order by name, det'))
               for db in [db_left, db_right]]
    (common, i0, i1), (ex0, ex1) = _reconcile_rs(*detsets)
    if ex0.any():
        bad_ds = detsets[0].subset(rows=ex0.nonzero()[0], keys=['name']).distinct()
        return failure_declaration(
            f'db_left contains {ex0.sum()} dets entries from {len(bad_ds)} detsets, '
            ' not found in db_right.',
            detail=bad_ds)
    new_dets = detsets[1].subset(rows=ex1.nonzero()[0])

    # The files table.
    files = [ResultSet.from_cursor(db.conn.execute(
        'select * from files order by obs_id, detset, name, sample_start'))
            for db in [db_left, db_right]]
    (common, i0, i1), (ex0, ex1) = _reconcile_rs(*files)
    if ex0.any():
        bad_obs = files[0].subset(rows=ex0.nonzero()[0], keys=['obs_id']).distinct()
        return failure_declaration(
            f'db_left contains {ex0.sum()} files entries from {len(bad_obs)} obs_ids, '
            'not found in db_right.',
            detail=bad_obs)
    new_files = files[1].subset(rows=ex1.nonzero()[0])
    new_files.keys[new_files.keys.index('name')] = 'filename'

    # Ok finally
    pd = {
        'remove_files': [],
        'remove_dets': [],
        'new_files': new_files,
        'new_dets': new_dets,
    }
    different = len(new_files) > 0 or len(new_dets) > 0
    return {
        'different': different,
        'patchable': True,
        'patch_data': pd,
    }


def patch_obsfiledb(patch_data, target_db):
    """Update an ObsFileDb with a batch of changes.

    Args:
      target_db (ObsFileDb): the database where changes should be made.
      patch_data (dict): patch information, as returned by
        diff_obsfiledbs.

    """
    assert len(patch_data['remove_files']) == 0
    assert len(patch_data['remove_dets']) == 0

    for file_entry in patch_data['new_files']:
        target_db.add_obsfile(**file_entry, commit=False)

    for det_entry in patch_data['new_dets']:
        target_db.add_detset(det_entry['name'], [det_entry['det']], commit=False)

    target_db.conn.commit()


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            epilog="""For details of individual modes, pass a dummy database argument
            followed by the mode and -h, e.g.: "%(prog)s x files -h" """)

    parser.add_argument('filename', help="Path to an ObsFileDb.",
                        metavar='obsfiledb.sqlite')

    cmdsubp = parser.add_subparsers(
        dest='mode')

    # "files"
    p = cmdsubp.add_parser(
        'files', usage="""Syntax:

    %(prog)s
    %(prog)s --all
    %(prog)s --clean

        This will print out a list of the files in the db,
        along with obs_id and detset.  Only a few lines
        will be shown, unless --all is passed.  To get a
        simple list of all files (for rsync or something),
        pass --clean.
        """,
        help="List the files referenced in the database.")

    p.add_argument('--clean', action='store_true',
                   help="Print a simple list of all files (for script digestion).")
    p.add_argument('--all', action='store_true',
                   help="Print all files, not an abbreviated list.")

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

        These operations will create a duplicate of the source
        ObsFileDb, with only the filenames (potentially) altered.  Any
        filename that starts with the first argument will be changed,
        in the output, to instead start with the second argument.
        When you do this you must either say where to write the output
        (-o) or give the program permission to overwrite your input
        database file.  Note that the first argument need not match
        all entries in the database; you can use it to pick out a
        subset (even a single entry).  """)
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

    # "diff"
    p = cmdsubp.add_parser(
        'diff', help=
        "Check database against some upstream target; report diff/patchability.",
        usage="""Syntax:

    %(prog)s [output options]
        """)
    p.add_argument('upstream_db')
    p.add_argument('--patch', action='store_true', help=
                   "If possible, patch target to match the upstream.")

    # "fix-db"
    p = cmdsubp.add_parser(
        'fix-db', help=
        "Upgrade database (schema fixes, etc).",
        usage="""Syntax:

    %(prog)s [output options]
        """)
    p.add_argument('--overwrite', action='store_true', help=
                   "Store modified database in the same file.")
    p.add_argument('--output-db', '-o', help=
                   "Store modified database in this file.")
    p.add_argument('--dry-run', action='store_true', help=
                   "Run the conversion steps but do not write the results anywhere.")

    return parser

def main(args=None, parser=None):
    """Entry point for the so-metadata tool."""
    if args is None:
        args = sys.argv[1:]
    elif not isinstance(args, argparse.Namespace):
        parser = get_parser()
        args = parser.parse_args(args)

    if args.mode is None:
        args.mode = 'summary'

    db = ObsFileDb.from_file(args.filename, force_new_db=False)

    if args.mode == 'files':
        # Get all files.
        rows = db.conn.execute(
            'select obs_id, name, detset from files '
            'order by obs_id, detset, name').fetchall()

        if args.clean:
            for obs_id, filename, detset in rows:
                print(filename)
        else:
            fmt = '  {obs_id} {detset} {filename}'
            hdr = fmt.format(obs_id="obs_id", detset="detset", filename="Filename")
            print(hdr)
            print('-' * (len(hdr) + 20))
            n = len(rows)
            if n > 20 and not args.all:
                rows = rows[:10]
            for obs_id, filename, detset in rows:
                print(fmt.format(obs_id=obs_id, filename=filename, detset=detset))
            if len(rows) < n:
                print(fmt.format(obs_id='...', filename='+%i others' % (n - len(rows)), detset=''))
                print('(Pass --all to show all results.)')
            print()

    elif args.mode == 'reroot':
        # Reconnect with write?
        if args.overwrite:
            if args.output_db:
                parser.error("Specify only one of --overwrite or --output-db.")
            db = ObsFileDb.from_file(args.filename, force_new_db=True)
            args.output_db = args.filename
        else:
            if args.output_db is None:
                parser.error("Specify an output database name with --output-db, "
                             "or pass --overwrite to clobber.")
            db = ObsFileDb.from_file(args.filename, force_new_db=True)

        # Get all files matching this prefix ...
        c = db.conn.execute('select name from files '
                            'where name like "%s%%"' % (args.old_prefix))
        rows = c.fetchall()
        print('Found %i records matching prefix ...'
               % len(rows))

        print('Converting to new prefix ...')
        n_examples = 1

        if not args.dry_run:
            c = db.conn.cursor()

        for (name, ) in rows:
            new_name = args.new_prefix + name[len(args.old_prefix):]
            if n_examples > 0:
                print(f'  Example: converting filename\n'
                      f'      "{name}"\n'
                      f'    to\n'
                      f'      "{new_name}"')
                n_examples -= 1
            if not args.dry_run:
                c.execute('update files set name=? where name=?', (new_name, name))

        print('Saving to %s' % args.output_db)
        if not args.dry_run:
            db.conn.commit()
            c.execute('vacuum')
            db.to_file(args.output_db)

    elif args.mode == 'diff':
        print(f'Comparing to {args.upstream_db} ...')
        db = ObsFileDb(args.filename)
        db_right = ObsFileDb(args.upstream_db)
        report = diff_obsfiledbs(db, db_right)
        if not report['different']:
            print(' ... databases are in sync.')
        elif report['patchable']:
            print(' ... upstream is different, but the target db can be patched to match.')
        else:
            print(' ... upstream and target have irreconcilable differences.')
            parser.exit(1)

        if args.patch and report['different']:
            print()
            print('Patching ...')
            patch_obsfiledb(report['patch_data'], db)
            print(' ... done')
            print()

    elif args.mode == 'fix-db':
        # Reconnect with write?
        if args.overwrite:
            if args.output_db:
                parser.error("Specify only one of --overwrite or --output-db.")
            db = ObsFileDb.from_file(args.filename, force_new_db=True)
            args.output_db = args.filename
        else:
            if args.output_db is None:
                parser.error("Specify an output database name with --output-db, "
                             "or pass --overwrite to clobber.")
            db = ObsFileDb.from_file(args.filename, force_new_db=True)

        # Get version ...
        v = db._get_version()
        print(f'Database reports as version = {v}')

        changes = False
        if v == 1:
            # Copy detsets to new table, where uniqueness has been
            # relaxed.  Re-do the meta table, where uniqueness is
            # enforced to prevent lots of redundant rows added by
            # _create().
            changes = True
            for line in [
                    'drop table meta',
                    'alter table detsets rename to old_detsets',
                    '*',
                    'insert into detsets (name, det) select name, det from old_detsets',
                    'drop table old_detsets',
            ]:
                if line == '*':
                    print('Creating updated tables.')
                    db._create()
                    continue
                print(f'Running: {line}')
                db.conn.execute(line)
            print()

            for index in TABLE_DEFS['_indices']:
                print('Adding indexes...')
                db.conn.execute(index)

        elif v == 2:
            changes = True
            for index in TABLE_DEFS['_indices']:
                print('Adding indexes...')
                db.conn.execute(index)
            print('Bumping version')
            db.conn.execute('insert or replace into meta (param,value) values (?,?)',
                            ('obsfiledb_version', 3))

        if changes:
            print('Saving to %s' % args.output_db)
            if not args.dry_run:
                db.conn.commit()
                db.conn.execute('vacuum')
                db.to_file(args.output_db)
        else:
            print('No changes to make.')

    else:
        parser.error(f'Unimplemented mode, "{args.mode}".')

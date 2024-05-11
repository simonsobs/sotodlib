import sqlite3
import gzip
import os


def sqlite_connect(filename=None, mode="w"):
    """Utility function for connecting to an sqlite3 DB.

    This provides a single function for opening an sqlite connection
    consistently across the code base.  When connecting to a file on
    disk we try to use options which will be more performant in the
    situation where the DB is on a networked / shared filesystem and
    being accessed from multiple processes.

    Args:
        filename (str):  The path on disk or None if using an
            in-memory DB.
        mode (str):  Either "r" or "w".

    Returns:
        (sqlite3.Connection):  The database connection.

    """
    if filename is None:
        # Memory-backed DB
        if mode == "r":
            raise ValueError("Cannot open memory DB in read-only mode")
        return sqlite3.connect(":memory:")

    # This timeout is in seconds.  If multiple processes are writing, they
    # might be blocked for a while until they get their turn.  This prevents
    # them from giving up too soon if other processes have a write lock.
    # https://www.sqlite.org/pragma.html#pragma_busy_timeout
    busy_time = 1000

    # Journaling options

    # Persistent journaling mode.  File creation / deletion can be expensive
    # on some filesystems.  This just writes some zeros to the header and
    # leaves the file.  This journal is a "side car" file next to the original
    # DB file and is safe to delete manually if one is sure that no processes
    # are accessing the DB.
    # https://www.sqlite.org/pragma.html#pragma_journal_mode
    journal_mode = "persist"

    # Max size of the journal.  Although it is being overwritten repeatedly,
    # if it gets too large we purge it and recreate.  This should not happen
    # for most normal operations.
    # https://www.sqlite.org/pragma.html#pragma_journal_size_limit
    journal_size = f"{10 * 1024 * 1024}"

    # Disk synchronization options

    # Using "normal" instead of the default "full" can avoid potentially expensive
    # (on network filesystems) sync operations.
    # https://www.sqlite.org/pragma.html#pragma_synchronous
    sync_mode = "normal"

    # Memory caching

    # The default page size in modern sqlite is 4096 bytes, and should be fine.
    # We set this explicitly to allow easy changing in the future or keeping it
    # fixed if the default changes.
    # https://www.sqlite.org/pragma.html#pragma_page_size
    page_size = 4096

    # The number of pages to cache in memory.  Setting this to a few MB of RAM
    # can have substantial performance benefits.  Total will be number of pages
    # times page size.
    # https://www.sqlite.org/pragma.html#pragma_cache_size
    n_cache_pages = 4000

    # Open connection
    if mode == "r":
        connstr = f"file:{filename}?mode=ro"
    else:
        connstr = f"file:{filename}?mode=rwc"
    conn = sqlite3.connect(connstr, uri=True, timeout=busy_time)

    # Set cache sizes
    conn.execute(f"pragma page_size={page_size}")
    conn.execute(f"pragma cache_size={n_cache_pages}")

    if mode == "r":
        # Read-only mode, all done.
        return conn

    # In write mode, set journaling / sync options
    conn.execute(f"pragma journal_mode={journal_mode}")
    conn.execute(f"pragma journal_size_limit={journal_size}")
    conn.execute(f"pragma synchronous={sync_mode}")

    # Other tuning options

    # Hold temporary tables in memory.
    # https://www.sqlite.org/pragma.html#pragma_temp_store
    conn.execute("pragma temp_store=memory")

    return conn


def sqlite_to_file(db, filename, overwrite=True, fmt=None):
    """Write an sqlite db to file.  Supports several output formats.

    Args:
      db (sqlite3.Connection): the sqlite3 database connection.
      filename (str): the path to the output file.
      overwrite (bool): whether an existing file should be
        overwritten.
      fmt (str): 'sqlite', 'dump', or 'gz'.  Defaults to 'sqlite'
        unless the filename ends with '.gz', in which case it is 'gz'.

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
        if os.path.exists(filename):
            os.remove(filename)
        new_db = sqlite_connect(filename=filename, mode='w')
        script = ' '.join(db.iterdump())
        new_db.executescript(script)
        new_db.commit()
    elif fmt == 'dump':
        with open(filename, 'w') as fout:
            for line in db.iterdump():
                fout.write(line)
    elif fmt == 'gz':
        with gzip.GzipFile(filename, 'wb') as fout:
            for line in db.iterdump():
                fout.write(line.encode('utf-8'))
    else:
        raise RuntimeError(f'Unknown format "{fmt}" requested.')

def sqlite_from_file(filename, fmt=None, force_new_db=True):
    """Instantiate an sqlite3.Connection and return it, with the data
    copied in from the specified file. The function can either map the database
    file directly, or map a copy of the database in memory (see force_new_db
    parameter).

    Args:
      filename (str): path to the file.
      fmt (str): format of the input; see to_file for details.
      force_new_db (bool): Used if connecting to an sqlite database. If True the
        databas is copied into memory and if False returns a connection to the
        database without reading it into memory

    """
    if fmt is None:
        fmt = 'sqlite'
        if filename.endswith('.gz'):
            fmt = 'gz'
    if fmt == 'sqlite':
        db0 = sqlite_connect(filename=filename, mode="r")
        if not force_new_db:
            return db0
        data = ' '.join(db0.iterdump())
    elif fmt == 'dump':
        with open(filename, 'r') as fin:
            data = fin.read()
    elif fmt == 'gz':
        with gzip.GzipFile(filename, 'r') as fin:
            data = fin.read().decode('utf-8')
    else:
        raise RuntimeError(f'Unknown format "{fmt}" requested.')
    db = sqlite_connect()
    db.executescript(data)
    return db


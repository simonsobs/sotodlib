import sqlite3
import gzip
import os


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
        new_db = sqlite3.connect(filename)
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
        db0 = sqlite3.connect(f'file:{filename}?mode=ro', uri=True)
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
    db = sqlite3.connect(':memory:')
    db.executescript(data)
    return db


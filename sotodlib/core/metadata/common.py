import psycopg
import gzip
import os

GET_TABLE_CREATE = """with table_info as (
                      select
                          c.column_name,
                          c.data_type,
                          c.character_maximum_length,
                          c.is_nullable,
                          tc.constraint_type,
                          tc.constraint_name
                      from
                          information_schema.columns as c
                      left join (
                          select
                              kcu.column_name,
                              tc.constraint_type,
                              tc.constraint_name,
                              kcu.table_name
                          from
                              information_schema.table_constraints as tc
                          join
                              information_schema.key_column_usage as kcu
                              on tc.constraint_name = kcu.constraint_name
                          where
                              tc.table_name = '%s'
                      ) as tc
                      on c.column_name = tc.column_name
                      where
                          c.table_name = '%s'
                  )
                  select
                      'create table your_table_name (' || string_agg(
                          column_name || ' ' || 
                          data_type || 
                          case 
                              when character_maximum_length is not null 
                                  then '(' || character_maximum_length || ')'
                              else ''
                          end || 
                          case 
                              when is_nullable = 'no' then ' not null'
                              else ''
                          end || 
                          case 
                              when constraint_type is not null then 
                                  ' constraint ' || constraint_name || ' ' || constraint_type
                              else ''
                          end,
                          ', '
                      ) || ');'
                  from
                      table_info;
"""


def dump_database(conn: psycopg.Connection) -> str:
    with conn.cursor() as cur:
        db_dump = ""
        # Fetch all table names
        cur.execute(
            "select table_name from information_schema.tables where table_schema='public'"
        )
        tables = cur.fetchall()

        for (table_name,) in tables:
            # Dump CREATE TABLE statement
            cur.execute(GET_TABLE_CREATE % (table_name, table_name))
            create_table = cur.fetchone()[0] + "\n"
            db_dump += create_table
            columns = cur.execute(
                "select column_name, data_type, character_maximum_length from "
                + "information_schema.columns where table_name = '%s';" % table_name
            ).fetchall()
            column_names = ", ".join(f"{col[0]}" for col in columns)
            # Dump data
            cur.execute(f"select {column_names} from {table_name}")
            rows = cur.fetchall()
            for row in rows:
                values = ", ".join(
                    "NULL" if value is None else f"'{str(value)}'" for value in row
                )
                db_dump += (
                    f"insert into {table_name} ({column_names}) values ({values});\n"
                )

    return db_dump


def postgres_to_file(
    db: psycopg.Connection, filename: str, overwrite: bool = True, fmt: str = None
) -> None:
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
            fmt = "dump"
    if os.path.exists(filename) and not overwrite:
        raise RuntimeError(f'File {filename} exists; remove or pass '
                           'overwrite=True.')
    if fmt == "dump":
        with open(filename, 'w') as fout:
            for line in dump_database(db):
                fout.write(line)
    elif fmt == 'gz':
        with gzip.GzipFile(filename, 'wb') as fout:
            for line in dump_database(db):
                fout.write(line)
    else:
        raise RuntimeError(f'Unknown format "{fmt}" requested.')


def postgres_from_file(filename: str, db: psycopg.Connection, fmt: str = None) -> None:
    """Instantiate an sqlite3.Connection and return it, with the data
    copied in from the specified file. The function can either map the database
    file directly, or map a copy of the database in memory (see force_new_db
    parameter).

    Args:
      filename (str): path to the file.
      db: A new DB connection.
      fmt (str): format of the input; see to_file for details.
      force_new_db (bool): Used if connecting to an sqlite database. If True the
        database is copied into memory and if False returns a connection to the
        database without reading it into memory

    """
    if fmt is None:
        fmt = "dump"
        if filename.endswith('.gz'):
            fmt = 'gz'
    if fmt == "dump":
        with open(filename, 'r') as fin:
            data = fin.readlines()
    elif fmt == 'gz':
        with gzip.GzipFile(filename, 'r') as fin:
            data = fin.readlines().decode("utf-8")
    else:
        raise RuntimeError(f'Unknown format "{fmt}" requested.')

    with db.cursor() as cursor:
        for datum in data:
            cursor.execute(datum.strip())
    db.commit()

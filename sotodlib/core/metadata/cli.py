"""Command-line interface for inspecting Context, obsdb and obsfiledb.

This program can be used to inspect the contents of ObsFileDb, ObsDb,
and ManifestDb ("metadata") sqlite3 databases.  In the case of
ObsFileDb and ManifestDb, it can also be used to perform batch updates
of filenames.

To summarize a database, pass the db type and then either the path to
the db file, or the path to the context.yaml file; some examples::

  %(prog)s obsdb /path/to/context.yaml
  %(prog)s obsfiledb /path/to/context/obsfiledb.sqlite


"""

import argparse
import os

from .. import Context
from . import ObsDb
from . import manifest


CLI_NAME = 'so-metadata'
HELP = """Read context / metadata databases and print summary or detailed
information.  You have to pass a db type (obsdb, obsfiledb, ...)
followed by the a db filename (or the path to a context.yaml, from
which the relevant db will be loaded).

"""

USAGE = "%(prog)s [db_type] [filename]"


def get_parser():
    parser = argparse.ArgumentParser(description=HELP)

    sps = parser.add_subparsers(dest='_subcmd')

    sp = sps.add_parser('obsdb', description=
                        'Inspect an ObsDb.  By default, prints out summary information.  '
                        'Pass --list to see a list of all observations; or pass an obs_id '
                        'to get detailed info about that obs.')
    sp.add_argument('db_file', help='Path to database (or context.yaml).')
    sp.add_argument('--type', choices=['context', 'db'], help=
                    'Specifies how to interpret the file (to override guessing '
                    'based on the filename extension.')
    sp.add_argument('--list', '-l', action='store_true', help=
                    'List all observations')
    sp.add_argument('--query', '-q', default='1', help=
                    'Restrict listed items with an ObsDb query string.')
    sp.add_argument('--key', '-k', action='append', default=[], help=
                    'If listing observations, also include the speficied db fields '
                    'in addition to obs_id.')
    sp.add_argument('obs_id', nargs='?')

    sp = sps.add_parser('metadata', description="Inspect (or modify) a ManifestDb.")
    manifest.get_parser(sp)

    return parser


def _set_type(args):
    if args.type is None:
        ext = os.path.splitext(args.db_file)[1]
        if ext in ['.yaml', '.yml']:
            args.type = 'context'
        elif ext in ['.sqlite', '.db']:
            args.type = 'db'
        else:
            parser.error(
                f'Not sure how to decode file with extension "{ext}"; pass --type=...')
    return args.type


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args._subcmd == 'obsdb':
        _set_type(args)

        if args.type == 'context':
            ctx = Context(args.db_file)
            db = ctx.obsdb
        elif args.type == 'db':
            db = ObsDb(args.db_file)

        # Summary mode?
        if args.obs_id is None:
            if args.list:
                rows = db.query(args.query)

                if args.list:
                    fields = ['obs_id']
                    for k in args.key:
                        fields.extend([_k.strip() for _k in k.split(',')])
                    for line in rows:
                        print(' '.join([str(line[k]) for k in fields]))
            else:
                info = db.info()
                print('  ObsDb summary')
                print('=================')
                print()
                if args.query != '1':
                    print('Note the "--query" argument is not applied to these results!')
                    print()
                print('There are %i observations' % info['count'])
                print()
                print(' Index fields (distinct values, examples)')
                print('------------------------------------------')
                for f, (count, eg_str) in info['fields'].items():
                    print(f'  {f:<15} ({count:d} distinct), e.g.: {eg_str}')
                print()
                print(' Tags defined (incidence)')
                print('--------------------------')
                for t, count in info['tags'].items():
                    print(f'  {t:<15} ({count:d} assignments)')
                print()
        else:
            print(' Observation info')
            print('-' * 60)
            item = db.get(args.obs_id)
            if item is None:
                print(f'  "{args.obs_id}" not found!')
            else:
                for k, v in db.get(args.obs_id).items():
                    print(f'  {k:<20}: {v}')

    elif args._subcmd == 'metadata':
        manifest.main(args)

    else:
        parser.error('Not implemented: {args._subcmd}')

if __name__ == '__main__':
    main()


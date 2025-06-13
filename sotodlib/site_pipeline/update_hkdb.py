"""
Command line / prefect tool for working with an hkdb.
"""


import argparse

from sotodlib.io import hkdb
from sotodlib.site_pipeline import util

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="Path to update_hkdb.yaml config file.")
    parser.add_argument('--verbose', action='store_true')

    sps = parser.add_subparsers(title="command", dest="command")

    # list
    sp = sps.add_parser('list', help="Summarize files known to the database.")
    sp.add_argument('--all', action='store_true', help=
                    "Show all records (otherwise limited to 10).")
    sp.add_argument('--compact', action='store_true', help=
                    "Show all records in compact format.")
    sp.add_argument('--status', action='append', help=
                    "Limit what types of records are shown.",
                    choices=['unindexed', 'indexed', 'failed'])

    # update
    sp = sps.add_parser('update', help="Scan for new files, and run indexing.")
    sp.add_argument('--lookback-days', type=float, help=
                    "Set the lookback time window in days; overrides any "
                    "file_idx_lookback_time setting in config file.")

    # reset-files
    sp = sps.add_parser('reset-files', help="Mark all 'failed' files as "
                        "unindexed, which will cause them to be rescanned. "
                        "Use --pattern to restrict to a subset.")
    sp.add_argument('--pattern', '-p', help=
                    "sql pattern to match with (e.g. '%_17244_%')")

    return parser


def cli_main(**kw):
    cmd = kw.get('command')
    cfg = hkdb.HkConfig.from_yaml(kw['cfg_file'])

    if kw.get('verbose'):
        util.init_logger('', logger=hkdb.log, verbosity=3)

    if cmd is None:
        print('Pass -h to get usage.')

    elif cmd == 'list':
        limit = 10
        if kw.get('all') or kw.get('compact'):
            limit = None
        index_status = kw.get('status')
        report = hkdb.get_files_info(cfg, limit=limit, index_status=index_status)
        if kw.get('compact'):
            last_r = None
            same = 0
            fmt = '  {r.path} {r.index_status}'
            for r in report:
                if last_r is None or r.index_status != last_r.index_status:
                    if same > 1:
                        print(' ... and %i more %s' % (same - 1, last_r.index_status))
                        print(fmt.format(r=last_r))
                    print(fmt.format(r=r))
                    same = 0
                else:
                    same += 1
                last_r = r
            if same > 1:
                print(' ... and %i more %s' % (same - 1, last_r.index_status))
                print(fmt.format(r=r))
        else:
            for r in report:
                print(f'{r.path} {r.index_status}')
            if limit is not None and len(report) >=  limit:
                print(' .... and possibly others (try --all or --compact)')

    elif cmd == 'update':
        if kw.get('lookback_days'):
            cfg.file_idx_lookback_time = kw['lookback_days'] * 86400
        report = hkdb.update_index_all(cfg)
        print('Report')
        for k, v in report.items():
            print(f'  {k}: {v}')

    elif cmd == 'reset-files':
        n = hkdb.reset_failed_files(cfg, pattern=kw.get('pattern'))
        print(f'Changed {n} entries.')


def simple_update(cfg_file: str):
    """Updates hkdb index databases"""
    util.init_logger('', logger=hkdb.log, verbosity=3)
    stats = hkdb.update_index_all(cfg_file)


# For prefect, map main to simple_update.
main = simple_update


if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    cli_main(**vars(args))

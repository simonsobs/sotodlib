# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""
Helper script to export a subset of SO data and copy it via rsync to the local
system.

Notes:

- You should run this from the local system (destination) where you want
  a copy of the data.

- You should set up your local ssh configuration so that you can ssh to the
  source machine without entering a password each time.  For NERSC, use sshproxy
  to get a 24-hour key and add that configuration to ~/.ssh/config.  For tiger3,
  see the documentation on confluence about using the jump host / proxy, and then
  use ssh-agent locally.  Similarly for the universe system in the UK.

- Be VERY careful if you choose to copy many observations.  Try to do that over
  an institutional network connection or someplace that is not metered for data
  usage.

- All of the databases used (the obsdb, obsfiledb, manifest dbs, etc) are fully
  copied, even though most of the data / obs referenced in those will not be
  present locally.  This is a useful feature:

    1.  You can query the DBs locally to look for observations you might want to
        sync.

    2.  You can later sync some additional data and the DBs will automatically
        still have references to past / older data from previous sync operations.

  However, the downside is that you have to pay attention to what data you have
  locally.  If you attempt to load data that does not exist, then Context.get_obs()
  will of course fail.

"""

import argparse
import atexit
import os
import re
import signal
import sqlite3
import subprocess as sp
import sys

import yaml

from ..core.util import tag_substr


class SshConnection:
    """Class representing a persistent SSH connection.

    Args:
        host (str):  The "user@host" string.
        socket (str):  The path to the ssh control socket.

    """
    def __init__(self, host, socket):
        self.socket = socket
        self.host = host

    def open(self):
        print(f"Open connection to {self.host}", flush=True)
        com = [
            "ssh",
            "-N",
            "-f",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPath='{self.socket}'",
            self.host,
        ]
        sp.run(com, check=True)

    def close(self):
        com = [
            "ssh",
            "-O",
            "exit",
            "-o",
            f"ControlPath='{self.socket}'",
            self.host,
        ]
        try:
            sp.run(com, check=True)
            print(f"Closed connection to {self.host}", flush=True)
        except Exception: #noqa
            # Socket may not exist
            pass

    def sync(self, remote_path, local_path):
        """Copy a remote file or directory to the local system.

        Args:
            remote_path (str):  The path on the remote system.
            local_path (str):  The local path.

        Returns:
            None.

        """
        local_dir = os.path.dirname(local_path)
        if local_path[-1] == "/":
            local_dir = os.path.dirname(local_dir)
        os.makedirs(local_dir, exist_ok=True)

        print(f"Syncing: {local_path}", flush=True)
        com = [
            "rsync",
            "-a",
            "--info=progress2",
            "-e",
            f"ssh -o 'ControlPath={self.socket}'",
            f"{self.host}:{remote_path}",
            local_path,
        ]
        sp.run(com, check=True)


"""Singleton for SSH connection"""
ssh_connection = None


def _signal_handler(sig, frame):
    global ssh_connection #noqa
    if ssh_connection is not None:
        ssh_connection.close()
    sys.exit(0)


@atexit.register
def _atexit_handler():
    global ssh_connection #noqa
    if ssh_connection is not None:
        ssh_connection.close()


def register_signals():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGQUIT, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    signal.signal(signal.SIGABRT, _signal_handler)
    signal.signal(signal.SIGSEGV, _signal_handler)


def load_obs_file(path):
    """Load the observation file into a list.

    Args:
        path (str):  Path to the file

    Returns:
        (list):  The list of observation IDs.

    """
    olist = []
    with open(path, "r") as f:
        for line in f:
            if re.match(r"^#.*", line) is None:
                olist.append(line.strip())
    return olist


def load_context_file(path):
    """Load a context file and apply substitutions.

    Args:
        path (str):  The path to the local context file.

    Returns:
        (dict):  The loaded context as a dictionary.

    """
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    # Do any tag substitutions
    tag_substr(conf, conf["tags"])
    return conf


def localize_file(in_path, out_path, subs):
    """Localize paths in a text file.

    This applies a list of regex expressions and replacement strings to a
    text-based file (e.g. a yaml file).

    Args:
        in_path (str):  The input file
        out_path (str):  The output file
        subs (list):  A list of expressions and replacements.

    Returns:
        None.

    """
    with open(out_path, "w") as fout, open(in_path, "r") as fin:
        for line in fin:
            for sreg, newpath in subs:
                line = sreg.sub(newpath, line)
            fout.write(line)


def localize_db(path, subs):
    """Localize paths in an sqlite database.

    This applies a list of regex expressions and replacement strings to an
    sqlite DB.  It dumps the DB to a text file, applies the path substitutions,
    and then constructs a new DB from the result.

    Args:
        path (str):  The DB file (modified in-place)
        subs (list):  A list of expressions and replacements.

    Returns:
        None.

    """
    # Temporary files
    temp_dump = f"{path}.temp_dump"
    temp_load = f"{path}.temp_load"
    temp_db = f"{path}.temp"

    if os.path.exists(temp_dump):
        os.remove(temp_dump)
    with open(temp_dump, "w") as tf:
        # Dump tables
        con = sqlite3.connect(path)
        tf.writelines([f"{x}\n" for x in con.iterdump()])
        con.close()

    # Pass through the dump file and replace paths
    localize_file(temp_dump, temp_load, subs)

    # Create new DB
    if os.path.exists(temp_db):
        os.remove(temp_db)
    with sqlite3.connect(temp_db) as newdb, open(temp_load, "r") as tmod:
        sqlcom = tmod.read()
        newdb.executescript(sqlcom)

    # Overwrite atomically
    os.replace(temp_db, path)

    # Clean up tempfiles
    os.remove(temp_load)
    os.remove(temp_dump)


def localize_context(path, subs):
    """Localize a context file in-place.

    This applies a list of regex expressions and replacement strings to a
    a context file.  A temp file is created and then moved into place.

    Args:
        path (str):  The context file, modified in-place.
        subs (list):  A list of expressions and replacements.

    Returns:
        None.

    """
    temp_ctxt = f"{path}.temp"
    localize_file(path, temp_ctxt, subs)
    os.replace(temp_ctxt, path)


def parse_context(ctxt, args):
    """Return the local and relative paths of all DBs.

    This uses the obsdb path to find the telescope name.

    Args:
        ctxt (str):  The path to the context file
        args (namespace):  The parsed commandline arguments.

    Returns:
        (tuple):  The (telescope name, obsdb path, obsfiledb path,
            and list of manifest DBs)

    """
    if "obsdb" not in ctxt:
        raise RuntimeError("Context has no obsdb key")
    obsdb = ctxt["obsdb"]
    if "obsfiledb" not in ctxt:
        raise RuntimeError("Context has no obsfiledb key")
    obsfiledb = ctxt["obsfiledb"]

    manifests = []
    for md in ctxt["metadata"]:
        if "db" not in md:
            continue
        dbpath = md["db"]
        manifests.append(dbpath)
    return obsdb, obsfiledb, manifests


def sync_manifest(db_path, args, obs_list):
    """Sync a metadata product

    The manifest DB is examined to see whether it has a mapping of obs_id
    to file path.  If not, all hdf5 files are synced.  If it does have a
    mapping, only our desired obs_ids are used to query the matching hdf5
    files to sync.

    Args:
        db_path (str):  The local path to the manifest DB.
        args (namespace):  The parsed commandline arguments.
        obs_list (list):  The list of obs_ids to sync.

    Returns:
        None

    """
    global ssh_connection #noqa

    rel_db = os.path.relpath(db_path, args.local_metadata)
    remote_db = os.path.join(args.remote_metadata, rel_db)

    # The metadata directory containing this DB
    local_mdir = os.path.dirname(db_path)
    remote_mdir = os.path.dirname(remote_db)

    # If the `map` table has the obs:obs_id column, then select the files
    # we want to copy.  Otherwise, copy the whole metadata product.

    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    have_obsid = False
    for row in cursor.fetchall():
        table = row[0]
        if table != "map":
            continue
        col_info = con.execute(f"PRAGMA table_info('{table}');").fetchall()
        for col in col_info:
            col_name = col[1]
            if col_name == "obs:obs_id":
                have_obsid = True
    con.close()

    if have_obsid:
        obs_str = ",".join([f"'{x}'" for x in obs_list])
        obs_str = f"({obs_str})"
        mfiles = []
        con = sqlite3.connect(db_path)
        cursor = con.cursor()
        cursor.execute(f"select name from files inner join map on files.id = map.file_id where \"obs:obs_id\" in {obs_str};")
        for row in cursor.fetchall():
            mfiles.append(row[0])
        mset = set(mfiles)
        mfiles = list(sorted(mset))
        print(f"Meta: {rel_db}: obs {obs_str} has meta files {mfiles}", flush=True)
        for mf in mfiles:
            ssh_connection.sync(f"{remote_mdir}/{mf}", f"{local_mdir}/")
        con.close()
    else:
        print(f"Meta: {rel_db}: sync full metadata product", flush=True)
        ssh_connection.sync(f"{remote_mdir}/*.h5", f"{local_mdir}/")


def sync_obs_g3book(obs, args):
    """Sync a single observation book.

    Args:
        obs (str):  The obs_id
        args (namespace):  The parsed commandline arguments.

    Returns:
        None.

    """
    global ssh_connection #noqa

    # Get the first 5 digits of the obs time
    mat = re.match(r"obs_(\d\d\d\d\d)\d+_([0-9a-z]+)_.*", obs)
    if mat is None:
        raise RuntimeError(f"obs {obs} does not have expected format")
    firstfive = mat.group(1)
    tele_name = mat.group(2)
    rel_path = os.path.join(tele_name, "obs", firstfive, obs)
    remote = os.path.join(args.remote_data, rel_path)
    local = os.path.join(args.local_data, rel_path)
    ssh_connection.sync(f"{remote}/", f"{local}/")


def main():
    parser = argparse.ArgumentParser(
        description="Copy observation data and metadata to the local system"
    )
    parser.add_argument(
        "--local_data",
        required=True,
        default=None,
        help="The local data directory.  Assumes <telescope> subdir.",
    )
    parser.add_argument(
        "--local_metadata",
        required=True,
        default=None,
        help="The local metadata directory.  Assumes <telescope>/manifests subdir.",
    )
    parser.add_argument(
        "--local_context",
        required=True,
        default=None,
        help="The local context file to create",
    )
    parser.add_argument(
        "--remote_data",
        required=True,
        default=None,
        help="The remote data directory.  Assumes <telescope> subdir.",
    )
    parser.add_argument(
        "--remote_metadata",
        required=True,
        default=None,
        help="The remote metadata directory.  Assumes <telescope>/manifests subdir.",
    )
    parser.add_argument(
        "--remote_context",
        required=True,
        default=None,
        help="The context file path on the remote system",
    )
    parser.add_argument(
        "--remote_host",
        required=True,
        default=None,
        help="The address of the remote host (user@remote.domain.name)",
    )
    parser.add_argument(
        "--obs_file",
        required=True,
        default=None,
        help="A text file with the obs_ids to sync",
    )
    parser.add_argument(
        "--ssh_control",
        required=False,
        default="~/.ssh/so_export.sock",
        help="The file for the persistent ssh control socket",
    )
    parser.add_argument(
        "--localize",
        default=False,
        action="store_true",
        help="If enabled, localize databases",
    )

    args = parser.parse_args()

    # Load observations
    obs_list = load_obs_file(args.obs_file)

    # Build the list of substitutions
    remote_data = args.remote_data.replace("/", r"\/")
    remote_metadata = args.remote_metadata.replace("/", r"\/")
    subs = [
        (re.compile(remote_data), args.local_data),
        (re.compile(remote_metadata), args.local_metadata),
    ]

    # Create the persistent ssh connection
    global ssh_connection
    ssh_connection = SshConnection(args.remote_host, args.ssh_control)

    # Register signal handlers so that the ssh connection is cleaned up,
    # even if the program crashes.
    register_signals()

    try:
        # Start up master ssh control
        ssh_connection.open()

        # The remote context file
        print(f"Using context file {args.remote_context}", flush=True)

        # Fetch the context file and localize
        ssh_connection.sync(args.remote_context, args.local_context)
        if args.localize:
            localize_context(args.local_context, subs)

        # Load the context file and find all the databases used.
        ctxt = load_context_file(args.local_context)
        obsdb, obsfiledb, manifests = parse_context(ctxt, args)

        # Sync all databases used in the context
        all_dbs = [obsdb, obsfiledb]
        all_dbs.extend(manifests)
        for local_db in all_dbs:
            rel = os.path.relpath(local_db, args.local_metadata)
            remote = os.path.join(args.remote_metadata, rel)
            ssh_connection.sync(remote, local_db)
            if args.localize:
                localize_db(local_db, subs)

        # Open each manifest DB and get the files matching our obs list.
        # Sync these metadata files.
        for mdb in manifests:
            sync_manifest(mdb, args, obs_list)

        # We want to sync the full books, not just the g3 files.
        for obs in obs_list:
            sync_obs_g3book(obs, args)

    except Exception: #noqa
        pass


if __name__ == "__main__":
    main()

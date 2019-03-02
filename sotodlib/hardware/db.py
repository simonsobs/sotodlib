# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for working with hardware databases.
"""

import os

from collections import OrderedDict

from contextlib import contextmanager

import numpy as np

import sqlite3

#from sotoddb import DetDB


# NOTE:  This source is not currently used.  It is kept here in case we revisit
# the use of databases.


class DataBase(object):
    """Class representing a hardware configuration database.

    This class provides an interface to an sqlite DB supported by the
    sotoddb package.  In addition to detector property tables, other tables
    are used to represent hardware features.

    Args:
        path (str, optional): Path to the DB file.  If the file does not
            exist, it is created and opened in write mode.  If it does exist,
            it is opened in the mode specified.
        mode (str, optional): The mode ("r" or "w") to use when opening the
            DB.  If not specified, the default is "r" if the file exists
            else "w" if it is being created.  In-memory DBs are always "w".
        conf (dict, optional): If the database is being created, this config
            dictionary is used to create auxilliary tables.
        dets (dict, optional): If the database is being created, this detector
            dictionary is used to create the detector property tables.

    """
    def __init__(self, path, mode=None, conf=None, dets=None):
        self._path = path
        self._mode = mode

        create = True
        if os.path.exists(self._path):
            create = False

        if self._mode == 'r' and create:
            raise RuntimeError("cannot open a non-existent DB in read-only "
                               " mode")

        self._connstr = None

        # This timeout is in seconds
        self._busytime = 1000

        # Journaling options
        self._journalmode = "persist"
        self._syncmode = "normal"

        if create:
            self.initdb(conf, dets)
        return

    def _open(self):
        try:
            # only python3 supports uri option
            if self._mode == 'r':
                self._connstr = 'file:{}?mode=ro'.format(self._path)
            else:
                self._connstr = 'file:{}?mode=rwc'.format(self._path)
            self._conn = sqlite3.connect(self._connstr, uri=True,
                                         timeout=self._busytime)
        except sqlite3.OperationalError:
            self._conn = sqlite3.connect(self._path,
                                         timeout=self._busytime)
        if self._mode == 'w':
            # In read-write mode, set the journaling
            self._conn.execute("pragma journal_mode={}"
                               .format(self._journalmode))
            self._conn.execute("pragma synchronous={}".format(self._syncmode))
            # Other tuning options
            self._conn.execute("pragma temp_store=memory")
            self._conn.execute("pragma page_size=4096")
            self._conn.execute("pragma cache_size=4000")
        return

    def _close(self):
        del self._conn
        self._conn = None
        return

    @contextmanager
    def cursor(self):
        self._open()
        cur = self._conn.cursor()
        cur.execute("begin transaction")
        try:
            yield cur
        except sqlite3.DatabaseError as err:
            cur.execute("rollback")
            raise err
        else:
            try:
                cur.execute("commit")
            except sqlite3.OperationalError:
                # sqlite3 in py3.5 can't commit a read-only finished
                # transaction.
                pass
        finally:
            del cur
            self._close()


    def initdb(self, conf, dets):
        """Initialize the database.

        We use sotoddb to create the database and populate the detector
        tables.  Then we directly create the other tables.

        Args:
            conf (dict): The hardware config dictionary used to create
                auxilliary tables.
            dets (dict): The detector dictionary used to create the detector
                property tables.

        """
        # Get the main detector schema from the first entry
        dkeys = list(dets.keys())
        dprops = dets[dkeys[0]]
        dschema = [
            "`det_id` integer",
            "`time0` integer",
            "`time1` integer",
            "`name` text"
        ]
        for k, v in dprops.items():
            if k == "quat":
                # We keep the pointing offsets in a separate table.
                continue
            coltype = "text"
            if isinstance(v, float):
                coltype = "float"
            elif isinstance(v, int):
                coltype = "integer"
            colstr = "`{}` {}".format(k, coltype)
            dschema.append(colstr)
        dqschema = [
            "`det_id` integer",
            "`time0` integer",
            "`time1` integer",
            "`qx` float",
            "`qy` float",
            "`qz` float",
            "`qw` float"
        ]

        # Create the DB and detector tables, then close.
        sodb = DetDB(map_file=self._path, init_db=True)
        sodb.create_table("detprops", dschema)
        sodb.create_table("detgeom", dqschema)
        del sodb

        # Now go and create our other tables and populate everything.

        # Go through the hardware config and grab the first row of each
        # dictionary to create the schema.
        for table, props in conf.items():
            pkeys = list(props.keys())
            row = pkeys[0]
            colprops = props[row]
            createstr = "create table {} (name text unique".format(table)
            for k, v in colprops.items():
                coltype = "text"
                if isinstance(v, float):
                    coltype = "float"
                elif isinstance(v, int):
                    coltype = "integer"
                createstr = "{}, {} {}".format(createstr, k, coltype)
            createstr = "{})".format(createstr)
            print(createstr, flush=True)
            with self.cursor() as cur:
                cur.execute(createstr)

        # Now go back and populate the hardware config tables.  One transaction
        # per table.
        for table, props in conf.items():
            with self.cursor() as cur:
                for name, prp in props.items():
                    colstr = "(name"
                    valstr = "('{}'".format(name)
                    for k, v in prp.items():
                        colstr += ", {}".format(k)
                        if isinstance(v, (float, int)):
                            valstr += ", {}".format(v)
                        else:
                            valstr += ", '{}'".format(v)
                    colstr += ")"
                    valstr += ")"
                    com = "insert into {} {} values {}".format(
                        table, colstr, valstr)
                    print(com, flush=True)
                    cur.execute(com)

        # Now populate the detector tables
        with self.cursor() as cur:
            for det, props in dets.items():
                detid = props["ID"]
                pcol = "(det_id, name"
                pval = "({}, '{}'".format(detid, det)
                gcol = "(det_id, qx, qy, qz, qw)"
                gval = "({}".format(detid)
                for k, v in props.items():
                    if k == "quat":
                        for i in range(4):
                            gval += ", {}".format(v[i])
                        gval += ")"
                    else:
                        pcol += ", {}".format(k)
                        if isinstance(v, (float, int)):
                            pval += ", {}".format(v)
                        else:
                            pval += ", '{}'".format(v)
                pcol += ")"
                pval += ")"
                com = "insert into detprops {} values {}".format(pcol, pval)
                print(com, flush=True)
                cur.execute(com)
                com = "insert into detgeom {} values {}".format(gcol, gval)
                print(com, flush=True)
                cur.execute(com)
        return

import h5py
import numpy as np
import sqlite3
from glob import glob
import os
import warnings

defaults = {"odir": "output",
            "atomic_db_path": "atomic_maps.db",
            "delete": False,
           }

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--odir", help='Output directory')
    parser.add_argument(
        "--atomic_db_path", help='Path of the atomic map database')
    parser.add_argument(
        "--delete", action="store_true",
        help='Delete the hdf files after reading them'
    )
    return parser

def make_db_from_outdir(odir, atomic_db_path):
    """Make an atomic db from all atomics in a directory and its subdirectories"""
    file_list = glob(os.path.join(odir, '**/*_info.hdf'), recursive=True)
    make_db(file_list, atomic_db_path)
    return read_db(atomic_db_path)
    
def make_db(info_file_list, atomic_db_path):
    if len(info_file_list) == 0:
        raise RuntimeError("Cannot make db from empty file list")
    conn = sqlite3.connect(atomic_db_path)
    cursor = conn.cursor()
    
    entries = [('obs_id', 'TEXT'), ('telescope', 'TEXT'), ('freq_channel', 'TEXT'), ('wafer', 'TEXT'), ('ctime', 'INTEGER'), ('split_label', 'TEXT'),
               ('split_detail', 'TEXT'), ('prefix_path', 'TEXT'), ('elevation', 'REAL'), ('azimuth', 'REAL'), ('pwv', 'REAL'), ('total_weight_qu', 'REAL'), ('median_weight_qu', 'REAL'), ('mean_weight_qu', 'REAL')]

    cmd = f"CREATE TABLE IF NOT EXISTS atomic ({', '.join([' '.join(tup) for tup in entries])})"
    cursor.execute(cmd)
    
    conn.commit()
    for info_file in info_file_list:
        info = parse_info(info_file)
        info_tuple = info_dict_to_tuple(info, entries)

        cursor.execute(f"INSERT INTO atomic VALUES ({', '.join(['?']*len(info_tuple))})", info_tuple)

    conn.commit()    
    conn.close()

def read_db(db_filename, *args):
    """Read from a db. 
    String args can be passed to make a query eg 'obs_id="xxx"'
    """
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    query = 'SELECT * from atomic'
    if len(args) > 0:
        query += ' where ' + " and ".join(args)
    res = cursor.execute(query)
    matches = res.fetchall()
    return matches
    
def load_h5(fn):
    if isinstance(fn, h5py._hl.files.File):
        return fn
    elif isinstance(fn, str):
        return h5py.File(fn, 'r')
    else:
        raise TypeError(f"Invalid type for hdf5 file(name) {type(fn)}")

def parse_info(h5):
    h5 = load_h5(h5)
    out = {}
    for key in h5.keys():
        item = np.squeeze(np.asarray(h5[key]))
        if item.ndim == 0:
            item = item[()] # Extract scalars
        if isinstance(item, bytes):
            item = item.decode('UTF-8') # Convert bytes to strings
        out[key] = item
    return out

def info_dict_to_tuple(dct, entry_list):
    out = []
    dtype_dict = {'TEXT':str, 'INTEGER':int, 'REAL':float}
    
    for entry in entry_list:
        dtype = dtype_dict[entry[1]]
        out.append(dtype(dct[entry[0]]))
    return tuple(out)

def main(defaults=defaults, **args):
    import sys
    cfg = dict(defaults)
    cfg.update({k: v for k, v in args.items() if v is not None})
    args = cfg
    warnings.simplefilter('ignore')
    make_db_from_outdir(args['odir'], args['atomic_db_path'])

if __name__ == '__main__':
    util.main_launcher(main, get_parser)

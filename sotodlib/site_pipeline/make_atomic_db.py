from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging, yaml, sqlite3, itertools, glob
from pixell import bunch
from . import util

defaults = {"odir": "./output",
            "atomic_db": "atomic_maps.db",
            "delete": False,
            }

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None, 
                     help="Path to atomic DB config.yaml file")
    parser.add_argument("--odir",
                        help='output directory')
    parser.add_argument("--atomic_db",
                        help='name of the atomic map database')
    parser.add_argument("--delete",
                        action="store_true",
                        help='delete the h5 files after reading them'
                       )
    return parser

def _get_config(config_file):
    return yaml.safe_load(open(config_file,'r'))

def main(config_file=None, defaults=defaults, **args):
    cfg = dict(defaults)
    # Update the default dict with values provided from a config.yaml file
    if config_file is not None:
        cfg_from_file = _get_config(config_file)
        cfg.update({k: v for k, v in cfg_from_file.items() if v is not None})
    else:
        print("No config file provided, assuming default values") 
    # Merge flags from config file and defaults with any passed through CLI
    cfg.update({k: v for k, v in args.items() if v is not None})
    # Certain fields are required. Check if they are all supplied here
    args = cfg
    warnings.simplefilter('ignore')
    
    # check if output exists
    if not(os.path.exists(args['odir'])):
        print('output does not exist, exiting')
        return True
    # Open a DB, if it doesn't exist it will be created
    # Write into the atomic map database.
    conn = sqlite3.connect('./'+args['atomic_db']) # open the conector, if the database exists then it will be opened, otherwise it will be created
    cursor = conn.cursor()
        
    # Check if the table exists, if not create it
    # the tags will be telescope, frequency channel, wafer, ctime, split_label, split_details, prefix_path, elevation, pwv
    cursor.execute("""CREATE TABLE IF NOT EXISTS atomic (
                      obs_id TEXT,
                      telescope TEXT, 
                      freq_channel TEXT, 
                      wafer TEXT, 
                      ctime INTEGER,
                      split_label TEXT,
                      split_detail TEXT,
                      prefix_path TEXT,
                      elevation REAL,
                      azimuth REAL,
                      RA_ref_start REAL,
                      RA_ref_stop REAL,
                      pwv REAL,
                      PRIMARY KEY(obs_id, freq_channel, wafer, split_label)
                      )""")
    conn.commit()
    
    # make a list of all *.hdf files
    files = glob.glob('%s/*/*.hdf'%args['odir'])
    list_ = []
    INSERT_COMMAND = """
    INSERT INTO atomic (obs_id, telescope, freq_channel, wafer, ctime, split_label, split_detail, prefix_path, elevation, azimuth, RA_ref_start, RA_ref_stop, pwv)
    VALUES ( :obs_id, :telescope, :freq_channel, :wafer, :ctime, :split_label, :split_detail, :prefix_path, :elevation, :azimuth, :RA_ref_start, :RA_ref_stop, :pwv);
    """
    for file in files:
        f = bunch.read(file)
        tuple_ = {"obs_id": f['obs_id'].decode(), 
                  "telescope": f['telescope'].decode(), 
                  "freq_channel": f['freq_channel'].decode(), 
                  "wafer": f['wafer'].decode(), 
                  "ctime":  float(f['ctime']), 
                  "split_label": f['split_label'].decode(), 
                  "split_detail": f['split_detail'].decode(), 
                  "prefix_path": f['prefix_path'].decode(), 
                  "elevation": f['elevation'], 
                  "azimuth": f['azimuth'], 
                  "RA_ref_start": f['RA_ref_start'], 
                  "RA_ref_stop": f['RA_ref_stop'], 
                  "pwv": float(f['pwv'])}
        list_.append(tuple_)
        if args['delete']:
            os.remove(file) # delete the files
    cursor.executemany(INSERT_COMMAND, list_)
    conn.commit()    
    conn.close()

    print("Done")
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)

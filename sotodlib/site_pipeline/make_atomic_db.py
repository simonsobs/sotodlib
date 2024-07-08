from argparse import ArgumentParser
import numpy as np, sys, time, warnings, os, so3g, logging, yaml, sqlite3, itertools, glob
from pixell import bunch
from . import util

defaults = {"odir": "./output",
            "atomic_db": "atomic_maps.db",
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
                      pwv REAL
                      )""")
    conn.commit()
    
    # make a list of all *.hdf files
    files = glob.glob('%s/*/*.hdf'%args['odir'])
    for file in files:
        f = bunch.read(file)
        tuple_ = (f['obs_id'].decode(), 
                  f['telescope'].decode(), 
                  f['freq_channel'].decode(), 
                  f['wafer'].decode(), 
                  float(f['ctime']), 
                  f['split_label'].decode(), 
                  f['split_detail'].decode(), 
                  f['prefix_path'].decode(), 
                  f['elevation'], 
                  f['azimuth'], 
                  f['RA_ref_start'], 
                  f['RA_ref_stop'], 
                  float(f['pwv']))    
        cursor.execute("INSERT INTO atomic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple_)    
    conn.commit()    
    conn.close()

    print("Done")
    return True

if __name__ == '__main__':
    util.main_launcher(main, get_parser)

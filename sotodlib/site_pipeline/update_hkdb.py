"""
Script for calling hkdb update function with a configuration file
"""
from sotodlib.io import hkdb
from sotodlib.site_pipeline import util

def main(cfg_file: str):
    """Updates hkdb index databases"""
    util.init_logger('', logger=hkdb.log)
    hkdb.update_index_all(cfg_file)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])

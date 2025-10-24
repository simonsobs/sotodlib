"""
This submodule contains classes to coadd atomic maps produced by the
make_atomic_filterbin_map site-pipeline script. The CoaddMapmaker class
creates a mapmaker object. Provided the atomic maps and related database,
this will coadd observations within some time interval. This is primarily meant
to be used by the make_coadd_atomic_map site-pipeline script, which calls
make_coadd_map.
"""
__all__ = ['CoaddMapmaker','make_coadd_map','setup_coadd_map','write_coadd_map']
import numpy as np
from pixell import enmap
import sqlite3
import os
from tqdm import tqdm
import datetime as dt
import logging

from . import utils

class CoaddMapmaker:
    """
    Class to initialize, create, and write coadded atomic maps
    
    Arguments
    ---------
    data : dict
        Data returned from atomic database query.
    shape : numpy.ndarray
        Shape of the output map geometry.
    wcs : wcs
        WCS of the output map geometry.
    geom_file_path : str
        Full path to geometry file to use.
    wmap_coadd : pixell.enmap
        Initial wmap to be populated
    weights_coadd : pixell.enmap
        Initial weights to be populated
    hits_coadd : pixell.enmap
        Initial hits to be populated
    logger : Logger, optional
        Logger for printing on the screen.
    """
    def __init__(self, data, shape, wcs, geom_file_path, 
                 wmap_coadd, weights_coadd, hits_coadd, logger=None):
        self.data = data
        self.shape = shape
        self.wcs = wcs
        self.geom_file_path = geom_file_path
        self.wmap_coadd = wmap_coadd
        self.weights_coadd = weights_coadd
        self.hits_coadd = hits_coadd
        self.map_coadd = None
        
        if logger is None:
            logger = logging.getLogger("coadd")
        self.logger = logger
        
    def add_map(self, file):
        """
        Accumulates an obs into the CoaddMapmaker object, adding to wmap,
        weights, and hits.
        
        Arguments
        ---------
        file : str
            Path to atomic map file.
        """
        file = os.path.normpath(file)
        wmap = enmap.read_map(file+'_wmap.fits', geometry=(self.shape,self.wcs))
        weights = enmap.read_map(file+'_weights.fits', geometry=(self.shape,self.wcs))
        hits = enmap.read_map(file+'_hits.fits', geometry=(self.shape,self.wcs))

        self.wmap_coadd += wmap
        for i in range(3):
            self.weights_coadd[i,i] += weights[i]
        self.hits_coadd += hits
            
    def write(self, output_root, output_db, band, platform, split_label, start_time, 
              stop_time, interval, unit='K'):
        time_str = f"{start_time:%Y%m%d}_{stop_time:%Y%m%d}"
        if isinstance(start_time, dt.datetime):
            start_time = start_time.timestamp()
        if isinstance(stop_time, dt.datetime):
            stop_time = stop_time.timestamp()
        oname = f'{output_root}/{interval}/coadd_{time_str}_{band}_{split_label}'
        self.logger.debug(f'output path prefix: {oname}')

        enmap.write_map(oname+'_map.fits', self.map_coadd, extra={'BUNIT':unit})
        enmap.write_map(oname+'_weights.fits', self.weights_coadd, extra={'BUNIT':unit})
        enmap.write_map(oname+'_hits.fits', self.hits_coadd)
        self.logger.info(f'wrote map to {oname}'+'_map.fits')
        self.logger.info(f'wrote weights to {oname}'+'_weights.fits')
        self.logger.info(f'wrote hits to {oname}'+'_hits.fits')

        obslist = ','.join(np.unique(self.data['obs_id']))
        conn = sqlite3.connect(output_db)
        cur = conn.cursor()
        col_names = ['telescope', 'freq_channel', 'split_label', 'prefix_path', 'geom_file_path', 'obslist', 'start_time', 'stop_time']
        placeholders = ','.join(['?'] * len(col_names))
        row_values = [platform, band, split_label, oname, self.geom_file_path, obslist, start_time, stop_time]
        insert_stmt = f'INSERT OR REPLACE INTO {interval} ({",".join(col_names)}) VALUES ({placeholders})'
        cur.execute(insert_stmt, row_values)
        conn.commit()
        conn.close()
        self.logger.info(f'added entry to {interval} table in {output_db}')
        
def setup_coadd_map(atomic_db, output_db, band, platform, split_label,
                    start_time, stop_time, interval, geom_file_prefix, 
                    overwrite=False, logger=None):
    """
    Queries the atomic database and setup the CoaddMapmaker object.
    """
    columns = ['obs_id', 'telescope', 'freq_channel', 'ctime', 'split_label', 'prefix_path']
    if isinstance(start_time, dt.datetime):
        start_time = start_time.timestamp()
    if isinstance(stop_time, dt.datetime):
        stop_time = stop_time.timestamp()
        
    if not overwrite:
        conn = sqlite3.connect(output_db)
        cur = conn.cursor()
        query_stmt = f"SELECT * FROM {interval} WHERE telescope = ? AND freq_channel = ? AND split_label = ? AND start_time = ? AND stop_time = ?"
        cur.execute(query_stmt, (platform,band,split_label,start_time,stop_time,))
        rows = cur.fetchall()
        conn.close()
        if len(rows) > 0:
            return False

    conn = sqlite3.connect(atomic_db)
    cur = conn.cursor()
    if '+' in band:
        query_stmt = f"SELECT {','.join(columns)} FROM atomic WHERE telescope = ? AND split_label = ? AND ctime >= ? AND ctime <= ?"
        cur.execute(query_stmt, (platform,split_label,start_time,stop_time,))
    else:
        query_stmt = f"SELECT {','.join(columns)} FROM atomic WHERE telescope = ? AND split_label = ? AND ctime >= ? AND ctime <= ? AND freq_channel = ?"
        cur.execute(query_stmt, (platform,split_label,start_time,stop_time,band,))
    rows = cur.fetchall()
    conn.close()

    data = {}
    for col in columns:
        data[col] = []
    for j, row in enumerate(rows):
        for k, item in enumerate(row):
            data[columns[k]].append(item)

    if '+' in band:
        band = band.split('+')[0]
    geom_file_path = geom_file_prefix+f'_{band}.fits'
    shape, wcs = enmap.read_map_geometry(geom_file_path)
    wmap_coadd = enmap.zeros((3,)+shape, wcs=wcs)
    weights_coadd = enmap.zeros((3,3,)+shape, wcs=wcs)
    hits_coadd = enmap.zeros((3,)+shape, wcs=wcs)
    
    mapmaker = CoaddMapmaker(data, shape, wcs, geom_file_path, 
                             wmap_coadd, weights_coadd, hits_coadd, logger=logger)
    
    return mapmaker
    
def write_coadd_map(mapmaker, output_root, output_db, band, platform, split_label, 
                    start_time, stop_time, interval, unit='K'):
    """
    Wrapper for CoaddMapmaker write function.
    """
    mapmaker.write(output_root, output_db, band, platform, split_label, start_time, 
                   stop_time, interval, unit=unit)
        
def make_coadd_map(atomic_db, output_root, output_db, band, platform, split_label,
                   start_time, stop_time, interval, geom_file_prefix, overwrite=False,
                   unit='K', logger=None):
    """
    Makes coadded atomic maps from a given time interval.
    
    Arguments
    ---------
    atomic_db : str
        Path to input atomic maps database.
    output_root : str
        Path to directory for output map files.
    output_db : str
        Path to output coadded maps database.
    band : str
        Band to coadd.
        Examples: ["f090", "f150", "f090+f150"]
    platform : str
        Telescope platform.
    split_label : str
        Split label from atomic map files.
    start_time : dt.datetime
        Start time for querying atomic maps.
    stop_time : dt.datetime
        Stop time for querying atomic maps.
    interval : str
        Interval to group atomic maps for coadding.
        Examples: ["daily", "weekly", "monthly"]
    geom_file_prefix : str
        Prefix path to geometry file, omitting the band.
    overwrite : bool
        Set to True to re-run coadding and overwrite database if time interval
        found in database.
    unit : str
        Temperature unit.
    logger : Logger, optional
        Logger for printing on the screen.
    """
    mapmaker = setup_coadd_map(atomic_db, output_db, band, platform, 
                               split_label, start_time, stop_time, interval, 
                               geom_file_prefix, overwrite=overwrite, logger=logger)
    
    if not mapmaker:
        return False

    mapmaker.logger.info(f"# of fits files: {len(mapmaker.data['prefix_path'])}")
    mapmaker.logger.info(f"Using geometry from {mapmaker.geom_file_path}")
    for file in tqdm(mapmaker.data['prefix_path'], total=len(mapmaker.data['prefix_path'])):
        mapmaker.add_map(file)

    iweights = utils.safe_invert_div(mapmaker.weights_coadd)
    mapmaker.map_coadd = enmap.map_mul(iweights, mapmaker.wmap_coadd)

    write_coadd_map(mapmaker, output_root, output_db, band, platform, split_label, 
                    start_time, stop_time, interval, unit=unit)

    
    return True

# -*- coding: utf-8 -*-
"""
To extract sky to secondary focus and secondary focus to array data from ID9 
locations are taken to be where the central ray crosses a surface
Also hardcoded is a dictionary between physical design and tube data.
Requires zemax to be running or set get_data to be False and 
@author: sdicker
"""
import numpy as np
import logging
from scipy.interpolate import interp2d

logger = logging.getLogger(__name__)

LAT_FOV=3.7 # LAT FOV in degrees
'''
Dictionary of zemax tube layout.
+ve x is to the right as seen from back of the cryostat *need to check*
          12
   6    4    5    7
      2    1    3
  10    8    9    11
          13
Below config assumes a 30 degree rotation
'''
LAT_TUBES={'c':1,'i1':4,'i2':2,'i3':8,'i4':9,'i5':3,'i6':5, 'o1':6,'o2':10,'o3':13,'o4':11,'o5':7,'o6':12}

def pix2sky(x,y,tube,rot=0,opt2cryo=0.):
    '''Routine to map pixels from arrays to sky
    x,y = position on focal plane (currently zemax coord)
    tube = which tube - integer 1 to 13, can look up names with tube_mapping
    rot = co-rotator position in degrees wrt elevation (TBD sort out where zero is)
    opt2cryo = the rotation to get from cryostat coordinates to zemax coordinates (TBD, prob 30 deg)
    Curretly uses global variables....'''
    d2r=np.pi/180.
    #TBD - put in check for MASK - values outside circle should not be allowed
    xz=x*np.cos(d2r*opt2cryo) - y*np.sin(d2r*opt2cryo) #get into zemax coord
    yz=y*np.cos(d2r*opt2cryo) + x*np.sin(d2r*opt2cryo)
    xs=array2secx[tube-1](xz,yz)# Where is it on (zemax secondary focal plane wrt LATR)
    ys=array2secy[tube-1](xz,yz)
    rot2=rot #may need to add offset here to account for physical vs ZEMAX
    xrot=xs*np.cos(d2r*rot2) - ys*np.sin(d2r*rot2) #get into LAT zemax coord
    yrot=ys*np.cos(d2r*rot2) + xs*np.sin(d2r*rot2)
    elev=sec2elev(xrot,yrot) # note these are around the telescope boresight
    xel=sec2xel(xrot,yrot)
    return elev,xel

def LAT_optics(zemax_dat):
    if type(zemax_dat) is str:
        try: 
            zemax_dat = np.load(zemax_dat, allow_pickle=True)
        except Exception as e:
            logger.error("Can't load data from " + zemax_dat)
            raise e
    elif type(zemax_dat) is not dict:
        logger.error("zemax_dat should either be a path or a dictionary")
        raise TypeError
    try:
        LAT=zemax_dat['LAT'][()]
    except Exception as e:
        logger.error("LAT key missing from dictionary")
        raise e
    
    gi=np.where(LAT['mask'] != 0.)
    sec2elev = interp2d(LAT['x'][gi],LAT['y'][gi],LAT['elev'][gi],bounds_error=True)
    sec2xel  = interp2d(LAT['x'][gi],LAT['y'][gi],LAT['xel'][gi],bounds_error=True)

    return sec2elev, sec2xel

def LATR_optics(zemax_dat, tube):
    if type(zemax_dat) is str:
        try: 
            zemax_dat = np.load(zemax_dat, allow_pickle=True)
        except Exception as e:
            logger.error("Can't load data from " + zemax_dat)
            raise e
    elif type(zemax_dat) is not dict:
        logger.error("zemax_dat should either be a path or a dictionary")
        raise TypeError
    try:
        LATR=zemax_dat['LATR'][()]
    except Exception as e:
        logger.error("LATR key missing from dictionary")
        raise e
    
    if type(tube) is str:
        tube_name = tube
        try:
            tube_num = LAT_TUBES[tube]
        except Exception as e:
            logger.error("Invalid tube name")
            raise e
    elif type(tube) is int:
        tube_num = tube
        try:
            tube_name = list(LAT_TUBES.keys())[tube_num]
        except Exception as e:
            logger.error("Invalid tube number")
            raise e

    logger.info('Working on LAT tube ' + tube_name)
    gi=np.where(LATR[tube_num]['mask'] != 0)
    array2secx = interp2d(LATR[tube_num]['array_x'][gi],LATR[tube_num]['array_y'][gi],LATR[tube_num]['sec_x'][gi],bounds_error=True)
    array2secy = interp2d(LATR[tube_num]['array_x'][gi],LATR[tube_num]['array_y'][gi],LATR[tube_num]['sec_y'][gi],bounds_error=True)

    return array2secx, array2secy

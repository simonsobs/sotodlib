# -*- coding: utf-8 -*-
"""
To extract sky to secondary focus and secondary focus to array data from ID9 
locations are taken to be where the central ray crosses a surface
Also hardcoded is a dictionary between physical design and tube data.
Requires zemax to be running or set get_data to be False and 
@author: sdicker
"""
get_data=False #True # set this if you need to connect to zemax and fit for the mapping
if get_data : import pyzdde.zdde as pyz
import numpy as np

#just the telescope
telescope_file='C://users/sdicker/Documents/'
#id9 (or similar) design file with all 13 tubes
#set dir to the path were the zemax and or the saved data file are
# dir='C://users/sdicker/Documents/from_medusa/sdicker/Documents/Simons/optics/CCAT_Richard_hills/ID9_final/'
dir='./'
id9='ID9_checked.zmx' #name of the zemax file - results saved as <filebasename>_trace_data.npz
fov=3.7 #telescope fov in degrees


def get_tube_dict():
    '''returns dictionary of tube layout, +ve x is to the right as seen from 
    back of the cryostat *need to check*
    #zemax tube layout 
    #          12
    #   6    4    5    7
    #      2    1    3
    #  10    8    9    11
    #          13
    below config assumes a 30 degree rotation'''
    tube2config={'c':1,'i1':4,'i2':2,'i3':8,'i4':9,'i5':3,'i6':5,
                     'o1':6,'o2':10,'o3':13,'o4':11,'o5':7,'o6':12}
    return tube2config

def get_points4telescope(lnk,fov,ds=0.1,secondary_surf=1):
    '''get arrays that map sky to telescope focal plane'''
    mode=0 
    n=int(2*fov//ds) #make it even so linspace is odd
    #print(n,fov,ds,fov//ds)
    xy=np.linspace(-1,1,n)
    sky_x,sky_y=np.meshgrid(xy,xy)#sky y = xelev sky_x=elevation
    sec_x=np.zeros(sky_x.shape) 
    sec_y=np.zeros(sky_x.shape) #y=+ve = in direction of beam
    mask=np.zeros(sky_x.shape) #where data is outside circular fov
    for x in range(sky_x.shape[0]) :
        for y in range(sky_x.shape[1]) :
            if sky_x[x,y]**2+sky_y[x,y]**2 < 1. :
                trace=lnk.zGetTrace(0,mode,secondary_surf,sky_x[x,y],sky_y[x,y],0,0)
                sec_x[x,y]=trace.x
                sec_y[x,y]=trace.y
                mask[x,y]=1.
                #print(mode,secondary_surf,sky_x[x,y],sky_y[x,y],trace.x,trace.y)
                #exit()
    return (sky_x,sky_y),(sec_x,sec_y),mask

def get_pointsLATR(lnk,numPnt,secondary_surf=1,array=-1):
    '''get numbers that map seconary focus onto detector arrays
    Note calculations are done in ZEMAX models x,y'''
    mode=0 
    xy=np.linspace(-1,1,numPnt)
    H_x,H_y=np.meshgrid(xy,xy)#rays to trace
    sec_x=np.zeros(H_x.shape) #where ray crosses secondary focus
    sec_y=np.zeros(H_x.shape) 
    focus_x=np.zeros(H_x.shape) #where ray hits detectors wrt optics tube
    focus_y=np.zeros(H_x.shape)
    mask=np.zeros(H_x.shape) #where data is outside circular fov
    for x in range(H_x.shape[0]) :
        for y in range(H_x.shape[1]) :
            if H_x[x,y]**2+H_y[x,y]**2 < 1. :
                trace=lnk.zGetTrace(0,mode,secondary_surf,H_x[x,y],H_y[x,y],0,0)
                sec_x[x,y]=trace.x
                sec_y[x,y]=trace.y
                mask[x,y]=1.
                trace=lnk.zGetTrace(0,mode,array,H_x[x,y],H_y[x,y],0,0)
                focus_x[x,y]=trace.x
                focus_y[x,y]=trace.y
    return sec_x,sec_y,focus_x,focus_y,mask

if __name__ == '__main__' and get_data:
    pyz.closeLink() #close all past links to zemax (max of 2 can exist)
    lnk = pyz.createLink()
    if lnk.zPushLens() != 0 :#pushes lens in server in user space - try without this
        print('ERROR - please allow program to push lenses (pg722 of manual)')
        exit()
    lnk.zSetConfig(1) #put into non-tipped format
    old_field=lnk.zGetField(0) # store old field for use with tubes
    lnk.zSetField(0,0,2,0)#set to just two fields
    lnk.zSetField(1,0,0)
    lnk.zSetField(2,fov,0)
    lnk.zPushLens() #seems to be needed to update fields.....
    numSurf=63 #should really get this from file
    for ns in range(numSurf) :
        comment=lnk.zGetSurfaceData(ns,1) 
        if comment == 'secondary focus' : 
            secondary_focus = ns
            break
    if secondary_focus == -1 :
        print('secondary focus not found, check the names in the lens file')
        exit()
    lnk.zPushLens() #not sure if this is needed but cannot hurt.....
    sky,sec,mask=get_points4telescope(lnk,fov,ds=0.1,secondary_surf=secondary_focus)
    LAT={'elev':sky[0]*fov,'xel':sky[1]*fov,'x':sec[0],'y':sec[1],'mask':mask} #y is in beam dir
    lnk.zSetField(2,0.65,0)
    exit()
    LATR=[]
    for cc in range(lnk.zGetConfig()[1]):
        lnk.zSetConfig(cc+1)
        lnk.zPushLens()
        sx,sy,fx,fy,m=get_pointsLATR(lnk,sky[0].shape[0],secondary_surf=secondary_focus,array=numSurf)
        LATR.append({'sec_x':sx,'sec_y':sy,'array_x':fx,'array_y':fy,'mask':m})
    #now for the seconary to focal plane mapping
    tube_mapping=get_tube_dict()
    print('Saving data to :',dir+id9.split('.')[0]+'_trace_data.npz')
    np.savez(dir+id9.split('.')[0]+'_trace_data.npz',tube_mapping=tube_mapping,
             LATR=LATR,LAT=LAT)
#exit()   
if get_data == False :
    dat=np.load(dir+id9.split('.')[0]+'_trace_data.npz',allow_pickle=True)
    tube_mapping=dat['tube_mapping'][()]
    LAT=dat['LAT'][()]
    LATR=dat['LATR'][()]

from scipy.interpolate import interp2d

gi=np.where(LAT['mask'] != 0.)
sec2elev = interp2d(LAT['x'][gi],LAT['y'][gi],LAT['elev'][gi],bounds_error=True)
sec2xel  = interp2d(LAT['x'][gi],LAT['y'][gi],LAT['xel'][gi],bounds_error=True)
array2secx=[] #set up optics tubes....
array2secy=[]
for n in range(len(LATR)):
    
    print('working on tube ',n+1)
    gi=np.where(LATR[n]['mask'] != 0)
    array2secx.append(
            interp2d(LATR[n]['array_x'][gi],LATR[n]['array_y'][gi],LATR[n]['sec_x'][gi],bounds_error=True))
    array2secy.append(
            interp2d(LATR[n]['array_x'][gi],LATR[n]['array_y'][gi],LATR[n]['sec_y'][gi],bounds_error=True))
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

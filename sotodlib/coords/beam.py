import numpy as np
import logging

import so3g
from pixell import enmap
from sotodlib import coords
from sotodlib.coords import demod as demod_mm
from sotodlib.coords import helpers


logger = logging.getLogger(__name__)

def get_each_detcen_P(tod, azpl, elpl, size=None, res=None, proj = 'car', flags=None, boresight_centered = False):
    """Get a standard Projection Matrix for detector-centered/boresight-centered coordinates.
    This is mainly used for beam characterization.
    Currently this must be done every single detector. See "coadded_maps()" in detail 

    Args:
      tod (float): axis manager that include one detector.
      azpl (float): planet azimuth, in radians.
      elpl (float): planet elevation, in radians.
      size (float): size of output Projection Matrix, in radians.
      res (float): resolution output Projection Matrix, in radians.
      proj (str): Default is CAR.
      flags: flags that use for map-making
      boresight_centered (bool): if True, the output map is in boresight-centered coordinate system instead.

    Returns a Projection Matrix
    """
    if res is None:
        res = 0.01 * coords.DEG
    if size is None:
        size = 5 * coords.DEG

    xi = tod.focal_plane.xi[0]
    eta = tod.focal_plane.eta[0]
    sight = so3g.proj.CelestialSightLine.for_horizon(tod.timestamps, tod.boresight.az, tod.boresight.el, roll= tod.boresight.roll)
    
    # planet quaternion in Az/El coordinate
    pq = so3g.proj.quat.rotation_lonlat(-azpl, elpl)
    # xieta quaternion for a given detector
    xieta_gamma0 = so3g.proj.quat.rotation_xieta(xi,eta,0)
    #sight.Q = ~xieta0 * ~sight.Q * pq * ~xieta0 # First xieta for moving to center for each detector, and the last one for canceling out
    if boresight_centered:
        detQ = ~sight.Q * pq # fixed Boresight center instead of detector center
    else:
        detQ = ~xieta_gamma0 * ~sight.Q * pq
    xis, etas, _  = so3g.proj.quat.decompose_xieta(detQ)
    sight.Q = so3g.proj.quat.rotation_xieta(xis, etas, 0) * ~xieta_gamma0
    # cut the gamma so that reference/cross polarization follows Ludwig 3-I definition. Ludwig 3-I definition is gamma of 0 at all xi, eta.
    # additional `~xieta_gamma0` here is used for cancelling the one that will be accounted for in so3g by default.
    rot = so3g.proj.quat.rotation_lonlat(0, 0)
    box = np.array([[-1, -1], [1, 1]]) * size
    geom = enmap.geometry(pos=box, res=res, proj = proj)
    P = coords.P.for_tod(tod, sight=sight, rot=rot,  geom=geom, hwp=True, comps='TQU', cuts=flags)

    return P

def calc_detcen_coadd_maps(tod, azpl, elpl, size=None, res=None, flag_str = "mapmaking"):
    """ Compute the coadded map including hit map from axis manager after preprcess applied.
    This function just executes get_each_detcen_P() every detector stored in axis manager and then coadded them.

    Args:
      tod (float): axis manager that include one detector.
      azpl (float): planet azimuth, in radians.
      elpl (float): planet elevation, in radians.
      size (float): size of output Projection Matrix, in radians.
      res (float): resolution output Projection Matrix, in radians.
      flag_str (str): name of flag for mapmaking. Default "mapmaking".
    
    Returns Coadded maps: hit, map, weighted_map, weight.
    """
    results = []
    for idet in tod.det_info.det_id:
        itod = tod.copy()
        itod.restrict('dets', itod.det_info.det_id == idet)
        iflags = itod.flags[flag_str]
        P = get_each_detcen_P(itod, azpl, elpl, size=size, res=res, flags=iflags, boresight_centered = False)

        result = demod_mm.make_map(itod, P=P, det_weights_demod=itod.inv_var)
        h = P.to_map(tod=itod, signal=np.ones(itod.dsT.shape, dtype = np.float32), det_weights=None, comps='T')[0]
        result['hit'] = h

        results.append(result)
    coadded_maps = coadd_maps(results)
    return coadded_maps

def coadd_maps(results):
    carwmaps = []
    carweights = []
    carhits = []
    for iresult in results:
        carwmaps.append(iresult['weighted_map'])
        carweights.append(iresult['weight'])
        carhits.append(iresult['hit'])

    deshit, dessummaps, desws, deswmaps = add_maps_all(carhits, carweights, carwmaps)
    ret = {'hit': deshit,
           'map': dessummaps,
           'weighted_map': deswmaps,
           'weight': desws}
    return ret

def add_maps_all(hits, weight, weightmaps):
    sumhit = np.sum(hits, axis = 0)
    sumws = np.sum(weight, axis = 0)
    sumwmaps = np.sum(weightmaps, axis = 0)
    summaps = exe_remove_weights(sumwmaps, sumws)
    deshit = enmap.full(hits[0].shape, hits[0].wcs, sumhit)
    desws = enmap.full(weight[0].shape, weight[0].wcs, sumws)
    deswmaps = enmap.full(weightmaps[0].shape, weightmaps[0].wcs, sumwmaps)
    dessummaps = enmap.full(weightmaps[0].shape, weightmaps[0].wcs, summaps)
    return deshit, dessummaps, desws, deswmaps

def exe_remove_weights(signal_map, weights_map, dest = None, eigentol=1e-4):
    inverse_weights_map = cal_inverse_weights(weights_map, dest = None, eigentol=eigentol)
    if dest is None:
        dest = np.zeros_like(signal_map)
    dest[:] = helpers._apply_inverse_weights_map(inverse_weights_map, signal_map)
    return dest

def cal_inverse_weights(weights_map, dest = None, eigentol=1e-4):
    if dest is None:
        dest = np.zeros_like(weights_map)
    dest[:] = helpers._invert_weights_map(weights_map, eigentol=eigentol, UPLO='U')
    return dest
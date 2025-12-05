import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import so3g
import yaml
from mapcat.database import DepthOneMapTable, TODDepthOneTable
from mapcat.helper import Settings
from pixell import bunch, enmap, mpi, utils

from sotodlib import coords, mapmaking
from sotodlib.core import FlagManager, metadata
from sotodlib.tod_ops import detrend_tod

DEFAULTS = {"query": "1",
            "odir": "./outputs",
            "comps": "T",
            "ntod": None,
            "tods": None,
            "nset": None,
            "site": 'so_lat',
            "nmat": "corr",
            "max_dets": None,
            "verbose": 0,
            "quiet": 0,
            "center_at": None,
            "window": 0.0,
            "nmat_dir": "/nmats",
            "nmat_mode": "build",
            "downsample": 1,
            "maxiter": 100,
            "tiled": 1,
            "wafer": None,
            "freq": None,
            "tasks_per_group":1,
            "cont":False,
            "rhs": False,
            "bin": False,
            "srcsamp": None,
            "unit": 'K',
            "mapcat_database_type": "sqlite",
            "mapcat_database_name": "mapcat.db",
            "mapcat_depth_one_parent": "./",
            "min_dets": 50,
           }

SENS_LIMITS = {"f030":120, "f040":80, "f090":100, "f150":140, "f220":300, "f280":750}

LoaderError = metadata.loader.LoaderError

class DataMissing(Exception):
    pass

def sensitivity_cut(rms_uKrts, sens_lim, med_tol=0.2, max_lim=100):
    # First reject detectors with unreasonably low noise
    good = rms_uKrts >= sens_lim
    # Also reject far too noisy detectors
    good &= rms_uKrts < sens_lim * max_lim
    # Then reject outliers
    if np.sum(good) == 0: 
        return good
    ref = np.median(rms_uKrts[good])
    good &= rms_uKrts > ref * med_tol
    good &= rms_uKrts < ref / med_tol
    return good

def measure_rms(tod, dt=1, bsize=32, nblock=10):
    tod = tod[:,:tod.shape[1]//bsize*bsize]
    tod = tod.reshape(tod.shape[0],-1,bsize)
    bstep = max(1,tod.shape[1]//nblock)
    tod = tod[:,::bstep,:][:,:nblock,:]
    rms = np.median(np.std(tod,-1),-1)
    # to µK√s units
    rms *= dt**0.5
    return rms

def _get_config(config_file):
    try:
        config = yaml.safe_load(open(config_file,'r'))
    except Exception as e:
        raise RuntimeError(f"Failed to load config {config_file}: {e}")
    return config

def tele2equ(coords, ctime, detoffs=[0,0], site="so_sat1"):

    # Broadcast and flatten input arrays
    coords, ctime = utils.broadcast_arrays(coords, ctime, npre=(1,0))
    cflat = utils.to_Nd(coords, 2, axis=-1)
    tflat = utils.to_Nd(ctime,  1, axis=-1)
    dflat, dshape = utils.to_Nd(detoffs, 2, axis=-1, return_inverse=True)
    nsamp, ndet = cflat.shape[1], dflat.shape[1]
    assert cflat.shape[1:] == tflat.shape, f"tele2equ coords and ctime have incompatible shapes {coords.shape} vs {ctime.shape}"

    # Set up the transform itself
    sight  = so3g.proj.CelestialSightLine.az_el(tflat, cflat[0], cflat[1],
            roll=cflat[2] if len(cflat) > 2 else 0, site=site, weather="toco")

    # To support other coordiante systems I would add
    # if rot is not None: sight.Q = rot * sight.Q
    dummy = np.arange(ndet)
    fp = so3g.proj.FocalPlane.from_xieta(dummy, dflat[0], dflat[1],
            dflat[2] if len(dflat) > 2 else 0)
    asm = so3g.proj.Assembly.attach(sight, fp)
    proj = so3g.proj.Projectionist()
    res = np.zeros((ndet,nsamp,4))

    # And actually perform it
    proj.get_coords(asm, output=res)

    # Finally unflatten
    res = res.reshape(dshape[1:]+coords.shape[1:]+(4,))
    return res

def find_scan_profile(context, my_tods, my_infos, comm=mpi.COMM_WORLD, npoint=100):

    # Pre-allocate empty profile since other tasks need a receive buffer
    profile = np.zeros([2,npoint])

    # Who has the first valid tod?
    first = np.where(comm.allgather([len(my_tods)]))[0][0]

    if comm.rank == first:
        tod, info = my_tods[0], my_infos[0]
        # Find our array's central pointing offset. 
        fp = tod.focal_plane
        xi0 = np.mean(utils.minmax(fp.xi))
        eta0 = np.mean(utils.minmax(fp.eta))
        # Build a boresight corresponding to a single az sweep at constant time
        azs  = info.az_center + np.linspace(-info.az_throw/2, info.az_throw/2, npoint)
        els  = np.full(npoint, info.el_center)
        profile[:] = tele2equ(np.array([azs, els])*utils.degree, info.timestamp, detoffs=[xi0, eta0]).T[1::-1] # dec,ra

    comm.Bcast(profile, root=first)

    return profile

def find_footprint(context, tods, ref_wcs, comm=mpi.COMM_WORLD, return_pixboxes=False, pad=1):
    # Measure the pixel bounds of each observation relative to our
    # reference wcs
    pixboxes = []
    for tod in tods:
        my_shape, my_wcs = coords.get_footprint(tod, ref_wcs)
        my_pixbox = enmap.pixbox_of(ref_wcs, my_shape, my_wcs)
        pixboxes.append(my_pixbox)
    pixboxes = utils.allgatherv(pixboxes, comm)
    if len(pixboxes) == 0:
        raise DataMissing("No usable obs to estimate footprint from")
    # Handle sky wrapping. This assumes cylindrical coordinates with sky-wrapping
    # in the x-direction, and that there's an integer number of pixels around the sky.
    # Could be done more generally, but would be much more involved, and this should be
    # good enough
    nphi = utils.nint(np.abs(360/ref_wcs.wcs.cdelt[0]))
    widths = pixboxes[:,1,1]-pixboxes[:,0,1]
    pixboxes[:,0,1] = utils.rewind(pixboxes[:,0,1], ref=pixboxes[0,0,1], period=nphi)
    pixboxes[:,1,1] = pixboxes[:,0,1] + widths
    # It's now safe to find the total pixel bounding box
    union_pixbox = np.array([np.min(pixboxes[:,0],0)-pad,np.max(pixboxes[:,1],0)+pad])
    # Use this to construct the output geometry
    shape = union_pixbox[1]-union_pixbox[0]
    # Cap xshape to nphi. To see why, consider this example:
    # Sky width: 100
    # box 0:   0  30
    # box 1: -40  10
    # box 2:  20  70
    # box 3:  45 110
    # union: -40 110: Wider than the whole sky!
    # But since we use union_pixbox[0] as the zero-pixel in
    # our output geometry, this overflow just results in
    # unhittable pixels for x >= nphi, which we can just chop off here
    shape[-1] = min(shape[-1], nphi)
    wcs = ref_wcs.deepcopy()
    wcs.wcs.crpix -= union_pixbox[0,::-1]
    # Make sure wcs crval follows so3g pointing matrix assumptions
    shape, wcs = coords.normalize_geometry(shape, wcs)
    if return_pixboxes:
        return shape, wcs, pixboxes
    else:
        return shape, wcs

def read_tods(context, obslist, inds=None, comm=mpi.COMM_WORLD, no_signal=False, site='so', L=None, min_dets=50):
    my_tods = []
    my_inds = []
    if inds is None:
        inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            tod = context.get_obs(obs_id, dets={"wafer_slot":detset, "wafer.bandpass":band}, no_signal=no_signal)
            tod = calibrate_obs(tod, band, site=site, L=L, min_dets=min_dets)
            my_tods.append(tod)
            my_inds.append(ind)
        except RuntimeError:
            continue
    return my_tods, my_inds

def calibrate_obs(obs, band, site='so', dtype_tod=np.float32, nocal=True, unit='K', L=None, min_dets=50):
    if obs.signal is not None and obs.dets.count < min_dets:
        return None
    if (not nocal) and (obs.signal is not None):
        # Check nans
        mask = np.logical_not(np.isfinite(obs.signal))
        if mask.sum() > 0:
            return None
        # Check all 0s
        zero_dets = np.sum(obs.signal, axis=1)
        mask = zero_dets == 0.0
        if mask.any():
            obs.restrict('dets', obs.dets.vals[np.logical_not(mask)])
    # Cut non-optical dets
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
    mapmaking.fix_boresight_glitches(obs, )
    srate = (obs.samps.count-1)/(obs.timestamps[-1]-obs.timestamps[0])
    # Add site and weather, since they're not in obs yet
    obs.wrap("weather", np.full(1, "toco"))
    if "site" not in obs:
        obs.wrap("site",    np.full(1, site))

    # add dummy glitch flags if not present
    if 'flags' not in obs._fields:
        obs.wrap('flags', FlagManager.for_tod(obs))
    if "glitch_flags" not in obs.flags:
        obs.flags.wrap('glitch_flags', so3g.proj.RangesMatrix.zeros(obs.shape[:2]),[(0,'dets'),(1,'samps')])

    if obs.signal is not None:
        detrend_tod(obs, method='linear')
        utils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(dtype_tod)

    if (not nocal) and (obs.signal is not None):
        rms = measure_rms(obs.signal, dt=1/srate)
        if unit=='K':
            good = sensitivity_cut(rms*1e6, SENS_LIMITS[band])
        elif unit == 'uK':
            good = sensitivity_cut(rms, SENS_LIMITS[band])
        utils.deslope(obs.signal, w=5, inplace=True)
    return obs

def map_to_calculate(map_name: str, inds_to_use: List[int], mapcat_settings: Dict[str,str])->bool:

    with Settings(**mapcat_settings).session() as session:
        existing_map = session.query(DepthOneMapTable).filter_by(map_name=map_name).first()
        map_tods = existing_map.tods if existing_map else []

        total_tods = np.sum([map_tod.wafer_count for map_tod in map_tods])

        if total_tods < len(inds_to_use):
            return True
    return False

def commit_depth1_tods(map_name:str, obslist: Dict[Tuple[int, str, str], List[Tuple[str, str, str, int]]], obs_infos: List[Tuple[str, float, float, float, int, str, str, str, str, str, str, int, str, float, float, float, float, float,  float, float, str, str]],
                       band: str, inds: List[int], mapcat_settings: Dict[str, str]) -> List[TODDepthOneTable]:
    with Settings(**mapcat_settings).session() as session:
        depth1map_obsids = np.unique([obslist[ind][0] for ind in inds])
        tods = []
        for obs_id in depth1map_obsids:
            obs_info = obs_infos[obs_infos["obs_id"] == obs_id][0]
            tod_depth1_entry = {"obs_id": obs_id,
                                "ctime": obs_info["timestamp"],
                                "start_time": obs_info["start_time"],
                                "stop_time": obs_info["stop_time"],
                                "nsamples": int(obs_info["n_samples"]),
                                "telescope": obs_info["telescope"],
                                "telescope_flavor": obs_info["telescope_flavor"],
                                "tube_slot": obs_info["tube_slot"],
                                "tube_flavor": obs_info["tube_flavor"],
                                "frequency": band,
                                "scan_type": obs_info["type"],
                                "subtype": obs_info["subtype"],
                                "wafer_count": int(obs_info["wafer_count"]),
                                "duration": obs_info["duration"],
                                "az_center": obs_info["az_center"],
                                "az_throw": obs_info["az_throw"],
                                "el_center": obs_info["el_center"],
                                "el_throw": obs_info["el_throw"],
                                "roll_center": obs_info["roll_center"],
                                "roll_throw": obs_info["roll_throw"],
                                "wafer_slots_list": obs_info["wafer_slots_list"],
                                "stream_ids_list": obs_info["stream_ids_list"]}
            existing_tod = session.query(TODDepthOneTable).filter_by(**tod_depth1_entry).first()
            tod = TODDepthOneTable(map_name=map_name, **tod_depth1_entry)
            if existing_tod is None:
                session.add(tod)
                tods.append(tod)
            else:
                tods.append(existing_tod)
        session.commit()
    return tods

def commit_depth1_map(map_name:str, prefix:str, detset:str, band:str, ctime:float, start_time:float, stop_time:float,
                      tods: List[TODDepthOneTable], mapcat_settings:Dict[str, str])->None:
    with Settings(**mapcat_settings).session() as session:
        existing_map = session.query(DepthOneMapTable).filter_by(map_name=map_name).first()
        depth1map_meta = DepthOneMapTable(map_id=existing_map.map_id if existing_map else None,
                                            map_name=map_name,
                                            map_path=prefix + "_map.fits",
                                            ivar_path=prefix + "_ivar.fits",
                                            time_path=prefix + "_time.fits",
                                            tube_slot=detset,
                                            frequency=band,
                                            ctime=ctime,
                                            start_time=start_time,
                                            stop_time=stop_time,
                                            tods=tods
                                            )
        session.merge(depth1map_meta)
        session.commit()

def write_depth1_map(prefix, data, dtype = np.float32, binned=False, rhs=False, unit='K'):
    data.signal.write(prefix, "map",  data.map.astype(dtype), unit=unit)
    data.signal.write(prefix, "ivar", data.ivar.astype(dtype), unit=f"{unit}^-2")
    data.signal.write(prefix, "time", data.tmap.astype(dtype))
    if binned:
        data.signal.write(prefix, "bin", data.bin.astype(dtype), unit=unit)
    if rhs:
        data.signal.write(prefix, "rhs", data.signal.rhs.astype(dtype), unit=f"{unit}^-1")

def write_depth1_info(oname, info):
    utils.mkdir(os.path.dirname(oname))
    bunch.write(oname, info)

def create_mapmaker_config(defaults: dict=DEFAULTS, config_file: Optional[str]=None, **args)->dict:
    config = dict(defaults)

    # Update the default dict with values provided from a config.yaml file
    if config_file is not None:
        config_from_file = _get_config(config_file)
        config.update({k: v for k, v in config_from_file.items() if v is not None})
    else:
        print("No config file provided, assuming default values") 

    # Merge flags from config file and defaults with any passed through CLI
    config.update({k: v for k, v in args.items() if v is not None})

    # Certain fields are required. Check if they are all supplied here
    required_fields = ['area','context']
    for req in required_fields:
        if req not in config.keys():
            raise KeyError(f"{req} is a required argument. Please supply it in a config file or via the command line")
    return config

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

import numpy as np
import so3g
from pixell import bunch, enmap, mpi
from pixell import utils as putils

from sotodlib import coords, mapmaking
from sotodlib.core import FlagManager, metadata
from sotodlib.site_pipeline.utils.config import _get_config
from sotodlib.tod_ops import detrend_tod

DEFAULTS = {
    "query": "1",
    "odir": "./outputs",
    "comps": "T",
    "ntod": None,
    "tods": None,
    "nset": None,
    "site": "so_lat",
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
    "tasks_per_group": 1,
    "cont": False,
    "rhs": False,
    "bin": False,
    "srcsamp": None,
    "unit": "K",
    "mapcat_database_type": "sqlite",
    "mapcat_database_name": "mapcat.db",
    "mapcat_depth_one_parent": "./",
    "min_dets": 50,
}

SENS_LIMITS = {
    "f030": 120,
    "f040": 80,
    "f090": 100,
    "f150": 140,
    "f220": 300,
    "f280": 750,
}

LoaderError = metadata.loader.LoaderError


class DataMissing(Exception):
    pass


def sensitivity_cut(
    rms_uKrts: np.ndarray, sens_lim: float, med_tol: float = 0.2, max_lim: float = 100
) -> np.ndarray:

    # First reject detectors with unreasonably low noise
    # Also reject far too noisy detectors
    good = rms_uKrts >= sens_lim
    good &= rms_uKrts < sens_lim * max_lim

    # Then reject outliers
    if np.sum(good) == 0:
        return good

    ref = np.median(rms_uKrts[good])
    good &= rms_uKrts > ref * med_tol
    good &= rms_uKrts < ref / med_tol

    return good


def measure_rms(
    tod: np.ndarray, dt: float = 1, bsize: int = 32, nblock: int = 10
) -> np.ndarray:

    tod = tod[:, : tod.shape[1] // bsize * bsize]
    tod = tod.reshape(tod.shape[0], -1, bsize)
    bstep = max(1, tod.shape[1] // nblock)
    tod = tod[:, ::bstep, :][:, :nblock, :]
    rms = np.median(np.std(tod, -1), -1)

    # to µK√s units
    rms *= dt**0.5

    return rms


def tele2equ(
    coords: List[np.ndarray],
    ctime: float,
    detoffs: List[int] = [0, 0],
    site: str = "so_sat1",
) -> np.ndarray:

    # Broadcast and flatten input arrays
    coords, ctime = putils.broadcast_arrays(coords, ctime, npre=(1, 0))
    cflat = putils.to_Nd(coords, 2, axis=-1)
    tflat = putils.to_Nd(ctime, 1, axis=-1)
    dflat, dshape = putils.to_Nd(detoffs, 2, axis=-1, return_inverse=True)
    nsamp, ndet = cflat.shape[1], dflat.shape[1]
    assert (
        cflat.shape[1:] == tflat.shape
    ), f"tele2equ coords and ctime have incompatible shapes {coords.shape} vs {ctime.shape}"

    # Set up the transform itself
    sight = so3g.proj.CelestialSightLine.az_el(
        tflat,
        cflat[0],
        cflat[1],
        roll=cflat[2] if len(cflat) > 2 else 0,
        site=site,
        weather="toco",
    )

    # To support other coordiante systems I would add
    # if rot is not None: sight.Q = rot * sight.Q
    dummy = np.arange(ndet)
    fp = so3g.proj.FocalPlane.from_xieta(
        dummy, dflat[0], dflat[1], dflat[2] if len(dflat) > 2 else 0
    )
    asm = so3g.proj.Assembly.attach(sight, fp)
    proj = so3g.proj.Projectionist()
    res = np.zeros((ndet, nsamp, 4))

    # And actually perform it
    proj.get_coords(asm, output=res)

    # Finally unflatten
    res = res.reshape(dshape[1:] + coords.shape[1:] + (4,))
    return res


def find_scan_profile(
    tods, infos, comm: "Comm" = mpi.COMM_WORLD, npoint: int = 100
) -> np.ndarray:

    # Pre-allocate empty profile since other tasks need a receive buffer
    profile = np.zeros([2, npoint])

    # Who has the first valid tod?
    first = np.where(comm.allgather([len(tods)]))[0][0]

    if comm.rank == first:
        tod, info = tods[0], infos[0]
        # Find our array's central pointing offset.
        fp = tod.focal_plane
        xi0 = np.mean(putils.minmax(fp.xi))
        eta0 = np.mean(putils.minmax(fp.eta))
        # Build a boresight corresponding to a single az sweep at constant time
        azs = info.az_center + np.linspace(
            -info.az_throw / 2, info.az_throw / 2, npoint
        )
        els = np.full(npoint, info.el_center)
        profile[:] = tele2equ(
            np.array([azs, els]) * putils.degree, info.timestamp, detoffs=[xi0, eta0]
        ).T[
            1::-1
        ]  # dec,ra

    comm.Bcast(profile, root=first)

    return profile


def find_footprint(
    tods,
    ref_wcs,
    comm: "Comm" = mpi.COMM_WORLD,
    return_pixboxes: bool = False,
    pad: int = 1,
) -> Tuple[Any, Any, Optional[np.ndarray]]:

    # Measure the pixel bounds of each observation relative to our
    # reference wcs
    pixboxes = []
    for tod in tods:
        my_shape, my_wcs = coords.get_footprint(tod, ref_wcs)
        my_pixbox = enmap.pixbox_of(ref_wcs, my_shape, my_wcs)
        pixboxes.append(my_pixbox)

    pixboxes = putils.allgatherv(pixboxes, comm)
    if len(pixboxes) == 0:
        raise DataMissing("No usable obs to estimate footprint from")

    # Handle sky wrapping. This assumes cylindrical coordinates with sky-wrapping
    # in the x-direction, and that there's an integer number of pixels around the sky.
    # Could be done more generally, but would be much more involved, and this should be
    # good enough
    nphi = putils.nint(np.abs(360 / ref_wcs.wcs.cdelt[0]))
    widths = pixboxes[:, 1, 1] - pixboxes[:, 0, 1]
    pixboxes[:, 0, 1] = putils.rewind(
        pixboxes[:, 0, 1], ref=pixboxes[0, 0, 1], period=nphi
    )
    pixboxes[:, 1, 1] = pixboxes[:, 0, 1] + widths

    # It's now safe to find the total pixel bounding box
    union_pixbox = np.array(
        [np.min(pixboxes[:, 0], 0) - pad, np.max(pixboxes[:, 1], 0) + pad]
    )

    # Use this to construct the output geometry
    shape = union_pixbox[1] - union_pixbox[0]

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
    wcs.wcs.crpix -= union_pixbox[0, ::-1]

    # Make sure wcs crval follows so3g pointing matrix assumptions
    shape, wcs = coords.normalize_geometry(shape, wcs)

    if return_pixboxes:
        return shape, wcs, pixboxes
    else:
        return shape, wcs, None


def read_tods(
    context,
    obslist,
    inds=None,
    comm=mpi.COMM_WORLD,
    no_signal=False,
    site="so",
    L=None,
    min_dets=50,
):
    my_tods = []
    my_inds = []
    if inds is None:
        inds = list(range(comm.rank, len(obslist), comm.size))
    for ind in inds:
        obs_id, detset, band, obs_ind = obslist[ind]
        try:
            tod = context.get_obs(
                obs_id,
                dets={"wafer_slot": detset, "wafer.bandpass": band},
                no_signal=no_signal,
            )
            tod, _ = calibrate_obs(tod, band, site=site, L=L, min_dets=min_dets)
            my_tods.append(tod)
            my_inds.append(ind)
        except RuntimeError:
            continue
    return my_tods, my_inds


def calibrate_obs(
    obs,
    band,
    site="so",
    dtype_tod=np.float32,
    nocal=True,
    unit="K",
    L=None,
    min_dets=50,
) -> Tuple[Optional[Any], Optional[np.ndarray]]:

    good = None
    if obs.signal is not None and obs.dets.count < min_dets:
        return None, None

    if (not nocal) and (obs.signal is not None):
        # Check nans
        mask = np.logical_not(np.isfinite(obs.signal))
        if mask.sum() > 0:
            return None, None
        # Check all 0s
        zero_dets = np.sum(obs.signal, axis=1)
        mask = zero_dets == 0.0
        if mask.any():
            obs.restrict("dets", obs.dets.vals[np.logical_not(mask)])
    # Cut non-optical dets
    obs.restrict("dets", obs.dets.vals[obs.det_info.wafer.type == "OPTC"])
    mapmaking.fix_boresight_glitches(
        obs,
    )
    srate = (obs.samps.count - 1) / (obs.timestamps[-1] - obs.timestamps[0])
    # Add site and weather, since they're not in obs yet
    obs.wrap("weather", np.full(1, "toco"))
    if "site" not in obs:
        obs.wrap("site", np.full(1, site))

    # add dummy glitch flags if not present
    if "flags" not in obs._fields:
        obs.wrap("flags", FlagManager.for_tod(obs))
    if "glitch_flags" not in obs.flags:
        obs.flags.wrap(
            "glitch_flags",
            so3g.proj.RangesMatrix.zeros(obs.shape[:2]),
            [(0, "dets"), (1, "samps")],
        )

    if obs.signal is not None:
        detrend_tod(obs, method="linear")
        putils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(dtype_tod)

    if (not nocal) and (obs.signal is not None):
        rms = measure_rms(obs.signal, dt=1 / srate)
        if unit == "K":
            good = sensitivity_cut(rms * 1e6, SENS_LIMITS[band])
        elif unit == "uK":
            good = sensitivity_cut(rms, SENS_LIMITS[band])
        putils.deslope(obs.signal, w=5, inplace=True)
    return obs, good


def write_depth1_map(
    prefix: str,
    data: np.ndarray,
    dtype: np.typing.DTypeLike = np.float32,
    binned: bool = False,
    rhs: bool = False,
    unit: str = "K",
):

    data.signal.write(prefix, "map", data.map.astype(dtype), unit=unit)
    data.signal.write(prefix, "ivar", data.ivar.astype(dtype), unit=f"{unit}^-2")
    data.signal.write(prefix, "time", data.tmap.astype(dtype))

    if binned:
        data.signal.write(prefix, "bin", data.bin.astype(dtype), unit=unit)

    if rhs:
        data.signal.write(
            prefix, "rhs", data.signal.rhs.astype(dtype), unit=f"{unit}^-1"
        )


def write_depth1_info(oname: str, info: Dict[Any, Any]):

    putils.mkdir(os.path.dirname(oname))
    bunch.write(oname, info)


def create_mapmaker_config(
    defaults: dict = DEFAULTS, config_file: Optional[str] = None, **args
) -> dict:

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
    required_fields = ["area", "context"]
    for req in required_fields:
        if req not in config.keys():
            raise KeyError(
                f"{req} is a required argument. Please supply it in a config file or via the command line"
            )

    return config

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

import numpy as np
import so3g
from pixell import bunch, enmap, mpi, tilemap
from pixell import utils as putils
from scipy import ndimage

from sotodlib import coords, mapmaking
from sotodlib.core import FlagManager, metadata
from sotodlib.mapmaking import utils as smutils
from sotodlib.site_pipeline.utils.config import _get_config
from sotodlib.tod_ops import detrend_tod

DEPTH1MAPMAKER_DEFAULTS = {
    "query": "type == 'obs' and subtype == 'cmb'",
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
    "update_delay": None,
    "min_dur": 300,
    "pretend_now_is": None,
    "gapfill_cuts": False,
    "holefill_mask": None,
    "holefill_thresh": 0.01,
    "holefill_buffer": 0.1,
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
    """Sensitivity cuts for mapmakers, based on white noise of individual detectors

    Parameters
    ----------
    rms_uKrts : ndarray
        RMS per det with shape Ndets, in units of uK*sqrt(s).
    sens_lim : float
        Fixed limit sensitivity.
    med_tol : float, optional
        Median tolerance for rejection of outliers.
    max_lim : float, optional
        Also reject very noisy detectors with noise higher than
        max_lim*sens_lim

    Returns
    -------
    good: ndarray
        A boolean array with good and bad detectors.
    """

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
    """Find an enmap geometry (shape, wcs) that encompass
    all TODs composing a depth-1 map. Useful to limit the
    size of the depth-1 map to only the necessary.

    Caveats: there might be edge cases covering a wide range
    of R.A. where the combined depth-1 map ends up wrapping on
    the other edge. The end result is a map which covers 360deg
    in R.A. The reason is that a priori you don't know where the
    center of the overall depth-1 map is. So if the depth-1 map
    exposes an area >180deg wide in R.A., you might end up with
    this issue. In the future we might add some mechanism that
    finds the middle-point of the hypothetical perfectly
    unwrapped patch, which would allow us to find the optimal
    wrapping.

    Parameters
    ----------
    tods : list
        List of Axis Managers containing the observations in a
        depth-1 map. These will don't have signal, but only the
        ancillary data required (boresight, focal_plane,
        timestamps, etc).
    ref_wcs : WCS dict
        Reference wcs to build the geometry of the map.
    comm : MPI communicator, optional
    return_pixboxes : bool, optional
        whether or not also return the pixboxes of the obs.
    pad : int, optional

    Returns
    -------
    shape : tuple
        Shape of the geometry, (N pixels in X, N pixels in Y).
    wcs : WCS dict
        WCS of the map.
    pixboxes : list, optional
        List of pixboxes for each input obs.
    """

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


def _set_glitch_flags(obs, mask):
    obs.flags.move("glitch_flags", None)
    obs.flags.wrap(
        "glitch_flags",
        so3g.proj.RangesMatrix.from_mask(mask),
        [(0, "dets"), (1, "samps")],
    )


def cut_bright(obs, sky, thresh, buffer):
    """Cut the samples where sky > thresh, plus buffer seconds on each side.

    The cuts are added to obs.flags.glitch_flags.

    Arguments:
        obs: AxisManager with pointing and glitch_flags.
        sky: (1, ny, nx) T map, in the same unit as thresh.
        thresh: brightness limit.
        buffer: padding along the scan, in seconds.

    Returns:
        (ndet, nsamp) bool array of the bright-strip samples.
    """
    ctime = obs.timestamps
    srate = (len(ctime) - 1) / (ctime[-1] - ctime[0])
    pmat = coords.pmat.P.for_tod(
        obs,
        comps="T",
        geom=sky.geometry,
        threads="domdir",
        weather=smutils.unarr(obs.weather),
        site=smutils.unarr(obs.site),
        interpol="nearest",
    )
    tod = np.zeros(obs.signal.shape, np.float32)
    pmat.from_map(sky, dest=tod)
    bright = tod > thresh
    del tod, pmat
    nbuf = int(round(buffer * srate))
    if nbuf > 0 and bright.any():
        bright = ndimage.binary_dilation(
            bright, structure=np.ones((1, 2 * nbuf + 1), bool)
        )
    _set_glitch_flags(obs, obs.flags.glitch_flags.mask() | bright)
    return bright


def strip_samples(obs, bright):
    """Get the bright-strip samples relative to the linear fill of the cuts.

    Call this after cut_bright and before gapfill_cuts. A strip with no good
    sample on one side (at a TOD end) uses the other end point on both sides,
    the same as np.interp in gapfill_cuts.

    Returns:
        Bunch of per-sample arrays: det, samp, left, right (fill end
        points), frac (position between the end points) and resid
        (signal - linear fill).
    """
    mask = obs.flags.glitch_flags.mask()
    rec = {k: [] for k in ("det", "samp", "left", "right", "frac", "resid")}
    for di in np.flatnonzero(bright.any(1)):
        good = np.flatnonzero(~mask[di])
        if good.size == 0:
            continue
        samp = np.flatnonzero(bright[di])
        pos = np.searchsorted(good, samp)
        left = good[np.maximum(pos - 1, 0)]
        right = good[np.minimum(pos, good.size - 1)]
        span = right - left
        frac = np.where(span > 0, (samp - left) / np.maximum(span, 1), 0.0)
        sig = obs.signal[di]
        rec["det"].append(np.full(samp.size, di))
        rec["samp"].append(samp)
        rec["left"].append(left)
        rec["right"].append(right)
        rec["frac"].append(frac)
        rec["resid"].append(sig[samp] - (1 - frac) * sig[left] - frac * sig[right])
    dtypes = dict(det=np.int32, samp=np.int32, left=np.int32, right=np.int32,
                  frac=np.float32, resid=np.float32)
    return bunch.Bunch(**{
        k: np.concatenate(v).astype(dtypes[k]) if v else np.zeros(0, dtypes[k])
        for k, v in rec.items()
    })


def gapfill_cuts(obs):
    """Fill the glitch_flags samples by linear interpolation, in place.

    The samples stay cut. The fill keeps bright signal in cut samples out of
    the noise model and the CG solution.
    """
    mask = obs.flags.glitch_flags.mask()
    x = np.arange(obs.samps.count)
    for di in np.flatnonzero(mask.any(1)):
        bad = mask[di]
        if bad.all():
            continue
        obs.signal[di, bad] = np.interp(x[bad], x[~bad], obs.signal[di, ~bad])


def holefill(signal_map, mapmaker, strips, map_):
    """Put the bright-strip samples back into the solved map.

    Each strip sample is its residual from the linear fill plus the solved
    map at the fill end points. The samples are binned with the white-noise
    weights and combined with the ML map by inverse variance. Only pixels hit
    by strip samples change.

    Arguments:
        signal_map: SignalMap after the solve.
        mapmaker: MLMapmaker.
        strips: dict of strip_samples output per obs name (this task only).
        map_: solved map.

    Returns:
        (map, div): the filled map and the total inverse covariance.
    """
    sm = signal_map
    ncomp = sm.ncomp
    if sm.tiled:
        rhs = tilemap.zeros(sm.geo_work, map_.dtype)
        div = tilemap.zeros(sm.geo_work.copy(pre=(ncomp, ncomp)), map_.dtype)
    else:
        rhs = enmap.zeros(sm.rhs.shape, sm.rhs.wcs, map_.dtype)
        div = enmap.zeros(sm.div.shape, sm.div.wcs, map_.dtype)
    mwork = sm.to_work(map_)
    for data in mapmaker.data:
        s = strips.get(data.id)
        if s is None or s.det.size == 0:
            continue
        pmap = sm.data[data.id].pmap
        w = data.nmat.ivar[s.det].astype(mapmaker.dtype)
        tod = np.zeros((data.ndet, data.nsamp), mapmaker.dtype)
        sm.forward(data.id, tod, mwork)
        val = s.resid + (1 - s.frac) * tod[s.det, s.left] + s.frac * tod[s.det, s.right]
        tod[:] = 0
        tod[s.det, s.samp] = val * w
        obs_rhs = pmap.zeros()
        pmap.to_map(dest=obs_rhs, signal=tod)
        rhs = rhs.insert(obs_rhs, op=np.ndarray.__iadd__)
        obs_div = pmap.zeros(super_shape=(ncomp, ncomp))
        for i in range(ncomp):
            obs_div[i] = 0
            obs_div[i, i] = 1
            tod[:] = 0
            pmap.from_map(obs_div[i], dest=tod)
            resp = tod[s.det, s.samp]
            tod[:] = 0
            tod[s.det, s.samp] = resp * w
            obs_div[i] = 0
            pmap.to_map(signal=tod, dest=obs_div[i])
        div = div.insert(obs_div, op=np.ndarray.__iadd__)
        del tod, obs_rhs, obs_div
    if sm.tiled:
        active = sm.rhs.geometry.active
        rhs = tilemap.redistribute(rhs, sm.comm, active)
        div = tilemap.redistribute(div, sm.comm, active)
        map_mul = tilemap.map_mul
    else:
        rhs = putils.allreduce(rhs, sm.comm)
        div = putils.allreduce(div, sm.comm)
        map_mul = enmap.map_mul
    tot = sm.div + div
    new = map_mul(smutils.safe_invert_div(tot), map_mul(sm.div, map_) + rhs)
    sel = div[0, 0] > 0
    out = map_.copy()
    out[..., sel] = new[..., sel]
    return out, tot


def create_mapmaker_config(
    defaults: dict = DEPTH1MAPMAKER_DEFAULTS, config_file: Optional[str] = None, args: dict = dict()
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

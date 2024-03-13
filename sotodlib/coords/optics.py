"""
Map positions on physical focal plane to sky using physical optics.
Also includes tools for computing transforms between point clouds.

LAT code adapted from code provided by Simon Dicker.
"""
import logging
from functools import lru_cache, partial
import numpy as np
from scipy.interpolate import interp1d, bisplrep, bisplev
from scipy.spatial.transform import Rotation as R
from sotodlib import core
from sotodlib.io.metadata import read_dataset
from so3g.proj import quat
import yaml

logger = logging.getLogger(__name__)

# Dictionary of zemax tube layout.
# +ve x is to the right as seen from back of the cryostat *need to check*
#           11
#    5    3    4    6
#       1    0    2
#    9    7    8    10
#           12
# Below config assumes a 30 degree rotation
LAT_TUBES = {
    "c1": 0,
    "i1": 3,
    "i2": 1,
    "i3": 7,
    "i4": 8,
    "i5": 2,
    "i6": 4,
    "o1": 5,
    "o2": 9,
    "o3": 12,
    "o4": 10,
    "o5": 6,
    "o6": 11,
}

# SAT Optics
# TODO: Maybe we want these to be provided in a config file?
SAT_X = (0.0, 29.7580, 59.4574, 89.5745, 120.550, 152.821, 163.986, 181.218)
SAT_LON = (
    0.0,
    0.0523597,
    0.10471958,
    0.15707946,
    0.20943951,
    0.26179764,
    0.27925093,
    0.30543087,
)


def _interp_func(x, y, spline):
    xr = np.atleast_1d(x).ravel()
    xa = np.argsort(xr)
    xs = np.argsort(xa)
    yr = np.atleast_1d(y).ravel()
    ya = np.argsort(yr)
    ys = np.argsort(ya)
    z = bisplev(xr[xa], yr[ya], spline)

    if np.isscalar(z):
        if np.isscalar(x):
            return z
        else:
            z = np.atleast_2d(z)
    z = z[(xs, ys)]
    if np.isscalar(x):
        return z[0]

    return z.reshape(x.shape)


def gen_pol_endpoints(x, y, pol):
    """
    Get end points of unit vectors that will be centered on the provided xy positions
    and have the angles specified by pol.

    Arguments:

        x: X positions nominally in mm.

        y: Y positions nominally in mm.

        pol: Angles in degrees.

    Returns:

        x_pol: X postions of endpoints where the even indices are starts
        and the odds are ends.

        y_pol: Y postions of endpoints where the even indices are starts
        and the odds are ends.
    """
    _x = np.cos(np.deg2rad(pol)) / 2
    _y = np.sin(np.deg2rad(pol)) / 2

    x_pol = np.column_stack((x - _x, x + _x)).ravel()
    y_pol = np.column_stack((y - _y, y + _y)).ravel()

    return x_pol, y_pol


def get_gamma(pol_xi, pol_eta):
    """
    Convert xi, eta endpoints to angles that correspond to gamma.

    Arguments:

        pol_xi: 2n xi values where the ones with even indices are
                starting points and odd are ending points where each pair
                forms a vector whose angle is gamma.

        pol_eta: 2n eta values where the ones with even indices are
                 starting points and odd are ending points where each pair
                 forms a vector whose angle is gamma.

    Returns:

        gamma: Array of n gamma values in radians.
    """
    xi = pol_xi.reshape((-1, 2))
    eta = pol_eta.reshape((-1, 2))

    d_xi = np.diff(xi, axis=1).ravel()
    d_eta = np.diff(eta, axis=1).ravel()

    gamma = np.arctan2(d_xi, d_eta) % (2 * np.pi)
    return gamma


@lru_cache(maxsize=None)
def load_ufm_to_fp_config(config_path):
    """
    Load and cache config file with the parameters to transform from UFM to focal_plane coordinates.

    Arguments:

        config_path: Path to the yaml config file.

    Returns:

        config: Dictionairy containing the config information.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@lru_cache(maxsize=None)
def get_ufm_to_fp_pars(telescope_flavor, wafer_slot, config_path):
    """
    Get (and cache) the parameters to transform from UFM to focal plane coordinates
    for a specific slot of a given telescope's focal plane.

    Arguments:

        telescope_flavor: The telescope, should be LAT or SAT.

        wafer_slot: The wafer slot to get parameters for.

        config_path: Path to the yaml with the parameters.

    Returns:

        transform_pars: Dict of transformation parameters that can be passed to ufm_to_fp.
    """
    config = load_ufm_to_fp_config(config_path)
    return config[telescope_flavor][wafer_slot]


def ufm_to_fp(aman, x=None, y=None, pol=None, theta=0, dx=0, dy=0):
    """
    Transform from coords internal to wafer to focal plane coordinates.

    Arguments:

        aman: AxisManager assumed to contain aman.det_info.wafer.
              If provided outputs will be wrapped into aman.focal_plane.

        x: X position in wafer's internal coordinate system in mm.
           If provided overrides the value from aman.

        y: Y position in wafer's internal coordinate system in mm.
           If provided overrides the value from aman.

        pol: Polarization angle in wafer's internal coordinate system in deg.
           If provided overrides the value from aman.

        theta: Internal rotation of the UFM in degrees.

        dx: X offset in mm.

        dy: Y offset in mm.

    Returns:

        x_fp: X position on focal plane.

        y_fp: Y position on focal plane.

        pol_fp: Pol angle on focal plane.
    """
    if x is None:
        x = aman.det_info.wafer.det_x
    if y is None:
        y = aman.det_info.wafer.det_y
    if pol is None:
        pol = aman.det_info.wafer.angle
    xy = np.column_stack((x, y, np.zeros_like(x)))

    rot = R.from_euler("z", theta, degrees=True)
    xy = rot.apply(xy)

    x_fp = xy[:, 0] + dx
    y_fp = xy[:, 1] + dy
    pol_fp = pol + theta

    if aman is not None:
        focal_plane = core.AxisManager(aman.dets)
        focal_plane.wrap("x_fp", x_fp, [(0, focal_plane.dets)])
        focal_plane.wrap("y_fp", y_fp, [(0, focal_plane.dets)])
        focal_plane.wrap("pol_fp", pol_fp, [(0, focal_plane.dets)])
        if "focal_plane" in aman:
            aman.focal_plane.merge(focal_plane)
        else:
            aman.wrap("focal_plane", focal_plane)

    return x_fp, y_fp, pol_fp


def LAT_pix2sky(x, y, sec2elev, sec2xel, array2secx, array2secy, rot=0, opt2cryo=30.0):
    """
    Routine to map pixels from arrays to sky.

    Arguments:

        x: X position on focal plane (currently zemax coord)

        y: Y position on focal plane (currently zemax coord)

        sec2elev: Function that maps positions on secondary to on sky elevation

        sex2xel: Function that maps positions on secondary to on sky xel.

        array2secx: Function that maps positions on tube's focal plane to x position on secondary.

        array2secy: Function that maps positions on tube's focal plane to y position on secondary.

        rot: Rotation about the line of site = elev - 60 - corotator.

        opt2cryo: The rotation to get from cryostat coordinates to zemax coordinates (TBD, prob 30 deg).

    Returns:

        elev: The on sky elevation in radians.

        xel: The on sky xel in radians.
    """
    d2r = np.pi / 180.0
    # TBD - put in check for MASK - values outside circle should not be allowed
    # get into zemax coord
    xz = x * np.cos(d2r * opt2cryo) - y * np.sin(d2r * opt2cryo)
    yz = y * np.cos(d2r * opt2cryo) + x * np.sin(d2r * opt2cryo)
    # Where is it on (zemax secondary focal plane wrt LATR)
    xs = array2secx(xz, yz)
    ys = array2secy(xz, yz)
    # get into LAT zemax coord
    # There is a 90 degree offset between how zemax coords and SO coords are defined
    rot -= 90
    xrot = xs * np.cos(d2r * rot) - ys * np.sin(d2r * rot)
    yrot = ys * np.cos(d2r * rot) + xs * np.sin(d2r * rot)
    # note these are around the telescope boresight
    elev = sec2elev(xrot, yrot)
    xel = sec2xel(xrot, yrot)

    return np.deg2rad(elev), np.deg2rad(xel)


@lru_cache(maxsize=None)
def load_zemax(path):
    """
    Load zemax_dat from path

    Arguments:

        path: Path to zemax data

    Returns:

        zemax_dat: Dictionairy with data from zemax
    """
    try:
        zemax_dat = np.load(path, allow_pickle=True)
    except Exception as e:
        logger.error("Can't load data from " + path)
        raise e
    return dict(zemax_dat)


@lru_cache(maxsize=None)
def LAT_optics(zemax_path):
    """
    Compute mapping from LAT secondary to sky.

    Arguments:

        zemax_path: Path to LAT optics data from zemax.

    Returns:

        sec2elev: Function that maps positions on secondary to on sky elevation

        sex2xel: Function that maps positions on secondary to on sky xel.
    """
    zemax_dat = load_zemax(zemax_path)
    try:
        LAT = zemax_dat["LAT"][()]
    except Exception as e:
        logger.error("LAT key missing from dictionary")
        raise e

    gi = np.where(LAT["mask"] != 0.0)
    x = LAT["x"][gi].ravel()
    y = LAT["y"][gi].ravel()
    elev = LAT["elev"][gi].ravel()
    xel = LAT["xel"][gi].ravel()

    s2e = bisplrep(x, y, elev, kx=3, ky=3)
    sec2elev = partial(_interp_func, spline=s2e)
    s2x = bisplrep(x, y, xel, kx=3, ky=3)
    sec2xel = partial(_interp_func, spline=s2x)

    return sec2elev, sec2xel


@lru_cache(maxsize=None)
def LATR_optics(zemax_path, tube_slot):
    """
    Compute mapping from LAT secondary to sky.

    Arguments:

        zemax_path: Path to LATR optics data from zemax.

        tube_slot: Either the tube name as a string or the tube number as an int.

    Returns:

        array2secx: Function that maps positions on tube's focal plane to x position on secondary.

        array2secy: Function that maps positions on tube's focal plane to y position on secondary.
    """
    zemax_dat = load_zemax(zemax_path)
    try:
        LATR = zemax_dat["LATR"][()]
    except Exception as e:
        logger.error("LATR key missing from dictionary")
        raise e

    if isinstance(tube_slot, str):
        tube_name = tube_slot
        try:
            tube_num = LAT_TUBES[tube_slot]
        except Exception as e:
            logger.error("Invalid tube name")
            raise e
    elif isinstance(tube_slot, int):
        tube_num = tube_slot
        try:
            tube_name = list(LAT_TUBES.keys())[tube_num]
        except Exception as e:
            logger.error("Invalid tube number")
            raise e
    else:
        raise ValueError("Invalid tube_slot")

    logger.info("Working on LAT tube " + tube_name)
    gi = np.where(LATR[tube_num]["mask"] != 0)

    array_x = LATR[tube_num]["array_x"][gi].ravel()
    array_y = LATR[tube_num]["array_y"][gi].ravel()
    sec_x = LATR[tube_num]["sec_x"][gi].ravel()
    sec_y = LATR[tube_num]["sec_y"][gi].ravel()

    a2x = bisplrep(array_x, array_y, sec_x, kx=3, ky=3)
    array2secx = partial(_interp_func, spline=a2x)
    a2y = bisplrep(array_x, array_y, sec_y, kx=3, ky=3)
    array2secy = partial(_interp_func, spline=a2y)

    return array2secx, array2secy


def LAT_focal_plane(aman, zemax_path, x=None, y=None, pol=None, roll=0, tube_slot="c1"):
    """
    Compute focal plane for a wafer in the LAT.

    Arguments:

        aman: AxisManager nominally containing aman.focal_plane.x_fp and aman.focal_plane.y_fp.
              If provided focal plane will be stored in aman.focal_plane.

        zemax_path: Path to LATR optics data from zemax.

        x: Detector x positions, if provided will override positions loaded from aman.

        y: Detector y positions, if provided will override positions loaded from aman.

        pol: Detector polarization angle, if provided will override positions loaded from aman.

        roll: Rotation about the line of site = elev - 60 - corotator.

        tube_slot: Either the tube name as a string or the tube number as an int.

    Returns:

        xi: Detector elev on sky from physical optics in radians.
            If aman is provided then will be wrapped as aman.focal_plane.xi.

        eta: Detector xel on sky from physical optics in radians.
             If aman is provided then will be wrapped as aman.focal_plane.eta.

        gamma: Detector gamma on sky from physical optics in radians.
               If aman is provided then will be wrapped as aman.focal_plane.eta.
    """
    if x is None:
        x = aman.focal_plane.x_fp
    if y is None:
        y = aman.focal_plane.y_fp
    if pol is None:
        pol = aman.focal_plane.pol_fp

    sec2elev, sec2xel = LAT_optics(zemax_path)
    array2secx, array2secy = LATR_optics(zemax_path, tube_slot)

    xi, eta = LAT_pix2sky(x, y, sec2elev, sec2xel, array2secx, array2secy, roll)

    pol_x, pol_y = gen_pol_endpoints(x, y, pol)
    pol_xi, pol_eta = LAT_pix2sky(
        pol_x, pol_y, sec2elev, sec2xel, array2secx, array2secy, roll
    )
    gamma = get_gamma(pol_xi, pol_eta)

    if aman is not None:
        focal_plane = core.AxisManager(aman.dets)
        focal_plane.wrap("xi", xi, [(0, focal_plane.dets)])
        focal_plane.wrap("eta", eta, [(0, focal_plane.dets)])
        focal_plane.wrap("gamma", gamma, [(0, focal_plane.dets)])
        if "focal_plane" in aman:
            aman.focal_plane.merge(focal_plane)
        else:
            aman.wrap("focal_plane", focal_plane)

    return xi, eta, gamma


@lru_cache(maxsize=None)
def sat_to_sky(x, theta):
    """
    Interpolate x and theta values to create mapping from SAT focal plane to sky.
    This function is a wrapper whose main purpose is the cache this mapping.

    Arguments:
        x: X values in mm, should be all positive.

        theta: Theta values in radians, should be all positive.
               Theta is defined by ISO coordinates.

    Return:
        sat_to_sky: Interp object with the mapping from the focal plane to sky.
    """
    return interp1d(x, theta, fill_value="extrapolate")


def SAT_focal_plane(aman, x=None, y=None, pol=None, roll=0, mapping_data=None):
    """
    Compute focal plane for a wafer in the SAT.

    Arguments:

        aman: AxisManager nominally containing aman.focal_plane.x_fp and aman.focal_plane.y_fp.
              If provided focal plane will be stored in aman.focal_plane.

        x: Detector x positions, if provided will override positions loaded from aman.

        y: Detector y positions, if provided will override positions loaded from aman.

        pol: Detector polarization angle, if provided will override positions loaded from aman.

        roll: Rotation about the line of site = -1*boresight.

        mapping_data: Tuple of (x, theta) that can be interpolated to map the focal plane to the sky.
                      Leave as None to use the default mapping.

    Returns:

        xi: Detector elev on sky from physical optics in radians.
            If aman is provided then will be wrapped as aman.focal_plane.xi.

        eta: Detector xel on sky from physical optics in radians.
             If aman is provided then will be wrapped as aman.focal_plane.eta.

        gamma: Detector gamma on sky from physical optics in radians.
               If aman is provided then will be wrapped as aman.focal_plane.eta.
    """
    if x is None:
        x = aman.focal_plane.x_fp
    if y is None:
        y = aman.focal_plane.y_fp
    if pol is None:
        pol = aman.focal_plane.pol_fp

    if mapping_data is None:
        fp_to_sky = sat_to_sky(SAT_X, SAT_LON)
    else:
        mapping_data = (tuple(val) for val in mapping_data)
        fp_to_sky = sat_to_sky(*mapping_data)

    # NOTE: lonlat coords are naturally centered at (1, 0, 0) and
    #       xieta at (0, 0, 1). The euler angle below does this recentering
    #       as well as flipping the sign of eta.
    #       There is also a sign flip of xi that is supresses the factor of
    #       -1 that would normally be applied when calculating lon since
    #       it has the opposite sign as x.
    #       The sign flips perform the flip about the origin from optics.
    minus_lon = np.sign(x) * fp_to_sky(np.abs(x))
    lat = np.sign(y) * fp_to_sky(np.abs(y))
    _xi, _eta, _ = quat.decompose_xieta(
        quat.euler(1, np.deg2rad(90)) * quat.rotation_lonlat(minus_lon, lat)
    )
    xi = _xi * np.cos(np.deg2rad(roll)) - _eta * np.sin(np.deg2rad(roll))
    eta = _eta * np.cos(np.deg2rad(roll)) + _xi * np.sin(np.deg2rad(roll))

    pol_x, pol_y = gen_pol_endpoints(x, y, pol)
    pol_minus_lon = np.sign(pol_x) * fp_to_sky(np.abs(pol_x))
    pol_lat = np.sign(pol_y) * fp_to_sky(np.abs(pol_y))
    _xi, _eta, _ = quat.decompose_xieta(
        quat.euler(1, np.deg2rad(90)) * quat.rotation_lonlat(pol_minus_lon, pol_lat)
    )
    pol_xi = _xi * np.cos(np.deg2rad(roll)) - _eta * np.sin(np.deg2rad(roll))
    pol_eta = _eta * np.cos(np.deg2rad(roll)) + _xi * np.sin(np.deg2rad(roll))
    gamma = get_gamma(pol_xi, pol_eta)

    if aman is not None:
        focal_plane = core.AxisManager(aman.dets)
        focal_plane.wrap("xi", xi, [(0, focal_plane.dets)])
        focal_plane.wrap("eta", eta, [(0, focal_plane.dets)])
        focal_plane.wrap("gamma", gamma, [(0, focal_plane.dets)])
        if "focal_plane" in aman:
            aman.focal_plane.merge(focal_plane)
        else:
            aman.wrap("focal_plane", focal_plane)

    return xi, eta, gamma


def get_focal_plane(
    aman,
    x=None,
    y=None,
    pol=None,
    roll=0,
    telescope_flavor=None,
    tube_slot=None,
    wafer_slot="ws0",
    config_path=None,
    zemax_path=None,
    mapping_data=None,
    return_fp=False,
):
    """
    Unified interface for LAT_focal_plane and SAT_focal_plane including the step of applying ufm_to_fp.

    Arguments:

        aman: AxisManager nominally containing aman.det_info.wafer.det_x and aman.det_info.wafer.det_y.
              If provided focal plane will be stored in aman.focal_plane.

        x: Detector x positions, if provided will override positions loaded from aman.

        y: Detector y positions, if provided will override positions loaded from aman.

        pol: Detector polarization angle, if provided will override positions loaded from aman.

        roll: Rotation about the line of sight.
              For the LAT this is elev - 60 - corotator.
              For the SAT this is -1*boresight.

        telescope_flavor: What the telescope flavor is ('LAT' or 'SAT').
                          Note that this is case insenstive.
                          If None and aman contains obs_info, will be extracted from there.

        tube_slot: Which optics tube of the telescope to use.
                   Only used by the LAT.
                   If None and aman contains obs_info, will be extracted from there.

        wafer_slot: Which wafer slot to use.

        config_path: Path to the ufm_to_fp config file.

        zemax_path: Path to the data file from Zemax.
                    Only used by the LAT.

        mapping_data: Tuple of (x, theta) that can be interpolated to map the focal plane to the sky.
                      Leave as None to use the default mapping.
                      Only used by the SAT.

        return_fp: If True also return the transformed focal plane in mm.

    Returns:

        xi: Detector elev on sky from physical optics in radians.
            If aman is provided then will be wrapped as aman.focal_plane.xi.

        eta: Detector xel on sky from physical optics in radians.
             If aman is provided then will be wrapped as aman.focal_plane.eta.

        gamma: Detector gamma on sky from physical optics in radians.
               If aman is provided then will be wrapped as aman.focal_plane.eta.

        x_fp: X position on focal plane.
              Only returned if return_fp is True.

        y_fp: Y position on focal plane.
              Only returned if return_fp is True.

        pol_fp: Pol angle on focal plane.
                Only returned if return_fp is True.
    """
    if aman is not None and "obs_info" in aman:
        if telescope_flavor is None:
            telescope_flavor = aman.obs_info.telescope_flavor
        if tube_slot is None:
            tube_slot = aman.obs_info.tube_slot
    if telescope_flavor is None or tube_slot is None:
        raise ValueError("Telescope or tube is None and unable to extract from aman")
    telescope_flavor = telescope_flavor.upper()
    if telescope_flavor not in ["LAT", "SAT"]:
        raise ValueError("Telescope should be LAT or SAT")

    ufm_to_fp_pars = get_ufm_to_fp_pars(telescope_flavor, wafer_slot, config_path)
    x_fp, y_fp, pol_fp = ufm_to_fp(aman, x=x, y=y, pol=pol, **ufm_to_fp_pars)
    if telescope_flavor == "LAT":
        if zemax_path is None:
            raise ValueError("Must provide zemax_path for LAT")
        xi, eta, gamma = LAT_focal_plane(
            aman, zemax_path, x=x_fp, y=y_fp, pol=pol_fp, roll=roll, tube_slot=tube_slot
        )
    else:
        xi, eta, gamma = SAT_focal_plane(
            aman, x=x_fp, y=y_fp, pol=pol_fp, roll=roll, mapping_data=mapping_data
        )

    if return_fp:
        return xi, eta, gamma, x_fp, y_fp, pol_fp
    return xi, eta, gamma


def gen_template(wafer_info, stream_id, **pointing_cfg):
    """
    Generate a pointing template from a wafer info ResultSet.

    Arguments:

        wafer_info: Either the path to a wafer_info ResultSet
                    or an open h5py File object.

        stream_id : The UFM to generate the template for (ie: ufm_mv5).

        **pointing_cfg: Arguments to pass to get_focal_plane,
                        should be arguments from rot onwards.

    Returns:

        template_det_ids: The detector ids of the detectors in the template.

        template: (ndet, 3) array of (xi, eta, gamma) for all dets in template.

        is_optical: (ndet) mask that is True for all optical dets.
    """
    ufm = stream_id.split("_")[-1].title()
    wafer = read_dataset(wafer_info, ufm)
    template_det_ids = wafer["dets:det_id"]
    det_x = wafer["dets:wafer.x"]
    det_y = wafer["dets:wafer.y"]
    det_pol = wafer["dets:wafer.angle"]
    det_type = wafer["dets:wafer.type"]
    is_optical = np.array(det_type) == "OPTC"

    xi, eta, gamma = get_focal_plane(
        None, x=det_x, y=det_y, pol=det_pol, **pointing_cfg
    )
    template = np.column_stack((xi, eta, gamma))

    # Dark dets should not be allowed to have pointing
    template[~is_optical] = np.nan

    return np.array(template_det_ids), template, is_optical

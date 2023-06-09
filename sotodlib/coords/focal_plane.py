"""
Map positions on physical focal plane to sky using physical optics.
Currently only works for the LAT.

LAT code adapted from code provided by Simon Dicker.
"""
import logging
from functools import lru_cache, partial
import numpy as np
from scipy.interpolate import interp1d, bisplrep, bisplev
from scipy.spatial.transform import Rotation as R
from sotodlib import core
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
    "c": 0,
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
SAT_THETA = (
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
    z = z[(xs, ys)]

    if np.isscalar(x):
        return z[0]
    return z.reshape(x.shape)


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
def get_ufm_to_fp_pars(telescope, slot, config_path):
    """
    Get (and cache) the parameters to transform from UFM to focal plane coordinates
    for a specific slot of a given telescope's focal plane.

    Arguments:

        telescope: The telescope, should be LAT or SAT.

        slot: The UFM slot to get parameters for.

        config_path: Path to the yaml with the parameters.

    Returns:

        transform_pars: Dictionairy of transformation parameters that can be passed to ufm_to_fp.
    """
    config = load_ufm_to_fp_config(config_path)
    return config[telescope][slot]


def ufm_to_fp(aman, x=None, y=None, theta=0, dx=0, dy=0):
    """
    Transform from coords internal to wafer to focal plane coordinates.

    Arguments:

        aman: AxisManager assumed to contain aman.det_info.wafer.
              If provided outputs will be wrapped into aman.focal_plane.

        x: X position in wafer's internal coordinate system in mm.
           If provided overrides the value from aman.

        y: Y position in wafer's internal coordinate system in mm.
           If provided overrides the value from aman.

        theta: Internal rotation of the UFM in degrees.

        dx: X offset in mm.

        dy: Y offset in mm.

    Returns:

        x_fp: X position on focal plane.

        y_fp: Y position on focal plane.
    """
    if x is None:
        x = aman.det_info.wafer.det_x
    if y is None:
        y = aman.det_info.wafer.det_y
    xy = np.column_stack((x, y, np.zeros_like(x)))

    rot = R.from_euler("z", theta, degrees=True)
    xy = rot.apply(xy)

    x_fp = xy[:, 0] + dx
    y_fp = xy[:, 1] + dy

    if aman is not None:
        focal_plane = core.AxisManager(aman.dets)
        focal_plane.wrap("x_fp", x_fp, [(0, focal_plane.dets)])
        focal_plane.wrap("y_fp", x_fp, [(0, focal_plane.dets)])
        if "focal_plane" in aman:
            aman.focal_plane.merge(focal_plane)
        else:
            aman.wrap("focal_plane", focal_plane)

    return x_fp, y_fp


def LAT_pix2sky(x, y, sec2elev, sec2xel, array2secx, array2secy, rot=0, opt2cryo=0.0):
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
    # We may need to add a rotation offset here to account for physical vs ZEMAX
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
def LATR_optics(zemax_path, tube):
    """
    Compute mapping from LAT secondary to sky.

    Arguments:

        zemax_path: Path to LATR optics data from zemax.

        tube: Either the tube name as a string or the tube number as an int.

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

    if isinstance(tube, str):
        tube_name = tube
        try:
            tube_num = LAT_TUBES[tube]
        except Exception as e:
            logger.error("Invalid tube name")
            raise e
    elif isinstance(tube, int):
        tube_num = tube
        try:
            tube_name = list(LAT_TUBES.keys())[tube_num]
        except Exception as e:
            logger.error("Invalid tube number")
            raise e

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


def LAT_focal_plane(aman, zemax_path, x=None, y=None, rot=0, tube="c"):
    """
    Compute focal plane for a wafer in the LAT.

    Arguments:

        aman: AxisManager nominally containing aman.focal_plane.x_fp and aman.focal_plane.y_fp.
              If provided focal plane will be stored in aman.focal_plane.

        zemax_path: Path to LATR optics data from zemax.

        x: Detector x positions, if provided will override positions loaded from aman.

        y: Detector y positions, if provided will override positions loaded from aman.

        rot: Rotation about the line of site = elev - 60 - corotator.

        tube: Either the tube name as a string or the tube number as an int.

    Returns:

        xi: Detector elev on sky from physical optics in radians.
            If aman is provided then will be wrapped as aman.focal_plane.xi.

        eta: Detector xel on sky from physical optics in radians.
             If aman is provided then will be wrapped as aman.focal_plane.eta.
    """
    if x is None:
        x = aman.focal_plane.x_fp
    if y is None:
        y = aman.focal_plane.y_fp

    sec2elev, sec2xel = LAT_optics(zemax_path)
    array2secx, array2secy = LATR_optics(zemax_path, tube)

    xi, eta = LAT_pix2sky(x, y, sec2elev, sec2xel, array2secx, array2secy, rot)

    if aman is not None:
        focal_plane = core.AxisManager(aman.dets)
        focal_plane.wrap("xi", xi, [(0, focal_plane.dets)])
        focal_plane.wrap("eta", eta, [(0, focal_plane.dets)])
        aman.wrap("focal_plane", focal_plane)

    return xi, eta


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


def SAT_focal_plane(aman, x=None, y=None, mapping_data=None):
    """
    Compute focal plane for a wafer in the SAT.

    Arguments:

        aman: AxisManager nominally containing aman.focal_plane.x_fp and aman.focal_plane.y_fp.
              If provided focal plane will be stored in aman.focal_plane.

        x: Detector x positions, if provided will override positions loaded from aman.

        y: Detector y positions, if provided will override positions loaded from aman.

        mapping_data: Tuple of (x, theta) that can be interpolated to map the focal plane to the sky.
                      Leave as None to use the default mapping.

    Returns:

        xi: Detector elev on sky from physical optics in radians.
            If aman is provided then will be wrapped as aman.focal_plane.xi.

        eta: Detector xel on sky from physical optics in radians.
             If aman is provided then will be wrapped as aman.focal_plane.eta.
    """
    if x is None:
        x = aman.focal_plane.x_fp
    if y is None:
        y = aman.focal_plane.y_fp

    if mapping_data is None:
        fp_to_sky = sat_to_sky(SAT_X, SAT_THETA)
    else:
        mapping_data = (tuple(val) for val in mapping_data)
        fp_to_sky = sat_to_sky(*mapping_data)
    # NOTE: The -1 does the flip about the origin
    theta = -1 * np.sign(x) * fp_to_sky(np.abs(x))
    phi = -1 * np.sign(y) * fp_to_sky(np.abs(y))
    xi, eta, gamma = quat.decompose_xieta(quat.rotation_iso(theta, phi))

    if aman is not None:
        focal_plane = core.AxisManager(aman.dets)
        focal_plane.wrap("xi", xi, [(0, focal_plane.dets)])
        focal_plane.wrap("eta", eta, [(0, focal_plane.dets)])
        if "focal_plane" in aman:
            aman.focal_plane.merge(focal_plane)
        else:
            aman.wrap("focal_plane", focal_plane)

    return xi, eta

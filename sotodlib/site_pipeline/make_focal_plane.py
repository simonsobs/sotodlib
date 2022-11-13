import argparse as ap
import numpy as np
import pandas as pd
from sotodlib.core import AxisManager
import sotodlib.io.g3tsmurf_utils as g3u
from scipy.spatial.transform import Rotation as R

from pycpd import AffineRegistration, ConstrainedDeformableRegistration


def LAT_coord_transform(xy, rot_fp, rot_ufm, r=72.645):
    """
    Transform from instrument model coords to LAT Zemax coords

    Arguments:

        xy: XY coords from instrument model.
            Should be a (2, n) array.

        rot_fp: Angle of array location on focal plane in deg.

        rot_ufm: Rotatation of UFM about its center.

    Returns:

        xy_trans: Transformed coords.
    """
    xy_trans = np.zeros((xy.shape[1], 3))
    xy_trans[:, :2] = xy.T

    r1 = R.from_euler("z", rot_fp, degrees=True)
    shift = r1.apply(np.array([r, 0, 0]))

    r2 = R.from_euler("z", rot_ufm, degrees=True)
    xy_trans = r2.apply(xy_trans) + shift

    return xy_trans.T[:2]


def rescale(xy):
    """
    Rescale pointing or template to [0, 1]

    Arguments:

        xy: Pointing or template, should have two columns.

    Returns:

        xy_rs: Rescaled array.
    """
    xy_rs = xy.copy()
    xy_rs[:, 0] /= xy[:, 0].max() - xy[:, 0].min()
    xy_rs[:, 0] -= xy_rs[:, 0].min()
    xy_rs[:, 1] /= xy[:, 1].max() - xy[:, 1].min()
    xy_rs[:, 1] -= xy_rs[:, 1].min()
    return xy_rs


def gen_priors(aman, prior, method="flat", width=1, basis=None):
    """
    Generate priors from detmap.
    Currently priors will be 1 everywhere but the location of the detmap's match.
    More complicated prior generation will be a part of this function eventually.

    Arguments:

        aman: AxisManager assumed to contain aman.det_info.det_id and aman.det_info.wafer.

        prior: Prior value at locations from the detmap.
               Should be greater than 1.

        method: What sort of priors to implement.
                Currently only 'flat' is accepted but at least 'gaussian' will be implemented later.

        width: Width of priors. When gaussian priors are added this will be sigma.

        basis: Basis to calculate width in.
               Currently not implemented so width will just be along the dets axis.
               At the very least radial distance will be added.

    Returns:

        priors: The 2d array of priors.
    """

    def _flat(arr, idx):
        arr[idx - width // 2 : idx + width // 2 + width % 2] = prior

    if method == "flat":
        prior_method = _flat
    else:
        raise ValueError("Method " + method + " not implemented")

    priors = np.ones((aman.dets.count, aman.dets.count))
    for i in aman.dets.count:
        priors[i] = prior_method(priors[i], i, width, prior)

    return priors


def match_template(
    focal_plane, template, out_thresh=0, avoid_collision=True, priors=None
):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Focal plane after either having optics model applied or being fit with channel map.
                     Should have columns: x, y or x, y, pol.

        template: Table containing template of focal plane.
                  Should have columns: x, y or x, y, pol.

        out_thresh: Threshold at which points will be considered outliers.
                    Should be in range [0, 1) and is checked against the
                    probability that a point matches its mapped point in the template.

        avoid_collision: Try to avoid collisions. May effect performance.

        priors: Priors to apply when matching.
                Should be be a n by n array where n is the number of points.
                The priors[i, j] is the prior on the i'th point in template matching the j'th point in focal_plane.

    Returns:

        mapping: Mapping between elements in template and focal_plane.
                 focal_plane[i] = template[mapping[i]]

        outliers: Indices of points that are outliers.
                  Note that this is in the basis of mapping and focal_plane, not template.
    """
    reg = AffineRegistration(**{"X": focal_plane, "Y": template})
    reg.register()
    P = reg.P

    if priors is not None:
        P *= priors
    if avoid_collision:
        # This should get the maximum probability without collisions
        inv = np.linalg.pinv(P)
        mapping = np.argmax(inv, axis=0)
    else:
        mapping = np.argmax(P, axis=1)

    outliers = np.array([])
    if out_thresh > 0:
        outliers = np.where(reg.P[range(reg.P.shape[0]), mapping] < out_thresh)[0]

    return mapping, outliers


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    # NOTE: Eventually all of this should just be metadata I can load from a single context?

    # Making some assumtions about pointing data that aren't currently true:
    # 1. I am assuming that the HDF5 file is a saved aman not a pandas results set
    # 2. I am assuming that it contains the measured detector polangs
    # 3. I am assuming it comtains aman.det_info
    parser.add_argument(
        "pointing_data",
        help="Location of HDF5 file containing pointing for each readout channel",
    )
    parser.add_argument(
        "detmap",
        help="Location of detmap file",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        help="Instrument mapping from sky to focal plane, feature not currently implemented",
    )
    args = parser.parse_args()

    # Load data
    aman = AxisManager.load(args.pointing_data)
    g3u.add_detmap_info(aman, args.detmap)

    # Apply instrument to pointing if availible
    # Otherwise try to match from channel map

    # Apply template
    # Get pixel for each channel

    # Set band based on chan map
    # Set pol from wiregrid

    # Add det_id to pointing table
    # Just an f string?
    # Add band and pol as well?
    # Save table


if __name__ == "__main__":
    main()

import sys
import argparse as ap
import numpy as np
import sotodlib.io.g3tsmurf_utils as g3u
from sotodlib.core import AxisManager
from pycpd import AffineRegistration


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
        prior_method(priors[i], i)

    return priors


def match_template(
    focal_plane, template, out_thresh=0, avoid_collision=True, priors=None
):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Measured pointing and optionally polarization angle.
                     Should be a (2, n) or (3, n) array with collums: xi, eta, pol.
                     Optionally an optics model can be preapplied to this to map the pointing
                     onto the physical focal plane in which case the collums are: x, y, pol.

        template: Designed x, y, and polarization angle of each detector.
                  Should be a (2, n) or (3, n) array with collums: x, y, pol.

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
    parser.add_argument("detmap", help="Location of detmap file")
    parser.add_argument("bias_map", help="Location of bias map file")
    parser.add_argument(
        "-i",
        "--instrument",
        help="Instrument mapping from sky to focal plane, feature not currently implemented",
    )
    # Should this point to a config file?
    parser.add_argument(
        "-p", "--priors", help="Amount to set prior to from detmap", default=1
    )
    parser.add_argument(
        "-nf",
        "--no_fit",
        action="store_true",
        help="Don't try to fit for mapping between detector positions on array and on sky, just directly copy det_id from detmap",
    )
    args = parser.parse_args()

    # Load data
    aman = AxisManager.load(args.pointing_data)
    g3u.add_detmap_info(aman, args.detmap)
    bg_map = np.load(args.bias_map, allow_pickle=True).item()

    if args.no_fit:
        aman.wrap("det_id", aman.det_info.det_id, [(0, aman.dets)])
        g3u.remove_detmap_info(aman)
        aman.save(args.pointing_data, overwrite=True)
        sys.exit()

    # TODO: apply instrument to pointing if availible

    # Split up by bandpass
    bc_aman = (
        aman.det_info.smurf.band.astype(int) << 32
    ) + aman.det_info.smurf.channel.astype(int)
    bc_bgmap = (bg_map["bands"] << 32) + bg_map["channels"]
    to_add = np.setdiff1d(bc_aman, bc_bgmap)
    to_remove = np.setdiff1d(bc_bgmap, bc_aman)
    msk = ~np.isin(bc_bgmap, to_remove)
    bg_map["bgmap"] = np.append(bg_map["bgmap"], -2 * np.ones(len(to_add)))
    bc_bgmap = np.append(bc_bgmap[msk], to_add)
    idx = np.argsort(bc_bgmap)[np.argsort(bc_aman)]
    bias_group = bg_map["bg_map"][idx]

    msk_bp1 = (
        (bias_group == 0)
        | (bias_group == 1)
        | (bias_group == 4)
        | (bias_group == 5)
        | (bias_group == 8)
        | (bias_group == 9)
    )
    msk_bp2 = (
        (bias_group == 2)
        | (bias_group == 3)
        | (bias_group == 6)
        | (bias_group == 7)
        | (bias_group == 10)
        | (bias_group == 11)
    )

    # Prep inputs
    priors = gen_priors(aman, args.prior, method="flat", width=1, basis=None)
    focal_plane = np.vstack((aman.xi, aman.eta, aman.polang))
    template = np.vstack(
        (
            aman.det_info.wafer.det_x,
            aman.det_info.wafer.det_y,
            aman.det_info.wafer.angle,
        )
    )

    # Do actual matching
    map_bp1, out_bp1 = match_template(
        focal_plane[:, msk_bp1],
        template[:, msk_bp1],
        out_thresh=0,
        avoid_collision=True,
        priors=priors[np.ix_(msk_bp1, msk_bp1)],
    )
    map_bp2, out_bp2 = match_template(
        focal_plane[:, msk_bp2],
        template[:, msk_bp2],
        out_thresh=0,
        avoid_collision=True,
        priors=priors[np.ix_(msk_bp1, msk_bp1)],
    )
    det_id = np.zeros(aman.dets.count, dtype=str)
    det_id[msk_bp1] = aman.det_info.det_id[msk_bp1][map_bp1]
    det_id[msk_bp2] = aman.det_info.det_id[msk_bp2][map_bp2]
    aman.wrap("det_id", det_id, [(0, aman.dets)])
    # TODO: Figure out what to do about outliers

    g3u.remove_detmap_info(aman)
    aman.save(args.pointing_data, overwrite=True)


if __name__ == "__main__":
    main()

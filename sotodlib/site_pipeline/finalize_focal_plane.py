import os
import sys
import argparse as ap
import numpy as np
import scipy.linalg as la
import yaml
import sotodlib.io.g3tsmurf_utils as g3u
from sotodlib.core import AxisManager, metadata, Context
from sotodlib.io.metadata import write_dataset
from sotodlib.site_pipeline import util
from sotodlib.coords import focal_plane as fpc

logger = util.init_logger(__name__, "finalize_focal_plane: ")


def get_affine(src, dst):
    """
    Get affine transformation between two point clouds.
    Transformation is dst = affine@src + shift

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

    Returns:

        affine: The transformation matrix.

        shift: Shift to apply after transformation.
    """
    M = np.vstack(
        (src - np.median(src, axis=1)[:, None], dst - np.median(dst, axis=1)[:, None])
    ).T
    u, s, vh = la.svd(M)
    vh_splits = [
        quad for half in np.split(vh.T, 2, axis=0) for quad in np.split(half, 2, axis=1)
    ]
    affine = np.dot(vh_splits[2], la.pinv(vh_splits[0]))

    transformed = affine @ src
    shift = np.median(dst - transformed, axis=1)

    return affine, shift


def decompose_affine(affine):
    """
    Decompose an affine transformation into its components.
    Note that this currently only works on a 2x2 matrix.

    Arguments:

        affine: The 2x2 affine transformation matrix.

    Returns:

        scale_0: The scale in the first dimension.

        scale_1: The scale in the second dimension.

        shear: The shear parameter.

        rot: The rotation angle in radians.
    """
    scale_0 = np.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2)
    rot = np.arctan2(affine[1, 0], affine[0, 0])

    ms = affine[0, 1] * np.cos(rot) + affine[1, 1] * np.sin(rot)
    if np.isclose(0, np.sin(rot)):
        scale_1 = (affine[1, 1] - ms * np.sin(rot)) / np.cos(rot)
    else:
        scale_1 = (ms * np.cos(rot) - affine[0, 1]) / np.sin(rot)

    shear = ms / scale_1

    return shear, scale_0, scale_1, rot


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load context
    # TODO: Currently this is just setup to load the single obs results
    # Need a way to load multi obs results
    ctx = Context(config["context"]["path"])
    name = config["context"]["position_match"]
    query = []
    if "query" in config["context"]:
        query = (ctx.obsdb.query(config["context"]["query"])["obs_id"],)
    obs_ids = np.append(config["context"].get("obs_ids", []), query)
    obs_ids = np.unique(obs_ids)
    if len(obs_ids) == 0:
        raise ValueError("No observations provided in configuration")

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    outpath = os.path.join(config["outdir"], f"{ufm}{append}.h5")
    dataset = "focal_plane"
    outpath = os.path.abspath(outpath)

    avg_fp = {}
    all_skipped = True
    for obs_id, detmap in zip(obs_ids, config["detmaps"]):
        logger.info("Loading information from observation " + obs_id)

        # Load data
        aman = ctx.get_meta(obs_id, dets=config["context"].get("dets", {}))
        if name not in aman:
            logger.warning(
                "\tNo position_match associated with this observation. Skipping."
            )
            continue

        # Put SMuRF band channel in the correct place
        smurf = AxisManager(aman.dets)
        smurf.wrap("band", aman[name].band, [(0, smurf.dets)])
        smurf.wrap("channel", aman[name].channel, [(0, smurf.dets)])
        aman.det_info.wrap("smurf", smurf)

        if detmap is not None:
            g3u.add_detmap_info(aman, detmap)
        have_wafer = "wafer" in aman.det_inFO
        if not have_wafer:
            logger.error("\tThis observation has no detmap results, skipping")
            continue

        focal_plane = np.column_stack(
            (
                aman[name].xi,
                aman[name].eta,
                aman.det_info.wafer.det_x,
                aman.det_info.wafer.det_y,
            )
        )
        out_msk = aman[name].outliers
        focal_plane[out_msk, :2] = np.nan
        for di, fp in zip(aman.det_info.detector_id, focal_plane):
            try:
                avg_fp[di].append(fp)
            except KeyError:
                avg_fp[di] = [fp]
        all_skipped = False

    if all_skipped:
        logger.error("No valid observations provided")
        sys.exit()

    # Compute the average focal plane while ignoring outliers
    focal_plane = []
    detector_ids = np.array(list(avg_fp.keys()))
    for did in detector_ids:
        avg_pointing = np.nanmedian(np.vstack(avg_fp[did]), axis=0)
        focal_plane.append(avg_pointing)
    focal_plane = np.column_stack(focal_plane)
    xi = focal_plane[0]
    eta = focal_plane[1]

    # Get nominal xi and eta
    transform_pars = fp.get_ufm_to_fp_pars(
        config["coord_transform"]["telescope"],
        config["coord_transform"]["slot"],
        config["coord_transform"]["config_path"],
    )
    x, y = fpc.ufm_to_fp(None, x=focal_plane[2], y=focal_plane[3], **transform_pars)
    if config["coord_transform"]["telescope"] == "LAT":
        xi_nominal, eta_nominal = fpc.LAT_focal_plane(
            None,
            config["coord_transform"]["zemax_path"],
            x,
            y,
            config["coord_transform"]["rot"],
            config["coord_transform"]["tube"],
        )
    elif config["coord_transform"]["telescope"] == "SAT":
        xi_nominal, eta_nominal = fpc.SAT_focal_plane(None, x, y)
    else:
        raise ValueError("Invalid telescope provided")

    # Compute transformation between the two nominal and measured pointing
    affine, shift = get_affine(
        np.vstack((xi, eta)), np.vstack((xi_nominal, eta_nominal))
    )
    s_xi, s_eta, shear, rot = decompose_affine(affine)

    # TODO: Make final outputs and save


if __name__ == "__main__":
    main()

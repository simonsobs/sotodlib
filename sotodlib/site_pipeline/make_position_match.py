import sys
import argparse as ap
import numpy as np
import yaml
import sotodlib.io.g3tsmurf_utils as g3u
from sotodlib.core import AxisManager
from sotodlib.io.metadata import read_dataset
from scipy.spatial.transform import Rotation as R
from pycpd import AffineRegistration
from detmap.inst_model import InstModel


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


def gen_priors(aman, template_det_ids, prior, method="flat", width=1, basis=None):
    """
    Generate priors from detmap.

    Arguments:

        aman: AxisManager assumed to contain aman.det_info.det_id and aman.det_info.wafer.

        template_det_ids: Array of det_ids in the same order as the template.

        prior: Prior value at locations from the detmap.
               Should be greater than 1.

        method: What sort of priors to implement.
                Currently only 'flat' and 'gaussian' are accepted.

        width: Width of priors. For gaussian priors this is sigma.

        basis: Basis to calculate width in.
               Currently not implemented so width will just be along the dets axis.
               At the very least radial distance will be added.

    Returns:

        priors: The 2d array of priors.
    """

    def _flat(arr, idx):
        arr[idx - width // 2 : idx + width // 2 + width % 2] = prior

    def _gaussian(arr, idx):
        arr = prior * np.exp(
            -0.5 * (np.arange(-1 * idx, len(arr) - idx, len(arr)) / width) ** 2
        )

    if method == "flat":
        prior_method = _flat
    elif method == "gaussian":
        prior_method = _gaussian
    else:
        raise ValueError("Method " + method + " not implemented")

    priors = np.ones((aman.dets.count, aman.dets.count))
    for i in aman.dets.count:
        prior_method(priors[i], i)

    det_ids = aman.det_info.det_ids
    if np.array_equal(det_ids, template_det_ids):
        return priors

    missing = np.setdiff1d(template_det_ids, det_ids)
    det_ids = np.concatenate(missing)
    priors = np.concatenate((priors, np.ones((len(missing), aman.dets.count))))
    asort = np.argsort(det_ids)
    template_map = np.argsort(np.argsort(template_det_ids))

    return priors[asort][template_map]


def match_template(
    focal_plane,
    template,
    out_thresh=0,
    avoid_collision=True,
    priors=None,
):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Measured pointing and optionally polarization angle.
                     Should be a (2, n) or (3, n) array with columns: xi, eta, pol.
                     Optionally an optics model can be preapplied to this to map the pointing
                     onto the physical focal plane in which case the columns are: x, y, pol.

        template: Designed x, y, and polarization angle of each detector.
                  Should be a (2, n) or (3, n) array with columns: x, y, pol.

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
    # 2. I am assuming it comtains aman.det_info
    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    pointing_paths = np.atleast_1d(config["pointing_data"])
    polangs_paths = np.atleast_1d(config["polangs"])

    # Load data
    pointings = []
    polangs = []
    for point_path, pol_path in zip(pointing_paths, polangs_paths):
        aman = AxisManager.load(point_path)
        g3u.add_detmap_info(aman, config["detmap"])

        rset = read_dataset(pol_path, "polarization_angle")
        pol_rid = rset["dets:readout_id"]
        pols = rset["polarization_angle"]
        rid_map = np.argsort(np.argsort(aman.det_info.readout_id))
        rid_sort = np.argsort(pol_rid)
        pols = pols[rid_sort][rid_map]

        pointings.append(aman)
        polangs.append(pols)
    bg_map = np.load(config["bias_map"], allow_pickle=True).item()
    bc_bgmap = (bg_map["bands"] << 32) + bg_map["channels"]

    if config["no_fit"]:
        for aman, path in zip(pointings, pointing_paths):
            aman.wrap("det_id", aman.det_info.det_id, [(0, aman.dets)])
            g3u.remove_detmap_info(aman)
            aman.save(path, overwrite=True)
        sys.exit()

    # TODO: apply instrument to pointing if availible

    bp1_bg = (0, 1, 4, 5, 8, 9)
    bp2_bg = (2, 3, 6, 7, 10, 11)

    gen_template = "gen_template" in config
    if gen_template:
        ufm = config["gen_template"]
        inst_model = InstModel(use_solution=False, array_names=(ufm))
        wafer = inst_model.map_makers[ufm].wafer_layout_data.wafer_info
        det_x = []
        det_y = []
        polang = []
        det_ids = []
        template_bg = []
        for mlp in wafer.values():
            for det in mlp.values():
                if not det.is_optical:
                    continue
                det_x.append(det.det_x)
                det_y.append(det.det_y)
                polang.append(det.angle_actual_deg)
                did = f"{ufm}_f{int(det.bandpass):03}_{det.rhomb}r{det.det_row:02}c{det.det_col:02}{det.pol}"
                det_ids.append(did)
                template_bg.append(det.bias_line)
        template = np.vstack((np.array(det_x), np.array(det_y), np.array(polang)))
        det_ids = np.array(det_ids)
        template_bp1 = np.isin(template_bg, bp1_bg)
        template_bp2 = np.isin(template_bg, bp2_bg)

    avg_fp = {}
    outliers = []
    master_template = []
    for aman, pol in zip(pointings, polangs):
        # Split up by bandpass
        bc_aman = (
            aman.det_info.smurf.band.astype(int) << 32
        ) + aman.det_info.smurf.channel.astype(int)
        to_add = np.setdiff1d(bc_aman, bc_bgmap)
        to_remove = np.setdiff1d(bc_bgmap, bc_aman)
        msk = ~np.isin(bc_bgmap, to_remove)
        bg_map["bgmap"] = np.append(bg_map["bgmap"], -2 * np.ones(len(to_add)))
        bc_bgmap = np.append(bc_bgmap[msk], to_add)
        idx = np.argsort(bc_bgmap)[np.argsort(bc_aman)]
        bias_group = bg_map["bg_map"][idx]

        msk_bp1 = np.isin(bias_group, bp1_bg)
        msk_bp2 = np.isin(bias_group, bp2_bg)

        # Prep inputs
        focal_plane = np.vstack((aman.xi, aman.eta, pol))
        if not gen_template:
            template = np.vstack(
                (
                    aman.det_info.wafer.det_x,
                    aman.det_info.wafer.det_y,
                    aman.det_info.wafer.angle,
                )
            )
            master_template.append(np.vstack((template, aman.det_info.det_id)))
            template_bp1 = msk_bp1
            template_bp2 = msk_bp2
            det_ids = aman.det_info.det_id
        priors = gen_priors(
            aman, det_ids, config["prior"], method="flat", width=1, basis=None
        )

        # Do actual matching
        map_bp1, out_bp1 = match_template(
            focal_plane[:, msk_bp1],
            template[:, template_bp1],
            out_thresh=config["out_thresh"],
            avoid_collision=True,
            priors=priors[np.ix_(template_bp1, msk_bp1)],
        )
        map_bp2, out_bp2 = match_template(
            focal_plane[:, msk_bp2],
            template[:, template_bp2],
            out_thresh=config["out_thresh"],
            avoid_collision=True,
            priors=priors[np.ix_(template_bp2, msk_bp2)],
        )

        out_msk = np.zeros(aman.dets.count, dtype=bool)
        out_msk[msk_bp1][out_bp1] = True
        out_msk[msk_bp2][out_bp2] = True
        outliers.append(out_msk)
        bp_msk = np.zeros(aman.dets.count)
        bp_msk[msk_bp1] = 1
        bp_msk[msk_bp2] = 2
        focal_plane = np.vstack((focal_plane, bp_msk))
        focal_plane = focal_plane.T
        focal_plane[out_msk] = np.nan
        for ri, fp in zip(aman.det_info.readout_id, focal_plane):
            try:
                avg_fp[ri].append(fp)
            except KeyError:
                avg_fp[ri] = [fp]

    if len(pointing_paths) == 1:
        det_id = np.zeros(aman.dets.count, dtype=str)
        det_id[msk_bp1] = det_ids[template_bp1][map_bp1]
        det_id[msk_bp2] = det_ids[template_bp2][map_bp2]
        aman.wrap("det_id", det_id, [(0, aman.dets)])
        aman.wrap("pointing_outliers", out_msk.astype(int), [(0, aman.dets)])
        g3u.remove_detmap_info(aman)
        aman.save(config["pointing_data"], overwrite=True)

    focal_plane = []
    readout_ids = np.array(list(avg_fp.keys()))
    for rid in readout_ids:
        avg_pointing = np.nanmedian(np.vstack(avg_fp[rid]), axis=0)
        focal_plane.append(avg_pointing)
    focal_plane = np.vstack(focal_plane).T
    bp_msk = focal_plane[-1].astype(int)
    msk_bp1 = bp_msk == 1
    msk_bp2 = bp_msk == 2
    focal_plane = focal_plane[:-1]

    # Do final matching
    if not gen_template:
        template = np.unique(np.hstack(master_template), axis=1)
        det_ids = template[-1]
        template = template[:-1].astype(float)

    map_bp1, out_bp1 = match_template(
        focal_plane[:, msk_bp1],
        template[:, template_bp1],
        out_thresh=config["out_thresh"],
        avoid_collision=True,
        priors=priors[np.ix_(template_bp1, msk_bp1)],
    )
    map_bp2, out_bp2 = match_template(
        focal_plane[:, msk_bp2],
        template[:, template_bp2],
        out_thresh=config["out_thresh"],
        avoid_collision=True,
        priors=priors[np.ix_(template_bp2, msk_bp2)],
    )
    det_id = np.zeros(len(readout_ids), dtype=str)
    det_id[msk_bp1] = det_ids[template_bp1][map_bp1]
    det_id[msk_bp2] = det_ids[template_bp2][map_bp2]
    out_msk = np.zeros(len(readout_ids), dtype=int)
    out_msk[msk_bp1][out_bp1] = 2
    out_msk[msk_bp2][out_bp2] = 2
    for aman, out, path in zip(pointings, outliers, pointing_paths):
        _, avg_srt, srt = np.intersect1d(
            readout_ids, aman.det_info.readout_id, return_indices=True
        )
        rid_map = np.argsort(srt)
        aman.wrap("det_id", det_id[avg_srt][rid_map], [(0, aman.dets)])
        aman.wrap(
            "pointing_outliers", out + out_msk[avg_srt][rid_map], [(0, aman.dets)]
        )
        g3u.remove_detmap_info(aman)
        aman.save(config["pointing_data"], overwrite=True)


if __name__ == "__main__":
    main()

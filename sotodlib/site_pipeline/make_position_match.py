import argparse as ap
import os
from functools import partial

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import sotodlib.io.g3tsmurf_utils as g3u
import yaml
from detmap.makemap import MapMaker
from pycpd import AffineRegistration
from scipy.cluster import vq
from scipy.optimize import linear_sum_assignment
from sotodlib.coords import affine as af
from sotodlib.coords import optics as op
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__, "make_position_match: ")

valid_bg = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def create_db(filename):
    """
    Create db for storing results if it doesn't already exist

    Arguments:

        filename: Path where database should be made.
    """
    if os.path.isfile(filename):
        return
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    scheme.add_data_field("dataset")

    metadata.ManifestDb(scheme=scheme).to_file(filename)


def gen_priors(aman, template_det_ids, prior, method="flat", width=1, basis=None):
    """
    Generate priors from detmap.

    Arguments:

        aman: AxisManager containing aman.det_info with det_id and wafer.

        template_det_ids: Array of det_ids in the same order as the template.

        prior: Prior value at locations from the detmap.
               Should be greater than 1.

        method: What sort of priors to implement.
                Currently only 'flat' and 'gaussian' are accepted.

        width: Width of priors. For gaussian priors this is sigma.

        basis: Basis to calculate width in.
               Nominally will load values from aman.det_info.wafer.
               Pass in None to use the indices as the basis.

    Returns:

        priors: The 2d array of priors.
    """

    def _flat(x_axis, idx):
        arr = np.ones_like(x_axis)
        lower_bound = x_axis[idx] - width // 2
        upper_bound = x_axis[idx] + width // 2 + width % 2
        prior_range = np.where((x_axis >= lower_bound) & (x_axis < upper_bound))[0]
        arr[prior_range] = prior
        return arr

    def _gaussian(x_axis, idx):
        arr = 1 + (prior - 1) * np.exp(
            -0.5 * ((x_axis - x_axis[idx]) ** 2) / (width**2)
        )
        return arr

    if method == "flat":
        prior_method = _flat
    elif method == "gaussian":
        prior_method = _gaussian
    else:
        raise ValueError("Method " + method + " not implemented")

    if basis is None:
        x_axis = np.arange(aman.dets.count)
    else:
        x_axis = aman.det_info.wafer[basis]

    priors = np.ones((aman.dets.count, len(template_det_ids)))
    _, msk, template_msk = np.intersect1d(
        aman.det_info.det_id, template_det_ids, return_indices=True
    )
    # TODO: Could probably vectorize this
    for i in range(aman.dets.count):
        _prior = prior_method(x_axis, i)
        priors[i, template_msk] = _prior[msk]

    return priors.T


def transform_from_detmap(aman, pointing_name, inliers, det_ids, template):
    """
    Do an approximate transformation of the pointing back to
    the focal plane using the mapping from the loaded detmap.

    Arguments:

        aman: AxisManager containing both pointing and detmap results.

        pointing_name: Name of sub-AxisManager containing pointing info

        inliers: Flag marking whick dets are inliers.

        det_ids: Detector IDs of the rows in the template

        template: The nominal pointing
    """
    # TODO: Gamma support
    dm_det_ids = aman.det_info.det_id.copy()
    dm_det_ids[~inliers] = "outlier"
    _, msk, template_msk = np.intersect1d(
        aman.det_info.det_id, det_ids, return_indices=True
    )
    if np.sum(msk) != aman.dets.count:
        logger.error("There are matched dets not found in the template")
    src = np.vstack((aman[pointing_name].xi0[msk], aman[pointing_name].eta0[msk]))
    mapping = np.argsort(np.argsort(aman.det_info.det_ids[msk]))

    template_sort = np.argsort(det_ids[template_msk])
    dst = template[template_msk][template_sort][mapping].T

    afn, sft = af.get_affine(src, dst)
    transformed = afn @ src + sft[..., None]

    aman[pointing_name].xi0[msk] = transformed[0]
    aman[pointing_name].eta0[msk] = transformed[1]

    return aman


def visualize(frame, frames, ax=None, bias_lines=True):
    """
    Visualize CPD matching process.
    Modified from the pycpd example scripts.

    Arguments:

        frame: The frame to display.

        frames: List of frames, each frame should be [iteration, error, X, Y]

        ax: Axes to use for plots.

        bias_lines: True if bias lines are included in points.
    """
    if ax is None:
        ax = plt.gca()
    iteration, error, X, Y = frames[frame]
    cmap = "Set3"
    if bias_lines:
        x = 1
        y = 2
        c_t = np.around(np.abs(X[:, 0])) / 11.0
        c_s = np.around(np.abs(Y[:, 0])) / 11.0
        srt = np.lexsort(X.T[1:3])
    else:
        x = 0
        y = 1
        c_t = np.zeros(len(X))
        c_s = np.ones(len(Y))
        srt = np.lexsort(X.T[0:2])
    ax.cla()
    for i in range(4):
        ax.scatter(
            X[:, x][srt[i::4]],
            X[:, y][srt[i::4]],
            c=c_t[srt[i::4]],
            cmap=cmap,
            alpha=0.5,
            marker=4 + i,
            vmin=0,
            vmax=1,
        )
    ax.scatter(
        Y[:, x], Y[:, y], c=c_s, cmap=cmap, alpha=0.5, marker="X", vmin=0, vmax=1
    )
    ax.text(
        0.87,
        0.92,
        "Iteration: {:d}\nQ: {:06.4f}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="x-large",
    )


def match_template(
    focal_plane,
    template,
    priors=None,
    bias_lines=True,
    reverse=False,
    vis=False,
    cpd_args={},
):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Measured pointing and optionally polarization angle.
                     Should be a (n, 4) or (n, 3) array with columns:
                     bias_line, xi, eta, gamma.
                     Nominal units for the columns are:
                     none, radians, radians, radians.

        template: Designed pointing of each detector.
                  Should be a (n, 4) or (n, 3) array with columns:
                  bias_line, x, y, pol.
                  Nominal units for the columns are:
                  none, mm, mm, degrees.

        priors: Priors to apply when matching.
                Should be be a n by n array where n is the number of points.
                priors[i, j] is the prior on the i'th point in template
                matching the j'th point in focal_plane.

        bias_lines: Include bias lines in matching.

        reverse: Reverse direction of match.

        vis: If True generate plots to watch the matching process.
             To save the plot pass in a path to save at instead.
             Note that displaying the animation with True is a blocking process
             so it should only be used when debugging with human interaction.
             If this is running headless you should pass in a path instead.

        cpd_args: Dictionairy of kwargs to be passed into AffineRegistration.
                  See the pycpd docs for what these can be.
    Returns:

        mapping: Mapping between elements in template and focal_plane.
                 focal_plane[i] = template[mapping[i]]

        P: The likelihood array without priors applied.

        TY: The transformed points.
    """
    if not bias_lines:
        focal_plane = focal_plane[:, 1:]
        template = template[:, 1:]
    if reverse:
        cpd_args.update({"X": focal_plane, "Y": template})
    else:
        cpd_args.update({"Y": focal_plane, "X": template})
    reg = AffineRegistration(**cpd_args)

    if vis:
        frames = []

        def store_frames(frames, iteration, error, X, Y):
            frames += [[iteration, error, X, Y]]

        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(store_frames, frames=frames)
        reg.register(callback)
        anim = ani.FuncAnimation(
            fig=fig,
            func=partial(
                visualize, frames=frames, ax=fig.axes[0], bias_lines=bias_lines
            ),
            frames=len(frames),
            interval=200,
        )
        if isinstance(vis, str):
            anim.save(vis)
            plt.close()
        else:
            plt.show()
    else:
        reg.register()

    P = reg.P.T
    if reverse:
        P = reg.P
    TY = reg.TY
    if not bias_lines:
        TY = np.column_stack((-2 + np.zeros(len(TY)), TY))

    if priors is None:
        priors = 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(P * priors, True)
    if len(row_ind) < len(focal_plane):
        mapping = np.argmax(P * priors, axis=0)
        mapping[col_ind] = row_ind
    else:
        mapping = row_ind[np.argsort(col_ind)]

    return mapping, P, TY


def _get_inliers(aman, rad_thresh, template=None):
    focal_plane = np.column_stack((aman.xi0, aman.eta0))
    inliers = np.ones(len(focal_plane), dtype=bool)
    if template is not None:
        cent = np.median(template[:, 1:3], axis=0)
        r = np.linalg.norm(focal_plane - cent, axis=1)
        inliers *= r <= rad_thresh

    fp_white = vq.whiten(focal_plane[inliers])
    codebook, _ = vq.kmeans(fp_white, 2)
    codes, _ = vq.vq(fp_white, codebook)

    c0 = codes == 0
    c1 = codes == 1
    m0 = np.median(focal_plane[inliers][c0], axis=0)
    m1 = np.median(focal_plane[inliers][c1], axis=0)
    dist = np.linalg.norm(m0 - m1)

    if dist < 1.5 * rad_thresh:
        cluster = c0 + c1
    elif np.sum(c0) >= np.sum(c1):
        cluster = c0
    else:
        cluster = c1

    cent = np.median(focal_plane[inliers][cluster], axis=0)
    r = np.linalg.norm(focal_plane[inliers] - cent, axis=1)
    inliers[inliers] *= cluster * (r <= rad_thresh)

    return inliers


def _scramble_bgs(bg):
    """
    Transform bias group information so that CPD is more strongly punished
    for swapping nearby bias lines.

    Currently this is just a simple sign flip on every other bias line.
    """
    bg[bg % 2 == 1] *= -1
    return bg


def _get_wafer(ufm):
    # TODO: Switch to det-match code here
    try:
        wafer = MapMaker(north_is_highband=False, array_name=ufm, verbose=False)
    except ValueError:
        wafer = MapMaker(
            north_is_highband=False,
            array_name=ufm,
            verbose=False,
            use_solution_as_design=False,
        )
    det_x = []
    det_y = []
    polang = []
    det_ids = []
    template_bg = []
    is_north = []
    for det in wafer.grab_metadata():
        if not det.is_optical:
            continue
        det_x.append(det.det_x)
        det_y.append(det.det_y)
        polang.append(det.angle_actual_deg)
        det_ids.append(det.detector_id)
        template_bg.append(det.bias_line)
        is_north.append(det.is_north)
    template_bg = np.array(template_bg)
    msk = np.isin(template_bg, valid_bg)
    det_ids = np.array(det_ids)[msk]
    template_n = np.array(is_north)[msk]
    template_bg = _scramble_bgs(template_bg)
    template = np.column_stack(
        (template_bg, np.array(det_x), np.array(det_y), np.array(polang))
    )[msk]

    return det_ids, template, template_n


def _get_pointing(wafer, pointing_cfg):
    xi, eta, gamma = op.get_focal_plane(
        None, x=wafer[:, 1], y=wafer[:, 2], pol=wafer[:, 3], **pointing_cfg
    )
    pointing = wafer.copy()
    pointing[:, 1] = xi
    pointing[:, 2] = eta
    pointing[:, 3] = gamma

    return pointing


def _load_template(template_path, ufm):
    template_rset = read_dataset(template_path, ufm)
    bg = np.array(template_rset["bg"], dtype=int)
    msk = np.isin(bg, valid_bg)
    det_ids = template_rset["dets:det_id"][msk]
    template = np.column_stack(
        (
            _scramble_bgs(bg.astype(float)),
            np.array(template_rset["xi"]),
            np.array(template_rset["eta"]),
            np.array(template_rset["gamma"]),
        )
    )[msk]
    template_n = template_rset["is_north"][msk]

    return det_ids, template, template_n


def _load_bg(aman, bg_path):
    logger.info("loading bg_map from " + bg_path)
    bg_map = np.load(bg_path, allow_pickle=True).item()
    bias_group = np.zeros(aman.dets.count) - 1
    for i in range(aman.dets.count):
        msk = np.all(
            [
                aman.det_info.smurf.band[i] == bg_map["bands"],
                aman.det_info.smurf.channel[i] == bg_map["channels"],
            ],
            axis=0,
        )
        bias_group[i] = bg_map["bgmap"][msk][0]
    msk_bg = np.isin(bias_group, valid_bg)

    return bias_group, msk_bg


def _update_vis(match_config, msk_str):
    vis = match_config.get("vis", False)
    # If we aren't saving a plot
    if isinstance(vis, bool):
        return match_config
    # Otherwise update the save path
    vis = os.path.join(vis, msk_str + ".webp")
    new_config = match_config.copy()
    new_config["vis"] = vis
    return new_config


def _do_match(
    det_ids, focal_plane, template, priors, msks, msk_strs, template_msks, match_config
):
    ndim = focal_plane.shape[1] - 1
    mapped_det_ids = np.zeros(len(focal_plane), dtype=det_ids.dtype)
    P = np.nan + np.zeros(len(focal_plane))
    transformed = np.nan + np.zeros((len(focal_plane), 3))
    mapped_template = np.nan + np.zeros_like(focal_plane)
    for msk, msk_str, t_msk, prior in zip(msks, msk_strs, template_msks, priors):
        logger.info("Performing match with mask: %s", msk_str)
        ndets = np.sum(msk)
        logger.info("\tMask has %d detectors", ndets)
        if ndets == 0:
            logger.info("\tSkipping...")
            continue
        _match_config = _update_vis(match_config, msk_str)
        _map, _P, _TY = match_template(
            focal_plane[msk],
            template[t_msk],
            priors=prior,
            **_match_config,
        )
        mapped_det_ids[msk] = det_ids[t_msk][_map]
        P[msk] = _P[_map, range(_P.shape[1])]
        transformed[msk, :ndim] = _TY[:, 1:]
        mapped_template[msk] = template[t_msk][_map]
    logger.info("Average matched likelihood = %f", np.nanmedian(P))

    return mapped_det_ids, P, transformed, mapped_template


def _mk_output(
    readout_id,
    det_id,
    band,
    channel,
    P_mapped,
    transformed,
    pointing_outlier,
    matched_bg,
    bg_mismap,
):
    out_dt = np.dtype(
        [
            ("dets:readout_id", readout_id.dtype),
            ("matched_det_id", det_id.dtype),
            ("band", int),
            ("channel", int),
            ("likelihood", np.float16),
            ("xi", np.float32),
            ("eta", np.float32),
            ("gamma", np.float32),
            ("pointing_outlier", bool),
            ("matched_bg", int),
            ("bg_mismap", bool),
        ]
    )
    data_out = np.fromiter(
        zip(
            readout_id,
            det_id,
            band,
            channel,
            P_mapped,
            *transformed.T,
            pointing_outlier,
            matched_bg,
            bg_mismap,
        ),
        out_dt,
        count=len(det_id),
    )
    rset_data = metadata.ResultSet.from_friend(data_out)

    return rset_data


def _load_ctx(config):
    ctx = Context(config["context"]["path"])
    pointing_name = config["context"]["pointing"]
    pol_name = config["context"]["polarization"]
    pol = True
    if pol_name is None:
        logger.warning("No polarization data in context")
        pol = False
    query = []
    if "query" in config["context"]:
        query = (ctx.obsdb.query(config["context"]["query"])["obs_id"],)
    obs_ids = np.append(config["context"].get("obs_ids", []), query)
    obs_ids = np.unique(obs_ids)
    if len(obs_ids) == 0:
        raise ValueError("No observations provided in configuration")
    elif len(obs_ids) > 1:
        logger.warning("More than one observation found, using %s", obs_ids[0])
    obs_id = obs_ids[0]
    aman = ctx.get_meta(obs_id, dets=config["context"].get("dets", {}))
    if pointing_name not in aman:
        raise ValueError("No pointing associated with this observation")

    return aman, obs_id, pol, pointing_name, pol_name


def _load_rset(config):
    obs_id = config["resultsets"].get("obs_id", "")
    pointing_rset = read_dataset(*config["resultsets"]["pointing"])
    pointing_aman = pointing_rset.to_axismanager(axis_key="dets:readout_id")
    aman = AxisManager(pointing_aman.dets)
    aman = aman.wrap("pointing", pointing_aman)

    pol = False
    if "polarization" in config["resultsets"]:
        polarization_rset = read_dataset(*config["resultsets"]["polarization"])
        polarization_aman = polarization_rset.to_axismanager(axis_key="dets:readout_id")
        aman = aman.wrap("polarization", polarization_aman)
        pol = True

    return aman, obs_id, pol, "pointing", "polarization"


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load data
    if "context" in config:
        aman, obs_id, pol, pointing_name, pol_name = _load_ctx(config)
    elif "resultsets" in config:
        aman, obs_id, pol, pointing_name, pol_name = _load_rset(config)
    else:
        raise ValueError("No valid inputs provided")

    # Build output path
    ufm = config["ufm"]
    append = config.get("append", "")
    if append:
        append = "_" + append
    if obs_id:
        obs_id = "_" + obs_id
    db = None
    if "manifest_db" in config:
        create_db(config["manifest_db"])
        db = metadata.ManifestDb(config["manifest_db"])
    outdir = os.path.abspath(config["outdir"])
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{ufm}{obs_id}{append}.h5")
    dataset = "focal_plane"

    # If a template is provided load it, otherwise generate one
    det_ids, template, template_n = (
        [],
        np.empty((0, 0)),
        [],
    )  # Just to make pyright shut up
    gen_template = "template" not in config
    if not gen_template:
        template_path = config["template"]
        if os.path.exists(template_path):
            det_ids, template, template_n = _load_template(template_path, ufm)
            plt.scatter(template[:, 1], template[:, 2])
            plt.savefig("/so/home/saianeesh/public_html/a.png")
            plt.close()
        else:
            logger.error("Provided template doesn't exist, trying to generate one")
            gen_template = True
    elif gen_template:
        logger.info(f"Generating template for {ufm}")
        if "pointing_cfg" not in config:
            raise ValueError("Need pointing_cfg to generate template")
        det_ids, template, template_n = _get_wafer(ufm)
        template = _get_pointing(template, config["pointing_cfg"])
    else:
        raise ValueError(
            "No template provided and unable to generate one for some reason"
        )

    match_config = config.get("matching", {})
    vis = match_config.get("vis", False)
    if isinstance(vis, str):
        os.makedirs(vis, exist_ok=True)
    reverse = match_config.get("reverse", False)
    if reverse:
        logger.warning(
            "Matching running in reverse mode. Transform will now be template -> fits."
        )

    # Add smurf band and channel
    smurf = AxisManager(aman.dets)
    have_band = "band" in aman[pointing_name]
    if have_band:
        template_n = np.array(template_n)
        template_msks = [template_n, np.logical_not(template_n)]
        smurf.wrap("band", aman[pointing_name].band, [(0, smurf.dets)])
        band = smurf.band.astype(int)
    else:
        template_msks = [np.ones(len(det_ids), dtype=bool)]
        band = -1 + np.zeros(aman.dets.counts, dtype=int)
        logger.error(
            "Input is missing band information.\n"
            + "\tWon't be able to load detmap or bgmap and north/south split cannot be performed."
        )
    have_ch = "channel" in aman[pointing_name]
    if have_ch:
        smurf.wrap("channel", aman[pointing_name].channel, [(0, smurf.dets)])
        channel = smurf.band.astype(int)
    else:
        channel = -1 + np.zeros(aman.dets.counts, dtype=int)
        logger.error(
            "Input is missing channel information.\n"
            + "\tWon't be able to load detmap or bgmap."
        )
    if "det_info" in aman:
        aman.det_info.wrap("smurf", smurf)
    else:
        det_info = AxisManager(aman.dets)
        det_info.wrap("smurf", smurf)
        aman.wrap("det_info", det_info)

    # Check if we can load a detmap and load if we can
    if "detmap" in config and have_band and have_ch:
        logger.info("Using detmap from " + config["detmap"])
        if not os.path.isfile(config["detmap"]):
            logger.error("Requested detmap doesn't exist. Running without one.")
            have_detmap = False
        g3u.add_detmap_info(aman, config["detmap"], columns="all")
    # Even if we didn't include a file, context could have been magic
    have_detmap = "det_id" in aman.det_info
    if not have_detmap:
        logger.warning("Running without detmap info. This can effect performance.")

    # Check if we can load a bgmap and load it if we can
    have_bgmap = "bias_map" in config
    if have_bgmap and not os.path.isfile(config["bias_map"]):
        logger.error("Requested bgmap doesn't exist. Running without bias line info.")
        have_bgmap = False
    if have_bgmap and have_band and have_ch:
        bias_group, msk_bg = _load_bg(aman, config["bias_map"])
        if have_detmap:
            bl_diff = np.sum(~(bias_group == aman.det_info.wafer.bias_line)) - np.sum(
                ~(msk_bg)
            )
            logger.info(
                "%d detectors have bias lines that don't match the detmap", bl_diff
            )
        bias_group = _scramble_bgs(bias_group)
    else:
        have_bgmap = False
        bias_group = np.nan + np.zeros(aman.dets.count)
        msk_bg = np.ones(aman.dets.count, dtype=bool)
        logger.warning("Running without bias line info. This can effect performance.")
        match_config["bias_lines"] = False

    # Cut outliers
    if config["outliers"].get("use_template", False):
        inliers = _get_inliers(
            aman[pointing_name], config["outliers"]["radial_thresh"], template
        )
    else:
        inliers = _get_inliers(aman[pointing_name], config["outliers"]["radial_thresh"])
    logger.info("Found %d detectors with bad pointing", np.sum(~inliers))

    if have_detmap and config["dm_transform"]:
        logger.info("Applying transformation from detmap")
        transform_from_detmap(aman, pointing_name, inliers, det_ids, template)

    msks = []
    msk_strs = []
    if have_band:
        north = np.isin(aman.det_info.smurf.band.astype(int), (0, 1, 2, 3))
        msks += [north * msk_bg * inliers, (~north) * msk_bg * inliers]
        msk_strs += ["valid_bg-and-north", "valid_bg-and-south"]
    else:
        msks += [inliers]
        msk_strs += ["no_mask"]
    # So that the plots will be associated with an obs_id
    msk_strs = [f"{ufm}{obs_id}{append}-{msk_str}" for msk_str in msk_strs]

    # Prep inputs
    if ("priors" in config) and have_detmap:
        _priors = gen_priors(
            aman,
            det_ids,
            config["priors"]["val"],
            config["priors"]["method"],
            config["priors"]["width"],
            config["priors"]["basis"],
        )
        priors = []
        for t_msk, msk in zip(template_msks, msks):
            priors.append(_priors[np.ix_(t_msk, msk)])
    else:
        priors = [None] * len(msks)

    if pol:
        pol_slice = slice(None)
        pol_val = aman[pol_name].polang
    else:
        pol_slice = slice(0, -1)
        pol_val = np.zeros_like(aman[pointing_name].eta0) + np.nan

    focal_plane = np.column_stack(
        (
            bias_group,
            aman[pointing_name].xi0,
            aman[pointing_name].eta0,
            pol_val,
        )
    )
    _focal_plane = focal_plane[:, pol_slice]
    _template = template[:, pol_slice]
    plt.scatter(focal_plane[msks[1], 1], focal_plane[msks[1], 2])
    plt.scatter(template[:, 1], template[:, 2])
    plt.savefig("/so/home/saianeesh/public_html/a.png")
    plt.close()

    # Do actual matching
    mapped_det_ids, P, transformed, mapped_template = _do_match(
        det_ids,
        _focal_plane,
        _template,
        priors,
        msks,
        msk_strs,
        template_msks,
        match_config,
    )
    transformed_fp = np.nan + np.zeros((aman.dets.count, 3))
    transformed_fp[inliers, pol_slice] = transformed[inliers, pol_slice]

    # BG mismap
    bias_group[msk_bg] = np.abs(bias_group[msk_bg])
    bias_group = np.nan_to_num(bias_group, nan=-2).astype(int)
    matched_bg = np.nan_to_num(np.abs(mapped_template[:, 0]), nan=-2).astype(int)
    bg_mismap = bias_group != matched_bg
    if have_bgmap:
        logger.info("%d bias line mismaps", np.sum(bg_mismap))

    # Another round of outlier flagging
    dist = np.linalg.norm(transformed_fp[:, :2] - mapped_template[:, 1:3], axis=1)
    print(np.nanmedian(dist))
    inliers *= dist < config["outliers"]["pixel_dist"]
    logger.info("Total of %d detectors with bad pointing", np.sum(~inliers))

    rset_data = _mk_output(
        aman.dets.vals,
        mapped_det_ids,
        band,
        channel,
        P,
        transformed,
        ~inliers,
        matched_bg,
        bg_mismap,
    )
    write_dataset(rset_data, outpath, dataset, overwrite=True)
    if db is not None:
        db.add_entry({"obs:obs_id": obs_id, "dataset": dataset}, outpath, replace=True)


if __name__ == "__main__":
    main()

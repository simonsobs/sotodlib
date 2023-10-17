import os
import sys
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import yaml
import sotodlib.io.g3tsmurf_utils as g3u
import sotodlib.io.load_smurf as ls
from functools import partial
from sotodlib.core import AxisManager, metadata, Context
from sotodlib.io.metadata import write_dataset
from sotodlib.site_pipeline import util
from scipy.optimize import linear_sum_assignment
from sqlalchemy.exc import OperationalError
from pycpd import AffineRegistration
from detmap.makemap import MapMaker

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
    scheme.add_data_field("input_paths")
    scheme.add_data_field("encoders")

    metadata.ManifestDb(scheme=scheme).to_file(filename)


def priors_from_result(
    fp_readout_ids,
    template_det_ids,
    final_fp_readout_ids,
    final_template_det_ids,
    likelihoods,
    normalization=0.2,
):
    """
    Generate priors from a previous run of the template matching.

    Arguments:

        fp_readout_ids: Array of readout_ids in the basis of the focal plane that was already matched.

        template_det_ids: Array of det_ids in the basis of the template that was already matched.

        final_fp_readout_ids: Array of readout_ids in the basis of the focal plane that will be matched.

        final_template_det_ids: Array of det_ids in the basis of the template that will be matched.

        likelihoods: Liklihood array from template matching.

        normalization: Value to normalize likelihoods to. The maximum prior will be 1+normalization.

    Returns:

        priors: The 2d array of priors in the basis of the focal plane and template that are to be matched.
    """
    likelihoods *= normalization
    priors = 1 + likelihoods

    missing = np.setdiff1d(final_template_det_ids, template_det_ids)
    template_det_ids = np.concatenate(missing)
    priors = np.concatenate((priors, np.ones((len(missing), len(fp_readout_ids)))))
    asort = np.argsort(template_det_ids)
    template_map = np.argsort(np.argsort(final_template_det_ids))
    priors = priors[asort][template_map]

    missing = np.setdiff1d(final_fp_readout_ids, fp_readout_ids)
    fp_readout_ids = np.concatenate(missing)
    priors = np.concatenate((priors.T, np.ones((len(missing), len(template_det_ids)))))
    asort = np.argsort(fp_readout_ids)
    fp_map = np.argsort(np.argsort(final_fp_readout_ids))
    priors = priors[asort][fp_map].T

    return priors


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


def transform_from_detmap(aman, pointing_name):
    """
    Do an approximate transformation of the pointing back to
    the focal plane using the mapping from the loaded detmap.

    This function works by making three roughly equal bins based on
    the pointing information in aman. These bins are then used to make
    three representative points each for both the pointing information
    and the xy positions of the detectors. These points then form two
    3x3 matrices that can be used to compute the transformation between
    the two spaces.

    Arguments:

        aman: AxisManager containing both pointing and datmap results.

        pointing_name: Name of sub-AxisManager containing pointing info
    """
    phi = np.arctan2(
        aman[pointing_name].eta0 - np.nanmedian(aman[pointing_name].eta0),
        aman[pointing_name].xi0 - np.nanmedian(aman[pointing_name].eta0),
    ) % (2 * np.pi)
    bins = np.nanpercentile(phi, [0.5, 33.5, 66.5, 99.5])
    x = []
    msks = []
    for i in range(3):
        msk = (phi > bins[i]) & (phi < bins[i + 1])
        msks.append(msk)
        x.append(
            np.array(
                (
                    np.nanmedian(aman[pointing_name].xi0[msk]),
                    np.nanmedian(aman[pointing_name].eta0[msk]),
                    1,
                )
            )
        )
    X = np.transpose(np.matrix(x))

    y = []
    for msk in msks:
        y.append(
            np.array(
                (
                    np.nanmedian(aman.det_info.wafer.det_x[msk]),
                    np.nanmedian(aman.det_info.wafer.det_y[msk]),
                    1,
                )
            )
        )
    Y = np.transpose(np.matrix(y))

    A2 = Y * X.I
    coords = np.vstack((aman[pointing_name].xi0, aman[pointing_name].eta0)).T
    transformed = [
        (A2 * np.vstack((np.matrix(pt).reshape(2, 1), 1)))[0:2, :] for pt in coords
    ]
    transformed = np.reshape(transformed, coords.shape)

    aman[pointing_name].xi0 = transformed[:, 0]
    aman[pointing_name].eta0 = transformed[:, 1]


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
    out_thresh=0,
    bias_lines=True,
    reverse=False,
    vis=False,
    cpd_args={},
):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Measured pointing and optionally polarization angle.
                     Should be a (n, 4) or (n, 3) array with columns: bias_line, xi, eta, pol.
                     Nominal units for the columns are: none, radians, radians, radians.

        template: Designed x, y, and polarization angle of each detector.
                  Should be a (n, 4) or (n, 3) array with columns: bias_line, x, y, pol.
                  Nominal units for the columns are: none, mm, mm, degrees.

        priors: Priors to apply when matching.
                Should be be a n by n array where n is the number of points.
                The priors[i, j] is the prior on the i'th point in template matching the j'th point in focal_plane.

        out_thresh: Threshold at which points will be considered outliers.
                    Should be in range [0, 1) and is checked against the
                    probability that a point matches its mapped point in the template.

        bias_lines: Include bias lines in matching.

        reverse: Reverse direction of match.

        vis: If True generate plots to watch the matching process.
             To save the plot pass in a path to save at instead.
             Note that with True is passed and the animation is displayed, that is a blocking process.
             so it should only be used when debugging with human interaction.
             If this is running headless you should pass in a path instead.

        cpd_args: Dictionairy containing kwargs to be passed into AffineRegistration.
                  See the pycpd docs for what these can be.
    Returns:

        mapping: Mapping between elements in template and focal_plane.
                 focal_plane[i] = template[mapping[i]]

        outliers: Indices of points that are outliers.
                  Note that this is in the basis of mapping and focal_plane, not template.

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
        TY = np.column_stack((np.zeros(len(TY)), TY))

    if priors is None:
        priors = 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(P * priors, True)
    if len(row_ind) < len(focal_plane):
        mapping = np.argmax(P * priors, axis=0)
        mapping[col_ind] = row_ind
    else:
        mapping = row_ind[np.argsort(col_ind)]

    outliers = np.array([])
    if out_thresh > 0:
        outliers = np.where(P[mapping, range(P.shape[1])] < out_thresh)[0]

    return mapping, outliers, P, TY


def _scramble_bgs(bg):
    """
    Transform bias group information so that CPD is more strongly punished
    for swapping nearby bias lines.

    Currently this is just a simple sign flip on every other bias line.
    """
    bg[bg % 2 == 1] *= -1
    return bg


def _gen_template(ufm):
    logger.info("Generating template for " + ufm)
    wafer = MapMaker(north_is_highband=False, array_name=ufm)
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
        polang.append(det.angle)
        det_ids.append(det.detector_id)
        template_bg.append(det.bias_line)
        is_north.append(det.is_north)
    det_ids = np.array(det_ids)
    template_bg = np.array(template_bg)
    template_msk = np.isin(template_bg, valid_bg)
    template_n = np.array(is_north) & template_msk
    template_s = ~np.array(is_north) & template_msk
    template_bg = _scramble_bgs(template_bg)
    template = np.column_stack(
        (template_bg, np.array(det_x), np.array(det_y), np.array(polang))
    )

    return template, template_n, template_s, det_ids


def _mk_template(aman):
    dm_msk = (
        np.isfinite(aman.det_info.wafer.det_x)
        | np.isfinite(aman.det_info.wafer.det_y)
        | np.isfinite(aman.det_info.wafer.angle)
    )
    dm_aman = aman.restrict("dets", aman.dets.vals[dm_msk], False)

    template_msk = np.isin(dm_aman.det_info.wafer.bias_line, valid_bg)
    bias_line = dm_aman.det_info.wafer.bias_line
    bias_line = _scramble_bgs(bias_line)

    template = np.column_stack(
        (
            bias_line,
            dm_aman.det_info.wafer.det_x,
            dm_aman.det_info.wafer.det_y,
            dm_aman.det_info.wafer.angle,
        )
    )
    det_ids = dm_aman.det_info.det_id
    if "is_north" in dm_aman.det_info:
        is_north = dm_aman.det_info.wafer.is_north == "True"
        template_n = is_north & template_msk
        template_s = ~(is_north) & template_msk
    else:
        logger.warning(
            "\tis_north is missing from the wafer info.\n"
            + "\t\tIf this is a sim this is probably fine. If not please check the inputs."
        )
        template_n = template_msk
        template_s = np.zeros_like(template_msk, dtype=bool)

    return template, template_n, template_s, det_ids


def _load_bg(aman, bg_map):
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
    msk_bp = np.isin(bias_group, valid_bg)

    return bias_group, msk_bp


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
    out_msk = np.zeros(len(focal_plane), dtype=bool)
    P = np.zeros(len(focal_plane))
    transformed = np.nan + np.zeros((len(focal_plane), 3))
    for msk, msk_str, t_msk, prior in zip(msks, msk_strs, template_msks, priors):
        logger.info("\tPerforming match with mask: %s", msk_str)
        _match_config = _update_vis(match_config, msk_str)
        _map, _out, _P, _TY = match_template(
            focal_plane[msk],
            template[t_msk],
            priors=prior,
            **_match_config,
        )
        mapped_det_ids[msk] = det_ids[t_msk][_map]
        P[msk] = _P[_map, range(_P.shape[1])]
        out_msk[np.flatnonzero(msk)[_out]] = True
        transformed[msk, :ndim] = _TY[:, 1:]
    out_msk[~np.any(msks, axis=0)] = True
    logger.info("\tAverage matched likelihood = " + str(np.median(P)))

    return mapped_det_ids, out_msk, P, transformed


def _get_encoder(aman, out_msk):
    if "az_avg" in aman:
        az = np.nanmedian(aman.az_avg[~out_msk])
    else:
        logger.warning("\tAz encoder information information not found, setting to nan")
        az = np.nan
    if "el_avg" in aman:
        el = np.nanmedian(aman.el_avg[~out_msk])
    else:
        logger.warning("\tEl encoder information information not found, setting to nan")
        el = np.nan
    if "bs_avg" in aman:
        bs = np.nanmedian(aman.bs_avg[~out_msk])
    else:
        logger.warning("\tBs encoder information information not found, setting to nan")
        bs = np.nan

    return az, el, bs


def _mk_output(
    out_dt,
    readout_id,
    det_id,
    dm_det_id,
    band_channel,
    focal_plane,
    transformed,
    P_mapped,
    out_msk,
    msks,
):
    logger.info(str(np.sum(np.any(msks, axis=0))) + " detectors matched")
    logger.info(str(np.unique(det_id).shape[0]) + " unique matches")
    if dm_det_id is not None:
        logger.info(str(np.sum(det_id == dm_det_id)) + " match with detmap")

    band_channel = np.nan_to_num(band_channel, nan=-1)
    data_out = np.fromiter(
        zip(
            readout_id,
            det_id,
            *band_channel.T,
            *focal_plane.T,
            *transformed.T,
            P_mapped,
            out_msk,
        ),
        out_dt,
        count=len(det_id),
    )
    rset_data = metadata.ResultSet.from_friend(data_out)

    return rset_data


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load context
    ctx = Context(config["context"]["path"])
    pointing_name = config["context"]["pointing"]
    pol_name = config["context"]["polarization"]
    if pol_name is None:
        logger.warning("Polarization data is disabled")
        pol = False
    query = []
    if "query" in config["context"]:
        query = (ctx.obsdb.query(config["context"]["query"])["obs_id"],)
    obs_ids = np.append(config["context"].get("obs_ids", []), query)
    obs_ids = np.unique(obs_ids)
    if len(obs_ids) == 0:
        raise ValueError("No observations provided in configuration")

    # Figure out the tuneset and check the list of obs
    try:
        SMURF = ls.G3tSmurf(db_path=ctx["obsdb"], archive_path=None)
        ses = SMURF.Session()
        tunefile = []
        tune_id = []
        for obs_id in obs_ids:
            try:
                obs = (
                    ses.query(ls.Observations)
                    .filter(ls.Observations.obs_id == obs_id)
                    .one()
                )
            except:
                raise ValueError(
                    obs_id + " not found. Please check obs_id list and context file."
                )
            tunefile.append(obs.tunesets[0].path)
            tune_id.append(obs.tunesets[0].id)
    except OperationalError:
        logger.info("Input seems to be from a sim")
        tunefile = ["sim"] * len(obs_ids)
        tune_id = ["sim"] * len(obs_ids)
    if not (
        np.all(np.array(tunefile) == tunefile[0])
        and np.all(np.array(tune_id) == tune_id[0])
    ):
        raise ValueError("Not all observations have the same tuneset")
    tunefile = tunefile[0]
    tune_id = tune_id[0]

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    if len(obs_ids) == 1:
        create_db(config["manifest_db"])
        db = metadata.ManifestDb(config["manifest_db"])
        outpath = os.path.join(config["outdir"], f"{ufm}_{obs_ids[0]}{append}.h5")
    else:
        outpath = os.path.join(config["outdir"], f"{ufm}_{tune_id}{append}.h5")
    dataset = "focal_plane"
    input_paths = "input_data_paths"
    outpath = os.path.abspath(outpath)

    # Make list of input paths for later reference
    types = ["config", "tunefile", "results", "context"]
    paths = [
        args.config_path,
        tunefile,
        outpath,
        config["context"]["path"],
    ]

    # If requested generate a template for the UFM with the instrument model
    gen_template = config["gen_template"]
    # If given a path instead of a bool load template from file
    if isinstance(gen_template, str):
        logger.info("Loading template from " + gen_template)
        if os.path.isfile(gen_template):
            # Assuming its a tsv with columns: det_id, bias_line, x, y, is_north
            template = np.genfromtxt(gen_template, dtype=str)
            det_ids = template[:, 0]
            template_n = template[:, -1].astype(bool)
            template_s = ~template_n
            template_msks = [template_n, template_s]
            template = template[:, 1:-1].astype(float)
            gen_template = True
            types.append("template")
            paths.append("gen_template")
        else:
            logger.error(
                "Requested template file doesn't exist. Generating template instead."
            )
            gen_template = True
            template, template_n, template_s, det_ids = _gen_template(ufm)
    elif gen_template:
        template, template_n, template_s, det_ids = _gen_template(ufm)
        template_msks = [template_n, template_s]

    # Check if we can load a bgmap and load it if we can
    have_bgmap = "bias_map" in config
    if have_bgmap:
        logger.info("loading bg_map from " + config["bias_map"])
        if os.path.isfile(config["bias_map"]):
            bg_map = np.load(config["bias_map"], allow_pickle=True).item()
            types.append("bgmap")
            paths.append(config["bias_map"])
        else:
            logger.error(
                "Requested bgmap doesn't exist. Running without bias line info."
            )
            have_bgmap = False
    else:
        logger.warning("Running without bias line info. This can effect performance.")
        config["matching"]["bias_lines"] = False

    # Check if we can load a detmap
    have_detmap = "detmap" in config
    if have_detmap:
        logger.info("Using detmap from " + config["detmap"])
        if os.path.isfile(config["detmap"]):
            types.append("detmap")
            paths.append(config["detmap"])
        else:
            logger.error("Requested detmap doesn't exist. Running without one.")
            have_detmap = False
    else:
        logger.warning("Running without detmap info. This can effect performance.")

    # Make ResultSet of inputs
    paths = [os.path.abspath(p) for p in paths]
    types += ["obs_id"] * len(obs_ids)
    paths += list(obs_ids)
    rset_paths = metadata.ResultSet(
        keys=["type", "path"],
        src=np.vstack((types, paths)).T,
    )

    base_match_config = config.get("matching", {})
    vis = base_match_config.get("vis", False)
    if isinstance(vis, str):
        os.makedirs(vis, exist_ok=True)
    reverse = base_match_config.get("reverse", False)
    if reverse:
        logger.warning(
            "Matching running in reverse mode. meas_x and meas_y will actually be fit pointing."
        )
    make_priors = ("priors" in config) and have_detmap
    avg_fp = {}
    master_template = []
    results = [[], [], []]
    encoder = []
    num = 0
    for obs_id in obs_ids:
        logger.info("Starting match on observation " + obs_id)

        # Load data
        aman = ctx.get_meta(obs_id, dets=config["context"].get("dets", {}))
        if pointing_name not in aman:
            logger.warning("\tNo pointing associated with this observation. Skipping.")
            continue

        if pol_name is not None:
            pol = True
            if pol_name not in aman:
                pol = False
                logger.warning("\tNo polang associated with this pointing")
        # Put SMuRF band channel in the correct place
        smurf = AxisManager(aman.dets)
        have_band = "band" in aman[pointing_name]
        if have_band:
            smurf.wrap("band", aman[pointing_name].band, [(0, smurf.dets)])
        else:
            logger.error(
                "\tInput is missing band information.\n"
                + "\t\tWon't be able to load detmap or bgmap and north/south split cannot be performed."
            )
        have_ch = "channel" in aman[pointing_name]
        if have_ch:
            smurf.wrap("channel", aman[pointing_name].channel, [(0, smurf.dets)])
        else:
            logger.error(
                "\tInput is missing channel information.\n"
                + "\t\tWon't be able to load detmap or bgmap."
            )
        aman.det_info.wrap("smurf", smurf)

        # Do a radial cut
        r = np.sqrt(
            (aman[pointing_name].xi0 - np.median(aman[pointing_name].xi0)) ** 2
            + (aman[pointing_name].eta0 - np.median(aman[pointing_name].eta0)) ** 2
        )
        r_msk = r < config["radial_thresh"] * np.median(r)
        aman = aman.restrict("dets", aman.dets.vals[r_msk])
        logger.info("\tCut " + str(np.sum(~r_msk)) + " detectors with bad pointing")

        if have_detmap and have_band and have_ch:
            g3u.add_detmap_info(aman, config["detmap"], columns="all")
        have_wafer = "wafer" in aman.det_info
        original = aman
        if have_wafer and config["dm_transform"]:
            logger.info("\tApplying transformation from detmap")
            original = aman.copy()
            transform_from_detmap(aman, pointing_name)

        # Load bias group info
        if have_bgmap and have_band and have_ch:
            bias_group, msk_bp = _load_bg(aman, bg_map)
            bl_diff = np.sum(~(bias_group == aman.det_info.wafer.bias_line)) - np.sum(
                ~(msk_bp)
            )
            logger.info(
                "\t"
                + str(bl_diff)
                + " detectors have bias lines that don't match the detmap"
            )
            bias_group = _scramble_bgs(bias_group)
        else:
            bias_group = np.nan + np.zeros(aman.dets.count)
            msk_bp = np.ones(aman.dets.count, dtype=bool)

        msks = []
        msk_strs = []
        if have_band:
            north = np.isin(aman.det_info.smurf.band.astype(int), (0, 1, 2, 3))
            msks += [north & msk_bp, (~north) & msk_bp]
            msk_strs += ["valid_bg-and-north", "valid_bg-and-south"]
        else:
            msks += [np.ones(aman.dets.count, dtype=bool)]
            msk_strs += ["no_mask"]
        # So that the plots will be associated with an obs_id
        msk_strs = [f"{obs_id}-{msk_str}" for msk_str in msk_strs]

        # Prep inputs
        if not gen_template:
            if not have_wafer:
                logger.warning(
                    "\tUnable to make template from wafer info for this observation, skipping."
                )
                logger.warning(
                    "\tYou may want to change settings and either provide a template or turn on gen_template."
                )
                continue
            template, template_n, template_s, det_ids = _mk_template(aman)
            template_msks = [template_n, template_s]
            c_msk = np.zeros(aman.dets.count)
            for i, msk in enumerate(template_msks):
                c_msk += i * msk
            master_template.append(np.column_stack((template, det_ids, c_msk)))
            if np.sum(template_n | template_s) < np.sum(np.any(msks, axis=0)):
                logger.warning(
                    "\tTemplate is smaller than input pointing, uniqueness of mapping is no longer gauranteed"
                )
        if have_band:
            _template_msks = template_msks
        else:
            _template_msks = [np.ones(len(det_ids), dtype=bool)]

        if make_priors and have_wafer:
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
            pol_orig = original[pol_name].polang
        else:
            pol_slice = slice(0, -1)
            pol_val = np.zeros_like(aman[pointing_name].eta0) + np.nan
            pol_orig = pol_val

        focal_plane = np.column_stack(
            (
                bias_group,
                aman[pointing_name].xi0,
                aman[pointing_name].eta0,
                pol_val,
            )
        )
        original_focal_plane = np.column_stack(
            (
                original[pointing_name].xi0,
                original[pointing_name].eta0,
                pol_orig,
            )
        )
        _focal_plane = focal_plane[:, pol_slice]
        _template = template[:, pol_slice]

        # Do actual matching
        match_config = base_match_config.copy()
        match_config["bias_lines"] &= have_bgmap and have_band and have_ch
        mapped_det_ids, out_msk, P, transformed = _do_match(
            det_ids,
            _focal_plane,
            _template,
            priors,
            msks,
            msk_strs,
            _template_msks,
            match_config,
        )

        # Store outputs for now
        results[0].append(aman.det_info.readout_id)
        results[1].append(det_ids)
        results[2].append(P)

        # Collapse mask and save focal plane
        c_msk = np.zeros(aman.dets.count)
        if have_band:
            band = aman.det_info.smurf.band
            for i, msk in enumerate(msks):
                c_msk += i * msk
        else:
            band = np.nan + np.zeros(aman.dets.count)
            c_msk += np.nan
        if have_ch:
            ch = aman.det_info.smurf.channel
        else:
            ch = np.nan + np.zeros(aman.dets.count)
        focal_plane = np.column_stack(
            (
                band,
                ch,
                original_focal_plane,
                focal_plane,
                c_msk,
            )
        )
        focal_plane[out_msk, 2:] = np.nan
        for ri, fp in zip(aman.det_info.readout_id, focal_plane):
            try:
                avg_fp[ri].append(fp)
            except KeyError:
                avg_fp[ri] = [fp]

        # Save nominal encoder angles
        encoder.append((obs_id,) + _get_encoder(aman[pointing_name], out_msk))
        num += 1

    encoder = np.array(
        encoder,
        dtype=np.dtype(
            [
                ("obs_id", obs_id.dtype),
                ("az", np.float32),
                ("el", np.float32),
                ("bs", np.float32),
            ]
        ),
    )
    rset_encoder = metadata.ResultSet.from_friend(encoder)
    encoders = "encoders"
    out_dt = np.dtype(
        [
            ("dets:readout_id", aman.det_info.readout_id.dtype),
            ("matched_det_id", det_ids.dtype),
            ("band", int),
            ("channel", int),
            ("xi", np.float32),
            ("eta", np.float32),
            ("polang", np.float32),
            ("meas_x", np.float32),
            ("meas_y", np.float32),
            ("meas_pol", np.float32),
            ("likelihood", np.float16),
            ("outliers", bool),
        ]
    )

    if num == 0:
        logger.error("No valid observations provided")
        sys.exit()
    elif num == 1:
        dm_det_id = None
        if have_wafer:
            dm_det_id = aman.det_info.det_id
        rset_data = _mk_output(
            out_dt,
            aman.det_info.readout_id,
            mapped_det_ids,
            dm_det_id,
            focal_plane[:, :2],
            focal_plane[:, 2:5],
            transformed,
            P,
            out_msk,
            msks,
        )
        write_dataset(rset_data, outpath, dataset, overwrite=True)
        write_dataset(rset_paths, outpath, input_paths, overwrite=True)
        write_dataset(rset_encoder, outpath, encoders, overwrite=True)
        db.add_entry(
            {
                "obs:obs_id": obs_id,
                "dataset": dataset,
                "input_paths": input_paths,
                "encoders": encoders,
            },
            outpath,
            replace=True,
        )
        sys.exit()

    if not gen_template:
        template = np.unique(np.vstack(master_template), axis=0)
        c_msk = template[:, -1].astype(int)
        template_msks = []
        for i in range(np.max(c_msk)):
            template_msks.append(c_msk == i)
        det_ids = template[:, -2]
        template = template[:, :-2].astype(float)

    # Compute the average focal plane while ignoring outliers
    focal_plane = []
    readout_ids = np.array(list(avg_fp.keys()))
    for rid in readout_ids:
        avg_pointing = np.nanmedian(np.vstack(avg_fp[rid]), axis=0)
        focal_plane.append(avg_pointing)
    focal_plane = np.column_stack(focal_plane)
    c_msk = focal_plane[-1].astype(int)
    if np.isnan(c_msk).all():
        msks = [np.ones_like(c_msk, dtype=bool)]
        msk_strs = ["no_mask"]
    else:
        msks = []
        for i in range(np.nanmax(c_msk)):
            msks.append(c_msk == i)
            msk_strs.append(f"avg_msk_{i}")
    bc_avg_pointing = focal_plane[:5].T
    focal_plane = focal_plane[5:9].T
    if np.isnan(focal_plane[:, -1]).all():
        focal_plane = focal_plane[:, :-1]
        template = template[:, :-1]

    # Build priors from previous results
    priors = 1
    for fp_readout_id, template_det_id, P in zip(*results):
        priors *= priors_from_result(
            fp_readout_id,
            template_det_id,
            readout_ids,
            det_ids,
            P,
            config["prior_normalization"],
        )
    priors = []
    for t_msk, msk in zip(template_msks, msks):
        priors.append(_priors[np.ix_(t_msk, msk)])

    # Do final matching
    match_config = config["matching"]
    match_config["bias_lines"] &= have_bgmap and np.isfinite(focal_plane[:, 0]).all()
    mapped_det_ids, out_msk, P, transformed = _do_match(
        det_ids,
        focal_plane,
        template,
        priors,
        msks,
        msk_strs,
        template_msks,
        match_config,
    )

    # Make final outputs and save
    rset_data = _mk_output(
        out_dt,
        readout_ids,
        mapped_det_ids,
        None,
        bc_avg_pointing[:, :2],
        bc_avg_pointing[:, 2:5],
        transformed,
        P,
        out_msk,
        msks,
    )
    write_dataset(rset_data, outpath, dataset, overwrite=True)
    write_dataset(rset_paths, outpath, input_paths, overwrite=True)
    write_dataset(rset_encoder, outpath, encoders, overwrite=True)


if __name__ == "__main__":
    main()

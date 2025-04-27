import logging
import os
from dataclasses import InitVar, dataclass, field
from functools import cached_property, partial
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import megham.transform as mt
import megham.utils as mu
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic
from sotodlib.coords import optics as op
from sotodlib.core import metadata
from sotodlib.io.metadata import read_dataset, write_dataset

logger = logging.getLogger("finalize_focal_plane")
plt.style.use("tableau-colorblind10")


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        dset.attrs[k] = v


@dataclass
class Transform:
    shift: NDArray[np.floating]  # (ndim,)
    affine: NDArray[np.floating]  # (ndim, ndim)
    scale: NDArray[np.floating] = field(init=False)  # (ndim,)
    shear: float = field(init=False)
    rot: float = field(init=False)

    def __post_init__(self):
        self.decompose()

    @classmethod
    def identity(cls):
        return Transform(np.zeros(3), np.eye(3))

    @classmethod
    def from_split(cls, shift, xieta_affine, gamma_scale):
        affine = np.eye(len(xieta_affine) + 1)
        affine[: len(xieta_affine), : len(xieta_affine)] = xieta_affine
        affine[-1, -1] = gamma_scale

        return Transform(shift, affine)

    def decompose(self):
        xieta_affine = self.affine[:2, :2]
        gamma_scale = self.affine[-1, -1]
        scale, shear, rot = mt.decompose_affine(xieta_affine)
        self.scale = np.array((*scale, gamma_scale))
        self.shear = shear.item()
        self.rot = mt.decompose_rotation(rot)[-1]

    def save(self, f, path, append=""):
        if path not in f:
            f.create_group(path)
        _add_attrs(
            f[path],
            {
                f"shift{append}": self.shift,
                f"scale{append}": self.scale,
                f"shear{append}": self.shear,
                f"rot{append}": self.rot,
                f"affine{append}": self.affine,
            },
        )

    @classmethod
    def load(cls, group, append=""):
        shift = group.attrs[f"shift{append}"]
        affine = group.attrs[f"affine{append}"]

        return Transform(shift, affine)


@dataclass
class Template:
    det_ids: NDArray[np.str_]  # (ndet,)
    fp: NDArray[np.floating]  # (ndet, ndim)
    optical: NDArray[np.bool_]  # (ndet,)
    pointing_cfg: InitVar[Dict]
    center: NDArray[np.floating] = field(init=False)  # (1, ndim)
    spacing: NDArray[np.floating] = field(init=False)  # (ndim,)
    bandpass: NDArray[np.str_] = field(init=False)
    pol: NDArray[np.str_] = field(init=False)
    rhombus: NDArray[np.str_] = field(init=False)

    def __post_init__(self, pointing_cfg):
        self.center = np.array(
            op.get_focal_plane(None, x=0, y=0, pol=0, **pointing_cfg)
        ).T
        xieta_spacing = mu.estimate_spacing(self.fp[self.optical, :2])
        # For gamma rather than the spacing in real space we want the difference between bins
        # This is a rough estimate but good enough for us
        gamma_spacing = np.percentile(np.diff(np.sort(self.fp[2])), 99.9)
        self.spacing = np.array([xieta_spacing, xieta_spacing, gamma_spacing])

        self.bandpass = np.zeros_like(self.det_ids)
        self.pol = np.zeros_like(self.det_ids)
        self.rhombus = np.zeros_like(self.det_ids)

    def add_wafer_info(self, aman, template_msk):
        self.__dict__.pop("id_strs", None)
        self.__dict__.pop("valid_ids", None)
        if not np.all(
            np.isin(["bandpass", "pol", "rhombus"], list(aman.det_info.wafer.keys()))
        ):
            logger.warning(
                "det_info.wafer seems to be missint metadata? Safe to ignore in rset mode."
            )
            return
        mapping = np.argsort(np.argsort(self.det_ids[template_msk]))
        srt = np.argsort(aman.det_info.det_id)
        self.bandpass[template_msk] = aman.det_info.wafer.bandpass[srt][mapping]
        self.pol[template_msk] = aman.det_info.wafer.pol[srt][mapping]
        self.rhombus[template_msk] = aman.det_info.wafer.rhombus[srt][mapping]

    @cached_property
    def id_strs(self):
        return np.array([bp + "_" + pol for bp, pol in zip(self.bandpass, self.pol)])

    @cached_property
    def valid_ids(self):
        ids = np.unique(self.id_strs)
        valid = ~(np.char.startswith(ids, "_") * np.char.endswith(ids, "_"))
        return ids[valid]


@dataclass
class FocalPlane:
    stream_id: str
    wafer_slot: str
    det_ids: NDArray[np.str_]  # (ndet,)
    avg_fp: NDArray[np.floating]  # (ndim, ndet)
    weights: NDArray[np.floating]  # (ndet,)
    transformed: NDArray[np.floating]  # (ndet, ndim)
    center: NDArray[np.floating]  # (1, ndim)
    center_transformed: NDArray[np.floating]  # (1, ndim)
    n_point: NDArray[np.int_]  # (ndet,)
    n_gamma: NDArray[np.int_]  # (ndet,)
    have_gamma: bool = field(default=False)
    template: Optional[Template] = field(default=None)
    transform: Transform = field(default_factory=Transform.identity)
    transform_nocm: Transform = field(default_factory=Transform.identity)
    full_fp: Optional[NDArray[np.floating]] = field(
        init=False, default=None
    )  # (ndet, ndim, n_aman)
    tot_weight: Optional[NDArray[np.floating]] = field(
        init=False, default=None
    )  # (ndet,)

    def __post_init__(self):
        if self.template is not None and not np.all(
            np.isclose(self.center, self.template.center)
        ):
            raise ValueError("Focalplane center does not match template")

    @property
    def diff(self):
        return self.avg_fp - self.transformed

    @property
    def isfinite(self):
        return np.isfinite(self.diff[:, 0])

    @property
    def dist(self):
        if self.have_gamma:
            return np.linalg.norm(self.diff, axis=1)
        return np.linalg.norm(self.diff[:, :2], axis=1)

    @classmethod
    def empty(cls, template, stream_id, wafer_slot, n_aman):
        if template is None:
            raise TypeError("template must be an instance of Template, not None")
        full_fp = np.full(template.fp.shape + (n_aman,), np.nan)
        tot_weight = np.zeros((len(template.det_ids), 2))
        avg_fp = np.full_like(template.fp, np.nan)
        weight = np.zeros((len(template.det_ids), 2))
        transformed = template.fp.copy()
        center = template.center.copy()
        center_transformed = template.center.copy()
        n_point = np.zeros_like(template.det_ids, dtype=int)
        n_gamma = np.zeros_like(template.det_ids, dtype=int)

        fp = FocalPlane(
            stream_id,
            wafer_slot,
            template.det_ids,
            avg_fp,
            weight,
            transformed,
            center,
            center_transformed,
            n_point,
            n_gamma,
            template=template,
        )
        fp.full_fp = full_fp
        fp.tot_weight = tot_weight

        return fp

    def map_by_det_id(self, aman):
        if self.template is not None and not np.array_equal(
            self.det_ids, self.template.det_ids
        ):
            raise ValueError("det_ids don't match template, mapping is not valid!")
        _, msk, template_msk = np.intersect1d(
            aman.det_info.det_id, self.det_ids, return_indices=True
        )
        if len(msk) != aman.dets.count:
            logger.warning("There are matched dets not found in the focal plane")
        mapping = np.argsort(np.argsort(self.det_ids[template_msk]))
        srt = np.argsort(aman.det_info.det_id[msk])
        xi = aman.pointing.xi[msk][srt][mapping]
        eta = aman.pointing.eta[msk][srt][mapping]
        r2 = np.nan + np.zeros_like(eta)
        if "r2" in aman.pointing:
            r2 = aman.pointing.R2[msk][srt][mapping]
        if "polarization" in aman:
            # name of field just a placeholder for now
            gamma = aman.polarization.polang[msk][srt][mapping]
        elif "gamma" in aman.pointing:
            gamma = aman.pointing.gamma[msk][srt][mapping]
        else:
            gamma = np.full(len(xi), np.nan)
        fp = np.column_stack((xi, eta, gamma))
        return fp, r2, template_msk

    def add_fp(self, i, fp, weights, template_msk):
        if self.full_fp is None or self.tot_weight is None:
            raise ValueError("full_fp or tot_weight not initialized")
        self.full_fp[template_msk, :, i] = fp * weights[:, 0][..., None]
        weights = np.nan_to_num(weights)
        self.tot_weight[template_msk] += weights

    def save(self, f, db_info, group):
        ndets = len(self.det_ids)
        outdt = [
            ("dets:det_id", self.det_ids.dtype),
            ("xi", np.float32),
            ("eta", np.float32),
            ("gamma", np.float32),
        ]
        fpout = np.fromiter(
            zip(self.det_ids, *(self.transformed.T)), dtype=outdt, count=ndets
        )
        write_dataset(
            metadata.ResultSet.from_friend(fpout),
            f,
            f"{group}/focal_plane",
            overwrite=True,
        )
        _add_attrs(
            f[f"{group}/focal_plane"],
            {"wafer_slot": str(self.wafer_slot), "measured_gamma": self.have_gamma},
        )
        entry = {"dets:stream_id": self.stream_id, "dataset": f"{group}/focal_plane"}
        entry.update(db_info[1])
        db_info[0].add_entry(entry, filename=os.path.basename(f.filename), replace=True)

        outdt_full = [
            ("dets:det_id", self.det_ids.dtype),
            ("xi_t", np.float32),
            ("eta_t", np.float32),
            ("gamma_t", np.float32),
            ("xi_m", np.float32),
            ("eta_m", np.float32),
            ("gamma_m", np.float32),
            ("weights", np.float32),
            ("r2", np.float32),
            ("n_point", np.int8),
            ("n_gamma", np.int8),
        ]
        fpfullout = np.fromiter(
            zip(
                self.det_ids,
                *(self.transformed.T),
                *(self.avg_fp.T),
                *(self.weights.T),
                self.n_point,
                self.n_gamma,
            ),
            dtype=outdt_full,
            count=ndets,
        )
        write_dataset(
            metadata.ResultSet.from_friend(fpfullout),
            f,
            f"{group}/focal_plane_full",
            overwrite=True,
        )

        self.transform.save(f, f"{group}/transform")
        self.transform_nocm.save(f, f"{group}/transform", "_nocm")
        _add_attrs(
            f[f"{group}"],
            {
                "fit_centers": self.center_transformed,
                "template_centers": self.center,
            },
        )

    @classmethod
    def load(cls, group):
        stream_id = group.name.split("/")[-1]
        fp_full = read_dataset(group.file, f"{group.name}/focal_plane_full")
        if fp_full.keys is None:
            raise ValueError("fp_full somehow has no keys")
        det_ids = fp_full["dets:det_id"]
        avg_fp = np.column_stack(
            (
                np.array(fp_full["xi_m"]),
                np.array(fp_full["eta_m"]),
                np.array(fp_full["gamma_m"]),
            )
        )
        # For backwards compatibility
        weights = np.array(fp_full["weights"])
        if "r2" in fp_full.keys:
            weights = np.column_stack((weights, np.array(fp_full["r2"])))
        transformed = np.column_stack(
            (
                np.array(fp_full["xi_t"]),
                np.array(fp_full["eta_t"]),
                np.array(fp_full["gamma_t"]),
            )
        )
        center = group.attrs["template_centers"]
        center_transformed = group.attrs["fit_centers"]
        n_point = fp_full["n_point"]
        n_gamma = fp_full["n_gamma"]
        have_gamma = group["focal_plane"].attrs["measured_gamma"]
        template = None
        transform = Transform.load(group["transform"])
        transform_nocm = Transform.load(group["transform"], "_nocm")
        if "wafer_slot" in group["focal_plane"].attrs:
            wafer_slot = group["focal_plane"].attrs["wafer_slot"]
        else:
            logger.warning("No wafer slot found in this focal plane, may be old.")
            wafer_slot = "ws?"

        return FocalPlane(
            stream_id,
            wafer_slot,
            np.array(det_ids),
            avg_fp,
            np.array(weights),
            transformed,
            center,
            center_transformed,
            np.array(n_point),
            np.array(n_gamma),
            have_gamma,
            template,
            transform,
            transform_nocm,
        )


@dataclass
class OpticsTube:
    name: str
    center: NDArray[np.floating]
    transform: Transform = field(default_factory=Transform.identity)
    transform_fullcm: Transform = field(default_factory=Transform.identity)
    focal_planes: List[FocalPlane] = field(default_factory=list)
    center_transformed: NDArray[np.floating] = field(init=False)

    def __post_init__(self):
        self.center_transformed = mt.apply_transform(
            self.center, self.transform_fullcm.affine, self.transform_fullcm.shift
        )

    @classmethod
    def from_pointing_cfg(cls, pointing_cfg):
        name = pointing_cfg["tube_slot"]
        telescope_flavor = pointing_cfg["telescope_flavor"].upper()
        if telescope_flavor not in ["LAT", "SAT"]:
            raise ValueError("Telescope should be LAT or SAT")

        if telescope_flavor == "LAT":
            if pointing_cfg["zemax_path"] is None:
                raise ValueError("Must provide zemax_path for LAT")
            xi, eta, gamma = op.LAT_focal_plane(
                None,
                pointing_cfg["zemax_path"],
                x=0,
                y=0,
                pol=0,
                roll=pointing_cfg.get("roll", 0),
                tube_slot=name,
            )
        else:
            xi, eta, gamma = op.SAT_focal_plane(
                None,
                x=0,
                y=0,
                pol=0,
                roll=pointing_cfg.get("roll", 0),
                mapping_data=pointing_cfg.get("mapping_data", None),
            )
        center = np.array((xi, eta, gamma)).reshape((1, 3))

        return OpticsTube(name, center)

    def save(self, f, db_info, group="/"):
        g = f[group]
        g.create_group(self.name)
        _add_attrs(
            g[self.name],
            {"center": self.center, "center_transformed": self.center_transformed},
        )
        tr_path = os.path.join(group, self.name, "transform")
        self.transform.save(f, tr_path)
        self.transform_fullcm.save(f, tr_path, "_fullcm")
        for focal_plane in self.focal_planes:
            focal_plane.save(
                f, db_info, os.path.join(group, self.name, focal_plane.stream_id)
            )

    @classmethod
    def load(cls, group):
        name = group.name.split("/")[-1]
        center = group.attrs["center"]
        transform = Transform.load(group["transform"])
        transform_fullcm = Transform.load(group["transform"], "_fullcm")
        fps = [
            FocalPlane.load(group[grp])
            for grp in group.keys()
            if "transform" not in grp
        ]

        return OpticsTube(name, center, transform, transform_fullcm, fps)


@dataclass
class Receiver:
    optics_tubes: List[OpticsTube] = field(default_factory=list)
    center: NDArray[np.floating] = field(
        default_factory=partial(np.zeros, shape=(1, 3))
    )
    include_cm: bool = field(default=False)
    transform: Transform = field(default_factory=Transform.identity)
    center_transformed: NDArray[np.floating] = field(init=False)

    def __post_init__(self):
        self.center_transformed = mt.apply_transform(
            self.center, self.transform.affine, self.transform.shift
        )

    def save(self, f, db_info, group="/"):
        _add_attrs(
            f[group],
            {
                "center": self.center,
                "center_transformed": self.center_transformed,
                "include_cm": self.include_cm,
            },
        )
        self.transform.save(f, os.path.join(group, "transform"))
        for ot in self.optics_tubes:
            ot.save(f, db_info, group)

    @classmethod
    def load(cls, f, group="/"):
        center = f[group].attrs["center"]
        include_cm = f[group].attrs["include_cm"]
        transform = Transform.load(f[group]["transform"])
        ots = [
            OpticsTube.load(f[group][grp])
            for grp in f[group].keys()
            if "transform" not in grp
        ]

        return Receiver(ots, center, include_cm, transform)

    @classmethod
    def load_file(cls, path):
        with h5py.File(path, "r") as f:
            # Check if its an old file:
            if "transform" in f.keys():
                return {"": cls.load(f)}
            return {grp: cls.load(f, grp) for grp in f.keys()}

    @property
    def focal_planes(self):
        fps = []
        for ot in self.optics_tubes:
            fps += ot.focal_planes
        return fps

    @property
    def lims(self):
        xmax = np.max([np.nanmax(fp.transformed[:, 0]) for fp in self.focal_planes])
        xmin = np.min([np.nanmin(fp.transformed[:, 0]) for fp in self.focal_planes])
        ymax = np.max([np.nanmax(fp.transformed[:, 1]) for fp in self.focal_planes])
        ymin = np.min([np.nanmin(fp.transformed[:, 1]) for fp in self.focal_planes])
        return (xmin, xmax), (ymin, ymax)


# Plotting Functions
def plot_ufm(focal_plane, plot_dir):
    nominal = focal_plane.template.fp
    measured = focal_plane.avg_fp
    transformed = focal_plane.transformed
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    # Plot pointing
    ax1.scatter(
        nominal[:, 0],
        nominal[:, 1],
        alpha=0.4,
        color="blue",
        label="nominal",
        marker="P",
    )
    ax1.scatter(
        transformed[:, 0],
        transformed[:, 1],
        alpha=0.4,
        color="black",
        label="transformed",
        marker="X",
    )
    ax1.scatter(measured[:, 0], measured[:, 1], alpha=0.4, color="orange", label="fit")
    ax1.set_xlabel("Xi (rad)")
    ax1.set_ylabel("Eta (rad)")
    ax1.set_aspect("equal")
    ax1.legend()

    # Histogram of differences
    dist = focal_plane.dist[focal_plane.isfinite] * 180 * 60 * 60 / np.pi
    dist_thresh = np.percentile(dist, 97)
    bins = max(int(len(dist) / 20), 10)
    ax2.hist(dist[dist < dist_thresh], bins=bins)
    ax2.set_xlabel("Residual (arcseconds)")

    fig.suptitle(f"{focal_plane.stream_id}")
    fig.set_size_inches(2 * fig.get_size_inches())
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plot_dir, f"{focal_plane.stream_id}.png"), bbox_inches="tight"
        )
        plt.clf()


def plot_ot(ot, plot_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    dists = [fp.dist[fp.isfinite] * 180 * 60 * 60 / np.pi for fp in ot.focal_planes]
    xis = [fp.transformed[fp.isfinite, 0] for fp in ot.focal_planes]
    etas = [fp.transformed[fp.isfinite, 1] for fp in ot.focal_planes]

    # Plot the radial dist
    r = np.sqrt(np.hstack(xis) ** 2 + np.hstack(etas) ** 2)
    dist = np.hstack(dists)
    dist_avg, edges, _ = binned_statistic(r, dist, "median", bins=50)
    r_bins = (edges[:-1] + edges[1:]) / 2
    max_dist = np.max(dist)

    ax1.scatter(r, dist, alpha=0.1)
    ax1.plot(r_bins, dist_avg, color="black")
    ax1.set_ylim((None, np.percentile(dist, 95)))
    ax1.set_xlabel("Radius (rad)")
    ax2.set_ylabel("Residual (arcseconds)")

    # Plot a heatmap
    cf = None
    for dist, xi, eta in zip(dists, xis, etas):
        cf = ax2.tricontourf(
            xi, eta, dist, levels=20, vmin=-1 * max_dist, vmax=max_dist, cmap="coolwarm"
        )
    if cf is not None:
        fig.colorbar(cf, ax=ax2, label="arcseconds")
    ax2.set_aspect("equal")
    ax2.set_xlabel("Xi (rad)")
    ax2.set_ylabel("Eta (rad)")
    fig.suptitle(f"{ot.name}")

    fig.set_size_inches(2 * fig.get_size_inches())
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{ot.name}.png"), bbox_inches="tight")
        plt.clf()


def plot_by_gamma(focal_plane, plot_dir):
    fig, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True)
    dist_thresh = np.percentile(focal_plane.dist[focal_plane.isfinite], 97)
    msk = (focal_plane.dist < dist_thresh) + focal_plane.isfinite
    rhombi = np.unique(focal_plane.template.rhombus)
    gammas = (focal_plane.template.fp[msk, 2] * 180 / np.pi) % 180.0
    bins = np.linspace(0, 180, 13)
    for i, name in enumerate(("xi", "eta", "gamma")):
        d = focal_plane.diff[msk, i] * 180 * 60 * 60 / np.pi
        axs[i].set_title(name)
        if np.sum(np.isfinite(d)) == 0:
            continue
        medians, *_ = binned_statistic(gammas, d, statistic="median", bins=bins)
        for rhombus in rhombi:
            rmsk = focal_plane.template.rhombus[msk] == rhombus
            if not (np.any(rmsk)):
                continue
            axs[i].scatter(gammas[rmsk], d[rmsk], alpha=0.2, label=f"Rhombus {rhombus}")
        axs[i].scatter(np.arange(7.5, 180, 15), medians, color="black")
        leg = axs[i].legend()
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    axs[0].set_ylabel("Diff (arcseconds)")
    axs[1].set_xlabel("Nominal Gamma (deg)")
    fig.suptitle(f"{focal_plane.stream_id} By Gamma")
    fig.set_size_inches(2 * fig.get_size_inches())
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plot_dir, f"{focal_plane.stream_id}_by_gamma.png"),
            bbox_inches="tight",
        )
        plt.clf()


def plot_receiver(receiver, plot_dir):
    max_diff = 0
    valid_ids = []
    for fp in receiver.focal_planes:
        diff = np.nanpercentile(np.abs(fp.diff), 97)
        if diff > max_diff:
            max_diff = diff
        valid_ids += list(fp.template.valid_ids)
    valid_ids = np.unique(valid_ids)
    max_diff *= 180 * 60 * 60 / np.pi
    xlims, ylims = receiver.lims

    fig, axs_all = plt.subplots(
        len(valid_ids), 3, sharex="col", sharey="row", constrained_layout=True
    )
    axs = axs_all.flat
    axs[0].set_title("Xi")
    axs[1].set_title("Eta")
    axs[2].set_title("Gamma")
    cf = None
    for i in range(len(valid_ids)):
        for j in range(3):
            axs[3 * i + j].set_aspect("equal")
            axs[3 * i + j].set_xlim(xlims)
            axs[3 * i + j].set_ylim(ylims)
        for fp in receiver.focal_planes:
            msk = (fp.template.id_strs == valid_ids[i]) * fp.isfinite
            if np.sum(msk) < 3:
                continue
            diff = fp.diff * 180 * 60 * 60 / np.pi
            cf = axs[3 * i + 0].tricontourf(
                fp.transformed[msk, 0],
                fp.transformed[msk, 1],
                diff[msk, 0],
                levels=20,
                vmin=-1 * max_diff,
                vmax=max_diff,
                cmap="coolwarm",
            )
            cf = axs[3 * i + 1].tricontourf(
                fp.transformed[msk, 0],
                fp.transformed[msk, 1],
                diff[msk, 1],
                levels=20,
                vmin=-1 * max_diff,
                vmax=max_diff,
                cmap="coolwarm",
            )
            if fp.have_gamma:
                cf = axs[3 * i + 2].tricontourf(
                    fp.transformed[msk, 0],
                    fp.transformed[msk, 1],
                    diff[msk, 2],
                    levels=20,
                    vmin=-1 * max_diff,
                    vmax=max_diff,
                    cmap="coolwarm",
                )
        axs[3 * i + 0].set_ylabel(f"{valid_ids[i]}\nEta (rad)")
    if cf is not None:
        fig.colorbar(cf, ax=axs_all.ravel().tolist(), label="arcsecs")
    axs[-3].set_xlabel("Xi (rad)")
    axs[-2].set_xlabel("Xi (rad)")
    axs[-1].set_xlabel("Xi (rad)")
    fig.suptitle("Full Receiver Residuals")
    fig.set_size_inches(2 * fig.get_size_inches())
    if plot_dir is None:
        plt.show()
    else:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"receiver.png"), bbox_inches="tight")
        plt.clf()

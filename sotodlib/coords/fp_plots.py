import os
from copy import deepcopy
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import binned_statistic

from .fp_containers import (
    DetectorOffsets,
    FocalPlane,
    FocalPlaneCollection,
    OpticsTube,
    Receiver,
    NuisanceParams,
)


def plot_detector_offsets(
    ax: mpl.axes._axes.Axes,  # type: ignore
    offsets: DetectorOffsets,
    text: Optional[str] = None,
    fontsize: int = 16,
    **scatter_kwargs,
):
    """
    Plot detector offsets on an existing axes.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axes on which to plot.
    offsets : DetectorOffsets
        Detector offsets to plot.
    text: Optional[str], default: None
        Label used for the detector group.
        Will be overlayed at the center of the array.
    fontsize : int, default=16
        Font size for the group label.
    **scatter_kwargs
        kwargs to pass to scatter.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot artist.
    """
    scatter = ax.scatter(offsets.xyz[:, 0], offsets.xyz[:, 1], **scatter_kwargs)
    if text is not None:
        x0 = np.nanmedian(offsets.xyz[:, 0])
        y0 = np.nanmedian(offsets.xyz[:, 1])
        ax.text(
            x0, y0, text, ha="center", va="center", fontsize=fontsize, fontweight="bold"
        )

    return scatter


def plot_receiver_templates(
    receiver: Receiver,
    ot_locs: dict,
    output_dir: str,
    fontsize: int = 16,
):
    """Plot the templates contained in a Receiver.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing the optics tubes and focal-plane templates.
    ot_locs : dict
        Optics-tube location information. Each entry should contain `dx`
        and `dy`, matching the structure of `optics_tubes.yaml`.
    output_dir : str
        Directory in which to save the plots.
    fontsize : int, default=16
        Font size used for focal-plane labels.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Generated figures keyed by plot name.
    """
    outdir = os.path.join(output_dir, receiver.epoch, "nominal")
    os.makedirs(outdir, exist_ok=True)

    # Give each OT a stable color.
    ot_names = [ot.name for ot in receiver.optics_tubes]
    ot_nums = {name: i + 1 for i, name in enumerate(ot_names)}

    def ot_color(ot_name):
        return plt.cm.tab20((ot_nums[ot_name] - 1) % 20 / 20.0)

    def add_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label_ in zip(handles, labels):
            unique[label_] = handle

        if unique:
            ax.legend(
                list(unique.values())[::-1],
                list(unique.keys())[::-1],
            )

    def save_and_finish(fig, name):
        fig.savefig(
            os.path.join(outdir, f"{receiver.name}_{receiver.epoch}_{name}.png")
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 15))
    for ot in receiver.optics_tubes:
        for fp, ws in zip(ot.focal_planes, ot.wafer_slots):
            plot_detector_offsets(
                ax,
                fp.template,
                text=f"{fp.name}\n({ws})",
                fontsize=fontsize,
                color=ot_color(ot.name),
                label=ot.name,
            )
    add_legend(ax)
    ax.set_xlabel("xi (rad)")
    ax.set_ylabel("eta (rad)")
    ax.set_title(f"{receiver.name.upper()} Template (On Sky)")
    ax.set_aspect("equal")
    save_and_finish(fig, "template_on_sky")

    # On sky, gamma colored
    fig, ax = plt.subplots(figsize=(15, 15))
    gamma_scatter = None
    for ot in receiver.optics_tubes:
        for fp, ws in zip(ot.focal_planes, ot.wafer_slots):
            gamma_scatter = plot_detector_offsets(
                ax,
                fp.template,
                text=f"{fp.name}\n({ws})",
                fontsize=fontsize,
                c=fp.template.gamma,
            )
    if gamma_scatter is not None:
        fig.colorbar(gamma_scatter, ax=ax, label="gamma (rad)")
    ax.set_xlabel("xi (rad)")
    ax.set_ylabel("eta (rad)")
    ax.set_title(f"{receiver.name.upper()} Template (On Sky)")
    ax.set_aspect("equal")
    save_and_finish(fig, "template_on_sky_gamma")

    # Back of receiver
    fig, ax = plt.subplots(figsize=(15, 15))
    for ot in receiver.optics_tubes:
        for fp, ws in zip(ot.focal_planes, ot.wafer_slots):
            plot_detector_offsets(
                ax,
                fp.template_ot,
                text=f"{fp.name}\n({ws})",
                fontsize=fontsize,
                color=ot_color(ot.name),
                label=ot.name,
            )
    add_legend(ax)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{receiver.name.upper()} Template (From Back)")
    ax.set_aspect("equal")
    save_and_finish(fig, "template_back")

    # Front of receiver.
    fig, ax = plt.subplots(figsize=(15, 15))
    for ot in receiver.optics_tubes:
        dx = ot_locs[ot.name]["dx"]
        dy = ot_locs[ot.name]["dy"]
        for fp, ws in zip(ot.focal_planes, ot.wafer_slots):
            template = deepcopy(fp.template_ot)
            x = template.xyz[:, 0] - dx
            y = template.xyz[:, 1] - dy
            x = -1 * (-1 * x + dx)
            y = -1 * y + dy
            template.xyz[:, 0] = x
            template.xyz[:, 1] = y
            plot_detector_offsets(
                ax,
                template,
                text=f"{fp.name}\n({ws})",
                fontsize=fontsize,
                color=ot_color(ot.name),
                label=ot.name,
            )
    add_legend(ax)
    ax.set_xlabel("x (mm), Not Accurate!")
    ax.set_ylabel("y (mm), Not Accurate!")
    ax.set_title(f"{receiver.name.upper()} Template (At Window, From Front)")
    ax.set_aspect("equal")
    save_and_finish(fig, "template_front")


def plot_ufm(
    focal_plane: FocalPlane | FocalPlaneCollection,
    output_dir: str,
    figsize: tuple[float, float] = (12, 6),
    with_pm: bool = True,
):
    """
    Plot the UFM focal-plane fit and residual distribution.

    Parameters
    ----------
    focal_plane : FocalPlane | FocalPlaneCollection
        Focal-plane object for a specefic observation.
    output_dir : str
        Directory in which to save the plot.
    figsize : tuple[float, float], default: (12, 6)
        Figure dimensions in inches.
    with_pm : bool, default: True
        If True then use `pm_data` if `False` use `data`.
        Only matters for `FocalPlane`.
    """
    wmsk = np.isfinite(focal_plane.weights) * (focal_plane.weights > 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    data = focal_plane.data
    transformed = focal_plane.transformed
    if isinstance(focal_plane, FocalPlane) and with_pm:
        data = focal_plane.pm_data
        transformed = focal_plane.pm_transformed

    # Plot focal-plane pointing.
    plot_detector_offsets(
        ax1,
        focal_plane.template,
        color="blue",
        alpha=0.4,
        label="template",
        marker="P",
    )
    plot_detector_offsets(
        ax1,
        transformed,
        color="black",
        alpha=0.4,
        label="transformed",
        marker="X",
    )
    plot_detector_offsets(
        ax1,
        data[wmsk],
        color="orange",
        alpha=0.4,
        label="measured",
    )
    ax1.set_xlabel("Xi (rad)")
    ax1.set_ylabel("Eta (rad)")
    ax1.set_aspect("equal")
    ax1.legend()

    # Plot histogram of residuals.
    dist = 3600 * np.linalg.norm(
        np.rad2deg((transformed.xyz - data.xyz)[wmsk, :2]), axis=1
    )
    if len(dist) > 0:
        dist_thresh = np.percentile(dist, 97)
        bins = max(int(len(dist) / 20), 10)
        ax2.hist(dist[dist < dist_thresh], bins=bins)

    ax2.set_xlabel('Residual (")')
    ax2.set_ylabel("Detectors")

    title = focal_plane.name
    if isinstance(focal_plane, FocalPlane):
        title += f" {focal_plane.meas_id}"
    fig.suptitle(title.replace(" ", "\n"))
    fig.savefig(
        os.path.join(output_dir, f"{title.replace(' ', '+')}.png"), bbox_inches="tight"
    )
    plt.close(fig)


def plot_optics_tube(
    ot: OpticsTube, output_dir, figsize: tuple[float, float] = (12, 6), bins: int = 50
):
    """
    Plot radial residuals for the fitted focal-plane collections in an optics tube.

    The signed residual is the projection of `transformed.xyz - data.xyz`
    onto the radial direction defined by `transformed.xyz`. Positive
    values indicate an outward radial residual, while negative values
    indicate an inward radial residual.

    Parameters
    ----------
    ot : OpticsTube
        OpticsTube containing fitted FocalPlaneCollection objects.
    output_dir : str
        Directory in which to save the plot.
    figsize : tuple[float, float], default=(12, 6)
        Figure dimensions in inches.
    bins : int, default=50
        Number of radial bins used for the median residual profile.
    """
    dists = []
    xis = []
    etas = []
    arcsec_per_rad = np.rad2deg(1.0) * 3600
    for fp in ot.focal_planes:
        if fp.pad:
            continue
        data = fp.data
        transformed = fp.transformed
        xi = transformed.xyz[:, 0]
        eta = transformed.xyz[:, 1]
        dxi = transformed.xyz[:, 0] - data.xyz[:, 0]
        deta = transformed.xyz[:, 1] - data.xyz[:, 1]
        radius = np.hypot(xi, eta)
        valid = (
            np.isfinite(fp.weights)
            * (fp.weights > 0)
            * np.isfinite(xi)
            * np.isfinite(eta)
            * np.isfinite(dxi)
            * np.isfinite(deta)
            * (radius > 0)
        )
        if not np.any(valid):
            continue
        radial_residual = (dxi[valid] * xi[valid] + deta[valid] * eta[valid]) / radius[
            valid
        ]
        dists.append(radial_residual * arcsec_per_rad)
        xis.append(xi[valid])
        etas.append(eta[valid])
    if not dists:
        return None

    dist = np.hstack(dists)
    xi = np.hstack(xis)
    eta = np.hstack(etas)
    radius = np.hypot(xi, eta)
    finite = (
        np.isfinite(dist) * np.isfinite(radius) * np.isfinite(xi) * np.isfinite(eta)
    )
    dist = dist[finite]
    xi = xi[finite]
    eta = eta[finite]
    radius = radius[finite]
    if len(dist) == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    dist_avg, edges, _ = binned_statistic(radius, dist, statistic="median", bins=bins)
    radius_bins = 0.5 * (edges[:-1] + edges[1:])
    ax1.scatter(radius, dist, s=6, alpha=0.1)
    ax1.plot(radius_bins, dist_avg, color="black", linewidth=2, label="Median")
    ax1.axhline(0, color="gray", linewidth=1)
    limit = np.nanpercentile(np.abs(dist), 95)
    if np.isfinite(limit) and limit > 0:
        ax1.set_ylim(-limit, limit)
    ax1.set_xlabel("Radius (rad)")
    ax1.set_ylabel("Signed radial residual (arcseconds)")
    ax1.legend()

    color_limit = np.nanpercentile(np.abs(dist), 99)
    cf = None
    if len(dist) >= 3 and np.isfinite(color_limit) and color_limit > 0:
        cf = ax2.scatter(
            xi, eta, c=dist, s=10, vmin=-color_limit, vmax=color_limit, cmap="coolwarm"
        )
    if cf is not None:
        fig.colorbar(cf, ax=ax2, label='Radial residual (")')
    ax2.set_aspect("equal")
    ax2.set_xlabel("Xi (rad)")
    ax2.set_ylabel("Eta (rad)")
    fig.suptitle(ot.name)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{ot.name}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_receiver(
    receiver: Receiver,
    output_dir: str,
    percentile: float = 97,
    levels: int = 21,
    min_points: int = 3,
    dpi: int = 180,
):
    """
    Plot receiver-wide residual maps grouped by detector split.

    Splits are grouped such that any splits that appear together within
    the same template are assigned to different rows, while splits that
    never co-occur may share a row. The resulting rows are labeled by
    the corresponding split names.

    If all focal-plane collections use fake gamma values
    (`fake_gamma=True`) then the gamma column is omitted.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing fitted optics tubes and focal-plane
        collections.
    output_dir : str
        Directory in which to save the plot.
    percentile : float, default=97
        Percentile of the absolute residual distribution used to define
        the symmetric color scale.
    levels : int, default=21
        Number of contour levels used for the residual maps.
    min_points : int, default=3
        Minimum number of valid detectors required to plot a given split
        within a focal-plane collection.
    dpi : int, default=180
        Resolution used when saving the figure.
    """
    arcsec_per_rad = 180 * 3600 / np.pi
    focal_planes = [fp for ot in receiver.optics_tubes for fp in ot.focal_planes]
    if len(focal_planes) == 0:
        return None

    # Decide whether gamma should be plotted.
    include_gamma = any(not fp.fake_gamma for fp in focal_planes)

    components = [0, 1]
    titles = ["Xi", "Eta"]
    if include_gamma:
        components.append(2)
        titles.append("Gamma")
    ncols = len(components)

    # Build split groupings.
    all_splits = set()
    neighbors = {}
    for fp in focal_planes:
        valid = np.isfinite(fp.weights) * (fp.weights > 0)
        splits = sorted(np.unique(np.asarray(fp.template.split)[valid]), key=str)
        all_splits.update(splits)
        for split in splits:
            neighbors.setdefault(split, set())
        for i, split_a in enumerate(splits):
            for split_b in splits[i + 1 :]:
                neighbors[split_a].add(split_b)
                neighbors[split_b].add(split_a)
    split_to_group = {}
    for split in sorted(all_splits, key=str):
        unavailable = {
            split_to_group[neighbor]
            for neighbor in neighbors[split]
            if neighbor in split_to_group
        }
        group = 0
        while group in unavailable:
            group += 1
        split_to_group[split] = group
    nrows = max(split_to_group.values()) + 1
    split_groups = [[] for _ in range(nrows)]
    for split, group in split_to_group.items():
        split_groups[group].append(split)
    for group in split_groups:
        group.sort(key=str)

    # Gather plotting data.
    entries = []
    all_residuals = []
    all_x = []
    all_y = []
    for fp in focal_planes:
        if fp.pad:
            continue
        transformed = fp.transformed.xyz
        data = fp.data.xyz
        splits = np.asarray(fp.template.split)
        residual = (transformed - data) * arcsec_per_rad
        valid = (
            np.isfinite(fp.weights)
            * (fp.weights > 0)
            * np.isfinite(transformed[:, 0])
            * np.isfinite(transformed[:, 1])
        )
        for split in np.unique(splits):
            msk = valid * (splits == split)
            if np.sum(msk) < min_points:
                continue
            entry = {
                "group": split_to_group[split],
                "split": split,
                "fp": fp,
                "x": transformed[msk, 0],
                "y": transformed[msk, 1],
                "residual": residual[msk],
            }
            entries.append(entry)
            all_x.append(entry["x"])
            all_y.append(entry["y"])
            all_residuals.append(entry["residual"][:, :2].ravel())
            if include_gamma and not fp.fake_gamma:
                all_residuals.append(entry["residual"][:, 2])
    if len(entries) == 0:
        return None
    all_residuals = np.concatenate(all_residuals)
    max_diff = np.nanpercentile(np.abs(all_residuals), percentile)

    if not np.isfinite(max_diff) or max_diff <= 0:
        max_diff = np.nanmax(np.abs(all_residuals))
    if not np.isfinite(max_diff) or max_diff <= 0:
        max_diff = 1.0

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0:
        dx = 1.0
    if dy <= 0:
        dy = 1.0
    xlims = (
        xmin - 0.05 * dx,
        xmax + 0.05 * dx,
    )
    ylims = (
        ymin - 0.05 * dy,
        ymax + 0.05 * dy,
    )

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols + 3, 6 * nrows + 3),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    contour_levels = np.linspace(
        -max_diff,
        max_diff,
        levels,
    )
    norm = Normalize(
        vmin=-max_diff,
        vmax=max_diff,
    )
    for col, title in enumerate(titles):
        axs[0, col].set_title(
            title,
            fontsize=18,
            fontweight="bold",
        )
    for row in range(nrows):
        split_label = " / ".join(map(str, split_groups[row]))
        axs[row, 0].set_ylabel(
            f"{split_label}\nEta (rad)",
            fontsize=14,
        )
        for col in range(ncols):
            ax = axs[row, col]
            ax.set_aspect("equal")
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.axhline(0, color="0.8", lw=0.5)
            ax.axvline(0, color="0.8", lw=0.5)

    # Plot residual maps
    for entry in entries:
        row = entry["group"]
        x = entry["x"]
        y = entry["y"]
        residual = entry["residual"]
        for col, component in enumerate(components):
            if component == 2 and entry["fp"].fake_gamma:
                continue
            z = residual[:, component]
            valid = np.isfinite(x) * np.isfinite(y) * np.isfinite(z)
            if np.sum(valid) < min_points:
                continue
            try:
                axs[row, col].tricontourf(
                    x[valid],
                    y[valid],
                    z[valid],
                    levels=contour_levels,
                    cmap="coolwarm",
                    norm=norm,
                    extend="both",
                )
            except (RuntimeError, ValueError):
                axs[row, col].scatter(
                    x[valid],
                    y[valid],
                    c=z[valid],
                    cmap="coolwarm",
                    norm=norm,
                    s=8,
                    linewidths=0,
                )

    # Labels
    for col in range(ncols):
        axs[-1, col].set_xlabel("Xi (rad)", fontsize=14)
    sm = ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])
    fig.colorbar(
        sm, ax=axs.ravel().tolist(), label="Residual (arcseconds)", shrink=0.92
    )
    fig.suptitle(f"{receiver.name} Receiver Residuals", fontsize=22, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            output_dir,
            f"{receiver.name}.png",
        ),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_displacements(
    x0: np.ndarray,
    y0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    colorbys: list[tuple[Optional[str], Optional[np.ndarray]]],
    output_dir: str,
    filename_prefix: str,
    title_prefix: str,
    plot_scale: float,
    scale_str: str,
    figsize: tuple[float, float] = (14, 14),
):
    """
    Plot displacement vectors, optionally colored by additional quantities.

    Parameters
    ----------
    x0, y0 : np.ndarray
        Starting coordinates of each displacement vector.
    dx, dy : np.ndarray
        Displacement components before plotting scale is applied.
    colorbys : list[tuple[Optional[str], Optional[np.ndarray]]]
        Quantities used to color the arrows.
    output_dir : str
        Directory in which to save the plots.
    filename_prefix : str
        Prefix used for output filenames.
    title_prefix : str
        Prefix used for plot titles.
    plot_scale : float
        Multiplicative scale applied to dx and dy.
    scale_str : str
        Description of the displacement scale for plot titles.
    figsize : tuple[float, float], default=(14, 14)
        Figure dimensions in inches.
    """
    os.makedirs(output_dir, exist_ok=True)
    dx = np.asarray(dx) * plot_scale
    dy = np.asarray(dy) * plot_scale
    nobs = len(dx)
    for name, color_data in colorbys:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        if name is None:
            ax.quiver(
                x0,
                y0,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
            )
            title = f"{title_prefix} Observation Displacements"
            fname = f"{filename_prefix}_displacements.png"
        else:
            if color_data is None:
                raise ValueError(f"{name} specified without data")
            if len(color_data) != nobs:
                raise ValueError(
                    f"{name}: expected {nobs} values but got {len(color_data)}"
                )
            q = ax.quiver(
                x0,
                y0,
                dx,
                dy,
                color_data,
                angles="xy",
                scale_units="xy",
                cmap="viridis",
            )
            fig.colorbar(q, ax=ax, label=name)
            title = (
                f"{title_prefix} Observation Displacements by {name}\n" f"{scale_str}"
            )
            fname = f"{filename_prefix}_displacements_{name}.png"
        ax.set_title(title)
        ax.set_xlabel("Xi (rad)")
        ax.set_ylabel("Eta (rad)")
        ax.set_aspect("equal")
        fig.savefig(
            os.path.join(output_dir, fname),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_receiver_displacements(
    receiver: Receiver,
    colorbys: list[tuple[Optional[str], Optional[np.ndarray]]],
    output_dir: str,
    figsize: tuple[float, float] = (14, 14),
):
    """
    Plot observation-by-observation focal-plane displacements relative
    to the final solved focal-plane collections.

    Each arrow corresponds to one FocalPlane measurement. Its location
    is given by the associated focal-plane template center and its
    direction is given by the median displacement between the
    observation and the solved collection coordinates.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing solved focal-plane collections.
    colorbys : list[tuple[Optional[str], Optional[np.ndarray]]]
        Quantities used to color the arrows. Each array must contain one
        value per measurement in the receiver. Passing (None, None)
        produces an uncolored plot.
    output_dir : str
        Directory in which to save the plots.
    figsize : tuple[float, float], default=(14, 14)
        Figure dimensions in inches.
    """
    arcmin_per_rad = 180 * 60 / np.pi
    plot_scale = 0.01 * arcmin_per_rad
    scale_str = "1 arcmin = 0.01 rad"
    x0 = []
    y0 = []
    dx = []
    dy = []
    for ot in receiver.optics_tubes:
        for fp in ot.focal_planes:
            if fp.pad or fp.template.center is None:
                continue
            xc, yc = fp.template.center[:2]
            for meas in fp.focal_planes:
                msk = np.isfinite(meas.weights) & (meas.weights > 0)
                if np.sum(msk) == 0:
                    continue
                disp_x = np.nanmedian(
                    meas.pm_transformed.xyz[msk, 0] - fp.transformed.xyz[msk, 0]
                )
                disp_y = np.nanmedian(
                    meas.pm_transformed.xyz[msk, 1] - fp.transformed.xyz[msk, 1]
                )
                x0.append(xc)
                y0.append(yc)
                dx.append(disp_x)
                dy.append(disp_y)

    plot_displacements(
        x0=np.asarray(x0),
        y0=np.asarray(y0),
        dx=np.asarray(dx),
        dy=np.asarray(dy),
        colorbys=colorbys,
        output_dir=output_dir,
        filename_prefix=receiver.name,
        title_prefix=receiver.name,
        plot_scale=plot_scale,
        scale_str=scale_str,
        figsize=figsize,
    )


def plot_receiver_nuisance_displacements(
    receiver: Receiver,
    nuisance_params: NuisanceParams,
    params_to_plot: list[tuple[str, str, Optional[str], str, float]],
    colorbys: list[tuple[Optional[str], Optional[np.ndarray]]],
    output_dir: str,
    figsize: tuple[float, float] = (14, 14),
):
    """
    Plot observation-by-observation focal-plane displacements using
    fitted nuisance parameters.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing the focal-plane measurements.
    nuisance_params : NuisanceParams
        Per-observation nuisance parameters.
    params_to_plot : list[tuple[str, str, Optional[str], str, float]]
        Parameter combinations to plot. Each tuple contains:
          - x: Nuisance parameter name for dx.
          - y: Nuisance parameter name for dy.
          - rot: Optional nuisance parameter name for rotation.
          - xy_units: Units of the x and y parameters.
          - scale: Parameter value corresponding to 0.01 rad.
    colorbys : list[tuple[Optional[str], Optional[np.ndarray]]]
        Quantities used to color the arrows. Each array must contain one
        value per measurement in the receiver. Passing (None, None)
        produces an uncolored plot.
    output_dir : str
        Directory in which to save the plots.
    figsize : tuple[float, float], default=(14, 14)
        Figure dimensions in inches.
    """
    if len(params_to_plot) == 0:
        return
    x0 = []
    y0 = []
    obs_ids = []
    for ot in receiver.optics_tubes:
        for fp in ot.focal_planes:
            if fp.pad or fp.template.center is None:
                continue
            xc, yc = fp.template.center[:2]
            for meas in fp.focal_planes:
                msk = np.isfinite(meas.weights) * (meas.weights > 0)
                if np.sum(msk) == 0:
                    continue
                x0.append(xc)
                y0.append(yc)
                obs_ids.append(meas.meas_id)
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    for x_name, y_name, rot_name, xy_units, scale in params_to_plot:
        dx = []
        dy = []
        for obs_id in obs_ids:
            pars, _ = nuisance_params[obs_id]
            if x_name not in pars or y_name not in pars:
                raise ValueError(
                    f"Missing nuisance parameters {x_name!r} or {y_name!r} for observation {obs_id}"
                )
            disp_x = pars[x_name]
            disp_y = pars[y_name]
            if rot_name is not None:
                if rot_name not in pars:
                    raise ValueError(
                        f"Missing rotation parameter {rot_name!r} for observation {obs_id}"
                    )
                theta = pars[rot_name]
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                disp_x, disp_y = (
                    cos_theta * disp_x - sin_theta * disp_y,
                    sin_theta * disp_x + cos_theta * disp_y,
                )
            dx.append(disp_x)
            dy.append(disp_y)
        param_names = f"{x_name}, {y_name}"
        if rot_name is not None:
            param_names += f", {rot_name}"
        label = f"{receiver.name} {param_names}"
        plot_displacements(
            x0=x0,
            y0=y0,
            dx=np.asarray(dx),
            dy=np.asarray(dy),
            colorbys=colorbys,
            output_dir=output_dir,
            filename_prefix=label.replace(" ", "_").replace(", ", "_"),
            title_prefix=label,
            plot_scale=0.01 / scale,
            scale_str=f"{scale:g} {xy_units} = .01 rad",
            figsize=figsize,
        )


def plot_receiver_nuisance_transform_displacements(
    receiver: Receiver,
    nuisance_params: NuisanceParams,
    colorbys: list[tuple[Optional[str], Optional[np.ndarray]]],
    output_dir: str,
    figsize: tuple[float, float] = (14, 14),
):
    """
    Plot observation-by-observation focal-plane displacements using
    the nuisance transforms stored in NuisanceParams.

    Each arrow corresponds to one FocalPlane measurement. Its location
    is given by the associated focal-plane template center, and its
    direction is given by the translation component of the observation's
    nuisance transform, rotated by the transform rotation angle.

    The first two elements of `transform.shift` are interpreted as dx
    and dy, in radians. The displacement is rotated by `transform.rot`.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing the focal-plane measurements.
    nuisance_params : NuisanceParams
        Per-observation nuisance parameters and transforms.
    colorbys : list[tuple[Optional[str], Optional[np.ndarray]]]
        Quantities used to color the arrows. Each array must contain one
        value per measurement in the receiver. Passing (None, None)
        produces an uncolored plot.
    output_dir : str
        Directory in which to save the plots.
    figsize : tuple[float, float], default=(14, 14)
        Figure dimensions in inches.
    """
    arcmin_per_rad = 180 * 60 / np.pi
    plot_scale = 0.01 * arcmin_per_rad
    scale_str = "1 arcmin = 0.01 rad"

    x0 = []
    y0 = []
    dx = []
    dy = []

    for ot in receiver.optics_tubes:
        for fp in ot.focal_planes:
            if fp.pad or fp.template.center is None:
                continue
            xc, yc = fp.template.center[:2]
            for meas in fp.focal_planes:
                msk = np.isfinite(meas.weights) & (meas.weights > 0)
                if np.sum(msk) == 0:
                    continue
                _, transform = nuisance_params[meas.meas_id]
                disp_x, disp_y = transform.shift[:2]
                theta = transform.rot
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                disp_x, disp_y = (
                    cos_theta * disp_x - sin_theta * disp_y,
                    sin_theta * disp_x + cos_theta * disp_y,
                )
                x0.append(xc)
                y0.append(yc)
                dx.append(disp_x)
                dy.append(disp_y)

    plot_displacements(
        x0=np.asarray(x0),
        y0=np.asarray(y0),
        dx=np.asarray(dx),
        dy=np.asarray(dy),
        colorbys=colorbys,
        output_dir=output_dir,
        filename_prefix=f"{receiver.name}_transforms",
        title_prefix=f"{receiver.name} Nuisance Transforms",
        plot_scale=plot_scale,
        scale_str=scale_str,
        figsize=figsize,
    )


def plot_receiver_resid_vs_enc(
    receiver: Receiver,
    output_dir: str,
    figsize: tuple[float, float] = (18, 10),
    alpha: float = 0.15,
    markersize: float = 4,
):
    """
    Plot focal-plane residuals versus encoder coordinates for an entire
    receiver.

    Each row corresponds to a component of `fpc_resid` and each column
    corresponds to an encoder coordinate `(az, el, roll)`. Residuals are
    shown in arcseconds and encoder coordinates are shown in degrees.

    Points are colored by focalplane name.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing fitted focal-plane collections.
    output_dir : str
        Directory in which to save the plot.
    figsize : tuple[float, float], default=(18, 10)
        Figure dimensions in inches.
    alpha : float, default=0.15
        Point transparency.
    markersize : float, default=4
        Marker size.
    """

    focal_planes = [fp for ot in receiver.optics_tubes for fp in ot.focal_planes]
    if len(focal_planes) == 0:
        return

    include_gamma = any(not fp.fake_gamma for fp in focal_planes)
    resid_indices = [0, 1]
    resid_labels = [
        'Xi Residual (")',
        'Eta Residual (")',
    ]
    if include_gamma:
        resid_indices.append(2)
        resid_labels.append('Gamma Residual (")')
    enc_labels = [
        "Az (deg)",
        "El (deg)",
        "Roll (deg)",
    ]
    nrows = len(resid_indices)
    ncols = 3

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
        sharex="col",
    )

    colors = plt.cm.tab20(np.linspace(0, 1, len(focal_planes)))
    for color, fp in zip(colors, focal_planes):
        added_label = False
        for meas in fp.focal_planes:
            valid = np.isfinite(meas.weights) * (meas.weights > 0)
            if np.sum(valid) == 0:
                continue

            enc = np.rad2deg(meas.enc.xyz)
            for row, resid_idx in enumerate(resid_indices):
                resid = meas.fpc_resid[:, resid_idx] * 180 * 3600 / np.pi
                valid_row = valid * np.isfinite(resid)
                if np.sum(valid_row) == 0:
                    continue
                for col in range(3):
                    msk = valid_row * np.isfinite(enc[:, col])
                    if np.sum(msk) == 0:
                        continue
                    axs[row, col].scatter(
                        enc[msk, col],
                        resid[msk],
                        s=markersize,
                        alpha=alpha,
                        color=color,
                        label=(
                            fp.name
                            if not added_label and row == 0 and col == 0
                            else None
                        ),
                    )
            added_label = True

    for row, label in enumerate(resid_labels):
        axs[row, 0].set_ylabel(label)
    for col, label in enumerate(enc_labels):
        axs[0, col].set_title(label)
        axs[-1, col].set_xlabel(label)
    handles, labels = axs[0, 0].get_legend_handles_labels()

    if len(handles) > 0:
        fig.legend(
            handles,
            labels,
            loc="outside right upper",
        )

    fig.suptitle(
        f"{receiver.name} Residuals vs Encoder Coordinates",
    )
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            output_dir,
            f"{receiver.name}_resid_vs_enc.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_receiver_residuals(
    receiver: Receiver,
    output_dir: str,
    bins: int = 100,
    percentile: float = 99,
    figsize: tuple[float, float] = (16, 5),
    dpi: int = 180,
):
    """
    Plot histograms of all focal-plane collection residuals in a system.

    Residuals from all measurements are collected across all receivers,
    optics tubes, and focal-plane collections. Only detectors with
    finite, positive weights are included.

    Parameters
    ----------
    receiver : Receiver
        Receiver containing fitted focal-plane collections.
    output_dir : str
        Directory in which to save the plot.
    bins : int, default=100
        Number of histogram bins.
    percentile : float, default=99
        Percentile of the absolute residual distribution used to limit
        the histogram range.
    figsize : tuple[float, float], default=(16, 5)
        Figure dimensions in inches.
    dpi : int, default=180
        Resolution used when saving the figure.
    """
    arcsec_per_rad = 180 * 3600 / np.pi
    residuals = [[], [], [], []]
    for ot in receiver.optics_tubes:
        for fp in ot.focal_planes:
            if fp.pad:
                continue
            for meas in fp.focal_planes:
                valid = np.isfinite(meas.weights) * (meas.weights > 0)
                if not np.any(valid):
                    continue
                resid = meas.fpc_resid * arcsec_per_rad
                ndim = 3 - int(meas.fake_gamma)
                for i in range(ndim):
                    msk = valid * np.isfinite(resid[:, i])
                    if np.any(msk):
                        residuals[i].append(resid[msk, i])

                resid_norm = np.linalg.norm(resid[:, :ndim], axis=1)
                msk = valid * np.isfinite(resid_norm)
                if np.any(msk):
                    residuals[3].append(resid_norm[msk])

    residuals = [
        np.concatenate(component) if component else np.array([])
        for component in residuals
    ]

    components = [0, 1]
    labels = ['Xi Residual (")', 'Eta Residual (")']
    if len(residuals[2]) > 0:
        components.append(2)
        labels.append('Gamma Residual (")')
    components.append(3)
    labels.append('Residual Norm (")')
    if not any(len(residuals[i]) > 0 for i in components):
        return None

    fig, axs = plt.subplots(
        1, len(components), figsize=figsize, squeeze=False, constrained_layout=True
    )
    for ax, idx, label in zip(axs[0], components, labels):
        data = residuals[idx]
        if len(data) == 0:
            continue

        limit = np.nanpercentile(np.abs(data), percentile)
        if not np.isfinite(limit) or limit <= 0:
            limit = np.nanmax(np.abs(data))
        if np.isfinite(limit) and limit > 0:
            data = data[np.abs(data) <= limit]
        ax.hist(data, bins=bins, histtype="stepfilled", alpha=0.7)

        if idx != 3:
            ax.axvline(0, color="black", linestyle="--", linewidth=1)

        median = np.median(data)
        ax.axvline(
            median,
            color="red",
            linestyle="-",
            linewidth=1.5,
            label=f"Median: {median:.3g}",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Detectors")
        ax.legend()

    fig.suptitle(f"{receiver.name} Residuals")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, f"{receiver.name}_residuals.png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

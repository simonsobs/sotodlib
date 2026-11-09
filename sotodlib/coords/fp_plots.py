import os
from copy import deepcopy
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .fp_containers import DetectorOffsets, FocalPlane, FocalPlaneCollection, Receiver


def plot_detector_offsets(
    ax: mpl.axes._axes.Axes,
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
        If True then use `data_pm` if `False` use `data`.
        Only matters for `FocalPlane`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    data = focal_plane.data
    transformed = focal_plane.transformed
    if isinstance(focal_plane, FocalPlane) and with_pm:
        data = focal_plane.data_pm
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
        data,
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
        np.rad2deg(
            (transformed.xyz - data.xyz)[
                np.isfinite(focal_plane.weights) * (focal_plane.weights > 0), :2
            ]
        ),
        axis=1,
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

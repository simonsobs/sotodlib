# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Hardware visualization tools.
"""

import numpy as np

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import quaternionarray as qa


default_band_colors = {
    "LF1": (0.4, 0.4, 1.0, 0.2),
    "LF2": (0.4, 0.4, 1.0, 0.2),
    "MFF1": (0.4, 1.0, 0.4, 0.2),
    "MFF2": (0.4, 1.0, 0.4, 0.2),
    "MFS1": (0.4, 1.0, 0.4, 0.2),
    "MFS2": (0.4, 1.0, 0.4, 0.2),
    "UHF1": (1.0, 0.4, 0.4, 0.2),
    "UHF2": (1.0, 0.4, 0.4, 0.2),
}


def plot_detectors(dets, width, height, outfile, labels=False, bandcolor=None):
    """Visualize a dictionary of detectors.

    This makes a simple plot of the detector positions on the projected
    focalplane.  If the detector dictionary contains a key "fwhm", that will
    be assumed to be in arcminutes.  Otherwise a nominal value is used.
    If the color and label options are specified, they will override the
    defaults.

    Args:
        dets (dict): Dictionary of detector properties.
        width (float): Width of plot in degrees.
        height (float): Height of plot in degrees.
        outfile (str): Output PDF path.
        labels (bool): If True, label each detector.
        bandcolor (dict, optional): Dictionary of color values for each band.

    Returns:
        None

    """
    if bandcolor is None:
        bandcolor = default_band_colors
    xfigsize = 10.0
    yfigsize = 10.0
    figdpi = 75
    yfigpix = int(figdpi * yfigsize)
    ypixperdeg = yfigpix / height

    fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    half_width = 0.5 * width
    half_height = 0.5 * height
    ax.set_xlabel("Degrees", fontsize="large")
    ax.set_ylabel("Degrees", fontsize="large")
    ax.set_xlim([-half_width, half_width])
    ax.set_ylim([-half_height, half_height])

    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for d, props in dets.items():
        band = props["band"]
        pixel = props["pixel"]
        pol = props["pol"]
        quat = props["quat"]
        fwhm = props["fwhm"]

        # radius in degrees
        detradius = 0.5 * fwhm / 60.0

        # rotation from boresight
        rdir = qa.rotate(quat, zaxis).flatten()
        ang = np.arctan2(rdir[1], rdir[0])

        orient = qa.rotate(quat, xaxis).flatten()
        polang = np.arctan2(orient[1], orient[0])

        mag = np.arccos(rdir[2]) * 180.0 / np.pi
        xpos = mag * np.cos(ang)
        ypos = mag * np.sin(ang)

        detface = bandcolor[band]

        circ = plt.Circle((xpos, ypos), radius=detradius, fc=detface,
                          ec="black", linewidth=0.05*detradius)
        ax.add_artist(circ)

        ascale = 1.5

        xtail = xpos - ascale * detradius * np.cos(polang)
        ytail = ypos - ascale * detradius * np.sin(polang)
        dx = ascale * 2.0 * detradius * np.cos(polang)
        dy = ascale * 2.0 * detradius * np.sin(polang)

        detcolor = "black"
        if pol == "A":
            detcolor = (1.0, 0.0, 0.0, 1.0)
        if pol == "B":
            detcolor = (0.0, 0.0, 1.0, 1.0)

        ax.arrow(xtail, ytail, dx, dy, width=0.1*detradius,
                 head_width=0.3*detradius, head_length=0.3*detradius,
                 fc=detcolor, ec="none", length_includes_head=True)

        if labels:
            # Compute the font size to use for detector labels
            fontpix = 0.1 * detradius * ypixperdeg
            ax.text((xpos), (ypos), pixel,
                    color='k', fontsize=fontpix, horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(fc='white', ec='none', pad=0.2, alpha=1.0))
            xsgn = 1.0
            if dx < 0.0:
                xsgn = -1.0
            labeloff = 1.0 * xsgn * fontpix * len(pol) / ypixperdeg
            ax.text((xtail+1.0*dx+labeloff), (ytail+1.0*dy), pol,
                    color='k', fontsize=fontpix, horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(fc='none', ec='none', pad=0, alpha=1.0))

    plt.savefig(outfile)
    plt.close()
    return

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


def plot_detectors(dets, outfile, width=None, height=None, labels=False,
                   bandcolor=None):
    """Visualize a dictionary of detectors.

    This makes a simple plot of the detector positions on the projected
    focalplane.  The size of detector circles are controlled by the detector
    "fwhm" key, which is in arcminutes.  If the bandcolor is specified it will
    override the defaults.

    Args:
        outfile (str): Output PDF path.
        dets (dict): Dictionary of detector properties.
        width (float): Width of plot in degrees (None = autoscale).
        height (float): Height of plot in degrees (None = autoscale).
        labels (bool): If True, label each detector.
        bandcolor (dict, optional): Dictionary of color values for each band.

    Returns:
        None

    """
    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    wmin = 1.0
    wmax = -1.0
    hmin = 1.0
    hmax = -1.0
    if (width is None) or (height is None):
        # We are autoscaling.  Compute the angular extent of all detectors
        # and add some buffer.
        for d, props in dets.items():
            quat = np.array(props["quat"]).astype(np.float64)
            dir = qa.rotate(quat, zaxis).flatten()
            if (dir[0] > wmax):
                wmax = dir[0]
            if (dir[0] < wmin):
                wmin = dir[0]
            if (dir[1] > hmax):
                hmax = dir[1]
            if (dir[1] < hmin):
                hmin = dir[1]
        wmin = np.arcsin(wmin) * 180.0 / np.pi
        wmax = np.arcsin(wmax) * 180.0 / np.pi
        hmin = np.arcsin(hmin) * 180.0 / np.pi
        hmax = np.arcsin(hmax) * 180.0 / np.pi
        wbuf = 0.1 * (wmax - wmin)
        hbuf = 0.1 * (hmax - hmin)
        wmin -= wbuf
        wmax += wbuf
        hmin -= hbuf
        hmax += hbuf
        width = wmax - wmin
        height = hmax - hmin
    else:
        half_width = 0.5 * width
        half_height = 0.5 * height
        wmin = -half_width
        wmax = half_width
        hmin = -half_height
        hmax = half_height

    if bandcolor is None:
        bandcolor = default_band_colors
    xfigsize = 10.0
    yfigsize = xfigsize * (height / width)
    figdpi = 75
    yfigpix = int(figdpi * yfigsize)
    ypixperdeg = yfigpix / height

    fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("Degrees", fontsize="large")
    ax.set_ylabel("Degrees", fontsize="large")
    ax.set_xlim([wmin, wmax])
    ax.set_ylim([hmin, hmax])

    for d, props in dets.items():
        band = props["band"]
        pixel = props["pixel"]
        pol = props["pol"]
        quat = np.array(props["quat"]).astype(np.float64)
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


class clr:
    WHITE = "\033[97m"
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"

    def disable(self):
        self.WHITE = ""
        self.PURPLE = ""
        self.BLUE = ""
        self.GREEN = ""
        self.YELLOW = ""
        self.RED = ""
        self.ENDC = ""


def summary_text(hw):
    """Print a textual summary of a hardware configuration.

    Args:
        hw (Hardware): A hardware dictionary.

    Returns:
        None

    """
    for obj, props in hw.data.items():
        nsub = len(props)
        print("{}{:<12}: {}{:5d} objects{}".format(clr.WHITE, obj, clr.RED,
                                                   nsub, clr.ENDC))
        if nsub <= 2000:
            line = ""
            for k in list(props.keys()):
                if (len(line) + len(k)) > 72:
                    print("    {}{}{}".format(clr.BLUE, line, clr.ENDC))
                    line = ""
                line = "{}{}, ".format(line, k)
            if len(line) > 0:
                print("    {}{}{}".format(clr.BLUE, line.rstrip(", "),
                      clr.ENDC))
        else:
            # Too many to print!
            print("    {}(Too many to print){}".format(clr.BLUE, clr.ENDC))

    return

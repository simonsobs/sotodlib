# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Hardware visualization tools.
"""

import numpy as np
import quaternionarray as qa
import warnings


default_band_colors = {
    "LAT_f030": (0.4, 0.4, 1.0, 0.2),
    "LAT_f040": (0.4, 0.4, 1.0, 0.2),
    "LAT_f090": (0.4, 1.0, 0.4, 0.2),
    "LAT_f150": (0.4, 1.0, 0.4, 0.2),
    "LAT_f230": (1.0, 0.4, 0.4, 0.2),
    "LAT_f290": (1.0, 0.4, 0.4, 0.2),
    "SAT_f030": (0.4, 0.4, 1.0, 0.2),
    "SAT_f040": (0.4, 0.4, 1.0, 0.2),
    "SAT_f090": (0.4, 1.0, 0.4, 0.2),
    "SAT_f150": (0.4, 1.0, 0.4, 0.2),
    "SAT_f230": (1.0, 0.4, 0.4, 0.2),
    "SAT_f290": (1.0, 0.4, 0.4, 0.2),
}


def set_matplotlib_pdf_backend():
    """Set the matplotlib backend to PDF

    This is necessary to render properly the focal plane plots"""

    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    return plt


def plot_detectors(
    dets, outfile, width=None, height=None, labels=False, bandcolor=None
):
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
    try:
        plt = set_matplotlib_pdf_backend()
    except:
        warnings.warn(
            """Couldn't set the PDF matplotlib backend,
focal plane plots will not render properly,
proceeding with the default matplotlib backend"""
        )
        import matplotlib.pyplot as plt

    # If you rotate the zaxis by quat, then the X axis is upwards in
    # the focal plane and the Y axis is to the right.  (This is what
    # TOAST expects.)  Below we use "x" and "y" for those directions,
    # and in plotting functions pass them in the order (y, x).

    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    quats = np.array([p['quat'] for p in dets.values()]).astype(float)
    pos = qa.rotate(quats, zaxis)  # shape (n_det, 3)

    # arc_factor returns a scaling that can be used to reproject (X,
    # Y) to units corresponding to the angle subtended from (0,0,1) to
    # (X, Y, sqrt(X^2 + Y^2)).  This is called "ARC" (Zenithal
    # Equidistant) projection in FITS.
    def arc_factor(x, y):
        r = (x**2 + y**2)**.5
        if r < 1e-6:
            return 1. + r**2/6
        return np.arcsin(r)/r
    detx = {k: p[0] * 180.0 / np.pi * arc_factor(p[0], p[1])
            for k, p in zip(dets.keys(), pos)}
    dety = {k: p[1] * 180.0 / np.pi * arc_factor(p[0], p[1])
            for k, p in zip(dets.keys(), pos)}

    # The detang is the polarization angle, measured CCW from vertical.
    pol = qa.rotate(quats, xaxis)  # shape (n_det, 3)
    detang = {k: np.arctan2(p[1], p[0])
              for k, p in zip(dets.keys(), pol)}

    wmin = 1.0
    wmax = -1.0
    hmin = 1.0
    hmax = -1.0
    if (width is None) or (height is None):
        # We are autoscaling.  Compute the angular extent of all detectors
        # and add some buffer.
        if len(detx):
            _y = np.array(list(dety.values()))
            _x = np.array(list(detx.values()))
            wmin, wmax = _y.min(), _y.max()
            hmin, hmax = _x.min(), _x.max()
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

    wafer_centers = dict()
    if labels:
        # We are plotting labels and will want to plot a wafer_slot label for each
        # wafer.  To decide where to place the label, we find the average location
        # of all detectors from each wafer and put the label there.
        for d, props in dets.items():
            dwslot = props["wafer_slot"]
            if dwslot not in wafer_centers:
                wafer_centers[dwslot] = []
            wafer_centers[dwslot].append((detx[d], dety[d]))
        for k in wafer_centers.keys():
            center = np.mean(wafer_centers[k], axis=0)
            size = (np.array(wafer_centers[k]) - center).std()
            wafer_centers[k] = (center, size)

    if bandcolor is None:
        bandcolor = default_band_colors
    wfigsize = 10.0
    hfigsize = wfigsize * (height / width)
    figdpi = 75
    hfigpix = int(figdpi * hfigsize)
    hpixperdeg = hfigpix / height

    fig = plt.figure(figsize=(wfigsize, hfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("Degrees", fontsize="large")
    ax.set_ylabel("Degrees", fontsize="large")
    ax.set_xlim([wmin, wmax])
    ax.set_ylim([hmin, hmax])

    # Draw wafer labels in the background
    if labels:
        # The font size depends on the wafer size ... but keep it
        # between (0.01 and 0.1) times the size of the figure.
        for k, (center, size) in wafer_centers.items():
            fontpix = np.clip(0.7 * size * hpixperdeg, 0.01 * hfigpix, 0.10 * hfigpix)
            ax.text(center[1] + fontpix/hpixperdeg, center[0], k,
                    color='k', fontsize=fontpix, horizontalalignment='center',
                    verticalalignment='center', zorder=100,
                    bbox=dict(fc='white', ec='none', pad=0.2, alpha=1.0))

    for d, props in dets.items():
        band = props["band"]
        pixel = props["pixel"]
        pol = props["pol"]
        quat = np.array(props["quat"]).astype(np.float64)
        fwhm = props["fwhm"]

        # radius in degrees
        detradius = 0.5 * fwhm / 60.0

        # Position and polarization angle
        xpos, ypos = detx[d], dety[d]
        polang = detang[d]

        detface = bandcolor[band]

        circ = plt.Circle((ypos, xpos), radius=detradius, fc=detface,
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

        ax.arrow(ytail, xtail, dy, dx, width=0.1*detradius,
                 head_width=0.3*detradius, head_length=0.3*detradius,
                 fc=detcolor, ec="none", length_includes_head=True)

        if labels:
            # Compute the font size to use for detector labels
            fontpix = 0.1 * detradius * hpixperdeg
            ax.text(ypos, xpos, pixel,
                    color='k', fontsize=fontpix, horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(fc='white', ec='none', pad=0.2, alpha=1.0))
            labeloff = fontpix * len(pol) / hpixperdeg
            if dy < 0:
                labeloff = -labeloff
            ax.text((ytail+1.0*dy+labeloff), (xtail+1.0*dx), pol,
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

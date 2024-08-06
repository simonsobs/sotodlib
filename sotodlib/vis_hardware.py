# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Hardware visualization tools.
"""

import astropy.units as u
import numpy as np
import quaternionarray as qa
import warnings

import toast

from so3g.proj import quat

from .core.hardware import LAT_COROTATOR_OFFSET

from .sim_hardware import sim_nominal


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
    dets,
    outfile,
    width=None,
    height=None,
    labels=False,
    bandcolor=None,
    xieta=False,
    lat_corotate=True,
    lat_elevation=None,
    show_centers=False,
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
        xieta (bool):  If True, plot in Xi / Eta / Gamma coordinates rather
            than focalplane X / Y / Z.
        show_centers (bool):  If True, label pixel centers.

    Returns:
        None

    """
    try:
        plt = set_matplotlib_pdf_backend()
    except:
        wmsg = (
            "Couldn't set the PDF matplotlib backend, "
            "focal plane plots will not render properly, "
            "proceeding with the default matplotlib backend"
        )
        warnings.warn(wmsg)
        import matplotlib.pyplot as plt

    xaxis, yaxis, zaxis = np.eye(3)

    # Get wafer to telescope map
    hw = sim_nominal()
    wfmap = hw.wafer_map()

    n_det = len(dets)
    detnames = list(dets.keys())
    quats = np.array(
        [dets[detnames[x]]["quat"] for x in range(n_det)], dtype=np.float64
    )

    # Skip bolometers that do not have a proper quaternion
    good = np.isfinite(quats[:, 0])
    bad = np.logical_not(good)
    nbad = np.sum(bad)
    if nbad != 0:
        print(f"Skipping {nbad} detectors without quaternions")
        for detname, is_bad in zip(detnames, bad):
            if is_bad:
                del dets[detname]
        detnames = list(dets.keys())
        quats = quats[good]
        n_det = len(dets)

    lat_rot = None
    lat_elstr = ""

    have_lat = False
    for d in range(n_det):
        dn = detnames[d]
        tele = wfmap["telescopes"][dets[dn]["wafer_slot"]]
        if tele == "LAT":
            have_lat = True

    if have_lat:
        if not lat_corotate:
            if lat_elevation is None:
                raise RuntimeError("Must specify elevation if not co-rotating")
            lat_elstr = f"({lat_elevation.to_value(u.degree):0.1f} Degrees Elevation)"
            lat_ang = lat_elevation.to_value(u.rad) - LAT_COROTATOR_OFFSET.to_value(
                u.rad
            )
            lat_rot = qa.rotation(zaxis, lat_ang)
        for d in range(n_det):
            dn = detnames[d]
            tele = wfmap["telescopes"][dets[dn]["wafer_slot"]]
            if tele == "LAT" and lat_rot is not None:
                quats[d] = qa.mult(lat_rot, quats[d])

    if n_det == 0:
        raise RuntimeError("No detectors specified")

    # arc_factor returns a scaling that can be used to reproject (X,
    # Y) to units corresponding to the angle subtended from (0,0,1) to
    # (X, Y, sqrt(X^2 + Y^2)).  This is called "ARC" (Zenithal
    # Equidistant) projection in FITS.
    def arc_factor(x, y):
        r = (x**2 + y**2) ** 0.5
        if r < 1e-6:
            return 1.0 + r**2 / 6
        return np.arcsin(r) / r

    if xieta:
        # Plotting as seen from observer in Xi / Eta / Gamma
        xi, eta, gamma = toast.instrument_coords.quat_to_xieta(quats)
        # xi, eta, gamma = quat.decompose_xieta(quats)

        detx = {
            detnames[k]: xi[k] * 180.0 / np.pi * arc_factor(xi[k], eta[k])
            for k in range(n_det)
        }
        dety = {
            detnames[k]: eta[k] * 180.0 / np.pi * arc_factor(xi[k], eta[k])
            for k in range(n_det)
        }

        for didx, dn in enumerate(detnames):
            wf = dets[dn]["wafer_slot"]
            px = int(dets[dn]["pixel"])
            if px == 0:
                q = quats[didx]
                dd = qa.rotate(q, zaxis)

        # In Xi / Eta coordinates, gamma is measured clockwise from line of
        # decreasing elevation.  Here we convert into visualization X/Y
        # coordinatates measured counter clockwise from the X axis.
        polangs = {detnames[k]: 1.5 * np.pi - gamma[k] for k in range(n_det)}
    else:
        # Plotting in focalplane X / Y / Z coordinates.
        # Compute direction and orientation vectors
        dir = qa.rotate(quats, zaxis)
        orient = qa.rotate(quats, xaxis)

        small = np.fabs(1.0 - dir[:, 2]) < 1.0e-12
        not_small = np.logical_not(small)
        xp = np.zeros(n_det, dtype=np.float64)
        yp = np.zeros(n_det, dtype=np.float64)

        mag = np.arccos(dir[not_small, 2])
        ang = np.arctan2(dir[not_small, 1], dir[not_small, 0])
        xp[not_small] = mag * np.cos(ang)
        yp[not_small] = mag * np.sin(ang)

        polangs = {
            detnames[k]: np.arctan2(orient[k, 1], orient[k, 0]) for k in range(n_det)
        }

        detx = {
            detnames[k]: xp[k] * 180.0 / np.pi * arc_factor(xp[k], yp[k])
            for k in range(n_det)
        }
        dety = {
            detnames[k]: yp[k] * 180.0 / np.pi * arc_factor(xp[k], yp[k])
            for k in range(n_det)
        }

    if (width is None) or (height is None):
        # We are autoscaling.  Compute the angular extent of all detectors
        # and add some buffer.
        _y = np.array(list(dety.values()))
        _x = np.array(list(detx.values()))
        wmin, wmax = _x.min(), _x.max()
        hmin, hmax = _y.min(), _y.max()
        wbuf = 0.2 * (wmax - wmin)
        hbuf = 0.2 * (hmax - hmin)
        wmin -= wbuf
        wmax += wbuf
        hmin -= hbuf
        hmax += hbuf
        width = wmax - wmin
        height = hmax - hmin
        half_width = 0.5 * width
        half_height = 0.5 * height
    else:
        half_width = 0.5 * width
        half_height = 0.5 * height
        wmin = -half_width
        wmax = half_width
        hmin = -half_height
        hmax = half_height

    wafer_centers = dict()
    if labels:
        # We are plotting labels and will want to plot a label for each
        # wafer.  To decide where to place the label, we find the average location
        # of all detectors from each wafer and put the label there.
        temp_centers = dict()
        for d, props in dets.items():
            dwslot = props["wafer_slot"]
            if dwslot not in temp_centers:
                temp_centers[dwslot] = {"x": list(), "y": list()}
            temp_centers[dwslot]["x"].append(detx[d])
            temp_centers[dwslot]["y"].append(dety[d])
        for k in list(temp_centers.keys()):
            xcenter = np.mean(temp_centers[k]["x"])
            ycenter = np.mean(temp_centers[k]["y"])
            ysize = 0.25 * np.std(np.array(temp_centers[k]["y"]) - ycenter)
            tube_slot = wfmap["tube_slots"][k]
            tube_props = hw.data["tube_slots"][tube_slot]
            tube_wafer_indx = hw.data["wafer_slots"][k]["tube_index"]
            ufm_slot = tube_props["wafer_ufm_slot"][tube_wafer_indx]
            if "wafer_ufm_loc" in tube_props:
                ufm_loc = tube_props["wafer_ufm_loc"][tube_wafer_indx]
                ltext = f"{k}:ws{ufm_slot}:{ufm_loc}"
            else:
                ltext = f"{k}:ws{ufm_slot}"
            wafer_centers[k] = ((xcenter, ycenter), ysize, ltext)

    if bandcolor is None:
        bandcolor = default_band_colors
    wfigsize = 10.0
    hfigsize = wfigsize * (height / width)
    figdpi = 75
    hfigpix = int(figdpi * hfigsize)
    hpixperdeg = hfigpix / height

    fig = plt.figure(figsize=(wfigsize, hfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim([wmin, wmax])
    ax.set_ylim([hmin, hmax])
    if xieta:
        ax.set_xlabel(r"Boresight $\xi$ Degrees", fontsize="large")
        ax.set_ylabel(r"Boresight $\eta$ Degrees", fontsize="large")
    else:
        ax.set_xlabel("Boresight X Degrees", fontsize="large")
        ax.set_ylabel("Boresight Y Degrees", fontsize="large")

    # Draw wafer labels in the background
    if labels:
        # The font size depends on the wafer size ... but keep it
        # between (0.01 and 0.1) times the size of the figure.
        for k, (center, size, ltext) in wafer_centers.items():
            fontpix = np.clip(0.7 * size * hpixperdeg, 0.01 * hfigpix, 0.10 * hfigpix)
            if fontpix < 1.0:
                fontpix = 1.0
            ax.text(
                center[0],
                center[1] + fontpix / hpixperdeg,
                ltext,
                color="k",
                fontsize=fontpix,
                horizontalalignment="center",
                verticalalignment="center",
                zorder=100,
                bbox=dict(fc="white", ec="none", pad=0.2, alpha=1.0),
            )

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
        polang = polangs[d]

        detface = bandcolor[band]

        circ = plt.Circle(
            (xpos, ypos),
            radius=detradius,
            fc=detface,
            ec="black",
            linewidth=0.05 * detradius,
        )
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

        ax.arrow(
            xtail,
            ytail,
            dx,
            dy,
            width=0.1 * detradius,
            head_width=0.3 * detradius,
            head_length=0.3 * detradius,
            fc=detcolor,
            ec="none",
            length_includes_head=True,
        )

        # Compute the font size to use for detector labels
        fontpix = 0.1 * detradius * hpixperdeg
        if fontpix < 1.0:
            fontpix = 1.0

        if show_centers:
            ysgn = -1.0
            if dx < 0.0:
                ysgn = 1.0
            ax.text(
                (xpos + 0.1 * dx),
                (ypos + 0.1 * ysgn * dy),
                f"({xpos:0.4f}, {ypos:0.4f})",
                color="green",
                fontsize=fontpix,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

        if labels:
            ax.text(
                xpos,
                ypos,
                pixel,
                color="k",
                fontsize=fontpix,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="white", ec="none", pad=0.2, alpha=1.0),
            )
            labeloff = fontpix * len(pol) / hpixperdeg
            if dy < 0:
                labeloff = -labeloff
            ax.text(
                (xtail + 1.0 * dx + labeloff),
                (ytail + 1.0 * dy),
                pol,
                color="k",
                fontsize=fontpix,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="none", ec="none", pad=0, alpha=1.0),
            )

    # Draw a "mini" coordinate axes for reference
    shortest = min(half_width, half_height)
    xmini = -0.7 * half_width
    ymini = -0.7 * half_height
    xlen = 0.06 * shortest
    ylen = 0.06 * shortest
    mini_width = 0.005 * shortest
    mini_head_width = 3 * mini_width
    mini_head_len = 3 * mini_width
    if xieta:
        aprops = [
            (xlen, 0, "-", r"$\xi$"),
            (0, ylen, "-", r"$\eta$"),
            (-xlen, 0, "--", "Y"),
            (0, -ylen, "--", "X"),
        ]
    else:
        aprops = [
            (xlen, 0, "-", "X"),
            (0, ylen, "-", "Y"),
            (-xlen, 0, "--", r"$\eta$"),
            (0, -ylen, "--", r"$\xi$"),
        ]
    for ap in aprops:
        lx = xmini + 1.5 * ap[0]
        ly = ymini + 1.5 * ap[1]
        lw = figdpi / 200.0
        ax.arrow(
            xmini,
            ymini,
            ap[0],
            ap[1],
            width=mini_width,
            head_width=mini_head_width,
            head_length=mini_head_len,
            fc="k",
            ec="k",
            linestyle=ap[2],
            linewidth=lw,
            length_includes_head=True,
        )
        ax.text(
            lx,
            ly,
            ap[3],
            color="k",
            fontsize=int(figdpi / 10),
            horizontalalignment="center",
            verticalalignment="center",
        )

    st = f"Focalplane Looking Towards Observer {lat_elstr}"
    if xieta:
        st = f"Focalplane on Sky From Observer {lat_elstr}"
    fig.suptitle(st)

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
        print(
            "{}{:<12}: {}{:5d} objects{}".format(
                clr.WHITE, obj, clr.RED, nsub, clr.ENDC
            )
        )
        if nsub <= 2000:
            line = ""
            for k in list(props.keys()):
                if (len(line) + len(k)) > 72:
                    print("    {}{}{}".format(clr.BLUE, line, clr.ENDC))
                    line = ""
                line = "{}{}, ".format(line, k)
            if len(line) > 0:
                print("    {}{}{}".format(clr.BLUE, line.rstrip(", "), clr.ENDC))
        else:
            # Too many to print!
            print("    {}(Too many to print){}".format(clr.BLUE, clr.ENDC))

    return

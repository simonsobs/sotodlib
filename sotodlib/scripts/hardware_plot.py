# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Plot a hardware model.
"""

import argparse

import astropy.units as u

from ..core import Hardware

from ..vis_hardware import plot_detectors, summary_text


def main():
    parser = argparse.ArgumentParser(
        description="This program reads a hardware model and plots the\
            detectors.  Note that you should pre-select detectors before\
            passing a hardware model to this function.  See so_hardware_trim.",
        usage="so_hardware_plot [options] (use --help for details)")

    parser.add_argument(
        "--hardware", required=True, default=None,
        help="Input hardware file"
    )

    parser.add_argument(
        "--out", required=False, default=None,
        help="Name of the output PDF file."
    )

    parser.add_argument(
        "--width", required=False, default=None,
        help="The width of the plot in degrees."
    )

    parser.add_argument(
        "--height", required=False, default=None,
        help="The height of the plot in degrees."
    )

    parser.add_argument(
        "--labels", required=False, default=False, action="store_true",
        help="Add pixel and polarization labels to the plot."
    )

    parser.add_argument(
        "--show_centers", required=False, default=False, action="store_true",
        help="Add labels with pixel center coordinates."
    )

    parser.add_argument(
        "--xieta", required=False, default=False, action="store_true",
        help="Plot in Xi / Eta coordinates."
    )

    parser.add_argument(
        "--lat_corotate",
        required=False,
        default=False,
        action="store_true",
        help="Rotate LAT receiver to maintain focalplane orientation",
    )

    parser.add_argument(
        "--lat_elevation_deg",
        required=False,
        default=60.0,
        type=float,
        help="Observing elevation of the LAT if not co-rotating",
    )

    args = parser.parse_args()

    outfile = args.out
    if outfile is None:
        fields = args.hardware.split(".")
        outfile = fields[0]

    print("Loading hardware file {}...".format(args.hardware), flush=True)
    hw = Hardware(args.hardware)
    # summary_text(hw)

    width = args.width
    if width is not None:
        width = float(width)
    height = args.height
    if height is not None:
        height = float(height)

    print("Generating detector plot...", flush=True)
    plot_detectors(
        hw.data["detectors"],
        outfile,
        width=width,
        height=height,
        labels=args.labels,
        xieta=args.xieta,
        lat_corotate=args.lat_corotate,
        lat_elevation=args.lat_elevation_deg * u.degree,
        show_centers=args.show_centers
    )

    return

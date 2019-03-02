# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Plot a hardware model.
"""

import argparse

from ..hardware import Hardware, plot_detectors, summary_text


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

    args = parser.parse_args()

    outfile = args.out
    if outfile is None:
        fields = args.hardware.split(".")
        outfile = fields[0]

    print("Loading hardware file {}...".format(args.hardware), flush=True)
    hw = Hardware(args.hardware)
    # summary_text(hw)

    print("Generating detector plot...", flush=True)
    plot_detectors(hw.data["detectors"], outfile, width=args.width,
                   height=args.height, labels=args.labels)

    return

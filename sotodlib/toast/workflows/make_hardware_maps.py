#!/usr/bin/env python

# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
from numpy.random import randn

from ... import sim_hardware as hardware


def write_hw(out, hw, plain=False, overwrite=True):
    if plain:
        outpath = "{}.toml".format(out)
    else:
        outpath = "{}.toml.gz".format(out)
    print("Dumping config to {}...".format(outpath))
    hw.dump(outpath, overwrite=overwrite, compress=(not plain))
    return


def main():
    parser = argparse.ArgumentParser(
        description="This program simulates the current nominal hardware "
        "model, perturbs it and dumps it to disk",
        usage="make_hardware_maps [options] (use --help for details)")

    parser.add_argument(
        "--out", required=False, default="hardware",
        help="Name (without extensions) of the output hardware file")
    parser.add_argument(
        "--plain", required=False, default=False, action="store_true",
        help="Write plain text (without gzip compression)")

    parser.add_argument(
        "--sigma_band_width", default=0.01, type=np.float,
        help="Proportional width of top hat bandpass width distribution.")
    parser.add_argument(
        "--sigma_band_center", default=0.01, type=np.float,
        help="Proportional width of top hat bandpass center distribution.")
    parser.add_argument(
        "--sigma_noise_net", default=0.01, type=np.float,
        help="Proportional width of noise NET distribution.")
    parser.add_argument(
        "--sigma_noise_fknee", default=0.01, type=np.float,
        help="Proportional width of noise fknee distribution.")
    parser.add_argument(
        "--sigma_noise_fmin", default=0.01, type=np.float,
        help="Proportional width of noise fmin distribution.")
    parser.add_argument(
        "--sigma_noise_alpha", default=0.01, type=np.float,
        help="Proportional width of noise alpha distribution.")
    parser.add_argument(
        "--sigma_beam_fwhm", default=0.01, type=np.float,
        help="Proportional width of beam FWHM distribution.")

    args = parser.parse_args()

    # Nominal configuration

    print("Getting nominal config...", flush=True)
    hw = hardware.sim_nominal()
    for tele, teleprops in hw.data["telescopes"].items():
        print("Simulating detectors for telescope {}...".format(tele),
              flush=True)
        hardware.sim_detectors_toast(hw, tele)
    write_hw(args.out + '.nominal', hw, plain=args.plain)

    # Perturb configuration

    print("Perturbing nominal config...", flush=True)
    for detname, detdata in hw.data["detectors"].items():
        bandname = detdata["band"]
        banddata = hw.data["bands"][bandname]
        # Perturb bandpass
        low = banddata["low"]
        center = banddata["center"]
        high = banddata["high"]
        halfband = .5 * (high - low) * (1 + randn() * args.sigma_band_width)
        center *= 1 + randn() * args.sigma_band_center
        low = center - halfband
        high = center + halfband
        detdata["low"] = low
        detdata["center"] = center
        detdata["high"] = high
        # Perturb noise
        detdata["NET"] = banddata["NET"] * (1 + randn() * args.sigma_noise_net)
        detdata["fknee"] = banddata["fknee"] * (1 + randn() * args.sigma_noise_fknee)
        detdata["fmin"] = banddata["fmin"] * (1 + randn() * args.sigma_noise_fmin)
        detdata["alpha"] = banddata["alpha"] * (1 + randn() * args.sigma_noise_alpha)
        # Perturb beam
        detdata["fwhm"] *= (1 + randn() * args.sigma_beam_fwhm)
        # Perturb position?  This would be the place to add errors in focal plane geometry
    write_hw(args.out + '.perturbed', hw, plain=args.plain)

    return


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright (c) 2026-2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""
Helper script for benchmarking G3 I/O.
"""

import argparse
import os
import time

import numpy as np
import so3g


try:
    from so3g.spt3g import core as g3c
    has_super = True
except (AttributeError, ImportError):
    from spt3g import core as g3c
    has_super = False

from sotodlib.io import load_book



def main():
    parser = argparse.ArgumentParser(description="Benchmark loading of G3 framefiles")
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="The output framefile to write",
    )
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="The input framefile to load",
    )
    parser.add_argument(
        "--n_dets",
        required=False,
        type=int,
        default=1760,
        help="The number of detectors to simulate",
    )
    parser.add_argument(
        "--n_samples",
        required=False,
        type=int,
        default=720000,
        help="The number of samples to simulate",
    )
    parser.add_argument(
        "--frame_samples",
        required=False,
        type=int,
        default=200 * 30,
        help="The number of samples per frame (sample rate * scan time)",
    )

    parser.add_argument(
        "--disable_obsinfo",
        required=False,
        action="store_true",
        default=False,
        help="Do not use knowledge of the observation size when loading",
    )

    args = parser.parse_args()

    if args.input is None and args.output is None:
        raise RuntimeError("Specify one of --input or --output")

    if args.input is not None and args.output is not None:
        raise RuntimeError("Specify only one of --input or --output")

    if has_super:
        container = "G3SuperTimestream"
    else:
        container = "G3TimestreamMap"

    if args.output is not None:
        # We are simulating and writing the file
        if os.path.isfile(args.output):
            os.remove(args.output)
        frameg = load_book._sim_g3_generator(
            args.n_dets,
            args.n_samples,
            frame_size=args.frame_samples,
        )
        start = time.time()
        w = g3c.G3Writer(args.output)
        for f in frameg:
            w.Process(f)
        del w
        stop = time.time()
        elapsed = stop - start
        st = os.stat(args.output)
        sz = st.st_size
        sz_mb = sz // 1e6
        print(f"Wrote {args.output} in {elapsed:8.3f} seconds", flush=True)
        print(f"  Frame file is {sz_mb} MB, using {container}", flush=True)

    if args.input is not None:
        samples = None
        if not args.disable_obsinfo:
            samples = (0, args.n_samples)
        st = os.stat(args.input)
        sz = st.st_size
        sz_mb = sz // 1e6
        start = time.time()
        aman = load_book.load_book_file(args.input, samples=samples)
        stop = time.time()
        elapsed = stop - start
        print(aman, flush=True)
        print(aman["signal"], aman["signal"].dtype, flush=True)
        print(f"Read {args.input} in {elapsed:8.3f} seconds", flush=True)
        print(f"  Frame file is {sz_mb} MB, using {container}", flush=True)


if __name__ == "__main__":
    main()


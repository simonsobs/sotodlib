# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.


import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger

from ...modules import OpDemod


def add_demodulation_args(parser):
    parser.add_argument(
        "--demodulate",
        required=False,
        action="store_true",
        help="Demodulate and downsample HWP-modulated signal",
        dest="demodulate"
    )
    parser.add_argument(
        "--no-demodulate",
        required=False,
        action="store_false",
        help="Do not demodulate HWP-modulated signal",
        dest="demodulate"
    )
    parser.set_defaults(demodulate=False)
    parser.add_argument(
        "--demod-wkernel",
        required=False,
        type=np.int,
        help="Width of demodulation kernel",
    )
    parser.add_argument(
        "--demod-fmax",
        required=False,
        type=np.float,
        help="Low-pass filter cut-off",
    )
    parser.add_argument(
        "--demod-nskip",
        type=np.int,
        default=3,
        help="Number of samples to skip in downsampling",
    )
    return


def demodulate(args, comm, data, name, detweights=None, madampars=None, verbose=True):
    if not args.demodulate:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()

    if detweights is not None:
        # Copy the detector weights to demodulated TOD
        modulated = [detname for detname in detweights if "demod" not in detname]
        for detname in modulated:
            detweight = detweights[detname]
            for demodkey in ["demod0", "demod4r", "demod4i"]:
                demod_name = "{}_{}".format(demodkey, detname)
                detweights[demod_name] = detweight
            del detweights[detname]

    if madampars is not None:
        # Filtering will affect the high frequency end of the noise PSD
        madampars["radiometers"] = False
        # Intensity and polarization will be decoupled in the noise matrix
        madampars["allow_decoupling"] = True

    demod = OpDemod(
        name=name, wkernel=args.demod_wkernel, fmax=args.demod_fmax, nskip=args.demod_nskip,
    )
    demod.exec(data)

    timer.report_clear("Demodulate")

    return

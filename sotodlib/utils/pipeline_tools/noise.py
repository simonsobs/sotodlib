# Copyright (c) 2019-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.timing import function_timer, Timer
from toast.tod import AnalyticNoise
from toast.utils import Logger
import toast.qarray as qa

from ...sim_hardware import get_example


def add_so_noise_args(parser):
    parser.add_argument(
        "--common-mode-noise",
        required=False,
        help="String defining analytical parameters of a per-tube "
        "common mode that is co-added with every detector: "
        "'fmin[Hz],fknee[Hz],alpha,NET[K]' OR "
        "'fmin[Hz],fknee[Hz],alpha,NET[K],center,width' where the last two "
        "define the coupling strenth distribution and default to (1, 0)"
        "Multiple common modes can be separated with ';'.",
    )
    parser.add_argument(
        "--common-mode-only",
        required=False,
        action="store_true",
        help="No individual detector noise",
        dest="common_mode_only"
    )
    parser.add_argument(
        "--no-common-mode-only",
        required=False,
        action="store_false",
        help="Include individual detector noise",
        dest="common_mode_only"
    )
    parser.set_defaults(common_mode_only=False)
    return


@function_timer
def get_elevation_noise(args, comm, data, key="noise"):
    """ Insert elevation-dependent noise

    """
    timer = Timer()
    timer.start()
    # fsample = args.sample_rate
    for obs in data.obs:
        tod = obs["tod"]
        fp = obs["focalplane"]
        noise = obs[key]
        for det in tod.local_dets:
            if det not in noise.keys:
                raise RuntimeError(
                    'Detector "{}" does not have a PSD in the noise object'.format(det)
                )
            A = fp[det]["A"]
            C = fp[det]["C"]
            psd = noise.psd(det)
            try:
                # Some TOD classes provide a shortcut to Az/El
                _, el = tod.read_azel(detector=det)
            except Exception:
                azelquat = tod.read_pntg(detector=det, azel=True)
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, _ = qa.to_position(azelquat)
                el = np.pi / 2 - theta
            el = np.median(el)
            # Scale the analytical noise PSD. Pivot is at el = 50 deg.
            psd[:] *= (A / np.sin(el) + C) ** 2
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Elevation noise")
    return


@function_timer
def get_analytic_noise(args, comm, focalplane, verbose=True):
    """ Create a TOAST noise object.

    Create a noise object from the 1/f noise parameters contained in the
    focalplane database.  Optionally add thermal common modes.

    """
    timer = Timer()
    timer.start()

    detectors = sorted(focalplane.detector_data.keys())
    fmins = {}
    fknees = {}
    alphas = {}
    NETs = {}
    rates = {}
    indices = {}
    for d in detectors:
        rates[d] = args.sample_rate
        fmins[d] = focalplane[d]["fmin"]
        fknees[d] = focalplane[d]["fknee"]
        alphas[d] = focalplane[d]["alpha"]
        NETs[d] = focalplane[d]["NET"]
        indices[d] = focalplane[d]["index"]

    ncommon = 0
    coupling_strength_distributions = []
    common_modes = []
    if args.common_mode_noise:
        # Add an extra "virtual" detector for common mode noise for
        # every optics tube
        for common_mode in args.common_mode_noise.split(";"):
            ncommon += 1
            try:
                fmin, fknee, alpha, net, center, width = np.array(
                    common_mode.split(",")
                ).astype(np.float64)
            except ValueError:
                fmin, fknee, alpha, net = np.array(common_mode.split(",")).astype(
                    np.float64
                )
                center, width = 1, 0
            coupling_strength_distributions.append([center, width])
            hw = get_example()
            for itube, tube in enumerate(sorted(hw.data["tubes"].keys())):
                d = "common_mode_{}_{}".format(ncommon - 1, tube)
                detectors.append(d)
                common_modes.append(d)
                rates[d] = args.sample_rate
                fmins[d] = fmin
                fknees[d] = fknee
                alphas[d] = alpha
                NETs[d] = net
                indices[d] = ncommon * 100000 + itube

    noise = AnalyticNoise(
        rate=rates,
        fmin=fmins,
        detectors=detectors,
        fknee=fknees,
        alpha=alphas,
        NET=NETs,
        indices=indices,
    )

    if args.common_mode_noise:
        mixmatrix = {}
        keys = set()
        if args.common_mode_only:
            detweight = 0
        else:
            detweight = 1
        for icommon in range(ncommon):
            # Update the mixing matrix in the noise operator
            center, width = coupling_strength_distributions[icommon]
            np.random.seed(1001 + icommon)
            couplings = center + np.random.randn(1000000) * width
            for det in focalplane.detector_data.keys():
                if det not in mixmatrix:
                    mixmatrix[det] = {det: detweight}
                    keys.add(det)
                tube = focalplane[det]["tube"]
                common = "common_mode_{}_{}".format(icommon, tube)
                index = focalplane[det]["index"]
                mixmatrix[det][common] = couplings[index]
                keys.add(common)
        # Add a diagonal entries, even if we wouldn't usually ask for
        # the common mode alone.
        for common in common_modes:
            mixmatrix[common] = {common : 1}
        # There should probably be an accessor method to update the
        # mixmatrix in the TOAST Noise object.
        if noise._mixmatrix is not None:
            raise RuntimeError("Did not expect non-empty mixing matrix")
        noise._mixmatrix = mixmatrix
        noise._keys = list(sorted(keys))

    focalplane._noise = noise

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Creating noise model")
    return noise

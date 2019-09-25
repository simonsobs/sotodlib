# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.timing import function_timer, Timer
from toast.tod import AnalyticNoise
from toast.utils import Logger
import toast.qarray as qa
from .. import hardware

def add_so_noise_args(parser):
    parser.add_argument(
        "--common-mode-noise",
        required=False,
        help="String defining analytical parameters of a per-tube "
        "common mode that is co-added with every detector: "
        "'fmin[Hz],fknee[Hz],alpha,NET[K]'",
    )
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
    focalplane database.

    """
    timer = Timer()
    timer.start()
    detectors = sorted(focalplane.keys())
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

    if args.common_mode_noise:
        # Add an extra "virtual" detector for common mode noise for
        # every optics tube
        fmin, fknee, alpha, net = np.array(args.common_mode_noise.split(",")).astype(
            np.float64
        )
        hw = hardware.get_example()
        for itube, tube in enumerate(sorted(hw.data["tubes"].keys())):
            d = "common_mode_{}".format(tube)
            detectors.append(d)
            rates[d] = args.sample_rate
            fmins[d] = fmin
            fknees[d] = fknee
            alphas[d] = alpha
            NETs[d] = net
            indices[d] = 100000 + itube

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
        # Update the mixing matrix in the noise operator
        mixmatrix = {}
        keys = set()
        for det in focalplane.keys():
            tube = focalplane[det]["tube"]
            common = "common_mode_{}".format(tube)
            mixmatrix[det] = {det: 1, common: 1}
            keys.add(det)
            keys.add(common)
        # There should probably be an accessor method to update the
        # mixmatrix in the TOAST Noise object.
        if noise._mixmatrix is not None:
            raise RuntimeError("Did not expect non-empty mixing matrix")
        noise._mixmatrix = mixmatrix
        noise._keys = list(sorted(keys))

    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Creating noise model")
    return noise

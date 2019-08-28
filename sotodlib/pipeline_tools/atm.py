# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.timing import function_timer, Timer
from toast.tod import atm_available_utils
from toast.utils import Logger

if atm_available_utils:
    from toast.tod.atm import (
        atm_atmospheric_loading,
        atm_absorption_coefficient,
        atm_absorption_coefficient_vec,
    )


@function_timer
def scale_atmosphere_by_bandpass(args, comm, data, totalname, mc, verbose=False):
    """ Scale atmospheric fluctuations by bandpass.
    Assume that cached signal under totalname is pure atmosphere
    and scale the absorption coefficient according to the bandpass.
    If the focalplane is included in the observation and defines
    bandpasses for the detectors, the scaling is computed for each
    detector separately.
    """
    if not args.simulate_atmosphere:
        return

    timer = Timer()
    log = Logger.get()

    if comm.world_rank == 0 and verbose:
        log.info("Scaling atmosphere by bandpass")

    timer.start()
    for obs in data.obs:
        tod = obs["tod"]
        todcomm = tod.mpicomm
        site_id = obs["site_id"]
        weather = obs["weather"]
        if "focalplane" in obs:
            focalplane = obs["focalplane"]
        else:
            focalplane = None
        start_time = obs["start_time"]
        weather.set(site_id, mc, start_time)
        altitude = obs["altitude"]
        air_temperature = weather.air_temperature
        surface_pressure = weather.surface_pressure
        pwv = weather.pwv
        # Use the entire processing group to sample the absorption
        # coefficient as a function of frequency
        freqmin = 0
        freqmax = 1000
        nfreq = 10001
        freqstep = (freqmax - freqmin) / (nfreq - 1)
        if todcomm is None:
            nfreq_task = nfreq
            my_ifreq_min = 0
            my_ifreq_max = nfreq
        else:
            nfreq_task = int(nfreq // todcomm.size) + 1
            my_ifreq_min = nfreq_task * todcomm.rank
            my_ifreq_max = min(nfreq, nfreq_task * (todcomm.rank + 1))
        my_nfreq = my_ifreq_max - my_ifreq_min
        if my_nfreq > 0:
            if atm_available_utils:
                my_freqs = freqmin + np.arange(my_ifreq_min, my_ifreq_max) * freqstep
                my_absorption = atm_absorption_coefficient_vec(
                    altitude,
                    air_temperature,
                    surface_pressure,
                    pwv,
                    my_freqs[0],
                    my_freqs[-1],
                    my_nfreq,
                )
            else:
                raise RuntimeError(
                    "Atmosphere utilities from libaatm are not available"
                )
        else:
            my_freqs = np.array([])
            my_absorption = np.array([])
        if todcomm is None:
            freqs = my_freqs
            absorption = my_absorption
        else:
            freqs = np.hstack(todcomm.allgather(my_freqs))
            absorption = np.hstack(todcomm.allgather(my_absorption))
        # loading = atm_atmospheric_loading(altitude, pwv, freq)
        for det in tod.local_dets:
            # Use detector bandpass from the focalplane
            center = focalplane[det]["bandcenter_ghz"]
            width = focalplane[det]["bandwidth_ghz"]
            nstep = 101
            # Interpolate the absorption coefficient to do a top hat
            # integral across the bandpass
            det_freqs = np.linspace(center - width / 2, center + width / 2, nstep)
            absorption_det = np.mean(np.interp(det_freqs, freqs, absorption))
            cachename = "{}_{}".format(totalname, det)
            ref = tod.cache.reference(cachename)
            ref *= absorption_det
            del ref

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Atmosphere scaling")
    return

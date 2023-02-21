# Copyright (c) 2018-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import pickle

import ephem
import h5py
import healpy as hp
import numpy as np
import traitlets
from astropy import constants
from astropy import units as u
from scipy.constants import au as AU
from scipy.interpolate import RectBivariateSpline
from toast import qarray as qa
from toast.data import Data
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer
from toast.traits import (Bool, Float, Instance, Int, Quantity, Unicode,
                          trait_docs)
from toast.utils import Environment, Logger, Timer, unit_conversion

XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimReadout(Operator):
    """Simulate various readout-related systematics"""

    API = Int(0, help="Internal interface version for this operator")

    realization = Int(0, help="The noise realization index")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask_readout = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    glitch_rate = Quantity(1.0 * u.Hz, help="Glitch occurrence rate")

    glitch_amplitude_center = Quantity(
        1 * u.K, help="Center of Gaussian distribution for glitch amplitude"
    )

    glitch_amplitude_sigma = Quantity(
        1 * u.K, help="Width of Gaussian distribution for glitch amplitude"
    )

    simulate_glitches = Bool(True, help="Enable glitch simulation")

    jump_rate = Quantity(0.001 * u.Hz, help="Jump occurrence rate")

    jump_amplitude_center = Quantity(
        0 * u.mK, help="Center of Gaussian distribution for jump amplitude"
    )

    jump_amplitude_sigma = Quantity(
        1 * u.mK, help="Width of Gaussian distribution for jump amplitude"
    )

    simulate_jumps = Bool(True, help="Enable jump simulation")

    misidentify_bolometers = Bool(True, help="Enable bolometer misidentification")

    misidentification_width = Quantity(
        1.0 * u.deg,
        help="Probability of misidentifying two channels is a Gaussian "
        "function of their distance on the focalplane.",
    )

    yield_center = Float(
        0.8, help="Center of Gaussian distribution for detector yield"
    )

    yield_sigma = Float(
        0.1, help="Width of Gaussian distribution for detector yield"
    )

    simulate_yield = Bool(True, help="Simulate failed bolometers")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _add_glitches(self, comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank):
        """Simulate glitches"""
        if not self.simulate_glitches:
            return
        log = Logger.get()
        fsample = focalplane.sample_rate
        events_per_sample = self.glitch_rate.to_value(u.Hz) / fsample.to_value(u.Hz)
        nsample = times.size
        nglitch = 0
        for det in local_dets:
            det_id = focalplane[det]["uid"]
            sig = signal[det]
            np.random.seed(
                int(obs_id + det_id  + self.realization) % 2**32
            )
            event_counts = np.random.poisson(events_per_sample, nsample)
            event_indices = np.argwhere(event_counts).ravel()
            nindex = event_indices.size
            # Drawing amplitudes for every affected sample is not the
            # same as drawing amplitudes for every event but it is fast.
            # If we routinely have more than one event occurring during
            # the same time stamp we are in trouble
            amplitudes = self.glitch_amplitude_center \
                         + np.random.randn(nindex) * self.glitch_amplitude_sigma
            events = amplitudes.to_value(signal.units) \
                     * event_counts[event_indices]
            sig[event_indices] += events
            nglitch += np.sum(event_counts)
        nglitch = comm.allreduce(nglitch)
        log.debug_rank(f"Simulated {nglitch} glitches", comm=comm)
        return

    @function_timer
    def _add_jumps(self, comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank):
        """Simulate baseline jumps"""
        if not self.simulate_jumps:
            return
        log = Logger.get()
        fsample = focalplane.sample_rate
        events_per_sample = self.jump_rate.to_value(u.Hz) / fsample.to_value(u.Hz)
        nsample = times.size
        njump = 0
        for det in local_dets:
            det_id = focalplane[det]["uid"]
            sig = signal[det]
            np.random.seed(
                int(obs_id + det_id  + self.realization + 45835364) % 2**32
            )
            event_counts = np.random.poisson(events_per_sample, nsample)
            event_indices = np.argwhere(event_counts).ravel()
            nindex = event_indices.size
            # Drawing amplitudes for every affected sample is not the
            # same as drawing amplitudes for every event but it is fast.
            # If we routinely have more than one event occurring during
            # the same time stamp we are in trouble
            amplitudes = self.jump_amplitude_center \
                         + np.random.randn(nindex) * self.jump_amplitude_sigma
            events = amplitudes.to_value(signal.units) \
                     * event_counts[event_indices]
            for index, event in zip(event_indices, events):
                sig[index:] += event
            njump += np.sum(event_counts)
        njump = comm.allreduce(njump)
        log.debug_rank(f"Simulated {njump} jumps", comm=comm)
        return

    @function_timer
    def _fail_bolometers(self, comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank):
        """Simulate bolometers that fail to bias or otherwise are unusable"""
        if not self.simulate_yield:
            return
        log = Logger.get()
        np.random.seed(int(obs_id + self.realization + 59127574) % 2**32)
        yield_ = self.yield_center + np.random.randn() * self.yield_sigma
        yield_ = min(1, max(0, yield_))
        nfail = 0
        for det in local_dets:
            det_id = focalplane[det]["uid"]
            np.random.seed(int(obs_id + self.realization + 59127574 + det_id) % 2**32)
            x = np.random.rand()
            if x > yield_:
                # FIXME: We should choose between various failure modes here
                #
                # Fail the bolometer by replacing the signal with
                # a linear trend.  We do *not* record the failure in the
                # detector flags so the data *cannot* be reduced without
                # some sort of mitigation
                start, stop = np.random.rand(2) * 10 * u.K
                x = (times - times[0]) / (times[-1] - times[0])
                trend = start + (stop - start) * x
                signal[det] = trend
                nfail += 1
        nfail = comm.allreduce(nfail)
        ndet = len(all_dets)
        log.debug_rank(
            f"Failed {nfail} / {ndet} = {nfail/ndet:.3f} bolometers.  "
            f"Target yield was {yield_:.3f}",
            comm=comm,
        )
        return

    @function_timer
    def _misidentify_bolometers(self, comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank):
        """Swap detector data between two confused readout channels"""
        if not self.misidentify_bolometers:
            return
        log = Logger.get()
        np.random.seed(int(obs_id + self.realization) % 2**32)
        nhit = 0
        for det1 in all_dets:
            for det2 in all_dets:
                if det1 == det2:
                    continue
                matched = True
                # Misidentification requires several properties to agree
                for prop in ["band", "card_slot", "wafer_slot", "bias", "pol"]:
                    if focalplane[det1][prop] != focalplane[det2][prop]:
                        matched = False
                        break
                if not matched:
                    continue
                # Detectors are matched, see if they get misidentified
                quat1 = focalplane[det1]["quat"]
                quat2 = focalplane[det2]["quat"]
                vec1 = qa.rotate(quat1, ZAXIS)
                vec2 = qa.rotate(quat2, ZAXIS)
                dist = np.arccos(np.dot(vec1, vec2))
                x = np.random.randn() * self.misidentification_width.to_value(u.rad)
                if np.abs(x) > dist:
                    nhit += 1
                    # Swap the signals.  This may require MPI communication
                    if det1 in local_dets and det2 in local_dets:
                        # Local operation
                        temp = signal[det1].copy()
                        signal[det1] = signal[det2]
                        signal[det2] = temp
                    elif det1 in local_dets:
                        # Send and receive signal
                        target = det2rank[det2]
                        comm.Send(signal[det1], dest=target, tag=nhit)
                        comm.Recv(signal[det1], source=target, tag=nhit)
                    elif det2 in local_dets:
                        # Receive and send signal
                        target = det2rank[det1]
                        sig2 = signal[det2].copy()
                        comm.Recv(signal[det2], source=target, tag=nhit)
                        comm.Send(sig2, dest=target, tag=nhit)
        ndet = len(all_dets)
        npair = ndet * (ndet - 1) // 2
        log.debug_rank(f"Misidentified {nhit} / {npair} detector pairs", comm=comm)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "shared_flags", "det_data":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimReadout")

        for obs in data.obs:
            comm = obs.comm.comm_group
            obs_id = obs.uid
            times = obs.shared[self.times].data
            all_dets = obs.all_detectors
            local_dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=local_dets, create_units=u.K)

            # Map detectors to their owning processes
            det2rank = {}
            for rank, dets in enumerate(obs.dist.dets):
                for det in dets:
                    det2rank[det] = rank
            focalplane = obs.telescope.focalplane
            signal = obs.detdata[self.det_data]
    
            self._add_glitches(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank
            )
            self._add_jumps(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank
            )
            self._fail_bolometers(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank
            )
            self._misidentify_bolometers(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank
            )
            
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.times, self.shared_flags],
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": [],
            "detdata": [self.det_data],
        }
        return prov

    def _accelerators(self):
        return list()

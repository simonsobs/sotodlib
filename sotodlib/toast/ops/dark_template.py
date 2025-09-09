# Copyright (c) 2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Operator for interfacing with the Maximum Likelihood Mapmaker.

"""

import os

from astropy import units as u
import numpy as np
import scipy.signal
import traitlets

import toast
from toast.traits import trait_docs, Unicode, Int, Instance, Bool, Float, List
from toast.ops import Operator
from toast.utils import Logger, Environment, rate_from_times
from toast.timing import function_timer, Timer
from toast.observation import default_values as defaults
from toast.fft import FFTPlanReal1DStore
from toast.instrument_coords import xieta_to_quat, quat_to_xieta


@trait_docs
class DarkTemplate(Operator):
    """Operator which assembles and projects out the dark template signal

    """

    API = Int(0, help="Internal interface version for this operator")

    naverage = Int(
        1000,
        help="Lowpass kernel size",
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("dtype_map")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check not in ["float", "float64"]:
            raise traitlets.TraitError(
                "Map data type must be float64 until so3g.ProjEng supports "
                "other map data types."
            )
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _get_dark_templates(self, ob):
        """ Construct low-passed dark bolometer templates
        """
        comm = ob.comm.comm_group

        # Gather dark tod to root process

        dark_tod = []
        fp = ob.telescope.focalplane
        for det in ob.local_detectors:
            if fp[det]["det_info:wafer:type"] != "DARK":
                continue
            tod = ob.detdata[self.det_data][det]
            # Check for typical pathologies
            if np.any(np.isnan(tod)):
                continue
            if np.std(tod) == 0:
                continue
            # Subtract the mean
            dark_tod.append(tod - np.mean(tod))
        if comm is not None:
            dark_tod = comm.gather(dark_tod)

        # Construct dark templates through PCA and lowpass

        if comm is None or comm.rank == 0:
            dark_tod = np.vstack(dark_tod)
            # PCA
            U, S, Vh = np.linalg.svd(dark_tod, full_matrices=False)
            templates = np.vstack(Vh)
            lowpassed = [np.ones(ob.n_local_samples)]  # always include the offset
            nsum = self.naverage
            lowpass = np.ones(nsum) / nsum
            nsample = ob.n_local_samples
            for template in templates:
                # zero-pad the template to avoid periodicity issues
                embedded = np.zeros(2 * nsample)
                embedded[:nsample] = template
                lowpassed_template = scipy.signal.convolve(
                    embedded, lowpass, mode="same"
                )
                lowpassed_template = lowpassed_template[:nsample]
                # Correct the ends for the zero padding (average includes zeros)
                frac = np.ones(template.size)
                frac[:nsum // 2] = nsum / np.arange(nsum // 2, nsum)
                frac[-nsum // 2:] = nsum / np.arange(nsum, nsum // 2, -1)
                lowpassed_template *= frac
                # Regularize the tail
                nsum_tail = nsum // 10
                lowpassed_template[:nsum_tail] = np.mean(template[:nsum_tail])
                lowpassed_template[-nsum_tail:] = np.mean(template[-nsum_tail:])                
                lowpassed.append(lowpassed_template)
            lowpassed = np.vstack(lowpassed)
        else:
            lowpassed = None

        # Broadcast

        if comm is not None:
            lowpassed = comm.bcast(lowpassed)

        return lowpassed

    @function_timer
    def _project_dark_templates(self, ob, templates):
        """ Project the provided templates out of every optical detector

        """
        if self.shared_flags is not None:
            common_flags = obs.shared[self.shared_flags].data & self.shared_flag_mask
        else:
            common_flags = np.zeros(ob.n_local_samples, dtype=np.uint8)

        fp = ob.telescope.focalplane
        last_good = None
        for det in ob.local_detectors:
            if fp[det]["det_info:wafer:type"] == "DARK":
                continue
            tod = ob.detdata[self.det_data][det]
            if self.det_flags is not None:
                det_flags = ob.detdata[self.det_data][det] & self.det_flag_mask
            else:
                det_flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
            good = np.logical_and(common_flags == 0, det_flags == 0)
            intervals = ob.intervals[self.view]
            if last_good is None or np.any(good != last_good):
                recompute = True
                last_good = good
                last_covs = []
                last_templates = []
            else:
                recompute = False
            for i, ival in enumerate(intervals):
                ind = slice(ival.first, ival.last)
                good_ind = good[ind].copy()
                if recompute:
                    templates_ind = templates[:, ind].copy()
                    masked_templates = templates_ind[:, good_ind].copy()
                    invcov = np.dot(masked_templates, masked_templates.T)
                    cov = np.linalg.inv(invcov)
                    last_covs.append(cov)
                    last_templates.append(masked_templates)
                else:
                    cov = last_covs[i]
                    masked_templates = last_templates[i]
                proj = np.dot(masked_templates, tod[ind][good_ind])
                coeff = np.dot(cov, proj)
                tod[ind] -= np.dot(coeff, templates_ind)

        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        comm = data.comm.comm_world
        gcomm = data.comm.comm_group
        timer.start()

        for ob in data.obs:
            templates = self._get_dark_templates(ob)
            self._project_dark_templates(ob, templates)

        return

    @function_timer
    def _finalize(self, data, **kwargs):
        pass

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"detdata": list()}
        if self.det_out is not None:
            prov["detdata"].append(self.det_out)
        return prov

    def _accelerators(self):
        return list()

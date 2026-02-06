# Copyright (c) 2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Operator for constructing and subtracting dark templates from timestream data.

This operator assembles a dark template signal from the data and projects it out,
enabling improved analysis of detector timestreams by removing correlated noise.
"""

import os
import pickle

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

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
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

    key = Unicode(
        "dark_templates",
        help="Shared data key for storing dark bolometer templates",
    )

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="If set, dark templates will be written out or loaded here.",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _gather_dark_tod(self, ob):
        """ Gather dark tod to root process

        """
        comm = ob.comm.comm_group
        dark_tod = []
        fp = ob.telescope.focalplane
        for det in ob.select_local_detectors(flagmask=self.det_mask):
            if fp[det]["det_info:wafer:type"] != "DARK":
                continue
            tod = ob.detdata[self.det_data][det]
            # Check for typical pathologies
            if np.any(np.isnan(tod)) or np.std(tod) == 0:
                continue
            # Subtract the mean
            dark_tod.append(tod - np.mean(tod))
        if comm is not None:
            dark_tod = comm.gather(dark_tod)
        return dark_tod

    @function_timer
    def _derive_templates(self, dark_tod):
        """ Derive dark templates from the dark TOD

        """
        dark_tod = np.vstack([x for x in dark_tod if len(x) > 0])
        nsample = dark_tod[0].size
        # PCA
        U, S, Vh = np.linalg.svd(dark_tod, full_matrices=False)
        templates = np.vstack(Vh)
        lowpassed = [np.ones(nsample)]  # always include the offset
        nsum = self.naverage
        lowpass = np.ones(nsum) / nsum
        for template in templates:
            # lowpass each template to suppress uncorrelated high frequency
            # noise.  Start by zero-padding the template to avoid
            # periodicity issues
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
        return lowpassed


    @function_timer
    def _load_dark_templates(self, ob):
        """ Try loading dark bolometer templates
        """
        log = Logger.get()
        comm = ob.comm.comm_group

        # See if the templates were already cached

        if self.cache_dir is not None:
            fname_cache = os.path.join(
                self.cache_dir,
                f"dark_templates.{ob.name}.pck",
            )
            if os.path.isfile(fname_cache):
                if comm is None or comm.rank == 0:
                    with open(fname_cache, "rb") as f:
                        templates = pickle.load(f)
                    log.debug(f"Loaded dark templates from {fname_cache}")
                else:
                    templates = None
                self._share_templates(ob, templates)
                return ob.shared[self.key]

        return None

    @function_timer
    def _share_templates(self, ob, templates):
        """ Put the dark templates in shared memory

        """
        comm = ob.comm.comm_group

        if templates is None:
            ntemplate, nsample = None, None
        else:
            ntemplate, nsample = templates.shape

        if comm is not None:
            ntemplate = comm.bcast(ntemplate)
            nsample = comm.bcast(nsample)

        ob.shared.create_column(
            self.key,
            shape=(nsample, ntemplate),
            dtype=np.float64,
        )
        if ob.comm.group_rank == 0:
            temp_trans = templates.T
        else:
            temp_trans = None
        ob.shared[self.key].set(temp_trans, offset=(0, 0), fromrank=0)
        return

    @function_timer
    def _save_dark_templates(self, ob, templates):
        """ Write the dark templates in a pickle file

        """
        # See if we are caching the templates

        log = Logger.get()
        comm = ob.comm.comm_group

        if self.cache_dir is None:
            return

        if comm is None or comm.rank == 0:
            fname_cache = os.path.join(
                self.cache_dir,
                f"dark_templates.{ob.name}.pck",
            )
            with open(fname_cache, "wb") as f:
                pickle.dump(templates.data.T, f)
            log.debug(f"Wrote dark templates to {fname_cache}")

        return

    @function_timer
    def _get_dark_templates(self, ob):
        """ Construct low-passed dark bolometer templates
        """
        log = Logger.get()
        comm = ob.comm.comm_group

        # gather all dark TOD to the root process

        dark_tod = self._gather_dark_tod(ob)

        # Construct dark templates through PCA and lowpass

        if comm is None or comm.rank == 0:
            templates = self._derive_templates(dark_tod)
        else:
            templates = None

        self._share_templates(ob, templates)

        return ob.shared[self.key]

    @function_timer
    def _project_dark_templates(self, ob, templates):
        """ Project the provided templates out of every optical detector

        """
        if self.shared_flags is not None:
            common_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
        else:
            common_flags = np.zeros(ob.n_local_samples, dtype=np.uint8)

        fp = ob.telescope.focalplane
        last_good = None
        for det in ob.select_local_detectors(flagmask=self.det_mask):
            if fp[det]["det_info:wafer:type"] == "DARK":
                continue
            tod = ob.detdata[self.det_data][det]
            if self.det_flags is not None:
                det_flags = ob.detdata[self.det_flags][det] & self.det_flag_mask
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
                    templates_ind = templates[ind].copy()
                    masked_templates = templates_ind[good_ind].copy()
                    invcov = np.dot(masked_templates.T, masked_templates)
                    cov = np.linalg.inv(invcov)
                    last_covs.append(cov)
                    last_templates.append(masked_templates)
                else:
                    cov = last_covs[i]
                    masked_templates = last_templates[i]
                proj = np.dot(masked_templates.T, tod[ind][good_ind])
                coeff = np.dot(cov, proj)
                tod[ind] -= np.dot(coeff, templates_ind.T)

        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        comm = data.comm.comm_world
        gcomm = data.comm.comm_group
        timer.start()

        if self.cache_dir is not None:
            if comm is None or comm.rank == 0:
                os.makedirs(self.cache_dir, exist_ok=True)

        for ob in data.obs:
            if ob.dist.comm_row_size != 1:
                raise RuntimeError(
                    "DarkTemplate only works with observations distributed "
                    "by detector"
                )
            templates = self._load_dark_templates(ob)
            if templates is None:
                # No precomputed templates found
                templates = self._get_dark_templates(ob)
                self._save_dark_templates(ob, templates)
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
        return prov

    def _accelerators(self):
        return list()

# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
import re
import warnings
from time import time

import h5py
import numpy as np
import traitlets
from astropy import units as u

from toast import qarray as qa
from toast.mpi import MPI, Comm, MPI_Comm
from toast.observation import default_values as defaults
from toast.timing import function_timer
from toast.traits import Int, Unicode, trait_docs
from toast.utils import Environment, Logger
from toast.ops import Operator
from toast.ops.demodulation import Lowpass


XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class DarkTemplates(Operator):
    """Operator that derives dark bolometer templates to use in destriping or
    filtering.

    """

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key to use as input"
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional shared flagging",
    )

    view = Unicode(
        None,
        allow_none=True,
        help="Use this view of the data in all observations",
    )

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="If specified, cache templates in this directory.",
    )

    optical_pattern = Unicode(
        f"^.*",
        allow_none=True,
        help="Regex pattern to match against optical detector names. "
        "Only detectors that match the pattern are assigned dark templates."
    )

    dark_pattern = Unicode(
        f"^demod0.*",
        allow_none=True,
        help="Regex pattern to match against dark detector names. Only detectors "
        "that match the pattern are used to derive dark templates."
    )

    mode = Unicode(
        "mean",
        help="Method of deriving the dark template. Supported modes:\n"
        "mean -- Average of all functioning dark bolometers per wafer\n"
        "median -- Median of all functioning dark bolometers per wafer\n"
        "PCA:nmode -- Perform Principal Component Analysis and pick `nmode` "
        "leading modes`"
        )

    submode = Unicode(
        "raw",
        help="Post processing of the dark template(s). Supported modes\n"
        "raw -- no post processing\n"
        "lowpass:<Quantity> -- low-pass the dark template\n"
        "split:<Quantity>,<Quantity>,... -- split the template into several "
        "complementary templates at given frequencies\n",
    )

    order = Int(
        1,
        help="Expansion order of the templates.  Zeroth order is never added "
        "to avoid degeneracies",
    )

    template_name = Unicode(
        "dark_templates",
        help="Observation key for recording the calculated templates",
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
            raise traitlets.TraitError(
                "Shared flag mask should be a positive integer"
            )
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError(
                "Det flag mask should be a positive integer"
            )
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        """Derive the templates

        Args:
            data (toast.Data): The distributed data.

        """
        log = Logger.get()
        wcomm = data.comm.comm_world
        gcomm = data.comm.comm_group

        if (wcomm is None or wcomm.rank == 0) and self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "ERROR: DarkTemplates assumes data are distributed "
                msg += "by detector"
                raise RuntimeError(msg)
            fp = ob.telescope.focalplane

            # Get the list of all detectors that are not cut

            local_dets = ob.select_local_detectors(
                detectors, flagmask=self.det_mask
            )
            local_dark_dets = []
            local_optical_dets = []
            for det in local_dets:
                if fp[det]["det_info:wafer:type"] == "DARK":
                    local_dark_dets.append(det)
                else:
                    local_optical_dets.append(det)

            # Broadcast the dark TOD to the group
            
            dark_tod = {}
            for dark_det in local_dark_dets:
                dark_tod[dark_det] = ob.detdata[self.det_data][dark_det]
            all_dark_tod = gcomm.allgather(dark_tod)
            for dark_dict in all_dark_tod:
                dark_tod.update(dark_dict)

            # See if we have cached templates

            cached_dark_tod = {}
            if self.cache_dir is not None:
                fname_cache = os.path.join(self.cache_dir, f"{ob.name}.pck")
                if os.path.isfile(fname_cache):
                    # Combine new and cached dark TOD
                    log.info_rank(
                        f"Loading dark TOD from "
                        f"{fname_cache}",
                        comm=gcomm,
                    )
                    with open(fname_cache, "rb") as f:
                        cached_dark_tod = pickle.load(f)
            else:
                fname_cache = None

            updated = False
            for key, value in dark_tod.items():
                if key not in cached_dark_tod:
                    cached_dark_tod[key] = value
                    updated = True
            dark_tod = cached_dark_tod

            # Optionally cache the dark TOD

            if fname_cache is not None and updated:
                if gcomm is None or gcomm.rank == 0:
                    with open(fname_cache, "wb") as f:
                        pickle.dump(dark_tod, f)
                log.info_rank(f"Cached dark TOD in {fname_cache}", comm=gcomm)

            # Optionally select a subset of dark detectors

            if self.dark_pattern is not None:
                pattern = re.compile(self.dark_pattern)
                keep = {}
                for det, tod in dark_tod.items():
                    if pattern.match(det) is not None:
                        keep[det] = tod
                dark_tod = keep

            ndark = len(dark_tod)
            if ndark == 0:
                msg = f"No dark detectors in {ob.name} match "
                msg += f"det_mask={self.det_mask}, "
                msg += f"pattern='{self.dark_pattern}'. "
                log.warning(msg)
                continue

            log.info_rank(
                f"Building dark templates from {ndark} detector streams.",
                comm=gcomm,
            )

            # Every process builds all dark templates.  This should be a negligible
            # amount of work

            nsample = ob.n_local_samples
            arr = np.zeros([ndark, nsample], dtype=float)
            for idet, det in enumerate(dark_tod):
                arr[idet] = dark_tod[det]

            dark_templates = {}
            if self.mode.lower() == "mean":
                dark_templates["dark_mean"] = np.mean(arr, 0)
            elif self.mode.lower() == "median":
                dark_templates["dark_median"] = np.median(arr, 0)
            elif self.mode.startswith("PCA:"):
                npca = int(self.mode.split(":")[1])
                npca = min(npca, ndark)
                U, S, Vh = np.linalg.svd(arr, full_matrices=False)
                for i in range(npca):
                    key = f"dark_pca_{i:02}"
                    dark_templates[key] = Vh[i]
            else:
                msg = f"Cannot parse mode = {self.mode}"
                raise RuntimeError(msg)

            # Optionally, lowpass/bandpass filter all computed templates
            # This happens after caching so cached templates are never
            # filtered.

            if self.submode != "raw":
                fsample = ob.telescope.focalplane.sample_rate
                single_lowpass = None
                lowpasses = None
                if self.submode.startswith("lowpass:"):
                    fmax = u.Quantity(self.submode.replace("lowpass:", ""))
                    single_lowpass = Lowpass(fmax, fsample)
                elif self.submode.startswith("split:"):
                    lowpasses = []
                    for freq in self.submode.replace("split:", "").split(","):
                        fmax = u.Quantity(freq)
                        lowpasses.append(Lowpass(fmax, fsample))
                else:
                    msg = f"Unknown submode: {self.submode}"
                    raise RuntimeError(msg)

                # Apply the filters to all templates

                for name in list(dark_templates.keys()):
                    template = dark_templates[name]
                    if single_lowpass is not None:
                        # Lowpass mode, replace original template
                        template[:] = single_lowpass(template)
                    else:
                        # Split mode, derive multiple templates and
                        # conserve total power
                        for i, lowpass in enumerate(lowpasses):
                            lowpassed = lowpass(template)
                            template -= lowpassed
                            dark_templates[f"{name}-lowpass{i}"] = lowpassed

            # Optionally add higher order templates

            if self.order > 1:
                for name, template in dark_templates.items:
                    for p in range(2, self.order + 1):
                        dark_templates[f"{name}-order{p}"] = template**p

            # Attach select detectors to the templates.
            # For now, there is only one key

            key = "optical"
            dark_templates = {key : dark_templates}

            if self.optical_pattern is None:
                pattern = None
            else:
                pattern = re.compile(self.optical_pattern)

            det_to_key = {}
            for det in local_optical_dets:
                if pattern is not None and pattern.match(det) is None:
                    # This detector does not need a template
                    continue
                det_to_key[det] = key

            # Store the dark templates in the observation

            dark_templates["det_to_key"] = det_to_key
            ob[self.template_name] = dark_templates

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

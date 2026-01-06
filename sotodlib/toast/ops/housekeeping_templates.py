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
class HouseKeepingTemplates(Operator):
    """Operator that derives housekeeping templates to use in destriping
    or filtering.

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

    pattern = Unicode(
        f"^.*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors "
        "that match the pattern are associated with the template."
    )

    mode = Unicode(
        "separate",
        help="Method of deriving the housekeeping templates. Supported modes:\n"
        "separate -- Process each housekeeping field separately\n"
        "mean -- Average of all housekeeping fields\n"
        "median -- Median of all housekeeping fields\n"
        "PCA:nmode -- Perform Principal Component Analysis and pick `nmode` "
        "leading modes`"
        )

    submode = Unicode(
        "raw",
        help="Post processing of the intensity template. Supported modes\n"
        "raw -- no post processing\n"
        "lowpass:<Quantity> -- low-pass the intensity template\n"
        "split:<Quantity>,<Quantity>,... -- split the template into several "
        "complementary templates at given frequencies\n",
    )

    order = Int(
        1,
        help="Expansion order of the templates.  Zeroth order is never added "
        "to avoid degeneracies",
    )

    hkkey = Unicode(
        None,
        allow_none=True,
        help="Housekeeping data key to extract and process into a template. "
        "It will be lowpassed or split according to `submode`. Multiple keys "
        "can be separated by commas."
    )

    template_name = Unicode(
        "housekeeping_templates",
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
        gcomm = data.comm.comm_group

        if self.hkkey is None:
            msg = f"You must set the `hkkey` trait before applying "
            msg += "HouseKeepingTemplates"
            raise RuntimeError(msg)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "ERROR: HouseKeepingTemplates assumes data are "
                msg += "distributed by detector"
                raise RuntimeError(msg)

            # Get the list of all detectors that are not cut
            local_dets = ob.select_local_detectors(
                detectors, flagmask=self.det_mask
            )
            # See if we already have cached templates

            hkfields = {}
            if self.cache_dir is not None:
                fname_cache = os.path.join(self.cache_dir, f"{ob.name}.pck")
                if os.path.isfile(fname_cache):
                    log.info_rank(
                        f"Loading housekeeping fields from "
                        f"{fname_cache}",
                        comm=gcomm,
                    )
                    with open(fname_cache, "rb") as f:
                        hkfields = pickle.load(f)
            else:
                fname_cache = None

            # If not loaded, process housekeeping data into templates

            updated = False
            for key in self.hkkey.split(","):
                if key in hkfields:
                    # This key is already loaded from cache.
                    pass
                else:
                    hkfield = ob.hk.interp(key)
                    hkfield -= np.mean(hkfield)
                    hkfields[key] = hkfield
                    updated = True

            # Optionally save the housekeeping fields

            if fname_cache is not None and updated:
                if gcomm is None or gcomm.rank == 0:
                    with open(fname_cache, "wb") as f:
                        pickle.dump(hkfields, f)
                log.info_rank(
                    f"Cached housekeeping fields in {fname_cache}", comm=gcomm
                )

            # Optionally discard unused housekeeping fields.  The cache
            # may have included fields that we do not want

            keep = {}
            for key in self.hkkey.split(","):
                keep[key] = hkfields[key]
            hkfields = keep

            nhk = len(hkfields)
            log.info_rank(
                f"Building housekeeping templates from {nhk} streams.",
                comm=gcomm,
            )

            # Optionally, combine the fields

            if self.mode == "separate":
                hktemplates = hkfields
            else:
                hktemplates = {}
                arr = []
                for template in hkfields.values():
                    arr.append(template)
                arr = np.vstack(arr)
                if self.mode.lower() == "mean":
                    hktemplates["housekeeping_mean"] = np.mean(arr, 0)
                elif self.mode.lower() == "median":
                    hktemplates["housekeeping_median"] = np.median(arr, 0)
                elif self.mode.startswith("PCA:"):
                    npca = int(self.mode.split(":")[1])
                    npca = min(npca, ndark)
                    U, S, Vh = np.linalg.svd(arr, full_matrices=False)
                    for i in range(npca):
                        key = f"housekeeping_pca_{i:02}"
                        hktemplates[key] = Vh[i]
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

                for name in list(hktemplates.keys()):
                    template = hktemplates[name]
                    if single_lowpass is not None:
                        # Lowpass mode, replace original template
                        template[:] = single_lowpass(template)
                    else:
                        # Split mode, derive multiple templates and
                        # conserve total power
                        for i, lowpass in enumerate(lowpasses):
                            lowpassed = lowpass(template)
                            template -= lowpassed
                            hktemplates[f"{name}-lowpass{i}"] = lowpassed

            # Optionally add higher order templates

            if self.order > 1:
                for name, template in hktemplates.items:
                    for p in range(2, self.order + 1):
                        hktemplates[f"{name}-order{p}"] = template**p

            # Attach select detectors to the templates.
            # For now, there is only one key

            key = "detector"
            hktemplates = {key : hktemplates}

            if self.pattern is None:
                pattern = None
            else:
                pattern = re.compile(self.pattern)

            det_to_key = {}
            for det in local_dets:
                if pattern is not None and pattern.match(det) is None:
                    # This detector does not need a template
                    continue
                det_to_key[det] = key

            # Store the dark templates in the observation

            hktemplates["det_to_key"] = det_to_key
            ob[self.template_name] = hktemplates

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

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


XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class IntensityTemplates(Operator):
    """Operator that derives intensity templates to use in destriping or
    filtering.  Typically run after demodulation or pair differencing
    when detector data is either pure intensity or pure polarization.

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
        f"^demod0.*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors "
        "that match the pattern are used to derive intensity templates."
    )

    mode = Unicode(
        "radius:1deg",
        help="Method of deriving the intensity template. Supported modes:\n"
        "radius:<Quantity> -- Average the intensity in the disc.  Use a small "
        "value to match only one pixel\n"
        "all -- Average all detectors matching `pattern`"
        )

    submode = Unicode(
        "raw",
        help="Post processing of the intensity template. Supported modes\n"
        "raw -- no post processing\n"
        "lowpass:<Quantity> -- low-pass the intensity template\n"
        "split:<Quantity>,<Quantity>,... -- split the template into several "
        "complementary templates at given frequencies\n",
    )

    fpkeys = Unicode(
        "det_info:wafer:type,det_info:wafer:bandpass",
        allow_none=True,
        help="Comma-separated list of focalplane keys to split the intensity "
        "detectors with",
    )

    hkkey = Unicode(
        None,
        allow_none=True,
        help="An additional housekeeping data key to extract and process into "
        "a template.  It will be lowpassed or split according to `submode` "
        "just like the intensity templates.  Multiple keys can be separated by "
        "commas."
    )

    template_name = Unicode(
        "intensity_templates",
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

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "ERROR: IntensityTemplates assumes data are distributed "
                msg += "by detector"
                raise RuntimeError(msg)

            # Get the list of all detectors that are not cut
            local_dets = ob.select_local_detectors(
                detectors, flagmask=self.det_mask
            )
            if self.pattern is None:
                local_intensity_dets = local_dets
            else:
                pat = re.compile(self.pattern)
                local_intensity_dets = [
                    det for det in local_dets if pat.match(det) is not None
                ]
                if len(local_intensity_dets) == 0:
                    msg = f"No detectors in {ob.name} match '{self.pattern}'"
                    raise RuntimeError(msg)
            if gcomm.size == 1:
                intensity_dets = local_intensity_dets
            else:
                proc_dets = gcomm.gather(local_intensity_dets)
                all_dets = None
                if gcomm.rank == 0:
                    all_set = set()
                    for pdets in proc_dets:
                        for d in pdets:
                            all_set.add(d)
                    intensity_dets = list(sorted(all_set))
                else:
                    intensity_dets = None
                intensity_dets = gcomm.bcast(intensity_dets)

            # Assemble and reduce the TOD for all qualifying detectors

            ndet_intensity = len(intensity_dets)
            nsample = ob.n_local_samples
            all_tod = np.zeros([ndet_intensity, nsample])
            for idet, intensity_det in enumerate(intensity_dets):
                if intensity_det not in ob.local_detectors:
                    continue
                all_tod[idet] = ob.detdata[self.det_data][intensity_det]
            all_tod = gcomm.allreduce(all_tod)

            # Get the relevant focalplane keys to go with the data.
            # We concatenate all the keys together and each unique
            # combination produces a separate template

            fp = ob.telescope.focalplane
            if self.fpkeys is None:
                fpkeys = None
            else:
                fpkeys = []
                for intensity_det in intensity_dets:
                    values = []
                    for key in self.fpkeys.split(","):
                        values.append(str(fp[intensity_det][key]))
                    keystring = "+".join(values)
                    fpkeys.append(keystring)
            unique_fpkeys = list(set(fpkeys))
            nunique = len(unique_fpkeys)
            log.info_rank(
                f"Building intensity templates for {nunique} key combinations: "
                f"{unique_fpkeys}",
                comm=gcomm,
            )

            # See if we already have cached templates

            intensity_templates = {}
            if self.cache_dir is not None:
                fname_cache = os.path.join(self.cache_dir, f"{ob.name}.pck")
                if os.path.isfile(fname_cache):
                    log.info_rank(
                        f"Loading precomputed intensity templates from "
                        f"{fname_cache}",
                        comm=gcomm,
                    )
                    with open(fname_cache, "rb") as f:
                        intensity_templates = pickle.load(f)
            else:
                fname_cache = None

            # See if we have a housekeeping template

            hkfields = {}
            if self.hkkey is not None:
                for key in self.hkkey.split(","):
                    hkfields[key] = ob.hk.interp(key)

            # Every process builds all intensity templates that it needs

            if self.mode.lower == "all":
                template_key = "all"
                radius = None
            else:
                template_key = "{pixel}"
                if self.mode.startswith("radius:"):
                    radius = u.Quantity(self.mode.split(":")[1])

            det_to_key = {}
            for det in local_dets:
                # See if this detector needs a template
                if det.startswith("demod4r") or det.startswith("demod4i"):
                    # Yes, derive the template key
                    pixel = det.replace("demod4r_", "").replace("demod4i_", "")
                    key = template_key.format(pixel=pixel)
                    if key not in intensity_templates:
                        intensity_templates[key] = {}
                    det_to_key[det] = key
                    # Get a shorthand for templates for this detector
                    templates = intensity_templates[key]
                    # See if we need to add housekeeping templates
                    for hkkey, hkfield in hkfields.items():
                        if hkkey not in templates:
                            templates[hkkey] = hkfield
                    # Compute the intensity-based template(s) for this key
                    quat = fp[det]["quat"]
                    vec = qa.rotate(quat, ZAXIS)
                    for fpkey in unique_fpkeys:
                        use_det = np.zeros(ndet_intensity, dtype=bool)
                        for idet, intensity_det in enumerate(intensity_dets):
                            if fpkeys[idet] != fpkey:
                                # mismatched key, do not include
                                continue
                            if radius is not None:
                                # Check for distance
                                iquat = fp[intensity_det]["quat"]
                                ivec = qa.rotate(iquat, ZAXIS)
                                dist = np.arccos(np.dot(vec, ivec))
                                if dist > radius.to_value(u.rad):
                                    # too far, do not include
                                    continue
                            use_det[idet] = True
                        if not np.any(use_det):
                            continue
                        mean_intensity = np.mean(all_tod[use_det], 0)
                        templates[fpkey] = mean_intensity

            # Optionally, cache all available templates

            if fname_cache is not None:
                if gcomm is not None:
                    all_intensity_templates = gcomm.gather(intensity_templates)
                    if gcomm.rank == 0:
                        for templates in all_intensity_templates:
                            intensity_templates.update(templates)
                if gcomm is None or gcomm.rank == 0:
                    with open(fname_cache, "wb") as f:
                        pickle.dump(intensity_templates, f)
                log.info_rank(f"Cached templates in {fname_cache}", comm=gcomm)

            # Optionally, lowpass/bandpass filter all computed templates
            # This happens after caching so cached templates are never
            # filtered.

            if self.submode != "raw":
                msg = "Filtering intensity templates not implemented yet"
                raise NotImplementedError(msg)

            # Store the intensity templates in the observation

            intensity_templates["det_to_key"] = det_to_key
            ob[self.template_name] = intensity_templates

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

# Copyright (c) 2025-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
import re
import warnings
from time import time, sleep

import h5py
import numpy as np
import traitlets
from astropy import units as u

from toast import qarray as qa
from toast.mpi import MPI, Comm, MPI_Comm
from toast.observation import default_values as defaults
from toast.timing import function_timer, Timer
from toast.traits import Int, Unicode, Quantity, trait_docs
from toast.utils import Environment, Logger
from toast.ops import Operator
from toast.ops.demodulation import Lowpass

from .utils import persistent_pickle_load

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

    source_pattern = Unicode(
        f"^demod0.*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors "
        "that match the pattern are used to derive intensity templates.",
    )

    target_pattern = Unicode(
        f"^demod4.*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors "
        "that match the pattern are assigned an intensity template.",
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

    order = Int(
        1,
        help="Expansion order of the templates.  Zeroth order is never added "
        "to avoid degeneracies",
    )

    fpkeys = Unicode(
        "det_info:wafer:type,det_info:wafer:bandpass",
        allow_none=True,
        help="Comma-separated list of focalplane keys to split the intensity "
        "detectors with",
    )

    template_name = Unicode(
        "intensity_templates",
        help="Observation key for recording the calculated templates",
    )

    focalplane_pixel_width = Quantity(
        0.01 * u.deg,
        help="Focalplane pixel width is used to assign detector positions into "
        "rows and columns.  Should be smaller than typical separation of "
        "nearest neighbor centers but not so small that detector in the same "
        "focalplane pixel can get separated.",
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
    def _det_to_pixel(self, det, focalplane):
        """Translate detector name into pixel name

        For now, this is done by translating the detector quaternion
        into a row and column on the focalplane.  If a suitable
        detector property is reliably available in the focalplane
        database, it should be used instead.
        """

        # Get a detector position on the focal plane by rotating the
        # boresight on the X-axis and taking the detector pointing
        # in lon/lat

        quat = focalplane[det]["quat"]
        yrot = qa.rotation(YAXIS, np.pi / 2)
        theta, phi, _ = qa.to_iso_angles(qa.mult(yrot, quat))
        lon = np.degrees(phi)
        lat = 90 - np.degrees(theta)

        wpix = self.focalplane_pixel_width.to_value(u.degree)
        col = int(lon / wpix - 0.5)
        row = int(lat / wpix - 0.5)
        pixel = f"({col},{row})"

        return pixel

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
                msg = "ERROR: IntensityTemplates assumes data are distributed "
                msg += "by detector"
                raise RuntimeError(msg)
            fp = ob.telescope.focalplane

            # Get the list of all detectors that are not cut
            local_dets = ob.select_local_detectors(
                detectors, flagmask=self.det_mask
            )
            local_intensity_dets = []
            for det in local_dets:
                key = "det_info:wafer:type"
                if key in fp.properties and fp[det]["det_info:wafer:type"] == "DARK":
                    is_optical = False
                else:
                    is_optical = True
                if is_optical:
                    local_intensity_dets.append(det)
            if self.source_pattern is not None:
                pat = re.compile(self.source_pattern)
                local_intensity_dets = [
                    det for det in local_intensity_dets
                    if pat.match(det) is not None
                ]
            if gcomm is None or gcomm.size == 1:
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

            if len(intensity_dets) == 0:
                msg = f"No detectors in {ob.name} match "
                msg += f"det_mask = {self.det_mask}, "
                msg += f"source_pattern = '{self.source_pattern}'. "
                log.warning(msg)
                continue

            # Assemble and reduce the TOD for all qualifying detectors

            ndet_intensity = len(intensity_dets)
            nsample = ob.n_local_samples
            all_tod = np.zeros([ndet_intensity, nsample])
            for idet, intensity_det in enumerate(intensity_dets):
                if intensity_det not in ob.local_detectors:
                    continue
                all_tod[idet] = ob.detdata[self.det_data][intensity_det]
            if gcomm is not None:
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
                result = persistent_pickle_load(fname_cache)
                if result is not None:
                    intensity_templates = result
            else:
                fname_cache = None

            # Every process builds all intensity templates that it needs

            if self.mode.lower() == "all":
                template_key = "all"
                radius = None
            else:
                template_key = "{pixel}"
                if self.mode.startswith("radius:"):
                    radius = u.Quantity(self.mode.split(":")[1])

            if self.target_pattern is None:
                pattern = None
            else:
                pattern = re.compile(self.target_pattern)

            det_to_key = {}
            for det in local_dets:
                # See if this detector needs a template
                if pattern is not None and pattern.match(det) is None:
                    # No need for a template
                    continue
                # Yes, derive the template key
                pixel = self._det_to_pixel(det, fp)
                key = template_key.format(pixel=pixel)
                if key not in intensity_templates:
                    intensity_templates[key] = {}
                det_to_key[det] = key
                # Get a shorthand for templates for this detector
                templates = intensity_templates[key]
                # Compute the intensity-based template(s) for this key
                quat = fp[det]["quat"]
                vec = qa.rotate(quat, ZAXIS)
                for fpkey in unique_fpkeys:
                    if fpkey in templates:
                        # Already cached
                        continue
                    use_det = np.zeros(ndet_intensity, dtype=bool)
                    for idet, intensity_det in enumerate(intensity_dets):
                        if fpkeys[idet] != fpkey:
                            # mismatched key, do not include
                            continue
                        if radius is not None:
                            # Check for distance
                            iquat = fp[intensity_det]["quat"]
                            ivec = qa.rotate(iquat, ZAXIS)
                            dp = np.dot(vec, ivec)
                            # arccos() will throw warnings if dp is not
                            # strictly in [-1, 1]
                            if dp < -1:
                                dp = -1
                            elif dp > 1:
                                dp = 1
                            dist = np.arccos(dp)
                            if dist > radius.to_value(u.rad):
                                # too far, do not include
                                continue
                        use_det[idet] = True
                    if not np.any(use_det):
                        continue
                    mean_intensity = np.mean(all_tod[use_det], 0)
                    mean_intensity -= np.mean(mean_intensity)
                    templates[fpkey] = mean_intensity

            # Optionally, cache all available templates

            if fname_cache is not None:
                if gcomm is not None:
                    templates_send = {}
                    if gcomm.rank != 0:
                        # Only send the templates that may have been updated
                        for key in det_to_key.values():
                            templates_send[key] = intensity_templates[key]
                    all_intensity_templates = gcomm.gather(templates_send)
                    if gcomm.rank == 0:
                        for templates in all_intensity_templates:
                            intensity_templates.update(templates)
                if gcomm is None or gcomm.rank == 0:
                    with open(fname_cache, "wb") as f:
                        pickle.dump(intensity_templates, f)
                log.info_rank(f"Cached templates in {fname_cache}", comm=gcomm)

            # Loading from cache causes there to be unused templates.
            # Discard all that are not relevant for local detectors
            unused_keys = set(intensity_templates.keys())
            for det in ob.local_detectors:
                if det in det_to_key:
                    key = det_to_key[det]
                    if key in unused_keys:
                        unused_keys.remove(key)
            for key in unused_keys:
                del intensity_templates[key]

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

                for key, templates in intensity_templates.items():
                    for name in list(templates.keys()):
                        template = templates[name]
                        if single_lowpass is not None:
                            # Lowpass mode, replace original template
                            template[:] = single_lowpass(template)
                        else:
                            # Split mode, derive multiple templates and
                            # conserve total power
                            for i, lowpass in enumerate(lowpasses):
                                lowpassed = lowpass(template)
                                template -= lowpassed
                                templates[f"{name}-lowpass{i}"] = lowpassed

            # Optionally add higher order templates

            if self.order > 1:
                for key, templates in intensity_templates.items():
                    for name, template in templates.items:
                        for p in range(2, self.order + 1):
                            templates[f"{name}-order{p}"] = template**p

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

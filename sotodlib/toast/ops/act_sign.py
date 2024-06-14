# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.utils import Logger
from toast.traits import trait_docs, Int, Unicode
from toast.timing import function_timer
from toast.observation import default_values as defaults
from toast.ops import Operator


@trait_docs
class ActSign(Operator):
    """Apply a sign flip to the responsivity of ACT detector data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    fp_column = Unicode(
        "det_info:optical_sign", help="Focalplane table column with sign factor"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            fp_data = {
                x: y for x, y in zip(
                    ob.telescope.focalplane.detector_data["name"],
                    ob.telescope.focalplane.detector_data[self.fp_column]
                )
            }
            for det in ob.local_detectors:
                ob.detdata[self.det_data][det, :] *= fp_data[det]

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "detdata": self.det_data,
        }
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()

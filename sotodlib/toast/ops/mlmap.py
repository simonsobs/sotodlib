# Copyright (c) 2021-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import pickle
import sys

import numpy as np

import toast
import toast.traits as ttraits


@ttraits.trait_docs
class MLMapmaker(toast.Operator):
    """Run the Maximum Likelihood mapmaker on TOAST data.
    
    This takes the distributed data and passes a named detdata object from each
    observation to the ML mapmaker.

    """

    # Class traits

    API = ttraits.Int(0, help="Internal interface version for this operator")

    det_data = ttraits.Unicode("signal", help="Observation detdata key for the timestream data")

    max_err = ttraits.Float(1.0e-6, help="Relative convergence limit")

    max_iter = ttraits.Int(100, help="Maximum number of iterations")

    output_dir = Unicode(
        None,
        allow_none=True,
        help="If specified, write output data products to this directory",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        pass

import numpy as np
import so3g

class PmatCut:
    """Implementation of cuts-as-extra-degrees-of-freedom for a single obs."""
    def __init__(self, cuts, model=None, params={"resolution":100, "nmax":100}):
        self.cuts   = cuts
        self.model  = model or "full"
        self.params = params
        self.njunk  = so3g.process_cuts(self.cuts.ranges, "measure", self.model, self.params, None, None)

    def forward(self, tod, junk):
        """Project from the cut parameter (junk) space for this scan to tod."""
        so3g.process_cuts(self.cuts.ranges, "insert", self.model, self.params, tod, junk)

    def backward(self, tod, junk):
        """Project from tod to cut parameters (junk) for this scan."""
        so3g.process_cuts(self.cuts.ranges, "extract", self.model, self.params, tod, junk)
        self.clear(tod)

    def clear(self, tod):
        junk = np.empty(self.njunk, tod.dtype)
        so3g.process_cuts(self.cuts.ranges, "clear", self.model, self.params, tod, junk)
class Signal:
    """This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc."""

    def __init__(self, name, ofmt, output, ext, **kwargs):
        """Initialize a Signal. It probably doesn't make sense to construct a generic signal
        directly, though. Use one of the subclasses.
        Arguments:
        * name: The name of this signal, e.g. "sky", "cut", etc.
        * ofmt: The format used when constructing output file prefix
        * output: Whether this signal should be part of the output or not.
        * ext: The extension used for the files.
        * **kwargs: additional keyword based parameters, accessible as class parameters
        """
        self.name = name
        self.ofmt = ofmt
        self.output = output
        self.ext = ext
        self.dof = None
        self.ready = False
        self.__dict__.update(kwargs)

    def add_obs(self, id, obs, nmat, Nd, **kwargs):
        pass

    def prepare(self):
        self.ready = True

    def forward(self, id, tod, x):
        pass

    def backward(self, id, tod, x):
        pass

    def precon(self, x):
        return x

    def to_work(self, x):
        return x.copy()

    def from_work(self, x):
        return x

    def wzeros(self):
        return 0

    def write(self, prefix, tag, x):
        pass

    def translate(self, other, x):
        return x

    def prior(self, xin, xout):
        pass

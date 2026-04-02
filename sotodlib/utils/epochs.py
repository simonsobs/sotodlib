from copy import deepcopy
from dataclasses import dataclass, field
from itertools import accumulate
from operator import add, and_, mul, or_
from typing import Any, Self, cast

import yaml
from deepdiff import DeepDiff

# TODO: Add check for a specific piece of data -> existance and strictness
# TODO: Add method to search via tags
# TODO: Make Invtervals aware of what Epochs they are members of and make Epochs aware of what Eras they are members of


@dataclass
class Interval:
    """
    Smallest unit of the epoch system.
    Here we have a time interval containing a set of tags and an arbitrary dict of data.
    This can be sliced with standard slicing notation, but the indices are ctime.

    Two `Interval` instances can be combined with the following operators,
    all operators are functionally addition but with differing checks:

    * `|`: Allow gaps between intervals, but data and tags can differ
    * `&`: Intervals must be overlapping, but data and tags can differ
    * `*`: Allow gaps between intervals, data must be the same but tags can differ
    * `+`: Invervals must be overlapping, data must be the same but tags can differ
    * `/`: Allow gaps between intervals, data can differ but tags must be the same
    * `-`: Invervals must be overlapping, data can differ but tags must be the same
    * `**`: Allow gaps between invervals, data and tags must be the same
    * `@`: Allow gaps between intervals, data and tags must be the same

    In cases where `data` is not the same but the `Invervals` are combined,
    the new `data` will keep values from the right hand `Inverval` for overlapping keys.
    """

    name: str
    start: float
    stop: float
    tags: set[str]
    data: dict[str, Any]

    def __getitem__(self: Self, val) -> Self:
        to_ret = deepcopy(self)
        if not isinstance(val, slice):
            val = slice(val)
        if val.start is not None:
            if self.start < val.start:
                raise IndexError(
                    f"Start of slice out of range, interval starts at {self.start} but slice starts at {val.start}!"
                )
            to_ret.start = val.start
        if val.stop is not None:
            if self.stop > val.stop:
                raise IndexError(
                    f"End of slice out of range, interval stops at {self.stop} but slice stops at {val.stop}!"
                )
            to_ret.stop = val.stop
        return to_ret

    def combine(
        self: Self,
        tocombine: Self,
        in_place: bool = True,
        allow_gap: bool = False,
        same_data: bool = False,
        same_tag: bool = False,
    ) -> Self:
        if not isinstance(tocombine, Interval):
            raise TypeError(
                f"Can only add intervals to other intervals, not {type(tocombine)}!"
            )
        combined = self
        if in_place:
            combined = deepcopy(self)
        if not allow_gap and (
            self.start > tocombine.stop or self.stop < tocombine.start
        ):
            raise ValueError(
                "Can't combine without overlapping time ranges with allow_gap=False"
            )
        combined.start = min(self.start, tocombine.start)
        combined.stop = max(self.stop, tocombine.stop)

        if not same_tag and self.tags != tocombine.tags:
            raise ValueError("Can't combine without identical tags with same_tag=False")
        combined.tags = self.tags.union(tocombine.tags)

        diff = DeepDiff(self.data, self.data)
        if not same_data and len(diff) != 0:
            raise ValueError(
                "Can't combine without identical data with same_data=False"
            )
        combined.data.update(tocombine.data)

        return combined

    def __or__(self, tocombine) -> Self:
        return self.combine(tocombine, False, False, False, False)

    def __ior__(self, tocombine) -> Self:
        return self.combine(tocombine, True, False, False, False)

    def __and__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, False, False)

    def __iand__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, False, False)

    def __mul__(self, tocombine) -> Self:
        return self.combine(tocombine, False, False, True, False)

    def __imul__(self, tocombine) -> Self:
        return self.combine(tocombine, True, False, True, False)

    def __add__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, True, False)

    def __iadd__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, True, False)

    def __div__(self, tocombine) -> Self:
        return self.combine(tocombine, False, False, False, True)

    def __idiv__(self, tocombine) -> Self:
        return self.combine(tocombine, True, False, False, True)

    def __sub__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, False, True)

    def __isub__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, False, True)

    def __pow__(self, tocombine) -> Self:
        return self.combine(tocombine, False, False, False, True)

    def __ipow__(self, tocombine) -> Self:
        return self.combine(tocombine, True, False, False, True)

    def __matmul__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, True, True)

    def __imatmul__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, True, True)


@dataclass
class Epoch:
    """
    Dataclass for storing a collection of `Invervals`s.
    `Inverval`s must be overlapping.

    Attributes
    ----------
    name : str
        Name for this epoch.
    covers: tuple[Interval]
        Collections of overlapping `Invervals` that make up an `Epoch`.
    strict : bool
        If `True` than in addition to being overlapping the `Intervals` must
        also contain the same data.
    """

    name: str
    covers: tuple[Interval]
    strict: bool
    _internal: Interval = field(init=False)

    def __post_init__(self):
        self._internal = cast(
            Interval, accumulate(self.covers, mul if self.strict else or_)
        )

    def __setattr__(self, name, value):
        if name == "covers" or name == "strict":
            self.__post_init__()
        return super().__setattr__(name, value)


@dataclass
class Era:
    """
    Dataclass for storing a collection of `Epoch`s.
    `Epoch`s must be non-overlapping but can have gaps in time between them.

    Attributes
    ----------
    name : str
        Name for this era.
    epochs : tuple[Epoch]
        Collections of non-overlapping `Epochs` that make up an `Era`.
    strict : bool
        If `True` than in addition to being non-overlapping the `Epochs` must
        also contain the same data.
    """

    name: str
    epochs: tuple[Epoch]
    strict: bool
    _internal: Interval = field(init=False)

    def __post_init__(self):
        self._internal = cast(
            Interval,
            accumulate(
                [e._internal for e in self.epochs], add if self.strict else and_
            ),
        )

    def __setattr__(self, name, value):
        if name == "epochs" or name == "strict":
            self.__post_init__()
        return super().__setattr__(name, value)


@dataclass
class Calendar:
    intervals: dict[str, Interval]
    eras: dict[str, Era]
    data: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, fpath: str):
        with open(fpath, "r") as f:
            cfg = yaml.safe_load(f)
        if "intervals" not in cfg:
            raise ValueError("At minimum an intervals section must be provided")

        # Collate information in string form
        intervals = cfg["intervals"]
        eras = {}
        data = {}
        for key, val in cfg:
            if key == "intervals":
                continue
            if key[0] == "_":
                data[key[1:]] = val
            eras[key] = val

        # Load intervals
        interval_dict = {}
        for name, interval in intervals.items():
            interval_data = {}
            for key, val in interval.get("data", {}).items():
                val = val.split(".")
                if len(val) != 2:
                    raise ValueError("Data must have format X.Y")
                interval_data[key] = data[val[0]][val[1]]
            interval_dict[name] = Interval(
                name,
                interval.get("start", 0),
                interval.get("stop", 20000000000),
                interval.get("tags", []),
                interval_data,
            )

        # Load epochs
        for ename, era in eras.items():
            strict_era = era.get("strict_era", False)
            strict_epoch = era.get("strict_epochs", True)
            epochs = []
            for name, ec in era.items():
                covers = []
                for cname in ec.get("covers", []):
                    iname, slstr = cname.split("[")
                    slstr = slstr[:-1]
                    ival = intervals[iname]
                    slc = slice(
                        *map(
                            lambda x: float(x.strip()) if x.strip() else None,
                            slstr.split(":"),
                        )
                    )
                    covers += [ival[slc]]
                epochs[name] = Epoch(name, tuple(covers), strict_epoch)
            eras[ename] = Era(ename, tuple(epochs), strict_era)

        return Calendar(interval_dict, eras, data)

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import accumulate
from operator import add, mul, or_
from typing import Any, Self

import yaml
from deepdiff import DeepDiff


@dataclass
class Interval:
    """
    Smallest unit of the epoch system.
    Here we have a time interval containing a set of tags and an arbitrary dict of data.
    This can be sliced with standard slicing notation, but the indices are ctime.

    Two `Interval` instances can be combined with the following operators,
    all operators are functionally addition but with differing checks:

    * `|`: naive addition, combines `tags` and `data` and simply returns the most inclusive time range
    * `+`: same as `|` but will error if time ranges are not overlapping
    * `&`: same as  `|` but will error if `tags` or `data` are not the same
    * `*`: same as `+` but will error if `data` is not the same
    * `@`: same as `*` but will error if `tags` is not the same

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
        time_check: bool = False,
        tag_check: bool = False,
        data_check: bool = False,
    ) -> Self:
        if not isinstance(tocombine, Interval):
            raise TypeError(
                f"Can only add intervals to other intervals, not {type(tocombine)}!"
            )
        combined = self
        if in_place:
            combined = deepcopy(self)
        if time_check and (self.start > tocombine.stop or self.stop < tocombine.start):
            raise ValueError(
                "Can't combine without overlapping time ranges with time_check=True"
            )
        combined.start = min(self.start, tocombine.start)
        combined.stop = max(self.stop, tocombine.stop)

        if tag_check and self.tags != tocombine.tags:
            raise ValueError("Can't combine without identical tags with tag_check=True")
        combined.tags = self.tags.union(tocombine.tags)

        diff = DeepDiff(self.data, self.data)
        if data_check and len(diff) != 0:
            raise ValueError(
                "Can't combine without identical data with data_check=True"
            )
        combined.data.update(tocombine.data)

        return combined

    def __or__(self, tocombine) -> Self:
        return self.combine(tocombine, False, False, False, False)

    def __ior__(self, tocombine) -> Self:
        return self.combine(tocombine, True, False, False, False)

    def __add__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, False, False)

    def __iadd__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, False, False)

    def __and__(self, tocombine) -> Self:
        return self.combine(tocombine, False, False, True, True)

    def __iand__(self, tocombine) -> Self:
        return self.combine(tocombine, True, False, True, True)

    def __mul__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, False, True)

    def __imul__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, False, True)

    def __matmul__(self, tocombine) -> Self:
        return self.combine(tocombine, False, True, True, True)

    def __imatmul__(self, tocombine) -> Self:
        return self.combine(tocombine, True, True, True, True)


@dataclass
class Epoch:
    name: str
    covers: tuple[Interval]
    strict: bool
    _internal: Interval = field(init=False)

    def __post_init__(self):
        self._internal = accumulate(self.covers, mul if self.strict else add)

    def __setattr__(self, name, value):
        if name == "covers" or name == "strict":
            self.__post_init__()
        return super().__setattr__(name, value)


@dataclass
class Era:
    name: str
    epochs: tuple[Epoch]
    strict: bool
    _internal: Interval = field(init=False)

    def __post_init__(self):
        self._internal = accumulate(
            [e._internal for e in self.epochs], add if self.strict else or_
        )

    def __setattr__(self, name, value):
        if name == "covers" or name == "strict":
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
            strict = era.get("strict", True)
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
                epochs[name] = Epoch(name, tuple(covers), strict)
            eras[ename] = Era(ename, tuple(epochs), strict)

        return Calendar(interval_dict, eras, data)

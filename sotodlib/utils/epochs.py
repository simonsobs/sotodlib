import re
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import accumulate
from operator import add, and_, mul, or_
from typing import Any, Optional, Self, cast

import yaml
from deepdiff import DeepDiff

# TODO: Add check for a specific piece of data -> existance and strictness
# TODO: Add method to search via tags


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

    def __repr__(self) -> str:
        return f"{self.name}[{self.start}:{self.stop}]"

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return False
        diff = DeepDiff(self.data, other.data)
        return bool(
            (str(self) == str(other)) * (self.tags == other.tags) * (len(diff) == 0)
        )

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

        diff = DeepDiff(self.data, tocombine.data)
        if not same_data and len(diff) != 0:
            raise ValueError(
                "Can't combine without identical data with same_data=False"
            )
        combined.data.update(tocombine.data)

        combined.name = self.name + "+" + tocombine.name

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
    era_name : str
        Name of the era that this epoch belongs to.
    covers: tuple[Interval]
        Collections of overlapping `Invervals` that make up an `Epoch`.
    strict : bool
        If `True` than in addition to being overlapping the `Intervals` must
        also contain the same data.
    """

    name: str
    era_name: str
    covers: tuple[Interval, ...]
    strict: bool
    _internal: Interval = field(init=False)

    def __post_init__(self):
        self._internal = cast(
            Interval, accumulate(self.covers, mul if self.strict else or_)
        )

    def __repr__(self) -> str:
        return (
            self.name
            + "("
            + ",".join([str(ival) for ival in self.covers])
            + ")"
            + ("*" * self.strict)
        )

    def __setattr__(self, name, value):
        if name == "covers" or name == "strict":
            self.__post_init__()
        return super().__setattr__(name, value)

    def __eq__(self, other):
        if not isinstance(other, Epoch):
            return False
        return str(self) == str(other)


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
    epochs: tuple[Epoch, ...]
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
    """
    Class to manage intervals, epochs, and eras in a unified way.
    It is reccomended to modify relationships between these using this class,
    the `Calender` is not aware of direct modifications.
    """

    intervals: dict[str, Interval]
    epochs: dict[str, Epoch]
    eras: dict[str, Era]
    data: dict[str, dict[str, Any]]
    orphan_epochs: list[Epoch]

    def change_strictness(
        self, strictness: bool, era: str, epoch: Optional[str] = None
    ):
        """
        Change the strictness for an `Era` or an `Epoch`.

        Parameters
        ----------
        strictness : bool
            The new strictness.
        era : str
            The name of the era to modify.
        epoch : Optional, default: None
            The name of epoch to modify.
            To modify and era set `epoch` to `None`.
        """
        if epoch is None:
            self.eras[era].strict = strictness
        else:
            self.epochs[f"{era}.{epoch}"].strict = strictness

    def add_epoch(self, era: str, epoch: Epoch):
        """
        Add an `Epoch` to an `Era`.
        If the `Epoch` is in the orphans list it will be removed.

        Parameters
        ----------
        era : str
            The name of the era to modify.
        epoch : Epoch
            The new epoch to add.
        """
        epoch.era_name = era
        epoch_list = list(self.eras[era].epochs) + [epoch]
        self.eras[era].epochs = tuple(epoch_list)
        self.epochs[f"{era}.{epoch.name}"]
        self.epoch_orphans = [e for e in self.epoch_orphans if e != epoch]

    def del_epoch(self, era: str, epoch: str):
        """
        Delete an `Epoch` from an `Era`.
        This will add it to the orphans list

        Parameters
        ----------
        era : str
            The name of the era to modify.
        epoch : str
            The name of the epoch to remove.
        """
        to_remove = self.epochs[f"{era}.{epoch}"]
        self.epoch_orphans += [to_remove]
        self.eras[era].epochs = tuple(
            e for e in self.eras[era].epochs if e != to_remove
        )
        del self.epochs[f"{era}.{epoch}"]

    def add_interval(self, era: str, epoch: str, interval: Interval):
        """
        Add an `Inverval` to an `Epoch`.

        Parameters
        ----------
        era : str
            The name of the era to modify.
        epoch : Epoch
            The name of epoch to modify,
        interval : Interval
            The interval to add.
        """
        self.epochs[f"{era}.{epoch}"].covers = tuple(
            list(self.epochs[f"{era}.{epoch}"].covers) + [interval]
        )
        self.intervals[str(interval)] = interval

    def del_interval(self, era: str, epoch: str, interval: str):
        """
        Delete an `Inverval` from an `Epoch`.

        Parameters
        ----------
        era : str
            The name of the era to modify.
        epoch : Epoch
            The name of epoch to modify,
        interval : str
            The name of interval (with its slice) to delete.
        """
        self.epochs[f"{era}.{epoch}"].covers = tuple(
            i
            for i in self.epochs[f"{era}.{epoch}"].covers
            if i != self.intervals[interval]
        )

    def find_slices(self, interval_name: str) -> list[Interval]:
        """
        Find all slices for an interval.

        Parameters
        ----------
        interval_name : str
            The name of the interval to find slices for.

        Returns
        -------
        slices : list[Interval]
            List of slices with the correct name.
        """
        ikeys = list(self.intervals.keys())
        r = re.compile(f"{interval_name}\\[.*:.*\\]")
        ikeys = filter(r.match, ikeys)
        slices = [self.intervals[ikey] for ikey in ikeys]

        return slices

    def get_membership(self, interval: str) -> list[tuple[Era, Epoch]]:
        """
        Get the `Epochs` than an interval belongs to.
        This does not search oprhaned `Epoch`s.

        Parameters
        ----------
        interval : str
            The name of the interval (with its slice) to search for.

        Returns
        -------
        membership : list[tuple[Era, Epoch]]
            A list a tuples where each element is an `(Era, Epoch)` pair
            where the `Epoch` contains the interval.
        """
        ival = self.intervals[interval]
        membership = []
        for era in self.eras.values():
            for epoch in era.epochs:
                for i in epoch.covers:
                    if i == ival:
                        membership += [(era, epoch)]
                        break
        return membership

    @classmethod
    def load(cls, fpath: str) -> Self:
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
            ival = Interval(
                name,
                interval.get("start", 0),
                interval.get("stop", 20000000000),
                interval.get("tags", []),
                interval_data,
            )
            interval_dict[str(ival)] = ival

        # Load epochs
        epoch_dict = {}
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
                    interval_dict[str(ival)] = ival
                    slc = slice(
                        *map(
                            lambda x: float(x.strip()) if x.strip() else None,
                            slstr.split(":"),
                        )
                    )
                    covers += [ival[slc]]
                epoch = Epoch(name, ename, tuple(covers), strict_epoch)
                epochs += [epoch]
                epoch_dict[f"{ename}.{name}"] = epoch
            eras[ename] = Era(ename, tuple(epochs), strict_era)

        return cls(interval_dict, epoch_dict, eras, data, [])

import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from operator import add, and_, mul, or_
from typing import Any, Literal, Optional, Self, cast, overload

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

    * `|`: Intervals must be overlapping, but data and tags can differ
    * `&`: Allow gaps between intervals, but data and tags can differ
    * `*`: Invervals must be overlapping, data must be the same but tags can differ
    * `+`: Allow gaps between intervals, data must be the same but tags can differ
    * `/`: Invervals must be overlapping, data can differ but tags must be the same
    * `-`: Allow gaps between intervals, data can differ but tags must be the same
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
            if self.start > val.start:
                raise IndexError(
                    f"Start of slice out of range, interval starts at {self.start} but slice starts at {val.start}!"
                )
            to_ret.start = val.start
        if val.stop is not None:
            if self.stop < val.stop:
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

        if same_tag and self.tags != tocombine.tags:
            raise ValueError("Can't combine without identical tags with same_tag=True")
        combined.tags = self.tags.union(tocombine.tags)

        diff = DeepDiff(self.data, tocombine.data)
        if same_data and len(diff) != 0:
            raise ValueError("Can't combine without identical data with same_data=True")
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
            Interval, reduce(mul if self.strict else or_, self.covers)
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
        if (name == "covers" or name == "strict") and "_internal" in self.__dict__:
            self.__post_init__()
        return super().__setattr__(name, value)

    def __eq__(self, other):
        if not isinstance(other, Epoch):
            return False
        return str(self) == str(other)

    def check_data(self, field: str, strict: bool = True) -> bool:
        """
        Check that all `Interval`s in this `Epoch` contain a certain data field.
        Optinally check that they each contain the same value for the field across all intervals.

        Parameters
        ----------
        field : str
            The name of the data field to check for.
        strict : str
            If True then all `Inverval`s within the `Epoch` must share the same
            value for the data field. If `False` they simply need to exist.

        """
        if field not in self._internal.data:
            return False
        if strict:
            tester = self if self.strict else deepcopy(self)
            try:
                tester.strict = True
            except ValueError:
                return False
            return True
        return cast(bool, reduce(and_, [field in ival.data for ival in self.covers]))


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
            reduce(add if self.strict else and_, [e._internal for e in self.epochs]),
        )

    def __setattr__(self, name, value):
        if (name == "epochs" or name == "strict") and "_internal" in self.__dict__:
            self.__post_init__()
        return super().__setattr__(name, value)

    def check_data(self, field: str, strict: bool = True) -> bool:
        """
        Check that all `Epoch`s in this `Era` contain a certain data field.
        Optinally check that all of their `Invervals` also contain the same value for the field.

        Parameters
        ----------
        field : str
            The name of the data field to check for.
        strict : str
            If True then all `Inverval`s within each `Epoch` must share the same
            value for the data field. If `False` they simply need to exist.

        """
        return cast(
            bool, reduce(and_, [e.check_data(field, strict) for e in self.epochs])
        )


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

    @overload
    def find_tagged(self, tag: str, search_in: Literal["eras"]) -> list[Era]:
        ...

    @overload
    def find_tagged(self, tag: str, search_in: Literal["epochs"]) -> list[Epoch]:
        ...

    @overload
    def find_tagged(self, tag: str, search_in: Literal["intervals"]) -> list[Interval]:
        ...

    def find_tagged(
        self, tag: str, search_in: Literal["intervals", "epochs", "eras"]
    ) -> list:
        """
        Return all objects containing a specific tag.
        The `search_in` parameter specifies which type of object to search for.

        Parameters
        ----------
        tag : str
            The tag to search for.
        search_in : Literal["intervals", "eras", "epochs"]
            What type of objects to search for.
            Note that orphaned epochs will not be searched.

        Returns
        ------
        found : list
            A list of obects of type `search_in` containing this tag.
        """
        if search_in == "intervals":
            return [ival for ival in self.intervals.values() if tag in ival.tags]
        elif search_in == "epochs":
            return [
                epoch for epoch in self.epochs.values() if tag in epoch._internal.tags
            ]
        elif search_in == "eras":
            return [era for era in self.eras.values() if tag in era._internal.tags]
        else:
            raise ValueError(f"Invalid search_in: {search_in}")

    @overload
    def find_data_field(self, field: str, search_in: Literal["eras"]) -> list[Era]:
        ...

    @overload
    def find_data_field(self, field: str, search_in: Literal["epochs"]) -> list[Epoch]:
        ...

    @overload
    def find_data_field(
        self, field: str, search_in: Literal["intervals"]
    ) -> list[Interval]:
        ...

    def find_data_field(
        self, field: str, search_in: Literal["intervals", "epochs", "eras"]
    ) -> list:
        """
        Return all objects containing a specific data field.
        The `search_in` parameter specifies which type of object to search for.
        To check that a specific `Era` has a field in all of it's epochs use `Era.check_data`.

        Parameters
        ----------
        field : str
            The name of the field to check
        search_in : Literal["intervals", "eras", "epochs"]
            What type of objects to search for.
            Note that orphaned epochs will not be searched.

        Returns
        ------
        found : list
            A list of obects of type `search_in` containing this data field.
        """
        if search_in == "intervals":
            return [ival for ival in self.intervals.values() if field in ival.data]
        elif search_in == "epochs":
            return [
                epoch for epoch in self.epochs.values() if field in epoch._internal.data
            ]
        elif search_in == "eras":
            return [era for era in self.eras.values() if field in era._internal.data]
        else:
            raise ValueError(f"Invalid search_in: {search_in}")

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
            To modify only the era set `epoch` to `None`.
            To modify all `Epoch`s in an era set to `all`.
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
        for key, val in cfg.items():
            if key == "intervals":
                continue
            if key[0] == "_":
                data[key[1:]] = val
                continue
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
                set(interval.get("tags", [])),
                interval_data,
            )
            interval_dict[str(ival)] = ival
            intervals[name] = ival

        # Load epochs
        epoch_dict = {}
        for ename, era in eras.items():
            strict_era = era.get("strict_era", False)
            strict_epoch = era.get("strict_epochs", True)
            epochs = []
            for name, ec in era.items():
                if name in ["strict_era", "strict_epochs"]:
                    continue
                covers = []
                for cname in ec.get("covers", []):
                    csplt = cname.split("[")
                    if len(csplt) == 1:
                        iname = cname
                        slstr = ":"
                    elif len(csplt) == 2:
                        iname, slstr = csplt
                        slstr = slstr[:-1]
                    else:
                        raise ValueError(f"Invalid cover string {cname}")
                    ival = intervals[iname]
                    interval_dict[str(ival)] = ival
                    slc = slice(
                        *map(
                            lambda x: float(x.strip()) if x.strip() else None,
                            slstr.split(":"),
                        )
                    )
                    covers += [ival[slc]]
                epoch = Epoch(name, ename, tuple(covers), strict=strict_epoch)
                epochs += [epoch]
                epoch_dict[f"{ename}.{name}"] = epoch
            eras[ename] = Era(ename, tuple(epochs), strict_era)

        return cls(interval_dict, epoch_dict, eras, data, [])

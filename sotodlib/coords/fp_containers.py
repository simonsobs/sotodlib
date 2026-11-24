"""
New Focal Plane contianers.
Supports transparent applications of all transforms including pointing model.
Calculated transforms can either be treated at properties for ease of
calculations or as static attributes for loading saved results.
"""

# TODO: Reprs

from __future__ import annotations

import logging
import os
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, reduce
from concurrent.futures import Executor
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Optional,
    Self,
    TypeVar,
    cast,
    overload,
)

import h5py
import megham.transform as mt
import megham.utils as mu
import numpy as np
from jaxtyping import Float, Integer, Shaped, Bool
from scipy.cluster import vq
from scipy.optimize import minimize
from so3g.proj import quat
from sotodlib.coords.pointing_model import apply_pointing_model, param_defaults
from sotodlib.core import AxisManager, IndexAxis, metadata
from sotodlib.core.metadata import ManifestDb, MetadataSpec
from sotodlib.io.metadata import SuperLoader, read_dataset, write_dataset
from sotodlib.utils import epochs

_COORDSYS_COLUMNS = {
    "xieta": ("xi", "eta", "gamma"),
    "horizon": ("az", "el", "roll"),
    "fp": ("x_fp", "y_fp", "pol_fp"),
    "ot": ("x_ot", "y_ot", "pol_ot"),
}

logger = logging.getLogger("solve_static_pointing")

T = TypeVar("T")


def make_manifestdb(system: PointingSystem, field: str, db_path: str, h5_path: str):
    """
    Create a ManifestDB mapping `(obs:timestamp, array) -> template dataset`
    for all templates contained in a PointingSystem.
    Note that if an existing db is on disk this will delete it

    Parameters
    ----------
    system : PointingSystem
        System containing template focal planes.
    field : str
        The field within each saved `FocalPlaneCollection` to point to.
    db_path : str
        Output sqlite database path.
    h5_path : str
        HDF5 file containing the saved templates.
    """
    if os.path.isfile(db_path):
        os.remove(db_path)
    scheme = metadata.ManifestScheme()
    scheme.add_range_match("obs:timestamp")
    scheme.add_exact_match("array")
    scheme.add_data_field("dataset")
    mdb = metadata.ManifestDb(scheme=scheme)
    mdb.to_file(db_path)
    mdb = metadata.ManifestDb(db_path)
    h5_rel = os.path.relpath(
        os.path.abspath(h5_path), os.path.dirname(os.path.abspath(db_path))
    )
    epoch_dict = {ep.name: ep for ep in system.era.epochs}
    seen = set()
    for rx_idx, rx in enumerate(system.receivers):
        for epoch_name in rx.epoch.split("+"):
            epoch = epoch_dict[epoch_name]
            for cover in epoch.covers:
                for ot in rx.optics_tubes:
                    for wafer_slot, fp in zip(
                        ot.wafer_slots,
                        ot.focal_planes,
                    ):
                        dataset = (
                            f"/receivers/"
                            f"{rx_idx}_{rx.name}_{rx.epoch}/"
                            f"{ot.name}/"
                            f"{wafer_slot}-{fp.name}/"
                            f"{field}"
                        )

                        key = (
                            cover.start,
                            cover.stop,
                            fp.name,
                        )
                        if key in seen:
                            raise ValueError(
                                f"Duplicate entry for array={fp.name} and range=({cover.start}, {cover.stop})"
                            )
                        seen.add(key)
                        mdb.add_entry(
                            {
                                "obs:timestamp": (cover.start, cover.stop),
                                "array": fp.name,
                                "dataset": dataset,
                            },
                            filename=h5_rel,
                        )


def make_per_obs_manifestdb(
    system: PointingSystem, field: str, db_path: str, h5_path: str
):
    """
    Create a ManifestDB mapping `(obs:obs_id, array) -> dataset`
    for a field stored on each FocalPlane measurement.

    Parameters
    ----------
    system : PointingSystem
        System containing per-observation measurements.
    field : str
        Field within each saved FocalPlane measurement to point to.
    db_path : str
        Output sqlite database path.
    h5_path : str
        HDF5 file containing the saved system.
    """
    if os.path.isfile(db_path):
        os.remove(db_path)
    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    scheme.add_exact_match("array")
    scheme.add_data_field("dataset")
    mdb = metadata.ManifestDb(scheme=scheme)
    mdb.to_file(db_path)
    mdb = metadata.ManifestDb(db_path)
    h5_rel = os.path.relpath(
        os.path.abspath(h5_path), os.path.dirname(os.path.abspath(db_path))
    )
    seen = set()
    for rx_idx, rx in enumerate(system.receivers):
        for ot in rx.optics_tubes:
            for wafer_slot, fp in zip(ot.wafer_slots, ot.focal_planes):
                for meas_idx, meas in enumerate(fp.focal_planes):
                    dataset = (
                        f"/receivers/"
                        f"{rx_idx}_{rx.name}_{rx.epoch}/"
                        f"{ot.name}/"
                        f"{wafer_slot}-{fp.name}/"
                        f"focal_planes/"
                        f"{meas_idx}_{meas.meas_id}/"
                        f"{field}"
                    )

                    key = (meas.meas_id, fp.name)

                    if key in seen:
                        raise ValueError(
                            f"Duplicate entry for obs_id={meas.meas_id}, array={fp.name}"
                        )
                    seen.add(key)
                    mdb.add_entry(
                        {
                            "obs:obs_id": meas.meas_id,
                            "array": fp.name,
                            "dataset": dataset,
                        },
                        filename=h5_rel,
                    )


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        if v is None:
            logger.warning("Not adding attribute %s because it is None", k)
            continue
        dset.attrs[k] = v


def _create_group(f, path, overwrite=True) -> h5py.Group:
    if path in f:
        if path == "/":
            return f[path]
        if not overwrite:
            raise ValueError(f"HDF5 path already exists: {path}")
        del f[path]
    return f.create_group(path)


def gamma_fit(src, dst):
    """
    Fit the transformation for gamma.
    Note that the periodicity here assumes things are in radians.

    Arguments:

        src: Source gamma in radians

        dst: Destination gamma in radians

    Returns:

       scale: Scale applied to src

       shift: Shift applied to scale*src
    """

    def _gamma_min(pars, src, dst):
        scale, shift = pars
        transformed = np.sin(src * scale + shift)
        diff = np.sin(dst) - transformed

        return np.sqrt(np.mean(diff**2))

    res = minimize(_gamma_min, (1.0, 0.0), (src, dst))
    return res.x


def _transformed(
    self: FocalPlane | FocalPlaneCollection, attr_name: str, transform: Transform
):
    to_ret = deepcopy(self.template)
    to_ret.xyz = mt.apply_transform(
        self.template.xyz, transform.affine, transform.shift
    )
    to_ret.center = None
    if self.template.center is not None:
        to_ret.center = mt.apply_transform(
            np.atleast_2d(self.template.center), transform.affine, transform.shift
        ).ravel()
    if self.autofreeze:
        setattr(self, attr_name, to_ret)
    return to_ret


def _get_aff(
    self: FocalPlane | FocalPlaneCollection,
    data_name: str,
    template_name: str,
    weights_name: str,
    attr_name: str,
    apply_wmsk: bool = True,
):
    to_ret = Transform.identity()
    wmsk = self.wmsk
    if np.sum(wmsk) < 4:
        return to_ret
    if apply_wmsk:
        msk = wmsk
    else:
        msk = slice(None)
    affine, shift = mt.get_affine_two_stage(
        getattr(self, template_name).xyz[msk, :2],
        getattr(self, data_name).xyz[msk, :2],
        getattr(self, weights_name)[msk],
    )
    to_ret.shift[:2] = shift
    to_ret.affine[:2, :2] = affine
    if not self.fake_gamma:
        to_ret.affine[-1, -1], to_ret.shift[-1] = gamma_fit(
            getattr(self, template_name).xyz[msk, 2],
            getattr(self, data_name).xyz[msk, 2],
        )
    if self.autofreeze:
        setattr(self, attr_name, to_ret)
    return to_ret


def setup_fit_params(
    system: PointingSystem, cfg: Namespace, old_system: Optional[PointingSystem]
) -> tuple[np.ndarray, list[tuple[float, float]], np.ndarray]:
    """
    Set up fit parameters, including nuisance parameters.

    Parameters
    ----------
    system : PointingSystem
        Pointing system containing the non-nuisance parameters.
    cfg : Namespace
        Configuration containing parameter initial values, bounds and static flags.
    old_system : Optional[PointingSystem]
        If not `None` then initial static and nuisance params/transforms
        will be pulled from this depending on if `cfg.pars_from_old`,
        `cfg.nuisance_from_old`, and `cfg.nuisance_transform_from_old` are True.
        For static params this can be overwridden by setting `overwrite_old = True`
        in the parameter config dict.

    Returns
    -------
    params : np.ndarray
        Initial fit parameter values, including nuisance parameters.
    bounds : list[tuple[float, float]]
        Parameter bounds, including nuisance parameter bounds.
    scales : np.ndarray
        Parameter scales, including nuisance parameter scales.
    """
    old_names = []
    if old_system is not None:
        old_names = old_system.parameter_names

    # Setup non-nuisance fit parameters
    params = system.parameters.copy()
    bounds = [(-np.inf, np.inf)] * len(params)
    scales = np.ones(len(params))

    for name, par in cfg.parameters.items():
        if name not in system.parameter_names:
            logger.warning("Unknown parameter %s", name)
            continue
        idx = system.parameter_names.index(name)
        params[idx] = float(par.get("init", params[idx]))
        if par.get("static", False):
            bounds[idx] = (params[idx], params[idx])
            continue
        bounds[idx] = tuple(float(b) for b in par.get("bounds", bounds[idx]))[:2]  # type: ignore
        lo, hi = bounds[idx]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            scales[idx] = hi - lo
        if (
            cfg.pars_from_old
            and not par.get("overwrite_old", False)
            and name in old_names
            and old_system is not None
        ):
            par[idx] = old_system.parameters[old_names.index(name)]

    # Setup nuisance fit parameters
    nn = len(system.nuisance_params.params_flat)
    nuisance_values = np.zeros(nn)
    nuisance_bounds = [(-np.inf, np.inf) for _ in range(nn)]
    nuisance_scales = np.ones(nn)
    for meas_id, nidx in system.nuisance_params.obs_mapping.items():
        if old_system is not None and meas_id in old_system.nuisance_params.obs_mapping:
            oidx = old_system.nuisance_params.obs_mapping[meas_id]
            if cfg.nuisance_from_old:
                system.nuisance_params.par_list[nidx] = (
                    old_system.nuisance_params.par_list[oidx]
                )
            if cfg.nuisance_transform_from_old:
                system.nuisance_params.transform_list[nidx] = (
                    old_system.nuisance_params.transform_list[oidx]
                )
        par_dict = system.nuisance_params.par_list[nidx]
        for name, value in par_dict.items():
            if name not in system.parameter_names:
                logger.warning(
                    "No equivalent non-nuisance parameter found for nuisance parameter %s",
                    name,
                )
                nuisance_values[nidx] = value
                nuisance_bounds[nidx] = (-np.inf, np.inf)
                nuisance_scales[nidx] = 1.0
                continue
            idx = system.parameter_names.index(name)
            nuisance_values[nidx] = value
            lo, hi = bounds[idx]
            nuisance_bounds[nidx] = (value - (value - lo) / cfg.nuisance_scale, value + (hi - value) / cfg.nuisance_scale)
            nuisance_scales[nidx] = scales[idx]

    # Append nuisance parameters
    params = np.concatenate([params, np.asarray(nuisance_values)])
    bounds.extend(nuisance_bounds)
    scales = np.concatenate([scales, np.asarray(nuisance_scales)])

    system.update_parameters(params)

    return params, bounds, scales

def _get_pm(meas):
    return meas._pm_data_dyn

class StaticOrCached(Generic[T]):
    """
    Wrapper that will get either:

    1. precomputed static attribute: `_{name}_stat`
    2. dynamic property computation: `_{name}_dyn`

    dependint on if `static` is true or false.
    """

    def __init__(self, name: str):
        self.name = name

    def __set_name__(self, owner, attr_name):
        self.attr_name = attr_name

    @overload
    def __get__(self, obj: None, owner: type) -> "StaticOrCached[T]": ...

    @overload
    def __get__(self, obj: Any, owner: type | None = None) -> T: ...

    def __get__(self, obj: Any, owner: type | None = None) -> T | "StaticOrCached[T]":
        if obj is None:
            return self

        if obj.static:
            if f"_{self.name}_stat" in obj.__dict__:
                res = getattr(obj, f"_{self.name}_stat")
                if res is not None:
                    return res
            res = getattr(obj, f"_{self.name}_dyn")
            setattr(obj, f"_{self.name}_stat", res)
            return res
        return getattr(obj, f"_{self.name}_dyn")


@dataclass
class NuisanceParams:
    """
    Map and store of per observation nuisance parameters and transforms.
    Can be indexed directly by `obs_id`.

    Attributes
    ----------
    obs_mapping : dict[str, int]
        Mapping from `obs_ids` to the matching
        element in `par_list`.
    par_list : list[dict[str, float]]
        Each row is a set of nuicance params
        for one observation, but `obs_mapping`
        can map multiple `obs_id`s to one row here
        to account for LAT OTs having seperate `obs_id`s.
    transform_list : list[Transform]
        Each element is a nuisance transform that can be applied when computing `pm_data`.
        In the same order as par_list.
    params_flat : Float[np.ndarray, "npar"]
        (Property) Flattened array of all nuisance parameter values, concatenated
        in `par_list` order and preserving the insertion order of each dictionary.
    """

    obs_mapping: dict[str, int]
    par_list: list[dict[str, float]]
    transform_list: list[Transform]

    def __getitem__(self, key) -> tuple[dict[str, float], Transform]:
        if key not in self.obs_mapping:
            return {}, Transform.identity()
        idx = self.obs_mapping[key]
        return self.par_list[idx], self.transform_list[idx]

    @property
    def params_flat(self) -> Float[np.ndarray, "npar"]:
        return np.array(
            [value for par_dict in self.par_list for value in par_dict.values()],
            dtype=float,
        )

    def update(self, params: Float[np.ndarray, "npar"]):
        """
        Update nuisance parameters from a flattened parameter array.

        Parameters
        ----------
        params : Float[np.ndarray, "npar"]
            Flattened nuisance parameter values, ordered according to
            `par_list` and the insertion order of each dictionary.
        """
        idx = 0
        for par_dict in self.par_list:
            for name in par_dict:
                par_dict[name] = float(params[idx])
                idx += 1

    @classmethod
    def from_system(
        cls, system: PointingSystem, nuisance_pars: dict[str, float], threshold: float
    ) -> NuisanceParams:
        """
        Construct nuisance parameters and transforms from a pointing system.

        Observations are grouped into epochs using their measurement IDs
        and timestamps. Each epoch is assigned a copy of `nuisance_pars`
        and an identity transform. Multiple observation IDs belonging
        to the same epoch map to the same nuisance parameter set.

        Parameters
        ----------
        system : PointingSystem
            Pointing system from which to collect observation IDs.
        nuisance_pars : dict[str, float]
            Initial nuisance parameter values assigned to each epoch.
        threshold : float
            Threshold passed to `epochs.group_obs` when grouping
            observations into epochs.

        Returns
        -------
        NuisanceParams
            Nuisance parameter mapping, parameter sets, and identity
            transforms corresponding to the observation epochs.
        """
        obs_ids = []
        timestamps = []
        for rx in system.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    for meas in fp.focal_planes:
                        obs_ids += [meas.meas_id]
                        timestamps += [float(meas.meas_id.split("_")[1])]
        obs_ids_split, _ = epochs.group_obs(
            np.asarray(obs_ids), np.asarray(timestamps), threshold
        )
        obs_mapping = {}
        for i, obs_list in enumerate(obs_ids_split):
            obs_mapping.update({obs_id: i for obs_id in obs_list})
        par_list = [nuisance_pars.copy() for _ in range(len(obs_ids_split))]
        transform_list = [Transform.identity() for _ in range(len(obs_ids_split))]
        return cls(obs_mapping, par_list, transform_list)

    @property
    def rev_mapping(self):
        rev_mapping = [[] for _ in range(len(self.transform_list))]
        for meas_id, idx in self.obs_mapping.items():
            rev_mapping[idx] += [meas_id]
        return rev_mapping

    def compute_transforms(self, system: PointingSystem):
        r"""
        Take in a system after a round of fitting and compute the nuisance transforms needed
        to make a clean focal plane. If there are existing nuisance transforms we assume that
        they were used to compute the current `pm_data` values and will be included in the net transform.

        We want to compute the transforms that reduce the obs-obs variance on `pm_data` without changing
        the average boresight pointing. To start lets assume we have `pm_data` and `pm_transform` for each `FocalPlane`
        and `cm_transform` for each `Receiver`. This we then compute `obs_transform` which is the common mode of `pm_transform`
        for all `FocalPlane`s in a given observation.
        The transform that then lines up `pm_data` along obs (relative to the boresight) is:

        $$
        T_{\mathrm{nuisance}} = \left(T_{\mathrm{obs}} \circ T_{\mathrm{cm}}^{-1}\right)^{-1}
        $$

        if we have some existing `nuisance_transform` then it was already included in

        $$
        \texttt{pm\_data} = T_{\mathrm{nuisance,old}}(\texttt{pointing\_model(data)})
        $$

        so the new nuisance transform should chain them as

        $$
        T_{\mathrm{nuisance}} = T_{\mathrm{nuisance,new}} \circ T_{\mathrm{nuisance,old}}
        $$

        giving us:

        $$
        \begin{aligned}
        \texttt{pm\_data}
        &= T_{\mathrm{nuisance}}(\texttt{pointing\_model(data)}) \\
        &= (T_{\mathrm{nuisance,new}} \circ T_{\mathrm{nuisance,old}})
        (\texttt{pointing\_model(data)}) \\
        &= T_{\mathrm{nuisance,new}}(\texttt{pm\_data\_old})
        \end{aligned}
        $$

        This function does assume that all `FocalPlane`s in an obs are in the same `Receiver`.
        If this is not true you set up some config wrong and should figure it out.

        Parameters
        ----------
        system : PointingSystem
            System after a round of fitting, containing the focal plane pointing
            transforms (`pm_transform`) and receiver common-mode transforms
            (`cm_transform`).


        Raises
        ------
        ValueError
            If there are obs split across multiple receivers or the total
            number of focal planes in the system does not match the mapping.
        """
        rev_mapping = self.rev_mapping
        done_idx = set()
        for rx in system.receivers:
            rx_aff_inv, rx_sft_inv = mt.invert_transform(
                rx.cm_transform.affine, rx.cm_transform.shift
            )
            affs = {}
            sfts = {}
            rx_ids = []
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    for meas in fp.focal_planes:
                        affs[meas.meas_id] = meas.pm_transform.affine
                        sfts[meas.meas_id] = meas.pm_transform.shift
                        rx_ids += [meas.meas_id]
            rx_ids = np.array(rx_ids)
            for i, meas_ids in enumerate(rev_mapping):
                if i in done_idx:
                    continue
                inrx = np.isin(meas_ids, rx_ids)
                if not np.all(inrx):
                    if np.any(inrx):
                        raise ValueError(
                            f"Somehow obs_ids {meas_ids} are not all in the same Receiver!"
                        )
                    continue
                cm_aff, cm_shift = mt.get_common_mode(
                    [affs[mid] for mid in meas_ids],
                    [sfts[mid] for mid in meas_ids],
                    -2,
                    True,
                    fast=False, #True,
                )
                n_aff, n_sft = mt.invert_transform( # cm_aff, cm_shift)
                    *mt.compose_transform(rx_aff_inv, rx_sft_inv, cm_aff, cm_shift)
                )
                trf_old = self.transform_list[i]
                n_aff, n_sft = mt.compose_transform(
                    trf_old.affine, trf_old.shift, n_aff, n_sft
                )
                self.transform_list[i] = Transform(n_sft, n_aff)
                done_idx.add(i)
        if len(done_idx) != len(self.transform_list):
            raise ValueError("Not all obs were completed! Check your receiver objects!")

    def save(self, f: h5py.File, path: str, overwrite: bool = True):
        """
        Save the nuisance params to an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            HDF5 file object to save to.
        path : str
            Path within the HDF5 file at which to save the pointing model.
        overwrite : bool
            If True, overwrite an existing pointing model at `path`.
        """
        obs_ids = np.array(list(self.obs_mapping.keys()))
        par_idx = np.array(list(self.obs_mapping.values()))
        dtype = [("obs:obs_id", obs_ids.dtype), ("par_idx", par_idx.dtype)]
        data = np.empty(len(obs_ids), dtype=dtype)
        data["obs:obs_id"] = obs_ids
        data["par_idx"] = par_idx
        write_dataset(
            metadata.ResultSet.from_friend(data),
            f,
            os.path.join(path, "obs_mapping"),
            overwrite=overwrite,
        )

        npar = len(self.par_list[0]) if self.par_list else 0
        par_idx = np.repeat(np.arange(len(self.par_list)), npar)
        par_name = np.array([name for pars in self.par_list for name in pars.keys()])
        par_value = np.array(
            [value for pars in self.par_list for value in pars.values()]
        )
        dtype = [
            ("par_idx", par_idx.dtype),
            ("par_name", par_name.dtype),
            ("par_value", par_value.dtype),
        ]
        data = np.empty(len(par_idx), dtype=dtype)
        data["par_idx"] = par_idx
        data["par_name"] = par_name
        data["par_value"] = par_value
        write_dataset(
            metadata.ResultSet.from_friend(data),
            f,
            os.path.join(path, "par_list"),
            overwrite=overwrite,
        )

        _ = _create_group(f, os.path.join(path, "transform_list"), overwrite)
        for i, trf in enumerate(self.transform_list):
            trf.save(f, os.path.join(path, "transform_list", f"{i}"), overwrite)

    @classmethod
    def load(cls, f: h5py.File, path: str) -> Self:
        """
        Load nuisance parameters from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            HDF5 file object containing the nuisance parameters.
        path : str
            Path within the HDF5 file from which to load the nuisance parameters.

        Returns
        -------
        NuisanceParams
            The loaded nuisance params.
        """
        data = read_dataset(f, os.path.join(path, "obs_mapping")).asarray()
        obs_mapping: dict[str, int] = {
            str(row["obs:obs_id"]): int(row["par_idx"]) for row in data
        }

        data = read_dataset(f, os.path.join(path, "par_list")).asarray()
        npar = int(np.max(data["par_idx"])) + 1 if len(data) else 0
        par_list: list[dict[str, float]] = [{} for _ in range(npar)]
        for row in data:
            name = row["par_name"]
            par_list[int(row["par_idx"])][str(name)] = float(row["par_value"])

        if os.path.join(path, "transform_list") in f:
            transform_list = [
                Transform.load(f, os.path.join(path, "transform_list", f"{i}"))
                for i in range(len(par_list))
            ]
        else:
            transform_list = [Transform.identity() for _ in range(len(par_list))]

        return cls(obs_mapping, par_list, transform_list)


@dataclass
class PointingModel:
    """
    Wrapper around a telescope pointing model and its application to
    detector offsets. In the future this should also be able to handle
    measurements from things like the starcamera.

    Attributes
    ----------
    epoch : str
        Human-readable identifier for the epoch associated with this
        data. If this spans multiple epochs they will be seperated with a '+'.
    parameters : dict
        Dictionary of pointing-model parameters accepted by
        `apply_pointing_model`. An empty dictionary corresponds to an
        identity pointing model and leaves detector offsets unchanged.
    force_zero_roll : bool
        If True, ignore the encoder roll angle when applying the pointing
        model and assume a zero-roll boresight solution.
    pad : bool, default: False
        If True, this model is excluded from fitting.
    """

    epoch: str
    parameters: dict
    nuisance_params: NuisanceParams
    force_zero_roll: bool
    pad: bool = False

    def apply_det(
        self,
        meas: FocalPlane,
        tol: float = 2.42e-6,
        max_iter: int = 5,
    ) -> DetectorOffsets:
        """
        Apply the inverse pointing model correction to a set of detector offsets.

        The correction is computed from the supplied encoder values using
        `apply_pointing_model` and then propagated to each detector through
        quaternion rotations. The returned detector offsets remain in the
        `xieta` coordinate system but have the pointing model corrections removed.

        Focal plane distortions are removed iteratively by applying the
        forward pointing model, estimating the resulting distortion, and
        subtracting it from the detector offsets until convergence.

        Nuisance parameters associated with the measurement are added to
        the pointing model parameters before computing the correction.
        The nuisance transform is applied to the detector offsets before returning.

        If the pointing model contains no parameters, or if no detectors
        are available for the inversion, the input detector offsets are
        returned unchanged.

        When `self.force_zero_roll=True`, the encoder roll angle is ignored
        during the correction and a zero-roll boresight solution is assumed.

        When `meas.fake_gamma=True`, detector polarization angles are ignored
        and only detector positions (`xi`, `eta`) are propagated during the
        iterative inversion. The returned detector offsets still contain
        all three coordinates.

        Parameters
        ----------
        meas : FocalPlane
            Focal plane measurement containing detector offsets, encoder
            values, detector weights, and associated metadata. Only
            detectors selected by `meas.wmsk` are included in the inversion.
        tol : float, default: 2.42e-6
            RMS convergence tolerance for the iterative distortion removal,
            in radians.
        max_iter : int, default: 5
            Maximum number of iterations for the distortion removal.

        Returns
        -------
        DetectorOffsets
            A copy of `meas.data` with the pointing-model correction removed.
            The returned object has the same metadata and detector ordering
            as the input but contains corrected detector coordinates.

        Raises
        ------
        RuntimeError
            If the iterative distortion removal does not converge within
            `max_iter` iterations.
        """
        to_ret = deepcopy(meas.data)
        if len(self.parameters) == 0:
            return to_ret

        # Apply nuisance
        params_use = deepcopy(self.parameters)
        npars, ntrf = self.nuisance_params[meas.meas_id]
        for key, val in npars.items():
            params_use[key] += val

        # Start with the input detector coordinates
        if not np.any(meas.wmsk):
            return to_ret
        estimate = meas.data_msk.xyz.copy()
        if meas.fake_gamma:
            estimate = estimate[:, :2]
        fp_target = meas.data_msk
        fp = meas.data_msk.fp
        det_enc_msk = meas.enc_msk
        ancil = det_enc_msk.ancil
        if self.force_zero_roll:
            q_enc = det_enc_msk.q_enc_zero_roll
        else:
            q_enc = det_enc_msk.q_enc

        err = np.inf
        prev_err = np.inf
        alpha = 1.0
        prev_distortion = np.zeros_like(estimate)
        
        for _ in range(max_iter):
            fp.xi = estimate[:, 0]
            fp.eta = estimate[:, 1]
        
            if meas.fake_gamma:
                q_fp = quat.rotation_xieta(estimate[:, 0], estimate[:, 1])
            else:
                fp.gamma = estimate[:, 2]
                q_fp = quat.rotation_xieta(
                        estimate[:, 0], estimate[:, 1], estimate[:, 2] # type: ignore
                )
        
            bs, fp_corrected = apply_pointing_model( None, params_use, ancil, fp)
            bs = cast(AxisManager, bs)
        
            xi, eta, gamma = quat.decompose_xieta(
                ~quat.euler(2, bs.roll)
                * ~quat.rotation_lonlat(-1 * np.asarray(bs.az), bs.el)
                * q_enc
                * q_fp
            )

            if fp_corrected is None:
                to_ret.xyz[meas.wmsk] = np.column_stack((xi, eta, gamma))
                to_ret.xyz = mt.apply_transform( to_ret.xyz, ntrf.affine, ntrf.shift)
                return to_ret
        
            if meas.fake_gamma:
                dx = fp_corrected.xi - fp.xi
                dy = fp_corrected.eta - fp.eta
        
                distortion = np.column_stack((dx, dy))
                resid = np.column_stack(( fp_corrected.xi - fp_target.xi, fp_corrected.eta - fp_target.eta,))
            else:
                dx = fp_corrected.xi - fp.xi
                dy = fp_corrected.eta - fp.eta
                dg = fp_corrected.gamma - fp.gamma
        
                distortion = np.column_stack((dx, dy, dg))
                resid = np.column_stack(( fp_corrected.xi - fp_target.xi, fp_corrected.eta - fp_target.eta, fp_corrected.gamma - fp_target.gamma,))
        
            err = np.sqrt(np.nanmean(np.sum(resid * resid, axis=1)))
        
            if err > prev_err:
                alpha *= 0.5
                distortion = prev_distortion
            else:
                alpha = min(1.0, alpha * 1.2)
                prev_distortion = distortion.copy()
                prev_err = err
        
            if err < tol:
                to_ret.xyz[meas.wmsk] = np.column_stack((xi, eta, gamma))
                to_ret.xyz = mt.apply_transform(
                    to_ret.xyz, ntrf.affine, ntrf.shift
                )
                return to_ret
        
            estimate -= alpha * distortion

        raise RuntimeError(
            f"Pointing-model inversion did not converge after {max_iter} iterations (error={err:.3e} rad)."
        )

    def save(self, f: h5py.File, path: str, overwrite: bool = True):
        """
        Save the pointing model to an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            HDF5 file object to save to.
        path : str
            Path within the HDF5 file at which to save the pointing model.
        overwrite : bool
            If True, overwrite an existing pointing model at `path`.
        """
        group = _create_group(f, path, overwrite)
        _add_attrs(
            group,
            {
                "epoch": self.epoch,
                "force_zero_roll": self.force_zero_roll,
                "pad": self.pad,
            },
        )

        parameters = AxisManager()
        for name, value in self.parameters.items():
            parameters.wrap(name, value)
        parameters.save(f, f"{path}/parameters")

    @classmethod
    def load(cls, f: h5py.File, path: str, nuisance_params: NuisanceParams) -> Self:
        """
        Load a pointing model from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            HDF5 file object containing the pointing model.
        path : str
            Path within the HDF5 file from which to load the pointing model.
        nuisance_params : NuisanceParams
            Nuisance parameters and transforms to associate with the
            loaded pointing model.

        Returns
        -------
        PointingModel
            The loaded pointing model.
        """
        group = cast(h5py.Group, f[path])
        parameters = AxisManager.load(group["parameters"])
        parameters = {name: parameters[name] for name in parameters._fields}

        return cls(
            epoch=cast(str, group.attrs["epoch"]),
            parameters=parameters,
            nuisance_params=nuisance_params,
            force_zero_roll=cast(bool, group.attrs["force_zero_roll"]),
            pad=cast(bool, group.attrs.get("pad", False)),
        )


@dataclass
class Transform:
    """
    Class to hold and decompose an affine transform.

    Attributes
    ----------
    shift : Float[np.ndarray, "ndim"]
        The shift to apply as part of the transform
    affine : Float[np.ndarray, "ndim ndim"]
        The affine transformation matrix.
    scale : Float[np.ndarray, "ndim"]
        (Property) The scale along each axis.
    shear : float
        (Property) The shear parameter of the `xieta` portion of the
        affine transformation.
    rot : float
        (Property) The rotation angle of the `xieta` plane.
    decompose : tuple[Float[np.ndarray, "ndim"], float, float]
        (Cached property) Decomposition of the affine transformation
        into scale, shear, and rotation. Cache is cleared if `shift`
        or `affine` is cleared.
    """

    shift: Float[np.ndarray, "ndim"]
    affine: Float[np.ndarray, "ndim ndim"]

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ("shift", "affine"):
            self.__dict__.pop("decompose", None)
        return super().__setattr__(name, value)

    @classmethod
    def identity(cls):
        return Transform(np.zeros(3), np.eye(3))

    @classmethod
    def from_split(cls, shift, xieta_affine, gamma_scale):
        affine = np.eye(len(xieta_affine) + 1)
        affine[: len(xieta_affine), : len(xieta_affine)] = xieta_affine
        affine[-1, -1] = gamma_scale

        return Transform(shift, affine)

    @cached_property
    def decompose(self) -> tuple[Float[np.ndarray, "ndim"], float, float]:
        """
        Decompose into scale, shear, and rotation matrices.

        Returns
        -------
        scale : Float["ndim"]
            The scale in each axis.
        shear : float
            The shear parameter of only the xieta portion.
        rot : float
            The rotation of the xieta plane.
        """
        xieta_affine = self.affine[:2, :2]
        gamma_scale = self.affine[-1, -1]
        scale, shear, rot = mt.decompose_affine(xieta_affine)
        scale = np.array((*scale, gamma_scale))
        shear = shear.item()
        rot = mt.decompose_rotation(rot)[-1]

        return scale, shear, rot

    @property
    def scale(self):
        return self.decompose[0]

    @property
    def shear(self):
        return self.decompose[1]

    @property
    def rot(self):
        return self.decompose[2]

    def save(self, f: h5py.File, path: str, overwrite: bool = True):
        """
        Save the transform to an HDF5 group.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file.
        path : str
            Path within the HDF5 file where the transform should be stored.
        overwrite : bool, default: True
            If True overwrite any existing h5groups.
        """
        group = _create_group(f, path, overwrite=overwrite)
        group.create_dataset("shift", data=self.shift)
        group.create_dataset("affine", data=self.affine)

    @classmethod
    def load(cls, f: h5py.File, path: str) -> Self:
        """
        Load a transform from an HDF5 group.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file.
        path : str
            Path within the HDF5 file containing the transform.

        Returns
        -------
        Transform
            Loaded transform.
        """
        group = cast(h5py.Group, f[path])
        shift = np.array(group["shift"])
        affine = np.array(group["affine"])

        return cls(shift=shift, affine=affine)


@dataclass
class DetectorOffsets:
    """
    Class for storing detector offsets in `xieta`, `horizon`, `fp`, or `ot`
    coordinates. The `xi`, `eta`, `gamma`, `az`, `el`, and `roll` properties
    provide convenient access to individual coordinates and raise a
    `ValueError` if accessed when `coordsys` is incompatible.

    Cached properties are cleared when `xyz` is assigned to.

    Attributes
    ----------
    xyz : Float[np.ndarray, "ndet 3"]
        Detector coordinates. The meaning of each column depends on
        `coordsys`:

        - `xieta`: (`xi`, `eta`, `gamma`).
        - `horizon`: (`az`, `el`, `roll`).
        - `fp` and `ot`: Coordinates in the corresponding coordinate system.
    det_id : Shaped[np.ndarray, "ndet"]
        Detector ID associated with each row of `xyz`.
    split : Shaped[np.ndarray, "ndet"]
        String used to group detectors when plotting. Nominally, this
        separates the four detectors behind the same feedhorn.
    center : Optional[Float[np.ndarray, "3"]]
        Center of the detector offsets, or `None` if unknown.
    coordsys : str
        Coordinate system of the detector offsets. Must be one of
        `xieta`, `horizon`, `fp`, or `ot`.
    xi : Float[np.ndarray, "ndet"]
        (Property) First coordinate of `xyz` when `coordsys="xieta"`.
    eta : Float[np.ndarray, "ndet"]
        (Property) Second coordinate of `xyz` when `coordsys="xieta"`.
    gamma : Float[np.ndarray, "ndet"]
        (Property) Third coordinate of `xyz` when `coordsys="xieta"`.
    az : Float[np.ndarray, "ndet"]
        (Property) First coordinate of `xyz` when `coordsys="horizon"`.
    el : Float[np.ndarray, "ndet"]
        (Property) Second coordinate of `xyz` when `coordsys="horizon"`.
    roll : Float[np.ndarray, "ndet"]
        (Property) Third coordinate of `xyz` when `coordsys="horizon"`.
    ancil : AxisManager
        (Cached property) Encoder ancillary data constructed from the
        horizon coordinates. Contains `az_enc`, `el_enc`, `roll_enc`,
        and `boresight_enc`, with angles converted to degrees.
    fp : AxisManager
        (Cached property) Focal plane coordinates containing `xi`, `eta`,
        and `gamma`.
    q_enc : Quaternion
        (Cached property) Quaternion representing encoder pointing,
        constructed from azimuth, elevation, and roll.
    q_enc_zero_roll : Quaternion
        (Cached property) Quaternion representing encoder pointing
        with roll set to zero.
    """

    xyz: Float[np.ndarray, "ndet 3"]
    det_id: Shaped[np.ndarray, "ndet"]
    split: Shaped[np.ndarray, "ndet"]
    center: Optional[Float[np.ndarray, "3"]]
    coordsys: str

    def __post_init__(self):
        if self.coordsys not in ("xieta", "horizon", "fp", "ot"):
            raise ValueError(f"Invalid coordsys {self.coordsys}")

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ("xyz"):
            self.__dict__.pop("ancil", None)
            self.__dict__.pop("fp", None)
            self.__dict__.pop("q_enc", None)
            self.__dict__.pop("q_enc_zero_roll", None)
        return super().__setattr__(name, value)

    def __getitem__(self, key) -> DetectorOffsets:
        return DetectorOffsets(
            xyz=self.xyz[key],
            det_id=self.det_id[key],
            split=self.split[key],
            center=self.center,
            coordsys=self.coordsys,
        )

    @classmethod
    def empty(cls, coordsys):
        xyz = np.zeros((0, 3))
        det_id = np.zeros(0, dtype=str)
        split = np.zeros(0, dtype=str)
        center = None
        return cls(xyz, det_id, split, center, coordsys)

    @property
    def xi(self):
        if self.coordsys != "xieta":
            raise ValueError("Must be in a xieta coordsys to get xi")
        return self.xyz[:, 0]

    @property
    def eta(self):
        if self.coordsys != "xieta":
            raise ValueError("Must be in a xieta coordsys to get eta")
        return self.xyz[:, 1]

    @property
    def gamma(self):
        if self.coordsys != "xieta":
            raise ValueError("Must be in a xieta coordsys to get gamma")
        return self.xyz[:, 2]

    @property
    def az(self):
        if self.coordsys != "horizon":
            raise ValueError("Must be in a horizon coordsys to get az")
        return self.xyz[:, 0]

    @property
    def el(self):
        if self.coordsys != "horizon":
            raise ValueError("Must be in a horizon coordsys to get el")
        return self.xyz[:, 1]

    @property
    def roll(self):
        if self.coordsys != "horizon":
            raise ValueError("Must be in a horizon coordsys to get roll")
        return self.xyz[:, 2]

    @cached_property
    def ancil(self):
        ancil = AxisManager(IndexAxis("samps", len(self.xyz)))
        ancil.wrap("az_enc", np.rad2deg(self.az))
        ancil.wrap("el_enc", np.rad2deg(self.el))
        ancil.wrap("roll_enc", np.rad2deg(self.roll))
        ancil.wrap("boresight_enc", -1 * np.rad2deg(self.roll))  # for SATs
        return ancil

    @cached_property
    def fp(self):
        fp = AxisManager(IndexAxis("samps", len(self.xyz)))
        fp.wrap("xi", self.xi)
        fp.wrap("eta", self.eta)
        fp.wrap("gamma", self.gamma)
        return fp

    @cached_property
    def q_enc(self):
        return quat.rotation_lonlat(-1 * self.az, self.el) * quat.euler(2, self.roll)

    @cached_property
    def q_enc_zero_roll(self):
        return quat.rotation_lonlat(-1 * self.az, self.el) * quat.euler(
            2, 0 * self.roll
        )

    def save(self, f: h5py.File, path: str, overwrite: bool = True):
        """Save detector offsets to an HDF5 ResultSet.

        The ResultSet contains one row per detector with columns for the
        detector ID, coordinate values, and split. Coordinate column names
        are determined by `coordsys`: `xi`, `eta`, `gamma` for
        `xieta` and `az`, `el`, `roll` for `horizon`.

        The coordinate system and center are stored as HDF5 attributes.

        Parameters
        ----------
        f : h5py.File
            HDF5 file to save to.
        path : str
            Path within the HDF5 file at which to save the ResultSet.
        overwrite : bool, default: True
            Whether to overwrite an existing dataset at `path`.
        """
        columns = _COORDSYS_COLUMNS[self.coordsys]
        dtype = [
            ("dets:det_id", self.det_id.dtype),
            *((name, self.xyz.dtype) for name in columns),
            ("split", self.split.dtype),
        ]

        data = np.empty(len(self.det_id), dtype=dtype)
        data["dets:det_id"] = self.det_id
        data[columns[0]] = self.xyz[:, 0]
        data[columns[1]] = self.xyz[:, 1]
        data[columns[2]] = self.xyz[:, 2]
        data["split"] = self.split

        write_dataset(
            metadata.ResultSet.from_friend(data),
            f,
            path,
            overwrite=overwrite,
        )
        _add_attrs(
            f[path],
            {
                "coordsys": self.coordsys,
                "center": self.center,
            },
        )

    @classmethod
    def load(cls, f: h5py.File, path: str) -> Self:
        """Load detector offsets from an HDF5 ResultSet.

        Parameters
        ----------
        f : h5py.File
            HDF5 file containing the detector offsets.
        path : str
            Path within the HDF5 file containing the ResultSet.

        Returns
        -------
        DetectorOffsets
            Detector offsets loaded from the ResultSet.
        """
        rs = read_dataset(f, path)
        coordsys = cast(str, f[path].attrs["coordsys"])
        columns = _COORDSYS_COLUMNS[coordsys]
        xyz = np.column_stack(
            [
                np.asarray(rs[columns[0]]),
                np.asarray(rs[columns[1]]),
                np.asarray(rs[columns[2]]),
            ]
        )
        center = (
            np.asarray(f[path].attrs["center"]) if "center" in f[path].attrs else None
        )

        return cls(
            xyz=xyz,
            det_id=np.asarray(rs["dets:det_id"]),
            split=np.asarray(rs["split"]),
            center=center,
            coordsys=coordsys,
        )


@dataclass
class FocalPlane:
    """
    Container for a single focal plane measurement and its derived transforms.

    A `FocalPlane` represents one measured realization of a focal plane
    (typically a single UFM) together with the information needed to compare
    it against a reference template. The class stores the measured detector
    positions, encoder values, detector weights, and an optional pointing
    model. From these inputs it can derive several useful quantities:

    - Pointing model corrected detector positions (`pm_data`).
    - The affine transform mapping the template onto the measured data
      (`transform`).
    - The transformed template (`transformed`).
    - The affine transform after applying the pointing model (`pm_transform`).
    - The transformed template in the pointing model corrected frame
      (`pm_transformed`).
    - The transformed template in the focal plane collection frame
      (`fpc_transformed`).

    Derived quantities can be evaluated dynamically or stored as static
    values. When `static=True`, the public properties return the frozen
    values stored by `freeze`. When `static=False`, the quantities are
    recomputed on access. If `autofreeze=True`, newly computed values
    are automatically cached into the corresponding static fields.

    Assigning a new value to `wmsk` clears the cached properties.

    Attributes
    ----------
    name : str
        Human-readable identifier for the focal plane (e.g. UFM name).
    meas_id : str
        Identifier for the measurement from which this focal plane was
        constructed.
    data : DetectorOffsets
        Measured detector offsets in focal plane coordinates.
    enc : DetectorOffsets
        Encoder values associated with the measurement. These are used when
        applying the pointing model and are typically stored in the
        `horizon` coordinate system.
    weights : Float[np.ndarray, "ndet"]
        Per-detector weights used when fitting transforms and combining
        measurements.
    wmsk : Bool[np.ndarray, "ndet"]
        Boolean mask selecting detectors included in fitting and pointing
        model inversion.
    parent : FocalPlaneCollection
        The collection that this measurement belongs to.
    pointing_model : PointingModel
        Pointing model to apply before computing corrected transforms.
    static : bool
        If True, return values from the frozen static fields instead of
        recomputing them.
    autofreeze : bool
        If True, cache dynamically computed quantities into their
        corresponding static fields.
    fake_gamma : bool
        If True, ignore gamma when fitting and applying transforms,
        effectively treating the focal plane as two-dimensional.
    pm_data : DetectorOffsets
        Pointing model corrected detector offsets. Dynamically computed
        using `pointing_model.apply_det` or returned from the frozen
        static value.
    transform : Transform
        Best-fit affine transform mapping `template` to `data`.
    transformed : DetectorOffsets
        Template detector offsets after applying `transform`.
    pm_transform : Transform
        Best-fit affine transform mapping `template` to `pm_data`.
    pm_transformed : DetectorOffsets
        Template detector offsets after applying `pm_transform`.
    fpc_transformed : DetectorOffsets
        Template detector offsets after applying the parent collection's
        transform.
    resid : Float[np.ndarray, "ndet 3"]
        Residuals between `data` and `transformed`.
    pm_resid : Float[np.ndarray, "ndet 3"]
        Residuals between `pm_data` and `pm_transformed`.
    fpc_resid : Float[np.ndarray, "ndet 3"]
        Residuals between `pm_data` and `fpc_transformed`.
    template : DetectorOffsets
        Reference detector layout used as the transform source on sky.
        Provided by `parent`.
    template_fp : DetectorOffsets
        Reference detector layout used as the transform source in focal
        plane coordinates. Provided by `parent`.
    template_ot : DetectorOffsets
        Reference detector layout used as the transform source in OT
        coordinates. Provided by `parent`.
    data_msk : DetectorOffsets
        (Cached property) Detector offsets selected by `wmsk`.
    enc_msk : DetectorOffsets
        (Cached property) Encoder values selected by `wmsk`.
    template_msk : DetectorOffsets
        (Cached property) Template detector offsets selected by `wmsk`.
    weights_msk : Float[np.ndarray, "nmsk"]
        (Cached property) Detector weights selected by `wmsk`.
    pm_data_msk : DetectorOffsets
        (Property) Pointing model corrected detector offsets selected by `wmsk`.
    """

    name: str
    meas_id: str
    data: DetectorOffsets
    enc: DetectorOffsets
    weights: Float[np.ndarray, "ndet"]
    wmsk: Bool[np.ndarray, "ndet"]
    parent: FocalPlaneCollection
    pointing_model: PointingModel
    static: bool
    autofreeze: bool
    fake_gamma: bool
    pm_data: ClassVar[StaticOrCached[DetectorOffsets]] = StaticOrCached("pm_data")
    transform: ClassVar[StaticOrCached[Transform]] = StaticOrCached("transform")
    transformed: ClassVar[StaticOrCached[DetectorOffsets]] = StaticOrCached(
        "transformed"
    )
    pm_transform: ClassVar[StaticOrCached[Transform]] = StaticOrCached("pm_transform")
    pm_transformed: ClassVar[StaticOrCached[DetectorOffsets]] = StaticOrCached(
        "pm_transformed"
    )
    fpc_transformed: ClassVar[StaticOrCached[DetectorOffsets]] = StaticOrCached(
        "fpc_transformed"
    )
    _pm_data_stat: DetectorOffsets = field(init=False)
    _transform_stat: Transform = field(init=False)
    _transformed_stat: DetectorOffsets = field(init=False)
    _pm_transform_stat: Transform = field(init=False)
    _pm_transformed_stat: DetectorOffsets = field(init=False)
    _fpc_transformed_stat: DetectorOffsets = field(init=False)

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ("wmsk"):
            self.__dict__.pop("data_msk", None)
            self.__dict__.pop("enc_msk", None)
            self.__dict__.pop("template_msk", None)
            self.__dict__.pop("weights_msk", None)
        return super().__setattr__(name, value)

    @property
    def template(self) -> DetectorOffsets:
        return self.parent.template

    @property
    def template_fp(self) -> DetectorOffsets:
        return self.parent.template_fp

    @property
    def template_ot(self) -> DetectorOffsets:
        return self.parent.template_ot

    @cached_property
    def data_msk(self) -> DetectorOffsets:
        return self.data[self.wmsk]

    @cached_property
    def enc_msk(self) -> DetectorOffsets:
        return self.enc[self.wmsk]

    @cached_property
    def template_msk(self) -> DetectorOffsets:
        return self.template[self.wmsk]

    @cached_property
    def weights_msk(self) -> Float[np.ndarray, "nmsk"]:
        return self.weights[self.wmsk]

    @property
    def pm_data_msk(self) -> DetectorOffsets:
        return self.pm_data[self.wmsk]

    @classmethod
    def from_aman(
        cls,
        name: str,
        meas_id: str,
        aman: AxisManager,
        parent: FocalPlaneCollection,
        pointing_model: PointingModel,
        autofreeze: bool = False,
        fake_gamma: bool = True,
        weight_factor: float = 1000.0,
    ) -> FocalPlane:
        """
        Create a `FocalPlane` instance from a context loaded AxisManager.
        Here we make the following assumptions:

        * The pointing information is in `aman.focal_plane`
        * Detector matching information exists with `det_ids` in `aman.det_info.det_id`.
        * The desired cuts (outlier, optical, hits, R2, etc.) have already been applied
        * We either have per detector encoder fits in `aman.focal_plane` or the observation average in `aman.obs_info`.

        Parameters
        ----------
        name : str
            Name assigned to the resulting focal plane instance.
        meas_id : str
            Measurement identifier, probably the `obs_id`.
        aman : AxisManager
            Input data container holding focal plane and detector metadata.
            See above for details on assumptions.
        parent : FocalPlaneCollection
            The collection that this measurement belongs to.
        pointing_model : PointingModel
            Pointing model associated with the observation.
        autofreeze : bool, default: False
            See class docstring.
        fake_gamma : bool, default: True
            See class docstring.
            If True will also set `gamma` to `nan`.
        weight_factor : float, default: 1000
            Scaling factor applied to spacing when computing weights.

        Returns
        -------
        focal_plane : FocalPlane
            A fully constructed focal plane object containing.
            Note that this will have `static = False`.
        """
        # Figure out mapping
        template = parent.template
        aman_det_ids = np.asarray(aman.det_info.det_id)
        template_det_ids = np.asarray(template.det_id)
        _, aman_idx, template_idx = np.intersect1d(
            aman_det_ids, template_det_ids, return_indices=True
        )
        order = np.argsort(template_idx)
        aman_idx = aman_idx[order]
        template_idx = template_idx[order]
        template_msk = np.zeros(len(template.xyz), dtype=bool)
        template_msk[template_idx] = True
        valid_template = np.all(np.isfinite(template.xyz), axis=1)
        valid_msk = template_msk * valid_template

        # Setup data
        data = deepcopy(template)
        xi = np.array(aman.focal_plane.xi)
        eta = np.array(aman.focal_plane.eta)
        gamma = (
            np.array(aman.focal_plane.gamma)
            if not fake_gamma
            else np.nan * np.ones_like(xi)
        )
        data.xyz[template_idx] = np.column_stack((xi, eta, gamma))[aman_idx]
        data.xyz[~template_msk] = np.nan

        # Setup enc
        enc = deepcopy(template)
        enc.coordsys = "horizon"
        az = (
            np.array(aman.focal_plane.az)
            if "az" in aman.focal_plane
            else np.deg2rad(np.array(aman.obs_info.az_center)) * np.ones_like(xi)
        )
        el = (
            np.array(aman.focal_plane.el)
            if "el" in aman.focal_plane
            else np.deg2rad(np.array(aman.obs_info.el_center)) * np.ones_like(xi)
        )
        roll = (
            np.array(aman.focal_plane.roll)
            if "roll" in aman.focal_plane
            else np.deg2rad(np.array(aman.obs_info.roll_center)) * np.ones_like(xi)
        )
        enc.xyz[template_idx] = np.column_stack((az, el, roll))[aman_idx]
        enc.xyz[~template_msk] = np.nan

        # Use kmeans to kill any ghosts
        fp_white = vq.whiten(data.xyz[valid_msk, :2])
        codebook, _ = vq.kmeans(fp_white, 2)
        codes, _ = vq.vq(fp_white, codebook)
        c0 = codes == 0
        c1 = codes == 1
        m0 = np.median(data.xyz[valid_msk, :2][c0], axis=0)
        m1 = np.median(data.xyz[valid_msk, :2][c1], axis=0)
        dist = np.linalg.norm(m0 - m1)
        # If centroids are too far from each other use the bigger one
        cent = template.center
        if cent is None:
            cent = np.nanmean(template.xyz, axis=0)
        rad_thresh = 1.1 * np.nanmax(
            np.linalg.norm(template.xyz[:, :2] - cent[:2], axis=1)
        )
        if dist < rad_thresh:
            cluster = c0 + c1
        elif np.sum(c0) >= np.sum(c1):
            cluster = c0
        else:
            cluster = c1
        # Flag anything too far away from the center
        cent = np.median(data.xyz[valid_msk, :2][cluster], axis=0)
        r = np.linalg.norm(data.xyz[valid_msk, :2] - cent, axis=1)
        valid_msk[valid_msk] *= cluster * (r <= rad_thresh)

        # Compute weights
        aff, sft = mt.get_rigid(data.xyz[valid_msk, :2], template.xyz[valid_msk, :2])
        aligned = mt.apply_transform(data.xyz[:, :2], aff, sft)
        xieta_spacing = mu.estimate_spacing(template.xyz[valid_msk, :2])
        # For gamma rather than the spacing in real space we want the difference between bins
        # This is a rough estimate but good enough for us
        gamma_sorted = np.sort(template.xyz[valid_msk, 2])
        gamma_spacing = np.percentile(np.diff(gamma_sorted), 99.9)
        spacing = np.array([xieta_spacing, xieta_spacing, gamma_spacing])

        weights = np.zeros(len(aligned))
        if not fake_gamma:
            gscale, gsft = gamma_fit(data.xyz[:, 2], template.xyz[:, 2])
            weights[valid_msk] = mu.gen_weights(
                np.column_stack((aligned[valid_msk], gscale * data.xyz[:, 2] + gsft)),
                template.xyz[valid_msk],
                spacing.ravel() / weight_factor,
            )
        else:
            weights[valid_msk] = mu.gen_weights(
                aligned[valid_msk],
                template.xyz[valid_msk, :2],
                spacing[:2].ravel() / weight_factor,
            )

        # ~1 sigma cut
        weights[weights < 0.61] = 0
        valid_msk *= weights == 0

        return cls(
            name,
            meas_id,
            data,
            enc,
            weights,
            valid_msk,
            parent,
            pointing_model,
            False,
            autofreeze,
            fake_gamma,
        )

    def freeze(self, static: bool, optimizer: bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        self.static = True
        self._pm_data_stat = self._pm_data_dyn
        self._fpc_transformed_stat = self._fpc_transformed_dyn
        if not optimizer:
            self._transform_stat = self._transform_dyn
            self._transformed_stat = self._transformed_dyn
            self._pm_transform_stat = self._pm_transform_dyn
            self._pm_transformed_stat = self._pm_transformed_dyn
        self.static = static

    def thaw(self, static: bool):
        """
        Clear the current values in static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after thawing.
        """
        self.__dict__.pop("_pm_data_stat", None)
        self.__dict__.pop("_fpc_transformed_stat", None)
        self.__dict__.pop("_transform_stat", None)
        self.__dict__.pop("_transformed_stat", None)
        self.__dict__.pop("_pm_transform_stat", None)
        self.__dict__.pop("_pm_transformed_stat", None)
        self.static = static

    @property
    def _pm_data_dyn(self) -> DetectorOffsets:
        to_ret = self.pointing_model.apply_det(self)
        if self.autofreeze:
            self._pm_data_stat = to_ret
        return to_ret

    @property
    def _transform_dyn(self) -> Transform:
        return _get_aff(
            self, "data_msk", "template_msk", "weights_msk", "_transform_stat", False
        )

    @property
    def _transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_transformed_stat", self.transform)

    @property
    def _pm_transform_dyn(self) -> Transform:
        return _get_aff(
            self,
            "pm_data_msk",
            "template_msk",
            "weights_msk",
            "_pm_transform_stat",
            False,
        )

    @property
    def _pm_transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_pm_transformed_stat", self.pm_transform)

    @property
    def _fpc_transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_pm_transformed_stat", self.parent.transform)

    @property
    def resid(self):
        return self.data.xyz - self.transformed.xyz

    @property
    def pm_resid(self):
        return self.pm_data.xyz - self.pm_transformed.xyz

    @property
    def fpc_resid(self):
        return self.pm_data.xyz - self.fpc_transformed.xyz

    def save(
        self,
        f: h5py.File,
        path: str,
        pointing_models: tuple[PointingModel, ...],
        overwrite: bool = True,
    ):
        """
        Save the focal plane to an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object to save to.
        path : str
            The path within the HDF5 file where the focal plane will be saved.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.
        overwrite : bool, default: True
            Whether to overwrite existing datasets.
        """
        group = _create_group(f, path, overwrite)
        pm_epcs = [pm.epoch for pm in pointing_models]
        try:
            pm_idx = pm_epcs.index(self.pointing_model.epoch)
        except ValueError:
            raise ValueError(
                f"Can't find pointing model with epoch {self.pointing_model.epoch} (in {self.name}+{self.meas_id})"
            )

        _add_attrs(
            group,
            {
                "name": self.name,
                "meas_id": self.meas_id,
                "static": self.static,
                "autofreeze": self.autofreeze,
                "fake_gamma": self.fake_gamma,
                "pm_idx": pm_idx,
            },
        )

        self.data.save(f, f"{path}/data", overwrite=overwrite)
        self.enc.save(f, f"{path}/enc", overwrite=overwrite)
        if f"{path}/weights" in f:
            if overwrite:
                del f[f"{path}/weights"]
            else:
                raise ValueError(f"{path}/weights already exists")
        f.create_dataset(f"{path}/weights", data=self.weights)
        f.create_dataset(f"{path}/wmsk", data=self.wmsk)

        if "_pm_data_stat" in self.__dict__:
            self._pm_data_stat.save(f, f"{path}/pm_data", overwrite=overwrite)
        if "_transform_stat" in self.__dict__:
            self._transform_stat.save(f, f"{path}/transform", overwrite=overwrite)
        if "_transformed_stat" in self.__dict__:
            self._transformed_stat.save(f, f"{path}/transformed", overwrite=overwrite)
        if "_pm_transform_stat" in self.__dict__:
            self._pm_transform_stat.save(f, f"{path}/pm_transform", overwrite=overwrite)
        if "_pm_transformed_stat" in self.__dict__:
            self._pm_transformed_stat.save(
                f, f"{path}/pm_transformed", overwrite=overwrite
            )
        if "_fpc_transformed_stat" in self.__dict__:
            self._fpc_transformed_stat.save(
                f, f"{path}/fpc_transformed", overwrite=overwrite
            )

    @classmethod
    def load(
        cls,
        f: h5py.File,
        path: str,
        parent: FocalPlaneCollection,
        pointing_models: tuple[PointingModel, ...],
    ) -> Self:
        """
        Load a focal plane from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object containing the focal plane.
        path : str
            The path within the HDF5 file where the focal plane is stored.
        template : DetectorOffsets
            The template shared by the focal planes in the collection.
        template_fp : DetectorOffsets
            Reference detector layout used as the transform source in focal plane coords.
        template_ot : DetectorOffsets
            Reference detector layout used as the transform source in OT coords.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.


        Returns
        -------
        FocalPlane
            The loaded focal plane.
        """
        group = cast(h5py.Group, f[path])
        data = DetectorOffsets.load(f, f"{path}/data")
        enc = DetectorOffsets.load(f, f"{path}/enc")
        weights = np.asarray(f[f"{path}/weights"])
        if "wmsk" in group:
            wmsk = np.asarray(f[f"{path}/wmsk"])
        else:
            wmsk = np.isfinite(weights) * (weights > 0)
        wmsk = np.isfinite(weights) * (weights > 0)

        pm_idx = cast(int, group.attrs["pm_idx"])
        fp = cls(
            name=cast(str, group.attrs["name"]),
            meas_id=cast(str, group.attrs["meas_id"]),
            data=data,
            enc=enc,
            weights=weights,
            wmsk=wmsk,
            parent=parent,
            pointing_model=pointing_models[pm_idx],
            static=cast(bool, group.attrs["static"]),
            autofreeze=cast(bool, group.attrs["autofreeze"]),
            fake_gamma=cast(bool, group.attrs["fake_gamma"]),
        )

        if "pm_data" in group:
            fp._pm_data_stat = DetectorOffsets.load(f, f"{path}/pm_data")
        if "transform" in group:
            fp._transform_stat = Transform.load(f, f"{path}/transform")
        if "transformed" in group:
            fp._transformed_stat = DetectorOffsets.load(f, f"{path}/transformed")
        if "pm_transform" in group:
            fp._pm_transform_stat = Transform.load(f, f"{path}/pm_transform")
        if "pm_transformed" in group:
            fp._pm_transformed_stat = DetectorOffsets.load(f, f"{path}/pm_transformed")
        if "fpc_transformed" in group:
            fp._fpc_transformed_stat = DetectorOffsets.load(
                f, f"{path}/fpc_transformed"
            )

        return fp


@dataclass
class FocalPlaneCollection:
    """
    Container for multiple measurements of the same focal plane.

    A `FocalPlaneCollection` combines several `FocalPlane` instances that
    share a common detector template and identical detector set. All
    measurements are expected to contain the same detectors in
    the same ordering.

    Individual measurements are merged using their detector weights to
    produce a weighted-average focal plane, which can then be compared
    against the reference template through a single affine transform.

    The collection operates on the pointing model corrected detector
    positions (`FocalPlane.pm_data`) from each measurement.
    Detector locations and weights are combined on a per-detector
    basis, with missing values represented by `nan`s.

    As with `FocalPlane`, derived quantities can be evaluated dynamically
    or stored as static values. When `static=True`, the public properties
    return frozen values previously stored by `freeze`. When
    `static=False`, quantities are recomputed on access. If
    `autofreeze=True`, newly computed values are automatically cached into
    the corresponding static fields.

    Attributes
    ----------
    name : str
        Human-readable identifier for the collection.
        This is most likely the name of the UFM.
    focal_planes : list[FocalPlane]
        Individual focal-plane measurements to combine. All measurements
        must contain the same detector set and detector ordering.
    template : DetectorOffsets
        Reference detector layout used as the transform source.
    template_fp : DetectorOffsets
        Reference detector layout used as the transform source in focal
        plane coordinates.
    template_ot : DetectorOffsets
        Reference detector layout used as the transform source in OT
        coordinates.
    static : bool
        If True, return values from the frozen static fields instead of
        recomputing them.
    autofreeze : bool
        If True, cache dynamically computed quantities into their
        corresponding static fields.
    pad : bool
        Flag indicating that this collection should be excluded from
        common-mode calculations. Typically used for padded or otherwise
        invalid focal-plane slots.
    fake_gamma : bool
        If True, ignore gamma when fitting transforms, effectively treating
        the focal plane as two-dimensional.
    parent : OpticsTube 
        The OT that this FocalPlane belongs to.
    data : DetectorOffsets
        Weighted-average detector offsets constructed from the
        pointing-model-corrected measurements in `focal_planes`.
        Dynamically computed or retrieved from the frozen static value.
    weights : Float[np.ndarray, "ndet"]
        Combined detector weights obtained by summing the weights from all
        constituent measurements.
    transform : Transform
        Best-fit affine transform mapping `template` to `data`.
    transformed : DetectorOffsets
        Template detector offsets after applying `transform`.
    resid : Float[np.ndarray, "ndet 3"]
        Residuals between `data` and `transformed`.
    ndet : int
        (Property) Number of detectors in the reference template.
    nobs : int
        (Property) Number of focal-plane measurements in the collection.
    wmsk : Bool[np.ndarray, "ndet"]
        (Property) Boolean mask selecting detectors with finite,
        positive combined weights.
    residuals : Float[np.ndarray, "ndet*nobs"]
        (Property) Flattened array of weighted residual magnitudes for
        all measurements. Residuals are computed relative to the
        collection's transformed template, using the number of
        coordinates determined by each measurement's `fake_gamma`
        setting. Detectors excluded by their measurement mask contribute
        zero.
    chisq : float
        (Property) Sum of squared values in `residuals`.
    """

    name: str
    focal_planes: list[FocalPlane]
    template: DetectorOffsets
    template_fp: DetectorOffsets
    template_ot: DetectorOffsets
    static: bool
    autofreeze: bool
    pad: bool
    fake_gamma: bool
    parent: OpticsTube
    data: ClassVar[StaticOrCached[DetectorOffsets]] = StaticOrCached("data")
    weights: ClassVar[StaticOrCached[Float[np.ndarray, "ndet"]]] = StaticOrCached(
        "weights"
    )
    transform: ClassVar[StaticOrCached[Transform]] = StaticOrCached("transform")
    transformed: ClassVar[StaticOrCached[DetectorOffsets]] = StaticOrCached(
        "transformed"
    )
    _data_stat: DetectorOffsets = field(init=False)
    _weights_stat: Float[np.ndarray, "ndet"] = field(init=False)
    _transform_stat: Transform = field(init=False)
    _transformed_stat: DetectorOffsets = field(init=False)
    _residual_buffer: Float[np.ndarray, "n_data"] = field(init=False)

    def __getitem__(self, key):
        fp_ids = [fp.meas_id for fp in self.focal_planes]
        if key in fp_ids:
            return self.focal_planes[fp_ids.index(key)]
        raise IndexError(f"Invalid key: {key}")

    def freeze(self, static: bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        if self.pad:
            return
        self.static = True
        self._data_stat = self._data_dyn
        self._weights_stat = self._weights_dyn
        self._transform_stat = self._transform_dyn
        self._transformed_stat = self._transformed_dyn
        self.static = static

    def thaw(self, static: bool):
        """
        Clear the current values in static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after thawing.
        """
        self.__dict__.pop("_weights_stat", None)
        self.__dict__.pop("_data_stat", None)
        self.__dict__.pop("_transform_stat", None)
        self.__dict__.pop("_transformed_stat", None)
        self.static = static

    @property
    def ndet(self) -> int:
        return len(self.template.xyz)

    @property
    def nobs(self) -> int:
        return len(self.focal_planes)

    @property
    def residuals(self) -> Float[np.ndarray, "ndet*nobs"]:
        n_data = self.ndet * self.nobs
        if n_data == 0:
            return np.empty(0)
        if "_residual_buffer" in self.__dict__ and len(self._residual_buffer) == n_data:
            resid = self._residual_buffer
        else:
            resid = np.empty(n_data)
            self._residual_buffer = resid
        i = 0
        for meas in self.focal_planes:
            ndim = 3 - int(meas.fake_gamma)
            r = np.linalg.norm(meas.fpc_resid[:, :ndim], axis=1) * meas.weights
            r[~meas.wmsk] = 0
            resid[i : (i + len(r))] = r
            i += len(r)
        to_ret = np.nan_to_num(
            resid, copy=False, nan=0.0, posinf=np.inf, neginf=-np.inf
        )
        if to_ret is None:
            raise TypeError
        return to_ret

    @property
    def chisq(self) -> float:
        return np.sum(self.residuals**2)

    @property
    def _data_dyn(self) -> DetectorOffsets:
        to_ret = deepcopy(self.template)
        to_ret.xyz[:] = 0
        fmsk = np.zeros(len(to_ret.xyz), dtype=bool)
        all_fake = True
        for meas in self.focal_planes:
            fmsk += meas.wmsk
            all_fake *= meas.fake_gamma
            ndim = 3 - int(meas.fake_gamma)
            to_ret.xyz[meas.wmsk, :ndim] += (
                    meas.pm_data.xyz[meas.wmsk, :ndim] * meas.weights[meas.wmsk, None]
            )
        to_ret.xyz[~fmsk] = np.nan
        to_ret.xyz[fmsk] /= self.weights[fmsk, None]
        if all_fake:
            to_ret.xyz[:, 2] = np.nan
        to_ret.center = None
        if self.autofreeze:
            self._data_stat = to_ret
        return to_ret

    @property
    def _weights_dyn(self) -> Float[np.ndarray, "ndet"]:
        weights_full = np.asarray([fp.weights for fp in self.focal_planes])
        to_ret = np.nansum(weights_full, axis=0)
        if self.autofreeze:
            self._weights_stat = to_ret
        return to_ret

    @property
    def _transform_dyn(self) -> Transform:
        if self.pad or np.sum(self.wmsk) == 0:
            return self.parent.cm_transform
        return _get_aff(self, "data", "template", "weights", "_transform_stat", True)

    @property
    def _transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_transformed_stat", self.transform)

    @property
    def resid(self):
        return self.data.xyz - self.transformed.xyz

    @property
    def wmsk(self):
        weights = self.weights
        return np.isfinite(weights) * (weights > 0)

    def save(
        self,
        f: h5py.File,
        path: str,
        pointing_models: tuple[PointingModel, ...],
        overwrite: bool = True,
    ):
        """
        Save the focal plane collection to an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object to save to.
        path : str
            The path within the HDF5 file where the collection will be saved.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.
        overwrite : bool, default: True
            Whether to overwrite existing datasets.
        """
        meas_ids = [fp.meas_id for fp in self.focal_planes]
        if len(meas_ids) != len(set(meas_ids)):
            raise ValueError("FocalPlaneCollection contains overlapping meas_ids")

        group = _create_group(f, path, overwrite)
        _add_attrs(
            group,
            {
                "name": self.name,
                "static": self.static,
                "autofreeze": self.autofreeze,
                "pad": self.pad,
                "fake_gamma": self.fake_gamma,
            },
        )
        self.template.save(f, f"{path}/template", overwrite=overwrite)
        self.template_fp.save(f, f"{path}/template_fp", overwrite=overwrite)
        self.template_ot.save(f, f"{path}/template_ot", overwrite=overwrite)
        for fp in self.focal_planes:
            fp.save(
                f,
                f"{path}/focal_planes/{fp.meas_id}",
                pointing_models=pointing_models,
                overwrite=overwrite,
            )

        if "_data_stat" in self.__dict__:
            self._data_stat.save(f, f"{path}/data", overwrite=overwrite)
        if "_weights_stat" in self.__dict__:
            if f"{path}/weights" in f:
                if overwrite:
                    del f[f"{path}/weights"]
                else:
                    raise ValueError(f"{path}/weights already exists")
            f.create_dataset(f"{path}/weights", data=self._weights_stat)
        if "_transform_stat" in self.__dict__:
            self._transform_stat.save(f, f"{path}/transform")
        if "_transformed_stat" in self.__dict__:
            self._transformed_stat.save(f, f"{path}/transformed", overwrite=overwrite)

    @classmethod
    def load(
            cls, f: h5py.File, path: str, pointing_models: tuple[PointingModel, ...], parent: OpticsTube,
    ) -> Self:
        """
        Load a focal plane collection from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object containing the collection.
        path : str
            The path within the HDF5 file where the collection is stored.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.
        parent : OpticsTube 
            The OT that this FocalPlane belongs to.

        Returns
        -------
        FocalPlaneCollection
            The loaded focal plane collection.
        """
        template = DetectorOffsets.load(
            f,
            f"{path}/template",
        )
        template_fp = DetectorOffsets.load(
            f,
            f"{path}/template_fp",
        )
        template_ot = DetectorOffsets.load(
            f,
            f"{path}/template_ot",
        )
        group = cast(h5py.Group, f[path])

        collection = cls(
            name=cast(str, group.attrs["name"]),
            focal_planes=[],
            template=template,
            template_fp=template_fp,
            template_ot=template_ot,
            static=cast(bool, group.attrs["static"]),
            autofreeze=cast(bool, group.attrs["autofreeze"]),
            pad=cast(bool, group.attrs["pad"]),
            fake_gamma=cast(bool, group.attrs["fake_gamma"]),
            parent=parent,
        )

        if "data" in group:
            collection._data_stat = DetectorOffsets.load(
                f,
                f"{path}/data",
            )
        if "weights" in group:
            collection._weights_stat = np.asarray(f[f"{path}/weights"])
        # if "scatter" in group:
        #     collection._scatter_stat = np.asarray(f[f"{path}/scatter"])
        if "transform" in group:
            collection._transform_stat = Transform.load(f, f"{path}/transform")
        if "transformed" in group:
            collection._transformed_stat = DetectorOffsets.load(
                f,
                f"{path}/transformed",
            )

        if "focal_planes" not in group:
            return collection
        for name in cast(h5py.Group, group["focal_planes"]):
            collection.focal_planes.append(
                FocalPlane.load(
                    f,
                    f"{path}/focal_planes/{name}",
                    parent=collection,
                    pointing_models=pointing_models,
                )
            )

        return collection


@dataclass
class OpticsTube:
    """
    An `OpticsTube` groups one or more `FocalPlaneCollection` instances
    associated with the same optics tube and
    calculates tube-level common-mode transforms. These common-mode
    transforms capture distortions that are shared across the focal planes
    within the tube and can be used to separate optics-tube effects from
    from per-array or full receiver effects.

    The tube common mode is computed from the affine transforms of all
    non-padded focal planes in the collection. If no valid focal planes are
    available, the identity transform is returned and the tube is marked as
    containing only padded data.

    When `static=True`, the public properties return frozen values previously
    stored by `freeze`. When `static=False`, quantities are recomputed on
    access. If `autofreeze=True`, newly computed values are automatically
    cached into the corresponding static fields.

    Attributes
    ----------
    name : str
        Human-readable identifier for the optics tube.
    focal_planes : list[FocalPlaneCollection]
        Focal-plane collections associated with this optics tube.
    wafer_slots : list[str]
        Wafer-slot identifiers corresponding to the entries in
        `focal_planes`. These are used to support lookup by wafer slot
        through `__getitem__`.
    static : bool
        If True, return values from the frozen static fields instead of
        recomputing them.
    autofreeze : bool
        If True, cache dynamically computed quantities into their
        corresponding static fields.
    parent : Receiver
        The Receiver that this OT belongs to.
    cm_transform : Transform
        Common-mode transform derived from the transforms of all
        non-padded focal-plane collections.
    cm_transform_norx : Transform
        Copy of the optics-tube common-mode transform with a receiver common mode removed.
    allpad : bool
        (Property) Flag indicating whether all focal-plane collections in the optics
        tube are padded and should therefore be excluded from common-mode
        calculations. Returns True if the collection is empty.
    """

    name: str
    focal_planes: list[FocalPlaneCollection]
    wafer_slots: list[str]
    static: bool
    autofreeze: bool
    parent: Receiver
    cm_transform: ClassVar[StaticOrCached[Transform]] = StaticOrCached("cm_transform")
    cm_transform_norx: ClassVar[StaticOrCached[Transform]] = StaticOrCached("cm_transform_norx")
    _cm_transform_stat: Transform = field(init=False)
    _cm_transform_norx_stat: Transform = field(init=False)

    def __getitem__(self, key):
        if key in self.wafer_slots:
            return self.focal_planes[self.wafer_slots.index(key)]
        fp_names = [fp.name for fp in self.focal_planes]
        if key in fp_names:
            return self.focal_planes[fp_names.index(key)]
        raise IndexError(f"Invalid key: {key}")

    def freeze(self, static: bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        if self.allpad:
            return
        self.static = True
        self._cm_transform_stat = self._cm_transform_dyn
        self.static = static

    def thaw(self, static: bool):
        """
        Clear the current values in static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after thawing.
        """
        self.__dict__.pop("_cm_transform_stat", None)
        self.static = static

    @property
    def allpad(self) -> bool:
        if len(self.focal_planes) == 0:
            return True
        return bool(np.all([fp.pad for fp in self.focal_planes]))

    @property
    def _cm_transform_dyn(self) -> Transform:
        transforms = [
            fp.transform.affine for fp in self.focal_planes if fp.pad is False
        ]
        shifts = [fp.transform.shift for fp in self.focal_planes if fp.pad is False]
        if len(transforms) == 0:
            iden = Transform.identity()
            cm_shift, cm_aff = iden.shift, iden.affine
        elif len(transforms) == 1:
            cm_aff, cm_shift = transforms[0], shifts[0]
        else:
            cm_aff, cm_shift = mt.get_common_mode(
                transforms, shifts, -2, False, fast=True
            )
        to_ret = Transform(cm_shift, cm_aff)
        if self.autofreeze:
            self._cm_transform_stat = to_ret
        return to_ret

    @property
    def _cm_transform_norx_dyn(self) -> Transform:
        if self.allpad:
            to_ret = Transform.identity()
        else:
            aff, sft = mt.decompose_transform(
                self.cm_transform.affine,
                self.cm_transform.shift,
                self.parent.cm_transform.affine,
                self.parent.cm_transform.shift,
            )
            to_ret = Transform(sft, aff)
        if self.autofreeze:
            self._cm_transform_norx_stat = to_ret
        return to_ret


    def save(
        self,
        f: h5py.File,
        path: str,
        pointing_models: tuple[PointingModel, ...],
        overwrite: bool = True,
    ):
        """
        Save the optics tube to an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object to save to.
        path : str
            The path within the HDF5 file where the optics tube will be saved.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.
        overwrite : bool, default: True
            Whether to overwrite existing datasets.
        """
        group = _create_group(f, path, overwrite)
        _add_attrs(
            group,
            {
                "name": self.name,
                "wafer_slots": self.wafer_slots,
                "static": self.static,
                "autofreeze": self.autofreeze,
            },
        )
        for ws, fp in zip(self.wafer_slots, self.focal_planes):
            fp.save(f, f"{path}/{ws}-{fp.name}", pointing_models, overwrite=overwrite)
        if "_cm_transform_stat" in self.__dict__:
            self._cm_transform_stat.save(f, f"{path}/cm_transform")
        if "_cm_transform_norx_stat" in self.__dict__:
            self.cm_transform_norx.save(f, f"{path}/cm_transform_norx")

    @classmethod
    def load(
            cls, f: h5py.File, path: str, pointing_models: tuple[PointingModel, ...], parent: Receiver
    ) -> Self:
        """
        Load an optics tube from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object containing the optics tube.
        path : str
            The path within the HDF5 file where the optics tube is stored.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.

        Returns
        -------
        OpticsTube
            The loaded optics tube.
        """
        group = cast(h5py.Group, f[path])
        tube = cls(
            name=cast(str, group.attrs["name"]),
            focal_planes=[],
            wafer_slots=list(np.asarray(group.attrs["wafer_slots"])),
            static=cast(bool, group.attrs["static"]),
            autofreeze=cast(bool, group.attrs["autofreeze"]),
            parent=parent,
        )

        if "cm_transform" in group:
            tube._cm_transform_stat = Transform.load(f, f"{path}/cm_transform")
        if "cm_transform_norx" in group:
            tube._cm_transform_norx_stat = Transform.load(f, f"{path}/cm_transform_norx")
        focal_planes = []
        group = cast(h5py.Group, f[path])
        for name in group:
            if name in {"cm_transform", "cm_transform_norx"}:
                continue
            focal_planes.append(
                FocalPlaneCollection.load(
                    f, f"{path}/{name}", pointing_models=pointing_models, parent=tube
                )
            )
        tube.focal_planes = focal_planes

        return tube


@dataclass
class Receiver:
    """
    A `Receiver` groups one or more `OpticsTube` instances and provides
    methods for calculating receive common-mode transforms. The
    receiver common mode captures distortions that are shared across
    multiple optics tubes and can be used to separate receiver wide effects
    from OT specific distortions.

    The receiver common mode is computed from the common-mode transforms of
    all optics tubes that contain valid, non-padded data. If no valid
    optics tubes are available, the identity transform is returned.
    Note that unlike the OT common mode, we only compute a rigid transformation here.

    After computing a receiver common mode, `remove_rx_all` can be used
    to propagate that common mode to all contained optics tubes, allowing
    each tube to store an OT only common-mode transform in
    `OpticsTube.cm_transform_norx`.

    When `static=True`, the public properties return frozen values previously
    stored by `freeze`. When `static=False`, quantities are recomputed on
    access. If `autofreeze=True`, newly computed values are automatically
    cached into the corresponding static fields.

    Optics tubes may be accessed by name through `__getitem__`.

    Attributes
    ----------
    name : str
        Human-readable identifier for the receiver.
        Probably just the telescope name.
    epoch : str
        Human-readable identifier for the epoch associated with this
        data. If this spans multiple epochs they will be seperated with a '+'.
    optics_tubes : list[OpticsTube]
        Optics tubes associated with this receiver.
    static : bool
        If True, return values from the frozen static fields instead of
        recomputing them.
    autofreeze : bool
        If True, cache dynamically computed quantities into their
        corresponding static fields.
    cm_transform : Transform
        Receiver-level common-mode transform derived from the common-mode
        transforms of all optics tubes containing valid data.
    allpad : bool
        (Property) Flag indicating whether all optics tube are padded
        and should therefore be excluded from fitting.
    """

    name: str
    epoch: str
    optics_tubes: list[OpticsTube]
    static: bool
    autofreeze: bool
    cm_transform: ClassVar[StaticOrCached[Transform]] = StaticOrCached("cm_transform")
    _cm_transform_stat: Transform = field(init=False)

    def __getitem__(self, key):
        ot_names = [ot.name for ot in self.optics_tubes]
        if key in ot_names:
            return self.optics_tubes[ot_names.index(key)]
        raise IndexError(f"Invalid key: {key}")

    def find_fpc(self, ot_name: str, fp_name: str) -> Optional[FocalPlaneCollection]:
        """
        Find a FocalPlaneCollection by its OpticsTube and collection names.

        Parameters
        ----------
        ot_name : str
            Name of the OpticsTube.
        fp_name : str
            Name of the FocalPlaneCollection.

        Returns
        -------
        fpc : Optional[FocalPlaneCollection]
            The matching focal-plane collection, or None if not found.
        """
        try:
            return self[ot_name][fp_name]
        except IndexError:
            return None

    def freeze(self, static: bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        if self.allpad:
            return
        self.static = True
        self._cm_transform_stat = self._cm_transform_dyn
        self.static = static

    def thaw(self, static: bool):
        """
        Clear the current values in static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after thawing.
        """
        self.__dict__.pop("_cm_transform_stat", None)
        self.static = static

    @property
    def allpad(self) -> bool:
        if len(self.optics_tubes) == 0:
            return True
        return bool(np.all([ot.allpad for ot in self.optics_tubes]))

    @property
    def _cm_transform_dyn(self) -> Transform:
        transforms = [
            ot.cm_transform.affine for ot in self.optics_tubes if ot.allpad is False
        ]
        shifts = [
            ot.cm_transform.shift for ot in self.optics_tubes if ot.allpad is False
        ]
        if len(transforms) == 0:
            iden = Transform.identity()
            cm_shift, cm_aff = iden.shift, iden.affine
        elif len(transforms) == 1:
            cm_aff, cm_shift = transforms[0], shifts[0]
        else:
            cm_aff, cm_shift = mt.get_common_mode(
                transforms, shifts, -2, True, fast=True
            )
        to_ret = Transform(cm_shift, cm_aff)
        if self.autofreeze:
            self._cm_transform_stat = to_ret
        return to_ret

    def remove_rx_all(self):
        """
        Remove the current receiver common mode from all optics tubes.

        This method calls `OpticsTube.remove_rx` for every optics tube in the
        receiver using the current value of `cm_transform`. After execution,
        each optics tube stores a receiver-corrected common-mode transform in
        `cm_transform_norx`.
        """
        for ot in self.optics_tubes:
            ot.remove_rx(self.cm_transform)

    def save(self, f, path, pointing_models: tuple[PointingModel, ...], overwrite=True):
        """
        Save the receiver and its optics tubes to an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            HDF5 file object to save to.
        path : str
            Path within the HDF5 file at which to save the receiver.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.
        overwrite : bool, default: True
            If True, overwrite an existing receiver at `path`.

        Raises
        ------
        ValueError
            If multiple optics tubes have the same name.
        """
        group = _create_group(f, path, overwrite)
        _add_attrs(
            group,
            {
                "name": self.name,
                "epoch": self.epoch,
                "static": self.static,
                "autofreeze": self.autofreeze,
            },
        )
        if "_cm_transform_stat" in self.__dict__:
            self._cm_transform_stat.save(f, f"{path}/cm_transform")
        names = [ot.name for ot in self.optics_tubes]
        if len(names) != len(set(names)):
            raise ValueError("Optics tube names must be unique")
        for ot in self.optics_tubes:
            ot.save(f, f"{path}/{ot.name}", pointing_models, overwrite=overwrite)

    @classmethod
    def load(
        cls, f: h5py.File, path: str, pointing_models: tuple[PointingModel, ...]
    ) -> Self:
        """
        Load a receiver from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object containing the optics tube.
        path : str
            The path within the HDF5 file where the optics tube is stored.
        pointing_models : tuple[PointingModel, ...]
            The pointing models used by the focal-planes.

        Returns
        -------
        Receiver
            The loaded receiver.
        """
        group = cast(h5py.Group, f[path])
        name = cast(str, group.attrs["name"])
        epoch = cast(str, group.attrs["epoch"])
        static = cast(bool, group.attrs["static"])
        autofreeze = cast(bool, group.attrs["autofreeze"])
        receiver = cls(
            name=name,
            epoch=epoch,
            optics_tubes=[],
            static=static,
            autofreeze=autofreeze,
        )

        if "cm_transform" in group:
            receiver._cm_transform_stat = Transform.load(f, f"{path}/cm_transform")
        optics_tubes = [
            OpticsTube.load(f, f"{path}/{ot_name}", pointing_models, receiver)
            for ot_name in group.keys()
            if ot_name != "cm_transform"
        ]
        receiver.optics_tubes = optics_tubes

        return receiver


@dataclass
class PointingSystem:
    """
    Collection of epoch-dependent pointing models and receiver data.

    A `PointingSystem` combines an observing era, a collection of pointing
    models, and the corresponding receiver-level focal-plane data. It provides
    a unified interface for managing pointing model parameters, evaluating
    focal-plane residuals, calculating common-mode transforms, and optimizing
    the complete pointing solution.

    Pointing model parameters are managed through `parameter_names` and
    `parameter_map`. These define the independent parameters of the system
    and specify which pointing models each parameter applies to. A parameter
    may be shared across multiple epochs or restricted to a subset of epochs.
    Nuisance parameters are managed separately through `NuisanceParams`.

    The receiver data are organized hierarchically:
    `PointingSystem -> Receiver -> OpticsTube -> FocalPlaneCollection -> FocalPlane`.

    The system provides two objective functions for optimization:

    - `objective` evaluates the total chi-squared, including regularization
      of receiver common-mode transforms.
    - `objective_resid` returns the corresponding residual vector, allowing
      least-squares optimizers to operate directly on the residuals.


    Attributes
    ----------
    era : epochs.Era
        Collection of observing epochs represented by the system.
    pointing_models : tuple[PointingModel, ...]
        Pointing model associated with each epoch or epoch group. The ordering
        must be consistent with the receiver and epoch organization.
    receivers : tuple[Receiver, ...]
        Receiver and focal-plane data associated with each epoch or epoch
        group. Each receiver contains its associated optics tubes and
        focal-plane collections.
    nuisance_params : NuisanceParams
        Additional nuisance parameters used by the pointing models. These
        parameters are managed separately from the primary pointing model
        parameters.
    parameter_names : tuple[str, ...]
        Names of the independent pointing model parameters. Names without
        parentheses represent parameters shared across all applicable epochs.
        Names with parentheses identify parameters restricted to the epochs
        listed within the parentheses.
    parameter_map : tuple[Integer[np.ndarray, "?npar"], ...]
        Mapping between independent parameters and the indices of the
        pointing models to which they apply. Each entry corresponds to the
        parameter with the same index in `parameter_names`.
    calender_str : str
        Serialized calendar YAML string used to reconstruct the observing
        calendar and associated era.
    state_meta : dict[str, str]
        String-based metadata associated with the current system state.
        Used to preserve provenance and support subsequent data forensics.
    parameters : Float[np.ndarray, "npar"]
        (Property) Current values of the independent pointing model parameters.
        Values are extracted from the corresponding pointing models according
        to `parameter_map`.
    parameters_full : Float[np.ndarray, "n"]
        (Property) Concatenation of the independent pointing model parameters and the
        flattened nuisance parameter values.
    n_data : int
        (Property) Total number of residual measurements included in the system,
        excluding padded focal-plane collections.
    residuals : Float[np.ndarray, "n_data"]
        (Property) Concatenated residual vector from all non-padded focal-plane collections.
    chisq : float
        (Property) Total chi-squared of the system, calculated from the residuals of
        all non-padded focal-plane collections. Does not include the
        receiver common-mode regularization penalty applied by `objective`.
    """

    era: epochs.Era
    pointing_models: tuple[PointingModel, ...]
    receivers: tuple[Receiver, ...]
    nuisance_params: NuisanceParams
    parameter_names: tuple[str, ...]
    parameter_map: tuple[Integer[np.ndarray, "?npar"], ...]
    calender_str: str
    state_meta: dict[str, str]
    _residual_buffer: Float[np.ndarray, "n_data"] = field(init=False)

    def __post_init__(self):
        # Force consistancy
        self.update_parameters(self.parameters)

    @property
    def parameters_full(self) -> Float[np.ndarray, "n"]:
        return np.hstack((self.parameters, self.nuisance_params.params_flat))

    def set_static(self, static: bool):
        """
        Set `static` for all relevent childen.

        Parameters
        ----------
        static : bool
            The value to set `static` to for all relevent children.
        """
        for rx in self.receivers:
            rx.static = static
            for ot in rx.optics_tubes:
                ot.static = static
                for fp in ot.focal_planes:
                    fp.static = static
                    for meas in fp.focal_planes:
                        meas.static = static

    def set_freeze(self, static: bool, optimizer: bool = False):
        """
        Run freeze for all relevent childen.

        Parameters
        ----------
        static : bool
            The value to set `static` to for all relevent children.
        """
        for rx in self.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    for meas in fp.focal_planes:
                        meas.freeze(True, optimizer)
                    fp.freeze(True)
                ot.freeze(True)
            rx.freeze(True)
        self.set_static(static)

    def set_thaw(self, static: bool, thaw_meas: Optional[list[str]] = None):
        """
        Run thaw for all relevent childen.

        Parameters
        ----------
        static : bool
            The value to set `static` to for all relevent children.
        """
        for rx in self.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    for meas in fp.focal_planes:
                        if thaw_meas is not None and meas.meas_id not in thaw_meas:
                            continue
                        meas.thaw(static)
                    fp.thaw(static)
                ot.thaw(static)
            rx.thaw(static)


    def warm_pm_data(self, executor : Executor):
        """
        Warm up the `FocalPlane.pm_data` cache in parallel.
        This will set `_pm_data_stat` to the output of `_pm_data_dyn`
        for all `FocalPlane`s, overwriting whatever the current cached static
        value is regardless of if `FocalPlane.static` is `True` or if it was recently
        frozen on thawed.

        This should be used with caution.

        Parameters
        ----------
        executor : Executor
            Executor used to evaluate `_pm_data_dyn` concurrently.
            Doesn't need to be from `concurrent.futures` but must
            implement a compatible API.
        """
        all_meas = [
            meas
            for rx in self.receivers
            for ot in rx.optics_tubes
            for fp in ot.focal_planes
            for meas in fp.focal_planes
        ]
    
        # futures = [
        #     executor.submit(_get_pm, meas)
        #     for meas in all_meas
        # ]
        # wait(futures)
    
        # for meas, future in zip(all_meas, futures):
        #     meas._pm_data_stat = future.result()

        results = executor.map(_get_pm, all_meas)

        for meas, pm_data in zip(all_meas, results):
            meas._pm_data_stat = pm_data

    @property
    def parameters(self) -> Float[np.ndarray, "npar"]:
        return np.array(
            [
                self.pointing_models[par_map[0]].parameters[par.split("(")[0]]
                for par, par_map in zip(self.parameter_names, self.parameter_map)
            ]
        )

    def update_parameters(self, new_pars: Float[np.ndarray, "npar"]):
        """
        Update the pointing model and nuisance parameter values.

        The independent pointing model parameters are distributed to the
        corresponding `PointingModel` instances according to `parameter_map`.
        Parameters mapped to multiple pointing models are updated consistently
        across all associated epochs.

        If additional values are provided beyond the independent pointing
        model parameters, they are passed to `NuisanceParams.update`.

        Updating parameters modifies the underlying `PointingModel` instances
        and `NuisanceParams` in place. It does not explicitly freeze or thaw
        the derived quantities of the pointing system.

        Parameters
        ----------
        new_pars : Float[np.ndarray, "npar"]
            New parameter values. The first `len(parameter_names)` entries
            correspond to the independent pointing model parameters. Any
            remaining entries correspond to the flattened nuisance parameters.
        """
        npar = len(self.parameter_names)
        if len(new_pars) < npar:
            raise ValueError(
                f"Expected at least {npar} parameters, got {len(new_pars)}"
            )
        if len(new_pars) > npar + len(self.nuisance_params.params_flat):
            raise ValueError("Too many parameters supplied")

        for val, par, par_map in zip(
            new_pars[:npar], self.parameter_names, self.parameter_map
        ):
            name = par.split("(")[0]
            for idx in par_map:
                self.pointing_models[idx].parameters[name] = val
        if npar < len(new_pars):
            self.nuisance_params.update(new_pars[npar:])

    @property
    def n_data(self) -> int:
        n = 0
        for rx in self.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    if fp.pad:
                        continue
                    n += fp.ndet * fp.nobs
        return n

    @property
    def residuals(self) -> Float[np.ndarray, "n_data"]:
        n_data = self.n_data
        if "_residual_buffer" in self.__dict__ and len(self._residual_buffer) == n_data:
            resid = self._residual_buffer
        else:
            resid = np.empty(n_data)
            self._residual_buffer = resid
        i = 0
        for rx in self.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    if fp.pad:
                        continue
                    fpr = fp.residuals
                    resid[i : (i + len(fpr))] = fpr
                    i += len(fpr)
        if i != n_data:
            raise RuntimeError(f"Residual size mismatch: expected {n_data}, got {i}")
        return resid

    @property
    def chisq(self) -> float:
        chisq = 0.0
        for rx in self.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    if fp.pad:
                        continue
                    chisq += fp.chisq
        return chisq

    def objective(
        self, params: Float[np.ndarray, "npar"], nresid: int, cm_fac: float, executor : Executor, thaw_meas = None 
    ) -> float:
        """
        Evaluate the total chi-squared objective function.

        Updates the pointing model and nuisance parameters, thaws the
        derived quantities throughout the pointing system, and evaluates
        the total focal-plane chi-squared.

        A regularization penalty is added for the receiver common-mode
        transforms to discourage receiver-wide distortions. The penalty
        includes the translational and rotational components of each
        receiver common mode.

        If a `RuntimeError` or `AssertionError` occurs during evaluation,
        a large objective value of `1e6` is returned to indicate an
        unsuccessful evaluation.

        Parameters
        ----------
        params : Float[np.ndarray, "npar"]
            Parameter values to evaluate, including nuisance parameters
            when applicable.
        nresid : int
            Number of residual measurements used to scale the common-mode
            regularization penalty.
        cm_fac : float
            Multiplicative factor controlling the strength of the
            common-mode regularization penalty.
        executor : Executor
            Executor used to evaluate `_pm_data_dyn` concurrently.
            See `warm_pm_data` for details.

        Returns
        -------
        objective : float
            Total objective value, consisting of the focal-plane chi-squared
            and the receiver common-mode regularization penalty.
        """
        print(".", end="", flush=True)
        self.update_parameters(params)
        self.set_thaw(True, thaw_meas)
        self.warm_pm_data(executor)
        try:
            chisq = self.chisq
            if np.isfinite(chisq):
                for rx in self.receivers:
                    chisq += (
                        np.sum((cm_fac * nresid * rx.cm_transform.shift) ** 2)
                        + (cm_fac * nresid * rx.cm_transform.rot) ** 2
                    )
            return chisq
        except (RuntimeError, AssertionError):
            return 1e6

    def objective_resid(
        self, params: Float[np.ndarray, "npar"], nresid: int, cm_fac: float, executor : Executor, thaw_meas = None
    ) -> Float[np.ndarray, "nresid_full"]:
        """
        Evaluate the residual vector used by least-squares optimization.

        Updates the pointing model and nuisance parameters, thaws the
        derived quantities throughout the pointing system, and evaluates
        the focal-plane residuals.

        Receiver common-mode shifts and rotations are appended to the
        residual vector as regularization terms. These terms penalize
        receiver-wide distortions and are scaled by `cm_fac` and `nresid`.

        If a `RuntimeError` or `AssertionError` occurs during evaluation,
        an array filled with `1e3` is returned.

        Parameters
        ----------
        params : Float[np.ndarray, "npar"]
            Parameter values to evaluate, including nuisance parameters
            when applicable.
        nresid : int
            Number of focal-plane residual measurements used to scale the
            common-mode regularization terms. Also determines the length
            of the fallback residual vector if evaluation fails.
        cm_fac : float
            Multiplicative factor controlling the strength of the
            common-mode regularization terms.
        executor : Executor
            Executor used to evaluate `_pm_data_dyn` concurrently.
            See `warm_pm_data` for details.

        Returns
        -------
        objective_resid : Float[np.ndarray, "nresid"]
            Concatenated residual vector containing the focal-plane
            residuals followed by the receiver common-mode shift and
            rotation regularization terms.
        """
        print(".", end="", flush=True)
        self.update_parameters(params)
        self.set_thaw(True, thaw_meas)
        self.warm_pm_data(executor)
        try:
            resid = [
                self.residuals,
            ]
            for rx in self.receivers:
                resid += [
                    (cm_fac * nresid * rx.cm_transform.shift).ravel(),
                    np.atleast_1d(cm_fac * nresid * rx.cm_transform.rot).ravel(),
                ]
            return np.hstack(resid)
        except (RuntimeError, AssertionError):
            return np.full(nresid, 1e3)

    def log_prob(
        self, params: Float[np.ndarray, "npar"], nresid: int, cm_fac: float, executor : Executor
    ) -> float:
        """
        Calculate the log probability of the current parameter values.

        This is defined as -0.5 times the objective function evaluated
        at the supplied parameters.

        Parameters
        ----------
        params : Float[np.ndarray, "npar"]
            Parameter values to evaluate.
        nresid : int
            Number of residuals used to scale the common-mode penalty.
        cm_fac : float
            Scaling factor applied to the common-mode penalty.
        executor : Executor
            Executor used to evaluate `_pm_data_dyn` concurrently.
            See `warm_pm_data` for details.

        Returns
        -------
        log_prob : float
            Log probability of the supplied parameter values.
        """
        return -0.5 * self.objective(params, nresid, cm_fac, executor, None)

    @overload
    def search(
        self, field: Literal["pointing_models"], timestamp: float
    ) -> Optional[PointingModel]: ...

    @overload
    def search(
        self, field: Literal["receivers"], timestamp: float
    ) -> Optional[Receiver]: ...

    def search(
        self, field: Literal["pointing_models", "receivers"], timestamp: float
    ) -> Optional[PointingModel | Receiver]:
        """
        Get the first object associated with a specifc timestamp.

        Parameters
        ----------
        field: Literal["pointing_models", "receivers"]
            The field to search in.
            Valid options are: "receivers" and "pointing_models".
        timestamp : float
            The time to search for.

        Returns
        -------
        found : Optional[PointingModel | Receiver]
            The first matching Epoch.
            Will be None if no matches are found.
        """
        if field not in ["pointing_models", "receivers"]:
            raise ValueError(f"Invalud field to search in: {field}.")
        dat = getattr(self, field)
        epoch = self.era.find_epoch(timestamp)
        if epoch is None:
            return None
        hits = [d for d in dat if epoch.name in d.epoch]
        if len(hits) == 0:
            return None
        return hits[0]

    def set_pad(self, pad_epochs: list[str], pad_empty: bool):
        """
        Set the padding status of pointing models and focal planes.

        Padding is enabled for pointing models and receiver associated
        with epochs specified in `pad_epochs`.
        If `pad_empty`` is True empty focal planes are also padded,
        regardless of their receiver's epoch.

        Parameters
        ----------
        pad_epochs : list of str
            Epoch identifiers for which padding should be enabled.
        pad_empty : bool
            If True, enable padding for focal planes collections that
            contain no focal planes.
        """
        for pm in self.pointing_models:
            pm.pad = pm.epoch in pad_epochs
        for rx in self.receivers:
            pad = rx.epoch in pad_epochs
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    fp.pad = pad or (pad_empty and (len(fp.focal_planes) == 0))

    def load_pm_from_mdb(self, mdb: ManifestDb):
        """
        Load pointing model parameters from a ManifestDb into the current era.

        This method retrieves pointing model parameters for each epoch defined in
        `self.era.epochs` from the provided ManifestDb, validates that the database
        schema is compatible, and assigns the loaded parameters to
        `self.pointing_models`. The checks performed are:

          * The ManifestDb must use a supported scheme with required params
            ['obs:timestamp'].
          * Each epoch's covering intervals must all resolve to the same database
            match.
          * A pointing model must exist for every epoch.


        Note that here we also assume that the loaded pointing model has a flat structure
        where each element is a scalar that maps to a pointing model parameter.
        Note also here that for pointing models that join multiple epochs we use
        the value from the latest epoch.

        Parameters
        ----------
        mdb : ManifestDb
            Database object containing pointing model data indexed by
            'obs:timestamp'.

        Raises
        ------
        ValueError
            If the ManifestDb scheme is unsupported.
            If no database matches are found for an epoch.
            If different intervals within an epoch resolve to inconsistent matches.
        """
        loader = SuperLoader()
        mspec = MetadataSpec.from_dict({"db": mdb, "unpack": "pointing_model"})
        if mdb.scheme.get_required_params() != ["obs:timestamp"]:
            raise ValueError("Unsupported scheme for pointing model db!")
        for epoch in self.era.epochs:
            matches = []
            for interval in epoch.covers:
                matches += [
                    mdb.match({"obs:timestamp": interval.start}),
                    mdb.match({"obs:timestamp": interval.stop - 1}),
                ]
            if len(matches) == 0:
                raise ValueError(f"No matches for epoch: {str(epoch)}")
            if not all(x == matches[0] for x in matches):
                raise ValueError(
                    f"Not all intervals have the same match in epoch: {str(epoch)}"
                )
            pm_aman = loader.load_one(
                mspec, {"obs:timestamp": epoch.covers[0].start}, []
            )
            pm_dict = {key: pm_aman[key] for key in pm_aman.keys()}
            pm = self.search("pointing_models", epoch.covers[0].start)
            if pm is None:
                raise ValueError(
                    f"Can't find pointing model for this epoch: {epoch.name}"
                )
            pm.parameters = pm_dict
        # Force consistancy
        self.update_parameters(self.parameters)

    @classmethod
    def empty(cls, era: epochs.Era, cfg: Namespace, calender_str: str) -> Self:
        """
        Create an empty pointing system from an observing era.

        This initializes a `PointingModel` and `Receiver` for each epoch,
        with empty focal-plane data constructed from the epoch's
        `ws_mapping`. Pointing models and receivers are then combined
        according to the configured epoch groups.

        Parameters
        ----------
        era : epochs.Era
            Era containing the epochs used to construct the pointing system.
            Each epoch must contain consistent `ws_mapping` data.
        cfg : Namespace
            Configuration containing the receiver name, epoch groups,
            pointing model settings, and focal plane options.
        calender_str : str
            A serialized calendar yaml string.

        Returns
        -------
        pointing_system : PointingSystem
            A pointing system containing the initialized pointing models
            and receivers.

        Raises
        ------
        ValueError
            If `era` has missing or inconsistent `ws_mapping` data.
            If any groups contain overlapping epochs.
            If any groups contais an unknown epoch.
            If an unknown pointing model parameter is passed.
        """
        if not era.check_data("ws_mapping", True):
            raise ValueError(
                "Era is either missing or has inconsistent values for ws_mapping"
            )
        nuisance_params = NuisanceParams({}, [], [])
        rxs = []
        pms = []
        for epoch in era.epochs:
            pms += [
                PointingModel(
                    epoch.name, {}, nuisance_params, cfg.force_zero_roll, False
                )
            ]
            ws_map = epoch.covers[0].data["ws_mapping"]
            ots = []
            for ot, arrays in ws_map.items():
                fps = []
                for _, (_, ufm) in arrays.items():
                    fps += [
                        FocalPlaneCollection(
                            ufm,
                            [],
                            DetectorOffsets.empty("xieta"),
                            DetectorOffsets.empty("fp"),
                            DetectorOffsets.empty("ot"),
                            False,
                            False,
                            True,
                            cfg.fake_gamma,
                        )
                    ]
                ots += [OpticsTube(ot, fps, list(arrays.keys()), False, False)]
            rxs += [Receiver(cfg.tel, epoch.name, ots, False, False)]

        def _group_epochs(dat, groups, desc, check_fields):
            if len(groups) == 0:
                return dat
            u, c = np.unique(np.concatenate(groups), return_counts=True)
            ng = [d for d in dat if d.epoch not in u]

            if np.any(c > 1):
                raise ValueError(f"Some {desc} groups have overlapping epochs!")

            ga = []
            for grp in groups:
                if len(grp) == 0:
                    continue
                for field in check_fields:
                    good = reduce(
                        epochs.OP_MAP["&"],
                        [
                            e.check_data(field, True)
                            for e in era.epochs
                            if e.name in grp
                        ],
                    )
                    if good:
                        allf = [
                            e._internal.data[field] for e in era.epochs if e.name in grp
                        ]
                        good = all(x == allf[0] for x in allf)
                    if not good:
                        raise ValueError(
                            f"Inconsistent {field} values in {desc} group {grp}"
                        )
                in_group = [d for d in dat if d.epoch in grp]
                if len(in_group) < len(grp):
                    raise ValueError(f"Unknown epoch in {desc} group {grp}")
                new = in_group[0]
                new.epoch = "+".join(d.epoch for d in in_group)
                ga.append(new)

            return ng + ga

        rxs = _group_epochs(rxs, cfg.rx_groups, "receiver", ["ws_mapping"])
        pms = _group_epochs(pms, cfg.pm_groups, "pointing model", [])

        par_names = []
        for pm in pms:
            pm_ver = cfg.pm_ver_override.get(pm.epoch, cfg.pm_ver)
            pars = deepcopy(param_defaults[pm_ver])
            pars.update(cfg.pm_defaults_overrides[pm_ver])
            par_names += [
                name
                for name, val in pars.items()
                if isinstance(val, (int, float, np.number))
            ]
            pars["version"] = pm_ver
            pm.parameters = pars
        par_names = np.unique(par_names).tolist()
        par_map = [np.arange(len(pms), dtype=int).tolist() for _ in par_names]
        pm_epochs = [pm.epoch for pm in pms]
        par: str
        for par, groups in cfg.par_groups.items():
            if par not in par_names:
                raise ValueError(f"Unknown parameter {par}")
            u, c = np.unique(np.concatenate(groups), return_counts=True)
            if np.any(c > 1):
                raise ValueError(f"Overlapping epochs in par group for {par}")
            if np.any(~np.isin(u, pm_epochs)):
                raise ValueError(f"Unknown epoch in par_group for {par}")
            par_idx = par_names.index(par)
            for grp in groups:
                epc_idxs = [pm_epochs.index(epc) for epc in grp]
                par_map[par_idx] = [i for i in par_map[par_idx] if i not in epc_idxs]
                par_names += [par + "(" + ",".join(grp) + ")"]
                par_map += [[epc_idxs]]
        n_mapped = [len(pmap) for pmap in par_map]
        par_names = tuple(name for name, n in zip(par_names, n_mapped) if n > 0)
        par_map = tuple(np.array(pmap) for pmap, n in zip(par_map, n_mapped) if n > 0)

        return cls(
            era,
            tuple(pms),
            tuple(rxs),
            nuisance_params,
            par_names,
            par_map,
            calender_str,
            {},
        )

    def set_pm_to_defaults(self, cfg):
        params = self.parameters.copy()
        def_pars = []
        for pm in self.pointing_models:
            pm_ver = cfg.pm_ver_override.get(pm.epoch, cfg.pm_ver)
            pars = deepcopy(param_defaults[pm_ver])
            pars.update(cfg.pm_defaults_overrides[pm_ver])
            def_pars += [pars]
        for i, (p_idx, name) in enumerate(
            zip(self.parameter_map, self.parameter_names)
        ):
            pars = def_pars[p_idx[0]]
            par_name = name.split("(")[0]
            params[i] = pars[par_name]
        self.update_parameters(params)

    def save(self, f: h5py.File, path: str, overwrite: bool = True):
        """
        Save the pointing system to an HDF5 file.

        The receiver-to-pointing-model relationship is stored explicitly as
        an integer array. Element `i` of `receiver_pointing_model` gives
        the index of the pointing model that should be passed to
        `Receiver.load` for receiver `i`.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file to save to.
        path : str
            Path within the HDF5 file at which to save the pointing system.
        overwrite : bool, default: True
            Whether to overwrite an existing pointing system at ``path``.

        Raises
        ------
        ValueError
            If there are no pointing models but receivers are present.
            If a receiver cannot be associated with a pointing model.
        """
        group = _create_group(f, path, overwrite=overwrite)
        _add_attrs(
            group,
            {
                "calender_str": self.calender_str,
                "era": self.era.name,
            },
        )
        state_group = _create_group(f, f"{path}/state", overwrite=overwrite)
        _add_attrs(state_group, self.state_meta)
        _ = _create_group(f, f"{path}/pointing_models", overwrite=overwrite)
        self.nuisance_params.save(f, f"{path}/nuisance", overwrite=overwrite)
        for idx, pointing_model in enumerate(self.pointing_models):
            pointing_model.save(
                f,
                f"{path}/pointing_models/{idx}_{pointing_model.epoch}",
                overwrite=overwrite,
            )
        _ = _create_group(f, f"{path}/receivers", overwrite=overwrite)
        for rx_idx, receiver in enumerate(self.receivers):
            receiver.save(
                f,
                f"{path}/receivers/{rx_idx}_{receiver.name}_{receiver.epoch}",
                self.pointing_models,
                overwrite=overwrite,
            )
        group.attrs["parameter_names"] = np.asarray(
            self.parameter_names, dtype=h5py.string_dtype()
        )
        parameter_map_group = cast(
            h5py.Group, _create_group(f, f"{path}/parameter_map", overwrite=overwrite)
        )
        for idx, par_map in enumerate(self.parameter_map):
            parameter_map_group.create_dataset(
                str(idx), data=np.asarray(par_map, dtype=np.int64)
            )

    @classmethod
    def load(cls, f: h5py.File, path: str) -> Self:
        """
        Load a pointing system from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file containing the pointing system.
        path : str
            Path within the HDF5 file containing the pointing system.

        Returns
        -------
        PointingSystem
            The reconstructed pointing system.
        """
        group = f[path]
        calender_str = cast(str, group.attrs["calender_str"])
        era_str = cast(str, group.attrs["era"])
        cal = epochs.Calendar.load(calender_str)
        era = cal.eras[era_str]
        parameter_names = tuple(
            str(name) for name in cast(Iterable[object], group.attrs["parameter_names"])
        )
        state_group = f[f"{path}/state"]
        state_meta = {key: val for key, val in state_group.attrs.items()}

        parameter_map_group = cast(h5py.Group, f[f"{path}/parameter_map"])
        parameter_map = tuple(
            np.asarray(parameter_map_group[name], dtype=np.int64)
            for name in sorted(parameter_map_group.keys())
        )

        if f"{path}/nuisance" in f:
            nuisance_params = NuisanceParams.load(f, f"{path}/nuisance")
        else:
            nuisance_params = NuisanceParams({}, [], [])
        pm_group = cast(h5py.Group, f[f"{path}/pointing_models"])
        pointing_models = tuple(
            PointingModel.load(f, f"{path}/pointing_models/{name}", nuisance_params)
            for name in sorted(pm_group.keys())
        )

        rx_group = cast(h5py.Group, f[f"{path}/receivers"])
        receivers = tuple(
            Receiver.load(f, f"{path}/receivers/{name}", pointing_models)
            for name in sorted(rx_group.keys())
        )

        return cls(
            era=era,
            pointing_models=pointing_models,
            receivers=receivers,
            nuisance_params=nuisance_params,
            parameter_names=parameter_names,
            parameter_map=parameter_map,
            calender_str=calender_str,
            state_meta=state_meta,
        )

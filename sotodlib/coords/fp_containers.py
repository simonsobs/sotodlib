"""
New Focal Plane contianers.
Supports transparent applications of all transforms including pointing model.
Calculated transforms can either be treated at properties for ease of
calculations or as static attributes for loading saved results.
"""
# TODO: Function to update pointing model pars that is aware of joint fits
# TODO: Serialization + ManifestDb
# TODO: Reprs

from __future__ import annotations
from typing import Optional, Any, cast
from dataclasses import dataclass, field
from jaxtyping import Float, Shaped
import numpy as np
import megham.transform as mt
import megham.utils as mu
from functools import cached_property
from copy import deepcopy
from scipy.optimize import minimize
from sotodlib.coords.pointing_model import apply_pointing_model
from so3g.proj import quat
from sotodlib.core import AxisManager, IndexAxis 
from sotodlib.utils import epochs
from sotodlib.io.metadata import SuperLoader
from sotodlib.core.metadata import MetadataSpec
from sotodlib.core.metadata import ManifestDb

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

def _transformed(self : FocalPlane | FocalPlaneCollection, attr_name : str):
    to_ret = deepcopy(self.template)
    to_ret.xyz = mt.apply_transform(self.template.xyz, self.transform.affine, self.transform.shift)
    to_ret.center = mt.apply_transform(self.template.center, self.transform.affine, self.transform.shift)
    if self.autofreeze:
        setattr(self, attr_name, to_ret)
    return to_ret

def _get_aff(self : FocalPlane | FocalPlaneCollection, data_name : str, attr_name : str):
    affine, shift = mt.get_affine_two_stage(
        self.template.xyz[:, :2],
        getattr(self, data_name).xyz[:, :2],
        self.weights[:, 0],
    )
    to_ret = Transform.identity()
    to_ret.shift[:2] = shift
    to_ret.affine[:2, :2] = affine
    if not self.fake_gamma:
        to_ret.affine[-1, -1], to_ret.shift[-1] = gamma_fit(
                    self.template.xyz[:, 2], getattr(self, data_name).xyz[:, 2]
                )
    if self.autofreeze:
        setattr(self, attr_name, to_ret)
    return to_ret

class StaticOrCached():
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

    def __get__(self, obj, owner):
        if obj is None:
            return self

        if obj.static:
            return getattr(obj, f"_{self.name}_stat")
        return getattr(obj, f"_{self.name}_dyn")


@dataclass
class PointingModel:
    """
    Wrapper around a telescope pointing model and its application to
    detector offsets. In the future this should also be able to handle
    measurements from things like the starcamera.

    Attributes 
    ----------
    parameters : dict
        Dictionary of pointing-model parameters accepted by
        `apply_pointing_model`. An empty dictionary corresponds to an
        identity pointing model and leaves detector offsets unchanged.
    force_zero_roll : bool
        If True, ignore the encoder roll angle when applying the pointing
        model and assume a zero-roll boresight solution.
    """
    parameters : dict
    force_zero_roll : bool

    def apply_det(
        self,
        det_off: DetectorOffsets,
        det_enc: DetectorOffsets,
        fake_gamma: bool,
    ) -> DetectorOffsets:
        """
        Apply the pointing model correction to a set of detector offsets.
    
        The correction is computed from the supplied encoder values using
        `apply_pointing_model` and then propagated to each detector through
        quaternion rotations. The returned detector offsets remain in the
        `xieta` coordinate system but are corrected for the boresight pointing
        model specified by this instance.

        If the pointing model contains no parameters, the input detector
        offsets are returned unchanged.
    
        When `force_zero_roll=True`, the encoder roll angle is ignored during
        the correction and a zero-roll boresight solution is assumed.
    
        Parameters
        ----------
        det_off : DetectorOffsets
            Detector offsets in the `xieta` coordinate system. These are the
            detector positions to which the pointing correction will be
            applied.
        det_enc : DetectorOffsets
            Telescope encoder values in the `horizon` coordinate system.
            These provide the boresight azimuth, elevation, and roll used to
            evaluate the pointing model.
        fake_gamma : bool
            If True, ignore detector polarization angles and propagate only
            detector position (`xi`, `eta`). If False, propagate the full
            (`xi`, `eta`, `gamma`) coordinate set.
    
        Returns
        -------
        det_pm : DetectorOffsets
            A copy of `det_off` with the pointing-model correction applied.
            The returned object has the same metadata and detector ordering
            as the input but contains corrected detector coordinates.
    
        Raises
        ------
        ValueError
            If `det_off` and `det_enc` do not contain the same number of
            detectors.
        """
        if len(self.parameters) == 0:
            return det_off
        if len(det_enc.xyz) != len(det_off.xyz):
            raise ValueError("Detector offsets and encoder values must have same shape!")
        bs = apply_pointing_model(None, self.parameters, det_off.ancil, False)
        if fake_gamma:
            q_fp = quat.rotation_xieta(det_off.xi, det_off.eta)
        else:
            q_fp = quat.rotation_xieta(det_off.xi, det_off.eta, det_off.gamma) # type: ignore
        to_ret = deepcopy(det_off)
        xi, eta, gamma = quat.decompose_xieta(
            ~quat.euler(2, bs.roll)
            * ~quat.rotation_lonlat(-1*bs.az, bs.el) # type: ignore
            * quat.rotation_lonlat(-1 * det_enc.az, det_enc.el)
            * quat.euler(2, (not self.force_zero_roll) * det_enc.roll)
            * q_fp
        )
        to_ret.xyz = np.column_stack((xi, eta, gamma))
        return to_ret

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
    """
    shift : Float[np.ndarray, "ndim"]
    affine : Float[np.ndarray, "ndim ndim"]

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

@dataclass
class DetectorOffsets:
    """
    Class for storing detector offsets in either `xieta` or `horizon` coordinates.
    `xi`, `eta`, `gamma`, `az`, `el`, and `roll` are included as properties for convenience.
    They will raise a `ValueError` if called with the wrong coordsys set.

    Attributes
    ----------
    xyz : Float[np.ndarray, "3 ndet"]
        The `x` axis of the detector offsets.
        For `xieta` this is `xi`.
        For `horizon` this is `az`.
        The `y` axis of the detector offsets.
        For `xieta` this is `eta`.
        For `horizon` this is `el`.
        The `z` axis of the detector offsets.
        For `xieta` this is `gamma`.
        For `horizon` this is `roll`.
    det_id : Shaped[np.ndarray, "ndet"]
        The detector id for each element in `xyz`.
    split : Shaped[np.ndarray, "ndet"]
        A string used to split up detectors when plotting.
        Nominally this should be to seperate the four detectors
        behind the same feedhorn.
    center : Optional[Float[np.ndarray, "3"]]
        The center of the detector offsets.
        Can be `None` if unknown.
    coordsys : str
        The coordinate system that these are in.
        Should be `xieta` or `horizon`.
    """
    xyz : Float[np.ndarray, "ndet 3"]
    det_id : Shaped[np.ndarray, "ndet"]
    split : Shaped[np.ndarray, "ndet"]
    center : Optional[Float[np.ndarray, "3"]]
    coordsys : str

    def __post_init__(self):
        if self.coordsys not in ("xieta", "horizon"):
            raise ValueError(f"Invalid coordsys {self.coordsys}")

    def __setattr__(self, name: str, value: Any, /) -> None:
        if name in ("xyz"):
            self.__dict__.pop("ancil", None)
        return super().__setattr__(name, value)


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
        ancil.wrap("boresight_enc", -1*np.rad2deg(self.roll)) # for SATs
        return ancil


@dataclass
class FocalPlane:
    """
    Container for a single focalplane measurement and its derived transforms.

    A `FocalPlane` represents one measured realization of a focal plane
    (typically a single UFM) together with the information needed to compare
    it against a reference template.  The class stores the measured detector
    positions, encoder values, detector weights, and an optional pointing
    model. From these inputs it can derive several useful properties:

    - Pointing model corrected detector positions (`data_pm`).
    - The affine transform mapping the template onto the measured data
      (`transform`).
    - The transformed template (`transformed`).
    - The affine transform after applying the pointing model
      (`pm_transform`).
    - The transformed template in the pointing model corrected frame
      (`pm_transformed`).

    Derived quantities can be evaluated dynamically or stored as static
    values. When `static=True`, the public properties return the frozen
    values stored by `freeze`. When `static=False`, the quantities
    are recomputed on access. If `autofreeze=True`, newly computed values
    are automatically cached into the corresponding static fields.

    Attributes
    ----------
    name : str
        Human-readable identifier for the focal plane (ie. UFM name).
    meas_id : str
        Identifier for the measurement from which this focal plane was
        constructed.
    data : DetectorOffsets
        Measured detector offsets in focal-plane coordinates.
    enc : DetectorOffsets
        Encoder values associated with the measurement. These are used when
        applying the pointing model and are typically stored in the
        `horizon` coordinate system.
    weights : Float[np.ndarray, "ndet"]
        Per-detector weights used when fitting transforms and combining
        measurements.
    template : DetectorOffsets
        Reference detector layout used as the transform source.
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
    data_pm : DetectorOffsets
        Measured detector offsets after application of the pointing model.
    transform : Transform
        Best-fit affine transform mapping `template` to `data`.
    transformed : DetectorOffsets
        Template detector offsets after applying `transform`.
    pm_transform : Transform
        Best-fit affine transform mapping `template` to `data_pm`.
    pm_transformed : DetectorOffsets
        Template detector offsets after applying `pm_transform`.
    resid : Float[np.ndarray, "ndet 3"]
        Residuals between `data` and `transformed`.
    pm_resid : Float[np.ndarray, "ndet 3"]
        Residuals between `data_pm` and `pm_transformed`.
    """
    name : str
    meas_id : str
    data : DetectorOffsets
    enc : DetectorOffsets
    weights : Float[np.ndarray, "ndet"]
    template : DetectorOffsets
    pointing_model : PointingModel
    static : bool
    autofreeze : bool
    fake_gamma : bool
    data_pm : DetectorOffsets = cast(DetectorOffsets,StaticOrCached("data_pm"))
    transform : Transform = cast(Transform, StaticOrCached("transform"))
    transformed : DetectorOffsets = cast(DetectorOffsets,StaticOrCached("transformed"))
    pm_transform : Transform = cast(Transform, StaticOrCached("pm_transform"))
    pm_transformed : DetectorOffsets = cast(DetectorOffsets, StaticOrCached("pm_transformed"))
    _data_pm_stat: DetectorOffsets = field(init=False)
    _transform_stat : Transform = field(init=False)
    _transformed_stat : DetectorOffsets = field(init=False)
    _pm_transform_stat : Transform = field(init=False)
    _pm_transformed_stat : DetectorOffsets = field(init=False)

    @classmethod
    def from_aman(cls, name : str, meas_id : str, aman : AxisManager, template : DetectorOffsets, pointing_model : PointingModel, autofreeze: bool=False, fake_gamma: bool = True, weight_factor: float = 1000.) -> FocalPlane:
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
        template : DetectorOffsets
            Template focal plane.
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
        _, msk, template_msk = np.intersect1d(
            np.array(aman.det_info.det_id), template.det_id, return_indices=True
        )
        mapping = np.argsort(np.argsort(template.det_id[template_msk]))
        srt = np.argsort(np.array(aman.det_info.det_id[msk]))

        # Setup data
        data = deepcopy(template)
        xi = np.array(aman.focal_plane.xi)
        eta = np.array(aman.focal_plane.eta)
        gamma = np.array(aman.focal_plane.gamma) if not fake_gamma else np.nan*np.ones_like(xi)
        data.xyz[template_msk] = np.column_stack((xi, eta, gamma))[msk][srt][mapping]
        data.xyz[~template_msk] = np.nan

        # Setup enc
        enc = deepcopy(template)
        enc.coordsys = "horizon"
        az = np.array(aman.focal_plane.az) if "az" in aman.focal_plane else np.deg2rad(np.array(aman.obs_info.az_center))*np.ones_like(xi)
        el = np.array(aman.focal_plane.el) if "el" in aman.focal_plane else np.deg2rad(np.array(aman.obs_info.el_center))*np.ones_like(xi)
        roll = np.array(aman.focal_plane.roll) if "roll" in aman.focal_plane else np.deg2rad(np.array(aman.obs_info.roll_center))*np.ones_like(xi)
        enc.xyz[template_msk] = np.column_stack((az, el, roll))[msk][srt][mapping]
        enc.xyz[~template_msk] = np.nan

        # Compute weights
        aff, sft = mt.get_rigid(
                data.xyz[:, :2], template.xyz[template_msk, :2]
            )
        aligned = mt.apply_transform(data.xyz[:, :2], aff, sft)
        xieta_spacing = mu.estimate_spacing(template.xyz[:, :2])
        # For gamma rather than the spacing in real space we want the difference between bins
        # This is a rough estimate but good enough for us
        gamma_spacing = np.percentile(np.diff(np.sort(template.xyz[:, 2])), 99.9)
        spacing = np.array([xieta_spacing, xieta_spacing, gamma_spacing])

        if not fake_gamma:
            gscale, gsft = gamma_fit(
                data.xyz[:, 2], template.xyz[template_msk, 2]
            )
            weights = mu.gen_weights(
                np.column_stack((aligned, gscale * data.xyz[:, 2] + gsft)),
                template.xyz[template_msk],
                spacing.ravel() / weight_factor,
            )
        else:
            weights = mu.gen_weights(
                aligned,
                template.xyz[template_msk, :2],
                spacing[:2].ravel() / weight_factor,
            )

        # ~1 sigma cut
        weights[weights < 0.61] = np.nan

        return cls(name, meas_id, data, enc, weights, template, pointing_model, False, autofreeze, fake_gamma)


    def freeze(self, static : bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        self._data_pm_stat = self._data_pm_dyn 
        self._transform_stat = self._transform_dyn
        self._transformed_stat = self._transformed_dyn
        self._pm_transform_stat = self._pm_transform_dyn
        self._pm_transformed_stat = self._pm_transformed_dyn
        self.static = static

    @property
    def _data_pm_dyn(self) -> DetectorOffsets:
        to_ret = self.pointing_model.apply_det(self.data, self.enc, self.fake_gamma)
        if self.autofreeze:
            self._data_pm_stat = to_ret
        return to_ret

    @property
    def _transform_dyn(self) -> Transform:
        return _get_aff(self, "data", "_transform_stat")

    @property
    def _transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_transformed_stat")

    @property
    def _pm_transform_dyn(self) -> Transform:
        return _get_aff(self, "pm_data", "_pm_transform_stat")

    @property
    def _pm_transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_pm_transformed_stat")

    @property
    def resid(self):
        return self.data.xyz - self.transformed.xyz

    @property
    def pm_resid(self):
        return self.data_pm.xyz - self.pm_transformed.xyz


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
    positions (`FocalPlane.data_pm`) from each measurement.
    Detector locations and weights are combined on a per-detector
    basis, missing values should be filled with `nan`s.
    
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
    data : DetectorOffsets
        Weighted-average detector offsets constructed from the
        pointing-model-corrected measurements in `focal_planes`.
    weights : Float[np.ndarray, "ndet"]
        Combined detector weights obtained by summing the weights from all
        constituent measurements.
    transform : Transform
        Best-fit affine transform mapping `template` to `data`.
    transformed : DetectorOffsets
        Template detector offsets after applying `transform`.
    resid : Float[np.ndarray, "ndet 3"]
        Residuals between `data` and `transformed`.
    """
    name : str
    focal_planes : list[FocalPlane] 
    template : DetectorOffsets
    static : bool
    autofreeze : bool
    pad : bool
    fake_gamma : bool
    data : DetectorOffsets = cast(DetectorOffsets, StaticOrCached("data"))
    weights : Float[np.ndarray, "ndet"] = cast(Float[np.ndarray, "ndet"], StaticOrCached("weights"))
    transform : Transform = cast(Transform, StaticOrCached("transform"))
    transformed : DetectorOffsets = cast(DetectorOffsets, StaticOrCached("transformed"))
    _data_stat: DetectorOffsets = field(init=False)
    _weights_stat : Float[np.ndarray, "ndet"] = field(init=False)
    _transform_stat : Transform = field(init=False)
    _transformed_stat : DetectorOffsets = field(init=False)

    def freeze(self, static : bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        self._data_stat = self._data_dyn 
        self._weights_stat = self._weights_dyn
        self._transform_stat = self._transform_dyn
        self._transformed_stat = self._transformed_dyn
        self.static = static

    @property
    def _data_dyn(self) -> DetectorOffsets:
        to_ret = deepcopy(self.template)
        dat_full = np.array([fp.data_pm.xyz * fp.weights[..., None] for fp in self. focal_planes])
        to_ret.xyz = np.nansum(dat_full, axis=-1)/self.weights[..., None]
        to_ret.center = None
        if self.autofreeze:
            self._data_stat = to_ret
        return to_ret

    @property
    def _weights_dyn(self) -> Float[np.ndarray, "ndet"]:
        weights_full = np.array([fp.weights for fp in self. focal_planes])
        to_ret = np.nansum(weights_full, axis=-1)
        if self.autofreeze:
            self._weights_stat = to_ret
        return to_ret

    @property
    def _transform_dyn(self) -> Transform:
        return _get_aff(self, "data", "_transform_stat")

    @property
    def _transformed_dyn(self) -> DetectorOffsets:
        return _transformed(self, "_transformed_stat")

    @property
    def resid(self):
        return self.data.xyz - self.transformed.xyz

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
    allpad : bool
        Flag indicating whether all focal-plane collections in the optics
        tube are padded and should therefore be excluded from common-mode
        calculations.
    cm_transform : Transform
        Common-mode transform derived from the transforms of all
        non-padded focal-plane collections.
    cm_transform_norx : Transform
        Copy of the optics-tube common-mode transform with a receiver
        common mode removed. This quantity is not computed dynamically and
        is updated only through `remove_rx`.
    """
    name : str
    focal_planes : list[FocalPlaneCollection]
    wafer_slots : list[str]
    static : bool
    autofreeze : bool
    allpad : bool
    cm_transform : Transform = cast(Transform, StaticOrCached("cm_transform"))
    cm_transform_norx : Transform = field(init=False)
    _cm_transform_stat : Transform = field(init=False)

    def __getitem__(self, key):
        if key in self.wafer_slots:
            return self.focal_planes[self.wafer_slots.index(key)]
        fp_names = [fp.name for fp in self.focal_planes]
        if key in fp_names:
            return self.focal_planes[fp_names.index(key)]
        raise IndexError(f"Invalid key: {key}")

    def freeze(self, static : bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        self._cm_transform_stat = self._cm_transform_dyn
        self.static = static

    @property
    def _cm_transform_dyn(self) -> Transform:
        transforms = [fp.transform.affine for fp in self.focal_planes if fp.pad is False]
        shifts = [fp.transform.shift for fp in self.focal_planes if fp.pad is False]
        allpad = False
        if len(transforms) == 0:
            allpad = True
            iden = Transform.identity()
            cm_shift, cm_aff = iden.shift, iden.affine
        elif len(transforms) == 1:
            cm_aff, cm_shift = transforms[0], shifts[0]
        else:
            cm_aff, cm_shift = mt.get_common_mode(transforms, shifts, -2, False)
        to_ret = Transform(cm_shift, cm_aff)
        if self.autofreeze:
            self._cm_transform_stat = to_ret
            self.allpad = allpad
        return to_ret

    def remove_rx(self, rx_cm : Transform):
        """
        Remove the receiver common-mode transform from the optics tube
        common-mode transform.
    
        Unlike `cm_transform`, the receiver-corrected transform is not a
        dynamic property and is updated only when this method is called.

        If `allpad=True`, no valid focal-plane collections are available for
        this optics tube and `cm_transform_norx` is set to the identity
        transform.
    
        Parameters
        ----------
        rx_cm : Transform
            Receiver common-mode transform to remove from the current
            optics tube common mode.
        """
        if self.allpad:
            self.cm_transform_norx = Transform.identity()
        aff, sft = mt.decompose_transform(
                self.cm_transform.affine,
                self.cm_transform.shift,
                rx_cm.affine,
                rx_cm.shift,
            )
        self.cm_transform_norx = Transform(sft, aff)

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
    """
    name : str
    optics_tubes : list[OpticsTube]
    static : bool
    autofreeze : bool
    cm_transform : Transform = cast(Transform, StaticOrCached("cm_transform"))
    _cm_transform_stat : Transform = field(init=False)

    def __getitem__(self, key):
        ot_names = [ot.name for ot in self.optics_tubes]
        if key in ot_names:
            return self.optics_tubes[ot_names.index(key)]
        raise IndexError(f"Invalid key: {key}")

    def freeze(self, static : bool):
        """
        Save the current dynamic properties to the static fields.

        Parameters
        ----------
        static : bool
            What to set `self.static` to after freezing.
        """
        self._cm_transform_stat = self._cm_transform_dyn
        self.static = static

    @property
    def _cm_transform_dyn(self) -> Transform:
        transforms = [ot.cm_transform.affine for ot in self.optics_tubes if ot.allpad is False]
        shifts = [ot.cm_transform.shift for ot in self.optics_tubes if ot.allpad is False]
        if len(transforms) == 0:
            iden = Transform.identity()
            cm_shift, cm_aff = iden.shift, iden.affine
        elif len(transforms) == 1:
            cm_aff, cm_shift = transforms[0], shifts[0]
        else:
            cm_aff, cm_shift = mt.get_common_mode(transforms, shifts, -2, True)
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


@dataclass
class PointingSystem:
    """
    Collection of epoch-dependent pointing models and receiver data.

    A `PointingSystem` maintains a one-to-one correspondence between the
    epochs in an `epochs.Era`, a set of `PointingModel`
    instances, and a set of `Receiver` instances. The ordering of these
    must be the same.

    Attributes
    ----------
    era : epochs.Era
        Collection of observing epochs represented by the system.
    pointing_models : tuple[PointingModel, ...]
        Pointing model associated with each epoch.
    receivers : tuple[Receiver, ...]
        Receiver and focal-plane data associated with each epoch.
    chisq : float
        The chisq of the current system.
        This computes the chisq from the residuals for each `FocalPlaneCollection`,
        not the individual `FocalPlane` measurements.
    """
    era : epochs.Era
    pointing_models : tuple[PointingModel, ...]
    receivers : tuple[Receiver, ...]

    def __post_init__(self):
        if len(self.era.epochs) != len(self.pointing_models) or len(self.era.epochs) != len(self.receivers):
            raise ValueError("Should have one pointing model and receiver per epoch!")

    @property
    def chisq(self) -> float:
        chisq = 0.
        for rx in self.receivers:
            for ot in rx.optics_tubes:
                for fp in ot.focal_planes:
                    chisq += np.nansum((fp.weights * fp.resid)**2).item()
        return chisq

    def load_pm_from_mdb(self, mdb : ManifestDb):
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
        mspec = MetadataSpec.from_dict({"db" : mdb, "unpack":"pointing_model"})
        if mdb.scheme.get_required_params() != ['obs:timestamp']:
            raise ValueError("Unsupported scheme for pointing model db!")
        for i, epoch in enumerate(self.era.epochs):
            matches = []
            for interval in epoch.covers:
                matches += [mdb.match({"obs:timestamp":interval.start}), mdb.match({"obs:timestamp":interval.stop-1})]
            if len(matches) == 0:
                raise ValueError(f"No matches for epoch: {str(epoch)}")
            if not all(x == matches[0] for x in matches):
                raise ValueError(f"Not all intervals have the same match in epoch: {str(epoch)}")
            pm_aman = loader.load_one(mspec, {"obs:timestamp":epoch.covers[0].start}, [])
            pm_dict = {key : pm_aman[key] for key in pm_aman.keys()}
            self.pointing_models[i].parameters = pm_dict

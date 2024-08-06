import logging
import os
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional

import megham.transform as mt
import megham.utils as mu
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import transform
from sotodlib.coords import optics as op
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset, read_dataset

logger = logging.getLogger("finalize_focal_plane")


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        dset.attrs[k] = v


@dataclass
class Transform:
    shift: NDArray[np.floating]  # (ndim,)
    affine: NDArray[np.floating]  # (ndim, ndim)
    scale: NDArray[np.floating] = field(init=False)  # (ndim,)
    shear: float = field(init=False)
    rot: float = field(init=False)

    def __post_init__(self):
        self.decompose()

    @classmethod
    def identity(cls):
        return Transform(np.zeros(3), np.eye(3))

    @classmethod
    def from_split(cls, shift, xieta_affine, gamma_scale):
        affine = np.eye(len(xieta_affine) + 1)
        affine[: len(xieta_affine), : len(xieta_affine)] = xieta_affine
        affine[-1, -1] = gamma_scale

        return Transform(shift, affine)

    def decompose(self):
        xieta_affine = self.affine[:2, :2]
        gamma_scale = self.affine[-1, -1]
        scale, shear, rot = mt.decompose_affine(xieta_affine)
        self.scale = np.array((*scale, gamma_scale))
        self.shear = shear.item()
        self.rot = mt.decompose_rotation(rot)[-1]

    def save(self, f, path, append=""):
        if path not in f:
            f.create_group(path)
        _add_attrs(
            f[path],
            {
                f"shift{append}": self.shift,
                f"scale{append}": self.scale,
                f"shear{append}": self.shear,
                f"rot{append}": self.rot,
                f"affine{append}": self.affine,
            },
        )

    @classmethod
    def load(cls, group, append=""):
        shift = group.attrs[f"shift{append}"]
        affine = group.attrs[f"affine{append}"]

        return Transform(shift, affine)


@dataclass
class Template:
    det_ids: NDArray[np.str_]  # (ndet,)
    fp: NDArray[np.floating]  # (ndet, ndim)
    optical: NDArray[np.bool_]  # (ndet,)
    pointing_cfg: InitVar[Dict]
    center: NDArray[np.floating] = field(init=False)  # (1, ndim)
    spacing: NDArray[np.floating] = field(init=False)  # (ndim,)

    def __post_init__(self, pointing_cfg):
        self.center = np.array(
            op.get_focal_plane(None, x=0, y=0, pol=0, **pointing_cfg)
        ).T
        xieta_spacing = mu.estimate_spacing(self.fp[self.optical, :2])
        # For gamma rather than the spacing in real space we want the difference between bins
        # This is a rough estimate but good enough for us
        gamma_spacing = np.percentile(np.diff(np.sort(self.fp[2])), 99.9)
        self.spacing = np.array([xieta_spacing, xieta_spacing, gamma_spacing])


@dataclass
class FocalPlane:
    stream_id: str
    det_ids: NDArray[np.str_]  # (ndet,)
    avg_fp: NDArray[np.floating]  # (ndim, ndet)
    weights: NDArray[np.floating]  # (ndet,)
    transformed: NDArray[np.floating]  # (ndet, ndim)
    center: NDArray[np.floating]  # (1, ndim)
    center_transformed: NDArray[np.floating]  # (1, ndim)
    n_point: NDArray[np.int_]  # (ndet,)
    n_gamma: NDArray[np.int_]  # (ndet,)
    have_gamma: bool = field(default=False)
    template: Optional[Template] = field(default=None)
    transform: Transform = field(default_factory=Transform.identity)
    transform_nocm: Transform = field(default_factory=Transform.identity)
    full_fp: Optional[NDArray[np.floating]] = field(
        init=False, default=None
    )  # (ndet, ndim, n_aman)
    tot_weight: Optional[NDArray[np.floating]] = field(
        init=False, default=None
    )  # (ndet,)

    def __post_init__(self):
        if self.template is not None and not np.all(
            np.isclose(self.center, self.template.center)
        ):
            raise ValueError("Focalplane center does not match template")

    @classmethod
    def empty(cls, template, stream_id, n_aman):
        if template is None:
            raise TypeError("template must be an instance of Template, not None")
        full_fp = np.full(template.fp.shape + (n_aman,), np.nan)
        tot_weight = np.zeros(len(template.det_ids))
        avg_fp = np.full_like(template.fp, np.nan)
        weight = np.zeros(len(template.det_ids))
        transformed = template.fp.copy()
        center = template.center.copy()
        center_transformed = template.center.copy()
        n_point = np.zeros_like(template.det_ids, dtype=int)
        n_gamma = np.zeros_like(template.det_ids, dtype=int)

        fp = FocalPlane(
            stream_id,
            template.det_ids,
            avg_fp,
            weight,
            transformed,
            center,
            center_transformed,
            n_point,
            n_gamma,
            template=template,
        )
        fp.full_fp = full_fp
        fp.tot_weight = tot_weight

        return fp

    def map_by_det_id(self, aman):
        if self.template is not None and not np.array_equal(
            self.det_ids, self.template.det_ids
        ):
            raise ValueError("det_ids don't match template, mapping is not valid!")
        _, msk, template_msk = np.intersect1d(
            aman.det_info.det_id, self.det_ids, return_indices=True
        )
        if len(msk) != aman.dets.count:
            logger.warning("There are matched dets not found in the focal plane")
        mapping = np.argsort(np.argsort(self.det_ids[template_msk]))
        srt = np.argsort(aman.det_info.det_id[msk])
        xi = aman.pointing.xi[msk][srt][mapping]
        eta = aman.pointing.eta[msk][srt][mapping]
        if "polarization" in aman:
            # name of field just a placeholder for now
            gamma = aman.polarization.polang[msk][srt][mapping]
        elif "gamma" in aman.pointing:
            gamma = aman.pointing.gamma[msk][srt][mapping]
        else:
            gamma = np.full(len(xi), np.nan)
        fp = np.column_stack((xi, eta, gamma))
        return fp, template_msk

    def add_fp(self, i, fp, weights, template_msk):
        if self.full_fp is None or self.tot_weight is None:
            raise ValueError("full_fp or tot_weight not initialized")
        self.full_fp[template_msk, :, i] = fp * weights[..., None]
        self.tot_weight[template_msk] += weights

    def save(self, f, db_info, group):
        ndets = len(self.det_ids)
        outdt = [
            ("dets:det_id", self.det_ids.dtype),
            ("xi", np.float32),
            ("eta", np.float32),
            ("gamma", np.float32),
        ]
        fpout = np.fromiter(
            zip(self.det_ids, *(self.transformed.T)), dtype=outdt, count=ndets
        )
        write_dataset(
            metadata.ResultSet.from_friend(fpout),
            f,
            f"{group}/focal_plane",
            overwrite=True,
        )
        _add_attrs(f[f"{group}/focal_plane"], {"measured_gamma": self.have_gamma})
        entry = {"dets:stream_id": self.stream_id, "dataset": f"{group}/focal_plane"}
        entry.update(db_info[1])
        db_info[0].add_entry(entry, filename=os.path.basename(f.filename), replace=True)

        outdt_full = [
            ("dets:det_id", self.det_ids.dtype),
            ("xi_t", np.float32),
            ("eta_t", np.float32),
            ("gamma_t", np.float32),
            ("xi_m", np.float32),
            ("eta_m", np.float32),
            ("gamma_m", np.float32),
            ("weights", np.float32),
            ("n_point", np.int8),
            ("n_gamma", np.int8),
        ]
        fpfullout = np.fromiter(
            zip(
                self.det_ids,
                *(self.transformed.T),
                *(self.avg_fp.T),
                self.weights,
                self.n_point,
                self.n_gamma,
            ),
            dtype=outdt_full,
            count=ndets,
        )
        write_dataset(
            metadata.ResultSet.from_friend(fpfullout),
            f,
            f"{group}/focal_plane_full",
            overwrite=True,
        )

        self.transform.save(f, f"{group}/transform")
        self.transform_nocm.save(f, f"{group}/transform", "_nocm")
        _add_attrs(
            f[f"{group}"],
            {
                "fit_centers": self.center_transformed,
                "template_centers": self.center,
            },
        )

    @classmethod
    def load(cls, group):
        stream_id = group.name.split("/")[-1]
        fp_full = read_dataset(group.file, f"{group.name}/focal_plane_full")
        det_ids = fp_full["dets:det_id"]
        avg_fp = np.column_stack(
            (np.array(fp_full["xi_m"]), np.array(fp_full["eta_m"]), np.array(fp_full["gamma_m"]))
        )
        weights = fp_full["weights"]
        transformed = np.column_stack(
            (np.array(fp_full["xi_t"]), np.array(fp_full["eta_t"]), np.array(fp_full["gamma_t"]))
        )
        center = group.attrs["template_centers"]
        center_transformed = group.attrs["fit_centers"]
        n_point = fp_full["n_point"]
        n_gamma = fp_full["n_gamma"]
        have_gamma = group["focal_plane"].attrs["measured_gamma"]
        template = None
        transform = Transform.load(group["transform"])
        transform_nocm = Transform.load(group["transform"], "_nocm")

        return FocalPlane(
            stream_id,
            np.array(det_ids),
            avg_fp,
            np.array(weights),
            transformed,
            center,
            center_transformed,
            np.array(n_point),
            np.array(n_gamma),
            have_gamma,
            template,
            transform,
            transform_nocm,
        )


@dataclass
class OpticsTube:
    name: str
    center: NDArray[np.floating]
    transform: Transform = field(default_factory=Transform.identity)
    transform_fullcm: Transform = field(default_factory=Transform.identity)
    focal_planes: List[FocalPlane] = field(default_factory=list)
    center_transformed: NDArray[np.floating] = field(init=False)

    def __post_init__(self):
        self.center_transformed = mt.apply_transform(
            self.center, self.transform_fullcm.affine, self.transform_fullcm.shift
        )

    @classmethod
    def from_pointing_cfg(cls, pointing_cfg):
        name = pointing_cfg["tube_slot"]
        telescope_flavor = pointing_cfg["telescope_flavor"].upper()
        if telescope_flavor not in ["LAT", "SAT"]:
            raise ValueError("Telescope should be LAT or SAT")

        if telescope_flavor == "LAT":
            if pointing_cfg["zemax_path"] is None:
                raise ValueError("Must provide zemax_path for LAT")
            xi, eta, gamma = op.LAT_focal_plane(
                None,
                pointing_cfg["zemax_path"],
                x=0,
                y=0,
                pol=0,
                roll=pointing_cfg.get("roll", 0),
                tube_slot=name,
            )
        else:
            xi, eta, gamma = op.SAT_focal_plane(
                None,
                x=0,
                y=0,
                pol=0,
                roll=pointing_cfg.get("roll", 0),
                mapping_data=pointing_cfg.get("mapping_data", None),
            )
        center = np.array((xi, eta, gamma)).reshape((1, 3))

        return OpticsTube(name, center)

    def save(self, f, db_info):
        f.create_group(self.name)
        _add_attrs(
            f[self.name],
            {"center": self.center, "center_transformed": self.center_transformed},
        )
        self.transform.save(f, f"{self.name}/transform")
        self.transform_fullcm.save(f, f"{self.name}/transform", "_fullcm")
        for focal_plane in self.focal_planes:
            focal_plane.save(f, db_info, f"{self.name}/{focal_plane.stream_id}")

    @classmethod
    def load(cls, group):
        name = group.name.split("/")[-1]
        center = group.attrs["center"]
        transform = Transform.load(group["transform"])
        transform_fullcm = Transform.load(group["transform"], "_fullcm")
        fps = [
            FocalPlane.load(group[grp])
            for grp in group.keys()
            if "transform" not in grp
        ]

        return OpticsTube(name, center, transform, transform_fullcm, fps)


@dataclass
class Receiver:
    optics_tubes: List[OpticsTube] = field(default_factory=list)
    center: NDArray[np.floating] = field(default=np.zeros((1, 3)))
    include_cm: bool = field(default=False)
    transform: Transform = field(default_factory=Transform.identity)
    center_transformed: NDArray[np.floating] = field(init=False)

    def __post_init__(self):
        self.center_transformed = mt.apply_transform(
            self.center, self.transform.affine, self.transform.shift
        )

    def save(self, f, db_info):
        _add_attrs(
            f["/"],
            {
                "center": self.center,
                "center_transformed": self.center_transformed,
                "include_cm": self.include_cm,
            },
        )
        self.transform.save(f, f"transform")
        for ot in self.optics_tubes:
            ot.save(f, db_info)

    @classmethod
    def load(cls, f):
        center = f["/"].attrs["center"]
        include_cm = f["/"].attrs["include_cm"]
        transform = Transform.load(f["transform"])
        ots = [OpticsTube.load(f[grp]) for grp in f.keys() if "transform" not in grp]

        return Receiver(ots, center, include_cm, transform)

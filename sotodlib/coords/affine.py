"""
Functions for working with affine transformations.
"""
# NOTE: These originated from some snippets I wrote for CLASS
#       that are now organized in github.com/skhrg/megham

import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as R


def get_affine(src, dst, centered=False):
    """
    Get affine transformation between two point clouds.
    Transformation is dst = affine@src + shift

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

        centered: If True then src and dst are assumed to be pre-centered.
                  If False the median is taken to be the center.

    Returns:

        affine: The transformation matrix.

        shift: Shift to apply after transformation.
    """
    msk = np.isfinite(src).all(axis=0) * np.isfinite(dst).all(axis=0)
    if np.sum(msk) < 7:
        raise ValueError("Not enough finite points to compute transformation")

    src_c, dst_c = src, dst
    if not centered:
        src_c = src - np.median(src[:, msk], axis=1)[:, None]
        dst_c = dst - np.median(dst[:, msk], axis=1)[:, None]
    M = np.vstack((src_c[:, msk], dst_c[:, msk])).T
    *_, vh = la.svd(M)
    vh_splits = [
        quad for half in np.split(vh.T, 2, axis=0) for quad in np.split(half, 2, axis=1)
    ]
    affine = np.dot(vh_splits[2], la.pinv(vh_splits[0]))

    transformed = affine @ src[:, msk]
    shift = np.median(dst[:, msk] - transformed, axis=1)

    return affine, shift


def decompose_affine(affine):
    """
    Decompose an affine transformation into its components.
    This decomposetion treats the affine matrix as: rotation * shear * scale.

    Arguments:

        affine: The affine transformation matrix.

    Returns:

        scale: Array of ndim scale parameters.

        shear: Array of shear parameters.

        rot: Rotation matrix.
             Not currently decomposed in this function because the easiest
             way to do that is not n-dimensional but the rest of this function is.
    """
    # Use the fact that rotation matrix times its transpose is the identity
    no_rot = affine.T @ affine
    # Decompose to get a matrix with just scale and shear
    no_rot = la.cholesky(no_rot).T

    scale = np.diag(no_rot)
    shear = (no_rot / scale[:, None])[np.triu_indices(len(no_rot), k=1)]
    rot = affine @ la.inv(no_rot)

    return scale, shear, rot


def decompose_rotation(rotation):
    """
    Decompose a rotation matrix into its xyz rotation angles.
    This currently won't work on anything higher than 3 dimensions.

    Arguments:

        rotation: (ndim, ndim) rotation matrix.

    Returns:

        angles: Array of rotation angles in radians.
                If the input is 2d then the first 2 angles will be nan.
    """
    ndim = len(rotation)
    if ndim > 3:
        raise ValueError("No support for rotations in more than 3 dimensions")
    if ndim < 2:
        raise ValueError("Rotations with less than 2 dimensions don't make sense")
    if rotation.shape != (ndim, ndim):
        raise ValueError("Rotation matrix should be ndim by ndim")
    _rotation = np.eye(3)
    _rotation[:ndim, :ndim] = rotation
    angles = R.from_matrix(_rotation).as_euler("xyz")

    if ndim == 2:
        angles[:2] = np.nan
    return angles


def weighted_shift(src, dst, weights):
    """
    Compute a weighted shift between two point clouds.
    Can be applied at dst = src + weights[..., None]

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

        weights (npoints) array of weights.

    Returns:

        shift: Shift computed to line up src and dst.
    """
    wdiff = weights * (dst - src)
    shift = np.nansum(wdiff, axis=1) / np.nansum(weights)

    return shift

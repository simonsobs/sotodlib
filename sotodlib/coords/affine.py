"""
Functions for working with affine transformations.
"""

import numpy as np
import scipy.linalg as la
from scipy.spatial import transform
from scipy.spatial.transform import Rotation as R
import scipy.spatial.distance as dist


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
    if np.sum(msk) < len(src):
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


def get_affine_weighted(src, dst, weights):
    """
    Get affine transformation between two point clouds with weights.
    Note that unlike get_affine this is a simple least squares solution
    so it is not as robust to outliers, WLRA version someday.
    Transformation is dst = affine@src + shift

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

        weights: (npoint,) array of weights.

    Returns:

        affine: The transformation matrix.

        shift: Shift to apply after transformation.
    """
    msk = (
        np.isfinite(src).all(axis=0)
        * np.isfinite(dst).all(axis=0)
        * np.isfinite(weights)
    )
    if np.sum(msk) < 7:
        raise ValueError("Not enough finite points to compute transformation")
    init_shift = weighted_shift(src, dst, weights)
    rt_weight = np.sqrt(weights[msk])
    wsrc = rt_weight * src[:, msk]
    wdst = rt_weight * (dst[:, msk] - init_shift[..., None])
    x, *_ = la.lstsq(
        np.vstack((wsrc, np.ones(wsrc.shape[-1]))).T, wdst.T, check_finite=False
    )
    shift = x[-1] + init_shift
    affine = x[:-1].T

    return affine, shift


def get_affine_two_stage(src, dst, weights):
    """
    Get affine transformation between two point clouds with a two stage solver.
    This first uses get_affine to do an intitial alignment and
    then uses get_affine_weighted to compute a correction on top of that.

    Transformation is dst = affine@src + shift

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

        weights: (npoint,) array of weights.

    Returns:

        affine: The transformation matrix.

        shift: Shift to apply after transformation.
    """
    # Do an initial alignment without weights
    affine_0, shift_0 = get_affine(src, dst)
    init_align = affine_0 @ src + shift_0[..., None]
    # Now compute the actual transform
    affine, shift = get_affine_weighted(init_align, dst, weights)
    # Compose the transforms
    affine = affine @ affine_0
    shift += (affine @ shift_0[..., None])[:, 0]
    # Now one last shift correction
    transformed = affine @ src + shift[..., None]
    shift += weighted_shift(transformed, dst, weights)

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
    Can be applied at dst = src + shift[..., None]

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


def estimate_var(src, dst):
    """
    Estimate the variance along each dimension between two point clouds.
    Useful for something like a gaussian mixture model.

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

    Returns:

        var: (ndim,) array of variances.
    """
    msk = np.isfinite(src).all(axis=0) * np.isfinite(dst).all(axis=0)
    norm = np.sum(msk) ** 2
    ndim, _ = src.shape
    var = np.zeros(ndim)
    for d in range(ndim):
        sq_diff = dist.cdist(src[d, msk, None], dst[d, msk, None], metric="sqeuclidean")
        var[d] = np.sum(sq_diff) / (norm)

    return var


def gen_weights(src, dst, var=None):
    """
    Generate weights between points in two registered point clouds.
    The weight here is just the liklihood from a gaussian.
    Note that this is not a GMM, each weight is computed from a single
    gaussian since we are assuming a known registration.

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

        var: (ndim,) array of variances.
             If None then estimate_var is used to compue this.

    Returns:

        weights: (npoints,) array of weights.
    """
    if var is None:
        var = estimate_var(src, dst)

    # Compute nd gaussian for each pair
    weights = np.prod(np.exp(-0.5 * (src - dst) ** 2 / var[..., None]), axis=0)

    return weights


def get_spacing(points):
    """
    Approximate the spacing of a (mostly) regular point cloud.

    Arguments:

        points: (ndim, npoints) array of points

    Returns:

        spacing: The approximate spacing between points
    """
    edm = dist.squareform(dist.pdist(points.T))
    edm[edm == 0] = np.nan
    nearest_dists = np.nanmin(edm, axis=0)

    return np.median(nearest_dists)

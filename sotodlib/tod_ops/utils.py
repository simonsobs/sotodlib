"""
Generically useful utility functions.
"""
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from so3g import block_moment, block_moment64


def get_block_moment(
    tod: NDArray[np.floating],
    block_size: int,
    moment: int = 1,
    central: bool = True,
    shift: int = 0,
    output: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Compute the n'th moment of data in blocks along each row.
    Note that the blocks are made to be exclusive,
    so any samples left at the end will be in a smaller standalone block.
    This is a wrapper around ``so3g.block_moment``.

    Arguments:

        tod: Data to compute the moment of.
             Should  be (ndet, nsamp) or (nsamp).
             Must be float32 or float64.

        block_size: Size of block to use.

        moment: Which moment to compute.
                Must be >= 1.

        central: If True compute the mean centered moment.

        shift: Sample to start the blocks at, will be 0 before this.

        output: Array to put the blocked moment into.
                If provided must be the same shape as tod.
                If None, will be intialized from tod.
    Returns:

        block_moment: The blocked moment.
                      Will have the same shape as tod.
                      If output is provided it is modified in place and retured here.
    """
    if not np.any(np.isfinite(tod)):
        raise ValueError("Only finite values allowed in tod")
    orig_shape = tod.shape
    dtype = tod.dtype.name
    tod = np.atleast_2d(tod)
    if len(tod.shape) > 2:
        raise ValueError("tod may not have more than 2 dimensions")
    if dtype not in ["float32", "float64"]:
        raise TypeError("tod must be float32 or float64")

    if output is None:
        output = np.ascontiguousarray(np.empty_like(tod))
    if output.shape != tod.shape:
        raise ValueError("output shape does not match tod")
    if output.dtype.name != dtype:
        raise TypeError("output type does not match tod")

    if moment < 1:
        raise ValueError("moment must be at least 1")

    if dtype == "float32":
        block_moment(tod, output, block_size, moment, central, shift)
    else:
        block_moment64(tod, output, block_size, moment, central, shift)

    return output.reshape(orig_shape)

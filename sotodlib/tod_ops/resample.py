import numpy as np
from sotodlib import core
from sotodlib.core import flagman as fm
from so3g.proj.ranges import Ranges, RangesMatrix
from scipy import interpolate

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

import logging

logger = logging.getLogger(__name__)


def spl_int(t1, t0, y):
    """
    Wrapper for scipy.interpolate spline interpolation to take the same
    arguments as np.interp
    
    Args
    -----
    t1 : ndarray
        Timestamps to interpolate onto.
    t0 : ndarray
        Timestamps of the data to interpolate.
    y : ndarray
        Data to interpolate

    Returns
    --------
    Interpolated values.
    """
    tck = interpolate.splrep(t0, y)
    return interpolate.splev(t1, tck)


def interp_aman(
    aman, 
    t0, 
    t1, 
    axis='samps', 
    interp_type='linear',
):
    """
    Function for interpolating an axis manager to some new set of timesamples.
    This is useful for upsampling, downsampling, or resampling between two
    datasets taken at similar rates. It handles interpolating numpy arrays,
    and RangesMatrices but currently does not interpolate csr matrices
    (it just drops these).

    Args
    -----
    aman : AxisManager
        Axis manager to interpolate
    t0 : ndarray
        Timestamps of axis manager.
    t1 : ndarray
        Timestamps to interpolate onto.
    axis : str
        Name of axis to interpolate along (defaults to samps).
        **Only tested on samps axis!!!**
    interp_type : str
        Either 'linear' or 'spline' type interpolation.

    Returns
    --------
    dest : AxisManager
        New axis manager interpolated to t1 timestamps.
    """

    ## resampling will not extrapolate
    assert t0[0] <= t1[0]
    assert t0[-1] >= t1[-1]

    new_axes = []
    for k, v in aman._axes.items():
        if k == axis:
            new_axes.append(core.OffsetAxis(axis, len(t1)))
        else:
            new_axes.append(v)
    dest = core.AxisManager(*new_axes)

    for k, assign in aman._assignments.items():
        if axis in assign:
            if isinstance(aman[k], core.AxisManager):
                dest.wrap(k, interp_aman(aman[k], t0, t1, axis=axis,
                          interp_type=interp_type))
            elif (isinstance(aman[k], RangesMatrix) or 
                    isinstance(aman[k], Ranges)):
                dest.wrap(k, fm.resample_cuts(aman[k], t0, t1))
                dest._assignments[k] = aman._assignments[k]
            elif isinstance(aman[k], csr_array):
                logger.warning('csr matrix is not supported in resampling.' +
                               f' {k} is a csr_matrix and is being dropped ' +
                               'from the returned axis manager')
                continue
            elif isinstance(aman[k], np.ndarray):
                shape = list(aman[k].shape)
                for i, a in enumerate(assign):
                    if a is not None:
                        shape[i] = a
                dest.wrap_new(k, shape=shape, dtype=aman[k].dtype)
                if len(shape) == 1:
                    if interp_type == 'linear':
                        dest[k][:] = np.interp(t1, t0, aman[k])
                    if interp_type == 'spline':
                        dest[k][:] = spl_int(t1, t0, aman[k])
                elif len(shape) == 2:
                    if (shape[-1] != axis):
                        logger.warning(f'dropping {k}')
                        continue
                    for i, y in enumerate(aman[k]):
                        if interp_type == 'linear':
                            dest[k][i, :] = np.interp(t1, t0, y)
                        if interp_type == 'spline':
                            dest[k][i, :] = spl_int(t1, t0, y)
            else:
                raise ValueError('Data type in axis manager not supported '+
                                 'in interpolation')
        else:
            dest.wrap(k, aman[k])
            dest._assignments[k] = aman._assignments[k]
    return dest


def decimate_aman(aman, fs_new):
    """
    Decimate axis manager to new sample rate.
    
    Args
    -----
    aman : AxisManager
        Axis manager to decimate.
    fs_new : int
        New sampling rate.

    Returns
    --------
    aman_new : AxisManager
        Decimated axis manager.
    """
    ts_new = np.arange(aman.timestamps[0], aman.timestamps[-1], 1/fs_new)
    aman_new = interp_aman(aman, aman.timestamps, ts_new)
    return aman_new


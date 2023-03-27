import so3g
from pixell import enmap

from .. import core, tod_ops, coords

def make_map(tod,
             resolution=0.1*coords.DEG,
             demodT_name='dsT',
             demodQ_name='demodQ',
             demodU_name='demodU',
             map_glitches=False):
    """
    Create a map, inverse variance weighted map, weight map of the temperature and polarization signals from a time-ordered data (TOD).
    This code hacks the sotodlib.coords and makes maps with demodulated Q, U signals.
    
    Parameters
    ----------
    tod : AxisManager
        The time-ordered data to use.
    resolution : float
        The pixel size of the map in radian. Default is 0.1 * coords.DEG.
    demodT_name : str, optional
        The name of the temperature signal in the TOD. Default is 'dsT'.
    demodQ_name : str, optional
        The name of the Q polarization signal in the TOD. Default is 'demodQ'.
    demodU_name : str, optional
        The name of the U polarization signal in the TOD. Default is 'demodU'.
    map_glitches : bool, optional
        Whether to create a map of the TOD with glitches masked out. Default is False.

    Returns
    -------
    mTQU : enmap.ndmap
        Solved maps of for the temperature and polarization. shape=(3, Ny, Nx).
    mTQU_weighted : enmap.ndmap
        "Inverse variance weighted maps" for the temperature and polarization. shape=(3, Ny, Nx)
    wTQU : enmap.ndmap
        The weight maps for mTQU. shape=(3, 3, Ny, Nx)
    """
    wcsk = coords.get_wcs_kernel('car', 0, 0, resolution)
    
    if (map_glitches):
        PT = coords.P.for_tod(tod=tod, wcs_kernel=wcsk,
                                cuts=tod.flags.glitches, comps='T')
        PQU = coords.P.for_tod(tod=tod, wcs_kernel=wcsk,
                                cuts=tod.flags.glitches, comps='QU')
        PTQU = coords.P.for_tod(tod=tod, wcs_kernel=wcsk,
                                cuts=tod.flags.glitches, comps='TQU')
    else:
        PT = coords.P.for_tod(tod=tod, wcs_kernel=wcsk, comps='T')
        PQU = coords.P.for_tod(tod=tod, wcs_kernel=wcsk, comps='QU')
        PTQU = coords.P.for_tod(tod=tod, wcs_kernel=wcsk, comps='TQU')
    
    # T map
    mT_weighted = PT.to_map(tod=tod, signal=tod[demodT_name])
    # T weight
    wT = PT.to_weights(tod, signal=tod[demodT_name])
    
    # Q/U maps 
    mQ_weighted = PQU.to_map(tod=tod, signal=tod[demodQ_name])
    mU_weighted = PQU.to_map(tod=tod, signal=tod[demodU_name])
    mQU_weighted = PQU.zeros()
    mQU_weighted[0][:] = mQ_weighted[0] - mU_weighted[1] # (= Q_{boresight coord}*cos(2 theta_pa) - U_{boresight coord}*sin(2 theta_pa) )
    mQU_weighted[1][:] = mQ_weighted[1] + mU_weighted[0] # (= Q_{boresight coord}*sin(2 theta_pa) + U_{boresight coord}*cos(2 theta_pa) )
    #, where theta_pa is detector's prallactic angle
    
    # QU weights
    wQ = PT.to_weights(tod, signal=tod[demodQ_name])
    wU = PT.to_weights(tod, signal=tod[demodU_name])
    wQU = (wQ  + wU)/2 # use the mean of Q and U weights as a weight
    
    # combine T and QU into TQU
    mTQU_weighted = enmap.zeros((3,) + PT.geom.shape, wcs=PT.geom.wcs)
    mTQU_weighted[0] = mT_weighted
    mTQU_weighted[1] = mQU_weighted[0]
    mTQU_weighted[2] = mQU_weighted[1]
    
    # TQU weights
    wTQU = enmap.zeros((3, 3) + PT.geom.shape, wcs=PT.geom.wcs)
    wTQU[0][0] = wT
    wTQU[1][1] = wQU
    wTQU[2][2] = wQU
    
    # remove weights
    mTQU = PTQU.remove_weights(signal_map=mTQU_weighted, weights_map=wTQU)
    return mTQU, mTQU_weighted, wTQU
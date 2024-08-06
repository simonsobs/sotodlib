import numpy as np
import so3g
from pixell import enmap

from .. import core, tod_ops, coords


def make_map(tod,
             P=None, wcs_kernel=None,
             res=0.1 * coords.DEG,
             dsT=None, demodQ=None, demodU=None,
             cuts=None,
             det_weights=None, det_weights_demod=None,
             center_on=None, flip_gamma=True):
    """Generates maps of temperature and polarization from demodulated HWP
    data.  It is assumed that each detector contributes a T signal
    (which has been low-pass filtered to avoid the modulated signal)
    stored in dsT, as well as separate Q and U timestreams,
    corresponding to the cosine-phase (demodQ = Q cos 4 chi) and
    sine-phase (demodU = U sin 4 chi) response of the HWP.

    The demodQ and demodU signals are assumed to have been computed
    without regard for the polarization angle of the detector, nor the
    on-sky parallactic angle.  The impact of these is handled by the
    projection routines in this function.

    Parameters
    ----------
    tod : sotodlib.core.AxisManager
        An AxisManager object
    P : sotodlib.coords.pmat
        Projection Matrix to be used for mapmaking. If None, it is
        generated from `wcs_kernel` or `center_on`.
    wcs_kernel : enlib.wcs.WCS or None, optional
        The WCS object used to generate the output map.
        If None, a new WCS object with a Cartesian projection and a resolution of `res` will be created.
    center_on: None or str
        Name of point source. If specified, Source-centered map with a tangent projection will be made.
        If None, wcs_kernel will be used to generate a projection matrix.
    size: float or None
        If specified, trim source-centored map to a local square with `size`, in radian.
        If None, not trimming will be applied. Only valid when `centor_on` is specified.
    res : float, optional
        The resolution of the output map, in radian.        
    dsT : array-like or None, optional
        The input dsT timestream data. If None, the 'dsT' field of `tod` will be used.
    demodQ : array-like or None, optional
        The input demodulated Q timestream data. If None, the 'demodQ' field of `tod` will be used.
    demodU : array-like or None, optional
        The input demodulated U timestream data. If None, the 'demodU' field of `tod` will be used.
    cuts : RangesMatrix or None, optional
        A RangesMatrix that identifies samples that should be excluded from projection operations.
        If None, no cuts will be applied.
    det_weights : array-like or None, optional
        The detector weights to use in the map-making for the dsT timestream.
    det_weights_demod : array-like or None, optional
        The detector weights to use in the map-making for the demodulated Q and U timestreams.
        If both of `det_weights` and `det_weights_demod` are None, uniform detector weights will be used.
        If only one of two are provided, the other weight is provided by `det_weights` = 2 * `det_weights_demod`.
    flip_gamma : bool or None
        Defaults to True.  When constructing the pointing matrix,
        reflect the nominal focal plane polarization angles about zero
        (gamma -> -gamma).  If you pass in your own P, make sure it
        was constructed with hwp=True.

    Returns
    -------
    A dictionary which contains:
    'map' : enmap.ndmap
        map of temperature and polarization
    'weighted_map' : enmap.ndmap
        The inverse variance weighted map of temperature and polarization
    'weight' : enmap.ndmap
        The map of inverse variance weights used in the map-making process.

    """
    if dsT is None:
        dsT = tod['dsT']
    if demodQ is None:
        demodQ = tod['demodQ']
    if demodU is None:
        demodU = tod['demodU']

    if flip_gamma is None:
        flip_gamma = True

    if P is None:
        if center_on is None:
            if wcs_kernel is None:
                wcs_kernel = coords.get_wcs_kernel('car', 0, 0, res)
            P = coords.P.for_tod(
                tod=tod, wcs_kernel=wcs_kernel, cuts=cuts, comps='QU', hwp=flip_gamma)
        else:
            P, X = coords.planets.get_scan_P(tod, planet=center_on, res=res, hwp=flip_gamma)
            

    if det_weights is None:
        if det_weights_demod is None:
            det_weights_demod = np.ones(tod.dets.count, dtype='float32')
        det_weights = det_weights_demod * 2.
    else:
        if det_weights_demod is None:
            det_weights_demod = det_weights / 2.

    # T map and weight
    mT_weighted = P.to_map(
        tod=tod, signal=dsT, comps='T', det_weights=det_weights)
    wT = P.to_weights(tod, signal=dsT, comps='T', det_weights=det_weights)

    # Q/U maps and weights
    mQ_weighted = P.to_map(tod=tod, signal=demodQ, comps='QU',
                             det_weights=det_weights_demod)
    mU_weighted = P.to_map(tod=tod, signal=demodU, comps='QU',
                             det_weights=det_weights_demod)

    mQU_weighted = P.zeros()
    mQU_weighted[0][:] = mQ_weighted[0] - mU_weighted[1]
    mQU_weighted[1][:] = mQ_weighted[1] + mU_weighted[0]
    del mQ_weighted, mU_weighted

    wQU = P.to_weights(tod, signal=demodQ, comps='T',
                         det_weights=det_weights_demod)

    # combine mT_weighted and mQU_weighted into mTQU_weighted
    mTQU_weighted = P.zeros(super_shape=3)
    mTQU_weighted[0] = mT_weighted
    mTQU_weighted[1] = mQU_weighted[0]
    mTQU_weighted[2] = mQU_weighted[1]

    # combine wT and wQU into wTQU
    wTQU = enmap.zeros((3, 3) + P.geom.shape, wcs=P.geom.wcs)
    wTQU[0][0] = wT
    wTQU[1][1] = wQU
    wTQU[2][2] = wQU

    # remove weights
    mTQU = P.remove_weights(signal_map=mTQU_weighted,
                              weights_map=wTQU, comps='TQU')
    
    output = {'map': mTQU,
             'weighted_map': mTQU_weighted,
             'weight': wTQU}
    return output

def from_map(tod, signal_map, cuts=None, flip_gamma=True, wrap=False, modulated=False):
    """
    Generate simulated TOD with HWP from a given signal map.

    Args:
        tod : an axisManager object
        signal_map: pixell.enmap.ndmap containing (Tmap, Qmap, Umap) representing the signal.
        cuts (RangesMatrix, optional): Cuts to apply to the data. Default is None.
        flip_gamma (bool, optional): Whether to flip detector coordinate. If you use the HWP, keep it `True`. Default is True.
        wrap (bool, optional): Whether to wrap the simulated data. Default is False.
        modulated (bool, optional): If True, return modulated signal. If False, return the demodulated signal 
        (`dsT`, `demodQ`, and `demodU`). Default is False. 

    Returns:
        `modulate==False`: A tuple containing the TOD (np.array) of dsT, demodQ and demodU.
        `modulate==True` : The modulated TOD (np.array)
        
    """
    Tmap, Qmap, Umap = signal_map
    
    P = coords.P.for_tod(tod=tod, geom=signal_map.geometry, cuts=cuts, 
                         comps='QU', hwp=flip_gamma)
    dsT_sim = P.from_map(Tmap, comps='T')
    demodQ_sim = P.from_map(enmap.enmap([Qmap, Umap]), comps='QU')
    demodU_sim = P.from_map(enmap.enmap([Umap, -Qmap]), comps='QU')
    
    if modulated is False:
        if wrap:
            tod.wrap('dsT', dsT_sim, [(0, 'dets'), (1, 'samps')])
            tod.wrap('demodQ', demodQ_sim, [(0, 'dets'), (1, 'samps')])
            tod.wrap('demodU', demodU_sim, [(0, 'dets'), (1, 'samps')])
        return dsT_sim, demodQ_sim, demodU_sim
    else:
        assert 'hwp_angle' in tod._fields
        signal_sim = dsT_sim + demodQ_sim*np.cos(4*tod.hwp_angle) + demodU_sim*np.sin(4*tod.hwp_angle)
        if wrap:
            tod.wrap('signal', signal_sim, [(0, 'dets'), (1, 'samps')])
        return signal_sim
    
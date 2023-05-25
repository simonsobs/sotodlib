import numpy as np
from numpy.polynomial import legendre as L
from scipy.optimize import curve_fit
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)


def get_sss(aman, signal=None, azpoint=None, nmodes=20, 
            flags=None, merge_stats=True, sss_stats_name='sss_stats',
            merge_model=True, sss_model_name='sss_model'):
    """
    Extracts scan synchronous signal (SSS) assuming it is represented by 
    a sum of legendre polynomials in Az-signal space using numpy polynomial fitter.

    Parameters
    ----------
    aman : AxisManager object
        The TOD to extract SSS from.
    signal : array-like, optional
        The TOD signal to use. If not provided, `aman.signal` will be used.
    nmodes : int, optional
        The max order of legendre polynomials to extract. Default is 20.
    flags : RangesMatrix, optional
        Flags to be masked out before extracting SSS. If Default is None, and no mask will be applied.
    merge_stats : bool, optional
        Whether to add the extracted SSS statistics to `aman` as new axes. Default is `True`.
    sss_stats_name : str, optional
        The name to use for the new field containing the SSS statistics if `merge_stats` is `True`. Default is 'sss_stats'.
    merge_model : bool, optional
        Whether to add the extracted SSS to `aman` as a new signal field. Default is `True`.
    sss_extract_name : str, optional
        The name to use for the new signal field containing the extracted SSS if `merge_extract` is `True`. Default is 'sss_extract'.

    Returns
    -------
    sss_stats : AxisManager object
        The extracted SSS and its statistics. The statistics include:

            - **coeffs** (n_dets x n_modes) : coefficients of the model

            .. math::
                y = \mathrm{coeffs}[0]L_0(x) + \mathrm{coeffs}[1]L_1(x) + \cdots + \mathrm{coeffs}[n]L_n(x)

            where x is the max-to-min az-range normalized to the +/-1 interval over which the legendre polynomials are defined, and the L_m's are the legendre polynomials of order m.
    
    """

    if signal is None:
        signal = aman.signal

    if azpoint is None:
        azpoint = aman.boresight.az

    x = (azpoint-(np.mean(azpoint)))/(np.ptp(azpoint)/2)

    # define sss_stats
    mode_names = []
    for mode in range(nmodes+1):
        mode_names.append(f'Legendre{mode}')

    sss_stats = core.AxisManager(aman.dets, core.LabelAxis(
        name='modes', vals=np.array(mode_names, dtype='<U10')))
    c = np.zeros((aman.dets.count, nmodes+1))

    if flags is None:
        m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)
    else:
        m = ~flags.mask()
   
    for i in range(aman.dets.count):
        c[i], stats = L.legfit(x[m[i]], signal[i, m[i]], nmodes, full=True)
        # Want to decide which stats are useful to wrap here besides the coeffs
    sss_stats.wrap('coeffs', c, [(0, 'dets'), (1, 'modes')])

    if merge_stats:
        aman.wrap(sss_stats_name, sss_stats)
    if merge_model:
        fitsig_tod = L.legval(x, sss_stats.coeffs.T)
        aman.wrap(sss_model_name, fitsig_tod, [(0, 'dets'), (1, 'samps')])
    return sss_stats

def subtract_sss(aman, signal=None, sss_template=None,
                 subtract_name='sss_remove'):
    """
    Subtract the scan synchronous signal (SSS) template from the
    signal in the given axis manager.

    Parameters
    ----------
    aman : AxisManager
        The axis manager containing the signal to which the SSS template will
        be applied.
    signal : ndarray, optional
        The signal from which the SSS template will be subtracted. If `signal` is
        None (default), the signal contained in the axis manager will be used.
    sss_template : ndarray, optional
        The SSS template to be subtracted from the signal. If `sss_template`
        is None (default), the SSS template stored in the axis manager under
        the key 'sss_extract' will be used.
    subtract_name : str, optional
        The name of the output axis manager field that will contain the SSS-subtracted 
        signal. Defaults to 'sss_remove'.

    Returns
    -------
    None
    """
    if signal is None:
        signal = aman.signal
    if sss_template is None:
        sss_template = aman['sss_model']

    aman.wrap(subtract_name, np.subtract(
              signal, sss_template), [(0, 'dets'), (1, 'samps')])

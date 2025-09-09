import numpy as np
import logging
from lmfit import Model as LmfitModel

from sotodlib import core
from sotodlib.hwp import demod_tod

logger = logging.getLogger(__name__)


def tau_model(fhwp, tau, AQ, AU, mode):
    return (AQ + 1j*AU) * np.exp(1j * mode * 2 * np.pi * tau * fhwp)


def get_tau_hwp(
    aman,
    width=1000,
    min_fhwp=1.,
    max_fhwp=2.02,
    demod_mode=4,
    wn=None,
    flag_name=None,
):
    """
    Analyze observation with hwp spinning up or spinning down and
    compute the timeconstant of detectors from hwp speed dependence of
    the angle of half-wave plate synchronous signal.

    A_mode_obs = A_mode_true * exp(i * 2 * pi * mode * f_hwp * tau_hwp)

    Parameters
    ----------
    aman : AxisManager
        AxisManager object containing the TOD data.
    flag_name : str or None
        Name of the flag field in `aman` to use for masking data. If None,
        no masking is applied.

    Returns
    -------
    out_aman : AxisManager
        An AxisManager containing time constants, their errors, and reduced
        chi-squared statistics.
    """
    if wn is None:
        wn = np.ones_like(aman.dets.count)

    hwp_rate = np.gradient(np.unwrap(aman.hwp_solution.hwp_angle)
                           * 200 / 2 / np.pi)
    aman.wrap('hwp_rate', np.abs(hwp_rate), [(0, 'samps')])

    logger.debug('wrap time vs phase rotation')
    idx = np.where((min_fhwp < hwp_rate) & (hwp_rate < max_fhwp))[0]
    start = idx[0]
    end = idx[-1]
    sections = core.OffsetAxis('sections', count=int((end-start)/width),
                               offset=0)
    result = core.AxisManager(aman.dets, sections)

    hwp_freq, demodQ, demodU = [], [], []
    for i in range(result.sections.count):
        s = aman.samps.offset + start + i * width
        e = s + width
        aman_short = aman.restrict('samps', (s, e), in_place=False)
        _, _demodQ, _demodU = demod_tod(aman_short, demod_mode=demod_mode,
                                        wrap=False)
        hwp_freq.append(np.median(aman_short.hwp_rate))
        demodQ.append(np.median(_demodQ, axis=1))
        demodU.append(np.median(_demodU, axis=1))

    hwp_freq = np.array(hwp_freq)
    result.wrap('hwp_freq', hwp_freq, axis_map=[(0, 'sections')])
    result.wrap('demodQ', np.transpose(demodQ),
                axis_map=[(0, 'dets'), (1, 'sections')])
    result.wrap('demodU', np.transpose(demodU),
                axis_map=[(0, 'dets'), (1, 'sections')])
    result.wrap('weights', np.sqrt(width/2/hwp_freq[None, :])/wn[:, None],
                axis_map=[(0, 'dets'), (1, 'sections')])

    logger.debug('Fit time constant')
    AQ = np.full(result.dets.count, np.nan)
    AQ_error = np.full(result.dets.count, np.nan)
    AU = np.full(result.dets.count, np.nan)
    AU_error = np.full(result.dets.count, np.nan)
    tau = np.full(result.dets.count, np.nan)
    tau_error = np.full(result.dets.count, np.nan)
    redchi2s = np.full(result.dets.count, np.nan)
    for i in range(result.dets.count):
        try:
            model = LmfitModel(tau_model, independent_vars=['fhwp'])
            model.set_param_hint('mode', vary=False)
            params = model.make_params(
                tau=1e-3,
                g=0,
                AQ=np.median(result.demodQ[i]),
                AU=np.median(result.demodU[i]),
                mode=demod_mode,
            )
            fit = model.fit(
                data=result.demodQ[i] + 1j * result.demodU[i],
                params=params,
                fhwp=result.hwp_freq,
                weights=result.weights[i],
            )
            AQ[i] = fit.params['AQ'].value
            AQ_error[i] = fit.params['AQ'].stderr
            AU[i] = fit.params['AU'].value
            AU_error[i] = fit.params['AU'].stderr
            tau[i] = fit.params['tau'].value
            tau_error[i] = fit.params['tau'].stderr
            redchi2s[i] = fit.redchi
        except Exception:
            logger.debug(f'Failed to fit {aman.dets.vals[i]}')

    result.wrap('AQ', AQ, axis_map=[(0, 'dets')])
    result.wrap('AQ_error', AQ_error, axis_map=[(0, 'dets')])
    result.wrap('AU', AU, axis_map=[(0, 'dets')])
    result.wrap('AU_error', AU_error, axis_map=[(0, 'dets')])
    result.wrap('tau_hwp', tau, axis_map=[(0, 'dets')])
    result.wrap('tau_hwp_error', tau_error, axis_map=[(0, 'dets')])
    result.wrap('redchi2s', redchi2s, axis_map=[(0, 'dets')])

    return result

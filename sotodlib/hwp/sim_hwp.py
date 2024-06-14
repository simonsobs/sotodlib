import numpy as np
import so3g
from spt3g import core
from sotodlib import core
import logging

logger = logging.getLogger(__name__)


def I_to_P_param(bandpass="f090", loading=10, inc_ang=5):
    """
    - HWP I to P leakage simulation function
    - Mueller matirices of three layer achromatic HWP \
      for each frequency and each incident angle are derived by RCWA simulation.
    - AR coating based on Mullite-Duroid coating for SAT1 at UCSD.\
      Thickness are used fabricated values.
    - 2f and 4f amplitudes and phases are extracted by HWPSS fitting of MIQ and MIU.
    - Amplitudes and phases are fitted by polynominal functions w.r.t each incident angle.
    - Only support MF because UHF and LF are design/fabrication phase. \
      We will add them after the fabrications.
    
    Args
    -----
    bandpass: str
        an 3-digit (front zero padded) integer "fXYZ" 
        where XYZ = 030/040/090/150/220/280
    loading: float
        intensity [Kcmb]
    inc_ang: float
        incident angle [deg]

    Returns
    --------
    amp_2f: float
        2f amplitude [mKcmb]
    amp_4f: float
        4f amplitude [mKcmb]
    phi_2f: float
        2f phase shift [rad]
    phi_4f: float
        4f phase shift [rad]
    """
    if bandpass == "f090":
        poly_A2_ang = np.array([-7.01e-06, -6.87e-05, 3.435e-02])
        poly_A4_ang = np.array([1.61e-07, 1.20e-05, 3.42e-06, 1.23e-07])
        phi_2f, phi_4f = 66.46, 55.57  # [deg]
    elif band == "f150":
        poly_A2_ang = np.array([-1.11e-05, 2.58e-05, 3.40e-02])
        poly_A4_ang = np.array([4.38e-08, 8.90e-06, 1.85e-06, 1.18e-07])
        phi_2f, phi_4f = 71.62, 54.43  # [deg]
    else:
        logger.error("Currently only supporting MF(f090/f150), stay tuned.")
    amp_2f = loading * np.poly1d(poly_A2_ang)(inc_ang)
    amp_4f = loading * np.poly1d(poly_A4_ang)(inc_ang)
    amp_2f, amp_4f = amp_2f * 1e3, amp_4f * 1e3
    phi_2f, phi_4f = np.deg2rad(phi_2f), np.deg2rad(phi_4f)
    if inc_ang == 0:
        amp_4f, phi_4f = 0, 0
    return amp_2f, amp_4f, phi_2f, phi_4f


def sim_hwpss(aman, name='hwpss_sim', hwp_freq=2., 
              bandpass='090', loading=10., inc_ang=5.,
              amp_2f_r=0.05, amp_4f_r=0.05, 
              phi_2f_r=0.1, phi_4f_r=0.1):
    """
    - HWPSS simulation function using 2f and 4f amp and phase from I_to_P_param function.
    - The simulated HWSS is added to the input AxisManager as “name”.

    Args
    -----
    aman: AxisManager
        target AxisManager
    hwp_freq: float
        HWP rotation speed, 2 [Hz] (default)
    bandpass: str
        an 3-digit (front zero padded) integer "XYZ" 
        where XYZ = 030/040/090/150/220/280
    loading: float
        intensity [Kcmb]
    inc_ang: float 
        incident angle [deg]
    amp_2f_r: float, optional
        2f HWPSS amplitude fluctuation
    amp_4f_r: float, optional
        4f HWPSS amplitude fluctuation
    phi_2f_r: float, optional
        2f HWPSS phase shift fluctuation
    phi_4f_r: float, optional
        4f HWPSS phase shift fluctuation

    """
    if name in aman.keys():
        aman.move(name, '')
    aman.wrap_new(name, ('dets', 'samps'))
    hwp_angle = np.mod(2 * np.pi * aman.timestamps * hwp_freq, 2 * np.pi)
    if 'hwp_angle' not in aman.keys():
        aman.wrap('hwp_angle', hwp_angle, [(0, 'samps')])
    else:
        aman.hwp_angle = hwp_angle
    nf = [2, 4]
    amp_2f, amp_4f, phi_2f, phi_4f = I_to_P_param(inc_ang=15)
    amp = [amp_2f, amp_4f]
    amp_r = [amp_2f * amp_2f_r, amp_4f * amp_4f_r]
    phi = [phi_2f, phi_4f]
    phi_r = [phi_2f * phi_2f_r, phi_4f * phi_4f_r]
    for n in range(len(nf)):
        amp_rnd = np.random.normal(
            loc=amp[n], scale=amp_r[n], size=(
                aman.dets.count,))
        phase_rnd = np.mod(
            np.random.normal(
                loc=phi[n],
                scale=phi_r[n],
                size=(
                    aman.dets.count,
                )),
            2 * np.pi)
        aman[name] += amp_rnd[:, None] * \
            np.cos(nf[n] * aman.hwp_angle + phase_rnd[:, None])


def sim_hwpss_2f4f(aman, name='hwpss_sim', hwp_freq=2., bandpass="090",
                   amp_2f=300, amp_4f=30, phi_2f=0, phi_4f=0,
                   amp_2f_r=0.05, amp_4f_r=0.05, phi_2f_r=0.1, phi_4f_r=0.1):
    """
    - HWPSS simulation function using given 2f and 4f amp and phase.
    - The simulated HWSS is added to the input AxisManager as “name”.

    Args
    -----
    aman: AxisManager
        target AxisManager
    hwp_freq: float
        2 [Hz] (default)
    bandpass: str 
        an 3-digit (front zero padded) integer "fXYZ"
        where XYZ = 030/040/090/150/220/280
    amp_2f: float
        2f HWPSS amplitude [Kcmb]
    amp_4f: float
        4f HWPSS amplitude [Kcmb]
    phi_2f: float
        2f HWPSS phase shift [deg.]
    phi_4f: float
        4f HWPSS phase shift [deg.]
    amp_2f_r: float 
        2f HWPSS amplitude fluctuation
    amp_4f_r: float
        4f HWPSS amplitude fluctuation
    phi_2f_r: float
        2f HWPSS phase shift fluctuation
    phi_4f_r: float 
        4f HWPSS phase shift fluctuation
    """
    if name in aman.keys():
        aman.move(name, '')
    aman.wrap_new(name, ('dets', 'samps'))
    hwp_angle = np.mod(2 * np.pi * aman.timestamps * hwp_freq, 2 * np.pi)
    if 'hwp_angle' not in aman.keys():
        aman.wrap('hwp_angle', hwp_angle, [(0, 'samps')])
    else:
        aman.hwp_angle = hwp_angle
    nf = [2, 4]
    amp = [amp_2f, amp_4f]
    amp_r = [amp_2f * amp_2f_r, amp_4f * amp_4f_r]
    phi = [phi_2f, phi_4f]
    phi_r = [phi_2f * phi_2f_r, phi_4f * phi_4f_r]
    for n in range(len(nf)):
        amp_rnd = np.random.normal(
            loc=amp[n], scale=amp_r[n], size=(
                aman.dets.count,))
        phase_rnd = np.mod(
            np.random.normal(
                loc=phi[n],
                scale=phi_r[n],
                size=(
                    aman.dets.count,
                )),
            2 * np.pi)
        aman[name] += amp_rnd[:, None] * \
            np.cos(nf[n] * aman.hwp_angle + phase_rnd[:, None])

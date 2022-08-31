import numpy as np
import so3g
from spt3g import core
from sotodlib import core

logger = logging.getLogger(__name__)

def I_to_P_param(bandpass="f090", loading=10, inc_ang=5):
    """ 
    ** I to P leakage simulation function **
    - Mueller matirices of three layer achromatic HWP for each frequency and each incident angle are derived by RCWA simulation.
    - AR coating based on Mullite-Duroid coating for SAT1 at UCSD.　Thickness are used fabricated values.
    - 2f and 4f amplitudes and phases are extracted by HWPSS fitting of MIQ and MIU.
    - Amplitudes and phases are fitted by polynominal functions w.r.t each incident angle.
    - Only support MF because UHF and LF are design/fabrication phase. We will add them after the fabrications.
    Args ---
    band: MF1=0, MF2=1
    loading: intensity [Kcmb]
    inc_ang: incident angle [deg]
    Return ---
    amp_2f/4f: 2f/4f amplitude [mKcmb]
    phi_2f/4f: 2f/4f phase shift [rad]
    """
    if bandpass == "f090":
        poly_A2_ang = np.array([-7.01e-06, -6.87e-05,  3.435e-02])
        poly_A4_ang = np.array([1.61e-07, 1.20e-05, 3.42e-06, 1.23e-07])
        phi_2f, phi_4f = 66.46, 55.57 #[deg]
    elif band == "f150":
        poly_A2_ang = np.array([-1.11e-05,  2.58e-05,  3.40e-02])    
        poly_A4_ang = np.array([4.38e-08, 8.90e-06, 1.85e-06, 1.18e-07])
        phi_2f, phi_4f = 71.62, 54.43 #[deg]
    else:
        logger.error("Currently only supporting MF(f090/f150), stay tuned.")
    amp_2f = loading * np.poly1d(poly_A2_ang)(inc_ang)
    amp_4f = loading * np.poly1d(poly_A4_ang)(inc_ang)
    amp_2f, amp_4f = amp_2f * 1e3, amp_4f * 1e3
    phi_2f, phi_4f = np.deg2rad(phi_2f), np.deg2rad(phi_4f)
    if inc_ang == 0: amp_4f, phi_4f = 0, 0
    return amp_2f, amp_4f, phi_2f, phi_4f

def sim_hwpss(aman, hwp_freq=2., band=0, loading=10., inc_ang=5.):
    """ 
    ** HWPSS simulation function **
    - 2f and 4f amp and phase are from I_to_P_param
    Args ---
    aman: AxisManager
    hwp_freq: 2 [Hz]
    band: MF1=0, MF2=1
    loading: intensity [Kcmb]
    inc_ang: incident angle [deg]
    Return ---
    Adding aman.hwpss
    """
    if 'hwpss' in aman.keys(): aman.move('hwpss','')
    aman.wrap_new( 'hwpss', ('dets', 'samps'))
    hwp_angle = np.mod(2*np.pi * aman.timestamps * hwp_freq, 2*np.pi)
    if not 'hwp_angle' in aman.keys(): aman.wrap('hwp_angle', hwp_angle, [(0,'samps')])
    else: aman.hwp_angle = hwp_angle
    amp_2f, amp_4f, phi_2f, phi_4f = I_to_P_param(inc_ang=15)
    amp = [0, 0, amp_2f, 0, amp_4f]
    amp_r = [1, 5, amp_2f*0.05, 1, amp_4f*0.05]
    phi = [0, 0, phi_2f, 0, phi_4f]
    phi_r = [0, np.deg2rad(1), phi_2f*0.1, np.deg2rad(1), phi_4f*0.1]
    nf = len(amp)
    for n in [2,4]:
        amp_rnd = np.random.normal(loc=amp[n], scale=amp_r[n], size=(aman.dets.count,))
        phase_rnd = np.mod(np.random.normal(loc=phi[n], scale=phi_r[n],size=(aman.dets.count,)), 2*np.pi)
        aman.hwpss = amp_rnd[:,None]*np.cos(n*aman.hwp_angle + phase_rnd[:,None])

def sim_hwpss_2f4f(aman, hwp_freq=2., band=0, amp_2f=300, amp_4f=30, phi_2f=0, phi_4f=0):
    """ 
    ** HWPSS simulation function **
    - 2f and 4f amp and phase are from I_to_P_param
    Args ---
    aman: AxisManager
    hwp_freq: 2 [Hz]
    band: MF1=0, MF2=1
    loading: intensity [Kcmb]
    inc_ang: incident angle [deg]
    Return ---
    Adding aman.hwpss
    """
    if 'hwpss' in aman.keys(): aman.move('hwpss','')
    aman.wrap_new( 'hwpss', ('dets', 'samps'))
    hwp_angle = np.mod(2*np.pi * aman.timestamps * hwp_freq, 2*np.pi)
    if not 'hwp_angle' in aman.keys(): aman.wrap('hwp_angle', hwp_angle, [(0,'samps')])
    else: aman.hwp_angle = hwp_angle
    amp = [0, 0, amp_2f, 0, amp_4f]
    amp_r = [1, 5, amp_2f*0.05, 1, amp_4f*0.05]
    phi = [0, 0, phi_2f, 0, phi_4f]
    phi_r = [0, np.deg2rad(1), phi_2f*0.1, np.deg2rad(1), phi_4f*0.1]
    nf = len(amp)
    for n in [2,4]:
        amp_rnd = np.random.normal(loc=amp[n], scale=amp_r[n], size=(aman.dets.count,))
        phase_rnd = np.mod(np.random.normal(loc=phi[n], scale=phi_r[n],size=(aman.dets.count,)), 2*np.pi)
        aman.hwpss = amp_rnd[:,None]*np.cos(n*aman.hwp_angle + phase_rnd[:,None])

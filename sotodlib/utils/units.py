"""Additions to the Astropy Units Package for SO Type Data. The Goal so to
enable unit coversion without converting between non equivalent types (ie. phase
measured by the smurf system and angle on the sky).
"""
import numpy as np
import astropy.units as u
_ns = globals()

DAC_Smurf = u.def_unit('DAC_Smurf', namespace=_ns,
                        format={'latex': r'DAC_{SM{\mu}RF}'},
                        doc="The raw data acquistion counts for smurf channels")

phase_Smurf = u.def_unit('phase_Smurf', namespace=_ns,
                          format={'latex': r'phase_{SM{\mu}RF}'},
                          doc="The phase change of a resonator observed \
                                over a flux ramp")

Amp_TES = u.def_unit(['A_TES', 'Ampere_TES', 'Amp_TES'],
                     prefixes=True,namespace=_ns,
                     format={'latex': r'A_{TES}' },
                     doc="Current measured through the TES")

Watt_TES = u.def_unit(['W_TES', 'Watt_TES'], prefixes=True, namespace=_ns,
                     format={'latex' : r'W_{TES}'},
                     doc="Power measured at the TES")

DAC_Bias = u.def_unit('DAC_Bias', namespace=_ns,
                       format={'latex': r'DAC_{Bias}'},
                       doc="The raw data acquistion counts for bias lines")

V_Bias = u.def_unit(['V_Bias','Volt_Bias'], prefixes=True, namespace=_ns,
                     format={'latex': r'V_{Bias}' },
                     doc="Voltage on a Bias Line")

### Be Judicious about hard coding things.
dac_to_phase = (np.pi / 2**15)
phase_to_pA = 9e6/(2*np.pi)

## pulled from sodetlib
rtm_bit_to_volt = 2*1.9073486328125e-05

smurf_equivalencies = [
    (DAC_Smurf, phase_Smurf, 
            lambda x: x*dac_to_phase, 
            lambda x: x/dac_to_phase),
    (phase_Smurf, Amp_TES, 
            lambda x: x*phase_to_pA*1e-12 , 
            lambda x: x/(1e-12*phase_to_pA)),
    (DAC_Smurf, Amp_TES, 
            lambda x: x*dac_to_phase*phase_to_pA*1e-12, 
            lambda x:x/(dac_to_phase*phase_to_pA*1e-12)),
    (DAC_Bias, V_Bias, 
            lambda x: x*rtm_bit_to_volt, 
            lambda x: x/rtm_bit_to_volt)
]

u.set_enabled_equivalencies(smurf_equivalencies);

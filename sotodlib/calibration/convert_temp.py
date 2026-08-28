import numpy as np
from scipy.constants import h, c, k

def Tth2brightness(Tbb, freq):
    """
    Args:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic.
        freq (float): frequency in Hz
    Returns:
        brightness (float): [W m^-2 sr^-1 Hz^-1] at the frequency and thermodynamic teperature (Planck function)
    """
    return 2 * h * freq**3 / c**2 / (np.exp(h * freq / k / Tbb) - 1)

def Trj2brightness(Trj, freq):
    """
    Args:
        Trj (float): Raileigh-Jeans temperature of the source in K_RJ.
        freq (float): frequency in Hz
    Returns:
        brightness (float): [W m^-2 sr^-1 Hz^-1] at the frequency and RJ teperature
    """
    return 2 * k * freq**2 / c**2 * Trj

def brightness2Tth(brightness, freq):
    """
    Args:
        brightness (float): [W m^-2 sr^-1 Hz^-1] at the frequency
        freq (float): frequency in Hz
    Returns:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic.
    """
    return h*freq/k / np.log(1 + 2*h*freq**3/c**2/brightness)

def brightness2Trj(brightness, freq):
    """
    Args:
        brightness (float): [W m^-2 sr^-1 Hz^-1] at the frequency
        freq (float): frequency in Hz
    Returns:
        Trj (float): Raileigh-Jeans temperature of the source in K_RJ.
    """
    return c**2/2/k/freq**2 * brightness

def dBdT(Tth, freq):
    """
    Args:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic.
        freq (float): frequency in Hz
    Returns:
        dBdT (float): dB/dT [W m^-2 sr^-1 Hz^-1 K_BB^-1] at the frequency and teperature (B = Planck function)
    """
    x_planck = h * freq / k / Tth # unitless x of the planck function
    return 2 * freq**2 * k / c**2 * x_planck**2 * np.exp(x_planck) / (np.exp(x_planck) - 1)**2

def dBdT_cmb(freq):
    T0_cmb = 2.7255
    return dBdT(T0_cmb, freq)

def Tkcmb2brightness(Tkcmb, freq):
    """
    Args:
        Tkcmb (float): Temperature in K_CMB unit
        freq (float): frequency in Hz
    Returns:
        brightness (float): [W m^-2 sr^-1 Hz^-1] at the frequency and CMB teperature
    """
    brightness = dBdT_cmb(freq) * Tkcmb
    return brightness

def brightness2Tkcmb(brightness, freq):
    """
    Args:
        brightness (float): [W m^-2 sr^-1 Hz^-1] at the frequency
        freq (float): frequency in Hz
    Returns:
        Tkcmb (float): Temperature in K_CMB unit
    """
    Tkcmb = brightness / dBdT_cmb(freq)
    return Tkcmb

def Tth2Trj(Tth, freq):
    """
    Args:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic.
        freq (float): frequency in Hz
    Returns:
        Trj (float): Teperature in K_RJ unit
    """
    brightness = Tth2brightness(Tth, freq)
    Trj = brightness2Trj(brightness, freq)
    return Trj

def Tth2Tkcmb(Tth, freq):
    """
    Args:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic.
        freq (float): frequency in Hz
    Returns:
        Tkcmb (float): Temperature in K_CMB unit
    """
    brightness = Tth2brightness(Tth, freq)
    Tkcmb = brightness2Tkcmb(brightness, freq)
    return Tkcmb

def Trj2Tth(Trj, freq):
    """
    Args:
        Trj (float): Rayleigh-JTeperature in K_RJ unit
        freq (float): frequency in Hz
    Returns:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic
    """
    brightness = Trj2brightness(Trj, freq)
    Tth = brightness2Tth(brightness, freq)
    return Tth

def Trj2Tkcmb(Trj, freq):
    """
    Args:
        Trj (float): Rayleigh-Jeans Teperature in K_RJ unit
        freq (float): frequency in Hz
    Returns:
        Tkcmb (float): Temperature in K_CMB unit
    """
    brightness = Trj2brightness(Trj, freq)
    Tkcmb = brightness2Tkcmb(brightness, freq)
    return Tkcmb

def Tkcmb2Tbb(Tkcmb, freq):
    """
    Args:
        Tkcmb (float): Temperature in K_CMB unit
        freq (float): frequency in Hz
    Returns:
        Tth (float): Thermondynamic temperature of the source in K_thermodynamic
    """
    brightness = brightness2Tkcmb(Tkcmb, freq)
    Tth = brightness2Tth(brightness, freq)
    return Tth

def Tkcmb2Trj(Tkcmb, freq):
    """
    Args:
        Tkcmb (float): Temperature in K_CMB unit
        freq (float): frequency in Hz
    Returns:
        T_RJ (float): Rayleigh-Jeans teperature in K_RJ unit
    """
    brightness = Tkcmb2brightness(Tkcmb, freq)
    Trj = brightness2Trj(brightness, freq)
    return Trj

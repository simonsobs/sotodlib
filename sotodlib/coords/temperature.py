import numpy as np
from scipy.constants import h, c, k

class TempConverter:
    self.T0cmb = 2.7255
    @staticmethod
    def Ttd_to_Inu(Ttd, nu):
        """Convert thermodynamic temperature to specific intensity at a given frequency.
        Args:
            Ttd (float): Thermodynamic temperature [K].
            nu (float): Frequency [Hz].
        Returns:
            Inu (float): Specific intensity [W/m^2/sr/Hz].
        """
        return 2 * h * nu**3 / c**2 / (np.exp(h * nu / k / Ttd) - 1)
    
    @staticmethod
    def Trj_to_Inu(Trj, nu):
         """Convert Rayleigh-Jeans temperature to specific intensity at a given frequency.
        Args:
            Trj (float): Rayleigh-Jeans temperature in Kelvin-rj.
            nu (float): Frequency in Hz.
        Returns:
            Inu (float): Specific intensity in W m^-2 sr^-1 Hz^-1.
        """
        return 2 * k * nu**2 / c**2 * Trj
    
    @staticmethod
    def Inu_to_Ttd(Inu, nu):
        """Convert specific intensity at frequency nu to thermodynamic temperature.
        Args:
            Inu (float): Specific intensity in W m^-2 sr^-1 Hz^-1.
            nu (float): Frequency in Hz.
        Returns:
            Ttd (float): Thermodynamic temperature in Kelvin.
        """
        return h*nu/k / np.log(1 + 2*h*nu**3/c**2/Inu)
    
    @staticmethod
    def Inu_to_Trj(Inu, nu):
        """Convert specific intensity at frequency nu to Rayleigh-Jeans temperature.
        Args:
            Inu (float): Specific intensity in W m^-2 sr^-1 Hz^-1.
            nu (float): Frequency in Hz. 
        Returns:
            Trj (float): Rayleigh-Jeans temperature in Kelvin-rj.
        """
        return c**2/2/k/nu**2 * brightness
    
    @staticmethod
    def dBnu_dTtd(Ttd, nu):
        """Calculate the derivative of Planck function with respect to thermodynamic temperature.
        Args:
            Ttd (float): Thermondynamic temperature of the source in thermo-dynamic temperature.
            nu (float): frequency in Hz
        Returns:
            dBnu/dTtd [W m^-2 sr^-1 Hz^-1 K^-1] at the frequency and teperature (Bnu = Planck function)
        """
        x_planck = h * nu / k / Ttd # unitless x of the planck function
        return 2 * nu**2 * k / c**2 * x_planck**2 * np.exp(x_planck) / (np.exp(x_planck) - 1)**2
    
    def dBnu_dTcmb(nu):
        """Calculate the derivative of Planck function with respect to CMB temperature.
        Args:
            nu (float): Frequency [Hz].
        Returns:
            dBnu_dTcmb (float): Derivative of Planck function [W/m^2/sr/Hz/K].
        """
        return self.dBnu_dTtd(self.T0cmb, nu)
    
    def Tcmb_to_Inu(Tcmb, nu):
        """
        Converts the CMB temperature to CMB specific intensity at a given frequency.
        Args:
            Tcmb (float): CMB temperature in Kelvin-cmb.
            nu (float): Frequency in Hz.
        Returns:
            CMB specific intensity at frequency nu in W/m^2/sr/Hz.
        """
        return self.dBnu_dTcmb(nu) * Tcmb
    
    def Inu_to_Tcmb(Inu, nu):
        """
        Convert specific intensity at frequency nu and thermodynamic temperature Ttd to CMB temperature.
        Args:
            Inu (float): Specific intensity in W m^-2 sr^-1 Hz^-1
            nu (float): Frequency in Hz
        Returns:
            Tcmb (float): CMB temperature in Kelvin-cmb
        """
        return Inu / self.dBnu_dTcmb(nu)
    
    def Ttd_to_Trj(Ttd, nu):
        """
        Convert thermodynamic temperature to Rayleigh-Jeans temperature.
        Args:
            Ttd (float): Thermondynamic temperature of the source in thermo-dynamic temperature.
            nu (float): frequency in Hz
        Returns:
            float: Rayleigh-Jeans temperature of the source in Kelvin-rj.
        """
        return self.Inu_to_Trj(self.Ttd_to_Inu(Ttd, nu), nu)
    
    def Ttd_to_Tcmb(Ttd, nu):
        """
        Convert thermodynamic temperature to CMB temperature.
        Args:
            Ttd (float): Thermondynamic temperature of the source in thermo-dynamic temperature.
            nu (float): frequency in Hz
        Returns:
            float: CMB temperature of the source in Kelvin-cmb.
        """
        return self.Inu_to_Tcmb(self.Ttd_to_Inu(Ttd, nu), nu)
    
    def Trj_to_Ttd(Trj, nu):
        """
        Convert Rayleigh-Jeans temperature to thermodynamic temperature.
        Args:
            Trj (float): Rayleigh-Jeans temperature of the source in Kelvin-rj.
            nu (float): frequency in Hz
        Returns:
            float: Thermodynamic temperature of the source in thermo-dynamic temperature.
        """
        return self.Inu_to_Ttd(self.Trj_to_Inu(Trj, nu), nu)
    
    def Trj_to_Tcmb(Trj, nu):
        """
        Convert Rayleigh-Jeans temperature to CMB temperature.
        Args:
            Trj (float): Rayleigh-Jeans temperature of the source in Kelvin-rj.
            nu (float): frequency in Hz
        Returns:
            float: CMB temperature of the source in Kelvin-cmb.
        """
        return self.Inu_to_Tcmb(self.Trj_to_Inu(Trj, nu), nu)
    
    def Tcmb_to_Ttd(Tcmb, nu):
        """
        Convert CMB temperature to thermodynamic temperature.
        Args:
            Tcmb (float): CMB temperature of the source in Kelvin-cmb.
            nu (float): frequency in Hz
        Returns:
            float: Thermodynamic temperature of the source in thermo-dynamic temperature.
        """
        return self.Inu_to_Ttd(self.Tcmb_to_Inu(Tcmb, nu), nu)
    
    def Tcmb_to_Trj(Tcmb, nu):
        """
        Convert CMB temperature to Rayleigh-Jeans temperature.
        Args:
            Tcmb (float): CMB temperature of the source in Kelvin-cmb.
            nu (float): frequency in Hz
        Returns:
            float: Rayleigh-Jeans temperature of the source in Kelvin-rj.
        """
        return self.Inu_to_Trj(self.Tcmb_to_Inu(Tcmb, nu), nu)


# Thermodynamic temperature of planets.
# Values come from Planck intermediate results LII. Planet flux densities (2017)
# (https://www.aanda.org/articles/aa/abs/2017/11/aa30311-16/aa30311-16.html)
planck_bands = ['p_f028', 'p_f044', 'p_f070', 'p_f100', 'p_f143', 'p_f217', 'p_f353', 'p_f545', 'p_f857']
planck_planet_Ttd_dict = {
    'Mars': {
        'p_f028': np.nan,
        'p_f044': np.nan,
        'p_f070': np.nan,
        'p_f100': 194.3,
        'p_f143': 198.4,
        'p_f217': 201.9,
        'p_f353': 209.9,
        'p_f545': 209.2,
        'p_f857': 213.6
    },
    'Jupiter': {
        'p_f028': 146.6,
        'p_f044': 160.9,
        'p_f070': 173.3,
        'p_f100': 172.6,
        'p_f143': 174.1,
        'p_f217': 175.8,
        'p_f353': 167.4,
        'p_f545': 137.4,
        'p_f857': 161.3
    },
    'Saturn': {
        'p_f028': 138.9,
        'p_f044': 147.3,
        'p_f070': 150.6,
        'p_f100': 145.7,
        'p_f143': 147.1,
        'p_f217': 145.1,
        'p_f353': 141.6,
        'p_f545': 102.5,
        'p_f857': 115.6
    },
    'Uranus': {
        'p_f028': np.nan,
        'p_f044': np.nan,
        'p_f070': np.nan,
        'p_f100': 120.5,
        'p_f143': 108.4,
        'p_f217': 98.5,
        'p_f353': 86.2,
        'p_f545': 73.9,
        'p_f857': 66.2
    },
    'Neptune': {
        'p_f028': np.nan,
        'p_f044': np.nan,
        'p_f070': np.nan,
        'p_f100': 117.4,
        'p_f143': 106.4,
        'p_f217': 97.4,
        'p_f353': 82.6,
        'p_f545': 72.3,
        'p_f857': 65.3
    }
}

# dictionary of thermodynamic temperature of planets for SO bands
# use nearest planck band from each SO band
SO_bands = ['f030', 'f040', 'f090', 'f150', 'f230', 'f290']
planet_Ttd_dict = {
    'Mars': {
        'f030': planck_planet_Ttd_dict['Mars']['p_f028'],
        'f040': planck_planet_Ttd_dict['Mars']['p_f044'],
        'f090': planck_planet_Ttd_dict['Mars']['p_f100'],
        'f150': planck_planet_Ttd_dict['Mars']['p_f143'],
        'f230': planck_planet_Ttd_dict['Mars']['p_f217'],
        'f290': planck_planet_Ttd_dict['Mars']['p_f353']
    },
    'Jupiter': {
        'f030': planck_planet_Ttd_dict['Jupiter']['p_f028'],
        'f040': planck_planet_Ttd_dict['Jupiter']['p_f044'],
        'f090': planck_planet_Ttd_dict['Jupiter']['p_f100'],
        'f150': planck_planet_Ttd_dict['Jupiter']['p_f143'],
        'f230': planck_planet_Ttd_dict['Jupiter']['p_f217'],
        'f290': planck_planet_Ttd_dict['Jupiter']['p_f353']
    },
    'Saturn': {
        'f030': planck_planet_Ttd_dict['Saturn']['p_f028'],
        'f040': planck_planet_Ttd_dict['Saturn']['p_f044'],
        'f090': planck_planet_Ttd_dict['Saturn']['p_f100'],
        'f150': planck_planet_Ttd_dict['Saturn']['p_f143'],
        'f230': planck_planet_Ttd_dict['Saturn']['p_f217'],
        'f290': planck_planet_Ttd_dict['Saturn']['p_f353']
    },
    'Uranus': {
        'f030': planck_planet_Ttd_dict['Uranus']['p_f028'],
        'f040': planck_planet_Ttd_dict['Uranus']['p_f044'],
        'f090': planck_planet_Ttd_dict['Uranus']['p_f100'],
        'f150': planck_planet_Ttd_dict['Uranus']['p_f143'],
        'f230': planck_planet_Ttd_dict['Uranus']['p_f217'],
        'f290': planck_planet_Ttd_dict['Uranus']['p_f353']
    },
    'Neptune': {
        'f030': planck_planet_Ttd_dict['Neptune']['p_f028'],
        'f040': planck_planet_Ttd_dict['Neptune']['p_f044'],
        'f090': planck_planet_Ttd_dict['Neptune']['p_f100'],
        'f150': planck_planet_Ttd_dict['Neptune']['p_f143'],
        'f230': planck_planet_Ttd_dict['Neptune']['p_f217'],
        'f290': planck_planet_Ttd_dict['Neptune']['p_f353']
    }
}


# the values comes from NASA JPl (https://ssd.jpl.nasa.gov/planets/phys_par.html)
# Now, the values are just equatorial radius.
planet_radius_dict = {'Mercury': 2440.53e3,
                      'Venus': 6051.8e3,
                      'Mars': 3396.1e3,
                      'Jupiter': 71492e3,
                      'Saturn': 60268e3,
                      'Uranus': 25559e3,
                      'Neptune': 24764e3}
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic, AltAz
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
from astropy.time import Time
import time

# Establish the observatory location
so_longitude = -1 * (67 + 47/60 + 15/3600) # degrees
so_latitude = -1 * (22 + 57/60 + 31/3600) # degrees
so_height = 5200 # meters

# Convert observatory location to geocentric
SO_loc = EarthLocation.from_geodetic(lon=so_longitude,
                                     lat=so_latitude,
                                     height=so_height)

def find_an_object(obj_name, obj_type, obs_time):
    """
    Find an object in (RA, dec) by searching its name. Requires internet
    connection to look up objects.

    Parameters
    ----------
    obj_name (str): name of the object you're looking up
    obj_type (str): effectively 'planet' or anything else
    obs_time (float): the time since the epoch of your observation

    Returns
    -------
    (obj_ra, obj_dec) (tuple): the RA and Dec of the object you looked up
    """
    # put the time in a readable format for astropy
    obs_time = Time(obs_time, format='unix_tai')

    # use package for finding objects within our solar system
    if obj_type == 'planet':
        with solar_system_ephemeris.set('builtin'):
            obj = get_body(obj_name, obs_time, SO_loc)
    # use package for finding objects outside of our solar system
    else:
        obj = SkyCoord.from_name(obj_name)
    obj_ra = obj.ra
    obj_dec = obj.dec
    return (obj_ra, obj_dec)

def azel_to_radec(obs_time, az, el):
    """
    For an individual timestamp, transform from (az, el) to (ra, dec)

    Parameters
    ----------
    obs_time (float): the time since the epoch of your observation (s)
    az (float): encoder azimuth position (degrees)
    el (float): encoder elevation position (degrees)

    Returns
    -------
    (RA, Dec) (tuple): the RA and Dec seen during the observation
    """
    # put the time in a readable format for astropy
    obs_time = Time(obs_time, format='unix_tai')

    # use the observatory location and time to find available observations
    SO_coord = AltAz(location=SO_loc, obstime=obs_time)

    # use the SO frame to get the astro az and el at that time
    AzEl = SkyCoord(az, el, unit="deg", frame=SO_coord)

    # convert to ra, dec
    RaDec = AzEl.transform_to('icrs')
    RA = RaDec.ra.deg
    Dec = RaDec.dec.deg
    return(RA, Dec)

def radec_to_azel(obs_time, ra, dec, ra_unit, dec_unit):
    """
    For an individual timestamp, transform from (ra, dec) to (az, el)

    Parameters
    ----------
    obs_time (float): the time since teh epoch of your observation (s)
    ra (float or str): the RA position you want to find az/el for
    dec (float or str): the Dec posiiton you want to find az/el for
    ra_unit (str): the units of RA ('deg' or 'hourangle')
    dec_unit (str): the units of Dec ('deg' or 'hourangle')

    Returns
    -------
    (az, el) (tuple): azimuth and elevation in degrees
    """
    obs_time = Time(obs_time, format='unix_tai')

    SO_coord = AltAz(location=SO_loc, obstime=obs_time)
    units = {'ra': ra_unit, 'dec': dec_unit}
    for rd in units.keys():
        if units[rd] == 'deg':
            units[rd] = u.deg
        elif units[rd] == 'hourangle':
            units[rd] = u.hourangle
        else:
            raise
    obs = SkyCoord(ra=ra, dec=dec, unit=(units['ra'], units['dec']), obstime=obs_time)
    altaz = obs.transform_to(SO_coord)
    az = altaz.az.deg
    el = altaz.alt.deg
    return (az, el)

if __name__ == "__main__":
    obs_time = time.time()
    # find jupiter
    jup = find_an_object('jupiter', 'planet', obs_time)
    print('Jupiter (ra, dec): ' + str(jup))

    # find the ra, dec of az=265, el=35
    tel = azel_to_radec(obs_time, az=265, el=35)
    print('Azimuth=265, Elevation=35 transforms to '+str(tel))

    # find the az, el of the Crab Nebula
    cn_ra, cn_dec = find_an_object('Crab Nebula', 'nebula', obs_time)
    az, el = radec_to_azel(obs_time, cn_ra, cn_dec, 'deg', 'deg')
    print('(Az, El) of Crab Nebula is ' + str((az, el)))

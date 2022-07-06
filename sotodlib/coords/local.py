import numpy as np
from astropy import units as u


"""
A set of functions used in to convert from GPS coordinates to Horizontal coordinates. 
These functions are used in the toast3 drone simulation operator. 

The coordinate systems introduced here are:

- ECEF: Earth Center, Earth Fixed Coordinate System (geocentric coordinates system). 
        It uses a cartesian system centered at the Earth Center-of-Mass to represent
        locations.
- Geodetic: Uses an angular coordinate system (longitude, latitude and geodetic height)
            to describe the Earth. The coordinate system is defined by a datum, which 
            describes the 3D shape of the Earth. The most common datum is the WGS-84, 
            which is also used by GPS and represents the Earth as an Ellipsoid. 
- ENU: Earth-North-Up local coordinate system. It is based on the tangent plane defined 
       by the local vertical direction and the Earth's axis of rotation. The resulting 
       cartesian system has three coordinates: one gives the position along the northern 
       axis (tangent to the meridians), one along the eastern axis (tangent to the paralles)
       and one representing the vertical direction (normal direction to the chosen geodetic
       datum). To maintain the right-hand convention, x is along the East axis, y along 
       the North axis and Z along the Up axis.

Most of the functions are adapted from pymap3d, to include error calulations 
"""

def ellipsoid(model='WGS84'):
    """Return the major and minor semiaxis given an ellipsoid model.
    
    Returns
    -------
    a : astropy.Quantity, semiaxis major in meter
    b : astropy.Quantity, semiaxis minor in meter
    
    """

    if model == 'WGS84':
        a = 6378137.0*u.meter         
        b = 6356752.31424518*u.meter 

    return a, b

def _check_quantity(val, unit):

    if isinstance(val, u.Quantity):
        return val.to(unit)
    else:
        return val * unit

def ecef2lonlat(x, y, z, ell='WGS84'):
    """Convert ECEF to geodetic coordinates

    Based on:  You, Rey-Jer. (2000). 'Transformation of Cartesian to Geodetic
    Coordinates without Iterations'.  Journal of Surveying Engineering. 
    doi: 10.1061/(ASCE)0733-9453

    Parameters
    ----------
    x : float, array (numpy or Quantity)
        target x ECEF coordinate. If numpy float or array, it is considered in meters
    y : float, array (numpy or Quantity)
        target y ECEF coordinate. If numpy float or array, it is considered in meters
    z : float, array (numpy or Quantity)
        target z ECEF coordinate. If numpy float or array, it is considered in meters
    ell : string, optional
        reference ellipsoid
    
    Returns
    -------
    lat : float, array (Quantity)
        target geodetic latitude 
    lon : float, array (Quantity)
        target geodetic longitude
    alt : float, array (Quantity)
        target altitude above geodetic ellipsoid
    
    """

    x = _check_quantity(x, u.meter)
    y = _check_quantity(y, u.meter)
    z = _check_quantity(z, u.meter)

    a, b = ellipsoid(ell)

    r = np.sqrt(x**2 + y**2 + z**2)

    E = np.sqrt(a**2 - b**2)

    # eqn. 4a
    U = np.sqrt(0.5*(r**2 - E**2) + 0.5*np.sqrt((r**2 - E**2)**2 + 4*E**2*z**2))

    Q = np.sqrt(x**2 + y**2)
    huE = np.sqrt(U**2 + E**2)

    Beta = np.arctan2(huE*z, U*Q)

    # eqn. 13
    eps = ((b*U - a*huE + E**2)*np.sin(Beta)) / (
        a*huE*1 / np.cos(Beta) - E**2*np.cos(Beta)
    )
    Beta += eps

    lat = np.arctan2(a*np.tan(Beta), b)

    lon = np.arctan2(y, x)

    # eqn. 7
    alt = np.hypot(z - b * np.sin(Beta), Q - a * np.cos(Beta))

    # inside ellipsoid?
    inside = (x**2/a**2 + y**2/a**2+ z**2/b**2 < 1)

    try:
        if inside.any():  # type: ignore
            # avoid all false assignment bug
            alt[inside] = -alt[inside]
    except (TypeError, AttributeError):
        if inside:
            alt = -alt

    return lon, lat, alt

def hor2enu(az, el, srange, deg=True):
    """Azimuth, Elevation, Slant range to target to East, North, Up
    
    Parameters
    ----------
    azimuth : float, array (numpy or Quantity)
        azimuth, if numpy float or array the unit is determined by the deg parameter
    elevation : float
        elevation, if numpy float or array the unit is determined by the deg parameter
    srange : float, array (numpy or Quantity)
        slant range. If numpy float or array, it is considered in meters
    deg : bool, optional
        if azimuth and elevation are not astropy quantities, if True set them to degrees 
    
    Returns
    --------
    E : Quantity
        East ENU coordinate (meters)
    N : float
        North ENU coordinate (meters)
    U : Quantity
        Up ENU coordinate (meters)
    
    """

    if deg:
        azel_unit = u.deg
    else:
        azel_unit = u.rad

    az = _check_quantity(az, azel_unit)
    el = _check_quantity(el, azel_unit)
    srange = _check_quantity(srange, u.meter)

    r = srange * np.cos(el)

    return r*np.sin(az), r*np.cos(az), srange*np.sin(el)

def enu2ecef(E, N, U, lon, lat, alt, ell='WGS84', deg=True):
    """East, North, Up to ECEF.

    Parameters
    ----------
    E : float, array (numpy or Quantity)
        target east ENU coordinate. If numpy float or array, it is considered in meters
    N : float, array (numpy or Quantity)
        target north ENU coordinate. If numpy float or array, it is considered in meters
    U : float, array (numpy or Quantity)
        target up ENU coordinate. If numpy float or array, it is considered in meters
    lon : float, array (numpy or Quantity)
        geodetic longitude of the observer, if numpy float or array the unit is 
        determined by the deg parameter
    lat : float, array (numpy or Quantity)
        geodetic latitude of the observer, if numpy float or array the unit is 
        determined by the deg parameter 
    alt : float, array (numpy or Quantity)
        altitude above geodetic ellipsoid of the observer, If numpy float or array, 
        it is considered in meters
    ell : string, optional
        reference ellipsoid
    deg : bool, optional
        if lon and lat are not quantities, if True set them to degrees 
    
    Returns
    -------
    x : Quantity
        target x ECEF coordinate
    y : Quantity
        target y ECEF coordinate
    z : Quantity
        target z ECEF coordinate
    
    """

    if deg:
        lonlat_unit = u.deg
    else:
        lonlat_unit = u.rad

    lon = _check_quantity(lon, lonlat_unit)
    lat = _check_quantity(lat, lonlat_unit)
    alt = _check_quantity(alt, u.meter)

    E = _check_quantity(E, u.meter)
    N = _check_quantity(N, u.meter)
    U = _check_quantity(U, u.meter)

    t = np.cos(lat) * U - np.sin(lat) * N
    dz = np.sin(lat) * U + np.cos(lat) * N

    dx = np.cos(lon) * t - np.sin(lon) * E
    dy = np.sin(lon) * t + np.cos(lon) * E

    x0, y0, z0, _, _, _ = lonlat2ecef(lon, lat, alt, ell, deg=deg)

    return x0 + dx, y0 + dy, z0 + dz

def lonlat2ecef(lon, lat, alt, ell='WGS84', deg=True, uncertainty=False, \
                delta_lon=0, delta_lat=0, delta_alt=0):
    """Convert geodetic coordinates to ECEF

    Parameters
    ----------
    lon : float, array (numpy or Quantity)
        geodetic longitude, if numpy float or array the unit is determined by the 
        deg parameter
    lat : float, array (numpy or Quantity)
        geodetic latitude, if numpy float or array the unit is determined by the 
        deg parameter 
    alt : float, array (numpy or Quantity)
        altitude above geodetic ellipsoid, If numpy float or array, it is considered 
        in meters
    ell : string, optional
        reference ellipsoid
    deg : bool, optional
        if azimuth and elevation are not astropy quantities, if True set them to 
        degrees 
    ell : string, optional
        reference ellipsoid
    uncertainty : bool, optional
        if True computes the uncertainties associated in ECEF coordinates
    delta_lon : float, array (numpy or Quantity)
        delta geodetic longitude, if numpy float or array the unit is determined by 
        the deg parameter
    delta_lat : float, array (numpy or Quantity)
        delta geodetic latitude, if numpy float or array the unit is determined by 
        the deg parameter
    delta_alt : float, array (numpy or Quantity)
        delta altitude above geodetic ellipsoid, if numpy float or array the unit is 
        determined by the deg parameter
    
    Returns
    -------
    x : Quantity
        target x ECEF coordinate
    y : Quantity
        target y ECEF coordinate
    z : Quantity
        target z ECEF coordinate
    
    """

    if deg:
        lonlat_unit = u.deg
    else:
        lonlat_unit = u.rad

    lon = _check_quantity(lon, lonlat_unit)
    lat = _check_quantity(lat, lonlat_unit)
    alt = _check_quantity(alt, u.meter)
    
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    a, b = ellipsoid(ell)
        
    N = a**2/np.sqrt(a**2*cos_lat**2 + b**2*sin_lat**2)

    x = (N + alt)*cos_lat*cos_lon
    y = (N + alt)*cos_lat*sin_lon
    z = (N*(b/a)**2 + alt)*sin_lat
    
    if uncertainty:

        delta_lat = _check_quantity(delta_lat, lonlat_unit)
        delta_lon = _check_quantity(delta_lon, lonlat_unit)
        delta_alt = _check_quantity(delta_alt, u.meter)
        
        delta_x, delta_y, delta_z = _lonlat2ecef_error(cos_lon, cos_lat, sin_lon, sin_lat, \
                                                       alt, a, b, delta_lat, delta_lon, delta_alt)
    
    else:
        delta_x, delta_y, delta_z = np.zeros_like(x)*u.meter, \
            np.zeros_like(y)*u.meter, \
            np.zeros_like(z)*u.meter      
    
    return x, y, z, delta_x, delta_y, delta_z

def _lonlat2ecef_error(cos_lon, cos_lat, sin_lon, sin_lat, alt, a, b, \
                       delta_lon, delta_lat, delta_alt):
    """Compute the error in ECEF coordinates.
    
    This uses parameters derived in the lonlat2ecef function. Check that function
    for a deeper explanation.

    """

    delta_x = np.sqrt(delta_alt**2*cos_lon**2*cos_lat**2+\
                      delta_lon**2*(a**2/np.sqrt(a**2*cos_lat**2 + \
                                                 b**2*sin_lat**2) + \
                                    alt)**2*sin_lon**2*cos_lat**2 +\
                      delta_lat**2*(a**2*(a**2*sin_lat*cos_lat - \
                                          b**2*sin_lat*cos_lat)*cos_lon*cos_lat/\
                                    (a**2*cos_lat**2 + b**2*sin_lat**2)**(3/2) - \
                                    (a**2/np.sqrt(a**2*cos_lat**2 + b**2*sin_lat**2) +\
                                    alt)*sin_lat*cos_lon)**2)
        
    delta_y = np.sqrt(delta_alt**2*sin_lon**2*cos_lat**2 + \
                      delta_lon**2*(a**2/np.sqrt(a**2*cos_lat**2 + \
                                                 b**2*sin_lat**2) + \
                                    alt)**2*cos_lon**2*cos_lat**2 + \
                      delta_lat**2*(a**2*(a**2*sin_lat*cos_lat - \
                                          b**2*sin_lat*cos_lat)*sin_lon*cos_lat/\
                                    (a**2*cos_lat**2 + b**2*sin_lat**2)**(3/2) - \
                                    (a**2/np.sqrt(a**2*cos_lat**2 + b**2*sin_lat**2) + \
                                        alt)*sin_lon*sin_lat)**2)
    
    delta_z = np.sqrt(delta_alt**2*sin_lat**2 + \
                      delta_lat**2*(b**2*(a**2*sin_lat*cos_lat - \
                                          b**2*sin_lat*cos_lat)*sin_lat/\
                                    (a**2*cos_lat**2 + b**2*sin_lat**2)**(3/2) +\
                                    (b**2/np.sqrt(a**2*cos_lat**2 + b**2*sin_lat**2) +\
                                        alt)*cos_lat)**2)

    return delta_x, delta_y, delta_z
    
def ecef2enu(ecef_obs_x, ecef_obs_y, ecef_obs_z, \
             ecef_target_x, ecef_target_y, ecef_target_z, \
             delta_obs_x, delta_obs_y, delta_obs_z, \
             delta_target_x, delta_target_y, delta_target_z, \
             lon, lat, deg=True):
    """Return the position relative to a refence point in a ENU system. 
    
    If one of delta_obs or delta_target are different from zeros then uncertainties is 
    calculated automatically. The array returns as EAST, NORTH, UP
    
    Parameters
    ----------
    ecef_obs_x : float or array, numpy or Quantity
        x ECEF coordinates array of the observer, if numpy float or array the unit
        is meter
    ecef_obs_y : float or array, numpy or Quantity
        y ECEF coordinates array of the observer, if numpy float or array the unit 
        is meter
    ecef_obs_z : float or array, numpy or Quantity
        z ECEF coordinates array of the observer, if numpy float or array the unit 
        is meter
    ecef_target_x : float or array, numpy or Quantity
        x ECEF coordinates array of the target, if numpy float or array the unit 
        is meter
    ecef_target_y : float or array, numpy or Quantity
        y ECEF coordinates array of the target, if numpy float or array the unit 
        is meter
    ecef_target_z : float or array, numpy or Quantity
        z ECEF coordinates array of the target, if numpy float or array the unit 
        is meter
    delta_obs_x : numpy or Quantity, numpy or Quantity
        x ECEF Coordinates error of the observer, if numpy array the unit is meter
    delta_obs_y : numpy or Quantity, numpy or Quantity
        x ECEF Coordinates error of the observer, if numpy array the unit is meter
    delta_obs_z : numpy or Quantity, numpy or Quantity
        x ECEF Coordinates error of the observer, if numpy array the unit is meter
    delta_target_x : numpy or Quantity, numpy or Quantity
        x ECEF Coordinates error of the target, if numpy array the unit is meter
    delta_target_y : numpy or Quantity, numpy or Quantity
        y ECEF Coordinates error of the target, if numpy array the unit is meter
    delta_target_z : numpy or Quantity, numpy or Quantity
        z ECEF Coordinates error of the target, if numpy array the unit is meter
    lon : float, array (numpy or Quantity)
        geodetic longitude, if numpy float or array the unit is determined by the deg parameter
    lat : float, array (numpy or Quantity)
        geodetic latitude, if numpy float or array the unit is determined by the deg parameter 
    deg : bool, optional
        if azimuth and elevation are not astropy quantities, if True set them to degrees

    Returns
    --------
    E : Quantity
        East ENU coordinate error
    N : float
        North ENU coordinate error
    U : Quantity
        Up ENU coordinate error
    delta_E : Quantity
        East ENU coordinate error
    delta_N : float
        North ENU coordinate error
    delta_U : Quantity
        Up ENU coordinate error
    
    """
    
    if deg:
        lonlat_unit = u.deg
    else:
        lonlat_unit = u.rad

    lon = _check_quantity(lon, lonlat_unit)
    lat = _check_quantity(lat, lonlat_unit)

    ecef_obs_x = _check_quantity(ecef_obs_x, u.meter)
    ecef_obs_y = _check_quantity(ecef_obs_y, u.meter)
    ecef_obs_z = _check_quantity(ecef_obs_z, u.meter)

    ecef_target_x = _check_quantity(ecef_target_x, u.meter)
    ecef_target_y = _check_quantity(ecef_target_y, u.meter)
    ecef_target_z = _check_quantity(ecef_target_z, u.meter)

    delta_obs_x = _check_quantity(delta_obs_x, u.meter)
    delta_obs_y = _check_quantity(delta_obs_y, u.meter)
    delta_obs_z = _check_quantity(delta_obs_z, u.meter)

    delta_target_x = _check_quantity(delta_target_x, u.meter)
    delta_target_y = _check_quantity(delta_target_y, u.meter)
    delta_target_z = _check_quantity(delta_target_z, u.meter)
    
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    mat = np.array([[-sin_lon, cos_lon, 0],
                    [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
                    [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]])

    ecef_target = np.vstack((ecef_target_x, ecef_target_y, ecef_target_z))
    ecef_obs = np.vstack((ecef_obs_x, ecef_obs_y, ecef_obs_z))
 
    diff = ecef_target-ecef_obs
    enu = np.matmul(mat, diff)

    delta_obs = np.vstack((delta_obs_x, delta_obs_y, delta_obs_z))
    delta_target = np.vstack((delta_target_x, delta_target_y, delta_target_z))

    if np.any(delta_obs != 0) or np.any(delta_target != 0):
        delta_diff = np.sqrt(delta_obs**2+delta_target**2)
        delta_enu = np.matmul(mat, delta_diff)
    else:
        delta_enu = np.zeros_like(enu)
        delta_enu = _check_quantity(delta_enu, u.meter)
    
    return enu[0], enu[1], enu[2], delta_enu[0], delta_enu[1], delta_enu[2]

def enu2hor(E, N, U, delta_E, delta_N, delta_U, degrees=True):
    """Compute Horizontal coordinates given the coordinates in the East-North-Up system.

    Parameters
    ----------
    E : float, array (numpy or Quantity)
        target east ENU coordinate. If numpy float or array, it is considered in meters
    N : float, array (numpy or Quantity)
        target north ENU coordinate. If numpy float or array, it is considered in meters
    U : float, array (numpy or Quantity)
        target up ENU coordinate. If numpy float or array, it is considered in meters

    """

    E = _check_quantity(E, u.meter)
    N = _check_quantity(N, u.meter)
    U = _check_quantity(U, u.meter)

    delta_E = _check_quantity(delta_E, u.meter)
    delta_N = _check_quantity(delta_N, u.meter)
    delta_U = _check_quantity(delta_U, u.meter)

    s = np.sqrt(E**2+N**2+U**2)
    el = np.arctan2(U, np.sqrt(E**2+N**2))
    az = np.arctan2(E, N) % (2*np.pi*u.rad)

    if np.any(delta_E != 0) or np.any(delta_N != 0) or np.any(delta_U != 0):
    
        delta_s = np.sqrt((E**2*delta_E**2+N**2*delta_N**2+U**2*delta_U**2)/s**2)

        delta_el = np.sqrt((delta_U**2*(E**2 + N**2)**2 + \
                            U**2*(delta_E**2*E**2 + delta_N**2*N**2))/\
                           ((E**2 + N**2)*s**4))*el.unit
        
        delta_az = np.sqrt((delta_E**2*N**2 + delta_N**2*E**2)/(E**2 + N**2)**2)*az.unit
    
    else:
        delta_s = np.zeros_like(s)*u.meter
        delta_az = np.zeros_like(az)*az.unit
        delta_el = np.zeros_like(el)*el.unit

    return az, el, s, delta_az, delta_el, delta_s

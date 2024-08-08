import numpy as np
import logging
logger = logging.getLogger(__name__)

import so3g
from so3g.proj import coords, quat
import sotodlib.coords.planets as planets

def get_sso(aman, sso, nstep=100):
    """
    Function for getting xi, eta position of given sso.

    Parameters
    ----------
    aman : AxisManager
        Input axis manager.
    sso : str
        Name of input sso.
    nstep : int
        Number of steps to downsample the TOD.

    Returns
    -------
    xi : array
        Array of xi positions.
    eta : array
        Array of eta positions.
    """

    az = aman.boresight.az
    el = aman.boresight.el
    ctime = aman.timestamps
    csl = so3g.proj.CelestialSightLine.az_el(ctime[::nstep], az[::nstep], el[::nstep],
                                             roll=aman.boresight.roll[0], weather="typical")
    q_bore = csl.Q

    ras, decs = [], []
    point_sources = {'tauA': (83.6272579, 22.02159891),
                     'rcw38': (134.7805107, -47.50911231),
                     'iras16183-4958': (245.5154, -50.09168292),
                     'iras19078+0901': (287.5575891, 9.107188994),
                     'rcw122': (260.0339538, -38.95673421),
                     'cenA': (201.3625336, -43.00797508),
                     '3c279': (194.0409868, -5.79174024),
                     '3c273': (187.2775626, 2.053532671),
                     'G025.36-00.14': (279.5264042, -6.793169326),
                     'QSO_J2253+1608': (343.4952422, 16.14301323),
                     'galactic_center': (-93.5833, -29.0078)}
    for d1_unix in ctime[::nstep]:
        if sso in point_sources:
            ra = point_sources[sso][0]
            dec = point_sources[sso][1]
            planet = planets.SlowSource(d1_unix*1., float(ra) * coords.DEG,
                                        float(dec) * coords.DEG)
        else:
            try:
                planet = planets.SlowSource.for_named_source(sso, d1_unix*1.)
            except ValueError:
                break
        ra0, dec0 = planet.pos(d1_unix)
        ras.append(ra0)
        decs.append(dec0)

    planet_q = quat.rotation_lonlat(np.array(ras), np.array(decs))
    if len(q_bore) != len(planet_q):
        raise ValueError("Q vectors have inconsistent lengths")
    q_total = ~q_bore * planet_q
    xi_p, eta_p, _ = quat.decompose_xieta(np.array(q_total))

    return xi_p, eta_p

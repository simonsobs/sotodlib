import numpy as np
import logging
logger = logging.getLogger(__name__)

import so3g
from so3g.proj import coords, quat
import sotodlib.coords.planets as planets

def get_sso(aman, sso, nstep=100):

    az = aman.boresight.az
    el = aman.boresight.el
    ctime = aman.timestamps
    csl = so3g.proj.CelestialSightLine.az_el(ctime[::nstep], az[::nstep], el[::nstep],
                                             roll=aman.boresight.roll[0], weather="typical")
    q_bore = csl.Q

    ras, decs = [], []
    for d1_idx, d1_unix in enumerate(ctime[::nstep]):
        
        planet = planets.SlowSource.for_named_source(sso, d1_unix*1.)
        ra0, dec0 = planet.pos(d1_unix)
        ras.append(ra0)
        decs.append(dec0)

    planet_q = quat.rotation_lonlat(np.array(ras), np.array(decs))
    q_total = ~q_bore * planet_q
    xi_p, eta_p, _ = quat.decompose_xieta(np.array(q_total))

    return xi_p, eta_p
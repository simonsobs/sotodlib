import numpy as np
from pixell import utils
from sotodlib import coords, mapmaking
import so3g

def get_cuts(aman, sidelobes="sidelobes", object_list=None, rise_tol=1*utils.degree):
    """
    Function to calculate cuts (RangesMatrix) for bright objects (e.g.
    Sun and Moon) from binary sidelobe masks in instrument-centered
    coordinates.

    The masks are taken from the observation metadata: aman[sidelobes]
    must be an AxisManager (e.g. as produced by the "Enmap" metadata
    loader) with a LabelAxis "fields" naming the objects (e.g.
    ['moon', 'sun']), and a "map" member holding a pixell enmap of
    shape [nfields, ny, nx].  Each map is a mask of 0s and 1s in the
    instrument-centered coordinates of that object (see
    coords.pmat.P.for_tod with instrument_centered), for the relevant
    tube/wafer; samples where the object falls on a non-zero pixel are
    cut.  The masks should be relatively low resolution since they are
    loaded as non-tiled enmaps.

    Parameters
    ----------
    aman : sotodlib.core.AxisManager
        An observation axis manager.
    sidelobes : str, optional
        Name of the member of aman holding the sidelobe masks.
    object_list : list, optional
        A list of strings indicating the objects to mask. These must be
        entries of the "fields" axis. If None, all fields are used.
    rise_tol : float, optional
        Objects that are below this elevation (in radians) at the start,
        middle and end of the observation are not masked.

    Returns
    -------
    cutss : list
        A list where the elements are the cuts for the objects requested.
        Each will be a RangesMatrix with shape (ndets,nsamps).
    """
    if sidelobes not in aman:
        raise KeyError("sidelobe masks '%s' not found in the observation metadata" % sidelobes)
    masks  = aman[sidelobes]
    fields = list(masks.fields.vals)
    if not hasattr(masks.map, "wcs"):
        raise ValueError("%s.map must be a pixell enmap" % sidelobes)
    if object_list is None: object_list = fields
    site    = mapmaking.unarr(aman.site) if "site" in aman else "so"
    weather = mapmaking.unarr(aman.weather) if "weather" in aman else "typical"
    cutss = []
    for name in object_list:
        if name not in fields:
            raise ValueError("sidelobe mask for %s not found in %s.fields %s" % (name, sidelobes, fields))
        mask = masks.map[fields.index(name)]
        cutss.append(make_cuts(aman, mask, name, site=site, weather=weather, rise_tol=rise_tol))
    return cutss

def make_cuts(aman, mask, objname, site="so", weather="typical", rise_tol=1*utils.degree):
    """Calculate the cuts (RangesMatrix, shape (ndets,nsamps)) for a single
    object, given its 2d sidelobe mask (enmap of 0s and 1s) in the
    instrument-centered coordinates of the object."""
    shape = (aman.dets.count, aman.samps.count)
    # First check if the object is above the horizon. We just check a few
    # samples to keep things simple
    ts    = aman.timestamps[[0, aman.samps.count//2, -1]]
    el    = np.array([coords.planets.get_source_azel(objname, t, site=site)[1] for t in ts])
    if not np.any(el > rise_tol): return _simple_cut(*shape)
    # Ok, it's above the horizon, check which samples are affected
    mask = mask.reshape((1,) + mask.shape[-2:]).astype(np.double) # from_map will complain if the map is not double
    pmat = coords.pmat.P.for_tod(aman, comps='T', geom=(mask.shape[-2:], mask.wcs),
        threads=False, weather=weather, site=site, instrument_centered=objname)
    # We want the raw mask value, independent of the detector T responsivity
    pmat.fp = so3g.proj.FocalPlane(quats=pmat.fp.quats)
    tod  = np.zeros(shape, dtype=np.float32)
    pmat.from_map(dest=tod, signal_map=mask, comps="T")
    return so3g.proj.RangesMatrix.from_mask(tod != 0)

def _simple_cut(ndets, nsamps):
    return so3g.proj.RangesMatrix.zeros((ndets,nsamps))

import numpy as np
from pixell import enmap, utils, coordsys
from sotodlib import coords, mapmaking
import so3g

def get_cuts(aman, args, sidelobe_cutters=None, object_list=None):
    """
    Function to calculate cuts (RangesMatrix) for Sun and Moon
    from a binary sidelobe mask with an arbitrary shape.
    The masks MUST BE full sky and relatively low resolution since
    the map is loaded as a non-tiled enmap.
    
    Parameters
    ----------
    aman : sotodlib.core.AxisManager
        An observation axis manager.
    args : dict
        An argument dictionary, which must contains the path to the
        masks in sun_mask and moon_mask. This is only used for these
        paths.
    sidelobe_cutters : dict, optional
        A dict holding SidelobeCutter for each sun and moon. In the
        first run this should be empty and will be filled. Subsequent
        runs will use these precalculated objects.
    object_list : list, optional
        A list of strings indicating the objects to mask.
        If None, sun and moon will be run.

    Returns
    -------
    cutss : list
        A list where the elements are the cuts for the objects requested.
        Each will be a RangesMatrix with shape (ndets,nsamps).
    """
    
    if object_list is None: object_list = ["sun", "moon"]
    if sidelobe_cutters is None: sidelobe_cutters = {}
    cutss = []
    for name in object_list:
        if name not in sidelobe_cutters:
            fname = args[name + "_mask"]
            if not fname: raise ValueError("config setting %s_mask missing for sidelobe cut" % name)
            sidelobe_cutters[name] = SidelobeCutter(fname, objname=name, dtype=aman.signal.dtype)
        cutter = sidelobe_cutters[name]
        cuts   = cutter.make_cuts(aman)
        cutss.append(cuts)
    return cutss

def _simple_cut(ndets, nsamps):
    return so3g.proj.RangesMatrix.zeros((ndets,nsamps))

class SidelobeCutter:
    def __init__(self, mask, objname, dtype=np.float32, rise_tol=1*utils.degree):
        self.mask = mask
        self.lmap = enmap.read_map(mask).astype(np.double) #from_map will complain if the map is not double
        self.objname= objname
        self.sys_sotodlib  = "%s,up=hor" % np.char.capitalize([objname])[0] # this is to capitalize the first letter
        self.sys_pixell = "hor,on=%s" % objname
        self.rise_tol = rise_tol
        self.dtype = dtype
    
    def make_cuts(self, aman):
        # First check if the object is above the horizon. We just check the endpoints
        # to keep things simple
        ts    = aman.timestamps[[0,-1]]
        hor   = coordsys.transform(self.sys_pixell, "hor", coordsys.Coords(ra=0, dec=0), ctime=ts, site="so", weather=mapmaking.unarr(aman.weather))
        risen = np.any(hor.el > self.rise_tol)
        if not risen: return _simple_cut(aman.signal.shape[0], aman.signal.shape[1])
        # Ok, it's above the horizon, check which samples are affected
        recenter = mapmaking.parse_recentering(self.sys_sotodlib)
        rot = mapmaking.recentering_to_quat_lonlat(*mapmaking.evaluate_recentering(
            recenter, ctime=aman.timestamps[len(aman.timestamps) // 2],
            geom=(self.lmap.shape, self.lmap.wcs),
            site=mapmaking.unarr(aman.site),
        ))
        pmat = coords.pmat.P.for_tod(aman, comps='T', geom=(self.lmap.shape, self.lmap.wcs),
            rot=rot, threads="domdir", weather=mapmaking.unarr(aman.weather),
            site=mapmaking.unarr(aman.site)
        )
        tod  = np.zeros(aman.signal.shape[:2], dtype=self.dtype)
        pmat.from_map(dest=tod, signal_map=self.lmap, comps="T",)
        cuts = so3g.proj.RangesMatrix.from_mask(tod.astype(int)) #here the mask is expected to be int
        return cuts

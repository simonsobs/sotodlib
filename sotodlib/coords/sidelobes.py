import numpy as np
from pixell import enmap, utils, coordsys
from sotodlib import coords, mapmaking
import so3g

def sidelobe_cut(obs, args, sidelobe_cutters, object_list=None):
    """
    Function to calculate cuts (RangesMatrix) for Sun and Moon
    from a binary sidelobe mask with an arbitrary shape.
    The masks MUST BE full sky and relatively low resolution since
    the map is loaded as a non-tiled enmap.

    Parameters
    ----------
    obs : sotodlib.core.AxisManager
        An observation axis manager.
    args : dict
        An argument dictionary, which must contains the path to the
        masks in mask_sun and mask_moon. This is only used for these
        paths.
    sidelobe_cutters : dict
        A dict holding SidelobeCutter for each sun and moon. In the
        first run this should be empty and will be filled. Subsequent
        runs will use these precalculated objects.
    object_list : list, optional
        A list of the objects we want to mask. If None, sun and moon
        will be run.

    Returns
    -------
    cutss : list
        A list where the elements are the cuts for the objects requested.
        Each will be a RangesMatrix with shape (ndets,nsamps).
    """

	if object_list is None: object_list = ["sun", "moon"]
	cutss = []
	for name in object_list:
		if name not in sidelobe_cutters:
			fname = args[name + "_mask"]
			if not fname: raise ValueError("config setting %s_mask missing for sidelobe cut" % name)
			sidelobe_cutters[name] = SidelobeCutter(fname, objname=name, dtype=obs.signal.dtype)
		cutter = sidelobe_cutters[name]
		cuts   = cutter.make_cuts(obs)
		cutss.append(cuts)
	return cutss

def Simplecut(ndets, nsamps):
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

    def make_cuts(self, obs):
        # First check if the object is above the horizon. We just check the endpoints
        # to keep things simple
        ts    = obs.timestamps[[0,-1]]
        hor   = coordsys.transform(self.sys_pixell, "hor", coordsys.Coords(ra=0, dec=0), ctime=ts, site="so", weather=mapmaking.unarr(obs.weather))
        risen = np.any(hor.el > self.rise_tol)
        if not risen: return Simplecut(obs.signal.shape[0], obs.signal.shape[1])
        # Ok, it's above the horizon, check which samples are affected
        try:
            recenter = mapmaking.parse_recentering(self.sys_sotodlib)
            rot = mapmaking.recentering_to_quat_lonlat(*mapmaking.evaluate_recentering(
                recenter, ctime=obs.timestamps[len(obs.timestamps) // 2],
                geom=(self.lmap.shape, self.lmap.wcs),
                site=mapmaking.unarr(obs.site),
            ))
            pmat = coords.pmat.P.for_tod(obs, comps='T', geom=(self.lmap.shape, self.lmap.wcs),
                rot=rot, threads="domdir", weather=mapmaking.unarr(obs.weather),
                site=mapmaking.unarr(obs.site)
            )
            tod  = np.zeros(obs.signal.shape[:2], dtype=self.dtype)
            forward(tod, self.lmap, pmat)
        except RuntimeError:
            # This happens when the interpolated pointing ends up outside the -npix:+2npix range
            # that gpu_mm allows. This can happen when a detector moves too close to the north pole
            # in the object-centered coordinates, which in CAR makes it seem to teleport by 180
            # degrees. When combined with unwinding, some jumps being sligthly less than 180 and
            # some slightly more than 180 can lead to the numbers drifting over a large range.
            # It might be better to catch this by just looking for values too close to the poles
            # explicitly instead of relying on gpu_mm to catch things itself
            #L.print("Error cutting %s sidelobes for %s: Pointing overflow. Skipping cut" % (self.objname, id), level=2, color=colors.red)
            return Simplecut(obs.signal.shape[0], obs.signal.shape[1])
        cuts = so3g.proj.RangesMatrix.from_mask(tod.astype(int)) #here the mask is expected to be int
        return cuts

def forward(tod, map, pmat):
    pmat.from_map(dest=tod, signal_map=map, comps="T",)

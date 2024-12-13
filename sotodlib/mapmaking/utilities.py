import numpy as np
from pixell import enmap, utils, fft, tilemap, resample
import so3g
import importlib

from .. import core
from .. import tod_ops
from .. import coords

def deslope_el(tod, el, srate, inplace=False):
    if not inplace: tod = tod.copy()
    utils.deslope(tod, w=1, inplace=True)
    f     = fft.rfftfreq(tod.shape[-1], 1/srate)
    fknee = 3.0
    with utils.nowarn():
        iN = (1+(f/fknee)**-3.5)**-1
    b  = 1/np.sin(el)
    utils.deslope(b, w=1, inplace=True)
    Nb = fft.irfft(iN*fft.rfft(b),b*0)
    amp= np.sum(tod*Nb,-1)/np.sum(b*Nb)
    tod-= amp[:,None]*b
    return tod

class ArrayZipper:
    def __init__(self, shape, dtype, comm=None):
        self.shape = shape
        self.ndof  = int(np.product(shape))
        self.dtype = dtype
        self.comm  = comm

    def zip(self, arr):  return arr.reshape(-1)

    def unzip(self, x):  return x.reshape(self.shape).astype(self.dtype, copy=False)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else self.comm.allreduce(np.sum(a*b))

class MapZipper:
    def __init__(self, shape, wcs, dtype, comm=None):
        self.shape, self.wcs = shape, wcs
        self.ndof  = int(np.product(shape))
        self.dtype = dtype
        self.comm  = comm

    def zip(self, map): return np.asarray(map.reshape(-1))

    def unzip(self, x): return enmap.ndmap(x.reshape(self.shape), self.wcs).astype(self.dtype, copy=False)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else utils.allreduce(np.sum(a*b), self.comm)

class TileMapZipper:
    def __init__(self, geo, dtype, comm):
        self.geo   = geo
        self.comm  = comm
        self.dtype = dtype
        self.ndof  = geo.size

    def zip(self, map):
        return np.asarray(map.reshape(-1))

    def unzip(self, x):
        return tilemap.TileMap(x.reshape(self.geo.pre+(-1,)).astype(self.dtype, copy=False), self.geo)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else utils.allreduce(np.sum(a*b), self.comm)

class MultiZipper:
    def __init__(self):
        self.zippers = []
        self.ndof    = 0
        self.bins    = []

    def add(self, zipper):
        self.zippers.append(zipper)
        self.bins.append([self.ndof, self.ndof+zipper.ndof])
        self.ndof += zipper.ndof

    def zip(self, *objs):
        return np.concatenate([zipper.zip(obj) for zipper, obj in zip(self.zippers, objs)])

    def unzip(self, x):
        res = []
        for zipper, (b1,b2) in zip(self.zippers, self.bins):
            res.append(zipper.unzip(x[b1:b2]))
        return res

    def dot(self, a, b):
        res = 0
        for (b1,b2), dof in zip(self.bins, self.zippers):
            res += dof.dot(a[b1:b2],b[b1:b2])
        return res

def inject_map(obs, map, recenter=None, interpol=None):
    # Infer the stokes components
    map = map.preflat
    if map.shape[0] not in [1,2,3]:
        raise ValueError("Map to inject must have either 1, 2 or 3 components, corresponding to T, QU and TQU.")
    comps = infer_comps(map.shape[0])
    # Support recentering the coordinate system
    if recenter is not None:
        ctime  = obs.timestamps
        rot    = recentering_to_quat_lonlat(*evaluate_recentering(recenter, ctime=ctime[len(ctime)//2], geom=(map.shape, map.wcs), site=unarr(obs.site)))
    else: rot = None
    # Set up our pointing matrix for the map
    pmat  = coords.pmat.P.for_tod(obs, comps=comps, geom=(map.shape, map.wcs), rot=rot, threads="domdir", interpol=self.interpol)
    # And perform the actual injection
    pmat.from_map(map.extract(shape, wcs), dest=obs.signal)

def safe_invert_div(div, lim=1e-2, lim0=np.finfo(np.float32).tiny**0.5):
    try:
        # try setting up a context manager that limits the number of threads
        from threadpoolctl import threadpool_limitse
        cm = threadpool_limits(limits=1, user_api="blas")
    except:
        # threadpoolctl not available, need a dummy context manager
        import contextlib
        cm = contextlib.nullcontext()
    with cm:
        hit = div[0,0] > lim0
        # Get the condition number of each pixel
        work    = np.ascontiguousarray(div[:,:,hit].T)
        E, V    = np.linalg.eigh(work)
        cond    = E[:,0]/E[:,-1]
        good    = cond >= lim
        # Invert the good ones
        inv_good= np.einsum("...ij,...j,...kj->...ik", V[good], 1/E[good], V[good])
        # Treat the bad ones as being purely T
        inv_bad = work[~good]*0
        inv_bad[:,0,0] = 1/work[~good,0,0]
        # Copy back
        work[good]  = inv_good
        work[~good] = inv_bad
        # And put into final output
        idiv = div*0
        idiv[:,:,hit] = work.T
    return idiv



def measure_cov(d, nmax=10000):
    d = d[:,::max(1,d.shape[1]//nmax)]
    n,m = d.shape
    step  = 10000
    res = np.zeros((n,n),d.dtype)
    for i in range(0,m,step):
        sub = mycontiguous(d[:,i:i+step])
        res += np.real(sub.dot(np.conj(sub.T)))
    return res/m

def project_out(d, modes): return d-modes.T.dot(modes.dot(d))

def project_out_from_matrix(A, V):
    # Used Woodbury to project out the given vectors from the covmat A
    if V.size == 0: return A
    Q = A.dot(V)
    return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))

def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None):
    # Translate from frequency to index
    binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
    # Make sure no bins have two few entries
    if nmin > 0:
        binds2 = [binds[0]]
        for b in binds:
            if b-binds2[-1] >= nmin: binds2.append(b)
        binds = binds2
    # Cap at nfreq and eliminate any resulting empty bins
    binds = np.unique(np.minimum(np.concatenate([[0],binds,[nfreq]]),nfreq))
    # Go from edges to [:,{from,to}]
    bins  = np.array([binds[:-1],binds[1:]]).T
    return bins

def mycontiguous(a):
    # I used this in act for some reason, but not sure why. I vaguely remember ascontiguousarray
    # causing weird failures later in lapack
    b = np.zeros(a.shape, a.dtype)
    b[...] = a[...]
    return b

def find_modes_jon(ft, bins, eig_lim=None, single_lim=0, skip_mean=False, verbose=False):
    ndet = ft.shape[0]
    vecs = np.zeros([ndet,0])
    if not skip_mean:
        # Force the uniform common mode to be included. This
        # assumes all the detectors have accurately measured gain.
        # Forcing this avoids the possibility that we don't find
        # any modes at all.
        vecs = np.concatenate([vecs,np.full([ndet,1],ndet**-0.5)],1)
    for bi, b in enumerate(bins):
        d    = ft[:,b[0]:b[1]]
        cov  = measure_cov(d)
        cov  = project_out_from_matrix(cov, vecs)
        e, v = np.linalg.eig(cov)
        e, v = e.real, v.real
        #e, v = e[::-1], v[:,::-1]
        accept = np.full(len(e), True, bool)
        if eig_lim is not None:
            # Compute median, exempting modes we don't have enough data to measure
            nsamp    = b[1]-b[0]+1
            median_e = np.median(np.sort(e)[::-1][:nsamp])
            accept  &= e/median_e >= eig_lim[bi]
        if verbose: print("bin %d: %4d modes above eig_lim" % (bi, np.sum(accept)))
        if single_lim is not None and e.size:
            # Reject modes too concentrated into a single mode. Since v is normalized,
            # values close to 1 in a single component must mean that all other components are small
            singleness = np.max(np.abs(v),0)
            accept    &= singleness < single_lim[bi]
        if verbose: print("bin %d: %4d modes also above single_lim" % (bi, np.sum(accept)))
        e, v = e[accept], v[:,accept]
        vecs = np.concatenate([vecs,v],1)
    return vecs

def measure_detvecs(ft, vecs):
    # Measure amps when we have non-orthogonal vecs
    rhs  = vecs.T.dot(ft)
    div  = vecs.T.dot(vecs)
    amps = np.linalg.solve(div,rhs)
    E    = np.mean(np.abs(amps)**2,1)
    # Project out modes for every frequency individually
    dclean = ft - vecs.dot(amps)
    # The rest is assumed to be uncorrelated
    Nu = np.mean(np.abs(dclean)**2,1)
    # The total auto-power
    Nd = np.mean(np.abs(ft)**2,1)
    return E, Nu, Nd

def sichol(A):
    iA = np.linalg.inv(A)
    try: return np.linalg.cholesky(iA), 1
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(-iA), -1

def safe_inv(a):
    with utils.nowarn():
        res = 1/a
        res[~np.isfinite(res)] = 0
    return res

def woodbury_invert(D, V, s=1):
    """Given a compressed representation C = D + sVV', compute a
    corresponding representation for inv(C) using the Woodbury
    formula."""
    V, D = map(np.asarray, [V,D])
    ishape = D.shape[:-1]
    # Flatten everything so we can be dimensionality-agnostic
    D = D.reshape(-1, D.shape[-1])
    V = V.reshape(-1, V.shape[-2], V.shape[-1])
    I = np.eye(V.shape[2])
    # Allocate our output arrays
    iD = safe_inv(D)
    iV = V*0
    # Invert each
    for i in range(len(D)):
        core = I*s + (V[i].T*iD[i,None,:]).dot(V[i])
        core, sout = sichol(core)
        iV[i] = iD[i,:,None]*V[i].dot(core)
    sout = -sout
    return iD, iV, sout

def apply_window(tod, nsamp, exp=1):
    """Apply a cosine taper to each end of the TOD."""
    if nsamp <= 0: return
    taper   = 0.5*(1-np.cos(np.arange(1,nsamp+1)*np.pi/nsamp))
    taper **= exp
    tod[...,:nsamp]  *= taper
    tod[...,-nsamp:] *= taper[::-1]

########################
###### Utilities #######
########################

def get_ids(query, context=None):
    try:
        with open(query, "r") as fname:
            return [line.split()[0] for line in fname]
    except IOError:
        return context.obsdb.query(query or "1")['obs_id']

def infer_comps(ncomp): return ["T","QU","TQU"][ncomp-1]

def parse_recentering(desc):
    """Parse an object centering description, as provided by the --center-at argument.
    The format is [from=](ra:dec|name),[to=(ra:dec|name)],[up=(ra:dec|name|system)]
    from: specifies which point is to be centered. Given as either
      * a ra:dec pair in degrees
      * the name of a pre-defined celestial object (e.g. Saturn), which should not move
        appreciably in celestial coordinates during a TOD
    to: the point at which to recenter. Optional. Given as either
      * a ra:dec pair in degrees
      * the name of a pre-defined celestial object
      Defaults to ra=0,dec=0 or ra=0,dec=90, depending on the projection
    up: which direction should point up after recentering. Optional. Given as either
      * the name of a coordinate system (e.g. hor, cel, gal), in which case
        up will point towards the north pole of that system
      * a ra:dec pair in degrees
      * the name of a pre-defined celestial object
      Defualts to the celestial north pole

    Returns "info", a bunch representing the recentering specification in more python-friendly
    terms. This can later be passed to evaluate_recentering to get the actual euler angles that perform
    the recentering.

    Examples:
      * 120.2:-13.8
        Centers on ra = 120.2°, dec = -13.8°, with up being celestial north
      * Saturn
        Centers on Saturn, with up being celestial north
      * Uranus,up=hor
        Centers on Uranus, but up is up in horizontal coordinates. Appropriate for beam mapping
      * Uranus,up=hor,to=0:90
        As above, but explicitly recenters on the north pole
    """
    # If necessary the syntax above could be extended with from_sys, to_sys and up-sys, which
    # so one could specify galactic coordiantes for example. Or one could generalize
    # from ra:dec to phi:theta[:sys], where sys would default to cel. But for how I think
    # this is enough.
    args = desc.split(",")
    info  = {"to":"auto", "up":"cel", "from_sys":"cel", "to_sys":"cel", "up_sys":"cel"}
    for ai, arg in enumerate(args):
        # Split into key,value
        toks = arg.split("=")
        if ai == 0 and len(toks) == 1:
            key, val = "from", toks[0]
        elif len(toks) == 2:
            key, val = toks
        else:
            raise ValueError("parse_recentering wants key=value format, but got %s" % (arg))
        # Handle the values
        if ":" in val:
            val = [float(w)*utils.degree for w in val.split(":")]
        info[key] = val
    if "from" not in info:
        raise ValueError("parse_recentering needs at least the from argument")
    return info

def evaluate_recentering(info, ctime, geom=None, site=None, weather="typical"):
    """Evaluate the quaternion that performs the coordinate recentering specified in
    info, which can be obtained from parse_recentering."""
    import ephem
    # Get the coordinates of the from, to and up points. This was a bit involved...
    def to_cel(lonlat, sys, ctime=None, site=None, weather=None):
        # Convert lonlat from sys to celestial coorinates. Maybe polish and put elswhere
        if sys == "cel" or sys == "equ":
            return lonlat
        elif sys == "hor":
            return so3g.proj.CelestialSightLine.az_el(ctime, lonlat[0], lonlat[1], site=site, weather=weather).coords()[0,:2]
        else:
            raise NotImplementedError
    def get_pos(name, ctime, sys=None):
        if isinstance(name, str):
            if name in ["hor", "cel", "equ", "gal"]:
                return to_cel([0,np.pi/2], name, ctime, site, weather)
            elif name == "auto":
                return np.array([0,0]) # would use geom here
            else:
                obj = getattr(ephem, name)()
                djd = ctime/86400 + 40587.0 + 2400000.5 - 2415020
                obj.compute(djd)
                return np.array([obj.a_ra, obj.a_dec])
        else:
            return to_cel(name, sys, ctime, site, weather)
    p1 = get_pos(info["from"], ctime, info["from_sys"])
    p2 = get_pos(info["to"],   ctime, info["to_sys"])
    pu = get_pos(info["up"],   ctime, info["up_sys"])
    return [p1,p2,pu]

def recentering_to_quat_lonlat(p1, p2, pu):
    """Return the quaternion that represents the rotation that takes point p1
    to p2, with the up direction pointing towards the point pu, all given as lonlat pairs"""
    from so3g.proj import quat
    # 1. First rotate our point to the north pole: Ry(-(90-dec1))Rz(-ra1)
    # 2. Apply the same rotation to the up point.
    # 3. We want the up point to be upwards, so rotate it to ra = 180°: Rz(pi-rau2)
    # 4. Apply the same rotation to the real point
    # 5. Rotate the point to its target position: Rz(ra2)Ry(90-dec2)
    ra1, dec1 = p1
    ra2, dec2 = p2
    rau, decu = pu
    qu    = quat.rotation_lonlat(rau, decu)
    R     = ~quat.rotation_lonlat(ra1, dec1)
    rau2  = quat.decompose_lonlat(R*qu)[0]
    R     = quat.euler(2, ra2)*quat.euler(1, np.pi/2-dec2)*quat.euler(2, np.pi-rau2)*R
    a = quat.decompose_lonlat(R*quat.rotation_lonlat(ra1,dec1))
    return R

def highpass(tod, fknee=1e-2, alpha=3):
    ft   = fft.rfft(tod)
    freq = fft.rfftfreq(tod.shape[1])
    ft  /= 1 + (freq/fknee)**-alpha
    return fft.irfft(ft, tod, normalize=True)

def find_boresight_jumps(vals, width=20, tol=0.1):
    # median filter array to get reference behavior
    bad   = np.zeros(vals.size,dtype=bool)
    width = int(width)//2*2+1
    fvals = utils.block_mean_filter(vals, width)
    bad  |= np.abs(vals-fvals) > tol
    return bad

def robust_unwind(a, period=2*np.pi, cut=None, tol=1e-3, mask=None):
    """Like utils.unwind, but only registers something as an angle jump if
    it is of just the right shape. If cut is specified, it should be a list
    of valid angle cut positions, which will further restrict when jumps are
    allowed. Only 1d input is supported."""
    # Find places where a jump would be acceptable
    period = float(period)
    diffs  = (a[1:]-a[:-1])/period
    valid  = np.abs(np.abs(diffs)-1) < tol/period
    diffs *= valid
    jumps  = np.concatenate([[0],np.round(diffs)])
    if mask is not None:
        jumps[mask] = 0
        jumps[:-1][mask[1:]] = 0
    if cut is not None:
        near_cut = np.zeros(a.size, bool)
        for cutval in cut:
            near_cut |= np.abs((a - cutval + period/2) % period + period/2) < tol
        jumps[~near_cut] = 0
    # Then correct our values
    return a - np.cumsum(jumps)*period

def find_elevation_outliers(el, tol=0.5*utils.degree):
    typ = np.median(el[::100])
    return np.abs(el-typ)>tol

def freq2ind(freqs, srate, nfreq, rfun=None):
    """Returns the index of the first fourier mode with greater than freq
    frequency, for each freq in freqs."""
    if freqs is None: return freqs
    if rfun  is None: rfun = np.ceil
    return rfun(np.asarray(freqs)/(srate/2.0)*nfreq).astype(int)

def rangemat_sum(rangemat):
    res = np.zeros(len(rangemat))
    for i, r in enumerate(rangemat):
        ra = r.ranges()
        res[i] = np.sum(ra[:,1]-ra[:,0])
    return res

def find_usable_detectors(obs, maxcut=0.1):
    ncut  = rangemat_sum(obs.flags.glitch_flags)
    good  = ncut < obs.samps.count * maxcut
    return obs.dets.vals[good]

def fix_boresight_glitches(obs, ang_tol=0.1*utils.degree, t_tol=1):
    az   = robust_unwind(obs.boresight.az)
    bad  = find_boresight_jumps(az,               tol=ang_tol)
    bad |= find_boresight_jumps(obs.boresight.el, tol=ang_tol)
    bad |= find_boresight_jumps(obs.timestamps,   tol=t_tol)
    bcut = so3g.RangesInt32.from_mask(bad)
    obs.boresight.az[:] = az
    tod_ops.get_gap_fill_single(obs.timestamps,   bcut, swap=True)
    tod_ops.get_gap_fill_single(obs.boresight.az, bcut, swap=True)
    tod_ops.get_gap_fill_single(obs.boresight.el, bcut, swap=True)

def unarr(a): return np.array(a).reshape(-1)[0]

def downsample_ranges(ranges, down):
    """Downsample either an array of ranges [:,{from,to}]
    or an so3gRangesInt32 object, by the integer factor down.
    The downsampling is inclusive: The output ranges will be
    as small as possible while fully encompassing the input ranges."""
    if isinstance(ranges, so3g.RangesInt32):
        return so3g.RangesInt32.from_array(downsample_ranges(ranges.ranges(), down), (ranges.count+down-1)//down)
    oranges = ranges.copy()
    # Lower range is simple
    oranges[:,0] //= down
    # End should be one above highest impacted index.
    oranges[:,1] = (oranges[:,1]-1)//down+1
    return oranges

def downsample_cut(cut, down):
    """Given an integer RangesMatrix respresenting samples to cut,
    return a new such RangesMatrix that describes which samples to
    cut if the timestream were to be downsampled by the integer
    factor down."""
    return so3g.proj.ranges.RangesMatrix([downsample_ranges(r,down) for r in cut.ranges])

def downsample_obs(obs, down):
    """Downsample AxisManager obs by the integer factor down.

    This implementation is quite specific and probably needs
    generalization in the future, but it should work correctly
    and efficiently for ACT-like data at least. In particular
    it uses fourier-resampling when downsampling the detector
    timestreams to avoid both aliasing noise and introducing
    a transfer function."""
    assert down == utils.nint(down), "Only integer downsampling supported, but got '%.8g'" % down
    # Compute how many samples we will end up with
    onsamp = (obs.samps.count+down-1)//down
    # Set up our output axis manager
    res    = core.AxisManager(obs.dets, core.IndexAxis("samps", onsamp))
    # Stuff without sample axes
    for key, axes in obs._assignments.items():
        if "samps" not in axes:
            val = getattr(obs, key)
            if isinstance(val, core.AxisManager):
                res.wrap(key, val)
            else:
                axdesc = [(k,v) for k,v in enumerate(axes) if v is not None]
                res.wrap(key, val, axdesc)
    # The normal sample stuff
    res.wrap("timestamps", obs.timestamps[::down], [(0, "samps")])
    bore = core.AxisManager(core.IndexAxis("samps", onsamp))
    for key in ["az", "el", "roll"]:
        bore.wrap(key, getattr(obs.boresight, key)[::down], [(0, "samps")])
    res.wrap("boresight", bore)
    res.wrap("signal", resample.resample_fft_simple(obs.signal, onsamp), [(0,"dets"),(1,"samps")])

    # The cuts
    # obs.flags will contain all types of flags. We should query it for glitch_flags and source_flags
    cut_keys = ["glitch_flags"]

    if "source_flags" in obs.flags:
        cut_keys.append("source_flags")

    # We need to add a res.flags FlagManager to res
    res = res.wrap('flags', core.FlagManager.for_tod(res))

    for key in cut_keys:
        res.flags.wrap(key, downsample_cut(getattr(obs.flags, key), down), [(0,"dets"),(1,"samps")])

    # Not sure how to deal with flags. Some sort of or-binning operation? But it
    # doesn't matter anyway
    return res

def import_optional(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except:
        return None

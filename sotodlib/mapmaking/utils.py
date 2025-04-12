from typing import Optional, Any, Union
from sqlalchemy import create_engine, exc
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker
import importlib
import numpy as np
import so3g
from pixell import enmap, fft, resample, tilemap, bunch, utils as putils

from .. import coords, core, tod_ops


def deslope_el(tod, el, srate, inplace=False):
    if not inplace: tod = tod.copy()
    putils.deslope(tod, w=1, inplace=True)
    f     = fft.rfftfreq(tod.shape[-1], 1/srate)
    fknee = 3.0
    with putils.nowarn():
        iN = (1+(f/fknee)**-3.5)**-1
    b  = 1/np.sin(el)
    putils.deslope(b, w=1, inplace=True)
    Nb = fft.irfft(iN*fft.rfft(b),b*0)
    amp= np.sum(tod*Nb,-1)/np.sum(b*Nb)
    tod-= amp[:,None]*b
    return tod

class ArrayZipper:
    def __init__(self, shape, dtype, comm=None):
        self.shape = shape
        self.ndof  = int(np.prod(shape))
        self.dtype = dtype
        self.comm  = comm

    def zip(self, arr):  return arr.reshape(-1)

    def unzip(self, x):  return x.reshape(self.shape).astype(self.dtype, copy=False)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else self.comm.allreduce(np.sum(a*b))

class MapZipper:
    def __init__(self, shape, wcs, dtype, comm=None):
        self.shape, self.wcs = shape, wcs
        self.ndof  = int(np.prod(shape))
        self.dtype = dtype
        self.comm  = comm

    def zip(self, map): return np.asarray(map.reshape(-1))

    def unzip(self, x): return enmap.ndmap(x.reshape(self.shape), self.wcs).astype(self.dtype, copy=False)

    def dot(self, a, b):
        return np.sum(a*b) if self.comm is None else putils.allreduce(np.sum(a*b), self.comm)

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
        return np.sum(a*b) if self.comm is None else putils.allreduce(np.sum(a*b), self.comm)

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
    pmat  = coords.pmat.P.for_tod(obs, comps=comps, geom=(map.shape, map.wcs), rot=rot, threads="domdir", interpol=interpol)
    # And perform the actual injection
    pmat.from_map(map.extract(map.shape, map.wcs), dest=obs.signal)

def safe_invert_div(div, lim=1e-2, lim0=np.finfo(np.float32).tiny**0.5):
    try:
        # try setting up a context manager that limits the number of threads
        from threadpoolctl import threadpool_limits
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

def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None, cap=True):
    # Urk, this function is ugly. It's an old one from when I
    # first started learning python.
    # Translate from frequency to index
    binds = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
    if cap: binds = np.concatenate([[0],binds,[nfreq]])
    # Cap at nfreq and eliminate any resulting empty bins
    binds = np.unique(np.minimum(binds,nfreq))
    # Make sure no bins have two few entries
    if nmin > 0:
        binds2 = [binds[0]]
        for b in binds[1:-1]:
            if b-binds2[-1] >= nmin: binds2.append(b)
        binds2.append(binds[-1])
        # If the last bin is too short, remove the second-to-last entry
        if binds2[-1]-binds2[-2] < nmin:
            del binds2[-2]
        binds = binds2
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

def measure_detvecs(ft, vecs, nper=2):
    # Allow too narrow bins
    nfull = vecs.shape[1]
    n     = min(nfull, ft.shape[-1]//nper+1)
    vecs  = vecs[:,:n]
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
    # Expand E to full requested len with low but
    # non-zero power so we don't need to special-case elsewhere
    small = np.min(Nu)*1e-6
    E  = np.pad(E, (0,nfull-n), constant_values=small)
    return E, Nu, Nd

def sichol(A):
    iA = np.linalg.inv(A)
    try: return np.linalg.cholesky(iA), 1
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(-iA), -1

def safe_inv(a):
    with putils.nowarn():
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

def get_subids(query, context=None, method="auto"):
    """A subid has the form obs_id:wafer_slot:band, and is a natural
    unit to use for mapmaking."""
    if method == "auto":
        try: return get_subids_file(query, context=context)
        except IOError: return get_subids_query(query, context=context)
    elif method == "file":
        return get_subids_file(query, context=context)
    elif method == "query":
        return get_subids_query(query, context=context)
    else:
        raise ValueError("Unrecognized method for get_subids: '%s'" % (str(method)))

def get_subids_query(query, context):
    obs_ids = context.obsdb.query(query or "1")['obs_id']
    sub_ids = expand_ids(obs_ids, context)
    return sub_ids

def get_subids_file(fname, context=None):
    with open(fname, "r") as fname:
        sub_ids = [line.split()[0] for line in fname]
    sub_ids = expand_ids(sub_ids, context=context)
    return sub_ids

def expand_ids(obs_ids, context=None, bands=None):
    """Given a list of ids that are either obs_ids or sub_ids, expand any obs_ids
    into sub_ids and return the resulting list.

    To infer the bands available either bands or context must be passed.
    If bands is passed then it should be a list of the multichroic bands
    available for all the wafers. Example: bands=["f090","f150"].

    Otherwise a Context should be passed, and the bands will be inferred
    per obs by querying its obsdb. This is the standard case.
    """
    if len(obs_ids) == 0: return []
    # Get the tube flavor for each. We will need this to get the bands.
    if context is not None:
        # Make sure we have plain obs_id regardless of whether obs_ids or sub_ids were passed
        actual_obs_ids = np.char.partition(obs_ids, ":")[:,0]
        # Get the tube flavor for each, and their meanings
        all_ids, flavors = [], []
        for row in context.obsdb.conn.execute("select obs_id, tube_flavor from obs"):
            all_ids.append(row[0])
            flavors.append(row[1])
        all_ids = np.array(all_ids)
        inds    = putils.find(all_ids, actual_obs_ids)
        flavors = [flavors[ind].lower() for ind in inds]
        flavor_map = {
            "lf": ("f030", "f040"),
            "mf": ("f090", "f150"),
            "hf": ("f150", "f220"),
            "uhf": ("f220", "f280"),
            None: ("f000",),
        }
    elif bands is not None:
        flavors    = ["a"]*len(obs_ids)
        flavor_map = {"a": bands}
    else:
        raise ValueError("Either bands or context must be passed")
    # Then loop through each and expand as necessary
    sub_ids = []
    for obs_id, flavor in zip(obs_ids, flavors):
        bands  = flavor_map[flavor]
        toks   = obs_id.split(":")
        if len(toks) == 3:
            # Already sub_id
            sub_ids.append(obs_id)
        elif len(toks) != 1:
            raise ValueError("Invalid obs_id '%s'" % (str(obs_id)))
        else:
            toks = obs_id.split("_")
            # SAT format: obs_1719902396_satp1_1111111
            # LAT format: 1696118940_i3_100 (why no obs in front?)
            wafer_mask = toks[-1]
            # Loop through wafer slots
            for si, status in enumerate(wafer_mask):
                if status == "1":
                    for band in bands:
                        sub_ids.append("%s:ws%d:%s" % (obs_id, si, band))
    return sub_ids

def filter_subids(subids, wafers=None, bands=None):
    subids = np.asarray(subids)
    if wafers is not None:
        wafs   = astr_tok(subids,":",1)
        subids = subids[np.isin(wafs, wafers)]
    if bands is not None:
        bpass  = astr_tok(subids,":",2)
        subids = subids[np.isin(bpass, bands)]
    return subids

def astr_cat(*arrs):
    res = np.char.add(arrs[0], arrs[1])
    for arr in arrs[2:]:
        res = np.char.add(res,arr)
    return res

def astr_tok(astr, sep, i):
    for j in range(i):
        astr = np.char.partition(astr,sep)[:,2]
    return np.char.partition(astr,sep)[:,0]

def split_subids(subids):
    ids, _, rest   = np.char.partition(subids, ":").T
    wafs, _, bands = np.char.partition(rest, ":").T
    return ids, wafs, bands

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
            val = [float(w)*putils.degree for w in val.split(":")]
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
    fvals = putils.block_mean_filter(vals, width)
    bad  |= np.abs(vals-fvals) > tol
    return bad

def robust_unwind(a, period=2*np.pi, cut=None, tol=1e-3, mask=None):
    """Like putils.unwind, but only registers something as an angle jump if
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

def find_elevation_outliers(el, tol=0.5*putils.degree):
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

def find_usable_detectors(obs, maxcut=0.1, glitch_flags: str = "flags.glitch_flags"):
    ncut  = rangemat_sum(obs[glitch_flags])
    good  = ncut < obs.samps.count * maxcut
    return obs.dets.vals[good]

def fix_boresight_glitches(obs, ang_tol=0.1*putils.degree, t_tol=1):
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

def get_wrappable(axman, key):
    val = getattr(axman, key)
    if isinstance(val, core.AxisManager):
        return key,val
    else:
        axes   = axman._assignments[key]
        axdesc = [(k,v) for k,v in enumerate(axes) if v is not None]
        return key, val, axdesc

def downsample_obs(obs, down, skip_signal=False, fft_resample=["signal"]):
    """Downsample AxisManager obs by the integer factor down.

    This implementation is quite specific and probably needs
    generalization in the future, but it should work correctly
    and efficiently for ACT-like data at least. In particular
    it uses fourier-resampling when downsampling the detector
    timestreams to avoid both aliasing noise and introducing
    a transfer function."""
    assert down == putils.nint(down), "Only integer downsampling supported, but got '%.8g'" % down
    if down == 1: return obs
    if "samps" not in obs: return obs
    # Compute how many samples we will end up with
    onsamp = (obs.samps.count+down-1)//down
    # Set up our output axis manager
    axes   = [obs[axname] for axname in obs._axes if axname != "samps"]
    res    = core.AxisManager(core.OffsetAxis("samps", onsamp), *axes)
    for key, axes in obs._assignments.items():
        # Stuff without sample axes
        if "samps" not in axes:
            res.wrap(*get_wrappable(obs, key))
        elif isinstance(obs[key], (core.AxisManager, core.FlagManager)):
            res.wrap(key, downsample_obs(obs["key"], down, skip_signal, fft_resample))
        elif isinstance(obs[key], so3g.proj.ranges.RangesMatrix):
            res.wrap(key, downsample_cut(obs[key], down))
        elif key in "fft_resample":
            if key == "signal" and skip_signal:
                continue
            # Make the axis that is samps the last one
            ax_idx = np.where(np.array(axes) == "samps")[0]
            dat = np.moveaxis(obs[key], ax_idx, -1)
            # Resample and return to original order
            dat = np.moveaxis(resample.resample_fft_simple(dat, onsamp), -1, ax_idx)
            res.wrap(key, dat, [(i, ax) for i, ax in enumerate(axes)])
        # Some naive slicing for everything else
        else:
            # Make the axis that is samps the last one
            ax_idx = np.where(np.array(axes) == "samps")[0]
            dat = np.moveaxis(obs[key], ax_idx, -1)
            # Resample and return to original order
            dat = np.moveaxis(dat[..., ::down][..., :onsamp], -1, ax_idx)
            res.wrap(key, dat, [(i, ax) for i, ax in enumerate(axes)])

    # Not sure how to deal with flags. Some sort of or-binning operation? But it
    # doesn't matter anyway
    return res

def get_flags(obs, flagnames):
    """Parse detector-set splits"""
    cuts_out = None
    if flagnames is None:
        return so3g.proj.RangesMatrix.zeros(obs.shape)
    det_splits = ['det_left','det_right','det_in','det_out','det_upper','det_lower']
    for flagname in flagnames:
        if flagname in det_splits:
            cuts = obs.det_flags[flagname]
        elif flagname == 'scan_left':
            cuts = obs.flags.left_scan
        elif flagname == 'scan_right':
            cuts = obs.flags.right_scan
        else:
            cuts = getattr(obs.flags, flagname) # obs.flags.flagname

        ## Add to the output matrix
        if cuts_out is None:
            cuts_out = cuts
        else:
            cuts_out += cuts
    return cuts_out

def import_optional(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except:
        return None

def setup_passes(downsample="1", maxiter="500", interpol="nearest", npass=None):
    """Set up information multipass mapmaking. Supports arguments
    of the form num or num,num,.... Infers the number of passes
    from the maximum length of these (or npass if explicitly given),
    and makes sure they all have this length by duplicating the item
    as necessary.

    Example:
     setup_passes(downsample="4,4,1", maxiter="300,300,50", interpol="linear")
     returns bunch.Bunch(downsample=[4,4,1], maxiter=[300,300,50],
      interpol=["linear","linear","linear"])"""
    tmp            = bunch.Bunch()
    tmp.downsample = putils.parse_ints(downsample)
    tmp.maxiter    = putils.parse_ints(maxiter)
    tmp.interpol   = interpol.split(",")
    # The entries may have different lengths. We use the max
    # and then pad the others by repeating the last element.
    # The final output will be a list of bunches
    if npass is None:
        npass = max([len(tmp[key]) for key in tmp])
    passes    = []
    for i in range(npass):
        entry = bunch.Bunch()
        for key in tmp:
            entry[key] = tmp[key][min(i,len(tmp[key])-1)]
        passes.append(entry)
    return passes

Base = declarative_base()

class AtomicInfo(Base):
    __tablename__ = "atomic"

    obs_id: Mapped[str] = mapped_column(primary_key=True)
    telescope: Mapped[str] = mapped_column(primary_key=True)
    freq_channel: Mapped[str] = mapped_column(primary_key=True)
    wafer: Mapped[str] = mapped_column(primary_key=True)
    ctime: Mapped[int] = mapped_column(primary_key=True)
    split_label: Mapped[str] = mapped_column(primary_key=True)
    valid: Mapped[Optional[bool]]
    split_detail: Mapped[Optional[str]]
    prefix_path: Mapped[Optional[str]]
    elevation: Mapped[Optional[float]]
    azimuth: Mapped[Optional[float]]
    pwv: Mapped[Optional[float]]
    dpwv: Mapped[Optional[float]]
    total_weight_qu: Mapped[Optional[float]]
    mean_weight_qu: Mapped[Optional[float]]
    median_weight_qu: Mapped[Optional[float]]
    leakage_avg: Mapped[Optional[float]]
    noise_avg: Mapped[Optional[float]]
    ampl_2f_avg: Mapped[Optional[float]]
    gain_avg: Mapped[Optional[float]]
    tau_avg: Mapped[Optional[float]]
    f_hwp: Mapped[Optional[float]]
    roll_angle: Mapped[Optional[float]]
    scan_speed: Mapped[Optional[float]]
    scan_acc: Mapped[Optional[float]]
    sun_distance: Mapped[Optional[float]]
    ambient_temperature: Mapped[Optional[float]]
    uv: Mapped[Optional[float]]
    ra_center: Mapped[Optional[float]]
    dec_center: Mapped[Optional[float]]

    def __init__(self, obs_id, telescope, freq_channel, wafer, ctime, split_label):
        self.obs_id = obs_id
        self.telescope = telescope
        self.freq_channel = freq_channel
        self.wafer = wafer
        self.ctime = ctime
        self.split_label = split_label

    def __repr__(self):
        return f"({self.obs_id},{self.telescope},{self.freq_channel},{self.wafer},{self.ctime},{self.split_label})"

def atomic_db_aux(atomic_db, info, valid = True):
    info.valid = valid
    engine = create_engine("sqlite:///%s" % atomic_db, echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(info)
        try:
            session.commit()
        except exc.IntegrityError:
            session.rollback()

# this class intends to produce a obslists dictionary that will be feed to the mapmaker. Every element of obslists will be mapped.
import numpy as np
from pixell import utils
from scipy import ndimage
import so3g.proj

from .utilities import *

def find_scan_periods(obs_info, ttol=60, atol=2*utils.degree, mindur=120):
    """Given a scan db, return the set of contiguous scanning periods in the form
    [:,{ctime_from,ctime_to}]."""
    atol = atol/utils.degree
    info = np.array([obs_info[a] for a in ["az_center", "el_center", "az_throw", "timestamp", "duration"]]).T
    # Get rid of nan entries
    bad  = np.any(~np.isfinite(info),1)
    # get rid of too short tods, since those don't have reliable az bounds
    bad |= info[:,-1] < mindur
    info = info[~bad]
    t1   = info[:,-2] # timestamp is start of tod, unlike in enki
    info = info[np.argsort(t1)]
    # Start, end
    t1   = info[:,-2]
    t2   = t1 + info[:,-1]
    # Remove angle ambiguities
    info[:,0] = utils.rewind(info[:,0], period=360)
    # How to find jumps:
    # 1. It's a jump if the scanning changes
    # 2. It's also a jump if a the interval between tod-ends and tod-starts becomes too big
    changes    = np.abs(info[1:,:3]-info[:-1,:3])
    jumps      = np.any(changes > atol,1)
    jumps      = np.concatenate([[0], jumps]) # from diff-inds to normal inds
    # Time in the middle of each gap
    gap_times = np.mean(find_period_gaps(np.array([t1,t2]).T, ttol=ttol),1)
    gap_inds  = np.searchsorted(t1, gap_times)
    jumps[gap_inds] = True
    # raw:  aaaabbbbcccc
    # diff: 00010001000
    # 0pre: 000010001000
    # cum:  000011112222
    labels  = np.cumsum(jumps)
    linds   = np.arange(np.max(labels)+1)
    t1s     = ndimage.minimum(t1, labels, linds)
    t2s     = ndimage.maximum(t2, labels, linds)
    # Periods is [nperiod,{start,end}] in ctime. Start is the start of the first tod
    # in the scanning period. End is the end of the last tod in the scanning period.
    periods = np.array([t1s, t2s]).T
    return periods

def find_scan_periods_perobs(obs_info, mindur=120):
    """Given a scan db, return the periods per obs, i.e. start and stop of each scan. This is a simplified version of find_scan_periods
    [:,{ctime_from,ctime_to}]."""
    info = np.array([obs_info[a] for a in ["timestamp", "duration"]]).T
    # Get rid of nan entries
    bad  = np.any(~np.isfinite(info),1)
    # get rid of too short tods, since those don't have reliable az bounds
    bad |= info[:,-1] < mindur
    info = info[~bad]
    t1   = info[:,-2] # timestamp is start of tod, unlike in enki
    info = info[np.argsort(t1)]
    # Start, end
    t1   = info[:,-2]
    t2   = t1 + info[:,-1]
    periods = np.array([t1, t2]).T
    return periods

def find_period_gaps(periods, ttol=60):
    """Helper for find_scan_periods. Given the [:,{ctime_from,ctime_to}] for all
    the individual scans, returns the times at which the gap between the end of
    a tod and the start of the next is greater than ttol (default 60 seconds)."""
    # We want to sort these and look for any places
    # where a to is followed by a from too far away. To to this we need to keep
    # track of which entries in the combined, sorted array was a from or a to
    periods = np.asarray(periods)
    types   = np.zeros(periods.shape, int)
    types[:,1] = 1
    types   = types.reshape(-1)
    ts      = periods.reshape(-1)
    order   = np.argsort(ts)
    ts, types = ts[order], types[order]
    # Now look for jumps
    jumps = np.where((ts[1:]-ts[:-1] > ttol) & (types[1:]-types[:-1] < 0))[0]
    # We will return the time corresponding to each gap
    gap_times = np.array([ts[jumps], ts[jumps+1]]).T
    return gap_times

def split_periods(periods, maxdur):
    # How long is each period
    durs   = periods[:,1]-periods[:,0]
    # How many parts to split each into
    nsplit = utils.ceil(durs/maxdur)
    nout   = np.sum(nsplit)
    # For each output period, find which input period it
    # corresponds to
    group  = np.repeat(np.arange(len(durs)), nsplit)
    sub    = np.arange(nout)-np.repeat(utils.cumsum(nsplit),nsplit)
    t1     = periods[group,0] + sub*maxdur
    t2     = np.minimum(periods[group,0]+(sub+1)*maxdur, periods[group,1])
    return np.array([t1,t2]).T

def build_period_obslists(obs_info, periods, context, nset=None, wafer=None, freq=None):
    """For each period for each detset-band, make a list of (id,detset,band)
    for the ids that fall inside that period. Returns a dictionary
    that maps (pid,deset,band) to those lists. pid is here the index into
    the periods, where periods is [nperiod,{ctime_from,ctime_to}]."""
    obslists = {}
    # 1. Figure out which period each obs belongs to
    ctimes_mid = obs_info.timestamp + obs_info.duration/2
    pids       = np.searchsorted(periods[:,0], ctimes_mid)-1
    # 2. Build our lists. Not sure how to do this without looping
    for i, row in enumerate(obs_info):
        if wafer is not None:
            wafer_list = [wafer]
        else:
            meta = context.get_meta(row.obs_id)
            wafer_list = np.unique(meta.det_info.wafer_slot)
        if freq is not None:
            band_list = [freq]
        else:
            band_list = ['f090','f150']
        for detset in wafer_list[:nset]:
            for band in band_list:
                key = (pids[i], detset, band)
                if key not in obslists: obslists[key] = []
                obslists[key].append((row.obs_id, detset, band, i))
    return obslists

def build_obslists(context, query, mode=None, nset=None, wafer=None, freq=None, ntod=None, tods=None, fixed_time=None, mindur=None, ):
    """ 
    Return an obslists dictionary (described in build_period_obslists), along with all ancillary data necessary for the mapmaker
    
    Parameters
    __________
    context : sotodlib.core.Context
            A context object
    query : str
            A list of tods. Can be a file with one obs_id per line, or a query that context.obsdb.query() will understand
    mode : str or none
            Optional, mode for selecting tods. Can be 'per_obs', 'fixed_interval', 'depth_1'. Default is 'per_obs'
    nset : int or None 
            Optional, the first nset sets will be mapped
    ntod : int or None
            Optional, the first ntod observations listed in query will be mapped
    tods : str or None
            Optional, a string with specific tods that will be mapped
    fixed_time : int or None
            Optional, if mode=='fixed_interval', this is the fixed time in seconds
    mindur : int or None
            Optional, minimum duration of an observation to be included in the mapping. If not defined it will be 120 seconds
           
    Output
    ______
    
    """
    
    # Get the full list of tods we will work with
    ids = get_ids(query, context=context)
    #print(ids)
    # restrict tod selection further. E.g. --tods [0], --tods[:1], --tods[::100], --tods[[0,1,5,10]], etc.
    if tods: ids = np.array(eval("ids"+tods)).reshape(-1)
    if ntod: ids = ids[:ntod]
    if len(ids) == 0:
        print("No tods found!")
        sys.exit(1)
    # Extract info about the selected ids
    obs_infos = context.obsdb.query("obs_id in (%s)" % ",".join(["'%s'" % id for id in ids]))
    obs_infos = obs_infos.asarray().view(np.recarray)
    ids       = obs_infos.obs_id # can't rely on ordering, so reget
    
    if mode is None or mode == 'per_obs':
        # We simply need to make and obslists dict with each key being one obs
        # In the future we will add sub-time and sub-focal plane splits, that would go here
        periods = find_scan_periods_perobs(obs_infos)
    elif mode=='depth_1':
        # In the future we will add sub-time and sub-focal plane splits, that would go here
        periods   = find_scan_periods(obs_infos, ttol=12*3600)
        periods   = split_periods(periods, 24*3600)
    elif mode=='fixed_interval':
        # In the future we will add sub-time and sub-focal-plane splits, that would go here
        if fixed_time is not None:
            periods   = find_scan_periods(obs_infos, ttol=12*3600)
            periods   = split_periods(periods, fixed_time)
        else:
            periods   = find_scan_periods(obs_infos, ttol=12*3600)
            periods   = split_periods(periods, 24*3600) # If fixed_time was not set, then we do 24 hrs by default and it will be the same as depth_1
    else:
        print("Invalid mode!")
        sys.exit(1)
    
    # We will make one map per period-detset-band
    obslists = build_period_obslists(obs_infos, periods, context, nset=nset, wafer=wafer, freq=freq)
    obskeys  = sorted(obslists.keys())
    return obslists, obskeys, periods, obs_infos

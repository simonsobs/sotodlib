"""
Functions for simulating glitches, jumps and other 
bad features in the data
"""
import numpy as np
from so3g.proj import Ranges, RangesMatrix


def add_random_glitches(tod, params={}, signal='glitches', flag='true_glitches',
                        overwrite=False, verbose=False):
    """Add glitches (spikes that just effect one data point) to the data.
    
    Args:
        tod (AxisManager): the tod 
        params (dictionary): Use to overwrite the default values
                n_glitch: the expected number of glitches per detector per observation
                sig_n_glitch: the width of the expected distribution
                h_glitch: the expected higher of the glitches
                sig_h_glitch: the expected higher of the glitches
        signal (string): name of the place to add the badness to the tod
        flag (string): name of the flag to store where the glitches are
        overwrite (bool): if true, write over signal. if false, add to signal
        verbose (bool): print the number of glitches added
    """
    gparams = {'n_glitch' : 1, 'sig_n_glitch' : 0.1, 
               'h_glitch' : 10, 'sig_h_glitch' : 0.01}
    gparams.update(params)
    params=gparams
    
    n_tot = int(np.abs(params['sig_n_glitch']*tod.dets.count*np.random.randn(1) 
                       + params['n_glitch']*tod.dets.count))
    places = np.random.randint( tod.dets.count*tod.samps.count, size=(n_tot,))
    heights = np.random.randn( n_tot)*params['sig_h_glitch'] + params['h_glitch']
    glitches = np.zeros( (tod.dets.count*tod.samps.count,) )
    glitches[places] = heights
    
    glitches = np.reshape( glitches, (tod.dets.count, tod.samps.count) )
    truth = RangesMatrix( [Ranges.from_mask( g!=0) for g in glitches])
    
    if verbose:
        print('Adding {} glitches to {} detectors'.format(n_tot, 
                                                          np.sum( [len(t.ranges())>0 for t in truth]),))
                                                                      
    if not signal in tod:
        tod.wrap(signal, glitches,
                 [(0,tod.dets), (1,(tod.samps))] )
    else:
        if overwrite:
            tod[signal] = glitches
        else:
            tod[signal] += glitches
            
    if not flag in tod.flags:
        tod.flags.wrap(flag, truth)
    else:
        if overwrite:
            tod.flags[flag] = truth
        else:
            tod.flags[flag] += truth    

def add_random_jumps(tod, params={}, signal='jumps', flag='true_jumps',
                        overwrite=False, verbose=False):
    """Add jumps (changes in DC level) to the data.
    
    Args:
        tod (AxisManager): the tod 
        params (dictionary): Use to overwrite the default values
                n_jump is the expected number of jumps per detector per observation
               sig_n_jump is the width of the expected distribution
               h_jump is the expected higher of the jumps
               sig_h_jump is the expected higher of the jumps
        signal (string): name of the place to add the badness to the tod
        flag (string): name of the flag to store where the glitches are
        overwrite (bool): if true, write over signal. if false, add to signal
        verbose (bool): print the number of glitches added
    """
  
    gparams = {'n_jump' : 0.2, 'sig_n_jump' : 0.05, 
               'h_jump' : 0.1, 'sig_h_jump' : 0.3}
    gparams.update(params)
    params=gparams
    
    n_tot = int(np.abs(params['sig_n_jump']*tod.dets.count*np.random.randn(1) 
                       + params['n_jump']*tod.dets.count))
    places = np.random.randint( tod.dets.count*tod.samps.count, size=(n_tot,))
    heights = np.random.randn( n_tot)*params['sig_h_jump'] + params['h_jump']
    jumps = np.zeros( (tod.dets.count*tod.samps.count,) )
    jumps[places] = heights
    
    jumps = np.reshape( jumps, (tod.dets.count, tod.samps.count) )
    truth = RangesMatrix( [Ranges.from_mask( g!=0) for g in jumps])
    ## actually make these jumps and not glitches
    jumps = np.cumsum(jumps, axis=1)    
    
    if verbose:
        print('Adding {} jumps to {} detectors'.format(n_tot, 
                                                          np.sum( [len(t.ranges())>0 for t in truth]),))
                                                                      
    if not signal in tod:
        tod.wrap(signal, jumps,
                 [(0,tod.dets), (1,(tod.samps))] )
    else:
        if overwrite:
            tod[signal] = jumps
        else:
            tod[signal] += jumps
            
    if not flag in tod.flags:
        tod.flags.wrap(flag, truth)
    else:
        if overwrite:
            tod.flags[flag] = truth
        else:
            tod.flags[flag] += truth

def add_random_offsets( tod, params={}, signal='offset',
                        overwrite=False, verbose=False):
    """Add uniformly distributed DC offsets to the data
    
    Args:
        tod (AxisManager): the tod 
        params (dictionary): Use to overwrite the default values
                offset_low: lowest possible offset
                offset_high: highest possible offset
        signal (string): name of the place to add the badness to the tod
        overwrite (bool): if true, write over signal. if false, add to signal
        verbose (bool): print that offsets are added
    """
    
    gparams = {'offset_low' : -20, 
               'offset_high':  20 }
    gparams.update(params)
    params=gparams
    
    offsets = params['offset_low'] + np.random.rand(tod.dets.count)*(params['offset_high']
                                                                     -params['offset_low'])
    offs = np.zeros( (tod.dets.count, tod.samps.count))
    offs += offsets[:,None]
    
    if verbose:
        print('Adding Offsets to detectors')
                                                                      
    if not signal in tod:
        tod.wrap(signal, offs,
                 [(0,tod.dets), (1,(tod.samps))] )
    else:
        if overwrite:
            tod[signal] = offs
        else:
            tod[signal] += offs

def add_random_trends( tod, params={}, signal='trends',
                        overwrite=False, verbose=False):
    """Add normally distributed trends offsets to the data
    
    Args:
        tod (AxisManager): the tod 
        params (dictionary): Use to overwrite the default values
                slope_mean: normal distribution mean
                slope_sig: normal distribution offset
        signal (string): name of the place to add the badness to the tod
        overwrite (bool): if true, write over signal. if false, add to signal
        verbose (bool): print that offsets are added
    """
    gparams = {'slope_mean' : 0.1, 
               'slope_sig':  0.01 }
    gparams.update(params)
    params=gparams
    
    s = np.random.randn(tod.dets.count)*params['slope_sig']+params['slope_mean']
    
    slopes = np.zeros( (tod.dets.count, tod.samps.count))
    slopes += s[:,None]*np.linspace(0,1,tod.samps.count)[None,:]
    
    if verbose:
        print('Adding Trends to Detectors')
                                                                      
    if not signal in tod:
        tod.wrap(signal, slopes,
                 [(0,tod.dets), (1,(tod.samps))] )
    else:
        if overwrite:
            tod[signal] = slopes
        else:
            tod[signal] +=slopes

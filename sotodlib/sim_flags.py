"""
Functions for simulating glitches, jumps and other 
bad features in the data
"""
import numpy as np
from so3g.proj import Ranges, RangesMatrix


def add_random_glitches(tod, params={}, signal='glitches', flag='true_glitches',
                        overwrite=False, verbose=False):
    """n_glitch is the expected number of glitches per detector per observation
       sig_n_glitch is the width of the expected distribution
       h_glitch is the expected higher of the glitches
       sig_h_glitch is the expected higher of the glitches
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
        
add_random_glitches(tod, verbose=True, overwrite=True)
    
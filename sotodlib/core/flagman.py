import numpy as np
from functools import reduce

from so3g.proj import Ranges, RangesMatrix
from . import AxisManager

class FlagManager(AxisManager):
    """An extension of the AxisManager class to make functions 
    more specifically associated with cuts and flags.
    
    FlagManager only expects to have items that are mapped to the 
    detector axis, the sample axis, or both. 
    
    Detector Flags can be passed as bitmasks or boolean arrays. To match with
    Ranges and RangesMatrix, the default is False and the exceptions
    are True
    """
    
    def __init__(self, det_axis, samp_axis):
        super().__init__(det_axis, samp_axis)
                
        if not 'dets' in self._axes:
            raise ValueError('FlagManagers require a dets axis')
        if not 'samps' in self._axes:
            raise ValueError('FlagManagers require a samps axis')
            
    def wrap(self, name, data, axis_map=None, *kwargs):
        """If axis_map is None, the data better be (dets,), (samps,),
            or (dets, samps)
        """
        
        if axis_map is None:
            try:
                s = data.shape
            except:
                ### catches if a detector mask is just a list
                s = np.shape(data)
            if len(s) == 1:
                if s[0] == self.dets.count:
                    ## detector only flag. Turn into RangesMatrix
                    axis_map=[(0,'dets')]
                elif s[0] == self.samps.count:
                    axis_map=[(0,'samps')]
                else:
                    raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
            elif len(s)==2:
                if s[0] == self.dets.count and s[1] == self.samps.count:
                    axis_map=[(0,'dets'), (1,'samps')]
                elif s[1] == self.dets.count and s[0] == self.samps.count:
                    axis_map=[(0,'samps'),(1,'dets')]
                else:
                    raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
            else:
                raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
        
        if len(axis_map)==1 and axis_map[0][1]=='dets':
            ### Change detector flags to RangesMatrix in the backend
            x = Ranges(self.samps.count)
            data = RangesMatrix([Ranges.ones_like(x) if Y 
                                 else Ranges.zeros_like(x) for Y in data])
            axis_map = [(0,'dets'),(1,'samps')]
        super().wrap(name, data, axis_map, *kwargs)
                    
    def buffer(self, n_buffer, flags=None):
        """Buffer all the samps cuts by n_buffer
        Like with Ranges / Ranges Matrix, buffer changes everything in place        
        
        Arguments:
        
            n_buffer: number of samples to buffer the samps cuts
            flags: List of flags to buffer. Uses their names
        """
        if flags is None:
            flags = self._fields
        
        for f in flags:
            self[f].buffer(n_buffer)
        
    def buffered(self, n_buffer, flags=None):
        """Return new FlagManager that has all the samps cuts buffered by n_buffer
        Like with Ranges / Ranges Matrix, buffered returns new object
        
        Arguments:
        
            n_buffer: number of samples to buffer the samps cuts
            flags: List of flags to buffer. Uses their names
        """
        new = self.copy()
        new.buffer(n_buffer, flags)
        return new
        
    def collapse(self, flags=None, method='union'):
        """Collapse flags in the FlagManager together. If flags is None
        it returns all flags
        
        Arguments:
        
            flags: List of flags to collapse together. Uses their names.
            method: How to collapse the data. Accepts 'union','intersect',
                        or function.
        """
        if flags is None:
            flags = self._fields

        to_reduce = [self._fields[f] for f in flags]
        if len(flags)==0:
            raise ValueError('Found zero flags to combine')
            
        out = RangesMatrix([Ranges(self.samps.count) for det in self.dets.vals])
        
        ## need to add out to prevent flag ordering from causing errors
        ### (Ranges can't add to RangeMatrix, only other way around)
        to_reduce[0] = out+to_reduce[0]
        
        if method == 'union': 
            op = lambda x, y: x+y
        elif method == 'intersect':
            op = lambda x, y: x*y
        else:
            op = method
        out = reduce(op, to_reduce)
        return out
    
    def reduce(self, new_flag, flags, method='union', keep=False):
        """reduce multiple flags into a new flag by chosen method.
        
        Arguments:
            
            new_flag: Name of new flag
            flags: list of flags to combine
            method: see collapse
            keep: if False, removes reduced flags from Manager
        """
        new = self.collapse(flags, method)
        if keep and (new_flag in flags):
            raise ValueError("keep == True would result in duplicate names.")
        
        # drop the fields if needed
        if not keep: 
            for f in flags: 
                self.move(f, None)
        
        self.wrap(new_flag, new)
        
    def reduce_all(self, new_flag, method='union', keep=False):
        """same as reduce but for all fields, to be more verbose"""
        return self.reduce(new_flag, list(self._fields.keys()), method, keep)
        
    @classmethod
    def for_tod(cls, tod):
        """Assumes tod is an AxisManager with dets and samps axes defined
        """
        return cls(tod.dets, tod.samps)

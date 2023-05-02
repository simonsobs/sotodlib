import numpy as np
from functools import reduce
from collections import OrderedDict as odict

from so3g.proj import Ranges, RangesMatrix
from . import AxisManager

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array

class FlagManager(AxisManager):
    """An extension of the AxisManager class to make functions 
    more specifically associated with cuts and flags.
    
    FlagManagers must have a dets axis and a samps axis when created.
    
    FlagManager only expects to have individual flags that are mapped to the 
    detector axis, the sample axis, or both. 
    
    Detector Flags can be passed as bitmasks or boolean arrays. To match with
    Ranges and RangesMatrix, the default is False and the exceptions
    are True
    """
    
    def __init__(self, dets_axis, samps_axis):
        self._dets_name = dets_axis.name
        self._samps_name = samps_axis.name
        
        super().__init__(dets_axis, samps_axis)
                
    
    def wrap(self, name, data, axis_map=None, **kwargs):
        """See core.AxisManager for basic usage
        
        If axis_map is None, the data better be (dets,), (samps,),
            or (dets, samps). Will not work if dets.count == samps.count
            
        """
        
        if axis_map is None:
            if self[self._dets_name].count == self[self._samps_name].count:
                raise ValueError("Cannot auto-detect axis_map when dets and "
                                 "samps axes have equal lengths. axis_map "
                                 "must be defined")
            s = _get_shape(data)
            
            if len(s) == 1:
                if s[0] == self[self._dets_name].count:
                    ## detector only flag. Turn into RangesMatrix
                    axis_map=[(0,self._dets_name)]
                elif s[0] == self[self._samps_name].count:
                    axis_map=[(0, self.samps)]
                else:
                    raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
            elif len(s) == 2:
                if s[0] == self[self._dets_name].count and s[1] == self[self._samps_name].count:
                    axis_map=[(0,self._dets_name), (1,self._samps_name)]
                elif s[1] == self[self._dets_name].count and s[0] == self[self._samps_name].count:
                    raise ValueError("FlagManager only takes 2D data aligned as"
                                     " (dets, samps). Data of shape {}"
                                     " is the wrong shape".format(s))
                else:
                    raise ValueError("FlagManager only takes 2D data aligned as"
                                     " (dets, samps). Data of shape {}"
                                     " is the wrong shape".format(s))
            else:
                raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
        
        if len(axis_map)==1 and axis_map[0][1]==self._dets_name:
            ### Change detector flags to RangesMatrix in the backend
            x = Ranges(self.samps.count)
            data = RangesMatrix([Ranges.ones_like(x) if Y 
                                 else Ranges.zeros_like(x) for Y in data])
            axis_map = [(0,self._dets_name),(1,self._samps_name)]

        super().wrap(name, data, axis_map, **kwargs)

    def wrap_dets(self, name, data):
        """Adding flag with just (dets,) axis.
        """
        s = _get_shape(data)
        if not len(s) == 1 or s[0] != self[self._dets_name].count:
            raise ValueError("Data of shape {} is cannot be aligned with"
                             "the detector axis".format(s))
        self.wrap(name, data, axis_map=[(0,self._dets_name)])
        
    def wrap_samps(self, name, data):
        """Adding flag with just (samps,) axis.
        """
        s = _get_shape(data)
        if not len(s) == 1 or s[0] != self[self._samps_name].count:
            raise ValueError("Data of shape {} is cannot be aligned with"
                             "the samps axis".format(s))
        self.wrap(name, data, axis_map=[(0,self._samps_name)])
        
    def wrap_dets_samps(self, name, data):
        """Adding flag with (dets, samps) axes.
        """
        s = _get_shape(data)
        if (not len(s) == 2 or s[0] != self[self._dets_name].count or
               s[1] != self[self._samps_name].count):
            raise ValueError("Data of shape {} is cannot be aligned with"
                             "the (dets,samps) axss".format(s))
        self.wrap(name, data, axis_map=[(0,self._dets_name), (1,self._samps_name)])

        
    def copy(self, axes_only=False):
        out = FlagManager(self[self._dets_name], self[self._samps_name])
        for k, v in self._axes.items():
            out._axes[k] = v
        if axes_only:
            return out
        for k, v in self._fields.items():
            out._fields[k] = v.copy()
        for k, v in self._assignments.items():
            out._assignments[k] = v.copy()
        return out
    
    def get_zeros(self, wrap=None):
        """
        Return a correctly sized RangesMatrix for building cuts for the FlagManager
        
        Args:
            wrap: if not None, it is a string with which to add to the FlagManager
        """
        out = RangesMatrix([Ranges(self[self._samps_name].count) for det in self[self._dets_name].vals])
        if not wrap is None:
            self.wrap_dets_samps( wrap, out)
            return self[wrap]
        return out
        
    def buffer(self, n_buffer, flags=None):
        """Buffer all the samps cuts by n_buffer
        Like with Ranges / Ranges Matrix, buffer changes everything in place        
        
        Args:
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
        
        Args:
            n_buffer: number of samples to buffer the samps cuts
            flags: List of flags to buffer. Uses their names
        
        Returns:
            new: FlagManager with all flags buffered
        """
        new = self.copy()
        new.buffer(n_buffer, flags)
        return new
        
    def reduce(self, flags=None, method='union', wrap=False, new_flag=None,
               remove_reduced=False):
        """Reduce (combine) flags in the FlagManager together. 
        
        Args:
            flags: List of flags to collapse together. Uses their names.
                   If flags is None then all flags are reduced
            method: How to collapse the data. Accepts 'union','intersect',
                        or function.
            wrap: if True, add reduced flag to self
            new_flag: name of new flag, required if wrap is True
            remove_reduced: if True, remove all reduced flags from self
        
        Returns:
            out: reduced flag
        """
        if flags is None:
            ## copy needed to no break things if removing flags
            flags = self._fields.copy()

        to_reduce = [self._fields[f] for f in flags]
        if len(flags)==0:
            raise ValueError('Found zero flags to combine')
            
        out = self.get_zeros()
        
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
        
        # drop the fields if needed
        if remove_reduced: 
            for f in flags: 
                self.move(f, None)
                
        if wrap:
            if new_flag is None:
                raise ValueError("new_flag cannot be None if wrap is True")
            self.wrap(new_flag, out)
            
        return out        
    
    def has_cuts(self, flags=None):
        """Return list of detector ids that have cuts
        
        Args: 
            flags: [optional] If not none it is the list of flags to combine to see
                    if cuts exist
        """
        c = self.reduce(flags=flags)
        idx = [len(x.ranges())>0 for x in c]
        return self[self._dets_name].vals[idx]

    @classmethod
    def for_tod(cls, tod, dets_name='dets', samps_name='samps'):
        """Create a Flag manager for an AxisManager tod which has axes for detectors
        and samples.
        
        Args:
            tod: AxisManager for the specific data
            dets_name: name of the axis that should be treated as detectors
            samps_name: name of the axis that should be treated as samples
        """
        
        return cls(tod[dets_name], tod[samps_name])

    @classmethod
    def promote(cls, src, dets_name, samps_name):
        """Create an instance of this class from an AxisManager.  The axes in
        src with names dets_name and samps_name are used for the dets
        and samps axes.

        This is a move-style constructor, and empties out src to
        minimize copies.

        """
        flagman = cls(src._axes.pop(dets_name),
                      src._axes.pop(samps_name))
        flagman._axes.update(src._axes)
        flagman._fields = dict(src._fields)
        flagman._assignments = dict(src._assignments)

        # Reset source.
        src._axes = odict()
        src._assignments = {}
        src._fields = odict()

        return flagman

def _get_shape(data):
    try:
        return data.shape
    except:
        ### catches if a detector mask is just a list
        return np.shape(data)

def has_any_cuts(flag):
    return np.array([len(x.ranges())>0 for x in flag], dtype='bool')
def has_all_cut(flag):
    return np.array(
        [len(x.complement().ranges())==0 for x in flag],
        dtype='bool'
    )
def count_cuts(flag):
    return np.array([len(x.ranges()) for x in flag], dtype='int')

def sparse_to_ranges_matrix(arr, buffer=0, close_gaps=0, val=True):
    """Convert a csr sparse array into a ranges matrix
    
    Arguments:
    
    arr: sparse csr boolean array
    buffer: integer samples to buffer around Ranges
    close_gaps: any integer sample gaps to close in the Ranges
    val: what value in the boolean array indicates a flag
    """
    x = RangesMatrix.zeros(arr.shape)
    for i in range(arr.shape[0]):
        slc = arr.indices[arr.indptr[i]:arr.indptr[i+1]]
        if not np.all(arr.data[arr.indptr[i]:arr.indptr[i+1]]==val):
            raise
        for s in slc:
            x[i].append_interval_no_check(int(s),int(s+1))
        x[i].buffer(buffer)
        x[i].close_gaps(close_gaps)
    return x


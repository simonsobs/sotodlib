# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Simons Observatory Data Processing.

This module contains the core data I/O and processing tools.

"""

from so3g.spt3g import core


class DataG3Module(object):
    """
    Base G3Module for processing or conditioning data.
    
    Classes inheriting from DataG3Module only need to override the process() function
    to build a G3Module that will call process() on every detector in a given 
    G3TimestreamMap. 
    
    Attributes:
        input (str): key of G3TimestreamMap of source data
        output (str or None): key of G3Timestream map of output data.
            if None, input will be overwritten with output
            
    TODO:
        * Add detector masking capabilities so we don't waste CPU time on cut detectors
        * Make an option for input/output to be a list, so processing can be done on multiple timestreams
        * Make it possible for input/output to be a G3Timestream, instead of a G3TimestreamMap
        * Decide if "Modules that add members to frames" should be a default type
    """
    def __init__(self, input='signal', output='signal_out'):
        self.input = input
        if output is None:
            self.output = self.input
        else:
            self.output = output
                
    def __call__(self, f):
        if f.type == core.G3FrameType.Scan:
            if self.input not in f.keys() or type(f[self.input]) != core.G3TimestreamMap:
                raise ValueError("""Frame is a Scan but does not have a G3Timestream map 
                                 named {}""".format(self.input))
        
            processing = core.G3TimestreamMap()
            
            for k in f[self.input].keys():
                processing[k] = core.G3Timestream( self.process(f[self.input][k], k) )
                                    
            processing.start = f[self.input].start
            processing.stop = f[self.input].stop   

            if self.input == self.output:
                f.pop(self.input)
            f[self.output] = processing
           
    def process(self, data, det_name):
        """
        Args:
            data (G3Timestream): data for a single detector
            det_name (str): the detector name in the focal plane,
                in case it's needed for accessing calibration info
        Returns:
            data (G3Timestream)
        """
        return data
    
    def apply(self, f):
        """
        Applies the G3Module on an individual G3Frame or G3TimestreamMap. Likely
        to be useful when debugging
        
        Args:
            f (G3Frame or G3TimestreamMap)
            
        Returns:
            G3Frame: if f is a G3Frame it returns the frame after the module has been applied
            G3TimestreamMap: if f is a G3TimestreamMap it returns self.output
        """
        if type(f) == core.G3Frame:
            if f.type != core.G3FrameType.Scan:
                raise ValueError("""Cannot apply {} on a {} frame""".format(self.__class__,
                                                                           f.type))
            self.__call__(f)
            return f
        if type(f) == core.G3TimestreamMap:
            frame = core.G3Frame()
            frame.type = core.G3FrameType.Scan
            frame[self.input] = f
            self.__call__(frame)
            return frame[self.output]
        raise ValueError('apply requires a G3Frame or a G3TimestreamMap, you gave me {}'.format(type(f)))
        
    @classmethod
    def from_function(cls, function, input='signal', output=None):
        """
        Allows inline creation of DataG3Modules with just a function definition
        
        Example:
            mean_sub = lambda data: data-np.mean(data)
            G3Pipeline.Add(DataG3Module.from_function(mean_sub) )

        Args:
            function (function): function that takes a G3Timestream (or ndarray)
                and returns an ndarray
        """
        dg3m = cls(input, output)
        process = lambda self, data, det_name: function(data)
        dg3m.process = process.__get__(dg3m, DataG3Module)
        return dg3m

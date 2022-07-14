import os
import numpy as np
import scipy.interpolate
import so3g
from spt3g import core
import argparse
import h5py

class analyze_wiregrid(): 
    
    def __init__(self, debug=False):

        """
        Class to analyze wiregrid calibration data

        Args
        -----
            debug: bool
                bit for dubug mode
        """

        
        self._debug = debug
        self._start = 0
        self._end = 0

    def load_data(self, start, end, archive_path, instance='HBA1'):

        """
        Loads house keeping data for a given time range. 
        For the specified time range, this function returns HWP parameters in HK g3 file

        Args
        -----
            start : timestamp or DateTime (timezone: UTC)
                start time for data
            end :  timestamp  or DateTime (timezone: UTC)
                end time for data
            archive_path : str
                path to HK g3 file
            instance : str          
                default = HBA1
        """
        self._start = start
        self._end = end
        hwp_keys=[
            'observatory.' + instance + '.feeds.HWPEncoder.rising_edge_count',
            'observatory.' + instance + '.feeds.HWPEncoder.irig_time',
            'observatory.' + instance + '.feeds.HWPEncoder_full.counter',
            'observatory.' + instance + '.feeds.HWPEncoder_full.counter_index',
            'observatory.' + instance + '.feeds.HWPEncoder.irig_synch_pulse_clock_time',
            'observatory.' + instance + '.feeds.HWPEncoder.irig_synch_pulse_clock_counts',   
            'observatory.' + instance + '.feeds.HWPEncoder.quad',
        ]
        alias=[key.split('.')[-1] for key in hwp_keys]

        if isinstance(start,np.datetime64): start = start.timestamp()
        if isinstance(end,np.datetime64): end = end.timestamp()
        # load housekeeping data with hwp keys
        if self._debug: print('loading HK data files ...')
        data = so3g.hk.load_range(start, end, fields=hwp_keys, alias=alias, data_dir=archive_path)
        
        return data
        
    def load_file(self, filename, instance='HBA1'):
        
        """
        Loads data with specified g3 files. 
        Return data from SO smurf data and HK data.
        
        Args
        -----
            filename : str or [str] or np,array(str)
                HK g3 filename (str or array)
            instance : str          
                default = HBA1
        """
        hwp_keys=[
            'observatory.' + instance + '.feeds.HWPEncoder.rising_edge_count',
            'observatory.' + instance + '.feeds.HWPEncoder.irig_time',
            'observatory.' + instance + '.feeds.HWPEncoder_full.counter',
            'observatory.' + instance + '.feeds.HWPEncoder_full.counter_index',
            'observatory.' + instance + '.feeds.HWPEncoder.irig_synch_pulse_clock_time',
            'observatory.' + instance + '.feeds.HWPEncoder.irig_synch_pulse_clock_counts',   
            'observatory.' + instance + '.feeds.HWPEncoder.quad',
        ]
        alias=[key.split('.')[-1] for key in hwp_keys]
        
        # load housekeeping files with hwp keys
        if self._debug: print('loading HK data files ...')
        scanner = so3g.hk.HKArchiveScanner()
        if isinstance(filename, list) or isinstance(filename, np.ndarray):
            for f in filename: scanner.process_file(f)
        else: scanner.process_file(filename)
        arc = scanner.finalize()
        if not any(arc.get_fields()[0]): 
            print('INFO: No HK data in input g3 files: ' + filename)
            self._start = 0
            self._end = 0
            return {}

        self._start = arc.simple([key for key in arc.get_fields()[0].keys()][0])[0][0]
        self._end = arc.simple([key for key in arc.get_fields()[0].keys()][0])[0][-1]
        for i in range(len(hwp_keys)):
            if not hwp_keys[i] in arc.get_fields()[0].keys():
                print('INFO: HWP is not spinning in input g3 files: ' + filename)
                return {}
            
        data_raw = arc.simple(hwp_keys)
        
        data = {'rising_edge_count':data_raw[0], 'irig_time':data_raw[1], 'counter':data_raw[2], 'counter_index':data_raw[3], \
                'irig_synch_pulse_clock_time':data_raw[4], 'irig_synch_pulse_clock_counts':data_raw[5], 'quad':data_raw[6]}

        return data     
    
    def analyze(self, data):
        
        """
        Analyze wiregrid data
        Args
        -----
            data : dict
                data from load_data
                
        Returns
        --------
            dict{angle}

            angle (float): detector angle
        """

        return {'readout': readout, 'angle': angle}

    def write_result(self, result):
        
        """
        Output calibrated detector angle as in HDF5 format

        File format
        --------
            'readout_id' : 
            'det_angle' : 
            'det_angle_error' : 
 
            readout_id (int): Detector readout ID
            det_angle (float): Detector angle calibrated by wiregrid (including HWP angle)
            det_angle_error (float): Statistical error on det_angle
        """

        return
    
    def main(args=None):
        
        parser = argparse.ArgumentParser(description='Analyze wiregrid data from level-1 smurf data and HK data.')

        parser.add_argument('-f','--file', action='store', required=True, help='A filename or list of filenames (to be loaded in order).', nargs='*')
        parser.add_argument('--output', default='./output.hdf5', help='A path to output HDF5 file')

        args = parser.parse_args()
        print('files=' + str(args.file))
        print('output=' + args.output)
        
        wiregrid = analyze_wiregrid()
        data = wiregrid.load_file(args.file)
        result = wiregrid.analyze(data)
        wiregrid.write_result(result, args.output)


if __name__ == '__main__':
    analyze_wiregrid.main() 

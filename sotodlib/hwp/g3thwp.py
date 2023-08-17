#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.interpolate
import so3g
from spt3g import core
import logging
import yaml
import datetime
import h5py
import sotodlib


logger = logging.getLogger(__name__)


class G3tHWP():

    def __init__(self, config_file=None):
        """
        Class to manage L2 HK data into HWP angle g3.

        Args
        -----
        config_file: str
            path to config yaml file
        """
        if config_file is not None:
            if os.path.exists(config_file):
                self.config_file = config_file
                self.configs = yaml.safe_load(open(self.config_file, "r"))
                logger.info("Loading config from " + self.config_file)
            else:
                logger.warning(
                    "Cannot find config file, use all default values")
                self.configs = {}
        else:
            logger.warning("Cannot find config file, use all default values")
            self.configs = {}

        self._start = 0
        self._end = 0
        self._file_list = None

        self._start = self.configs.get('start', 0)
        self._end = self.configs.get('end', 0)

        self._file_list = self.configs.get('file_list', None)
        self._data_dir = self.configs.get('data_dir', None)
        self._margin = self.configs.get('margin', 10)

        # 1st/2nd encoder readout
        self._field_instance = self.configs.get('field_instance',
                                                'observatory.HBA.feeds.HWPEncoder')
        self._field_instance_sub = self.configs.get('field_instance_sub',
                                                    'observatory.HBA2.feeds.HWPEncoder')

        self._field_list = self.configs.get('field_list',
                                            ['rising_edge_count', 'irig_time', 'counter',
                                             'counter_index', 'irig_synch_pulse_clock_time',
                                             'irig_synch_pulse_clock_counts', 'quad'])

        # Size of pakcets sent from the BBB
        # 120 in the latest version, 150 in the previous version
        self._pkt_size = self.configs.get('pkt_size', 120)

        # IRIG type
        # 0: 1Hz IRIG (default), 1: 10Hz IRIG
        self._irig_type = self.configs.get('irig_type', 0)

        # Number of encoder slits per HWP revolution
        self._num_edges = self.configs.get('num_edges', 570 * 2)

        # Reference slit edgen width
        self._ref_edges = self.configs.get('ref_edges', 2)

        # Reference slit angle
        self._delta_angle = 2 * np.pi / self._num_edges

        # Reference slit indexes
        self._ref_indexes = []

        # Search range of reference slot
        self._ref_range = self.configs.get('ref_range', 0.1)

        # Threshoild for outlier data to calculate nominal slit width
        self._slit_width_lim = self.configs.get('slit_width_lim', 0.1)

        # force to quad value
        # 0: use readout quad value (default)
        # 1: positive rotation direction, -1: negative rotation direction
        self._force_quad = int(self.configs.get('force_quad', 0))
        if np.abs(self._force_quad) > 1:
            logger.warning("force_quad in config file must be 0 or 1 or -1")
            if self._force_quad > 1:
                self._force_quad = 1
            else:
                self._force_quad = -1

        # Output path + filename
        self._output = self.configs.get('output', None)

    def load_data(self, start=None, end=None,
                  data_dir=None, instance=None):
        """
        Loads house keeping data for a given time range and
        returns HWP parameters in L2 HK .g3 file

        Args
        -----
            start : timestamp or datetime
                start time for data, assumed to be in UTC unless specified
            end :  timestamp or datetime
                end time for data, assumed to be in UTC unless specified
            data_dir : str or None
                path to HK g3 file, overwrite config file
            instance : str or None
                instance of field list, overwrite config file
                ex.) 'HBA' or 'observatory.HBA.feeds.HWPEncoder'

        Returns
        ----
        dict
            {alias[i] : (time[i], data[i])}
        """
        if start is not None and end is not None:
            self._start = start
            self._end = end
        if self._start is None:
            logger.error("Cannot find time range")
            return {}

        if isinstance(start, datetime.datetime):
            if start.tzinfo is None:
                logger.warning('No tzinfo info in start argument, set to utc timezone')
                start = start.replace(tzinfo=datetime.timezone.utc)
            self._start = start.timestamp()
        if isinstance(end, datetime.datetime):
            if end.tzinfo is None:
                logger.warning('No tzinfo info in end argument, set to utc timezone')
                end = start.replace(tzinfo=datetime.timezone.utc)
            self._end = end.timestamp()

        if data_dir is not None:
            self._data_dir = data_dir
        if self._data_dir is None:
            logger.error("Cannot find data directory")
            return {}
        if instance is not None:
            if 'observatory' in instance:
                self._field_instance = instance
            else:
                self._field_instance = 'observatory.' + instance + '.feeds.HWPEncoder'

        # load housekeeping data with hwp keys
        logger.info('Loading HK data files ')
        logger.info("input time range: " +
                    str(self._start) + " - " + str(self._end))

        fields, alias = self._key_formatting()

        data = so3g.hk.load_range(
            self._start,
            self._end,
            fields,
            alias,
            data_dir=self._data_dir)
        if not any(data):
            logger.info('HWP is not spinning in time range {' + str(
                self._start) + ' - ' + str(self._end) + '}, data is empty')

        return data

    def load_file(self, file_list=None, instance=None):
        """
        Loads house keeping data with specified g3 files.
        Return HWP parameters from SO HK data.

        Args
        -----
            file_list : str or list or None
                path and file name of input level 2 HK g3 file
            instance : str or None
                instance of field list, overwrite config file 

                ex.) 'HBA' or 'observatory.HBA.feeds.HWPEncoder'
        Returns
        ----
        dict
            {alias[i] : (time[i], data[i])}
        """

        if file_list is None and self._file_list is None:
            logger.error('Cannot find input g3 file')
            return {}
        if file_list is not None:
            self._file_list = file_list

        if instance is not None:
            if 'observatory' in instance:
                self._field_instance = instance
            else:
                self._field_instance = 'observatory.' + instance + '.feeds.HWPEncoder'

        # load housekeeping files with hwp keys
        scanner = so3g.hk.HKArchiveScanner()
        if not (isinstance(self._file_list, list)
                or isinstance(self._file_list, np.ndarray)):
            self._file_list = [self._file_list]
        for f in self._file_list:
            if not os.path.exists(f):
                logger.error('Cannot find input g3 file')
                return {}
            scanner.process_file(f)
        logger.info("Loading HK data files: {}".format(
            ' '.join(map(str, self._file_list))))

        arc = scanner.finalize()
        if not any(arc.get_fields()[0]):
            self._start = 0
            self._end = 0
            return {}

        fields, alias = self._key_formatting()

        if not np.any([f in arc.get_fields()[0].keys() for f in fields]):
            logger.info(
                "HWP is not spinning in input g3 files or cannot find field")
            return {}
        if self._start == 0 and self._end == 0:
            self._start = np.min([arc.simple(f)[0][0]
                                 for f in fields if f in arc.get_fields()[0].keys()])
            self._end = np.max([arc.simple(f)[0][-1]
                               for f in fields if f in arc.get_fields()[0].keys()])

        data = {a: arc.simple(f) for a, f in zip(
            alias, fields) if f in arc.get_fields()[0].keys()}

        return data

    def _key_formatting(self):
        """
        Formatting hwp housekeeping field names and aliases

        Return
        -----
        fields, alias
        """
        # 1st encoder readout
        fields = [self._field_instance + '_full.' + f if 'counter' in f
                  else self._field_instance + '.' + f for f in self._field_list]
        alias = self._field_list

        # 2nd encoder readout
        if self._field_instance_sub is not None:
            fields += [self._field_instance_sub + '_full.' + f if 'counter' in f
                       else self._field_instance_sub + '.' + f for f in self._field_list]
            alias += [a + '_2' for a in self._field_list]

        return fields, alias

    def _data_formatting(self, data, suffix=''):
        """
        Formatting encoder data

        Args
        -----
        data : dict
            HWP HK data from load_data
        suffix: Specify whether to use 1st or 2nd encoder, '' or '_2'
            '' for 1st encoder, '_2' for 2nd encoder

        Returns
        --------
        dict
            {'rising_edge_count', 'irig_time', 'counter', 'counter_index', 'quad', 'quad_time'}
        """
        enc_key = {'': '1st', '_2': '2nd'}

        keys = ['rising_edge_count', 'irig_time',
                'counter', 'counter_index', 'quad', 'quad_time']
        out = {k: data[k+suffix][1] if k+suffix in data.keys() else []
               for k in keys}

        # irig part
        if 'irig_time'+suffix not in data.keys():
            logger.warning(
                f'All IRIG time is not correct for {enc_key[suffix]} encoder')
            return out

        if self._irig_type == 1:
            out['irig_time'] = data['irig_synch_pulse_clock_time'+suffix][1]
            out['rising_edge_count'] = data['irig_synch_pulse_clock_counts'+suffix][1]

        logger.info('IRIG timing quality check.')
        out['irig_time'], out['rising_edge_count'] = self._irig_quality_check(
            out['irig_time'], out['rising_edge_count'])

        # encoder part
        if 'counter'+suffix not in data.keys():
            logger.warning(
                f'No encoder data is available for {enc_key[suffix]} encoder')
            return out

        out['quad'] = self._quad_form(data['quad'+suffix][1])
        out['quad_time'] = data['quad'+suffix][0]

        return out

    def _slowdata_process(self, fast_time, irig_time):
        """ Diagnose hwp status and output status flags

        Returns
        --------
        dict
            {stable, locked, hwp_rate, slow_time}

        Notes
        ------
        - Time definition -
        if fast_time exists: slow_time = fast_time
        elif: irig_time exists but no fast_time, slow_time = irig_time
        else: slow_time is per 10 sec array
        """
        slow_time = np.arange(self._start, self._end, 10)

        if len(irig_time) == 0:
            out = {
                'locked': np.zeros_like(slow_time, dtype=bool),
                'stable': np.zeros_like(slow_time, dtype=bool),
                'hwp_rate': np.zeros_like(slow_time, dtype=np.float32),
                'slow_time': slow_time,
            }
            return out

        if len(fast_time) == 0:
            fast_irig_time = irig_time
            locked = np.zeros_like(irig_time, dtype=bool)
            stable = np.zeros_like(irig_time, dtype=bool)
            hwp_rate = np.zeros_like(irig_time, dtype=np.float32)

        else:
            # hwp speed calc. (approximate using ref)
            hwp_rate_ref = 1 / np.diff(fast_time[self._ref_indexes])
            hwp_rate = [hwp_rate_ref[0] for i in range(self._ref_indexes[0])]
            for n in range(len(np.diff(self._ref_indexes))):
                hwp_rate += [hwp_rate_ref[n]
                             for r in range(np.diff(self._ref_indexes)[n])]
            hwp_rate += [hwp_rate_ref[-1] for i in range(len(fast_time) -
                                                         self._ref_indexes[-1])]

            fast_irig_time = fast_time
            locked = np.ones_like(fast_time, dtype=bool)
            locked[np.where(hwp_rate == 0)] = False
            stable = np.ones_like(fast_time, dtype=bool)

            # irig only status
            irig_only_time = irig_time[np.where(
                (irig_time < fast_time[0]) | (irig_time > fast_time[-1]))]
            irig_only_locked = np.zeros_like(irig_only_time, dtype=bool)
            irig_only_hwp_rate = np.zeros_like(irig_only_time, dtype=np.float32)

            fast_irig_time = np.append(irig_only_time, fast_time)
            fast_irig_idx = np.argsort(fast_irig_time)
            fast_irig_time = fast_irig_time[fast_irig_idx]
            locked = np.append(irig_only_locked, locked)[fast_irig_idx]
            hwp_rate = np.append(irig_only_hwp_rate, hwp_rate)[fast_irig_idx]
            stable = np.ones_like(fast_irig_time, dtype=bool)

        # slow status
        slow_time = slow_time[np.where(
            (slow_time < fast_irig_time[0]) | (slow_time > fast_irig_time[-1]))]
        slow_locked = np.zeros_like(slow_time, dtype=bool)
        slow_stable = np.zeros_like(slow_time, dtype=bool)
        slow_hwp_rate = np.zeros_like(slow_time, dtype=np.float32)

        slow_time = np.append(slow_time, fast_irig_time)
        slow_idx = np.argsort(slow_time)
        slow_time = slow_time[slow_idx]
        locked = np.append(slow_locked, locked)[slow_idx]
        stable = np.append(slow_stable, stable)[slow_idx]
        hwp_rate = np.append(slow_hwp_rate, hwp_rate)[slow_idx]

        locked[np.where(hwp_rate == 0)] = False

        return {'locked': locked, 'stable': stable, 'hwp_rate': hwp_rate, 'slow_time': slow_time}

    def analyze(self, data, ratio=None, mod2pi=True, fast=True):
        """
        Analyze HWP angle solution
        to be checked by hardware that 0 is CW and 1 is CCW from (sky side) consistently for all SAT

        Args
        -----
            data : dict
                HWP HK data from load_data
            ratio : float, optional
                parameter for referelce slit
                threshold = 2 slit distances +/- ratio
            mod2pi : bool, optional
                If True, return hwp angle % 2pi
            fast : bool, optional
                If True, run fast fill_ref algorithm

        Returns
        --------
        dict
            {fast_time, angle, slow_time, stable, locked, hwp_rate}


        Notes
        ------
            * fast_time: timestamp
                * IRIG synched timing (~2kHz)
                * angle (float): IRIG synched HWP angle in radian
            * slow_time: timestamp
                * time list of slow block
            * stable: bool
                * if non-zero, indicates the HWP spin state is known.
                * i.e. it is either spinning at a measurable rate, or stationary.
                * When this flag is non-zero, the hwp_rate field can be taken at face value.
            * locked: bool
                * if non-zero, indicates the HWP is spinning and the position solution is working.
                * In this case one should find the hwp_angle populated in the fast data block.
            * hwp_rate: float:
                * the "approximate" HWP spin rate, with sign, in revs / second.
                * Use placeholder value of 0 for cases when not "stable".
        """

        if not any(data):
            logger.info("no HWP field data")

        d = self._data_formatting(data)
        if 'irig_time_2' in data.keys() and 'counter_2' in data.keys():
            d2 = self._data_formatting(data, suffix='_2')
            if len(d['counter']) < len(d2['counter']):
                logger.info('Use 2nd encoder.')
                d = d2
        if len(d['irig_time']) == 0:
            logger.warning('There is no correct IRIG timing. Stop analyze.')
            return {}

        # hwp angle calc.
        if ratio is not None:
            logger.info(f"Overwriting reference slit threshold by {ratio}.")
            self._ref_range = ratio
        fast_time, angle = self._hwp_angle_calculator(
            d['counter'], d['counter_index'], d['irig_time'],
            d['rising_edge_count'], d['quad_time'], d['quad'],
            mod2pi, fast)
        if len(fast_time) == 0:
            logger.warning('analyzed encoder data is None')

        # hwp status calc.
        out = self._slowdata_process(fast_time, d['irig_time'])
        out['fast_time'] = fast_time
        out['angle'] = angle

        return out

    def eval_angle(self, solved, poly_order=3):
        """
        Evaluate the non-uniformity of hwp angle timestamp and subtract
        The raw hwp angle timestamp is kept.

        Args
        -----
        solved: dict
          dict data from analyze

        Returns
        --------
        output: dict
            {fast_time, fast_time_raw, angle, slow_time, stable, locked, hwp_rate, fast_time_moving_ave, angle_moving_ave}

        Notes
        ------
        non-uniformity of hwp angle comes from following reasons,
            - non-uniformity of encoder slits
            - sag of rotor
            - bad tuning of the comparator threshold of DriverBoard
            - degradation of LED
        and the non-uniformity can be time-dependent.

        Need to evaluate and subtract it before interpolating hwp angle into Smurf timestamps.
        The non-uniformity of encoder slots creates additional hwp angle jitter.
        The maximum possible additional jitter is comparable to the requirement of angle jitter.

        The simple method to subrtact the non-uniformity
        is to take the moving average of hwp angle per one revolution.
        The advantages of moving averaging method is is it's simplicity and robustness.
        The disadvantage is that this method is assuming no real angle fluctuation within one revolution.

        The more carful and accurate method is to make an template of encoder slits,
        and subtract it from the timestamp.
        """
        if 'fast_time_raw' in solved.keys():
            logger.info('Non-uniformity is already subtracted. Calculation is skipped.')
            return
                    
        def moving_average(array, n):
            return np.convolve(array, np.ones(n), 'valid')/n

        logger.info('Remove non-uniformity from hwp angle and overwrite')
        solved['fast_time_moving_ave'] = moving_average(
            solved['fast_time'], self._num_edges)
        solved['angle_moving_ave'] = moving_average(
            solved['angle'], self._num_edges)

        def detrend(array, deg=poly_order):
            x = np.linspace(-1, 1, len(array))
            p = np.polyfit(x, array, deg=deg)
            pv = np.polyval(p, x)
            return array - pv

        # template subtraction
        ft = solved['fast_time'][self._ref_indexes[0]:self._ref_indexes[-2]+1]
        # remove rotation frequency drift for making a template of encoder slits
        ft = detrend(ft, deg=3)
        # make template
        template_slit = np.diff(ft).reshape(
            len(self._ref_indexes)-2, self._num_edges)
        template_slit = np.average(template_slit, axis=0)
        average_slit = np.average(template_slit)
        # subtract template, keep raw timestamp
        subtract = np.cumsum(np.roll(np.tile(template_slit-average_slit, len(
            self._ref_indexes) + 1), self._ref_indexes[0] + 1)[:len(solved['fast_time'])])
        solved['fast_time_raw'] = solved['fast_time']
        solved['fast_time'] = solved['fast_time'] - subtract

    def write_solution(self, solved, output=None):
        """
        Output HWP angle + flags as SO HK g3 format

        Args
        -----
        solved: dict
          dict data from analyze
        output: str or None
          output path + file name, overwirte config file

        Notes
        -----------
        Output file format 

        * Provider: 'hwp'
            * Fast block
                * 'hwp.hwp_angle'
            * Slow block
                * 'hwp.stable'
                * 'hwp.locked'
                * 'hwp.hwp_rate'

        - fast_time: timestamp
            IRIG synched timing (~2kHz)
        - angle: float
            IRIG synched HWP angle in radian
        - slow_time: timestamp
            time list of slow block
        - stable: bool
            if non-zero, indicates the HWP spin state is known. 

            i.e. it is either spinning at a measurable rate, or stationary. 

            When this flag is non-zero, the hwp_rate field can be taken at face value. 

        - locked: bool
            if non-zero, indicates the HWP is spinning and the position solution is working. 

            In this case one should find the hwp_angle populated in the fast data block. 

        - hwp_rate: float
            the "approximate" HWP spin rate, with sign, in revs / second. 

            Use placeholder value of 0 for cases when not "locked".
        """
        if self._output is None and output is None:
            logger.warning('Not specified output file')
            return
        if output is not None:
            self._output = output
        if len(solved) == 0:
            logger.warning('input data is empty, skip writing')
            return
        if len(solved['fast_time']) == 0:
            logger.info('write no rotation data, skip writing')
            return
        session = so3g.hk.HKSessionHelper(hkagg_version=2)
        writer = core.G3Writer(output)
        writer.Process(session.session_frame())
        prov_id = session.add_provider('hwp')
        writer.Process(session.status_frame())

        # Divide the full time span into equal intervals
        start_time = solved['slow_time'].min()
        end_time = solved['slow_time'].max()
        if np.any(solved['fast_time']):
            start_time = min(start_time, solved['fast_time'].min())
            end_time = max(end_time, solved['fast_time'].max())
        frame_length = 60  # seconds

        while start_time < end_time:
            t0, t1 = start_time, start_time + frame_length

            # Write a slow frame?
            s = (t0 <= solved['slow_time']) * (solved['slow_time'] < t1)
            if np.any(s):
                frame = session.data_frame(prov_id)

                slow_block = core.G3TimesampleMap()
                slow_block.times = core.G3VectorTime(
                    [core.G3Time(_t * core.G3Units.s) for _t in solved['slow_time'][s]])
                slow_block['stable'] = core.G3VectorInt(solved['stable'][s])
                slow_block['locked'] = core.G3VectorInt(solved['locked'][s])
                slow_block['hwp_rate'] = core.G3VectorDouble(
                    solved['hwp_rate'][s])
                frame['block_names'].append('slow')
                frame['blocks'].append(slow_block)
                writer.Process(frame)

            # Write a fast frame?
            s = (t0 <= solved['fast_time']) * (solved['fast_time'] < t1)
            if np.any(s):
                frame = session.data_frame(prov_id)

                fast_block = core.G3TimesampleMap()
                fast_block.times = core.G3VectorTime(
                    [core.G3Time(_t * core.G3Units.s) for _t in solved['fast_time'][s]])
                fast_block['hwp_angle'] = core.G3VectorDouble(
                    solved['angle'][s])
                frame['block_names'].append('fast')
                frame['blocks'].append(fast_block)
                writer.Process(frame)

            start_time += frame_length

        return
    
    def _set_empty_axes(self, aman):
        
        aman.wrap_new('timestamps', shape=('samps', ), dtype=np.float64)
        aman.wrap_new('hwp_angle_ver1', shape=('samps', ), dtype=np.float64)
        aman.wrap_new('hwp_angle_ver2', shape=('samps', ), dtype=np.float64)
        aman.wrap_new('stable', shape=('samps', ), dtype=bool)
        aman.wrap_new('locked', shape=('samps', ), dtype=bool)
        aman.wrap_new('hwp_rate', shape=('samps', ), dtype=np.float16)
        aman.wrap_new('eval', shape=('samps', ), dtype=bool)
        
        return 
        
    def _write_empty_solution_h5(self, tod, output=None, h5_address=None):
        
        logger.info('Writing empty solutions')
        aman = sotodlib.core.AxisManager(tod.dets, tod.samps)
        self._set_empty_axes(aman)
        aman.timestamps[:] = tod.timestamps
        aman.save(output, h5_address, overwrite=True)
        
        return

    def _bool_interpolation(self, timestamp1, data, timestamp2):
        interp = scipy.interpolate.interp1d(timestamp1, data, kind='linear', bounds_error=False)(timestamp2)
        result = []
        for i in interp:
            if i > .999:
                result.append(True)
            else:
                result.append(False)
        return result

    
    def write_solution_h5(self, tod, output=None, h5_address=None):
        """
        Output HWP angle + flags as AxisManager format

        Args
        -----
        data: g3 file
          data = G3tHWP.load_data(start, end)
        output: str or None
          output path + file name, overwrite config file

        Notes
        -----------

       Output file format 

        - timestamp:
            SMuRF synched timestamp
        - hwp_angle_ver1: float
            SMuRF synched HWP angle (calculated from raw encoder signals) in radian
        - hwp_angle_ver2: float
            SMuRF synched HWP angle after the template subtraction (the systematics from the non-uniform encoder slot pattern is subtracted ) in radian. 

            if 'eval' is zero, template subtraction is not completed and its value is same as 'hwp_angle_ver1'.
        - stable: bool
            if non-zero, indicates the HWP spin state is known. 

            i.e. it is either spinning at a measurable rate, or stationary. 

            When this flag is non-zero, the hwp_rate field can be taken at face value. 

        - locked: bool
            if non-zero, indicates the HWP is spinning and the position solution is working. 

            In this case one should find the hwp_angle populated in the fast data block. 

        - hwp_rate: float
            the "approximate" HWP spin rate, with sign, in revs / second. 

            Use placeholder value of 0 for cases when not "locked".
        - eval: bool
            if non-zero, the template subtraction is completed.
        """
        if self._output is None and output is None:
            logger.warning('Output file not specified')
            return
        if output is not None:
            self._output = output
            
        if len(tod.timestamps) == 0:
            logger.warning('Input data is empty.')
            return 
        
        start = int(tod.timestamps[0])-self._margin
        end = int(tod.timestamps[-1])+self._margin
        try:
            data = self.load_data(start, end)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while loading HWP data. The specified encoder field is missing.")
            self._write_empty_solution_h5(tod, output, h5_address)
            return
            
        if len(data) == 0:
            logger.warning('No HWP data in the specified timestamps.')
            self._write_empty_solution_h5(tod, output, h5_address)
            return
        
        # calculate HWP angle 
        logger.debug("analyze")
        
        try:
            solved = self.analyze(data, mod2pi=False)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while calculating HWP angle. Encoder signal might have too much noise.")
            self._write_empty_solution_h5(tod, output, h5_address)
            return 
            
        if len(solved['fast_time']) == 0:
            logger.info('No rotation data in the specified timestamps.')
            self._write_empty_solution_h5(tod, output, h5_address)
            return
            
        # calculate template subtracted angle
        try:
            self.eval_angle(solved)
        except Exception as e:
            logger.error(f"Exception '{e}' thrown while the template subtraction")
            pass
        
        # write solution
        aman = sotodlib.core.AxisManager(tod.dets, tod.samps)
        self._set_empty_axes(aman)
        if solved['fast_time'][0] > tod.timestamps[0] or solved['fast_time'][-1] < tod.timestamps[-1]:
            logger.error("The angle solution contains empty data at the beginning or end of the timestamps.")
        aman.timestamps[:] = tod.timestamps
        aman.stable[:] = self._bool_interpolation(solved['slow_time'], solved['stable'], tod.timestamps)
        aman.locked[:] = self._bool_interpolation(solved['slow_time'], solved['locked'], tod.timestamps)
        aman.hwp_rate[:] = scipy.interpolate.interp1d(solved['slow_time'], solved['hwp_rate'], kind='linear', bounds_error=False)(tod.timestamps)
        if 'fast_time_raw' in solved.keys():
            aman.hwp_angle_ver1[:] = np.mod(scipy.interpolate.interp1d(solved['fast_time_raw'], solved['angle'], kind='linear',bounds_error=False)(tod.timestamps),2*np.pi)
            aman.hwp_angle_ver2[:] = np.mod(scipy.interpolate.interp1d(solved['fast_time'], solved['angle'], kind='linear',bounds_error=False)(tod.timestamps),2*np.pi)
            aman.eval[:] = np.ones(len(tod.timestamps))
        else:
            logger.info('Template subtraction failed')
            aman.hwp_angle_ver1[:] = np.mod(scipy.interpolate.interp1d(solved['fast_time'], solved['angle'], kind='linear', bounds_error=False)(tod.timestamps),2*np.pi)
            aman.hwp_angle_ver2[:] = np.mod(scipy.interpolate.interp1d(solved['fast_time'], solved['angle'], kind='linear', bounds_error=False)(tod.timestamps),2*np.pi)
    
        aman.save(output, h5_address, overwrite=True)

        return
    
    def _hwp_angle_calculator(
            self,
            counter,
            counter_idx,
            irig_time,
            rising_edge,
            quad_time,
            quad,
            mod2pi,
            fast):

        #   counter: BBB counter values for encoder signal edges
        self._encd_clk = counter
        #   counter_index: index numbers for detected edges by BBB
        self._encd_cnt = counter_idx
        #   irig_time: decoded time in second since the unix epoch
        self._irig_time = irig_time
        # rising_edge_count: BBB clcok count values for the IRIG on-time
        # reference marker risinge edge
        self._rising_edge = rising_edge
        #   quad: quadrature signal to determine rotation direction
        self._quad_time = quad_time
        self._quad = quad

        # return arrays
        self._time = []
        self._angle = []

        # check packet drop
        self._encoder_packet_sort()
        self._find_dropped_packets()

        # assign IRIG synched timestamp
        self._time = scipy.interpolate.interp1d(
            self._rising_edge,
            self._irig_time,
            kind='linear',
            fill_value='extrapolate')(self._encd_clk)
        # Reject unexpected counter
        idx = np.where((1 / np.diff(self._time) / self._num_edges) > 5.0)[0]
        if len(idx) > 0:
            self._encd_clk = np.delete(self._encd_clk, idx)
            self._encd_cnt = self._encd_cnt[0] + \
                np.arange(len(self._encd_cnt) - len(idx))
            self._time = np.delete(self._time, idx)

        # reference finding and fill its angle
        _status_find_ref = self._find_refs()
        if _status_find_ref == -1:
            return [], []
        if fast:
            self._fill_refs_fast()
        else:
            self._fill_refs()

        # re-assign IRIG synched timestamp
        self._time = scipy.interpolate.interp1d(
            self._rising_edge,
            self._irig_time,
            kind='linear',
            fill_value='extrapolate')(self._encd_clk)

        # calculate hwp angle with IRIG timing
        self._calc_angle_linear(mod2pi)

        logger.debug('qualitycheck')
        logger.debug('_time:        ' + str(len(self._time)))
        logger.debug('_angle:       ' + str(len(self._angle)))
        logger.debug('_encd_cnt:    ' + str(len(self._encd_cnt)))
        logger.debug('_encd_clk:    ' + str(len(self._encd_clk)))
        logger.debug('_ref_cnt:     ' + str(len(self._ref_cnt)))
        logger.debug('_ref_indexes: ' + str(len(self._ref_indexes)))

        if len(self._time) != len(self._angle):
            logger.warning('Failed to calculate hwp angle!')
            return [], []
        logger.info('hwp angle calculation is finished.')
        return self._time, self._angle

    def _find_refs(self):
        """ Find reference slits """
        self._ref_indexes = []
        # Calculate spacing between all clock values
        diff = np.ediff1d(self._encd_clk)  # [1:]
        n = 0
        diff_split = []
        for i in range(len(diff)):
            diff_split.append(diff[n:n + (self._num_edges - 2):1])
            n += (self._num_edges - 2)
            if n >= len(diff):
                break
        offset = 1
        # Conditions for idenfitying the ref slit
        # Slit distance somewhere between 2 slits:
        # 2 slit distances (defined above) +/- 10%
        for i in range(len(diff_split)):
            _diff = diff_split[i]
            # eliminate upper/lower _slit_width_lim
            _diff_upperlim = np.percentile(
                _diff, (1 - self._slit_width_lim) * 100)
            _diff_lowerlim = np.percentile(_diff, self._slit_width_lim * 100)
            __diff = _diff[np.where(
                (_diff < _diff_upperlim) & (_diff > _diff_lowerlim))]
            # Define mean value as nominal slit distance
            if len(__diff) == 0:
                continue
            slit_dist = np.mean(__diff)

            # Conditions for idenfitying the ref slit
            # Slit distance somewhere between 2 slits:
            # 2 slit distances (defined above) +/- ref_range
            ref_hi_cond = ((self._ref_edges + 2) *
                           slit_dist * (1 + self._ref_range))
            ref_lo_cond = ((self._ref_edges + 1) *
                           slit_dist * (1 - self._ref_range))
            # Find the reference slit locations (indexes)
            _ref_idx = np.argwhere(np.logical_and(
                _diff < ref_hi_cond, _diff > ref_lo_cond)).flatten()
            if len(_ref_idx) != 1:
                continue
            self._ref_indexes.append(_ref_idx[0] + offset)
            offset += len(diff_split[i])
        # Define the reference slit line to be the line before
        # the two "missing" lines
        # Store the count and clock values of the reference lines
        self._ref_indexes = np.array(self._ref_indexes)
        if len(self._ref_indexes) == 0:
            if len(diff) < self._num_edges:
                logger.warning(
                    'cannot find reference points, # of data is less than # of slit')
            else:
                logger.warning(
                    'cannot find reference points, please adjust parameters!')
            return -1

        ## delete unexpected ref slit indexes ##
        self._ref_indexes = np.delete(self._ref_indexes, np.where(
            np.diff(self._ref_indexes) < self._num_edges - 10)[0])
        self._ref_clk = self._encd_clk[self._ref_indexes]
        self._ref_cnt = self._encd_cnt[self._ref_indexes]
        logger.debug('found {} reference points'.format(
            len(self._ref_indexes)))

        return 0

    def _fill_refs(self):
        """ Fill in the reference edges """
        # If no references, say that the first sample is theta = 0
        # This case comes up for testing with a function generator
        if len(self._ref_clk) == 0:
            self._ref_clk = [self._encd_clk[0]]
            self._ref_cnt = [self._encd_cnt[0]]
            return
        # Loop over all of the reference slits
        for ii in range(len(self._ref_indexes)):
            logger.debug("\r {:.2f} %".format(
                100. * ii / len(self._ref_indexes)), end="")
            # Location of this slit
            ref_index = self._ref_indexes[ii]
            # Linearly interpolate the missing slits
            clks_to_add = np.linspace(
                self._encd_clk[ref_index - 1], self._encd_clk[ref_index], self._ref_edges + 2)[1:-1]
            self._encd_clk = np.insert(self._encd_clk, ref_index, clks_to_add)
            # Adjust the encoder count values for the added lines
            # Add 2 to all future counts and interpolate the counts
            # for the two added slits
            self._encd_cnt[ref_index:] += self._ref_edges
            cnts_to_add = np.linspace(
                self._encd_cnt[ref_index - 1], self._encd_cnt[ref_index], self._ref_edges + 2)[1:-1]
            self._encd_cnt = np.insert(self._encd_cnt, ref_index, cnts_to_add)
            # Also adjsut the reference count values in front of
            # this one for the added lines
            self._ref_cnt[ii + 1:] += self._ref_edges
            # Adjust the reference index values in front of this one
            # for the added lines
            self._ref_indexes[ii + 1:] += self._ref_edges
        return

    def _fill_refs_fast(self):
        """ Fill in the reference edges """
        # If no references, say that the first sample is theta = 0
        # This case comes up for testing with a function generator
        if len(self._ref_clk) == 0:
            self._ref_clk = [self._encd_clk[0]]
            self._ref_cnt = [self._encd_cnt[0]]
            return
        # insert interpolate clk to reference points
        lastsub = np.split(self._encd_clk, self._ref_indexes)[-1]
        self._encd_clk = np.concatenate(
            np.array(
                [[sub_clk, np.linspace(self._encd_clk[ref_index - 1], self._encd_clk[ref_index], self._ref_edges + 2)[1:-1]]
                 for ref_index, sub_clk
                 in zip(self._ref_indexes, np.split(self._encd_clk, self._ref_indexes))], dtype=object
            ).flatten()
        )
        self._encd_clk = np.append(self._encd_clk, lastsub)

        self._encd_cnt = self._encd_cnt[0] + np.arange(
            len(self._encd_cnt) + len(self._ref_indexes) * self._ref_edges)
        self._ref_indexes += np.arange(len(self._ref_indexes)
                                       ) * self._ref_edges
        self._ref_cnt = self._encd_cnt[self._ref_indexes]

        return

    def _flatten_counter(self):
        cnt_diff = np.diff(self._encd_cnt)
        loop_indexes = np.argwhere(cnt_diff <= -(self._max_cnt - 1)).flatten()
        for ind in loop_indexes:
            self._encd_cnt[(ind + 1):] += -(cnt_diff[ind] - 1)
        return

    def _calc_angle_linear(self, mod2pi=True):

        quad = self._quad_form(
            scipy.interpolate.interp1d(
                self._quad_time,
                self._quad,
                kind='linear',
                fill_value='extrapolate')(
                self._time))
        if self._force_quad == 0:
            direction = list(map(lambda x: 1 if x == 0 else -1, quad))
        else:
            direction = self._force_quad

        self._encd_cnt_split = np.split(self._encd_cnt, self._ref_indexes)
        angle_first_revolution = (self._encd_cnt_split[0] - self._ref_cnt[0]) * \
            (2 * np.pi / self._num_edges) % (2 * np.pi)
        angle_last_revolution = (self._encd_cnt_split[-1] - self._ref_cnt[-1]) * \
            (2 * np.pi / self._num_edges) % (2 * np.pi) + \
            len(self._ref_cnt) * 2 * np.pi
        self._angle = np.concatenate([(self._encd_cnt_split[i] - self._ref_cnt[i]) *
                                      (2 * np.pi /
                                       np.diff(self._ref_indexes)[i - 1])
                                      % (2 * np.pi) + i * 2 * np.pi
                                      for i in range(1, len(self._encd_cnt_split) - 1)])
        self._angle = np.concatenate(
            [angle_first_revolution, self._angle.flatten(), angle_last_revolution])
        self._angle = direction * self._angle
        if mod2pi:
            self._angle = self._angle % (2 * np.pi)

        return

    def _find_dropped_packets(self):
        """ Estimate the number of dropped packets """
        cnt_diff = np.diff(self._encd_cnt)
        dropped_samples = np.sum(cnt_diff[cnt_diff >= self._pkt_size])
        self._num_dropped_pkts = dropped_samples // (self._pkt_size - 1)
        if self._num_dropped_pkts > 0:
            logger.warning('{} dropped packets are found, performing fill process'.format(
                self._num_dropped_pkts))

            idx = np.where(np.diff(self._encd_cnt) > 1)[0]
            offset = 0
            for i in idx:
                i += offset
                _diff = np.diff(self._encd_cnt)[i]
                clk = (self._encd_clk[i + 1] - self._encd_clk[i]) / _diff
                gap = np.array(
                    [self._encd_clk[i] + clk * ii for ii in range(1, _diff)])
                self._encd_clk = np.insert(self._encd_clk, i + 1, gap)
                offset += _diff - 1
            self._encd_cnt = self._encd_cnt[0] + np.arange(len(self._encd_clk))
        return

    def _encoder_packet_sort(self):
        cnt_diff = np.diff(self._encd_cnt)
        if np.any(cnt_diff != 1):
            logger.warning(
                'a part of the counter is incorrect')
            if np.any(cnt_diff < 0):
                if 1 - self._pkt_size in cnt_diff:
                    logger.warning(
                        'Packet flip found, performing sort process')
                idx = np.argsort(self._encd_cnt)
                self._encd_clk = self._encd_clk[idx]
            else:
                logger.warning('maybe packet drop exists')
        else:
            logger.debug('no need to fix encoder index')

    def _quad_form(self, quad):
        
        if self._force_quad==1: 
            return np.ones_like(quad)
        # bit process
        quad[(quad >= 0.5)] = 1
        quad[(quad < 0.5)] = 0
        offset = 0
        logger_bit = True
        for quad_split in np.array_split(quad, 1 + np.floor(len(quad) / 100)):
            if quad_split.mean() > 0.1 and quad_split.mean() < 0.9:
                if logger_bit:
                    logger.warning(
                        "flipping quad is corrected by mean value, please consider to use force_quad")
                    logger_bit = False
                for j in range(len(quad_split)):
                    quad[j + offset] = int(quad_split.mean() + 0.5)
                offset += len(quad_split)
                continue

            outlier = np.argwhere(
                np.abs(
                    quad_split.mean() -
                    quad_split) > 0.5).flatten()
            for i in outlier:
                if i == 0:
                    ii, iii = i + 1, i + 2
                elif i == outlier[-1]:
                    ii, iii = i - 1, i - 2
                else:
                    ii, iii = i - 1, i + 1
                if quad_split[i] + quad_split[ii] + quad_split[iii] == 1:
                    quad[i + offset] = 0
                if quad_split[i] + quad_split[ii] + quad_split[iii] == 2:
                    quad[i + offset] = 1
            offset += len(quad_split)

        return quad

    def _irig_quality_check(self, irig_time, rising_edge):
        idx = np.where(np.diff(irig_time) == 1)[0]
        if self._irig_type == 1:
            idx = np.where(np.isclose(np.diff(irig_time), np.full(len(irig_time)-1, 0.1)))[0]
        if len(irig_time) - 1 == len(idx):
            return irig_time, rising_edge
        elif len(irig_time) > len(idx) and len(idx) > 0:
            if np.any(np.diff(irig_time) > 5):
                logger.warning(
                    'a part of the IRIG time is incorrect, performing the correction process...')
            irig_time = irig_time[idx]
            rising_edge = rising_edge[idx]
            logger.debug('deleted wrong irig_time, indices: ' +
                         str(np.where(np.diff(irig_time) != 1)[0]))
        else:
            irig_time = np.array([])
            rising_edge = np.array([])
        return irig_time, rising_edge

    def interp_smurf(self, smurf_timestamp):
        smurf_angle = scipy.interpolate.interp1d(
            self._time,
            self._angle,
            kind='linear',
            fill_value='extrapolate')(smurf_timestamp)
        return smurf_angle

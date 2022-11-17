#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import scipy.interpolate
import so3g
from spt3g import core
import argparse
from tqdm import tqdm
import logging
import yaml

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
                    "Can not find config file, use all default values")
                self.configs = {}
        else:
            logger.warning("Can not find config file, use all default values")
            self.configs = {}

        self._start = 0
        self._end = 0
        self._file_list = None

        self._start = self.configs.get('start', 0)
        self._end = self.configs.get('end', 0)

        self._file_list = self.configs.get('file_list', None)
        self._data_dir = self.configs.get('data_dir', None)

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
            logger.error("force_quad in config file must be 0 or 1 or -1")
            sys.exit(1)

        # Output path + filename
        self._output = self.configs.get('output', None)

    def load_data(self, start=None, end=None,
                  data_dir=None, instance='HBA'):
        """
        Loads house keeping data for a given time range and 
        returns HWP parameters in L2 HK .g3 file

        Args
        -----
            start : timestamp or DateTime
                start time for data, assumed to be in UTC unless specified
            end :  timestamp or DateTime
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
            logger.error("Can not find time range")

        if isinstance(start, np.datetime64):
            start = start.timestamp()
        if isinstance(end, np.datetime64):
            end = end.timestamp()

        if data_dir is not None:
            self._data_dir = data_dir
        if self._data_dir is None:
            logger.error("Can not find data directory")
            sys.exit(1)
        if instance is not None:
            if 'observatory' in instance:
                self._field_instance = instance
            else:
                self._field_instance = 'observatory.' + instance + '.feeds.HWPEncoder'
        else:
            logger.error("Can not find field instance")
            sys.exit(1)

        # load housekeeping data with hwp keys
        logger.info('Loading HK data files ')
        logger.info("input time range: " +
                    str(self._start) + " - " + str(self._end))

        hwp_keys = []
        for i in range(len(self._field_list)):
            if 'counter' in self._field_list[i]:
                hwp_keys.append(self._field_instance +
                                '_full.' + self._field_list[i])
            else:
                hwp_keys.append(self._field_instance +
                                '.' + self._field_list[i])
        alias = self._field_list

        # 2nd encoder readout
        hwp_keys_2 = []
        alias_2 = []
        if self._field_instance_sub is not None:
            for i in range(len(self._field_list)):
                if 'counter' in self._field_list[i]:
                    hwp_keys_2.append(self._field_instance_sub +
                                      '_full.' + self._field_list[i])
                else:
                    hwp_keys_2.append(self._field_instance_sub +
                                      '.' + self._field_list[i])
            alias_2 = [a + '_2' for a in self._field_list]

        data = so3g.hk.load_range(
            self._start,
            self._end,
            fields=hwp_keys + hwp_keys_2,
            alias=alias + alias_2,
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
                instance of field list, overwrite config file \n
                ex.) 'HBA' or 'observatory.HBA.feeds.HWPEncoder'
        Returns
        ----
        dict 
            {alias[i] : (time[i], data[i])} 
        """

        if file_list is None and self._file_list is None:
            logger.error('Can not find input g3 file')
            sys.exit(1)
        if file_list is not None:
            self._file_list = file_list

        if instance is not None:
            if 'observatory' in instance:
                self._field_instance = instance
            else:
                self._field_instance = 'observatory.' + instance + '.feeds.HWPEncoder'

        hwp_keys = []
        for i in range(len(self._field_list)):
            if 'counter' in self._field_list[i]:
                hwp_keys.append(self._field_instance +
                                '_full.' + self._field_list[i])
            else:
                hwp_keys.append(self._field_instance +
                                '.' + self._field_list[i])
        alias = self._field_list

        # 2nd encoder readout
        hwp_keys_2 = []
        alias_2 = []
        if self._field_instance_sub is not None:
            for i in range(len(self._field_list)):
                if 'counter' in self._field_list[i]:
                    hwp_keys_2.append(self._field_instance_sub +
                                      '_full.' + self._field_list[i])
                else:
                    hwp_keys_2.append(self._field_instance_sub +
                                      '.' + self._field_list[i])
            alias_2 = [a + '_2' for a in self._field_list]

        # load housekeeping files with hwp keys
        scanner = so3g.hk.HKArchiveScanner()
        if not (isinstance(self._file_list, list)
                or isinstance(self._file_list, np.ndarray)):
            self._file_list = [self._file_list]
        for f in self._file_list:
            if not os.path.exists(f):
                logger.error('Can not find input g3 file')
                sys.exit(1)
            scanner.process_file(f)
        logger.info("Loading HK data files: {}".format(
            ' '.join(map(str, self._file_list))))

        arc = scanner.finalize()
        if not any(arc.get_fields()[0]):
            self._start = 0
            self._end = 0
            return {}

        for i in range(len(hwp_keys)):
            if not hwp_keys[i] in arc.get_fields()[0].keys():
                logger.info(
                    "HWP is not spinning in input g3 files or can not find field")
                return {}
            elif self._start == 0 and self._end == 0:
                self._start = arc.simple(hwp_keys[0])[0][0]
                self._end = arc.simple(hwp_keys[0])[0][-1]

        data = {}
        for i in range(len(alias)):
            data = dict(**data, **{alias[i]: arc.simple(hwp_keys[i])})

        for i in range(len(alias_2)):
            if not hwp_keys_2[i] in arc.get_fields()[0].keys():
                continue
            data = dict(**data, **{alias_2[i]: arc.simple(hwp_keys_2[i])})

        return data

    def analyze(self, data, ratio=0.25, fast=True):
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
        ## Analysis parameters ##
        # Fast block
        counter = np.array([])
        counter_idx = np.array([])
        quad_time = np.array([])
        quad = np.array([])
        if 'counter' in data.keys():
            counter = data['counter'][1]
            counter_idx = data['counter_index'][1]
            quad_time = data['quad'][0]
            quad = self._quad_form(data['quad'][1])

        irig_time = np.array([])
        rising_edge = np.array([])
        if 'irig_time' in data.keys():
            irig_time = data['irig_time'][1]
            rising_edge = data['rising_edge_count'][1]
            if self._irig_type == 1:
                irig_time = data['irig_synch_pulse_clock_time'][1]
                rising_edge = data['irig_synch_pulse_clock_counts'][1]
            logger.info('IRIG timing quality check.')
            irig_time, rising_edge = self._irig_quality_check(
                irig_time, rising_edge)
            if len(irig_time) == 0:
                logger.warning('All IRIG time is not correct in 1st encoder')

        irig_time_2 = np.array([])
        rising_edge_2 = np.array([])
        if 'irig_time_2' in data.keys() and 'counter_2' in data.keys():
            logger.info('checking 2nd encoder.')
            irig_time_2 = data['irig_time_2'][1]
            rising_edge_2 = data['rising_edge_count_2'][1]
            if self._irig_type == 1:
                irig_time_2 = data['irig_synch_pulse_clock_time_2'][1]
                rising_edge_2 = data['irig_synch_pulse_clock_counts_2'][1]
            irig_time_2, rising_edge_2 = self._irig_quality_check(
                irig_time_2, rising_edge_2)
            if len(irig_time_2) == 0:
                logger.info('All IRIG time is not correct in 2nd encoder')

        if len(irig_time) == 0 and len(irig_time_2) == 0:
            logger.warning('There is no correct IRIG timing.')
        if len(irig_time) < len(irig_time_2):
            logger.info('Use 2nd encoder IRIG timing.')
            irig_time = irig_time_2
            rising_edge = rising_edge_2
            counter = data['counter_2']
            counter_index = data['counter_index_2']
            quad_time = data['quad_2'][0]
            quad = self._quad_form(data['quad_2'][1])

        fast_time = np.array([])
        angle = np.array([])
        hwp_rate = np.array([])
        if len(counter) > 0 and len(irig_time) > 0:
            fast_time, angle = self._hwp_angle_calculator(
                counter, counter_idx, irig_time, rising_edge, quad_time, quad, ratio, fast)

            # hwp speed calc. (approximate using ref)
            hwp_rate_ref = 1 / np.diff(fast_time[self._ref_indexes])
            hwp_rate = [hwp_rate_ref[0] for i in range(self._ref_indexes[0])]
            for n in range(len(np.diff(self._ref_indexes))):
                hwp_rate += [hwp_rate_ref[n]
                             for r in range(np.diff(self._ref_indexes)[n])]
            hwp_rate += [hwp_rate_ref[-1] for i in range(len(fast_time) -
                                                         self._ref_indexes[-1])]

        # Slow block
        # - Time definition -
        # if fast_time exists: slow_time = fast_time
        # elif: irig_time exists but no fast_time, slow_time = irig_time
        # else: slow_time is per 10 sec array
        if len(fast_time) != 0 and len(irig_time) != 0:
            fast_irig_time = fast_time
            locked = np.ones(len(fast_time), dtype=bool)
            locked[np.where(hwp_rate == 0)] = False
            stable = np.ones(len(fast_time), dtype=bool)

            irig_only_time = irig_time[np.where(
                (irig_time < fast_time[0]) | (irig_time > fast_time[-1]))]
            irig_only_locked = np.zeros(len(irig_only_time), dtype=bool)
            irig_only_hwp_rate = np.zeros(len(irig_only_time), dtype=float)
            fast_irig_time = np.append(irig_only_time, fast_time).flatten()
            fast_irig_idx = np.argsort(fast_irig_time)
            fast_irig_time = fast_irig_time[fast_irig_idx]
            locked = (np.append(irig_only_locked, locked).flatten())[
                fast_irig_idx]
            hwp_rate = (np.append(irig_only_hwp_rate, hwp_rate).flatten())[
                fast_irig_idx]
            stable = np.ones(len(fast_irig_time), dtype=bool)
        elif len(fast_time) == 0 and len(irig_time) != 0:
            fast_irig_time = irig_time
            locked = np.zeros(len(fast_irig_time), dtype=bool)
            stable = np.ones(len(fast_irig_time), dtype=bool)
            hwp_rate = np.zeros(len(fast_irig_time), dtype=float)
        else:
            fast_irig_time = []
            locked = []
            stable = []
            hwp_rate = []

        slow_only_time = np.arange(self._start, self._end, 10)
        if len(fast_irig_time) != 0:
            irig_only_locked = np.zeros(len(irig_only_time), dtype=bool)
            irig_only_hwp_rate = np.zeros(len(irig_only_time), dtype=float)
            fast_irig_time = np.append(irig_only_time, fast_time).flatten()
            fast_irig_idx = np.argsort(fast_irig_time)
            fast_irig_time = fast_irig_time[fast_irig_idx]
            locked = (np.append(irig_only_locked, locked).flatten())[
                fast_irig_idx]
            hwp_rate = (np.append(irig_only_hwp_rate, hwp_rate).flatten())[
                fast_irig_idx]
            stable = np.ones(len(fast_irig_time), dtype=bool)

            slow_only_time = slow_only_time[np.where(
                (slow_only_time < fast_irig_time[0]) | (slow_only_time > fast_irig_time[-1]))]
            slow_only_locked = np.zeros(len(slow_only_time), dtype=bool)
            slow_only_stable = np.zeros(len(slow_only_time), dtype=bool)
            slow_only_hwp_rate = np.zeros(len(slow_only_time), dtype=float)

            slow_time = np.append(slow_only_time, fast_irig_time).flatten()
            slow_idx = np.argsort(slow_time)
            slow_time = slow_time[slow_idx]
            locked = (np.append(slow_only_locked, locked).flatten())[slow_idx]
            stable = (np.append(slow_only_stable, stable).flatten())[slow_idx]
            hwp_rate = (np.append(slow_only_hwp_rate,
                        hwp_rate).flatten())[slow_idx]
            locked[np.where(hwp_rate == 0)] = False
        else:
            slow_time = slow_only_time
            locked = np.zeros(len(slow_time), dtype=bool)
            stable = np.zeros(len(slow_time), dtype=bool)
            hwp_rate = np.zeros(len(slow_time), dtype=float)

        return {'fast_time': fast_time, 'angle': angle, 'slow_time': slow_time,
                'stable': stable, 'locked': locked, 'hwp_rate': hwp_rate}

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
        Output file format \n
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
            if non-zero, indicates the HWP spin state is known. \n
            i.e. it is either spinning at a measurable rate, or stationary. \n
            When this flag is non-zero, the hwp_rate field can be taken at face value. \n
        - locked: bool 
            if non-zero, indicates the HWP is spinning and the position solution is working. \n
            In this case one should find the hwp_angle populated in the fast data block. \n
        - hwp_rate: float
            the "approximate" HWP spin rate, with sign, in revs / second. \n
            Use placeholder value of 0 for cases when not "locked". 
        """
        if self._output is None and output is None:
            logger.error('Not specified output file')
            return
        if output is not None:
            self._output = output
        if len(solved['slow_time']) == 0:
            logger.error('input data is empty')
            return
        if len(solved['fast_time']) == 0:
            logger.info('write no rotation data')
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

    def _hwp_angle_calculator(
            self,
            counter,
            counter_idx,
            irig_time,
            rising_edge,
            quad_time,
            quad,
            ratio,
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
        idx = np.where((1/np.diff(self._time)/self._num_edges) > 5.0)[0]
        if len(idx) > 0:
            self._encd_clk = np.delete(self._encd_clk, idx)
            self._encd_cnt = self._encd_cnt[0] + \
                np.arange(len(self._encd_cnt) - len(idx))
            self._time = np.delete(self._time, idx)

        # reference finding and fill its angle
        self._find_refs()
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
        self._calc_angle_linear()

        logger.debug('qualitycheck')
        logger.debug('_time:        ' + str(len(self._time)))
        logger.debug('_angle:       ' + str(len(self._angle)))
        logger.debug('_encd_cnt:    ' + str(len(self._encd_cnt)))
        logger.debug('_encd_clk:    ' + str(len(self._encd_clk)))
        logger.debug('_ref_cnt:     ' + str(len(self._ref_cnt)))
        logger.debug('_ref_indexes: ' + str(len(self._ref_indexes)))

        if len(self._time) != len(self._angle):
            raise ValueError('Failed to calculate hwp angle!')
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
            diff_split.append(diff[n:n+(self._num_edges-2):1])
            n += (self._num_edges - 2)
            if n >= len(diff):
                break
        offset = 0
        # Conditions for idenfitying the ref slit
        # Slit distance somewhere between 2 slits:
        # 2 slit distances (defined above) +/- 10%
        for i in range(len(diff_split)):
            _diff = diff_split[i]
            # eliminate upper/lower _slit_width_lim
            _diff_upperlim = np.percentile(
                _diff, (1 - self._slit_width_lim)*100)
            _diff_lowerlim = np.percentile(_diff, self._slit_width_lim*100)
            __diff = _diff[np.where(
                (_diff < _diff_upperlim) & (_diff > _diff_lowerlim))]
            # Define mean value as nominal slit distance
            if len(__diff) == 0: continue
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
            if len(_ref_idx) != 1: continue
            self._ref_indexes.append(_ref_idx[0] + offset)
            offset += len(diff_split[i])
        # Define the reference slit line to be the line before
        # the two "missing" lines
        # Store the count and clock values of the reference lines
        self._ref_indexes = np.array(self._ref_indexes)
        if len(self._ref_indexes) == 0:
            if len(diff) < self._num_edges:
                logger.error(
                    'can not find reference points, # of data is less than # of slit')
            else:
                logger.error(
                    'can not find reference points, please adjust parameters!')
            sys.exit(1)

        ## delete unexpected ref slit indexes ##
        self._ref_indexes = np.delete(self._ref_indexes, np.where(
            np.diff(self._ref_indexes) < self._num_edges-10)[0])
        self._ref_clk = self._encd_clk[self._ref_indexes]
        self._ref_cnt = self._encd_cnt[self._ref_indexes]
        logger.debug('found {} reference points'.format(
            len(self._ref_indexes)))

        return

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

    def _calc_angle_linear(self):

        quad = self._quad_form(
            scipy.interpolate.interp1d(
                self._quad_time,
                self._quad,
                kind='linear',
                fill_value='extrapolate')(
                self._time),
            interp=True)
        if self._force_quad == 0:
            direction = list(map(lambda x: 1 if x == 0 else -1, quad))
        else:
            direction = self._force_quad

        self._encd_cnt_split = np.split(self._encd_cnt, self._ref_indexes)
        self._angle = (self._encd_cnt_split[0] - self._ref_cnt[0]) * \
            (2 * np.pi / self._num_edges) % (2 * np.pi)
        self._angle = np.append(self._angle,
                                np.concatenate(
                                    np.array(
                                        [((self._encd_cnt_split[i] - self._ref_cnt[i]) *
                                          (2 * np.pi /
                                            np.diff(self._ref_indexes)[i-1])
                                            % (2 * np.pi)).flatten()
                                            for i in range(1, len(self._encd_cnt_split)-1)])))
        self._angle = np.append(self._angle,
                                (self._encd_cnt_split[-1] - self._ref_cnt[-1]) *
                                (2 * np.pi / self._num_edges) % (2 * np.pi))
        self._angle = direction * self._angle % (2 * np.pi)

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
                clk = (self._encd_clk[i+1] - self._encd_clk[i]) / _diff
                gap = np.array(
                    [self._encd_clk[i] + clk*ii for ii in range(1, _diff)])
                self._encd_clk = np.insert(self._encd_clk, i+1, gap)
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

    def _quad_form(self, quad, interp=False):
        # treat 30 sec span noise
        if not interp:
            quad_diff = np.ediff1d(quad, to_begin=0)
            for i in np.argwhere(quad_diff == 1).flatten():
                if i != 0 and i != np.argwhere(quad_diff == 1).flatten()[-1]:
                    if quad[i - 1] == 0 and quad[i] > 0 and quad[i + 1] == 0:
                        quad[i] = 0
        # bit process
        quad[(quad >= 0.5)] = 1
        quad[(quad > 0) & (quad < 0.5)] = 0
        return quad

    def _irig_quality_check(self, irig_time, rising_edge):
        idx = np.where(np.diff(irig_time) == 1)[0]
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
            self.angle,
            kind='linear',
            fill_value='extrapolate')(smurf_timestamp)
        return smurf_angle

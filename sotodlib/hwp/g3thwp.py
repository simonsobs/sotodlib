#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.interpolate
import h5py
from copy import copy
import so3g
from spt3g import core
import logging
import yaml
import datetime
import sotodlib
import traceback


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
                                                'satp1.hwp-bbb-e1.feeds.HWPEncoder')
        self._field_instance_sub = self.configs.get('field_instance_sub',
                                                    'satp1.hwp-bbb-e2.feeds.HWPEncoder')

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

        # Number of encoder slits per HWP revolution pre-process
        self._edges_per_rev = self._num_edges - self._ref_edges

        # Reference slit angle
        self._delta_angle = 2 * np.pi / self._num_edges

        # Search range of reference slot
        self._ref_range = self.configs.get('ref_range', 0.1)

        # Threshoild for outlier data to calculate nominal slit width
        self._slit_width_lim = self.configs.get('slit_width_lim', 0.1)

        # The distance from the hwp center to the fine encoder slots (mm)
        self._encoder_disk_radius = self.configs.get(
            'encoder_disk_radius', 346.25)

        # The time period and amount of irig desynchronization
        # [ start_time, stop_time, amount of time shift ]
        self._irig_desync = self.configs.get('irig_desync', None)

        # Output path + filename
        self._output = self.configs.get('output', None)

        # encoder suffixes
        self._suffixes = ['_1', '_2']

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
                ex.) lab data 'observatory.HBA.feeds.HWPEncoder'
                ex.) site data 'satp1.hwp-bbb-e1.feeds.HWPEncoder'
                ex.) site data 'satp3.hwp-bbb-a1.feeds.HWPEncoder'

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
                logger.warning(
                    'No tzinfo info in start argument, set to utc timezone')
                start = start.replace(tzinfo=datetime.timezone.utc)
            self._start = start.timestamp()
        if isinstance(end, datetime.datetime):
            if end.tzinfo is None:
                logger.warning(
                    'No tzinfo info in end argument, set to utc timezone')
                end = start.replace(tzinfo=datetime.timezone.utc)
            self._end = end.timestamp()

        if data_dir is not None:
            self._data_dir = data_dir
        if self._data_dir is None:
            logger.error("Cannot find data directory")
            return {}
        if instance is not None:
            self._field_instance = instance

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
                ex.) lab data 'observatory.HBA.feeds.HWPEncoder'
                ex.) site data 'satp1.hwp-bbb-e1.feeds.HWPEncoder'
                ex.) site data 'satp3.hwp-bbb-a1.feeds.HWPEncoder'
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
            self._field_instance = instance

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
        alias = [a + '_1' for a in self._field_list]

        # 2nd encoder readout
        if self._field_instance_sub is not None:
            fields += [self._field_instance_sub + '_full.' + f if 'counter' in f
                       else self._field_instance_sub + '.' + f for f in self._field_list]
            alias += [a + '_2' for a in self._field_list]

        # metadata key
        meta_keys = {
            'pid_direction': 'hwp-pid.feeds.hwppid.direction',
        }
        platform = self._field_instance.split('.')[0]
        for k, f in meta_keys.items():
            alias.append(k)
            fields.append(platform + '.' + f)

        return fields, alias

    def _data_formatting(self, data, suffix):
        """
        Formatting encoder data

        Args
        -----
        data : dict
            HWP HK data from load_data
        suffix: Specify whether to use 1st or 2nd encoder, '_1' or '_2'
            '_1' for 1st encoder, '_2' for 2nd encoder

        Returns
        --------
        dict
            {'rising_edge_count', 'irig_time', 'counter', 'counter_index', 'quad', 'quad_time'}
        """
        keys = ['rising_edge_count', 'irig_time',
                'counter', 'counter_index', 'quad', 'quad_time']
        out = {k: data[k+suffix][1] if k+suffix in data.keys() else []
               for k in keys}

        # irig part
        if 'irig_time'+suffix not in data.keys():
            logger.warning(
                'All IRIG time is not correct for encoder' + suffix)
            return out

        if self._irig_type == 1:
            out['irig_time'] = data['irig_synch_pulse_clock_time'+suffix][1]
            out['rising_edge_count'] = data['irig_synch_pulse_clock_counts'+suffix][1]

        # encoder part
        if 'counter'+suffix not in data.keys():
            logger.warning(
                'No encoder data is available for encoder'+suffix)
            return out

        out['quad'] = data['quad'+suffix][1]
        out['quad_time'] = data['quad'+suffix][0]

        return out

    def _slowdata_process(self, fast_time, irig_time, suffix):
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
                'locked'+suffix: np.zeros_like(slow_time, dtype=bool),
                'stable'+suffix: np.zeros_like(slow_time, dtype=bool),
                'hwp_rate'+suffix: np.zeros_like(slow_time, dtype=np.float32),
                'slow_time'+suffix: slow_time,
            }
            return out

        if len(fast_time) == 0:
            fast_irig_time = irig_time
            locked = np.zeros_like(irig_time, dtype=bool)
            stable = np.zeros_like(irig_time, dtype=bool)
            hwp_rate = np.zeros_like(irig_time, dtype=np.float32)

        else:
            # hwp speed calc. (approximate using ref)
            ref_indexes = self._ref_indexes
            if isinstance(self._ref_indexes, tuple):
                ref_indexes = self._ref_indexes[0]
            hwp_rate_ref = 1 / np.diff(fast_time[ref_indexes])
            hwp_rate = [hwp_rate_ref[0] for i in range(ref_indexes[0])]
            for n in range(len(np.diff(ref_indexes))):
                hwp_rate += [hwp_rate_ref[n]
                             for r in range(np.diff(ref_indexes)[n])]
            hwp_rate += [hwp_rate_ref[-1] for i in range(len(fast_time) -
                                                         ref_indexes[-1])]

            fast_irig_time = fast_time
            locked = np.ones_like(fast_time, dtype=bool)
            locked[hwp_rate == 0] = False
            stable = np.ones_like(fast_time, dtype=bool)

            # irig only status
            irig_only_time = irig_time[
                (irig_time < fast_time[0]) | (irig_time > fast_time[-1])]
            irig_only_locked = np.zeros_like(irig_only_time, dtype=bool)
            irig_only_hwp_rate = np.zeros_like(
                irig_only_time, dtype=np.float32)

            fast_irig_time = np.append(irig_only_time, fast_time)
            fast_irig_idx = np.argsort(fast_irig_time)
            fast_irig_time = fast_irig_time[fast_irig_idx]
            locked = np.append(irig_only_locked, locked)[fast_irig_idx]
            hwp_rate = np.append(irig_only_hwp_rate, hwp_rate)[fast_irig_idx]
            stable = np.ones_like(fast_irig_time, dtype=bool)

        # slow status
        slow_time = slow_time[
            (slow_time < fast_irig_time[0]) | (slow_time > fast_irig_time[-1])]
        slow_locked = np.zeros_like(slow_time, dtype=bool)
        slow_stable = np.zeros_like(slow_time, dtype=bool)
        slow_hwp_rate = np.zeros_like(slow_time, dtype=np.float32)

        slow_time = np.append(slow_time, fast_irig_time)
        slow_idx = np.argsort(slow_time)
        slow_time = slow_time[slow_idx]
        locked = np.append(slow_locked, locked)[slow_idx]
        stable = np.append(slow_stable, stable)[slow_idx]
        hwp_rate = np.append(slow_hwp_rate, hwp_rate)[slow_idx]

        locked[hwp_rate == 0] = False

        return {'locked'+suffix: locked, 'stable'+suffix: stable, 'hwp_rate'+suffix: hwp_rate, 'slow_time'+suffix: slow_time}

    def analyze(self, data, ratio=None, mod2pi=True, fast=True, suffix='_1'):
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
            {fast_time, angle, slow_time, stable, locked, hwp_rate, ref_indexes, filled_flag}


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
            * hwp_rate: float
                * the "approximate" HWP spin rate, with sign, in revs / second.
                * Use placeholder value of 0 for cases when not "stable".
            * ref_indexes: int
                * Indexes of of reference slots.
            * filled_flag: bool
                * Flag indicating the points that are filled due to packet drop.
            * num_dropped_packets
                * number of dropped encoder packets
            * num_dropped_packets_irig
                * number of dropped irig packets
            * num_glitches
                * number of encoder data point glitches, unexpected data points
            * num_value_glitches
                * number of encoder value glitches, points with value shift due to glitches
            * num_glitches_irig
                * number of irig data point glitches, unexpected data points
            * num_value_glitches_irig
                * number of irig value glitches, points with value shift due to glitches
            * num_dead_rots
                * number rotations that failed to fix glitches
            * num_dropped_slits
                * number rotations that failed to fix glitches
        """

        if not any(data):
            logger.info("no HWP field data")

        d = self._data_formatting(data, suffix)

        # hwp angle calc.
        if ratio is not None:
            logger.info(f"Overwriting reference slit threshold by {ratio}.")
            self._ref_range = ratio

        out = {}

        logger.info("Start calclulating angle.")
        if len(d['irig_time']) == 0:
            logger.warning('There is no correct IRIG timing. Stop analyze.')
        else:
            fast_time, angle = self._hwp_angle_calculator(
                d['counter'], d['counter_index'], d['irig_time'],
                d['rising_edge_count'], d['quad_time'], d['quad'],
                mod2pi, fast)
            if len(fast_time) == 0:
                logger.warning('analyzed encoder data is None')
            out.update(self._slowdata_process(
                fast_time, d['irig_time'], suffix))
            out['fast_time'+suffix] = fast_time
            out['angle'+suffix] = angle
            out['quad'+suffix] = self._quad_corrected
            out['ref_indexes'+suffix] = self._ref_indexes
            # generate flags
            filled_flag = np.zeros_like(fast_time, dtype=bool)
            out['filled_flag'+suffix] = filled_flag
            out['num_dropped_packets'+suffix] = int(self._num_dropped_pkts)
            out['num_dropped_packets_irig'+suffix] = int(self._num_dropped_pkts_irig)
            out['num_glitches'+suffix] = int(self._num_glitches)
            out['num_glitches_irig'+suffix] = int(self._num_glitches_irig)
            out['num_value_glitches'+suffix] = int(self._num_value_glitches)
            out['num_value_glitches_irig'+suffix] = int(self._num_value_glitches_irig)
            out['num_dead_rots'+suffix] = int(self._num_dead_rots)
            out['num_dropped_slits'+suffix] = int(self._num_dropped_slits)
        return out

    def eval_angle(self, solved, poly_order=3, suffix='_1'):
        """
        Evaluate the non-uniformity of hwp angle timestamp (template) and subtract it
        The raw hwp angle timestamp is kept.

        Args
        -----
        solved: dict
            dict data from analyze
        poly_order:
            order of polynomial filtering for removing drift of hwp speed
            for evaluating the non-uniformity of hwp angle.
        suffix:
            '_1' for 1st encoder, '_2' for 2nd encoder

        Returns
        --------
        output: dict
            {fast_time, fast_time_raw, template, template_err, ...}


        Notes
        ------
            * template: float array (ratio)
                * Averaged non-uniformity of the hwp angle
                * normalized by the step of the angle encoder
            * template_err: float array
                * Error bar of template

        non-uniformity of hwp angle comes from following reasons,
            - non-uniformity of encoder slits
            - sag of rotor
            - bad tuning of the comparator threshold of DriverBoard
            - degradation of LED
        and the non-uniformity can be time-dependent.

        Need to evaluate and subtract it before interpolating hwp angle into Smurf timestamps.
        The non-uniformity of encoder slots creates additional hwp angle jitter.
        The maximum possible additional jitter is comparable to the requirement of angle jitter.
        We make an template of encoder slits and subtract it from the timestamp.
        """
        if 'fast_time_raw'+suffix in solved.keys():
            logger.info(
                'Non-uniformity is already subtracted. Calculation is skipped.')
            return

        def detrend(array, deg):
            x = np.linspace(-1, 1, len(array))
            p,_ ,_ ,_ ,_ = np.polyfit(x, array, deg=deg, full=True)  # supress rank warning
            pv = np.polyval(p, x)
            return array - pv

        logger.info('Remove non-uniformity from hwp angle and overwrite')
        ref_indexes = solved['ref_indexes'+suffix]
        fast_time = solved['fast_time'+suffix]

        # Trim only the timestamps of integer revolutions
        ft = fast_time[ref_indexes[0]:ref_indexes[-2]+1]
        # remove rotation frequency drift for making a template of encoder slits
        ft = detrend(ft, deg=poly_order)
        # make template from difference of time
        template_slit = np.diff(ft).reshape(len(solved['ref_indexes'+suffix])-2, self._num_edges)
        template_err = np.std(template_slit, axis=0)
        template_slit = np.average(template_slit, axis=0) # take average of all revolutions
        template_slit = np.cumsum(template_slit)
        template_slit -= np.average(template_slit) # remove global time ofset
        subtract = np.roll(np.tile(template_slit, int(np.ceil(len(fast_time)/self._num_edges))), ref_indexes[0]+1)
        subtract = subtract[:len(fast_time)]
        # subtract template, keep raw timestamp
        solved['fast_time_raw'+suffix] = copy(fast_time)
        solved['fast_time'+suffix] = fast_time - subtract
        # Normalize template by the width of slit
        average_dt_slit = np.average(np.diff(fast_time - subtract))
        solved['template'+suffix] = template_slit / average_dt_slit
        solved['template_err'+suffix] = template_err / average_dt_slit


    def template_subtraction(self, solved, suffix='_1'):
        """ Template subtraction taking into account the drift of hwp rotation speed
        """
        ref_indexes = solved['ref_indexes'+suffix]
        fast_time = solved['fast_time'+suffix]

        counter = np.arange(len(fast_time))
        spl = scipy.interpolate.CubicSpline(counter[ref_indexes], fast_time[ref_indexes])
        dt_smoothed = spl(counter)
        dt_derivative = spl.derivative()(counter)

        template = np.split(fast_time - dt_smoothed, ref_indexes)[1:-1]
        template_err = np.std(template, axis=0)
        template = np.average(template, axis=0)
        template -= np.average(template)  #  no global shift

        template_model = np.roll(template, ref_indexes[0])
        template_model = np.tile(template_model, int(np.ceil(len(fast_time)/self._num_edges)))[:len(fast_time)]
        template_model = template_model * dt_derivative / np.average(dt_derivative)

        solved['fast_time_raw'+suffix] = copy(fast_time)
        solved['fast_time'+suffix] = fast_time - template_model

        # Normalize template by the width of slit
        average_dt_slit = np.average(np.diff(fast_time - template_model))
        solved['template'+suffix] = np.diff(template, append=template[0]) / average_dt_slit
        solved['template_t'+suffix] = template / average_dt_slit
        solved['template_t_err'+suffix] = template_err / average_dt_slit
        return


    def eval_offcentering(self, solved):
        """
        Evaluate the off-centering of the hwp from the phase difference between two encoders.
        Assume that slot pattern subraction is already applied

        * Definition of offcentering must be clear.

        Args
        -----
        solved: dict
            dict solved from template_subtraction
            {fast_time_1, angle_2, fast_time_2, angle_2, ...}

        Returns
        --------
        output: dict
            {offcenter_idx1, offcenter_idx2, offcentering, offset_time}

        Notes
        ------
            * offcenter_idx1: int
                * index of the solved['fast_time_1'] for which offcentering is estimated.
            * offcenter_idx2: int
                * index of the solved['fast_time_2'] for which offcentering is estimated.
            * offcentering: float
                * Offcentering (mm) at solved['fast_time(_2)'][offcenter_idx1(2)].
            * offset_time: float
                * Offset time of the encoder signals induced by the offcentering.
                * Offset time is the delayed (advanced) timing of the encoder1 (2) in sec.

        """

        logger.info('Remove offcentering effect from hwp angle and overwrite')
        # Calculate offcentering from where the first reference slot was detected by the 2nd encoder.
        if solved["ref_indexes_1"][0] > self._num_edges/2-1:
            offcenter_idx1_start = int(solved["ref_indexes_1"][0]-self._num_edges/2)
            offcenter_idx2_start = int(solved["ref_indexes_2"][0])
        else:
            offcenter_idx1_start = int(solved["ref_indexes_1"][1]-self._num_edges/2)
            offcenter_idx2_start = int(solved["ref_indexes_2"][0])
        # Calculate offcentering to the end of the shorter encoder data.
        if len(solved["fast_time_1"][offcenter_idx1_start:]) > len(solved["fast_time_2"][offcenter_idx2_start:]):
            idx_length = len(solved["fast_time_2"][offcenter_idx2_start:])
        else:
            idx_length = len(solved["fast_time_1"][offcenter_idx1_start:])
        offcenter_idx1 = np.arange(
            offcenter_idx1_start, offcenter_idx1_start+idx_length-1)
        offcenter_idx2 = np.arange(
            offcenter_idx2_start, offcenter_idx2_start+idx_length-1)
        # Calculate the offset time of the encoders induced by the offcentering.
        offset_time = (solved["fast_time_1"][offcenter_idx1] -
                       solved["fast_time_2"][offcenter_idx2])/2
        # Calculate the offcentering (mm).
        period = (solved["fast_time_1"][offcenter_idx1+1] -
                  solved["fast_time_1"][offcenter_idx1])*self._num_edges
        # When data is extremely noisy, period gets zero and offcentering becomes nan
        period[period == 0] = np.median(period)
        offset_angle = 2 * np.pi * offset_time / period
        offcentering = np.tan(offset_angle) * self._encoder_disk_radius

        solved['offcenter_idx1'] = offcenter_idx1
        solved['offcenter_idx2'] = offcenter_idx2
        solved['offcentering'] = offcentering
        solved['offset_time'] = offset_time

        return

    def correct_offcentering(self, solved):
        """
        Correct the timing of solved['fast_time'] which is delayed (advanced) by the offcentering.

        Args
        -----
        solved: dict
            dict solved from template_subtraction
            {fast_time_1, angle_1, fast_time_2, angle_2, ...}
        offcentering: dict
            dict solved from eval_offcentering
            {offcenter_idx1, offcenter_idx2, offcentering, offset_time}

        Returns
        --------
        output: dict
            {fast_time, angle, fast_time_2, angle_2, ...}

        Notes
        ------
            * offcenter_idx1: int
                * index of the solved['fast_time_1'] for which offcentering is estimated.
            * offcenter_idx2: int
                * index of the solved['fast_time_2'] for which offcentering is estimated.
            * offcentering: float
                * Offcentering (mm) at solved['fast_time_1(2)'][offcenter_idx1(2)].
            * offset_time: float
                * Offset time of the encoder signals induced by the offcentering.
                * Offset time is the delayed (advanced) timing of the encoder1 (2) in sec.

        * We should allow to correct the offcentering by external input, since offcentering measurement is not always available.
        """

        # Skip the correction when the offcentering estimation doesn't exist.
        if 'offcentering' not in solved.keys():
            logger.warning(
                'Offcentering info does not exist. Offcentering correction is skipped.')
            return

        offcenter_idx1 = solved['offcenter_idx1']
        offcenter_idx2 = solved['offcenter_idx2']
        offset_time = solved['offset_time']

        solved['fast_time_raw_1'] = solved['fast_time_raw_1'][offcenter_idx1]
        solved['fast_time_raw_2'] = solved['fast_time_raw_2'][offcenter_idx2]
        solved['fast_time_1'] = solved['fast_time_1'][offcenter_idx1] - offset_time
        solved['fast_time_2'] = solved['fast_time_2'][offcenter_idx2] + offset_time
        solved['angle_1'] = solved['angle_1'][offcenter_idx1]
        solved['angle_2'] = solved['angle_2'][offcenter_idx2]

        return

    def write_solution(self, solved, output=None):
        """
        Output HWP angle + flags as SO HK g3 format

        Args
        -----
        solved: dict
          dict data from analyze
        output: str or None
          output path + file name, override config file

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

    def _set_empty_axes(self, aman, suffix=None):
        if suffix is None:
            aman.wrap_new('hwp_angle', shape=('samps', ), dtype=np.float64)
            aman.wrap('primary_encoder', 0)
            aman.wrap('version', 0)
            aman.wrap('pid_direction', 0)
            aman.wrap('offcenter_direction', 0)
            aman.wrap_new('offcenter', shape=(2,), dtype=np.float64)
            aman.wrap('sotodlib_version', sotodlib.__version__)
        else:
            aman.wrap_new('hwp_angle_ver1'+suffix,
                          shape=('samps', ), dtype=np.float64)
            aman.wrap_new('hwp_angle_ver2'+suffix,
                          shape=('samps', ), dtype=np.float64)
            aman.wrap_new('hwp_angle_ver3'+suffix,
                          shape=('samps', ), dtype=np.float64)
            aman.wrap_new('quad'+suffix, shape=('samps', ), dtype=int)
            aman.wrap('quad_direction'+suffix, 0)
            aman.wrap_new('stable'+suffix, shape=('samps', ), dtype=bool)
            aman.wrap_new('locked'+suffix, shape=('samps', ), dtype=bool)
            aman.wrap_new('hwp_rate'+suffix, shape=('samps', ), dtype=np.float16)
            aman.wrap_new('template'+suffix, shape=(self._num_edges, ), dtype=np.float64)
            aman.wrap_new('template_t'+suffix, shape=(self._num_edges, ), dtype=np.float64)
            aman.wrap_new('template_t_err'+suffix, shape=(self._num_edges, ), dtype=np.float64)
            aman.wrap_new('filled_flag'+suffix, shape=('samps', ), dtype=bool)
            aman.wrap('num_dropped_packets'+suffix, 0)
            aman.wrap('num_dropped_packets_irig'+suffix, 0)
            aman.wrap('num_glitches'+suffix, 0)
            aman.wrap('num_glitches_irig'+suffix, 0)
            aman.wrap('num_value_glitches'+suffix, 0)
            aman.wrap('num_value_glitches_irig'+suffix, 0)
            aman.wrap('num_dead_rots'+suffix, 0)
            aman.wrap('num_dropped_slits'+suffix, 0)
            aman.wrap('version'+suffix, 1)
            aman.wrap('logger'+suffix, 'Not set')
        return aman

    def _set_raw_axes(self, aman, data):
        """ Set raw encoder data in aman """
        for k, v in data.items():
            aman.wrap('raw_' + k, np.array(v))
        return aman

    def _load_raw_axes(self, aman, output, h5_address):
        """ Load raw encoder data from h5 """
        fileds, alias = self._key_formatting()
        data = {}
        f = h5py.File(output)
        for a in alias:
            if 'raw_' + a in f[h5_address].keys():
                v = f[h5_address]['raw_' + a][:]
                data[a] = v
        f.close()
        return data

    def _angle_interpolation(self, timestamp1, angle, timestamp2):
        """Linearly interpolate the angle to the timestamp of the detector readout.
        Fill outside the data range constant by the first and last values of the angle."""
        return np.interp(timestamp2, timestamp1, angle, left=angle[0], right=angle[-1])

    def _bool_interpolation(self, timestamp1, data, timestamp2, fill, round_option):
        """Linearly interpolate the boolean array. fill values by False outside of the
        data range
        Args
            fill: Outside of data range is filled by this value, 0 or 1
            round_option: round option, 'floor' or 'ceil'
        """
        interp = np.interp(timestamp2, timestamp1, data, left=fill, right=fill)
        if round_option == 'floor':
            interp = np.floor(interp)
        elif round_option == 'ceil':
            interp = np.ceil(interp)
        else:
            raise ValueError(f'round option {round_option} is not supported.')
        return interp.astype(bool)

    def set_data(self, tod, h5_filename=None):
        """
        Output HWP hk data as AxisManager format. The results are stored in HDF5 files.
        We save the copy of raw hwp encoder hk data into HDF5 file, to save time for
        re-calculating the hwp angle solutions.

        Args
        ----
        tod: AxisManager

        h5_filename:
            If this is not None, try to load raw encoder data from hdf5 file

        Notes
        -----
        Output file format

        - Raw output
            x is a number different for each data and each observation

            - raw_rising_edge_count_1/2: int (2, x)

            - raw_irig_time_1/2: float (2, x)

            - raw_counter_1/2: float (2, x)

            - raw_counter_index_1/2: int (2, x)

            - raw_irig_synch_pulse_clock_time_1/2: float (2, x)

            - raw_irig_synch_pulse_clock_counts_1/2: int (2, x)

            - raw_quad_1/2: bool (2, x)

            - raw_pid_direction: bool (2, x)
        """

        if len(tod.timestamps) == 0:
            logger.warning('Input data is empty.')
            return

        aman = sotodlib.core.AxisManager(tod.samps)
        start = int(tod.timestamps[0])-self._margin
        end = int(tod.timestamps[-1])+self._margin

        self.data = {}
        if h5_filename is not None:
            logger.info('Loading raw encoder data from h5')
            obs_id = tod.obs_info['obs_id']
            self.data = self._load_raw_axes(aman, h5_filename, obs_id)
        else:
            try:
                self.data = self.load_data(start, end)

            except Exception as e:
                logger.error(
                    f"Exception '{e}' thrown while loading HWP data. The specified encoder field is missing.")
                print(traceback.format_exc())

            finally:
                self._set_raw_axes(aman, self.data)

        return aman

    def make_solution(self, tod):
        """
        Output HWP angle, flags, metadata as AxisManager format
        The results are stored in HDF5 files. Since HWP angle solution HDF5 files are large,
        we automatically split into the new output files.

        Args
        ----
        tod: AxisManager

        Notes
        -----
        Output file format

        - Primary output

            - timestamp: float (samps,)
                SMuRF synched timestamp

            - hwp_angle: float (samps,)
                The latest version of the SMuRF synched HWP angle in radian.
                * ver1: HWP angle calculated from the raw encoder signal.
                * ver2: HWP angle after the template subtraction.
                * ver3: HWP angle after the template and off-centering subtraction.
                The field 'version' indicates which version this hwp_angle is.

            - primaty encoder: int
                This field indicates which encoder is used for hwp_angle, 1 or 2

            - version: int
                This field indicates the version of the HWP angle in hwp_angle.

        - Supplementary output
            Suffix _1/2 indicates the encoder_1 or encoder_2

            - version_1/2: int
                This field indicates the version of the HWP angle of each encoder.

            - hwp_angle_ver1/2/3_1/2: float (samps,)
                This field stores the ver1/2/3 angle data.

            - stable_1/2: bool (samps,)
                If non-zero, indicates the HWP spin state is known.
                i.e. it is either spinning at a measurable rate, or stationary.
                When this flag is non-zero, the hwp_rate field can be taken at face value.

            - locked_1/2: bool (samp,)
                If non-zero, indicates the HWP is spinning and the position solution is working.
                In this case one should find the hwp_angle populated in the fast data block.

            - hwp_rate_1/2: float (samps,)
                The "approximate" HWP spin rate, with sign, in revs / second.
                Use placeholder value of 0 for cases when not "locked".

            - logger_1/2: str
                Log message for angle calculation status
                'No HWP data', 'HWP data too short',
                'Angle calculation failed', 'Angle calculation succeeded'

            - filled_flag_1/2: bool (samps,)
                Array to indicate the data points that are filled due to packet drop.

            - quad_1/2: int (quad,)
                0 or 1 or -1. 0 means no data

            - template_1/2: float (1140,)
                Template of the non uniformity of hwp encoder plate

            - template_err_1/2: float (1140,)
                Error bar of template

            - offcenter: float (2,)
                - (average offcenter, std of offcenter) unit is (mm)

        - Rotation direction
            Rotation direction estimated by several methods.
            0 or 1 or -1. 0 means no data.

            - quad_direction_1/2: int
                Estimation by median encoder quadrature for each encoder

            - pid_direction: int
                Estimation by median pid controller commanded direction

            - offcenter_direction: int
                Estimation by the offcentering measured by the time offset between two encoders.

            - template_direction: int
                Estimation by the template of encoder plate.
                To be implemented.

            - scan_direction: int
                Estimation by scan synchronous modulation of rotation speed.
                To be implemented.

        """

        aman = sotodlib.core.AxisManager(tod.samps)
        aman.wrap_new('timestamps', ('samps', ))[:] = tod.timestamps
        self._set_empty_axes(aman)

        if 'pid_direction' in self.data.keys():
            pid_direction = np.nanmedian(self.data['pid_direction'][1])*2 - 1
            if pid_direction in [1, -1]:
                aman['pid_direction'] = pid_direction
            else:
                aman['pid_direction'] = 0

        solved = {}
        success = []

        # version 0
        # No data or angle calculation is failed
        for suffix in self._suffixes:
            logger.info('Start analyzing encoder'+suffix)
            self._set_empty_axes(aman, suffix)
            # load data
            if not 'counter' + suffix in self.data.keys():
                logger.warning('No HWP data in the specified timestamps.')
                aman['logger'+suffix] = 'No HWP data'
                success.append(False)
            try:
                solved.update(self.analyze(self.data, mod2pi=False, suffix=suffix))
            except Exception as e:
                logger.error(
                    f"Exception '{e}' thrown while calculating HWP angle. Angle calculation failed.")
                aman['logger'+suffix] = 'Angle calculation failed'
                success.append(False)
                print(traceback.format_exc())
            if len(solved) == 0 or ('fast_time'+suffix not in solved.keys()) or len(solved['fast_time'+suffix]) == 0:
                logger.info(
                    'No correct rotation data in the specified timestamps.')
                aman['logger'+suffix] = 'No HWP data'
                success.append(False)
            aman['logger'+suffix] = 'Angle calculation succeeded'
            success.append(True)

        # Correct ambiguity of references when angle solution is succeeded
        # and only one of the encoders have ambiguity
        ref_ambiguous = [isinstance(solved['angle' + suffix], tuple) for suffix in self._suffixes]
        if sum(ref_ambiguous) == 2:
            logger.error('Both encoders have ambiguity in references. Abort angle calculation')
            for suffix in self._suffixes:
                aman['logger'+suffix] += ', failed to correct ambiguous references'
            success = [False, False]
        if sum(ref_ambiguous) == 1 and sum(success) < 2:
            logger.error('Cannot correct ambiguity of references. Abort angle calculation')
            for suffix in self._suffixes:
                aman['logger'+suffix] += ', failed to correct ambiguous references'
            success = [False, False]
        if sum(ref_ambiguous) == 1 and sum(success) == 2:
            ambiguous_suffix = np.array(self._suffixes)[ref_ambiguous][0]
            good_suffix = np.array(self._suffixes)[np.logical_not(ref_ambiguous)][0]
            good_angle = np.interp(solved['fast_time' + ambiguous_suffix],
                         solved['fast_time' + good_suffix], solved['angle' + good_suffix])
            for i, ambiguous_angle in enumerate(solved['angle' + ambiguous_suffix]):
                median_diff = np.median(ambiguous_angle - good_angle)
                if abs(abs(median_diff) - np.pi) < 2 * np.arctan(5 / self._encoder_disk_radius):
                    solved['angle' + ambiguous_suffix] = solved['angle' + ambiguous_suffix][i]
                    solved['ref_indexes' + ambiguous_suffix] = solved['ref_indexes' + ambiguous_suffix][i]
                    logger.warning(f'Corrected ambiguous references of encoder{ambiguous_suffix}')
                    break
            if isinstance(solved['angle' + ambiguous_suffix], tuple):
                logger.error('Cannot correct ambiguity of references. Abort angle calculation')
                for suffix in self._suffixes:
                    aman['logger'+suffix] += ', failed to correct ambiguous references'
                success = [False, False]

        # version 1
        # angle calculation succeeded
        for i, suffix in enumerate(self._suffixes):
            if not success[i]:
                continue
            aman['version'+suffix] = 1
            aman['stable'+suffix] = self._bool_interpolation(
                solved['slow_time'+suffix], solved['stable'+suffix], tod.timestamps, 0, 'floor')
            aman['locked'+suffix] = self._bool_interpolation(
                solved['slow_time'+suffix], solved['locked'+suffix], tod.timestamps, 0, 'floor')
            aman['hwp_rate'+suffix] = np.interp(
                tod.timestamps, solved['slow_time'+suffix], solved['hwp_rate'+suffix], left=0, right=0)

            quad = scipy.interpolate.interp1d(
                solved['fast_time'+suffix], solved['quad'+suffix], kind='linear', fill_value='extrapolate')(tod.timestamps)
            aman['quad'+suffix] = np.array([1 if q else -1 for q in quad])
            aman['quad_direction'+suffix] = np.nanmedian(aman['quad'+suffix])

            filled_flag = np.zeros_like(solved['fast_time'+suffix], dtype=bool)
            filled_flag[solved['filled_flag'+suffix]] = 1
            aman['filled_flag'+suffix] = self._bool_interpolation(
                solved['fast_time'+suffix], filled_flag, tod.timestamps, 1, 'ceil')
            aman['hwp_angle_ver1'+suffix] = np.mod(self._angle_interpolation(
                solved['fast_time'+suffix], solved['angle'+suffix], tod.timestamps), 2*np.pi)
            aman['num_dropped_packets'+suffix] = solved['num_dropped_packets'+suffix]
            aman['num_dropped_packets_irig'+suffix] = solved['num_dropped_packets_irig'+suffix]
            aman['num_glitches'+suffix] = solved['num_glitches'+suffix]
            aman['num_glitches_irig'+suffix] = solved['num_glitches_irig'+suffix]
            aman['num_value_glitches'+suffix] = solved['num_value_glitches'+suffix]
            aman['num_value_glitches_irig'+suffix] = solved['num_value_glitches_irig'+suffix]
            aman['num_dead_rots'+suffix] = solved['num_dead_rots'+suffix]
            aman['num_dropped_slits'+suffix] = solved['num_dropped_slits'+suffix]

        # version 2
        # calculate template subtracted angle
        for i, suffix in enumerate(self._suffixes):
            if not success[i]:
                continue
            try:
                self.template_subtraction(solved, suffix=suffix)
                aman['hwp_angle_ver2'+suffix] = np.mod(self._angle_interpolation(
                    solved['fast_time'+suffix], solved['angle'+suffix], tod.timestamps), 2*np.pi)
                aman['version'+suffix] = 2
                aman['template'+suffix] = solved['template'+suffix]
                aman['template_t'+suffix] = solved['template_t'+suffix]
                aman['template_t_err'+suffix] = solved['template_t_err'+suffix]
            except Exception as e:
                logger.error(
                    f"Exception '{e}' thrown while the template subtraction.")
                print(traceback.format_exc())

        # version 3
        # calculate off-centering corrected angle
        if (aman.version_1 == 2 and aman.version_2 == 2):
            try:
                self.eval_offcentering(solved)
                self.correct_offcentering(solved)
                for suffix in self._suffixes:
                    aman['hwp_angle_ver3'+suffix] = np.mod(self._angle_interpolation(
                        solved['fast_time'+suffix], solved['angle'+suffix], tod.timestamps), 2*np.pi)
                    aman['version'+suffix] = 3
                aman['offcenter'] = np.array([np.average(solved['offcentering']), np.std(solved['offcentering'])])
                aman.offcenter_direction = np.sign(aman['offcenter'][0])
            except Exception as e:
                logger.error(
                    f"Exception '{e}' thrown while the off-centering correction.")
                print(traceback.format_exc())
        else:
            logger.warning(
                'Offcentering calculation is only available when two encoders are operating. Skipped.')

        # make the hwp angle solution with highest version as hwp_angle
        highest_version = int(np.max([aman.version_1, aman.version_2]))
        primary_encoder = int(np.argmax([aman.version_1, aman.version_2]) + 1)
        logger.info(f'Save hwp_angle_ver{highest_version}_{primary_encoder} as hwp_angle')
        aman.hwp_angle = aman[f'hwp_angle_ver{highest_version}_{primary_encoder}']
        aman.primary_encoder = primary_encoder
        aman.version = highest_version

        return aman

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

        # counter: BBB counter values for encoder signal edges
        self._encd_clk = counter

        # counter_index: index numbers for detected edges by BBB
        self._encd_cnt = counter_idx

        # irig_time: decoded time in second since the unix epoch
        self._irig_time = irig_time

        # rising_edge_count: BBB clcok count values for the IRIG on-time
        # reference marker risinge edge
        self._rising_edge = rising_edge

        # reference slit indexes
        self._ref_indexes = []
        # temporary placeholder for full references
        self._ref_clk_tmp = None

        # quad: quadrature signal to determine rotation direction
        self._quad_time = quad_time
        self._quad = quad

        self._quad_corrected = []

        # return arrays
        self._time = []
        self._angle = []

        # metadata of packet drop
        self._num_dropped_pkts = 0
        self._num_dropped_pkts_irig = 0

        self._num_dropped_slits = 0

        # glitch statistics
        self._num_glitches = 0
        self._num_value_glitches = 0
        self._num_glitches_irig = 0
        self._num_value_glitches_irig = 0
        self._num_dead_rots = 0

        # if no data, skip analysis
        if len(self._encd_clk) == 0:
            return [], []

        # check duplication in data
        self._duplication_check()

        # check IRIG timing quality
        self._irig_quality_check()

        self._fix_irig_desync()

        # check 32 bit internal counter overflow glitch
        self._process_counter_overflow_glitch()

        # treat counter index reset due to agent reboot
        self._process_counter_index_reset()

        # check packet drop
        self._encoder_packet_sort()
        self._fill_dropped_packets()

        # reference finding
        _status_find_ref = self._find_refs()
        if not _status_find_ref:
            return [], []

        # correct references with glitches
        self._correct_bad_refs()

        # reference filling
        if fast:
            self._fill_refs_fast()
        else:
            self._fill_refs()

        # If encoder slit is missing, temporarily "correct" refereces
        # to fix glitches. ambiguity of references needs to be corrected
        # by comparing two encoders
        for i in range(1, 3):
            if np.median(np.diff(self._ref_indexes[::i + 1])) - self._num_edges < 3:
                logger.warning(f'Detected {i} missing slit, '
                               'ambiguous references needs to be corrected')
                self._ref_clk_tmp = self._ref_clk
                self._num_dropped_slits = i

                refs = self._find_true_refs()
                self._ref_indexes = refs[0]
                break

        # glitch removal
        self._fix_datapoint_glitches()
        self._fix_value_glitches()

        # assign IRIG synched timestamp
        self._time = scipy.interpolate.interp1d(
            self._rising_edge,
            self._irig_time,
            kind='linear',
            fill_value='extrapolate')(self._encd_clk)

        self._quad_corrected = self._quad_form(
            scipy.interpolate.interp1d(
                self._quad_time,
                self._quad_form(self._quad),
                kind='linear',
                fill_value='extrapolate')(self._time))

        # If references are ambiguous, list up all possible patterns of
        # reference indexes and angles
        if self._num_dropped_slits > 0:
            _angle = []
            _ref_indexes = []
            # restore all the reference indexes
            self._ref_indexes = np.where(np.isin(self._encd_clk, self._ref_clk_tmp))[0]
            refs = self._find_true_refs()
            for ref in refs:
                self._ref_indexes = ref
                self._generate_sub_data(ref_clk=True)
                self._calc_angle_linear(mod2pi)
                _angle.append(self._angle)
                _ref_indexes.append(self._ref_indexes)
            self._angle = tuple(_angle)
            self._ref_indexes = tuple(_ref_indexes)

        else:
            self._calc_angle_linear(mod2pi)

        logger.debug('qualitycheck')
        logger.debug('_time:        ' + str(len(self._time)))
        logger.debug('_angle:       ' + str(len(self._angle)))
        logger.debug('_encd_cnt:    ' + str(len(self._encd_cnt)))
        logger.debug('_encd_clk:    ' + str(len(self._encd_clk)))
        logger.debug('_ref_cnt:     ' + str(len(self._ref_cnt)))
        logger.debug('_ref_indexes: ' + str(len(self._ref_indexes)))

        if isinstance(self._angle, tuple):
            if np.any([len(self._time) != len(angle) for angle in self._angle]):
                logger.warning('Failed to calculate hwp angle!')
                return [], []
        else:
            if len(self._time) != len(self._angle):
                logger.warning('Failed to calculate hwp angle!')
                return [], []
        logger.info('hwp angle calculation is finished.')
        return self._time, self._angle

    def _find_refs(self):
        """ Find reference slits """
        # Function to find the mean of an array without outliers
        def _mean(arr):
            high = np.percentile(arr, 90)
            low = np.percentile(arr, 10)
            return np.mean(arr[(arr < high) & (arr > low)])
        # Function to find the mean of an array larger than the median without outliers
        def _mean_high(arr):
            high = np.percentile(arr, 90)
            low = np.percentile(arr, 55)
            return np.mean(arr[(arr < high) & (arr > low)])

        # Generate self._encd_diff
        self._generate_sub_data(ref_clk=False)

        # Find the instantaneous average encoder clk spacing between datapoints
        rot_spacing = np.arange(0, len(self._encd_diff), 100*(self._num_edges - 2))
        diff_split = np.split(self._encd_diff, rot_spacing[1:])
        slit_dist = np.concatenate([np.ones(len(_diff)) * _mean(_diff) for _diff in diff_split])
        slit_dist_high = np.concatenate([np.ones(len(_diff)) * _mean_high(_diff) for _diff in diff_split])

        # Generate thresholds to distinguish references and glitched references
        ref_hi_cond = (self._ref_edges + 1) * slit_dist * (1 + self._ref_range)
        ref_lo_cond = (self._ref_edges + 1) * slit_dist * (1 - self._ref_range)
        ref_gl_cond = slit_dist_high * (1 + self._ref_range)

        # Normal references
        ref_ind = np.where((self._encd_diff > ref_lo_cond) & (self._encd_diff < ref_hi_cond))[0]
        # Potential glitched references
        ref_ind_glitch = np.where((self._encd_diff < ref_lo_cond) & (self._encd_diff > ref_gl_cond))[0]

        # Find and add glitched references to normal references
        used_ind_pairs = []
        def _before(x, ind):
            return abs(x - ind) + 1e8*(1 + np.sign(x - ind))
        def _after(x, ind):
            return abs(x - ind) + 1e8*(1 - np.sign(x - ind))

        if len(ref_ind) > 0:
            for ind in ref_ind_glitch:
                # Check that there is sufficient space before and after the glitched reference
                before = ref_ind[np.argmin(_before(ref_ind, ind))]
                after = ref_ind[np.argmin(_after(ref_ind, ind))]
                if all(np.array([after - ind, ind - before]) > 0.9 * self._num_edges):
                    if (before, after) in used_ind_pairs:
                        continue
                    used_ind_pairs.append((before, after))
                    ref_ind = np.append(ref_ind, ind)

        # Define the reference slit line to be the line before
        # the two "missing" lines
        # Store the count and clock values of the reference lines
        self._ref_indexes = np.sort(ref_ind)
        if len(self._ref_indexes) == 0:
            if len(self._encd_diff) < self._num_edges:
                logger.warning(
                    'cannot find reference points, # of data is less than # of slit')
            else:
                logger.warning(
                    'cannot find reference points, please adjust parameters!')
            return False

        # check quality of ref_indexes
        number_of_bad_refs = np.sum(np.diff(self._ref_indexes) != self._num_edges - 2)
        if number_of_bad_refs > 0:
            logger.warning(f'There are {number_of_bad_refs} bad ref indexes')

        self._ref_clk = self._encd_clk[self._ref_indexes]
        self._ref_cnt = self._encd_cnt[self._ref_indexes]
        logger.debug('found {} reference points'.format(
            len(self._ref_indexes)))

        return True

    def _generate_sub_data(self, ref_clk=True):
        """ Re-generates datafields derived from _encd_clk, _encd_cnt, and _ref_indexes """
        self._encd_diff = np.ediff1d(self._encd_clk, to_begin=self._encd_clk[1]-self._encd_clk[0])
        # Only generate if self._ref_indexes exists
        if ref_clk:
            self._ref_clk = np.take(self._encd_clk, self._ref_indexes)
            self._ref_cnt = np.take(self._encd_cnt, self._ref_indexes)

        return True

    def _correct_bad_refs(self, plot=False, **kwargs):
        """ Corrects glitches in reference slits """
        # Generate statistics of the references
        self._ref_mean = np.mean(self._encd_diff[self._ref_indexes])
        self._ref_std = np.std(self._encd_diff[self._ref_indexes])

        # Find the glitched reference correction size
        ref_corr_size = np.zeros(len(self._ref_indexes))
        for index, ind in enumerate(self._ref_indexes):
            # Only correct references 5 stds away from the mean
            if self._encd_diff[ind] < self._ref_mean - 5*self._ref_std:
                # Find the expected correction value and step size
                # in the forward direction
                cvf, csf = self._find_corr_size(ind, 1)
                # Find the expected correction value and step size
                # in the reverse direction
                cvr, csr = self._find_corr_size(ind, -1)
                # Return the step size which corrects better
                ref_corr_size[index] = csf if cvf < cvr else -csr

        # Mask out points that are glitched
        encd_mask = np.ones(len(self._encd_clk))
        for index, ind in enumerate(np.copy(self._ref_indexes)):
            corr = int(ref_corr_size[index])
            if corr == 0:
                continue
            elif corr > 0:
                # Create glitch mask and edit other reference
                encd_mask[ind:ind + corr] *= 0
                self._ref_indexes[index + 1:] -= corr
            else:
                # Create glitch mask and edit other reference
                encd_mask[ind + corr:ind] *= 0
                self._ref_indexes[index:] += corr

        # Apply mask
        self._encd_clk = self._encd_clk[encd_mask.astype(bool)]
        for ind, value in enumerate(encd_mask.astype(bool)):
            if not value:
                self._encd_cnt[ind:] -= 1
        self._encd_cnt = self._encd_cnt[encd_mask.astype(bool)]
        # Re-generate various arrays
        self._generate_sub_data()

        return True

    def _find_corr_size(self, ind, direction, glitch_steps = 10):
        """ Finds how many datapoints to sum over in order to fix the glitch """
        # Residual of summing up differences
        running_avg = self._encd_diff[ind] - self._ref_mean
        # Sum in a direction until the resisual starts increasing
        for step in np.arange(1, 1 + glitch_steps):
            # Make sure the next step index is reasonable
            step_index = min(max(0, int(ind + direction*step)), len(self._encd_diff) - 1)
            new_avg = running_avg + self._encd_diff[step_index]
            if abs(new_avg) < abs(running_avg):
                running_avg = new_avg
            else:
                # Return the minimum residual and the number of steps to reach it
                return abs(running_avg), step - 1
        else:
            logger.warning('Error: Reference optimization took too long')
            return self._ref_mean, 0

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
            return True
        # Split the encoder clk/cnt arrays based on the references
        encd_clk_split = np.split(self._encd_clk, self._ref_indexes)
        filled_clks = [encd_clk_split[0]]
        encd_cnt_split = np.split(self._encd_cnt, self._ref_indexes)
        filled_cnts = [encd_cnt_split[0]]

        # Loop over every split
        for i, ref_ind in enumerate(np.copy(self._ref_indexes)):
            start_clk = self._encd_clk[ref_ind-1]
            end_clk = self._encd_clk[ref_ind]

            # Insert extra clks to fill the reference
            filled_clks.append([int(start_clk+(end_clk-start_clk)/3),
                                int(start_clk+(end_clk-start_clk)*2/3)])
            filled_clks.append(encd_clk_split[i+1])

            # Adjust the cnt for added clks
            filled_cnts.append(2*i + encd_cnt_split[i+1][0] + np.array([0, 1]))
            filled_cnts.append(2*(i + 1) + encd_cnt_split[i+1])

            # Adjust the ref for added clks
            self._ref_indexes[i+1:] += self._ref_edges

        # Convert split arrays back into single array
        self._encd_clk = np.array([clk for entry in filled_clks for clk in entry])
        self._encd_cnt = np.array([cnt for entry in filled_cnts for cnt in entry])
        self._generate_sub_data()

        return True

    def _find_true_refs(self):
        # If encoder slit is missing, true references and fake references are mixed
        # find true references from the distance between references
        refs = [[]] * (self._num_dropped_slits + 1)
        for i in range(self._num_dropped_slits + 1):
            refs[i] = [self._ref_indexes[i]]
            for ref_i in self._ref_indexes[i + 1:]:
                if ref_i - refs[i][-1] >= self._num_edges and \
                        ref_i - refs[i][-1] < self._num_edges + 10:
                    refs[i].append(ref_i)
        # sort by length, very short ones are not true references
        refs = sorted(refs, key=len, reverse=True)
        return refs

    def _fix_datapoint_glitches(self):
        """ Removes glitches that add encoder datapoints """
        # Find how many extra datapoints there are per revolution
        self._glitches = np.ediff1d(self._ref_indexes, to_end=self._num_edges) - self._num_edges
        encd_diff_split = np.split(self._encd_diff, self._ref_indexes)

        bad_fills = []
        dead_rots = []
        total_mask = [np.ones(len(encd_diff_split[0]))]
        # Loop through every rotation and find which points to mask out
        for i, diff in enumerate(encd_diff_split[1:]):
            # Only process rotations which have a glitch
            if self._glitches[i] != 0:
                # Assuming the signal has some static duty cycle, the clk difference between
                # two points follows a bimodal distribution. These lines find the average
                # clk diff peaks of that distribution as well as whether the rotation starts
                # with a high or low clock difference
                high, low = self._find_high_low_values(diff)
                start_high = self._find_rot_start_type(diff)

                # Generate a glitch mask for this rotation
                result, mask = self._glitch_mask(diff, high, low, start_high)
                # If the mask could not be generated, check if the duty cycle was incorrectly calculated
                if not result:
                    result, mask = self._glitch_mask(diff, high, low, not start_high)
                # If the mask could still not be generated, check if it was because of a packet drop or remove the rotation
                if not result:
                    start_clk = self._ref_clk[i]
                    end_clk = self._ref_clk[i+1]
                    # If the rotation had a packet drop
                    if np.any([edges[0] < end_clk and start_clk < edges[1] for edges in self._edges_dropped_pkts]):
                        # Fill the rotation with data from an adjacent rotation
                        if i == 0:
                            gap_clk = self._encd_clk[self._ref_indexes[i+1]:self._ref_indexes[i+2]] \
                                    - self._encd_clk[self._ref_indexes[i+1]] + self._encd_clk[self._ref_indexes[i]]
                        else:
                            if self._ref_indexes[i - 1] in bad_fills:
                                bad_fills.append(self._ref_indexes[i])
                            gap_clk = self._encd_clk[(self._ref_clk[i-1] <= self._encd_clk) &
                                                     (self._encd_clk < self._ref_clk[i])]
                            gap_clk = gap_clk[total_mask[-1]] - self._ref_clk[i-1] + self._ref_clk[i]

                        corr_factor = (self._ref_clk[i+1] - self._ref_clk[i]) * \
                                      (self._num_edges - 1) / (gap_clk[-1] - gap_clk[0]) / (self._num_edges)
                        gap_clk = corr_factor * (gap_clk - gap_clk[0]) + gap_clk[0]

                        fill_clk = np.zeros(len(diff))
                        fill_clk[:self._num_edges] = gap_clk
                        self._encd_clk[(self._ref_clk[i] <= self._encd_clk) &
                                       (self._encd_clk < self._ref_clk[i+1])] = fill_clk
                        mask = np.full(len(diff), False)
                        mask[:self._num_edges] = np.logical_not(mask[:self._num_edges])
                    else:
                        logger.debug(i, ' dead rots index')
                        mask = np.full(len(diff), False)
                        dead_rots.append(i)

                total_mask.append(mask)

                # Find how many glitches were removed
                num_glitches = len(mask) - np.sum(mask)
                self._ref_indexes[i+1:] -= num_glitches
                logger.debug(f'{num_glitches}, {np.where(~np.array(mask))[0]}')
            elif min(diff) < 0.1*np.median(diff):
                # Sometimes a dropped packet gets filled with glitched data
                bad_fills.append(self._ref_indexes[i])
                total_mask.append(np.ones(len(diff), dtype=bool))
            else:
                total_mask.append(np.ones(len(diff), dtype=bool))

        # Join the individual rotation masks into a single overall mask
        total_mask = np.array([bool(mask) for entry in total_mask for mask in entry])
        self._num_glitches = int(sum(~total_mask))
        if self._num_glitches > 0:
            logger.warning(f'{self._num_glitches} glitches are removed')
        if len(dead_rots) > 0:
            self._num_dead_rots = len(dead_rots)
            logger.warning(f'Could not remove glitches from {len(dead_rots)} rotations')
        self._encd_clk = self._encd_clk[total_mask]
        self._encd_cnt -= np.cumsum(np.logical_not(total_mask).astype(int))
        self._encd_cnt = self._encd_cnt[total_mask]
        self._ref_indexes = np.delete(self._ref_indexes, dead_rots)

        # Correct for packet drop fills which have glitches
        for ref in bad_fills:
            before_ref = max(self._ref_indexes[self._ref_indexes < ref])
            after_ref = min(self._ref_indexes[self._ref_indexes > ref])
            fill_ref = before_ref if before_ref else after_ref
            fill_values = (self._encd_clk[ref + self._num_edges] - self._encd_clk[ref]) * \
                          (self._encd_clk[fill_ref:fill_ref + self._num_edges] - self._encd_clk[fill_ref]) / \
                          (self._encd_clk[fill_ref + self._num_edges] - self._encd_clk[fill_ref]) + \
                          self._encd_clk[ref]
            self._encd_clk[ref:ref + self._num_edges] = fill_values

        self._generate_sub_data()

        return True

    def _find_high_low_values(self, arr):
        """ Finds the two peaks of an array with a bimodal distribution """
        # This function leads to warning of empty array when arr is very short
        med = np.median(arr[arr > 0.3*np.median(arr)])
        high = np.median(arr[arr > med])
        low = np.median(arr[np.logical_and(arr < med, arr > 0.3*med)])

        return high, low

    def _find_rot_start_type(self, arr):
        """ Finds the starting pattern of an array with a bimodal distribution """
        even = np.median(np.diff(arr)[::2][:20])
        odd = np.median(np.diff(arr)[1::2][:20])

        return True if even < odd else False

    def _glitch_mask(self, diffs, high, low, start):
        """
        Generates a mask of 'good' points in an array which is expected to follow
        a bimodal distribution
        """
        return_mask = [True, True]
        toggle = not start
        diff_sum = 0
        prev_res = 0
        # Iterate over ever point in the array, checking the rolling sum
        # against what it is expected to be. If the residual doesn't follow
        # the expected pattern, mask that point as a glitch
        for ind, diff in enumerate(diffs[2:]):
            diff_sum += diff
            res = abs(high-diff_sum) if toggle else abs(low-diff_sum)
            if prev_res > res:
                diff_forward = diff_sum + diffs[ind+3] if len(diffs) > ind + 3 else diff_sum
                comp = 2*low + high if toggle else low + 2*high
                if abs(comp-diff_forward) > abs(high+low-diff_forward):
                    return_mask.append(False)
                    prev_res = res
                    continue

            return_mask.append(True)
            diff_sum = diff
            toggle = not toggle
            prev_res = abs(high-diff_sum) if toggle else abs(low-diff_sum)
        else:
            return_mask.append(True)
            return_mask = return_mask[1:]

        if np.sum(return_mask) == self._num_edges + 1 and diffs[-1] < 0.3 * low:
            return_mask[-1] = False

        # Check that the number of valid points is what we expect in one rotation
        if np.sum(return_mask) == self._num_edges:
            return True, return_mask
        else:
            return False, None

    def _fix_value_glitches(self):
        """ Removes glitches that change the encoder value """
        diff_matrix = np.split(self._encd_diff, self._ref_indexes)
        # Find the average value in each rotation
        norm_matrix = np.array([[np.mean(diff[1:])] for diff in diff_matrix])
        # Average rotation template
        template = np.median(diff_matrix[1:-1]/norm_matrix[1:-1], axis=0)
        expectation = (norm_matrix*template).flatten()

        # Handle the first and last rotations
        cut_start = self._num_edges - len(diff_matrix[0])
        cut_end = len(diff_matrix[-1]) - self._num_edges if len(diff_matrix[-1]) != self._num_edges else None
        expectation = expectation[cut_start:cut_end]

        # Find which encd values differ from expectation in a way that
        # looks like a glitch
        error = self._encd_diff - expectation
        for i in self._ref_indexes[:-1]:
            for j in range(self._num_edges):
                dist = abs(error[i+j]) - 0.01*expectation[i+j]
                if dist > 0 and dist < expectation[i+j]:
                    dist_inds = 3 if j == self._num_edges - 1 else 1
                    dist_sign = np.sign(error[i+j])

                    try:
                        error[i+j] -= dist_sign*dist
                        error[i+j+1:i+j+1+dist_inds] += dist_sign*dist/dist_inds
                        self._num_value_glitches += 1
                    except IndexError:
                        pass

        # Recreate the encoder clock after accounting for value glitches
        self._encd_clk = self._encd_clk + np.cumsum(expectation + error - self._encd_diff)
        self._generate_sub_data()

        return True

    def _calc_angle_linear(self, mod2pi=True):
        """ Calculate hwp angle of encoder counters for each revolution """
        self._encd_cnt_split = np.split(self._encd_cnt, self._ref_indexes)
        angle_first_revolution = (self._encd_cnt_split[0] - self._ref_cnt[0]) * \
            (2 * np.pi / self._num_edges)
        angle_last_revolution = (self._encd_cnt_split[-1] - self._ref_cnt[-1]) * \
            (2 * np.pi / self._num_edges) + (len(self._ref_cnt) - 1) * 2 * np.pi
        self._angle = np.concatenate(
            [(self._encd_cnt_split[i] - self._ref_cnt[i]) *
             (2 * np.pi /np.diff(self._ref_indexes)[i - 1]) + i * 2 * np.pi
             for i in range(1, len(self._encd_cnt_split) - 1)]
        )
        self._angle = np.concatenate(
            [angle_first_revolution, self._angle.flatten(), angle_last_revolution])

        if mod2pi:
            self._angle = self._angle % (2 * np.pi)
        return

    def _duplication_check(self):
        """ Check the duplication in hk data and remove it """
        unique_array, unique_index = np.unique(
            self._encd_cnt, return_index=True)
        if len(unique_array) != len(self._encd_cnt):
            logger.warning(
                'Duplication is found in encoder data, performing correction.')
            self._encd_cnt = unique_array
            self._encd_clk = self._encd_clk[unique_index]
        unique_array, unique_index = np.unique(
            self._rising_edge, return_index=True)
        if len(unique_array) != len(self._rising_edge):
            logger.warning(
                'Duplication is found in IRIG data, performing correction.')
            self._rising_edge = unique_array
            self._irig_time = self._irig_time[unique_index]

    def _irig_quality_check(self):
        """ IRIG timing quality check """
        idx = np.where(np.diff(self._irig_time) == 1)[0]
        if self._irig_type == 1:
            idx = np.where(np.isclose(np.diff(self._irig_time),
                           np.full(len(self._irig_time)-1, 0.1)))[0]
        if len(self._irig_time) - 1 == len(idx):
            return

        # check packet drop or data point glitches
        elif len(self._irig_time) > len(idx) and len(idx) > 0:
            self._num_glitches_irig = np.sum(np.diff(self._irig_time) < 1)
            if self._num_glitches_irig > 0:
                logger.warning(f'{self._num_glitches_irig} additional irig time is detected')
            self._num_dropped_pkts_irig = np.sum(np.diff(self._irig_time) > 1)
            if self._num_dropped_pkts_irig > 0:
                logger.warning(f'{self._num_dropped_pkts_irig} irig packet drops is detected')
            if np.any(np.diff(self._irig_time) > 5):
                logger.warning(
                    'a part of the IRIG time is incorrect, performing the correction process...')
            self._irig_time = self._irig_time[idx]
            self._rising_edge = self._rising_edge[idx]
            logger.warning('deleted wrong irig_time, indices: ' +
                           str(np.where(np.diff(self._irig_time) != 1)[0]))
        else:
            self._irig_time = np.array([])
            self._rising_edge = np.array([])
            return

        # check value glitches of irig rising edges
        def min_distance_clk(time, interval):
            mod = np.mod(time, interval)
            return np.min([mod, interval - mod], axis=0)

        bbb_clk = np.diff(self._rising_edge)
        avg_bbb_clk = np.median(bbb_clk)
        self._num_value_glitches_irig = np.sum(min_distance_clk(bbb_clk, avg_bbb_clk) >= 1e5)
        if self._num_value_glitches_irig > 0:
            logger.warning(f'{self._num_value_glitches_irig} glitched irig_time is detected')
            idx = np.where(min_distance_clk(bbb_clk, avg_bbb_clk) < 1e5)[0]
            if len(self._irig_time) > len(idx) and len(idx) > 0:
                self._irig_time = self._irig_time[idx]
                self._rising_edge = self._rising_edge[idx]

    def _fix_irig_desync(self):
        """ Fix IRIG desynchronization by adding constant time offset """
        if self._irig_desync is None:
            return

        for t0, t1, dt in self._irig_desync:
            desynced = (t0 <= self._irig_time) & (self._irig_time <= t1)
            if np.any(desynced):
                logger.warning('irig time has known desynchronization, apply correction')
                self._irig_time[desynced] -= dt

    def _process_counter_overflow_glitch(self):
        """
        Treat glitches due to 32 bit internal counter overflow
        We suspect that this is a glitch caused by the very occasional failure of the encoder counter overflow correction
        due to latency or other problems on the pc running the encoder agent.
        """
        idx = np.where((np.diff(self._encd_clk)>=2**32-1) & (np.diff(self._encd_clk)<2**32+1e6))[0] + 1
        if len(idx) > 0:
            logger.warning(f'{len(idx)} counter overflow glitches are found, perform correction.')
        for i in idx:
            self._encd_clk[i] -= 2**32

    def _process_counter_index_reset(self):
        """ Treat counter index reset due to agent reboot """
        idx = np.where(np.diff(self._encd_cnt) < -1e4)[0] + 1
        if len(idx) > 0:
            logger.warning(f'{len(idx)} counter resets are found, perform correction.')
        for i in idx:
            self._encd_cnt[i:] = self._encd_cnt[i:] + abs(np.diff(self._encd_cnt)[i-1]) + 1

    def _fill_dropped_packets(self):
        """ Estimate the number of dropped packets """
        cnt_diff = np.diff(self._encd_cnt)
        dropped_samples = np.sum(cnt_diff[cnt_diff >= self._pkt_size] - 1)
        self._num_dropped_pkts = dropped_samples // self._pkt_size
        if self._num_dropped_pkts > 0:
            logger.warning('{} dropped packets are found, performing fill process'.format(
                self._num_dropped_pkts))

        self._edges_dropped_pkts = []
        for _ in np.where(np.diff(self._encd_cnt) > 1)[0]:
            index = np.where(np.diff(self._encd_cnt) > 1)[0][0]
            gap = int(np.diff(self._encd_cnt)[index]) - 1
            # Fill dropped counters with counters one before or one after rotation.
            # This filling method works even when the reference slot counter is dropped.
            rot_needed = int(np.ceil(gap/self._edges_per_rev)) + 1
            self._edges_dropped_pkts.append((self._encd_clk[index], self._encd_clk[index+1]))

            if index - rot_needed*self._edges_per_rev >= 0:
                rot_before_index = index - rot_needed*self._edges_per_rev
                gap_clk = self._encd_clk[rot_before_index:rot_before_index + gap + 2] \
                        - self._encd_clk[rot_before_index] + self._encd_clk[index]
                corr_factor = (self._encd_clk[index + 1] - self._encd_clk[index])/(gap_clk[-1] - gap_clk[0])
                gap_clk = (corr_factor*(gap_clk - gap_clk[0]) + gap_clk[0])[1:-1]
            else:
                rot_after_index = index + rot_needed*self._edges_per_rev + 1
                gap_clk = self._encd_clk[rot_after_index - gap - 1:rot_after_index + 1] \
                        - self._encd_clk[rot_after_index] + self._encd_clk[index + 1]
                corr_factor = (self._encd_clk[index + 1] - self._encd_clk[index])/(gap_clk[-1] - gap_clk[0])
                gap_clk = (corr_factor*(gap_clk - gap_clk[-1]) + gap_clk[-1])[1:-1]

            gap_cnt = np.arange(self._encd_cnt[index] + 1, self._encd_cnt[index+1])
            self._encd_cnt = np.insert(self._encd_cnt, index + 1, gap_cnt)
            self._encd_clk = np.insert(self._encd_clk, index + 1, gap_clk)

        self._edges_dropped_pkts = np.array(self._edges_dropped_pkts)

        return True

    def _encoder_packet_sort(self):
        cnt_diff = np.diff(self._encd_cnt)
        if np.any(cnt_diff != 1):
            logger.debug(
                'a part of the counter is incorrect')
            if np.any(cnt_diff < 0):
                if 1 - self._pkt_size in cnt_diff:
                    logger.warning(
                        'Packet flip found, performing sort process')
                idx = np.argsort(self._encd_cnt)
                self._encd_clk = self._encd_clk[idx]
            else:
                logger.warning('Packet drop exists')
        else:
            logger.debug('no need to fix encoder index')
        return

    def _quad_form(self, quad):
        # bit process
        quad[(quad >= 0.5)] = 1
        quad[(quad < 0.5)] = 0
        offset = 0
        outlier_count = 0
        for quad_split in np.array_split(quad, 1 + np.floor(len(quad) / 100)):
            if quad_split.mean() > 0.1 and quad_split.mean() < 0.9:
                for j in range(len(quad_split)):
                    quad[j + offset] = int(quad_split.mean() + 0.5)
                offset += len(quad_split)
                continue

            outlier = np.argwhere(
                np.abs(quad_split.mean() - quad_split) > 0.5).flatten()
            if len(outlier) > 5:
                outlier_count += 1
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
        if outlier_count > 0:
            logger.warning("flipping quad was corrected by mean value "
                           f"in {outlier_count} sections.")

        return quad

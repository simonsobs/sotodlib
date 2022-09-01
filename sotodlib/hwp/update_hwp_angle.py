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
# logging.basicConfig(level=logging.DEBUG)


class update_hwp_angle():

    def __init__(self, config_file=None):
        """
        Class to manage L2 HK data into HWP angle g3.

        Args
        -----
        config_file: str
            path to config yaml file
            yaml file should include below info.
            -----------------------
            verbose
            data_dir
            field_instance
            field_list
            packet_size
            num_edges
            ref_edges
            ref_range
            output
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

        if 'verbose' in self.configs.keys():
            if self.configs['verbose'] >= 1:
                logger.setLevel('INFO')
            if self.configs['verbose'] >= 2:
                logger.setLevel('WARNING')
            if self.configs['verbose'] >= 3:
                logger.setLevel('DEBUG')
        else:
            logger.debug('verbose is not set')

        self._start = 0
        self._end = 0
        self._file_list = None
        if 'start' in self.configs.keys():
            self._start = self.configs['start']
        if 'end' in self.configs.keys():
            self._end = self.configs['end']

        self._file_list = None
        if 'file_list' in self.configs.keys():
            self._file_list = self.configs['file_list']

        self._data_dir = None
        if 'data_dir' in self.configs.keys():
            self._data_dir = self.configs['data_dir']

        self._field_instance: 'observatory.HBA.feeds.HWPEncoder'
        if 'field_instance' in self.configs.keys():
            self._field_instance = self.configs['field_instance']

        self._field_list = ['rising_edge_count', 'irig_time', 'counter',
                            'counter_index', 'irig_synch_pulse_clock_time',
                            'irig_synch_pulse_clock_counts', 'quad']
        if 'field_list' in self.configs.keys():
            self._field_list = self.configs['field_list']

        # Size of pakcets sent from the BBB
        # 120 in the latest version, 150 in the previous version
        self._pkt_size = 120
        if 'packet_size' in self.configs.keys():
            self._pkt_size = self.configs['packet_size']

        # Number of encoder slits per HWP revolution

        self._num_edges = 570 * 2
        if 'num_edges' in self.configs.keys():
            self._num_edges = self.configs['num_edges']

        # Reference slit edgen width
        self._ref_edges = 2
        if 'ref_edges' in self.configs.keys():
            self._ref_edges = self.configs['ref_edges']

        # Reference slit angle
        self._delta_angle = 2 * np.pi / self._num_edges

        # Reference slit indexes
        self._ref_indexes = []

        self._output = None
        if 'output' in self.configs.keys():
            self._output = self.configs['output']

    def load_data(self, start=None, end=None,
                  archive_path=None, instance=None):
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
                path to HK g3 file, default = None
                assuming to input from hwp config file
                overwrite if inputting here as argument
            instance : str
                instance of field list, default = None
                assuming to input from hwp config file
                overwrite if inputting here as argument
                accept both for example;
                    just 'HBA' or 'observatory.HBA.feeds.HWPEncoder'
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

        if archive_path is not None:
            self._data_dir = archive_path
        if self._data_dir is None:
            logger.error("Can not find data directory")
            sys.exit(1)
        if instance is not None:
            if 'observatory' in instance:
                self._field_instance = instance
            else:
                self._field_instance = 'observatory.' + instance + '.feeds.HWPEncoder'

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

        data = so3g.hk.load_range(
            self._start,
            self._end,
            fields=hwp_keys,
            alias=alias,
            data_dir=self._data_dir)
        if not any(data):
            logger.info('HWP is not spinning in time range {' + str(
                self._start) + ' - ' + str(self._end) + '}, data is empty')

        return data

    def load_file(self, filename=None, instance=None):
        """
        Loads house keeping data with specified g3 files.
        Return HWP parameters from SO HK data.

        Args
        -----
            filename : str or [str] or np,array(str)
                default = None
                path and filename of HK g3 (str or array)
            instance : str
                default = None
                instance of field list
                 - assuming to input from hwp config file
                 - overwrite if inputting here as argument
                 - accept both for example;
                     just 'HBA' or 'observatory.HBA.feeds.HWPEncoder'
        """

        if filename is None and self._file_list is None:
            logger.error('Can not find input g3 file')
            sys.exit(1)
        if filename is not None:
            self._file_list = filename

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
            logger.info("Loading HK data files: {}".format(
                ' '.join(map(str, self._file_list))))
            self._start = 0
            self._end = 0
            return {}

        self._start = arc.simple(
            [key for key in arc.get_fields()[0].keys()][0])[0][0]
        self._end = arc.simple(
            [key for key in arc.get_fields()[0].keys()][0])[0][-1]
        for i in range(len(hwp_keys)):
            if not hwp_keys[i] in arc.get_fields()[0].keys():
                logger.info("HWP is not spinning in input g3 files")
                return {}

        data = {}
        for i in range(len(alias)):
            data = dict(**data, **{alias[i]: arc.simple(hwp_keys)[i]})

        return data

    def analyze(self, data, ratio=0.25, irig_type=0, fast=True):
        """
        Analyze HWP angle solution
        *** to be checked by hardware that 0 is CW and 1 is CCW from (sky side) consistently　for all SAT ***
        Args
        -----
            data : dict
                HWP HK data from load_data
            ratio : float, optional
                0.1 (default)
                parameter for referelce slit , threshold = 2 slit distances +/- 10%
            irig_type : 0 or 1, optional
                If 0, use 1 Hz IRIG timing (default)
                If 1, use 10 Hz IRIG timing
            fast : bool, optional
                If True, run fast fill_ref algorithm

        Returns
        --------
            dict{fast_time, angle, slow_time, stable, locked, hwp_rate}

            fast_time: IRIG synched timing (~2kHz)
            angle (float): IRIG synched HWP angle in radian
            slow_time: time list of slow block
            stable (flag): if non-zero, indicates the HWP spin state is known.
                           i.e. it is either spinning at a measurable rate, or stationary.
                           When this flag is non-zero, the hwp_rate field can be taken at face value.
            locked (flag): if non-zero, indicates the HWP is spinning and the position solution is working.
                           In this case one should find the hwp_angle populated in the fast data block.
            hwp_rate (float): the "approximate" HWP spin rate, with sign, in revs / second.
                            Use placeholder value of 0 for cases when not "stable".
        """

        if not any(data):
            logger.info("no HWP field data")
        ## Analysis parameters ##
        # Fast block
        if 'counter' in data.keys():
            counter = data['counter'][1]
            counter_idx = data['counter_index'][1]
            quad_time = data['quad'][0]
            quad = self._quad_form(data['quad'][1])
        else:
            counter = []
            counter_idx = []
            quad_time = []
            quad = []

        if 'irig_time' in data.keys():
            irig_time = data['irig_time'][1]
            rising_edge = data['rising_edge_count'][1]
            if irig_type == 1:
                irig_time = data['irig_synch_pulse_clock_time'][1]
                rising_edge = data['irig_synch_pulse_clock_counts'][1]
        else:
            irig_time = []
            rising_edge = []

        if 'counter' in data.keys() and 'irig_time' in data.keys():
            # Reject unexpected counter
            time = scipy.interpolate.interp1d(
                rising_edge,
                irig_time,
                kind='linear',
                fill_value='extrapolate')(counter)
            idx = np.where(
                (time >= data['irig_time'][0][0] - 2) & (time <= data['irig_time'][0][-1] + 2))
            counter = counter[idx]
            counter_idx = counter_idx[idx]

            fast_time, angle = self._hwp_angle_calculator(
                counter, counter_idx, irig_time, rising_edge, quad_time, quad, ratio, fast)

            # hwp speed calc. (approximate using ref)
            hwp_rate_ref = 1 / np.diff(fast_time[self._ref_indexes])
            hwp_rate = [0 for i in range(self._ref_indexes[0])]
            for n in range(len(np.diff(self._ref_indexes))):
                hwp_rate += [hwp_rate_ref[n]
                             for r in range(np.diff(self._ref_indexes)[n])]
            hwp_rate += [0 for i in range(len(fast_time) -
                                          self._ref_indexes[-1])]
        else:
            fast_time = []
            angle = []
            hwp_rate = []
        fast_time = np.array(fast_time)
        angle = np.array(angle)
        hwp_rate = np.array(hwp_rate)

        # Slow block
        # - Time definition -
        # if fast_time exists, slow_time = fast_time
        # else if irig_time exists but no fast_time, slow_time = irig_time
        # else: slow_time is per 10 sec array
        if len(fast_time) != 0 and len(irig_time) != 0:
            locked = np.ones(len(fast_time), dtype=bool)
            locked[np.where(hwp_rate == 0)] = 0
            irig_time = irig_time[np.where(
                (irig_time < fast_time[0]) | (irig_time > fast_time[-1]))]
            fast_irig_time = np.append(irig_time, fast_time).flatten()
            fast_irig_idx = np.argsort(fast_irig_time)
            fast_irig_time = fast_irig_time[fast_irig_idx]
            locked = (
                np.append(
                    np.zeros(
                        len(irig_time),
                        dtype=bool),
                    locked).flatten())[fast_irig_idx]
            hwp_rate = (
                np.append(
                    np.zeros(
                        len(irig_time),
                        dtype=float),
                    hwp_rate).flatten())[fast_irig_idx]
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

        slow_only_time = (np.arange(self._start, self._end, 10))
        if len(fast_irig_time) != 0:
            slow_only_time = slow_only_time[np.where(
                (slow_only_time < fast_irig_time[0]) | (slow_only_time > fast_irig_time[-1]))]
            slow_time = np.append(slow_only_time, fast_irig_time).flatten()
            slow_idx = np.argsort(slow_time)
            slow_time = slow_time[slow_idx]
            locked = np.append(np.zeros(len(slow_only_time),
                               dtype=bool), locked)[slow_idx]
            stable = np.append(np.zeros(len(slow_only_time),
                               dtype=bool), stable)[slow_idx]
            hwp_rate = np.append(
                np.zeros(
                    len(slow_only_time),
                    dtype=float),
                hwp_rate)[slow_idx]
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

        File format
        --------
        Provider: 'hwp'
            Fast block
                'hwp.hwp_angle'
            Slow block
                'hwp.stable'
                'hwp.locked'
                'hwp.hwp_rate'

            fast_time: IRIG synched timing (~2kHz)
            angle (float): IRIG synched HWP angle in radian
            slow_time: time list of slow block
            stable (flag): if non-zero, indicates the HWP spin state is known.
                           i.e. it is either spinning at a measurable rate, or stationary.
                            When this flag is non-zero, the hwp_rate field can be taken at face value.
            locked (flag): if non-zero, indicates the HWP is spinning and the position solution is working.
                           In this case one should find the hwp_angle populated in the fast data block.
            hwp_rate (float): the "approximate" HWP spin rate, with sign, in revs / second.
                            Use placeholder value of 0 for cases when not "locked".
        """
        if self._output is None and output is None:
            logger.error('Not specified output file')
            return
        if output is not None:
            self._output = output
        if solved['slow_time'].size == 0:
            logger.error('input data is empty')
            return
        if solved['fast_time'].size == 0:
            logger.info('write no rotation data')
            return
        session = so3g.hk.HKSessionHelper(hkagg_version=2)
        writer = core.G3Writer(output)
        writer.Process(session.session_frame())
        prov_id = session.add_provider('hwp')
        writer.Process(session.status_frame())
        frame = session.data_frame(prov_id)

        slow_block = core.G3TimesampleMap()
        slow_block.times = core.G3VectorTime(
            [core.G3Time(_t * core.G3Units.s) for _t in solved['slow_time']])
        slow_block['stable'] = core.G3VectorInt(solved['stable'])
        slow_block['locked'] = core.G3VectorInt(solved['locked'])
        slow_block['hwp_rate'] = core.G3VectorDouble(solved['hwp_rate'])
        frame['block_names'].append('slow')
        frame['blocks'].append(slow_block)

        if len(solved['fast_time']) != 0:
            fast_block = core.G3TimesampleMap()
            fast_block.times = core.G3VectorTime(
                [core.G3Time(_t * core.G3Units.s) for _t in solved['fast_time']])
            fast_block['hwp_angle'] = core.G3VectorDouble(solved['angle'])
            frame['block_names'].append('fast')
            frame['blocks'].append(fast_block)

        writer.Process(frame)

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

        # reference finding and fill its angle
        self._find_refs(ratio)
        if fast:
            self._fill_refs_fast()
        else:
            self._fill_refs()

        # assign IRIG synched timestamp
        self._time = scipy.interpolate.interp1d(
            self._rising_edge,
            self._irig_time,
            kind='linear',
            fill_value='extrapolate')(
            self._encd_clk)
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

    def _find_refs(self, dev):
        """ Find reference slits """
        self._ref_indexes = []
        # Calculate spacing between all clock values
        diff = np.ediff1d(self._encd_clk, to_begin=0)  # [1:]

        split = int(len(diff) / self._num_edges)
        diff_split = np.array_split(diff, split)
        offset = 0
        # Conditions for idenfitying the ref slit
        # Slit distance somewhere between 2 slits:
        # 2 slit distances (defined above) +/- 10%
        for i in range(split):
            _diff = diff_split[i]
            # Define median value as nominal slit distance
            slit_dist = np.median(_diff)
            # Conditions for idenfitying the ref slit
            # Slit distance somewhere between 2 slits:
            # 2 slit distances (defined above) +/- 10%
            ref_hi_cond = ((self._ref_edges + 2) * slit_dist * (1 + dev))
            ref_lo_cond = ((self._ref_edges + 1) * slit_dist * (1 - dev))
            # Find the reference slit locations (indexes)
            _ref_idx = np.argwhere(np.logical_and(
                _diff < ref_hi_cond, _diff > ref_lo_cond)).flatten()
            if len(_ref_idx) == 1:
                self._ref_indexes.append(_ref_idx[0] + offset)
                offset += len(diff_split[i])
            if len(_ref_idx) == 2:
                self._ref_indexes.append(_ref_idx[0] + offset)
                self._ref_indexes.append(_ref_idx[1] + offset)
                offset += len(diff_split[i])
        # Define the reference slit line to be the line before
        # the two "missing" lines
        # Store the count and clock values of the reference lines

        self._ref_indexes = np.array(self._ref_indexes)
        if len(self._ref_indexes) == 0:
            print(
                'WARNING: can not find reference points, please adjust ratio parameter!')
            sys.exit(1)
        self._ref_clk = np.take(self._encd_clk, self._ref_indexes)
        self._ref_cnt = np.take(self._encd_cnt, self._ref_indexes)
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
            print("\r {:.2f} %".format(
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
            # print(clks_to_add)
            # print(cnts_to_add)
            #print(self._ref_cnt, np.diff(self._ref_cnt), print(self._ref_indexes))
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
            self._encd_cnt.size + self._ref_indexes.size * self._ref_edges)
        self._ref_cnt += np.arange(self._ref_cnt.size) * self._ref_edges
        self._ref_indexes += np.arange(self._ref_indexes.size) * \
            self._ref_edges

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
        direction = list(map(lambda x: 1 if x == 0 else -1, quad))
        self._angle = direction * \
            (self._encd_cnt - self._ref_cnt[0]
             ) * self._delta_angle % (2 * np.pi)
        return

    def _find_dropped_packets(self):
        """ Estimate the number of dropped packets """
        cnt_diff = np.diff(self._encd_cnt)
        dropped_samples = np.sum(cnt_diff[cnt_diff >= self._pkt_size])
        self._num_dropped_pkts = dropped_samples // (self._pkt_size - 1)
        if self._num_dropped_pkts > 0:
            logger.warning('WARNING: {} dropped packets are found.'.format(
                self._num_dropped_pkts))
        return

    def _encoder_packet_sort(self):
        cnt_diff = np.diff(self._encd_cnt)
        if np.any(cnt_diff != 1):
            logger.warning(
                'WARNING: a part of the counter is incorrect, performing the correction process... ')
            if np.any(cnt_diff < 0):
                if 1 - self._pkt_size in cnt_diff:
                    logger.warning(
                        'WARNING: Packet flip found, sorting process performed... ')
                idx = np.argsort(self._encd_cnt)
                self._encd_clk = self._encd_clk[idx]
            else:
                logger.warning('WARNING: maybe packet drop exists ...')
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

    def interp_smurf(self, smurf_timestamp):
        smurf_angle = scipy.interpolate.interp1d(
            self._time,
            self.angle,
            kind='linear',
            fill_value='extrapolate')(smurf_timestamp)
        return smurf_angle

    def main(args=None):

        if args is None:
            parser = argparse.ArgumentParser(
                description='Analyze HWP encoder data from level-2 HK data, \
                                                        and produce HWP angle solution for all times.')
            parser.add_argument(
                '-c', '--config-file', default=None, type=str,
                help="Configuration File for running update_hwp_angle")
            parser.add_argument(
                '-t', '--time', action='store', default=None, type=int,
                help='time range ex) --time [start] [end]',
                nargs=2)
            parser.add_argument(
                '-d', '--data-dir', action='store', default=None, type=str,
                help='input data directory')
            parser.add_argument(
                '-f', '--file', action='store', default=None, type=str, nargs='*',
                help='filename or list of filenames (to be loaded in order).')
            parser.add_argument(
                '-o', '--output', action='store', default=None, type=str,
                help='path to output g3 file')
        args = parser.parse_args()

        logger.info("Starting update_hwp_angle")

        configs = yaml.safe_load(open(args.config_file, "r"))

        logger.info("instance update_hwp_angle class")
        hwp = update_hwp_angle(args.config_file)

        # Load data from arguments or config file
        logger.info("load_data")
        data = None
        if args.time is not None and args.data_dir is not None:
            data = hwp.load_data(
                args.time[0], args.time[1], archive_path=args.data_dir)
        elif args.time is not None and args.data_dir is None:
            data = hwp.load_data(args.time[0], args.time[1])
        elif args.file is not None:
            data = hwp.load_file(args.file)
        elif 'start' in configs.keys() and 'end' in configs.keys():
            data = hwp.load_data(configs['start'], configs['end'])
        elif 'file' in configs.keys():
            data = hwp.load_file(configs['file'])
        else:
            logger.error("Not specified time range and filenames")
            sys.exit(1)

        logger.info("analyze")
        solved = hwp.analyze(data)

        logger.info("write_solution")
        if args.output is not None:
            output = args.output
        elif 'output' in configs.keys():
            output = configs['output']
        hwp.write_solution(solved, output)
        
        logger.info("output file: " + output)
        logger.info("Finised update_hwp_angle")


if __name__ == "__main__":
    update_hwp_angle.main()

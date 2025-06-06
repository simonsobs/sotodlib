from typing import Optional, Dict
from dataclasses import dataclass, fields

import so3g
from so3g.proj import Ranges
from spt3g import core
import itertools
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
from tqdm.auto import tqdm
import os
import logging
import sys
import shutil
import yaml
import datetime as dt
from zipfile import ZipFile, ZIP_DEFLATED
import sotodlib
from sotodlib.site_pipeline.util import init_logger
from .datapkg_utils import walk_files


log = logging.getLogger('bookbinder')
if not log.hasHandlers():
    init_logger('bookbinder')

class TimingSystemOff(Exception):
    """Exception raised when we try to bind books where the timing system is found to be off and the books have imprecise timing counters"""
    pass

class NoScanFrames(Exception):
    """Exception raised when we try and bind a book but the SMuRF file contains not Scan frames (so no detector data)"""
    pass

MAX_DROPPED_SAMPLES = 100
class BadTimeSamples(Exception):
    """Exception raised when there are drops in the time samples in the 
    UFM timestreams"""
    pass

class NoHKFiles(Exception):
    """Exception raised when we cannot find any HK data around the book time"""
    pass

class NoMountData(Exception):
    """Exception raised when we cannot find mount data"""
    pass

class NoHWPData(Exception):
    """Exception raised when we cannot find HWP data"""
    pass

class DuplicateAncillaryData(Exception):
    """Exception raised when we find the HK data has copies of the same timestamps"""
    pass

class NonMonotonicAncillaryTimes(Exception):
    """Exception raised when we find the HK data has timestamps that are not strictly increasing monotonically"""
    pass

class BookDirHasFiles(Exception):
    """Exception raised when files already exist in a book directory"""
    pass

def setup_logger(logfile=None):
    """
    This setups up a logger for bookbinder. If a logfile is passed, it will
    write to that file as well as stdout. It is useful to create a one-off
    logger here instead of using `getLogger` because it allows us to set
    a separate log-file for each bookbinder instance.
    """
    fmt = '%(asctime)s: %(message)s (%(levelname)s)'
    log = logging.Logger('bookbinder', level=logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if logfile is not None:
        ch = logging.FileHandler(logfile)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)

    return log

def get_frame_iter(files):
    """
    Returns a continuous iterator over frames for a list of files.
    """
    return itertools.chain(*[core.G3File(f) for f in files])

def close_writer(writer):
    """
    Closes out a G3FileWriter with an end-processing frame. If None is passed,
    this will not do anything.
    """
    if writer is None:
        return
    writer(core.G3Frame(core.G3FrameType.EndProcessing))

def next_scan(it):
    """
    Returns the next Scan frame, along with any intermediate frames for an
    iterator.
    """
    interm_frames = []
    for frame in it:
        if frame.type == core.G3FrameType.Scan:
            return frame, interm_frames
        interm_frames.append(frame)
    return None, interm_frames

class HkDataField:
    """
    Class containing HK Data for a single field.

    Args
    -----
    instance_id: str
        Instance id of the agent producing the data.
    feed: str
        Feed name for hk data feed.
    field: str
        Field name for hk data feed.

    Attributes
    -----------
    times: np.ndarray
        HK sample timestamps.
    data: np.ndarray
        HK sample data.
    finalized: bool
        True if datas has been processed and finalized.
    """
    def __init__(self, instance_id: str, feed: str, field: str):
        self.instance_id = instance_id
        self.feed = feed
        self.field = field

        self.times =[]
        self.data = []
        self.finalized = False

    def __len__(self):
        return len(self.times)

    @property
    def addr(self):
        """Returns full address of field"""
        return f"{self.instance_id}.{self.feed}.{self.field}"

    def process_frame(self, frame):
        """Update data based on G3Frame"""
        address = frame['address']  # "<site>.<instance_id>.feeds.<feed_name>""
        spl = address.split('.')
        instance_id, feed = spl[1], spl[3]
        if instance_id != self.instance_id or feed != self.feed:
            return

        for block in frame['blocks']:
            if self.field not in block:
                continue
            self.times.append(np.array(block.times) / core.G3Units.s)
            self.data.append(block[self.field])

    def finalize(self, drop_duplicates=False, require_monotonic_times=True):
        """Finalize data, and store in numpy array"""
        self.times = np.hstack(self.times, dtype=np.float64)
        self.data = np.hstack(self.data)
        self.finalized = True

        # Check for duplicates
        clean_times, idxs = np.unique(self.times, return_index=True)
        if len(self.times) != len(clean_times):
            if not drop_duplicates:
                raise DuplicateAncillaryData(
                    f"HK data from {self.addr} has" 
                    " duplicate timestamps"
                )
            else:
                log.warning(
                    f"HK data from {self.addr} has duplicate timestamps"
                )
            self.times = self.times[idxs]
            self.data = self.data[idxs]
        if not np.all(np.diff(self.times)>0):
            bad = np.sum( np.diff(self.times) <= 0)
            msg = f"Times from {self.addr} have {bad} samples that are " \
                "not increasing"
            if require_monotonic_times:
                raise NonMonotonicAncillaryTimes(msg)
            else:
                log.warning(msg)
        
@dataclass
class HkData:
    """
    Class containing HkData for bookbinding, including ACU and HWP data.
    """
    az: Optional[HkDataField] = None
    el: Optional[HkDataField] = None
    boresight: Optional[HkDataField] = None
    corotator_enc: Optional[HkDataField] = None
    az_mode: Optional[HkDataField] = None
    hwp_freq: Optional[HkDataField] = None

    @classmethod
    def from_dict(cls, d: Dict[str, str]):
        """
        Creates HkData object from a dict of field addresses. Addresses must be formatted like ``<instance_id>.<feed>.<field>``.
        Keys of dict must be valide fields of HkData class, such as ``az`` or ``el``.
        """
        kw = {}
        for k, v in d.items():
            try:
                instance, feed, field = v.split('.')
            except Exception as exc:
                raise ValueError(f"Could not parse field: {v}. "
                                 "Must be formatted <instance_id>.<feed>.<field>") from exc
            kw[k] = HkDataField(instance, feed, field)
        return cls(**kw)

    def process_frame(self, frame):
        """Processes G3frame, and updates relevant HkDataFields"""
        for fld in fields(self):
            f = getattr(self, fld.name)
            if isinstance(f, HkDataField):
                f.process_frame(frame)

    def finalize(self, drop_duplicates=True, require_monotonic_times=True):
        """Finalizes HkDatafields"""
        for fld in fields(self):
            f = getattr(self, fld.name)
            if isinstance(f, HkDataField):
                f.finalize(
                    drop_duplicates=drop_duplicates,
                    require_monotonic_times=require_monotonic_times,
                )

class AncilProcessor:
    """
    Processor for ancillary (ACU) data

    Params
    --------
    files : list
        List of HK files to process.
    book_id: str
        ID of book being bound.
    hk_fields: dict
        Dictionary of fields corresponding to relevant HK Data. See the HkData
        class for what housekeeping fields are allowed.  For example::

            >> hk_fields = {
                'az': 'acu.acu_udp_stream.Corrected_Azimuth',
                'el': 'acu.acu_udp_stream.Corrected_Elevation',
                'boresight': 'acu.acu_udp_stream.Corrected_Boresight',
                # corotator_enc: acu.acu_status.Corotator_current_position # (For LAT)
                'az_mode':  'acu.acu_status.Azimuth_mode',
                'hwp_freq': 'hwp-bbb-e1.HWPEncoder.approx_hwp_freq',
            }

    Attributes
    -----------
    hkdata : HkData
        Class containing relevant Hk data for the duration of the book.
    times : np.ndarray
        Timestamps for anc data. This will be populated after preprocess.
    anc_frame_data : List[G3TimestreamMap]
        List of G3TimestreamMaps saved for each bound frame. This will be
        populated on bind and should be used to add copies of the anc data to
        the detector frames.
    """
    def __init__(self, files, book_id, hk_fields: Dict, 
                 drop_duplicates=False, require_hwp=True, 
                 require_acu=True, require_monotonic_times=True, 
                 log=None
                 ):
        self.hkdata: HkData = HkData.from_dict(hk_fields)

        self.files = files
        self.anc_frame_data = None
        self.out_files = []
        self.book_id = book_id
        self.preprocessed = False
        self.drop_duplicates = drop_duplicates
        self.require_hwp = require_hwp
        self.require_acu = require_acu
        self.require_monotonic_times = require_monotonic_times
        if log is None:
            self.log = logging.getLogger('bookbinder')
        else:
            self.log = log

        if len(self.files) == 0:
            if self.require_acu or self.require_hwp:
                raise NoHKFiles("No HK files specified for book")
            self.log.warning("No HK files found for book")
            for fld in ['az', 'el', 'boresight', 'corotator_enc','az_mode', 'hwp_freq']:
                setattr(self.hkdata, fld, None)

        if self.require_acu and self.hkdata.az is None:
            self.log.warning("No ACU data specified in hk_fields!")

        if self.require_hwp and self.hkdata.hwp_freq is None:
            self.log.warning("No HWP Freq data is specified in hk_fields.")

    def preprocess(self):
        """
        Preprocesses HK data and populates the `data` and `times` objects.
        """
        if self.preprocessed:
            return

        self.log.info("Preprocessing HK Data")
        frame_iter = get_frame_iter(self.files)

        for fr in frame_iter:
            if fr['hkagg_type'] != 2:
                continue
            self.hkdata.process_frame(fr)

        # look for ACU fields that are configured but not found in HK data files
        # will not check fields that are not in the configuration file
        for fld in ['az', 'el', 'boresight', 'corotator_enc']:
            f = getattr(self.hkdata, fld)
            if f is not None:
                if self.require_acu and len(f) == 0:
                    raise NoMountData(
                        f"Did not find ACU data in {self.files} for {fld}",
                    )
                elif len(f) == 0:
                    ## requiring Az data is fails and we didn't find any
                    self.log.warning(
                        f"Did not find ACU data for {fld}. Bypassed because "
                        "require_acu is false"
                    )
                    setattr(self.hkdata, fld, None)

        # look for HWP data if HWP fields are in configuration file
        if self.hkdata.hwp_freq is not None:
            if self.require_hwp and len(self.hkdata.hwp_freq) == 0:
                raise NoHWPData(
                    f"Did not find HWP data in {self.files}",
                )
            elif len(self.hkdata.hwp_freq) == 0:
                ## requiring HWP data is false and we didn't find any
                self.log.warning(
                    f"Did not find HWP data in data. Bypassed because "
                    "require_hwp is false"
                )
                self.hkdata.hwp_freq = None
        
        self.hkdata.finalize(
            drop_duplicates=self.drop_duplicates,
            require_monotonic_times=self.require_monotonic_times,
        )
        self.preprocessed = True

    def bind(self, outdir, times, frame_idxs, file_idxs):
        """
        Binds ancillary data.

        Params
        ----------
        outdir : path
            Path where files should be written
        times : np.ndarray
            Array of timestamps to write to book
        frame_idxs : np.ndarray
            Array mapping sample (in times) to output frame index
        file_idxs : np.ndarray
            Array mapping output frame idx to output file idx
        """
        self.log.info("Binding ancillary data")

        # Handle file writers
        writer = None
        cur_file_idx = None
        out_files = []

        def validate_mount_field(hk_field: HkDataField, max_dt=None):
            m = (times[0] <= hk_field.times) & (hk_field.times <= times[-1])
            if m.sum() < 2:
                raise NoMountData(
                    f"No mount data overlapping with detector data: {hk_field.addr}"
                )
            if max_dt is not None:
                _max_dt = np.max(np.diff(hk_field.times[m]))
                if _max_dt > max_dt:
                    raise NoMountData(
                        f"Max data spacing {_max_dt}s is higher than {max_dt}s for {hk_field.addr}. "
                        "Interpolation may be questionable."
                    )

        # go through and interpolate ACU times to detector times
        acu_interp_data = {}
        for fld in ['az', 'el', 'boresight', 'corotator_enc']:
            f = getattr(self.hkdata, fld)
            if f is not None:
                try:
                    validate_mount_field(f, max_dt=10)
                    acu_interp_data[fld] = np.interp(
                        times, f.times, f.data
                    )
                except NoMountData as e:
                    if self.require_acu:
                        raise e
                    else:
                        self.log.warning(e)
                        acu_interp_data[fld] = None
            else: 
                acu_interp_data[fld] = None

        az = acu_interp_data['az']
        el = acu_interp_data['el']
        boresight = acu_interp_data['boresight']
        corotator_enc = acu_interp_data['corotator_enc']

        anc_frame_data = []
        for oframe_idx in np.unique(frame_idxs):
            # Update file writer if starting a new output file
            if file_idxs[oframe_idx] != cur_file_idx:
                close_writer(writer)
                cur_file_idx = file_idxs[oframe_idx]
                fname = os.path.join(outdir, f'A_ancil_{cur_file_idx:0>3}.g3')
                out_files.append(fname)
                writer = core.G3Writer(fname)

            m = frame_idxs == oframe_idx
            ts = times[m]

            oframe = core.G3Frame(core.G3FrameType.Scan)

            i0, i1 = np.where(m)[0][[0, -1]]
            oframe['sample_range'] = core.G3VectorInt([int(i0), int(i1+1)])
            oframe['book_id'] = self.book_id

            anc_data = core.G3TimesampleMap()
            anc_data.times = core.G3VectorTime(ts * core.G3Units.s)
            if az is not None:
                anc_data['az_enc'] = core.G3VectorDouble(az[m])
                anc_data['el_enc'] = core.G3VectorDouble(el[m])
            if boresight is not None:
                anc_data['boresight_enc'] = core.G3VectorDouble(boresight[m])
            if corotator_enc is not None:
                anc_data['corotator_enc'] = core.G3VectorDouble(corotator_enc[m])
            oframe['ancil'] = anc_data
            writer(oframe)
            anc_frame_data.append(anc_data)

            self.add_acu_summary_info(oframe, ts[0], ts[-1])

        # Save this to be added to detector files
        self.anc_frame_data = anc_frame_data
        self.out_files = out_files

    def add_acu_summary_info(self, frame, t0, t1):
        """
        Adds ACU summary information to a G3Frame. This will add the following
        info if data is present in the HK dataset::
            - azimuth_mode: (str)
                Azimuth_mode, pulled from the ACU summary data. `ProgramTrack`
                means that this frame contains scan data, and `Preset` means the
                telescope is slewing.
            - azimuth_velocity_mean / azimuth_velocity_std: (float / float)
                Mean and standard deviation of the az velocity (deg / sec)
            - elevation_velocity_mean / elevaction_velocity_std: (float / float)
                Mean and standard deviation of the el velocity (deg / sec)

        Params
        ----------
        frame : G3Frame
            Frame to add data to
        t0 : float
            Start time of frame (unix time), inclusive
        t1 : float
            Stop time of frame (unix time), inclusive
        """
        az_mode = self.hkdata.az_mode
        az = self.hkdata.az
        el = self.hkdata.el
        if az_mode is not None:
            m = (t0 <= az_mode.times) & (az_mode.times <= t1)
            if not np.any(m):
                frame['azimuth_mode'] = 'None'
            elif 'ProgramTrack' in az_mode.data[m]:
                frame['azimuth_mode'] = 'ProgramTrack' # Scanning
            else:
                frame['azimuth_mode'] = 'Preset'  # Slewing

        if az is not None:
            m = (t0 <= az.times) & (az.times <= t1)
            if m.sum() >= 2:
                dt = np.diff(az.times[m]).mean()
                az_vel = np.diff(az.data[m]) / dt
                frame['azimuth_velocity_mean'] = np.mean(az_vel)
                frame['azimuth_velocity_stdev'] = np.std(az_vel)
        for k in ['azimuth_velocity_mean', 'azimuth_velocity_stdev']:
            if k not in frame:
                frame[k] = np.nan

        if el is not None:
            m = (t0 <= el.times) & (el.times <= t1)
            if m.sum() >= 2:
                dt = np.diff(el.times[m]).mean()
                el_vel = np.diff(el.data[m]) / dt
                frame['elevation_velocity_mean'] = np.mean(el_vel)
                frame['elevation_velocity_stdev'] = np.std(el_vel)
        for k in ['elevation_velocity_mean', 'elevation_velocity_stdev']:
            if k not in frame:
                frame[k] = np.nan

class SmurfStreamProcessor:
    def __init__(self, obs_id, files, book_id, readout_ids,
                 log=None, allow_bad_timing=False):
        self.files = files
        self.obs_id = obs_id
        self.stream_id = None
        self.times = None
        self.frame_idxs = None
        self.nchans = None
        self.nframes = None
        self.bias_names = None
        self.primary_names = None
        self.timing_paradigm = None
        self.session_id = None
        self.slow_primary = None
        self.sostream_version = None
        self.readout_ids = readout_ids
        self.out_files = []
        self.book_id = book_id
        self.allow_bad_timing = allow_bad_timing

        if log is None:
            self.log = logging.getLogger('bookbinder')
        else:
            self.log = log

    def preprocess(self):
        """
        Extracts file times, nchans, and nframes from file list
        """
        if self.times is not None:  # Already preprocessed
            return

        self.log.info(f"Preprocessing smurf obsid {self.obs_id}")

        self.nframes = 0
        ts = []
        smurf_frame_counters = [] # smurf frame-counter
        fc_idx = None
        frame_idxs = []
        frame_idx = 0
        timing = True
        for frame in get_frame_iter(self.files):
            if frame.type != core.G3FrameType.Scan:
                continue

            # Populate attributes from the first scan frame
            if self.nchans is None:
                self.nchans = len(self.readout_ids)
                self.primary_names = frame['primary'].names
                fc_idx = list(self.primary_names).index("FrameCounter")
                self.bias_names = frame['tes_biases'].names
                self.timing_paradigm = frame['timing_paradigm']
                self.session_id = frame['session_id']
                if 'slow_primary' in frame:
                    self.slow_primary = frame['slow_primary']
                self.sostream_version = frame['sostream_version']
                self.stream_id = frame['sostream_id']

            good, t = get_frame_times(frame, self.allow_bad_timing)
            timing = timing and good
            ts.append(t)
            smurf_frame_counters.append(frame['primary'].data[fc_idx])
            frame_idxs.append(np.full(len(t), frame_idx, dtype=np.int32))

            self.nframes += 1
            frame_idx += 1

        if len(ts) == 0:
            raise NoScanFrames(f"{self.obs_id} has no detector data")
        self.times = np.hstack(ts)
        self.smurf_frame_counters = np.hstack(smurf_frame_counters)
        self.frame_idxs = np.hstack(frame_idxs)
 
        if np.any( np.diff(self.times) < 0):
            raise BadTimeSamples(
                f"{self.obs_id} has time samples not increasing"
            )
        
        timing = timing and (not self.timing_paradigm=='Low Precision')
        
        if (not self.allow_bad_timing) and (not timing):
            raise TimingSystemOff(
                f"Observation {self.obs_id} does not have high precision timing"
                " information. Pass `allow_bad_timing=True` to bind anyway"
            )
        
        # If low-precision, we need to linearize timestamps in order for
        # bookbinder to work properly
        if not timing:
            self.log.warning(
                "Timestamps are Low Precision, linearizing from frame-counter"
            )
            dt, offset = np.polyfit(self.smurf_frame_counters, self.times, 1)
            self.times = offset + dt * self.smurf_frame_counters

    def bind(self, outdir, times, frame_idxs, file_idxs, pbar=False, ancil=None,
             atol=1e-4):
        """
        Binds SMuRF data
        
        Params
        ------ 
        outdir : str
            Output directory to put bound files
        times : np.ndarray
            Full array of timestamps that should be contained in the books
        frame_idxs : np.ndarray
            Output frame idx for each specified timestamp
        file_idxs : np.ndarray
            Output file idx for each specified output frame
        pbar : bool or tqdm.tqdm
            If True, will create a new progress bar. If False, will disable.
            If a progress bar is passed, will use that.
        ancil : AncilProcessor
            Ancillary processor object (must be already bound). Ancil data will
            be copied into the output frames.
        atol : float
            Absolute tolerance between smurf-timestamps and book-timestamps.
            Samples mapped to times that are further away than atol will
            be considered unmapped.
        """
        if pbar is True:
            pbar = tqdm(total=self.nframes)
        elif pbar is False:
            pbar = tqdm(total=self.nframes, disable=True)

        pbar.set_description(f"Binding {self.stream_id}")

        self.log.info(f"Binding smurf obsid {self.obs_id}")

        # Here `times` is the full array of times in the book (gapless reference
        # times) and `self.times`` is timestamps from the pre-processed L2 data
        # (may contain gaps)
        sample_map = find_ref_idxs(times, self.times)
        mapped = np.abs(times[sample_map] - self.times) < atol

        oframe_idxs = frame_idxs[sample_map]  # out-frame idx for each in sample
        oframe_idxs[~mapped] = -1 
        _, offsets = np.unique(frame_idxs, return_index=True)
        # Sample idx within each out-frame for every input sample
        out_offset_idxs = sample_map - offsets[frame_idxs[sample_map]]

        iframe_idxs = self.frame_idxs  #in-frame idx for each in sample
        _, offsets = np.unique(self.frame_idxs, return_index=True)
        # Sample idx within each in-frame for every input sample
        in_offset_idxs = np.arange(len(self.times)) - offsets[self.frame_idxs]

        # Mask to use to separate out non-resonator signal from resonator signal
        tracked_chans = np.array(['NONE' not in r for r in self.readout_ids], dtype=bool)
        readout_id_arr = np.array(self.readout_ids)

        # Handle file writers
        writer = None
        cur_file_idx = None
        out_files = []

        inframe_iter = get_frame_iter(self.files)
        iframe, interm_frames = next_scan(inframe_iter)
        iframe_idx = 0
        oframe_num = 0
        pbar.update()

        for oframe_idx in np.unique(frame_idxs):
            # Update writer
            if file_idxs[oframe_idx] != cur_file_idx:
                close_writer(writer)
                cur_file_idx = file_idxs[oframe_idx]
                fname = os.path.join(
                    outdir, f'D_{self.stream_id}_{cur_file_idx:0>3}.g3')
                out_files.append(fname)
                writer = core.G3Writer(fname)

            # Initialize stuff
            m = frame_idxs == oframe_idx
            nsamp = np.sum(m)
            ts = times[m]
            data = np.zeros((self.nchans, nsamp), dtype=np.int32)
            biases = np.zeros((len(self.bias_names), nsamp), dtype=np.int32)
            primary = np.zeros((len(self.primary_names), nsamp), dtype=np.int64)
            filled = np.zeros(nsamp, dtype=bool)
            
            # Loop through in_frames filling current out_frame
            while True:
                # First, write any intermediate frames like observation and wiring
                for fr in interm_frames:
                    if 'frame_num' in fr:
                        del fr['frame_num']
                    fr['frame_num'] = oframe_num  # Update this so they remain ordered
                    oframe_num += 1
                    writer(fr)

                m = (oframe_idxs == oframe_idx) & (iframe_idxs == iframe_idx)
                outsamps = out_offset_idxs[m]
                insamps = in_offset_idxs[m]

                # Below is equivalent to:
                #    >> data[:, outsamps] = iframe['data'][:, insamps]
                #    >> ...
                # However it is much faster to copy arrays by mapping contiguous
                # chunks to contiguous chunks using slices, like:
                #    >> arr_out[:, o0:o1] = arr_in[:, i0:i1]
                # since numpy does not need to create a temporary copy of the
                # data, and can just do a direct mem-map. This speeds up binding
                # by a factor of ~4.
                #
                # Here we are splitting outsamps and insamps into a list
                # of ranges where both arrays are contiguous. Then we loop
                # through each sub-range and copy data via slicing.
                split_idxs = 1 + np.where(
                    (np.diff(outsamps) > 1) | (np.diff(insamps) > 1))[0]
                outsplits = np.split(outsamps, split_idxs)
                insplits = np.split(insamps, split_idxs)
                for i in range(len(outsplits)):
                    if len(insplits[i]) == 0:
                        continue

                    in0, in1 = insplits[i][0], insplits[i][-1] + 1
                    out0, out1 = outsplits[i][0], outsplits[i][-1] + 1

                    data[:, out0:out1] = iframe['data'].data[:, in0:in1]
                    biases[:, out0:out1] = iframe['tes_biases'].data[:, in0:in1]
                    primary[:, out0:out1] = iframe['primary'].data[:, in0:in1]
                    filled[out0:out1] = 1

                # If there are any remaining samples in the next in_frame, pull it and repeat
                if np.any((oframe_idxs == oframe_idx) & (iframe_idxs > iframe_idx)):
                    iframe, interm_frames = next_scan(inframe_iter)
                    iframe_idx += 1
                    pbar.update()
                    continue
                else:
                    break

            # Interpolate data where there are gaps
            if np.all(~filled):
                self.log.error(
                    f"No samples properly mapped in oframe frame {oframe_idx}!"
                    "Cannot properly interpolate for this frame."
                )
                raise ValueError(f"Cannot finish binding {self.obs_id}")
            elif np.any(~filled):
                self.log.warning(
                    f"{np.sum(~filled)} missing samples in out-frame {oframe_idx}"
                )
                # Missing samples at the beginning / end of a frame will be
                # filled with the first / last sample in the frame
                fill_value = (data[:, filled][:, 0], data[:, filled][:, -1])
                data[:, ~filled] = interp1d(
                    ts[filled], data[:, filled], axis=1, assume_sorted=True,
                    kind='linear', fill_value=fill_value, bounds_error=False
                )(ts[~filled])

            m = frame_idxs == oframe_idx
            i0, i1 = np.where(m)[0][[0, -1]]

            oframe = core.G3Frame(core.G3FrameType.Scan)

            if ancil is not None:
                t0, t1 = ts[0], ts[-1]
                oframe['ancil'] = ancil.anc_frame_data[oframe_idx]
                ancil.add_acu_summary_info(oframe, t0, t1)

            oframe['book_id'] = self.book_id 
            oframe['sample_range'] = core.G3VectorInt([int(i0), int(i1+1)])
            oframe['flag_smurfgaps'] = core.G3VectorBool(~filled)

            ts = core.G3VectorTime(ts * core.G3Units.s)
            oframe['signal'] = so3g.G3SuperTimestream(readout_id_arr[tracked_chans], ts, data[tracked_chans, :])
            oframe['untracked'] = so3g.G3SuperTimestream(readout_id_arr[~tracked_chans], ts, data[~tracked_chans, :])
            oframe['primary'] = so3g.G3SuperTimestream(self.primary_names, ts, primary)
            oframe['tes_biases'] = so3g.G3SuperTimestream(self.bias_names, ts, biases)
            oframe['stream_id'] = self.stream_id
            oframe['frame_num'] = oframe_num

            oframe_num += 1
            writer(oframe)
        
        close_writer(writer)
        self.out_files = out_files

        # In case there were remaining frames left over
        pbar.update(self.nframes - iframe_idx - 1)
        if pbar.n >= pbar.total:
            pbar.close()

class BookBinder:
    """
    Class for combining smurf and hk L2 data to create books containing detector
    timestreams.

    Parameters
    ----------
    book : sotodlib.io.imprinter.Books
        Book object to bind
    obsdb : dict
        Result of imprinter.get_g3tsmurf_obs_for_book. This should be a dict
        from obs-id to G3tSmurf Observations.
    filedb : dict
        Result of imprinter.get_files_for_book. This should be a dict from
        obs-id to the list of smurf-files for that observation.
    hkfiles : list
        List of HK files to process.
    max_samps_per_frame : int
        Max number of samples per frame. This will be used to split frames
        when the ACU data isn't present or cannot be used.
    max_file_size : int
        Max file size in bytes. This will be used to rotate files.
    readout_ids : dict, optional
        Dict of readout_ids to use for each stream_id. If provided, these
        will be used to set the `names` in the signal frames. If not provided,
        names will be taken from the input frames.
    ignore_tags : bool, optional
        if true, will ignore tags if the level 2 observations have unmatched
        tags
    ancil_drop_duplicates: bool, optional
        if true, will drop duplicate timestamp data from ancillary files. added
        to deal with an occassional hk aggregator error where it is picking up
        multiple copies of the same data
    require_acu: bool, optional
        if true, will throw error if we do not find Mount data
    require_monotonic_times: bool, optional
        if true, will throw error if we ever see timestamps not increasing or going backwards
    require_hwp: bool, optional
        if true, will throw error if we do not find HWP data
    allow_bad_time: bool, optional
        if not true, books will not be bound if the timing systems signals are not found. 
    min_ctime: float, optional
        if not None, will cut book to this minimum ctime
    max_ctime: float optional
        if not None, will cut book to this maximum ctime
    
    Attributes
    -----------
    ancil : AncilProcessor
        Processor for ancillary data
    streams : dict
        Dict of SmurfStreamProcessor objects, keyed by stream_id
    times : np.ndarray
        Array of times for all samples in the book
    frame_idxs : np.ndarray
        Array of output frame indices for all samples in the book
    file_idxs : np.ndarray
        Array of output file indices for all output frames in the book
    """
    def __init__(self, book, obsdb, filedb, data_root, readout_ids, 
                outdir, hk_fields, max_samps_per_frame=50_000, max_file_size=1e9, ignore_tags=False, ancil_drop_duplicates=False, 
                require_hwp=True, require_acu=True,
                require_monotonic_times=True,
                allow_bad_timing=False,
                min_ctime=None, max_ctime=None):
        self.filedb = filedb
        self.book = book
        self.data_root = data_root
        self.hk_root = os.path.join(data_root, 'hk')
        self.meta_root = os.path.join(data_root, 'smurf')
        
        self.obsdb = obsdb
        self.outdir = outdir

        assert book.schema==0, "obs/oper books only have schema=0"

        self.max_samps_per_frame = max_samps_per_frame
        self.max_file_size = max_file_size
        self.ignore_tags = ignore_tags
        self.allow_bad_timing = allow_bad_timing

        if os.path.exists(outdir):
            # don't count hidden files, possibly from NFS processes
            nfiles = len([f for f in os.listdir(outdir) if f[0] != '.'])
            if nfiles > 1:
                raise BookDirHasFiles(
                    f"Output directory {outdir} contains files. Delete to retry"
                      " bookbinding"
                )

            elif nfiles == 1:
                assert os.listdir(outdir)[0] == 'Z_bookbinder_log.txt', \
                    f"only acceptable file in new book path {outdir} is " \
                    " Z_bookbinder_log.txt"

        else:
            os.makedirs(outdir)

        logfile = os.path.join(outdir, 'Z_bookbinder_log.txt')
        self.log = setup_logger(logfile)

        try:
            self.hkfiles = get_hk_files(
                self.hk_root, 
                book.start.timestamp(),
                book.stop.timestamp()
            )
        except NoHKFiles as e:
            if require_hwp or require_acu:
                self.log.error(
                    "HK files are required if we require ACU or HWP data"
                )
                raise e
            self.log.warning(
                "Found no HK files during book time, binding anyway because "
                "require_acu and require_hwp are False"
            )
            self.hkfiles = []            

        self.ancil = AncilProcessor(
            self.hkfiles, 
            book.bid, 
            hk_fields,
            log=self.log, 
            drop_duplicates=ancil_drop_duplicates,
            require_hwp=require_hwp,
            require_acu=require_acu,
            require_monotonic_times=require_monotonic_times,
        )
        self.streams = {}
        for obs_id, files in filedb.items():
            obs = self.obsdb[obs_id]
            stream_id = obs.stream_id
            if not obs.timing and not self.allow_bad_timing:
                raise TimingSystemOff(
                    f"Observation {obs_id} does not have high precision timing "
                    "information. Pass `allow_bad_timing=True` to bind anyway"
                )
            self.streams[stream_id] = SmurfStreamProcessor(
                obs_id, files, book.bid, readout_ids[obs_id], log=self.log,
                allow_bad_timing=self.allow_bad_timing,
            )

        self.min_ctime = min_ctime
        self.max_ctime = max_ctime
        self.times = None
        self.frame_idxs = None
        self.file_idxs = None
        self.meta_files = None
        
    def preprocess(self):
        """
        Runs preprocessing steps for the book. Preprocesses smurf and ancillary
        data. Creates full list of book-times, the output frame idx for each
        sample, and the output file idx for each frame.
        """
        if self.times is not None:
            return

        for stream in self.streams.values():
            stream.preprocess()

        t0 = np.max([s.times[0] for s in self.streams.values()])
        if self.min_ctime is not None:
            assert self.min_ctime >= t0, \
                f"{self.min_ctime} is less than the first time found in"\
                f" the detector data {t0}"
            self.log.warning(
                f"Over-riding minimum ctime from {t0} to {self.min_ctime}"
            )
            t0 = self.min_ctime
        else:
            self.min_ctime = t0
        t1 = np.min([s.times[-1] for s in self.streams.values()])
        if self.max_ctime is not None:
            assert self.max_ctime <= t1, \
                f"{self.max_ctime} is greater than the last time found in"\
                f" the detector data {t1}"
            self.log.warning(
                f"Over-riding maximum ctime from {t1} to {self.max_ctime}"
            )
        else:
            self.max_ctime = t1
        # prioritizes the last stream
        # implicitly assumes co-sampled (this is where we could throw errors
        # after looking for co-sampled data)
        ts, _ = fill_time_gaps(stream.times) 
        m = (t0 <= ts) & (ts <= t1)
        ts = ts[m]

        self.ancil.preprocess()

        # Divide up frames, only look within detector data and +/-30 seconds
        frame_splits = find_frame_splits(self.ancil, ts[0]-30, ts[-1]+30)

        if frame_splits is None:
            frame_idxs = np.arange(len(ts)) // self.max_samps_per_frame
        else:
            frame_idxs = np.digitize(ts, frame_splits)
            frame_idxs -= frame_idxs[0]
            new_frame_idxs = frame_idxs.copy()
            # Divide up frames that are too long
            for fidx in np.unique(frame_idxs):
                m = frame_idxs == fidx
                new_frame_idxs += np.cumsum(m) // self.max_samps_per_frame
            frame_idxs = new_frame_idxs

        # Divide up files
        samp_size = 4 # bytes
        max_chans = np.max([s.nchans for s in self.streams.values()])
        totsize = samp_size * max_chans * np.arange(len(ts))
        file_idxs = []
        for fr in np.unique(frame_idxs):
            idx = np.where(frame_idxs == fr)[0][-1]
            file_idxs.append(totsize[idx] // self.max_file_size)
        file_idxs = np.array(file_idxs, dtype=int)

        self.times = ts
        self.frame_idxs = frame_idxs
        self.file_idxs = file_idxs

        self.check_timesamples()
        self.log.info("Finished preprocessing data")

    def check_timesamples(self, atol=1e-4):
        """
        Checks for missing timesamples in individual streams relative to the 
        book times. Makes sure individual readout slots haven't dropped too many
        points
        """
        if self.times is None:
            raise ValueError(
                "Preprocess must have been run to check_timesamples"
            )
        
        self.dropped = {}
        for u, s in self.streams.items():
            sample_map = find_ref_idxs(self.times, s.times)
            mapped = np.abs(self.times[sample_map] - s.times) < atol
            diffs = np.diff(sample_map[mapped])
            idx = np.where( diffs>1)[0]
            self.dropped[u] = sum( [diffs[i]-1 for i in idx] )
        
        if np.all( [x==0 for x in self.dropped.values()] ):
            ## no dropped samples from any slot
            return
        msg = '\n'.join([
            f"\t{self.streams[u].obs_id}: {x}" for u, x in self.dropped.items()
        ])
        if np.any( [x>MAX_DROPPED_SAMPLES for x in self.dropped.values()]):
            if (not self.allow_bad_timing):
                raise BadTimeSamples(
                    f"Streams have more than {MAX_DROPPED_SAMPLES} time samples"
                    f" missing. Pass `allow_bad_timing=True` to bind anyway. "
                    "Missing samples:\n" + msg
                )
            else:
                self.log.warning(
                    f"Streams have more than {MAX_DROPPED_SAMPLES} time samples"
                    f" missing. Missing Samples: \n" + msg
                )
        else:
            self.log.warning(
                f"Streams have time samples missing. Missing Samples: \n" + msg
            )


    def copy_smurf_files_to_book(self):
        """
        Copies smurf ancillary files to an operation book.
        """
        if self.book.type != 'oper':
            return

        self.log.info("Copying smurf ancillary files to book")

        files = []
        for obs in self.obsdb.values():
            files.extend(get_smurf_files(obs, self.meta_root))
        
        smurf_dirname = 'Z_smurf'
        os.makedirs(os.path.join(self.outdir, smurf_dirname), exist_ok=True)

        meta_files = {}
        for f in files:
            relpath = os.path.join(smurf_dirname, os.path.basename(f))
            dest = os.path.join(self.outdir, relpath)
            self.log.info(f"Copying to {dest}")
            shutil.copyfile(f, dest)

            if f.endswith('iv_analysis.npy'):
                meta_files['iv'] = relpath
            elif f.endswith('bg_map.npy'):
                meta_files['bgmap'] = relpath
            elif f.endswith('bias_step_analysis.npy'):
                meta_files['bias_steps'] = relpath
            elif f.endswith('take_noise.npy'):
                meta_files['noise'] = relpath

        self.meta_files = meta_files

    def write_M_files(self, telescope, tube_config):
        # write M_book file
        m_book_file = os.path.join(self.outdir, "M_book.yaml")
        book_meta = {}
        book_meta["book"] = {
            "type": self.book.type,
            "schema_version": self.book.schema,
            "book_id": self.book.bid,
            "finalized_at": dt.datetime.utcnow().isoformat(),
        }
        book_meta["bookbinder"] = {
            "codebase": sotodlib.__file__,
            "version": sotodlib.__version__,
            # leaving this in but KH doesn't know what it's supposed to be for
            "context": "unknown", 
        }
        with open(m_book_file, "w") as f:
            yaml.dump(book_meta, f)
        
        mfile = os.path.join(self.outdir, "M_index.yaml")
        with open(mfile, "w") as f:
            yaml.dump(
                self.get_metadata(
                    telescope=telescope,
                    tube_config=tube_config,
                ), f
            )

    def get_metadata(self, telescope=None, tube_config={}):
        """
        Returns metadata dict for the book
        """
        self.preprocess()

        meta = {}
        meta['book_id'] = self.book.bid
        meta['type'] = self.book.type

        meta['start_time'] = float(self.times[0])
        meta['stop_time'] = float(self.times[-1])
        meta['n_frames'] = len(np.unique(self.frame_idxs))
        meta['n_samples'] = len(self.times)
        meta['session_id'] = self.book.bid.split('_')[1]
        meta['filled_samples'] = {k:int(x) for k,x in self.dropped.items()}

        sample_ranges = []
        for file_idx in np.unique(self.file_idxs):
            fr_idxs = np.where(self.file_idxs == file_idx)[0]
            i0 = int(np.where(self.frame_idxs == fr_idxs[0])[0][0])
            i1 = int(np.where(self.frame_idxs == fr_idxs[-1])[0][-1])
            sample_ranges.append([i0, i1+1])
        meta['sample_ranges'] = sample_ranges

        if telescope is None:
            self.log.warning(
                "telescope not explicitly defined. guessing from book"
            )
            meta['telescope'] = self.book.tel_tube[:3].lower()
        else: 
            meta['telescope'] = telescope

        if 'tube_slot' not in tube_config:
            self.log.warning("tube_slot key missing from tube_config. guessing")
        meta['tube_slot'] = tube_config.get(
            'tube_slot',
            self.book.tel_tube.lower().replace("sat","satst")[3:]
        )
        meta['tube_flavor'] = tube_config.get('tube_flavor')
        meta['wafer_slots'] = tube_config.get('wafer_slots')

        detsets = []
        tags = []

        # build detset list in same order as slots
        meta['stream_ids'] = self.book.slots.split(',')
        for sid in meta['stream_ids']:
            detsets.append(
                [obs.tunesets[0].name for _,obs in self.obsdb.items() 
                    if obs.stream_id == sid ][0]
            )
        # just append all tags, order doesn't matter
        for _, g3tobs in self.obsdb.items():
            tags.append(g3tobs.tag)
        meta['detsets'] = detsets

        hwp_freq = self.ancil.hkdata.hwp_freq
        meta['hwp_freq_mean'] = None
        meta['hwp_freq_stdev'] = None
        t0, t1 = self.times[0], self.times[-1]
        if hwp_freq is not None:
            m = (t0 < hwp_freq.times) & (hwp_freq.times < t1)
            if m.any():
                meta['hwp_freq_mean'] = float(np.mean(hwp_freq.data[m]))
                meta['hwp_freq_stdev'] = float(np.std(hwp_freq.data[m]))
        
        az = self.ancil.hkdata.az
        meta['az_speed_mean'] = None
        meta['az_speed_stdev'] = None
        if az is not None:
            m = (t0 < az.times) & (az.times <= t1)
            if np.sum(m) >= 2:
                dt = np.diff(az.times[m]).mean()
                az_speed = np.abs(np.diff(az.data[m]) / dt)
                meta['az_speed_mean'] = float(np.mean(az_speed))
                meta['az_speed_stdev'] = float(np.std(az_speed))

        # make sure all tags are the same for obs in the same book
        tags = list(set(tags))
        if not self.ignore_tags:
            assert len(tags) == 1
        else:
            tags = [','.join(tags)]
        tags = tags[0].split(',')
        # book should have at least one tag
        assert len(tags) > 0
        meta['subtype'] = tags[1] if len(tags) > 1 else ""
        # sanitize rest of tags
        meta['tags'] = [t.strip() for t in tags[2:] if t.strip() != '']
        
        if (self.book.type == 'oper') and self.meta_files:
            meta['meta_files'] = self.meta_files
        return meta

    def bind(self, pbar=False):
        """
        Binds data.

        Params
        ---------
        pbar : bool
            If True, will enable a progress bar.
        """
        self.preprocess()
        
        self.log.info(f"Binding data to {self.outdir}")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Copy smurf ancillary files if they exist
        self.copy_smurf_files_to_book()

        # Bind Ancil Data
        self.ancil.bind(self.outdir, self.times, self.frame_idxs, self.file_idxs)
        
        tot = np.sum([s.nframes for s in self.streams.values()])
        pbar = tqdm(total=tot, disable=(not pbar))
        for stream in self.streams.values():
            stream.bind(self.outdir, self.times, self.frame_idxs,
                        self.file_idxs, pbar=pbar, ancil=self.ancil)

        self.log.info("Finished binding data. Exiting.")
        return True

class TimeCodeBinder:
    """Class for building the timecode based books, smurf, stray, and hk books. 
    These books are built primarily just by copying specified files from level 
    2 locations to new locations at level 2.
    """

    def __init__(
        self, book, timecode, indir, outdir, file_list=None, 
        ignore_pattern=None,
    ):
        self.book = book
        self.timecode = timecode
        self.indir = indir
        self.outdir = outdir
        self.file_list = file_list
        if ignore_pattern is not None:
            self.ignore_pattern = ignore_pattern
        else:
            self.ignore_pattern = []
        
        if book.type == 'smurf' and book.schema > 0:
            self.compress_output = True
        else:
            self.compress_output = False    

    def get_metadata(self, telescope=None, tube_config={}):
        return {
            "book_id": self.book.bid,
            # dummy start and stop times
            "start_time": float(self.timecode) * 1e5,
            "stop_time": (float(self.timecode) + 1) * 1e5,
            "telescope": telescope,
            "type": self.book.type,
        }
    
    def write_M_files(self, telescope, tube_config):
        # write M_book file
        
        book_meta = {}
        book_meta["book"] = {
            "type": self.book.type,
            "schema_version": self.book.schema,
            "book_id": self.book.bid,
            "finalized_at": dt.datetime.utcnow().isoformat(),
        }
        book_meta["bookbinder"] = {
            "codebase": sotodlib.__file__,
            "version": sotodlib.__version__,
            # leaving this in but KH doesn't know what it's supposed to be for
            "context": "unknown", 
        }
        if self.compress_output:
            with ZipFile(self.outdir, mode='a') as zf:
                zf.writestr("M_book.yaml", yaml.dump(book_meta))
        else:
            m_book_file = os.path.join(self.outdir, "M_book.yaml")
            with open(m_book_file, "w") as f:
                yaml.dump(book_meta, f)
        
        index = self.get_metadata(
            telescope=telescope,
            tube_config=tube_config,
        )
        if self.compress_output:
            with ZipFile(self.outdir, mode='a') as zf:
                zf.writestr("M_index.yaml", yaml.dump(index))
        else:
            mfile = os.path.join(self.outdir, "M_index.yaml")
            with open(mfile, "w") as f:
                yaml.dump(index, f)

    def bind(self, pbar=False):
        if self.compress_output:
            if self.file_list is None:
                self.file_list = walk_files(self.indir, include_suprsync=True)
                ignore = shutil.ignore_patterns(*self.ignore_pattern)
                to_ignore = ignore("", self.file_list)
                self.file_list = sorted(
                    [f for f in self.file_list if f not in to_ignore]
                )
            with ZipFile(self.outdir, mode='x') as zf:
                for f in self.file_list:
                    relpath = os.path.relpath(f, self.indir)
                    zf.write(f, arcname=relpath, compress_type=ZIP_DEFLATED)
        elif self.file_list is None:
            shutil.copytree(
                self.indir,
                self.outdir,
                ignore=shutil.ignore_patterns(
                    *self.ignore_pattern,
                ),
            )
        else:
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            for f in self.file_list:
                relpath = os.path.relpath(f, self.indir)
                path = os.path.join(self.outdir, relpath)
                base, _ = os.path.split(path)
                if not os.path.exists(base):
                    os.makedirs(base)
                shutil.copy(f, os.path.join(self.outdir, relpath))

def fill_time_gaps(ts):
    """
    Fills gaps in an array of timestamps.

    Parameters
    -------------
    ts : np.ndarray
        List of timestamps of length `n`, potentially with gaps
    
    Returns
    --------
    new_ts : np.ndarray
        New list of timestamps of length >= n, with gaps filled.
    mask : np.ndarray
        Returns a mask that tells you which elements of `new_ts` are
        taken from real data and which are interpolated. `~mask` gives
        the indices of the interpolated elements.
    """
    # Find indices where gaps occur and how long each gap is
    dts = np.diff(ts)
    dt = np.median(dts)
    missing = np.round(dts/dt - 1).astype(int)
    total_missing = int(np.sum(missing))

    # Create new array with the correct number of samples
    new_ts = np.full(len(ts) + total_missing, np.nan)

    # Insert old timestamps into new array with offsets that account for gaps
    offsets = np.concatenate([[0], np.cumsum(missing)])
    i0s = np.arange(len(ts))
    new_ts[i0s + offsets] = ts

    # Use existing data to interpolate and fill holes
    m = np.isnan(new_ts)
    xs = np.arange(len(new_ts))
    interp = interp1d(xs[~m], new_ts[~m])
    new_ts[m] = interp(xs[m])

    return new_ts, ~m

_primary_idx_map = {}
def get_frame_times(frame, allow_bad_timing=False):
    """
    Returns timestamps for a G3Frame of detector data.

    Parameters
    --------------
    frame : G3Frame
        Scan frame containing detector data
    allow_bad_timing: bool, optional
        if not true, raises an error if it finds data with imprecise timing

    Returns
    --------------
    high_precision : bool
        If true, timestamps are computed from timing counters. If not, they are
        software timestamps
    
    timestamps : np.ndarray
        Array of timestamps (sec) for samples in the frame

    """
    if len(_primary_idx_map) == 0:
        for i, name in enumerate(frame['primary'].names):
            _primary_idx_map[name] = i
        
    c0 = frame['primary'].data[_primary_idx_map['Counter0']]
    c2 = frame['primary'].data[_primary_idx_map['Counter2']]

    counters = np.all( np.diff(c0)!=0 ) and np.all( np.diff( c2 )!=0)

    if counters:
        return True, counters_to_timestamps(c0, c2)
    elif allow_bad_timing:
        return False, np.array(frame['data'].times) / core.G3Units.s
    else:
        ## don't change this error message. used in Imprinter CLI
        raise TimingSystemOff("Timing counters not incrementing")

def split_ts_bits(c):
    """
    Split up 64 bit to 2x32 bit
    """
    NUM_BITS_PER_INT = 32
    MAXINT = (1 << NUM_BITS_PER_INT) - 1
    a = (c >> NUM_BITS_PER_INT) & MAXINT
    b = c & MAXINT
    return a, b

def counters_to_timestamps(c0, c2):
    s, ns = split_ts_bits(c2)

    # Add 20 years in seconds (accounting for leap years) to handle
    # offset between EPICS time referenced to 1990 relative to UNIX time.
    c2 = s + ns*1e-9 + 5*(4*365 + 1)*24*60*60
    ts = np.round(c2 - (c0 / 480000) ) + c0 / 480000
    return ts

def find_ref_idxs(refs, vs):
    """
    Creates a mapping from a list of timestamps (vs) to a list of reference
    timestamps (refs). Returns an index-map of shape `vs.shape`, that maps each
    timestamp of the array `vs` to the closest timestamp in the array `refs`.

    This assumes that `refs` and `vs` are sorted in ascending order.

    Parameters
    ----------
    refs : array_like
        List of reference timestamps
    vs : array_like
        List of timestamps

    Returns
    -------
    idxs : array_like
        Map of shape `vs.shape` that maps each element of `vs` to the closest
        element in `refs`.
    """
    # Find the indices of the samples in the list of timestamps (vs)
    # that are closest to the reference timestamps
    idx = np.searchsorted(refs, vs, side='left')
    idx = np.clip(idx, 1, len(refs)-1)
    # shift indices to the closest sample
    left = refs[idx-1]
    right = refs[idx]
    idx -= vs - left < right - vs
    return idx

def get_hk_files(hkdir, start, stop, tbuff=10*60):
    """
    Gets HK files for dat between start and stop dirHWP files that may have data
    between start and stop time
    """
    files = []
    for subdir in os.listdir(hkdir):
        try:
            tcode = int(subdir)
        except:
            continue

        if not start//1e5 - 1 <= tcode <= stop//1e5 + 1:
            continue

        subpath = os.path.join(hkdir, subdir)
        files.extend([os.path.join(subpath, f) for f in os.listdir(subpath)])

    files = np.array(sorted(files))
    file_times = np.array(
        [int(os.path.basename(f).split('.')[0]) for f in files]
    )

    m = (start-tbuff <= file_times) & (file_times < stop+tbuff)
    if not np.any(m):
        check = np.where( file_times <= start )
        if len(check) < 1 or len(check[0]) < 1:
            raise NoHKFiles(
                f"Cannot find HK files between {start} and {stop}"
            )
        fidxs = [check[0][-1]]
        m[fidxs] = 1
    else:
        fidxs = np.where(m)[0]
    # Add files before and after for good measure
    
    i0, i1 = fidxs[0], fidxs[-1]
    if i0 > 0:
        m[i0 - 1] = 1
    if i1 < len(m) - 1:
        m[i1 + 1] = 1

    return files[m].tolist()

def locate_scan_events(
        times, az, 
        vel_thresh=0.01, # mount noise for satps is 0.015 deg/s level
        min_gap=200, 
        filter_window=100
    ):
    """
    Locate places where the azimuth velocity changes sign, including starts and
    stops. These locations are where we should start determining the scan 
    framing.

    Parameters
    ----------
    times : ndarray float
        times 
    az: ndarray float
        azimuth positions
    vel_thresh : float, optional
        threshold for what is considered stopped
    min_gap : int, optional
        Length of gap (in samples) longer than which events are considered separate

    Returns
    -------
    events : list
        Full list containing all zero-crossings, starts, and stops that should become frame edges
    """

    if len(az) < 1:
        return []

    offset = 0
    vel = np.diff(az)/np.diff(times)

    if filter_window is not None:
        win = np.hanning(filter_window) / np.sum(np.hanning(filter_window))
        vel = convolve(vel, win, mode='same')[filter_window:-filter_window]
        offset = filter_window

    ## find places with "zero" velocity
    zero_vel = np.abs(vel) < vel_thresh
    if np.all(zero_vel):
        return []
    
    zeros = Ranges.from_mask(zero_vel)
    zeros.close_gaps(min_gap)
    
    ## find places where velocity changes sign in case the velocity
    ## is so fast it never gets close enough to zero
    x = np.where( np.sign(vel) > 0 )[0]
    y = np.where( np.diff(x) > 1 )[0]
    cross = Ranges.zeros_like(zeros)
    for z in y:
        cross.add_interval( x[z]+1, x[z]+2)
    x = np.where( np.sign(vel) < 0 )[0]
    y = np.where( np.diff(x) > 1 )[0]
    for z in y:
        cross.add_interval( x[z]+1, x[z]+2)
    
    zeros = zeros + cross
    events = []
    
    for c in zeros.ranges():   
        # if zero period is longer than min_gap, it's a start or stop add each side to the list
        if c[1] - c[0] > min_gap:
            if c[0] != 0:
                events.append( c[0] )
            if c[1] != len(vel):
                events.append( c[1] )
        # otherwise, it's a zero crossing, add mean
        else:
            events.append( int(round( sum(c)/2 )) )
    
    return np.array(events, dtype='int')+offset

def find_frame_splits(ancil, t0=None, t1=None):
    """
    Determines timestamps of frame-splits from ACU data. If it cannot determine
    frame-splits, returns None.

    Arguments
    ----------
    ancil: AncillaryProcesser
    t0: float (optional)
        start time to analyze ACU behavior 
    t1: float (optional)
        stop time to analyze ACU behavior
    """
    az = ancil.hkdata.az
    if az is None:
        return None

    if t0 is None:
        t0 = az.times[0]
    if t1 is None:
        t1 = az.times[-1]

    msk = np.all(
        [az.times >= t0, az.times <= t1],
        axis=0
    )
    idxs = locate_scan_events(az.times[msk], az.data[msk], filter_window=100)
    return az.times[msk][idxs]

def get_smurf_files(obs, meta_path, all_files=False):
    """
    Returns a list of smurf files that should be copied into a book.

    Parameters
    ------------
    obs : G3tObservations
        Observation to pull files from
    meta_path : path
        Smurf metadata path
    all_files : bool
        If true will return all found metadata files

    Returns
    -----------
    files : List[path]
        List of copyable files
    """

    def copy_to_book(file):
        if all_files:
            return True
        return file.endswith('npy')

    tscode = int(obs.action_ctime//1e5)
    files = []

    # check adjacent folders in case action falls on a boundary
    for tc in [tscode-1, tscode, tscode + 1]:
        action_dir = os.path.join(
            meta_path,
            str(tc),
            obs.stream_id,
            f'{obs.action_ctime}_{obs.action_name}'
        )

        if not os.path.exists(action_dir):
            continue

        for root, _, fs in os.walk(action_dir):
            files.extend([os.path.join(root, f) for f in fs])

    return [f for f in files if copy_to_book(f)]

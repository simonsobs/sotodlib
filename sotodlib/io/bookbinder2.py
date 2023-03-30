import so3g
from spt3g import core
import itertools
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import os
import yaml
import logging

from sotodlib.site_pipeline.util import init_logger


log = logging.getLogger('bookbinder')
if not log.hasHandlers():
    log = init_logger('bookbinder')


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


class AncilProcessor:
    """
    Processor for ancillary (ACU) data
    
    Params
    --------
    files : list
        List of HK files to process

    Attributes
    -----------
    data : dict
        Dict containing ancillary data timestreams read from HK files. This
        will be populated after preprocess.
    times : np.ndarray
        Timestamps for anc data. This will be populated after preprocess.
    anc_frame_data : List[G3TimestreamMap]
        List of G3TimestreamMaps saved for each bound frame. This will be
        populated on bind and should be used to add copies of the anc data to
        the detector frames. 
    """
    def __init__(self, files):
        self.files = files
        self.times = None
        self.data = None
        self.anc_frame_data = None
    
    def preprocess(self):
        """
        Preprocesses HK data and populates the `data` and `times` objects.
        """
        log.info("Preprocessing HK Data")
        frame_iter = get_frame_iter(self.files)

        data = {}
        times = []
        block_idx = None
        for fr in frame_iter:
            if fr['hkagg_type'] != 2:
                continue
            if 'ACU_broadcast' not in fr['block_names']:
                continue
            if block_idx is None:
                block_idx = list(fr['block_names']).index('ACU_broadcast')

            block = fr['blocks'][block_idx]
            times.append(np.array(block.times) / core.G3Units.s)
            for k, v in block.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        
        if not times:
            times = np.array([], dtype=np.float64)
            data = {}
        else:
            times = np.hstack(times)
            for k, v in data.items():
                data[k] = np.hstack(v)

        self.data = data
        self.times = times
    
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
        log.info("Binding ancilary data")

        # Handle file writers
        writer = None
        cur_file_idx = None

        anc_frame_data = []
        for oframe_idx in np.unique(frame_idxs):
            if file_idxs[oframe_idx] != cur_file_idx:
                close_writer(writer)
                cur_file_idx = file_idxs[oframe_idx]
                fname = f'A_ancil_{cur_file_idx:0>3}.g3'
                writer = core.G3Writer(os.path.join(outdir, fname))

            m = np.where(frame_idxs == oframe_idx)
            ts = times[m]

            oframe = core.G3Frame(core.G3FrameType.Scan)
            anc_data = core.G3TimesampleMap()
            anc_data.times = core.G3VectorTime(ts * core.G3Units.s)
            oframe['ancil'] = anc_data
            writer(oframe)
            anc_frame_data.append(anc_data)

        # Save this to be added to detector files
        self.anc_frame_data = anc_frame_data
        

class SmurfStreamProcessor:
    def __init__(self, obs_id, files):
        self.files = files
        self.obs_id = obs_id
        self.stream_id = None
        self.times = None
        self.frame_idxs = None
        self.nchans = None
        self.nframes = None
        self.chan_names = None
        self.bias_names = None
        self.primary_names = None
        self.timing_paradigm = None
        self.session_id = None
        self.slow_primary = None
        self.sostream_version = None

    def preprocess(self):
        """
        Extracts file times, nchans, and nnframes from file list
        """
        if self.times is not None:  # Already preprocessed
            return

        log.info(f"Preprocessing smurf obsid {self.obs_id}")

        self.nframes = 0
        ts = []
        frame_idxs = []
        frame_idx = 0
        for frame in self.frame_iter():
            if frame.type != core.G3FrameType.Scan:
                continue

            if self.nchans is None:
                self.chan_names = frame['data'].names
                self.nchans = len(self.chan_names)
                self.primary_names = frame['primary'].names
                self.bias_names = frame['tes_biases'].names
                self.timing_paradigm = frame['timing_paradigm']
                self.session_id = frame['session_id']
                self.slow_primary = frame['slow_primary']
                self.sostream_version = frame['sostream_version']
                self.stream_id = frame['sostream_id']

            t = get_frame_times(frame)[1]
            ts.append(t)
            frame_idxs.append(np.full(len(t), frame_idx, dtype=np.int))

            self.nframes += 1
            frame_idx += 1

        self.times = np.hstack(ts)
        self.frame_idxs = np.hstack(frame_idxs)
        self.preprocessed = True

    def bind(self, outdir, times, frame_idxs, file_idxs, pbar=False, ancil=None):
        if pbar is True:
            pbar = tqdm(total=self.nframes)
        elif pbar is False:
            pbar = tqdm(total=self.nframes, disable=True)
        pbar.set_description(f"Binding {self.stream_id}")

        log.info(f"Binding smurf obsid {self.obs_id}")

        # Mapping to input sample idx to output sample idx
        atol = 0.001
        out_sample_map = find_ref_idxs(times, self.times)
        mapped = np.abs(times[out_sample_map] - self.times) < atol
        _, out_frame_offsets = np.unique(frame_idxs, return_index=True)
        # Map from input sample idx to output frame idx
        oframe_idxs = frame_idxs[out_sample_map]
        oframe_idxs[~mapped] = -1 

        iframe_idxs = self.frame_idxs

        # Map from input sample idx to output sample idx relative to the out frame it belongs to
        out_offset_samp_idxs = out_sample_map - out_frame_offsets[frame_idxs[out_sample_map]]

        _, offsets = np.unique(self.frame_idxs, return_index=True)
        in_offset_samp_idxs = np.arange(len(self.times)) - offsets[self.frame_idxs]
        # Handle file writers
        writer = None
        cur_file_idx = None

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
                fname = f'D_{self.stream_id}_{cur_file_idx:0>3}.g3'
                writer = core.G3Writer(os.path.join(outdir, fname))

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

                # The fastest way to copy data into a numpy array is to use
                # direct slicing like ``arr_out[o0:o1] = arr_in[i0:i1]`` since
                # it doesn't need to create a temporary copy of the array, and
                # can just do a direct mem-map. Doing this speeds up the binding
                # by a factor of 3-4x compared to other methods I've tried.
                outsamps = out_offset_samp_idxs[m]
                insamps = in_offset_samp_idxs[m]

                split_idxs = 1 + np.where((np.diff(outsamps) > 1) \
                                        & (np.diff(insamps) > 1))[0]
                outsplits = np.split(outsamps, split_idxs)
                insplits = np.split(insamps, split_idxs)
                for i in range(len(outsplits)):
                    in0, in1 = insplits[i][0], insplits[i][-1] + 1
                    out0, out1 = outsplits[i][0], outsplits[i][-1] + 1

                    data[:, out0:out1] = iframe['data'].data[:, in0:in1]
                    biases[:, out0:out1] = iframe['tes_biases'].data[:, in0:in1]
                    primary[:, out0:out1] = iframe['primary'].data[:, in0:in1]
                    filled[out0:out1] = 1

                # If there are any remaining samples in the next in_frame, pull it and repeat
                if np.any((oframe_idxs == oframe_idx) & (iframe_idxs == iframe_idx + 1)):
                    iframe, interm_frames = next_scan(inframe_iter)
                    iframe_idx += 1
                    pbar.update()
                    continue
                else:
                    break
            
            print(np.sum(filled)-len(filled))
            oframe = core.G3Frame(core.G3FrameType.Scan)
            ts = core.G3VectorTime(ts * core.G3Units.s)
            oframe['data'] = so3g.G3SuperTimestream(self.chan_names, ts, data)
            oframe['primary'] = so3g.G3SuperTimestream(self.primary_names, ts, primary)
            oframe['tes_biases'] = so3g.G3SuperTimestream(self.bias_names, ts, biases)
            oframe['timing_paradigm'] = self.timing_paradigm
            oframe['session_id'] = self.session_id
            oframe['slow_primary'] = self.slow_primary
            oframe['sostream_version'] = self.sostream_version
            oframe['sostream_id'] = self.stream_id
            oframe['frame_num'] = oframe_num
            oframe['num_samples'] = len(ts)
            if ancil is not None:
                oframe['ancil'] = ancil.anc_frame_data[oframe_idx]

            oframe_num += 1

            writer(oframe)
            
        close_writer(writer)
        if pbar.n >= pbar.total:
            pbar.close()


class BookBinder:
    def __init__(self, book, obsdb, filedb, hkfiles, max_samps_per_frame=10_000):
        self.filedb = filedb
        self.book = book
        self.hkfiles = hkfiles
        self.obsdb = obsdb

        self.max_samps_per_frame = max_samps_per_frame
        self.ancil = AncilProcessor(hkfiles)
        self.streams = {}
        for obs_id, files in filedb.items():
            stream_id = '_'.join(obs_id.split('_')[1:-1])
            self.streams[stream_id] = SmurfStreamProcessor(obs_id, files)

        self.times = None
        self.frame_idxs = None
        self.file_idxs = None
        
    def _get_full_times(self):
        for stream in self.streams.values():
            stream.preprocess()

        t0 = np.max([s.times[0] for s in self.streams.values()])
        t1 = np.min([s.times[-1] for s in self.streams.values()])

        mask, times = None, None
        for x in self.streams.values():
            ts, filled = fill_time_gaps(x.times)
            m = (t0 < ts) & (ts < t1)
            if times is None:
                times = ts[m]
                mask = filled[m]
            else:
                gaps = ~mask
                times[gaps] = ts[m][gaps]
                mask[gaps] = filled[m][gaps]

        return times, mask
    
    def preprocess(self):
        if self.times is not None:
            return

        for stream in self.streams.values():
            stream.preprocess()
        self.ancil.preprocess()

        times, _ = self._get_full_times()

        # Divide up frames
        nsamps = len(times)
        frame_idxs = np.arange(nsamps) // self.max_samps_per_frame

        nframes = len(np.unique(frame_idxs))
        file_idxs = [0 for _ in range(nframes)]

        self.times = times
        self.frame_idxs = frame_idxs
        self.file_idxs = file_idxs

    def get_metadata(self):
        self.preprocess()

        meta = {}
        meta['book_id'] = self.book.bid
        meta['start_time'] = float(self.times[0])
        meta['end_time'] = float(self.times[-1])
        meta['n_frames'] = len(np.unique(self.frame_idxs))
        meta['n_samples'] = len(self.times)
        meta['session_id'] = self.book.bid.split('_')[1]

        sample_ranges = []
        for file_idx in np.unique(self.file_idxs):
            fr_idxs = np.where(self.file_idxs == file_idx)[0]
            i0 = int(np.where(self.frame_idxs == fr_idxs[0])[0][0])
            i1 = int(np.where(self.frame_idxs == fr_idxs[-1])[0][-1])
            sample_ranges.append([i0, i1+1])
        meta['sample_ranges'] = sample_ranges

        meta['telescope'] = self.book.tel_tube[:3].lower()
        # parse e.g., sat1 -> st1, latc1 -> c1
        meta['tube_slot'] = self.book.tel_tube.lower().replace("sat","satst")[3:]
        meta['type'] = self.book.type
        detsets = []
        tags = []
        for _, g3tobs in self.obsdb.items():
            detsets.append(g3tobs.tunesets[0].name)
            tags.append(g3tobs.tag)
        meta['detsets'] = detsets
        # make sure all tags are the same for obs in the same book
        tags = list(set(tags))
        assert len(tags) == 1
        tags = tags[0].split(',')
        # book should have at least one tag
        assert len(tags) > 0
        meta['subtype'] = tags[1] if len(tags) > 1 else ""
        meta['tags'] = tags[2:]
        meta['stream_ids'] = self.book.slots.split(',')
        return meta

    def bind(self, outdir, pbar=False):
        """
        Binds 
        """
        self.preprocess()

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Write M_file
        mfile = os.path.join(outdir, 'M_index.yaml')
        with open(mfile, 'w') as f:
            yaml.dump(self.get_metadata(), f)

        # Bind Ancil Data
        self.ancil.bind(outdir, self.times, self.frame_idxs, self.file_idxs)
        
        tot = np.sum([s.nframes for s in self.streams.values()])
        pbar = tqdm(total=tot, disable=(not pbar))
        for stream in self.streams.values():
            stream.bind(outdir, self.times, self.frame_idxs, self.file_idxs,
                        pbar=pbar, ancil=self.ancil)


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
    """
    # Find indices where gaps occur and how long each gap is
    dts = np.diff(ts)
    dt = np.median(dts)
    missing = np.round(dts/dt - 1).astype(int)
    total_missing = int(np.sum(missing))

    # Create new  array with the correct number of samples
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
def get_frame_times(frame):
    """
    Returns timestamps for a G3Frame of detector data.

    Parameters
    --------------
    frame : G3Frame
        Scan frame containing detector data

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

    if np.any(c0):
        return True, counters_to_timestamps(c0, c2)
    else:
        return False, np.array(frame['data'].times) / core.G3Units.s


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
    # offset between EPOCH time referenced to 1990 relative to UNIX time.
    c2 = s + ns*1e-9 + 5*(4*365 + 1)*24*60*60
    ts = np.round(c2 - (c0 / 480000) ) + c0 / 480000
    return ts


def find_ref_idxs(refs, vs, atol=0.5):
    """
    Find missing samples in a list of timestamps (vs) given a list of
    reference timestamps (refs). The reference timestamps are assumed
    to be the "true" timestamps, and the list of timestamps (vs) are
    assumed to be missing some of the samples. The function returns
    the missing samples in the list of timestamps (vs).

    Parameters
    ----------
    refs : array_like
        List of reference timestamps
    vs : array_like
        List of timestamps
    atol : float
        Absolute tolerance for the difference between the reference

    Returns
    -------
    i_missing : array_like
        indices (in refs) of the missing samples in vs
    t_missing : array_like
        values (in refs) of the missing samples in vs
    """
    # return
    # Find the indices of the samples in the list of timestamps (vs)
    # that are closest to the reference timestamps
    idx = np.searchsorted(refs, vs, side='left')
    idx = np.clip(idx, 1, len(refs)-1)
    # shift indices to the closest sample
    left = refs[idx-1]
    right = refs[idx]
    idx -= vs - left < right - vs
    return idx

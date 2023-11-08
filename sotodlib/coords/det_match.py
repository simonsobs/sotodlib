import warnings
from dataclasses import dataclass, fields, asdict
from typing import List, Optional, Tuple, Iterator
from copy import deepcopy
import h5py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from sotodlib.coords import optics
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset



def map_band_chans(b1, c1, b2, c2, chans_per_band=512):
    """
    Returns an index mapping of length nchans1 from one set of bands and
    channels to another. Note that unmapped indices are returned as -1, so this
    must be handled before (or after) indexing or else you'll get weird
    results.
    Args:
        b1 (np.ndarray):
            Array of length nchans1 containing the smurf band of each channel
        c1 (np.ndarray):
            Array of length nchans1 containing the smurf channel of each channel
        b2 (np.ndarray):
            Array of length nchans2 containing the smurf band of each channel
        c2 (np.ndarray):
            Array of length nchans2 containing the smurf channel of each channel
        chans_per_band (int):
            Lets just hope this never changes.
    """
    acs1 = b1 * chans_per_band + c1
    acs2 = b2 * chans_per_band + c2

    mapping = np.full_like(acs1, -1, dtype=int)
    for i, ac in enumerate(acs1):
        x = np.where(acs2 == ac)[0]
        if len(x > 0):
            mapping[i] = x[0]
    return mapping


@dataclass
class Resonator:
    """
    Data structure to hold any resonator information.
    """
    idx: int

    is_north: int
    res_freq: float
    res_qi: float = np.nan
    smurf_res_idx: int = -1
    smurf_band: int = -1
    smurf_channel: int = -1
    smurf_subband: int = -1
    readout_id: str = ''

    xi: float = np.nan
    eta: float = np.nan
    gamma: float = np.nan

    bg: int = -1
    det_x: float = np.nan
    det_y: float = np.nan
    det_row: int = 0
    det_col: int = 0
    pixel_num: int = 0
    det_rhomb: str = ''
    det_pol: str = ''
    det_freq: int = 0
    det_bandpass: str = ''
    det_angle_raw_deg: float = np.nan
    det_angle_actual_deg: float = np.nan
    det_type: str = ''
    det_id: str = ''
    is_optical: int = 1
    mux_bondpad: int = 0
    mux_subband: str = ''
    mux_band: int = -1
    mux_channel: int = -1
    mux_layout_pos: int = -1

    matched: int = 0
    match_idx: int = -1
 

def apply_design_properties(smurf_res, design_res, in_place=False):
    """
    Combines two resonators into one, taking smurf-properties such as res-idx,
    smurf-band, and smurf-channel one, and design properties such as det
    position and polarization from the other.

    Args:
        smurf_res (Resonator):
            The resonator to take smurf properties from
        design_res (Resonator):
            The resonator to take design properties from
        in_place (bool):
            If True, the src_res will be modified. Otherwise, a new Resonator
            will be created and returned.
    """
    if in_place:
        r = smurf_res
    else:
        r = deepcopy(smurf_res)
    
    design_props = [
        'bg', 'det_x', 'det_y', 'det_row', 'det_col', 'pixel_num', 'det_rhomb',
        'det_pol', 'det_freq', 'det_bandpass', 'det_angle_raw_deg',
        'det_angle_actual_deg', 'det_type', 'det_id', 'is_optical',
        'mux_bondpad', 'mux_subband', 'mux_band', 'mux_channel',
        'mux_layout_pos'
    ]

    for prop in design_props:
        setattr(r, prop, getattr(design_res, prop))

    return r


class ResSet:
    """
    Class to hold a group of resonances. This provides easy interfaces for
    accessing Resonance fields as np arrays for fast computations
    and provides initialization functions from different data sources.
    """

    def __init__(self, resonances: List[Resonator], name=None):
        self.resonances: List[Resonator] = resonances
        self.name = name

    def __iter__(self):
        yield from self.resonances

    def __getitem__(self, k):
        return self.resonances[k]

    def __len__(self):
        return len(self.resonances)

    @classmethod
    def from_array(cls, arr):
        resonators = []
        names = arr.dtype.names
        for a in arr:
            resonators.append(Resonator(**dict(zip(names, a))))
        return cls(resonators)


    def as_array(self):
        """
        Returns resonance data in the form of a numpy structured array.
        """
        dtype = []
        data = []
        for field in fields(Resonator):
            if field.type == str:
                typ = '<U50'
            else:
                typ = field.type
            dtype.append((field.name, typ))
        for r in self.resonances:
            data.append(tuple(getattr(r, name) for name, _ in dtype))
        return np.array(data, dtype=dtype)

    @classmethod
    def from_tunefile(cls, tunefile, name=None, north_is_highband=True,
                      resfit_file=None, bgmap_file=None):
        """
        Creates an instance based on a smurf-tune file. If a resfit or bgmap
        file is included, that data will be added to the Resonance objects as
        well.

        Args:
            tunefile (str):
                Path to Pysmurf tunefile
            name (str):
                Name to label this ResSet
            north_is_highband (bool):
                True if the north-side of the array corresponds to bands
                4-7
            resfit_file (str):
                Path to file containing resonance fit data
            bgmap_file (str):
                Path to file containing bgmap data
        """
        tune = np.load(tunefile, allow_pickle=True).item()
        resonances = []
        idx = 0
        for band, _v in tune.items():
            if 'resonances' not in _v:
                continue

            for res_idx, d in _v['resonances'].items():
                is_north = north_is_highband ^ (band < 4)
                res = Resonator(
                    idx=idx, smurf_res_idx=res_idx, res_freq=d['freq'],
                    smurf_band=band, is_north=is_north
                )
                if d['channel'] != -1:
                    res.smurf_channel = d['channel']

                if res.smurf_band >= 4:
                    res.res_freq -= 2000

                resonances.append(res)
                idx += 1
        rs = cls(resonances, name=name)
        if resfit_file is not None:
            rs.add_resfit_data(resfit_file)
        if bgmap_file is not None:
            rs.add_bgmap_data(bgmap_file)
        return rs

    @classmethod
    def from_solutions(cls, sol_file, north_is_highband=True, name=None, 
                       fp_pars=None, platform='SAT'):
        """
        Creates an instance from an input-solution file. This will include both design data, along with smurf-band
        and smurf-channel info. Resonance frequencies used here are the VNA
        freqs measured by Kaiwen.

        Args:
            sol_file (str):
                Path to solutions file
            name (str):
                Name to label this ResSet
            north_is_highband (bool):
                True if the north-side of the array corresponds to bands
                4-8
            fp_pars (dict):
                Result of the function ``sotododlib.coords.optics.get_ufm_to_fp_pars``. If this is None, detector positions will
                not be mapped to pointing angles.
            platform (str):
                'SAT' or 'LAT'. Used to determine which focal plane function to
                use for pointing
        """
        resonances = []

        if fp_pars is not None:
            theta = -np.deg2rad(fp_pars['theta'])
            dx, dy = fp_pars['dx'], fp_pars['dy']

            if platform.upper() == 'SAT':
                fp_func = optics.SAT_focal_plane
            elif platform.upper() == 'LAT':
                fp_func = optics.LAT_focal_plane
    
        with open(sol_file, 'r') as f:
            labels  = f.readline().split(',') # Read header
            for i in range(len(labels)):
                labels[i] = labels[i].strip()

            #Helper needed for when sol file has saved ints as floats
            def _int(val, null_val=None): 
                if val == 'null' and null_val is not None:
                    return null_val
                return int(float(val))

            i = 0
            for line in f.readlines():
                d = dict(zip(labels, line.split(',')))
                if not d['bias_line']:  # Skip incomplete lines
                    continue
                # is_north = north_is_highband ^ (_int(d['smurf_band']) < 4)
                # is_north = d['is_north'].lower().strip() == 'true'
                is_north = _int(d['bias_line']) < 6
                is_optical = (d['is_optical'].lower() == 'true')
                r = Resonator(
                    i, res_freq=float(d['freq_mhz']), smurf_band=_int(d['smurf_band']),
                    bg=_int(d['bias_line']), det_x=float(d['det_x']),
                    det_y=float(d['det_y']), det_rhomb=d['rhomb'],
                    det_row=_int(d['det_row']), det_col=_int(d['det_col']),
                    pixel_num=_int(d['pixel_num'], null_val=-1),
                    det_type=d['det_type'], det_id=d['detector_id'],
                    mux_band=_int(d['mux_band']), mux_channel=_int(d['mux_channel']),
                    mux_subband=d['mux_subband'], mux_bondpad=d['bond_pad'],
                    det_angle_raw_deg=d['angle_raw_deg'],
                    det_angle_actual_deg=d['angle_actual_deg'],
                    mux_layout_pos=_int(d['mux_layout_position']),
                    det_bandpass=d['bandpass'], det_pol=d['pol'],
                    is_optical=is_optical,
                    is_north=is_north
                )

                if _int(d['smurf_channel']) != -1:
                    r.smurf_channel = _int(d['smurf_channel'])

                if fp_pars is not None:
                    x, y = float(d['det_x']), float(d['det_y'])
                    xp = x * np.cos(theta) - y * np.sin(theta) + dx
                    yp = x * np.sin(theta) + y * np.cos(theta) + dy
                    xi, eta, gamma = fp_func(None, x=xp, y=yp, pol=0)
                    r.xi = xi
                    r.eta = eta
                    r.gamma = gamma

                resonances.append(r)
                i += 1

        return cls(resonances, name=name)

    def add_resfit_data(self, resfit_file):
        """
        Adds resonator quality data from a res_fit file.
        """
        resfits = np.load(resfit_file, allow_pickle=True).item()
        for r in self.resonances:
            d = resfits[r.smurf_band][r.smurf_res_idx]
            r.res_qi = d['derived_params']['Qi']

    def add_bgmap_data(self, bgmap_file):
        """
        Adds bias group data from an sodetlib bgmap file.
        """
        bgmap = np.load(bgmap_file, allow_pickle=True).item()
        bands = np.array([res.smurf_band for res in self.resonances], dtype=int)
        chans = np.array([res.smurf_channel for res in self.resonances], dtype=int)
        idxmap = map_band_chans(bands, chans, bgmap['bands'], bgmap['channels'])

        for i, r in enumerate(self.resonances):
            idx = idxmap[i]
            if idx == -1:
                continue
            bg = bgmap['bgmap'][idx]
            if bg != -1:
                r.bg = bg

    def add_pointing(self, bands, chans, xis, etas):
        """
        Adds measured detector pointing to resonators.
        """
        bs = np.array([res.smurf_band for res in self.resonances], dtype=int)
        cs = np.array([res.smurf_channel for res in self.resonances], dtype=int)
        idxmap = map_band_chans(bs, cs, bands, chans)
        for i, r in enumerate(self.resonances):
            idx = idxmap[i]
            if idx == -1:
                continue
            r.xi = xis[idx]
            r.eta = etas[idx]

@dataclass
class MatchParams:
    """
    Any constants / hardcoded values go here to be fiddled with

    Args:
        unassigned_slots (int):
            Number of additional "unassigned" node to use per-side
        freq_offset_mhz (float):
            constant offset between resonator-set frequencies to use in
            matching.
        freq_width (float):
            width of exponential to use in the frequency cost function (MHz).
        dist_width (float)
            width of exponential to use in the pointing cost function (rad)
        unmatched_good_res_pen (float):
            penalty to apply to leaving a resonator with a good qi unassigned
        good_res_qi_thresh (float):
            qi threshold that is considered "good"
        force_src_pointing (bool):
            If true, will assign a np.inf penalty to leaving a src resonator
            with a provided pointing unmatched.
        assigned_bg_unmatched_pen (float):
            Penalty to apply to leaving a resonator with an assigned bg
            unmatched
        unassigned_bg_unmatched_pen (float):
            Penalty to apply to leaving a resonator with an unassigned bg
            unmatched
        assigned_bg_mismatch_pen (float):
            Penalty to apply for matching a resonator with an assigned bias line
            to another one with a mis-matched bias line.
        unassigned_bg_mismatch_pen (float):
            Penalty to apply for matching a resonator with no bias line to
            another one with a mis-matched bias line.
    """
    unassigned_slots: int = 1000
    freq_offset_mhz: float = 0.0
    freq_width: float = 2.
    dist_width: float =0.01
    unmatched_good_res_pen: float = 10.
    good_res_qi_thresh: float = 100e3
    force_src_pointing: bool = True
    assigned_bg_unmatched_pen: float = 100000
    unassigned_bg_unmatched_pen: float = 10000
    assigned_bg_mismatch_pen: float = 100000
    unassigned_bg_mismatch_pen: float = 1


@dataclass
class MatchingStats:
    unmatched_src: int = 0
    unmatched_dst: int = 0
    matched_chans: int = 0
    mismatched_bg: int = 0
    freq_diff_avg: float = 0.0
    freq_err_avg: float = 0.0
    pointing_err_avg: float = 0.0


class Match:
    """ 
    Class for performing a Resonance Matching between two sets of resonators,
    labeled `src` and `dst`. In the matching algorithm there is basically no
    difference between `src` and `dst` res-sets, except:
     - When merged, smurf-data such as band, channel, and res-idx will be taken
       from the `src` res-set
     - The `force_src_pointing` param can be used to assign a very high penalty
       to leaving any `src` resonator that has pointing info unassigned.
    
    Args:
        src (ResSet):
            The source resonator set
        dst (ResSet):
            The dest resonator set
        match_pars (MatchParams):
            MatchParams object used in the matching algorithm. This
            can be used to tune the cost-function and matching.
    
    Attributes:
        src (ResSet):
            The source resonator set
        dst (ResSet):
            The dest resonator set
        match_pars (MatchParams):
            MatchParams object used in the matching algorithm. This
            can be used to tune the cost-function and matching.
        matching (np.ndarray):
            A 2xN array of indices, where the first row corresponds to the
            indices of the src resonators, and the second row corresponds to
            the indices of the dst resonators.  If the index is larger than the
            size of the corresponding res-set, that means the paired
            resonator was not matched.
        merged (ResSet):
            A ResSet containing the merged resonators. This is created by
            applying the "design" properties of the dst resonators to the source
            resonators. If a source resonator is not matched to any dest
            resonator, it will be copied as-is.
        stats (MatchingStats):
            A MatchingStats object containing some statistics about the
            matching.
    """
    def __init__(self, src: ResSet, dst: ResSet, match_pars: Optional[MatchParams]=None):
        self.src = src
        self.dst = dst

        if match_pars is None:
            self.match_pars = MatchParams()
        else:
            self.match_pars = match_pars

        self.matching, self.merged = self._match()
        self.stats = self.get_stats()

    def _get_biadjacency_matrix(self):
        src_arr = self.src.as_array()
        dst_arr = self.dst.as_array()

        mat = np.zeros((len(self.src), len(self.dst)), dtype=float)

        # N/S mismatch
        m = src_arr['is_north'][:, None] != dst_arr['is_north'][None, :]
        mat[m] = np.inf

        # Frequency offset
        df = src_arr['res_freq'][:, None] - dst_arr['res_freq'][None, :]
        df -= self.match_pars.freq_offset_mhz
        mat += np.exp((np.abs(df / self.match_pars.freq_width)) ** 2)

        # BG mismatch
        bgs_mismatch = src_arr['bg'][:, None] != dst_arr['bg'][None, :]
        bgs_unassigned = (src_arr['bg'][:, None] == 1) | (dst_arr['bg'][None, :] == -1)

        m = bgs_mismatch & bgs_unassigned
        mat[m] += self.match_pars.unassigned_bg_mismatch_pen
        m = bgs_mismatch & (~bgs_unassigned)
        mat[m] += self.match_pars.assigned_bg_mismatch_pen

        # If pointing, add cost if assigned too far
        dd = np.sqrt(
              (src_arr['xi'][:, None] - dst_arr['xi'][None, :])**2 \
            + (src_arr['eta'][:, None] - dst_arr['eta'][None, :])**2)
        m = ~np.isnan(dd)
        mat[m] += np.exp((np.abs(dd[m] / self.match_pars.dist_width)) ** 2)

        return mat

    def _get_unassigned_costs(self, rs, force_if_pointing=True):
        ra = rs.as_array()

        arr = np.zeros(len(rs))

        # Additional cost for leaving good resonator unassigned
        is_good_res = ra['res_qi'] > self.match_pars.good_res_qi_thresh
        is_good_res[np.isnan(is_good_res)] = 0
        arr[is_good_res] += self.match_pars.unmatched_good_res_pen

        bg_assigned = ra['bg'] != -1
        arr[bg_assigned] += self.match_pars.assigned_bg_unmatched_pen
        arr[~bg_assigned] += self.match_pars.unassigned_bg_unmatched_pen

        # Infinite cost if has pointing
        if force_if_pointing:
            m = ~np.isnan(ra['xi'])
            arr[m] = np.inf

        return arr

    def _match(self):
        nside = max(len(self.src), len(self.dst)) + self.match_pars.unassigned_slots

        # Keep this square so all resonators are included in final matching
        mat_full = np.zeros((nside, nside), dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='overflow encountered in exp*',
                category=RuntimeWarning
            )

            mat_full[:len(self.src), :len(self.dst)] = self._get_biadjacency_matrix()
            mat_full[:len(self.src), len(self.dst):] = \
                self._get_unassigned_costs(
                    self.src,
                    force_if_pointing=self.match_pars.force_src_pointing
                )[:, None]
            mat_full[len(self.src):, :len(self.dst)] = \
                self._get_unassigned_costs(self.dst, force_if_pointing=False)[None, :]
            mat_full[len(self.src):, len(self.dst):] = 0

        self.matching = np.array(linear_sum_assignment(mat_full))

        for r1, r2 in self.get_match_iter(include_unmatched=True):
            if r1 is None:
                r2.matched = 0
                continue
            if r2 is None:
                r1.matched = 0
                continue

            r1.matched = 1
            r1.match_idx = r2.idx
            r2.matched = 1
            r2.match_idx = r1.idx

        resonances = []
        for r1 in self.src:
            if r1.matched:
                # print(r1.match_idx)
                r2 = self.dst[r1.match_idx]
                r = apply_design_properties(r1, r2, in_place=False)
            else:
                r = deepcopy(r1)
            resonances.append(r)
        self.merged = ResSet(resonances, name='merged')

        return self.matching, self.merged

    def get_match_iter(self, include_unmatched=True) \
        -> Iterator[Tuple[Optional[Resonator], Optional[Resonator]]]:
        """
        Returns an iterator over matched resonators (r1, r2).

        Args:
            include_unmatched (bool):
                If True, will include unmatched resonators, with the pair set
                to None.
        """
        for i, j in zip(*self.matching):
            if i < len(self.src):
                r1 = self.src[i]
            else:
                r1 = None
            if j < len(self.dst):
                r2 = self.dst[j]
            else:
                r2 = None

            if (r1 is None) and (r2 is None):
                continue

            if (r1 is None) or (r2 is None):
                if not include_unmatched:
                    continue

            yield r1, r2

    def get_stats(self) -> MatchingStats:
        """
        Gets stats associated with current matching.
        """
        stats = MatchingStats()
        nchans_with_pointing = 0
        src_arr = self.src.as_array()
        dst_arr = self.dst.as_array()

        stats.unmatched_src = np.sum(~src_arr['matched'].astype(bool))
        stats.unmatched_dst = np.sum(~dst_arr['matched'].astype(bool))

        for r1, r2 in self.get_match_iter(include_unmatched=False):
            stats.matched_chans += 1
            if r1.bg != r2.bg:
                stats.mismatched_bg += 1
            stats.freq_diff_avg += r1.res_freq - r2.res_freq
            stats.freq_err_avg += np.abs(
                r1.res_freq - r2.res_freq - self.match_pars.freq_offset_mhz
            )
            if not np.isnan(r1.xi):
                pointing_err = np.sqrt(
                    (r1.xi - r2.xi)**2 + (r1.eta - r2.eta)**2 
                )
                nchans_with_pointing += 1
                stats.pointing_err_avg += pointing_err
        stats.freq_diff_avg /= stats.matched_chans
        stats.freq_diff_avg = float(stats.freq_diff_avg)

        stats.freq_err_avg /= stats.matched_chans
        stats.freq_err_avg = float(stats.freq_err_avg)

        if nchans_with_pointing != 0:
            stats.pointing_err_avg /= nchans_with_pointing
            stats.pointing_err_avg = float(np.rad2deg(stats.pointing_err_avg))
        else:
            stats.pointing_err_avg = np.nan

        return stats

    def save(self, path):
        """Saves match to HDF5 file"""
        with h5py.File(path, 'w') as fout:
            fout.create_group('meta')
            fout.create_group('meta/match_pars')
            for k, v in asdict(self.match_pars).items():
                fout['meta/match_pars'][k] = v
            fout['meta/src_name'] = self.src.name
            fout['meta/dst_name'] = self.dst.name

            write_dataset(
                metadata.ResultSet.from_friend(self.src.as_array()), fout, 'src')
            write_dataset(
                metadata.ResultSet.from_friend(self.dst.as_array()), fout, 'dst')
            write_dataset(np.array(self.matching), fout, 'matching')
            write_dataset(
                metadata.ResultSet.from_friend(self.merged.as_array()), fout, 'merged')


def plot_match_freqs(m: Match, is_north=True, show_offset=False, xlim=None):
    """
    Plots src and dst ResSet freqs in a match, along with their matching
    assignments.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    if not show_offset:
        offset = m.match_pars.freq_offset_mhz
    else:
        offset = 0

    yw= 0.4
    def plot_res(ax, r: Resonator, y=0, offset=0):
        if r is None:
            return
        if r.is_north != is_north:
            return

        xs = [r.res_freq+offset, r.res_freq+offset]
        ys = [y - yw, y+yw]

        if r.bg == -1:
            c = 'grey'
        else:
            c = f'C{r.bg}'

        ax.plot(xs, ys, c=c)

    for r1, r2 in m.get_match_iter():
        plot_res(ax, r1, y=0)
        plot_res(ax, r2, y=1, offset=offset)

    for r1, r2 in m.get_match_iter(include_unmatched=False):
        if r1.is_north != is_north:
            continue
        ax.plot([r1.res_freq, r2.res_freq+offset], [yw, 1-yw], 
                c='grey', alpha=.3)

    ax.set_yticks([yw/2, 1-yw/2])
    labels = ['ResSet1', 'ResSet2']
    for i, rs in enumerate([m.src, m.dst]):
        if rs.name is not None:
            labels[i] = rs.name

    ax.set_yticklabels(labels, fontsize=14, rotation=90, va='center')

    ax.set_ylim(0, 1)
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Resonance Freq (MHz)", fontsize=14)
    ax.set_title(f"Freq Matching (north={is_north})")
    return fig, ax


def plot_match_pointing(match: Match, show_pairs=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    sa, da = match.src.as_array(), match.dst.as_array()
    plt.plot(np.rad2deg(da['xi']), np.rad2deg(da['eta']), '.')
    plt.plot(np.rad2deg(sa['xi']), np.rad2deg(sa['eta']), '.')
    if show_pairs:
        for r1, r2 in match.get_match_iter(include_unmatched=False):
            plt.plot(
                *np.rad2deg(np.array([[r1.xi, r2.xi], [r1.eta, r2.eta]])),
                color='red', alpha=.2
            )
    ax.set_xlabel("Xi (deg)")
    ax.set_ylabel("Eta (deg)")

    return fig, ax

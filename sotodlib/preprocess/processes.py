import numpy as np
from operator import attrgetter

from so3g.proj import Ranges, RangesMatrix

import sotodlib.core as core
import sotodlib.tod_ops as tod_ops
import sotodlib.obs_ops as obs_ops
from sotodlib.hwp import hwp, hwp_angle_model
import sotodlib.coords.planets as planets

from sotodlib.core.flagman import (has_any_cuts, has_all_cut,
                                   count_cuts,
                                   sparse_to_ranges_matrix)

from .pcore import _Preprocess, _FracFlaggedMixIn
from .. import flag_utils
from ..core import AxisManager


class FFTTrim(_Preprocess):
    """Trim the AxisManager to optimize for faster FFTs later in the pipeline.
    All processing configs go to `fft_trim`

    .. autofunction:: sotodlib.tod_ops.fft_trim
    """
    name = "fft_trim"
    def process(self, aman, proc_aman):
        tod_ops.fft_trim(aman, **self.process_cfgs)

class Detrend(_Preprocess):
    """Detrend the signal. All processing configs go to `detrend_tod`

    .. autofunction:: sotodlib.tod_ops.detrend_tod
    """
    name = "detrend"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)
    
    def process(self, aman, proc_aman):
        tod_ops.detrend_tod(aman, signal_name=self.signal,
                            **self.process_cfgs)

class DetBiasFlags(_FracFlaggedMixIn, _Preprocess):
    """
    Derive poorly biased detectors from IV and Bias Step data. Save results
    in proc_aman under the "det_bias_cuts" field. 

    .. autofunction:: sotodlib.tod_ops.flags.get_det_bias_flags
    """
    name = "det_bias_flags"
    _influx_field = "det_bias_flags_frac"

    def calc_and_save(self, aman, proc_aman):
        dbc_aman = tod_ops.flags.get_det_bias_flags(aman, merge=False, full_output=True,
                                                    **self.calc_cfgs)
        self.save(proc_aman, dbc_aman)
    
    def save(self, proc_aman, dbc_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("det_bias_flags", dbc_aman)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        keep = ~proc_aman.det_bias_flags.det_bias_flags
        meta.restrict("dets", meta.dets.vals[has_all_cut(keep)])
        return meta
    
    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_det_bias_flags
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            plot_det_bias_flags(aman, proc_aman['det_bias_flags'], rfrac_range=self.calc_cfgs['rfrac_range'],
                                psat_range=self.calc_cfgs['psat_range'], filename=filename.replace('{name}', f'{ufm}_bias_cuts_venn'))


class Trends(_FracFlaggedMixIn, _Preprocess):
    """Calculate the trends in the data to look for unlocked detectors. All
    calculation configs go to `get_trending_flags`.

    Saves results in proc_aman under the "trend" field. 

    Data selection can have key "kind" equal to "any" or "all."

     Example config block::

        - name : "trends"
          signal: "signal" # optional
          calc:
            max_trend: 2.5
            t_piece: 100
          save: True
          plot: True
          select:
            kind: "any"
    
    .. autofunction:: sotodlib.tod_ops.flags.get_trending_flags
    """
    name = "trends"
    _influx_field = "trend_flags_frac"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)
    def calc_and_save(self, aman, proc_aman):
        _, trend_aman = tod_ops.flags.get_trending_flags(
            aman, merge=False, full_output=True,
            signal=aman[self.signal], **self.calc_cfgs)
        aman.wrap("trends", trend_aman)
        self.save(proc_aman, trend_aman)
    
    def save(self, proc_aman, trend_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("trends", trend_aman)
    
    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        if self.select_cfgs["kind"] == "any":
            keep = ~has_any_cuts(proc_aman.trends.trend_flags)
        elif self.select_cfgs["kind"] == "all":
            keep = ~has_all_cut(proc_aman.trends.trend_flags)
        else:
            raise ValueError(f"Entry '{self.select_cfgs['kind']}' not"
                                "understood. Expect 'any' or 'all'")
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_trending_flags
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            plot_trending_flags(aman, proc_aman['trends'], filename=filename.replace('{name}', f'{ufm}_trending_flags'))


class GlitchDetection(_FracFlaggedMixIn, _Preprocess):
    """Run glitch detection algorithm to find glitches. All calculation configs
    go to `get_glitch_flags` 

    Saves retsults in proc_aman under the "glitches" field.

    Data section should define a glitch significant "sig_glitch" and a maximum
    number of glitches "max_n_glitch."

    Example configuration block::
        
      - name: "glitches"
        calc:
          signal_name: "hwpss_remove"
          t_glitch: 0.00001
          buffer: 10
          hp_fc: 1
          n_sig: 10
          subscan: False
        save: True
        plot:
            plot_ds_factor: 50
        select:
          max_n_glitch: 10
          sig_glitch: 10

    .. autofunction:: sotodlib.tod_ops.flags.get_glitch_flags
    """
    name = "glitches"
    _influx_field = "glitch_flags_frac"

    def calc_and_save(self, aman, proc_aman):
        _, glitch_aman = tod_ops.flags.get_glitch_flags(aman,
            merge=False, full_output=True, **self.calc_cfgs
        ) 
        aman.wrap("glitches", glitch_aman)
        self.save(proc_aman, glitch_aman)
        if self.calc_cfgs.get('save_plot', False):
            flag_utils.plot_glitch_stats(aman, save_path=self.calc_cfgs['save_plot'])
    
    def save(self, proc_aman, glitch_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("glitches", glitch_aman)
 
    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        flag = sparse_to_ranges_matrix(
            proc_aman.glitches.glitch_detection > self.select_cfgs["sig_glitch"]
        )
        n_cut = count_cuts(flag)
        keep = n_cut <= self.select_cfgs["max_n_glitch"]
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_signal_diff, plot_flag_stats
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            plot_signal_diff(aman, proc_aman.glitches, flag_type='glitches', flag_threshold=self.select_cfgs.get("max_n_glitch", 10), 
                             plot_ds_factor=self.plot_cfgs.get("plot_ds_factor", 50), filename=filename.replace('{name}', f'{ufm}_glitch_signal_diff'))
            plot_flag_stats(aman, proc_aman.glitches, flag_type='glitches', filename=filename.replace('{name}', f'{ufm}_glitch_stats'))


class FixJumps(_Preprocess):
    """
    Repairs the jump heights given a set of jump flags and heights.

    Example config block::

      - name: "fix_jumps"
        signal: "signal" # optional
        process:
        jumps_aman: "jumps_2pi"

    .. autofunction:: sotodlib.tod_ops.jumps.jumpfix_subtract_heights
    """
    name = "fix_jumps"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        field = self.process_cfgs['jumps_aman']
        aman[self.signal] = tod_ops.jumps.jumpfix_subtract_heights(
            aman[self.signal], proc_aman[field].jump_flag.mask(),
            inplace=True, heights=proc_aman[field].jump_heights)


class Jumps(_FracFlaggedMixIn, _Preprocess):
    """Run generic jump finding and fixing algorithm.
    
    calc_cfgs should have 'function' defined as one of 
    'find_jumps', 'twopi_jumps' or 'slow_jumps'. Any additional configs to the
    jump function goes in 'jump_configs'. 

    Saves results in proc_aman under the "jumps" field.

    Data section should define a maximum number of jumps "max_n_jumps".

    Example config block::

      - name: "jumps"
        calc:
          function: "twopi_jumps"
        save:
          jumps_name: "jumps_2pi"
        plot:
            plot_ds_factor: 50
        select:
            max_n_jumps: 5
        

    .. autofunction:: sotodlib.tod_ops.jumps.find_jumps
    """

    name = "jumps"
    _influx_field = "jump_flags_frac"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        function = self.calc_cfgs.get("function", "find_jumps")
        cfgs = self.calc_cfgs.get('jump_configs', {})

        if function == 'find_jumps':
            func = tod_ops.jumps.find_jumps
        elif function == 'twopi_jumps':
            func = tod_ops.jumps.twopi_jumps
        elif function == 'slow_jumps':
            func = tod_ops.jumps.slow_jumps
        else:
            raise ValueError("function must be 'find_jumps', 'twopi_jumps' or" 
                            f"'slow_jumps'. Received {function}")

        jumps, heights = func(aman, merge=False, fix=False,
                              signal=aman[self.signal], **cfgs)
        jump_aman = tod_ops.jumps.jumps_aman(aman, jumps, heights)
        self.save(proc_aman, jump_aman)

    def save(self, proc_aman, jump_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            name = self.save_cfgs.get('jumps_name', 'jumps')
            proc_aman.wrap(name, jump_aman)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        name = self.save_cfgs.get('jumps_name', 'jumps')

        n_cut = count_cuts(proc_aman[name].jump_flag)
        keep = n_cut <= self.select_cfgs["max_n_jumps"]
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_signal_diff, plot_flag_stats
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            name = self.save_cfgs.get('jumps_name', 'jumps')
            plot_signal_diff(aman, proc_aman[name], flag_type='jumps', flag_threshold=self.select_cfgs.get("max_n_jumps", 5), 
                             plot_ds_factor=self.plot_cfgs.get("plot_ds_factor", 50), filename=filename.replace('{name}', f'{ufm}_jump_signal_diff'))
            plot_flag_stats(aman, proc_aman[name], flag_type='jumps', filename=filename.replace('{name}', f'{ufm}_jumps_stats'))


class PSDCalc(_Preprocess):
    """ Calculate the PSD of the data and add it to the Preprocessing AxisManager under the
    "psd" field.

    Example config block::

      - "name : "psd"
        "signal: "signal" # optional
        "wrap": "psd" # optional
        "calc":
          "psd_cfgs": # optional, kwargs to scipy.welch
            "nperseg": 1024
          "wrap_name": "psd" # optional
          "subscan": False
        "save": True

    .. autofunction:: sotodlib.tod_ops.fft_ops.calc_psd
    """
    name = "psd"
    
    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')
        self.wrap = step_cfgs.get('wrap', 'psd')

        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        freqs, Pxx = tod_ops.fft_ops.calc_psd(aman, signal=aman[self.signal],
                                              **self.calc_cfgs)

        fft_aman = core.AxisManager(aman.dets,
                                    core.OffsetAxis("nusamps", len(freqs)))
        pxx_axis_map = [(0, "dets"), (1, "nusamps")]
        if self.calc_cfgs.get('subscan', False):
            fft_aman.wrap("Pxx_ss", Pxx, pxx_axis_map+[(2, aman.subscans)])
            Pxx = np.nanmean(Pxx, axis=-1) # Mean of subscans

        fft_aman.wrap("freqs", freqs, [(0,"nusamps")])
        fft_aman.wrap("Pxx", Pxx, pxx_axis_map)

        self.save(proc_aman, fft_aman)

    def save(self, proc_aman, fft_aman):
        if not(self.save_cfgs is None):
            proc_aman.wrap(self.wrap, fft_aman)
    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_psd

            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            filename = filename.replace('{name}', f'{ufm}_{self.wrap}')

            plot_psd(aman, signal=attrgetter(f"{self.wrap}.Pxx")(proc_aman),
                     xx=attrgetter(f"{self.wrap}.freqs")(proc_aman), filename=filename, **self.plot_cfgs)


class GetStats(_Preprocess):
    """ Get basic statistics from a TOD or its power spectrum.

    Example config block:

      - name : "tod_stats"
        signal: "signal" # optional
        wrap: "tod_stats" # optional
        calc:
          stat_names: ["median", "std"]
          split_subscans: False # optional
          psd_mask: # optional, for cutting a power spectrum in frequency
            freqs: "psd.freqs"
            low_f: 1
            high_f: 10
        save: True

    """
    name = "tod_stats"
    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')
        self.wrap = step_cfgs.get('wrap', 'tod_stats')

        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        if self.calc_cfgs.get('psd_mask') is not None:
            mask_dict = self.calc_cfgs.get('psd_mask')
            _f = attrgetter(mask_dict['freqs'])
            try:
                freqs = _f(aman)
            except KeyError:
                freqs = _f(proc_aman)
            low_f, high_f = mask_dict['low_f'], mask_dict['high_f']
            fmask = np.all([freqs >= low_f, freqs <= high_f], axis=0)
            self.calc_cfgs['mask'] = fmask
            del self.calc_cfgs['psd_mask']

        _f = attrgetter(self.signal)
        try:
            signal = _f(aman)
        except KeyError:
            signal = _f(proc_aman)
        stats_aman = tod_ops.flags.get_stats(aman, signal, **self.calc_cfgs)
        self.save(proc_aman, stats_aman)

    def save(self, proc_aman, stats_aman):
        if not(self.save_cfgs is None):
            proc_aman.wrap(self.wrap, stats_aman)

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_signal

            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            filename = filename.replace('{name}', f'{ufm}_{self.signal}')

            plot_signal(aman, signal_name=self.signal, x_name="timestamps", filename=filename, **self.plot_cfgs)

class Noise(_Preprocess):
    """Estimate the white noise levels in the data. Assumes the PSD has been
    wrapped into the preprocessing AxisManager. All calculation configs goes to `calc_wn`.

    Saves the results into the "noise" field of proc_aman. 

    Can run data selection of a "max_noise" value. 

    Example config block::

     - name: "noise"
       fit: False
       subscan: False
       calc:
         low_f: 5
         high_f: 10
       save: True
       select:
         max_noise: 2000

    If ``fit: True`` this operation will run
    :func:`sotodlib.tod_ops.fft_ops.fit_noise_model`, else it will run
    :func:`sotodlib.tod_ops.fft_ops.calc_wn`.

    """
    name = "noise"

    def __init__(self, step_cfgs):
        self.psd = step_cfgs.get('psd', 'psd')
        self.fit = step_cfgs.get('fit', False)
        self.subscan = step_cfgs.get('subscan', False)

        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        if self.psd not in proc_aman:
            raise ValueError("PSD is not saved in Preprocessing AxisManager")
        psd = proc_aman[self.psd]
        pxx = psd.Pxx_ss if self.subscan else psd.Pxx

        if self.calc_cfgs is None:
            self.calc_cfgs = {}

        if self.fit:
            if self.calc_cfgs.get('subscan') is None:
                self.calc_cfgs['subscan'] = self.subscan
            calc_aman = tod_ops.fft_ops.fit_noise_model(aman, pxx=pxx,
                                                        f=psd.freqs, 
                                                        merge_fit=True,
                                                        **self.calc_cfgs)
        else:
            wn = tod_ops.fft_ops.calc_wn(aman, pxx=pxx,
                                         freqs=psd.freqs,
                                         **self.calc_cfgs)
            if not self.subscan:
                calc_aman = core.AxisManager(aman.dets)
                calc_aman.wrap("white_noise", wn, [(0,"dets")])
            else:
                calc_aman = core.AxisManager(aman.dets, aman.subscan_info.subscans)
                calc_aman.wrap("white_noise", wn, [(0,"dets"), (1,"subscans")])

        self.save(proc_aman, calc_aman)
    
    def save(self, proc_aman, noise):
        if self.save_cfgs is None:
            return

        if isinstance(self.save_cfgs, bool):
            if self.save_cfgs:
                proc_aman.wrap("noise", noise)
                return

        if self.save_cfgs['wrap_name'] is None:
            proc_aman.wrap("noise", noise)
        else:
            proc_aman.wrap(self.save_cfgs['wrap_name'], noise)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta

        if proc_aman is None:
            proc_aman = meta.preprocess

        if 'wrap_name' in self.save_cfgs:
            self.select_cfgs['name'] = self.select_cfgs.get('name', self.save_cfgs['wrap_name'])
        else:
            self.select_cfgs['name'] = self.select_cfgs.get('name', 'noise')

        if self.fit:
            wn = proc_aman[self.select_cfgs['name']].fit[:,1]
            fk = proc_aman[self.select_cfgs['name']].fit[:,0]
        else:
            wn = proc_aman[self.select_cfgs['name']].white_noise
            fk = None 
        if self.subscan:
            wn = np.nanmean(wn, axis=-1) # Mean over subscans
            if fk is not None:
                fk = np.nanmean(fk, axis=-1) # Mean over subscans
        keep = np.ones_like(wn, dtype=bool)
        if "max_noise" in self.select_cfgs.keys():
            keep &= (wn <= np.float64(self.select_cfgs["max_noise"]))
        if "min_noise" in self.select_cfgs.keys():
            keep &= (wn >= np.float64(self.select_cfgs["min_noise"]))
        if fk is not None and "max_fknee" in self.select_cfgs.keys():
            keep &= (fk <= np.float64(self.select_cfgs["max_fknee"]))
        meta.restrict("dets", meta.dets.vals[keep])
        return meta
    
class Calibrate(_Preprocess):
    """Calibrate the timestreams based on some provided information.

    Type of calibration is decided by process["kind"]

    1. "single_value" : multiplies entire signal by the single value
    process["val"]

    2. "array" : takes the dot product of the array with the entire signal. The
    array is specified by ``process["cal_array"]``, which must exist in
    ``aman``. The array can be nested within additional ``AxisManager``
    objects, for instance ``det_cal.phase_to_pW``.

    Example config block(s)::

      - name: "calibrate"
        process:
          kind: "single_value"
          divide: True # If true will divide instead of multiply.
          # phase_to_pA: 9e6/(2*np.pi)
          val: 1432394.4878270582
      - name: "calibrate"
        process:
          kind: "array"
          cal_array: "cal.array"

    """
    name = "calibrate"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        if self.process_cfgs["kind"] == "single_value":
            if self.process_cfgs.get("divide", False):
                aman[self.signal] /= self.process_cfgs["val"]
            else:
                aman[self.signal] *= self.process_cfgs["val"]
        elif self.process_cfgs["kind"] == "array":
            field = self.process_cfgs["cal_array"]
            _f = attrgetter(field)
            if self.process_cfgs.get("proc_aman_cal", False):
                cal_arr = _f(proc_aman)
            else:
                cal_arr = _f(aman)
            if self.process_cfgs.get("divide", False):
                aman[self.signal] = np.divide(aman[self.signal].T, cal_arr).T
            else:
                aman[self.signal] = np.multiply(aman[self.signal].T, cal_arr).T
        else:
            raise ValueError(f"Entry '{self.process_cfgs['kind']}'"
                              " not understood")

class EstimateHWPSS(_Preprocess):
    """
    Builds a HWPSS Template. Calc configs go to ``hwpss_model``.
    Results of fitting saved if field specified by calc["name"].

    Example config block::

      - "name : "estimate_hwpss"
        "calc":
          "signal_name": "signal" # optional
          "hwpss_stats_name": "hwpss_stats"
        "save": True

    .. autofunction:: sotodlib.hwp.hwp.get_hwpss
    """
    name = "estimate_hwpss"
    _influx_field = "hwpss_coeffs"
    _influx_percentiles = [0, 50, 75, 90, 95, 100]

    def calc_and_save(self, aman, proc_aman):
        hwpss_stats = hwp.get_hwpss(aman, **self.calc_cfgs)
        self.save(proc_aman, hwpss_stats)

    def save(self, proc_aman, hwpss_stats):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap(self.calc_cfgs["hwpss_stats_name"], hwpss_stats)

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_4f_2f_counts, plot_hwpss_fit_status
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]
            plot_4f_2f_counts(aman, filename=filename.replace('{name}', f'{ufm}_4f_2f_counts'))
            plot_hwpss_fit_status(aman, proc_aman[self.calc_cfgs["hwpss_stats_name"]], filename=filename.replace('{name}', f'{ufm}_hwpss_stats'))

    @classmethod
    def gen_metric(cls, meta, proc_aman):
        """ Generate a QA metric for the coefficients of the HWPSS fit.
        Coefficient percentiles and mean are recorded for every mode and detset.

        Arguments
        ---------
        meta : AxisManager
            The full metadata container.
        proc_aman : AxisManager
            The metadata containing just the output of this process.

        Returns
        -------
        line : dict
            InfluxDB line entry elements to be fed to
            `site_pipeline.monitor.Monitor.record`
        """
        # record one metric per wafer_slot per bandpass
        # extract these tags for the metric
        tag_keys = ["wafer_slot", "tel_tube", "wafer.bandpass"]
        tags = []
        vals = []
        from ..qa.metrics import _get_tag, _has_tag
        import re
        for bp in np.unique(meta.det_info.wafer.bandpass):
            for ws in np.unique(meta.det_info.wafer_slot):
                subset = np.where(
                    (meta.det_info.wafer_slot == ws) & (meta.det_info.wafer.bandpass == bp)
                )[0]

                # get the coefficients for every detector
                coeff = proc_aman.hwpss_stats.coeffs[subset]
                # mask those that were not set
                nonzero = np.any(coeff != 0.0, axis=1)

                # calculate amplitude of each mode
                mode_labels = list(proc_aman.hwpss_stats.modes.vals)
                num_re = re.compile("^[SC](\d+)$")
                nums = sorted(list(set([num_re.match(l).group(1) for l in mode_labels])))
                coeff_amp = np.zeros((coeff.shape[0], len(nums)), coeff.dtype)
                amp_labels = []
                for i, n in enumerate(nums):
                    c_ind = mode_labels.index(f"C{n}")
                    s_ind = mode_labels.index(f"S{n}")
                    coeff_amp[:, i] = np.sqrt(coeff[:, c_ind]**2 + coeff[:, s_ind]**2)
                    amp_labels.append(f"A{n}")

                # record percentiles over detectors and fraction of samples flagged
                perc = np.percentile(coeff_amp[nonzero], cls._influx_percentiles, axis=0)
                mean = coeff_amp[nonzero].mean(axis=0)

                tags_base = {
                    k: _get_tag(meta.det_info, k, subset[0]) for k in tag_keys if _has_tag(meta.det_info, k)
                }
                tags_base["telescope"] = meta.obs_info.telescope

                # loop over percentiles and coefficient labels
                for pi, p in enumerate(cls._influx_percentiles):
                    for l in amp_labels:
                        t_new = tags_base.copy()
                        t_new.update({"mode": l, "det_stat": f"percentile_{p}"})
                        tags.append(t_new)
                    vals += list(perc[pi])

                # finally also record the mean
                for l in amp_labels:
                    t_new = tags_base.copy()
                    t_new.update({"mode": l, "det_stat": "mean"})
                    tags.append(t_new)
                vals += list(mean)

        obs_time = [meta.obs_info.timestamp] * len(tags)
        return {
            "field": cls._influx_field,
            "values": vals,
            "timestamps": obs_time,
            "tags": tags,
        }

class SubtractHWPSS(_Preprocess):
    """Subtracts a HWPSS template from signal. 

    .. autofunction:: sotodlib.hwp.hwp.subtract_hwpss
    """
    name = "subtract_hwpss"

    def __init__(self, step_cfgs):
        self.hwpss_stats = step_cfgs.get('hwpss_stats', 'hwpss_stats')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        if not(proc_aman[self.hwpss_stats] is None):
            modes = [int(m[1:]) for m in proc_aman[self.hwpss_stats].modes.vals[::2]]
            template = hwp.harms_func(aman.hwp_angle, modes,
                                  proc_aman[self.hwpss_stats].coeffs)
            if 'hwpss_model' in aman._fields:
                aman.move('hwpss_model', None)
            aman.wrap('hwpss_model', template, [(0, 'dets'), (1, 'samps')])
            hwp.subtract_hwpss(
                aman,
                subtract_name = self.process_cfgs["subtract_name"]
                )

class Apodize(_Preprocess):
    """Apodize the edges of a signal. All process configs go to `apodize_cosine`

    .. autofunction:: sotodlib.tod_ops.apodize.apodize_cosine
    """
    name = "apodize"

    def process(self, aman, proc_aman):
        tod_ops.apodize.apodize_cosine(aman, **self.process_cfgs)

class Demodulate(_Preprocess):
    """Demodulate the tod. All process confgis go to `demod_tod`.

    .. autofunction:: sotodlib.hwp.hwp.demod_tod
    """
    name = "demodulate"

    def process(self, aman, proc_aman):
        hwp.demod_tod(aman, **self.process_cfgs["demod_cfgs"])
        if self.process_cfgs.get("trim_samps"):
            trim = self.process_cfgs["trim_samps"]
            proc_aman.restrict('samps', (aman.samps.offset + trim,
                                         aman.samps.offset + aman.samps.count - trim))
            aman.restrict('samps', (aman.samps.offset + trim,
                                    aman.samps.offset + aman.samps.count - trim))


class AzSS(_Preprocess):
    """Estimates Azimuth Synchronous Signal (AzSS) by binning signal by azimuth of boresight and subtract.
    All process confgis go to `get_azss`. If `method` is 'interpolate', no fitting applied
    and binned signal is directly used as AzSS model. If `method` is 'fit', Legendre polynominal
    fitting will be applied and used as AzSS model. If `subtract_in_place` is True, subtract AzSS model
    from signal in place.

    Example configuration block::

      - name: "azss"
        calc:
          signal: 'demodQ'
          azss_stats_name: 'azss_statsQ'
          range: [-1.57079, 7.85398]
          bins: 1080
          flags: 'glitch_flags'
          merge_stats: False
          merge_model: False
          subtract_in_place: True
        save: True
        process:
          subtract: True

    If we estimate and subtract azss in left going scans only,
    make union of gltich_flags and scan_flags first

      - name : "union_flags"
        process:
          flag_labels: ['glitches.glitch_flags', 'turnaround_flags.right_scan']
          total_flags_label: 'glitch_flags_left'

      - name: "azss"
        calc:
          signal: 'demodQ'
          azss_stats_name: 'azss_statsQ_left'
          range: [-1.57079, 7.85398]
          bins: 1080
          flags: 'glitch_flags_left'
          scan_flags: 'left_scan'
          merge_stats: False
          merge_model: False
          subtract_in_place: True
        save: True
        process:
          subtract: True

    .. autofunction:: sotodlib.tod_ops.azss.get_azss
    """
    name = "azss"

    def calc_and_save(self, aman, proc_aman):
        if self.process_cfgs:
            self.save(proc_aman, aman[self.calc_cfgs['azss_stats_name']])
        else:
            calc_aman, _ = tod_ops.azss.get_azss(aman, **self.calc_cfgs)
            self.save(proc_aman, calc_aman)

    def save(self, proc_aman, azss_stats):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap(self.calc_cfgs["azss_stats_name"], azss_stats)

    def process(self, aman, proc_aman, sim=False):
        if self.calc_cfgs.get('azss_stats_name') in proc_aman and self.process_cfgs["subtract"]:
            if sim:
                tod_ops.azss.get_azss(aman, **self.calc_cfgs)
            else:
                tod_ops.azss.subtract_azss(
                    aman,
                    proc_aman.get(self.calc_cfgs.get('azss_stats_name')),
                    signal=self.calc_cfgs.get('signal', 'signal'),
                    scan_flags=self.calc_cfgs.get('scan_flags'),
                    method=self.calc_cfgs.get('method', 'interpolate'),
                    max_mode=self.calc_cfgs.get('max_mode'),
                    range=self.calc_cfgs.get('range'),
                    in_place=True
                )
        else:
            tod_ops.azss.get_azss(aman, **self.calc_cfgs)

class SubtractAzSSTemplate(_Preprocess):
    """Subtract Azimuth Synchronous Signal (AzSS) common template.
    Make common template by weighted mean or pca.
    This requires to calculate AzSS beforehand.

    Example configuration block::

      - name: "subtract_azss_template"
        process:
          signal: 'signal'
          azss: 'azss_stats_left'
          method: 'interpolate'
          scan_flags: 'left_scan'
          pca_modes: 1
          subtract: True

    .. autofunction:: sotodlib.tod_ops.azss.subtract_azss_template
    """
    name = "subtract_azss_template"

    def process(self, aman, proc_aman):
        tod_ops.azss.subtract_azss_template(aman, **self.process_cfgs)

class GlitchFill(_Preprocess):
    """Fill glitches. All process configs go to `fill_glitches`.
    Notes on flags. If flags are provided as step_cfgs, `proc_aman.get(flags)` is used.
    If provided as process_cfgs, `aman.get(glitch_flags)` is used instead.

    Example configuration block::

      - name: "glitchfill"
        signal: "hwpss_remove"
        flags: "glitches.glitch_flags" # optional
        process:
          nbuf: 10
          use_pca: False
          modes: 1
          in_place: True
          glitch_flags: "glitch_flags"
          wrap: None

    .. autofunction:: sotodlib.tod_ops.gapfill.fill_glitches
    """
    name = "glitchfill"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')
        self.flags = step_cfgs.get('flags')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        if self.flags is not None:
            glitch_flags=proc_aman.get(self.flags)
            tod_ops.gapfill.fill_glitches(
                aman, signal=aman[self.signal],
                glitch_flags=glitch_flags,
                **self.process_cfgs)
        else:
            tod_ops.gapfill.fill_glitches(
                aman, signal=aman[self.signal],
                **self.process_cfgs)

class FlagTurnarounds(_Preprocess):
    """From the Azimuth encoder data, flag turnarounds, left-going, and right-going.
        All process configs go to ``get_turnaround_flags``. If the ``method`` key
        is not included in the preprocess config file calc configs then it will
        default to 'scanspeed'.
    
    .. autofunction:: sotodlib.tod_ops.flags.get_turnaround_flags
    """
    name = 'flag_turnarounds'

    def calc_and_save(self, aman, proc_aman):
        if self.calc_cfgs is None:
            self.calc_cfgs = {}
            self.calc_cfgs['method'] = 'scanspeed'
        elif not('method' in self.calc_cfgs):
            self.calc_cfgs['method'] = 'scanspeed'

        if self.calc_cfgs['method'] == 'scanspeed':
            ta, left, right = tod_ops.flags.get_turnaround_flags(aman, **self.calc_cfgs)
            calc_aman = core.AxisManager(aman.dets, aman.samps)
            calc_aman.wrap('turnarounds', ta, [(0, 'dets'), (1, 'samps')])
            calc_aman.wrap('left_scan', left, [(0, 'dets'), (1, 'samps')])
            calc_aman.wrap('right_scan', right, [(0, 'dets'), (1, 'samps')])

        if self.calc_cfgs['method'] == 'az':
            ta = tod_ops.flags.get_turnaround_flags(aman, **self.calc_cfgs)
            calc_aman = core.AxisManager(aman.dets, aman.samps)
            calc_aman.wrap('turnarounds', ta, [(0, 'dets'), (1, 'samps')])

        if ('merge_subscans' not in self.calc_cfgs) or (self.calc_cfgs['merge_subscans']):
            calc_aman.wrap('subscan_info', aman.subscan_info)

        self.save(proc_aman, calc_aman)

    def save(self, proc_aman, turn_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("turnaround_flags", turn_aman)

    def process(self, aman, proc_aman):
        tod_ops.flags.get_turnaround_flags(aman, **self.process_cfgs)

class SubPolyf(_Preprocess):
    """Fit TOD in each subscan with polynominal of given order and subtract it.
        All process configs go to `sotodlib.tod_ops.sub_polyf`.
    
    .. autofunction:: sotodlib.tod_ops.subscan_polyfilter
    """
    name = 'sub_polyf'
    
    def process(self, aman, proc_aman):
        tod_ops.sub_polyf.subscan_polyfilter(aman, **self.process_cfgs)

class SSOFootprint(_Preprocess):
    """Find nearby sources within a given distance and get SSO footprint and plot
    each source on the focal plane.

    .. autofunction:: sotodlib.obs_ops.sources.get_sso
    """
    name = 'sso_footprint'

    def calc_and_save(self, aman, proc_aman):
        if self.calc_cfgs.get("source_list", None):
            ssos = self.calc_cfgs["source_list"]
        else:
            ssos = planets.get_nearby_sources(tod=aman, distance=self.calc_cfgs.get("distance", 20))
            if not ssos:
                raise ValueError("No sources found within footprint")
            ssos = [i[0] for i in ssos]
        sso_aman = core.AxisManager()
        nstep = self.calc_cfgs.get("nstep", 100)
        onsamp = (aman.samps.count+nstep-1)//nstep
        for sso in ssos:
            planet = sso
            xi_p, eta_p = obs_ops.sources.get_sso(aman, planet, nstep=nstep)
            planet_aman = core.AxisManager(core.OffsetAxis('ds_samps', count=onsamp,
                                                            offset=aman.samps.offset,
                                                            origin_tag=aman.samps.origin_tag))
            # planet_aman = core.AxisManager(core.OffsetAxis("samps", onsamp))
            planet_aman.wrap("xi_p", xi_p, [(0, "ds_samps")])
            planet_aman.wrap("eta_p", eta_p, [(0, "ds_samps")])
            sso_aman.wrap(planet, planet_aman)
        self.save(proc_aman, sso_aman)
        
    def save(self, proc_aman, sso_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("sso_footprint", sso_aman)

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_sso_footprint
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            for sso in proc_aman.sso_footprint._assignments.keys():
                planet_aman = proc_aman.sso_footprint[sso]
                plot_sso_footprint(aman, planet_aman, sso, filename=filename.replace('{name}', f'{sso}_sso_footprint'), **self.plot_cfgs)

class DarkDets(_Preprocess):
    """Find dark detectors in the data.

    Saves results in proc_aman under the "dark_dets" field. 

     Example config block::

        - name : "dark_dets"
          signal: "signal" # optional
          calc: True
          save: True
          select: True
    
    .. autofunction:: sotodlib.tod_ops.flags.get_dark_dets
    """
    name = "dark_dets"

    def calc_and_save(self, aman, proc_aman):
        mskdarks = tod_ops.flags.get_dark_dets(aman, merge=False)
        
        dark_aman = core.AxisManager(aman.dets, aman.samps)
        dark_aman.wrap('darks', mskdarks, [(0, 'dets'), (1, 'samps')])
        self.save(proc_aman, dark_aman)
    
    def save(self, proc_aman, dark_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("darks", dark_aman)
    
    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        keep = ~has_all_cut(proc_aman.darks.darks)
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

class SourceFlags(_Preprocess):
    """Calculate the source flags in the data.
    All calculation configs go to `get_source_flags`.

    Saves results in proc_aman under the "source_flags" field. 

     Example config block::

        - name : "source_flags"
          calc:
            mask: {'shape': 'circle',
                   'xyr': [0, 0, 1.]}
            center_on: ['jupiter', 'moon'] # list of str
            res: 20 # arcmin
            max_pix: 4000000 # max number of allowed pixels in map
            distance: 0 # max distance of footprint from source in degrees
          save: True
          select: True # optional

    .. autofunction:: sotodlib.tod_ops.flags.get_source_flags
    """
    name = "source_flags"

    def calc_and_save(self, aman, proc_aman):
        source_list = np.atleast_1d(self.calc_cfgs.get('center_on', 'planet'))
        if source_list == ['planet']:
            from sotodlib.coords.planets import SOURCE_LIST
            source_list = [x for x in aman.tags if x in SOURCE_LIST]
            if len(source_list) == 0:
                raise ValueError("No tags match source list")

        # find if source is within footprint + distance
        positions = planets.get_nearby_sources(tod=aman, source_list=source_list,
                                               distance=self.calc_cfgs.get('distance', 0))

        source_aman = core.AxisManager(aman.dets, aman.samps)
        for p in positions:
            source_flags = tod_ops.flags.get_source_flags(aman,
                                                          merge=self.calc_cfgs.get('merge', False),
                                                          overwrite=self.calc_cfgs.get('overwrite', True),
                                                          source_flags_name=self.calc_cfgs.get('source_flags_name', None),
                                                          mask=self.calc_cfgs.get('mask', None),
                                                          center_on=p[0],
                                                          res=self.calc_cfgs.get('res', None),
                                                          max_pix=self.calc_cfgs.get('max_pix', None))

            source_aman.wrap(p[0], source_flags, [(0, 'dets'), (1, 'samps')])

        # add sources that were not nearby from source list
        for source in source_list:
            if source not in source_aman._fields:
                source_aman.wrap(source, RangesMatrix.zeros([aman.dets.count, aman.samps.count]),
                                 [(0, 'dets'), (1, 'samps')])

        self.save(proc_aman, source_aman)

    def save(self, proc_aman, source_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("source_flags", source_aman)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            source_flags = meta.preprocess.source_flags
        else:
            source_flags = proc_aman.source_flags

        source_list = np.atleast_1d(self.calc_cfgs.get('center_on', 'planet'))
        if source_list == ['planet']:
            from sotodlib.coords.planets import SOURCE_LIST
            source_list = [x for x in aman.tags if x in SOURCE_LIST]
            if len(source_list) == 0:
                raise ValueError("No tags match source list")

        for source in source_list:
            if source in source_flags._fields:
                keep = ~has_all_cut(source_flags[source])
                meta.restrict("dets", meta.dets.vals[keep])
                source_flags.restrict("dets", source_flags.dets.vals[keep])
        return meta

class HWPAngleModel(_Preprocess):
    """Apply hwp angle model to the TOD.

    Saves results in proc_aman under the "hwp_angle" field. 

     Example config block::

        - name : "hwp_angle_model"
          process: True
          calc:
            on_sign_ambiguous: 'fail'
          save: True
          
    .. autofunction:: sotodlib.hwp.hwp_angle_model.apply_hwp_angle_model
    """
    name = "hwp_angle_model"

    def process(self, aman, proc_aman):
        if (not 'hwp_angle' in aman._fields) and ('hwp_angle' in proc_aman._fields):
            aman.wrap('hwp_angle', proc_aman['hwp_angle']['hwp_angle'],
                      [(0, 'samps')])
        else:
            return

    def calc_and_save(self, aman, proc_aman):
        hwp_angle_model.apply_hwp_angle_model(aman, **self.calc_cfgs)
        hwp_angle_aman = core.AxisManager(aman.samps)
        hwp_angle_aman.wrap('hwp_angle', aman.hwp_angle, [(0, 'samps')])
        self.save(proc_aman, hwp_angle_aman)

    def save(self, proc_aman, hwp_angle_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("hwp_angle", hwp_angle_aman)


class FourierFilter(_Preprocess):
    """
    Applies a fourier filter (defined in fft_ops) to the data.

    Example config file entry::

      - name: "fourier_filter"
        wrap_name: "lpf_sig"
        signal_name: "signal"
        process:
          filt_function: "low_pass_sine2"
          trim_samps: 2000
          filter_params:
            cutoff: 1
            width: 0.1
    
    or with params from a noise fit::

      - name: "fourier_filter"
        wrap_name: "lpf_sig"
        signal_name: "signal"
        process:
          filt_function: "low_pass_sine2"
          trim_samps: 2000
          noise_fit_array: "noiseQ_fit"

    See :ref:`fourier-filters` documentation for more details.
    """
    name = 'fourier_filter'

    def __init__(self, step_cfgs):
        self.signal_name = step_cfgs.get('signal_name', 'signal')
        # By default signal is overwritted by the filtered signal
        self.wrap_name = step_cfgs.get('wrap_name', 'signal')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        filt_function = self.process_cfgs.get(
            "filt_function",
            "high_pass_butter4"
        )
        _f = getattr(
            tod_ops.filters,
            filt_function
        )

        if self.process_cfgs.get("noise_fit_array"):
            field = self.process_cfgs["noise_fit_array"]
            _noise_fit = attrgetter(field)
            noise_fit = _noise_fit(proc_aman)
        else:
            noise_fit = None
        
        filter_params = tod_ops.fft_ops.build_hpf_params_dict(
            filt_function,
            noise_fit=noise_fit,
            filter_params=self.process_cfgs.get("filter_params", None)
        )

        filt = _f(**filter_params)

        filt_tod= tod_ops.filters.fourier_filter(aman, filt,
                                                 signal_name=self.signal_name)
        if self.wrap_name in aman._fields:
            aman.move(self.wrap_name, None)
        aman.wrap(self.wrap_name, filt_tod, [(0, 'dets'), (1, 'samps')])
        if self.process_cfgs.get("trim_samps"):
            trim = self.process_cfgs["trim_samps"]
            aman.restrict('samps', (aman.samps.offset + trim,
                                    aman.samps.offset + aman.samps.count - trim))
            proc_aman.restrict('samps', (proc_aman.samps.offset + trim,
                                         proc_aman.samps.offset + proc_aman.samps.count - trim))

class PCARelCal(_Preprocess):
    """
    Estimate the relcal factor from the atmosphere using PCA.

    Example configuration file entry::

      - name: 'pca_relcal'
        signal: 'lpf_sig'
        pca_run: 'run1'
        calc:
            pca:
                xfac: 2
                yfac: 1.5
                calc_good_medianw: True
            lpf:
                type: "sine2"
                cutoff: 1
                trans_width: 0.1
            trim_samps: 2000
        save: True
        plot:
            plot_ds_factor: 20

    See :ref:`pca-background` for more details on the method.
    """
    name = 'pca_relcal'

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')
        self.run = step_cfgs.get('pca_run', 'run1')
        self.run_name = f'{self.signal}_{self.run}'

        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        self.plot_signal = self.signal
        if self.calc_cfgs.get("lpf") is not None:
            filt = tod_ops.filters.get_lpf(self.calc_cfgs.get("lpf"))
            filt_tod = tod_ops.fourier_filter(aman, filt, signal_name='signal')

            filt_aman = core.AxisManager(aman.dets, aman.samps)
            filt_aman.wrap(self.signal, filt_tod, [(0, 'dets'), (1, 'samps')])

            if self.calc_cfgs.get("trim_samps") is not None:
                trim = self.calc_cfgs["trim_samps"]
                proc_aman.restrict('samps', (proc_aman.samps.offset + trim,
                                             proc_aman.samps.offset + proc_aman.samps.count - trim))
                filt_aman.restrict('samps', (filt_aman.samps.offset + trim,
                                             filt_aman.samps.offset + filt_aman.samps.count - trim))
            if self.plot_cfgs:
                self.plot_signal = filt_aman[self.signal]

        bands = np.unique(aman.det_info.wafer.bandpass)
        bands = bands[bands != 'NC']
        # align samps w/ proc_aman to include samps restriction when loading back from db.
        rc_aman = core.AxisManager(proc_aman.dets, proc_aman.samps)
        pca_det_mask = np.full(aman.dets.count, False, dtype=bool)
        relcal = np.zeros(aman.dets.count)
        pca_weight0 = np.zeros(aman.dets.count)
        for band in bands:
            m0 = aman.det_info.wafer.bandpass == band
            rc_aman.wrap(f'{band}_idx', m0, [(0, 'dets')])
            band_aman = aman.restrict('dets', aman.dets.vals[m0], in_place=False)

            filt_aman = filt_aman.restrict('dets', aman.dets.vals[m0], in_place=False)
            band_aman.merge(filt_aman)

            pca_out = tod_ops.pca.get_pca(band_aman,signal=band_aman[self.signal])
            pca_signal = tod_ops.pca.get_pca_model(band_aman, pca_out,
                                        signal=band_aman[self.signal])
            if self.calc_cfgs.get("pca") is None:
                result_aman = tod_ops.pca.pca_cuts_and_cal(band_aman, pca_signal)
            else:
                result_aman = tod_ops.pca.pca_cuts_and_cal(band_aman, pca_signal, **self.calc_cfgs.get("pca"))

            pca_det_mask[m0] = np.logical_or(pca_det_mask[m0], result_aman['pca_det_mask'])
            relcal[m0] = result_aman['relcal']
            pca_weight0[m0] = result_aman['pca_weight0']
            rc_aman.wrap(f'{band}_pca_mode0', result_aman['pca_mode0'], [(0, 'samps')])
            rc_aman.wrap(f'{band}_xbounds', result_aman['xbounds'])
            rc_aman.wrap(f'{band}_ybounds', result_aman['ybounds'])
            rc_aman.wrap(f'{band}_median', result_aman['median'])

        rc_aman.wrap('pca_det_mask', pca_det_mask, [(0, 'dets')])
        rc_aman.wrap('relcal', relcal, [(0, 'dets')])
        rc_aman.wrap('pca_weight0', pca_weight0, [(0, 'dets')])

        self.save(proc_aman, rc_aman)

    def save(self, proc_aman, pca_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap(self.run_name, pca_aman)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        keep = ~proc_aman[self.run_name]['pca_det_mask']
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

    def plot(self, aman, proc_aman, filename):
        if self.plot_cfgs is None:
            return
        if self.plot_cfgs:
            from .preprocess_plot import plot_pcabounds
            filename = filename.replace('{ctime}', f'{str(aman.timestamps[0])[:5]}')
            filename = filename.replace('{obsid}', aman.obs_info.obs_id)
            det = aman.dets.vals[0]
            ufm = det.split('_')[2]

            bands = np.unique(aman.det_info.wafer.bandpass)
            bands = bands[bands != 'NC']
            for band in bands:
                pca_aman = aman.restrict('dets', aman.dets.vals[proc_aman[self.run_name][f'{band}_idx']], in_place=False)
                band_aman = proc_aman[self.run_name].restrict('dets', aman.dets.vals[proc_aman[self.run_name][f'{band}_idx']], in_place=False)
                plot_pcabounds(pca_aman, band_aman, filename=filename.replace('{name}', f'{ufm}_{band}_pca'), signal=self.plot_signal, band=band, plot_ds_factor=self.plot_cfgs.get('plot_ds_factor', 20))

class PCAFilter(_Preprocess):
    """
    Applies a pca filter to the data.

    example config file entry::

      - name: "pca_filter"
        signal: "signal"
        process:
          n_modes: 10

    See :ref:`pca-background` for more details on the method.
    """
    name = 'pca_filter'

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        n_modes = self.process_cfgs.get('n_modes')
        signal = aman.get(self.signal)
        if aman.dets.count < n_modes:
            raise ValueError(f'The number of pca modes {n_modes} is '
                             f'larger than the number of detectors {aman.dets.count}.')
        model = tod_ops.pca.get_pca_model(aman, signal=signal, n_modes=n_modes)
        _ = tod_ops.pca.add_model(aman, model, signal=signal, scale=-1)

class FilterForSources(_Preprocess):
    """
    Mask and gap-fill the signal at samples flagged by source_flags.
    Then PCA the resulting time ordered data.

    example config file entry::

      - name: "filter_for_sources"
        signal: "signal"
        process:
          n_modes: 10
          source_flags: "source_flags"

    .. autofunction:: sotodlib.coords.planets.filter_for_sources
    """
    name = 'filter_for_sources'

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        n_modes = self.process_cfgs.get('n_modes')
        signal = aman.get(self.signal)
        flags = aman.flags.get(self.process_cfgs.get('source_flags'))
        if aman.dets.count < n_modes:
            raise ValueError(f'The number of pca modes {n_modes} is '
                             f'larger than the number of detectors {aman.dets.count}.')
        planets.filter_for_sources(aman, signal=signal, source_flags=flags, n_modes=n_modes)

class PTPFlags(_Preprocess):
    """Find detectors with anomalous peak-to-peak signal.

    Saves results in proc_aman under the "ptp_flags" field. 

     Example config block::

        - name : "ptp_flags"
          calc:
            signal_name: "dsT"
            kurtosis_threshold: 6
          save: True
          select: True

    .. autofunction:: sotodlib.tod_ops.flags.get_ptp_flags
    """
    name = "ptp_flags"

    def calc_and_save(self, aman, proc_aman):
        mskptps = tod_ops.flags.get_ptp_flags(aman, **self.calc_cfgs)

        ptp_aman = core.AxisManager(aman.dets, aman.samps)
        ptp_aman.wrap('ptp_flags', mskptps, [(0, 'dets'), (1, 'samps')])
        self.save(proc_aman, ptp_aman)

    def save(self, proc_aman, ptp_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("ptp_flags", ptp_aman)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        keep = ~has_all_cut(proc_aman.ptp_flags.ptp_flags)
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

class InvVarFlags(_Preprocess):
    """Find detectors with too high inverse variance.

    Saves results in proc_aman under the "inv_var_flags" field. 

     Example config block::

        - name : "inv_var_flags"
          calc:
            signal_name: "demodQ"
            nsigma: 6
          save: True
          select: True

    .. autofunction:: sotodlib.tod_ops.flags.get_inv_var_flags
    """
    name = "inv_var_flags"

    def calc_and_save(self, aman, proc_aman):
        msk = tod_ops.flags.get_inv_var_flags(aman, **self.calc_cfgs)

        inv_var_aman = core.AxisManager(aman.dets, aman.samps)
        inv_var_aman.wrap('inv_var_flags', msk, [(0, 'dets'), (1, 'samps')])
        self.save(proc_aman, inv_var_aman)

    def save(self, proc_aman, inv_var_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("inv_var_flags", inv_var_aman)

    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        keep = ~has_all_cut(proc_aman.inv_var_flags.inv_var_flags)
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

class EstimateT2P(_Preprocess):
    """Estimate T to P leakage coefficients.

    Saves results in proc_aman under the "t2p" field. 

     Example config block::

        - name : "estimate_t2p"
          calc:
            T_sig_name: 'dsT'
            Q_sig_name: 'demodQ'
            U_sig_name: 'demodU'
            joint_fit: True
            trim_samps: 2000
            lpf_cfgs:
              type: 'sine2'
              cutoff: 0.5
              trans_width: 0.1
          save: True
    
    .. autofunction:: sotodlib.tod_ops.t2pleakage.get_t2p_coeffs
    """
    name = "estimate_t2p"

    def calc_and_save(self, aman, proc_aman):
        t2p_aman = tod_ops.t2pleakage.get_t2p_coeffs(aman, **self.calc_cfgs)
        self.save(proc_aman, t2p_aman)

    def save(self, proc_aman, t2p_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("t2p", t2p_aman)

class SubtractT2P(_Preprocess):
    """Subtract T to P leakage.

     Example config block::

        - name : "subtract_t2p"
          process:
            Q_sig_name: 'demodQ'
            U_sig_name: 'demodU'
    
    .. autofunction:: sotodlib.tod_ops.t2pleakage.subtract_t2p
    """
    name = "subtract_t2p"

    def process(self, aman, proc_aman):
        tod_ops.t2pleakage.subtract_t2p(aman, proc_aman['t2p'],
                                        **self.process_cfgs)

class SplitFlags(_Preprocess):
    """Get flags used for map splitting/bundling.

    Saves results in proc_aman under the "split_flags" field.

     Example config block::

        - name : "split_flags"
          calc:
            high_gain: 0.115
            high_noise: 3.5e-5
            high_tau: 1.5e-3
            det_A: A
            pol_angle: 35
            det_top: B
            high_leakage: 1.0e-3
            high_2f: 1.5e-3
            right_focal_plane: 0
            top_focal_plane: 0
            central_pixels: 0.071
          save: True

    .. autofunction:: sotodlib.obs_ops.flags.get_split_flags
    """
    name = "split_flags"

    def calc_and_save(self, aman, proc_aman):
        split_flg_aman = obs_ops.splits.get_split_flags(aman, proc_aman, split_cfg=self.calc_cfgs)

        self.save(proc_aman, split_flg_aman)

    def save(self, proc_aman, split_flg_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("split_flags", split_flg_aman)

class UnionFlags(_Preprocess):
    """Do the union of relevant flags for mapping
    Typically you would include turnarounds, glitches, etc.

    Saves results for aman under the "flags.[total_flags_label]" field.

     Example config block::

        - name : "union_flags"
          process:
            flag_labels: ['jumps_2pi.jump_flag', 'glitches.glitch_flags', 'turnaround_flags.turnarounds']
            total_flags_label: 'glitch_flags'

    """
    name = "union_flags"

    def process(self, aman, proc_aman):
        from so3g.proj import RangesMatrix
        total_flags = RangesMatrix.zeros([proc_aman.dets.count, proc_aman.samps.count]) # get an empty flags with shape (Ndets,Nsamps)
        for label in self.process_cfgs['flag_labels']:
            _label = attrgetter(label)
            total_flags += _label(proc_aman) # The + operator is the union operator in this case

        if 'flags' not in aman._fields:
            from sotodlib.core import FlagManager
            aman.wrap('flags', FlagManager.for_tod(aman))
        if self.process_cfgs['total_flags_label'] in aman['flags']:
            aman['flags'].move(self.process_cfgs['total_flags_label'], None)
        aman['flags'].wrap(self.process_cfgs['total_flags_label'], total_flags)

class RotateQU(_Preprocess):
    """Rotate Q and U components to/from telescope coordinates.

    Example config block::

        - name : "rotate_qu"
          process:
            sign: 1 
            offset: 0 
            update_focal_plane: True

    .. autofunction:: sotodlib.coords.demod.rotate_demodQU
    """
    name = "rotate_qu"

    def process(self, aman, proc_aman):
        from sotodlib.coords import demod
        demod.rotate_demodQU(aman, **self.process_cfgs)

class SubtractQUCommonMode(_Preprocess):
    """Subtract Q and U common mode.

    Example config block::

        - name : 'subtract_qu_common_mode'
          signal_name_Q: 'demodQ'
          signal_name_U: 'demodU'
          process: True
          calc: True
          save: True

    .. autofunction:: sotodlib.tod_ops.deproject.subtract_qu_common_mode
    """
    name = "subtract_qu_common_mode"

    def __init__(self, step_cfgs):
        self.signal_name_Q = step_cfgs.get('signal_Q', 'demodQ')
        self.signal_name_U = step_cfgs.get('signal_U', 'demodU')
        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        self.save(proc_aman, aman)

    def save(self, proc_aman, aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap('qu_common_mode_coeffs', aman['qu_common_mode_coeffs'])

    def process(self, aman, proc_aman):
        if 'qu_common_mode_coeffs' in proc_aman:
            tod_ops.deproject.subtract_qu_common_mode(aman, self.signal_name_Q, self.signal_name_U,
                                                      coeff_aman=proc_aman['qu_common_mode_coeffs'], 
                                                      merge=False)
        else:
            tod_ops.deproject.subtract_qu_common_mode(aman, self.signal_name_Q,
                                                      self.signal_name_U, merge=True)

class FocalplaneNanFlags(_Preprocess):
    """Find additional detectors which have nans 
       in their focal plane coordinates.

    Saves results in proc_aman under the "fp_flags" field. 

     Example config block::

        - name : "fp_flags"
          signal: "signal" # optional
          calc:
              merge: False
          save: True
          select: True
    
    .. autofunction:: sotodlib.tod_ops.flags.get_focalplane_flags
    """
    name = "fp_flags"

    def calc_and_save(self, aman, proc_aman):
        mskfp = tod_ops.flags.get_focalplane_flags(aman, **self.calc_cfgs)
        fp_aman = core.AxisManager(aman.dets, aman.samps)
        fp_aman.wrap('fp_nans', mskfp, [(0, 'dets'), (1, 'samps')])
        self.save(proc_aman, fp_aman)
    
    def save(self, proc_aman, fp_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("fp_flags", fp_aman)
    
    def select(self, meta, proc_aman=None):
        if self.select_cfgs is None:
            return meta
        if proc_aman is None:
            proc_aman = meta.preprocess
        keep = ~has_all_cut(proc_aman.fp_flags.fp_nans)
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

class PointingModel(_Preprocess):
    """Apply pointing model to the TOD.

    Saves results in proc_aman under the "pointing" field. 

     Example config block::

        - name : "pointing_model"
          process: True

    .. autofunction:: sotodlib.coords.pointing_model.apply_pointing_model
    """
    name = "pointing_model"

    def process(self, aman, proc_aman):
        from sotodlib.coords import pointing_model
        if self.process_cfgs:
            pointing_model.apply_pointing_model(aman)

class CorrectIIRParams(_Preprocess):
    """Correct missing iir_params by default values.
    This corrects iir_params only when the observation is within the time_range
    that is known to have problem.

    Example config block::

        - name: "correct_iir_params"
          process: True

    .. autofunction:: sotodlib.obs_ops.utils.correct_iir_params
    """
    name = "correct_iir_params"

    def process(self, aman, proc_aman):
        from sotodlib.obs_ops import correct_iir_params
        correct_iir_params(aman)


_Preprocess.register(SplitFlags)
_Preprocess.register(SubtractT2P)
_Preprocess.register(EstimateT2P)
_Preprocess.register(InvVarFlags)
_Preprocess.register(PTPFlags)
_Preprocess.register(PCARelCal)
_Preprocess.register(PCAFilter)
_Preprocess.register(FilterForSources)
_Preprocess.register(FourierFilter)
_Preprocess.register(Trends)
_Preprocess.register(FFTTrim)
_Preprocess.register(Detrend)
_Preprocess.register(GlitchDetection)
_Preprocess.register(Jumps)
_Preprocess.register(FixJumps)
_Preprocess.register(PSDCalc)
_Preprocess.register(Noise)
_Preprocess.register(Calibrate)
_Preprocess.register(EstimateHWPSS)
_Preprocess.register(SubtractHWPSS)
_Preprocess.register(Apodize)
_Preprocess.register(Demodulate)
_Preprocess.register(AzSS)
_Preprocess.register(SubtractAzSSTemplate)
_Preprocess.register(GlitchFill)
_Preprocess.register(FlagTurnarounds)
_Preprocess.register(SubPolyf)
_Preprocess.register(DetBiasFlags)
_Preprocess.register(SSOFootprint)
_Preprocess.register(DarkDets)
_Preprocess.register(SourceFlags)
_Preprocess.register(HWPAngleModel)
_Preprocess.register(GetStats)
_Preprocess.register(UnionFlags)
_Preprocess.register(RotateQU)
_Preprocess.register(SubtractQUCommonMode)
_Preprocess.register(FocalplaneNanFlags)
_Preprocess.register(PointingModel)
_Preprocess.register(CorrectIIRParams)

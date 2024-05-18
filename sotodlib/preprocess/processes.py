import numpy as np
from operator import attrgetter

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
            n_pieces: 10
          save: True
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
        elif self.select_cfgs == "all":
            keep = ~has_all_cut(proc_aman.trends.trend_flags)
        else:
            raise ValueError(f"Entry '{self.select_cfgs['kind']}' not"
                                "understood. Expect 'any' or 'all'")
        meta.restrict("dets", meta.dets.vals[keep])
        return meta


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
        save: True
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
        signal: "hwpss_remove"
        calc:
          function: "twopi_jumps"
        save:
          jumps_name: "jumps_2pi"

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

class PSDCalc(_Preprocess):
    """ Calculate the PSD of the data and add it to the AxisManager under the
    "psd" field.

    Example config block::

      - "name : "psd"
        "signal: "signal" # optional
        "wrap": "psd" # optional
        "process":
          "psd_cfgs": # optional, kwargs to scipy.welch
            "nperseg": 1024
          "wrap_name": "psd" # optional
        "calc": True
        "save": True

    .. autofunction:: sotodlib.tod_ops.fft_ops.calc_psd
    """
    name = "psd"
    
    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')
        self.wrap = step_cfgs.get('wrap', 'psd')

        super().__init__(step_cfgs)
        

    def process(self, aman, proc_aman):
        freqs, Pxx = tod_ops.fft_ops.calc_psd(aman, signal=aman[self.signal],
                                              **self.process_cfgs)
        fft_aman = core.AxisManager(
            aman.dets, 
            core.OffsetAxis("nusamps",len(freqs))
        )
        fft_aman.wrap("freqs", freqs, [(0,"nusamps")])
        fft_aman.wrap("Pxx", Pxx, [(0,"dets"), (1,"nusamps")])
        aman.wrap(self.wrap, fft_aman)

    def calc_and_save(self, aman, proc_aman):
        self.save(proc_aman, aman[self.wrap])

    def save(self, proc_aman, fft_aman):
        if not(self.save_cfgs is None):
            proc_aman.wrap(self.wrap, fft_aman)

class Noise(_Preprocess):
    """Estimate the white noise levels in the data. Assumes the PSD has been
    wrapped into the AxisManager. All calculation configs goes to `calc_wn`. 

    Saves the results into the "noise" field of proc_aman. 

    Can run data selection of a "max_noise" value. 

    Example config block::

     - name: "noise"
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

        super().__init__(step_cfgs)

    def calc_and_save(self, aman, proc_aman):
        if self.psd not in aman:
            raise ValueError("PSD is not saved in AxisManager")
        psd = aman[self.psd]
        
        if self.calc_cfgs is None:
            self.calc_cfgs = {}
        
        if self.fit:
            calc_aman = tod_ops.fft_ops.fit_noise_model(aman, pxx=psd.Pxx, 
                                                        f=psd.freqs, 
                                                        merge_fit=True,
                                                        **self.calc_cfgs)
        else:
            wn = tod_ops.fft_ops.calc_wn(aman, pxx=psd.Pxx,
                                         freqs=psd.freqs,
                                         **self.calc_cfgs)
            calc_aman = core.AxisManager(aman.dets)
            calc_aman.wrap("white_noise", wn, [(0,"dets")])

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

        self.select_cfgs['name'] = self.select_cfgs.get('name','noise')

        if self.fit:
            keep = proc_aman[self.select_cfgs['name']].fit[:,1] <= self.select_cfgs["max_noise"]
        else:
            keep = proc_aman[self.select_cfgs['name']].white_noise <= self.select_cfgs["max_noise"]

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
            aman[self.signal] *= self.process_cfgs["val"]
        elif self.process_cfgs["kind"] == "array":
            field = self.process_cfgs["cal_array"]
            _f = attrgetter(field)
            aman[self.signal] = np.multiply(aman[self.signal].T, _f(aman)).T
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
            hwp.subtract_hwpss(
                aman,
                hwpss_template = template,
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
        hwp.demod_tod(aman, **self.process_cfgs)


class EstimateAzSS(_Preprocess):
    """Estimates Azimuth Synchronous Signal (AzSS) by binning signal by azimuth of boresight.
    All process confgis go to `get_azss`. If `method` is 'interpolate', no fitting applied 
    and binned signal is directly used as AzSS model. If `method` is 'fit', Legendre polynominal
    fitting will be applied and used as AzSS model.

    .. autofunction:: sotodlib.tod_ops.azss.get_azss
    """
    name = "estimate_azss"

    def calc_and_save(self, aman, proc_aman):
        calc_aman, _ = tod_ops.azss.get_azss(aman, **self.calc_cfgs)
        self.save(proc_aman, calc_aman)
    
    def save(self, proc_aman, azss_stats):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap(self.calc_cfgs["azss_stats_name"], azss_stats)

class GlitchFill(_Preprocess):
    """Fill glitches. All process configs go to `fill_glitches`.

    Example configuration block::

      - name: "glitchfill"
        signal: "hwpss_remove"
        flag_aman: "jumps_2pi"
        flag: "jump_flag"
        process:
          nbuf: 10
          use_pca: False
          modes: 1

    .. autofunction:: sotodlib.tod_ops.gapfill.fill_glitches
    """
    name = "glitchfill"

    def __init__(self, step_cfgs):
        self.signal = step_cfgs.get('signal', 'signal')
        self.flag_aman = step_cfgs.get('flag_aman', 'glitches')
        self.flag = step_cfgs.get('flag', 'glitch_flags')

        super().__init__(step_cfgs)

    def process(self, aman, proc_aman):
        tod_ops.gapfill.fill_glitches(
            aman, signal=aman[self.signal],
            glitch_flags=proc_aman[self.flag_aman][self.flag],
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
          signal: "signal" # optional
          calc:
            mask: {'shape': 'circle',
                   'xyr': (0, 0, 1.)}
            center_on: 'jupiter'
            res: 0.005817764173314432 # np.radians(20/60)
            max_pix: 4e6
          save: True
    
    .. autofunction:: sotodlib.tod_ops.flags.get_source_flags
    """
    name = "source_flags"
    
    def calc_and_save(self, aman, proc_aman):
        center_on = self.calc_cfgs.get('center_on', 'planet')
        # Get source from tags
        if center_on == 'planet':
            from sotodlib.coords.planets import SOURCE_LIST
            matches = [x for x in aman.tags if x in SOURCE_LIST]
            if len(matches) != 0:
                source = matches[0]
            else:
                raise ValueError("No tags match source list")
        else:
            source = center_on
        source_flags = tod_ops.flags.get_source_flags(aman, merge=False, center_on=source,
                                                      mask=self.calc_cfgs.get('mask', None),
                                                      res=self.calc_cfgs.get('res', None),
                                                      max_pix=self.calc_cfgs.get('max_pix', None))

        source_aman = core.AxisManager(aman.dets, aman.samps)
        source_aman.wrap('source_flags', source_flags, [(0, 'dets'), (1, 'samps')])
        self.save(proc_aman, source_aman)
    
    def save(self, proc_aman, source_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("sources", source_aman)

class HWPAngleModel(_Preprocess):
    """Apply hwp angle model to the TOD.

    Saves results in proc_aman under the "hwp_angle" field. 

     Example config block::

        - name : "hwp_angle_model"
          calc:
            on_sign_ambiguous: 'fail'
          save: True
    
    .. autofunction:: sotodlib.hwp.hwp_angle_model.apply_hwp_angle_model
    """
    name = "hwp_angle_model"
    
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
_Preprocess.register(EstimateAzSS)
_Preprocess.register(GlitchFill)
_Preprocess.register(FlagTurnarounds)
_Preprocess.register(SubPolyf)
_Preprocess.register(DetBiasFlags)
_Preprocess.register(SSOFootprint)
_Preprocess.register(DarkDets)
_Preprocess.register(SourceFlags)
_Preprocess.register(HWPAngleModel)

import numpy as np

import sotodlib.core as core
import sotodlib.flags as flags
import sotodlib.tod_ops as tod_ops
from sotodlib.hwp import hwp

from sotodlib.core.flagman import (has_any_cuts, has_all_cut,
                                   count_cuts,
                                    sparse_to_ranges_matrix)

from .core import _Preprocess


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
    def process(self, aman, proc_aman):
        tod_ops.detrend_tod(aman, **self.process_cfgs)
        
class Trends(_Preprocess):
    """Calculate the trends in the data to look for unlocked detectors. All
    calculation configs go to `get_trending_flags`.

    Saves results in proc_aman under the "trend" field. 

    Data selection can have key "kind" equal to "any" or "all."
    
    .. autofunction:: sotodlib.flags.get_trending_flags
    """
    name = "trends"
    
    def calc_and_save(self, aman, proc_aman):
        trend_cut, trend_aman = flags.get_trending_flags(
            aman, merge=False, full_output=True, 
            **self.calc_cfgs
        )
        aman.wrap("trends", trend_aman)
        self.save(proc_aman, trend_aman)
    
    def save(self, proc_aman, trend_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("trends", trend_aman)
    
    def select(self, meta):
        if self.select_cfgs is None:
            return meta
        if self.select_cfgs["kind"] == "any":
            keep = ~has_any_cuts(meta.preprocess.trends.trend_flags)
        elif self.select_cfgs == "all":
            keep = ~has_all_cut(meta.preprocess.trends.trend_flags)
        else:
            raise ValueError(f"Entry '{self.select_cfgs['kind']}' not"
                                "understood. Expect 'any' or 'all'")
        meta.restrict("dets", meta.dets.vals[keep])
        return meta

class GlitchDetection(_Preprocess):
    """Run glitch detection algorithm to find glitches. All calculation configs
    go to `get_glitch_flags` 

    Saves retsults in proc_aman under the "glitches" field.

    Data section should define a glitch significant "sig_glitch" and a maximum
    number of glitches "max_n_glitch."

    .. autofunction:: sotodlib.flags.get_glitch_flags
    """
    name = "glitches"
    
    def calc_and_save(self, aman, proc_aman):
        glitch_cut, glitch_aman = flags.get_glitch_flags(
            aman, merge=False, full_output=True,
            **self.calc_cfgs
        ) 
        aman.wrap("glitches", glitch_aman)
        self.save(proc_aman, glitch_aman)
    
    def save(self, proc_aman, glitch_aman):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap("glitches", glitch_aman)
 
    def select(self, meta):
        if self.select_cfgs is None:
            return meta
        
        flag = sparse_to_ranges_matrix(
            meta.preprocess.glitches.glitch_detection > self.select_cfgs["sig_glitch"]
        )
        n_cut = count_cuts(flag)
        keep = n_cut <= self.select_cfgs["max_n_glitch"]
        meta.restrict("dets", meta.dets.vals[keep])
        return meta
    
class PSDCalc(_Preprocess):
    """ Calculate the PSD of the data and add it to the AxisManager under the
    "psd" field. All process configs goes to `calc_psd`

    .. autofunction:: sotodlib.tod_ops.fft_ops.calc_psd
    """
    name = "psd"
    
    def process(self, aman, proc_aman):
        freqs, Pxx = tod_ops.fft_ops.calc_psd(aman, **self.process_cfgs)
        fft_aman = core.AxisManager(
            aman.dets, 
            core.OffsetAxis("fsamps",len(freqs))
        )
        fft_aman.wrap("freqs", freqs, [(0,"fsamps")])
        fft_aman.wrap("Pxx", Pxx, [(0,"dets"),(1,"fsamps")])
        aman.wrap("psd", fft_aman)

    def calc_and_save(self, aman, proc_aman):
        self.save(proc_aman, aman.psd)
    
    def save(self, proc_aman, fft_aman):
        if self.save_cfgs:
            proc_aman.wrap("psd", fft_aman)

class Noise(_Preprocess):
    """Estimate the white noise levels in the data. Assumes the PSD has been
    wrapped into the AxisManager. All calculation configs goes to `calc_wn`. 

    Saves the results into the "noise" field of proc_aman. 

    Can run data selection of a "max_noise" value. 
    
    .. autofunction:: sotodlib.tod_ops.fft_ops.calc_wn
    """
    name = "noise"
    
    def calc_and_save(self, aman, proc_aman):
        if "psd" not in aman:
            raise ValueError("PSD is not saved in AxisManager")

        wn = tod_ops.fft_ops.calc_wn(
            aman, 
            pxx=aman.psd.Pxx, 
            freqs=aman.psd.freqs, 
            **self.calc_cfgs
        )
        noise = core.AxisManager(aman.dets)
        noise.wrap("white_noise", wn, [(0,"dets")])
        aman.wrap("noise", noise)
        self.save(proc_aman, noise)
    
    def save(self, proc_aman, noise):
        if self.save_cfgs:
            proc_aman.wrap("noise", noise)

    def select(self, meta):
        if self.select_cfgs is None:
            return meta
        keep = meta.preprocess.noise.white_noise <= self.select_cfgs["max_noise"]
        meta.restrict("dets", meta.dets.vals[keep])
        return meta
    
class Calibrate(_Preprocess):
    """Calibrate the timestreams based on some provided information.

    Type of calibration is decided by process["kind"]

    1. "single_value" : multiplies entire signal by the single value
    process["val"]

    2. to be expanded
    """
    name = "calibrate"
    
    def process(self, aman, proc_aman):
        if self.process_cfgs["kind"] == "single_value":
            aman.signal *=  self.process_cfgs["val"]
        else:
            raise ValueError(f"Entry '{self.process_cfgs['kind']}'"
                              " not understood")

class EstimateHWPSS(_Preprocess):
    """
    Builds a HWPSS Template. Calc configs go to ``hwpss_model``.
    Results of fitting saved if field specified by calc["name"]

    .. autofunction:: sotodlib.hwp.hwp.get_hwpss
    """
    name = "estimate_hwpss"

    def calc_and_save(self, aman, proc_aman):
        hwpss_stats = hwp.get_hwpss(aman, **self.calc_cfgs)
        self.save(proc_aman, hwpss_stats)

    def save(self, proc_aman, hwpss_stats):
        if self.save_cfgs is None:
            return
        if self.save_cfgs:
            proc_aman.wrap(self.calc_cfgs["hwpss_stats_name"], hwpss_stats)

class SubtractHWPSS(_Preprocess):
    """Subtracts a HWPSS template from signal. 

    .. autofunction:: sotodlib.hwp.hwp.subtract_hwpss
    """
    name = "subtract_hwpss"

    def process(self, aman, proc_aman):
        hwp.subtract_hwpss(
            aman,
            hwpss_template = aman[self.process_cfgs["hwpss_extract"]],
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

_Preprocess.register(Trends.name, Trends)
_Preprocess.register(FFTTrim.name, FFTTrim)
_Preprocess.register(Detrend.name, Detrend)
_Preprocess.register(GlitchDetection.name, GlitchDetection)
_Preprocess.register(PSDCalc.name, PSDCalc)
_Preprocess.register(Noise.name, Noise)
_Preprocess.register(Calibrate.name, Calibrate)
_Preprocess.register(EstimateHWPSS.name, EstimateHWPSS)
_Preprocess.register(SubtractHWPSS.name, SubtractHWPSS)
_Preprocess.register(Apodize.name, Apodize)
_Preprocess.register(Demodulate.name, Demodulate)

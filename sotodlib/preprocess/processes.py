import numpy as np

import sotodlib.core as core
import sotodlib.flags as flags
import sotodlib.tod_ops as tod_ops

from sotodlib.core.flagman import (has_any_cuts, has_all_cut,
                                   count_cuts,
                                    sparse_to_ranges_matrix)

from .core import _Preprocess


class FFT_Trim(_Preprocess):
    name = "fft_trim"    
    def process(self, aman):
        tod_ops.fft_trim(aman, **self.process_cfgs)

class Detrend(_Preprocess):
    name = "detrend"
    def process(self, aman):
        tod_ops.detrend_tod(aman, **self.process_cfgs)
        
class Trends(_Preprocess):
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

class Glitch_Detection(_Preprocess):
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
    
class PSD_Calc(_Preprocess):
    name = "psd"
    
    def process(self, aman):
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
    name = "calibrate"
    
    def process(self, aman):
        if self.process_cfgs["kind"] == "single_value":
            aman.signal *=  self.process_cfgs["val"]
        else:
            raise ValueError(f"Entry '{self.process_cfgs['kind']}'"
                              " not understood")

_Preprocess.register(Trends.name, Trends)
_Preprocess.register(FFT_Trim.name, FFT_Trim)
_Preprocess.register(Detrend.name, Detrend)
_Preprocess.register(Glitch_Detection.name, Glitch_Detection)
_Preprocess.register(PSD_Calc.name, PSD_Calc)
_Preprocess.register(Noise.name, Noise)
_Preprocess.register(Calibrate.name, Calibrate)


#!/usr/bin/env python3
 
# Copyright (c) 2019-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
 
"""
This workflow analyses a simulated drone observation and makes I,Q,U maps of the  
source
 
- A single TOAST HDF5 observation of an artificial (drone-borne) source is loaded
  into a sotodlib AxisManager.
 
- Samples close to the drone are flagged with a radial mask, the chopper phase is
  estimated, and the data are double-demodulated (chopper + HWP).
  
This script is meant for post-processing simulations that already exist on disk.
It does not simulate data. It runs serially (no MPI).
 
Dump the default options with:
 
    drone_pol_analysis --help
 
Example:
 
    drone_pol_analysis --sims_file source-1-0_w25.h5 --tele satp1 --band f090 \\
        --chopper-freq 37 --out_dir drone_out --plots
"""
 
import argparse
import datetime
import os
import sys
import traceback
 
import numpy as np
 
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
import scipy.signal
 
from sotodlib import core, coords
import sotodlib.coords.helpers as helpers
import sotodlib.tod_ops as tod_ops
from sotodlib.tod_ops import filters
from sotodlib.tod_ops.detrend import detrend_tod
from sotodlib.tod_ops.filters import fourier_filter
from sotodlib.coords.demod import make_map as make_map_dm, rotate_demodQU
from sotodlib.coords.planets import get_det_weights
from sotodlib.sim_hardware import sim_nominal, sim_detectors_toast
 
import so3g
from so3g.proj import quat
 
from pixell import enmap, utils
 
import toast
from toast.timing import Timer
 
# Approximate beam FWHM per band, in arcminutes.
FWHM_ARCMIN = {"f090": 27.4, "f150": 19.0, "f220": 14.0, "f280": 12.0}
hwp_wobble = {'satp1':{"f090":[np.deg2rad(0.4 / 60), 4.33], 
                       "f150":[np.deg2rad(0.4 / 60), 4.30]}, 
              'satp3':{"f090":[np.deg2rad(2.82 / 60), 4.06], 
                       "f150":[np.deg2rad(2.96 / 60), 4.08]}}
 
def read_toast_h5_drone(filename):
    """Load a TOAST drone-simulation HDF5 file into an AxisManager.
 
    The source (drone) az/el are read from the ``shared/source`` column, which
    stores [-az, el] in degrees.
    """
    import h5py
 
    with h5py.File(filename, "r") as f:
        dets = [v["name"].decode("UTF-8") for v in f["instrument"]["focalplane"]]
        count = f["detdata"]["signal"].shape[1]
 
        aman = core.AxisManager(
            core.LabelAxis("dets", dets),
            core.OffsetAxis("samps", count, 0),
        )
        aman.wrap_new("signal", ("dets", "samps"), dtype="float32")
        aman.wrap_new("timestamps", ("samps",), dtype="float64")
        aman.timestamps[:] = f["shared"]["times"]
        aman.signal[:] = f["detdata"]["signal"]
 
        # Boresight pointing
        bman = core.AxisManager(aman.samps.copy())
        bman.wrap("az", np.array(f["shared"]["azimuth"]), [(0, "samps")])
        bman.wrap("el", np.array(f["shared"]["elevation"]), [(0, "samps")])
        bman.wrap("roll", np.zeros(count), [(0, "samps")])
        aman.wrap("boresight", bman)
 
        # Drone position
        source = np.array(f["shared"]["source"])
        aman.wrap("az_drone", source[:, 0], [(0, "samps")])
        aman.wrap("el_drone", source[:, 1], [(0, "samps")])
 
        if "hwp_angle" in f["shared"]:
            aman.wrap("hwp_angle", np.array(f["shared"]["hwp_angle"]), [(0, "samps")])
 
        focalplane = np.array(f["instrument"]["focalplane"])
        aman.wrap("focalplane", focalplane, [(0, "dets")])
 
        qboresight_azel = np.array(f["shared"]["boresight_azel"])
        aman.wrap("qboresight_azel", qboresight_azel)
 
    # Detector focal-plane coordinates
    q = helpers.ScalarLastQuat(aman.focalplane["quat"]).to_g3()
    xi, eta, gamma = quat.decompose_xieta(q)
    az_det, el_det, roll_det = quat.decompose_lonlat(aman.focalplane["quat"])
 
    fp = core.AxisManager(aman.dets)
    for key, value in zip(
        ["xi", "eta", "gamma", "az_det", "el_det", "roll_det"],
        [xi, eta, gamma, az_det, el_det, roll_det],
    ):
        fp.wrap_new(key, shape=("dets",))[:] = value
    aman.wrap("focal_plane", fp)
 
    return aman
 
def get_proj_matrix(aman, tele, band, size=10, res=1, defl_correction=True):
    """Projection matrix in drone-centered coordinates.
 
    Args:
        boresight_angle (float): boresight roll in degrees.
        size (float): map size in degrees.
        res (float): map resolution in arcminutes.
    """
    size = size * utils.degree
    res = res * utils.arcmin
 
    box = np.array([[-size / 2, -size / 2], [size / 2, size / 2]])
    geom = enmap.geometry(pos=box, res=res)
 
    sight = so3g.proj.CelestialSightLine.for_horizon(
        aman.timestamps,
        aman.boresight.az,
        aman.boresight.el,
        aman.boresight.roll,
    )
 
    pq = so3g.proj.quat.rotation_lonlat(-np.radians(aman.az_drone), aman.el_drone)
 
    if defl_correction is True:
        ph_def, ph_hwp = hwp_wobble[tele][band]
        dxi = ph_def*np.cos(aman.hwp_angle-ph_hwp) 
        deta = -ph_def*np.sin(aman.hwp_angle-ph_hwp)
        deflq = so3g.proj.quat.rotation_xieta(xi = dxi, eta = deta)
 
        pq *= deflq
        
    sight.Q = so3g.proj.quat.rotation_lonlat(0, 0) * ~pq * sight.Q
 
    fp = so3g.proj.quat.rotation_xieta(
        aman.focal_plane.xi, aman.focal_plane.eta, aman.focal_plane.gamma
    )
    return coords.pmat.P(sight=sight, fp=fp, geom=geom, comps="TQU")
 
def drone_radial_mask(aman, tele, band, boresight_angle=0.0, mask=(0.0, 0.0, 2.5)):
    """Flag samples within a circular region [x0, y0, r] (degrees) around the drone."""
    P = get_proj_matrix(aman, tele, band)
    x, y, r = mask
    mask_map = P.zeros()
    d = enmap.distance_from(
        mask_map.shape, mask_map.wcs, [[y * coords.DEG], [x * coords.DEG]]
    )
    mask_map += 1.0 * (d < r * coords.DEG)
    a = P.from_map(mask_map)
    return so3g.proj.RangesMatrix(
        [so3g.proj.Ranges.from_mask(row != 0) for row in a]
    )
 
def radial_mask_array(aman, tele, band, boresight_angle=0.0, fwhm_scale=3.0):
    """Build the drone radial mask and return it as a float array with NaN off-drone."""
    radius_deg = fwhm_scale * FWHM_ARCMIN[band] / 60
    source_flags = drone_radial_mask(
        aman, tele, band, boresight_angle=boresight_angle, mask=(0, 0, radius_deg)
    )
    mask_array = source_flags.mask().astype(int).astype(float)
    mask_array[np.where(source_flags == 0.0)] = np.nan
    return source_flags, mask_array
 
def compute_dist_flags(aman, boresight_angle=0.0):
    """Per-sample angular distance (a map projected to TOD) from the drone center."""
    P = get_proj_matrix(aman, boresight_angle=boresight_angle)
    mask_map = P.zeros()
    d = enmap.distance_from(mask_map.shape, mask_map.wcs, [[0.0], [0.0]])
    mask_map += d
    return P.from_map(mask_map)
 
def get_dets_w_drone(aman, mask_array, threshold=10):
    """Indices of detectors with a masked signal peak above ``threshold``."""
    dets_w_drone = []
    for n in range(aman.dets.count):
        peak_idxs, _ = scipy.signal.find_peaks(aman.signal[n], distance=10000)
        peak_idxs_sorted = peak_idxs[np.argsort(aman.signal[n][peak_idxs])][::-1][:5]
        for peak in peak_idxs_sorted:
            if mask_array[n][peak] == 1 and aman.signal[n][peak] > threshold:
                dets_w_drone.append(n)
    return dets_w_drone
 
def get_psds(aman, mask_array, threshold=10):
    """Stack the raw signal over drone crossings and return (mean rFFT, stacked TOD)."""
    psds = []
    signal_copy = np.zeros(len(aman.timestamps))
 
    for det_idx in range(aman.dets.count):
        det_sees_drone = np.where(mask_array[det_idx] == 1)
        if len(det_sees_drone) == 0:
            continue
 
        # Split the flagged samples into contiguous drone-crossing segments.
        diff_idxs = np.diff(det_sees_drone)[0]
        peak_stops = np.where(diff_idxs > 1)[0]
 
        for peak_num in range(len(peak_stops) + 1):
            idx_start = 0 if peak_num == 0 else peak_stops[peak_num - 1] + 1
            idx_stop = -1 if peak_num == len(peak_stops) else peak_stops[peak_num]
 
            try:
                signal_temp = aman.signal[det_idx][det_sees_drone][idx_start:idx_stop]
                if np.nanmax(signal_temp) < threshold:
                    continue
                min_samp = det_sees_drone[0][idx_start]
                max_samp = det_sees_drone[0][idx_stop]
                signal_copy[min_samp:max_samp] += signal_temp
                psds.append(np.fft.rfft(signal_copy))
            except (ValueError, IndexError):
                continue
 
    psds = np.array(psds)
    psd_sum = psds.sum(axis=0) / len(psds)
    return psd_sum, signal_copy
 
def estimate_chopper_phase(aman, signal_copy, chopper_freq):
    """Estimate the chopper demodulation phase from the stacked drone signal."""
    dt = aman.timestamps - aman.timestamps[0]
 
    aman2 = core.AxisManager(
        core.LabelAxis("dets", ["q", "u"]),
        core.OffsetAxis("samps", len(dt)),
    )
    aman2.wrap_new("signal", shape=("dets", "samps"), dtype="float32")
    aman2.wrap("timestamps", dt, axis_map=[(0, "samps")])
 
    phasor = np.exp(2j * np.pi * chopper_freq * dt)
    demod = phasor * signal_copy
    aman2.signal[0] = demod.real
    aman2.signal[1] = demod.imag
 
    lpf = tod_ops.filters.low_pass_sine2(1.0, 2.0)
    sig2 = tod_ops.fourier_filter(aman2, lpf)
 
    # Signal-to-noise weighted mean phase.
    sig_strength = (sig2 ** 2).sum(axis=0)
    qu_mean = np.dot(sig2, sig_strength) / sig_strength.sum()
    return np.arctan2(qu_mean[1], qu_mean[0])
 
def demod_tod_double(
    aman,
    tele,
    band,
    signal_name="signal",
    demod_mode=4,
    freq_chopper=47.0,
    phase_chopper=0.0,
    bpf_cfg=None,
    lpf_cfg=None,
):
    """Double demodulation (chopper + HWP), writing dsT, demodQ, demodU into ``aman``.
 
    The chopper is demodulated by multiplying the signal with a cosine reference,
    then the HWP sideband is band-passed and demodulated with the effective-angle
    phasor as usual. 
    """
    speed = np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) 
    speed /= (aman.timestamps[-1] - aman.timestamps[0]) * (2 * np.pi)
    bandwidth = 2 * 0.95 * speed
 
    if bpf_cfg is None:
        bpf_cfg = {
            "type": "sine2",
            "center": demod_mode * speed,
            "width": bandwidth,
            "trans_width": 0.1,
        }
    bpf = filters.get_bpf(bpf_cfg)
 
    # HWP effective angle (accounts for non-ideality).
    epsilon, psi = hwp_wobble[tele][band]
    theta_effective = aman.hwp_angle + epsilon * np.cos(aman.hwp_angle - psi)
    phasor = np.exp(np.abs(demod_mode) * 1.0j * theta_effective)
 
    # Chopper demodulation (in place on the signal).
    time = aman.timestamps - aman.timestamps[0]
    phasor_chopper = np.cos(2 * np.pi * freq_chopper * time + phase_chopper)
    aman.signal *= phasor_chopper
 
    demod = fourier_filter(aman, bpf, detrend=None, signal_name=signal_name) * phasor
 
    if lpf_cfg is None:
        lpf_cfg = {"type": "sine2", "cutoff": speed * 0.95, "trans_width": 0.1}
    lpf = filters.get_lpf(lpf_cfg)
 
    aman.wrap_new("dsT", dtype="float32", shape=("dets", "samps"))
    aman["dsT"] = fourier_filter(aman, lpf, signal_name="signal", detrend=None) * 4
 
    aman.wrap_new("demodQ", dtype="float32", shape=("dets", "samps"))
    aman["demodQ"] = demod.real
    aman["demodQ"] = fourier_filter(aman, lpf, signal_name="demodQ", detrend=None) * 2.0 * 4
 
    aman.wrap_new("demodU", dtype="float32", shape=("dets", "samps"))
    aman["demodU"] = demod.imag
    aman["demodU"] = fourier_filter(aman, lpf, signal_name="demodU", detrend=None) * 2.0 * 4
 
    return aman
 
def make_demod_map(aman, tele, band, size=5, outlier_clip=2.0):
    """Demodulated I/Q/U map in drone-centered coordinates."""
    P = get_proj_matrix(aman, tele, band, size=size)
    det_weights = get_det_weights(aman, signal=aman.signal, outlier_clip=outlier_clip)
    return make_map_dm(aman, P=P, det_weights_demod=det_weights)["map"]
 
def get_hw_positions(band, wafer):
    """Nominal hardware xi/eta/gamma focal-plane positions for a band/wafer."""
    hw = sim_nominal()
    sim_detectors_toast(hw, "SAT1")
 
    qdr, names_all = [], []
    for name in hw.data["detectors"].keys():
        if band in name and wafer in name:
            q = hw.data["detectors"][name]["quat"]
            qdr.append([q[3]] + list(q[:3]))
            names_all.append(name)
 
    quat_det = so3g.proj.quat.G3VectorQuat(np.array(qdr))
    xi_h, eta_h, gamma_h = so3g.proj.quat.decompose_xieta(quat_det)
    return xi_h, eta_h, gamma_h, names_all
 
def plot_maps(map_i, out_dir):
    """Three-panel dB I/Q/U map."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300)
    norm = np.nanmax(map_i[0])
    ax1.imshow(10 * np.log10(np.abs(map_i[0] / norm)), vmin=-30, vmax=0, cmap="magma")
    ax2.imshow(10 * np.log10(np.abs(map_i[1] / norm)), vmin=-30, vmax=0, cmap="magma")
    im = ax3.imshow(
        10 * np.log10(np.abs(map_i[2] / norm)), vmin=-30, vmax=0, cmap="magma"
    )
 
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([30, 90, 150, 210, 270], [-2, -1, 0, 1, 2])
    ax1.set_yticks([30, 90, 150, 210, 270], [-2, -1, 0, 1, 2])
    ax1.set_ylabel("Elevation [deg]", size=11)
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax1.set_title("I", size=13)
    ax2.set_title("Q", size=13)
    ax3.set_title("U", size=13)
 
    cbar_axis = fig.add_axes([0.92, 0.3319, 0.02, 0.3265])
    cbar = fig.colorbar(im, cax=cbar_axis)
    cbar.set_label("dB", rotation=270, size=14, labelpad=20)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(os.path.join(out_dir, "example_maps.png"), bbox_inches="tight")
    plt.close(fig)
 
### Basic analysis run ###
 
def run_analysis(args):
    """Load the observation and recover corrected per-detector polarization angles."""
    log = toast.utils.Logger.get()
 
    log.info(f"Loading {args.sims_file}")
    aman = read_toast_h5_drone(args.sims_file)
    detrend_tod(aman)
 
    # Radial mask around the drone (a few beam FWHM).
    _, mask_array = radial_mask_array(
        aman, args.tele, args.band, boresight_angle=args.boresight_angle, fwhm_scale=args.mask_fwhm_scale
    )
 
    dets_w_drone = get_dets_w_drone(aman, mask_array, threshold=args.peak_threshold)
    log.info(f"Detectors seeing the drone: {len(dets_w_drone)}")
 
    # Estimate the chopper phase from the stacked drone crossings, then demodulate.
    _, signal_copy = get_psds(aman, mask_array, threshold=args.peak_threshold)
    phase = estimate_chopper_phase(aman, signal_copy, args.chopper_freq)
    log.info(f"Estimated chopper phase: {np.degrees(phase):.3f} deg")
 
    demod_tod_double(
        aman,
        tele=args.tele,
        band=args.band,
        demod_mode=args.demod_mode,
        freq_chopper=args.chopper_freq,
        phase_chopper=-phase,
    )
    rotate_demodQU(aman)
 
    maps = make_demod_map(aman, args.tele, args.band)
 
    if args.plots:
        plot_maps(
            maps,
            args.out_dir, 
        )
 
 
def main():
    log = toast.utils.Logger.get()
    timer = Timer()
    timer.start()
 
    log.info(f"Starting drone polarization analysis at {datetime.datetime.now()}")
 
    parser = argparse.ArgumentParser(description="Drone polarization analysis pipeline")
 
    parser.add_argument(
        "--sims_file",
        required=True,
        type=str,
        help="Input TOAST drone-simulation HDF5 file",
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="drone_out",
        help="The output directory",
    )
    parser.add_argument(
        "--band",
        required=False,
        type=str,
        default="f090",
        help="Frequency band (f090, f150, f220, f280)",
    )
    parser.add_argument(
        "--tele",
        required=False,
        type=str,
        default="satp1",
        help="Telescope (satp1, satp3)",
    )
    parser.add_argument(
        "--wafer",
        required=False,
        type=str,
        default="w25",
        help="Wafer slot (used only for diagnostic focal-plane plots)",
    )
    parser.add_argument(
        "--boresight-angle",
        required=False,
        type=float,
        default=0.0,
        help="Boresight roll angle in degrees",
    )
    parser.add_argument(
        "--chopper-freq",
        required=False,
        type=float,
        default=37.0,
        help="Source chopping frequency in Hz",
    )
    parser.add_argument(
        "--demod-mode",
        required=False,
        type=int,
        default=4,
        help="HWP demodulation mode",
    )
    parser.add_argument(
        "--map-size",
        required=False,
        type=float,
        default=5.0,
        help="Demodulated map size in degrees",
    )
    parser.add_argument(
        "--mask-fwhm-scale",
        required=False,
        type=float,
        default=3.0,
        help="Drone mask radius as a multiple of the beam FWHM",
    )
    parser.add_argument(
        "--peak-threshold",
        required=False,
        type=float,
        default=10.0,
        help="Minimum signal peak height to count as a drone crossing",
    )
    parser.add_argument(
        "--phase-offset",
        required=False,
        type=float,
        default=100.0,
        help="Injected phase offset, used only for labeling outputs",
    )
    parser.add_argument(
        "--chopping-mode",
        required=False,
        type=str,
        default="square",
        help="Chopping waveform, used only for labeling outputs",
    )
    parser.add_argument(
        "--proper-demod",
        required=False,
        type=str,
        default="yes",
        help="Demodulation label, used only for labeling outputs",
    )
    parser.add_argument(
        "--plots",
        required=False,
        default=False,
        action="store_true",
        help="Save diagnostic figures to the output directory",
    )
 
    args = parser.parse_args()
 
    # Create our output directory
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
 
    run_analysis(args)
 
    log.info_rank("Workflow completed in", timer=timer)
 
 
def cli():
    try:
        main()
    except Exception:
        print("Proc failed with exception:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
 
 
if __name__ == "__main__":
    cli()

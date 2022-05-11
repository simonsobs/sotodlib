#!/usr/bin/env python3
import so3g
from so3g.proj import coords, quat
import sotodlib.coords.planets as planets
from sotodlib import core
from sotodlib.io.load import load_file
from sotodlib.toast import sim_sso

import h5py
import numpy as np
import scipy
import scipy.stats
import scipy.signal
import time, ephem, os
from scipy.optimize import curve_fit
from glob import glob
from datetime import datetime
import argparse as ap
import pandas as pd
from mpi4py import MPI

opj = os.path.join
INITIAL_PARA_FILE = "data/initial_parameters.hdf5"


def highpass_filter(data, cutoff, fs, order=5):
    """High-pass filtering used especiallyy for SAT

    Args:
    ----
    cutoff: cuttoff frequency
    fs: sampling rate

    Returns:
    -------
    High-passed timestream
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="high", analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def read_toast_h5(filename):
    """Convert h5 file to AxisManager object"""

    f = h5py.File(filename, "r")

    dets = [v["name"] for v in f["instrument"]["focalplane"]]
    count = f["detdata"]["signal"].shape[1]

    aman = core.AxisManager(
        core.LabelAxis("dets", dets),
        core.OffsetAxis("samps", count, 0),
    )

    aman.wrap_new("signal", ("dets", "samps"), dtype="float32")
    aman.wrap_new("timestamps", ("samps",), dtype="float64")

    bman = core.AxisManager(aman.samps.copy())
    bman.wrap("az", f["shared"]["azimuth"], [(0, "samps")])
    bman.wrap("el", f["shared"]["elevation"], [(0, "samps")])
    bman.wrap("roll", 0 * bman.az, [(0, "samps")])
    aman.wrap("boresight", bman)

    aman.timestamps[:] = f["shared"]["times"]
    aman.signal[:] = f["detdata"]["signal"]

    if "hwp_angle" in f["shared"]:
        aman.wrap("hwp_angle", f["shared"]["hwp_angle"].value, [(0, "samps")])

    aman.wrap("toast_focalplane", f["instrument"]["focalplane"], [(0, "dets")])

    return aman


def read_context(filename):
    """Convert Context to AxisManager object"""

    ctx = Context(opj(filename, "context", "context.yaml"))
    obs_id = ctx.obsdb.get()[0]["obs_id"]
    tod = ctx.get_obs(obs_id)

    dets = tod.focal_plane.dets.vals
    count = tod.signal.shape[1]

    aman = core.AxisManager(
        core.LabelAxis("dets", dets),
        core.OffsetAxis("samps", count, 0),
    )

    aman.wrap_new("signal", ("dets", "samps"), dtype="float32")
    aman.wrap_new("timestamps", ("samps",), dtype="float64")

    bman = core.AxisManager(aman.samps.copy())
    bman.wrap("az", tod.boresight.az, [(0, "samps")])
    bman.wrap("el", tod.boresight.el, [(0, "samps")])
    aman.wrap("boresight", bman)

    aman.timestamps[:] = tod.timestamps
    aman.signal[:] = tod.signal

    if "hwp_angle" in dir(tod):
        aman.wrap("hwp_angle", tod.hwp_angle, [(0, "samps")])

    bman = core.AxisManager(aman.dets.copy())
    bman.wrap("xi", tod.focal_plane.xi, [(0, "dets")])
    bman.wrap("eta", tod.focal_plane.eta, [(0, "dets")])
    bman.wrap("gamma", tod.focal_plane.gamma, [(0, "dets")])
    aman.wrap("toast_focalplane", bman)

    return aman


def load_data(path, suffix, f_format=None):
    """Load simulation from folder/database
        -- this should be changed

    Args
    ------
    path: location of the simulation
            to be loaded
    band: frequency bands available 'f030', 'f040',
            'f090', 'f150', 'f230', 'f290'
    tele: telescope name to be chosen from 'LAT',
            'SAT1', 'SAT2', 'SAT3' (, 'SAT4')

    Return
    ------
    full tod AxisManager object
    """

    ## All naming connections are subject to change and need to be decided

    ## This is just an example block

    if f_format != "g3" and f_format != "h5" and f_format != "context":
        raise ValueError("Invalid observation format provided")

    try:
        if f_format == "g3":
            full_path = opj(path, suffix)
            tod = load_file(sorted(glob(full_path)))
        elif f_format == "h5":
            tod = read_toast_h5(path)
        else:
            tod = read_context(path)

    except:
        ## since this fitting will happen for multiple sims perhaps
        ## we should not raise an extension and instead move to the
        ## next sim
        raise NameError("The requested simulation is not in database")

    return tod


def get_xieta_src_centered(ctime, az, el, data, sso_name, threshold=5):
    """Create a planet centered coordinate system for single detector data

    Args
    ------
    ctime, az, el: unix time and boresight position coordinates
    data: single detector data
    sso_name: celestial source to insert to pyephem
    threshold: minimal ~SNR required for the detector timestream
               to be fitted.

    Return
    ------
    xi, eta : full coordinate arrays
    xi0, eta0 : position of planet crossing
    q_det : detector quaternion
    q_bore : boresight quaternion
    q_obj : source quaternion
    """

    csl = so3g.proj.CelestialSightLine.az_el(ctime, az, el, weather="typical")
    q_bore = csl.Q

    ### Signal strength criterion - a chance to flag detectors (should be extra rare)
    peaks, _ = scipy.signal.find_peaks(data, height=threshold * np.std(data))
    if not list(peaks):
        return

    ## The mean planet position is used
    dt = datetime.fromtimestamp(np.mean(ctime))
    dt_str = dt.strftime("%Y/%m/%d %H:%M:%S")
    obj = getattr(ephem, sso_name)()
    obj.compute(dt_str)
    ra_obj = obj.ra
    dec_obj = obj.dec
    q_obj = quat.rotation_lonlat(ra_obj, dec_obj)

    ### rotate coord system around max point
    src_idx = np.where(data == np.max(data[peaks]))[0][0]
    q_bs_t = coords.quat.quat(*np.array(q_bore)[src_idx])
    xi0, eta0, _ = quat.decompose_xieta(~q_bs_t * q_obj)
    q_det = quat.rotation_xieta(xi0, eta0)
    q_total = ~q_det * ~q_bore * q_obj

    xi, eta, _ = quat.decompose_xieta(q_total)
    xi0, eta0, _ = quat.decompose_xieta(q_det)

    return np.array([xi, eta]), np.array([xi0, eta0]), q_det, q_bore, q_obj


def define_fit_params(band, tele, return_beamsize=False):
    """
    Define initial position and allowed offsets for detector centroids
    and beam parameters according to frequency and telescope specifications.
    The time constant initial guess and boundary conditions remain the same.
    Args
    ------
    band, tele : frequency band and telescope
    Return
    ------
    initial_guess, bounds : arguments for the least square fits

    """

    hf = h5py.File(INITIAL_PARA_FILE, "r")
    # band=bytes(f"{band}", encoding="utf-8")
    ## unecessary / write differently
    if tele != "LAT":
        import re

        tele = re.split("(\d+)", tele)[0]
    try:
        idx = list(hf.get(tele)["frequency-band name"]).index(band)
    except:
        raise KeyError("Telescope name must be one of LAT,SAT")

    beamsize = (hf.get(tele)["beam size"])[idx]
    amp = (hf.get(tele)["detector response"])[idx]
    ## perhaps reconsider this bound
    beamsize = np.radians(beamsize / 60)
    offset_bound = 2 * beamsize

    ## allow for max 10% bias
    ## 10% is large but allowing for a large bias helps tracking down
    ## problems in the fitting method
    ## outliers should be eliminated afterwards
    ## We can make this a user input parameter
    fwhm_bound = 0.1 * beamsize
    fwhm_min = beamsize - fwhm_bound
    fwhm_max = beamsize + fwhm_bound

    initial_guess = [amp, 0, 0, beamsize, beamsize, np.pi / 6, 3e-3]

    bounds_min = (0, -offset_bound, -offset_bound, fwhm_min, fwhm_min, 0.0, 1e-3)
    bounds_max = (2 * amp, offset_bound, offset_bound, fwhm_max, fwhm_max, np.pi, 10e-3)
    bounds = np.array(
        (
            bounds_min,
            bounds_max,
        )
    )

    return initial_guess, bounds, offset_bound


def gaussian2d(xieta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi):
    """Simulate a time stream with an Gaussian beam model
    Args
    ------
    xieta: 2d array of float
        containing xieta in the detector-centered coordinate system
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated) and the rotation angle (in radians)

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi, eta = xieta
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data


def tconst_convolve(ctime, data, tau):
    """Convolve the time stream with a given time constant
    Args
    ------
    ctime: 1d array of float
        containting the ctime of each data point
    data: 1d array of float
        time stream data for a given detector
    tau: float
        time constant to be added to the time stream (in seconds)

    Return
    ------
    data_tau: 1d array of float
        time-constant filtered time stream
    """
    k = int(np.ceil(np.log(len(data)) / np.log(2)))
    n = 2 ** k
    delta_time = np.mean(np.diff(ctime))
    freqs = np.fft.fftfreq(n, delta_time)
    data_pad = np.zeros(n)
    data_pad[: len(data)] = data
    fft_data = np.fft.fft(data_pad)
    fft_data *= 1.0 / (1.0 + 2.0j * np.pi * tau * freqs)
    data_tau = np.fft.ifft(fft_data).real[: len(data)]
    return data_tau


def tod_sim_pointing(space_time_pointing, a, xi0, eta0):
    """Time simulator including only the pointing information
    Args
    ------
    space_time_pointing: dict
        containing pointing and time info keyed as 'xieta' and 'time',
        'idx_main' as the indice within the continous band data for the
        main beam; 'fwhm_xi', 'fwhm_eta', 'phi', 'tau' for set beam
        information and the time constant
    a, xi0, eta0: all floats
        parameters for the pointing information, as elaborated in the gaussian2d function

    Return
    ------
    data_sim_tau: 1d array of float
        simulated data with a specific pointing
    """
    fwhm_xi = space_time_pointing["fwhm_xi"]
    fwhm_eta = space_time_pointing["fwhm_eta"]
    phi = space_time_pointing["phi"]
    tau = space_time_pointing["tau"]
    data_sim = gaussian2d(
        space_time_pointing["xieta"], a, xi0, eta0, fwhm_xi, fwhm_eta, phi
    )
    if tau is None:
        return data_sim
    data_sim_tau = tconst_convolve(space_time_pointing["time"], data_sim, tau)
    idx_main = space_time_pointing["idx_main"]
    return data_sim_tau[idx_main]


def tod_sim_beam(space_time_beam, fwhm_xi, fwhm_eta, phi):
    """Time simulator including beam morphology information
    Args
    ------
    space_time: dict
        containing pointing and time info keyed as 'xieta' and 'time',
        'idx_main' as the indice within the continous band data for the
        main beam; 'a', 'xi_off', 'eta_off', 'tau' for set pointing and
        time constant
    fwhm_xi, fwhm_eta, phi: all floats
        parameters for the beam model, as elaborated in the gaussian2d function

    Return
    ------
    data_sim_tau: 1d array of float
        simulated data with a specific beam model and tau value
    """
    a = space_time_beam["a"]
    xi0 = space_time_beam["xi_off"]
    eta0 = space_time_beam["eta_off"]
    tau = space_time_beam["tau"]
    data_sim = gaussian2d(
        space_time_beam["xieta"], a, xi0, eta0, fwhm_xi, fwhm_eta, phi
    )
    if tau is None:
        return data_sim
    data_sim_tau = tconst_convolve(space_time_beam["time"], data_sim, tau)
    idx_main = space_time_beam["idx_main"]
    return data_sim_tau[idx_main]


def tod_sim_all(space_time, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, tau):
    """Time simulator including beam model and time constant
    Args
    ------
    space_time: dict
        containing pointing and time info keyed as 'xieta' and 'time'
    a, xi0, eta0, fwhm_xi, fwhm_eta, phi: all floats
        parameters for the beam model, as elaborated in the gaussian2d function
    tau: float
        time constant

    Return
    ------
    data_sim_tau: 1d array of float
        simulated data with a specific beam model and tau value
    """
    data_sim = gaussian2d(space_time["xieta"], a, xi0, eta0, fwhm_xi, fwhm_eta, phi)
    if tau is None:
        return data_sim
    data_sim_tau = tconst_convolve(space_time["time"], data_sim, tau)
    idx_main = space_time["idx_main"]
    return data_sim_tau[idx_main]


def fit_params(
    tod, ctime, az, el, band, sso_name, det_idx, highpass, cutoff, init_params
):
    """Function that fits individual time-streams and returns the parameters

    Args
    ----
    tod: AxisManager
        The tod loaded simulation object
    band: string
        The frequency-band
    sso_name: string
        Name of the celestial source
    det_idx: int
        The index of the detector as ordered in the AxisManager object
    highpass: bool
        If True use a butterworth filter on the data
    """

    initial_guess, bounds, radius_main = init_params
    radius_cut = 2 * radius_main

    data = tod.signal[det_idx, :]
    data -= np.mean(data)
    sample_rate = 1.0 / np.mean(np.diff(tod.timestamps))

    if highpass and (cutoff is not None):
        data = highpass_filter(data, cutoff, sample_rate)
        # Eliminate the phase delay caused by the filter in a more
        # elegant way
        data[data < 0] = 0
    coord_transforms = get_xieta_src_centered(ctime, az, el, data, sso_name)

    if coord_transforms is None:
        all_params = np.empty(9)
        all_params[:] = np.nan
        return all_params

    total_coords, peak_coords, q_det, q_bore, q_obj = coord_transforms

    xi_det_center, eta_det_center = total_coords
    xi0, eta0 = peak_coords
    radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)

    ## We can get away with much less subdivisions of the data
    ## This can be changed after agreed upon
    idx_band = np.where(abs(eta_det_center) < radius_cut)[0]

    ctime_band = ctime[idx_band.min() : idx_band.max()]
    xi_band = xi_det_center[idx_band.min() : idx_band.max()]
    eta_band = eta_det_center[idx_band.min() : idx_band.max()]
    data_band = data[idx_band.min() : idx_band.max()]

    radius_band = np.sqrt(xi_band ** 2 + eta_band ** 2)
    idx_main_in_band = np.where(radius_band < radius_main)[0]
    idx_disk = np.where(radius < radius_cut)[0]
    ctime_disk = ctime[idx_disk]
    xi_disk = xi_det_center[idx_disk]
    eta_disk = eta_det_center[idx_disk]
    data_disk = data[idx_disk]
    radius_disk = np.sqrt(xi_disk ** 2 + eta_disk ** 2)
    idx_main_in_disk = np.where(radius_disk < radius_main)[0]
    idx_main_out_disk = np.where(radius_disk > radius_main)[0]
    xi_main = xi_disk[np.array(idx_main_in_disk)]
    eta_main = eta_disk[idx_main_in_disk]
    data_main = data_disk[idx_main_in_disk]

    p0 = initial_guess
    xieta = np.vstack((xi_band, eta_band))
    space_time_pointing = {
        "xieta": xieta,
        "time": ctime_band,
        "idx_main": idx_main_in_band,
        "fwhm_xi": p0[3],
        "fwhm_eta": p0[4],
        "phi": p0[5],
        "tau": p0[6],
    }

    p0_pointing = p0[:3]
    bounds_pointing = bounds[:, :3]
    try:
        popt_pointing, _ = curve_fit(
            tod_sim_pointing,
            space_time_pointing,
            data_main,
            p0=p0_pointing,
            bounds=bounds_pointing,
        )
    except:
        all_params = np.empty(9)
        all_params[:] = np.nan
        return all_params

    p0[0] = popt_pointing[0]

    # Re-organizing the data after corrrecting for the pointing
    q_t = quat.rotation_xieta(xi0, eta0)
    q_delta = quat.rotation_xieta(popt_pointing[1], popt_pointing[2])  # xi0 and eta0
    xi_t, eta_t, _ = quat.decompose_xieta(q_delta * q_t)  # xi0 and eta0

    q_det = quat.rotation_xieta(xi_t, eta_t)
    xi_det_center, eta_det_center, psi_det_center = quat.decompose_xieta(
        ~q_det * ~q_bore * q_obj
    )
    radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)

    ####################### same for this block #######################

    idx_band = np.where(abs(eta_det_center) < radius_cut)[0]
    ctime_band = ctime[idx_band.min() : idx_band.max()]
    xi_band = xi_det_center[idx_band.min() : idx_band.max()]
    eta_band = eta_det_center[idx_band.min() : idx_band.max()]
    data_band = data[idx_band.min() : idx_band.max()]
    radius_band = np.sqrt(xi_band ** 2 + eta_band ** 2)
    idx_main_in_band = np.where(radius_band < radius_main)[0]
    # cut out the data within an radius
    idx_disk = np.where(radius < radius_cut)[0]
    ctime_disk = ctime[idx_disk]
    xi_disk = xi_det_center[idx_disk]
    eta_disk = eta_det_center[idx_disk]
    data_disk = data[idx_disk]
    radius_disk = np.sqrt(xi_disk ** 2 + eta_disk ** 2)
    idx_main_in_disk = np.where(radius_disk < radius_main)[0]
    idx_main_out_disk = np.where(radius_disk > radius_main)[0]
    data_main = data_disk[np.array(idx_main_in_disk)]

    #####################################################################

    # Fitting fwhm_xi, fwhm_eta, phi
    xieta = np.vstack((xi_band, eta_band))
    space_time_beam = {
        "xieta": xieta,
        "time": ctime_band,
        "idx_main": idx_main_in_band,
        "a": p0[0],
        "xi_off": p0[1],
        "eta_off": p0[2],
        "tau": p0[6],
    }
    p0_beam = p0[3:6]
    bounds_beam = bounds[:, 3:6]

    try:
        popt_beam, _ = curve_fit(
            tod_sim_beam,
            space_time_beam,
            data_main,
            p0=p0_beam,
            bounds=bounds_beam,
        )
    except:
        all_params = np.empty(9)
        all_params[:] = np.nan
        return all_params

    p0[3] = popt_beam[0]  # fwhm_xi
    p0[4] = popt_beam[1]  # fwhm_eta
    p0[5] = popt_beam[2]  # phi

    # Fitting all seven parameters simultaneously
    xieta = np.vstack((xi_band, eta_band))
    space_time = {"xieta": xieta, "time": ctime_band, "idx_main": idx_main_in_band}

    try:
        popt, pcov = curve_fit(
            tod_sim_all,
            space_time,
            data_main,
            p0=p0,
            bounds=bounds,
        )

    except:
        all_params = np.empty(9)
        all_params[:] = np.nan
        return all_params

    q_t = quat.rotation_xieta(xi_t, eta_t)
    q_delta = quat.rotation_xieta(popt[1], popt[2])  # xi0 and eta0
    popt[1], popt[2], _ = quat.decompose_xieta(q_delta * q_t)
    noise = np.nanstd(data[np.where(radius > radius_cut)[0]])
    snr = popt[0] / noise

    all_params = np.array([*popt, tod.dets.vals[det_idx], snr])

    return all_params


def make_maps(path, outdir, ofilename, sso_name, beamsize, n_modes):
    ctx = core.Context(opj(path, "context", "context.yaml"))
    obs_id = ctx.obsdb.get()[0]["obs_id"]
    tod = ctx.get_obs(obs_id)

    ofilename = opj(outdir, ofilename)
    mask_dict = {"shape": "circle", "xyr": (0, 0, 5 * beamsize)}

    source_flags = planets.compute_source_flags(
        tod=tod, mask=mask_dict, center_on=sso_name, res=0.02
    )

    ## Make the maps in degrees
    ## mask ~ 5*beamsize
    ## map size and mask radius could be user inputs
    planets.make_map(
        tod,
        center_on=sso_name,
        scan_coords=True,
        n_modes=n_modes,
        source_flags=source_flags,
        size=10 * beamsize,
        filename=ofilename,
        comps="T",
    )


def run(
    indir,
    outdir,
    f_format,
    tube,
    band,
    wafer,
    sso_name,
    year,
    month,
    day,
    tele,
    highpass,
    cutoff,
    map_make,
    n_modes,
):
    """
    Initiate an MPI environment and fit a simulation in the time domain
    """

    t1 = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ## Standardize output names
    file_name = "calobs_%s_%s_%s_%s_%s_%s-%02d-%02d" % (
        tele,
        tube,
        band,
        wafer,
        sso_name,
        year,
        int(month),
        int(day),
    )
    suffix = "tod/CES-Atacama-" + tele + "-" + sso_name + "-1-0/*.g3"
    path = opj(indir, file_name)
    tod = load_data(path, suffix, f_format)

    ## Get the initial parameters
    init_params = define_fit_params(band, tele)
    ctime, az, el = tod.timestamps, tod.boresight.az, tod.boresight.el

    N = len(tod["dets"].vals)
    rank_batchsize = int(np.floor(N / size))
    quotient, remainder = divmod(N, size)

    # given potential mismatch between number of detectors and processes
    # correct for non-indentical batch sizes, by adding extra values to
    # lower ranks starting with zero rank

    if rank <= (remainder - 1):
        rank_batchsize += 1

    # create iterable vector based on individual process batch size
    batch = np.arange(0, rank_batchsize, dtype=np.int32)
    # modify iterable by shifting array to correct position for process split
    batch += rank * (rank_batchsize)
    # given potential mismatch between number of detectors and processes correct
    # for non-identical batch sizes
    if rank > remainder:
        batch += int(remainder)

    df = pd.DataFrame(
        columns=[
            "amp",
            "xi0",
            "eta0",
            "fwhm_xi",
            "fwhm_eta",
            "phi",
            "tau",
            "det_name",
            "snr",
        ],
        index=batch,
    )

    for det in batch:
        params = fit_params(
            tod, ctime, az, el, band, sso_name, det, highpass, cutoff, init_params
        )
        df.loc[det, :] = np.array(params)
    all_dfs = comm.gather(df, root=0)

    if rank == 0:
        full_df = pd.concat(all_dfs)
        full_df.dropna()
        full_df = full_df.set_index(full_df["det_name"])

        ## Convert to structured array and store file - minor problem with string encoding
        # dt = h5py.special_dtype(vlen=str)
        new_dtypes = {
            "amp": np.float64,
            "xi0": np.float64,
            "eta0": np.float64,
            "fwhm_xi": np.float64,
            "fwhm_eta": np.float64,
            "phi": np.float64,
            "tau": np.float64,
            "det_name": str,
            "snr": np.float64,
        }
        full_df = full_df.astype(new_dtypes)

        # calculating relative and absolute calibration
        amp = full_df.amp.values
        full_df["rel_cal"] = amp / np.mean(amp)

        beam_file = "/mnt/so1/shared/site-pipeline/bcp/%s_%s_beam.h5" % (tele, band)
        sso_obj = sim_sso.OpSimSSO(sso_name, beam_file)

        sso_obj._get_planet_temp(sso_name)
        datestr = "%s-%02d-%02d" % (year, int(month), int(day))
        sso_obj.sso.compute(datestr)
        ttemp = np.interp(float(band[1:]), sso_obj.t_freqs, sso_obj.ttemp)
        beam, _ = sso_obj._get_beam_map(sso_obj.sso.size, ttemp)
        amp_ref = beam(0, 0)[0][0]
        full_df["abs_cal"] = amp / amp_ref

        ### Again need to decide on the naming convention
        obs_id = "%s_%s_%s_%s_%s_%s-%02d-%02d" % (
            tele,
            tube,
            band,
            wafer,
            sso_name,
            year,
            int(month),
            int(day),
        )
        h_name = "fitting_table_" + obs_id

        # hf = h5py.File(opj(outdir, h_name), 'w')
        # hf.create_dataset('fitted_params_'+obs_id, data=summary_data)
        # hf.close()

        ## Assert tables dependency is properly satisfied
        out_folder = opj(outdir, tele, tube, wafer)
        if not os.path.exists(out_folder):
            # Create a new directory because it does not exist
            os.makedirs(out_folder)
        full_df.to_hdf(opj(out_folder, h_name + ".h5"), key="parameter_table", mode="w")

        t2 = time.time()
        print("Time to run fittings for a full wafer is {} seconds".format(t2 - t1))

        ## check why the mapmaker was working only with context and fix this issue
        if map_make and f_format == "context" and n_modes is not None:
            beamsize = np.degrees(init_params[2]) / 2
            make_maps(path, outdir, obs_id, sso_name, beamsize, n_modes)


def main():
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--indir",
        action="store",
        dest="indir",
        help="The location of the simulated files.",
    )
    parser.add_argument(
        "--outdir",
        action="store",
        dest="outdir",
        help="The location for the .h5 output files to be stored.",
    )
    parser.add_argument(
        "--f_format",
        action="store",
        dest="f_format",
        help="The format of the simulations [g3, h5, context].",
    )
    parser.add_argument(
        "--tele", action="store", dest="tele", help="The telescope name [LAT, SAT]."
    )
    parser.add_argument(
        "--tube",
        action="store",
        dest="tube",
        help="The tube name [c1, i6, o6, ...] for LAT, [SAT1, SAT2, ...] for SAT.",
    )
    parser.add_argument(
        "--band",
        action="store",
        dest="band",
        help="Frequency band [30,40,90,150,230,290].",
    )
    parser.add_argument(
        "--wafer",
        action="store",
        dest="wafer",
        help="The wafer name [wafers number increases from the center and \
    anti-clockwise] across the focalplane.",
    )
    parser.add_argument(
        "--sso_name",
        action="store",
        dest="sso_name",
        help="Calibration source (astrophysical only for the time being).",
    )
    parser.add_argument(
        "--year",
        action="store",
        dest="year",
        help="The year of the calibration observation.",
    )
    parser.add_argument(
        "--month",
        action="store",
        dest="month",
        help="The month of the calibration observation.",
    )
    parser.add_argument(
        "--day",
        action="store",
        dest="day",
        help="The day of the calibration observation",
    )
    ## The telescope can be defined from the wafer name
    ## No need for extra parser argument --fix later

    parser.add_argument(
        "--highpass",
        action="store_true",
        dest="highpass",
        help="If True, use highpass butterworth filters (especially useful for SATs).",
    )
    parser.add_argument(
        "--cutoff",
        action="store",
        dest="cutoff",
        default=None,
        help="The cutoff frequency to be used in the filtering.",
        type=float,
    )
    # parser.add_argument('--sample_rate', action='store', dest='sample_rate', default=None,
    # help='Sampling rate of the observation.', type=int)
    parser.add_argument(
        "--map_make",
        action="store_true",
        dest="map_make",
        help="Make planet maps of the simulations fitted.",
    )
    parser.add_argument(
        "--n_modes",
        action="store",
        dest="n_modes",
        default=None,
        help="Number of PCA modes to be removed in the planet mapmaker.",
        type=int,
    )

    args = parser.parse_args()

    run(
        args.indir,
        args.outdir,
        args.f_format,
        args.tube,
        args.band,
        args.wafer,
        args.sso_name,
        args.year,
        args.month,
        args.day,
        args.tele,
        args.highpass,
        args.cutoff,
        args.map_make,
        args.n_modes,
    )


if __name__ == "__main__":
    main()

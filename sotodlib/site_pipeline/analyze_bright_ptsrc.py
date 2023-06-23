#!/usr/bin/env python3
import so3g
from so3g.proj import coords, quat
import sotodlib.coords.planets as planets
from sotodlib import core
from sotodlib.toast.ops import sim_sso
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset, read_dataset
from astropy import units as u

import numpy as np
import scipy.signal
import time, ephem, os
from scipy.optimize import curve_fit
from datetime import datetime
import argparse as ap

from sotodlib.site_pipeline import util


opj = os.path.join
INITIAL_PARA_FILE = "/mnt/so1/shared/site-pipeline/bcp/initial_parameters.h5"


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
    
    init_rs = read_dataset(INITIAL_PARA_FILE, tele[:3])
    
    idx_t = np.where(init_rs['frequency-band name'] == band.encode(encoding='UTF-8'))[0][0]
    beamsize = init_rs[idx_t]["beam size"] # in arcmin
    amp = init_rs[idx_t]["detector response"]
    beamsize = np.radians(beamsize / 60)
    offset_bound = 2 * beamsize

    ## allow for max 10% bias
    ## 10% is large but allowing for a large bias helps tracking down
    fwhm_bound_frac = 0.1
    fwhm_bound = fwhm_bound_frac * beamsize
    fwhm_min = beamsize - fwhm_bound
    fwhm_max = beamsize + fwhm_bound
    
    tau_init = 3e-3 # seconds
    initial_guess = [amp, 0, 0, beamsize, beamsize, np.pi / 6, tau_init]

    bounds_min = (0, -offset_bound, -offset_bound, fwhm_min, fwhm_min, 0.0, 1e-3)
    bounds_max = (2 * amp, offset_bound, offset_bound, fwhm_max, fwhm_max, np.pi, 10e-3)
    bounds = np.array((bounds_min, bounds_max,))

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
        readout_id, data,
        ctime, az, el, band, sso_name, highpass, cutoff, init_params
):
    """Function that fits individual time-streams and returns the parameters

    Args
    ----
    readout_id: str
        Detector id string.
    data: array of float
        Vector of time-ordered data for this detector.
    ctime: array of float
        timestamps
    az: array of float
        azimuth of boresight
    el: array of float
        elevation of boresight
    obs_id: obs_id
        obs_id for loading the tod
    band: string
        The frequency-band
    sso_name: string
        Name of the celestial source
    highpass: bool
        If True use a butterworth filter on the data
    """
    initial_guess, bounds, radius_main = init_params
    radius_cut = 2 * radius_main

    data -= np.mean(data)
    sample_rate = 1.0 / np.mean(np.diff(ctime))

    if highpass and (cutoff is not None):
        data = highpass_filter(data, cutoff, sample_rate)
        data[data < 0] = 0
    coord_transforms = get_xieta_src_centered(ctime, az, el, data, sso_name)

    if coord_transforms is None:
        return np.array([readout_id, *np.full(8, np.nan)])

    total_coords, peak_coords, q_det, q_bore, q_obj = coord_transforms

    xi_det_center, eta_det_center = total_coords
    xi0, eta0 = peak_coords
    radius = np.sqrt(xi_det_center ** 2 + eta_det_center ** 2)

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

    # Thing to return on error ...
    failed_params = np.array([readout_id, *np.full(8, np.nan)])

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
        return failed_params

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
        return failed_params

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
        return failed_params

    q_t = quat.rotation_xieta(xi_t, eta_t)
    q_delta = quat.rotation_xieta(popt[1], popt[2])  # xi0 and eta0
    popt[1], popt[2], _ = quat.decompose_xieta(q_delta * q_t)
    noise = np.nanstd(data[np.where(radius > radius_cut)[0]])
    snr = popt[0] / noise

    all_params = np.array([readout_id, *popt, snr])

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


def main(
    ctx_file,
    ctx_idx,
    outdir,
    band,
    sso_name,
    tele,
    highpass,
    cutoff,
    map_make,
    n_modes,
    test_mode=False,
    logger=None,
):
    """
    Initiate an MPI environment and fit a simulation in the time domain
    """
    # Two local imports, to avoid docs depenency.
    from mpi4py import MPI
    import pandas as pd

    if logger is None:
        logger = util.init_logger(__name__)
    t1 = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ntask = comm.Get_size()

    ctx = core.Context(ctx_file)
    obs = ctx.obsdb.query()

    obs_id = obs[ctx_idx]['obs_id']
    logger.warning(f'Loading {obs_id} in task {rank}')
    tod = ctx.get_obs(obs_id, no_signal=True)
    rd_ids = tod.dets.vals

    ## Get the initial parameters
    init_params = define_fit_params(band, tele)
    ctime, az, el = tod.timestamps, tod.boresight.az, tod.boresight.el

    nrd = len(rd_ids)
    nsample_task = nrd // ntask + 1
    rd_idx_rng = np.arange(rank * nsample_task,
                           min((rank + 1) * nsample_task, nrd))
    
    df = pd.DataFrame(
        columns=[
            "dets:readout_id",
            "amp",
            "xi0",
            "eta0",
            "fwhm_xi",
            "fwhm_eta",
            "phi",
            "tau",
            "snr",
        ], index=rd_idx_rng,)

    # Load signal for only my dets.
    tod = ctx.get_obs(obs_id, dets=np.array(rd_ids)[rd_idx_rng])

    count = 0
    for _i, rd_idx in enumerate(rd_idx_rng):
        rd_id = rd_ids[rd_idx]
        params = fit_params(rd_id, tod.signal[_i],
                            ctime, az, el, 
                            band, sso_name, 
                            highpass, cutoff, init_params)
        snr = float(params[-1])
        logger.info(f'Solved {rd_idx:<5d} "{rd_id}" with S/N={snr:.2f}')
        df.loc[rd_idx, :] = np.array(params)
        count += 1
        if test_mode and count >= 2:
            break
    all_dfs = comm.gather(df, root=0)

    if rank == 0:
        full_df = pd.concat(all_dfs)
        full_df.dropna()
        full_df = full_df.set_index(full_df["dets:readout_id"])

        new_dtypes = {
            "dets:readout_id": str,
            "amp": np.float64,
            "xi0": np.float64,
            "eta0": np.float64,
            "fwhm_xi": np.float64,
            "fwhm_eta": np.float64,
            "phi": np.float64,
            "tau": np.float64,
            "snr": np.float64,
        }
        full_df = full_df.astype(new_dtypes)

        # calculating relative and absolute calibration
        amp = full_df.amp.values
        full_df["rel_cal"] = amp / np.mean(amp)

        beam_file = "/mnt/so1/shared/site-pipeline/bcp/%s_%s_beam.h5" % (tele, band)
        sso_obj = sim_sso.SimSSO(beam_file=beam_file, sso_name=sso_name)
        freq_arr_GHz, temp_arr = sso_obj._get_sso_temperature(sso_name)
        dt_obs = datetime.fromtimestamp(np.average(tod.timestamps))
        sso_ephem = getattr(ephem, sso_name)()
        sso_ephem.compute(dt_obs.strftime('%Y-%m-%d'))
        ttemp = np.interp(float(band[1:])*u.GHz, freq_arr_GHz, temp_arr)
        beam, _ = sso_obj._get_beam_map(None, sso_ephem.size*u.arcsec, ttemp)
        amp_ref = beam(0, 0)[0][0]
        full_df["abs_cal"] = amp / amp_ref

        ## Assert tables dependency is properly satisfied
        out_folder = opj(outdir, tele)
        if not os.path.exists(out_folder):
            # Create a new directory because it does not exist
            os.makedirs(out_folder)
        result_arr = full_df.to_records(index=False)
        result_rs = metadata.ResultSet.from_friend(result_arr)
        print(opj(out_folder,'cal_obs_%s.h5'%tele))
        write_dataset(result_rs, opj(out_folder,'cal_obs_%s.h5'%tele), obs_id, overwrite=True)

        t2 = time.time()
        print("Time to run fittings for %s is %.2f seconds."%(obs_id , t2 - t1))

        ## check why the mapmaker was working only with context and fix this issue
        if map_make and n_modes is not None:
            beamsize = np.degrees(init_params[2]) / 2
            make_maps(path, outdir, obs_id, sso_name, beamsize, n_modes)


def get_parser(parser=None):
    if parser is None:
        parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--ctx_file",
        action="store",
        dest="ctx_file",
        help="The location of the context file.",
    )
    
    parser.add_argument(
        "--ctx_idx",
        action="store",
        dest="ctx_idx",
        help="Index of the observation in the context file.",
        type=int,
    )
    
    parser.add_argument(
        "--outdir",
        action="store",
        dest="outdir",
        help="The location for the .h5 output files to be stored.",
    )

    parser.add_argument(
        "--tele", action="store", dest="tele", help="The telescope name [LAT, SAT]."
    )
    parser.add_argument(
        "--band",
        action="store",
        dest="band",
        help="Frequency band [30,40,90,150,230,290].",
    )
    
    parser.add_argument(
        "--sso_name",
        action="store",
        dest="sso_name",
        help="Calibration source (astrophysical only for the time being).",
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run analysis on a subset of detectors, to quickly check for problems."
    )

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

    return parser


if __name__ == "__main__":
    util.main_launcher(main, get_parser)

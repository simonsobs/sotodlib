# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import warnings

import astropy.units as u
import numpy as np

try:
    import pandas as pd
    from detmap.data_io.solution_select import AvailableSolutions

    available_solutions = AvailableSolutions()
    detmap_available = True
except ImportError:
    detmap_available = False


def wafer_chis(dets, pos, freq, is_north, mux_band, bond_pad, alpha, collision):
    """Given detector properties on one wafer, compute the crosstalk.

    Args:
        dets (list):  The list of detector names
        pos (list):  The list of tuple x, y postitions
        is_north (array):  Array of 'N' / 'S' values
        mux_band (array):  The mux band for each det
        bond_pad (array):  The bond pad for each det
        alpha (float):  The crosstalk prefactor
        collision (float):  Collistion threshold

    Returns:
        (dict):  Wafer chi values.

    """
    ndet = len(dets)
    wchis = {}
    for idet1, det1 in enumerate(dets):
        x1, y1 = pos[idet1]
        freq1 = freq[idet1]
        if freq1 == 0:
            # Detector was not found in the mapping and
            # there is no resonator frequency
            continue
        is_north1 = is_north[idet1]
        mux_band1 = mux_band[idet1]
        bond_pad1 = bond_pad[idet1]
        for idet2 in range(idet1 + 1, ndet):
            freq2 = freq[idet2]
            if freq2 == 0:
                # Detector was not found in the mapping and
                # there is no resonator frequency
                continue
            det2 = dets[idet2]
            is_north2 = is_north[idet2]
            mux_band2 = mux_band[idet2]
            bond_pad2 = bond_pad[idet2]
            # Short-circuit chi-calculation if the detectors
            # cannot cross-talk
            if is_north1 != is_north2:
                continue
            if mux_band1 != mux_band2:
                continue
            if np.abs(bond_pad1 - bond_pad2) != 4:
                continue
            # Check that this truly is the nearly frequency neighbor
            neighbors = np.argwhere(
                (is_north == is_north1) & (mux_band == mux_band1)
            ).flatten()
            freq_neighbors = np.argsort(np.abs(freq[neighbors] - freq1)).flatten()
            if idet2 not in neighbors[freq_neighbors[1:3]]:
                # freq_neighbors[0] is idet1
                # frequency neighbors on either side
                continue
            x2, y2 = pos[idet2]
            # Translate frequencies to chi
            df = freq1 - freq2
            avg_f = (freq1 + freq2) / 2
            chi = alpha * np.power(avg_f, 4) * np.power(df, -2) * 1e12
            # Check for anomalously high chi in absense of resonator collision
            if chi > 1 and np.abs(df) > collision:
                msg = "Anomalously high chi at"
                msg += f" {det1}@({x1}, {y1}), {det2}@({x2}, {y2})"
                msg += f" df: {df}"
                msg += f" : {chi}"
                raise RuntimeError(msg)
            # Check for resonator collision, if so set flag to zero TOD
            if np.abs(df) < collision:
                chi = np.nan
            wchis[(det1, det2)] = chi
            wchis[(det2, det1)] = chi
    return wchis


def pos_to_detmap_ind(pos, bp, pol, mapping, tol=1.0, verbose=False):
    """Match position to a detector index on the synthetic UFM.

    Args:
        pos (tuple): x, y coordinates of first detector (mm) relative to UFM center
        bp (int): detector bandpass
        pol (str): detector polarization, 'A' or 'B'
        mapping (object): pandas object from DetMap CSV of UFM characteristics
        tol (float):  Maximum allowed distance between provided and
            matched detector positions.

    Returns:
        (int):  The index.

    """

    # Compute Pythagorean distances to all detectors in the mapping
    x, y = pos
    dist = np.sqrt(np.square(mapping["det_x"] - x) + np.square(mapping["det_y"] - y))

    # Send all out of bandpass and polarization distances to infinity
    dist[np.logical_or(mapping["bandpass"] != str(bp), mapping["pol"] != pol)] = np.inf

    # Match detector to position
    ind = dist.argmin()

    # Check to make sure the match is credible
    dist_min = dist.iloc[ind]
    if dist_min > tol:
        if verbose:
            msg = f"Failed to match ({x} mm, {y} mm) to a detector in "
            msg += f"mapping.  Minimum distance is {dist_min} mm"
            warnings.warn(msg)
        return None

    if verbose:
        print(f"Minimum distance is at {ind} : {dist_min} mm", flush=True)

    return ind


def chi_simulated_obs(focalplane, dets, alpha, tol, collision):
    """Compute the crosstalk magnitude for a synthetic focalplane.

    This takes a hardcoded mapping from synthetic wafer slot to UFM name
    that is compatible with the conventions used in DetMap files.

    It uses DetMap to obtain the various parameters needed for the calcuation.

    Args:
        focalplane (SOFocalplane):  Focalplane object
        dets (iterable):  detector names to consider
        alpha:  crosstalk prefactor (Hz^-2), from John Groh (via FastHenry),
            valid for nearest physical neighbors
        tol (float):  Maximum allowed distance between provided and
            matched detector positions.
        collision (float):  The threshold to consider a resonator collision.

    Returns:
        (dict):  The chi for every detector pair.

    """
    # SATp1 MF wafers  = w25-w31
    # SATp2 UHF wafers = w06-w12
    # SATp3 MF wafers  = w32-w38
    # SATp4 LF wafers  = w42-w48
    # LAT LF wafers    = w39-w41
    # LAT MF wafers    = w13-w24
    # LAT UHF wafers   = w00-w05

    # Mapping between wafer slots and array names
    wafer_to_array = {
        "dummy00": "Cv4",
        "dummy01": "Cv5",
        "dummy04": "Mv9",
        "dummy05": "Sv5",
        "w00": "Uv31",  # LAT OTc1, only one LAT UHF wafer in DetMap
        "w01": "Uv31",  # LAT OTc1, only one LAT UHF wafer in DetMap
        "w02": "Uv31",  # LAT OTc1, only one LAT UHF wafer in DetMap
        "w03": "Uv31",  # LAT OTi5, only one LAT UHF wafer in DetMap
        "w04": "Uv31",  # LAT OTi5, only one LAT UHF wafer in DetMap
        "w05": "Uv31",  # LAT OTi5, only one LAT UHF wafer in DetMap
        "w06": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w07": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w08": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w09": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w10": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w11": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w12": "Uv8",   # SATp2, only one SAT UHF wafer in DetMap
        "w13": "Mv25",  # LAT OTi6
        "w14": "Mv26",  # LAT OTi6
        "w15": "Mv11",  # LAT OTi6
        "w16": "Mv21",  # LAT OTi1
        "w17": "Mv24",  # LAT OTi1
        "w18": "Mv28",  # LAT OTi1
        "w19": "Mv13",  # LAT OTi3
        "w20": "Mv20",  # LAT OTi3
        "w21": "Mv28",  # LAT OTi3, Mv34 missing from DetMap
        "w22": "Mv14",  # LAT OTi4
        "w23": "Mv32",  # LAT OTi4
        "w24": "Mv11",  # LAT OTi4, Mv49 missing from DetMap
        "w25": "Mv19",  # SATp1
        "w26": "Mv27",  # SATp1, Mv48 missing from DetMap
        "w27": "Mv12",  # SATp1, Mv50r2 missing from DetMap
        "w28": "Mv22",  # SATp1
        "w29": "Mv18",  # SATp1
        "w30": "Mv33",  # SATp1, Mv52r1 missing from DetMap
        "w31": "Mv17",  # SATp1, Mv51 missing from DetMap
        "w32": "Mv19",  # SATp3, Mv5 mising from DetMap
        "w33": "Mv27",  # SATp3
        "w34": "Mv22",  # SATp3, Mv35 missing from DetMap
        "w35": "Mv12",  # SATp3
        "w36": "Mv23",  # SATp3
        "w37": "Mv33",  # SATp3
        "w38": "Mv17",  # SATp3
    }

    # Get the bandpass, polarization and position for every detector
    bandpasses = []
    pols = []
    positions = []
    wafers = []

    for det in dets:
        wafers.append(focalplane[det]["wafer_slot"])
        bandpasses.append(int(focalplane[det]["band"][-3:]))
        pols.append(focalplane[det]["pol"])
        positions.append(
            (
                focalplane[det]["wafer_x"].to_value(u.mm),
                focalplane[det]["wafer_y"].to_value(u.mm),
            )
        )

    # Load the appropriate mappings
    mappings = {}
    wafer_set = set(wafers)
    for wafer in wafer_set:
        if wafer not in wafer_to_array:
            msg = f"Could not map {wafer} to an array in DetMap. "
            msg += f"Mapped wafer slots are {sorted(wafer_to_array.keys())}"
            raise RuntimeError(msg)
        array = wafer_to_array[wafer]
        try:
            datafile = available_solutions.get_solution_file(array)
        except ValueError as e:
            msg = f"{array} does not appear to be a valid DetMap name:\n{e}\n "
            msg += "Perhaps wafer_to_array is out of date?"
        mappings[wafer] = pd.read_csv(datafile)

    # chi for every detector pair
    chis = {}
    for wafer in wafer_set:
        # Collect detector and positions that match wafer
        m = mappings[wafer]
        if len(m) == 0:
            msg = "No match in DetMap."
            raise RuntimeError(msg)
        detector_subset = []
        position_subset = []
        freq_subset = []
        is_north_subset = []
        mux_band_subset = []
        bond_pad_subset = []
        for d, w, b, p, pos in zip(dets, wafers, bandpasses, pols, positions):
            if w == wafer:
                detector_subset.append(d)
                position_subset.append(pos)
                ind = pos_to_detmap_ind(pos, b, p, m, tol=tol)
                if ind is not None:
                    # Get resonator frequency
                    freq_subset.append(m.iloc[ind]["freq_mhz"])
                    # Get other helpful variables
                    is_north_subset.append(m.iloc[ind]["is_north"])
                    mux_band_subset.append(m.iloc[ind]["mux_band"])
                    bond_pad_subset.append(m.iloc[ind]["bond_pad"])
                else:
                    # This position is not in the mapping.  For now, we
                    # keep the detector and disable crosstalk for it
                    freq_subset.append(0)
                    is_north_subset.append(None)
                    mux_band_subset.append(None)
                    bond_pad_subset.append(None)

        # Convert to numpy objects
        freq_subset = np.array(freq_subset)
        is_north_subset = np.array(is_north_subset)
        mux_band_subset = np.array(mux_band_subset)
        bond_pad_subset = np.array(bond_pad_subset)

        chis.update(
            wafer_chis(
                detector_subset,
                position_subset,
                freq_subset,
                is_north_subset,
                mux_band_subset,
                bond_pad_subset,
                alpha,
                collision,
            )
        )

    return chis


def chi_real_obs(focalplane, dets, alpha, collision):
    """Compute the crosstalk magnitude for a real focalplane.

    This uses metadata found in real observations to compute the crosstalk.
    One simplification is that real data is always organized with a single wafer
    per observation.

    Args:
        focalplane (SOFocalplane):  Focalplane object
        dets (iterable):  detector names to consider
        alpha:  crosstalk prefactor (Hz^-2), from John Groh (via FastHenry),
            valid for nearest physical neighbors
        collision (float):  The threshold to consider a resonator collision.

    Returns:
        (dict):  The chi for every detector pair.

    """
    # Get the bandpass, polarization and position for every detector
    pols = []
    positions = []
    wafers = []
    freq = []
    is_north = []
    mux_band = []
    bond_pad = []

    for det in dets:
        wafers.append(focalplane[det]["det_info:wafer_slot"])
        pols.append(focalplane[det]["det_info:wafer:pol"])
        positions.append(
            (
                focalplane[det]["det_info:wafer:x"],
                focalplane[det]["det_info:wafer:y"],
            )
        )
        freq.append(focalplane[det]["det_info:smurf:frequency"])
        # Get other helpful variables
        is_north.append(focalplane[det]["det_info:wafer:coax"])
        mux_band.append(focalplane[det]["det_info:wafer:mux_band"])
        bond_pad.append(focalplane[det]["det_info:wafer:bond_pad"])

    freq = np.array(freq)
    is_north = np.array(is_north)
    mux_band = np.array(mux_band)
    bond_pad = np.array(bond_pad)

    return wafer_chis(
        dets,
        positions,
        freq,
        is_north,
        mux_band,
        bond_pad,
        alpha,
        collision,
    )


def pos_to_chi(focalplane, dets, alpha=9.64e-30, tol=1.0, collision=1.0):
    """Calculate the crosstalk magnitude for all detector pairs.

    This uses either real metadata or DetMap solutions to get the detector
    positions and properties.

    Args:
        focalplane (SOFocalplane):  Focalplane object
        dets (iterable):  detector names to consider
        alpha:  crosstalk prefactor (Hz^-2), from John Groh (via FastHenry),
            valid for nearest physical neighbors
        tol (float):  Maximum allowed distance between provided and
            matched detector positions (only in DetMap case).
        collision (float):  The threshold to consider a resonator collision.

    Returns:
        (dict):  The chi for every detector pair.

    """
    # Make sure we have the necessary focalplane keys or the DetMap package
    # available.
    required_cols = [
        "det_info:wafer_slot",
        "det_info:wafer:mux_band",
        "det_info:wafer:bond_pad",
        "det_info:wafer:coax",
        "det_info:wafer:pol",
        "det_info:wafer:x",
        "det_info:wafer:y",
        "det_info:smurf:frequency",
    ]
    have_fp_cols = True
    for colname in required_cols:
        if colname not in focalplane.detector_data.colnames:
            have_fp_cols = False
            break
    if not have_fp_cols and not detmap_available:
        raise RuntimeError("Cannot evaluate chi: no DetMap or required focalplane info")

    if have_fp_cols:
        chis = chi_real_obs(focalplane, dets, alpha, collision)
    else:
        chis = chi_simulated_obs(focalplane, dets, alpha, tol, collision)

    return chis

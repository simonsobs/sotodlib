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
except:
    detmap_available = False

mappings = {}

# SAT1 MF wafers  = w25-w31
# SAT2 MF wafers  = w32-w38
# SAT3 HF wafers  = w06-w12
# SAT4 LF wafers  = w42-w48
# LAT LF wafers  = w39-w41
# LAT MF wafers  = w13-w24
# LAT HF wafers  = w00-w05

# Arbitrary mapping between wafer slots and array names
wafer_to_array = {
    "dummy00" : "Cv4",  # LAT
    "dummy01" : "Cv5",
    "w13" : "Mv6",   # LAT  1/12
    "w14" : "Mv7",   # LAT  2/12
    "dummy04" : "Mv9",
    "w15" : "Mv11",  # LAT  3/12
    "w16" : "Mv12",  # LAT  4/12
    "w25" : "Mv13",  # SAT1 1/7
    "w26" : "Mv14",  # SAT1 2/7
    # "Mv15",  # LAT (missing)
    "w17" : "Mv17",  # LAT  5/12
    "w27" : "Mv18",  # SAT1 3/7
    "w28" : "Mv19",  # SAT1 4/7
    "w29" : "Mv22",  # SAT1 5/7
    "w30" : "Mv23",  # SAT1 6/7
    "w31" : "Mv24",  # SAT1 7/7
    "w18" : "Mv25",  # LAT  6/12
    "w19" : "Mv26",  # LAT  7/12
    "w20" : "Mv27",  # LAT  8/12
    "w21" : "Mv28",  # LAT  9/12
    "w22" : "Mv29",  # LAT 10/12
    "w23" : "Mv32",  # LAT 11/12
    "w24" : "Mv33",  # LAT 12/12
    "dummy05" : "Sv5",
    "w06" : "Uv31", # Only one SAT HF wafer in DetMap
    "w07" : "Uv31", # Only one SAT HF wafer in DetMap
    "w08" : "Uv31", # Only one SAT HF wafer in DetMap
    "w09" : "Uv31", # Only one SAT HF wafer in DetMap
    "w10" : "Uv31", # Only one SAT HF wafer in DetMap
    "w11" : "Uv31", # Only one SAT HF wafer in DetMap
    "w12" : "Uv31", # Only one SAT HF wafer in DetMap
    "w00" : "Uv8",  # Only one LAT HF wafer in DetMap
    "w01" : "Uv8",  # Only one LAT HF wafer in DetMap
    "w02" : "Uv8",  # Only one LAT HF wafer in DetMap
    "w03" : "Uv8",  # Only one LAT HF wafer in DetMap
    "w04" : "Uv8",  # Only one LAT HF wafer in DetMap
    "w05" : "Uv8",  # Only one LAT HF wafer in DetMap
}



def pos_to_ind(pos, bp, pol, mapping, tol=1.0, verbose=False):
    """
        Match position to a detector index on the UFM

        pos: x, y coordinates (tuple) of first detector (mm) relative to UFM center
        bp: detector bandpass (int)
        pol: detector polarization (string), 'A' or 'B'
        mapping: pandas object from DetMap CSV of UFM characteristics
        tol (float):  Maximum allowed distance between provided and
            matched detector positions.
    """

    # Compute Pythagorean distances to all detectors in the mapping
    x, y = pos
    dist = np.sqrt(np.square(mapping["det_x"] - x) + np.square(mapping["det_y"] - y))

    # Send all out of bandpass and polarization distances to infinity
    dist[np.logical_or(mapping['bandpass'] != str(bp), mapping['pol'] != pol)] = np.inf

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


def pos_to_chi(focalplane, dets, alpha=9.64e-30, tol=1.0):
    """
        Calculate magnitude of crosstalk for all detector pairs given
        their physical positions and mapping

        focalplane (SOFocalplane):  Focalplane object
        dets (iterable):  detector names to consider
        alpha:  crosstalk prefactor (Hz^-2), from John Groh (via FastHenry),
            valid for nearest physical neighbors
        tol (float):  Maximum allowed distance between provided and
            matched detector positions.
    """
    if not detmap_available:
        raise RuntimeError("Cannot evaluate chi -- no DetMap available")

    # Get the bandpass, polarization and position for every detector
    bandpasses = []
    pols = []
    positions = []
    wafers = []
    for det in dets:
        wafers.append(focalplane[det]["wafer_slot"])
        bandpasses.append(int(focalplane[det]["band"][-3:]))
        pols.append(focalplane[det]["pol"])
        positions.append((
            focalplane[det]["wafer_x"].to_value(u.mm),
            focalplane[det]["wafer_y"].to_value(u.mm)
        ))

    # Load the appropriate mappings
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
            msg += f"Perhaps wafer_to_array is out of date?"
        mappings[wafer] = pd.read_csv(datafile)

    # chi for every detector pair
    chis = {}
    for wafer in wafer_set:
        # Collect detector and positions that match wafer
        m = mappings[wafer]
        if len(m) == 0:
            msg = f"No match in DetMap."
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
                ind = pos_to_ind(pos, b, p, m, tol=tol)
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
        # Compute chi for all detector pairs in this subset
        ndet = len(detector_subset)
        for idet1, det1 in enumerate(detector_subset):
            x1, y1 = position_subset[idet1]
            freq1 = freq_subset[idet1]
            if freq1 == 0:
                # Detector was not found in the mapping and
                # there is no resonator frequency
                continue
            is_north1 = is_north_subset[idet1]
            mux_band1 = mux_band_subset[idet1]
            bond_pad1 = bond_pad_subset[idet1]
            for idet2 in range(idet1 + 1, ndet):
                freq2 = freq_subset[idet2]
                if freq2 == 0:
                    # Detector was not found in the mapping and
                    # there is no resonator frequency
                    continue
                det2 = detector_subset[idet2]
                is_north2 = is_north_subset[idet2]
                mux_band2 = mux_band_subset[idet2]
                bond_pad2 = bond_pad_subset[idet2]
                # Short-circuit chi-calculation if the detectors
                # cannot cross-talk
                if is_north1 != is_north2:
                    continue
                if mux_band1 != mux_band2:
                    continue
                if np.abs(bond_pad1 - bond_pad2) != 4:
                    continue
                x2, y2 = position_subset[idet2]
                # Translate frequencies to chi
                df = freq1 - freq2
                avg_f = (freq1 + freq2) / 2
                chi = alpha * np.power(avg_f, 4) * np.power(df, -2) * 1e12
                if chi > 1:
                    msg = f"Anomalously high chi at"
                    msg += f" {det1}@({x1}, {y1}), {det2}@({x2}, {y2})"
                    msg += f" : {chi}"
                    raise RuntimeError(msg)
                chis[(det1, det2)] = chi
                chis[(det2, det1)] = chi

    return chis

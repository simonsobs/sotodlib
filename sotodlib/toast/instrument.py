# Copyright (c) 2019-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import sys
from datetime import datetime, timezone
import re

import numpy as np

import astropy.units as u
from astropy.table import QTable
from toast.instrument import Focalplane, GroundSite, Telescope
from toast.utils import Logger, name_UID

from ..core.hardware import Hardware, build_readout_id, parse_readout_id
from ..sim_hardware import sim_nominal
from .sim_focalplane import sim_telescope_detectors


FOCALPLANE_RADII = {
    "LAT": u.Quantity(3.6, u.deg),
    "SAT1": u.Quantity(17.8, u.deg),
    "SAT2": u.Quantity(17.8, u.deg),
    "SAT3": u.Quantity(17.8, u.deg),
    "SAT4": u.Quantity(17.2, u.deg),
}


def get_telescope(telescope, wafer_slots, tube_slots):
    """Determine which telescope matches the detector selections"""
    if telescope is not None:
        return telescope
    # User did not set telescope so we infer it from the
    # tube and wafer slots
    hwexample = sim_nominal()
    if wafer_slots is not None:
        wafer_slots = wafer_slots.split(",")
        wafer_map = hwexample.wafer_map()
        tube_slots = [wafer_map["tube_slots"][ws] for ws in wafer_slots]
    elif tube_slots is not None:
        tube_slots = tube_slots.split(",")
    else:
        raise RuntimeError("Must set telescope, wafer_slots or tube_slots.")
    for tube_slot in tube_slots:
        for telescope_name, telescope_data in hwexample.data["telescopes"].items():
            if tube_slot in telescope_data["tube_slots"]:
                if telescope is None:
                    telescope = telescope_name
                elif telescope != telescope_name:
                    raise RuntimeError(
                        f"Tubes '{tube_slots}' span more than one telescope"
                    )
        if telescope is None:
            raise RuntimeError(
                f"Failed to match tube_slot = '{tube_slot}' with a telescope"
            )
    return telescope


def get_tele_wafer_band_name(tele, tube, wafer, band):
    """Return a simplified name for one band of a wafer.

    There are multiple places where we need to construct a unique string
    composed of the telescope name, optics tube, wafer and band within a
    wafer.  The formal band name includes the telescope, which results
    in annoying duplication.

    Args:
        tele (str):  Name of the telescope
        tube (str):  The optics tube slot
        wafer (str):  The wafer slot name
        band (str):  The name of the band in the hardware dictionary

    Returns:
        (str):  The name.

    """
    bf_pat = re.compile(r"(.*_)")
    band_freq = bf_pat.sub("", band)
    return f"{tele}_{tube}_{wafer}_{band_freq}"


class SOSite(GroundSite):
    def __init__(
        self,
        name="ATACAMA",
        lat=-22.958064 * u.degree,
        lon=-67.786222 * u.degree,
        alt=5200 * u.meter,
        **kwargs,
    ):
        super().__init__(
            name,
            lat,
            lon,
            alt,
            **kwargs,
        )


class SOFocalplane(Focalplane):
    """SO Focalplane class.

    This can be constructed in several ways:
        - From a file while applying selection criteria.
        - From an existing (pre-selected) Hardware instance.
        - From a nominal sim on the fly with selections.

    Note that to support serialization, the simulated hardware dictionary elements
    have values in standard units.  When constructing our focalplane table we restore
    those units and build Quantities.

    Args:
        hw (Hardware):  If specified, construct from a Hardware object in memory.
        hwfile (str):  If specified, load this hardware model from disk.
        det_info_file (str):  If simulating a Hardware model, optionally specify
            a det_info format file to load
        det_info_version (str):  The version of the det_info file format.
        telescope (str):  If not None, select only detectors from this telescope.
        sample_rate (Quantity):  Use this sample rate for all detectors.
        bands (str):  Comma separated string of bands to use.
        wafer_slots (str):  Comma separated string of wafers to use.
        tube_slots (str):  Comma separated string of tubes to use.
        thinfp (int):  The factor by which to reduce the number of detectors.
        creation_time (float):  Optional timestamp to use when building readout_id.
        comm (MPI.Comm):  Optional MPI communicator.

    """

    def __init__(
        self,
        hw=None,
        hwfile=None,
        det_info_file=None,
        det_info_version=None,
        telescope=None,
        sample_rate=u.Quantity(10.0, u.Hz),
        bands=None,
        wafer_slots=None,
        tube_slots=None,
        thinfp=None,
        creation_time=None,
        comm=None,
    ):
        log = Logger.get()
        meta = dict()
        meta["telescope"] = get_telescope(telescope, wafer_slots, tube_slots)
        field_of_view = 2 * FOCALPLANE_RADII[meta["telescope"]]

        if creation_time is None:
            # Use the current time
            creation_time = datetime.now(tz=timezone.utc).timestamp()

        if hw is None:
            if comm is None or comm.rank == 0:
                if hwfile is not None:
                    log.debug(f"Loading hardware configuration from {hwfile}...")
                    hw = Hardware(hwfile)
                elif meta["telescope"] in ["LAT", "SAT1", "SAT2", "SAT3", "SAT4"]:
                    log.debug("Simulating default hardware configuration")
                    hw = sim_nominal()
                    sim_telescope_detectors(
                        hw,
                        meta["telescope"],
                        det_info=(det_info_file, det_info_version),
                        no_darks=det_info_file is not None,
                    )
                else:
                    raise RuntimeError(
                        "Must provide a path to file or a valid telescope name"
                    )

                if (
                    bands is not None
                    or wafer_slots is not None
                    or tube_slots is not None
                ):
                    match = dict()
                    if bands is not None:
                        match["band"] = bands.replace(",", "|")
                    if wafer_slots is not None:
                        match["wafer_slot"] = wafer_slots.split(",")
                    if tube_slots is not None:
                        tube_slots = tube_slots.split(",")
                    hw = hw.select(tube_slots=tube_slots, match=match)

                if thinfp is not None:
                    dets = list(hw.data["detectors"].keys())
                    for det in dets:
                        pixel = hw.data["detectors"][det]["pixel"]
                        try:
                            pixel_id = int(pixel)
                        except ValueError:
                            pixel_id = name_UID(pixel)
                        if pixel_id % thinfp != 0:
                            del hw.data["detectors"][det]

                ndet = len(hw.data["detectors"])
                if ndet == 0:
                    raise RuntimeError(
                        f"No detectors match query: telescope={meta['telescope']}, "
                        f"tube_slots={tube_slots}, wafer_slots={wafer_slots}, "
                        f"bands={bands}, thinfp={thinfp}"
                    )
                else:
                    log.debug(
                        f"{ndet} detectors match query: telescope={meta['telescope']}, "
                        f"tube_slots={tube_slots}, wafer_slots={wafer_slots}, "
                        f"bands={bands}, thinfp={thinfp}"
                    )

            if comm is not None:
                hw = comm.bcast(hw)

        def get_par_float(ddata, key, default):
            if key in ddata:
                return float(ddata[key])
            else:
                return float(default)

        (
            readout_id,
            names,
            quats,
            bands,
            wafer_slots,
            tube_slots,
            nets,
            net_corrs,
            fknees,
            fmins,
            alphas,
            As,
            Cs,
            bandcenters,
            bandwidths,
            ids,
            pixels,
            fwhms,
            pols,
            pol_angs,
            pol_angs_wafer,
            pol_orientations_wafer,
            gamma,
            card_slots,
            channels,
            AMCs,
            biases,
            readout_freqs,
            bondpads,
            mux_positions,
            tele_wf_band,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for det_name, det_data in hw.data["detectors"].items():
            readout_id.append(
                build_readout_id(
                    creation_time, det_data["wafer_slot"], det_data["channel"]
                )
            )
            names.append(det_name)
            quats.append(np.array([float(x) for x in det_data["quat"]]))
            ids.append(int(det_data["ID"]))
            # Replace pixel index on the wafer with a unique identifier that
            # combines telescope, wafer and pixel index
            pixels.append(
                f"{meta['telescope']}_{det_data['wafer_slot']}_p{det_data['pixel']}"
            )
            fwhms.append(float(det_data["fwhm"]) * u.arcmin)
            pols.append(det_data["pol"])
            pol_angs.append(det_data["pol_ang"] * u.degree)
            pol_angs_wafer.append(det_data["pol_ang_wafer"] * u.degree)
            pol_orientations_wafer.append(det_data["pol_orientation_wafer"] * u.degree)
            gamma.append(det_data["pol_ang"] * u.degree)
            card_slots.append(det_data["card_slot"])
            channels.append(det_data["channel"])
            AMCs.append(det_data["AMC"])
            biases.append(det_data["bias"])
            readout_freqs.append(float(det_data["readout_freq_GHz"]) * u.GHz)
            bondpads.append(det_data["bondpad"])
            mux_positions.append(det_data["mux_position"])
            # Band is used to retrieve band-averaged values
            band = det_data["band"]
            band_data = hw.data["bands"][band]
            bands.append(band)
            # Get the wafer slot and translate into tube slot
            wafer_slot = det_data["wafer_slot"]
            wafer_slots.append(wafer_slot)
            # Determine which tube_slot has this wafer
            for tube_slot, tube_data in hw.data["tube_slots"].items():
                if wafer_slot in tube_data["wafer_slots"]:
                    break
            else:
                raise RuntimeError(f"{wafer_slot} is not in any tube_slot")
            tube_slots.append(tube_slot)
            tele_wf_band.append(
                get_tele_wafer_band_name(meta["telescope"], tube_slot, wafer_slot, band)
            )
            # Get noise parameters.  If detector-specific entries are
            # absent, use band averages
            nets.append(
                get_par_float(det_data, "NET", band_data["NET"])
                * 1.0e-6
                * u.K
                * u.s**0.5
            )
            net_corrs.append(get_par_float(det_data, "NET_corr", band_data["NET_corr"]))
            fknees.append(get_par_float(det_data, "fknee", band_data["fknee"]) * u.mHz)
            fmins.append(get_par_float(det_data, "fmin", band_data["fmin"]) * u.mHz)
            alphas.append(get_par_float(det_data, "alpha", band_data["alpha"]))
            As.append(get_par_float(det_data, "A", band_data["A"]))
            Cs.append(get_par_float(det_data, "C", band_data["C"]))
            # bandpass
            lower = get_par_float(det_data, "low", band_data["low"]) * u.GHz
            # center = get_par_float(det_data, "center", band_data["center"]) * u.GHz
            upper = get_par_float(det_data, "high", band_data["high"]) * u.GHz
            bandcenters.append(0.5 * (lower + upper))
            bandwidths.append(upper - lower)

        meta["platescale"] = hw.data["telescopes"][meta["telescope"]]["platescale"] \
                             * u.deg / u.mm

        detdata = QTable(
            [
                readout_id,
                names,
                ids,
                quats,
                bands,
                card_slots,
                wafer_slots,
                tube_slots,
                fwhms,
                nets,
                net_corrs,
                fknees,
                fmins,
                alphas,
                As,
                Cs,
                bandcenters,
                bandwidths,
                pixels,
                pols,
                pol_angs,
                pol_angs_wafer,
                pol_orientations_wafer,
                gamma,
                channels,
                AMCs,
                biases,
                readout_freqs,
                bondpads,
                mux_positions,
                tele_wf_band,
            ],
            names=[
                "readout_id",
                "name",
                "uid",
                "quat",
                "band",
                "card_slot",
                "wafer_slot",
                "tube_slot",
                "FWHM",
                "psd_net",
                "NET_corr",
                "psd_fknee",
                "psd_fmin",
                "psd_alpha",
                "elevation_noise_a",
                "elevation_noise_c",
                "bandcenter",
                "bandwidth",
                "pixel",
                "pol",
                "pol_ang",
                "pol_ang_wafer",
                "pol_orientation_wafer",
                "gamma",
                "channel",
                "AMC",
                "bias",
                "readout_freq",
                "bondpad",
                "mux_position",
                "tele_wf_band",
            ],
            meta=meta,
        )

        super().__init__(
            detector_data=detdata,
            field_of_view=field_of_view,
            sample_rate=sample_rate,
        )


def update_creation_time(det_data, creation_time):
    """Update the readout_id column of a focalplane with a new creation time.

    Args:
        det_data (QTable):  The detector properties table.
        creation_time (float):  The updated time for use in the readout_id

    Returns:
        None

    """
    for row in range(len(det_data)):
        wf, ct, chan = parse_readout_id(det_data["readout_id"][row])
        det_data["readout_id"][row] = build_readout_id(creation_time, wf, chan)


def simulated_telescope(
    hw=None,
    hwfile=None,
    det_info_file=None,
    det_info_version=None,
    telescope_name=None,
    sample_rate=10 * u.Hz,
    bands=None,
    wafer_slots=None,
    tube_slots=None,
    thinfp=None,
    comm=None,
):
    if hw is not None and telescope_name is None:
        # get it from the hw
        if len(hw.data["telescopes"]) != 1:
            raise RuntimeError("Input Hardware has multiple telescopes")
        telescope_name = list(hw.data["telescopes"].keys())[0]
    focalplane = SOFocalplane(
        hw=hw,
        hwfile=hwfile,
        det_info_file=det_info_file,
        det_info_version=det_info_version,
        telescope=telescope_name,
        sample_rate=sample_rate,
        bands=bands,
        wafer_slots=wafer_slots,
        tube_slots=tube_slots,
        thinfp=thinfp,
        comm=comm,
    )
    site = SOSite()
    # The focalplane construction above will lookup the telescope name if needed
    # from the wafer / tube information
    telescope = Telescope(
        focalplane.detector_data.meta["telescope"],
        focalplane=focalplane,
        site=site,
    )
    return telescope

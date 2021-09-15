# Copyright (c) 2019-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import sys

import astropy.units as u
from astropy.table import QTable
from toast.instrument import Focalplane
from toast.utils import Logger

from ..sim_hardware import get_example, sim_telescope_detectors


FOCALPLANE_RADII = {
    "LAT" : 3.6 * u.deg,
    "SAT1" : 17.8 * u.deg,
    "SAT2" : 17.8 * u.deg,
    "SAT3" : 17.8 * u.deg,
    "SAT4" : 17.2 * u.deg,
}


def get_telescope(telescope, wafer_slots, tube_slots):
    """ Determine which telescope matches the detector selections
    """
    if telescope is not None:
        return telescope
    # User did not set telescope so we infer it from the
    # tube and wafer slots
    hwexample = get_example()
    if wafer_slots is not None:
        wafer_slots = wafer_slots.split(",")
        wafer_map = hwexample.wafer_map()
        tube_slots = [wafer_map["tube_slots"][ws] for ws in wafer_slots]
    elif tube_slots is not None:
        tube_slots = tube_slots.split(",")
    else:
        raise RuntimeError("Must set telescope, wafer_slots or tube_slots.")
    for tube_slot in tube_slots:
        for telescope_name, telescope_data in hwexample.data[
            "telescopes"
        ].items():
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


class SOFocalplane(Focalplane):
    """ SO Focalplane class
    """

    def __init__(
            self,
            hwfile=None,
            telescope=None,
            sample_rate=10 * u.Hz,
            bands=None,
            wafer_slots=None,
            tube_slots=None,
            thinfp=None,
            comm=None,
    ):
        log = Logger.get()
        self.telescope = get_telescope(telescope, wafer_slots, tube_slots)
        field_of_view = 2 * FOCALPLANE_RADII[self.telescope]

        if comm is None or comm.rank == 0:
            if hwfile is not None:
                log.info(f"Loading hardware configuration from {hwfile}...")
                hw = Hardware(args.hardware)
            elif self.telescope in ["LAT", "SAT1", "SAT2", "SAT3"]:
                log.info("Simulating default hardware configuration")
                hw = get_example()
                hw.data["detectors"] = sim_telescope_detectors(
                    hw, self.telescope,
                )
            else:
                raise RuntimeError(
                    "Must provide a path to file or a valid telescope name"
                )

            match = {"band": bands.replace(",", "|")}
            if wafer_slots is not None:
                match["wafer_slot"]  = wafer_slots.split(",")
            if tube_slots is not None:
                tube_slots = tube_slots.split(",")
            hw = hw.select(tube_slots=tube_slots, match=match)

            if thinfp is not None:
                dets = list(hw.data["detectors"].keys())
                for det in dets:
                    pixel = hw.data["detectors"][det]["pixel"]
                    if int(pixel) % thinfp != 0:
                        del hw.data["detectors"][det]

            ndet = len(hw.data["detectors"])
            if ndet == 0:
                raise RuntimeError(
                    f"No detectors match query: telescope={self.telescope}, "
                    f"tube_slots={tube_slots}, wafer_slots={wafer_slots}, "
                    f"bands={bands}, thinfp={thinfp}"
                )
            else:
                log.info(
                    f"{ndet} detectors match query: telescope={self.telescope}, "
                    f"tube_slots={tube_slots}, wafer_slots={wafer_slots}, "
                    f"bands={bands}, thinfp={thinfp}"
                )
        else:
            hw = None

        if comm is not None:
            hw = comm.bcast(hw)

        def get_par(key, default):
            if key in det_data:
                return det_data[key]
            else:
                return default

        (
            names, quats, bands, wafer_slots, tube_slots, nets, net_corrs,
            fknees, fmins, alphas, As, Cs, bandcenters, bandwidths,
            ids, pixels, fwhms, pols, card_slots, channels, AMCs, biases,
            readout_freqs, bondpads, mux_positions,
        ) = (
            [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], [], [], [],
        )
        for det_name, det_data in hw.data["detectors"].items():
            names.append(det_name)
            quats.append(det_data["quat"])
            ids.append(det_data["ID"])
            pixels.append(det_data["pixel"])
            fwhms.append(det_data["fwhm"] * u.arcmin)
            pols.append(det_data["pol"])
            card_slots.append(det_data["card_slot"])
            channels.append(det_data["channel"])
            AMCs.append(det_data["AMC"])
            biases.append(det_data["bias"])
            readout_freqs.append(det_data["readout_freq_GHz"] * u.GHz)
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
            # Get noise parameters.  If detector-specific entries are
            # absent, use band averages
            nets.append(get_par("NET", band_data["NET"]) * u.uK * u.s ** .5)
            net_corrs.append(get_par("NET_corr", band_data["NET_corr"]))
            fknees.append(get_par("fknee", band_data["fknee"]) * u.mHz)
            fmins.append(get_par("fmin", band_data["fmin"]) * u.mHz)
            #alphas.append(get_par("alpha", band_data["alpha"]))
            alphas.append(1)  # hardwire a sensible number. 3.5 is not realistic.
            As.append(get_par("A", band_data["A"]))
            Cs.append(get_par("C", band_data["C"]))
            # bandpass
            lower = get_par("low", band_data["low"]) * u.GHz
            #center = get_par("center", band_data["center"]) * u.GHz
            upper = get_par("high", band_data["high"]) * u.GHz
            bandcenters.append(0.5 * (lower + upper))
            bandwidths.append(upper - lower)

        detdata = QTable(
            [
                names, ids, quats, bands, card_slots, wafer_slots, tube_slots,
                fwhms, nets, net_corrs, fknees, fmins, alphas,
                As, Cs,
                bandcenters, bandwidths, pixels, pols, channels, AMCs, biases,
                readout_freqs, bondpads, mux_positions
            ],
            names=[
                "name", "uid", "quat", "band", "card_slot", "wafer_slot", "tube_slot",
                "FWHM", "psd_net", "NET_corr", "psd_fknee", "psd_fmin", "psd_alpha",
                "elevation_noise_a", "elevation_noise_c",
                "bandcenter", "bandwidth", "pixel", "pol", "channel", "AMC", "bias",
                "readout_freq", "bondpad", "mux_position",
            ])

        super().__init__(
            detector_data=detdata,
            field_of_view=field_of_view,
            sample_rate=sample_rate,
        )

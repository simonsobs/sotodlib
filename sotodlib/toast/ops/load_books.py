# Copyright (c) 2022-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import re
import datetime

import numpy as np
import traitlets
from astropy import units as u
import yaml

from toast.timing import function_timer, Timer
from toast.traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance, List
from toast.ops.operator import Operator
from toast.utils import Environment, Logger
from toast.dist import distribute_discrete
from toast.observation import default_values as defaults

from ..io import parse_book_name, parse_book_time, read_book
from ...sim_hardware import sim_nominal
from ..instrument import SOFocalplane


@trait_docs
class LoadBooks(Operator):
    """Load Level-3 Books into observations.

    If no focalplane directory is specified, a minimal focalplane is created using only
    the stream_id and readout names in the G3 files.  Similarly for noise models.  If
    the noise_dir is specified, noise models are loaded from there.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    books = List(list(), help="List of observation book directories")

    focalplane_dir = Unicode(
        None, allow_none=True, help="Directory for focalplane models"
    )

    detset_key = Unicode(
        None,
        allow_none=True,
        help="If specified, use this column of the focalplane detector_data to group detectors",
    )

    noise_dir = Unicode(None, allow_none=True, help="Directory for noise models")

    wafers = List(list(), help="Only load detectors from these wafers / stream_ids")

    bands = List(list(), help="Only load detectors from these bands")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key for detector signal",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for detector flags",
    )

    azimuth = Unicode(
        defaults.azimuth, help="Observation shared key for boresight Azimuth"
    )

    elevation = Unicode(
        defaults.elevation, help="Observation shared key for boresight Elevation"
    )

    boresight_azel = Unicode(
        defaults.boresight_azel,
        help="Observation shared key for boresight Az/El quaternions",
    )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        help="Observation shared key for boresight RA/DEC quaternions",
    )

    corotator_angle = Unicode(
        defaults.corotator_angle,
        allow_none=True,
        help="Observation shared key for corotator_angle (if it is used)",
    )

    boresight_angle = Unicode(
        defaults.boresight_angle,
        allow_none=True,
        help="Observation shared key for boresight rotation angle (if it is used)",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle,
        allow_none=True,
        help="Observation shared key for HWP angle",
    )

    frame_intervals = Unicode(
        None,
        allow_none=True,
        help="Observation interval key for frame boundaries",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        if self.books is None or len(self.books) == 0:
            raise RuntimeError("No books specified, nothing to load")

        # Extract the frequency "name" from any specified bands
        band_freqs = list()
        band_pat = re.compile(r".*(\d+)")
        if self.bands is not None:
            for bnd in self.bands:
                mat = band_pat.match(bnd)
                if mat is None:
                    raise RuntimeError(
                        "If specified, band names must end with the frequency"
                    )
                band_freqs.append(int(mat.group(1)))

        book_props = None
        if data.comm.group_rank == 0:
            # We are only using the tube / wafer / band information here
            # (not detector properties) so the nominal hardware model should be fine.
            hw = sim_nominal()

            # Parse the book names and corresponding information
            book_pat = re.compile(r"obs_(\d+)_(.*)_(\d+)")
            book_props = dict()
            for path in self.books:
                book_name = os.path.basename(path)
                book_timestamp, book_tele, book_tube, book_wafers = parse_book_name(
                    book_name
                )
                # print(
                #     f"Load found book {book_timestamp}, {book_tele}, {book_tube}, {book_wafers}"
                # )
                wf_props = dict()
                for wf in book_wafers:
                    if len(self.wafers) > 0 and wf not in self.wafers:
                        # print(f"  wafer {wf} not in list, skipping")
                        continue
                    bands = hw.data["wafer_slots"][wf]["bands"]
                    npix = hw.data["wafer_slots"][wf]["npixel"]
                    use_bands = list()
                    for bnd in bands:
                        bmat = band_pat.match(bnd)
                        # if bmat is None:
                        #     print(f"  band {bnd} does not match- should never happen!")
                        bfreq = int(bmat.group(1))
                        if len(band_freqs) > 0 and bfreq not in band_freqs:
                            # print(f"  band {bnd} ({bfreq}) not in list, skipping")
                            continue
                        use_bands.append(bnd)
                    if len(use_bands) > 0:
                        # print(f"  Adding wafer {wf}")
                        wf_props[wf] = {
                            "npixel": npix,
                            "bands": use_bands,
                        }
                    # else:
                    #     print(f"skipping wafer {wf} with no selected bands")
                if len(wf_props) == 0:
                    msg = (
                        f"book '{book_name}' had no wafers matching wafer / band lists"
                    )
                    log.warning(msg)
                else:
                    # Use this book.
                    book_props[book_name] = {
                        "path": path,
                        "timestamp": book_timestamp,
                        "telescope": book_tele,
                        "tube": book_tube,
                        "wafers": wf_props,
                    }
        if data.comm.comm_group is not None:
            book_props = data.comm.comm_group.bcast(book_props, root=0)

        if len(book_props) == 0:
            raise RuntimeError("No books found matching wafer / band criteria")

        # Determine the approximate size of each book, for load balancing among groups

        book_names = list(sorted(book_props.keys()))
        book_sizes = list()
        if data.comm.group_rank == 0:
            for name in book_names:
                props = book_props[name]
                m_obs_file = os.path.join(props["path"], "M_index.yaml")
                if not os.path.isfile(m_obs_file):
                    msg = f"Book directory {path} does not contain 'M_index.yaml'"
                    log.error(msg)
                    raise RuntimeError(msg)
                with open(m_obs_file, "r") as f:
                    m_obs = yaml.load(f, Loader=yaml.FullLoader)
                start = parse_book_time(m_obs["timestamp_start"])
                end = parse_book_time(m_obs["timestamp_end"])
                duration = (end - start).total_seconds() / 3600.0
                total_pix = np.sum(
                    [
                        (v["npixel"] * len(v["bands"]))
                        for k, v in props["wafers"].items()
                    ]
                )
                book_sizes.append(duration * total_pix)
        if data.comm.comm_group is not None:
            book_sizes = data.comm.comm_group.bcast(book_sizes, root=0)

        # Dictionary of observation field names
        obs_fields = {
            "times": self.times,
            "shared_flags": self.shared_flags,
            "det_data": self.det_data,
            "det_flags": self.det_flags,
            "hwp_angle": self.hwp_angle,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "boresight_azel": self.boresight_azel,
            "boresight_radec": self.boresight_radec,
            "corotator_angle": self.corotator_angle,
            "boresight_angle": self.boresight_angle,
        }

        # Distribute books among groups.

        groupdist = distribute_discrete(book_sizes, data.comm.ngroups)

        # Every process group creates observations from their books

        group_firstbook = groupdist[data.comm.group][0]
        group_numbooks = groupdist[data.comm.group][1]

        fp_cache = dict()

        for book_indx in range(group_firstbook, group_firstbook + group_numbooks):
            bname = book_names[book_indx]
            bprops = book_props[bname]
            telename = bprops["telescope"]

            msg = f"Process group {data.comm.group} loading book {bname}"
            log.info_rank(msg, comm=data.comm.comm_group)
            if telename in fp_cache:
                nominal_fp = fp_cache[telename]
            else:
                # We are only using this for a nominal channel map- the
                # sample rate does not matter.
                nominal_fp = SOFocalplane(
                    sample_rate=1.0 * u.Hz,
                    telescope=telename,
                )
                fp_cache[telename] = nominal_fp

            session_obs = read_book(
                data.comm,
                nominal_fp,
                bprops["path"],
                self.focalplane_dir,
                self.noise_dir,
                obs_fields,
                bprops["tube"],
                bprops["wafers"],
                detset_key=self.detset_key,
                frame_intervals=self.frame_intervals,
            )

            # Redistribute each observation and append to the output.  Also
            # create a set of intervals corresponding to the frame boundaries.
            for ob in session_obs:
                ob.redistribute(data.comm.group_size, times=obs_fields["times"])
                data.obs.append(ob)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {
            "shared": [
                self.times,
                self.boresight_azel,
            ],
            "detdata": [self.det_data],
        }
        if self.boresight_radec is not None:
            prov["shared"].append(self.boresight_radec)
        if self.boresight_angle is not None:
            prov["shared"].append(self.boresight_angle)
        if self.corotator_angle is not None:
            prov["shared"].append(self.corotator_angle)
        if self.hwp_angle is not None:
            prov["shared"].append(self.hwp_angle)
        return prov

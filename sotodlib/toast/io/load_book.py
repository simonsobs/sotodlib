# Copyright (c) 2020-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for loading Level-3 Book format data.

"""

import os
import sys
from datetime import datetime, timezone
import re
import glob

import yaml
import numpy as np
from astropy import units as u
from astropy.table import Table, QTable
import h5py

# Import so3g before any other packages that import spt3g
import so3g
from spt3g import core as c3g

import toast
import toast.spt3g as t3g
import toast.qarray as qa
from toast.observation import default_values as defaults
from toast.instrument import Telescope, Focalplane, Session, GroundSite
from toast.utils import Logger

from ...core.hardware import parse_readout_id
from ...sim_hardware import telescope_tube_wafer
from ..instrument import (
    SOSite,
    SOFocalplane,
    update_creation_time,
    get_tele_wafer_band_name,
)


def parse_book_name(name):
    """Parse an observation book name.

    This returns the book timestamp, telescope, tube, and list of wafers / streams.

    Args:
        name (str):  The name of the book.

    Returns:
        (tuple):  The timestamp, telescope, tube, and wafers.

    """
    log = toast.utils.Logger.get()

    book_pat = re.compile(r"obs_(\d+)_(.*)_(\d+)")
    book_mat = book_pat.match(name)
    if book_mat is None:
        raise RuntimeError(f"'{name}' is not an observation book")
    book_timestamp = float(book_mat.group(1))
    book_teltube = book_mat.group(2)
    book_slots = book_mat.group(3)

    lat_pat = re.compile(r"lat(.*)")
    lat_mat = lat_pat.match(book_teltube)
    if lat_mat is None:
        # This is a sat
        tele_name = book_teltube.upper()
        tube_name = f"ST{tele_name[-1]}"
    else:
        tele_name = "LAT"
        tube_name = lat_mat.group(1)

    ttw = telescope_tube_wafer()
    wafers = list()
    for bit, wf in zip(book_slots, ttw[tele_name][tube_name]):
        if bit == "1":
            wafers.append(wf)

    return book_timestamp, tele_name, tube_name, wafers


def parse_book_time(dstr):
    """Parse a book observation timestamp.

    Args:
        dstr (str):  The timestamp string.

    Returns:
        (datetime):  The datetime object.

    """
    return datetime.strptime(dstr, "%Y-%m-%dT%H:%M:%S%z")


class MetaCollector(object):
    """Class to parse frames and build metadata."""

    def __init__(self):
        self.frame_sizes = list()
        self.readout_ids = None
        self.sample_rate = list()

    def __call__(self, frame):
        if frame is not None and frame.type != c3g.G3FrameType.EndProcessing:
            if frame.type == c3g.G3FrameType.Scan:
                # We have a Scan frame, find the number of samples
                srange = np.array(frame["sample_range"])
                if len(srange) != 2:
                    raise RuntimeError(
                        f"frame sample range has {len(srange)} elements instead of 2"
                    )
                if srange[0] >= srange[1]:
                    raise RuntimeError(
                        f"frame has no samples (range = {srange[0]} - {srange[1]}"
                    )
                self.frame_sizes.append(srange[1] - srange[0])

                # If this is the first Scan frame, get the list of detector readout IDs
                if self.readout_ids is None:
                    self.readout_ids = list(frame["signal"].names)
                    # print(f"metacollect first frame, set readout_ids to {self.readout_ids[0]} .. {self.readout_ids[-1]}", flush=True)

                # Estimate the sample rate from the timestamps
                stamps = t3g.from_g3_time(frame["ancil"].times)
                (rate, dt, dt_min, dt_max, dt_std) = toast.utils.rate_from_times(stamps)
                self.sample_rate.append(rate)
        return


def parse_frame_meta(frame_files):
    """Parse the frame information.

    Args:
        frame_files (list):  List of frame files for one wafer.

    Returns:
        (tuple):  The frame sizes, readout names, sample rate.

    """
    meta_collect = MetaCollector()
    for ffile in frame_files:
        # print(f"running metacollect pipe on {ffile}")
        load_pipe = c3g.G3Pipeline()
        load_pipe.Add(c3g.G3Reader(ffile))
        load_pipe.Add(meta_collect)
        load_pipe.Run()
    return (
        np.array(meta_collect.frame_sizes, dtype=np.int32),
        list(meta_collect.readout_ids),
        np.median(meta_collect.sample_rate),
    )


class DataSpreader(object):
    """Class to distribute frames.

    This class receives frames on the rank zero process of a group and uses
    point-to-point communication to send frames to the correct process.  Because
    This only accumulates and sends one process worth of data at a time, the
    entire observation never exists on the root process.  The tradeoff however
    is a performance hit compared to gathering all data and performing a scatterv.

    """

    def __init__(
        self,
        obs,
        obs_fields,
    ):
        self.obs = obs
        self.obs_fields = obs_fields
        self.dist = obs.dist
        # The distribution sample sets represent the frame sizes.  Make a mapping of
        # frame index to process.
        foff = 0
        self.frame_target = dict()
        for iproc, prange in enumerate(self.dist.samp_sets):
            for ifr in range(prange.n_elem):
                # Each sample set contains exactly one frame
                self.frame_target[foff] = iproc
                foff += 1
        self.cur_frame = 0
        self.cur_proc = 0
        self.send_buffer = list()
        self.recv_buffer = list()

    def __call__(self, frame):
        # This function will only be called on rank 0
        if frame is not None and frame.type == c3g.G3FrameType.Scan:
            # Accumulate scan frame to send buffer
            if self.frame_target[self.cur_frame] != self.cur_proc:
                # Done with this process
                if self.cur_proc == 0:
                    # Copy to our own receive buffer
                    self.recv_buffer = [x for x in self.send_buffer]
                else:
                    self.obs.comm.comm_group.send(
                        self.send_buffer, self.cur_proc, tag=self.cur_proc
                    )
                self.cur_proc = self.frame_target[self.cur_frame]
                self.send_buffer.clear()
            self.send_buffer.append(frame)
            self.cur_frame += 1

    def flush(self):
        if self.obs.comm.group_rank != 0:
            # Wait to receive our list of frames.  This code only runs if the group
            # has more than one process, so no need to check comm for None.
            self.recv_buffer = self.obs.comm.comm_group.recv(
                source=0,
                tag=self.obs.comm.group_rank,
            )
            self._unpack_frames(self.recv_buffer)
        else:
            # Root process flushes remaining buffer
            if self.cur_proc == 0:
                # This means there was only one process- just unpack the
                # send buffer.
                self._unpack_frames(self.send_buffer)
            else:
                # We were accumulating the buffer for another process when
                # the frame stream ended.  Send it now.
                self.obs.comm.comm_group.send(
                    self.send_buffer, self.cur_proc, tag=self.cur_proc
                )
                # Also unpack our own frames
                self._unpack_frames(self.recv_buffer)
        # Reset counters
        self.cur_frame = 0
        self.cur_proc = 0
        self.recv_buffer = list()
        self.send_buffer = list()

    def _unpack_frames(self, buffer):
        # Since the data is distributed in the sample direction, every process has
        # exclusive use of the column shared objects.  This means it is safe to
        # set those buffers directly.
        off = 0
        detmap = None
        # print(f"UNPACK frames for telescope {self.obs.telescope.name}", flush=True)
        fpdata = self.obs.telescope.focalplane.detector_data
        # print(f"  FPDATA = {fpdata[:3]}", flush=True)
        for frm in buffer:
            ancil = frm["ancil"]
            times = t3g.from_g3_time(ancil.times)
            n = len(times)
            self.obs.shared[self.obs_fields["times"]].data[off : off + n] = times
            self.obs.shared[self.obs_fields["azimuth"]].data[off : off + n] = np.array(
                ancil["az_enc"], copy=False
            )
            self.obs.shared[self.obs_fields["elevation"]].data[off : off + n] = ancil[
                "el_enc"
            ]
            if "flags" in ancil:
                self.obs.shared[self.obs_fields["shared_flags"]].data[
                    off : off + n
                ] = np.array(ancil["flags"]).astype(np.uint8)

            self.obs.shared[self.obs_fields["boresight_azel"]].data[
                off : off + n, :
            ] = t3g.from_g3_quats(frm["qboresight_azel"])

            if "qboresight_radec" in frm:
                self.obs.shared[self.obs_fields["boresight_radec"]].data[
                    off : off + n, :
                ] = t3g.from_g3_quats(frm["qboresight_radec"])

            if "hwp_enc" in ancil:
                self.obs.shared[self.obs_fields["hwp_angle"]].data[
                    off : off + n
                ] = np.array(ancil["hwp_enc"]).astype(np.float64)

            if "corotation_enc" in ancil:
                self.obs.shared[self.obs_fields["corotator_angle"]].data[
                    off : off + n
                ] = np.array(ancil["corotation_enc"]).astype(np.float64)

            if "boresight_enc" in ancil:
                self.obs.shared[self.obs_fields["boresight_angle"]].data[
                    off : off + n
                ] = np.array(ancil["boresight_enc"]).astype(np.float64)

            if detmap is None:
                # Compute the mapping between observation detectors and rows of the
                # supertimestream.
                # print(f"  First frame, signal names = {frm['signal'].names[0]} .. {frm['signal'].names[-1]}")
                super_name_to_row = {y: x for x, y in enumerate(frm["signal"].names)}
                # print(super_name_to_row, flush=True)
                detmap = np.array([super_name_to_row[x] for x in fpdata["readout_id"]])
                # print(f"{self.obs.name} detmap = {detmap}")
            # print(f"DBG LOAD frame {off}:{off+n} signal = {frm['signal'].data[:5,:]}")
            # print(f"LOAD frame {off}:{off+n} signal = {frm['signal'].data[detmap,:]}")
            self.obs.detdata[self.obs_fields["det_data"]][:, off : off + n] = frm[
                "signal"
            ].data[detmap, :]
            if "signal_units" in frm and frm["signal_units"] is not None:
                # We have some unit information
                # print(f"Frame units = {frm['signal_units']}")
                self.obs.detdata[self.obs_fields["det_data"]].update_units(
                    t3g.from_g3_unit(frm["signal_units"])
                )
            self.obs.detdata[self.obs_fields["det_flags"]][:, off : off + n] = np.array(
                frm["flags"].data[detmap, :], dtype=np.uint8
            )

            off += n


def import_obs_data(
    frame_files,
    ob,
    fields,
):
    # Create observation fields
    ob.shared.create_column(
        fields["times"],
        shape=(ob.n_local_samples,),
        dtype=np.float64,
    )
    ob.shared.create_column(
        fields["azimuth"],
        shape=(ob.n_local_samples,),
        dtype=np.float64,
    )
    ob.shared.create_column(
        fields["elevation"],
        shape=(ob.n_local_samples,),
        dtype=np.float64,
    )
    ob.shared.create_column(
        fields["boresight_azel"],
        shape=(ob.n_local_samples, 4),
        dtype=np.float64,
    )
    ob.shared.create_column(
        fields["boresight_radec"],
        shape=(ob.n_local_samples, 4),
        dtype=np.float64,
    )
    if fields["hwp_angle"] is not None:
        ob.shared.create_column(
            fields["hwp_angle"],
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if fields["boresight_angle"] is not None:
        ob.shared.create_column(
            fields["boresight_angle"],
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if fields["corotator_angle"] is not None:
        ob.shared.create_column(
            fields["corotator_angle"],
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
    if fields["shared_flags"] is not None:
        ob.shared.create_column(
            fields["shared_flags"],
            shape=(ob.n_local_samples,),
            dtype=np.uint8,
        )

    ob.shared.create_column(
        defaults.position,
        shape=(ob.n_local_samples, 3),
        dtype=np.float64,
    )
    ob.shared.create_column(
        defaults.velocity,
        shape=(ob.n_local_samples, 3),
        dtype=np.float64,
    )

    ob.detdata.create(fields["det_data"], dtype=np.float64)
    ob.detdata.create(fields["det_flags"], dtype=np.uint8)

    # Distribute frames and copy into place

    spreader = DataSpreader(ob, fields)
    if ob.comm.group_rank == 0:
        # Read and send
        for ffile in frame_files:
            load_pipe = c3g.G3Pipeline()
            load_pipe.Add(c3g.G3Reader(ffile))
            load_pipe.Add(spreader)
            load_pipe.Run()
    # Unpack local frames
    spreader.flush()


@toast.timing.function_timer
def read_book(
    comm,
    fp_nominal,
    book_dir,
    focalplane_dir,
    noise_dir,
    obs_fields,
    tube,
    wafer_bands,
    detset_key=None,
    frame_intervals=None,
    ignore_sim=False,
):
    """Read a book into single observing session.

    Observations are created for each selected wafer and band.

    Args:
        comm (toast.Comm):  The toast communicator.
        fp_nominal (SOFocalplane):  The nominal focalplane.
        book_dir (str):  The directory containing the book.
        focalplane_dir (str):  Path to directory of focalplane files.
        noise_dir (str):  Path to directory of noise models.
        obs_fields (dict):  Dictionary of observation fields to use.
        tube (str):  The tube slot.
        wafer_bands (dict):  Dictionary of bands to load for each wafer.
        frame_intervals (str):  The intervals to use for frame boundaries.
        ignore_sim (bool):  If True, do not load toast simulation information
            in the M_observation.yaml file.  This will emulate loading real data,
            but the resulting observation cannot be used for reproducing the
            simulated content.

    Returns:
        (list):  The list of observations for this session.

    """
    log = Logger.get()
    book_name = os.path.basename(book_dir)
    # One process loads book metadata
    book_meta = None
    obs_meta = None
    if comm.group_rank == 0:
        book_meta_file = os.path.join(book_dir, f"M_book.yaml")
        # print(f"Loading {book_meta_file}")
        with open(book_meta_file, "r") as f:
            book_meta = yaml.load(f, Loader=yaml.Loader)
        # print(f"{book_dir}:  {book_meta}")
        obs_meta_file = os.path.join(book_dir, f"M_observation.yaml")
        # print(f"Loading {obs_meta_file}")
        with open(obs_meta_file, "r") as f:
            obs_meta = yaml.load(f, Loader=yaml.Loader)
        # print(f"{book_dir}:  {obs_meta}")
    if comm.comm_group is not None:
        book_meta = comm.comm_group.bcast(book_meta, root=0)
        obs_meta = comm.comm_group.bcast(obs_meta, root=0)

    # All of the observation metadata is currently duplicated between the observation
    # frame and the M_observation.yaml file.  However, if that changes we could grab
    # the observation frame from any of the files here.

    book_session = False
    tele_uid = None
    if not ignore_sim:
        # Attempt to restore the session and site information that was used during
        # the sim.
        if "toast" in obs_meta:
            toast_meta = obs_meta["toast"]
            try:
                tele_uid = toast_meta["telescope_uid"]
            except KeyError:
                msg = f"Book {book_name} toast sim info does not have telescope UID "
                msg += f"Will use default value based on telescope name."
                log.warning_rank(msg, comm=comm.comm_group)
            try:
                sname = toast_meta["observing_session"]
                sstart = toast_meta["observing_session_start"]
                send = toast_meta["observing_session_end"]
                suid = None
                if "observing_session_uid" in toast_meta:
                    suid = toast_meta["observing_session_uid"]
                session = Session(
                    sname,
                    uid=suid,
                    start=datetime.fromtimestamp(sstart).astimezone(timezone.utc),
                    end=datetime.fromtimestamp(send).astimezone(timezone.utc),
                )
            except KeyError:
                msg = f"Book {book_name} toast sim info does not have valid observing "
                msg += f" session.  Will use nominal values."
                log.warning_rank(msg, comm=comm.comm_group)
                book_session = True

            weather = None
            try:
                wname = toast_meta["site_weather_name"]
                wreal = toast_meta["site_weather_realization"]
                wtime = toast_meta["site_weather_time"]
                wsuid = toast_meta["site_weather_uid"]
                wmaxpwv = toast_meta["site_weather_max_pwv"]
                if wmaxpwv == "NONE":
                    wmaxpwv = None
                else:
                    wmaxpwv = u.Quantity(wmaxpwv)
                use_median = bool(toast_meta["site_weather_use_median"])
                weather = toast.weather.SimWeather(
                    time=datetime.fromtimestamp(wtime).astimezone(timezone.utc),
                    name=wname,
                    site_uid=wsuid,
                    realization=wreal,
                    max_pwv=wmaxpwv,
                    median_weather=use_median,
                )
            except KeyError:
                # weather info may not exist
                pass

            try:
                sialt = toast_meta["site_alt_m"]
                silat = toast_meta["site_lat_deg"]
                silon = toast_meta["site_lon_deg"]
                siname = toast_meta["site_name"]
                siuid = None
                if "site_uid" in toast_meta:
                    siuid = toast_meta["site_uid"]
                site = GroundSite(
                    siname,
                    silat * u.degree,
                    silon * u.degree,
                    sialt * u.meter,
                    uid=siuid,
                    weather=weather,
                )
            except KeyError:
                msg = f"Book {book_name} toast sim info does not have valid site "
                msg += f" information.  Will use nominal values."
                log.warning_rank(msg, comm=comm.comm_group)
                book_session = True
        else:
            msg = f"Cannot load simulation info from book {book_name}, "
            msg += f"use ignore_sim=True?"
            log.warning_rank(msg, comm=comm.comm_group)
    else:
        book_session = True

    if book_session:
        # Create a nominal Site and a Session based only on the book information
        session = Session(
            book_name,
            start=datetime.fromisoformat(obs_meta["timestamp_start"]),
            end=datetime.fromisoformat(obs_meta["timestamp_end"]),
        )
        site = SOSite()

    session_obs = list()

    # Compute the mapping from wafer and channel to row of the nominal focalplane
    wchan = dict()
    for wafer, props in wafer_bands.items():
        wchan[wafer] = dict()
    for irow, row in enumerate(fp_nominal.detector_data):
        if row["wafer_slot"] in wafer_bands:
            wchan[row["wafer_slot"]][row["channel"]] = (irow, row["band"])

    for wafer, props in wafer_bands.items():
        bands = props["bands"]
        framefiles = glob.glob(os.path.join(book_dir, f"D_{wafer}_*.g3"))
        framefiles.extend(glob.glob(os.path.join(book_dir, f"D_{wafer}_*.g3.gz")))

        # Esure that frame files are processed in order!
        framefiles.sort()

        # One process gets the D-frame info
        frame_sizes = None
        readout_ids = None
        sample_rate = None
        if comm.group_rank == 0:
            frame_sizes, readout_ids, sample_rate = parse_frame_meta(framefiles)
        if comm.comm_group is not None:
            frame_sizes = comm.comm_group.bcast(frame_sizes, root=0)
            readout_ids = comm.comm_group.bcast(readout_ids, root=0)
            sample_rate = comm.comm_group.bcast(sample_rate, root=0)
        sample_rate = u.Quantity(sample_rate * u.Hz)
        _, creation_time, _ = parse_readout_id(readout_ids[0])

        # Use the frame boundaries to create sample sets each with one frame
        sample_sets = [[x] for x in frame_sizes]
        total_samples = np.sum(frame_sizes)

        for bnd in bands:
            # Do we have a focalplane model for this data?  If not, we will create a
            # nominal one as a starting point.
            use_nominal = True
            if focalplane_dir is not None:
                fp_file = os.path.join(focalplane_dir, f"{book_name}_{wafer}_{bnd}.h5")
                # print(f"Loading focalplane {fp_file}", flush=True)
                if os.path.isfile(fp_file):
                    # We have a file, load it
                    focalplane = toast.instrument.Focalplane(sample_rate=sample_rate)
                    with toast.io.H5File(
                        fp_file, "r", comm=comm.comm_group, force_serial=True
                    ) as f:
                        focalplane.load_hdf5(f.handle, comm=comm.comm_group)
                    update_creation_time(focalplane.detector_data, creation_time)
                    use_nominal = False
                else:
                    log.warning(
                        f"Focalplane file {fp_file} does not exist, using nominal values"
                    )
            if use_nominal:
                # Find the rows of the frame data which match our band
                fp_rows = list()
                for rid in readout_ids:
                    wf, ct, chan = parse_readout_id(rid)
                    fpr, fpband = wchan[wf][chan]
                    if fpband == bnd:
                        fp_rows.append(fpr)

                # Create a focalplane by extracting these rows from the nominal one.
                detprops = QTable(fp_nominal[fp_rows])
                update_creation_time(detprops, creation_time)
                focalplane = toast.instrument.Focalplane(
                    detector_data=detprops, sample_rate=sample_rate
                )

            # Create the telescope

            tele_name = get_tele_wafer_band_name(
                obs_meta["telescope"],
                tube,
                wafer,
                bnd,
            )
            telescope = Telescope(
                tele_name, uid=tele_uid, focalplane=focalplane, site=site
            )
            obs_name = f"{session.name}_{tele_name}"

            # Create the observation

            if detset_key is None:
                detsets = None
            else:
                detsets = focalplane.detector_groups(detset_key)

            obs = toast.Observation(
                comm,
                telescope,
                total_samples,
                name=obs_name,
                session=session,
                detector_sets=detsets,
                sample_sets=sample_sets,
                process_rows=1,
            )

            # Load frame data into the observation
            # print(f"CALL import obs data:  {framefiles} | {obs_fields}")
            import_obs_data(
                framefiles,
                obs,
                obs_fields,
            )

            # Position and velocity of the observatory are simply computed
            position, velocity = site.position_velocity(obs.shared[obs_fields["times"]])
            obs.shared[defaults.position].data[:] = position
            obs.shared[defaults.velocity].data[:] = velocity

            if noise_dir is not None:
                # FIXME: we could regex match on a glob here to find the
                # observation key name to use.
                noise_file = os.path.join(
                    noise_dir, f"{book_name}_{wafer}_{bnd}_noise_model.h5"
                )
                # print(f"Loading focalplane {fp_file}", flush=True)
                if os.path.isfile(noise_file):
                    # We have a file, load it
                    nse = toast.noise.Noise()
                    with toast.io.H5File(noise_file, "r", comm=comm.comm_group) as f:
                        nse.load_hdf5(f.handle, obs)
                    obs["noise_model"] = nse
                else:
                    log.warning_rank(
                        f"Noise model file {noise_file} does not exist, skipping",
                        comm=comm.comm_group,
                    )

            session_obs.append(obs)

    return session_obs

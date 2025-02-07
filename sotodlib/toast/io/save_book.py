# Copyright (c) 2020-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for saving Level-3 Book format data.

"""

import os
import sys
from datetime import datetime, timezone

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

from ...sim_hardware import telescope_tube_wafer
from ...__init__ import __version__ as sotodlib_version


def obs_wafer_list(obs_list):
    """Get the list of wafers covered by a set of observations.

    Args:
        obs_list (list):  The list of observations.

    Returns:
        (tuple):  The (tube name, list of wafers).

    """
    tube_check = None
    wafers = list()

    for obs in obs_list:
        focalplane = obs.telescope.focalplane
        props = focalplane.detector_data

        obs_tube_check = set(props[:]["tube_slot"])
        if len(obs_tube_check) != 1:
            msg = f"Observation has multiple tube_slots ({obs_tube_check})"
            raise RuntimeError(msg)
        tube_name = obs_tube_check.pop()

        if tube_check is None:
            tube_check = tube_name
        else:
            if tube_name != tube_check:
                msg = f"Observations in session have multiple tubes"
                raise RuntimeError(msg)

        wafer_set = set(props[:]["wafer_slot"])
        for wf in wafer_set:
            if wf not in wafers:
                wafers.append(wf)
    return tube_check, wafers


def book_name_from_obs(obs_list):
    """Return the name of the book, given a list of observations.

    This uses focalplane information to build the name of the book directory.  The
    observations should all be from the same observing session.

    Args:
        obs_list (list):  The list of observations.

    Returns:
        (str):  The book name.

    """
    log = toast.utils.Logger.get()
    timestamp = int(obs_list[0].session.start.timestamp())

    ttw = telescope_tube_wafer()
    tele_name = obs_list[0].telescope.focalplane.detector_data.meta["telescope"]
    tube_name, wafers = obs_wafer_list(obs_list)

    if tele_name == "LAT":
        tele_tube = f"lat{tube_name}"
    else:
        tele_tube = tele_name

    wafer_bitstr = ""
    for wf in ttw[tele_name][tube_name]:
        if wf in wafers:
            wafer_bitstr = f"{wafer_bitstr}1"
        else:
            wafer_bitstr = f"{wafer_bitstr}0"
    return f"obs_{timestamp:10d}_{tele_tube}_{wafer_bitstr}"


def format_book_time(dt):
    """Format a datetime in the desired way for books.

    Args:
        dt (datetime):  A timezone-aware datetime object.

    Returns:
        (str):  The formatted time string.

    """
    return datetime.strftime(dt, "%Y-%m-%dT%H:%M:%S%z")


def write_frame_files(comm, frames, froot, gzip=False):
    fname = f"{froot}_{comm.group_rank:03d}.g3"
    tempname = f"{froot}_{comm.group_rank:03d}_temp.g3"
    if gzip:
        fname += ".gz"
        tempname += ".gz"
    writer = c3g.G3Writer(tempname)
    for frm in frames:
        # if frm.type == c3g.G3FrameType.Scan and "signal" in frm:
        #     print(f"PROCESS {frm['signal'].data}")
        writer.Process(frm)
    del writer
    if comm.comm_group is not None:
        # Wait for all writers to finish
        comm.comm_group.barrier()
    # Move temp file into place
    os.rename(tempname, fname)


def export_obs_meta(obs):
    """Extract Observation metadata.

    Return general metadata as an Observation frame and a corresponding
    dictionary suitable for writing to the book metadata.

    Args:
        obs (Observation):  One observation in the session.

    Returns:
        (tuple):  The (frame, dictionary).

    """
    log = toast.utils.Logger.get()

    # Construct observation frame and dictionary
    ob = c3g.G3Frame(c3g.G3FrameType.Observation)
    obmeta = dict()

    session = obs.session
    log.verbose(f"Create observation frame for session {session.name}")

    ob["telescope_name"] = c3g.G3String(
        obs.telescope.focalplane.detector_data.meta["telescope"]
    )
    obmeta["telescope_name"] = str(
        obs.telescope.focalplane.detector_data.meta["telescope"]
    )
    ob["telescope_uid"] = c3g.G3Int(obs.telescope.uid)
    obmeta["telescope_uid"] = int(obs.telescope.uid)

    ob["observing_session"] = c3g.G3String(session.name)
    obmeta["observing_session"] = str(session.name)

    ob["observing_session_uid"] = c3g.G3Int(session.uid)
    obmeta["observing_session_uid"] = int(session.uid)

    ob["observing_session_start"] = t3g.to_g3_time(session.start.timestamp())
    obmeta["observing_session_start"] = float(session.start.timestamp())

    ob["observing_session_end"] = t3g.to_g3_time(session.end.timestamp())
    obmeta["observing_session_end"] = float(session.end.timestamp())

    site = obs.telescope.site
    siteclass = toast.utils.object_fullname(site.__class__)
    ob["site_name"] = c3g.G3String(site.name)
    ob["site_class"] = c3g.G3String(siteclass)
    ob["site_uid"] = c3g.G3Int(site.uid)
    obmeta["site_name"] = str(site.name)
    obmeta["site_class"] = str(siteclass)
    obmeta["site_uid"] = int(site.uid)

    if isinstance(site, toast.instrument.GroundSite):
        ob["site_lat_deg"] = c3g.G3Double(site.earthloc.lat.to_value(u.degree))
        ob["site_lon_deg"] = c3g.G3Double(site.earthloc.lon.to_value(u.degree))
        ob["site_alt_m"] = c3g.G3Double(site.earthloc.height.to_value(u.meter))
        obmeta["site_lat_deg"] = float(site.earthloc.lat.to_value(u.degree))
        obmeta["site_lon_deg"] = float(site.earthloc.lon.to_value(u.degree))
        obmeta["site_alt_m"] = float(site.earthloc.height.to_value(u.meter))

        if site.weather is not None:
            if hasattr(site.weather, "name"):
                # This is a simulated weather object, dump it.
                ob["site_weather_name"] = c3g.G3String(site.weather.name)
                obmeta["site_weather_name"] = str(site.weather.name)
                ob["site_weather_realization"] = c3g.G3Int(site.weather.realization)
                obmeta["site_weather_realization"] = int(site.weather.realization)
                if site.weather.max_pwv is None:
                    ob["site_weather_max_pwv"] = c3g.G3String("NONE")
                    obmeta["site_weather_max_pwv"] = "NONE"
                else:
                    ob["site_weather_max_pwv"] = c3g.G3Double(site.weather.max_pwv)
                    obmeta["site_weather_max_pwv"] = float(site.weather.max_pwv)
                ob["site_weather_time"] = t3g.to_g3_time(site.weather.time.timestamp())
                obmeta["site_weather_time"] = float(site.weather.time.timestamp())
                ob["site_weather_uid"] = c3g.G3Int(site.weather.site_uid)
                obmeta["site_weather_uid"] = int(site.weather.site_uid)
                ob["site_weather_use_median"] = c3g.G3Bool(site.weather.median_weather)
                obmeta["site_weather_use_median"] = bool(site.weather.median_weather)
    return ob, obmeta


def export_obs_ancil(
    obs,
    book_id,
    times,
    shared_flags,
    boresight_azel,
    boresight_radec=None,
    corotator_angle=None,
    boresight_angle=None,
    hwp_angle=None,
    frame_intervals=None,
):
    """Extract Observation ancillary data.

    This extracts the shared data fields.  The input observation is one within the
    observing session, since all such observations are guaranteed to have the same
    telescope pointing and flags.

    Args:
        obs (Observation):  One observation in the session.
        book_id (str):  The book ID.
        times (str):  The field containing timestamps.
        shared_flags (str):  The field containing shared flags.
        boresight_azel (str):  The boresight Az / El quaternions.
        boresight_radec (str):  The boresight RA / DEC quaternions.
        corotator_angle (str):  The field containing the corotator angle.
        boresight_angle (str):  The field containing the boresight rotation angle.
        hwp_angle (str):  The field containing the HWP rotation angle.
        frame_intervals (str):  The name of the intervals to use for frame boundaries.
            If not specified, the observation sample sets are used.

    Returns:
        (list):  The frames.

    """
    log = toast.utils.Logger.get()
    log.verbose(f"Create ancillary frames for {book_id}")

    output = list()
    for ifrm, frmint in enumerate(obs.intervals[frame_intervals]):
        slc = slice(frmint.first, frmint.last + 1, 1)
        # Construct the Scan frame
        frame = c3g.G3Frame(c3g.G3FrameType.Scan)
        frame["book_id"] = c3g.G3String(book_id)
        frame["sample_range"] = c3g.G3VectorInt(
            np.array(
                [
                    frmint.first,
                    frmint.last + 1,
                ],
                dtype=np.int64,
            )
        )

        ancil = c3g.G3TimesampleMap()
        ancil.times = t3g.to_g3_time(obs.shared[times].data[slc])

        theta, phi, pa = qa.to_iso_angles(obs.shared[boresight_azel].data[slc, :])
        ancil["az_enc"] = c3g.G3VectorDouble(-phi)
        ancil["el_enc"] = c3g.G3VectorDouble((np.pi / 2) - theta)
        # ancil["flags"] = c3g.G3VectorUnsignedChar(obs.shared[shared_flags].data[slc])
        ancil["flags"] = c3g.G3VectorInt(
            obs.shared[shared_flags].data[slc].astype(np.int32)
        )
        frame["qboresight_azel"] = t3g.to_g3_quats(
            obs.shared[boresight_azel].data[slc, :]
        )
        if boresight_radec is not None:
            frame["qboresight_radec"] = t3g.to_g3_quats(
                obs.shared[boresight_radec].data[slc, :]
            )

        if hwp_angle is not None and hwp_angle in obs.shared:
            ancil["hwp_enc"] = c3g.G3VectorDouble(obs.shared[hwp_angle].data[slc])

        if corotator_angle is not None and corotator_angle in obs.shared:
            ancil["corotation_enc"] = c3g.G3VectorDouble(
                obs.shared[corotator_angle].data[slc]
            )

        if boresight_angle is not None and boresight_angle in obs.shared:
            ancil["boresight_enc"] = c3g.G3VectorDouble(
                obs.shared[boresight_angle].data[slc]
            )

        frame["ancil"] = ancil
        output.append(frame)

    return output


def export_obs_data(
    wafer_obs,
    full_fp,
    times,
    det_data,
    det_flags,
    frame_intervals=None,
):
    """Extract Observation detector data.

    This returns just the frame objects that are detector related.  The ancil
    frame data is merged in at a higher level.  The wafer_obs list is a subset
    of observations in the session which contain detectors at multiple frequencies
    within the same wafer.  Any readout_ids that are not included in the observations
    have their data set to zero and their flags set to invalid.

    Args:
        wafer_obs (list):  The observations spanning a single wafer.
        full_fp (SOFocalplane):  A focalplane instance containing all detectors in the
            session.
        times (str):  The field containing timestamps.
        det_data (str):  The detector data name to export.
        det_flags (str):  The detector flags name to export.
        frame_intervals (str):  The name of the intervals to use for frame boundaries.
            If not specified, the observation sample sets are used.

    Returns:
        (list):  The frames.

    """
    log = toast.utils.Logger.get()

    obs_names = [x.name for x in wafer_obs]
    log.verbose(f"Create data frames for {obs_names}")

    wafer_check = set()
    for ob in wafer_obs:
        wob = set(ob.telescope.focalplane.detector_data[:]["wafer_slot"])
        wafer_check.update(wob)
    if len(wafer_check) != 1:
        msg = f"Observation list {obs_names} contains multiple wafers ({wafer_check})"
        log.error(msg)
        raise RuntimeError(msg)

    the_wafer = wafer_check.pop()

    # Extract just the detectors in this wafer
    wafer_rows = full_fp.detector_data["wafer_slot"] == the_wafer
    wafer_data = full_fp.detector_data[wafer_rows]

    det_names = wafer_data["readout_id"]

    # For each observation, compute the indexing from the detectors into the full-wafer
    # data

    wafer_indices = list()
    full_name_to_row = {y: x for x, y in enumerate(wafer_data["name"])}
    for ob in wafer_obs:
        wafer_indices.append(
            np.array([full_name_to_row[x] for x in ob.all_detectors], dtype=np.int32)
        )

    # We are going to walk through the observations in lock step and for each frame
    # we will extract the relevant detectors.  Since all of these observations should
    # have identical frame boundaries, we get those from the first observation.

    output = list()

    for ifrm, frmint in enumerate(wafer_obs[0].intervals[frame_intervals]):
        slc = slice(frmint.first, frmint.last + 1, 1)

        # Get the timestamps from the first observation
        g3times = t3g.to_g3_time(wafer_obs[0].shared[times].data[slc])

        # Construct the Scan frame
        frame = c3g.G3Frame(c3g.G3FrameType.Scan)
        frame["stream_id"] = c3g.G3String(wafer_data["wafer_slot"][0])

        # Create timestream containers for detector data and flags.
        # FIXME:  we could eventually enable compression here, but first
        # need to decide how we want to reduce our precision and we should
        # modify unit tests to perform the same operations in memory on
        # the input data.

        dts = so3g.G3SuperTimestream()
        dts.names = det_names
        dts.times = g3times
        dts.quanta = np.ones(len(det_names))
        dts.options(enable=0)
        dts_temp = np.zeros(
            (len(det_names), frmint.last - frmint.first + 1),
            dtype=wafer_obs[0].detdata[det_data].dtype,
        )

        fts = so3g.G3SuperTimestream()
        fts.names = det_names
        fts.times = g3times
        fts.options(enable=0)
        fts_temp = np.zeros(
            (len(det_names), frmint.last - frmint.first + 1), dtype=np.int32
        )

        # Timestream units
        utype, uscale = t3g.to_g3_unit(ob.detdata[det_data].units)
        frame["signal_units"] = utype

        # Fill the data
        for iob, ob in enumerate(wafer_obs):
            for idet, det in enumerate(ob.all_detectors):
                dts_temp[wafer_indices[iob][idet], :] = ob.detdata[det_data][
                    idet, slc
                ] * uscale
                fts_temp[wafer_indices[iob][idet], :] = ob.detdata[det_flags][
                    idet, slc
                ]

        dts.data = dts_temp
        # dts.options(enable=1)
        frame["signal"] = dts
        del dts

        fts.data = fts_temp
        # fts.options(enable=1)
        frame["flags"] = fts
        del fts

        output.append(frame)
    return output


@toast.timing.function_timer
def write_book(
    obs_list,
    full_fp,
    book_dir,
    focalplane_dir,
    noise_dir,
    times,
    shared_flags,
    det_data,
    det_flags,
    boresight_azel,
    boresight_radec,
    corotator_angle,
    boresight_angle,
    hwp_angle,
    gzip=False,
    frame_intervals=None,
    redist_sampsets=False,
):
    """Write a set of observations as a book.

    An observation book is created inside the parent `book_dir`.  The observations
    passed in should be from the same observing "session".  For Simons Observatory,
    a TOAST session is mapped to a telescope tube of data for some time span.

    When building the book, observations from the same wafer at different frequencies
    are combined to write the detector data files for that wafer.  If data for one
    frequency is missing, the timestreams for those detectors is set to zero.

    To-Do:  We could add more options to dump out multiple flavors of detector
    data rather than just the defaults.

    Args:
        obs_list (list):  The list of observations.
        full_fp (SOFocalplane):  A focalplane instance containing all detectors in the
            session.
        book_dir (str):  The parent directory of the output book.
        focalplane_dir (str):  Path to directory for dumping per-wafer focalplane info.
        noise_dir (str):  Path to directory for dumping per-wafer noise models.
        times (str):  The field containing timestamps.
        shared_flags (str):  The field containing shared flags.
        det_data (str):  The detector data name to export.
        det_flags (str):  The detector flags name to export.
        boresight_azel (str):  The boresight Az / El quaternions.
        boresight_radec (str):  The boresight RA / DEC quaternions.
        corotator_angle (str):  The field containing the corotator angle.
        boresight_angle (str):  The field containing the boresight rotation angle.
        hwp_angle (str):  The field containing the HWP rotation angle.
        gzip (bool):  If True, gzip the frame files.
        frame_intervals (str):  The intervals to use for frame boundaries.
        redist_sampsets (list):  When redistributing, use these new sample sets.

    Returns:
        None.

    """
    log = toast.utils.Logger.get()
    if len(obs_list) == 0:
        # Nothing to do!
        log.warning("List of observations to write is empty, skipping")
        return

    # All the observations are on the same group
    comm = obs_list[0].comm

    # Get the current time as the "book finalization" time.
    now = None
    if comm.group_rank == 0:
        now = datetime.now(tz=timezone.utc)
    if comm.comm_group is not None:
        now = comm.comm_group.bcast(now)

    # Get the start and end time of the session
    timestamp_start = None
    timestamp_end = None
    session = obs_list[0].session
    if session.start is None or session.end is None:
        # We have to get the time range from the observations
        if comm.group_rank == 0:
            timestamp_start = obs_list[0].shared[times][0]
        if comm.group_rank == comm.group_size - 1:
            timestamp_end = obs_list[0].shared[times][-1]
    else:
        # Use the session times
        timestamp_start = session.start.timestamp()
        timestamp_end = session.end.timestamp()
    if comm.comm_group is not None:
        timestamp_start = comm.comm_group.bcast(timestamp_start, root=0)
        timestamp_end = comm.comm_group.bcast(timestamp_end, root=(comm.group_size - 1))

    log.verbose_rank(
        f"Writing book for session {session.name} with {len(obs_list)} observations",
        comm=comm.comm_group,
    )

    # Get the observation frame from the first observation of the session
    obs_frame, obs_props = export_obs_meta(obs_list[0])

    # Redistribute observations so that every process has a time slice
    for obs in obs_list:
        obs.redistribute(
            1,
            times=times,
            override_sample_sets=redist_sampsets,
        )

    # Create book directory
    book_name = book_name_from_obs(obs_list)
    book_path = os.path.join(book_dir, book_name)
    if comm.group_rank == 0:
        if not os.path.isdir(book_path):
            os.makedirs(book_path)
    if comm.comm_group is not None:
        comm.comm_group.barrier()

    # Get the ancillary frames
    ancil_frames = export_obs_ancil(
        obs_list[0],
        book_name,
        times,
        shared_flags,
        boresight_azel,
        boresight_radec,
        corotator_angle=corotator_angle,
        boresight_angle=boresight_angle,
        hwp_angle=hwp_angle,
        frame_intervals=frame_intervals,
    )

    # Collect the total sample ranges of all frames in each
    # file, for writing into the index later.
    local_sample_ranges = [
        int(ancil_frames[0]["sample_range"][0]),
        int(ancil_frames[-1]["sample_range"][1]),
    ]
    if comm.comm_group is None:
        file_sample_ranges = [local_sample_ranges]
    else:
        file_sample_ranges = comm.comm_group.gather(local_sample_ranges, root=0)

    # Write ancillary frames
    froot = os.path.join(book_path, "A_ancil")
    write_frames = [obs_frame]
    write_frames.extend(ancil_frames)
    write_frame_files(comm, write_frames, froot, gzip=gzip)

    # Split up observations into lists per wafer
    all_wafers = dict()
    for ob in obs_list:
        wf = ob.telescope.focalplane.detector_data["wafer_slot"][0]
        if wf not in all_wafers:
            all_wafers[wf] = list()
        all_wafers[wf].append(ob)

    # Export per-wafer frames
    for wf, wafer_obs in all_wafers.items():
        det_frames = export_obs_data(
            wafer_obs,
            full_fp,
            times,
            det_data,
            det_flags,
            frame_intervals=frame_intervals,
        )
        # Add the ancillary data into the detector frames
        for aframe, dframe in zip(ancil_frames, det_frames):
            dframe["book_id"] = aframe["book_id"]
            dframe["sample_range"] = aframe["sample_range"]
            dframe["ancil"] = aframe["ancil"]
            dframe["qboresight_azel"] = aframe["qboresight_azel"]
            if "qboresight_radec" in aframe:
                dframe["qboresight_radec"] = aframe["qboresight_radec"]
        # Write
        froot = os.path.join(book_path, f"D_{wf}")
        write_frames = [obs_frame]
        write_frames.extend(det_frames)
        write_frame_files(comm, write_frames, froot, gzip=gzip)

    del write_frames
    del det_frames
    del ancil_frames

    # Write out remaining metadata

    if comm.group_rank == 0:
        if not os.path.isdir(focalplane_dir):
            os.makedirs(focalplane_dir)
        if noise_dir is not None:
            if not os.path.isdir(noise_dir):
                os.makedirs(noise_dir)
    if comm.comm_group is not None:
        comm.comm_group.barrier()

    for wf, wafer_obs in all_wafers.items():
        for ob in wafer_obs:
            fp = ob.telescope.focalplane
            band = fp.detector_data["band"][0]
            fp_file = os.path.join(focalplane_dir, f"{book_name}_{wf}_{band}.h5")
            with toast.io.H5File(fp_file, "w", comm=comm.comm_group) as f:
                fp.save_hdf5(f.handle, comm=comm.comm_group)
            # Go through the observation and find any noise models
            for k in list(sorted(ob.keys())):
                v = ob[k]
                if isinstance(v, toast.noise.Noise):
                    noise_file = os.path.join(
                        noise_dir, f"{book_name}_{wf}_{band}_{k}.h5"
                    )
                    with toast.io.H5File(noise_file, "w", comm=comm.comm_group) as f:
                        v.save_hdf5(f.handle, ob)

    # Write out observation metadata as a yaml file, in addition to the observation
    # frame.  NOTE:  we copy what is done in the BookBinder class.
    if comm.group_rank == 0:
        obs_meta = dict()
        obs_meta["obs_id"] = book_name
        obs_meta["type"] = "obs"
        obs_meta["book_id"] = book_name
        obs_meta["observatory"] = "Simons Observatory"
        obs_meta["telescope"] = obs_props["telescope_name"]
        obs_meta["stream_ids"] = [str(x) for x in all_wafers.keys()]
        obs_meta["detsets"] = obs_meta["stream_ids"]
        obs_meta["start_time"] = timestamp_start
        obs_meta["stop_time"] = timestamp_end
        obs_meta["timestamp_start"] = format_book_time(
            datetime.fromtimestamp(timestamp_start).astimezone(timezone.utc)
        )
        obs_meta["timestamp_end"] = format_book_time(
            datetime.fromtimestamp(timestamp_end).astimezone(timezone.utc)
        )
        obs_meta["sample_ranges"] = file_sample_ranges
        obs_meta["n_samples"] = file_sample_ranges[-1][1]
        obs_meta["tags"] = []
        obs_meta_file = os.path.join(book_path, f"M_index.yaml")
        try:
            obs_meta_content = yaml.dump(obs_meta, default_flow_style=False)
            with open(obs_meta_file, "w") as f:
                f.write(obs_meta_content)
        except yaml.YAMLError:
            log.error("Cannot write M_index.yaml")
            raise

        # Write out toast properties
        toast_meta_file = os.path.join(book_path, f"M_toast.yaml")
        try:
            toast_meta_content = yaml.dump(obs_props, default_flow_style=False)
            with open(toast_meta_file, "w") as f:
                f.write(toast_meta_content)
        except yaml.YAMLError:
            log.error("Cannot write M_toast.yaml")
            raise

        # Write out book metadata last- so we can use the existence of this file as
        # an indicator of success
        meta = dict()
        book_meta = dict()
        book_meta["type"] = "obs"
        book_meta["book_id"] = book_name
        book_meta["schema_version"] = 0
        book_meta["finalized_at"] = format_book_time(now)
        meta["book"] = book_meta
        binder_meta = dict()
        binder_meta["codebase"] = "sotodlib.toast"
        binder_meta[
            "version"
        ] = f"sotodlib={sotodlib_version},toast={toast.__version__}"
        binder_meta["context"] = "testing"
        meta["bookbinder"] = binder_meta
        meta_file = os.path.join(book_path, f"M_book.yaml")
        try:
            meta_content = yaml.dump(meta, default_flow_style=False)
            with open(meta_file, "w") as f:
                f.write(meta_content)
        except yaml.YAMLError:
            log.error("Cannot write M_book.yaml")
            raise

    if comm.comm_group is not None:
        comm.comm_group.barrier()

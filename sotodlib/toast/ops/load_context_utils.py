# Copyright (c) 2022-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Helper functions for LoadContext operator."""

import os
import numpy as np

from astropy import units as u
from astropy.table import Column, QTable

import so3g

import toast
from toast.mpi import MPI
from toast.timing import function_timer, Timer
from toast.utils import Environment, Logger
from toast.dist import distribute_discrete, distribute_uniform

from ...core import Context, AxisManager, FlagManager
from ...core.axisman import AxisInterface
from ...preprocess import Pipeline as PreProcPipe
from ...hwp.hwp_angle_model import apply_hwp_angle_model


def open_context(context=None, context_file=None):
    if context is None:
        # The user did not specify a context- create a temporary
        # one from the file
        return Context(context_file)
    else:
        # Just return the user-specified context.
        return context


def close_context(context=None, context_file=None):
    if context_file is None:
        # We are using a pre-created context, do nothing
        return
    del context


def read_and_preprocess_wafers(
    obs_name,
    session_name,
    gcomm,
    wafer_readers,
    wafer_dets,
    pconf=None,
    context=None,
    context_file=None,
):
    """Read the wafer data.

    Each process reads zero or more wafers and applies preprocessing.

    Args:
        obs_name (str):  The observation name.
        session_name (str):  The observing session.
        gcomm (MPIComm):  The observation group communicator, or None.
        wafer_readers (dict):  For each wafer name, the group rank assigned
            to read this wafer.
        wafer_dets (dict):  For each wafer name, the list of detectors.
        pconf (dict):  The preprocessing configuration to apply (or None).
        context (Context):  The pre-existing Context or None.
        context_file (str):  The context file to open or None.

    Returns:
        (dict):  The AxisManager data for each wafer on this process (or None).

    """
    log = Logger.get()
    timer = Timer()
    timer.start()
    if gcomm is None:
        rank = 0
    else:
        rank = gcomm.rank

    results = dict()
    for wf, reader in wafer_readers.items():
        if reader == rank:
            ctx = open_context(context=context, context_file=context_file)
            axtod = ctx.get_obs(session_name, dets=wafer_dets[wf])
            close_context(ctx, context_file=context_file)
            timer.stop()
            elapsed = timer.seconds()
            timer.start()
            log.debug(
                f"LoadContext {obs_name} loaded wafer {wf} in {elapsed} seconds",
            )

            # If the axis manager has a HWP angle solution, apply it.
            if "hwp_solution" in axtod:
                axtod = apply_hwp_angle_model(axtod)
                timer.stop()
                elapsed = timer.seconds()
                timer.start()
                log.debug(
                    f"LoadContext {obs_name} HWP model wafer {wf} in {elapsed} seconds",
                )

            if pconf is not None:
                prepipe = PreProcPipe(pconf["process_pipe"], logger=log)
                prepipe.run(axtod, axtod.preprocess)
                timer.stop()
                elapsed = timer.seconds()
                timer.start()
                msg = f"LoadContext {obs_name} apply preproc to {wf}"
                msg += f" in {elapsed} seconds"
                log.debug(msg)
            results[wf] = axtod
    return results


def parse_metadata(axman, obs_meta, fp_cols, path_sep, det_axis, obs_base, fp_base):
    """Parse Context metadata.

    This is a recursive function which starts with the AxisManger returned
    by `Context.get_meta()` and splits meta data into "detector properties"
    which will be placed in the focalplane model and "other metadata" which
    will be stored in the Observation dictionary.

    For each element of the input AxisManager we do the following:

    - If the object has one axis and it is the detector axis,
      treat it as a column in the detector property table
    - If the object has one axis and it is the sample axis,
      it will be loaded later as shared data.
    - If the object has multiple axes, load it later
    - If the object has no axes, then treat it as observation
      metadata
    - If the object is a nested AxisManager, descend and apply
      the same steps as above.

    Args:
        axman (AxisManager):  The input axismanager.
        obs_meta (dict):  The working dictionary of observation metadata.
        fp_cols (dict):  Dictionary of table Columns of detector properties.
        path_sep (str):  The path separation char when building flattened names.
        det_axis (str):  The name of the detector axis.
        obs_base (str):  The current key in the nested observation metadata.
        fp_base (str):  The current key to use when appending focalplane columns.

    Returns:
        None

    """
    if obs_base is None:
        om = obs_meta
    else:
        obs_meta[obs_base] = dict()
        om = obs_meta[obs_base]
    for key in axman.keys():
        if fp_base is not None:
            fp_key = f"{fp_base}{path_sep}{key}"
        else:
            fp_key = key
        if isinstance(axman[key], AxisInterface):
            # This is one of the axes
            continue
        if isinstance(axman[key], AxisManager):
            # Descend
            parse_metadata(
                axman[key], obs_meta, fp_cols, path_sep, det_axis, key, fp_key
            )
        else:
            # FIXME:  It would be nicer if this information was available
            # through a public member...
            field_axes = axman._assignments[key]
            if len(field_axes) == 0:
                # This data is not associated with an axis.
                om[key] = axman[key]
            elif len(field_axes) == 1 and field_axes[0] == det_axis:
                # This is a detector property
                if fp_key in fp_cols:
                    msg = f"Context meta key '{fp_key}' is duplicated in nested"
                    msg += " AxisManagers"
                    raise RuntimeError(msg)
                fp_cols[fp_key] = Column(name=fp_key, data=np.array(axman[key]))
    if obs_base is not None and len(om) == 0:
        # There were no meta data keys- delete this dict
        del obs_meta[obs_base]


def distribute_detector_props(obs, wafer_key):
    """Compute the communication patterns for detector data.

    This computes the range of local detector indices for each wafer on all
    processes, and also the corresponding range of indices in the wafer data
    on the reading process.

    Args:
        obs (Observation):  The observation.
        wafer_key (str):  The column of the focalplane table with the wafer name.

    Returns:
        (tuple):  The (wafer_dets, wafer_readers, wafer_proc_dets, proc_wafer_dets)

    """
    gcomm = obs.comm.comm_group
    rank = obs.comm.group_rank
    gsize = obs.comm.group_size

    det_table = obs.telescope.focalplane.detector_data
    det_names = obs.all_detectors
    det_wafers = list(det_table[wafer_key])

    proc_wafer_dets = dict()
    wafer_proc_dets = dict()
    wafer_readers = dict()
    wafer_dets = dict()

    wafer_off = 0
    cur_wafer = det_wafers[0]
    wafer_proc_dets[cur_wafer] = dict()
    wafer_readers[cur_wafer] = 0
    wafer_dets[cur_wafer] = list()
    for proc in range(gsize):
        proc_wafer_dets[proc] = dict()
        full_indices = obs.dist.det_indices[proc]
        full_slc = slice(
            full_indices.offset, full_indices.offset + full_indices.n_elem, 1
        )
        local_det_names = det_names[full_slc]
        local_det_wafers = det_wafers[full_slc]
        local_wafers = list(sorted(set(local_det_wafers)))
        if local_det_wafers[0] != cur_wafer:
            # This process starts on a new wafer
            cur_wafer = local_det_wafers[0]
            wafer_off = 0
        # Check for wafer transitions in this process's data
        wafer_change = [
            x
            for x, (y, z) in enumerate(zip(local_det_wafers[:-1], local_det_wafers[1:]))
            if y != z
        ]
        # Add the detector ranges to the mapping dictionaries
        wfirst = 0
        for wc in wafer_change:
            wlast = wc
            nwblock = wlast - wfirst + 1
            # Extend list of detectors for this wafer
            wafer_dets[cur_wafer].extend(local_det_names[wfirst : wlast + 1])
            # Detector indices for this wafer in the local data
            proc_wafer_dets[proc][cur_wafer] = (wfirst, wlast + 1)
            # Detector indices for this process's local data within the full
            # wafer data.
            wafer_proc_dets[cur_wafer][proc] = (wafer_off, wafer_off + nwblock)
            wfirst = wlast + 1
            # We are now on a new wafer.  Reset the wafer detector offset and
            # also assign the reading to this process.
            cur_wafer = local_det_wafers[wlast + 1]
            wafer_off = 0
            wafer_dets[cur_wafer] = list()
            wafer_readers[cur_wafer] = proc
            wafer_proc_dets[cur_wafer] = dict()
        # Handle final block
        wlast = len(local_det_wafers) - 1
        nwblock = wlast - wfirst + 1
        wafer_dets[cur_wafer].extend(local_det_names[wfirst : wlast + 1])
        proc_wafer_dets[proc][cur_wafer] = (wfirst, wlast + 1)
        wafer_proc_dets[cur_wafer][proc] = (wafer_off, wafer_off + nwblock)
        wafer_off += nwblock

    return (wafer_dets, wafer_readers, wafer_proc_dets, proc_wafer_dets)


def distribute_detector_data(
    obs,
    field,
    axwafers,
    axfield,
    axdtype,
    wafer_readers,
    wafer_proc_dets,
    proc_wafer_dets,
    is_flag=False,
    flag_invert=False,
    flag_mask=None,
):
    gcomm = obs.comm.comm_group
    rank = obs.comm.group_rank
    gsize = obs.comm.group_size

    # If the blocks of detector data exceed 2^30 elements in total, they might hit
    # MPI limitations on the communication message sizes.  Work around that here.
    try:
        from mpi4py.util import pkl5

        if gcomm is not None:
            gcomm = pkl5.Intracomm(gcomm)
    except Exception:
        pass

    def _process_flag(mask, inbuf, outbuf, invert):
        temp = mask * np.ones_like(inbuf)
        if invert:
            temp[inbuf != 0] = 0
        else:
            temp[inbuf == 0] = 0
        outbuf |= temp

    # Keep a handle to our send buffers to ensure that any temporary objects
    # remain in existance until after all receives have happened.
    send_data = dict()
    send_req = list()
    recv_data = None

    # The tag value stride, to ensure unique tags for every sender / receiver / wafer
    # combination.
    tag_stride = 1000

    wf_index = {y: x for x, y in enumerate(wafer_readers.keys())}

    for wafer, reader in wafer_readers.items():
        if reader == rank:
            send_data[wafer] = dict()
            for receiver, send_dets in wafer_proc_dets[wafer].items():
                recv_dets = proc_wafer_dets[receiver][wafer]
                # Is this some detector flag data using ranges instead of samples?
                # If so, we construct a temporary buffer and build sample flags
                # from the ranges.
                n_send_det = send_dets[1] - send_dets[0]
                flat_size = n_send_det * obs.n_local_samples
                if isinstance(axwafers[wafer][axfield], so3g.proj.RangesMatrix):
                    # Yes, flagged ranges
                    sdata = np.empty(
                        flat_size,
                        dtype=np.uint8,
                    )
                    if flag_invert:
                        sdata[:] = flag_mask
                        for idet in range(n_send_det):
                            off = idet * obs.n_local_samples
                            for rg in axwafers[wafer][axfield][idet].ranges():
                                sdata[off + rg[0] : off + rg[1]] = 0
                    else:
                        sdata[:] = 0
                        for idet in range(n_send_det):
                            off = idet * obs.n_local_samples
                            for rg in axwafers[wafer][axfield][idet].ranges():
                                sdata[off + rg[0] : off + rg[1]] = flag_mask
                else:
                    # Either normal sample flags or signal data
                    sdata = np.empty(
                        flat_size,
                        dtype=axdtype,
                    )
                    sdata[:] = np.ravel(
                        axwafers[wafer][axfield][send_dets[0] : send_dets[1], :]
                    )
                if receiver == rank:
                    # We just need to process the data locally
                    if is_flag:
                        _process_flag(
                            flag_mask,
                            sdata.reshape((n_send_det, -1)),
                            obs.detdata[field][recv_dets[0] : recv_dets[1], :],
                            flag_invert,
                        )
                    else:
                        obs.detdata[field][recv_dets[0] : recv_dets[1], :] = (
                            sdata.reshape((n_send_det, -1))
                        )
                else:
                    # Send asynchronously.
                    tag = (rank * gsize + receiver) * tag_stride + wf_index[wafer]
                    req = gcomm.isend(sdata, dest=receiver, tag=tag)
                    # Save a handle to this buffer while send operation is in progress
                    send_data[wafer][axfield] = sdata
                    send_req.append(req)

    my_wafer_dets = proc_wafer_dets[rank]
    for wafer, recv_dets in my_wafer_dets.items():
        sender = wafer_readers[wafer]
        if sender == rank:
            # This data was already copied locally above
            continue
        else:
            # Receive from sender.  We always allocate a contiguous temporary buffer.
            tag = (sender * gsize + rank) * tag_stride + wf_index[wafer]
            n_recv_det = recv_dets[1] - recv_dets[0]
            recv_data = gcomm.recv(source=sender, tag=tag)
            if is_flag:
                _process_flag(
                    flag_mask,
                    recv_data.reshape((n_recv_det, -1)),
                    obs.detdata[field][recv_dets[0] : recv_dets[1], :],
                    flag_invert,
                )
            else:
                # Just assign
                obs.detdata[field][recv_dets[0] : recv_dets[1], :] = recv_data.reshape(
                    (n_recv_det, -1)
                )
            del recv_data

    # Wait for communication
    for req in send_req:
        req.wait()
    if gcomm is not None:
        gcomm.barrier()

    # Now safe to delete our dictionary of isend buffer handles, which might include
    # temporary buffers of ranges flags.
    del send_data


def compute_boresight_pointing(
    obs,
    times_key,
    az_key,
    el_key,
    roll_key,
    quat_azel_key,
    quat_radec_key,
    position_key,
    velocity_key,
    flag_key,
    flag_mask,
):
    """Compute boresight pointing in parallel.

    Use the loaded Az / El / Roll angles to compute the boresight Az / El
    quaternions and also the RA / DEC boresight quaternions.  Also compute
    the observatory position and velocity.

    Args:
        obs (Observation):  The observation to modify.
        times_key (str):  The shared times field.
        az_key (str):  The shared Azimuth field.
        el_key (str):  The shared Elevation field.
        roll_key (str):  The shared Roll field.
        quat_azel_key (str):  The shared boresight Az/El quaternion field.
        quat_radec_key (str):  The shared boresight Ra/Dec quaternion field.
        position_key (str):  The shared telescope position field.
        velocity_key (str):  The shared telescope velocity field.
        flag_key (str):  The shared flag field.
        flag_mask (int):  The shared flag mask.

    Returns:
        None

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    comm = obs.comm
    gcomm = comm.comm_group
    rank = comm.group_rank

    site = obs.telescope.site

    # Position and velocity of the observatory are simply computed.  Only the
    # first row of the process grid needs to do this.
    position = None
    velocity = None
    if rank == 0:
        position, velocity = site.position_velocity(obs.shared[times_key])
    obs.shared[position_key].set(position, offset=(0, 0), fromrank=0)
    obs.shared[velocity_key].set(velocity, offset=(0, 0), fromrank=0)
    log.debug_rank(
        f"LoadContext {obs.name} site position/velocity took",
        comm=gcomm,
        timer=timer,
    )

    # Since this step takes some time, all processes in the group
    # contribute
    pnt_dist = distribute_uniform(obs.n_all_samples, comm.group_size)

    bore_azel = None
    bore_radec = None
    bore_flags = None

    # The sample counts and displacement for each process
    sample_count = [x.n_elem for x in pnt_dist]
    sample_displ = [x.offset for x in pnt_dist]

    # The local sample range as a slice
    slc = slice(sample_displ[rank], sample_displ[rank] + sample_count[rank], 1)

    bore_bad = np.logical_or(
        np.isnan(obs.shared[az_key].data[slc]),
        np.logical_or(
            np.isnan(obs.shared[el_key].data[slc]),
            np.isnan(obs.shared[roll_key].data[slc]),
        ),
    )
    bore_flags = np.array(obs.shared[flag_key].data[slc])
    bore_flags[bore_bad] |= flag_mask
    temp_az = np.array(obs.shared[az_key].data[slc])
    temp_el = np.array(obs.shared[el_key].data[slc])
    temp_roll = np.array(obs.shared[roll_key].data[slc])
    temp_az[bore_bad] = 0
    temp_el[bore_bad] = 0
    temp_roll[bore_bad] = 0
    bore_azel = toast.qarray.from_lonlat_angles(
        -temp_az,
        temp_el,
        temp_roll,
    )
    bore_radec = toast.coordinates.azel_to_radec(
        site,
        obs.shared[times_key].data[slc],
        bore_azel,
        use_qpoint=True,
    )

    # Gather all samples to rank zero and set the shared object elements.
    ftype = None
    qtype = None
    if gcomm is not None:
        ftype = MPI.UNSIGNED_CHAR
        qtype = MPI.DOUBLE
    for name, local_data, nnz, mtype in [
        (flag_key, bore_flags, 1, ftype),
        (quat_azel_key, bore_azel, 4, qtype),
        (quat_radec_key, bore_radec, 4, qtype),
    ]:
        all_data = None
        final_data = None

        if gcomm is None:
            all_data = local_data
        else:
            counts = [x * nnz for x in sample_count]
            displ = [x * nnz for x in sample_displ]

            if rank == 0:
                # Flat buffer for the gather
                all_data = np.empty(obs.n_all_samples * nnz, dtype=local_data.dtype)
            gcomm.Gatherv(
                local_data.reshape((-1)), [all_data, counts, displ, mtype], root=0
            )

            if rank == 0:
                if nnz == 1:
                    final_data = all_data
                else:
                    final_data = all_data.reshape((-1, nnz))
        obs.shared[name].set(final_data, fromrank=0)
        del final_data
        del all_data

    del bore_flags
    del bore_azel
    del bore_radec

    log.debug_rank(
        f"LoadContext {obs.name} boresight pointing conversion took",
        comm=gcomm,
        timer=timer,
    )

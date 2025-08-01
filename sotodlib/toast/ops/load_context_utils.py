# Copyright (c) 2022-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Helper functions for LoadContext operator."""

import re

import numpy as np
from astropy.table import Column

import so3g

import toast
from toast.mpi import MPI
from toast.timing import function_timer, Timer
from toast.utils import Logger
from toast.dist import distribute_uniform
from toast.observation import default_values as defaults

from ...core import Context, AxisManager
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


@function_timer
def read_and_preprocess_wafers(
    obs_name,
    session_name,
    gcomm,
    wafer_readers,
    wafer_dets,
    preconfig=None,
    context=None,
    context_file=None,
):
    """Read the wafer data.

    Each process reads zero or more wafers and applies preprocessing.

    If the preprocessing config is specified, it should include the archive stanza,
    which will be inserted into the context dictionary.

    Args:
        obs_name (str):  The observation name.
        session_name (str):  The observing session.
        gcomm (MPIComm):  The observation group communicator, or None.
        wafer_readers (dict):  For each wafer name, the group rank assigned
            to read this wafer.
        wafer_dets (dict):  For each wafer name, the list of detectors.
        preconfig (dict):  The preprocessing configuration to apply (or None).
        context (Context):  The pre-existing Context or None.
        context_file (str):  The context file to open or None.

    Returns:
        (dict):  The AxisManager data for each wafer on this process.

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
            if preconfig is not None:
                # Ensure that the preprocessing archive is defined.  First
                # check if it already exists in the context.
                have_ctx_archive = False
                if ctx.get("metadata") is None:
                    ctx["metadata"] = []
                for key in ctx.get("metadata"):
                    if key.get("name") == "preprocess":
                        have_ctx_archive = True
                        break
                # Check if the archive exists in the preprocess config
                if "archive" in preconfig:
                    have_pre_archive = True
                else:
                    have_pre_archive = False
                # The archive should only be defined in one place...
                if have_ctx_archive and have_pre_archive:
                    msg = "Both the context file AND the preprocess config define"
                    msg += " a preprocess archive.  There can be only one."
                    raise RuntimeError(msg)
                if not have_ctx_archive:
                    if not have_pre_archive:
                        msg = "Either the context or the preprocess config must "
                        msg += "specify the preprocess archive."
                        raise RuntimeError(msg)
                    else:
                        ctx["metadata"].append(
                            {"db": preconfig["archive"]["index"], "name": "preprocess"}
                        )
            # Load the data and immediately close context
            axtod = ctx.get_obs(session_name, dets=wafer_dets[wf])
            if context_file is not None:
                del ctx
            timer.stop()
            elapsed = timer.seconds()
            timer.start()
            log.debug(
                f"LoadContext {obs_name} loaded wafer {wf} in {elapsed} seconds",
            )

            # If the axis manager has a HWP angle solution, apply it.
            if "hwp_solution" in axtod:
                # Did we already apply it in the preprocessing?
                if (
                    preconfig is None
                    or "hwp_angle_model" not in preconfig["process_pipe"]
                ):
                    axtod = apply_hwp_angle_model(axtod)
                    timer.stop()
                    elapsed = timer.seconds()
                    timer.start()
                    log.debug(
                        f"LoadContext {obs_name} HWP model wafer {wf} in {elapsed} seconds",
                    )

            if preconfig is not None:
                prepipe = PreProcPipe(preconfig["process_pipe"], logger=log)
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


def _local_process_dets(obs, rank):
    """Get the list of local detectors for a specific rank."""
    # Full detector list
    det_names = obs.all_detectors
    # The range of full detector indices on this process.
    full_indices = obs.dist.det_indices[rank]
    # The same info, as a slice.
    full_slc = slice(full_indices.offset, full_indices.offset + full_indices.n_elem, 1)
    # The names of the detectors on this process.
    return det_names[full_slc]


def distribute_detector_props(obs, wafer_key):
    """Compute the communication patterns for detector data.

    This computes the range of local detector indices for each wafer on all
    processes, and also the corresponding range of indices in the wafer data
    on the reading process.  For example, imagine we have 2 wafers and 4
    processes:

    +====================+    +====================+
    |       Wafer 0      |    |     Process 0      |
    |                    |    |--------------------|
    |                    |    |     Process 1      |
    +====================+    |                    |
    |       Wafer 1      |    |--------------------|
    |                    |    |     Process 2      |
    |                    |    |--------------------|
    |                    |    |     Process 3      |
    +====================+    +--------------------+

    In this case, Process 0 will read Wafer 0 and Process 1 will read Wafer 1.
    The data from Wafer 0 needs to be sent to Processes 0 and 1, and the data
    from Wafer 1 needs to be sent to Processes 1, 2, and 3.

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

    # Wafer names for all detectors
    det_wafers = list(det_table[wafer_key])

    # For each process, this has a key for each wafer on that process.
    # The value for each wafer key is the destination local detector indices for
    # that wafer.
    proc_wafer_dets = dict()

    # For each wafer, this has a key for each destination process.  The value
    # for each process key is the range of detector indices in the wafer data
    # that should be sent to that process.
    wafer_proc_dets = dict()

    # For each wafer, the process that should do the reading.  This is the first
    # process rank that has any data from that wafer.
    wafer_readers = dict()

    # For each wafer, the list of local detector names within that wafer.
    wafer_dets = dict()

    wafer_off = 0
    cur_wafer = det_wafers[0]
    wafer_proc_dets[cur_wafer] = dict()
    wafer_readers[cur_wafer] = 0
    wafer_dets[cur_wafer] = list()
    for proc in range(gsize):
        proc_wafer_dets[proc] = dict()
        # The range of full detector indices on this process.
        full_indices = obs.dist.det_indices[proc]
        # The same info, as a slice.
        full_slc = slice(
            full_indices.offset, full_indices.offset + full_indices.n_elem, 1
        )
        # The names of the detectors on this process.
        local_det_names = _local_process_dets(obs, proc)
        # The wafer names of the detectors on this process.
        local_det_wafers = det_wafers[full_slc]

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


@function_timer
def distribute_detector_data(
    obs,
    field,
    axwafers,
    axis_dets,
    axis_samples,
    axfield,
    axdtype,
    wafer_readers,
    wafer_proc_dets,
    proc_wafer_dets,
    path_sep,
    is_flag=False,
    flag_invert=False,
    flag_mask=None,
):
    """Communicate detector data from the reading processes to the destinations.

    Data for each wafer is loaded on exactly one process.  This function uses the
    pre-computed data communication pattern built by `distribute_detector_props()`.
    However, the loaded wafer data AxisManager may have been truncated with a
    symmetric sample buffer on either side and some detectors may have been removed.
    This function must account for this when copying data from the AxisManager into
    The full-size observation.

    The observation data is modified in-place.

    Args:
        obs (Observation):  The observation.
        field (str):  The detdata field in the Observation.
        axwafers (dict):  The dictionary of wafer data loaded on this process.
        axis_dets (str):  The name of the detector LabelAxis in the AxisManagers.
        axis_samples (str):  The name of the sample OffsetAxis in the AxisManagers.
        axfield (str):  The name of the data field in the AxisManagers.
        axdtype (np.dtype):  The dtype of the source buffer for sending the data
            from the AxisManager.
        wafer_readers (dict):  The reading process for each wafer.
        wafer_proc_dets (dict):  For each wafer, the dictionary of destination
            processes and the detector indices to send.
        proc_wafer_dets (dict):  For each process, the dictionary of wafer name to
            local detector indices to receive.
        path_sep (str):  The path separation char when building flattened names.
        is_flag (bool):  If True, this field is a flag.
        flag_invert (bool):  If True, invert the meaning of the flag values.
        flag_mask (np.uint8):  The flag mask (or None).

    Returns:
        None

    """
    log = Logger.get()
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

    def _process_flag(mask, inbuf, detdata, det_begin, det_end, invert):
        n_det = det_end - det_begin
        temp = mask * np.ones_like(inbuf)
        if invert:
            temp[inbuf != 0] = 0
        else:
            temp[inbuf == 0] = 0
        for idet in range(n_det):
            detdata[idet + det_begin, :] |= temp[idet]

    # Keep a handle to our send buffers to ensure that any temporary objects
    # remain in existance until after all receives have happened.
    send_data = dict()
    send_req = list()
    recv_data = None

    # The tag value stride, to ensure unique tags for every sender / receiver / wafer
    # combination.
    tag_stride = 2000

    wf_index = {y: x for x, y in enumerate(wafer_readers.keys())}

    for wafer, reader in wafer_readers.items():
        if reader == rank:
            send_data[wafer] = dict()
            # The wafer axis manager may have a different set of detectors than
            # the full data, if some detectors have been cut.  It may also have
            # a symmetric buffer of samples cut on either side of the observation.
            # here we construct the mapping from full data into this restricted
            # axis manager.
            restricted_dets = axwafers[wafer][axis_dets].vals
            restricted_indices = {y: x for x, y in enumerate(restricted_dets)}

            # If the field is nested, descend
            field_path = axfield.split(path_sep)
            axroot = axwafers[wafer]
            axobject = axroot[field_path[0]]
            for ch in field_path[1:]:
                axobject = axobject[ch]

            for receiver, send_dets in wafer_proc_dets[wafer].items():
                # "send_dets" is the un-restricted range of wafer detectors.
                # We will send a full-size (unrestricted) buffer to ease
                # bookkeeping, and just copy un-cut detector data into that
                # buffer.
                n_send_det = send_dets[1] - send_dets[0]

                # The names of the detectors on the receiving process.
                local_det_names = _local_process_dets(obs, receiver)

                # The receiving detector names and their relative indices
                recv_dets = proc_wafer_dets[receiver][wafer]
                n_recv_det = recv_dets[1] - recv_dets[0]
                recv_det_names = local_det_names[recv_dets[0] : recv_dets[1]]
                recv_det_indices = {y: x for x, y in enumerate(recv_det_names)}

                # Build the mapping of restricted indices to send buffer indices
                restrict_to_send = {
                    restricted_indices[x]: y
                    for x, y in recv_det_indices.items()
                    if x in restricted_indices
                }

                # The per-detector flags, so that detectors are cut properly on the
                # receiver.
                det_flags = np.ones(
                    n_send_det,
                    dtype=np.uint8,
                )
                for idet_ax, idet_send in restrict_to_send.items():
                    det_flags[idet_send] = 0

                # Does the axis manager have a truncated number of samples?
                restricted_samps = axwafers[wafer][axis_samples].count
                if restricted_samps != obs.n_local_samples:
                    ax_shift = axwafers[wafer][axis_samples].offset
                else:
                    ax_shift = 0

                # flat-packed buffer size of send buffer
                flat_size = n_send_det * obs.n_local_samples

                # Is this some detector flag data using ranges instead of samples?
                # If so, we construct a temporary buffer and build sample flags
                # from the ranges.
                if isinstance(axobject, so3g.proj.RangesMatrix):
                    # Yes, flagged ranges.  We may have restricted sample ranges
                    # for our flags, and so we initialize the full buffer to
                    # the invalid mask.
                    sdata = defaults.det_mask_invalid * np.ones(
                        flat_size,
                        dtype=np.uint8,
                    )
                    if flag_invert:
                        for idet_ax, idet_send in restrict_to_send.items():
                            # Set this detector's values to the mask, and then
                            # we will "unflag" the specified ranges.
                            off = idet_send * obs.n_local_samples + ax_shift
                            sdata[off : off + restricted_samps] = flag_mask
                            for rg in axobject[idet_ax].ranges():
                                sdata[off + rg[0] : off + rg[1]] = 0
                    else:
                        for idet_ax, idet_send in restrict_to_send.items():
                            # Set this detector's values to good, and then
                            # we will flag the specified ranges.
                            off = idet_send * obs.n_local_samples + ax_shift
                            sdata[off : off + restricted_samps] = 0
                            for rg in axobject[idet_ax].ranges():
                                sdata[off + rg[0] : off + rg[1]] = flag_mask
                else:
                    # Either normal sample flags or signal data
                    if is_flag:
                        # We may have restricted sample ranges for our flags, and so
                        # we initialize the full buffer to the invalid mask.
                        sdata = defaults.det_mask_invalid * np.ones(
                            flat_size,
                            dtype=axdtype,
                        )
                    else:
                        # Signal data is initialized to zero.  Restricted / missing
                        # samples will be indicated by flags.
                        sdata = np.zeros(
                            flat_size,
                            dtype=axdtype,
                        )
                    for idet_ax, idet_send in restrict_to_send.items():
                        off = idet_send * obs.n_local_samples + ax_shift
                        sdata[off : off + restricted_samps] = axobject[idet_ax, :]

                if receiver == rank:
                    # We just need to process the data locally
                    sdata_2d = sdata.reshape((n_send_det, -1))
                    if is_flag:
                        _process_flag(
                            flag_mask,
                            sdata_2d,
                            obs.detdata[field],
                            recv_dets[0],
                            recv_dets[1],
                            flag_invert,
                        )
                    else:
                        for idet in range(n_send_det):
                            obs.detdata[field][idet + recv_dets[0], :] = sdata_2d[idet]
                    # Update per-detector flags
                    dflags = {
                        obs.local_detectors[recv_dets[0] + x]: defaults.det_mask_invalid
                        for x in range(n_recv_det)
                        if det_flags[x] != 0
                    }
                    obs.update_local_detector_flags(dflags)
                else:
                    # Send asynchronously.
                    tag = (rank * gsize + receiver) * tag_stride + 2 * wf_index[wafer]
                    flag_tag = tag + 1
                    req = gcomm.isend(sdata, dest=receiver, tag=tag)
                    req_flags = gcomm.isend(det_flags, dest=receiver, tag=flag_tag)
                    # Save a handle to this buffer while send operation is in progress
                    send_data[wafer][axfield] = sdata
                    send_data[wafer][f"{axfield}_flags"] = det_flags
                    send_req.append(req)
                    send_req.append(req_flags)

    my_wafer_dets = proc_wafer_dets[rank]
    for wafer, recv_dets in my_wafer_dets.items():
        n_recv_det = recv_dets[1] - recv_dets[0]
        sender = wafer_readers[wafer]
        if sender == rank:
            # This data was already copied locally above
            continue
        else:
            # Receive from sender.  We always allocate a contiguous temporary buffer.
            tag = (sender * gsize + rank) * tag_stride + 2 * wf_index[wafer]
            flag_tag = tag + 1
            recv_data = gcomm.recv(source=sender, tag=tag)
            recv_2d = recv_data.reshape((n_recv_det, -1))
            det_flags = gcomm.recv(source=sender, tag=flag_tag)
            if is_flag:
                _process_flag(
                    flag_mask,
                    recv_2d,
                    obs.detdata[field],
                    recv_dets[0],
                    recv_dets[1],
                    flag_invert,
                )
            else:
                # Just assign
                for idet in range(n_recv_det):
                    obs.detdata[field][idet + recv_dets[0], :] = recv_2d[idet]
            # Update per-detector flags
            dflags = {
                obs.local_detectors[recv_dets[0] + x]: defaults.det_mask_invalid
                for x in range(n_recv_det)
                if det_flags[x] != 0
            }
            obs.update_local_detector_flags(dflags)
            del recv_2d
            del recv_data
            del det_flags

    # Wait for communication
    for req in send_req:
        req.wait()
    if gcomm is not None:
        gcomm.barrier()

    # Now safe to delete our dictionary of isend buffer handles, which might include
    # temporary buffers of ranges flags.
    del send_data

    # Every process checks its local data for NaN values.  If any are found, a warning
    # is printed and the detector is cut.
    dflags = dict()
    for det in obs.local_detectors:
        nnan = np.count_nonzero(np.isnan(obs.detdata[field][det]))
        if nnan > 0:
            msg = f"{obs.name}:{det} has {nnan} NaN values.  Cutting."
            log.warning(msg)
            dflags[det] = defaults.det_mask_invalid
    obs.update_local_detector_flags(dflags)


@function_timer
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
    if gcomm is None:
        ftype = None
        qtype = None
    else:
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
            final_data = local_data
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


def ax_name_fp_subst(var, fp_array, det=None):
    """Substitute a focalplane column value into a string.

    If det is not specified, then there should be a unique name for the whole table
    column when substituting.

    Args:
        var (str):  The input variable string
        fp_array (Table):  The focalplane astropy Table of detector properties.
        det (str):  If not None, use the value for this row of the table.

    Returns:
        (str):  The string with substitutions made.

    """
    if var is None:
        return None
    out = ""
    last = 0
    for match in re.finditer(r"(\{.*\})", var):
        out += var[last:match.start()]
        colname = match.group()
        colname = colname.replace("{", "")
        colname = colname.replace("}", "")
        if colname not in fp_array.keys():
            msg = f"Column '{colname}' not in focalplane table"
            raise RuntimeError(msg)
        if det is None:
            vals = np.unique(fp_array[colname])
            if len(vals) == 1:
                out += f"{vals[0]}"
            else:
                msg = f"Column '{colname}' has multiple values ({vals}), cannot"
                msg += " use in name"
                raise RuntimeError(msg)
        else:
            mask = fp_array[colname] == det
            out += f"{fp_array[colname][mask][0]}"
        last = match.end()
    out += var[last:]
    return out

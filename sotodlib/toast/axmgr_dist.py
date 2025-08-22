# Copyright (c) 2025-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for redistribution of AxisManager data."""

import numpy as np
from astropy import units as u
from astropy.table import Column, QTable

import so3g

import toast
from toast.mpi import MPI
from toast.timing import function_timer, Timer
from toast.utils import Logger
from toast.observation import default_values as defaults


def local_process_dets(obs, rank):
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
        local_det_names = local_process_dets(obs, proc)
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
                local_det_names = local_process_dets(obs, receiver)

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
def _load_data(self, ob, have_pointing, pconf):
    """Recursively load AxisManager data.

    The first process with detectors from a given wafer will read
    all data for that wafer.  Some examples:

    - 1 processes, 7 wafers:  rank 0 reads all wafers
    - 2 processes, 7 wafers:  rank 0 reads 3 wafers, rank 1 reads 4 wafers
    - 16 processes, 7 wafers:  ranks 0, 2, 4, 6, 9, 11, 13 read one wafer each

    Each reading process will optionally apply the preprocessing
    to the loaded data prior to distribution. The rank zero process (which
    is always a reader) extracts any shared data and additional metadata that
    is found.  Reading processes communicate slices of detector data to all
    other processes.

    Args:
        ob (Observation):  The observation to populate.
        have_pointing (bool):  True if the data contains boresight pointing.
        pconf (dict):  The optional preprocessing config.

    Returns:
        None

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    comm = ob.comm
    gcomm = comm.comm_group
    rank = ob.comm.group_rank

    # Compute the data distribution of detectors
    wafer_key = f"det_info{self.ax_pathsep}{self.ax_detinfo_wafer_key}"
    wafer_dets, wafer_readers, wafer_proc_dets, proc_wafer_dets = (
        distribute_detector_props(ob, wafer_key)
    )

    # Load data and apply preprocessing.
    axwafers = read_and_preprocess_wafers(
        ob.name,
        ob.session.name,
        gcomm,
        wafer_readers,
        wafer_dets,
        preconfig=pconf,
        context=self.context,
        context_file=self.context_file,
    )

    log.debug_rank(
        f"LoadContext {ob.name} AxisManager data loaded in",
        comm=gcomm,
        timer=timer,
    )

    # Track the fields we are extracting
    fp_array = ob.telescope.focalplane.detector_data
    ax_times = ax_name_fp_subst(self.ax_times, fp_array)
    ax_boresight_az = ax_name_fp_subst(self.ax_boresight_az, fp_array)
    ax_boresight_el = ax_name_fp_subst(self.ax_boresight_el, fp_array)
    ax_boresight_roll = ax_name_fp_subst(self.ax_boresight_roll, fp_array)
    ax_hwp_angle = ax_name_fp_subst(self.ax_hwp_angle, fp_array)
    ax_det_signal = ax_name_fp_subst(self.ax_det_signal, fp_array)
    ax_flags = list()
    for axname, bit in self.ax_flags:
        full_name = ax_name_fp_subst(axname, fp_array)
        ax_flags.append((full_name, bit))
    ax_det_flags = list()
    for axname, bit in self.ax_det_flags:
        full_name = ax_name_fp_subst(axname, fp_array)
        ax_det_flags.append((full_name, bit))

    shared_ax_to_obs = {ax_times: self.times}
    if have_pointing:
        shared_ax_to_obs[ax_boresight_az] = self.azimuth
        shared_ax_to_obs[ax_boresight_el] = self.elevation
        shared_ax_to_obs[ax_boresight_roll] = self.roll
    if self.hwp_angle is not None and ax_hwp_angle is not None:
        shared_ax_to_obs[ax_hwp_angle] = self.hwp_angle
    shared_flag_invert = {x[0]: (x[1] < 0) for x in ax_flags}
    shared_flag_fields = {x[0]: abs(x[1]) for x in ax_flags}
    det_flag_invert = {x[0]: (x[1] < 0) for x in ax_det_flags}
    det_flag_fields = {x[0]: abs(x[1]) for x in ax_det_flags}

    # The results of the recursive parsing
    extra_meta = dict()
    shared_data = dict()
    det_data = dict()
    interval_data = dict()

    # Recursively parse axis managers and build list of data to
    # populate in the observation.  Since all wafers have the same
    # ancil data (and the same field names, just with different
    # detectors), we can just do this on one process and broadcast
    # the result.
    #
    # FIXME:  In the case of multiple wafers (and readers) per
    # observation, every reader must parse every wafer and merge
    # all the metadata information.
    #
    temp_shared = None
    restricted_samps = None
    ax_shift = None
    if ob.comm.group_rank == 0:
        first_wafer = list(axwafers.keys())[0]
        self._parse_data(
            ob,
            axwafers[first_wafer],
            shared_ax_to_obs,
            shared_flag_fields,
            shared_flag_invert,
            det_flag_fields,
            det_flag_invert,
            extra_meta,
            shared_data,
            det_data,
            interval_data,
            None,
        )
        temp_shared = {x: None for x, y in shared_data.items()}

        # Does the axis manager have a truncated number of samples?
        restricted_samps = axwafers[first_wafer][self.axis_sample].count
        if restricted_samps != ob.n_local_samples:
            ax_shift = axwafers[first_wafer][self.axis_sample].offset
        else:
            ax_shift = 0

    if gcomm is not None:
        extra_meta = gcomm.bcast(extra_meta, root=0)
        temp_shared = gcomm.bcast(temp_shared, root=0)
        if ob.comm.group_rank != 0:
            shared_data = temp_shared
        det_data = gcomm.bcast(det_data, root=0)
        interval_data = gcomm.bcast(interval_data, root=0)
        restricted_samps = gcomm.bcast(restricted_samps, root=0)
        ax_shift = gcomm.bcast(ax_shift, root=0)

    # Add extra metadata that was discovered.
    ob.update(extra_meta)

    log.debug_rank(
        f"LoadContext {ob.name} AxisManager field parsing took",
        comm=gcomm,
        timer=timer,
    )

    # Create the intervals
    for intr_name, sampspans in interval_data.items():
        ob.intervals[intr_name] = toast.intervals.IntervalList(
            ob.shared[self.times], samplespans=sampspans
        )

    # Collectively store shared data.  All readers have a full copy of
    # this data, but we only set this from rank zero.  If there are
    # cut samples at the beginning and end, ensure that timestamps are
    # always valid.
    for shr_obs_name, shrbuf in shared_data.items():
        bf = shrbuf
        if shr_obs_name == self.times and restricted_samps != ob.n_local_samples:
            if rank == 0:
                msg = f"Axman samples {restricted_samps} != {ob.n_local_samples}"
                msg += ", extrapolating timestamps"
                log.debug(msg)
                bf = np.zeros(ob.n_local_samples, dtype=np.float64)
                (rate, dt, _, _, _) = toast.utils.rate_from_times(shrbuf)
                bf[ax_shift:ax_shift+restricted_samps] = shrbuf
                bf[0:ax_shift] = shrbuf[0] + dt * np.arange(
                    -ax_shift, 0, 1, dtype=np.float64
                )
                end_gap = ob.n_local_samples - restricted_samps - ax_shift
                bf[ax_shift + restricted_samps:] = shrbuf[-1] + dt * np.arange(
                    1, end_gap + 1, 1, dtype=np.float64
                )
        ob.shared[shr_obs_name].set(bf, fromrank=0)

    log.debug_rank(
        f"LoadContext {ob.name} Shared data copy took",
        comm=gcomm,
        timer=timer,
    )

    # Now that we have timestamps loaded, update our focalplane sample rate
    # and observing session times.
    (rate, dt, dt_min, dt_max, dt_std) = toast.utils.rate_from_times(
        ob.shared[self.times].data
    )
    ob.telescope.focalplane.sample_rate = rate * u.Hz

    ob.session.start = datetime.fromtimestamp(
        ob.shared[self.times].data[0]
    ).astimezone(timezone.utc)
    ob.session.end = datetime.fromtimestamp(
        ob.shared[self.times].data[-1]
    ).astimezone(timezone.utc)

    log.debug_rank(
        f"LoadContext {ob.name} sample rate calculation took",
        comm=gcomm,
        timer=timer,
    )

    # Distribute detector data
    for field, merge_list in det_data.items():
        for ax_field, ax_dtype, mask in merge_list:
            do_invert = False
            if ax_field in det_flag_invert:
                do_invert = det_flag_invert[ax_field]
            distribute_detector_data(
                ob,
                field,
                axwafers,
                self.axis_detector,
                self.axis_sample,
                ax_field,
                ax_dtype,
                wafer_readers,
                wafer_proc_dets,
                proc_wafer_dets,
                str(self.ax_pathsep),
                is_flag=(mask is not None),
                flag_invert=do_invert,
                flag_mask=mask,
            )

    # Original wafer data no longer needed.  AxisManager does not seem to
    # have a clean destructor, so do it manually.
    def _ax_del_children(ax):
        for k in list(ax.keys()):
            if isinstance(ax[k], AxisManager):
                _ax_del_children(ax[k])
            del ax[k]
    for wfname in list(axwafers.keys()):
        _ax_del_children(axwafers[wfname])
        del axwafers[wfname]
    del axwafers

    log.debug_rank(
        f"LoadContext {ob.name} Detector data distribution took",
        comm=gcomm,
        timer=timer,
    )

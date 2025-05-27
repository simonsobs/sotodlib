# Copyright (c) 2023-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Template regression mapmaking.
"""
import os

import numpy as np
from astropy import units as u
import toast
from toast.mpi import flatten
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops
from .job import workflow_timer
from .proc_noise_est import select_mapmaking_noise_model


def setup_splits(operators):
    """Add commandline args and operators for SAT mapmaking splits.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    operators.append(so_ops.Splits(name="splits", enabled=False))


@workflow_timer
def splits(job, otherargs, runargs, data):
    """Apply mapmaking splits.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    job_ops = job.operators
    splits = job.operators.splits

    if splits.enabled:
        mapmaker_select_noise_and_binner(job, otherargs, runargs, data)
        if job_ops.mapmaker.enabled:
            splits.mapmaker = job_ops.mapmaker
        elif job_ops.filterbin.enabled:
            splits.mapmaker = job_ops.filterbin
        else:
            msg = "No mapmaker is enabled!"
            raise RuntimeError(msg)
        splits.output_dir = splits.mapmaker.output_dir
        mapmaker_run(job, otherargs, runargs, data, splits)


def setup_mapmaker(operators, templates):
    """Add commandline args, operators, and templates for TOAST mapmaker.

    Args:
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    templates.append(
        toast.templates.Offset(
            name="baselines",
            step_time=2.0 * u.second,
            enabled=False,
        )
    )
    templates.append(
        toast.templates.Periodic(
            name="azss",
            key=defaults.azimuth,
            flags=defaults.shared_flags,
            flag_mask=defaults.shared_mask_invalid,
            increment=np.pi / 180.0,  # One degree, az field is in radians
            bins=None,
            enabled=False,
        )
    )
    templates.append(
        toast.templates.Hwpss(
            name="hwpss",
            hwp_angle=defaults.hwp_angle,
            harmonics=5,
            enabled=False,
        )
    )
    templates.append(
        toast.templates.Fourier2D(
            name="fourier2d",
            correlation_length=5.0 * u.second,
            correlation_amplitude=10.0,
            order=1,
            fit_subharmonics=False,
        )
    )
    operators.append(toast.ops.BinMap(name="binner", pixel_dist="pix_dist"))
    operators.append(
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        )
    )
    operators.append(toast.ops.MapMaker(name="mapmaker", det_data=defaults.det_data))


def mapmaker_select_noise_and_binner(job, otherargs, runargs, data):
    """Helper function to setup noise model and binner for mapmapking.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()

    # Configured templates for this job
    job_tmpls = job.templates

    # Configured operators for this job
    job_ops = job.operators

    if job_ops.mapmaker.enabled:
        job_ops.mapmaker.binning = job_ops.binner
        job_ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
            templates=[job_tmpls.baselines, job_tmpls.azss, job_tmpls.hwpss]
        )
        job_ops.mapmaker.map_binning = job_ops.binner_final
        job_ops.mapmaker.output_dir = otherargs.out_dir
        tmsg = "  "
        for tmpl in job_ops.mapmaker.template_matrix.templates:
            estr = "(enabled)" if tmpl.enabled else "(disabled)"
            tmsg += f"{tmpl.name} {estr}, "
        log.info_rank(
            "  Using regression templates:",
            comm=data.comm.comm_world,
        )
        log.info_rank(
            tmsg,
            comm=data.comm.comm_world,
        )
        # Noise model.  If noise estimation is not enabled, and no existing noise model
        # is found, then create a fake noise model with uniform weighting.
        noise_model = select_mapmaking_noise_model(job, otherargs, runargs, data)
        if noise_model is None:
            for ob in data.obs:
                (estrate, _, _, _, _) = toast.utils.rate_from_times(
                    ob.shared[defaults.times].data
                )
                ob["fake_noise"] = toast.noise_sim.AnalyticNoise(
                    detectors=ob.all_detectors,
                    rate={x: estrate * u.Hz for x in ob.all_detectors},
                    fmin={x: 1.0e-5 * u.Hz for x in ob.all_detectors},
                    fknee={x: 0.0 * u.Hz for x in ob.all_detectors},
                    alpha={x: 1.0 for x in ob.all_detectors},
                    NET={
                        x: 1.0 * u.K * np.sqrt(1.0 * u.second)
                        for x in ob.all_detectors
                    },
                )
            log.info_rank(
                "  Using fake noise model with uniform weighting",
                comm=data.comm.comm_world,
            )
            noise_model = "fake_noise"
        job_ops.binner.noise_model = noise_model
        job_ops.binner_final.noise_model = noise_model

        if job_tmpls.baselines.enabled:
            job_tmpls.noise_model = noise_model


@workflow_timer
def mapmaker_run(job, otherargs, runargs, data, map_op):
    """Run a mapmaker, optionally per observation.

    This runs the mapmaker either in single shot or per
    detector/observation.  Currently this supports instances of the
    `Mapmaker` and `Splits` operators.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.
        map_op (Operator):  The operator to run.

    Returns:
        None

    """
    log = toast.utils.Logger.get()

    if map_op.enabled:
        do_obsmaps = hasattr(otherargs, "obsmaps") and otherargs.obsmaps
        do_detmaps = hasattr(otherargs, "detmaps") and otherargs.detmaps
        do_intervalmaps = (
            hasattr(otherargs, "intervalmaps") and otherargs.intervalmaps
        )
        # See if user wants separate detector maps
        if do_detmaps:
            my_dets = data.all_local_detectors(flagmask=defaults.det_mask_invalid)
            if data.comm.comm_world is None:
                all_dets = my_dets
            else:
                all_dets = data.comm.comm_world.allgather(my_dets)
                all_dets = sorted(set(flatten(all_dets)))
        else:
            all_dets = [None]
        if do_obsmaps and do_intervalmaps:
            log.warning_rank(
                "--intervalmaps overrides --obsmaps", data.comm.comm_world
            )
        if do_obsmaps or do_intervalmaps:
            log.debug_rank(
                f"{data.comm.group}: Running observation or interval maps",
                comm=data.comm.comm_world,
            )

        mapmaker_name = map_op.name
        for det in all_dets:
            if det is None:
                # Map all detectors together
                detectors = None
            else:
                # Single detector mode, append detector name to all
                # data products
                map_op.name = f"{mapmaker_name}_{det}"
                detectors = [det]
            if do_obsmaps or do_intervalmaps:
                # Map each observation separately
                timer_obs = toast.timing.Timer()
                timer_obs.start()
                group = data.comm.group
                orig_name = map_op.name
                orig_outdir = map_op.output_dir
                orig_comm = data.comm
                new_comm = toast.Comm(world=data.comm.comm_group)
                for iobs, obs in enumerate(data.obs):
                    log.info_rank(
                        f"{group} : mapping observation {iobs + 1} "
                        f"/ {len(data.obs)}.",
                        comm=new_comm.comm_world,
                    )
                    # Set the observation name used in output file names.
                    # This is the base observation name without any "demod_"
                    # string.
                    obs_str = obs.name.replace("demod_", "")
                    # Data object that only covers one observation
                    obs_data = data.select(obs_uid=obs.uid)
                    # Set the output directory to the session name
                    map_op.output_dir = os.path.join(orig_outdir, obs.session.name)

                    # Delete any solver or other products
                    import re
                    for k in list(obs_data.keys()):
                        if re.match(r".*_solve_.*", k) is not None:
                            del obs_data[k]

                    # Replace comm_world with the group communicator
                    obs_data._comm = new_comm
                    if isinstance(map_op, so_ops.Splits):
                        binner = map_op.mapmaker.binning
                    else:
                        binner = map_op.binning

                    if binner.pixel_pointing.pixels in obs:
                        del obs[binner.pixel_pointing.pixels]
                    if binner.stokes_weights.weights in obs:
                        del obs[binner.stokes_weights.weights]
                    if binner.pixel_pointing.detector_pointing.quats in obs:
                        del obs[binner.pixel_pointing.detector_pointing.quats]

                    orig_view = binner.pixel_pointing.view
                    if do_intervalmaps and orig_view is not None:
                        if isinstance(map_op, so_ops.Splits):
                            msg = "Interval mapping cannot be used with Splits"
                            raise RuntimeError(msg)
                        # Map each interval separately
                        ob = obs_data.obs[0]
                        times = ob.shared[defaults.times].data
                        views = ob.intervals[orig_view]
                        for iview, view in enumerate(views):
                            # Add a view for this specific interval
                            single_view = f"{orig_view}-{iview}"
                            ob.intervals[single_view] = toast.IntervalList(
                                times, timespans=[(view.start, view.stop)]
                            )
                            binner.pixel_pointing.view = single_view
                            map_op.name = f"{orig_name}_{obs_str}-{iview}"
                            map_op.reset_pix_dist = True
                            try:
                                map_op.apply(obs_data, detectors=detectors)
                                log.info_rank(
                                    f"{group} : Mapped det={det} "
                                    f"{obs_str}-{iview} / {len(views)} in",
                                    comm=new_comm.comm_world,
                                    timer=timer_obs,
                                )
                            except Exception as e:
                                log.info_rank(
                                    f"{group} : Failed to map "
                                    f"{obs_str}-{iview} / {len(views)} (e) in",
                                    comm=new_comm.comm_world,
                                    timer=timer_obs,
                                )
                        binner.pixel_pointing.view = orig_view
                    else:
                        # Map the observation as a whole
                        # Rename the operator with the observation suffix
                        map_op.name = f"{orig_name}_{obs_str}"
                        if isinstance(map_op, so_ops.Splits):
                            # Reset the pixel distribution of the underlying
                            # mapmaker
                            map_op.mapmaker.reset_pix_dist = True
                        else:
                            # Reset the trait on this mapmaker
                            map_op.reset_pix_dist = True

                    # Map this observation
                    map_op.apply(obs_data, detectors=detectors)

                    log.info_rank(
                        f"{group} : Mapped det={det} obs={obs_str} in",
                        comm=new_comm.comm_world,
                        timer=timer_obs,
                    )
                log.info_rank(
                    f"{group} : Done mapping {len(data.obs)} observations.",
                    comm=new_comm.comm_world,
                )
                map_op.name = orig_name
                map_op.output_dir = orig_outdir
                data._comm = orig_comm
                del new_comm
            else:
                log.debug_rank(
                    f"{data.comm.group}: Calling mapmaker.apply() directly, "
                    f"det={det}",
                    comm=data.comm.comm_world,
                )
                map_op.apply(data, detectors=detectors)


@workflow_timer
def mapmaker(job, otherargs, runargs, data):
    """Run the TOAST mapmaker.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    log = toast.utils.Logger.get()

    # Configured templates for this job
    job_tmpls = job.templates

    # Configured operators for this job
    job_ops = job.operators

    mapmaker_select_noise_and_binner(job, otherargs, runargs, data)
    mapmaker_run(job, otherargs, runargs, data, job_ops.mapmaker)

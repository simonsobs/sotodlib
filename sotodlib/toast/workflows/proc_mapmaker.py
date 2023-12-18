# Copyright (c) 2023-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Template regression mapmaking.
"""

import numpy as np
from astropy import units as u
import toast
import toast.ops
from toast.observation import default_values as defaults

from .. import ops as so_ops
from .job import workflow_timer


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
    operators.append(toast.ops.BinMap(name="binner", pixel_dist="pix_dist"))
    operators.append(
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        )
    )
    operators.append(toast.ops.MapMaker(name="mapmaker", det_data=defaults.det_data))


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

    if job_ops.mapmaker.enabled:
        job_ops.mapmaker.binning = job_ops.binner
        job_ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
            templates=[job_tmpls.baselines, job_tmpls.azss]
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
        noise_model = None
        if job_ops.demodulate.enabled:
            # We will use the noise estimate made after demodulation
            log.info_rank("  Using demodulated noise model", comm=data.comm.comm_world)
            noise_model = job_ops.demod_noise_estim_fit.out_model
        elif job_ops.noise_estim.enabled and job_ops.noise_estim_fit.enabled:
            # We have a noise estimate
            log.info_rank("  Using estimated noise model", comm=data.comm.comm_world)
            noise_model = job_ops.noise_estim_fit.out_model
        else:
            have_noise = True
            for ob in data.obs:
                if "noise_model" not in ob:
                    have_noise = False
            if have_noise:
                log.info_rank(
                    "  Using noise model from data files", comm=data.comm.comm_world
                )
                noise_model = "noise_model"
            else:
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

        if otherargs.obsmaps:
            # Map each observation separately
            timer_obs = toast.timing.Timer()
            timer_obs.start()
            group = data.comm.group
            orig_name = job_ops.mapmaker.name
            orig_comm = data.comm
            new_comm = toast.Comm(world=data.comm.comm_group)
            for iobs, obs in enumerate(data.obs):
                log.info_rank(
                    f"{group} : mapping observation {iobs + 1} / {len(data.obs)}.",
                    comm=new_comm.comm_world,
                )
                # Data object that only covers one observation
                obs_data = data.select(obs_uid=obs.uid)
                # Replace comm_world with the group communicator
                obs_data._comm = new_comm
                job_ops.mapmaker.name = f"{orig_name}_{obs.name}"
                job_ops.mapmaker.reset_pix_dist = True
                job_ops.mapmaker.apply(obs_data)
                log.info_rank(
                    f"{group} : Mapped {obs.name} in",
                    comm=new_comm.comm_world,
                    timer=timer_obs,
                )
            log.info_rank(
                f"{group} : Done mapping {len(data.obs)} observations.",
                comm=new_comm.comm_world,
            )
            data._comm = orig_comm
        else:
            job_ops.mapmaker.apply(data)

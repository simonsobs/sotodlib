# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np

from toast.timing import function_timer, Timer
from toast.tod import OpSimPySM
from toast.utils import Logger

try:
    import pysm
except:
    pysm = None

try:
    import so_pysm_models
except Exception as e:
    print('Failed to load so_pysm_models: "{}"'.format(e))
    so_pysm_models = None


def add_pysm_args(parser):
    parser.add_argument(
        "--pysm-model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. s3,d6,f1,a2" '
        "this pipeline also supports the SO specific models from "
        "so_pysm_models, see https://github.com/simonsobs/so_pysm_models "
        "currently the most complete PySM model for simulations is:"
        '"SO_d0,SO_s0,SO_a0,SO_f0,SO_x1_cib,SO_x1_tsz,SO_x1_ksz,SO_x1_cmb_lensed_solardipole"',
    )
    parser.add_argument(
        "--pysm-apply-beam",
        required=False,
        action="store_true",
        help="Convolve sky with detector beam",
        dest="pysm_apply_beam",
    )
    parser.add_argument(
        "--no-pysm-apply-beam",
        required=False,
        action="store_false",
        help="Do not convolve sky with detector beam.",
        dest="pysm_apply_beam",
    )
    parser.set_defaults(pysm_apply_beam=True)
    return


@function_timer
def simulate_sky_signal(args, comm, data, focalplanes, subnpix, localsm, signalname=None):
    """ Use PySM to simulate smoothed sky signal.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()
    # Convolve a signal TOD from PySM
    if comm.world_rank == 0:
        log.info("Simulating sky signal with PySM")

    map_dist = (
        None
        if comm is None
        else pysm.MapDistribution(nside=args.nside, mpi_comm=comm.comm_rank)
    )
    pysm_component_objects = []
    pysm_model = []
    for model_tag in args.pysm_model.split(","):

        if not model_tag.startswith("SO"):
            pysm_model.append(model_tag)
        else:
            if so_pysm_models is None:
                raise RuntimeError(
                    "{} requires so_pysm_models".format(model_tag))
            if model_tag == "SO_x1_cib":
                pysm_component_objects.append(
                    so_pysm_models.WebSkyCIB(
                        websky_version="0.3",
                        interpolation_kind="linear",
                        nside=args.nside,
                        map_dist=map_dist,
                    )
                )
            elif model_tag == "SO_x1_ksz":
                pysm_component_objects.append(
                    so_pysm_models.WebSkySZ(
                        version="0.3",
                        nside=args.nside,
                        map_dist=map_dist,
                        sz_type="kinetic",
                    )
                )
            elif model_tag == "SO_x1_tsz":
                pysm_component_objects.append(
                    so_pysm_models.WebSkySZ(
                        version="0.3",
                        nside=args.nside,
                        map_dist=map_dist,
                        sz_type="thermal",
                    )
                )
            elif model_tag.startswith("SO_x1_cmb"):
                lensed = "unlensed" not in model_tag
                include_solar_dipole = "solar" in model_tag
                pysm_component_objects.append(
                    so_pysm_models.WebSkyCMBMap(
                        websky_version="0.3",
                        lensed=lensed,
                        include_solar_dipole=include_solar_dipole,
                        seed=1,
                        nside=args.nside,
                        map_dist=map_dist,
                    )
                )
            else:
                pysm_component_objects.append(
                    so_pysm_models.get_so_models(
                        model_tag, args.nside, map_dist=map_dist
                    )
                )

    if signalname is None:
        signalname = "pysmsignal"
    op_sim_pysm = OpSimPySM(
        comm=comm.comm_rank,
        out=signalname,
        pysm_model=pysm_model,
        pysm_component_objects=pysm_component_objects,
        focalplanes=focalplanes,
        nside=args.nside,
        subnpix=subnpix,
        localsm=localsm,
        apply_beam=args.pysm_apply_beam,
        coord="G",  # setting G doesn't perform any rotation
        map_dist=map_dist,
    )
    assert args.coord in "CQ", "Input SO models are always in Equatorial coordinates"
    op_sim_pysm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("PySM")

    return signalname

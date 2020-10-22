# Copyright (c) 2018-2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST data loading.

This module contains code for loading data from disk into memory.

"""
import os
import sys
import re

import traceback

import numpy as np

import traitlets

import numpy as np

from scipy.constants import degree

import healpy as hp

from astropy import units as u

from .. import qarray as qa

from ..utils import Environment, name_UID, Logger, rate_from_times

from ..dist import distribute_uniform

from ..timing import function_timer, Timer

from ..intervals import Interval, regular_intervals

import toast.traits as trts

from toast import Operator, Observation

from toast.instrument import Telescope

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g

from toast.dist import Data, distribute_discrete

import toast.qarray as qa
from toast.tod import TOD, Noise, Interval
from toast.tod import spt3g_utils as s3utils
from toast.tod.interval import Interval
from toast.todmap import TODGround
from toast.utils import Logger, Environment, memreport, rate_from_times

from toast.timing import Timer

from toast.mpi import MPI

from ..core import Hardware

from .frame_utils import frame_to_tod


class SOTelescope(Telescope):
    def __init__(self, name, focalplane):
        site = Site("Atacama", lat="-22.958064", lon="-67.786222", alt=5200)
        super().__init__(name, site=site)
        self.id = {
            # Use the same telescope index for all SATs to re-use the
            # atmospheric simulation
            #'LAT' : 0, 'SAT0' : 1, 'SAT1' : 2, 'SAT2' : 3, 'SAT3' : 4
            "LAT": 0,
            "SAT0": 4,
            "SAT1": 4,
            "SAT2": 4,
            "SAT3": 4,
        }[name]


def select_obs_and_dets(
    context,
    obs_ids=None,
    obs_queries=None,
    obs_regex=None,
    detsets=None,
    dets=None,
    dets_regex=None,
    union=False,
):
    """Given a context, select observation IDs and detectors.

    This takes all the selection criteria and computes either the intersection or the
    union of the results.

    Args:
        obs_ids (list):  List of explicit observation IDs.
        obs_queries (list):  List of query strings for observations.
        obs_regex (str):  A regex string to apply to observation IDs for selection.
        detsets (list):  An explicit list of detector sets.
        dets (list):  An explicit list of detectors.
        dets_regex (str):  A regex string to apply to the detector names.
        union (bool):  If True, take the union of results, not the intersection.

    Returns:
        (tuple):  The list of observation IDs and detectors to use.

    """
    final_obs = list()
    final_dets = list()

    return final_obs, final_dets


def load_data(context, obs_ids, dets, comm=toast.Comm()):
    """Load specified observations and detectors into memory.

    The full observation and detector lists should be pre-selected (see for example
    the `select_obs_and_dets()` function).  This function uses the provided context
    to load each observation on one process and broadcast to the group of processes
    assigned to that observation.  Since the loading involves reading and concatenating
    frame files, it is best to do this once and communicate the result.

    NOTE:  currently the context exists on all processes (including loading the
    yaml file and opening databases).  This will likely not scale and we should
    refactor the Context class to handle this scenario.

    Args:
        context (Context):  The context to use.
        obs_ids (list):  The list of observation IDs to load.
        dets (list):  The list of detectors to load from each observation.
        comm (toast.Comm):  The toast communicator.

    Returns:
        (toast.Data):  The distributed toast Data object.

    """
    log = Logger.get()

    # the global communicator
    cworld = comm.comm_world
    # the communicator within the group
    cgroup = comm.comm_group

    # Normally, here is where we would (on one process) query the size of all
    # observations and distribute them among the process groups.  Unfortunately
    # Context.obsfiledb.get_files() does not reliably return a sample range for each
    # observation, so we cannot get the relative sizes of the observations.  For now,
    # just distribute them with equal weight.

    # One process gets the list of observation directories
    obslist = obs_ids
    # weight = dict()
    weight = {x: 1.0 for x in obslist}

    # worldrank = 0
    # if cworld is not None:
    #     worldrank = cworld.rank
    #
    # if worldrank == 0:
    #     # Get the weights...
    #
    # if cworld is not None:
    #     obslist = cworld.bcast(obslist, root=0)
    #     weight = cworld.bcast(weight, root=0)

    # Distribute observations based on approximate size
    dweight = [weight[x] for x in obslist]
    distobs = distribute_discrete(dweight, comm.ngroups)

    # Distributed data
    data = Data(comm=comm)

    # Now every group loads its observations

    firstobs = distobs[comm.group][0]
    nobs = distobs[comm.group][1]
    for ob in range(firstobs, firstobs + nobs):
        telematch = re.match(r"CES-Atacama-(\w+)-.*", obslist[ob])
        if telematch is None:
            msg = "Cannot extract telescope name from {}".format(obslist[ob])
            raise RuntimeError(msg)
        telename = telematch.group(1)

        axmgr = None
        samples = None
        focal_plane = None
        if comm.group_rank == 0:
            # Load the data
            try:
                axmgr = context.get_obs(obs_id=obslist[ob], dets=dets)
                # Number of samples
                samples = axmgr.samps.count
                # Effective sample rate
                sample_rate = rate_from_times(axmgr.timestamps)
                # Create a Focalplane and Telescope and extract other metadata
                dets = axmgr.dets.vals
                quats = axmgr.focal_plane.quat
                det_quat = {d: q for d, q in zip(dets, quats)}
                focal_plane = Focalplane(
                    detector_data=det_quat, sample_rate=sample_rate
                )
                site = Site()
                tele = Telescope

            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                lines = ["Proc {}: {}".format(worldrank, x) for x in lines]
                print("".join(lines), flush=True)
                if cworld is not None:
                    cworld.Abort()

        # Broadcast meta

        # Create the observation.
        telescope = SOTelescope(telename)

        obs = Observation(telescope, name=obslist[ob], samples=samples)

        # Create data members.

        # Add to the data object
        data.obs.append(obs)

    return data

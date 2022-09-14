"""Utility functions for interacting with level 2 data and g3tsmurf
"""

import os
import numpy as np
import logging
import datetime as dt

from sotodlib.io.load_smurf import load_file, SmurfStatus, make_datetime
from sotodlib.io.g3tsmurf_db import Observations, Files


logger = logging.getLogger(__name__)



def get_obs_folder(obs_id, archive):
    """
    Get the folder associated with the observation action. Assumes
    everything is following the standard suprsync formatting.
    """
    session = archive.Session()
    obs = session.query(Observations).filter(Observations.obs_id == obs_id).one()
    session.close()
    
    return os.path.join(
        archive.meta_path, 
        str(obs.action_ctime)[:5], 
        obs.stream_id, 
        str(obs.action_ctime)+"_"+obs.action_name,
    )
    
def get_obs_outputs(obs_id, archive):
    """
    Get the output files associated with the observation action. Assumes
    everything is following the standard suprsync formatting. Returns 
    absolute paths since everything in g3tsmurf is absolute paths.
    """
    path = os.path.join(
        get_obs_folder(obs_id,archive),
        "outputs"
    )
    
    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist, how does {obs.obs_id} exist?")
    return [os.path.join(path,f) for f in os.listdir(path)]


def get_obs_plots(obs_id, archive):
    """
    Get the output plots associated with the observation action. Assumes
    everything is following the standard suprsync formatting. Returns 
    absolute paths since everything in g3tsmurf is absolute paths.
    """
    path = os.path.join(
        get_obs_folder(obs_id,archive),
        "plots"
    )
    
    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist, how does {obs.obs_id} exist?")
    return [os.path.join(path,f) for f in os.listdir(path)]

def get_batch( 
    obs_id, 
    archive,
    ram_limit=None, 
    n_det_chunks=None,
    n_samp_chunks=None, 
    n_dets=None, 
    n_samps=None, 
    det_chunks=None, 
    samp_chunks=None,
    start = None,
    end = None,
    startend_buffer = 10,
    test = False,
    load_file_args={},
):
    """A Generator to loop through and load AxisManagers of sections of
    Observations. When run with none of the optional arguments it will default
    to returning the full observation. Some arguments over-write others as
    described in the docstrings below. When splitting the Observations by both
    detectors and samples, the chunks of samples for the same detectors are
    looped through first (samples is the inner for loop).

    Example usage::

        for aman in get_batch(obs_id, archive, ram_limit=2e9):
            run_analysis_pipeline(aman)

        for aman in get_batch(obs_id, archive, n_dets=200):
            run_analysis_pipeline(aman)
        
    
    Arguments
    ----------
    obs_id : string, or Observatin object
        Level 2 observation IDs
    archive : G3tSmurf Instance 
        The G3tSmurf database connected to the obs_id
    ram_limit : None or float
        A (very simplistically calculated) limit on RAM per AxisManager. If 
        specified it overrides all other inputs for how the AxisManager is split.
    n_det_chunks : None or int
        number of chunks of detectors to split the observation by. Each
        AxisManager will have N_det = N_obs_det / n_det_chunks. If specified, 
        it overrides n_dets and det_chunks arguments.
    n_samp_chunks: None or int
        number of chunks of samples to split the observation by. Each AxisManager
        will have N_samps = N_obs_samps / n_samps_chunks. If specified, it overrides 
        n_samps and samp_chunks arguments.
    n_dets : None or int
        number of detectors to load per AxisManager. If specified, it overrides the
        det_chunks argument.
    n_samps : None or int
        number of samples to load per AxisManager. If specified it overrides the 
        samps_chunks arguments.
    det_chunks: None or list of lists, tuples, or ranges
        if specified, each entry in the list is successively passed to load the 
        AxisManagers as  `load_file(... channels = list[i] ... )`
    samp_chunks: None or list of tuples
        if specified, each entry in the list is successively passed to load the
        AxisManagers as  `load_file(... samples = list[i] ... )`
    start: Datetime or string. Begin batching at this time.
    end: Datetime or string.  End betching at this time.
    startend_buffer: int.  Seconds of buffer to add on either side samples estimated
        to correspond with start, end based on sampling rate.
    test: bool
        If true, yields a tuple of (det_chunks, samp_chunks) instead of a loaded
        AxisManager
    load_file_kwargs: dict
        additional arguments to pass to load_smurf
    
    Yields
    --------
    AxisManagers with loaded sections of data
    """
    
    session = archive.Session()

    if isinstance(obs_id, Observations):
        obs = obs_id
        obs_id = obs.obs_id
    else:
        obs = session.query(Observations).filter(Observations.obs_id==obs_id).one()
    db_files = session.query(Files).filter(Files.obs_id==obs_id).order_by(Files.start)
    filenames = sorted( [f.name for f in db_files])
    
    ts = obs.tunesets[0] ## if this throws an error we have some fallbacks
    obs_dets, obs_samps = len(ts.dets), obs.n_samples
    session.close()

    ## should we let folks overwrite this here?
    if "archive" in load_file_args:
        archive = load_file_args.pop("archive")
        
    if "status" in load_file_args:
        status = load_file_args.pop("status")
    else:
        status = SmurfStatus.from_file(filenames[0])

    samp_s, samp_e = 0, obs_samps
    partial = False
    if start != None or end != None:
         partial = True
          samprate = 4e3 / (status.downsample_enabled * status.downsample_factor)
           if start != None:
                samp_s = (make_datetime(start).timestamp() - obs.start.timestamp()) * samprate \
                    - startend_buffer * samprate
                samp_s = max(int(samp_s), 0)
            if end != None:
                samp_e = obs_samps - (obs.stop.timestamp() - make_datetime(end).timestamp()) * samprate \
                    + startend_buffer * samprate
                samp_e = min(obs_samps, int(samp_e))
            obs_samps = samp_e-samp_s

    
    if n_det_chunks is not None and n_dets is not None:
        logger.warning("Both n_det_chunks and n_dets specified, n_det_chunks overrides")
    if n_samp_chunks is not None and n_samps is not None:
        logger.warning("Both n_samp_chunks and n_samps specified, n_samp_chunks overrides")
    
    logger.debug(f"{obs_id} has (n_dets, n_samps): ({obs_dets}, {obs_samps})")
    if ram_limit is not None:
        pts_limit = int(ram_limit // 4)
        n_samp_chunks = 1
        n_samps = obs_samps
        n_dets = pts_limit // n_samps   
        while n_dets == 0:
            n_samp_chunks += 1
            n_samps = obs_samps//n_samp_chunks
            n_dets= pts_limit // (n_samps)
        n_det_chunks = int(np.ceil( obs_dets/n_dets ))
    
    if n_det_chunks is not None:
        n_dets = int(np.ceil(obs_dets/n_det_chunks))
        det_chunks = [range(i*n_dets,min((i+1)*n_dets,obs_dets)) for i in range(n_det_chunks)]
    if n_samp_chunks is not None:
        n_samps = int(np.ceil(obs_samps/n_samp_chunks))
        samp_chunks = [(samp_s+i*n_samps,min((i+1)*n_samps+samp_s,samp_e)) for i in range(n_samp_chunks)]

    if n_dets is not None:
        n_det_chunks = int(np.ceil(obs_dets/n_dets))
        det_chunks = [range(i*n_dets,min((i+1)*n_dets,obs_dets)) for i in range(n_det_chunks)]
    if n_samps is not None:
        n_samp_chunks = int(np.ceil(obs_samps/n_samps))
        samp_chunks = [(samp_s+i*n_samps,min((i+1)*n_samps+samp_s,samp_e)) for i in range(n_samp_chunks)]
        
    if det_chunks is None:
        det_chunks = [range(0,obs_dets)]
    if samp_chunks is None:
        samp_chunks = [(samp_s,samp_e)]
            
    logger.debug(f"Loading data with det_chunks: {det_chunks}.")
    logger.debug(f"Loading data in samp_chunks: {samp_chunks}.")
    
    try:
        for det_chunk in det_chunks:
            for samp_chunk in samp_chunks:
                if test:
                    yield (det_chunk, samp_chunk)
                else:
                    aman = load_file( 
                        filenames, 
                        channels=det_chunk, 
                        samples=samp_chunk, 
                        archive=archive,
                        status=status,
                        **load_file_args,
                    )
                    if partial:
                        msk = np.all(
                            [aman.timestamps >= start.timestamp(), aman.timestamps < end.timestamp()],
                            axis=0,
                        )
                        idx = np.where(msk)[0]
                        if len(idx) == 0:
                            aman.restrict("samps", (aman.samps.offset, aman.samps.offset))
                        else:
                            aman.restrict("samps", (aman.samps.offset+idx[0], aman.samps.offset+idx[-1]))
                    yield aman

    except GeneratorExit:
        pass
    

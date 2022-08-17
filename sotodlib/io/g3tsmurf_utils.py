"""Utility functions for interacting with level 2 data and g3tsmurf
"""

from sotodlib.io.g3tsmurf_db import Observations

def get_obs_folder(obs_id, archive):
    """
    Get the folder associated with the observation action. Assumes
    everything is following the standard suprsync formatting.
    """
    session = archive.Session()
    obs = session.query(Observations).filter(Observations.obs_id == obs_id).one()
    
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
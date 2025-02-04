import os
import yaml

from ..core.util import tag_substr

def get_tags(env_file=None, env_var="DATAPKG_ENV"):
    if env_file is None:
        env_file = os.environ.get(env_var)
    if env_file is None:
        tags = {}
    else:
        tags = yaml.safe_load( open(env_file, "r") )
    return tags

def load_configs(cfg_file, env_file=None, env_var="DATAPKG_ENV"):
    """ load a yaml config file and replace string paths tags
    
    Example config file with a tag::

        my_path : /complete/path
        other_path: {tag}/path
    
    matches with an environment file which would contain::

        tag: /other/relevant/thing

    so that the returned config dictionary contains::

        'other_path' : '/other/relevant/thing/path'
    
    Arguments
    ----------
    cfg_file: str
        path to a .yaml configuration file
    env_file: str (optional)
        path to a .yaml file where each entry is a possible tag.
    env_var: str (optional)
        

    Returns
    --------
    configs: dict
        loaded configuration dictionary where all strings with tags have been
        replaced with the values loaded from the environment file
    """
    tags = get_tags(env_file=env_file, env_var=env_var)

    configs = yaml.safe_load( open(cfg_file, "r"))
    try:
        configs = tag_substr(configs, tags)
    except KeyError as e:
        raise ValueError(
            f"Config file {cfg_file} expects tag {e.args[0]} but this tag is "
            f"not found in environment file '{env_file}'"
        )
    return configs


def get_imprinter_config( platform, env_file=None, env_var="DATAPKG_ENV"):
    tags = get_tags(env_file=env_file, env_var=env_var)

    if 'configs' not in tags:
        raise ValueError(f"configs not found in tags {tags}")

    return os.path.join( tags['configs'], platform, 'imprinter.yaml')

def walk_files(path, include_suprsync=False):
    """get a list of the files in a timecode folder, optional flag to ignore
    suprsync files
    
    Arguments
    ----------
    path: path to a level 2 timecode folder, either smurf or timestreams
    include_suprsync: optional, bool
        if true, includes the suprsync files in the returned list

    Returns
    --------
    files (list): list of the absolute paths to all files in a timecode folder
    """
    if not os.path.exists(path):
        return []
    flist = []
    for root, _, files in os.walk(path):
        if not include_suprsync and 'suprsync' in root:
            continue
        for f in files:
            flist.append( os.path.join(path, root, f))
    return flist

def just_suprsync(path):
    """check if timecode folder only has suprsync folder in it
    """
    flist = os.listdir( path )
    if len(flist) == 1 and flist[0] == "suprsync":
        return True
    return False

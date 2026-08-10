from typing import Dict, List, Tuple

import numpy as np
from mapcat.database import DepthOneMapTable, TODDepthOneTable
from mapcat.helper import Settings
from sqlmodel import select


def map_to_calculate(
    map_name: str, inds_to_use: List[int], mapcat_settings: Dict[str, str]
) -> bool:
    """Check whether a depth-1 map needs to be (re)calculated.

    Compares the total number of wafers already recorded in the map
    catalog against the number of indices requested. Returns True if
    the existing TOD count is less than what is requested.

    Parameters
    ----------
    map_name : str
        Unique name identifying the depth-1 map.
    inds_to_use : list of int
        Indices into the observation list that should contribute to
        this map.
    mapcat_settings : dict
        Connection settings forwarded to ``mapcat.helper.Settings``.

    Returns
    -------
    bool
        True if the map should be calculated, False otherwise.
    """
    with Settings(**mapcat_settings).session() as session:
        map_query = select(DepthOneMapTable).where(DepthOneMapTable.map_name == map_name)
        existing_map = session.execute(map_query).first()
        map_tods = existing_map[0].tods if existing_map else []

        total_tods = np.sum([map_tod.wafer_count for map_tod in map_tods])

        if total_tods < len(inds_to_use):
            return True
    return False


def commit_depth1_tods(
    map_name: str,
    obslist: Dict[Tuple[int, str, str], List[Tuple[str, str, str, int]]],
    obs_infos: np.recarray,
    band: str,
    inds: List[int],
    mapcat_settings: Dict[str, str],
) -> List[TODDepthOneTable]:
    """Commit TOD entries for a depth-1 map to the map catalog.

    For each unique observation id create a
    ``TODDepthOneTable`` row (if one does not already exist) and
    associates it with the given map name if possible.

    Parameters
    ----------
    map_name : str
        Unique name identifying the depth-1 map.
    obslist : dict
        Mapping from index to list of (obs_id, ...) tuples describing
        the observations that contribute to the map.
    obs_infos : np.recarray
        Record array of observation metadata, keyed by ``obs_id``.
    band : str
        Frequency band identifier (e.g. ``'f150'``).
    inds : list of int
        Indices into ``obslist`` selecting the TODs to commit.
    mapcat_settings : dict
        Connection settings forwarded to ``mapcat.helper.Settings``.

    Returns
    -------
    list of TODDepthOneTable
        The TOD entries.
    """
    with Settings(**mapcat_settings).session() as session:
        depth1map_obsids = np.unique([obslist[ind][0] for ind in inds])
        tods = []
        for obs_id in depth1map_obsids:
            obs_info = obs_infos[obs_infos["obs_id"] == obs_id][0]
            tod_depth1_entry = {
                "obs_id": obs_id,
                "ctime": obs_info["timestamp"],
                "start_time": obs_info["start_time"],
                "stop_time": obs_info["stop_time"],
                "nsamples": int(obs_info["n_samples"]),
                "telescope": obs_info["telescope"],
                "telescope_flavor": obs_info["telescope_flavor"],
                "tube_slot": obs_info["tube_slot"],
                "tube_flavor": obs_info["tube_flavor"],
                "frequency": band,
                "scan_type": obs_info["type"],
                "subtype": obs_info["subtype"],
                "wafer_count": int(obs_info["wafer_count"]),
                "duration": obs_info["duration"],
                "az_center": obs_info["az_center"],
                "az_throw": obs_info["az_throw"],
                "el_center": obs_info["el_center"],
                "el_throw": obs_info["el_throw"],
                "roll_center": obs_info["roll_center"],
                "roll_throw": obs_info["roll_throw"],
                "wafer_slots_list": obs_info["wafer_slots_list"],
                "stream_ids_list": obs_info["stream_ids_list"],
            }
            tod_select_values = [
                getattr(TODDepthOneTable, key) == value
                for key, value in tod_depth1_entry.items()
            ]
            tod_query = select(TODDepthOneTable).where(*tod_select_values)
            existing_tod = session.execute(tod_query).first()
            tod = TODDepthOneTable(map_name=map_name, **tod_depth1_entry)
            if existing_tod is None:
                session.add(tod)
                tods.append(tod)
            else:
                tods.append(existing_tod[0])
        session.commit()
    return tods


def commit_depth1_map(
    map_name: str,
    prefix: str,
    detset: str,
    band: str,
    ctime: float,
    start_time: float,
    stop_time: float,
    tods: List[TODDepthOneTable],
    mapcat_settings: Dict[str, str],
) -> None:
    """Commit or update a depth-1 map entry in the map catalog.

    Creates a ``DepthOneMapTable`` row with paths to the map, inverse-
    variance, and time FITS files. If a row with the same ``map_name``
    already exists it is merged (updated) rather than duplicated.

    Parameters
    ----------
    map_name : str
        Unique name identifying the depth-1 map.
    prefix : str
        File path prefix; ``_map.fits``, ``_ivar.fits``, and
        ``_time.fits`` are appended to form the output paths.
    detset : str
        Tube slot / detector set identifier.
    band : str
        Frequency band identifier (e.g. ``'f150'``).
    ctime : float
        Representative ctime for the map.
    start_time : float
        Start time of the earliest contributing observation.
    stop_time : float
        Stop time of the latest contributing observation.
    tods : list of TODDepthOneTable
        TOD entries to associate with this map.
    mapcat_settings : dict
        Connection settings forwarded to ``mapcat.helper.Settings``.
    """
    with Settings(**mapcat_settings).session() as session:
        map_query = select(DepthOneMapTable).where(
            DepthOneMapTable.map_name == map_name
        )
        existing_map = session.execute(map_query).first()
        depth1map_meta = DepthOneMapTable(
            map_id=existing_map[0].map_id if existing_map else None,
            map_name=map_name,
            map_path=prefix + "_map.fits",
            ivar_path=prefix + "_ivar.fits",
            time_path=prefix + "_time.fits",
            tube_slot=detset,
            frequency=band,
            ctime=ctime,
            start_time=start_time,
            stop_time=stop_time,
            tods=tods,
        )
        session.merge(depth1map_meta)
        session.commit()

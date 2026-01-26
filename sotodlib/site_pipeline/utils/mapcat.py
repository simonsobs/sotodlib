from typing import Dict, List, Tuple

import numpy as np
from mapcat.database import DepthOneMapTable, TODDepthOneTable
from mapcat.helper import Settings
from sqlmodel import select


def map_to_calculate(
    map_name: str, inds_to_use: List[int], mapcat_settings: Dict[str, str]
) -> bool:

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
    obs_infos: List[
        Tuple[
            str,
            float,
            float,
            float,
            int,
            str,
            str,
            str,
            str,
            str,
            str,
            int,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            str,
            str,
        ]
    ],
    band: str,
    inds: List[int],
    mapcat_settings: Dict[str, str],
) -> List[TODDepthOneTable]:
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

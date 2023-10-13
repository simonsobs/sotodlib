""" Utility functions for querying and/or working with the G3tHK databases
"""

import numpy as np
import logging

from sotodlib.io.g3thk_db import G3tHk, HKFiles, HKAgents, HKFields

logger = logging.getLogger(__name__)


def pysmurf_monitor_control_list(agent, start=None, stop=None, HK=None, logger=logger):
    """Return list of stream_ids controlled by a pysmurf-monitor agent

    Arguments
    ----------
    agent: HKAgent instance or instance_id
    start: timestamp
        if agent is instance_id, used to find relevant agent instance
    stop: timestamp
        if agent is instance_id, used to find relevant agent instance
    HK: G3tHK Archive
        if agent is instance_id, used to find relevant agent instance

    Returns
    -------
    stream_ids: list of stream_ids monitored by agent
    """
    if isinstance(agent, str):
        if start is None or stop is None:
            raise ValueError(
                "start and stop are required when agent is the" " instance id"
            )
        if HK is None:
            raise ValueError("need database to search if agent is instance id")
        agent_list = HK.get_db_agents(agent, start, stop)
        if len(agent_list) == 0: 
            logger.warning(f"Agent {agent} not running between {start} and {stop}")
            return np.array([], dtype='<U8') 
        logger.info(f"Total agents to go through: {len(agent_list)}")
        return np.unique(
                np.concatenate([pysmurf_monitor_control_list(agent, logger=logger) for agent in agent_list])
        )
    stream_ids = []
    for field in agent.fields:
        if not "_meta" in field.field:
            continue
        splits = field.field.split(".")
        sid = [x for x in splits if "_meta" in x][0].strip("_meta")
        stream_ids.append(sid)
    return np.unique(stream_ids)


def check_was_streaming(stream_id, start, stop, cfgs=None, HK=None, servers=None):
    """Query the HK database to see if a specific stream_id was
    streaming during a time range.

    Arguments
    ----------
    stream_id: string
    start: timestamp
    stop: timestamp
    cfgs: optional, dictionary
        configuration dictionary with finalization informtion. Format matches
        what is described in G3tSmurf.from_configs. If not given, HK and servers
        arguments are required.
    servers: optional, list
        list of dictionaries containing the server to instance id mapping as
        described in G3tSmurf.from_configs. In not given the cfgs dictionary is
        used instead.
    HK: optional, G3tHK instance
        if not given, loaded using configs dictionary

    Returns
    -------
    check: boolean
        if true, pysmurf-monitor has recorded the g3 stream was open
        during the time between start and stop.

    Raises
    ------
    ValueError if stream_id is found in multiple pysmurf-monitors
    """

    if HK is None:
        HK = G3tHk.from_configs(cfgs)
    if servers is None:
        servers = cfgs["finalization"]["servers"]
    assert HK.get_last_update() >= stop

    agents = None

    for server in servers:
        pm = server.get("pysmurf-monitor")
        if pm is None:
            continue
        alist = HK.get_db_agents(pm, start, stop)
        sids = np.unique([pysmurf_monitor_control_list(agent) for agent in alist])
        if np.any([stream_id == s for s in sids]):
            if agents is not None:
                logger.error(f"found {stream_id} in multiple pysmurf-montiors")
                raise ValueError(
                    f"found {stream_id} in multiple "
                    f"pysmurf-montiors, do not know which one to "
                    "use"
                )
            agents = alist

    if agents is None:
        ## found no agents, either not streaming or db not updated
        logger.warning(
            f"Found no pysmurf monitor agents monitoring {stream_id}"
            " during this time"
        )
        return False

    fields = []
    search = f"{stream_id}_meta.AMCcSmurfProcessorSOStreamopen_g3stream"

    for agent in agents:
        idx = np.where([search in field.field for field in agent.fields])[0]
        if len(idx) == 0:
            continue
        fields.append(agent.fields[idx[0]])

    if len(fields) == 0:
        logger.warning(
            f"Found no fields with key {search}. Was metadata" "reading out correctly?"
        )
        return False

    data = HK.load_data(fields)
    if len(data) != 1:
        logger.error("Data returned from field list has more than one key")

    k, val = data.popitem()
    msk = np.all([val[0] >= start, val[0] <= stop], axis=0)

    return np.any(val[1][msk])

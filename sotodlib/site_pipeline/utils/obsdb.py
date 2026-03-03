"""Observation database query utilities for site_pipeline."""

import time


def get_obslist(context, query=None, obs_id=None, min_ctime=None, max_ctime=None,  # multilayer_preprocess_tod, preprocess_obs, preprocess_tod, update_preprocess_plots
                update_delay=None, tags=None, planet_obs=False):
    """Query the obs database with a given query.

    Parameters
    ----------
    context : core.Context
        The context to use for the obsdb.
    query : str, optional
        A query string for the obsdb.
    obs_id : str, optional
        The specific obsid to retrieve.
    min_ctime : int, optional
        The minimum ctime of obs to retrieve.
    max_ctime : int, optional
        The maximum ctime of obs to retrieve.
    update_delay : int, optional
        The number of days to subtract from the current time to set the minimum ctime.
    tags : list of str, optional
        A list of tags to use for the query.
    planet_obs : bool, optional
        If True, format query and tags for planet obs.

    Returns
    -------
    obs_list : list
        The list of obs found from the query.
    """

    try:
        with open(query, "r") as fname:
            return [context.obsdb.get(line.split()[0]) for line in fname]
    except FileNotFoundError:
        if (min_ctime is None) and (update_delay is not None):
            # If min_ctime is provided it will use that..
            # Otherwise it will use update_delay to set min_ctime.
            min_ctime = int(time.time()) - update_delay*86400

        if obs_id is not None:
            tot_query = f"obs_id=='{obs_id}'"
        else:
            tot_query = "and "
            if min_ctime is not None:
                tot_query += f"timestamp>={min_ctime} and "
            if max_ctime is not None:
                tot_query += f"timestamp<={max_ctime} and "
            if query is not None:
                tot_query += query + " and "
            tot_query = tot_query[4:-4]
            if tot_query=="":
                tot_query="1"

        if not(tags is None):
            for i, tag in enumerate(tags):
                tags[i] = tag.lower()
                if '=' not in tag:
                    tags[i] += '=1'

        if planet_obs:
            obs_list = []
            for tag in tags:
                obs_list.extend(context.obsdb.query(tot_query, tags=[tag]))
        else:
            obs_list = context.obsdb.query(tot_query, tags=tags)

        return obs_list

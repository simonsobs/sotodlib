#!/usr/bin/env python3

# Copyright (c) 2024-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
Helper script for batching observations.

The results are printed to stdout to enable use of this
script inside a slurm script.

"""

import argparse
import os
import re
import time


def load_obs_file(path):
    """Load the observation file into a list.

    Args:
        path (str):  Path to the file

    Returns:
        (list):  The list of observation IDs.

    """
    olist = list()
    with open(path, "r") as f:
        for line in f:
            if re.match(r"^#.*", line) is None:
                olist.append(line.strip())
    return olist


def get_state_file(out_root, obs, name="state"):
    """Construct the path to the state file.

    Args:
        out_root (str):  The top-level output directory
        obs (str):  The observation ID

    Returns:
        (str):  The state file path.

    """
    obs_dir = os.path.join(out_root, obs)
    state_file = os.path.join(obs_dir, name)
    return state_file


def get_obs_state(out_root, obs):
    """Find the current processing state of an observation.
    """
    state_file = get_state_file(out_root, obs)
    if os.path.isfile(state_file):
        with open(state_file, "r") as f:
            state = f.readline()
        return state.rstrip()
    return None


def clear_obs_state(out_root, obs):
    """Clear the current processing state of an observation.
    """
    obs_dir = os.path.join(out_root, obs)
    if not os.path.isdir(obs_dir):
        return
    state_file = get_state_file(out_root, obs)
    if os.path.isfile(state_file):
        os.remove(state_file)


def set_obs_state(out_root, obs, state):
    """Set the current processing state of an observation.
    """
    obs_dir = os.path.join(out_root, obs)
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)
    state_file = get_state_file(out_root, obs)
    temp_state = get_state_file(out_root, obs, name="state_temp")
    # Although the temp file only exists for a fraction of a second,
    # check if it was somehow left behind by a previous process.
    if os.path.isfile(temp_state):
        msg = "Temporary state file {temp_state} exists."
        msg += "  This means that a previous batch_control job was "
        msg += "killed while setting the state.  You should cleanup "
        msg += "manually."
        raise RuntimeError(msg)
    with open(temp_state, "w") as f:
        f.write(f"{state}\n")
    os.rename(temp_state, state_file)


def find_obs(all_obs, n_job_obs, out_root, ignore_running=False, timeout_hours=24):
    """Get the list of obs to run for a given job.

    Args:
        all_obs (list):  The list of all observation IDs to consider.
        n_job_obs (int):  The number of observations to select.
        out_root (str):  The output root directory.
        ignore_running (bool):  If True, consider "running" jobs to be eligible
            for re-running.
        timeout_hours (float):  Number of hours after which a "running" job is
            considered dead, regardless of ignore_running option.

    Returns:
        None

    """
    timeout_sec = timeout_hours * 3600
    selected = list()
    for obs in all_obs:
        state = get_obs_state(out_root, obs)
        if state is not None:
            if state == "done":
                continue
            if state == "running" and not ignore_running:
                # Check how long this job has been "running"
                cur_time = time.time()
                state_file = get_state_file(out_root, obs)
                state_time = os.path.getmtime(state_file)
                elapsed = cur_time - state_time
                if elapsed < timeout_sec:
                    # The job has not timed out yet
                    continue
        # We are going to consider this obs
        selected.append(obs)
        if len(selected) >= n_job_obs:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Get observation IDs for a job")
    parser.add_argument(
        "--out_root",
        required=True,
        default=None,
        help="The output root directory",
    )
    parser.add_argument(
        "--observations",
        required=False,
        default=None,
        help="File of observation IDs",
    )
    parser.add_argument(
        "--get_batch",
        required=False,
        type=int,
        default=None,
        help="Get the next batch of observations",
    )
    parser.add_argument(
        "--batch_list",
        required=False,
        action="store_true",
        default=False,
        help="Print the batch as a comma-separated list instead of one per line",
    )
    parser.add_argument(
        "--get_state",
        required=False,
        type=str,
        default=None,
        help="Get the state of this observation",
    )
    parser.add_argument(
        "--set_state_done",
        required=False,
        type=str,
        default=None,
        help="Set the state of this observation to 'done'",
    )
    parser.add_argument(
        "--set_state_running",
        required=False,
        type=str,
        default=None,
        help="Set the state of this observation to 'running'",
    )
    parser.add_argument(
        "--clear_state",
        required=False,
        type=str,
        default=None,
        help="Clear the state of this observation",
    )
    parser.add_argument(
        "--cleanup",
        required=False,
        action="store_true",
        default=False,
        help="Remove the running state from all observations",
    )

    args = parser.parse_args()

    if args.get_batch is not None:
        # We are getting the next batch of observations
        all_obs = load_obs_file(args.observations)
        batch_obs = find_obs(all_obs, args.get_batch, args.out_root)
        if args.batch_list:
            batch_str = ",".join(batch_obs)
            print(f"{batch_str}", flush=True)
        else:
            for obs in batch_obs:
                print(f"{obs}", flush=True)
    elif args.get_state is not None:
        state = get_obs_state(args.out_root, args.get_state)
        print(f"{state}", flush=True)
    elif args.clear_state is not None:
        clear_obs_state(args.out_root, args.clear_state)
    elif args.set_state_done is not None:
        set_obs_state(args.out_root, args.set_state_done, "done")
    elif args.set_state_running is not None:
        set_obs_state(args.out_root, args.set_state_running, "running")
    elif args.cleanup:
        all_obs = load_obs_file(args.observations)
        for obs in all_obs:
            state = get_obs_state(args.out_root, obs)
            if state is not None and state == "running":
                clear_obs_state(args.out_root, obs)


if __name__ == "__main__":
    main()

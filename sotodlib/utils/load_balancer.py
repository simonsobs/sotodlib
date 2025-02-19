"""
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
from typing import Dict, List, Tuple, Optional

class ObsToRankBalancer(object):
    '''
    This class implemements an Obs to rank planner based on Heterogeneous Earliest
    Finish Time (HEFT) scheduling algorithm. 

    For reference:
    H. Topcuoglu, S. Hariri, and Min-You Wu. Performance-effective and 
    low-complexity task scheduling for heterogeneous computing. 
    IEEE Transactions on Parallel and Distributed Systems, March 2002.

    Constractor parameters:
    observations: A list of observations ids along with their wafer.
    mpi_size: The number of MPI ranks to use.

    The class implements a plan method that returns a plan, a list of tuples. 

    Each tuple will have the obs_id, selected rank, total number of samples before this
    observation, and the total number of samples after this observation.
    '''
    def __init__(self, observations: List[Dict[str, int]], comm_size: int=1) -> None:


        self.observations = [(ob['obs_id'], ob['nsamples'] * ob['detectors']) for ob in observations]
        self.resources = [{'performance': 1, 'name': f'rank{i:04d}'} for i in range(comm_size)]
        self.num_samples =  [x[1] for x in self.observations]
        self._plan = list()


    def plan(self, rank: Optional[int] = None) -> List[Tuple[Tuple[str, int], Dict[str, int], int, int]]:
        '''
        This method implements the basic HEFT algorithm. It returns a list of tuples
        Each tuple contains: Workflow ID, Resource ID, Start Time, End Time.

        The plan method takes as input a campaign, resources and num_oper in case
        any of these has changed. They default to `None`
        *Parameters:*
            rank: The rank to plan for. If None, it will plan for all ranks.
        *Returns:*
            List[Tuple[Tuple[str, int], Dict[str, int], int, int]]
        '''

        tmp_cmp = self.observations
        tmp_res = self.resources
        # Reset the plan in case of a recall
        self._plan = list()

        # Get the indices of the sorted list.
        av_est_idx_sorted = [i[0] for i in sorted(enumerate(self.num_samples),
                                                  key=lambda x: x[1],
                                                  reverse=True)]
        # This list tracks when a resource whould be available.

        resource_free = [0] * len(tmp_res)

        for sorted_idx in av_est_idx_sorted:
            wf_est_tx = self.num_samples[sorted_idx]
            if wf_est_tx != 0:
                min_end_time = float('inf')
                for i in range(len(tmp_res)):
                    tmp_str_time = resource_free[i]
                    tmp_end_time = tmp_str_time + wf_est_tx
                    if tmp_end_time < min_end_time:
                        min_end_time = tmp_end_time
                        tmp_min_idx = i
                self._plan.append((tmp_cmp[sorted_idx],
                                tmp_res[tmp_min_idx],
                                resource_free[tmp_min_idx], 
                                resource_free[tmp_min_idx] +
                                    wf_est_tx))
                resource_free[tmp_min_idx] = resource_free[tmp_min_idx] + \
                                            wf_est_tx
        if rank is not None:
            self._plan = [x for x in self._plan if x[1]['name'] == f'rank{rank:04d}']
        return self._plan

from __future__ import annotations
from typing import Tuple, Union, List, Callable, Optional


def _get_mpi_comm() -> (
    Tuple[bool, int, Optional["MPICommExecutor"], Optional[Callable]]
):
    """This private function tries to create an MPICommExecutor object and returns it if successful.

    Returns
    -------
    bool
        Whether an MPICommExecutor object was created successfully. This is important as only rank 0 creates the executor and all other ranks return None as an executor.
    int
        The global rank of the master process.
    Optional["MPICommExecutor"]
        If the executor was created successfully, this is the MPICommExecutor object. Otherwise, it is None.
    Optional[Callable]
        If the executor was created successfully, this is the as_completed function. Otherwise, it is None.
    """
    try:
        from mpi4py.futures import MPICommExecutor
        from mpi4py.futures import as_completed
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        rank = comm.Get_rank()
        max_workers = comm.Get_size() - 1

        return (
            True,
            rank,
            MPICommExecutor(comm, root=0, max_workers=max_workers).__enter__(),
            as_completed,
        )
    except (NameError, ImportError):
        return False, 0, None, None


def _get_concurrent_comm(
    nprocs: Optional[int] = None,
) -> Tuple[int, "ProcessPoolExecutor", Callable]:
    """This private function tries to create an ProcessPoolExecutor object and returns it if successful.

    Returns
    -------
    bool
        Whether an ProcessPoolExecutor object was created successfully. This is important as only rank 0 creates the executor and all other ranks return None as an executor.
    int
        The global rank of the master process.
    Optional["ProcessPoolExecutor"]
        If the executor was created successfully, this is the ProcessPoolExecutor object. Otherwise, it is None.
    Optional[Callable]
        If the executor was created successfully, this is the as_completed function. Otherwise, it is None.
    """

    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import as_completed
    from multiprocessing import set_start_method

    set_start_method("spawn")
    return True, 0, ProcessPoolExecutor(max_workers=nprocs), as_completed


def get_exec_env(
    nprocs: int = None, priority: List[str] = ["mpi", "process_pool"]
) -> Tuple[int, Union["MPICommExecutor", "ProcessPoolExecutor"], Callable]:
    """This function sets up the execution environment for parallel processing based on the specified priority list.

    Since this function returns a rank value the main function needs to go under an
    >>> if rank == 0

    block to ensure that the code is only run by the master process.

    In addition the main function needs to take the executor and as_completed_callable as arguments.

    Let's assume that the main function is called main_func and it current state is as follows:
    >>> def main_func():
    >>>     with ProcessPoolExecutor() as executor:
    >>>         futures = []
    >>>         for i in range(10):
    >>>             futures.append(executor.submit(some_func, i))
    >>>         for future in as_completed(futures):
    >>>             result = future.result()
    >>>             print(result)
    >>> if __name__ == "__main__":
    >>>     main_func()


    Utilizing different executors requires to change the main function as follows:
    >>> def main_func(executor, as_completed_callable):
    >>>     futures = []
    >>>     for i in range(10):
    >>>         futures.append(executor.submit(some_func, i))
    >>>     for future in as_completed_callable(futures):
    >>>         result = future.result()
    >>>         print(result)
    >>> if __name__ == "__main__":
    >>>     rank, executor, as_completed_callable = get_exec_env(nprocs=4)
    >>>     if rank == 0:
    >>>         main_func(executor, as_completed_callable)


    Parameters
    ----------
    nprocs : int, optional
        The number of processes to use for the process pool executor. If not specified, the default is None.
    priority : List[str], optional

    Returns
    -------
    int
        The rank of the process that is the master process for this type of execution

    Union["MPICommExecutor", "ProcessPoolExecutor"]
        The executor that is used to run the code in parallel. This can be either an MPICommExecutor or a ProcessPoolExecutor depending on the execution environment.

    Callable
        The as_completed function that is used to get the results of the parallel execution based on the type of the executor

    Raises
    ------
    ValueError
        When a valid executor is not available based on the execution priority list.
    """
    user_priority = list(priority)
    executor_created = False
    while not executor_created:
        executor_mode = user_priority.pop(0)
        match executor_mode:
            case "mpi":
                executor_created, rank, executor, as_completed_callable = (
                    _get_mpi_comm()
                )
            case "process_pool":
                executor_created, rank, executor, as_completed_callable = (
                    _get_concurrent_comm(nprocs)
                )
            case _:
                raise ValueError("No executor available")

    return rank, executor, as_completed_callable

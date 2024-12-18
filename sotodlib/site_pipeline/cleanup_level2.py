import numpy as np
import datetime as dt
from typing import Optional
import argparse

from sotodlib.io.imprinter import Imprinter
from sotodlib.io.datapkg_completion import DataPackaging
from sotodlib.site_pipeline.util import init_logger

logger = init_logger(__name__, "cleanup_level2: ")

def level2_completion(
    dpk: DataPackaging, 
    lag: Optional[float] = 14,
    min_timecode: Optional[int] = None, 
    max_timecode: Optional[int] = None, 
    raise_incomplete: Optional[bool] = True,
):

    ## build time range where we require timecodes to be complete
    if min_timecode is None:
        min_timecode = dpk.get_first_timecode_on_disk()
    if max_timecode is None:
        x = dt.datetime.now() - dt.timedelta(days=lag)
        max_timecode = int( x.timestamp() // 1e5)

    logger.info(
        f"Checking Timecode completion from {min_timecode} to "
        f"{max_timecode}."
    )

    check_list = []
    for timecode in range(min_timecode, max_timecode):
        check = dpk.make_timecode_complete(timecode)
        if not check[0]:
            check_list.append( (timecode, check[1]) )
            continue
        check = dpk.verify_timecode_deletable(
            timecode, include_hk=True, 
            verify_with_librarian=False,
        ) 
        if not check[0]:
            check_list.append( (timecode, check[1]) )

    if len( check_list ) > 0 and raise_incomplete:
        raise ValueError(
            f"Data Packaging cannot be completed for {check_list}"
        )

def do_delete_level2(
    dpk: DataPackaging, 
    lag: Optional[float] = 28,
    min_timecode: Optional[int] = None, 
    max_timecode: Optional[int] = None, 
    raise_incomplete: Optional[bool] =True,
):
    ## build time range where we should be deleting
    if min_timecode is None:
        min_timecode = dpk.get_first_timecode_on_disk()

    if max_timecode is None:
        x = dt.datetime.now() - dt.timedelta(days=lag)
        max_timecode = int( x.timestamp() // 1e5)
    
    logger.info(
        f"Removing Level 2 data from {min_timecode} to "
        f"{max_timecode}."
    )
    delete_list = []
    for timecode in range(min_timecode, max_timecode):
        check = dpk.check_and_delete_timecode(timecode)
        if not check[0]:
            logger.error(f"Failed to remove level 2 for {timecode}")
            delete_list.append( (timecode, check[1]))
            continue
    if len( delete_list ) > 0 and raise_incomplete:
        raise ValueError(
            f"Level 2 Deletion not finished for {delete_list}"
        )

def do_delete_staged(
    dpk: DataPackaging, 
    lag: Optional[float] = 14,
    min_timecode: Optional[int] = None, 
    max_timecode: Optional[int] = None, 
    raise_incomplete: Optional[bool] =True,
):
    ## build time range where we should be deleting
    if min_timecode is None:
        min_timecode = dpk.get_first_timecode_in_staged()

    if max_timecode is None:
        x = dt.datetime.now() - dt.timedelta(days=lag)
        max_timecode = int( x.timestamp() // 1e5)
        
    logger.info(
        f"Removing staged from {min_timecode} to "
        f"{max_timecode}."
    )
    delete_list = []
    for timecode in range(min_timecode, max_timecode):
        check = dpk.make_timecode_complete(timecode)
        if not check[0]:
            delete_list.append( (timecode, check[1]) )
            continue
        check = dpk.verify_timecode_deletable(
            timecode, include_hk=True, 
            verify_with_librarian=False,
        ) 
        if not check[0]:
            delete_list.append( (timecode, check[1]) )
            continue
        check = dpk.delete_timecode_staged(timecode)
        if not check[0]:
            logger.error(f"Failed to remove staged for {timecode}")
            delete_list.append( (timecode, check[1]))
            continue
    if len( delete_list ) > 0 and raise_incomplete:
        raise ValueError(
            f"Staged Deletion not finished for {delete_list}"
        )


def main(
    platform: str,
    check_complete: Optional[bool]= False,
    delete_staged: Optional[bool] = False,
    delete_lvl2: Optional[bool]= False,
    completion_lag: Optional[float] = 14,
    min_complete_timecode: Optional[int] = None,
    max_complete_timecode: Optional[int] = None,
    staged_deletion_lag: Optional[float] = 14,
    min_staged_delete_timecode: Optional[int] = None,
    max_staged_delete_timecode: Optional[int] = None,
    lvl2_deletion_lag: Optional[float] = 28,
    min_lvl2_delete_timecode: Optional[int] = None,
    max_lvl2_delete_timecode: Optional[int] = None,
    ):
    """
    Use the imprinter database to clean up already bound level 2 files. 

    Parameters
    ----------
    platform : str
        platform we're running for
    completion_lag : float, optional
        The number of days in the past where we expect data packaging to be
        fully complete.
    min_complete_timecode : Optional[datetime], optional
        The lowest timecode to run completion checking. over-rides the "start
        from beginning" behavior.
    max_complete_timecode : Optional[datetime], optional
        The highest timecode to run completion checking. over-rides the
        completion_lag calculated value.
    dry_run : Optional[bool], 
        If true, only prints deletion to logger
    """
    dpk = DataPackaging(platform)

    if check_complete:
        level2_completion(
            dpk, completion_lag,  
            min_complete_timecode, max_complete_timecode,
        )
    
    if delete_staged:
        do_delete_staged(
            dpk, staged_deletion_lag, 
            min_staged_delete_timecode, max_staged_delete_timecode
        )
    
    if delete_lvl2:
        do_delete_level2(
            dpk, lvl2_deletion_lag,  
            min_lvl2_delete_timecode, max_lvl2_delete_timecode,
        )



def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('platform', type=str, help="Platform for Imprinter")
    parser.add_argument('--check-complete', action="store_true", 
        help="If passed, run completion check")
    parser.add_argument('--delete-lvl2', action="store_true", 
        help="If passed, delete lvl2 raw data")
    parser.add_argument('--delete-staged', action="store_true", 
        help="If passed, delete lvl2 staged data")

    parser.add_argument('--completion-lag', type=float, default=14, 
        help="Buffer days before we start failing completion")
    parser.add_argument('--min-complete-timecode', type=int,
        help="Minimum timecode to start completion check. Overrides starting "
        "from the beginning")
    parser.add_argument('--max-complete-timecode', type=int,
        help="Maximum timecode to stop completion check. Overrides the "
        "completion-lag setting")
    
    parser.add_argument('--lvl2-deletion-lag', type=float, default=28, 
        help="Buffer days before we start deleting level 2 raw data")
    parser.add_argument('--min-lvl2-delete-timecode', type=int,
        help="Minimum timecode to start level 2 raw data deletion. Overrides "
        "starting from the beginning")
    parser.add_argument('--max-lvl2-delete-timecode', type=int,
        help="Maximum timecode to stop level 2 raw data deletion. Overrides the"
        " lvl2-deletion-lag setting")
    
    parser.add_argument('--staged-deletion-lag', type=float, default=28, 
        help="Buffer days before we start deleting level 2 staged data")
    parser.add_argument('--min-staged-delete-timecode', type=int,
        help="Minimum timecode to start level 2 staged data deletion. Overrides"
        " starting from the beginning")
    parser.add_argument('--max-staged-delete-timecode', type=int,
        help="Maximum timecode to stop level 2 staged data deletion. Overrides"
        " the lvl2-deletion-lag setting")

    return parser

if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))

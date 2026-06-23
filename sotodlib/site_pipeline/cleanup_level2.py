import numpy as np
import datetime as dt
from typing import Optional
import argparse

from sotodlib.io.datapkg_completion import DataPackaging
from sotodlib.site_pipeline.utils.logging import init_logger

logger = init_logger(__name__, "cleanup_level2: ")

def log_response(resp):
    if resp[0]:
        logger.info(resp[1])
    else:
        logger.error(resp[1])

def main(
    platform: str,
    delete_staged: Optional[bool] = False,
    delete_lvl2: Optional[bool] = False,
    completion_lag: Optional[float] = 14,
    min_complete_timecode: Optional[int] = None,
    max_complete_timecode: Optional[int] = None,
    staged_deletion_lag: Optional[float] = 14,
    min_staged_delete_timecode: Optional[int] = None,
    max_staged_delete_timecode: Optional[int] = None,
    lvl2_deletion_lag: Optional[float] = 28,
    min_lvl2_delete_timecode: Optional[int] = None,
    max_lvl2_delete_timecode: Optional[int] = None,
    dry_run: Optional[bool] = False,
    max_runtime: Optional[float] = None,
):
    """
    Use the imprinter database to clean up already bound level 2 files.

    Always runs make_timecode_complete and verify_timecode_deletable for every
    timecode in the completion-check range. The completion-check range must
    fully contain any deletion ranges — an error is raised at startup if not.
    Each of those checks is called at most once per timecode.

    Parameters
    ----------
    platform : str
        Platform we're running for.
    delete_staged : bool, optional
        If True, delete staged data for each timecode.
    delete_lvl2 : bool, optional
        If True, delete level 2 raw data for each timecode.
    completion_lag : float, optional
        Days in the past before which we require data packaging to be
        complete. Used to set the max of the completion-check range.
    min_complete_timecode : int, optional
        Override the lowest timecode for completion checking.
    max_complete_timecode : int, optional
        Override the highest timecode for completion checking.
    staged_deletion_lag : float, optional
        Days in the past before which we delete staged data.
    min_staged_delete_timecode : int, optional
        Override the lowest timecode for staged deletion.
    max_staged_delete_timecode : int, optional
        Override the highest timecode for staged deletion.
    lvl2_deletion_lag : float, optional
        Days in the past before which we delete level 2 raw data.
    min_lvl2_delete_timecode : int, optional
        Override the lowest timecode for level 2 deletion.
    max_lvl2_delete_timecode : int, optional
        Override the highest timecode for level 2 deletion.
    dry_run : bool, optional
        If True, log what would be done without making any changes.
    max_runtime : float, optional
        Maximum number of minutes to run before stopping the loop early.
        The loop stops cleanly at the next timecode boundary; any failures
        accumulated up to that point are still raised.
    """
    if dry_run:
        logger.info("DRY RUN: no changes will be made.")

    dpk = DataPackaging(platform)

    def lag_to_max_timecode(lag: float) -> int:
        x = dt.datetime.now() - dt.timedelta(days=lag)
        return int(x.timestamp() // 1e5)

    # Build Time Ranges where we're running completion checks, deleting staged
    # and deleting level 2.
    
    tc_check_min = ( 
        min_complete_timecode
        if min_complete_timecode is not None
        else dpk.get_first_timecode_on_disk()
    )
    tc_check_max = (
        max_complete_timecode
        if max_complete_timecode is not None
        else lag_to_max_timecode(completion_lag)
    )
    logger.info(
        f"Completion check range: {tc_check_min} to {tc_check_max}."
    )

    if delete_staged:
        tc_staged_min = (
            min_staged_delete_timecode
            if min_staged_delete_timecode is not None
            else dpk.get_first_timecode_in_staged()
        )
        tc_staged_max = (
            max_staged_delete_timecode
            if max_staged_delete_timecode is not None
            else lag_to_max_timecode(staged_deletion_lag)
        )
        logger.info(
            f"Staged deletion range: {tc_staged_min} to {tc_staged_max}."
        )
    else:
        tc_staged_min = tc_staged_max = None

    if delete_lvl2:
        tc_lvl2_min = (
            min_lvl2_delete_timecode
            if min_lvl2_delete_timecode is not None
            else dpk.get_first_timecode_on_disk()
        )
        tc_lvl2_max = (
            max_lvl2_delete_timecode
            if max_lvl2_delete_timecode is not None
            else lag_to_max_timecode(lvl2_deletion_lag)
        )
        logger.info(
            f"Level 2 deletion range: {tc_lvl2_min} to {tc_lvl2_max}."
        )
    else:
        tc_lvl2_min = tc_lvl2_max = None

    # --- Validate that the completion-check range covers all deletions ----
    if delete_staged and (
        tc_check_min > tc_staged_min or tc_check_max < tc_staged_max
    ):
        raise ValueError(
            f"Completion check range [{tc_check_min}, {tc_check_max}) "
            f"does not cover staged deletion range "
            f"[{tc_staged_min}, {tc_staged_max}). "
            f"Increase --completion-lag or adjust the completion timecode bounds."
        )
    if delete_lvl2 and (
        tc_check_min > tc_lvl2_min or tc_check_max < tc_lvl2_max
    ):
        raise ValueError(
            f"Completion check range [{tc_check_min}, {tc_check_max}) "
            f"does not cover level 2 deletion range "
            f"[{tc_lvl2_min}, {tc_lvl2_max}). "
            f"Increase --completion-lag or adjust the completion timecode bounds."
        )

    # --- Runtime deadline -------------------------------------------------
    start_time = dt.datetime.now()
    deadline = (
        start_time + dt.timedelta(minutes=max_runtime)
        if max_runtime is not None
        else None
    )

    # --- Accumulate failures per operation --------------------------------
    complete_failures = []
    staged_failures = []
    lvl2_failures = []

    # --- Unified per-timecode loop ----------------------------------------
    for timecode in range(tc_check_min, tc_check_max):

        if deadline is not None and dt.datetime.now() >= deadline:
            logger.info(
                f"Max runtime of {max_runtime} minutes reached at timecode "
                f"{timecode}; stopping early."
            )
            break

        # 1. Completion check — always runs for every timecode in the range.
        #    make_timecode_complete and verify_timecode_deletable are called
        #    exactly once per timecode; deletion steps below reuse the result.
        if dry_run:
            logger.info(
                f"DRY RUN: would run make_timecode_complete and "
                f"verify_timecode_deletable for {timecode}."
            )
        else:
            check = dpk.make_timecode_complete(timecode)
            log_response(check)
            if not check[0]:
                complete_failures.append((timecode, check[1]))
                continue
            check = dpk.verify_timecode_deletable(
                timecode, include_hk=True,
                verify_with_librarian=False,
            )
            log_response(check)
            if not check[0]:
                complete_failures.append((timecode, check[1]))
                continue

        # 2. Staged deletion.
        if delete_staged and tc_staged_min <= timecode < tc_staged_max:
            if dry_run:
                logger.info(
                    f"DRY RUN: would delete staged data for {timecode}."
                )
            else:
                check = dpk.delete_timecode_staged(timecode, verify_with_librarian=False)
                log_response(check)
                if not check[0]:
                    logger.error(f"Failed to remove staged for {timecode}")
                    staged_failures.append((timecode, check[1]))
                    continue

        # 3. Level 2 deletion.
        if delete_lvl2 and tc_lvl2_min <= timecode < tc_lvl2_max:
            if dry_run:
                logger.info(
                    f"DRY RUN: would delete level 2 data for {timecode}."
                )
            else:
                check = dpk.delete_timecode_level2(timecode, dry_run=False, 
                    verify_with_librarian=True
                )
                log_response(check)
                dpk.cleanup_level2_folders(timecode)
                if not check[0]:
                    logger.error(f"Failed to remove level 2 for {timecode}")
                    lvl2_failures.append((timecode, check[1]))
                    continue

    # --- Raise on any failures --------------------------------------------
    errors = []
    if complete_failures:
        errors.append(
            f"Data Packaging cannot be completed for {complete_failures}"
        )
    if staged_failures:
        errors.append(
            f"Staged Deletion not finished for {staged_failures}"
        )
    if lvl2_failures:
        errors.append(
            f"Level 2 Deletion not finished for {lvl2_failures}"
        )
    if errors:
        raise ValueError("\n".join(errors))


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('platform', type=str, help="Platform for Imprinter")
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

    parser.add_argument('--dry-run', action="store_true",
        help="If passed, log what would be done without making any changes")
    parser.add_argument('--max-runtime', type=float, default=None,
        help="Maximum number of minutes to run before stopping the loop early")

    return parser

if __name__ == "__main__":
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args))

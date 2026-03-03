"""Logging utilities for site_pipeline."""

import logging
import sys
import time


class _ReltimeFormatter(logging.Formatter):
    """Logging formatter that displays relative timestamps."""

    def __init__(self, *args, t0=None, **kw):
        super().__init__(*args, **kw)
        if t0 is None:
            t0 = time.time()
        self.start_time = t0

    def formatTime(self, record, datefmt=None):
        if datefmt is None:
            datefmt = '%8.3f'
        return datefmt % (record.created - self.start_time)


def init_logger(name, announce='', verbosity=2, logger=None):  # analyze_bright_ptsrc, check_book, cleanup_level2, finalize_focal_plane, make_coadd_atomic_map, make_cosamp_hk, make_det_info_wafer, make_hwp_solutions, make_read_det_match, make_tau_hwp, make_uncal_beam_map, multilayer_preprocess_tod, preprocess_obs, preprocess_tod, record_qa, update_book_plan, update_det_cal, update_hkdb, update_hwp_angle, update_librarian, update_obsdb, update_preprocess_plots, update_smurf_caldbs
    """Configure and return a logger for site_pipeline elements.  It is
    disconnected from general sotodlib (propagate=False) and displays
    relative instead of absolute timestamps.

    """
    if logger is None:
        logger = logging.getLogger(name)

    if verbosity == 0:
        level = logging.ERROR
    elif verbosity == 1:
        level = logging.WARNING
    elif verbosity == 2:
        level = logging.INFO
    elif verbosity == 3:
        level = logging.DEBUG

    # add handler only if it doesn't exist
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler(sys.stdout)
        formatter = _ReltimeFormatter('%(asctime)s: %(message)s (%(levelname)s)')

        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        i, r = formatter.start_time // 1, formatter.start_time % 1
        text = (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(i))
              + (',%03d' % (r*1000)))
        logger.info(f'{announce}Log timestamps are relative to {text}')
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)
                break

    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    return logger

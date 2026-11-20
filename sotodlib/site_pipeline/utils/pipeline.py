"""Pipeline execution utilities for site_pipeline."""

import sys


def main_launcher(main_func, parser_func, args=None):  # analyze_bright_ptsrc, check_book, cli, generate_reports, make_coadd_atomic_map, make_cosamp_hk, make_det_info_wafer, make_ml_map, make_source_flags, make_tau_hwp, make_uncal_beam_map, multilayer_preprocess_tod, preprocess_obs, preprocess_tod, record_qa, update_det_cal, update_preprocess_plots
    """Launch an element's main entry point function, after generating
    a parser and executing it on the command line arguments (or args
    if it is passed in).

    Args:
      main_func: the main entry point for a pipeline element.
      parser_func: the argument parser generation function for a pipeline
        element.
      args (list of str): arguments to parse (default is None, which
        will lead to sys.argv[1:]).

    Returns:
      Whatever main_func returns.

    """
    if args is None:
        args = sys.argv[1:]
    return main_func(**vars(parser_func().parse_args(args=args)))

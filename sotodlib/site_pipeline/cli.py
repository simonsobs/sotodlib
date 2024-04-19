"""In order to plug in nicely to the command line wrapper, element
submodules should expose functions called ``main()`` and
``get_parser()``.

The ``get_parser()`` function should look like this::

  def get_parser(parser=None):
    if parser is None:
      parser = argparse.ArgumentParser()
    # element-specific args:
    parser.add_argument('obs_id', help="The obs_id to analyze.")
    parser.add_arugment('--config-file', help="Config file.")
    return parser

When called by the ``so-site-pipeline`` wrapper, a parser will be
passed in.

The ``main()`` function is the entry point to be called from the CLI
or from Prefect.  The arguments should include the arguments defined
through the ArgumentParser, as well as any additional support for
Prefect (such as a logger argument).  For example::

  def main(obs_id=None, config_file=None, logger=None):
    ...


If you want the submodule to be executable directly as a script (or
through ``python -m``), add a ``__main__`` handling block like this
one::

  if __name__ == '__main__':
    util.main_launcher(main, get_parser)


To register a properly organized submodule in the ``so-site-pipeline``
command line wrapper, edit ``cli.py`` and see comments inline.

"""

import argparse

from . import (
    analyze_bright_ptsrc,
    check_book,
    make_det_info_wafer,
    make_ml_map,
    make_source_flags,
    make_uncal_beam_map,
    preprocess_tod,
    update_g3tsmurf_db,
    update_obsdb,
    make_level3_hk
)

# Dictionary matching element name to a submodule (which must have
# been imported above).  Note for these to work, the submodule must
# have defined a get_parser() and a main() function as described in
# the module comments above.

ELEMENTS = {
    'analyze-bright-ptsrc': analyze_bright_ptsrc,
    'check-book': check_book,
    'make-det-info-wafer': make_det_info_wafer,
    'make-ml-map': make_ml_map,
    'make-source-flags': make_source_flags,
    'make-uncal-beam-map': make_uncal_beam_map,
    'preprocess-tod': preprocess_tod,
    'update-g3tsmurf-db': update_g3tsmurf_db,
    'update-obsdb': update_obsdb,
    'make_level3_hk': make_level3_hk,
}

CLI_NAME = 'so-site-pipeline'

def get_parser():
    parser = argparse.ArgumentParser()
    # Make sure all args here are redirected to vars starting with
    # '_'.  We are going to clean those off before passing to the
    # subcommand.
    parser.add_argument('--bash-completion', action='store_true',
                        dest='_bash_completion')
    sps = parser.add_subparsers(dest='_pipemod')

    for name, module in ELEMENTS.items():
        sp = sps.add_parser(name)
        module.get_parser(sp)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Extract top-level args ...
    top_args = {k: v for k, v in vars(args).items()
                if k[0] == '_'}
    [delattr(args, k) for k in top_args]
    top_args = argparse.Namespace(**top_args)

    if top_args._bash_completion:
        names = list(ELEMENTS.keys())
        print('complete -W "%s" %s' % (' '.join(names), CLI_NAME))
        parser.exit()
    
    if top_args._pipemod is None:
        parser.error('First argument must be a sub-command.')

    module = ELEMENTS[top_args._pipemod]
    module.main(**vars(args))

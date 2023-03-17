import argparse

from . import (
    make_source_flags,
    make_uncal_beam_map,
)

# Dictionary matching element name to a submodule (which must have
# been imported above).  Note for these to work, the submodule must
# have defined a get_parser() and a main() function.
#
# Those functions should be set up like this:
#
#   def get_parser(parser=None):
#       if parser is None:
#           parser = argparse.ArgumentParser(...)
#       ...
#
#   def main(args=None):
#       args = util.get_args(args, get_parser)
#

ELEMENTS = {
    'make-source-flags': make_source_flags,
    'make-uncal-beam-map': make_uncal_beam_map,
}

CLI_NAME = 'so-site-pipeline'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash-completion', action='store_true')
    sps = parser.add_subparsers(dest='_pipemod')

    for name, module in ELEMENTS.items():
        sp = sps.add_parser(name)
        module.get_parser(sp)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.bash_completion:
        names = list(ELEMENTS.keys())
        print('complete -W "%s" %s' % (' '.join(names), CLI_NAME))
    else:
        module = ELEMENTS[args._pipemod]
        module.main(args)
    

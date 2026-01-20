"""Configuration and parsing utilities for site_pipeline."""

import argparse
import inspect
import yaml


class ArgumentParser(argparse.ArgumentParser):  # No direct usage found
    """A variant of ArgumentParser that allows the defaults
    to be overriden by values in a yaml config files. Thus the
    priority order becomes, from highest to lowest:

    1. Arguments passed on the command line
    2. The config file
    3. Defaults defined with add_argument()

    The config file is specified using the --config-file option,
    which this class adds automatically. It should therefore not
    be added manually.
    """
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--config-file", type=str, default=None,
            help="Optional yaml file containing overrides for the default values")

    def parse_args(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file", type=str, default=None)
        args, _ = parser.parse_known_args()
        if args.config_file != None:
            # Ok, we have a config file, parse it and use it to
            # replace our defaults
            with open(args.config_file, "r") as ifile:
                config = yaml.safe_load(ifile)
            for action in self._actions:
                try:
                    action.default  = config[action.dest]
                    # We mark it as non-required so we can run
                    # without even normally required arguments
                    # if they're provided by the config file
                    action.required = False
                except (KeyError, AttributeError) as e:
                    pass
        # Then parse again, taking into account any default update
        return argparse.ArgumentParser.parse_args(self, argv)


def parse_quantity(val, default_units=None):  # make_source_flags, make_uncal_beam_map
    """Convert an expression with units into an astropy Quantity.

    Args:
      val: the expression (see Notes).
      default_units: the units to assume if they are not provided in
        val.

    Returns:
      The astropy Quantity decoded from the argument.  Note the
      quantity is converted to the default_units, if they are
      provided.

    Notes:
      The default_units, if provided, should be "unit-like", by which
      we mean it is either:

      - An astropy Unit.
      - A string that astropy.units.Unit() can parse.

      The val can be any of the following:

      - A tuple (x, u) or list [x, u], where x is a float and u is
        unit-like.
      - A string (x), where x can be parsed by astropy.units.Quantity.
      - A float (x), but only if default_units is not None.

    Examples:
      >>> parse_quantity('100 arcsec')
      <Quantity 100. arcsec>

      >>> parse_quantity([12., 'deg'])
      <Quantity 12. deg>

      >>> parse_quantity('15 arcmin', 'deg')
      <Quantity 0.25 deg>

      >>> parse_quantity(100, 'm')
      <Quantity 100. m>

    """
    # Heavy to import, and we want this module to be fast to import
    # because it provides an ArgumentParser that should inform us
    # of incorrect arguments with as low latency as possible
    from astropy import units as u

    if default_units is not None:
        default_units = u.Unit(default_units)

    if isinstance(val, str):
        q = u.Quantity(val)
    elif isinstance(val, (list, tuple)):
        q = val[0] * u.Unit(val[1])
    elif isinstance(val, (float, int)):
        if default_units is None:
            raise ValueError(
                f"Cannot decode argument '{val}' without default_units.")
        q = val * default_units

    if default_units is not None:
        q = q.to(default_units)
    return q


def _filter_dict(d, bad_keys=['_stop_here']):
    """Filter out special keys from a dictionary.

    Helper function for lookup_conditional.
    """
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items()
            if k not in bad_keys}


def lookup_conditional(source, key, tags=None, default=KeyError):  # make_source_flags, make_uncal_beam_map
    """Lookup a value in a dict, with the possibility of descending
    through nested dictionaries using tags provided by the user.

    This function returns the returns source[key] unless source[key]
    is a dict, in which case the tags (a list of strings) are each
    tested in the dict to see if they lead to a sub-setting.

    For example, if the source dictionary is {'number': {'a': 1, 'b':
    2}} and the user requests key 'number', with tags=['a'], then the
    returned value will be 1.

    If you want a dict to be returned literally, and not crawled
    further, include a dummy key '_stop_here', with arbitrary value
    (this key will be removed from the result before returning to the
    user).

    The key '_default' will always cause a match, even if none of the
    other tags match.  (This _default value also becomes the default
    if further recursion fails to yield an exact match.)

    Args:
      source (dict): The parameter tree to search.
      key (str): The key to terminate the search on.
      tags (list of str or None): tags that may be auto-descended.
      default: Value to return if the search does not resolve.  The
        special value KeyError will instead cause a KeyError to be
        raised if the search is not resolved.

    Examples::

      source = {
        'my_param': {
          '_default': 100.,
          'f150': 90.
        }
      }

      lookup_conditional(source, 'my_param')
        => 100.

      lookup_conditional(source, 'my_param', tags=['f090'])
        => 100.

      lookup_conditional(source, 'my_param', tags=['f150'])
        => 90.

      lookup_conditional(source, 'my_other_param')
        KeyError!

      lookup_conditional(source, 'my_other_param', default=0)
        => 0

      # Note _default takes precedence over default argument.
      lookup_conditional(source, 'my_param', default=0)
        => 100.

      # Nested example:
      source = {
        'fit_params': {
          '_default': {
            'a': 12,
            'b': 100,
            '_stop_here': None,  # don't descend any further.
          },
          'f150': {
            'SAT': {
              'a': 1000,
              'b': 1200,
              '_stop_here': None,
            },
            'LAT': {
              'a': 1,
              'b': 2,
              '_stop_here': None,
            },
          },
        },
      }

      lookup_conditional(source, 'fit_params', tags=['f150', 'LAT'])
        => {'a': 1, 'b': 2}

      lookup_conditional(source, 'fit_params', tags=['LAT'])
        => {'a': 12, 'b': 100}

      lookup_conditional(source, 'fit_params', tags=['f150'])
        => {'a': 12, 'b': 100}

    """
    if tags is None:
        tags = []
    if key is not None:
        # On entry, key is not None.
        result = default
        if key in source:
            result = lookup_conditional(source[key], None, tags=tags, default=default)
        if inspect.isclass(result) and issubclass(result, Exception):
            raise result(f"Failed to find key '{key}' in {source}")
        return result
    else:
        # This block is entered on recursion.
        if not isinstance(source, dict):
            return source
        if '_stop_here' in source:
            return _filter_dict(source)
        # Update default?
        if '_default' in source:
            default = _filter_dict(source['_default'])
        # Find a tag.
        for t in tags:
            if t in source:
                return lookup_conditional(source[t], None, tags=tags, default=default)
        return default

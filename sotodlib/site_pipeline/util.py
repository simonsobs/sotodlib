import math
import os
import inspect
import logging
import time
import sys
import copy
import argparse
import yaml
import copy

class ArchivePolicy:
    """Storage policy assistance.  Helps to determine the HDF5
    filename and dataset name for a result.

    Make me better!

    """
    @staticmethod
    def from_params(params):
        if params['type'] == 'simple':
            return ArchivePolicy(**params)
        if params['type'] == 'directory':
            return DirectoryArchivePolicy(**params)
        raise ValueError('No handler for "type"="%s"' % params['type'])

    def __init__(self, **kwargs):
        self.filename = kwargs['filename']

    def get_dest(self, product_id):
        """Returns (hdf_filename, dataset_addr).

        """
        return self.filename, product_id

class ArgumentParser(argparse.ArgumentParser):
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

class DirectoryArchivePolicy:
    """Storage policy for stuff organized directly on the filesystem.

    """
    def __init__(self, **kwargs):
        self.root_dir = kwargs['root_dir']
        self.pattern = kwargs['pattern']

    def get_dest(self, **kw):
        """Returns full path to destination directory.

        """
        return os.path.join(self.root_dir, self.pattern.format(**kw))


def parse_quantity(val, default_units=None):
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
    if not isinstance(d, dict):
        return d
    # Support for lookup_conditional
    return {k: v for k, v in d.items()
            if k not in bad_keys}


def lookup_conditional(source, key, tags=None, default=KeyError):
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

class _ReltimeFormatter(logging.Formatter):
    def __init__(self, *args, t0=None, **kw):
        super().__init__(*args, **kw)
        if t0 is None:
            t0 = time.time()
        self.start_time = t0

    def formatTime(self, record, datefmt=None):
        if datefmt is None:
            datefmt = '%8.3f'
        return datefmt % (record.created - self.start_time)

def init_logger(name, announce='', verbosity=2, logger=None):
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

def main_launcher(main_func, parser_func, args=None):
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

def get_obslist(context, query=None, obs_id=None, min_ctime=None, max_ctime=None, 
                update_delay=None, tags=None, planet_obs=False):
    """Query the obs database with a given query.
    
    Parameters
    ----------
    context : core.Context
        The context to use for the obsdb.
    query : str, optional
        A query string for the obsdb.
    obs_id : str, optional
        The specific obsid to retrieve.
    min_ctime : int, optional
        The minimum ctime of obs to retrieve.
    max_ctime : int, optional
        The maximum ctime of obs to retrieve.
    update_delay : int, optional
        The number of days to subtract from the current time to set the minimum ctime.
    tags : list of str, optional
        A list of tags to use for the query.
    planet_obs : bool, optional
        If True, format query and tags for planet obs.
    
    Returns
    -------
    obs_list : list
        The list of obs found from the query.
    """
    if (min_ctime is None) and (update_delay is not None):
        # If min_ctime is provided it will use that..
        # Otherwise it will use update_delay to set min_ctime.
        min_ctime = int(time.time()) - update_delay*86400

    if obs_id is not None:
        tot_query = f"obs_id=='{obs_id}'"
    else:
        tot_query = "and "
        if min_ctime is not None:
            tot_query += f"timestamp>={min_ctime} and "
        if max_ctime is not None:
            tot_query += f"timestamp<={max_ctime} and "
        if query is not None:
            tot_query += query + " and "
        tot_query = tot_query[4:-4]
        if tot_query=="":
            tot_query="1"

    if not(tags is None):
        for i, tag in enumerate(tags):
            tags[i] = tag.lower()
            if '=' not in tag:
                tags[i] += '=1'

    if planet_obs:
        obs_list = []
        for tag in tags:
            obs_list.extend(context.obsdb.query(tot_query, tags=[tag]))
    else:
        obs_list = context.obsdb.query(tot_query, tags=tags)
    
    return obs_list

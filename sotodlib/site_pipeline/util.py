import math
import os
import inspect
import logging
import time

from .. import core

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


def parse_angle(angle, default_units='deg', desired_units='deg'):
    """Convert an expression for an angle into a float in the desired
    angle units.  This accepts floats or tuples that also state the
    unit.  For example:

      parse_angle(123.)
        => 123.

      parse_angle((60, 'arcmin'))
        => 1.

      parse_angle(1, desired_units='rad')
        => 0.017453292519943295

      parse_angle(1, default_units='rad')
        => 57.29577951308232

    The units must be one of ['rad', 'deg', 'arcmin', 'arcsec'].

    """
    units = {
        'rad': 1.,
        'deg': math.pi/180,
        'arcmin': math.pi/180/60,
        'arcsec': math.pi/180/60/60,
    }
    try:
        angle = float(angle)
    except TypeError:
        angle, default_units = angle  # e.g. (12, 'arcmin')
    return angle * units[default_units] / units[desired_units]


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

def init_logger(name, announce=''):
    """Configure and return a logger for site_pipeline elements.  It is
    disconnected from general sotodlib (propagate=False) and displays
    relative instead of absolute timestamps.

    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = _ReltimeFormatter('%(asctime)s: %(message)s (%(levelname)s)')

    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    i, r = formatter.start_time // 1, formatter.start_time % 1
    text = (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(i))
            + (',%03d' % (r*1000)))
    logger.info(f'{announce}Log timestamps are relative to {text}')

    return logger

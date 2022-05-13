from collections import OrderedDict as odict
import yaml
import os
import importlib
import logging

from . import metadata
from .axisman import AxisManager

logger = logging.getLogger(__name__)

class Context(odict):
    # Sets of special handlers may be registered in this class variable, then
    # requested by name in the context.yaml key "context_hooks".
    hook_sets = {}

    def __init__(self, filename=None, site_file=None, user_file=None,
                 data=None, load_list='all'):
        """Construct a Context object.  Note this is an ordereddict with a few
        attributes added on.

        Args:

          filename (str): Path to the dataset context file.  If None,
            that's fine.
          site_file (str): Path to the site file.  If None, then the
            value of SOTODLIB_SITECONFIG environment variable is used;
            unless that's unset in which case the file site.yaml in
            the current directory will be used.
          user_file (str): Path to the user file.  If None, then the
            value of SOTODLIB_USERCONFIG environment variable is used;
            unless that's unset in which case the file
            ~/.sotodlib.yaml will be used.
          data (dict or None): Optional dict of context data to merge
            in, after loading the site, user and main context files.
            Note the data are merged in with the usual rules (so items
            in data['tags'] will me merged into self['tags'].)
          load_list (str or list): A list of databases to load; some
            combination of 'obsdb', 'detdb', 'obsfiledb', or the
            string 'all' to load all of them (default).

        """
        super().__init__()
        # Start with site and user config.
        site_ok, site_file, site_cfg = _read_cfg(
            site_file, 'SOTODLIB_SITECONFIG',
            os.path.join(os.getcwd(), 'site.yaml'))
        logger.info(f'Using site_file={site_file}.')
        user_ok, user_file, user_cfg = _read_cfg(
            user_file, 'SOTODLIB_USERCONFIG',
            os.path.expanduser('~/.sotodlib.yaml'))
        logger.info(f'Using user_file={user_file}.')

        self.update(site_cfg)
        self.update_context(user_cfg)

        ok, full_filename, context_cfg = _read_cfg(filename)
        if filename is not None and not ok:
            raise RuntimeError(
                'Could not load requested context file %s' % filename)
        logger.info(f'Using context_file={full_filename}.')
        self.update_context(context_cfg)

        # Update with anything the user passed in.
        if data is not None:
            self.update_context(data)

        self.site_file = site_file
        self.user_file = user_file
        self.filename = full_filename

        self.obsdb = None
        self.detdb = None
        self.obsfiledb = None
        self.obs_detdb = None

        for to_import in self.get('imports', []):
            importlib.import_module(to_import)

        # Activate the requested hook set
        if self.get('context_hooks'):
            self._hooks = self.hook_sets[self['context_hooks']]
        else:
            self._hooks = {}

        # Check-default 'tags' dict.
        self['tags'] = self._get_warn_missing('tags', {})

        # Perform recursive substitution on strings defined in tags.
        self._subst(self)

        # Load basic databases.
        self.reload(load_list)

        # Call a post-processing hook before returning to user?
        self._call_hook('on-context-ready')

    def _call_hook(self, hook_key, *args, **kwargs):
        hook_func = self._hooks.get(hook_key)
        if hook_func is None:
            return
        logger.info('Calling hook for %s: %s' % (hook_key, hook_func))
        hook_func(self, *args, **kwargs)

    def _subst(self, dest, max_recursion=20):
        # Do string substitution of all our tags into dest (in-place
        # if dest is a dict).
        assert(max_recursion > 0)  # Too deep this dictionary.
        if isinstance(dest, str):
            # Keep subbing until it doesn't change any more...
            new = dest.format(**self['tags'])
            while dest != new:
                dest = new
                new = dest.format(**self['tags'])
            return dest
        if isinstance(dest, list):
            return [self._subst(x) for x in dest]
        if isinstance(dest, tuple):
            return (self._subst(x) for x in dest)
        if isinstance(dest, dict):
            for k, v in dest.items():
                dest[k] = self._subst(v, max_recursion-1)
            return dest
        return dest

    def _get_warn_missing(self, k, default=None):
        if not k in self:
            logger.warning(f'Key "{k}" not present in context.')
            return default
        return self[k]

    def update_context(self, new_stuff):
        appendable = ['metadata']
        mergeable = ['tags']

        for k, v in new_stuff.items():
            if k in appendable and k in self:
                self[k].extend(v)
            elif k in mergeable and k in self:
                self[k].update(v)
            else:
                self[k] = v

    def reload(self, load_list='all'):
        """Load (or reload) certain databases associated with this dataset.
        (Note we don't load any per-observation metadata here.)

        """
        # Metadata support databases.
        for key, cls in [('detdb', metadata.DetDb),
                         ('obsdb', metadata.ObsDb),
                         ('obsfiledb', metadata.ObsFileDb)]:
            if (load_list == 'all' or key in load_list) and key in self:
                db_file = self[key]
                if not db_file.startswith('/'):
                    # Relative to context file.
                    db_file = os.path.join(os.path.split(self.filename)[0], db_file)
                db_file = os.path.abspath(db_file)
                logger.info(f'Loading {key} from {self[key]} -> {db_file}.')
                try:
                    db = cls.from_file(db_file, force_new_db=False)
                except Exception as e:
                    logger.error(f'DB failure when loading {key} from {self[key]} -> {db_file}\n')
                    raise e
                setattr(self, key, db)
        # The metadata loader.
        if load_list == 'all' or 'loader' in load_list:
            self.loader \
                = metadata.SuperLoader(self)

    def get_obs(self, obs_id=None, dets=None, detsets=None,
                free_tags=None, samples=None, no_signal=None,
                loader_type=None, filename=None, meta=None):
        """Load TOD and supporting metadata for some observation.

        Args:
          obs_id (multiple): The observation to load (see Notes).
          dets (list): The detectors to read (default: all).
          detsets (list): The detsets to read (default: all).
          free_tags (list): Strings to match against the
            obs_colon_tags fields for detector restrictions.
          samples (tuple of ints): The start and stop sample indices
            (default: all).
          no_signal (bool): If True, the .signal will be set to None.
            This is a way to get the axes and pointing info without
            the (large) TOD blob.
          loader_type (str): Name of the registered TOD loader
            function to use (this will override whatever is specified
            in context.yaml).
          filename (str): Instead of loading based on obs_id, figure
            out what observation, detectors, and sample range sample
            range are covered by this file and load that (along with
            supporting metadata).
          meta (AxisManager): Instead of loading based on obs_id, use
            an AxisManager returned by get_meta (with possible
            additional restrictions applied) as the way to determine
            obs_id, sample range, and detectors to load.  Using this
            suppresses metadata loading through the present function.

        Notes:

          The observation to load can be specified through the obs_id
          argument, or through filename (if obs_id is None).  If
          obs_id is a dict, it will be interpreted as the output from
          an ObsDb query and expected to contain the key
          ``'obs:obs_id'``.  If obs_id is a string, then it will first
          be split on the ':' character, and the result converted to
          {'obs:obs_id': tokens[0]}; the other tokens are appended to
          "free_tags".

          Finally if obs_id is an AxisManager, it's treated as if it
          had been passed in through the meta argument.

          The detector subset to load is determined by the dets and
          detsets arguments and any free_tags.  Arbitrary fields known
          to det_info can be passed through dets, as lists or
          individual strings to match.  For example::

            dets={'dets:readout_id': ['my_det0', 'my_det6']}

          which is equivalent to

            dets=['my_det0', 'my_det6']

          Batches can be selected using other det_info fields, e.g.::

            dets={'dets:band': 'f150', 'dets:wafer_slot': ['w21', 'w22']}

          You activate detsets through::

            detsets=['fileset0', 'fileset1']

          which is equivalent to including

            dets={'dets:detset': ['fileset0', 'fileset1']}

          The sample range to load is determined by the samples
          argument.  Use Python start/stop indexing; for example
          samples=(0, -2) will try to read all but the last two
          samples and samples=(100, None) will read all samples except
          the first 100.

          Note on interface stability: it's safe to assume that the
          first arg will always be obs_id.  But all other arguments
          should be passed by keyword.

        """
        meta = None
        detspec = {}

        if filename is not None:
            if obs_id is not None or detsets is not None or samples is not None:
                logger.warning(f'Passing filename={filename} to get_obs causes '
                               f'obs_id={obs_id} and detsets={detsets} and '
                               f'samples={samples} be ignored.')
            # Resolve this to an obs_id / detset combo.
            info = self.obsfiledb.lookup_file(filename, resolve_paths=True)
            obs_id = info['obs_id']
            detsets = info['detsets']
            if info['sample_range'] is None or None in info['sample_range']:
                samples = None
                logger.warning('Due to incomplete ObsFileDb info, passing filename=... '
                               'will cause *all* files for the detset covered '
                               'by that file to be loaded.')
            else:
                samples = info['sample_range']

        # Handle some special cases for obs_id; at the end of this
        # checks and conversion, obs_id should be a string.

        if isinstance(obs_id, AxisManager):
            # Just move that to the meta argument.
            assert(meta is None)  # obs_id and meta both passed in?
            obs_id, meta = None, obs_id

        elif isinstance(obs_id, dict):
            obs_id = obs_id['obs_id']  # You passed in a dict.

        # If the obs_id has colon-coded free tags, extract them.
        free_tags = list(free_tags) if free_tags else []
        if isinstance(obs_id, str) and ':' in obs_id:
            tokens = obs_id.split(':')
            obs_id = tokens[0]
            free_tags.extend(tokens[1:])

        if meta is None:
            # Load the metadata.
            request = {'obs:obs_id': obs_id}
            if detsets is not None:
                request['dets:detset'] = detsets
            if isinstance(dets, list):
                request['dets:readout_id'] = dets
            elif dets is not None:
                request.update({'dets:' + k: v for k, v in dets.items()})
            meta = self.get_meta(request, free_tags=free_tags)

        # Use the obs_id, dets, and samples (if possible) from meta.
        obs_id = meta['obs_info']['obs_id']
        dets = list(meta.det_info['readout_id'])
        if samples is None and 'samps' in meta:
            samples = (meta.samps.offset, meta.samps.offset + meta.samps.count)

        # Make sure standard obsloaders are registered ...
        from ..io import load as _

        # Load TOD.
        if loader_type is None:
            loader_type = self.get('obs_loader_type', 'default')
        loader_func = OBSLOADER_REGISTRY[loader_type]  # Register your loader?
        aman = loader_func(self.obsfiledb, obs_id, dets,
                           samples=samples, no_signal=no_signal)

        if aman is None:
            return meta
        if meta is not None:
            aman.merge(meta)
        return aman

    def get_meta(self, request, samples=None, check=False,
                 free_tags=None, free_tag_fields=None,
                 det_info_scan=False):
        """Load and return the supporting metadata for an observation.  The
        request parameter can be a simple observation id as a string,
        or else a request dict like the kind passed in from get_obs.

        """
        if free_tag_fields is None:
            free_tag_fields = self.get('obs_colon_tags', [])

        free_tags = list(free_tags) if free_tags else []
        if isinstance(request, str):
            if len(free_tag_fields):
                obs_id = request.split(':')[0]
                free_tags.extend(request.split(':')[1:])
            else:
                obs_id = request
            request = {'obs:obs_id': obs_id}

        # Call a hook after preparing obs_id but before loading obs
        obs_id = request['obs:obs_id']
        self._call_hook('before-use-detdb', obs_id=obs_id)

        # Identify whether we should use a detdb or an obs_detdb
        # If there is an obs_detdb, use that.
        # Otherwise, use whatever is in self.detdb, even if that is None.
        if self.obs_detdb is not None:
            detdb = self.obs_detdb
        else:
            detdb = self.detdb

        # Initialize det_info, starting with detdb.
        if detdb is not None:
            det_info = detdb.props()

            # Backwards compatibility -- add "readout_id" if not found.
            if 'readout_id' not in det_info.keys:
                logger.warning('DetDb does not contain "readout_id"; aliasing from "name".')
                det_info.merge(metadata.ResultSet(
                    ['readout_id'], [(name,) for name in det_info['name']]))

        # Incorporate detset info from obsfiledb.
        detsets_info = self.obsfiledb.get_det_table(obs_id)
        if det_info is None:
            det_info = detsets_info
        else:
            det_info = metadata.merge_det_info(det_info, detsets_info,
                                               ['readout_id'])

        metadata_list = self._get_warn_missing('metadata', [])
        return self.loader.load(metadata_list, request, det_info=det_info, check=check,
                                free_tags=free_tags, free_tag_fields=free_tag_fields,
                                det_info_scan=det_info_scan)

    def check_meta(self, request):
        """Check for existence of required metadata.

        """
        return self.get_meta(request, check=True)


def _read_cfg(filename=None, envvar=None, default=None):
    """Load a YAML file.  If filename is None, use the filename specified
    in the environment variable called envvar.  If that is not defined
    or decodes to None or an empty string, use the filename specified
    in default.

    Returns (ok, full_path, data) where ok is a boolean indicating
    whether the file at full_path was found on the file-system,
    full_path is the full path to the resolved filename (or None if
    not resolved), and data is the OrderedDict containing the data (or
    {} if not decoded).

    """
    if filename is None and envvar is not None:
        filename = os.getenv(envvar)
        if filename is None or filename == '':
            filename = None
    if filename is None and default is not None:
        filename = default
    if filename is None:
        return False, None, odict()
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return False, filename, odict()
    return True, filename, yaml.safe_load(open(filename, 'r'))


def obsloader_template(db, obs_id, dets=None, prefix=None, samples=None,
                       no_signal=None,
                       **kwargs):
    """This function is here to document the API for "obsloader" functions
    used by the Context system.  "obsloader" functions are used to
    load time-ordered detector data (rather than supporting metadata)
    from file archives, and return an AxisManager.

    Args:
      db (ObsFileDB): The database supporting the data files.
      obs_id (str): The obs_id (as recognized by ObsFileDb).
      dets (list of str): The dets to load.  If None, all dets are
        loaded.  If an empty list, ancillary data for the observation
        is still loaded.
      samples (tuple): The (start, end) indices of samples which
        should be loaded.  If start is None, 0 is used.  If end is
        None, sample_count is used.  Passing None is equivalent to
        passing (None, None).
      prefix (str): The root address of the data files, if not already
        known to the ObsFileDb.  (This is passed through to ObsFileDb
        prefix= argument.)
      no_signal (bool): If True, loader should avoid reading signal
        data (if possible) and should set .signal=None in the output.
        Passing None is equivalent to passing False.

    Notes:
      This interface is subject to further extension.  When possible
      such extensions should take the form of optional arguments,
      whose default value is None and which are not activated except
      when needed.  This permits existing loaders to future-proof
      themselves by including ``**kwargs`` in the function signature
      but raising an exception if kwargs contains anything strange.
      See the body of this example function for template code to
      reject unexpected kwargs.

    Returns:
      An AxisManager with the data.

    """
    if any([v is not None for v in kwargs.values()]):
        raise RuntimeError(
            f"This loader function does not understand these kwargs: f{kwargs}")
    raise NotImplementedError("This is just a template function.")


#: OBSLOADER_REGISTRY will be accessed by the Context system to load
#: TOD.  The function obsloader_template, in this module, shows the
#: signature and describes the interface.

OBSLOADER_REGISTRY = {}

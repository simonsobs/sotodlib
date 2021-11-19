from collections import OrderedDict as odict
import yaml
import os
import importlib
import logging

from . import metadata

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
                loader_type=None, logic_only=False):
        """Load TOD and supporting metadata for a particular observation id.
        The detectors to read can be specified through colon-coding in
        obs_id, through dets, or through detsets.

        After figuring out what detectors you want, loader_type will
        be used to fund a loader function in the OBSLOADER_REGISTRY.

        """
        detspec = {}

        # Handle the case that this is a row from a obsdb query.
        if isinstance(obs_id, dict):
            obs_id = obs_id['obs_id']  # You passed in a dict.

        # If the obs_id is colon-coded, decode them.
        if ':' in obs_id:
            tokens = obs_id.split(':')
            obs_id = tokens[0]
            allowed_fields = self.get('obs_colon_tags', [])
            # Create a map from option value to option key.
            prop_map = {}
            for f in allowed_fields[::-1]:
                for v in self.detdb.props(props=[f]).distinct()[f]:
                    prop_map[v] = f
            for t in tokens[1:]:
                prop = prop_map.get(t)
                if prop is None:
                    raise ValueError('obs_id included tag "%s" but that is not '
                                     'a value for any of DetDb.%s.' % (t, allowed_fields))
                if prop in detspec:
                    raise ValueError('obs_id included tag "%s" and that resulted '
                                     'in re-restriction on property "%s"' % (t, prop))
                detspec[prop] = t

        # Start the list of detector selectors.
        dets_selection = [detspec]

        # Did user give a list of dets (or detspec)?
        if dets is not None:
            dets_selection.append(dets)

        # Default detsets should be only those listed in obsfiledb
        if detsets is None and self.obsfiledb is not None:
            detsets = self.obsfiledb.get_detsets(obs_id)

        # Intersect with detectors allowed by the detsets argument?
        if detsets is not None:
            all_detsets = self.obsfiledb.get_detsets(obs_id)
            ddets = []
            for ds in detsets:
                if isinstance(ds, int):
                    # So user can pass in detsets=[0] as a shortcut.
                    ds = all_detsets[ds]
                ddets.extend(self.obsfiledb.get_dets(ds))
            dets_selection.append(ddets)

        # Make the final list of dets -- force resolve it to a list,
        # not a detspec.
        dets = self.detdb.intersect(*dets_selection,
                                    resolve=True)

        # The request to the metadata loader should include obs_id and
        # the detector selection.
        request = {'obs:obs_id': obs_id}
        request.update({'dets:'+k: v for k, v in detspec.items()})

        if logic_only:
            # Return the results of detector and obs resolution.
            return {'request': request,
                    'detspec': detspec,
                    'dets': dets}

        # How to load?
        if loader_type is None:
            loader_type = self.get('obs_loader_type', 'default')

        # Load metadata.
        metadata_list = self._get_warn_missing('metadata', [])
        # edge case when only one entry in metadata
        if not isinstance(metadata_list, list):
            raise ValueError(f"Expected metadata list not {type(metadata_list)}."
                             " Check .yaml Formatting")
        meta = self.loader.load(metadata_list, request)

        # Load TOD.
        from ..io.load import OBSLOADER_REGISTRY
        loader_func = OBSLOADER_REGISTRY[loader_type]  # Register your loader?
        aman = loader_func(self.obsfiledb, obs_id, dets)

        if aman is None:
            return meta
        if meta is not None:
            aman.merge(meta)
        return aman

    def get_meta(self, request):
        """Load and return the supporting metadata for an observation.  The
        request parameter can be a simple observation id as a string,
        or else a request dict like the kind passed in from get_obs.

        """
        if isinstance(request, str):
            request = {'obs:obs_id': request}
        metadata_list = self._get_warn_missing('metadata', [])
        return self.loader.load(metadata_list, request)

    def check_meta(self, request):
        """Check for existence of required metadata.

        """
        if isinstance(request, str):
            request = {'obs:obs_id': request}
        metadata_list = self._get_warn_missing('metadata', [])
        return self.loader.check(metadata_list, request)


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

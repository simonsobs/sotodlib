from collections import OrderedDict as odict
import yaml
import os

from ..  import metadata


class Context(odict):
    def __init__(self, filename=None, site_file=None, user_file=None,
                 load_list='all'):
        super().__init__()
        # Start with site and user config.
        site_ok, site_file, site_cfg = _read_cfg(
            site_file, 'SOTODLIB_SITECONFIG',
            os.path.join(os.getcwd(), 'site.yaml'))
        user_ok, user_file, user_cfg = _read_cfg(
            user_file, 'SOTODLIB_USERCONFIG',
            os.path.expanduser('~/.sotodlib.yaml'))

        self.update(site_cfg)
        self.update_context(user_cfg)

        ok, filename, context_cfg = _read_cfg(filename)
        if not ok:
            raise RuntimeError(
                'Could not load requested context file %s' % filename)
        self.update_context(context_cfg)

        self.site_file = site_file
        self.user_file = user_file
        self.filename = filename

        self._subst(self)

        self.obsdb = None
        self.detdb = None
        self.obsfiledb = None

        self.reload(load_list)

    def _subst(self, dest, max_recursion=20):
        # Do string substitution of all our tags into dest (in-place
        # if dest is a dict).
        assert(max_recursion > 0) # Too deep this dictionary.
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
        if load_list == 'all' or 'detdb' in load_list:
            self.detdb \
                = metadata.DetDB.from_file(self['detdb'])
        if load_list == 'all' or 'obsfiledb' in load_list:
            self.obsfiledb \
                = metadata.ObsFileDB.from_file(self['obsfiledb'])
        if load_list == 'all' or 'loader' in load_list:
            self.loader \
                = metadata.SuperLoader(self)

    def get_obs(self, obs_id=None, dets=None, detsets=None,
                loader_type=None):
        """Load TOD and supporting metadata for a particular observation id.
        The detectors to read can be specified through colon-coding in
        obs_id, through dets, or through detsets.

        After figuring out what detectors you want, loader_type will
        be used to fund a loader function in the OBSLOADER_REGISTRY.

        """
        detspec = {}
        if ':' in obs_id:
            tokens = obs_id.split(':')
            obs_id = tokens[0]
            allowed_fields = self.get('obs_colon_tags', [])
            options = self.detdb.props(props=allowed_fields).distinct()
            for f in allowed_fields:
                field_options = list(set(options[f]))
                for t in tokens:
                    if t in field_options:
                        detspec[f] = t

        # Start the list of detector selectors.
        dets_selection = [detspec]

        # Did user give a list of dets (or detspec)?
        if dets is not None:
            dets_selection.append(dets)

        # Intersect with detectors allowed by the detsets argument?
        if detsets is not None:
            ddets = []
            for ds in detsets:
                ddets.extend(self.obsfiledb.get_dets(ds))
            dets_selection.append(ddets)

        # Make the final list of dets -- force resolve it to a list,
        # not a detspec.
        dets = self.detdb.intersect(*dets_selection,
                                    resolve=True)

        # The request to the metadata loader should include obs_id and
        # the detector selection.
        request = {'obs:obs_id': obs_id}
        request.update(detspec)

        # Load metadata.
        meta = self.loader.load(self['metadata'][:], request)

        # How to load?
        if loader_type is None:
            loader_type = self.get('obs_loader_type', 'default')

        # Load TOD.
        from sotodlib.data.load import OBSLOADER_REGISTRY
        loader_func = OBSLOADER_REGISTRY[loader_type]  # Did you register your loader?
        aman = loader_func(self.obsfiledb, obs_id, dets)

        if aman is None:
            return meta
        if meta is not None:
            aman.merge(meta)
        return aman

    def get_meta(self, request):
        return self.loader.load(self['metadata'][:], request)


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

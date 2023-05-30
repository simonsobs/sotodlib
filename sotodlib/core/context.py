from collections import OrderedDict as odict
import yaml
import os
import importlib
import logging
import numpy as np

from . import metadata
from .axisman import AxisManager, OffsetAxis, AxisInterface

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
                = metadata.SuperLoader(self, obsdb=self.obsdb)

    def get_obs(self,
                obs_id=None,
                dets=None,
                samples=None,
                filename=None,
                detsets=None,
                meta=None,
                ignore_missing=None,
                free_tags=None,
                no_signal=None,
                loader_type=None,
    ):
        """Load TOD and supporting metadata for some observation.

        Most arguments to this function are also accepted by (and in
        fact passed directly to) :func:`get_meta`, but are documented
        here.

        Args:
          obs_id (multiple): The observation to load (see Notes).
          dets (list, array, dict or ResultSet): The detectors to
            read.  If None, all dets will be read.
          samples (tuple of ints): The start and stop sample indices.
            If None, read all samples.  (Note that some loader
            functions might not support this argument.)
          filename (str): The path to a file to load, instead of using
            obs_id.  It is still required that this file appear in the
            obsfiledb, but this shortcut will automatically determine
            the obs_id and the detector and sample range selections
            that correspond to this single file.
          detsets (list, array): The detsets to read (with None
            equivalent to requesting all detsets).
          meta (AxisManager): An AxisManager returned by get_meta
            (though possibly with additional axis restrictions
            applied) to use as a starting point for detector selection
            and sample range.  (This will eventually be passed back to
            get_meta in the meta= argument, to fill in any missing
            metadata fields.)
          free_tags (list): Strings to match against the
            obs_colon_tags fields for detector restrictions.
          ignore_missing (bool): If True, don't fail when a metadata
            item can't be loaded, just try to proceed without it.
          no_signal (bool): If True, the .signal will be set to None.
            This is a way to get the axes and pointing info without
            the (large) TOD blob.  Not all loaders may support this.
          loader_type (str): Name of the registered TOD loader
            function to use (this will override whatever is specified
            in context.yaml).

        Notes:
          It is acceptable to pass the ``obs_id`` argument by position
          (first), but all other arguments should be passed by
          keyword.

          The ``obs_id`` can be any of the following:

          - a string -- this is interpreted as the literal obs_id as
            used in the ObsDb and ObsFileDb.  Note however that this
            string may include "free tags" (see below).
          - a dict -- this is understood to be an ObsDb record, and
            the value under key 'obs_id' will be used as the obs_id
            (the other items will be ignored).
          - an AxisManager -- this is a short-hand for passing an
            object through meta=... .  I.e., ``get_obs(obs_is=axisman)``
            is treated the same way as ``get_obs(obs_id=None, meta=axisman)``.

          Detector subselection is achieved through the ``dets``
          argument.  If this is a dict, the keys must all be fields
          appearing in det_info.  Typically det_info will include at
          least readout_id and detset (this is the indexing
          information from ObsFileDb).  Some examples are::

            dets={'readout_id': ['det_00', 'det_01']}
            dets={'detset': 'wafer21'}
            dets={'band': ['f090']}
            dets={'detset': ['wafer21', 'wafer22'], 'band': ['f150']}

          Each value in ``dets`` can be a single item, or a list or
          numpy array of items.  The keys may include an optional
          'dets:' prefix.

          If ``dets`` is passed as a list or numpy array, that is
          equivalent to passing that value in through a dict with key
          'readout_id'; e.g.::

            dets=['det_00', 'det_01']

          You can instead pass a "det_info" ResultSet directly into
          the dets argument; that is equivalent to passing
          dets=det_info['readout_id'].  This is to accomodate the
          following sort of pattern::

            det_info = context.get_det_info(obs_id)
            det_info = det_info.subset(rows=(det_info['band'] == 'f090'))
            tod = context.get_obs(obs_id, dets=det_info)

          The sample range to load is determined by the samples
          argument.  Use Python start/stop indexing; for example
          samples=(0, -2) will try to read all but the last two
          samples and samples=(100, None) will read all samples except
          the first 100.

          When passing in ``meta``, the obs_id, detector list, and
          sample range will be extracted from that object.  It is an
          error to also specify ``obs_id``, ``dets``, ``samples``,
          ``filename``, or ``free_ags`` (but this could change).

        """
        meta = self.get_meta(obs_id=obs_id, dets=dets, samples=samples,
                             filename=filename, detsets=detsets, meta=meta,
                             free_tags=free_tags, ignore_missing=ignore_missing)

        # Use the obs_id, dets, and samples from meta.
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
        aman = loader_func(self.obsfiledb, obs_id, dets=dets,
                           samples=samples, no_signal=no_signal)
 
        if aman is None:
            return meta
        if meta is not None:
            if 'det_info' in aman and 'det_info' in meta:
                # If the loader added det_info, then perform a special
                # merge.  Duplicate keys should be avoided, because
                # checking the values are the same is annoying.
                _det_info = aman['det_info']
                del aman['det_info']
                _det_info.restrict_axes([meta.dets])
                for k in meta.det_info._fields:
                    if k in _det_info._fields:
                        try:
                            check = np.all([meta['det_info'][k] ==_det_info[k]])
                            if check:
                                _det_info.move(k, None)
                                continue
                        except Exception as e:
                            pass
                        logger.error(f'Key "{k}" is present in det_info returned by '
                                     f'observation loader as well as in metadata '
                                     f'databases; The two versions are not '
                                     f'comparable. dropping the loader version.')
                        _det_info.move(k, None)
                meta.det_info.merge(_det_info)
            aman.merge(meta)
        return aman

    def get_meta(self,
                 obs_id=None,
                 dets=None,
                 samples=None,
                 filename=None,
                 detsets=None,
                 meta=None,
                 free_tags=None,
                 check=False,
                 ignore_missing=False,
                 det_info_scan=False):
        """Load supporting metadata for an observation and return it in an
        AxisManager.

        The arguments shared with :func:`get_obs` (``obs_id``,
        ``dets``, ``samples`, ``filename``, ``detsets`, ``meta``,
        ``free_tags``) have the same meaning as in that function and
        are treated in the same way.

        Args:
          check (bool): If True, run in a check mode where an attempt
            is made to load each metadata entry, but the results are
            not kept and instead the function returns a report on what
            entries could / could not be loaded
          det_info_scan (bool): If True, only process the metadata
            entries that explicitly modify det_info.

        Returns:
          AxisManager with a .dets LabelAxis and .det_info and
          .obs_info entries.  If samples is specified, or if any
          metadata loads triggered its creation, then the .samps
          OffsetAxis is also created.

        Notes:
          When ``meta`` is passed in, it will be used to figure out
          the obs_id and detector and sample selections; however a new
          metadata AxisManager is returned.  Users should not rely on
          this; future improvements might modify meta in place, and
          try to re-use entries already present rather than loading
          them a second time.

        """
        def _warn_conflict(preamble, **kwargs):
            fails = {k: v for k, v in kwargs.items() if v is not None}
            if len(fails):
                logger.warning(f'{preamble}: arguments ignored: {fails}')

        free_tag_fields = self.get('obs_colon_tags', [])
        free_tags = list(free_tags) if free_tags else []

        if filename is not None:
            _warn_conflict(
                'Passing filename={filename} to get_obs with incompatible other args',
                obs_id=obs_id, detsets=detsets, samples=samples)
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
            _warn_conflict(
                'Argument obs_id=<AxisManager> is incompatible with other args',
                meta=meta)
            obs_id, meta = None, obs_id

        elif isinstance(obs_id, dict):
            obs_id = obs_id['obs_id']  # You passed in a dict.

        elif isinstance(obs_id, str):
            # If the obs_id has colon-coded free tags, extract them.
            if ':' in obs_id:
                tokens = obs_id.split(':')
                obs_id = tokens[0]
                free_tags.extend(tokens[1:])

        if meta is not None:
            _warn_conflict(
                'Argument meta=<AxisManager> causes det/sample args to be ignored',
                samples=samples, dets=dets, detsets=detsets)
            obs_id = meta.obs_info['obs_id']
            dets = {'dets:readout_id': list(meta.dets.vals)}
            if 'samps' in meta:
                samples = meta.samps.offset, meta.samps.offset + meta.samps.count

        # Call a hook after preparing obs_id but before loading obs
        self._call_hook('before-use-detdb', obs_id=obs_id)

        # Identify whether we should use a detdb or an obs_detdb
        # If there is an obs_detdb, use that.
        # Otherwise, use whatever is in self.detdb, even if that is None.
        if self.obs_detdb is not None:
            detdb = self.obs_detdb
        else:
            detdb = self.detdb

        # Initialize det_info, starting with detdb.
        det_info = None
        if detdb is not None:
            det_info = detdb.props()

            # Backwards compatibility -- add "readout_id" if not found.
            if 'readout_id' not in det_info.keys:
                logger.warning('DetDb does not contain "readout_id"; aliasing from "name".')
                det_info.merge(metadata.ResultSet(
                    ['readout_id'], [(name,) for name in det_info['name']]))

        # Incorporate detset info from obsfiledb.
        detsets_info = self.obsfiledb.get_det_table(obs_id)
        det_info = metadata.merge_det_info(det_info, detsets_info,
                                           ['readout_id'])

        # Make the request for SuperLoader
        request = {'obs:obs_id': obs_id}
        if detsets is not None:
            request['dets:detset'] = detsets

        # Convert dets argument to request entry(s)
        if isinstance(dets, dict):
            for k, v in dets.items():
                if not k.startswith('dets:'):
                    k = 'dets:' + k
                if k in request:
                    raise ValueError(f'Duplicate specification of dets field "{k}"')
                request[k] = v
        elif isinstance(dets, metadata.ResultSet):
            request['dets:readout_id'] = dets['readout_id']
        elif hasattr(dets, '__getitem__'):
            # lists, tuples, arrays ...
            request['dets:readout_id'] = dets
        elif dets is not None:
            # Try a cast ...
            request['dets:readout_id'] = list(dets)

        metadata_list = self._get_warn_missing('metadata', [])
        meta = self.loader.load(metadata_list, request, det_info=det_info, check=check,
                                free_tags=free_tags, free_tag_fields=free_tag_fields,
                                det_info_scan=det_info_scan, ignore_missing=ignore_missing)
        if check:
            return meta

        if samples is not None:
            if 'samps' in meta:
                meta.restrict('samps', slice(*samples))
            else:
                start, stop = samples
                assert(start >= 0 and stop >= 0)  # This could be loosened using obsfiledb
                axm = AxisManager(OffsetAxis('samps', stop - start, start, obs_id))
                meta = meta.merge(axm)
        return meta

    def get_det_info(self,
                     obs_id=None,
                     dets=None,
                     samples=None,
                     filename=None,
                     detsets=None,
                     meta=None,
                     free_tags=None):
        """Pass all arguments to :func:`get_meta(det_info_scan=True)`, and
        then return only the det_info, as a ResultSet.

        """
        if meta is None:
            meta = self.get_meta(obs_id=obs_id, dets=dets, samples=samples,
                                 filename=filename, detsets=detsets, free_tags=free_tags,
                                 det_info_scan=True)
        # Convert
        def _unpack(aman):
            items = []
            for k in aman.keys():
                if isinstance(aman[k], AxisManager):
                    sub_items = _unpack(aman[k])
                    for _k, _c in sub_items:
                        items.append((f'{k}.{_k}', _c))
                elif isinstance(aman[k], AxisInterface):
                    pass
                else:
                    items.append((k, aman[k]))
            return items

        items = _unpack(meta.det_info)
        return metadata.ResultSet([k for k, v in items],
                                  zip(*[v for k, v in items]))


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

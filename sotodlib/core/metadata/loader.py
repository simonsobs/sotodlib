from sotodlib import core
import os

REGISTRY = {
    '_default': 'DefaultHdf'
}


class SuperLoader:
    def __init__(self, context=None, detdb=None, obsdb=None, working_dir=None):
        """Metadata batch loader.

        Args:
          context (Context): context, from which detdb and obsdb will
            be pulled unless they are specified explicitly.
          detdb (DetDb): detdb to use when resolving detector axis.
          obsdb (ObsDb): obsdb to use when resolving obs axis.
          working_dir (str): base directory for any metadata specified
            as relative paths.  If None, but if context is not None,
            then the path of context.filename is used.

        """
        if context is not None:
            if detdb is None:
                detdb = context.detdb
            if obsdb is None:
                obsdb = context.obsdb
            if working_dir is None:
                working_dir = os.path.split(context.filename)[0]
        self.detdb = detdb
        self.obsdb = obsdb
        self.manifest_cache = {}
        self.working_dir = working_dir

    @staticmethod
    def register_metadata(name, loader_class):
        """Globally register a metadata "Loader Class".

        Args:
          name (str): Name under which to register the loader.
            Metadata archives will request the loader class using
            this name.
          loader_class: Metadata loader class.

        """
        REGISTRY[name] = loader_class

    def load_raw(self, spec_list, request, detdb=None):
        """Loads metadata objects and returns them in their Natural
        containers.

        Args:
          spec_list (list of dict): A list of metadata specification
            dictionaries.
          request (dict): A metadata request dictionary.
          detdb (core.metadata.DetDb): A DetDb-like object for use
            loading metadata.

        Notes:
          Each entry in spec_list must be a dictionary with the
          following keys:

            ``db`` (str)
                The path to a ManifestDb file.

            ``name`` (str)
                Naively, the name to give to the extracted data.  But
                the string may encode more complicated instructions,
                which are understood by the Unpacker class in this
                module.

            ``loader`` (str, optional)
                The name of the loader class to use when loading the
                data.  This is normally unnecessary, and will override
                any value declared in the ManifestDb.

          Before being passed to the loader, any filenames returned
          from the ManifestDb will be resolved relative to the ``db``
          location, unless the filename begins with a '/' in which
          case it is treated as an absolute path.  Similarly, the
          ``db`` path should be a relative location relative to the
          context file, or else an absolute path starting with '/'.

        Returns:
          A list of tuples (unpacker, item), corresponding to ecah
          entry in spec_list.  The unpacker is an Unpacker object
          created based on the 'name' field.  The item is the metadata
          in its native format (which could be a ResultSet or
          AxisManager), with all restrictions specified in request
          already applied.

        """
        if detdb is None:
            detdb = self.detdb
        items = []
        for spec_dict in spec_list:
            dbfile = spec_dict['db']
            if dbfile[0] != '/':
                dbfile = os.path.join(self.working_dir, dbfile)
            dbfile_path = os.path.split(dbfile)[0]
            names = spec_dict['name']
            loader = spec_dict.get('loader', None)

            # Load the database, match the request,
            if dbfile not in self.manifest_cache:
                if dbfile.endswith('sqlite'):
                    man = core.metadata.ManifestDb.readonly(dbfile)
                else:
                    man = core.metadata.ManifestDb.from_file(dbfile)
                self.manifest_cache[dbfile] = man
            man = self.manifest_cache[dbfile]

            # Provide any extrinsic boosting.  Downstream products
            # (like an HDF5 table) might require some parameters from
            # the ObsDb, and we can't tell that from here.  So query
            # what you can, and pass it along.
            if self.obsdb is not None and 'obs:obs_id' in request:
                obs_info = self.obsdb.get(request['obs:obs_id'], add_prefix='obs:')
                if obs_info is not None:
                    obs_info.update(request)
                    request = obs_info

            missing_keys = man.scheme.get_required_params()
            for k in request.keys():
                if k in missing_keys:
                    missing_keys.remove(k)
            obs_keys = [k for k in missing_keys if k.startswith('obs:')]
            if len(obs_keys):
                if self.obsdb is None:
                    reason = 'no ObsDb was passed in'
                elif 'obs:obs_id' not in request:
                    reason = 'obs_id was not specified in the request'
                elif obs_info is None:
                    reason = 'ObsDb has no info on %s' % request['obs:obs_id']
                else:
                    reason = 'ObsDb does not provide a value for %s' % obs_keys
                raise RuntimeError(
                    'Metadata request could not be constructed because: %s' % reason)

            try:
                index_lines = man.match(request, multi=True)
            except Exception as e:
                # Catch any errors and provide a bunch of context to
                # help user fix their config.
                text = str(e)
                raise RuntimeError(
                    'An exception was raised while decoding the following spec:\n'
                    + '  ' + str(spec_dict) + '\n'
                    + 'with the following request:\n'
                    + '  ' + str(request) + '\n'
                    + 'The exception is:\n  %s' % text)

            # Make files relative to db location.
            for line in index_lines:
                if 'filename' in line:
                    line['filename'] = os.path.join(dbfile_path, line['filename'])

            # Load and reduce each Index line
            results = []
            for index_line in index_lines:
                # Augment the index_line with info from the request.
                skip_this = False
                for k in request:
                    if k in index_line:
                        if request[k] != index_line[k]:
                            skip_this = True
                if skip_this:
                    continue
                index_line.update(request)
                if loader is None:
                    loader = index_line.get('loader')
                if loader is None:
                    loader = REGISTRY['_default']
                try:
                    loader_class = REGISTRY[loader]
                except KeyError:
                    raise RuntimeError('No metadata loader registered under name "%s"' % loader)
                loader_object = loader_class(detdb=detdb, obsdb=self.obsdb)
                mi1 = loader_object.from_loadspec(index_line)
                # restrict to index_line...
                if (detdb is None and
                    len([k for k in index_line if k.startswith('dets:')])):
                    raise ValueError(f"Metadata not loadable without detdb: {index_line}")
                mi2 = mi1.restrict_dets(index_line, detdb=detdb)
                results.append(mi2)

            # Check that we got results, then combine them in to single ResultSet.
            assert(len(results) > 0)
            result = results[0].concatenate(results)

            # Get list of fields and decode name map.
            if isinstance(result, core.AxisManager):
                fields = list(result._fields.keys())
            else:
                fields = result.keys
            unpackers = Unpacker.decode(names, fields)

            items.append((unpackers, result))
        return items

    def unpack(self, packed_items, dest=None, detdb=None):
        """Unpack items from packed_items, and return then in a single
        AxisManager.

        """
        if detdb is None:
            detdb = self.detdb
        if dest is None:
            dest = core.AxisManager()
        for unpackers, metadata_instance in packed_items:
            # Convert to AxisManager...
            if isinstance(metadata_instance, core.AxisManager):
                child_axes = metadata_instance
            else:
                child_axes = metadata_instance.axismanager(detdb=detdb)
            fields_to_delete = list(child_axes._fields.keys())
            # Unpack to requested field names.
            for unpack in unpackers:
                if unpack.src is None:
                    dest.wrap(unpack.dest, child_axes)
                    break
                else:
                    fields_to_delete.remove(unpack.src)
                    if unpack.src != unpack.dest:
                        child_axes.move(unpack.src, unpack.dest)
            else:
                for f in fields_to_delete:
                    child_axes.move(f, None)
                dest.merge(child_axes)
        return dest

    def load(self, spec_list, request, detdb=None, dest=None, check=False):
        """Loads metadata objects and processes them into a single
        AxisManager.  This is equivalent to running load_raw and then
        unpack, though the two are intermingled.

        If check=True, this won't store and return the loaded
        metadata; it will instead return a list of the same length as
        spec_list, with either None (if the entry loaded successful)
        or the Exception raised when trying to load that entry.

        """
        if detdb is None:
            detdb = self.detdb

        if check:
            errors = []
            for spec in spec_list:
                try:
                    item = self.load_raw([spec], request, detdb)
                    errors.append((spec, None))
                except Exception as e:
                    errors.append((spec, e))
            return errors

        for spec in spec_list:
            try:
                item = self.load_raw([spec], request, detdb)
                dest = self.unpack(item, dest=dest, detdb=detdb)
            except Exception as e:
                e.args = e.args + (
                    "\n\nThe above exception arose while processing "
                    "the following metadata spec:\n"
                    f"  spec:    {spec}\n"
                    f"  request: {request}\n\n"
                    "Does your database expose this product for this observation?",)
                raise e
        return dest

    def check(self, spec_list, request):
        """Runs the same loading code as self.load, but does not keep the
        results, and will not raise an error due to missing metadata.

        Instead, this function returns information about whether
        metadata caused any trouble.

        """
        errors = []
        ok = True
        for spec in spec_list:
            try:
                item = self.load_raw([spec], request)
                errors.append((spec, None))
            except Exception as e:
                errors.append((spec, e))
                ok = False
        return ok, errors


class Unpacker:
    """Encapsulation of instructions for what information to extract from
    some source container, and what to call it in the destination
    container.

    The classmethod :ref:method:``decode`` is used populate Unpacker
    objects from metadata instructions; see docstring.

    Attributes:
      dest (str): The field name in the destination container.
      src (str): The field name in the source container.  If this is
        None, then the entire source container (or whatever it may be)
        should be stored in the destination container under the dest
        name.

    """
    @classmethod
    def decode(cls, coded, wildcard=[]):
        """
        Args:
          coded (list of str or str): Each entry of the string is a
            "coded request" which is converted to an Unpacker, as
            described below.  Passing a string here yields the same
            results as passing a single-element list containing that
            string.
          wildcard (list of str): source names from which to draw, in
            the case that the coded request contains a wildcard.
            Wildcard decoding, if requested, will fail unless the list
            has exactly 1 entry.

        Returns:
          A list of Unpacker objects, one per entry in coded.

        Notes:
          Each coded request must be in one of 4 possible forms, shown
          below, to the left of the :.  The resulting assignment
          operation is shown to the right of the colon::

            'dest_name&source_name'  : dest[dest_name] = source['source_name']
            'dest_name&'             : dest[dest_name] = source['dest_name']
            'dest_name&*'            : dest[dest_name] = source[wildcard[0]]
            'dest_name'              : dest[dest_name] = source

        """
        if isinstance(coded, str):
            coded = [coded]
        # Make a plan based on the name list.
        unpackers = []
        wrap_name = None
        for name in coded:
            if '&' in name:
                assert(wrap_name is None) # You already initiated a merge...
                dest_name, src_name = name.split('&') # check count...
                if src_name == '':
                    src_name = dest_name
                elif src_name == '*':
                    assert(len(wildcard) == 1)
                    src_name = wildcard[0]
                unpackers.append(cls(dest_name, src_name))
            else:
                assert(len(unpackers) == 0) # You already initiated a wrap...
                assert(wrap_name is None) # Multiple 'merge' names? Use & to multiwrap.
                wrap_name = name
                unpackers.append(cls(wrap_name, None))
        return unpackers

    def __init__(self, dest, src):
        self.dest, self.src = dest, src

    def __repr__(self):
        if self.src is None:
            return f'<Unpacker:{self.dest}>'
        return f'<Unpacker:{self.dest}<-{self.src}>'


class LoaderInterface:
    """Base class for "Loader" classes.  Subclasses must define, at least,
    the from_loadspec method.

    """
    def __init__(self, detdb=None, obsdb=None):
        """Args:
          detdb (DetDb): db against which to reconcile dets: index data.
          obsdb (ObsDb): db against which to reconcile obs: index data.

        References to the input database are cached for later use.

        """
        self.detdb = detdb
        self.obsdb = obsdb

    def from_loadspec(self, load_params):
        """Retrieve a metadata result.

        Arguments:
          load_params: an index dictionary.

        Returns:
          A ResultSet or similar metadata object.

        """
        raise NotImplementedError

    def batch_from_loadspec(self, load_params):
        """Retrieves a batch of metadata results.  load_params should be a
        list of valid index data specifications.  Returns a list of
        objects, corresponding to the elements of load_params.

        The default implementation simply calls self.from_loadspec
        repeatedly; but subclasses are free to do something more optimized.

        """
        return [self.from_loadspec(p) for p in load_params]

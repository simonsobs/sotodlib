from sotodlib import core
from ..axisman import get_coindices

import logging
import os
import numpy as np

from . import ResultSet

logger = logging.getLogger(__name__)

REGISTRY = {
    '_default': 'DefaultHdf'
}


def _filter_items(prefix, d, remove=True):
    # Restrict d to only items that start with prefix; if d is a dict,
    # return a dict with only the keys that satisfy that condition.
    if isinstance(d, dict):
        return {k: d[prefix*remove + k]
                for k in _filter_items(prefix, list(d.keys()), remove=remove)}
    return [k[len(prefix)*remove:] for k in d if k.startswith(prefix)]


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
            then the path of context.filename is used; otherwise cwd
            is used.

        """
        if context is not None:
            if detdb is None:
                detdb = context.detdb
            if obsdb is None:
                obsdb = context.obsdb
            if working_dir is None:
                working_dir = os.path.split(context.filename)[0]
        if working_dir is None:
            working_dir = os.getcwd()
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

    def load_one(self, spec, request, det_info):
        """Process a single metadata entry (spec) by loading a ManifestDb and
        reading metadata for a particular observation.  The request
        must be pre-augmented with all ObsDb info that might be
        needed.  det_info is used to screen the returned data for the
        various index_lines.

        Args:
          spec (dict): A metadata specification dict (corresponding to
            a metadata list entry in context.yaml).
          request (dict): A metadata request dict (stating what
            observation and detectors are of interest).
          det_info (ResultSet): Table of detector properties to use
            when resolving metadata that is indexed with dets:*
            fields.

        Notes:
          The metadata ``spec`` dict has the following schema:

            ``db`` (str)
                The path to a ManifestDb file.

            ``name`` (str)
                Naively, the name to give to the extracted data.  But
                the string may encode more complicated instructions,
                which are understood by the Unpacker class in this
                module.

            ``loader`` (str, optional)
                The name of the loader class to use when loading the
                data.  This will take precedence over what is
                specified in the ManifestDb, and is normally
                unnecessary but can be used for debugging /
                work-arounds.

          Any filenames in the ManifestDb that are given as relative
          paths will be resolved relative to the directory where the
          db file is found.

          The ``request`` dict specifies what times and detectors are
          of interest.  If the metadata archive is indexed by
          timestamp and wafer_slot, then you might pass in::

             {'obs:timestamp': 1234567000.,
              'dets:wafer_slot': 'w01'}

          When this function is invoked from self.load, the request
          dict will have been automatically "augmented" using the
          ObsDb.  The main purpose of this is to provide obs:timestamp
          (and any other useful indexing fields) from ObsDb based on
          obs:obs_id.

          The ``det_info`` object comes into play in cases where a
          loaded metadata result refers to some large group of
          detectors, but the metadata index, or the user request,
          expresses that the result should be limited to only a subset
          of those detectors.  This is notated in practice by
          including dets:* fields in the index data in the ManifestDb,
          or in the request dict.

        Returns:
          A list of tuples (unpacker, item), corresponding to each
          entry in spec_list.  The unpacker is an Unpacker object
          created based on the 'name' field.  The item is the metadata
          in its native format (which could be a ResultSet or
          AxisManager), with all restrictions specified in request
          already applied.

        """
        # Load the database, match the request,
        dbfile = os.path.join(self.working_dir, spec['db'])
        dbpath = os.path.split(dbfile)[0]
        if dbfile not in self.manifest_cache:
            if dbfile.endswith('sqlite'):
                man = core.metadata.ManifestDb.readonly(dbfile)
            else:
                man = core.metadata.ManifestDb.from_file(dbfile)
            self.manifest_cache[dbfile] = man
        man = self.manifest_cache[dbfile]

        # Do we have all the keys we need?
        provided_obs_keys = _filter_items(
            'obs:', man.scheme.get_required_params(), remove=False)
        missing_obs_keys = (set(provided_obs_keys) - set(request.keys()))
        if len(missing_obs_keys):
            raise RuntimeError(
                f'Metadata request is indexed by {request.keys()} but '
                f'request info includes only {provided_obs_keys}.')

        # Lookup.
        try:
            index_lines = man.match(request, multi=True, prefix=dbpath)
        except Exception as e:
            text = str(e)
            raise RuntimeError(
                'An exception was raised while decoding the following spec:\n'
                + '  ' + str(spec) + '\n'
                + 'with the following request:\n'
                + '  ' + str(request) + '\n'
                + 'The exception is:\n  %s' % text)

        # Load each index_line.
        results = []
        for index_line in index_lines:
            logger.debug(f'Processing index_line={index_line}')
            # Augment the index_line with info from the request; if
            # the request and index_line share a key but conflict on
            # the value, we don't want this item.
            skip_this = False
            for k in request:
                if k in index_line:
                    if request[k] != index_line[k]:
                        skip_this = True
            if skip_this:
                continue
            index_line.update(request)
            loader = spec.get('loader', None)
            if loader is None:
                loader = index_line.get('loader', REGISTRY['_default'])
            try:
                loader_class = REGISTRY[loader]
            except KeyError:
                raise RuntimeError('No metadata loader registered under name "%s"' % loader)

            loader_object = loader_class()  # pass obs info?
            mi1 = loader_object.from_loadspec(index_line)

            # Restrict returned values according to the specs in index_line.

            if isinstance(mi1, ResultSet):
                # For simple tables, the restrictions can be
                # integrated into the table, to be dealt with later.
                det_restricts = _filter_items('dets:', index_line, remove=False)
                mask = np.ones(len(mi1), bool)
                keep_cols = list(mi1.keys)
                new_cols = []
                for k, v in det_restricts.items():
                    if k in mi1.keys:
                        mask *= (mi1[k] == v)
                    else:
                        new_cols.append((k, v))
                a = mi1.subset(keys=keep_cols, rows=mask)
                mi2 = ResultSet([k for k, v in new_cols],
                                [[v for k, v in new_cols]] * len(a))
                mi2.merge(a)

            elif isinstance(mi1, core.AxisManager):
                # For AxisManager results, the dets axis *must*
                # reconcile 1-to-1 with some field in det_info, and
                # that may be used to toss things out based on
                # index_line.
                det_restricts = _filter_items('dets:', index_line, remove=True)
                dets_key = 'name'
                new_dets, i_new, i_info = get_coindices(mi1.dets.vals, det_info['name'])
                mask = np.ones(len(i_new), bool)
                for k, v in det_restricts.items():
                    mask *= (det_info[k][i_info] == v)
                if mask.all() and len(new_dets) == mi1.dets.count:
                    mi2 = mi1
                else:
                    mi2 = mi1.restrict('dets', new_dets[mask])

            else:
                raise RuntimeError(
                    'Returned object is non-specialized type {}: {}'
                    .format(mi1.__class__, mi1))

            results.append(mi2)

        # Check that we got results, then combine them in to single ResultSet.
        assert(len(results) > 0)
        result = results[0].concatenate(results)
        return result

    def load(self, spec_list, request, det_info=None, dest=None, check=False):
        """Loads metadata objects and processes them into a single
        AxisManager.

        Args:
          spec_list (list of dicts): Each dict is a metadata spec, as
            described in load_one.
          request (dict): A request dict.
          det_info (AxisManager): Detector info table to use for
            reconciling 'dets:*' field restrictions.
          dest (AxisManager or None): Destination container for the
            metadata (if None, a new one is created).
          check (bool): If True, run in check mode (see Notes).

        Returns:
          In normal mode, an AxisManager containing the metadata
          (dest).  In check mode, a list of tuples (spec, exception).

        Notes:
          If check=True, this won't store and return the loaded
          metadata; it will instead return a list of the same length
          as spec_list, with either None (if the entry loaded
          successful) or the Exception raised when trying to load that
          entry.  When check=False, metadata retrieval errors will
          raise some kind of error.  When check=True, those are caught
          and returned to the caller.

        """
        # Augmented request.
        if self.obsdb is not None and 'obs:obs_id' in request:
            obs_info = self.obsdb.get(request['obs:obs_id'], add_prefix='obs:')
            if obs_info is not None:
                obs_info.update(request)
                request = obs_info

        def reraise(e, spec):
            e.args = e.args + (
                "\n\nThe above exception arose while processing "
                "the following metadata spec:\n"
                f"  spec:    {spec}\n"
                f"  request: {request}\n\n"
                "Does your database expose this product for this observation?",)
            raise e

        # Process each item.
        items = []
        for spec in spec_list:
            logger.debug(f'Processing metadata spec={spec}')
            try:
                item = self.load_one(spec, request, det_info)
                error = None
            except Exception as e:
                if check:
                    error = e
                else:
                    reraise(e, spec)

            if spec.get('det_info'):
                # Things that augment det_info need to be resolved
                # before the check==True escape.  det_info things
                # should be ResultSets where all keys are prefixed
                # with dets:!
                if any([not k.startswith('dets:') for k in item.keys]):
                    reraise(RuntimeError(
                        f'New det_info metadata has keys without prefix "dets:": '
                        f'{item}'))

                det_key = spec['det_key']
                key = det_key[len('dets:'):]
                both, i0, i1 = get_coindices(item[det_key], det_info[key])

                logger.debug(f' ... updating det_info (row count '
                             f'{len(det_info)} -> {len(i1)})')
                det_info = det_info.subset(rows=i1)
                item = item.subset([k for k in item.keys if k != det_key],
                                   rows=i0)
                item.keys = [k[len('dets:'):] for k in item.keys]
                det_info.merge(item)
                item = None

            if check:
                items.append((spec, error))
                continue

            if item is None:
                # Exit for the det_info case.
                continue

            # Make everything an axisman.
            if not isinstance(item, core.AxisManager):
                item = item.axismanager(det_info=det_info)

            # Unpack it.
            try:
                unpackers = Unpacker.decode(spec['name'], item)
                for unpacker in unpackers:
                    dest = unpacker.unpack(item, dest=dest)
            except Exception as e:
                reraise(e, spec)

            logger.debug(f'load(): dest now has shape {dest.shape}')

        if check:
            return items

        dest.wrap('det_info', convert_det_info(det_info))

        return dest


def convert_det_info(det_info, dets=None):
    """
    Convert det_info ResultSet into nested AxisManager.
    """
    children = {}
    if dets is None:
        dets = det_info['name']
    output = core.AxisManager(core.LabelAxis('dets', dets))
    subtables = {}
    for k in det_info.keys:
        if '.' in k:
            prefix, subkey = k.split('.', 1)
            if not prefix in subtables:
                subtables[prefix] = []
            subtables[prefix].append(subkey)
        else:
            output.wrap(k, det_info[k], [(0, 'dets')])
    for subtable, subkeys in subtables.items():
        sub_info = det_info.subset(keys=[f'{subtable}.{k}' for k in subkeys])
        sub_info.keys = subkeys
        child = convert_det_info(sub_info, dets)
        output.wrap(subtable, child)
    return output

class Unpacker:
    """Encapsulation of instructions for what information to extract from
    some source container, and what to call it in the destination
    container.

    The classmethod :ref:method:``decode`` is used to populate
    Unpacker objects from metadata instructions; see docstring.

    Attributes:
      dest (str): The field name in the destination container.
      src (str): The field name in the source container.  If this is
        None, then the entire source container (or whatever it may be)
        should be stored in the destination container under the dest
        name.

    """
    @classmethod
    def decode(cls, coded, target=None, wildcard=None):
        """Args:
          coded (list of str or str): Each entry of the string is a
            "coded request" which is converted to an Unpacker, as
            described below.  Passing a string here yields the same
            results as passing a single-element list containing that
            string.
          target (AxisManager): The object from which the targets will
            be unpacked.  This is only accessed if wildcard is None.
          wildcard (list of str): source_name values to draw from if
            the user has requested wildcard matching.  Currently only
            a single wildcard item may be extracted, so the list must
            have length 1.  If not passed explicitly, wildcard list
            will be taken from ``target``.  Passing [] for this option
            will effectively disable the wildcard feature.

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

        if wildcard is None and target is not None:
            wildcard = list(target._fields.keys())[:1]

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

    def unpack(self, item, dest=None):
        """Extract desired fields from an AxisManager and merge them into
        another one.

        Args:

          item (AxisManager): Source object from which to extract
            fields.
          dest (AxisManager): Place to put them.

        Returns:
          dest, or a new AxisManager if dest=None was passed in.

        """
        if dest is None:
            dest = core.AxisManager()
        fields_to_delete = list(item._fields.keys())
        # Unpack to requested field names.
        if self.src is None:
            dest.wrap(self.dest, item)
            return dest
        else:
            fields_to_delete.remove(self.src)
            if self.src != self.dest:
                item.move(self.src, self.dest)
        for f in fields_to_delete:
            item.move(f, None)
        dest.merge(item)
        return dest


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
        #self.detdb = detdb
        #self.obsdb = obsdb

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

from sotodlib import core

import collections
import logging
import os
import numpy as np
import warnings

from . import ResultSet

logger = logging.getLogger(__name__)

REGISTRY = {
    '_default': 'DefaultHdf'
}


class LoaderError(RuntimeError):
    """
    Use with two args: (pithy_summary, formatted_detail)
    """


class IncompleteMetadataError(RuntimeError):
    pass


class IncompleteDetInfoError(RuntimeError):
    pass


class MetadataSpecError(RuntimeError):
    pass


class MetadataSpecWarning(Warning):
    pass


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
            if working_dir is None and context.filename is not None:
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
            a metadata list entry in context.yaml), or MetadataSpec
            object.
          request (dict): A metadata request dict (stating what
            observation and detectors are of interest).
          det_info (ResultSet): Table of detector properties to use
            when resolving metadata that is indexed with dets:...
            fields.

        Notes:
          If passing ``spec`` as a dict, see the schema described in
          :class:`MetadataSpec`.

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
          including dets:... fields in the index data in the ManifestDb,
          or in the request dict.  Only fields already present in
          det_info may be included in the request dict.

        Returns:
          A list of tuples (unpacker, item), corresponding to each
          entry in spec_list.  The unpacker is an Unpacker object
          created based on the 'name' field.  The item is the metadata
          in its native format (which could be a ResultSet or
          AxisManager), with all restrictions specified in request
          already applied.

        """
        if isinstance(spec, dict):
            spec = MetadataSpec.from_dict(spec)

        # Load the database, match the request,
        if isinstance(spec.db, str):
            # The usual case.
            dbfile = os.path.join(self.working_dir, spec.db)
            dbpath = os.path.split(dbfile)[0]
            if dbfile not in self.manifest_cache:
                if dbfile.endswith('sqlite'):
                    man = core.metadata.ManifestDb.readonly(dbfile)
                else:
                    man = core.metadata.ManifestDb.from_file(dbfile)
                self.manifest_cache[dbfile] = man
            man = self.manifest_cache[dbfile]
        elif isinstance(spec.db, core.metadata.ManifestDb):
            # Useful for testing and hacking
            dbpath = self.working_dir
            man = spec.db

        # Do we have all the keys we need?
        required_obs_keys = _filter_items(
            'obs:', man.scheme.get_required_params(), remove=False)
        missing_obs_keys = (set(required_obs_keys) - set(request.keys()))
        if len(missing_obs_keys):
            raise RuntimeError(
                f'Metadata request is indexed by {request.keys()} but '
                f'ManifestDb requires {required_obs_keys}.')

        required_dets_keys = _filter_items(
            'dets:', man.scheme.get_required_params(), remove=False)
        missing_dets_keys = list((set(required_dets_keys) - set(request.keys())))

        if len(missing_dets_keys):
            # Make request to ManifestDb for each detector bundle.
            short_keys = _filter_items('dets:', missing_dets_keys)
            try:
                subreqs = det_info.subset(keys=short_keys).distinct()
            except:
                raise RuntimeError(
                    f'Metadata request requires keys={missing_dets_keys} '
                    f'but det_info={det_info}.')
            subreqs.keys = missing_dets_keys # back with dets: prefix ...
        else:
            subreqs = ResultSet([], [()])  # length 1!

        index_lines = []
        for subreq in subreqs:
            # Reject any subreq that explicitly contradicts request on any key.
            if any([subreq.get(k, v) != v for k, v in request.items()]):
                continue
            subreq.update(request)

            try:
                _lines = man.match(subreq, multi=True, prefix=dbpath)
            except Exception as e:
                text = str(e)
                raise LoaderError('Exception when matching subrequest.',
                                  f"An exception occurred while processing sub-request:\n\n"
                                  f"  subreq={subreq}\n\n")
            for _line in _lines:
                # Now reject any _line if they contradict subreq.
                if any([subreq.get(k, v) != v for k, v in _line.items()]):
                    continue
                _line.update(subreq)
                index_lines.append(_line)

        # Pre-screen the index_lines for dets:* assignments; plan to
        # skip lines that aren't relevant according to det_info.
        to_skip = []
        for index_line in index_lines:
            logger.debug(f'Pre-screening index_line={index_line}')
            skip_this = len(det_info) == 0
            if not skip_this:
                mask = np.ones(len(det_info), bool)
                for k, v in _filter_items('dets:', index_line, remove=True).items():
                    if k not in det_info.keys:
                        raise IncompleteDetInfoError(
                            f"Entry requires det_info['{k}'], and that is not defined.")
                    mask *= (det_info[k] == v)
                skip_this = (mask.sum() == 0)
            to_skip.append(skip_this)

        if len(index_lines) == 0:
            # If we come out the other side with no data to load,
            # invent one so that we at least get the structure of the
            # metadata (even though we'll throw out all the actual
            # results).  You can get here if someone passes dets=[].
            candidate_index_lines = man.inspect(request, False, prefix=dbpath)
            index_lines.append(candidate_index_lines[0])
            to_skip = [False]

        elif all(to_skip):
            # Load at least one item, or we won't have the structure of
            # the output.
            to_skip[0] = False

        # Load each index_line.
        results = []
        for skip, index_line in zip(to_skip, index_lines):
            if skip:
                logger.debug(f'Skipping load for index_line={index_line}')
                continue
            logger.debug(f'Loading for index_line={index_line}')

            loader = spec.loader
            if loader is None:
                loader = index_line.get('loader', REGISTRY['_default'])
            try:
                loader_class = REGISTRY[loader]
            except KeyError:
                raise LoaderError(
                    'Loader function not found.',
                    f'No metadata loader registered under name "{loader}"')

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
                dets_key = 'readout_id'
                new_dets, i_new, i_info = core.util.get_coindices(
                    mi1.dets.vals, det_info[dets_key])

                mask = np.ones(len(i_new), bool)
                if len(i_info):
                    for k, v in det_restricts.items():
                        mask *= (det_info[k][i_info] == v)
                if mask.all() and len(new_dets) == mi1.dets.count:
                    mi2 = mi1
                else:
                    mi2 = mi1.restrict('dets', new_dets[mask])

            else:
                raise LoaderError(
                    'Invalid metadata carrier.',
                    'Returned object is non-specialized type {}: {}'
                    .format(mi1.__class__, mi1))

            results.append(mi2)

        # Check that we got results, then combine them in to single ResultSet.
        logger.debug(f'Concatenating {len(results)} results: {results}')
        assert(len(results) > 0)
        result = results[0].concatenate(results)
        return result

    def load(self, spec_list, request, det_info=None, free_tags=[],
             free_tag_fields=[], dest=None, check=False, det_info_scan=False,
             ignore_missing=False, on_missing=None):
        """Loads metadata objects and processes them into a single
        AxisManager.

        Args:
          spec_list (list of dicts): Each dict is a metadata spec, as
            described in load_one.
          request (dict): A request dict.
          det_info (AxisManager): Detector info table to use for
            reconciling 'dets:...' field restrictions.
          free_tags (list of str): Strings that restrict the detector
            to any detector that matches the string in any of the
            det_info fields listed in free_tag_fields.
          free_tag_fields (list of str): Fields (of the form dets:x)
            that can be inspected to match free_tags.
          dest (AxisManager or None): Destination container for the
            metadata (if None, a new one is created).
          check (bool): If True, run in check mode (see Notes).
          det_info_scan (bool): If True, *only* process entries that
            directly update det_info.
          ignore_missing (bool): If True, don't fail when a metadata
            item can't be loaded, just try to proceed without it.
          on_missing (dict): If a key here matches the label of a
            metadata entry, the value will override the on_missing
            entry of the metadata entry.  (Each value must be "trim",
            "skip" or "fail".)

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
        # Augmented request -- note that dets:* restrictions from
        # request will be added back into this by check tags.
        aug_request = _filter_items('obs:', request, False)

        if on_missing is None:
            on_missing = {}

        if self.obsdb is not None and 'obs:obs_id' in request:
            if dest is None:
                dest = core.AxisManager()
            obs_man = core.AxisManager()
            obs_info = self.obsdb.get(request['obs:obs_id'], add_prefix='obs:')
            if obs_info is None:
                logger.warning(
                    f"Observation {request['obs:obs_id']} not found in obsdb; "
                    "trying to proceed anyway. You might have metadata failures.")
                obs_man.wrap('obs_id', request['obs:obs_id'])
            else:
                obs_info.update(aug_request)
                aug_request.update(obs_info)
                for k, v in _filter_items('obs:', obs_info).items():
                    obs_man.wrap(k, v)
            dest.wrap('obs_info', obs_man)

        def reraise(spec, e):
            logger.error(
                f"An error occurred while processing a meta entry:\n\n"
                f"  spec:    {spec}\n\n"
                f"  request: {request}\n\n")
            if isinstance(e, LoaderError):
                # Present all args to logger instead...
                for a in e.args[1:]:
                    logger.error(a)
                e = LoaderError(e.args[0])
            raise e

        def check_tags(det_info, aug_request, final=False):
            mask = np.ones(len(det_info), bool)
            unmatched = list(free_tags)
            for tag in free_tags:
                for field in free_tag_fields:
                    if field in det_info.keys:
                        s = (det_info[field] == tag)
                        if s.any():
                            mask *= s
                            unmatched.remove(tag)
            if final and len(unmatched):
                raise RuntimeError(
                    f'One or more free tags was left unconsumed: {unmatched}')

            det_reqs = _filter_items('dets:', request, True)
            unmatched = []
            for k, v in det_reqs.items():
                if k in det_info.keys:
                    if isinstance(v, (list, np.ndarray)):
                        mask *= (core.util.get_multi_index(v, det_info[k]) >= 0)
                    else:
                        mask *= (det_info[k] == v)
                        aug_request['dets:' + k] = v
                else:
                    unmatched.append('dets:' + k)
            if final and len(unmatched):
                raise RuntimeError(
                    f'One or more dets:* selections was left unconsumed: {unmatched}')

            if not np.all(mask):
                logger.debug(f' ... free tags / request reduce det_info (row count '
                             f'{len(det_info)} -> {mask.sum()})')
                det_info = det_info.subset(rows=mask)

            if len(mask) > 0 and len(det_info) == 0:
                logger.warning(f'All detectors have been eliminated from processing.')
                logger.warning(f'  dets:*: {det_reqs}')
                logger.warning(f'  free_tags: {free_tags}')

            return det_info, aug_request

        det_info, aug_request = check_tags(det_info, aug_request)
        n_dets = len(det_info)

        # Process each item.
        items = []
        for spec in spec_list:
            spec, _spec = MetadataSpec.from_dict(spec), spec
            if det_info_scan and not spec.det_info:
                continue

            logger.debug(f'Processing metadata spec={_spec} with augmented '
                         f'request={aug_request}')

            label = spec.label
            _on_missing = spec.on_missing
            if label is not None and label in on_missing:
                _on_missing = on_missing[label]
                logger.debug(f'User overrides on_missing={_on_missing} for {label}')

            assert _on_missing in ['trim', 'skip', 'fail']

            try:
                item = self.load_one(spec, aug_request, det_info)
                error = None
            except Exception as e:
                if check:
                    error = e
                elif ignore_missing or _on_missing == 'skip':
                    logger.warning(f'Failed to load metadata for spec={_spec}; ignoring.')
                    continue
                else:
                    reraise(_spec, e)

            if spec.det_info and error is None:
                item_keys = _filter_items('dets:', item.keys)
                try:
                    det_info = merge_det_info(
                        det_info, item,
                        on_missing=_on_missing)
                except IncompleteMetadataError as e:
                    if check:
                        # I guess we report this, either way.
                        error = e
                    elif _on_missing == 'fail':
                        reraise(_spec, e)
                    elif _on_missing == 'skip':
                        # print a warning I guess
                        logger.warning(f'Skipping failed det_info load, spec={_spec}')

                item = None

                # The check_tags call can cause truncation of the
                # dataset, and that's ok.
                det_info, aug_request = check_tags(det_info, aug_request)
                n_dets = len(det_info)

            if check:
                items.append((spec, error))
                continue

            if item is None:
                # Exit for the det_info case.
                continue

            # Make everything an axisman.
            if isinstance(item, ResultSet):
                # Note this might raise an IncompleteDetInfoError.
                item = broadcast_resultset(item, det_info=det_info)

            elif not isinstance(item, core.AxisManager):
                logger.error(
                    f'The decoded item {item} is not an AxisManager or '
                    f'other well-understood type.  Request was: {request}.')

            # You have to check for detector loss here -- compare
            # item.dets.vals to what's in det_info.
            i0 = core.util.get_multi_index(
                item.dets.vals, det_info['readout_id'])

            n_dets_item = len(set(i0[i0>=0]))
            if n_dets_item < n_dets:
                message = (f"Only {n_dets_item} of {n_dets} detectors "
                           "have data for metadata specified by "
                           f"spec={_spec}. ")
                if _on_missing == 'trim':
                    logger.warning(message + 'Trimming.')
                elif _on_missing == 'fail':
                    raise IncompleteMetadataError(message)
                else:  # skip
                    logger.warning(message + 'Discarding.')
                    continue

            # Unpack it.
            try:
                dest = unpack_item(spec.unpack, item, dest=dest)
                #unpacker = _Unpacker.from_spec(spec, target=item)
                #dest = unpacker.unpack(item, dest=dest)
            except Exception as e:
                reraise(_spec, e)

            logger.debug(f'load(): dest now has shape {dest.shape}')
            n_dets = dest.dets.count

        check_tags(det_info, aug_request, final=True)

        if check:
            return items

        dest.wrap('det_info', convert_det_info(det_info))

        return dest


def _filter_items(prefix, d, remove=True):
    # Restrict d to only items that start with prefix; if d is a dict,
    # return a dict with only the keys that satisfy that condition.
    if isinstance(d, dict):
        return {k: d[prefix*remove + k]
                for k in _filter_items(prefix, list(d.keys()), remove=remove)}
    return [k[len(prefix)*remove:] for k in d if k.startswith(prefix)]


def merge_det_info(det_info, new_info, multi=True, on_missing='trim'):
    """Args:

      det_info (ResultSet or None): The det_info table to start from,
        with columns *without* the 'dets:' prefix.
      new_info (ResultSet): New data to merge/check against
        det_info; only columns with dets: prefix are processed.
      multi (bool): whether to permit some rows to match multiple
        rows; this is True by default.
      on_missing (str): what to do if the new_info does not fully cover
        the dets in det_info.

    Returns:
      A (possibly) new det_info table, containing updates and
      restrictions from new_info.  This will contain at least the
      columns that det_info had, and at most as many rows.

    Notes:
      The input arguments det_info and new_info may be modified by
      this function.  Passing in None for det_info is convenient to
      initialize a det_info from a new_info where the columns are
      named dets:... .

    """
    new_keys = _filter_items('dets:', new_info.keys)
    if (len(new_keys) != len(new_info.keys)):
        raise RuntimeError(
            f'New det_info metadata has keys without prefix "dets:": '
            f'{new_info}')
    new_info.keys = new_keys

    if det_info is None:
        return new_info

    join_on = list(set(new_info.keys).intersection(det_info.keys))
    if len(join_on) == 0:
        raise IncompleteMetadataError(
            f'Cannot merge det_info: no common keys in '
            f'{det_info} and {new_info}.')

    new_index = list(zip(*[new_info[k] for k in join_on]))
    det_index = list(zip(*[det_info[k] for k in join_on]))

    # Perform the row-row matching, accepting that rows of new_info
    # may match multiple entries of det_info.  (This is so new_info
    # can have a det_id = 'NO_MATCH' entry.)
    i0 = core.util.get_multi_index(
        new_index, det_index)
    i1 = np.arange(len(det_index))
    i0, i1 = i0[i0>=0], i1[i0>=0]

    if not multi and len(i0) != len(set(i0)):
        offenders = collections.Counter(i0)
        offenders = [new_index[i] for i, v in offenders.items() if v > 1][:10]
        raise ValueError(
            'Some new items matched up to more than one entry in the '
            'existing det_info.  Should this entry have multi=True? '
            f'Offenders: {join_on} = {offenders}')

    # Common fields need to be in accordance, then drop them.
    common_keys = set(new_info.keys) & set(det_info.keys)
    for k in common_keys:
        if len(i0) and np.any(new_info[k][i0] != det_info[k][i1]):
            raise ValueError(
                'When reconciling new det_info, a disagreement in the value '
                f'of field {k} was observed.  If this is due to ambiguity '
                f'in the det_id, maybe add {k} to the match_keys?')

    if len(det_info) != len(i1) and on_missing != 'trim':
        raise IncompleteMetadataError('{len(det_info)} -> {len(i1)})')

    logger.debug(f' ... updating det_info (row count '
                 f'{len(det_info)} -> {len(i1)})')
    det_info = det_info.subset(rows=i1)
    new_info = new_info.subset(
        [k for k in new_info.keys if k not in common_keys],
        rows=i0)
    det_info.merge(new_info)
    return det_info


def convert_det_info(det_info, dets=None):
    """Convert det_info ResultSet into nested AxisManager.

    Args:
      dets (list of str): The labels to use for the LabelAxis.  You
        probably want to use the default, det_info['readout_id'], or
        pass in det_info[something_else].

    Returns:

      Nested AxisManager with a LabelAxis called dets, containing all
      the columns from det_info.  Keys in det_info are split on '.';
      so for example det_info['sky.x'] will show up at output.sky.x,
      a.k.a. output['sky']['x'].

    """
    children = {}
    if dets is None:
        dets = det_info['readout_id']
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

def unconvert_det_info(aman):
    """Convert a det_info-style AxisManager (back) into a ResultSet... the
    opposite of convert_det_info.

    """
    def get_cols(aman, prefix=''):
        columns = []
        for k, v in aman._fields.items():
            if isinstance(v, core.AxisManager):
                columns.extend(get_cols(v, prefix=k + '.'))
            else:
                columns.append((prefix + k, v))
        return columns
    keys, columns = zip(*get_cols(aman))
    return ResultSet(keys, zip(*columns))


def broadcast_resultset(
        rs, det_info, axis_name='dets', axis_key='readout_id',
        prefix='dets:'):
    """Convert rs from a ResultSet into an AxisManager, but reconciling
    against the det_info table to make sure the output's .dets axis
    matches an indexing column of det_info (such as readout_id).

    Args:
      rs (ResultSet): target to convert.  Each column will become a
        vector in the output, unless it begins with 'dets:' in which
        case it will be reconciled and/or discarded (see Notes).
      det_info (ResultSet): table of detector info to use for
        broadcasting rs onto the output axis.
      axis_name (str): Name to use for the LabelAxis of the output.
      axis_key (str): Name of the column of rs that should be used for
        the values in the LabelAxis of the output.
      prefix (str): The prefix that should be used to identify
        indexing columns in rs.

    Returns:
      AxisManager.

    Notes:
      This function is to be applied to specially formatted ResultSet
      tables that contains a mixture of "indexing columns" and "data
      columns".  The "indexing columns" are the ones that start with
      ``prefix``, and are used to associate values in the data columns
      with specific detectors.  For example::

        >>> print(rs)
        ResultSet<[dets:readout_id,a,b], 1280 rows>
        >>> print(det_info)
        ResultSet<[readout_id,x,y,detset], 1280 rows>
        >>> aman = broadcast_resultset(rs, det_info)
        >>> print(aman, aman.dets)
        AxisManager(a[dets], b[dets], dets:LabelAxis(1280))

      In this case the rs column dets:readout_id contains all the same
      values as det_info['readout_id'], and the AxisManager thus also
      contains all 1280 entries.  The .dets axis has the values in the
      order of det_info['readout_id'].

      To broadcast, fields with the stated prefix must also be found
      in det_info.  For example::

        >>> print(rs)
        ResultSet<[dets:band,dets:wafer,abscal], 14 rows>
        >>> print(det_info)
        ResultSet<[readout_id,band,wafer,detset], 1760 rows>
        >>> aman = broadcast_resultset(rs, det_info)
        >>> print(aman)
        AxisManager(abscal[dets], dets:LabelAxis(1738))

      To use database terminology, the rs table is joined with the
      det_info table on the fields prefixed with ``prefix`` (in this
      case, 'band' and 'wafer').  The remaining columns from rs (just
      'abscal') are populated as fields in the output.  Note how the
      output .dets axis has fewer elements than det_info -- the 14
      rows of rs specifying values for different (band, wafer)
      combinations were enough to match only 1738 of the rows in
      det_info.  (This can happen even when not broadcasting.)

    """
    from sotodlib import core

    # Store short names for each index column.
    index_cols = {}
    for k in rs.keys:
        if k.startswith(prefix):
            index_cols[k] = k[len(prefix):]

    # Construct a map that takes a (tuple of dets:* values) to
    # specific row index of rs.
    row_map = {}
    for i, row in enumerate(rs):
        key = tuple([row[k] for k in index_cols.keys()])
        if key in row_map:
            raise ValueError("Duplicate entries for combined unique key.")
        row_map[key] = i

    # Get index of rs that corresponds to each row in det_info.
    missing_keys = [k for k in index_cols.values()
                    if k not in det_info.keys]
    if len(missing_keys):
        raise IncompleteDetInfoError(
            f"Loaded metadata requires det_info keys {missing_keys}.")

    indices = np.array(
        [row_map.get(tuple(row.values()), -1)
         for row in det_info.subset(keys=index_cols.values())],
        dtype=int)
    mask = (indices >= 0)
    dets = det_info[axis_key][mask]
    indices = indices[mask]  # drop any det_info items not matched

    # Re-order and wrap each field.
    aman = core.AxisManager(core.LabelAxis(axis_name, dets))
    for k in rs.keys:
        if not k.startswith(prefix):
            aman.wrap(k, rs[k][indices], [(0, axis_name)])
    return aman


class MetadataSpec:
    """Container for the canonical metadata specification.

    When constructed from a dict, the following attributes are set
    directly from the corresponding key:

    ``db`` (str, ManifestDb)
        The path to a ManifestDb file. For testing and other purposes,
        this may be passed as a ManifestDb object.  Defaults to None.

    ``det_info`` (bool)
        If True, treat the metadata as a contribution to
        det_info. The metadata will be merged into the active
        det_info object.  Defaults to False.

    ``label`` (str)
        A short string describing the metadata.  This is
        used to target the entry when overriding the default
        ``on_missing`` behavior.  Defaults to None.

    ``loader`` (str)
        The name of the loader class to use when loading the
        data.  This will take precedence over what is
        specified in the ManifestDb, and is normally
        unnecessary but can be used for debugging /
        work-arounds.  Defaults to None.

    ``on_missing`` (str)
        String describing how to proceed in the event that the
        metadata is incomplete (or missing entirely) for the target
        Observation.  The value should one of 'trim', 'skip', or
        'fail'.  Defaults to 'trim'.

    ``unpack`` (list of str)
        Instructions for how to populate the destination AxisManager
        with fields found in this metadata item.  See notes below.

    The following dict keys are deprecated, but are processed for
    backwards compatibility.

    ``name`` (str or list of str)
        (Deprecated.)  This has been renamed as "unpack", and will be
        copied into that attribute if unpack is not otherwise set.


    In the ``unpack`` list, each must be in one of 4 possible forms,
    shown below, to the left of the ``:``.  The resulting assignment
    operation is shown to the right of the ``:``.  ::

            'dest_name&source_name'  : dest[dest_name] = source[source_name]
            'dest_name&'             : dest[dest_name] = source[dest_name]
            'dest_name&*'            : dest[dest_name] = source[wildcard[0]]
            'dest_name'              : dest[dest_name] = source

    The first three forms cause a single field to be extracted.  The
    3rd form is used to extract a single field and rename it, assuming
    that name is the only one in source (it is an error otherwise).

    The ``unpack`` list may include multiple single field extraction
    entries, but each source field may only be referenced once. (So
    for example ``['a&a', 'b&b']`` is valid but ``['a&a, 'b&a']`` is
    not).

    The fourth form causes the entire item to be merged into the
    target at dest_name.  This can operate alongside any number of
    individual field extractions.

    """

    db = None
    det_info = False
    label = None
    loader = None
    on_missing = 'trim'
    unpack = None

    @classmethod
    def from_dict(cls, spec):
        self = cls()
        # canonical ...
        for k in ['db', 'label', 'unpack', 'det_info', 'loader', 'on_missing']:
            if k in spec:
                setattr(self, k, spec[k])
        # "name" used to be unpacking instructions.
        if 'name' in spec:
            name = spec['name']
            if isinstance(name, str):
                name = [name]
            if self.label is None:
                self.label = name[0].split('&')[0]
            if self.unpack is None:
                self.unpack = name
            else:
                # Prefer self.unpack but warn.
                warnings.warn(
                    "metadata spec contains 'unpack' and 'name' entries; "
                    "ignoring the latter (except to set the 'label', maybe).",
                    MetadataSpecWarning)
        # Make sure unpack is non-empty.
        if self.unpack is None:
            self.unpack = ['&']
        elif isinstance(self.unpack, str):
            self.unpack = [self.unpack]
        return self


def unpack_item(unpack, item, dest=None, wildcard=None):
    # Execute the unpacking of item described in unpack (assumed from
    # MetadataSpec.unpack).
    if wildcard is None:
        wildcard = list(item._fields.keys())[:1]

    # Make a plan based on the unpacking list
    instructions = []
    for name in unpack:
        if '&' in name:
            dest_name, src_name = name.split('&') # check count...
            if src_name == '':
                src_name = dest_name
            elif src_name == '*':
                assert(len(wildcard) == 1)
                src_name = wildcard[0]
            instructions.append(('extract', dest_name, src_name))
        else:
            instructions.append(('full', name, None))

    if dest is None:
        dest = core.AxisManager()

    # Based on instructions, we may need multiple copies of this
    # item; one for any "full" extraction, and one to extract data
    # members from.
    copy_count = sum([inst[0] == 'full' for inst in instructions])
    extr_counts = collections.Counter([inst[2] for inst in instructions
                                       if inst[0] == 'extract'])
    extr_max_count = max(extr_counts.values(), default=0)
    assert extr_max_count <= 1  # Multiple extraction of child fields not supported.
    copy_count += extr_max_count

    # Start with the full copies.
    for inst, dest_name, src_name in instructions:
        if inst != 'full':
            continue
        assert src_name is None
        copy_count -= 1
        _item = item
        if copy_count > 0:
            _item = item.copy()
        dest.wrap(dest_name, _item)

    if extr_max_count == 0:
        return dest

    # And now the partial copies.  By assertion, each field is
    # extracted at most once.
    fields_to_delete = list(item._fields.keys())

    for inst, dest_name, src_name in instructions:
        if inst != 'extract':
            continue
        fields_to_delete.remove(src_name)
        if src_name != dest_name:
            item.move(src_name, dest_name)

    for f in fields_to_delete:
        del item[f]

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


def load_metadata(tod, spec, unpack=False):
    """Process a metadata entry for an AxisManager.

    Args:

      tod (AxisManager): The data structure from which to source any
        obs_info and det_info that are needed to process the metadata
        specification.  This
      spec (dict): a metadata specification, such as one might find as
        an element of the "metadata" list in a context.yaml file.
      unpack (bool): if True, and if the spec does not identify as
        det_info, try to unpack the result into an AxisManager and
        return that.  This will result in broadcasting of items
        indexed by det_info fields into a full .dets axis.

    Returns:
      The loaded metadata item, which could be an AxisManager or
      ResultSet.  In the AxisManager case, the axes have likely not
      been resolved against the provided `tod`, so the sample count
      and detector count / ordering may be different.  (The caller can
      merge after the fact.)

    Notes:
      The ``tod`` container needs to contain ``obs_info`` and
      ``det_info`` (including the ``dets`` axis), in order to follow
      any branching instructions for loading the metadata.  This would
      normally be an AxisManager returned by ``Context.get_obs()`` or
      ``get_meta()``.

    """
    loader = SuperLoader()
    det_info = unconvert_det_info(tod.det_info)
    request = {}
    for k, v in tod.obs_info._fields.items():
        request[f'obs:{k}'] = v
    spec = MetadataSpec.from_dict(spec)
    item = loader.load_one(spec, request, det_info)
    if not unpack or spec.det_info:
        return item

    if isinstance(item, ResultSet):
        # Note this might raise an IncompleteDetInfoError.
        item = broadcast_resultset(item, det_info=det_info)

    return unpack_item(spec.unpack, item)

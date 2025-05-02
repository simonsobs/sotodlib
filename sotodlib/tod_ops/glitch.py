import numpy as np
from functools import partial
from so3g.proj import Ranges
from ..core import AxisManager


def ranges_from_n_flagged(n_flagged, n_thres=2, buffer=5):
    return Ranges.from_bitmask(n_flagged >= n_thres).buffer(buffer)

def get_det_mask(ranges_matrix, ranges):
    """
    Given a per-detector ranges matrix and a second set of ranges, return a
    mask of the detectors that overlap (intersect) the ranges in the latter.

    Parameters
    ----------
    ranges_matrix: RangesMatrix
        The ranges matrix containing a Ranges object for each detector.
    ranges: Ranges
        The ranges to be checked for overlap (intersection) with ranges_matrix.

    Returns
    -------
    det_mask: list
        A list of lists containing boolean values indicating whether each
        detector overlaps (intersects) the ranges in ranges_affected.
        Dimensions:
        len(outer list) = len(ranges)
        len(inner list) = n_detectors
    """
    det_mask = [[len(o.ranges()) > 0 for o in (ranges_matrix * Ranges.from_array(ranges.ranges()[i:i+1], ranges.count))]
                for i in range(len(ranges.ranges()))]
    return det_mask

def ranges2slices(r, offset=0):
    slices = [slice(r_[0]+offset, r_[1]+offset) for r_ in r.ranges()]
    return slices

def build_snippet_layouts(aman, slices, dets_affected):
    """
    Build snippet layouts from lists of the affected detectors and slices.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the data.
    slices: list
        The slices of the data that are affected.
    dets_affected: list
        The affected detectors for each slice.

    Returns
    -------
    snippets: list
        A list of AxisManagers, each containing (only) the affected detectors and
        samples for that snippet, e.g.:
        AxisManager(dets:LabelAxis(n_dets_affected), samps:OffsetAxis(n_samps_affected))
    """
    snippets = [AxisManager(
                    aman.dets.restriction(aman.dets.vals[dets])[0],
                    aman.samps.restriction(sl)[0],
                ) for (sl, dets) in zip(slices, dets_affected)]
    return snippets

def extract_snippet(aman, snippet_layout, in_place=False):
    """Restricts the aman data according to the snippet_layout (affected dets, samps)
    """
    return aman.restrict_axes(snippet_layout._axes, in_place=in_place)

def extract_snippets(aman, snippet_layouts):
    """
    Helper function to run extract_snippet for a list of snippet_layouts.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the data from which to extract the
        snippets.
    snippet_layouts: list
        List of axis managers containing information on the detectors and
        samples affected by each glitch snippet.

    Returns
    -------
    snippets: list
        A list of AxisManagers, each containing the data for the corresponding
        snippet layout.
    """
    return list(map(partial(extract_snippet, aman), snippet_layouts))

def get_snippets(aman, glitch_ranges, det_mask, offset=0):
    """
    Given the ranges of the glitches and masks of the affected detectors,
    return the snippets of data affected by the glitches.

    Parameters
    ----------
    aman: AxisManager
        The axis manager containing the data.
    glitch_ranges: Ranges
        The ranges of the glitches.
    det_mask: list
        A list of lists of boolean values identifying which detectors are
        affected by the glitches in glitch_ranges.
    offset: int
        The offset to be added to the start of each range. Default is 0.

    Returns
    -------
    snippets: list
        A list of AxisManagers, each containing the data for the corresponding
        glitch snippet.
    """
    # compile slices for each range
    slices = ranges2slices(glitch_ranges, offset=offset)

    # from the det_mask, get the indices of the affected detectors
    det_idxs = [np.where(det_mask[i])[0] for i in range(len(det_mask))]

    # build snippet layouts, each of which is an axis manager containing restricted axes
    snippet_layouts = build_snippet_layouts(aman, slices, det_idxs)

    # extract the snippets from aman
    snippets = extract_snippets(aman, snippet_layouts)

    return snippets

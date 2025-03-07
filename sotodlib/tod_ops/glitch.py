from functools import partial
from so3g.proj import Ranges
from ..core import AxisManager


def ranges_from_n_affected(n_affected, n_thres=2, buffer=5):
    return Ranges.from_bitmask(n_affected >= n_thres).buffer(buffer)

def dets_in_ranges(ranges_matrix, ranges):
    intervals = ranges.ranges()
    dets = []
    for i in range(len(intervals)):
        # make a new range for each interval
        r = Ranges.from_array(intervals[i:i+1], ranges.count)

        # get the dets affected in that range
        # note: ranges matrix needs to come first
        overlap = ranges_matrix * r
        dets.append([i for (i, r) in enumerate(overlap) if len(r.ranges()) > 0])
    return dets

def ranges2slices(r, offset=0):
    slices = []
    for r_ in r.ranges():
        slices.append(slice(r_[0]+offset, r_[1]+offset))
    return slices

def build_snippet_layouts(aman, slices, dets_affected):
    snippets = []
    for (sl, dets) in zip(slices, dets_affected):
        snippet = AxisManager(
            aman.dets.restriction(aman.dets.vals[dets])[0],
            aman.samps.restriction(sl)[0],
        )
        snippets.append(snippet)
    return snippets

def extract_snippet(aman, snippet_layout, in_place=False):
    return aman.restrict_axes(snippet_layout._axes, in_place=in_place)

def extract_snippets(aman, snippet_layouts):
    return list(map(partial(extract_snippet, aman), snippet_layouts))

# Copyright (c) 2023-2025 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import numpy as np
import so3g

from sotodlib import core
from sotodlib.mapmaking.utils import filter_subids, find_usable_detectors

from ._helpers import quick_tod

# ---------------------------------------------------------------------------
# filter_subids
# ---------------------------------------------------------------------------

SUBIDS = np.array([
    "obs_1750770479_satp3_1111111:ws3:f090",
    "obs_1750774090_satp3_1111111:ws0:f150",
    "obs_1750777688_satp3_1111011:ws0:f090",
    "obs_1750781294_satp3_1110111:ws0:f150",
    "obs_1750785970_satp3_1000000:ws0:f090"
])


def test_filter_subids_no_filter():
    result = filter_subids(SUBIDS)
    np.testing.assert_array_equal(result, SUBIDS)


def test_filter_subids_by_wafer():
    result = filter_subids(SUBIDS, wafers=["ws0"])
    assert all(":ws0:" in s for s in result)
    assert len(result) == 4


def test_filter_subids_by_band():
    result = filter_subids(SUBIDS, bands=["f090"])
    assert all(s.endswith("f090") for s in result)
    assert len(result) == 3


def test_filter_subids_by_ot():
    result = filter_subids(SUBIDS, ots=["satp3"])
    assert all("satp3" in s for s in result)
    assert len(result) == 5


def test_filter_subids_no_match():
    result = filter_subids(SUBIDS, wafers=["ws99"])
    assert len(result) == 0


# ---------------------------------------------------------------------------
# find_usable_detectors
# ---------------------------------------------------------------------------

def _add_glitch_flags(tod, cut_ranges_per_det):
    """Add flags.glitch_flags to a quick_tod AxisManager.

    cut_ranges_per_det: list of lists of [start, end] pairs, one per detector.
    """
    n_dets = tod.dets.count
    n_samps = tod.samps.count
    ranges = []
    for i in range(n_dets):
        r = cut_ranges_per_det[i]
        if r:
            ri = so3g.RangesInt32.from_array(
                np.array(r, dtype=np.int32).reshape(-1, 2), n_samps
            )
        else:
            ri = so3g.RangesInt32(n_samps)
        ranges.append(ri)
    glitch_flags = so3g.proj.ranges.RangesMatrix(ranges)
    flags_aman = core.AxisManager(tod.dets, tod.samps)
    flags_aman.wrap("glitch_flags", glitch_flags, [(0, "dets"), (1, "samps")])
    tod.wrap("flags", flags_aman)
    return tod


def test_find_usable_detectors_all_good():
    tod = quick_tod(3, 1000)
    tod = _add_glitch_flags(tod, [[], [], []])
    result = find_usable_detectors(tod)
    np.testing.assert_array_equal(result, np.array(tod.dets.vals))


def test_find_usable_detectors_one_cut():
    # det0001 has 20 % of samples flagged → removed with default maxcut=0.1
    tod = quick_tod(3, 1000)
    tod = _add_glitch_flags(tod, [[], [[0, 200]], []])
    result = find_usable_detectors(tod)
    assert tod.dets.vals[1] not in result
    assert tod.dets.vals[0] in result
    assert tod.dets.vals[2] in result


def test_find_usable_detectors_maxcut_threshold():
    # det0: 5 % flagged — kept at maxcut=0.1, removed at maxcut=0.04
    tod = quick_tod(2, 1000)
    tod = _add_glitch_flags(tod, [[[0, 50]], []])
    assert tod.dets.vals[0] in find_usable_detectors(tod, maxcut=0.1)
    assert tod.dets.vals[0] not in find_usable_detectors(tod, maxcut=0.04)

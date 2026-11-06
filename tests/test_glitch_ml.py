# Copyright (c) 2026 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check the glitch classification module (tod_ops.glitch_ml).

"""

import os
import pickle
import tempfile
import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from so3g.proj import Ranges, RangesMatrix

from sotodlib import core
from sotodlib.tod_ops import glitch_ml as gml
from sotodlib.preprocess.processes import GlitchClassification


def get_tod(n_dets=8, n_samps=1000, offset=0, events=(), same_pixel=(), no_pointing=(), seed=0):
    """Return a TOD with glitch flags in ``tod.flags.glitches``.

    Each event ``(dets, start, stop)`` flags *dets* in ``[start, stop)`` and
    adds a step to their signal there.  The detectors in *same_pixel* share
    one focal-plane position, and those in *no_pointing* have NaN positions.
    """
    rng = np.random.default_rng(seed)
    dets = core.LabelAxis("dets", [f"det{i:02d}" for i in range(n_dets)])
    samps = core.OffsetAxis("samps", n_samps, offset)
    tod = core.AxisManager(dets, samps)
    tod.wrap("signal", rng.normal(size=(n_dets, n_samps)).astype(np.float32),
             [(0, "dets"), (1, "samps")])
    boresight = core.AxisManager(samps)
    boresight.wrap("roll", np.full(n_samps, 0.1), [(0, "samps")])
    tod.wrap("boresight", boresight)
    xi = np.radians(rng.uniform(-1, 1, n_dets))
    eta = np.radians(rng.uniform(-1, 1, n_dets))
    for d in same_pixel:
        xi[d], eta[d] = xi[same_pixel[0]], eta[same_pixel[0]]
    xi[list(no_pointing)] = np.nan
    focal_plane = core.AxisManager(dets)
    focal_plane.wrap("xi", xi, [(0, "dets")])
    focal_plane.wrap("eta", eta, [(0, "dets")])
    tod.wrap("focal_plane", focal_plane)
    flags = [Ranges(n_samps) for _ in range(n_dets)]
    for event_dets, start, stop in events:
        for d in event_dets:
            flags[d].add_interval(start, stop)
            tod.signal[d, start:stop] += 20
    tod.wrap("flags", core.AxisManager(dets, samps))
    tod.flags.wrap("glitches", RangesMatrix(flags), [(0, "dets"), (1, "samps")])
    return tod


def get_forest(classes=4, seed=0):
    """Return a small forest trained on random stats."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(200, len(gml.get_all_stat_names())))
    y = np.arange(200) % classes
    return gml.train_forest(X, y, n_trees=5, max_depth=3)


class GetSnippetsTest(unittest.TestCase):
    def test_ranges_and_det_mask(self):
        tod = get_tod(n_dets=6, n_samps=100, offset=50,
                      events=[([0], 10, 12), ([1], 11, 13), ([2, 3], 50, 51), ([0], 80, 81)])
        snippets = gml.get_snippets(tod, tod.flags.glitches, n_thres=2, buffer=5)
        # >= 2 dets flagged at samples 11 and 50 only; padded by 5
        assert_array_equal(snippets.snippet_ranges.ranges(), [[6, 17], [45, 56]])
        assert_array_equal(snippets.det_mask, [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]])
        assert_array_equal(snippets.dets.vals, tod.dets.vals)
        self.assertEqual((snippets.samps.offset, snippets.samps.count), (50, 100))

    def test_merging_and_edges(self):
        tod = get_tod(n_dets=6, n_samps=100,
                      events=[([0, 1], 2, 3), ([0, 1], 30, 31), ([2, 3], 38, 39), ([4, 5], 98, 99)])
        snippets = gml.get_snippets(tod, tod.flags.glitches, n_thres=2, buffer=5)
        # Padding is clipped at the ends; snippets overlapping after padding merge
        assert_array_equal(snippets.snippet_ranges.ranges(), [[0, 8], [25, 44], [93, 100]])
        assert_array_equal(snippets.det_mask, [[1, 1, 0, 0, 0, 0],
                                               [1, 1, 1, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 1]])

    def test_n_thres(self):
        tod = get_tod(n_dets=6, n_samps=100, events=[([0, 1], 10, 12), ([2, 3, 4], 50, 52)])
        snippets = gml.get_snippets(tod, tod.flags.glitches, n_thres=3, buffer=0)
        assert_array_equal(snippets.snippet_ranges.ranges(), [[50, 52]])
        assert_array_equal(snippets.det_mask, [[0, 0, 1, 1, 1, 0]])

    def test_no_snippets(self):
        tod = get_tod(n_dets=6, n_samps=100, events=[([0], 10, 12)])
        snippets = gml.get_snippets(tod, tod.flags.glitches)
        self.assertEqual(snippets.snippets.count, 0)
        self.assertEqual(snippets.det_mask.shape, (0, 6))

    def test_shape_mismatch(self):
        tod = get_tod(n_dets=6, n_samps=100)
        with self.assertRaises(ValueError):
            gml.get_snippets(tod, RangesMatrix([Ranges(100) for _ in range(5)]))


class StatsTest(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(gml.get_all_stat_names(), [
            "Number of Detectors", "Y and X Extent Ratio", "Mean abs(Correlation)",
            "Mean abs(Time Lag)", "Y Hist Max and Adjacent/Number of Detectors",
            "Within 0.1 of Y Hist Max/Number of Detectors", "Number of Peaks"])
        self.assertIs(gml.get_stat("Number of Detectors"), gml.num_dets)
        self.assertEqual(gml.get_stat("Y and X Extent Ratio").requires, ("x_pos", "y_pos"))

    def test_register_stat_unknown_input(self):
        with self.assertRaises(ValueError):
            @gml.register_stat("Test: Unknown Input")
            def bad_stat(signal, z_pos):
                return 0
        self.assertNotIn("Test: Unknown Input", gml.get_all_stat_names())

    def test_num_dets(self):
        self.assertEqual(gml.num_dets(x_pos=np.zeros(7)), 7)

    def test_yx_extent_ratio(self):
        self.assertEqual(gml.yx_extent_ratio(x_pos=np.array([0., 1., 2.]),
                                             y_pos=np.array([0., 0.5, 3.])), 1.5)
        self.assertEqual(gml.yx_extent_ratio(x_pos=np.ones(3), y_pos=np.arange(3.)), np.inf)
        self.assertTrue(np.isnan(gml.yx_extent_ratio(x_pos=np.ones(3), y_pos=np.ones(3))))

    def test_mean_abs_correlation(self):
        a = np.array([1., -1., 1., -1.])
        b = np.array([1., 1., -1., -1.])
        # Upper triangle including the diagonal: (1 + |r| + 1) / 3
        self.assertAlmostEqual(gml.mean_abs_correlation(np.array([a, b])), 2 / 3)
        self.assertAlmostEqual(gml.mean_abs_correlation(np.array([a, -a])), 1)
        self.assertAlmostEqual(gml.mean_abs_correlation(np.array([a])), 1)
        self.assertTrue(np.isnan(gml.mean_abs_correlation(np.ones((3, 1)))))

    def test_mean_abs_time_lag(self):
        pulse = np.zeros(20)
        pulse[5] = 1
        signal = np.array([pulse, np.roll(pulse, 2), np.roll(pulse, 5)])
        # Pairwise lags 2, 5 and 3
        self.assertAlmostEqual(gml.mean_abs_time_lag(signal), 10 / 3)
        # A forward shift by 18 of 20 samples is a backward shift by 2
        self.assertEqual(gml.mean_abs_time_lag(np.array([pulse, np.roll(pulse, 18)])), 2)
        self.assertTrue(np.isnan(gml.mean_abs_time_lag(np.array([pulse]))))
        self.assertTrue(np.isnan(gml.mean_abs_time_lag(np.ones((3, 1)))))

    def test_frac_max_adjacent_y(self):
        # Histogram bins are [0, 1), [1, 2), ..., [9, 10]
        self.assertEqual(gml.frac_max_adjacent_y(np.array([0, 3.5, 4.5, 4.5, 4.5, 5.5, 7.5, 10])), 5 / 8)
        # Peak in the first or last bin, which only has one neighbor
        self.assertEqual(gml.frac_max_adjacent_y(np.array([0, 0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 10])), 6 / 8)
        self.assertEqual(gml.frac_max_adjacent_y(np.array([0, 8.5, 9.5, 9.5, 10])), 4 / 5)

    def test_frac_near_max_y(self):
        # Bins are 0.06 wide; the peak bin is [0.24, 0.30), so the edges
        # within 0.1 of 0.24 are 0.18, 0.24 and 0.30, and only the bins
        # [0.18, 0.24) and [0.24, 0.30) have both edges within 0.1
        y = np.array([0, 0.21, 0.21, 0.27, 0.27, 0.27, 0.27, 0.33, 0.33, 0.33, 0.6])
        self.assertEqual(gml.frac_near_max_y(y), 6 / 11)
        # All bins are within 0.1 if the extent is below 0.1; none if above 1
        self.assertEqual(gml.frac_near_max_y(np.array([0, 0.01, 0.02, 0.05])), 1)
        self.assertEqual(gml.frac_near_max_y(np.array([0, 0, 0, 2])), 0)

    def test_num_peaks(self):
        signal = np.zeros((5, 200))
        signal[:, [50, 150]] = 10
        self.assertEqual(gml.num_peaks(signal), 2)
        # A spike in a single detector shows in the maximum across detectors
        signal = np.zeros((10, 200))
        signal[3, 100] = 10
        self.assertEqual(gml.num_peaks(signal), 1)
        self.assertEqual(gml.num_peaks(np.zeros((5, 200))), 0)


class ComputeSummaryStatsTest(unittest.TestCase):
    def test_stats_order(self):
        tod = get_tod(events=[([0, 1, 2], 100, 105), ([3, 4], 500, 503)])
        snippets = gml.get_snippets(tod, tod.flags.glitches)
        names = gml.get_all_stat_names()
        stats = gml.compute_summary_stats(tod, snippets)
        self.assertEqual(stats.shape, (2, len(names)))
        assert_array_equal(stats[:, 0], [3, 2])
        reordered = gml.compute_summary_stats(tod, snippets, stats=names[::-1])
        assert_array_equal(reordered, stats[:, ::-1])

    def test_signal_unchanged(self):
        # A snippet spanning all detectors used to detrend tod.signal in place
        tod = get_tod(events=[(range(8), 100, 105), ([3, 4], 500, 503)])
        signal = tod.signal.copy()
        gml.compute_summary_stats(tod, gml.get_snippets(tod, tod.flags.glitches))
        assert_array_equal(tod.signal, signal)

    def test_missing_positions(self):
        tod = get_tod(events=[([0, 1], 100, 105), ([2, 3, 4], 500, 503)], no_pointing=[0, 1, 4])
        snippets = gml.get_snippets(tod, tod.flags.glitches)
        with self.assertLogs("sotodlib.tod_ops.glitch_ml", level="WARNING"):
            stats = gml.compute_summary_stats(tod, snippets, stats=[
                "Number of Detectors", "Mean abs(Correlation)"])
        # No detector with a position: no stats
        self.assertTrue(np.all(np.isnan(stats[0])))
        # Detectors without positions are excluded from all stats
        start, stop = snippets.snippet_ranges.ranges()[1]
        signal = tod.signal[[2, 3], start:stop].copy()
        signal -= np.median(signal, axis=1, keepdims=True)
        self.assertEqual(stats[1, 0], 2)
        self.assertEqual(stats[1, 1], gml.mean_abs_correlation(signal))

    def test_detrend_method_and_custom_stat(self):
        @gml.register_stat("Test: Mean Signal")
        def mean_signal(signal):
            return np.mean(signal, dtype=np.float64)
        self.addCleanup(gml._STAT_REGISTRY.pop, "Test: Mean Signal")

        tod = get_tod(events=[([0, 1], 100, 103)])
        snippets = gml.get_snippets(tod, tod.flags.glitches)
        means = [gml.compute_summary_stats(tod, snippets, stats=["Test: Mean Signal"],
                                           detrend_method=method)[0, 0]
                 for method in ["mean", "median"]]
        # The glitch steps skew the median-detrended signal upward
        self.assertAlmostEqual(means[0], 0, places=4)
        self.assertGreater(means[1], 1)
        with self.assertRaises(ValueError):
            gml.compute_summary_stats(tod, snippets, detrend_method="bogus")


class ClassifySnippetsTest(unittest.TestCase):
    def setUp(self):
        # The last event is in a single pixel, so it has no finite extent ratio
        self.tod = get_tod(events=[([0, 1, 2], 100, 105), ([3, 4], 400, 403),
                                   ([0, 5], 700, 702), ([6, 7], 900, 903)],
                           same_pixel=[6, 7])
        self.forest = get_forest()

    def classify(self, forest, **kwargs):
        snippets = gml.get_snippets(self.tod, self.tod.flags.glitches)
        return gml.classify_snippets(self.tod, snippets, forest, **kwargs)

    def test_predictions(self):
        snippets = self.classify(self.forest)
        assert_array_equal(snippets.stat_names.vals, gml.get_all_stat_names())
        assert_array_equal(snippets.pred_cols.vals, gml.PRED_COLUMNS)
        self.assertEqual(snippets.detrend_method, "median")
        preds = snippets.predictions
        self.assertEqual(preds.shape, (4, len(gml.PRED_COLUMNS)))
        # Snippets with non-finite stats are not classified
        valid = np.all(np.isfinite(snippets.stats), axis=1)
        assert_array_equal(valid, [True, True, True, False])
        self.assertTrue(np.all(np.isnan(preds[~valid])))
        # Each classified snippet is in its original row
        for i in np.flatnonzero(valid):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                probs = self.forest.predict_proba(snippets.stats[i:i + 1])[0]
            assert_array_equal(preds[i, 1:], probs)
            self.assertEqual(preds[i, 0], np.argmax(probs))

    def test_forest_from_file(self):
        expected = self.classify(self.forest).predictions
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "forest.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.forest, f)
            for name in [path, path[:-len(".pkl")]]:
                assert_array_equal(self.classify(name).predictions, expected)

    def test_n_jobs(self):
        self.forest.n_jobs = 3
        self.classify(self.forest, n_jobs=None)
        self.assertEqual(self.forest.n_jobs, 3)
        self.classify(self.forest)
        self.assertEqual(self.forest.n_jobs, 1)

    def test_no_feature_name_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.classify(self.forest)
        self.assertFalse(any("feature names" in str(w.message) for w in caught))

    def test_wrong_classes(self):
        with self.assertRaises(ValueError):
            self.classify(get_forest(classes=3))

    def test_no_snippets(self):
        tod = get_tod(events=[([0], 100, 105)])
        snippets = gml.classify_snippets(tod, gml.get_snippets(tod, tod.flags.glitches), self.forest)
        self.assertEqual(snippets.predictions.shape, (0, len(gml.PRED_COLUMNS)))
        self.assertEqual(snippets.stats.shape, (0, len(gml.get_all_stat_names())))


class TrainForestTest(unittest.TestCase):
    def test_feature_names(self):
        self.assertEqual(list(get_forest().feature_names_in_), gml.get_all_stat_names())
        X, y = np.zeros((8, 2)), np.arange(8) % 4
        stats = ["Number of Detectors", "Number of Peaks"]
        self.assertEqual(list(gml.train_forest(X, y, stats=stats).feature_names_in_), stats)
        with self.assertRaises(ValueError):
            gml.train_forest(X, y)

    def test_n_jobs(self):
        rng = np.random.default_rng(1)
        X, y = rng.uniform(size=(200, 7)), np.arange(200) % 4
        forests = [gml.train_forest(X, y, n_trees=10, n_jobs=n_jobs) for n_jobs in [1, 2]]
        forests[1].n_jobs = 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            assert_array_equal(forests[0].predict_proba(X), forests[1].predict_proba(X))


class GlitchClassificationTest(unittest.TestCase):
    def run_step(self, events, tmp):
        tod = get_tod(n_samps=2000, offset=777, events=events)
        proc_aman = core.AxisManager(tod.dets, tod.samps)
        proc_aman.wrap("glitches", core.AxisManager(tod.dets, tod.samps))
        proc_aman.glitches.wrap("glitch_flags", tod.flags.glitches, [(0, "dets"), (1, "samps")])
        forest_name = os.path.join(tmp, "forest")
        with open(forest_name + ".pkl", "wb") as f:
            pickle.dump(get_forest(), f)
        step = GlitchClassification({"name": "classify_glitches",
                                     "calc": {"trained_forest_name": forest_name},
                                     "save": True})
        signal = tod.signal.copy()
        with self.assertLogs("preprocess", level="INFO") as logs:
            step.calc_and_save(tod, proc_aman)
        self.assertIn("glitch snippets", logs.output[0])
        assert_array_equal(tod.signal, signal)
        return proc_aman, forest_name

    def check_round_trip(self, proc_aman, tmp):
        path = os.path.join(tmp, "proc.h5")
        proc_aman.save(path, "proc")
        loaded = core.AxisManager.load(path, "proc").glitch_snippets
        saved = proc_aman.glitch_snippets
        for field in ["det_mask", "stats", "predictions"]:
            assert_array_equal(loaded[field], saved[field])
        assert_array_equal(loaded.snippet_ranges.ranges(), saved.snippet_ranges.ranges())
        for axis in ["stat_names", "pred_cols"]:
            assert_array_equal(loaded[axis].vals, saved[axis].vals)
        for field in ["trained_forest_used", "detrend_method"]:
            self.assertEqual(loaded[field], saved[field])

    def test_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc_aman, forest_name = self.run_step([(range(8), 500, 510), ([2, 5], 1200, 1205)], tmp)
            snippets = proc_aman.glitch_snippets
            self.assertEqual(snippets.snippets.count, 2)
            self.assertEqual(snippets.predictions.shape, (2, len(gml.PRED_COLUMNS)))
            self.assertEqual(snippets.trained_forest_used, os.path.basename(forest_name) + ".pkl")
            self.assertEqual(snippets.detrend_method, "median")
            self.check_round_trip(proc_aman, tmp)

    def test_step_no_snippets(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc_aman, _ = self.run_step([([3], 100, 105)], tmp)
            self.assertEqual(proc_aman.glitch_snippets.snippets.count, 0)
            self.check_round_trip(proc_aman, tmp)


if __name__ == "__main__":
    unittest.main()

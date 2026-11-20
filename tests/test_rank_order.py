import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data import raw
from data.constants import step_length
from data.rank_order import (
    CUTOFF_SEQUENCE_VERSION,
    antitonic_regression,
    project_ranked_cumulative,
)


class RankProjectionTests(unittest.TestCase):
    def test_pava_equal_weight_optimum_and_chained_blocks(self):
        np.testing.assert_array_equal(antitonic_regression([100, 120, 110, 0]), [110, 110, 110, 0])
        np.testing.assert_allclose(antitonic_regression([10, 0, 12, 8]), [10, 20/3, 20/3, 20/3])
        np.testing.assert_array_equal(antitonic_regression([]), [])

    def test_ordered_values_and_input_are_unchanged(self):
        values = np.array([[0., 0., 0.], [100., 80., 0.], [200., 90., 0.]])
        original = values.copy()
        result = project_ranked_cumulative(values, zero_tiers=[False, False, True])
        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(values, original)
        self.assertFalse(np.shares_memory(values, result))

    def test_projection_preserves_time_endpoints_and_structural_zero(self):
        values = np.array([[0., 0., 0.], [80., 100., 0.], [180., 130., 0.], [200., 160., 0.]])
        result = project_ranked_cumulative(values, zero_tiers=[False, False, True])
        np.testing.assert_array_equal(result[1], [90., 90., 0.])
        np.testing.assert_array_equal(result[[0, -1]], values[[0, -1]])
        self.assertTrue((np.diff(result, axis=0) >= 0).all())
        self.assertTrue((np.diff(result, axis=1) <= 0).all())

    def test_unobserved_padding_cannot_constrain_a_covered_rank(self):
        values = np.array([[0., 0.], [1., 1.], [1., 10.], [1., 20.]])
        result = project_ranked_cumulative(values, last_valid_indices=[1, 3])
        np.testing.assert_array_equal(result, values)

    def test_expiring_peer_cannot_cause_a_temporal_drop(self):
        values = np.array([[0., 0.], [0., 8.], [1., 8.], [12., 8.]])
        result = project_ranked_cumulative(values, last_valid_indices=[3, 1])
        np.testing.assert_array_equal(result[:, 0], [0., 4., 4., 12.])
        self.assertTrue((np.diff(result, axis=0) >= 0).all())

    def test_random_monotone_series_with_staggered_coverage(self):
        rng = np.random.default_rng(12)
        for _ in range(40):
            values = np.cumsum(rng.uniform(0, 100, size=(40, 21)), axis=0)
            limits = rng.integers(0, 40, size=21)
            result = project_ranked_cumulative(values, limits)
            self.assertTrue((np.diff(result, axis=0) >= -1e-10).all())
            for step, row in enumerate(result):
                self.assertTrue((np.diff(row[limits >= step]) <= 1e-10).all())

    def test_invalid_observations_are_not_silently_hidden(self):
        for values in ([[1., 2.], [0., 3.]], [[-1., 0.]], [[np.nan, 0.]]):
            with self.assertRaises(ValueError):
                project_ranked_cumulative(values)
        with self.assertRaisesRegex(ValueError, "structural zero"):
            project_ranked_cumulative([[0., 10.]], zero_tiers=[True, False])
        with self.assertRaises(ValueError):
            project_ranked_cumulative([[0., 0.]], last_valid_indices=[0.5, 0.5])


class SequenceRankIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.start = 1_700_000_000_000
        self.end = self.start + 4 * step_length
        self.events = {str(event): {"eventType": "story", "characters": []} for event in (309, 314)}
        for context in (
            patch.object(raw.config, "raw_dir", str(self.root)),
            patch.object(raw, "_load_local_metadata", return_value=(self.events, {})),
            patch.object(raw, "get_start_at", return_value=self.start),
            patch.object(raw, "get_end_at", return_value=self.end),
            patch.object(raw, "get_cn_holidays", return_value=set()),
        ):
            context.start()
            self.addCleanup(context.stop)

    def write_rank(self, event, rank, samples):
        path = self.root / '3' / str(event) / f'{rank}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"time": self.start + step * step_length, "ep": ep} for step, ep in samples]
        path.write_text(json.dumps(data), encoding='utf-8')
        if not data:
            report_path = self.root / '_merge_reports' / '3' / str(event) / f'{rank}.json'
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({
                'sources': {'bestdori': {'status': 'ok', 'received': 0}},
                'merge': {'merged': 0},
            }), encoding='utf-8')
        return path

    def generate(self, event_ids=None, tier_list=None, now=None):
        return raw.generate_sequences(
            event_ids=event_ids, server_list=[3], tier_list=tier_list,
            current_time_ms=self.end if now is None else now,
        )

    def test_single_rank_and_scan_use_the_same_event_projection(self):
        # Both raw curves are ordered at every shared timestamp. PCHIP
        # nevertheless crosses when they have different interior knots.
        first = self.write_rank(314, 500, [(0, 0), (1, 10), (4, 200)])
        second = self.write_rank(314, 1000, [(0, 0), (2, 100), (4, 160)])
        self.write_rank(314, 100000, [])
        original_files = {path: path.read_bytes() for path in (first, second)}
        all_seqs = self.generate([314])
        grid = np.arange(5) * step_length
        raw_curves = np.column_stack([
            raw.create_pchip_for_data(json.loads(path.read_text()), self.start, self.end)(grid)
            for path in (first, second)
        ])
        self.assertTrue((np.diff(raw_curves, axis=1) > 0).any())
        corrected = np.column_stack([np.r_[0., np.cumsum(seq['y'][:, 0], dtype=np.float64)] for seq in all_seqs])
        np.testing.assert_allclose(corrected[:, :2], project_ranked_cumulative(raw_curves), atol=1e-5)
        self.assertTrue((np.diff(corrected, axis=1) <= 1e-5).all())
        np.testing.assert_allclose(corrected[-1], [200., 160., 0.], atol=1e-5)

        single = self.generate([314], [1000])[0]
        scan = self.generate(tier_list=[1000])[0]
        np.testing.assert_array_equal(single['y'], all_seqs[1]['y'])
        np.testing.assert_array_equal(scan['y'], single['y'])
        for path, contents in original_files.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_supported_new_1500_rank_participates_even_when_not_requested(self):
        for event in (309, 314):
            self.write_rank(event, 1000, [(0, 0), (1, 10), (4, 200)])
            self.write_rank(event, 1500, [(0, 0), (2, 100), (4, 160)])
        old = self.generate([309], [1000])[0]
        new = self.generate([314], [1000])[0]
        self.assertFalse(np.allclose(old['y'], new['y']))
        self.assertEqual(self.generate([309], [1500]), [])

    def test_missing_file_is_not_an_empty_zero_rank(self):
        self.write_rank(314, 1000, [(0, 0), (4, 100)])
        seq = self.generate([314], [1000])[0]
        np.testing.assert_allclose(seq['y'].sum(), 100.)
        self.assertEqual(len(self.generate([314])), 1)

    def test_active_synthetic_end_cannot_lower_a_newer_lower_rank(self):
        self.write_rank(314, 500, [(0, 0), (1, 1)])
        path = self.write_rank(314, 1000, [(0, 0), (1, 1), (2, 10)])
        seq = self.generate([314], [1000], now=self.start + 2 * step_length)[0]
        np.testing.assert_allclose(np.sum(seq['y'][:2]), 10.)
        # Generation/projection did not move the real latest-observation bound.
        self.assertEqual(raw.get_latest_cutoff_time(314, 3, 1000), self.start + 2 * step_length)
        self.assertEqual(json.loads(path.read_text())[-1]['ep'], 10)

    def test_pchip_is_retained_and_missing_end_uses_last_ep(self):
        path = self.write_rank(314, 500, [(0, 0), (1, 30), (3, 100)])
        seq = self.generate([314])[0]
        curve = raw.create_pchip_for_data(json.loads(path.read_text()), self.start, self.end)
        expected = np.diff(curve(np.arange(5) * step_length))
        np.testing.assert_allclose(seq['y'][:, 0], expected, rtol=1e-6)
        np.testing.assert_allclose(seq['y'].sum(), 100.)
        self.assertEqual(float(seq['y'][-1, 0]), 0.)

    def test_hdf5_roundtrip_and_processing_version(self):
        import h5py
        from data.sequences import load_sequences_from_hdf5

        self.write_rank(314, 500, [(0, 0), (4, 100)])
        seqs = self.generate([314])
        path = self.root / 'sequences.h5'
        raw.save_sequences_to_hdf5(seqs, str(path))
        with h5py.File(path) as handle:
            self.assertEqual(handle['meta'].attrs['cutoff_sequence_version'], CUTOFF_SEQUENCE_VERSION)
        loaded = load_sequences_from_hdf5(str(path))
        np.testing.assert_array_equal(loaded[0]['y'], seqs[0]['y'])

    def test_legacy_hdf5_is_readable_but_warns_about_missing_projection(self):
        import h5py
        from data.sequences import load_sequences_from_hdf5

        self.write_rank(314, 500, [(0, 0), (4, 100)])
        path = self.root / 'legacy.h5'
        raw.save_sequences_to_hdf5(self.generate([314]), str(path))
        with h5py.File(path, 'a') as handle:
            del handle['meta'].attrs['cutoff_sequence_version']
        with self.assertWarnsRegex(RuntimeWarning, '尚未应用当前跨档保序'):
            self.assertEqual(len(load_sequences_from_hdf5(str(path))), 1)

    def test_preprocess_preserves_version_and_corrected_increment_inputs(self):
        import h5py
        import preprocess

        self.write_rank(309, 500, [(0, 0), (4, 100)])
        self.write_rank(314, 500, [(0, 0), (4, 100)])
        seqs = self.generate([309, 314])
        seqs[0]['split'], seqs[1]['split'] = 'train', 'val'
        source, target = self.root / 'raw.h5', self.root / 'pre.h5'
        raw.save_sequences_to_hdf5(seqs, str(source))
        with patch('sys.argv', ['preprocess.py', '--raw_h5', str(source), '--out_h5', str(target), '--t_min', '2']):
            with self.assertWarnsRegex(RuntimeWarning, '仅包含一种奖励规则'):
                preprocess.main()
        with h5py.File(target) as handle:
            self.assertEqual(handle['meta'].attrs['cutoff_sequence_version'], CUTOFF_SEQUENCE_VERSION)
            for split in ('train', 'val'):
                bucket = next(iter(handle[split].values()))
                th = int(bucket['hist_len'][0])
                np.testing.assert_array_equal(bucket['x_hist'][0, :th], seqs[0]['y'][:th])


class CacheNamespaceTests(unittest.TestCase):
    def test_default_cache_is_versioned_without_deleting_legacy_files(self):
        from service.cache import PredictionCache

        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'legacy.json'
            legacy.write_text('{"history": []}', encoding='utf-8')
            with patch('service.cache.os.makedirs'), patch.object(PredictionCache, '_load_all'):
                cache = PredictionCache()
            self.assertEqual(Path(cache.cache_dir).name, CUTOFF_SEQUENCE_VERSION)
            self.assertTrue(legacy.exists())
            custom = PredictionCache(cache_dir=directory)
            self.assertEqual(custom.cache_dir, directory)


if __name__ == '__main__':
    unittest.main()

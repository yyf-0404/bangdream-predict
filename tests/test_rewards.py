import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from data.rewards import RewardTable, REWARD_FEATURE_NAMES
from tests.test_data_contract import timestamp


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.table = RewardTable()

    def encode(self, when, ranks, event=226):
        return self.table.encode(3, event, timestamp(when + 'T15:00:00'), ranks)

    def test_star_intervals_before_and_after_2025_adjustment(self):
        ranks = [2000, 2001, 3000, 5000, 5001, 10000, 10001, 20000, 20001, 100000]
        old = self.encode('2025-09-25', ranks)
        new = self.encode('2025-09-26', ranks)
        np.testing.assert_allclose(old['reward_features'][:, 0], [.5,.25,.25,.25,.25,.25,.25,.25,.2,.2])
        np.testing.assert_allclose(new['reward_features'][:, 0], [1,.75,.75,.75,.5,.5,.25,.25,.2,.2])
        self.assertTrue(new['reward_feature_mask'][:, :4].all())

    def test_voice_known_from_226_and_2026_cosmetic_rewards_stack(self):
        old = self.encode('2024-03-14', [500, 1000, 1500, 2000])
        np.testing.assert_array_equal(old['reward_features'][:, 1:4], [[1,0,0],[1,0,0],[0,0,0],[0,0,0]])
        new = self.encode('2026-05-26', [500, 1000, 1500, 2000], event=311)
        np.testing.assert_array_equal(new['reward_features'][:, 1:4], [[1,1,1],[1,0,1],[1,0,0],[0,0,0]])
        self.assertTrue(old['reward_feature_mask'][:, :4].all())
        self.assertFalse(old['reward_feature_mask'][:, 6:].any())

    def test_distance_on_both_sides_not_masked_by_award_eligibility(self):
        result = self.encode('2026-05-26', [1000, 1500, 2000], event=311)
        distances = result['reward_features'][:, 5]
        self.assertLess(distances[0], 0)
        self.assertEqual(distances[1], 0)
        self.assertGreater(distances[2], 0)
        self.assertTrue(result['reward_feature_mask'][:, 5].all())
        self.assertEqual(result['reward_features'][-1, 1], 0)

    def test_old_ids_312_313_use_actual_dates_not_id_order(self):
        for event in (312, 313):
            result = self.encode('2025-11-10', [1500], event)
            self.assertEqual(result['reward_stage'], 'from_2025_09_26')
            np.testing.assert_array_equal(result['reward_features'][0, :4], [1,0,0,0])
        result = self.encode('2026-06-05', [1500], 310)
        np.testing.assert_array_equal(result['reward_features'][0, :4], [1,1,0,0])

    def test_audit_stage_name_never_affects_content_features(self):
        document = json.loads(json.dumps(self.table.document))
        for index, stage in enumerate(document['stages']):
            stage['name'] = f'unrelated-name-{index}'
        document['version'] = 'other-audit-version'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rewards.json'
            path.write_text(json.dumps(document))
            other = RewardTable(path).encode(3, 311, timestamp('2026-05-26T15:00:00'), [500,1000,1500])
        original = self.encode('2026-05-26', [500,1000,1500], 311)
        np.testing.assert_array_equal(other['reward_features'], original['reward_features'])
        np.testing.assert_array_equal(other['reward_feature_mask'], original['reward_feature_mask'])
        self.assertNotEqual(other['reward_schedule_sha256'], original['reward_schedule_sha256'])
        self.assertEqual(len(REWARD_FEATURE_NAMES), 8)

    def test_unknown_scope_is_not_silently_extrapolated(self):
        for ranks in ([100001], [0], [1500.5]):
            with self.assertRaises(ValueError):
                self.encode('2026-05-26', ranks)
        with self.assertRaises(ValueError):
            self.encode('2024-03-01', [1000], 225)
        with self.assertRaises(ValueError):
            self.table.encode(0, 226, timestamp('2026-05-26T15:00:00'), [1000])


if __name__ == '__main__':
    unittest.main()

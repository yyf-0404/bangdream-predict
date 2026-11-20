"""Cache changes must preserve every input, mask, weight and source integrity check."""
import os
import shutil
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from data.joint_dataset import JointDataset, validate_joint_event
from tests import test_joint_data


class JointCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_joint_data.JointDataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.save()
        self.path = self.fixture.path

    def test_all_cache_validates_each_event_once_and_preserves_every_batch(self):
        before = self.path.read_bytes()
        cached, lru = JointDataset(self.path, 'train'), JointDataset(self.path, 'train', cache_events=1)
        options = dict(shuffle=True, seed=9, boundaries=(4,9,18))
        with patch('data.joint_dataset.validate_joint_event', wraps=validate_joint_event) as validate:
            first = list(cached.batches(5, **options))
            second = list(cached.batches(5, **options))
            self.assertEqual(validate.call_count, 2)
        legacy = list(lru.batches(5, **options))
        for a,b,c in zip(first, second, legacy):
            for index in (0,1):
                for key in a[index]:
                    np.testing.assert_array_equal(a[index][key], b[index][key])
                    np.testing.assert_array_equal(a[index][key], c[index][key])
            np.testing.assert_array_equal(a[2], b[2])
            np.testing.assert_array_equal(a[2], c[2])
        self.assertEqual(cached.cache_info()['misses'], 2)
        self.assertEqual(cached.cache_info()['evictions'], 0)
        self.assertGreater(lru.cache_info()['misses'], 2)
        self.assertEqual(cached.norm, lru.norm)
        self.assertEqual(cached.weight_report, lru.weight_report)
        self.assertEqual(self.path.read_bytes(), before)

    def test_cached_arrays_are_read_only_but_slices_remain_independent(self):
        data = JointDataset(self.path, 'train')
        event = data.event(232)
        with self.assertRaises(ValueError):
            event['cutoff'][0,0] = 10
        x, _, _ = data.slice(0)
        x['cutoff_hist'][0,0] = 123
        self.assertEqual(data.event(232)['cutoff'][0,0], 0)
        self.assertGreater(data.cache_info()['resident_array_bytes'], 0)

    def test_hit_detects_in_place_edit_and_atomic_replacement(self):
        data = JointDataset(self.path, 'train')
        data.event(232)
        replacement = self.path.with_suffix('.copy')
        shutil.copyfile(self.path, replacement)
        os.replace(replacement, self.path)
        with self.assertRaisesRegex(ValueError, 'changed during iteration'):
            data.event(232)
        data = JointDataset(self.path, 'train')
        data.event(232)
        with h5py.File(self.path, 'a') as handle:
            handle['meta'].attrs['data_snapshot_sha256'] = 'modified'
        with self.assertRaisesRegex(ValueError, 'changed during iteration'):
            data.event(232)

    def test_initial_event_validation_is_not_skipped(self):
        with h5py.File(self.path, 'a') as handle:
            handle['events/232/cutoff'][1,0] = -1
        data = JointDataset(self.path, 'train')
        with self.assertRaises(ValueError):
            data.event(232)

    def test_capacity_and_split_guards(self):
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                JointDataset(self.path, 'train', cache_events=invalid)
        data = JointDataset(self.path, 'train')
        self.assertEqual(data.cache_events, 2)
        with self.assertRaisesRegex(ValueError, 'split'):
            data.event(234)

"""Tiny raw -> joint HDF5 -> finite batches integration, with no training framework."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from data.constants import step_length, tier_list_for_event
from data.cutoff_quality import cutoff_digest
from data.event_split import EventSplit
from data.joint_dataset import (JointDataset, fit_joint_normalization, prepare_joint_artifact,
                                save_joint_hdf5)
from data.joint_sequences import build_joint_event, JOINT_RANKS, validate_joint_event
from data.joint_slicing import MODEL_INPUT_KEYS, joint_model_input, joint_training_slice
from data.joint_weights import build_joint_weight_plan
from data.rewards import RewardTable
from tests.test_data_contract import timestamp


class JointDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = EventSplit({
            'strategy':'fixed-rule-type-holdout', 'version':'tiny-joint-v1', 'server':3,
            'temporal_holdout':False,
            'events_by_type': {'challenge': {'old':[232,234], 'new':[311,317]}},
            'validation_by_type': {'challenge': {'old':[234], 'new':[317]}},
        })
        self.rewards = RewardTable()
        dates = {232:'2024-04-01',234:'2024-04-20',311:'2026-05-26',317:'2026-07-16'}
        self.metadata = {}
        for event_id, day in dates.items():
            start, end = timestamp(day+'T15:00:00'), timestamp(day+'T23:00:00')
            self.metadata[str(event_id)] = dict(eventType='challenge', characters=[],
                                               startAt=[None]*3+[str(start)], endAt=[None]*3+[str(end)])
            for rank in tier_list_for_event(3, event_id):
                points = [] if rank == 100000 else [
                    {'time':start, 'ep':0},
                    {'time':end, 'ep':int(1e8/rank)*(2 if event_id in (234,317) else 1)},
                ]
                self.write_rank(event_id, rank, points)
        self.events = [self.build(e) for e in sorted(self.manifest.event_ids)]
        self.train = [e for e in self.events if e['split'] == 'train']
        self.val = [e for e in self.events if e['split'] == 'val']
        self.path = self.root / 'joint.h5'

    def write_rank(self, event_id, rank, points):
        path = self.root / '3' / str(event_id) / f'{rank}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(points))
        report = self.root / '_merge_reports' / '3' / str(event_id) / f'{rank}.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({'cutoff_sha256':cutoff_digest(points),
                                     'sources':{'bestdori':{'status':'ok','received':len(points)}},
                                     'merge':{'merged':len(points)}}))

    def build(self, event_id, **options):
        now = options.pop('current_time_ms', timestamp('2026-09-01T00:00:00'))
        return build_joint_event(event_id, raw_dir=self.root, metadata=self.metadata, characters={},
                                 reward_table=self.rewards, event_split=self.manifest,
                                 current_time_ms=now, **options)

    def save(self):
        return save_joint_hdf5(self.events, self.path, event_split=self.manifest,
                               reward_table=self.rewards, t_min=2)

    def test_old_1500_is_unknown_and_confirmed_zero_is_valid(self):
        index1500, index100000 = JOINT_RANKS.index(1500), JOINT_RANKS.index(100000)
        for event in self.events:
            self.assertEqual(event['cutoff'].shape, (17,21))
            self.assertTrue(event['available_mask'][:, index100000].all())
            self.assertTrue(event['target_mask'][:, index100000].all())
            self.assertTrue((event['cutoff'][:, index100000] == 0).all())
            if event['balance_group'] == 0:
                self.assertFalse(event['available_mask'][:, index1500].any())
                self.assertFalse(event['target_mask'][:, index1500].any())
            else:
                self.assertTrue(event['target_mask'][:, index1500].all())
        self.assertTrue(self.events[0]['reward_feature_mask'][index1500, :4].all())

    def test_no_supported_file_is_silently_dropped_from_training(self):
        (self.root / '3/311/1500.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing supported'):
            self.build(311)
        incomplete = self.build(311, require_complete=False)
        broken = [incomplete if e['event_id'] == 311 else e for e in self.events]
        with self.assertRaisesRegex(ValueError, 'Missing joint training rank'):
            prepare_joint_artifact(broken, self.manifest, t_min=2)

    def test_tail_forward_fill_keeps_real_coverage_and_provenance(self):
        event = self.events[0]
        start, end = event['start_at'], event['end_at']
        self.write_rank(232, 20, [{'time':start,'ep':0}, {'time':end-step_length,'ep':10000000}])
        rebuilt = self.build(232)
        self.assertEqual(rebuilt['latest_observation_ms'][0], end-step_length)
        self.assertEqual(rebuilt['provenance'][-1,0], 4)
        self.assertTrue(rebuilt['target_mask'][-1,0])
        self.assertEqual(rebuilt['cutoff'][-1,0], rebuilt['cutoff'][-2,0])

    def test_active_mask_requested_coverage_and_unknown_placeholders(self):
        start = self.events[2]['start_at']  # event 311
        self.write_rank(311, 500, [{'time':start,'ep':0},{'time':start+3*step_length,'ep':1000}])
        self.write_rank(311, 1000, [{'time':start,'ep':0},{'time':start+5*step_length,'ep':1000}])
        active = self.build(311, current_time_ms=start+5*step_length)
        self.assertFalse(active['target_mask'].any())
        x = joint_model_input(active, 4, requested_rank=1000)
        self.assertFalse(x['anchor_mask'][JOINT_RANKS.index(500)])
        self.assertEqual(x['anchor'][JOINT_RANKS.index(500)], 0)
        self.assertTrue(x['anchor_mask'][JOINT_RANKS.index(1000)])
        with self.assertRaisesRegex(ValueError, 'Requested rank'):
            joint_model_input(active, 4, requested_rank=500)
        with self.assertRaisesRegex(ValueError, 'Active events'):
            joint_training_slice(active, 4, 1., np.ones(21))

    def test_loss_masks_and_audit_metadata_are_not_network_inputs(self):
        event = copy.deepcopy(self.events[0])
        original = joint_model_input(event, 3)
        event['target_mask'][:] = False
        event['balance_group'] = 1
        event['reward_stage'] = 'unrelated'
        event['reward_schedule_version'] = 'changed-audit-label'
        event['event_id'] = 99999
        changed = joint_model_input(event, 3)
        self.assertEqual(set(original), MODEL_INPUT_KEYS)
        for key in original:
            np.testing.assert_array_equal(original[key], changed[key])

    def test_masks_shapes_partial_supervision_and_leakage_rejected(self):
        bad = copy.deepcopy(self.events[0])
        bad['target_mask'][-1,0] = False
        with self.assertRaisesRegex(ValueError, 'Partial future'):
            validate_joint_event(bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate joint'):
            prepare_joint_artifact(self.events + [self.events[0]], self.manifest, t_min=2)
        bad = copy.deepcopy(self.events)
        bad[0]['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'fixed split'):
            prepare_joint_artifact(bad, self.manifest, t_min=2)

    def test_joint_effective_weights_equal_with_1500_and_zero_supervision(self):
        plan = build_joint_weight_plan(self.train, t_min=2)
        self.assertEqual(plan.report['joint_samples'], 28)
        self.assertEqual(plan.report['rank_supervision_units'], [280,294])
        np.testing.assert_allclose(plan.report['rule_final_weight_sums'], [14,14])
        weights, groups = [], []
        by_id = {e['event_id']:e for e in self.train}
        for row, (event_id, t0) in enumerate(plan.index):
            event = by_id[event_id]
            _, _, w = joint_training_slice(event, int(t0), plan.sample_scale[row], plan.tier_weights)
            weights.append(w)
            groups.append(event['balance_group'])
            if event['balance_group'] == 0:
                self.assertEqual(w[JOINT_RANKS.index(1500)], 0)
            self.assertGreater(w[-1], 0)  # true zero participates
        weights, groups = np.array(weights), np.array(groups)
        np.testing.assert_allclose([weights[groups==g].sum() for g in (0,1)], [14,14])
        self.assertAlmostEqual(weights.sum(axis=1).mean(), 1.)
        self.assertEqual(plan.tier_weights[JOINT_RANKS.index(1500)], 1.)
        self.assertEqual(plan.tier_weights[JOINT_RANKS.index(1000)], 1.)

    def test_validation_uses_one_global_factor_and_no_training_statistics(self):
        plan = build_joint_weight_plan(self.val, t_min=2, balance_rules=False)
        self.assertFalse(plan.report['rule_balance_applied'])
        self.assertEqual(*plan.report['rule_factors'])
        self.assertNotEqual(*plan.report['rule_final_weight_sums'])
        with self.assertRaisesRegex(ValueError, 'Validation weights'):
            build_joint_weight_plan(self.val, t_min=2)
        with self.assertRaisesRegex(ValueError, 'separately'):
            build_joint_weight_plan(self.events, t_min=2)

    def test_normalization_counts_events_once_and_ignores_unknown_1500(self):
        _, _, norm = prepare_joint_artifact(self.events, self.manifest, t_min=2)
        expected = np.concatenate([np.log1p(e['cutoff'][e['available_mask']]) for e in self.train])
        self.assertAlmostEqual(norm['log_cutoff_mean'], expected.mean())
        self.assertAlmostEqual(norm['log_cutoff_std'], expected.std())
        self.assertEqual(norm['valid_grid_values'], expected.size)
        changed = copy.deepcopy(self.events)
        for event in changed:
            if event['split'] == 'val':
                event['cutoff'] *= 1000
        _, _, other = prepare_joint_artifact(changed, self.manifest, t_min=2)
        self.assertEqual(norm, other)
        with self.assertRaisesRegex(ValueError, 'training events only'):
            fit_joint_normalization(self.val)

    def test_hdf5_roundtrip_finite_batches_padding_tails_and_bounded_cache(self):
        reports = self.save()
        self.assertEqual(reports['train']['joint_samples'], 28)
        dataset = JointDataset(self.path, 'train', cache_events=1)
        batches = list(dataset.batch_rows(5, shuffle=True, seed=9, boundaries=(4,9,18)))
        rows = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(rows), np.arange(28))
        self.assertTrue(any(len(b) < 5 for b in batches))
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(batches, dataset.batch_rows(5, shuffle=True, seed=9, boundaries=(4,9,18)))))
        by_id = {e['event_id']:e for e in self.events}
        x, y, weights = dataset.slice(0)
        event_id, t0 = map(int, dataset.index[0])
        original = joint_model_input(by_id[event_id], t0)
        for key in original:
            np.testing.assert_array_equal(x[key], original[key])
        np.testing.assert_array_equal(y['cutoff_future'], by_id[event_id]['cutoff'][t0+1:].astype(np.float32))
        sample_count, total_mass = 0, 0.
        for x, y, w in dataset.batches(5, shuffle=True, seed=9, boundaries=(4,9,18)):
            sample_count += len(w)
            total_mass += w.sum()
            self.assertEqual(set(x), MODEL_INPUT_KEYS)
            self.assertLessEqual(len(dataset._cache), 1)
            for row, length in enumerate(x['future_len']):
                self.assertFalse(y['target_mask'][row, length:].any())
                self.assertFalse(x['future_time_mask'][row, length:].any())
                self.assertTrue((x['future_time_features'][row, length:] == 0).all())
        self.assertEqual(sample_count, 28)
        self.assertAlmostEqual(total_mass, 28.)
        with h5py.File(self.path) as handle:
            self.assertEqual(len(handle['events']), 4)
            self.assertNotIn('x_hist', handle['train'])
            self.assertEqual(handle['train/index'].shape, (28,2))

    def test_refuses_overwrite_and_tampered_cut_index(self):
        self.save()
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(self.path.read_bytes(), original)
        with h5py.File(self.path, 'a') as handle:
            handle['train/index'][0] = [232,999]
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            JointDataset(self.path, 'train')

    def test_cli_audit_and_generation_use_independent_v2_path(self):
        import generate_joint_h5
        (self.root / 'events_all.json').write_text(json.dumps(self.metadata))
        (self.root / 'characters_main.json').write_text('{}')
        manifest_path = self.root / 'split.json'
        manifest_path.write_text(json.dumps(self.manifest.manifest))
        args = ['generate_joint_h5.py','--raw_dir',str(self.root),'--split_manifest',str(manifest_path),
                '--t_min','2','--out_h5',str(self.path)]
        with patch.object(sys, 'argv', args + ['--audit_only']), contextlib.redirect_stdout(io.StringIO()):
            generate_joint_h5.main()
        self.assertFalse(self.path.exists())
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
            generate_joint_h5.main()
        self.assertEqual(len(JointDataset(self.path, 'val').index), 28)

    def test_validation_extension_reuses_parent_events_and_weight_settings(self):
        import generate_joint_h5
        parent = copy.deepcopy(self.manifest.manifest)
        parent['version'] = 'tiny-parent'
        for field in ('events_by_type', 'validation_by_type'):
            parent[field]['challenge']['new'].remove(317)
        base = self.root / 'parent.h5'
        save_joint_hdf5([e for e in self.events if e['event_id'] != 317], base,
                        event_split=EventSplit(parent), reward_table=self.rewards, t_min=2)
        (self.root / 'events_all.json').write_text(json.dumps(self.metadata))
        (self.root / 'characters_main.json').write_text('{}')
        manifest_path = self.root / 'split.json'
        manifest_path.write_text(json.dumps(dict(self.manifest.manifest, parent_version='tiny-parent')))
        args = ['generate_joint_h5.py', '--raw_dir', str(self.root), '--split_manifest', str(manifest_path),
                '--extend-validation-from', str(base), '--out_h5', str(self.path)]
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(generate_joint_h5, 'build_joint_event', wraps=build_joint_event) as build:
            generate_joint_h5.main()
        self.assertEqual([call.args[0] for call in build.call_args_list], [317])
        old_train, new_train = (JointDataset(path, 'train') for path in (base, self.path))
        self.assertEqual(new_train.norm, old_train.norm)
        np.testing.assert_array_equal(new_train.sample_scale, old_train.sample_scale)
        self.assertEqual(JointDataset(self.path, 'val').event_ids, {234, 317})


if __name__ == '__main__':
    unittest.main()

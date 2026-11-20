import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data import raw
from data.constants import step_length
from data.joint_sequences import build_joint_event
from data.joint_slicing import joint_model_input
from model.conditional_curve import DEFAULT_MODEL_PATH, VERSION
from service.conditional_forecasts import ConditionalForecasts
from utils.files import file_sha256


class ConditionalReleaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        date = 1788505200000
        info = dict(startAt=[None]*3+[date], endAt=[None]*3+[date+48*step_length],
                    characters=[], eventType='challenge')
        self.meta = {str(eid):info for eid in (306, 328)}
        (self.root/'events_all.json').write_text(json.dumps(self.meta))
        (self.root/'characters_main.json').write_text('{}')
        for target, key, value in [(raw.config,'raw_dir',str(self.root)), (raw,'_event_data_cache',None),
                                   (raw,'_char_data_cache',None), (raw,'download_cutoffs',lambda **opts: None)]:
            change = patch.object(target,key,value)
            change.start()
            self.addCleanup(change.stop)
        self.start = raw.get_start_at(3,328)
        self.end = raw.get_end_at(3,328)
        self.steps = (self.end-self.start)//step_length
        for eid in (306,328):
            folder = self.root/'3'/str(eid)
            folder.mkdir(parents=True)
            for j, tier in enumerate((500,1000,1500,2000)):
                if eid == 306 and tier == 1500:
                    continue
                points = [dict(time=self.start+t*step_length, ep=t*(7-j)*1000)
                          for t in (12, 48, 51, 57, self.steps)]
                (folder/f'{tier}.json').write_text(json.dumps(points))
        self.service = ConditionalForecasts(DEFAULT_MODEL_PATH,cache_root=self.root/'cache')

    def test_packaged_model_and_filename(self):
        model = self.service.predictor.model
        self.assertEqual(file_sha256(DEFAULT_MODEL_PATH),'867e2ffac8b320509675dae290e6771b2b5cc2f2c33a201ab6e22fef211e320b')
        self.assertEqual(model.metadata['version'],VERSION)
        self.assertEqual(model.metadata['fit_scope'],'train')
        self.assertEqual(len(model.metadata['train_events']),79)
        self.assertFalse(set(range(322,329)).intersection(model.metadata['train_events']))

    def test_training_entry_defaults_to_the_published_model_and_training_split(self):
        import train
        from config import CURVE_DATA
        with patch.object(sys, 'argv', ['train.py']), patch('training.conditional_curve.train_conditional_curve') as fit:
            train.main()
        self.assertEqual(fit.call_args.args, (CURVE_DATA, 'outputs/conditional-curve'))
        self.assertEqual(fit.call_args.kwargs['fit_scope'], 'train')
        self.assertFalse(fit.call_args.kwargs.get('observed_covariance', False))

    def test_existing_joint_entry_reads_the_published_artifact(self):
        from service.joint_predictor import JointPredictor
        old_entry = JointPredictor()
        issue = self.start + 54*step_length
        a = old_entry.predict_as_of(328, issue, ranks=[500,1000], raw_dir=self.root, metadata=self.meta, characters={})
        b = self.service.predictor.predict_as_of(328, issue, ranks=[500,1000], raw_dir=self.root, metadata=self.meta, characters={})
        self.assertEqual(a['model_sha256'], b['model_sha256'])
        np.testing.assert_allclose(a['pred_cutoff'], b['pred_cutoff'], atol=1e-6, rtol=2e-11)

    def test_live_and_finished_http_payloads_match_independent_truncation(self):
        issue = self.start+54*step_length+1000
        reply = self.service.get_predictions(328,3,1500,now_ms=issue)
        latest = reply['latest_slice']
        self.assertEqual(latest['t0'],51)
        self.assertEqual(latest['issued_at_ms'],issue)
        self.assertEqual(latest['anchor_at_ms'], self.start+51*step_length)
        event = build_joint_event(328,raw_dir=self.root,metadata=self.meta,characters={},current_time_ms=issue,
                                  reward_table=self.service.predictor.model.reward_table,require_complete=False,recorded_only=True)
        x = joint_model_input(event,51)
        _, direct = self.service.predictor.model.predict(x)
        column = event['ranks'].tolist().index(1500)
        self.assertEqual([p['time'] for p in latest['prefix_curve']], event['grid_ms'][52:].tolist())
        self.assertEqual([p['ep'] for p in latest['prefix_curve']], direct[:,column].astype(int).tolist())
        self.assertEqual(reply['history'][-1]['t0'],54)
        self.assertEqual(reply['history'][-1]['anchor_t0'],51)
        self.assertEqual(reply['history'][-1]['ep'],latest['total_sum_hat'])
        completed = self.service.get_predictions(328,3,1500,now_ms=self.end)
        self.assertIsNone(completed['latest_slice'])
        self.assertEqual(completed['history'][:len(reply['history'])],reply['history'])
        self.assertEqual(completed['history'][-1]['t0'],self.steps-1)
        reader = ConditionalForecasts(DEFAULT_MODEL_PATH, cache_root=self.root/'cache')
        with patch.object(reader.predictor,'predict_at',side_effect=AssertionError('cache miss')):
            self.assertEqual(reader.get_predictions(328,3,1500,now_ms=self.end),completed)

    def test_coverage_and_snapshot_changes(self):
        self.assertIsNone(self.service.get_predictions(306,3,1500,now_ms=self.end))
        self.assertIsNone(self.service.get_predictions(328,3,500,now_ms=self.start+47*step_length))
        self.assertIsNone(self.service.get_predictions(328,1,500,now_ms=self.end))
        self.assertIsNone(self.service.get_predictions(328,3,70000,now_ms=self.end))
        before = self.service.get_predictions(328,3,1500,now_ms=self.end)
        path = self.root/'3/328/1500.json'
        points = json.loads(path.read_text())
        points[3]['ep'] += 4000
        path.write_text(json.dumps(points))
        after = self.service.get_predictions(328,3,1500,now_ms=self.end)
        self.assertEqual(before['history'][:9],after['history'][:9])
        self.assertNotEqual(before['history'][-1]['ep'],after['history'][-1]['ep'])
        self.assertIsNotNone(self.service.get_predictions(306,3,500,now_ms=self.end))

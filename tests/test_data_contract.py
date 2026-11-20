"""Data correctness checks, including saved empty provenance and host timezone independence."""
import contextlib
from datetime import datetime, date
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from data import raw
from data.cutoff_quality import cutoff_digest, load_cutoff_snapshot
from data.cutoff_sources import fetch_and_merge_cutoffs
from data.time_features import server_datetime, time_features


def timestamp(local):
    return int(datetime.fromisoformat(local).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp() * 1000)


class TimeContractTests(unittest.TestCase):
    def test_cn_boundaries_and_features_are_independent_of_host_timezone(self):
        original = os.environ.get('TZ')
        metadata = {str(e): {'startAt': [None]*3 + [timestamp('2026-05-26T10:00:00')],
                              'endAt': [None]*3 + [timestamp('2026-06-03T22:59:59')]}
                    for e in (226, 293, 312, 313)}
        results = []
        try:
            for zone in ('UTC', 'Asia/Shanghai', 'America/Los_Angeles'):
                os.environ['TZ'] = zone
                time.tzset()
                bounds = [(raw.get_start_at(3, e, metadata=metadata), raw.get_end_at(3, e, metadata=metadata))
                          for e in (226, 293, 312, 313)]
                features = time_features([timestamp('2026-05-26T23:30:00')], holidays=set())
                results.append((bounds, features.tolist()))
        finally:
            if original is None:
                os.environ.pop('TZ', None)
            else:
                os.environ['TZ'] = original
            time.tzset()
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        self.assertEqual([server_datetime(start).hour for start, _ in results[0][0]], [15, 16, 14, 14])
        self.assertEqual([server_datetime(end).hour for _, end in results[0][0]], [23, 23, 20, 20])
        self.assertEqual(results[0][1][0][0], 47)

    def test_all_four_holiday_states_are_retained(self):
        stamps = [timestamp(f'2026-05-{d:02d}T12:00:00') for d in range(1, 5)]
        features = time_features(stamps, holidays={date(2026, 5, 2), date(2026, 5, 3)})
        np.testing.assert_array_equal(features[:, 2], [1, 2, 3, 0])


class EmptyContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / '3' / '309' / '100000.json'
        self.report_path = self.root / '_merge_reports' / '3' / '309' / '100000.json'
        self.path.parent.mkdir(parents=True)
        self.report_path.parent.mkdir(parents=True)
        self.start = timestamp('2026-05-16T15:00:00')
        self.end = timestamp('2026-05-16T23:00:00')

    def fetch(self, payload):
        return fetch_and_merge_cutoffs(3, 309, 100000, start_at=self.start, end_at=self.end,
                                      fetch_json=lambda url: payload)

    def save(self, points, report=None):
        self.path.write_text(json.dumps(points))
        if report is not None:
            self.report_path.write_text(json.dumps(report))

    def load(self):
        return load_cutoff_snapshot(self.root, 3, 309, 100000)

    def test_nonempty_illegal_responses_are_failures_not_structural_zero(self):
        points, report = self.fetch({'cutoffs': [{'time': 'bad', 'ep': -1}]})
        self.assertEqual(points, [])
        self.assertEqual(report['successful_sources'], [])
        self.assertEqual(report['result_status'], 'unknown')
        self.assertTrue(all(s['status'] == 'filtered_empty' for s in report['sources'].values()))
        self.save(points, report)
        with self.assertRaisesRegex(ValueError, 'not a confirmed'):
            self.load()

    def test_outside_window_only_is_not_evidence_of_insufficient_players(self):
        _, report = self.fetch({'cutoffs': [{'time': self.end + 1, 'ep': 123}]})
        self.assertEqual(report['successful_sources'], [])
        self.assertEqual(report['result_status'], 'unknown')

    def test_real_empty_and_legacy_empty_evidence_are_supported(self):
        points, report = self.fetch({'cutoffs': []})
        self.assertEqual(report['result_status'], 'structural_zero')
        self.save(points, report)
        self.assertEqual(self.load(), [])
        self.save([], {'sources': {'bestdori': {'status':'ok', 'received':0}}, 'merge':{'merged':0}})
        self.assertEqual(self.load(), [])

    def test_old_invalid_ok_report_and_missing_report_cannot_prove_zero(self):
        self.save([])
        with self.assertRaisesRegex(ValueError, 'not a confirmed'):
            self.load()
        self.save([], {'sources': {'bestdori': {'status':'ok', 'received':1, 'valid':0}}, 'merge':{'merged':0}})
        with self.assertRaisesRegex(ValueError, 'not a confirmed'):
            self.load()

    def test_hash_mismatch_and_invalid_saved_points_are_rejected(self):
        points = [{'time': self.start, 'ep': 0}]
        self.save(points, {'cutoff_sha256':cutoff_digest([])})
        with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
            self.load()
        self.report_path.unlink()
        for bad in ([{'time':self.start, 'ep':-1}], [{'time':self.start, 'ep':True}],
                    [{'time':self.start, 'ep':2}, {'time':self.end, 'ep':1}]):
            self.save(bad)
            with self.assertRaisesRegex(ValueError, 'non-monotone'):
                self.load()

    def test_one_real_empty_can_coexist_with_failed_sources(self):
        points, report = fetch_and_merge_cutoffs(3, 309, 100000, start_at=self.start, end_at=self.end,
            fetch_json=lambda url: {'cutoffs':[]} if 'bestdori.com' in url else {'cutoffs':[{'ep':-1}]})
        self.assertEqual(report['successful_sources'], ['bestdori'])
        self.save(points, report)
        self.assertEqual(self.load(), [])

    def test_download_failure_preserves_old_raw_and_its_report(self):
        points = [{'time':self.start, 'ep':0}, {'time':self.end, 'ep':100}]
        old_report = {'cutoff_sha256':cutoff_digest(points)}
        self.save(points, old_report)
        originals = self.path.read_bytes(), self.report_path.read_bytes()
        failed = self.fetch({'cutoffs':[{'time':'bad', 'ep':-1}]})
        with (patch.object(raw.config, 'raw_dir', str(self.root)),
              patch.object(raw, '_cache_metadata'),
              patch.object(raw, '_load_local_metadata', return_value=({'309':{'startAt':[None]*3+[self.start]}}, {})),
              patch.object(raw, 'get_start_at', return_value=self.start),
              patch.object(raw, 'get_end_at', return_value=self.end),
              patch.object(raw, 'fetch_and_merge_cutoffs', return_value=failed),
              contextlib.redirect_stdout(io.StringIO())):
            raw.download_cutoffs([309], [3], [100000], force=True)
        self.assertEqual((self.path.read_bytes(), self.report_path.read_bytes()), originals)
        self.assertTrue((self.root / '_fetch_reports/3/309/100000.json').exists())


if __name__ == '__main__':
    unittest.main()

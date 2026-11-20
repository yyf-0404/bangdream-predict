"""Serve joint, as-of forecasts through the existing per-tier HTTP contract."""
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from data import raw
from data.constants import is_supported_tier, step_length, tier_list_for_event
from data.curve_history import read_observations
from model.conditional_curve import DAY
from service.conditional_predictor import ConditionalPredictor
from utils.http_utils import save_json

logger = logging.getLogger(__name__)


def history_signature(event, series):
    digest = hashlib.sha256()
    for key in ('grid_ms', 'time_features', 'event_context', 'reward_features', 'reward_feature_mask'):
        digest.update(event[key].tobytes())
    for rank, (times, values) in sorted(series.items()):
        digest.update(np.asarray([rank, len(times)], dtype='<i8').tobytes())
        digest.update(times.tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def latest_slice(result, event, column):
    if result is None or not result['anchor_mask'][column]:
        return None
    cut = result['t0']
    future = result['pred_cutoff'][:, column]
    anchor = float(result['anchor'][column])
    return dict(t0=cut, future_len=len(future), y_hat=np.diff(future, prepend=anchor).tolist(),
                total_sum_hat=int(future[-1]), future_sum_hat=int(future[-1]-anchor),
                prefix_curve=[dict(time=int(t), ep=int(ep)) for t, ep in zip(event['grid_ms'][cut+1:], future)],
                issued_at_ms=result['issued_at_ms'], anchor_at_ms=result['anchor_at_ms'],
                model_sha256=result['model_sha256'])


class ConditionalForecasts:
    def __init__(self, model_path, *, cache_root='service/cache_data', update_interval=1800):
        self.predictor = ConditionalPredictor(model_path)
        self.cache_dir = Path(cache_root) / 'conditional-curve-http-v1' / self.predictor.model_hash
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = update_interval
        self.last_fetch = {}
        self.locks = {}
        self.locks_guard = threading.Lock()
        self.stop_event = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._periodic, name='conditional-curve-updater', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _event_lock(self, eid):
        with self.locks_guard:
            return self.locks.setdefault(eid, threading.Lock())

    def _periodic(self):
        while not self.stop_event.is_set():
            try:
                raw._cache_metadata(force=True)
                for eid in raw.get_active_events(3):
                    with self._event_lock(eid):
                        self._refresh_cutoffs(eid, active=True)
            except Exception:
                logger.exception('Could not refresh active CN cutoff data')
            self.stop_event.wait(self.interval)

    def _refresh_cutoffs(self, eid, active):
        if time.monotonic() - self.last_fetch.get(eid, float('-inf')) < self.interval:
            return
        root = Path(raw.config.raw_dir) / '3' / str(eid)
        tiers = tier_list_for_event(3, eid)
        if active or any(not (root / f'{rank}.json').is_file() for rank in tiers):
            raw.download_cutoffs(event_list=[eid], server_list=[3], force=active, max_workers=4)
        self.last_fetch[eid] = time.monotonic()

    def _history(self, event, series, last_issue):
        signature = history_signature(event, series)
        path = self.cache_dir / f"{event['event_id']}.json"
        record = {}
        if path.is_file():
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                logger.warning('Rebuilding unreadable forecast history %s', path)
        if not isinstance(record, dict) or record.get('signature') != signature:
            record = dict(signature=signature, through=DAY-1, history={})
        if last_issue <= record['through']:
            return record['history']
        history = record['history']
        for issue in range(record['through']+1, last_issue+1):
            issued = int(event['grid_ms'][issue])
            result = self.predictor.predict_at(event, series, issued)
            if result is None:
                continue
            for col in np.flatnonzero(result['anchor_mask']):
                rank = str(event['ranks'][col])
                history.setdefault(rank, []).append(dict(t0=issue, time=issued, anchor_t0=result['t0'],
                                                         ep=int(result['pred_cutoff'][-1, col])))
        record['through'] = last_issue
        if not save_json(record, path, indent=None):
            logger.warning('Could not persist forecast history %s', path)
        return history

    def get_predictions(self, event_id, server, tier, *, now_ms=None):
        if server != 3 or not is_supported_tier(server, event_id, tier):
            return None
        raw._cache_metadata()
        now = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
        start = raw.get_start_at(server, event_id)
        end = raw.get_end_at(server, event_id)
        if start is None or end is None or now < start + DAY*step_length:
            return None
        with self._event_lock(event_id):
            event = self.predictor.calendar(event_id)
            active = now < end
            self._refresh_cutoffs(event_id, active)
            series = read_observations(raw.config.raw_dir, event, require_complete=False)
            last_issue = min((now-start)//step_length, len(event['grid_ms'])-2)
            entries = self._history(event, series, last_issue).get(str(tier), [])
            history = [entry for entry in entries if entry['time'] <= now]
            # The latest request can see genuine observations between two scheduled issues.
            result = self.predictor.predict_at(event, series, now) if active else None
            latest = latest_slice(result, event, event['ranks'].tolist().index(tier))
            if not history and latest is None:
                return None
            return dict(event_id=event_id, server=server, tier=tier, history=history, latest_slice=latest)

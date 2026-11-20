"""Fit the population mean and functional covariance, counting each event once."""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import svd

from data.joint_dataset import FullFitDataset, JointDataset
from data.joint_sequences import JOINT_RANKS
from model.conditional_curve import (ConditionalCurveModel, DAY, FEATURE_COUNT, KNOTS,
                                     RANK_MODES, SCORE_OFFSET, SHARED_MODES, VERSION,
                                     activity_clock, context_features)


def fit_clock(events):
    # Estimate time of day from complete 24-hour blocks away from the two edges.
    total = np.full((DAY, len(JOINT_RANKS)), 2.)
    for event in events:
        steps = len(event['cutoff'])-1
        blocks = (steps-2*DAY) // DAY
        if blocks < 1:
            continue
        right = DAY * (blocks+1)
        delta = np.diff(event['cutoff'][DAY:right+1], axis=0).reshape(blocks, DAY, -1)
        daily = delta.sum(axis=1, keepdims=True)
        relative = np.divide(delta * DAY, daily, out=np.zeros_like(delta), where=daily > 0)
        slots = event['time_features'][DAY+1:2*DAY+1, 0]
        total[slots] += relative.sum(axis=0) / np.maximum(1, (daily[:, 0] > 0).sum(axis=0))
    total = (np.roll(total, 1, axis=0) + 2*total + np.roll(total, -1, axis=0)) / 4.
    total = np.maximum(total / total.mean(axis=0), .05)
    return total / total.mean(axis=0)


def _modes(matrix, count):
    """Return unit-prior loadings, padded for small or degenerate training sets."""
    modes = np.zeros((count, matrix.shape[1]))
    if len(matrix) >= 2:
        _, singular, vectors = svd(matrix, full_matrices=False)
        n = min(count, len(singular))
        modes[:n] = vectors[:n] * (singular[:n, None] / np.sqrt(len(matrix)-1))
        approximation = (matrix @ vectors[:n].T) @ vectors[:n]
    else:
        approximation = np.zeros_like(matrix)
    return modes, approximation


def fit_curves(events, metadata):
    events = list(events)
    if not events or any(not e['ended'] or e['split'] != 'train' for e in events):
        raise ValueError('Fit requires completed training events')
    if len({int(e['event_id']) for e in events}) != len(events):
        raise ValueError('Duplicate training event')
    if any(len(e['cutoff']) <= DAY+1 for e in events):
        raise ValueError('Training events must extend past the first day')
    if any(e['reward_schedule_sha256'] != metadata['reward_schedule_sha256'] for e in events):
        raise ValueError('Training event and artifact reward schedules differ')

    count, ranks, knots = len(events), len(JOINT_RANKS), len(KNOTS)
    profile = fit_clock(events)
    curves = np.zeros((count, ranks, knots))
    features = np.zeros((count, ranks, FEATURE_COUNT))
    valid = np.zeros((count, ranks), dtype=bool)
    for i, event in enumerate(events):
        scores = event['cutoff']
        clock = activity_clock(profile, event['time_features'][:, 0])
        phase = clock / clock[-1]
        valid[i] = event['target_mask'][0]
        features[i] = context_features(scores[DAY], len(scores)-1, event['event_context'][0],
                                       event['reward_features'], event['reward_feature_mask'])
        y = np.log((scores+SCORE_OFFSET) / (scores[DAY]+SCORE_OFFSET))
        y -= np.log((clock+1.) / (clock[DAY]+1.))
        for r in np.flatnonzero(valid[i]):
            curves[i, r] = np.interp(KNOTS, phase[:, r], y[:, r])

    center = np.zeros((ranks, FEATURE_COUNT))
    mean = np.zeros((ranks, knots))
    slopes = np.zeros((ranks, FEATURE_COUNT, knots))
    scale = np.ones(ranks)
    residual = np.zeros_like(curves)
    seen = np.flatnonzero(valid.any(axis=0))
    if not len(seen):
        raise ValueError('No observed training ranks')
    for r in seen:
        x, y = features[valid[:, r], r], curves[valid[:, r], r]
        center[r], mean[r] = x.mean(axis=0), y.mean(axis=0)
        x, y = x-center[r], y-mean[r]
        slopes[r] = np.linalg.solve(x.T@x + 5.*np.eye(FEATURE_COUNT), x.T@y)
        error = y - x @ slopes[r]
        scale[r] = max(.05, np.sqrt(np.mean(error**2)))
        residual[valid[:, r], r] = np.clip(error/scale[r], -4., 4.)

    flat, fitted = _modes(residual.reshape(count, -1), SHARED_MODES)
    shared = flat.reshape(SHARED_MODES, ranks, knots).transpose(1, 0, 2)
    shared = shared * scale[:, None, None]
    remainder = residual - fitted.reshape(residual.shape)
    private = np.zeros((ranks, RANK_MODES, knots))
    for r in seen:
        modes, _ = _modes(remainder[valid[:, r], r], RANK_MODES)
        private[r] = modes * scale[r]
    noise = np.maximum(.045, scale*.5)
    # A split predating a tier can still serve it using the closest fitted rank prior.
    for r in np.flatnonzero(~valid.any(axis=0)):
        source = seen[np.argmin(abs(np.log(np.array(JOINT_RANKS)[seen] / JOINT_RANKS[r])))]
        for array in (center, mean, slopes, shared, private, noise):
            array[r] = array[source]
        profile[:, r] = profile[:, source]
    metadata = dict(metadata, version=VERSION, ranks=list(JOINT_RANKS),
                    train_events=sorted(int(e['event_id']) for e in events),
                    last_training_end_ms=max(int(e['end_at']) for e in events),
                    observed_events_per_rank=valid.sum(axis=0).tolist())
    return ConditionalCurveModel(metadata, profile=profile, center=center, mean=mean, slopes=slopes,
                                 shared=shared, private=private, noise=noise)


def train_conditional_curve(data_path, out_dir, *, fit_scope='train'):
    if fit_scope not in ('train', 'all'):
        raise ValueError('fit_scope must be train or all')
    dataset = FullFitDataset(data_path) if fit_scope == 'all' else JointDataset(data_path, 'train')
    with h5py.File(data_path, 'r') as handle:
        attrs = handle['meta'].attrs
        metadata = dict(reward_table=json.loads(attrs['reward_table']),
                        reward_schedule_sha256=str(attrs['reward_schedule_sha256']),
                        data_snapshot_sha256=str(attrs['data_snapshot_sha256']), fit_scope=fit_scope)
    model = fit_curves((dataset.event(i) for i in sorted(dataset.event_ids)), metadata)
    path = Path(out_dir) / 'conditional-curve.npz'
    model.save(path)
    print(f'Saved {path} ({len(dataset.event_ids)} training events, scope={fit_scope})', flush=True)
    return path

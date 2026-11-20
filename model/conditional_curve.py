"""Conditional functional PCA for a joint surface of cumulative rank scores."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from data.constants import NUM_EVENT_TYPES
from data.joint_sequences import JOINT_RANKS
from data.rewards import RewardTable
from model.v2.projection import project_future


VERSION = 'conditional-curve-v1'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'saved_models' / 'conditional-curve.npz'
DAY = 48  # half-hour steps
SCORE_OFFSET = 1000.
KNOTS = np.r_[.01, np.arange(1, 39) / 40., .965, .975, .985, .9925, .9975, 1.]
SHARED_MODES = 4
RANK_MODES = 2
FEATURE_COUNT = 3 + NUM_EVENT_TYPES + 4


def activity_clock(profile, slots):
    """Elapsed typical activity per rank. The initial grid point has zero mass."""
    mass = profile[np.asarray(slots, dtype=int)[1:]]
    return np.vstack([np.zeros(mass.shape[1]), np.cumsum(mass, axis=0)])


def context_features(first_day, steps, event_type, reward, reward_mask):
    count = len(first_day)
    types = np.broadcast_to(np.arange(NUM_EVENT_TYPES) == int(event_type), (count, NUM_EVENT_TYPES))
    return np.column_stack([np.log1p(first_day / SCORE_OFFSET), first_day == 0,
                            np.full(count, np.log(steps / 400.)), types,
                            np.where(reward_mask[:, :4], reward[:, :4], 0.)])


def curve_values(values, phase):
    """Linear interpolation along the last (activity-progress) axis."""
    phase = np.clip(phase, KNOTS[0], KNOTS[-1])
    left = np.searchsorted(KNOTS, phase, side='right').clip(1, len(KNOTS)-1) - 1
    weight = (phase - KNOTS[left]) / (KNOTS[left+1] - KNOTS[left])
    return values[..., left] * (1.-weight) + values[..., left+1] * weight


class ConditionalCurveModel:
    """The shared activity factors and each rank's own factors have unit priors."""

    def __init__(self, metadata, *, profile, center, mean, slopes, shared, private, noise):
        ranks, knots = len(JOINT_RANKS), len(KNOTS)
        if metadata.get('version') != VERSION or metadata.get('ranks') != list(JOINT_RANKS):
            raise ValueError('Incompatible conditional curve artifact')
        self.metadata = metadata
        self.reward_table = RewardTable(document=metadata['reward_table'])
        if self.reward_table.sha256 != metadata['reward_schedule_sha256']:
            raise ValueError('Inconsistent conditional curve reward schedule')
        shapes = {'profile': (DAY, ranks), 'center': (ranks, FEATURE_COUNT), 'mean': (ranks, knots),
                  'slopes': (ranks, FEATURE_COUNT, knots), 'shared': (ranks, SHARED_MODES, knots),
                  'private': (ranks, RANK_MODES, knots), 'noise': (ranks,)}
        arrays = dict(profile=profile, center=center, mean=mean, slopes=slopes,
                      shared=shared, private=private, noise=noise)
        for name, values in arrays.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != shapes[name] or not np.isfinite(values).all():
                raise ValueError(f'Invalid curve model array: {name}')
            setattr(self, name, values)
        if (self.profile <= 0).any() or (self.noise <= 0).any():
            raise ValueError('Clock rates and noise scales must be positive')

    def save(self, path):
        """Publish a complete artifact without replacing an existing model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix='.curve-', suffix='.npz') as tmp:
            np.savez_compressed(tmp, metadata=json.dumps(self.metadata), profile=self.profile,
                                center=self.center, mean=self.mean, slopes=self.slopes,
                                shared=self.shared, private=self.private, noise=self.noise)
            tmp.flush()
            os.link(tmp.name, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(json.loads(str(archive['metadata'])),
                       **{key: archive[key] for key in archive.files if key != 'metadata'})

    def predict(self, inputs):
        """Consume one unpadded joint_model_input, returning raw and publishable curves."""
        history = np.asarray(inputs['cutoff_hist'], dtype=np.float64)
        available = np.asarray(inputs['history_available_mask'], dtype=bool)
        t0 = len(history)-1
        if t0 < DAY:
            raise ValueError('Conditional curve prediction needs at least 24 hours of history')
        if not np.array_equal(inputs['ranks'], JOINT_RANKS):
            raise ValueError('Conditional curve rank axis mismatch')
        anchor_mask = inputs['anchor_mask']
        slots = np.r_[inputs['history_time_features'][:, 0], inputs['future_time_features'][:, 0]] - 1
        clock = activity_clock(self.profile, slots)
        phase = clock / clock[-1]
        first_day = np.where(available[DAY], history[DAY], 0.)
        for r in np.flatnonzero(anchor_mask & ~available[DAY]):
            known = np.flatnonzero(available[:, r])
            first_day[r] = np.interp(clock[DAY, r], clock[known, r], history[known, r])
        features = context_features(first_day, len(clock)-1, inputs['event_context'][0],
                                    inputs['reward_features'], inputs['reward_feature_mask'])
        prior = self.mean + np.einsum('rf,rfk->rk', features-self.center, self.slopes)
        # Use separated observations: interpolated half-hours are not independent evidence.
        times = np.unique(np.r_[np.arange(12, t0, 24), max(1, t0-12), t0])
        times = times[times != DAY]  # the normalizing point has a known zero residual
        width = SHARED_MODES + len(JOINT_RANKS) * RANK_MODES
        rows, residuals = [], []
        features_by_rank = np.concatenate([self.shared, self.private], axis=1)
        for r in np.flatnonzero(anchor_mask):
            observed = times[available[times, r]]
            if not len(observed):
                continue
            p = phase[observed, r]
            y = np.log((history[observed, r]+SCORE_OFFSET) / (first_day[r]+SCORE_OFFSET))
            y -= np.log((clock[observed, r]+1.) / (clock[DAY, r]+1.))
            basis = curve_values(features_by_rank[r], p).T
            design = np.zeros((len(p), width))
            design[:, :SHARED_MODES] = basis[:, :SHARED_MODES]
            start = SHARED_MODES + r * RANK_MODES
            design[:, start:start+RANK_MODES] = basis[:, SHARED_MODES:]
            rows.append(design / self.noise[r])
            residuals.append((y - curve_values(prior[r], p)) / self.noise[r])
        factors = np.zeros(width)
        if rows:
            design, target = np.vstack(rows), np.concatenate(residuals)
            precision = np.eye(width)
            factors = np.linalg.solve(precision + design.T @ design, design.T @ target)
            # Bound the influence of a sparse or irregular rank on the shared factors.
            weights = np.minimum(1., 2.5 / np.maximum(abs(target-design@factors), 1e-9))
            weighted = design.T * weights
            factors = np.linalg.solve(precision + weighted @ design, weighted @ target)

        fitted = prior + np.einsum('rmk,m->rk', self.shared, factors[:SHARED_MODES])
        local_factors = factors[SHARED_MODES:].reshape(-1, RANK_MODES)
        fitted += np.einsum('rmk,rm->rk', self.private, local_factors)
        raw = np.zeros((len(clock)-t0-1, len(JOINT_RANKS)))
        for r in np.flatnonzero(anchor_mask):
            levels = curve_values(fitted[r], phase[t0:, r]) + np.log1p(clock[t0:, r])
            ratio = np.exp(np.clip(levels[1:] - levels[0], -20., 20.))
            raw[:, r] = np.maximum(0., (history[-1, r]+SCORE_OFFSET) * ratio - SCORE_OFFSET)
        projected = project_future(raw[None], history[-1:], anchor_mask[None])[0]
        return raw, projected

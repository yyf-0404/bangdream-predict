"""Joint-cut weighting: sum rank losses, then mean samples; never renormalize a batch."""
from dataclasses import dataclass
import math
import warnings

import numpy as np

from .constants import model_tier_index
from .joint_sequences import JOINT_RANKS, validate_joint_event
from .sample_weights import compute_sample_weight, compute_tier_weight

JOINT_WEIGHT_VERSION = "joint-rank-masked-rule-balanced-v1"


@dataclass
class JointWeightPlan:
    index: np.ndarray                 # [N,2]: event ID, t0
    sample_scale: np.ndarray          # [N]: progress * group factor (not a network input)
    tier_weights: np.ndarray          # [R], multiply by each event's target rank mask
    report: dict


def build_joint_weight_plan(events, *, t_min=48, balance_rules=True,
                            weight_mode="sqrt_floor", weight_floor=0.2,
                            tier_center=9, tier_alpha=0.5, tier_floor=0.1,
                            tier_1500_weight=1.0):
    if type(t_min) is not int or t_min < 0:
        raise ValueError("t_min must be a nonnegative integer")
    tier_weights = np.array([compute_tier_weight(
        model_tier_index(r), center=tier_center, alpha=tier_alpha, floor=tier_floor,
        tier=r, tier_1500_weight=tier_1500_weight,
    ) for r in JOINT_RANKS], dtype=np.float64)
    indices, progress, rules = [], [], []
    base_sums, counts, units = np.zeros(2), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    seen, splits = set(), set()
    for event in events:
        validate_joint_event(event)
        if not event['ended'] or event['split'] not in ('train', 'val'):
            raise ValueError("Weights require completed training/validation events")
        identity = (event['server'], event['event_id'])
        if identity in seen:
            raise ValueError("Duplicate joint event")
        seen.add(identity)
        splits.add(event['split'])
        if len(splits) > 1:
            raise ValueError("Compute joint weights separately for train and validation")
        if event['split'] == 'val' and balance_rules:
            raise ValueError("Validation weights must not force old/new balance")
        T = len(event['cutoff']) - 1
        if T <= t_min:
            raise ValueError(f"Event {event['event_id']} has no cuts at t_min={t_min}")
        rank_mask = event['target_mask'][0]
        rank_mass = float(np.sum(tier_weights[rank_mask]))
        if rank_mass <= 0:
            raise ValueError("Event has no positive supervised rank weight")
        group = event['balance_group']
        event_progress = [compute_sample_weight(T-t0, T, weight_mode, weight_floor)
                          for t0 in range(t_min, T)]
        indices.extend((event['event_id'], t0) for t0 in range(t_min, T))
        progress.extend(event_progress)
        rules.extend([group] * len(event_progress))
        counts[group] += len(event_progress)
        units[group] += len(event_progress) * int(rank_mask.sum())
        base_sums[group] += rank_mass * math.fsum(event_progress)
    N = len(indices)
    if not N or not np.isfinite(base_sums).all() or ((counts > 0) & (base_sums <= 0)).any():
        raise ValueError("Joint plan has no positive effective supervision")
    balanced = bool(balance_rules and (counts > 0).all())
    if balance_rules and not balanced:
        warnings.warn("Joint training has only one balance group; 1:1 is unavailable", RuntimeWarning)
    factors = N / (2 * base_sums) if balanced else np.full(2, N / base_sums.sum())
    sample_scale = np.asarray(progress) * factors[np.asarray(rules)]
    return JointWeightPlan(
        index=np.asarray(indices, dtype=np.int64), sample_scale=sample_scale,
        tier_weights=tier_weights,
        report=dict(version=JOINT_WEIGHT_VERSION, joint_samples=N,
                    rule_sample_counts=counts.tolist(), rank_supervision_units=units.tolist(),
                    rule_base_weight_sums=base_sums.tolist(), rule_factors=factors.tolist(),
                    rule_final_weight_sums=(base_sums * factors).tolist(),
                    rule_balance_requested=bool(balance_rules), rule_balance_applied=balanced,
                    t_min=t_min, weight_mode=weight_mode, weight_floor=weight_floor,
                    tier_center=tier_center, tier_alpha=tier_alpha, tier_floor=tier_floor,
                    tier_1500_weight=tier_1500_weight),
    )


def weight_plan_digest(dataset, sample_scale=None):
    """Identity of row order and exact effective weight arrays, not just metadata."""
    import hashlib
    result = hashlib.sha256()
    scale = dataset.sample_scale if sample_scale is None else sample_scale
    for value in (dataset.index, scale, dataset.tier_weights):
        array = np.ascontiguousarray(value)
        result.update(str((array.shape, array.dtype.str)).encode())
        result.update(array.tobytes())
    return result.hexdigest()

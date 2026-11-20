"""Sample-weight planning without TensorFlow or training-environment imports."""

from dataclasses import dataclass
import math
import warnings

from .constants import MISSING_TIER_IDX, oldEventTierListOfServer, uses_new_event_tiers


SAMPLE_WEIGHT_VERSION = "rule-balanced-full-data-v1"


def compute_sample_weight(K, T, mode="sqrt_floor", floor=0.2):
    if T <= 0 or not 0 <= K <= T:
        raise ValueError("Expected 0 <= remaining steps <= total steps, with total > 0")
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("Weight floor must be finite and nonnegative")
    ratio = K / T
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return math.sqrt(ratio)
    if mode == "linear":
        return ratio
    if mode == "old":
        return ratio ** 1.5
    if mode == "sqrt_floor":
        return max(math.sqrt(ratio), floor)
    if mode == "linear_floor":
        return max(ratio, floor)
    raise ValueError(f"Unknown weight mode: {mode}")


def compute_tier_weight(tier_idx, center=9, alpha=0.15, floor=0.3,
                        *, tier=None, tier_1500_weight=1.0):
    """Preserve existing tier weights, overriding real rank 1500 explicitly.

    A missing embedding index is not a ranking distance. It must not cause
    rank 1500 to receive the old far-tail floor weight.
    """
    if any(not math.isfinite(v) or v < 0 for v in (alpha, floor, tier_1500_weight)):
        raise ValueError("Tier weight parameters must be finite and nonnegative")
    if tier is not None and int(tier) == 1500:
        return float(tier_1500_weight)
    if int(tier_idx) == MISSING_TIER_IDX:
        raise ValueError("Missing tier index requires an explicit supported rank")
    return max(math.exp(-alpha * abs(int(tier_idx) - int(center))), floor)


def sequence_rule_and_tier(seq):
    context = seq['context_cat']
    if len(context) < 5:
        raise ValueError("Sample weighting requires context_cat with server and event ID")
    tier_idx, server, event = map(int, context[2:5])
    rule = int(uses_new_event_tiers(server, event))
    if 'tier' in seq:
        tier = int(seq['tier'])
    elif rule and tier_idx == MISSING_TIER_IDX:
        # Compatibility with earlier HDF5 files: only new CN rank 1500 used
        # this missing index. Never reinterpret an old-event missing index.
        tier = 1500
    else:
        tiers = oldEventTierListOfServer.get(server, [])
        if not 0 <= tier_idx < len(tiers):
            raise ValueError("Cannot recover a real rank from this sequence")
        tier = tiers[tier_idx]
    if tier == 1500 and not rule:
        raise ValueError("Old-rule activities cannot contain rank 1500 samples")
    return rule, tier


@dataclass(frozen=True)
class SampleWeightPlan:
    rules: tuple
    tiers: tuple
    tier_weights: tuple
    counts: tuple
    base_sums: tuple
    factors: tuple
    balanced: bool


def build_sample_weight_plan(sequences, t0_lists, *, balance_rules=True,
                             weight_mode="sqrt_floor", weight_floor=0.2,
                             tier_center=9, tier_alpha=0.15, tier_floor=0.3,
                             tier_1500_weight=1.0):
    """Equalize total *effective* old/new weight, preserving every cut.

    Base weights include both progress and tier weights. If N cuts and two
    rule groups are present, group g receives factor N/(2 * base_sum[g]).
    Thus each group's final weight sum is N/2 and the global mean is 1.
    Unbalanced/one-group splits retain relative base weights, mean-normalized.
    Call independently for train and validation; never use held-out totals.
    """
    if len(sequences) != len(t0_lists):
        raise ValueError("Each sequence must have exactly one cut-point list")
    rules, tiers, tier_weights = [], [], []
    counts, sums = [0, 0], [0.0, 0.0]
    for seq, cuts in zip(sequences, t0_lists):
        rule, tier = sequence_rule_and_tier(seq)
        tier_weight = compute_tier_weight(
            int(seq['context_cat'][2]), center=tier_center, alpha=tier_alpha,
            floor=tier_floor, tier=tier, tier_1500_weight=tier_1500_weight,
        )
        T = len(seq['y'])
        for cut in cuts:
            if not 0 <= cut < T or int(cut) != cut:
                raise ValueError("Cut points must be integer indices inside the sequence")
        mass = math.fsum(
            compute_sample_weight(T - cut, T, weight_mode, weight_floor) * tier_weight
            for cut in cuts
        )
        counts[rule] += len(cuts)
        sums[rule] += mass
        rules.append(rule)
        tiers.append(tier)
        tier_weights.append(tier_weight)

    total = sum(counts)
    if not total:
        return SampleWeightPlan(tuple(rules), tuple(tiers), tuple(tier_weights),
                                tuple(counts), tuple(sums), (1.0, 1.0), False)
    if any(count and mass <= 0 for count, mass in zip(counts, sums)):
        raise ValueError("A rule group has samples but no positive base weight")
    balanced = bool(balance_rules and all(counts))
    if balance_rules and not balanced:
        warnings.warn(
            "训练切片仅包含一种奖励规则，无法达到新旧 1:1；未补造或移入验证样本。"
            "请检查训练/验证活动清单；旧的 split_threshold=290 划分没有新规则。",
            RuntimeWarning, stacklevel=2,
        )
    if balanced:
        factors = tuple(total / (2.0 * mass) for mass in sums)
    else:
        factors = (total / math.fsum(sums),) * 2
    return SampleWeightPlan(tuple(rules), tuple(tiers), tuple(tier_weights),
                            tuple(counts), tuple(sums), factors, balanced)

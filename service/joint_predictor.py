"""Compatibility entry point for the published joint conditional predictor."""
import numpy as np

from data.constants import is_supported_tier
from service.conditional_predictor import ConditionalPredictor


class JointPredictor(ConditionalPredictor):
    def predict_current_curve(self, event, t0, rank):
        result = self.predict_event(event, t0, requested_rank=rank)
        r = list(map(int, event['ranks'])).index(rank)
        return rank_curve_result(event, result, r)


def latest_covered_cut(event, ranks=None):
    if ranks is None:
        ranks = [r for r in (500, 1000, 1500, 2000)
                 if is_supported_tier(event['server'], event['event_id'], r)]
    if not len(ranks) or not all(is_supported_tier(event['server'], event['event_id'], r) for r in ranks):
        raise ValueError('Requested a missing or unsupported event tier')
    indices = [list(event['ranks']).index(r) for r in ranks]
    covered = event['available_mask'][:, indices].all(axis=1)
    cuts = np.flatnonzero(covered & (event['grid_ms'] <= event['snapshot_at_ms'])
                          & (event['grid_ms'] < event['end_at']))
    if not len(cuts):
        raise ValueError('No jointly observed anchor at the issue time')
    return int(cuts[-1])


def rank_curve_result(event, result, rank_index):
    """Adapt cumulative V2 output to the existing per-rank API meanings."""
    r, t0 = rank_index, int(result['t0'])
    if not result['anchor_mask'][r]:
        raise ValueError('Cannot publish a rank without a reliable anchor')
    future = np.asarray(result['pred_cutoff'][:, r], np.float64)
    anchor = float(result['anchor'][r])
    increments = np.diff(np.concatenate([[anchor], future]))
    # Historical holes remain null, not zero. No need to fabricate service observations.
    historical = [float(v) if valid else None for v, valid in
                  zip(event['cutoff'][1:t0+1, r], event['available_mask'][1:t0+1, r])]
    return dict(t0=t0, future_len=len(future), y_hat=increments.tolist(),
                total_sum_hat=float(future[-1]), future_sum_hat=float(future[-1]-anchor),
                prefix_curve=historical+future.tolist(), model_sha256=result['model_sha256'])

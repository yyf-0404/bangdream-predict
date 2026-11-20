"""Deployment constraints in EP space; intentionally independent of TensorFlow.

Unlike historical data projection, this never locks a zero rank's future to zero.
Only the fixed set of publishable, anchored ranks can influence the projection.
"""
import numpy as np

from data.rank_order import antitonic_regression

PROJECTION_VERSION = 'anchored-future-pava-v1'


def project_future(raw_cutoff, anchor, anchor_mask, future_time_mask=None):
    raw = np.asarray(raw_cutoff, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    valid = np.asarray(anchor_mask, dtype=bool)
    if raw.ndim != 3 or anchor.shape != (raw.shape[0], raw.shape[2]) or valid.shape != anchor.shape:
        raise ValueError('Expected [batch, future, rank] and [batch, rank] anchors/masks')
    times = (np.ones(raw.shape[:2], dtype=bool) if future_time_mask is None
             else np.asarray(future_time_mask, dtype=bool))
    if times.shape != raw.shape[:2] or (np.diff(times.astype(int), axis=1) > 0).any():
        raise ValueError('Future time mask must be a right-padded prefix')
    if not np.isfinite(anchor[valid]).all() or (anchor[valid] < 0).any():
        raise ValueError('Anchors must be finite and nonnegative')
    result = np.zeros_like(raw)
    for b in range(len(raw)):
        ranks = np.flatnonzero(valid[b])
        if not len(ranks):
            raise ValueError('No publishable anchor')
        previous = anchor[b, ranks].copy()
        if (np.diff(previous) > 0).any():
            raise ValueError('Anchors must already be rank ordered; they cannot be moved')
        steps = np.flatnonzero(times[b])
        rows = raw[b][np.ix_(steps, ranks)]
        if not np.isfinite(rows).all() or (rows < 0).any():
            raise ValueError('Publishable predictions must be finite and nonnegative')
        for t in np.flatnonzero((np.diff(rows, axis=1) > 0).any(axis=1)):
            rows[t] = antitonic_regression(rows[t])
        result[b][np.ix_(steps, ranks)] = np.maximum.accumulate(np.maximum(rows, previous), axis=0)
    return result


def surface_diagnostics(cutoff, anchor, rank_mask, time_mask):
    """Counts/magnitudes on adjacent valid ranks, skipping unknown columns."""
    cutoff = np.asarray(cutoff, dtype=np.float64)
    result = dict(rank_crossings=0, max_rank_violation=0., time_decreases=0,
                  max_time_violation=0.)
    for b in range(len(cutoff)):
        ranks = np.flatnonzero(rank_mask[b])
        rows = cutoff[b][np.asarray(time_mask[b], bool)][:, ranks]
        if not len(ranks) or not len(rows):
            continue
        rank_diff = np.diff(rows, axis=1)
        time_diff = -np.diff(np.concatenate([np.asarray(anchor[b])[None, ranks], rows]), axis=0)
        for prefix, values, count_key in (
            ('rank', rank_diff, 'rank_crossings'), ('time', time_diff, 'time_decreases')
        ):
            result[count_key] += int((values > 1e-6).sum())
            result[f'max_{prefix}_violation'] = max(
                result[f'max_{prefix}_violation'], float(np.maximum(values, 0).max(initial=0)))
    return result


def predict_joint_batch(model, inputs, *, return_components=False):
    """Same postprocessing for validation and offline/service prediction."""
    output = model(inputs, training=False)
    raw = output['raw_cutoff']
    raw = raw.numpy() if hasattr(raw, 'numpy') else np.asarray(raw)
    if raw.shape != (len(inputs['anchor']), inputs['future_time_mask'].shape[1], inputs['anchor'].shape[1]):
        raise ValueError('V2 model output shape does not match the joint input')
    projected = project_future(raw, inputs['anchor'], inputs['anchor_mask'], inputs['future_time_mask'])
    if return_components:
        components = {key: np.asarray(output[key]) for key in
                      ('base_remaining', 'remaining_total', 'direct_remaining',
                       'moe_weights', 'moe_shape_js') if key in output}
        return raw, projected, components
    return raw, projected

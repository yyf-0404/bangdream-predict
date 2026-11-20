"""A single input constructor shared by future V2 training and inference."""
import numpy as np

from .joint_sequences import JOINT_RANKS

# Identity, reward versions, balance groups, provenance and future target masks
# are deliberately excluded from this whitelist.
MODEL_INPUT_KEYS = frozenset({
    'cutoff_hist', 'history_available_mask', 'ranks', 'rank_position',
    'reward_features', 'reward_feature_mask', 'event_context',
    'history_time_features', 'future_time_features', 'history_progress', 'future_progress',
    'anchor', 'anchor_mask', 'history_time_mask', 'future_time_mask',
    'history_len', 'future_len',
})


def joint_model_input(event, t0, *, requested_rank=None):
    T = len(event['cutoff']) - 1
    if type(t0) not in (int, np.int64, np.int32) or not 0 <= t0 < T:
        raise ValueError("Joint t0 must leave at least one future step")
    available = event['available_mask'][:t0+1].copy()
    anchor_mask = available[-1].copy()
    if not anchor_mask.any():
        raise ValueError("No available anchor at this cut")
    if requested_rank is not None:
        if requested_rank not in JOINT_RANKS or not anchor_mask[JOINT_RANKS.index(requested_rank)]:
            raise ValueError("Requested rank lacks exact/right coverage at this cut")
    history = np.where(available, event['cutoff'][:t0+1], 0).astype(np.float32)
    return dict(
        cutoff_hist=history, history_available_mask=available,
        ranks=event['ranks'].copy(), rank_position=np.log1p(event['ranks']).astype(np.float32),
        reward_features=event['reward_features'].copy(), reward_feature_mask=event['reward_feature_mask'].copy(),
        event_context=event['event_context'].copy(),
        # Embedding index 0 is reserved for time padding, unlike true zero EP.
        history_time_features=event['time_features'][:t0+1] + 1,
        future_time_features=event['time_features'][t0+1:] + 1,
        history_progress=np.arange(t0+1, dtype=np.float32) / T,
        future_progress=np.arange(t0+1, T+1, dtype=np.float32) / T,
        anchor=history[-1].copy(), anchor_mask=anchor_mask,
        history_time_mask=np.ones(t0+1, dtype=bool), future_time_mask=np.ones(T-t0, dtype=bool),
        history_len=np.int32(t0+1), future_len=np.int32(T-t0),
    )


def joint_training_slice(event, t0, sample_scale, tier_weights):
    if not event['ended']:
        raise ValueError("Active events cannot supply training labels")
    x = joint_model_input(event, t0)
    mask = event['target_mask'][t0+1:].copy()
    if not np.all(mask == mask[0]):
        raise ValueError("Partial future supervision needs a separate weighting audit")
    target = np.where(mask, event['cutoff'][t0+1:], 0).astype(np.float32)
    rank_weights = sample_scale * np.asarray(tier_weights, dtype=np.float64) * mask[0]
    return x, {'cutoff_future': target, 'target_mask': mask}, rank_weights


def collate_joint_inputs(inputs):
    """Shared training/inference padding, without constructing fake future labels."""
    if not inputs:
        raise ValueError("Cannot collate an empty batch")
    if any(set(x) != MODEL_INPUT_KEYS for x in inputs):
        raise ValueError("Joint model input keys do not match the schema")
    history_keys = {'cutoff_hist', 'history_available_mask', 'history_time_features',
                    'history_progress', 'history_time_mask'}
    future_keys = {'future_time_features', 'future_progress', 'future_time_mask'}
    max_history = max(int(x['history_len']) for x in inputs)
    max_future = max(int(x['future_len']) for x in inputs)

    def stack_pad(arrays, length=None):
        if length is None:
            return np.stack(arrays)
        result = np.zeros((len(arrays), length, *arrays[0].shape[1:]), dtype=arrays[0].dtype)
        for row, array in enumerate(arrays):
            result[row, :len(array)] = array
        return result

    x = {}
    for key in MODEL_INPUT_KEYS:
        length = max_history if key in history_keys else max_future if key in future_keys else None
        x[key] = stack_pad([sample[key] for sample in inputs], length)
    return x


def collate_joint_slices(slices):
    """Pad only this batch, retaining every real sample (including the tail)."""
    x = collate_joint_inputs([sample[0] for sample in slices])
    max_future = x['future_time_mask'].shape[1]
    y = {}
    for key in ('cutoff_future', 'target_mask'):
        arrays = [sample[1][key] for sample in slices]
        y[key] = np.zeros((len(arrays), max_future, *arrays[0].shape[1:]), dtype=arrays[0].dtype)
        for row, array in enumerate(arrays):
            y[key][row, :len(array)] = array
    return x, y, np.stack([sample[2] for sample in slices])

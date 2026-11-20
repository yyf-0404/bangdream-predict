"""One cumulative time/rank surface per event. NumPy/SciPy only, no TF."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import raw
from .constants import (eventTypeMap, newEventTierListOfServer, step_length, tier_list_for_event,
                        uses_new_event_tiers)
from .cutoff_quality import cutoff_digest, load_cutoff_snapshot
from .rewards import RewardTable
from .time_features import time_features

JOINT_SCHEMA_VERSION = "rank-joint-cumulative-v1"
JOINT_RANKS = tuple(newEventTierListOfServer[3])


def event_calendar(event_id, server, metadata, characters, reward_table):
    """The published event schedule, rank rewards and covariates available at issue time."""
    if server != 3:
        raise ValueError("Joint forecasts support CN only")
    start = raw.get_start_at(server, event_id, metadata=metadata)
    end = raw.get_end_at(server, event_id, metadata=metadata)
    if start is None or end is None or end <= start or (end-start) % step_length:
        raise ValueError(f"Missing or invalid event times: {event_id}")
    info = metadata[str(event_id)]
    ranks = np.asarray(JOINT_RANKS, dtype=np.int32)
    grid = np.arange(start, end+1, step_length, dtype=np.int64)
    return dict(event_id=int(event_id), server=server, start_at=start, end_at=end, grid_ms=grid,
                time_features=time_features(grid, server), ranks=ranks,
                event_context=np.asarray([eventTypeMap[info['eventType']], raw.get_band(info, characters), server],
                                         dtype=np.int32),
                balance_group=int(uses_new_event_tiers(server, event_id)),
                **reward_table.encode(server, event_id, start, ranks))


def build_joint_event(event_id, *, server=3, raw_dir=None, metadata=None, characters=None,
                      reward_table=None, event_split=None, current_time_ms=None,
                      require_complete=True, recorded_only=False):
    """In recorded-only replay every included tier needs a timestamped observation by issue time."""
    metadata = raw.get_event_data() if metadata is None else metadata
    characters = raw.get_character_data() if characters is None else characters
    calendar = event_calendar(event_id, server, metadata, characters, reward_table or RewardTable())
    start, end = calendar['start_at'], calendar['end_at']
    now = int(datetime.now(timezone.utc).timestamp() * 1000) if current_time_ms is None else int(current_time_ms)
    if now < start:
        raise ValueError("Cannot construct a joint event before its start")
    if recorded_only and now >= end:
        raise ValueError("An as-of forecast must be issued before the end")
    root = Path(raw.config.raw_dir if raw_dir is None else raw_dir)
    supported = tier_list_for_event(server, event_id)
    samples, hashes = {}, {}
    for rank in supported:
        path = root / str(server) / str(event_id) / f"{rank}.json"
        if not path.exists():
            if require_complete:
                raise FileNotFoundError(f"Missing supported cutoff rank: {path}")
            continue
        points = load_cutoff_snapshot(root, server, event_id, rank)
        if not recorded_only:
            hashes[str(rank)] = cutoff_digest(points)
        # An active snapshot cannot use observations later than that snapshot.
        if now < end and points:
            points = [p for p in points if p['time'] <= now]
            if not points:
                continue  # not a structural zero
        if recorded_only:
            if not points:
                continue  # an archived empty list has no observation timestamp
            hashes[str(rank)] = cutoff_digest(points)
        samples[rank] = points
    surface = raw.build_cumulative_grid(samples, start, end, now)
    shape = (len(surface['grid_ms']), len(JOINT_RANKS))
    result = {
        'cutoff': np.zeros(shape, dtype=np.float64),
        'available_mask': np.zeros(shape, dtype=bool),
        'target_mask': np.zeros(shape, dtype=bool),
        'provenance': np.zeros(shape, dtype=np.uint8),
        'rank_repaired_mask': np.zeros(shape, dtype=bool),
        'latest_observation_ms': np.full(len(JOINT_RANKS), -1, dtype=np.int64),
    }
    indices = [JOINT_RANKS.index(int(r)) for r in surface['ranks']]
    for key in ('cutoff', 'available_mask', 'provenance', 'rank_repaired_mask'):
        result[key][:, indices] = surface[key]
    result['latest_observation_ms'][indices] = surface['latest_observation_ms']
    if now >= end:
        result['target_mask'][:] = result['available_mask']
    result.update(
        **calendar, snapshot_at_ms=now, ended=now >= end,
        split=event_split.split_for(server, event_id) if event_split is not None else 'inference',
        source_hashes=hashes,
    )
    validate_joint_event(result)
    return result


def validate_joint_event(event):
    ranks = event['ranks']
    if not np.array_equal(ranks, JOINT_RANKS):
        raise ValueError("Joint rank axis mismatch")
    cutoff = event['cutoff']
    shape = cutoff.shape
    if len(shape) != 2 or shape[1] != len(JOINT_RANKS) or shape[0] < 2:
        raise ValueError("Invalid joint cumulative shape")
    if not np.isfinite(cutoff).all() or (cutoff < 0).any():
        raise ValueError("Joint EP must be finite and nonnegative")
    expected_grid = event['start_at'] + np.arange(shape[0], dtype=np.int64) * step_length
    if not np.array_equal(event['grid_ms'], expected_grid) or expected_grid[-1] != event['end_at']:
        raise ValueError("Joint timestamp grid mismatch")
    if bool(event['ended']) != (event['snapshot_at_ms'] >= event['end_at']):
        raise ValueError("Joint snapshot/ended flag mismatch")
    for key in ('available_mask', 'target_mask'):
        mask = event[key]
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError(f"Invalid {key}")
    if event['time_features'].shape != (shape[0], 3):
        raise ValueError("Time grid mismatch")
    time_values = event['time_features']
    if (not np.issubdtype(time_values.dtype, np.integer) or (time_values < 0).any()
            or (time_values >= [48,7,4]).any()):
        raise ValueError("Time feature categories outside schema")
    if event['event_context'].shape != (3,) or event['event_context'][2] != event['server']:
        raise ValueError("Joint event context mismatch")
    if event['reward_features'].shape != (len(ranks), 8) or event['reward_feature_mask'].shape != (len(ranks), 8):
        raise ValueError("Reward feature shape mismatch")
    if not np.isfinite(event['reward_features']).all():
        raise ValueError("Nonfinite reward features")
    if event['reward_feature_mask'].dtype != np.bool_:
        raise ValueError("Invalid reward field mask")
    if (event['provenance'].shape != shape or event['rank_repaired_mask'].shape != shape
            or event['latest_observation_ms'].shape != (len(ranks),)):
        raise ValueError("Joint provenance/coverage shape mismatch")
    supported = np.isin(ranks, tier_list_for_event(event['server'], event['event_id']))
    if (event['available_mask'][:, ~supported].any() or event['target_mask'][:, ~supported].any()
            or (cutoff[:, ~supported] != 0).any()):
        raise ValueError("Unsupported ranks must remain unknown, not zero labels")
    if event['balance_group'] != int(uses_new_event_tiers(event['server'], event['event_id'])):
        raise ValueError("Incorrect balance group")
    if (np.diff(cutoff, axis=0) < -1e-7).any():
        raise ValueError("Cumulative time order violation")
    for values, mask in zip(cutoff, event['available_mask']):
        if (np.diff(values[mask]) > 1e-7).any():
            raise ValueError("Cumulative rank order violation")
    if event['ended']:
        if not np.all(event['target_mask'] == event['target_mask'][0]):
            raise ValueError("Partial future labels need a new joint-weight audit")
        if not np.array_equal(event['target_mask'], event['available_mask']):
            raise ValueError("Finished supervision/availability mismatch")
    elif event['target_mask'].any():
        raise ValueError("Active event must not provide future training labels")


def joint_split_report(events, event_split):
    seen = set()
    for event in events:
        validate_joint_event(event)
        key = (event['server'], event['event_id'])
        if key in seen:
            raise ValueError("Duplicate joint event / event leakage")
        seen.add(key)
        if event['split'] != event_split.split_for(*key) or not event['ended']:
            raise ValueError("Joint training requires the fixed split and finished events")
        expected_type = eventTypeMap[event_split.event_types[event['event_id']]]
        if int(event['event_context'][0]) != expected_type:
            raise ValueError("Joint event type mismatch")
        if not np.array_equal(event['target_mask'][0], np.isin(event['ranks'], tier_list_for_event(*key))):
            raise ValueError("Missing joint training rank supervision")
    if seen != {(event_split.server, e) for e in event_split.event_ids}:
        raise ValueError("Incomplete joint event split")
    return dict(version=event_split.manifest['version'], strategy=event_split.manifest['strategy'],
                manifest=event_split.manifest, counts_by_type=event_split.counts_by_type(),
                train_events=[[event_split.server, e] for e in sorted(event_split.training_ids)],
                val_events=[[event_split.server, e] for e in sorted(event_split.validation_ids)])

"""Compact V2 HDF5: store each event once, stream cut batches without TensorFlow."""
from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .cutoff_quality import QUALITY_VERSION
from .event_split import EventSplit, full_fit_scope
from .joint_sequences import JOINT_SCHEMA_VERSION, JOINT_RANKS, joint_split_report, validate_joint_event
from .joint_slicing import collate_joint_slices, joint_training_slice
from .joint_weights import JOINT_WEIGHT_VERSION, build_joint_weight_plan, weight_plan_digest
from .rank_order import CUTOFF_SEQUENCE_VERSION
from .rewards import REWARD_FEATURE_NAMES, REWARD_FEATURE_VERSION
from .time_features import TIME_FEATURE_VERSION

ARRAY_FIELDS = ('cutoff', 'available_mask', 'target_mask', 'provenance', 'rank_repaired_mask',
                'latest_observation_ms', 'ranks', 'grid_ms', 'time_features', 'event_context',
                'reward_features', 'reward_feature_mask')
ATTR_FIELDS = ('event_id', 'server', 'start_at', 'end_at', 'snapshot_at_ms', 'ended',
               'balance_group', 'split', 'reward_stage', 'reward_schedule_version', 'reward_schedule_sha256')


def fit_joint_normalization(events):
    """Train events only, count each valid grid value once (not once per cut)."""
    count, mean, m2, delta_count, delta_sum = 0, 0., 0., 0, 0.
    for event in events:
        if event['split'] != 'train':
            raise ValueError("Normalization must use training events only")
        valid = event['available_mask']
        values = np.log1p(event['cutoff'][valid])
        n = len(values)
        if n:
            local_mean = float(values.mean())
            diff = local_mean - mean
            m2 += float(np.square(values - local_mean).sum()) + diff**2 * count * n / (count + n)
            mean += diff * n / (count + n)
            count += n
        increments = np.diff(event['cutoff'], axis=0)[valid[1:] & valid[:-1]]
        if (increments < -1e-7).any():
            raise ValueError("Negative training increments")
        delta_sum += float(np.maximum(increments, 0).sum())
        delta_count += increments.size
    if not count or not delta_count:
        raise ValueError("No valid training values for normalization")
    return dict(log_cutoff_mean=mean, log_cutoff_std=max((m2 / count)**0.5, 1e-6),
                delta_scale=max(delta_sum / delta_count, 1.), valid_grid_values=count,
                valid_increments=delta_count)


def prepare_joint_artifact(events, event_split, **weight_options):
    split_report = joint_split_report(events, event_split)
    plans = {split: build_joint_weight_plan(
        [e for e in events if e['split'] == split],
        balance_rules=split == 'train', **weight_options,
    ) for split in ('train', 'val')}
    return split_report, plans, fit_joint_normalization([e for e in events if e['split'] == 'train'])


def save_joint_hdf5(events, filename, *, event_split, reward_table, overwrite=False, **weight_options):
    path = Path(filename)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {path}; pass overwrite=True explicitly")
    split_report, plans, norm = prepare_joint_artifact(events, event_split, **weight_options)
    if any(e['reward_schedule_sha256'] != reward_table.sha256 for e in events):
        raise ValueError("Mixed reward table snapshots")
    source_identity = {str(e['event_id']): e['source_hashes'] for e in events}
    data_hash = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, 'w') as handle:
            meta = handle.create_group('meta')
            meta.attrs.update(schema_version=JOINT_SCHEMA_VERSION,
                              cutoff_sequence_version=CUTOFF_SEQUENCE_VERSION,
                              time_feature_version=TIME_FEATURE_VERSION, quality_version=QUALITY_VERSION,
                              reward_feature_version=REWARD_FEATURE_VERSION,
                              reward_schedule_version=reward_table.version,
                              reward_schedule_sha256=reward_table.sha256,
                              sample_weight_version=JOINT_WEIGHT_VERSION,
                              data_snapshot_sha256=data_hash,
                              event_split_report=json.dumps(split_report, ensure_ascii=False, sort_keys=True),
                              reward_table=json.dumps(reward_table.document, ensure_ascii=False, sort_keys=True),
                              reward_feature_names=json.dumps(REWARD_FEATURE_NAMES),
                              provenance_codes=json.dumps({0:'unknown', 1:'synthetic_start', 2:'exact',
                                                           3:'pchip', 4:'terminal_fill', 5:'structural_zero'}))
            meta.create_dataset('ranks', data=JOINT_RANKS)
            handle.create_group('norm').attrs.update(norm)
            parent = handle.create_group('events')
            for event in events:
                group = parent.create_group(str(event['event_id']))
                for key in ARRAY_FIELDS:
                    group.create_dataset(key, data=event[key], compression='gzip')
                for key in ATTR_FIELDS:
                    group.attrs[key] = event[key]
                group.attrs['source_hashes'] = json.dumps(event['source_hashes'], sort_keys=True)
            for split, plan in plans.items():
                group = handle.create_group(split)
                group.create_dataset('index', data=plan.index, compression='gzip')
                group.create_dataset('sample_scale', data=plan.sample_scale, compression='gzip')
                group.create_dataset('tier_weights', data=plan.tier_weights)
                group.attrs['weight_report'] = json.dumps(plan.report, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {split: plan.report for split, plan in plans.items()}


class JointDataset:
    """Cache compact events, never expanded cut batches. None retains the whole split.

    Each retained event is read/validated once. An explicit positive capacity keeps
    the legacy LRU policy for memory-constrained users and benchmark controls.
    The loader is single-producer, as required by the tf.data adapter.
    """
    def __init__(self, filename, split, *, cache_events=None):
        if split not in ('train', 'val') or (cache_events is not None and
                (type(cache_events) is not int or cache_events < 1)):
            raise ValueError("Invalid joint dataset split/cache size")
        self.filename, self.split, self.cache_events = str(filename), split, cache_events
        self.cache_policy = 'all_split' if cache_events is None else 'lru'
        self._cache = OrderedDict()
        self._cache_hits = self._cache_misses = self._cache_evictions = 0
        self._file_identity = self._identity()
        with h5py.File(filename, 'r') as handle:
            attrs = handle['meta'].attrs
            for key, expected in (('schema_version', JOINT_SCHEMA_VERSION),
                                  ('time_feature_version', TIME_FEATURE_VERSION),
                                  ('sample_weight_version', JOINT_WEIGHT_VERSION),
                                  ('reward_feature_version', REWARD_FEATURE_VERSION),
                                  ('quality_version', QUALITY_VERSION),
                                  ('cutoff_sequence_version', CUTOFF_SEQUENCE_VERSION)):
                if attrs.get(key) != expected:
                    raise ValueError(f"Incompatible joint data {key}; rebuild V2 HDF5")
            if not np.array_equal(handle['meta/ranks'][:], JOINT_RANKS):
                raise ValueError("Incompatible joint rank axis")
            self.event_split_report = json.loads(attrs['event_split_report'])
            manifest = EventSplit(self.event_split_report['manifest'])
            table = json.loads(attrs['reward_table'])
            table_hash = hashlib.sha256(json.dumps(table, ensure_ascii=False, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if table_hash != attrs['reward_schedule_sha256']:
                raise ValueError("Reward table content hash mismatch")
            self.reward_hash = table_hash
            self.snapshot_hash = attrs['data_snapshot_sha256']
            self.norm = dict(handle['norm'].attrs)
            self.index = handle[split]['index'][:]
            self.sample_scale = handle[split]['sample_scale'][:]
            self.tier_weights = handle[split]['tier_weights'][:]
            self.weight_report = json.loads(handle[split].attrs['weight_report'])
            if (self.index.ndim != 2 or self.index.shape[1] != 2
                    or self.sample_scale.shape != (len(self.index),)
                    or self.tier_weights.shape != (len(JOINT_RANKS),)
                    or not np.issubdtype(self.index.dtype, np.integer)
                    or not np.isfinite(self.sample_scale).all() or (self.sample_scale < 0).any()
                    or not np.isfinite(self.tier_weights).all() or (self.tier_weights < 0).any()):
                raise ValueError("Invalid joint index/weights")
            ids = set(map(int, self.index[:, 0]))
            self.event_ids = frozenset(ids)
            if self.cache_events is None:
                self.cache_events = len(ids)
            expected_ids = manifest.training_ids if split == 'train' else manifest.validation_ids
            if ids != expected_ids:
                raise ValueError("Joint cut index differs from fixed event split")
            for event_id in ids:
                group = handle['events'][str(event_id)]
                T = group['cutoff'].shape[0] - 1
                cuts = self.index[self.index[:, 0] == event_id, 1]
                if (group.attrs['split'] != split or not group.attrs['ended']
                        or not np.array_equal(np.sort(cuts), np.arange(self.weight_report['t_min'], T))):
                    raise ValueError("Incomplete, duplicate or misassigned joint cuts")
            # Audit effective mass from stored scales/masks, not only reported factors.
            mass = np.zeros(2)
            for event_id in ids:
                group = handle['events'][str(event_id)]
                rank_mask = group['target_mask'][0]
                rule = int(group.attrs['balance_group'])
                if rule not in (0,1):
                    raise ValueError("Invalid stored balance group")
                mass[rule] += self.sample_scale[self.index[:,0] == event_id].sum() * self.tier_weights[rank_mask].sum()
            if (self.weight_report.get('version') != JOINT_WEIGHT_VERSION
                    or self.weight_report['joint_samples'] != len(self.index)
                    or not np.allclose(mass, self.weight_report['rule_final_weight_sums'], rtol=1e-10, atol=1e-8)
                    or not np.isclose(mass.sum(), len(self.index), rtol=1e-10, atol=1e-8)):
                raise ValueError("Stored joint weight mass mismatch")
            if split == 'val' and self.weight_report['rule_balance_applied']:
                raise ValueError("Validation must not force rule balance")
        self._check_snapshot()

    def _identity(self):
        value = os.stat(self.filename)
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _check_snapshot(self):
        # Even an all-hit cache must detect replacement/in-place editing. Stat is
        # cheap; no HDF5 open, decompression or whole-event validation on hits.
        if self._identity() != self._file_identity:
            raise ValueError('Joint dataset changed during iteration; reopen the loader')

    def cache_info(self):
        resident = tuple(self._cache.values())  # Producer may prefetch while metrics are read.
        return dict(policy=self.cache_policy, capacity=self.cache_events,
                    events_in_split=len(self.event_ids), resident_events=len(resident),
                    hits=self._cache_hits, misses=self._cache_misses, evictions=self._cache_evictions,
                    resident_array_bytes=sum(event[key].nbytes for event in resident
                                             for key in ARRAY_FIELDS))

    def event(self, event_id):
        self._check_snapshot()
        if event_id not in self.event_ids:
            raise ValueError('Event does not belong to this dataset split')
        if event_id not in self._cache:
            with h5py.File(self.filename, 'r') as handle:
                if (handle['meta'].attrs['data_snapshot_sha256'] != self.snapshot_hash
                        or handle['meta'].attrs['reward_schedule_sha256'] != self.reward_hash):
                    raise ValueError("Joint dataset changed during iteration")
                group = handle['events'][str(event_id)]
                event = {key: group[key][:] for key in ARRAY_FIELDS}
                event.update({key: group.attrs[key] for key in ATTR_FIELDS})
                event['source_hashes'] = json.loads(group.attrs['source_hashes'])
            validate_joint_event(event)
            self._check_snapshot()
            for key in ARRAY_FIELDS:
                event[key].setflags(write=False)
            self._cache_misses += 1
            self._cache[event_id] = event
            if len(self._cache) > self.cache_events:
                self._cache.popitem(last=False)
                self._cache_evictions += 1
        else:
            self._cache_hits += 1
        self._cache.move_to_end(event_id)
        return self._cache[event_id]

    def slice(self, row):
        event_id, t0 = map(int, self.index[row])
        return joint_training_slice(self.event(event_id), t0, self.sample_scale[row], self.tier_weights)

    def batch_rows(self, batch_size, *, shuffle=False, seed=1, boundaries=(160, 224, 320, 448, 688)):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        boundaries = np.asarray(boundaries)
        if boundaries.ndim != 1 or (np.diff(boundaries) <= 0).any():
            raise ValueError("Bucket boundaries must be strictly increasing")
        buckets = np.searchsorted(boundaries, self.index[:, 1] + 1, side='left')
        rng = np.random.default_rng(seed)
        batches = []
        for bucket in np.unique(buckets):
            rows = np.flatnonzero(buckets == bucket)
            if shuffle:
                rng.shuffle(rows)
            batches.extend(rows[start:start+batch_size] for start in range(0, len(rows), batch_size))
        if shuffle:
            rng.shuffle(batches)
        yield from batches

    def batches(self, batch_size, **options):
        for rows in self.batch_rows(batch_size, **options):
            yield collate_joint_slices([self.slice(int(row)) for row in rows])


class FullFitDataset(JointDataset):
    """Reuse slicing/batching, but explicitly consume BOTH original source splits.

    Source HDF5 labels, arrays, weights and normalization are never changed.
    Fresh full-fit normalization/weights are fitted once per complete event.
    """
    def __init__(self, filename, *, cache_events=None):
        self.sources = {name:JointDataset(filename,name,cache_events=cache_events) for name in ('train','val')}
        train, val = self.sources['train'], self.sources['val']
        if (train.event_ids & val.event_ids or train.event_split_report != val.event_split_report
                or train.snapshot_hash != val.snapshot_hash or train.reward_hash != val.reward_hash):
            raise ValueError('Full fit requires matching, disjoint audited source splits')
        self.filename, self.split = train.filename, 'train'
        self.snapshot_hash, self.reward_hash = train.snapshot_hash, train.reward_hash
        self.event_split_report = copy.deepcopy(train.event_split_report)
        self.full_fit_scope = full_fit_scope(self.event_split_report)
        self.event_ids = train.event_ids | val.event_ids
        if set(self.full_fit_scope['event_ids']) != self.event_ids:
            raise ValueError('Full fit must include every source event')
        self.original_split = {eid:name for name,source in self.sources.items() for eid in source.event_ids}
        options = {key:train.weight_report[key] for key in
            ('t_min','weight_floor','tier_center','tier_alpha','tier_floor','tier_1500_weight')}
        if any(val.weight_report[key] != value for key,value in options.items()):
            raise ValueError('Source split weighting settings differ')
        events = [self.event(eid) for eid in sorted(self.event_ids)]
        plan = build_joint_weight_plan(events, balance_rules=True, weight_mode='none', **options)
        self.index, self.sample_scale, self.tier_weights = plan.index, plan.sample_scale, plan.tier_weights
        self.weight_report = plan.report
        self.norm = fit_joint_normalization(events)
        source_keys = {tuple(map(int,row)) for source in self.sources.values() for row in source.index}
        if len(source_keys) != len(self.index) or source_keys != {tuple(map(int,row)) for row in self.index}:
            raise ValueError('Full-fit cut index omitted or duplicated source observations')
        if not self.weight_report['rule_balance_applied']:
            raise ValueError('Full-fit MoE requires both reward-rule groups')
        np.testing.assert_allclose(self.weight_report['rule_final_weight_sums'],len(self.index)/2.,rtol=1e-10)
        self.training_weighting = dict(version='joint-full-fit-weighting-v1', requested='none',
            scope='all-ended-events', effective_weight_plan_sha256=weight_plan_digest(self),
            effective_report=copy.deepcopy(self.weight_report),
            source_weight_plan_sha256={name:weight_plan_digest(source) for name,source in self.sources.items()})

    def event(self, event_id):
        if event_id not in self.original_split:
            raise ValueError('Event is outside the full-fit dataset')
        source_split = self.original_split[event_id]
        original = self.sources[source_split].event(event_id)
        if not original['ended']:
            raise ValueError('An active event cannot supply full-fit labels')
        # Metadata copy only; arrays remain the original read-only views.
        return dict(original, split='train', source_split=source_split)

    def cache_info(self):
        reports = {name:source.cache_info() for name,source in self.sources.items()}
        return dict(policy='full-fit-source-union', events_in_split=len(self.event_ids), sources=reports,
                    resident_array_bytes=sum(r['resident_array_bytes'] for r in reports.values()))

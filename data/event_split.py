"""Fixed event-level development splits; no training-framework dependencies."""
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

from .constants import eventTypeMap, uses_new_event_tiers, tier_list_for_event


DEFAULT_SPLIT_MANIFEST = str(Path(__file__).resolve().parent / 'splits' / 'cn_type_balanced_v1.json')


@dataclass
class EventSplit:
    manifest: dict

    def __post_init__(self):
        m = self.manifest
        if m.get('strategy') != 'fixed-rule-type-holdout' or not m.get('version'):
            raise ValueError('Expected a versioned fixed-rule-type-holdout manifest')
        if m.get('temporal_holdout') is not False:
            raise ValueError('This split must not be represented as a temporal holdout')
        self.server = int(m['server'])
        self.event_types = {}
        self.validation_ids = set()
        candidates = m['events_by_type']
        validation = m['validation_by_type']
        if not candidates or set(candidates) != set(validation):
            raise ValueError('Candidate/validation event types must agree')
        for kind, groups in candidates.items():
            if kind not in eventTypeMap:
                raise ValueError(f'Unknown event type: {kind}')
            if set(groups) != {'old', 'new'} or set(validation[kind]) != {'old', 'new'}:
                raise ValueError('Every event type must specify old and new groups')
            for rule in ('old', 'new'):
                pool, heldout = groups[rule], validation[kind][rule]
                if len(set(pool)) != len(pool) or len(set(heldout)) != len(heldout):
                    raise ValueError('Duplicate event ID in manifest')
                if not set(heldout) <= set(pool):
                    raise ValueError('Validation events must belong to their candidate type/rule')
                for event in pool:
                    if type(event) is not int or event <= 0 or event in self.event_types:
                        raise ValueError(f'Invalid or repeated event: {event}')
                    if uses_new_event_tiers(self.server, event) != (rule == 'new'):
                        raise ValueError(f'Event {event} has an incorrect reward-rule group')
                    self.event_types[event] = kind
                self.validation_ids.update(heldout)
        self.event_ids = set(self.event_types)
        self.training_ids = self.event_ids - self.validation_ids
        if not self.training_ids or not self.validation_ids:
            raise ValueError('Both train and validation must contain events')
        for kind in candidates:
            for ids in (self.training_ids, self.validation_ids):
                if not any(self.event_types[event] == kind for event in ids):
                    raise ValueError(f'Event type {kind} must occur in both splits')

    def split_for(self, server, event):
        if server != self.server or event not in self.event_ids:
            raise ValueError(f'Event {server}/{event} is outside the fixed split manifest')
        return 'val' if event in self.validation_ids else 'train'

    def validate_metadata(self, metadata):
        for event, kind in self.event_types.items():
            actual = metadata.get(str(event), {}).get('eventType')
            if actual != kind:
                raise ValueError(f'Event {event} metadata type mismatch: expected {kind}, got {actual}')

    def counts_by_type(self):
        result = {}
        for split, ids in (('train', self.training_ids), ('val', self.validation_ids)):
            result[split] = {
                kind: {rule: sum(self.event_types[event] == kind and
                                uses_new_event_tiers(self.server, event) == (rule == 'new')
                                for event in ids) for rule in ('old', 'new')}
                for kind in sorted(self.manifest['events_by_type'])
            }
        return result

    def validation_metric_groups(self):
        """Only include groups actually present; absent types must not report 0 loss."""
        groups = {}
        for event in sorted(self.validation_ids):
            rule = 'new' if uses_new_event_tiers(self.server, event) else 'old'
            for name in (rule, f'{rule}_{self.event_types[event]}'):
                groups.setdefault(name, []).append([self.server, event])
        return groups


def load_event_split(path=DEFAULT_SPLIT_MANIFEST):
    with open(path, encoding='utf-8') as handle:
        return EventSplit(json.load(handle))


def audit_sequence_split(sequences, event_split=None):
    """Reject event leakage and, for fixed manifests, omitted events/ranks."""
    assignments, ranks = {}, {}
    for seq in sequences:
        ctx = seq['context_cat']
        key = (int(ctx[3]), int(ctx[4]))
        split = seq.get('split', 'train')
        if split not in ('train', 'val'):
            raise ValueError(f'Invalid split for event {key}: {split}')
        if key in assignments and assignments[key] != split:
            raise ValueError(f'Event {key} appears in both train and validation')
        assignments[key] = split
        if event_split is None:
            continue
        if split != event_split.split_for(*key):
            raise ValueError(f'Event {key} does not match the fixed split assignment')
        if int(ctx[0]) != eventTypeMap[event_split.event_types[key[1]]]:
            raise ValueError(f'Event {key} sequence type does not match its manifest')
        tier = int(seq['tier'])
        seen = ranks.setdefault(key, set())
        if tier in seen:
            raise ValueError(f'Duplicate sequence for event {key}, tier {tier}')
        seen.add(tier)
    if event_split is not None:
        expected = {(event_split.server, event) for event in event_split.event_ids}
        if set(assignments) != expected:
            raise ValueError(f'Fixed split is incomplete; missing events: {sorted(expected - set(assignments))}')
        for key, tiers in ranks.items():
            if tiers != set(tier_list_for_event(*key)):
                raise ValueError(f'Event {key} has missing or unsupported ranks')
    report = {
        'strategy': 'explicit-sequence-labels', 'version': 'explicit-sequence-labels-v1',
        'train_events': [list(key) for key in sorted(assignments) if assignments[key] == 'train'],
        'val_events': [list(key) for key in sorted(assignments) if assignments[key] == 'val'],
    }
    if event_split is not None:
        report.update(strategy=event_split.manifest['strategy'], version=event_split.manifest['version'],
                      manifest=event_split.manifest, counts_by_type=event_split.counts_by_type())
    return report


def read_event_split_report(h5_file):
    meta = h5_file['meta'].attrs if 'meta' in h5_file else {}
    payload = meta.get('event_split_report')
    if payload is None:
        warnings.warn(
            'HDF5 缺少活动划分记录，可能仍使用旧编号阈值；'
            '请运行 generate_raw_h5.py 和 preprocess.py 重建数据。',
            RuntimeWarning, stacklevel=2,
        )
        return None
    return json.loads(payload)


FULL_FIT_VERSION = 'joint-all-ended-full-fit-v1'


def full_fit_scope(split_report):
    split = EventSplit(split_report['manifest'])
    return dict(version=FULL_FIT_VERSION, kind='all-ended-events',
        source_splits=['train', 'val'], event_ids=sorted(split.event_ids),
        independent_validation=False)


def validate_full_fit_scope(scope, split_report):
    expected = full_fit_scope(split_report)
    if (not isinstance(scope, dict) or scope != expected
            or type(scope.get('independent_validation')) is not bool
            or not isinstance(scope.get('event_ids'), list)
            or any(type(i) is not int for i in scope['event_ids'])):
        raise ValueError('Invalid full-fit scope: every source event must be training-only')
    return expected


def chronological_scope(split_report, validation_ids):
    """Use the frozen source union for training and later events for development."""
    scope = full_fit_scope(split_report)
    if (not isinstance(validation_ids, list) or not validation_ids
            or any(type(i) is not int or i <= max(scope['event_ids']) for i in validation_ids)
            or validation_ids != sorted(set(validation_ids))):
        raise ValueError('Chronological validation must contain distinct later events')
    return dict(scope, version='joint-chronological-development-v1', kind='chronological-development',
                validation_event_ids=validation_ids, independent_validation=True)


def validate_model_fit_scope(scope, split_report):
    if isinstance(scope, dict) and scope.get('kind') == 'chronological-development':
        expected = chronological_scope(split_report, scope.get('validation_event_ids'))
        if scope != expected or scope.get('independent_validation') is not True:
            raise ValueError('Invalid chronological development scope')
        return expected
    return validate_full_fit_scope(scope, split_report)

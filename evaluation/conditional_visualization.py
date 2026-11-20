"""Replay rolling validation issues and draw the existing slice-sum plots for joint curves."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from data.joint_dataset import JointDataset
from evaluation.visualization import plot_slice_forecasts, reconstruction_points, slice_points
from evaluation.validation_gallery import use_plot_font, write_gallery
from model.conditional_curve import ConditionalCurveModel, DAY
from utils.files import file_sha256
from data.curve_history import causal_input, read_observations


FOCUS_RANKS = (500, 1000, 1500, 2000)


def _render_event(job):
    model_path, event, raw_dir, out, ranks, labels, t_min, max_slices = job
    use_plot_font()
    model = ConditionalCurveModel.load(model_path)
    series = read_observations(raw_dir, event)
    eid = int(event['event_id'])
    length = len(event['grid_ms']) - 1
    issues = np.asarray(slice_points(length, t_min, max_slices), dtype=int)
    cols = [list(event['ranks']).index(rank) for rank in ranks]
    endpoints = np.full((len(issues), len(ranks)), np.nan)
    anchors = np.full_like(endpoints, np.nan)
    cuts = np.full(len(issues), -1, dtype=int)
    desired = set(reconstruction_points(length, t_min))
    segments = {}
    for i, issue in enumerate(issues):
        built = causal_input(event, series, int(issue))
        if built is None:
            continue
        inputs, cut = built
        cuts[i] = cut
        _, predicted = model.predict(inputs)
        valid = inputs['anchor_mask'][cols]
        endpoints[i, valid] = predicted[-1, cols][valid]
        anchors[i, valid] = inputs['anchor'][cols][valid]
        if issue in desired:
            seg = np.vstack([inputs['anchor'][cols], predicted[:, cols]])
            seg[:, ~valid] = np.nan
            segments[int(issue)] = (cut, seg)

    finals = np.array([labels.get((eid, rank), event['cutoff'][-1, col]) for rank, col in zip(ranks, cols)])
    paths = {f'segment_{i}': values for i, (_, values) in segments.items()}
    np.savez_compressed(out / 'rolling' / f'{eid}.npz', issues=issues, cuts=cuts, ranks=ranks,
                        anchors=anchors, endpoints=endpoints, finals=finals, **paths)
    chart_rows = []
    for j, (rank, col) in enumerate(zip(ranks, cols)):
        forecasts = endpoints[:, j]
        mask = np.isfinite(forecasts)
        count = int(mask.sum())
        record = dict(rank=rank, points=count, final=float(finals[j]), mape=None, image=None)
        if count:
            image = f'val_slice_viz/event-{eid}-rank-{rank}.png'
            rank_segments = {i:(cut, curve[:, j]) for i, (cut, curve) in segments.items()
                             if np.isfinite(curve[0, j])}
            plot_slice_forecasts(event['cutoff'][:, col], issues, anchors[:, j], forecasts,
                                 finals[j], rank_segments, f'国服 {eid} 期 · {rank} 档 · 条件曲线验证',
                                 out / image, show_error=True)
            record['image'] = image
            if finals[j] > 0:
                record['mape'] = float(np.mean(np.abs(forecasts[mask] - finals[j])) * 100 / finals[j])
        chart_rows.append(record)
    return dict(event=eid, rule='new' if event['balance_group'] else 'old', length=length,
                scheduled=len(issues), covered=int((cuts >= 0).sum()), ranks=chart_rows)


def visualize_conditional_validation(model_path, joint_h5, raw_dir, out_dir, *, final_labels=None,
                                     n_seq=1000, t_min=DAY, max_slices=1024, seed=1, workers=4):
    if t_min < DAY or min(workers, n_seq, max_slices) < 1:
        raise ValueError('Conditional curves require at least 24h of history and positive sampling counts')
    dataset = JointDataset(joint_h5, 'val')
    model = ConditionalCurveModel.load(model_path)
    if dataset.event_ids.intersection(model.metadata['train_events']):
        raise ValueError('The supplied model was fitted on validation events')
    if dataset.reward_hash != model.reward_table.sha256:
        raise ValueError('The model and validation data have different reward tables')
    labels = {}
    if final_labels:
        for row in json.loads(Path(final_labels).read_text()):
            key, value = (int(row['event_id']), int(row['tier'])), float(row['final'])
            if not np.isfinite(value) or value < 0 or (key in labels and labels[key] != value):
                raise ValueError(f'Invalid or conflicting final label for {key}')
            labels[key] = value
    out = Path(out_dir)
    for name in ('rolling', 'val_slice_viz'):
        (out / name).mkdir(parents=True, exist_ok=True)
    events = {eid: dataset.event(eid) for eid in sorted(dataset.event_ids)}
    available = [(eid, int(rank)) for eid, e in events.items() for rank in e['ranks'][e['target_mask'][-1]]]
    if n_seq < len(available):
        pick = sorted(np.random.default_rng(seed).choice(len(available), size=n_seq, replace=False))
        available = [available[i] for i in pick]
    selected = defaultdict(list)
    for eid, rank in available:
        selected[eid].append(rank)
    jobs = [(model_path, events[eid], raw_dir, out, ranks, labels, t_min, max_slices)
            for eid, ranks in sorted(selected.items(), reverse=True)]
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for event in pool.map(_render_event, jobs):
            results.append(event)
            print(f"CN {event['event']} | rolling issues {event['covered']}/{event['scheduled']} "
                  f"| events {len(results)}/{len(jobs)}", flush=True)
    report = dict(model_sha256=file_sha256(model_path), data_sha256=file_sha256(joint_h5),
                  t_min=t_min, max_slices=max_slices, events=results)
    for event in results:
        errors = [r['mape'] for r in event['ranks'] if r['rank'] in FOCUS_RANKS and r['mape'] is not None]
        event['focus_mape'] = float(np.mean(errors)) if errors else None
    report['focus_mape'] = {}
    for group in ('old', 'new', 'all'):
        errors = [e['focus_mape'] for e in results if (group == 'all' or e['rule'] == group)
                  and e['focus_mape'] is not None]
        report['focus_mape'][group] = float(np.mean(errors)) if errors else None
    tmp = out / 'validation.tmp'
    tmp.write_text(json.dumps(report, allow_nan=False, indent=2) + '\n')
    tmp.replace(out / 'validation.json')
    write_gallery(report, out, FOCUS_RANKS)
    return report

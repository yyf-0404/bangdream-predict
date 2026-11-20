"""Build the same timestamp-truncated history for offline replay and live forecasts."""
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .constants import step_length, tier_list_for_event
from .cutoff_quality import load_cutoff_snapshot
from .joint_sequences import JOINT_RANKS
from .joint_slicing import joint_model_input
from .rank_order import project_ranked_cumulative


def read_observations(raw_dir, event, *, require_complete=True):
    series = {}
    server = event.get('server', 3)
    for rank in tier_list_for_event(server, event['event_id']):
        path = Path(raw_dir) / str(server) / str(event['event_id']) / f'{rank}.json'
        if not require_complete and not path.is_file():
            continue
        points = load_cutoff_snapshot(raw_dir, server, event['event_id'], rank)
        if points:
            times = np.asarray([p['time']-event['start_at'] for p in points], dtype=np.int64)
            values = np.asarray([p['ep'] for p in points], dtype=np.float64)
            if times[0] < 0 or times[-1] > event['end_at']-event['start_at']:
                raise ValueError('Observation outside the event window')
            series[rank] = times, values
    return series


def as_of_input(event, series, issued_at_ms, *, ranks=None):
    """Return model input and the last common half-hour anchor for an active forecast."""
    elapsed = int(issued_at_ms) - event['start_at']
    duration = event['end_at'] - event['start_at']
    if elapsed < 0 or elapsed >= duration:
        return None
    tiers = tier_list_for_event(event.get('server', 3), event['event_id'])
    focus = [r for r in (500, 1000, 1500, 2000) if r in tiers] if ranks is None else list(ranks)
    if not focus or any(r not in tiers for r in focus):
        raise ValueError('Requested a missing or unsupported event tier')
    counts = {r:int(np.searchsorted(t, elapsed, side='right')) for r,(t, _) in series.items()}
    if any(not counts.get(r) for r in focus):
        return None
    measured = [r for r in sorted(series) if counts[r]]
    limits = {r:int(series[r][0][counts[r]-1]//step_length) for r in measured}
    cut = min(limits[r] for r in focus)
    if cut < 48:
        return None
    grid = np.arange(cut+1, dtype=np.int64) * step_length
    samples = np.empty((cut+1, len(measured)))
    for col, rank in enumerate(measured):
        t, v = (arr[:counts[rank]] for arr in series[rank])
        if t[0] > 0:
            t, v = np.r_[0, t], np.r_[0., v]
        t, v = np.r_[t, duration], np.r_[v, v[-1]]
        samples[:, col] = PchipInterpolator(t, v)(grid)
    valid_to = np.minimum(cut, [limits[r] for r in measured])
    ordered = project_ranked_cumulative(samples, valid_to)
    columns = [JOINT_RANKS.index(r) for r in measured]
    shape = (len(event['grid_ms']), len(JOINT_RANKS))
    scores = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    scores[:cut+1, columns] = ordered
    mask[:cut+1, columns] = np.arange(cut+1)[:, None] <= valid_to
    return joint_model_input(dict(event, cutoff=scores, available_mask=mask), cut), cut


def causal_input(event, series, issue):
    return as_of_input(event, series, event['start_at'] + int(issue)*step_length)

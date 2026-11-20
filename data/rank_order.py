"""Rank constraints for derived cumulative EP grids, not raw observations."""

import numpy as np


CUTOFF_SEQUENCE_VERSION = "pchip-rank-pava-v1"


def antitonic_regression(values):
    """Equal-weight least-squares non-increasing fit using PAVA.

    Kept local because the project's SciPy 1.11 does not provide PAVA.
    Already ordered values are unchanged; ties are allowed.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("PAVA expects a finite one-dimensional array")
    means, counts = [], []
    for value in values:
        means.append(float(value))
        counts.append(1)
        while len(means) > 1 and means[-2] < means[-1]:
            count = counts[-2] + counts[-1]
            mean = (means[-2] * counts[-2] + means[-1] * counts[-1]) / count
            means[-2:] = [mean]
            counts[-2:] = [count]
    return np.repeat(means, counts).astype(np.float64)


def project_ranked_cumulative(cumulative, last_valid_indices=None, zero_tiers=None):
    """Project a (time, rank) grid ordered by increasing numeric rank.

    Only an observed prefix of each active-event rank participates. Synthetic
    end padding must not constrain ranks which have newer observations. Valid
    rank sets therefore only shrink along time. A temporal floor prevents a
    rank falling when an earlier PAVA constraint disappears with a stale peer.
    PAVA is order-preserving: projecting above the preceding ordered row also
    stays above that row. For a fixed rank set the floor is redundant.

    Uncovered tails remain placeholders, never newly available observations.
    Structural zero ranks are immutable; contradictory positive lower ranks
    are rejected rather than turning confirmed zero into a positive estimate.
    """
    values = np.asarray(cumulative, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a finite nonnegative (time, rank) EP grid")
    steps, ranks = values.shape
    if (np.diff(values, axis=0) < 0).any():
        raise ValueError("Cumulative EP must be non-decreasing before rank projection")
    if last_valid_indices is None:
        last_valid_indices = np.full(ranks, steps - 1, dtype=np.int64)
    limits = np.asarray(last_valid_indices)
    if limits.shape != (ranks,) or not np.issubdtype(limits.dtype, np.integer):
        raise ValueError("Expected one integer coverage limit per rank")
    if (limits < -1).any() or (limits >= steps).any():
        raise ValueError("Coverage limits must be between -1 and the final grid index")
    zeros = np.zeros(ranks, dtype=bool) if zero_tiers is None else np.asarray(zero_tiers, dtype=bool)
    if zeros.shape != (ranks,):
        raise ValueError("Expected one structural-zero flag per rank")
    if zeros.any() and (values[:, np.flatnonzero(zeros)[0]:] != 0).any():
        raise ValueError("A structural zero rank conflicts with positive EP at that or a lower rank")

    result = values.copy()
    for step in range(steps):
        if step:
            # Also keep the unused tail placeholders temporally non-decreasing.
            result[step] = np.maximum(result[step], result[step - 1])
        valid = np.flatnonzero(limits >= step)
        row = result[step, valid]
        if (np.diff(row) > 0).any():
            result[step, valid] = antitonic_regression(row)
    return result

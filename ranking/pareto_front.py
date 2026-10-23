"""Vectorized NSGA-II style non-dominated sorting for multi-objective ranking."""

from typing import List, Optional

import numpy as np
import pandas as pd

# JSON cannot represent Infinity, so boundary points of a front (which have infinite
# crowding distance by definition) are stored with this large finite value instead.
CROWDING_CAP = 1_000_000.0

# Missing objective values are treated as the worst possible (minimization) value.
MISSING_PENALTY = 99.0

# Max elements in one broadcast comparison block (keeps memory bounded for large n).
_BLOCK_ELEMS = 20_000_000


def _non_dominated_mask(coords: np.ndarray) -> np.ndarray:
    """Return a boolean mask of rows not dominated by any other row (minimization)."""
    m, d = coords.shape
    nd = np.ones(m, dtype=bool)
    if m <= 1:
        return nd
    block = max(1, _BLOCK_ELEMS // max(1, m * d))
    for start in range(0, m, block):
        cand = coords[start:start + block]                      # (b, d)
        le = np.all(coords[:, None, :] <= cand[None, :, :], axis=2)  # (m, b): i <= j everywhere
        lt = np.any(coords[:, None, :] < cand[None, :, :], axis=2)   # (m, b): i < j somewhere
        nd[start:start + block] = ~np.any(le & lt, axis=0)
    return nd


def _crowding_distance(coords: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Standard NSGA-II crowding distance for the members of one front."""
    k = len(front)
    dist = np.zeros(k, dtype=float)
    if k <= 2:
        dist[:] = np.inf
        return dist
    pts = coords[front]
    for obj in range(pts.shape[1]):
        order = np.argsort(pts[:, obj], kind="stable")
        span = pts[order[-1], obj] - pts[order[0], obj]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if span == 0:
            continue
        gaps = (pts[order[2:], obj] - pts[order[:-2], obj]) / span
        dist[order[1:-1]] += gaps
    return dist


def compute_pareto_fronts(
    df: pd.DataFrame,
    minimize_cols: List[str],
    maximize_cols: List[str],
    stop_after: Optional[int] = None,
) -> pd.DataFrame:
    """Assign `pareto_rank`, `crowding_distance` and `on_pareto_front` to each row.

    Objectives in `minimize_cols` are minimized and those in `maximize_cols` maximized.
    NaNs are penalized as worst-case. If `stop_after` is given, peeling stops once at
    least that many rows have been ranked; the remaining rows all receive one extra,
    lowest rank (they are never ranked against each other), which is enough to pick a
    top-k list cheaply.
    """
    if df.empty or not (minimize_cols or maximize_cols):
        df["pareto_rank"] = 1
        df["crowding_distance"] = 0.0
        df["on_pareto_front"] = True
        return df

    obj_cols = list(minimize_cols) + [c for c in maximize_cols if c in df.columns]
    n_min = len(minimize_cols)
    data = df[obj_cols].astype(float).copy()
    for col in obj_cols[n_min:]:
        data[col] = -data[col]
    coords = data.fillna(MISSING_PENALTY).to_numpy()
    n = len(coords)

    ranks = np.zeros(n, dtype=int)
    crowding = np.zeros(n, dtype=float)
    remaining = np.arange(n)
    rank = 0
    ranked = 0

    while remaining.size:
        if stop_after is not None and ranked >= stop_after:
            ranks[remaining] = rank + 1
            crowding[remaining] = 0.0
            break
        rank += 1
        mask = _non_dominated_mask(coords[remaining])
        front = remaining[mask]
        ranks[front] = rank
        crowding[front] = _crowding_distance(coords, front)
        ranked += front.size
        remaining = remaining[~mask]

    crowding = np.where(np.isinf(crowding), CROWDING_CAP, crowding)

    df["pareto_rank"] = ranks
    df["crowding_distance"] = crowding
    df["on_pareto_front"] = ranks == 1
    return df

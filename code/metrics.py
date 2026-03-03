"""
Evaluation metric functions for post-disaster housing recovery.

All metrics return scalars or dicts computed from a completed episode's
building list.  Used by evaluate.py to fill the master output table.

5 metrics:
  1. Makespan        — total days until all buildings repaired
  2. Mean wait time  — average days from arrival to work start
  3. Recovery curve  — cumulative completion fraction C(t) vs t
  4. Priority response time — per-PL mean response time
  5. Recovery AUC    — normalised area under recovery curve
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# We import Building only for type hints; avoid circular import.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .env import Building


def makespan(buildings: List[Building]) -> int:
    """T* = max done_day over all buildings.
    Source: standard JSSP metric (Artigues 2025)."""
    done_days = [b.done_day for b in buildings if b.done_day >= 0]
    return max(done_days) if done_days else -1


def mean_wait_time(buildings: List[Building]) -> float:
    """Average days from arrival to work start, over all buildings.
    Source: Lv et al. (2025), mean flow time."""
    waits = [
        b.start_day - b.arrive_day
        for b in buildings
        if b.start_day >= 0
    ]
    return float(np.mean(waits)) if waits else 0.0


def priority_response_times(buildings: List[Building]) -> Dict[int, float]:
    """Per-PL mean response time (arrival → completion)."""
    by_pl: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    for b in buildings:
        if b.done_day >= 0:
            by_pl[b.pl].append(b.done_day - b.arrive_day)
    return {
        pl: float(np.mean(vs)) if vs else 0.0
        for pl, vs in by_pl.items()
    }


def recovery_curve(buildings: List[Building]) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative completion curve C(t) = n_completed(t) / N.

    Returns (days, fracs) arrays for plotting."""
    total = len(buildings)
    if total == 0:
        return np.array([0]), np.array([0.0])

    done_days = sorted(b.done_day for b in buildings if b.done_day >= 0)
    if not done_days:
        return np.array([0]), np.array([0.0])

    max_day = done_days[-1]
    days = np.arange(0, max_day + 1)
    fracs = np.zeros(len(days), dtype=np.float64)
    completed = 0
    idx = 0
    for d in range(len(days)):
        while idx < len(done_days) and done_days[idx] <= d:
            completed += 1
            idx += 1
        fracs[d] = completed / total

    return days, fracs


def recovery_auc(buildings: List[Building]) -> float:
    """Normalised AUC of recovery curve: ∫C(t)dt / T*.
    Higher = faster overall recovery."""
    days_arr, fracs_arr = recovery_curve(buildings)
    if len(days_arr) <= 1:
        return 0.0
    ms = int(days_arr[-1])
    if ms <= 0:
        return 1.0
    return float(np.trapezoid(fracs_arr, days_arr) / ms)


def all_metrics(buildings: List[Building]) -> Dict[str, float]:
    """Compute all 5 scalar metrics at once."""
    prt = priority_response_times(buildings)
    return {
        "makespan": makespan(buildings),
        "mean_wait": mean_wait_time(buildings),
        "response_pl0": prt[0],
        "response_pl1": prt[1],
        "response_pl2": prt[2],
        "auc": recovery_auc(buildings),
    }

"""
Baseline scheduling policies for post-disaster housing recovery.

Each policy is a callable:  policy(obs: ndarray) → action: ndarray(5,)
where action = [w_remain, w_wait, w_total_work, w_priority, w_crew_cap].

These 5 weights define a linear scoring function applied to each building:
    score(b) = w^T · [remain/50, wait/50, total/50, pl/2, crew/6]

Buildings are sorted by score (descending) and crews greedily assigned.

5 baselines covering Lv et al. (2025) core rule families:
  - LJF / SJF     : workload-oriented (Longest / Shortest Job First)
  - StrictPriority : priority-oriented (real BNPB policy simulation)
  - FIFO           : fairness-oriented
  - Random         : lower-bound reference
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import config as C

# Type alias for a baseline policy function.
Policy = Callable[[np.ndarray], np.ndarray]


# ──────────────────────────────────────────────────────────────────────
# Feature order: [remain_work, wait_time, total_work, priority, crew_cap]
# ──────────────────────────────────────────────────────────────────────

def ljf_policy(obs: np.ndarray) -> np.ndarray:
    """Longest Job First: weight remaining_work positively.

    Source: Lv et al. (2025), LPT/LJF rule.
    """
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def sjf_policy(obs: np.ndarray) -> np.ndarray:
    """Shortest Job First: weight remaining_work negatively.

    Source: Lv et al. (2025), SPT rule — optimal for minimising mean response
    time in single-machine scheduling.
    """
    return np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def strict_priority_policy(obs: np.ndarray) -> np.ndarray:
    """StrictPriority: weight priority_level positively.

    Simulates real BNPB policy where critical infrastructure (hospitals,
    schools) is repaired first.
    Source: BNPB PERKA No.4/2012; Liang et al. (2026).
    """
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def fifo_policy(obs: np.ndarray) -> np.ndarray:
    """FIFO: weight waiting_time positively.

    Source: standard fairness baseline.
    """
    return np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# Stateful RNG for random policy (seeded per evaluation run).
_random_rng = np.random.default_rng(42)


def random_policy(obs: np.ndarray) -> np.ndarray:
    """Random: uniform random weights.  Lower-bound reference.

    Source: standard baseline in RL evaluation.
    """
    return _random_rng.standard_normal(C.ACT_DIM).astype(np.float32)


def seed_random_policy(seed: int) -> None:
    """Re-seed the random policy's RNG (call before each eval episode)."""
    global _random_rng
    _random_rng = np.random.default_rng(seed)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

ALL_BASELINES: dict[str, Policy] = {
    "LJF": ljf_policy,
    "SJF": sjf_policy,
    "StrictPriority": strict_priority_policy,
    "FIFO": fifo_policy,
    "Random": random_policy,
}

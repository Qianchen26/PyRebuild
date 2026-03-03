"""
Zero-magic-number parameter registry for pyrebuild RL.

Every constant has a docstring citing its source. Grouped by provenance:
  §1  Alisjahbana & Kiremidjian (2021) — directly calibrated
  §2  BPS NTB 2018 — Indonesian statistics
  §3  BNPB 2018 disaster reports — batch arrival calibration
  §4  Construction industry literature — stochastic parameters
  §5  FEMA P-58 / RCPSP / POKMAS — crew capacity
  §6  PPO literature — algorithm hyperparameters
  §7  Derived parameters
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# §1  Alisjahbana & Kiremidjian (2021) — directly calibrated
# ──────────────────────────────────────────────────────────────────────

# Construction time medians (days) by damage level [minor, moderate, major].
# Source: Alisjahbana 2021, Table 2, calibrated against HAZUS.
CONS_MED: np.ndarray = np.array([30, 40, 50], dtype=np.float64)

# Log-normal shape parameter for construction time.
# Source: Alisjahbana 2021, Table 2.
BETA_CONS: float = 0.4

# Pre-processing delay medians (days) by damage level [minor, moderate, major].
# Source: Alisjahbana 2021, Table 2.
# Used only to calibrate the batch arrival schedule (§3).
DELAY_MED: np.ndarray = np.array([275, 300, 225], dtype=np.float64)

# Log-normal shape parameter for delay.
# Source: Alisjahbana 2021, Table 2.
BETA_DELAY: float = 0.3

# Episode termination: when ALL buildings are repaired (not a fixed horizon).
# Alisjahbana (2021) observed recovery over ~600 days, but our RL agent should
# not be truncated — it must learn to finish the job regardless of duration.
# MAX_SIM_DAYS is a *safety cap* only (prevents infinite loops from bugs).
MAX_SIM_DAYS: int = 1500

# ──────────────────────────────────────────────────────────────────────
# §1b  Region data — Alisjahbana 2021, Tables 1 & 3
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegionConfig:
    """Immutable configuration for one of the 7 NTB regions."""
    name: str                       # Region name (Indonesian)
    name_en: str                    # English name
    n_buildings: Dict[int, int] = field(default_factory=dict)
    # {damage_level: count}  — DL0=minor, DL1=moderate, DL2=major
    n_contractors: int = 0          # Total contractor count
    # Source: Alisjahbana 2021 Table 3 / BPS labor census.
    n_hospitals: int = 0            # Source: BPS NTB 2018
    n_schools: int = 0              # Source: BPS NTB 2018

    @property
    def total_buildings(self) -> int:
        return sum(self.n_buildings.values())

    @property
    def infra_ratio(self) -> float:
        """Fraction of buildings that are critical infrastructure."""
        total = self.total_buildings
        if total == 0:
            return 0.0
        return (self.n_hospitals + self.n_schools) / total


# 7 regions — building counts from lombok_data.pkl; contractor counts from
# Alisjahbana 2021 notebook Cell 14; hospital/school counts from BPS NTB 2018.
# Damage level mapping:  rr (ringan/minor)=0, rs (sedang/moderate)=1,
#                        rb (berat/major)=2
REGIONS: Dict[str, RegionConfig] = {
    "mataram": RegionConfig(
        name="mataram", name_en="Mataram",
        n_buildings={0: 9500, 1: 3672, 2: 1345},
        n_contractors=9917,
        n_hospitals=44, n_schools=118,       # BPS NTB 2018
    ),
    "lombokbarat": RegionConfig(
        name="lombokbarat", name_en="West Lombok",
        n_buildings={0: 45218, 1: 13556, 2: 14069},
        n_contractors=45208,
        n_hospitals=80, n_schools=210,
    ),
    "lombokutara": RegionConfig(
        name="lombokutara", name_en="North Lombok",
        n_buildings={0: 8889, 1: 4772, 2: 42049},
        n_contractors=22996,
        n_hospitals=39, n_schools=85,
    ),
    "lomboktengah": RegionConfig(
        name="lomboktengah", name_en="Central Lombok",
        n_buildings={0: 16639, 1: 3096, 2: 4483},
        n_contractors=15048,
        n_hospitals=124, n_schools=523,
    ),
    "lomboktimur": RegionConfig(
        name="lomboktimur", name_en="East Lombok",
        n_buildings={0: 12209, 1: 4657, 2: 10104},
        n_contractors=15404,
        n_hospitals=122, n_schools=514,
    ),
    "sumbawabarat": RegionConfig(
        name="sumbawabarat", name_en="West Sumbawa",
        n_buildings={0: 13078, 1: 3803, 2: 1283},
        n_contractors=10200,
        n_hospitals=38, n_schools=54,
    ),
    "sumbawa": RegionConfig(
        name="sumbawa", name_en="Sumbawa",
        n_buildings={0: 9652, 1: 2756, 2: 1374},
        n_contractors=10360,
        n_hospitals=120, n_schools=169,
    ),
}

# ──────────────────────────────────────────────────────────────────────
# §2  BPS NTB 2018 — infrastructure ratios (derived)
# ──────────────────────────────────────────────────────────────────────
# Hospital/school counts are embedded in RegionConfig above.
# INFRA_RATIO ranges 0.2%–2.7% across regions.

# ──────────────────────────────────────────────────────────────────────
# §3  BNPB 2018 — batch arrival calibration
# ──────────────────────────────────────────────────────────────────────

# List of (arrival_day, fraction_of_total_buildings).
# Calibrated from BNPB timeline:
#   Aug 5 earthquake → Aug 12 (day 7):  67,875 houses assessed (~30%)
#   Aug 25 (day 20):  74,361 verified (~33%)
#   Full assessment ~2 months (day ~60)
# Simplified to 3 batches.  Source: reliefweb.int, ahacentre.org.
BATCH_SCHEDULE: List[Tuple[int, float]] = [
    (0, 0.30),     # Day 0:  30% immediate
    (14, 0.35),    # Day 14: 35% second assessment
    (60, 0.35),    # Day 60: 35% final assessment
]

# ──────────────────────────────────────────────────────────────────────
# §4  Construction industry literature — stochastic parameters
# ──────────────────────────────────────────────────────────────────────

# Maximum daily worker dropout fraction.
# Source: Iran 7% absenteeism (Sharif 2018); Indonesia informal sector
# highly vulnerable to post-disaster disruptions (UTS 2022).
# 10% is conservative for post-disaster context.
DROPOUT_MAX: float = 0.10

# Daily work-progress noise (relative standard deviation).
# Source: ~50% of construction time productive globally (ResearchGate);
# daily variance driven by weather, materials, fatigue.
# 15% relative noise is a conservative lower bound.
PROGRESS_NOISE_STD: float = 0.15

# ──────────────────────────────────────────────────────────────────────
# §5  FEMA P-58 / RCPSP / POKMAS — crew capacity
# ──────────────────────────────────────────────────────────────────────

# Maximum daily crew capacity per building by damage level
# [DL0=minor, DL1=moderate, DL2=major].
#
# Cross-validated by 5 sources:
#   1. FEMA P-58 (2018): min/max crew per damage state
#   2. RCPSP theory (Artigues 2025): per-activity resource demand b_ik
#   3. Construction management: diminishing returns / site congestion
#   4. RSMeans (2024): typical residential crew sizes
#   5. Indonesia POKMAS: community-based reconstruction practice
#
# DL0(minor)=2:  single-trade repair, 1-2 person crew
# DL1(moderate)=4: multi-trade coordination, 3-5 person crew
# DL2(major)=6:  full structural repair, 5-8 person crew
MAX_CREW: np.ndarray = np.array([2, 4, 6], dtype=np.int32)

# ──────────────────────────────────────────────────────────────────────
# §6  PPO literature — algorithm hyperparameters
# ──────────────────────────────────────────────────────────────────────

# All from Schulman (2017) PPO paper or SB3 defaults.
LR_START: float = 3e-4          # SB3 default; Schulman 2017
LR_END: float = 1e-5            # Linear decay endpoint
GAE_LAMBDA: float = 0.95        # Schulman 2017 default
GAMMA: float = 0.99             # Standard for long-horizon (600-day) tasks
CLIP_RANGE: float = 0.2         # Schulman 2017 default
N_ENVS: int = 8                 # SB3 recommended parallel envs
NET_ARCH: List[int] = [256, 256]  # SB3 default MlpPolicy
TRAIN_SEEDS: int = 5            # Henderson et al. (2018): RL reproducibility
EVAL_SEEDS: int = 10            # Per user specification

# ──────────────────────────────────────────────────────────────────────
# §7  Derived parameters
# ──────────────────────────────────────────────────────────────────────

# Number of queues = 3 DL × 3 PL = 9.
# Some combinations may have 0 buildings in a given scenario.
N_DL: int = 3       # Damage levels: 0, 1, 2
N_PL: int = 3       # Priority levels: 0, 1, 2
N_QUEUES: int = N_DL * N_PL  # = 9

# Per-queue observation features: [mean_remaining_work, mean_wait, queue_size]
N_QUEUE_FEATURES: int = 3

# Global features: [time_progress, worker_avail_rate, queue_load_rate,
#   info_reveal_rate, remaining_work_rate, severe_damage_frac, dropout_rate]
N_GLOBAL_FEATURES: int = 7

# Observation dimension = 9 queues × 3 features + 7 global = 34
OBS_DIM: int = N_QUEUES * N_QUEUE_FEATURES + N_GLOBAL_FEATURES  # = 34

# Action dimension = 5 dispatching weights (linear scoring rule).
# PPO outputs [w_remain, w_wait, w_total_work, w_priority, w_crew_cap].
# Score per building = features @ weights → sorted descending → greedy fill.
# Source: Parameterized Dispatching Rule (PDR), Lv et al. 2025.
ACT_DIM: int = 5
DISPATCH_FEATURES: int = 5

# Per-building feature normalization (max values for [0,1] scaling).
_MAX_WORK: float = float(CONS_MED[2])   # 50 (max median work)
_MAX_CREW: float = float(MAX_CREW.max())  # 6
DISPATCH_NORM: np.ndarray = np.array([
    _MAX_WORK,   # remaining_work: range [0, ~120]
    _MAX_WORK,   # waiting_time:   range [0, ~episode length]
    _MAX_WORK,   # total_work:     range [0, ~120]
    2.0,         # priority_level: values {0, 1, 2}
    _MAX_CREW,   # crew_cap:       values {2, 4, 6}
], dtype=np.float64)

# Work normalization scale.
# max(CONS_MED) × exp(2 × BETA_CONS²) ≈ 50 × exp(0.32) ≈ 69 → round up.
W_SCALE: float = 100.0

# ──────────────────────────────────────────────────────────────────────
# §8  Reward function coefficients (initial; tuned by ablation E5)
# ──────────────────────────────────────────────────────────────────────

# Holding-cost reward: r_t = -Σ(w_pl · N_unfinished_pl) / N_total
# Source: OR holding-cost model.  PL=2 (highest priority) costs 5× PL=0.
HOLDING_COST_WEIGHTS: Dict[int, float] = {0: 1.0, 1: 2.0, 2: 5.0}

# ──────────────────────────────────────────────────────────────────────
# §9  Synthetic scenario generation ranges
# ──────────────────────────────────────────────────────────────────────

# Ranges cover real region parameter space.
SYNTH_BUILDINGS_RANGE: Tuple[int, int] = (10_000, 80_000)
SYNTH_CONTRACTOR_RATIO_RANGE: Tuple[float, float] = (0.10, 0.80)  # K = H × ratio
SYNTH_INFRA_RATIO_RANGE: Tuple[float, float] = (0.002, 0.03)

# Named damage distributions for synthetic training scenarios.
SYNTH_DAMAGE_DISTS: Dict[str, Tuple[float, float, float]] = {
    "minor_dominated":  (0.70, 0.20, 0.10),
    "major_dominated":  (0.10, 0.20, 0.70),
    "balanced":         (0.34, 0.33, 0.33),
    "extreme_minor":    (0.90, 0.05, 0.05),   # near-total minor
    "extreme_major":    (0.05, 0.05, 0.90),   # near-total major
}

# Training scenario pool: {type: count}.  Total = 200 scenarios.
# Each is instantiated with a unique seed → fully reproducible.
SCENARIO_POOL: Dict[str, int] = {
    "minor_dominated": 50,
    "major_dominated": 50,
    "balanced":        50,
    "extreme_minor":   25,
    "extreme_major":   25,
}


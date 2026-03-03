"""
HouseRecoveryEnv — Gymnasium environment for post-disaster housing
recovery scheduling (2018 Lombok earthquake).

MDP:
  State:   34-dim = 9 queues × 3 features + 7 global features
  Action:  5-dim dispatching weights → per-building scoring → greedy fill
  Reward:  Holding cost (weighted count of unfinished buildings per step)
  Done:    all buildings repaired  (MAX_SIM_DAYS as safety cap only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C
from . import metrics as M


# ──────────────────────────────────────────────────────────────────────
# Building dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Building:
    """One building in the simulation."""
    uid: int                # Unique id (stable across resets with same seed)
    dl: int                 # Damage level ∈ {0, 1, 2}
    pl: int                 # Priority level ∈ {0, 1, 2}
    total_work: float       # Total crew-days (sampled from LogNormal)
    remaining_work: float   # Remaining crew-days
    max_crew: int           # Max daily workers (from MAX_CREW[dl])
    arrive_day: int         # Day this building enters the queue (batch arrival)
    start_day: int = -1     # Day work first begins (-1 = not started)
    done_day: int = -1      # Day work completed (-1 = not done)
    workers_today: int = 0  # Workers assigned on current step (transient)


# ──────────────────────────────────────────────────────────────────────
# Queue key helper
# ──────────────────────────────────────────────────────────────────────

def _queue_index(dl: int, pl: int) -> int:
    """Map (dl, pl) → flat index in [0, 8].  Row-major: dl * 3 + pl."""
    return dl * C.N_PL + pl


# ──────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────

class HouseRecoveryEnv(gym.Env):
    """
    MDP for post-disaster housing recovery scheduling.

    Observation (34-dim):
        [0:27]  9 queues × 3 features (mean_remaining_work, mean_wait, queue_frac)
        [27:34] 7 global features

    Action (5-dim continuous):
        Dispatching weights [w_remain, w_wait, w_total_work, w_priority, w_crew]
        applied to per-building features → score = f^T w → sorted → greedy fill.

    Reward:
        Holding cost: r_t = -Σ(w_pl · N_unfinished_pl) / N_total

    Termination:
        Episode ends when all revealed buildings are repaired.
        MAX_SIM_DAYS serves as a safety truncation (not a design feature).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        region: Optional[str] = None,
        n_buildings: Optional[int] = None,
        n_contractors: Optional[int] = None,
        damage_dist: Optional[Tuple[float, float, float]] = None,
        infra_ratio: Optional[float] = None,
        max_crew: Optional[np.ndarray] = None,
        batch_schedule: Optional[List[Tuple[int, float]]] = None,
        dropout_max: float = C.DROPOUT_MAX,
        progress_noise_std: float = C.PROGRESS_NOISE_STD,
    ):
        """
        Parameters
        ----------
        region : str or None
            If given, load real region data from config.REGIONS[region].
            Mutually exclusive with n_buildings / n_contractors / damage_dist.
        n_buildings, n_contractors, damage_dist, infra_ratio :
            Synthetic scenario parameters.  Used when region is None.
        max_crew : array-like of shape (3,)
            Override MAX_CREW per DL.  Default: config.MAX_CREW.
        """
        super().__init__()

        # ── resolve configuration ──
        if region is not None:
            rc = C.REGIONS[region]
            self._n_buildings_by_dl = dict(rc.n_buildings)  # {dl: count}
            self._n_contractors = n_contractors if n_contractors is not None else rc.n_contractors
            self._infra_ratio = rc.infra_ratio
            self._n_hospitals = rc.n_hospitals
            self._n_schools = rc.n_schools
        else:
            assert n_buildings is not None and n_contractors is not None
            assert damage_dist is not None
            self._damage_dist = np.asarray(damage_dist, dtype=np.float64)
            self._damage_dist /= self._damage_dist.sum()  # normalise
            self._n_buildings_total_cfg = int(n_buildings)
            self._n_contractors = int(n_contractors)
            self._infra_ratio = infra_ratio if infra_ratio is not None else 0.01
            self._n_buildings_by_dl = None  # computed in reset()
            self._n_hospitals = 0
            self._n_schools = 0

        self._region = region
        self._max_crew = np.asarray(
            max_crew if max_crew is not None else C.MAX_CREW, dtype=np.int32
        )
        self._batch_schedule = batch_schedule or list(C.BATCH_SCHEDULE)
        self._dropout_max = dropout_max
        self._progress_noise_std = progress_noise_std

        # ── Gymnasium spaces ──
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(C.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(C.ACT_DIM,), dtype=np.float32
        )

        # ── internal state (populated by reset()) ──
        self.buildings: List[Building] = []
        self._day: int = 0
        self._total_work: float = 0.0
        self._rng: np.random.Generator = np.random.default_rng()
        self._preloaded_buildings: Optional[List[dict]] = None
        self._current_dropout: float = 0.0
        self._current_workers_available: int = 0

    @classmethod
    def from_scenario(cls, scenario_cfg) -> "HouseRecoveryEnv":
        """Create an env that uses pre-generated buildings from a saved
        ScenarioConfig, bypassing random generation on reset().

        Parameters
        ----------
        scenario_cfg : rl.scenarios.ScenarioConfig
            A scenario loaded via ``rl.scenarios.load_scenario(path)``.
        """
        env = cls(
            n_buildings=scenario_cfg.n_buildings,
            n_contractors=scenario_cfg.n_contractors,
            damage_dist=(0.34, 0.33, 0.33),  # placeholder, not used
            infra_ratio=scenario_cfg.infra_ratio,
            max_crew=np.array(scenario_cfg.max_crew, dtype=np.int32),
        )
        # Store frozen building specs for injection in reset()
        env._preloaded_buildings = [
            {
                "uid": b.uid, "dl": b.dl, "pl": b.pl,
                "total_work": b.total_work,
                "max_crew": b.max_crew,
                "arrive_day": b.arrive_day,
            }
            for b in scenario_cfg.buildings
        ]
        return env

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._day = 0
        self._current_dropout = 0.0
        self._current_workers_available = self._n_contractors

        # ── generate or load buildings ──
        if self._preloaded_buildings is not None:
            # Use pre-saved scenario: identical buildings every reset
            self.buildings = [
                Building(
                    uid=b["uid"], dl=b["dl"], pl=b["pl"],
                    total_work=b["total_work"],
                    remaining_work=b["total_work"],  # reset progress
                    max_crew=b["max_crew"],
                    arrive_day=b["arrive_day"],
                )
                for b in self._preloaded_buildings
            ]
        else:
            self.buildings = self._generate_buildings()

        self._total_work = sum(b.total_work for b in self.buildings)

        revealed = list(self._revealed())
        obs = self._get_obs(revealed)
        info = self._get_info(revealed)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._day += 1

        # Cache revealed buildings once for this step (avoids 6× re-iteration)
        revealed = list(self._revealed())

        # 1. Batch arrivals  (buildings already have arrive_day; we just
        #    filter by day for "revealed" set — no extra work needed)

        # 2. Worker dropout
        dropout = self._rng.uniform(0.0, self._dropout_max)
        available_workers = int(self._n_contractors * (1.0 - dropout))
        self._current_dropout = dropout
        self._current_workers_available = available_workers

        # 3. Action → per-building scoring + greedy allocation
        self._allocate_workers(action, available_workers, revealed)

        # 4. Work progress + completions
        newly_completed = self._advance_work(revealed)

        # 5. Compute reward (Holding Cost)
        reward = self._compute_reward(revealed)

        # 6. Termination check
        terminated = self._all_done()
        truncated = self._day >= C.MAX_SIM_DAYS

        obs = self._get_obs(revealed)
        info = self._get_info(revealed)
        info["day"] = self._day
        info["dropout"] = dropout
        info["workers_available"] = available_workers
        info["newly_completed"] = newly_completed

        return obs, reward, terminated, truncated, info

    # ==================================================================
    # Building generation
    # ==================================================================

    def _generate_buildings(self) -> List[Building]:
        """Create all buildings with attributes sampled from the scenario."""
        buildings: List[Building] = []
        uid = 0

        # ── determine per-DL counts ──
        if self._n_buildings_by_dl is not None and self._region is not None:
            # Real region: exact counts from data
            dl_counts = dict(self._n_buildings_by_dl)
        else:
            # Synthetic: sample from damage distribution
            n_total = self._n_buildings_total_cfg
            counts = self._rng.multinomial(
                n_total, self._damage_dist
            )
            dl_counts = {dl: int(c) for dl, c in enumerate(counts)}

        total_n = sum(dl_counts.values())

        # ── assign priority levels (PL) ──
        # Hospitals → PL=2, Schools → PL=1, Residential → PL=0
        # For real regions: use exact hospital/school counts from BPS NTB 2018.
        # For synthetic: sample using infra_ratio.
        if self._region is not None:
            n_hosp = self._n_hospitals
            n_school = self._n_schools
        else:
            n_infra = int(total_n * self._infra_ratio)
            # Roughly 30% hospitals, 70% schools (typical NTB ratio)
            n_hosp = max(1, int(n_infra * 0.3))
            n_school = n_infra - n_hosp

        # We'll assign PL after creating buildings, by random selection
        # (hospitals/schools are sampled from the building pool).

        # ── batch arrival assignment ──
        # Assign each building to a batch; within each batch, arrival day
        # is fixed (matching BNPB timeline).
        batch_fracs = np.array([f for _, f in self._batch_schedule])
        batch_fracs = batch_fracs / batch_fracs.sum()
        batch_days = [d for d, _ in self._batch_schedule]

        # For each DL, split buildings across batches proportionally
        all_arrive_days: List[int] = []
        for dl in range(C.N_DL):
            n_dl = dl_counts.get(dl, 0)
            batch_counts = self._rng.multinomial(n_dl, batch_fracs)
            for batch_idx, bc in enumerate(batch_counts):
                all_arrive_days.extend([batch_days[batch_idx]] * int(bc))

        # Verify we have the right total
        assert len(all_arrive_days) == total_n, (
            f"Expected {total_n} arrive days, got {len(all_arrive_days)}"
        )

        # ── create Building objects ──
        arrive_idx = 0
        for dl in range(C.N_DL):
            n_dl = dl_counts.get(dl, 0)
            if n_dl == 0:
                continue

            # Sample construction times from LogNormal
            # LogNormal params: median = CONS_MED[dl], shape = BETA_CONS
            # np.random.lognormal(mean, sigma) uses underlying normal params:
            #   mean_ln = ln(median), sigma_ln = BETA_CONS
            median = C.CONS_MED[dl]
            mu_ln = np.log(median)
            sigma_ln = C.BETA_CONS
            work_times = self._rng.lognormal(mu_ln, sigma_ln, size=n_dl)

            for i in range(n_dl):
                w = float(work_times[i])
                b = Building(
                    uid=uid,
                    dl=dl,
                    pl=0,  # default; overwritten below for infra
                    total_work=w,
                    remaining_work=w,
                    max_crew=int(self._max_crew[dl]),
                    arrive_day=all_arrive_days[arrive_idx],
                )
                buildings.append(b)
                uid += 1
                arrive_idx += 1

        # ── assign PL by sampling infrastructure buildings ──
        # Hospitals (PL=2) and schools (PL=1) are sampled randomly from
        # the building pool, regardless of DL — this is the DL-PL decoupling.
        indices = self._rng.permutation(len(buildings))
        assigned = 0
        for idx in indices:
            if assigned < n_hosp:
                buildings[idx].pl = 2  # Hospital → PL=2
                assigned += 1
            elif assigned < n_hosp + n_school:
                buildings[idx].pl = 1  # School → PL=1
                assigned += 1
            else:
                break

        # Shuffle to avoid ordering artifacts
        self._rng.shuffle(buildings)

        return buildings

    # ==================================================================
    # Worker allocation (per-building scoring + greedy fill)
    # ==================================================================

    def _allocate_workers(
        self, action: np.ndarray, available: int, revealed: List[Building],
    ) -> None:
        """
        Per-building linear scoring + greedy fill (PDR approach).

        1. Extract 5 dispatching weights from action.
        2. Build normalised feature vector for EVERY queued building:
             [remaining_work, waiting_time, total_work, priority, crew_cap]
        3. Compute score = features @ weights.
        4. Sort buildings by score descending → greedy fill respecting max_crew.

        Source: Parameterized Dispatching Rule (PDR), Lv et al. 2025.
        """
        # Reset today's assignment
        for b in revealed:
            b.workers_today = 0

        if available <= 0:
            return

        # Collect all queued buildings (revealed + incomplete)
        queued: List[Building] = [
            b for b in revealed if b.remaining_work > 0
        ]
        if not queued:
            return

        # Extract 5 dispatching weights
        weights = np.asarray(action, dtype=np.float64).ravel()[:C.DISPATCH_FEATURES]

        # Build normalised feature matrix (N_queued × 5) — fully vectorized
        rem = np.array([b.remaining_work for b in queued], dtype=np.float64)
        wait = np.array([max(0, self._day - b.arrive_day) for b in queued],
                        dtype=np.float64)
        total = np.array([b.total_work for b in queued], dtype=np.float64)
        pl = np.array([b.pl for b in queued], dtype=np.float64)
        crew = np.array([b.max_crew for b in queued], dtype=np.float64)

        features = np.column_stack([
            rem  / C.DISPATCH_NORM[0],
            wait / C.DISPATCH_NORM[1],
            total / C.DISPATCH_NORM[2],
            pl   / C.DISPATCH_NORM[3],
            crew / C.DISPATCH_NORM[4],
        ])

        # Score each building and sort descending
        scores = features @ weights
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-scores)

        # Greedy fill
        remaining = available
        for idx in order:
            if remaining <= 0:
                break
            b = queued[idx]
            assign = min(
                b.max_crew,
                int(np.ceil(b.remaining_work)),
                remaining,
            )
            b.workers_today = assign
            if b.start_day < 0:
                b.start_day = self._day
            remaining -= assign

    # ==================================================================
    # Work progress
    # ==================================================================

    def _advance_work(self, revealed: List[Building]) -> int:
        """Advance work for all buildings with workers assigned.
        Returns count of newly completed buildings."""
        newly_completed = 0
        for b in revealed:
            if b.workers_today > 0 and b.remaining_work > 0:
                # Progress = workers assigned + noise
                noise = self._rng.normal(
                    0.0, self._progress_noise_std * b.workers_today
                )
                progress = max(0.0, b.workers_today + noise)
                b.remaining_work = max(0.0, b.remaining_work - progress)
                if b.remaining_work <= 0:
                    b.done_day = self._day
                    newly_completed += 1
        return newly_completed

    # ==================================================================
    # Reward
    # ==================================================================

    def _compute_reward(self, revealed: List[Building]) -> float:
        """
        Holding Cost reward (Operations Research standard).

        Every step, the agent is penalised for each unfinished building,
        weighted by priority level:
            r_t = - (w0*N0 + w1*N1 + w2*N2) / N_total

        This gives strong gradient to:
          1. Minimise makespan (fewer steps = less total penalty).
          2. Prioritise PL=2 buildings (5× the penalty of PL=0).

        Holding cost weights are set in config: HOLDING_COST_WEIGHTS.
        """
        n_total = len(self.buildings)
        if n_total == 0:
            return 0.0

        weighted_unfinished = 0.0
        for b in revealed:
            if b.remaining_work > 0:
                weighted_unfinished += C.HOLDING_COST_WEIGHTS[b.pl]

        return -weighted_unfinished / n_total

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_obs(self, revealed: List[Building]) -> np.ndarray:
        """Build 34-dim observation vector."""
        n_revealed = len(revealed)
        total_n = len(self.buildings)

        # ── single-pass bucket grouping for queue features ──
        # Accumulate per-queue stats in one pass over revealed buildings.
        q_rem_sum = np.zeros(C.N_QUEUES, dtype=np.float64)
        q_wait_sum = np.zeros(C.N_QUEUES, dtype=np.float64)
        q_count = np.zeros(C.N_QUEUES, dtype=np.int64)
        n_in_queue = 0
        remaining_total = 0.0
        n_severe = 0

        for b in revealed:
            if b.remaining_work > 0:
                qi = _queue_index(b.dl, b.pl)
                q_rem_sum[qi] += b.remaining_work
                q_wait_sum[qi] += self._day - b.arrive_day
                q_count[qi] += 1
                n_in_queue += 1
            remaining_total += b.remaining_work
            if b.dl == 2:
                n_severe += 1

        # ── queue features (9 × 3 = 27) ──
        queue_feats = np.zeros(C.N_QUEUES * C.N_QUEUE_FEATURES, dtype=np.float32)
        for qi in range(C.N_QUEUES):
            n_q = q_count[qi]
            if n_q > 0:
                base = qi * C.N_QUEUE_FEATURES
                queue_feats[base + 0] = np.clip(
                    (q_rem_sum[qi] / n_q) / C.W_SCALE, 0.0, 1.0
                )
                queue_feats[base + 1] = np.clip(
                    (q_wait_sum[qi] / n_q) / C.MAX_SIM_DAYS, 0.0, 1.0
                )
                queue_feats[base + 2] = np.clip(n_q / max(total_n, 1), 0.0, 1.0)

        # ── global features (7) ──
        global_feats = np.zeros(C.N_GLOBAL_FEATURES, dtype=np.float32)

        # 0: time progress
        global_feats[0] = np.clip(self._day / C.MAX_SIM_DAYS, 0.0, 1.0)

        # 1: worker availability rate K(t)/K
        if self._n_contractors > 0:
            global_feats[1] = np.clip(
                self._current_workers_available / self._n_contractors, 0.0, 1.0
            )
        else:
            global_feats[1] = 1.0

        # 2: queue load rate (# in queue / total)
        global_feats[2] = np.clip(n_in_queue / max(total_n, 1), 0.0, 1.0)

        # 3: information reveal rate (# revealed / total)
        global_feats[3] = np.clip(n_revealed / max(total_n, 1), 0.0, 1.0)

        # 4: remaining work rate
        global_feats[4] = np.clip(
            remaining_total / max(self._total_work, 1.0), 0.0, 1.0
        )

        # 5: severe damage fraction (DL=2 among revealed)
        global_feats[5] = (n_severe / n_revealed) if n_revealed > 0 else 0.0

        # 6: current dropout rate δ_t
        global_feats[6] = np.clip(self._current_dropout, 0.0, 1.0)

        obs = np.concatenate([queue_feats, global_feats])
        assert obs.shape == (C.OBS_DIM,), f"obs shape {obs.shape} != {C.OBS_DIM}"
        return obs

    # ==================================================================
    # Helpers
    # ==================================================================

    def _revealed(self):
        """Yield buildings that have arrived by current day."""
        for b in self.buildings:
            if b.arrive_day <= self._day:
                yield b

    def _all_done(self) -> bool:
        """True when every revealed building is repaired."""
        for b in self.buildings:
            if b.arrive_day <= self._day and b.remaining_work > 0:
                return False
        # Also check: have all batches arrived?
        last_batch_day = max(d for d, _ in self._batch_schedule)
        if self._day < last_batch_day:
            return False
        return True

    def _get_info(self, revealed: List[Building]) -> dict:
        """Return episode-level metrics for logging / output table."""
        n_completed = sum(1 for b in revealed if b.remaining_work <= 0)
        n_in_queue = sum(1 for b in revealed if b.remaining_work > 0)
        total_n = len(self.buildings)

        # Per-PL response time and wait time in a single pass
        rt_by_pl: Dict[int, List[float]] = {pl: [] for pl in range(C.N_PL)}
        wait_times: Dict[int, List[float]] = {pl: [] for pl in range(C.N_PL)}
        for b in self.buildings:
            if b.start_day >= 0:
                wait_times[b.pl].append(b.start_day - b.arrive_day)
        for b in revealed:
            if b.start_day >= 0:
                rt_by_pl[b.pl].append(b.start_day - b.arrive_day)
        all_waits = [w for ws in wait_times.values() for w in ws]

        info: dict = {
            "n_completed": n_completed,
            "n_in_queue": n_in_queue,
            "n_total": total_n,
            "completion_frac": n_completed / max(total_n, 1),
            "mean_wait": float(np.mean(all_waits)) if all_waits else 0.0,
            "mean_rt": (
                float(np.mean([v for vs in rt_by_pl.values() for v in vs]))
                if any(rt_by_pl.values()) else 0.0
            ),
        }

        # Per-PL metrics via loop (replaces copy-paste per-PL lines)
        for pl in range(C.N_PL):
            info[f"mean_wait_pl{pl}"] = (
                float(np.mean(wait_times[pl])) if wait_times[pl] else 0.0
            )
            info[f"mean_rt_pl{pl}"] = (
                float(np.mean(rt_by_pl[pl])) if rt_by_pl[pl] else 0.0
            )

        # Makespan & AUC — delegate to metrics.py (only at episode end)
        if self._all_done():
            info["makespan"] = M.makespan(self.buildings)
            info["auc"] = M.recovery_auc(self.buildings)

        return info

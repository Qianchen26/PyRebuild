"""
Automated tests for HouseRecoveryEnv.

Covers:
  1.  test_gymnasium_api       — env_checker passes
  2.  test_obs_shape           — (34,)
  3.  test_action_shape        — (5,)
  4.  test_obs_bounds          — all in [0, 1]
  5.  test_episode_terminates  — episode ends (all buildings repaired)
  6.  test_batch_arrivals      — buildings arrive at correct days
  7.  test_all_workers_used    — greedy allocation uses all workers
  8.  test_reward_negative     — holding cost reward is always ≤ 0
  9.  test_baselines_runnable  — each baseline completes an episode
  10. test_real_region_config  — each of 7 regions loads
  11. test_scenario_fairness   — same seed → identical buildings across resets
"""

from __future__ import annotations

import numpy as np
import pytest

from code import config as C
from code.baselines import ALL_BASELINES
from code.env import HouseRecoveryEnv, _queue_index


# ── Small-scale env for fast tests ──────────────────────────────────

def _small_env(**kwargs) -> HouseRecoveryEnv:
    """Create a small synthetic env for fast testing."""
    defaults = dict(
        n_buildings=200,
        n_contractors=500,
        damage_dist=(0.4, 0.3, 0.3),
        infra_ratio=0.05,
        dropout_max=0.0,        # deterministic workers for tests
        progress_noise_std=0.0, # deterministic progress for tests
    )
    defaults.update(kwargs)
    return HouseRecoveryEnv(**defaults)


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════

class TestGymnasiumAPI:
    """Test 1: Gymnasium env_checker passes."""

    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env
        env = _small_env()
        # check_env runs reset + multiple steps, checks spaces, etc.
        check_env(env, skip_render_check=True)


class TestObservation:
    """Tests 2-4: observation shape and bounds."""

    def test_obs_shape(self):
        env = _small_env()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (C.OBS_DIM,), f"Expected ({C.OBS_DIM},), got {obs.shape}"

    def test_action_shape(self):
        env = _small_env()
        assert env.action_space.shape == (C.ACT_DIM,)

    def test_obs_bounds(self):
        env = _small_env()
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            assert np.all(obs >= 0.0), f"obs has negative values: {obs[obs < 0]}"
            assert np.all(obs <= 1.0), f"obs has values > 1: {obs[obs > 1]}"
            if term or trunc:
                break


class TestEpisodeDynamics:
    """Tests 5-7: episode termination, batch arrivals, worker allocation."""

    def test_episode_terminates(self):
        """Episode must terminate (all buildings repaired)."""
        env = _small_env(n_buildings=50, n_contractors=200)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < C.MAX_SIM_DAYS:
            action = np.ones(C.ACT_DIM, dtype=np.float32)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1
        assert done, f"Episode did not terminate within {C.MAX_SIM_DAYS} steps"
        assert steps < C.MAX_SIM_DAYS, "Hit safety cap — buildings should finish sooner"

    def test_batch_arrivals(self):
        """Buildings arrive at the configured batch days."""
        env = _small_env(n_buildings=100, n_contractors=50)
        env.reset(seed=7)

        batch_days = set(d for d, _ in C.BATCH_SCHEDULE)
        arrive_days = set(b.arrive_day for b in env.buildings)

        # All arrive_days should be in the batch schedule
        assert arrive_days == batch_days, (
            f"Arrive days {arrive_days} != batch days {batch_days}"
        )

        # Check proportions roughly match schedule fracs
        total = len(env.buildings)
        for day, expected_frac in C.BATCH_SCHEDULE:
            actual_count = sum(1 for b in env.buildings if b.arrive_day == day)
            actual_frac = actual_count / total
            assert abs(actual_frac - expected_frac) < 0.15, (
                f"Day {day}: expected ~{expected_frac:.0%}, got {actual_frac:.0%}"
            )

    def test_all_workers_used(self):
        """Greedy allocation should use all available workers
        (when there are enough buildings to absorb them)."""
        env = _small_env(
            n_buildings=500,
            n_contractors=100,
            dropout_max=0.0,  # no dropout → exact 100 workers
        )
        obs, _ = env.reset(seed=1)

        # Uniform action — all queues equal priority
        action = np.ones(C.ACT_DIM, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        workers_assigned = sum(b.workers_today for b in env._revealed())
        # Should use all 100 workers (plenty of buildings to absorb)
        assert workers_assigned == 100, (
            f"Expected 100 workers used, got {workers_assigned}"
        )


class TestReward:
    """Test 8: holding-cost reward is negative while buildings remain."""

    def test_reward_negative_holding_cost(self):
        env = _small_env(n_buildings=50, n_contractors=100)
        obs, _ = env.reset(seed=0)
        action = np.ones(C.ACT_DIM, dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        # Holding cost: negative while unfinished buildings remain
        assert reward < 0, f"Expected negative reward, got {reward}"


class TestBaselines:
    """Test 9: each baseline completes an episode without error."""

    @pytest.mark.parametrize("name", list(ALL_BASELINES.keys()))
    def test_baselines_runnable(self, name):
        env = _small_env(n_buildings=50, n_contractors=100)
        policy_fn = ALL_BASELINES[name]
        obs, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < C.MAX_SIM_DAYS:
            action = policy_fn(obs)
            assert action.shape == (C.ACT_DIM,), f"{name} returned wrong shape"
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1
        assert done, f"{name} did not complete episode"


class TestRegionConfigs:
    """Test 10: all 7 real regions load valid configurations."""

    @pytest.mark.parametrize("region", list(C.REGIONS.keys()))
    def test_real_region_config(self, region):
        env = HouseRecoveryEnv(region=region)
        obs, info = env.reset(seed=0)
        assert obs.shape == (C.OBS_DIM,)
        assert info["n_total"] > 0
        assert env._n_contractors > 0


class TestScenarioFairness:
    """Test 11: same seed → identical buildings across resets."""

    def test_scenario_fairness(self):
        env = _small_env(n_buildings=100, n_contractors=200)

        # Reset twice with same seed
        env.reset(seed=123)
        buildings_1 = [
            (b.uid, b.dl, b.pl, round(b.total_work, 6), b.arrive_day)
            for b in sorted(env.buildings, key=lambda b: b.uid)
        ]

        env.reset(seed=123)
        buildings_2 = [
            (b.uid, b.dl, b.pl, round(b.total_work, 6), b.arrive_day)
            for b in sorted(env.buildings, key=lambda b: b.uid)
        ]

        assert buildings_1 == buildings_2, (
            "Same seed should produce identical buildings across resets"
        )


class TestCrossPolicyFairness:
    """Test 12: DIFFERENT policies with same env seed see identical buildings.

    This is the Step 0 scenario fairness test from the verification plan:
    fixed seed → each policy → verify buildings (uid, DL, PL, work, arrive_day)
    are identical."""

    def test_cross_policy_same_buildings(self):
        env = _small_env(n_buildings=200, n_contractors=300)
        seed = 42

        fingerprints = {}
        for policy_name in ALL_BASELINES:
            env.reset(seed=seed)
            fp = tuple(sorted(
                (b.uid, b.dl, b.pl, round(b.total_work, 6), b.arrive_day)
                for b in env.buildings
            ))
            fingerprints[policy_name] = fp

        ref_name = list(fingerprints.keys())[0]
        ref_fp = fingerprints[ref_name]
        for name, fp in fingerprints.items():
            assert fp == ref_fp, (
                f"{name} buildings differ from {ref_name} with same seed"
            )


class TestBaselineOrdering:
    """Test 13: each baseline produces correct dispatching weights.

    Feature order: [remain_work, wait_time, total_work, priority, crew_cap].
    With per-building scoring, the baselines emit constant weight vectors.
    We verify the weight vector matches the scheduling rule."""

    def test_ljf_weights(self):
        """LJF should set positive weight on remaining_work (index 0)."""
        obs = np.zeros(C.OBS_DIM, dtype=np.float32)
        weights = ALL_BASELINES["LJF"](obs)
        assert weights[0] > 0, "LJF must weight remaining_work positively"
        assert all(weights[i] == 0 for i in [1, 2, 3, 4]), "LJF: only w_remain should be nonzero"

    def test_sjf_weights(self):
        """SJF should set negative weight on remaining_work (index 0)."""
        obs = np.zeros(C.OBS_DIM, dtype=np.float32)
        weights = ALL_BASELINES["SJF"](obs)
        assert weights[0] < 0, "SJF must weight remaining_work negatively"
        assert all(weights[i] == 0 for i in [1, 2, 3, 4]), "SJF: only w_remain should be nonzero"

    def test_strict_priority_weights(self):
        """StrictPriority should set positive weight on priority (index 3)."""
        obs = np.zeros(C.OBS_DIM, dtype=np.float32)
        weights = ALL_BASELINES["StrictPriority"](obs)
        assert weights[3] > 0, "SP must weight priority positively"
        assert all(weights[i] == 0 for i in [0, 1, 2, 4]), "SP: only w_priority should be nonzero"

    def test_fifo_weights(self):
        """FIFO should set positive weight on waiting_time (index 1)."""
        obs = np.zeros(C.OBS_DIM, dtype=np.float32)
        weights = ALL_BASELINES["FIFO"](obs)
        assert weights[1] > 0, "FIFO must weight wait_time positively"
        assert all(weights[i] == 0 for i in [0, 2, 3, 4]), "FIFO: only w_wait should be nonzero"



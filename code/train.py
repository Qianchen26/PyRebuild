"""
SB3 PPO training for HouseRecoveryEnv with domain randomization.

Training protocol:
    200 scenarios across 5 categories (seed-controlled, fully reproducible).
    Each env randomly samples a scenario from the pool on every reset().
    Monitor wrapper ensures ep_rew_mean is logged to tensorboard.

Usage:
    python -m rl.train --total-timesteps 1000000
    python -m rl.train --total-timesteps 2000000 --n-envs 16 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import config as C
from .env import HouseRecoveryEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "output" / "models"


# ─────────────────────────────────────────────────────────────────
# Scenario pool construction
# ─────────────────────────────────────────────────────────────────

def build_scenario_pool(
    base_seed: int = 0,
) -> List[Tuple[str, int, np.ndarray, int, int, float]]:
    """Build the 200-scenario pool.

    Returns list of (scenario_type, seed, damage_dist,
                     n_buildings, n_contractors, infra_ratio)
    tuples.  Each is deterministic given base_seed + index.
    """
    pool = []
    idx = 0
    for stype, count in C.SCENARIO_POOL.items():
        damage_dist = np.array(
            C.SYNTH_DAMAGE_DISTS[stype], dtype=np.float64
        )
        for i in range(count):
            seed = base_seed + idx
            rng = np.random.default_rng(seed)
            n_buildings = int(rng.integers(*C.SYNTH_BUILDINGS_RANGE))
            contractor_ratio = float(rng.uniform(*C.SYNTH_CONTRACTOR_RATIO_RANGE))
            n_contractors = max(1, int(round(n_buildings * contractor_ratio)))
            infra_ratio = float(rng.uniform(*C.SYNTH_INFRA_RATIO_RANGE))
            pool.append((stype, seed, damage_dist, n_buildings,
                         n_contractors, infra_ratio))
            idx += 1
    return pool


# ─────────────────────────────────────────────────────────────────
# Domain-randomization env factory
# ─────────────────────────────────────────────────────────────────

class DomainRandomizationEnv(HouseRecoveryEnv):
    """Wrapper that randomly samples a scenario from the pool on
    each reset(), providing maximum training diversity.

    The env is fully reproducible: pool construction is seeded,
    and scenario selection per reset is seeded by the pool RNG.
    """

    def __init__(self, scenario_pool, env_seed: int = 0):
        # Initialize with dummy params; real params set in reset()
        first = scenario_pool[0]
        super().__init__(
            n_buildings=first[3],
            n_contractors=first[4],
            damage_dist=tuple(first[2]),
            infra_ratio=first[5],
        )
        self._scenario_pool = scenario_pool
        self._pool_rng = np.random.default_rng(env_seed)

    def reset(self, *, seed=None, options=None):
        # Pick a random scenario from the pool
        idx = self._pool_rng.integers(0, len(self._scenario_pool))
        stype, scen_seed, damage_dist, n_bld, n_ctr, infra = \
            self._scenario_pool[idx]

        # Reconfigure env params for this scenario
        self._damage_dist = damage_dist / damage_dist.sum()
        self._n_buildings_total_cfg = n_bld
        self._n_contractors = n_ctr
        self._infra_ratio = infra
        self._n_buildings_by_dl = None  # force synthetic generation
        self._region = None

        # Use the scenario-specific seed for building generation
        return super().reset(seed=scen_seed, options=options)


def make_dr_env(scenario_pool, env_idx: int):
    """Factory for a domain-randomization env."""
    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        env = DomainRandomizationEnv(
            scenario_pool, env_seed=env_idx * 1000
        )
        return Monitor(env)
    return _thunk


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        raise ImportError(
            "stable-baselines3 required. Install: pip install stable-baselines3"
        )

    parser = argparse.ArgumentParser(description="Train PPO on HouseRecoveryEnv")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=C.N_ENVS)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for scenario pool + PPO")
    parser.add_argument("--tag", type=str, default="dr_v1")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"ppo_seed{args.seed}_{args.tag}"
    save_path = MODEL_DIR / run_name

    # ── Build scenario pool ──
    pool = build_scenario_pool(base_seed=args.seed)
    type_counts = {}
    for stype, *_ in pool:
        type_counts[stype] = type_counts.get(stype, 0) + 1
    print(f"Scenario pool: {len(pool)} scenarios")
    for stype, cnt in type_counts.items():
        print(f"  {stype:20s}: {cnt}")

    # ── Create vectorised envs with Monitor ──
    env_fns = [make_dr_env(pool, i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # ── VecNormalize: auto-normalise obs & reward ──
    from stable_baselines3.common.vec_env import VecNormalize
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=C.GAMMA,
    )

    print(f"\nTraining PPO | {args.total_timesteps:,} steps | {args.n_envs} envs")
    print(f"Seed: {args.seed} | Save: {save_path}")
    print(f"VecNormalize: obs=True, reward=True")

    # ── Linear learning rate schedule ──
    def lr_schedule(progress_remaining: float) -> float:
        return C.LR_END + (C.LR_START - C.LR_END) * progress_remaining

    # ── PPO model ──
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=C.GAMMA,
        gae_lambda=C.GAE_LAMBDA,
        clip_range=C.CLIP_RANGE,
        ent_coef=0.01,
        policy_kwargs={"net_arch": C.NET_ARCH},
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(PROJECT_ROOT / "output" / "tb_logs"),
    )

    # ── Callbacks ──
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total_timesteps // 10, 1),
        save_path=str(save_path / "checkpoints"),
        name_prefix="ppo",
    )

    # ── Train ──
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb],
        progress_bar=True,
    )

    # ── Save final model + VecNormalize stats ──
    final_path = str(save_path / "final_model")
    model.save(final_path)
    vec_env.save(str(save_path / "vecnormalize.pkl"))
    print(f"\nFinal model saved → {final_path}")
    print(f"VecNormalize saved → {save_path / 'vecnormalize.pkl'}")

    vec_env.close()


if __name__ == "__main__":
    main()

"""
evaluate.py — Run baselines (+ optional PPO) on real-region scenarios.

Supports multi-seed evaluation for statistical robustness.

Usage:
    # Baselines only, single seed:
    python -m rl.evaluate --tag test_eval

    # Full evaluation: 7 regions × all policies × 10 seeds:
    python -m rl.evaluate --tag full_eval --n-seeds 10

    # With PPO model:
    python -m rl.evaluate --tag ppo_eval --ppo-path output/models/ppo_final.zip --n-seeds 10

    # With PPO model + VecNormalize stats:
    python -m rl.evaluate --tag ppo_eval \\
        --ppo-path output/models/ppo_final.zip \\
        --vecnorm-path output/models/vecnormalize.pkl \\
        --n-seeds 10
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from code import config as C
from code.env import HouseRecoveryEnv
from code.baselines import ALL_BASELINES, seed_random_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# Baseline policies — imported from baselines.py
# Wrap (obs) → (obs, env) signature for evaluate_policy() compatibility.
# ──────────────────────────────────────────────────────────────────────

def _wrap(fn):
    """Adapt baselines.py policy(obs) → evaluate.py policy(obs, env)."""
    def wrapper(obs: np.ndarray, env: HouseRecoveryEnv) -> np.ndarray:
        return fn(obs)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper

BASELINES = {name: _wrap(fn) for name, fn in ALL_BASELINES.items()}


# ──────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────

def evaluate_policy(
    policy_fn,
    region: str,
    seed: int,
    deterministic: bool = True,
    ppo_model=None,
    vecnorm_env=None,
    dropout_max: float = C.DROPOUT_MAX,
    worker_scale: float = 1.0,
) -> dict:
    """Run one episode for a policy on a region, return info dict."""
    # Apply worker scaling to region's n_contractors
    rc = C.REGIONS[region]
    n_contractors_override = max(1, int(rc.n_contractors * worker_scale))
    env = HouseRecoveryEnv(
        region=region, dropout_max=dropout_max,
        n_contractors=n_contractors_override if worker_scale != 1.0 else None,
    )
    obs, _ = env.reset(seed=seed)

    # If using VecNormalize, normalize obs
    if vecnorm_env is not None:
        obs = vecnorm_env.normalize_obs(obs)

    total_reward = 0.0
    step_count = 0

    while True:
        if ppo_model is not None:
            action, _ = ppo_model.predict(obs, deterministic=deterministic)
        else:
            action = policy_fn(obs, env)

        obs, reward, terminated, truncated, info = env.step(action)
        if vecnorm_env is not None:
            obs = vecnorm_env.normalize_obs(obs)
        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    info["total_reward"] = total_reward
    info["steps"] = step_count
    info["region"] = region
    info["seed"] = seed
    return info


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies on real regions")
    parser.add_argument("--tag", type=str, required=True, help="Eval run tag")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds to evaluate (default: 1)")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed (seeds = base_seed .. base_seed + n_seeds - 1)")
    parser.add_argument("--ppo-path", type=str, default=None, help="Path to saved PPO .zip")
    parser.add_argument("--vecnorm-path", type=str, default=None,
                        help="Path to VecNormalize .pkl (required with PPO)")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="Regions to evaluate (default: all 7)")
    parser.add_argument("--dropout-max", type=float, default=None,
                        help="Override dropout_max for sensitivity test")
    parser.add_argument("--worker-scale", type=float, default=1.0,
                        help="Scale n_contractors by this factor (e.g. 0.5 = 50%%)")
    args = parser.parse_args()

    regions = args.regions or list(C.REGIONS.keys())
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    # Load PPO if available
    ppo_model = None
    vecnorm_env = None
    if args.ppo_path:
        from stable_baselines3 import PPO
        ppo_model = PPO.load(args.ppo_path)
        print(f"Loaded PPO model from {args.ppo_path}")

        if args.vecnorm_path:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            # Create a dummy env to load VecNormalize stats
            dummy_env = DummyVecEnv([lambda: HouseRecoveryEnv(region=regions[0])])
            vecnorm_env = VecNormalize.load(args.vecnorm_path, dummy_env)
            vecnorm_env.training = False
            vecnorm_env.norm_reward = False
            print(f"Loaded VecNormalize from {args.vecnorm_path}")

    # Build policy list
    policies = dict(BASELINES)
    if ppo_model is not None:
        policies["PPO"] = None  # sentinel

    # Run evaluation
    results = []
    total_runs = len(regions) * len(policies) * len(seeds)
    run_idx = 0

    for region in regions:
        region_cfg = C.REGIONS[region]
        for policy_name, policy_fn in policies.items():
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {policy_name:8s} x "
                      f"{region_cfg.name_en:20s} x seed={seed} ...",
                      end="", flush=True)

                info = evaluate_policy(
                    policy_fn=policy_fn,
                    region=region,
                    seed=seed,
                    ppo_model=ppo_model if policy_name == "PPO" else None,
                    vecnorm_env=vecnorm_env if policy_name == "PPO" else None,
                    dropout_max=args.dropout_max if args.dropout_max is not None else C.DROPOUT_MAX,
                    worker_scale=args.worker_scale,
                )
                info["policy"] = policy_name

                results.append(info)
                makespan = info.get("makespan", "N/A")
                print(f"  makespan={makespan}  reward={info['total_reward']:.3f}")

    # Save raw results
    out_dir = PROJECT_ROOT / "output" / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {out_path}")

    # Build summary DataFrame
    df = pd.DataFrame(results)

    # Aggregate: mean ± std across seeds for each (policy, region)
    agg_cols = ["makespan", "total_reward", "mean_wait", "mean_wait_pl0",
                "mean_wait_pl1", "mean_wait_pl2", "mean_rt_pl0", "mean_rt_pl1",
                "mean_rt_pl2", "auc"]
    # Only aggregate columns that exist
    agg_cols = [c for c in agg_cols if c in df.columns]
    summary = df.groupby(["policy", "region"])[agg_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    csv_path = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Summary saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*110}")
    print(f"{'Policy':8s} | {'Region':20s} | "
          f"{'Makespan':>12s} | {'Reward':>14s} | "
          f"{'RT PL2':>12s} | {'RT PL0':>12s} | {'AUC':>12s}")
    print(f"{'-'*110}")

    for _, row in summary.iterrows():
        ms = f"{row.get('makespan_mean', 0):.0f}±{row.get('makespan_std', 0):.0f}"
        rw = f"{row.get('total_reward_mean', 0):.1f}±{row.get('total_reward_std', 0):.1f}"
        rt2 = f"{row.get('mean_rt_pl2_mean', 0):.1f}±{row.get('mean_rt_pl2_std', 0):.1f}"
        rt0 = f"{row.get('mean_rt_pl0_mean', 0):.1f}±{row.get('mean_rt_pl0_std', 0):.1f}"
        auc = f"{row.get('auc_mean', 0):.3f}±{row.get('auc_std', 0):.3f}"
        region_name = C.REGIONS[row["region"]].name_en
        print(f"{row['policy']:8s} | {region_name:20s} | "
              f"{ms:>12s} | {rw:>14s} | "
              f"{rt2:>12s} | {rt0:>12s} | {auc:>12s}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()

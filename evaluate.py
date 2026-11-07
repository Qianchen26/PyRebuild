"""Evaluate every policy in the seven Lombok regions with environment seeds 1000-1009.

Usage: OMP_NUM_THREADS=1 python evaluate.py [--workers 14]   (12 policies x 7 regions x 10 seeds, about 1 min)

Policies: the fixed rules SJF, LJF, FIFO, PriorityOnly, the FCR (results/fcr_search), the historical rule
used for calibration, Random (new U(-1, 1) weights every day), and the five trained PPO models.
Writes results/episodes.csv with one row per policy, region and seed: the episode metrics, the
episode-mean weights (w_work ... w_damage) used by the weight figure, and a hash of the initial houses,
which must be the same for every policy in a region and seed.
"""
import argparse
import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from calibrate import historical
from env import HouseRecoveryEnv, region_scenario

HERE = Path(__file__).resolve().parent
SEEDS = range(1000, 1010)
RANDOM_SEED_BASE = 20261017   # Random reseeds every episode: base + 1000 * region index + env seed
WEIGHTS = ["w_work", "w_wait", "w_pri", "w_damage"]
ALT_HC = {"hc_pl_1_1_1": ("pl", [1, 1, 1]), "hc_pl_1_5_25": ("pl", [1, 5, 25]),
          "hc_damage_1_2_5": ("dl", [1, 2, 5])}   # holding cost recomputed under other weights (sensitivity)


def fcr():
    """The FCR weights: the best vector on the confirmation scenarios."""
    top = pd.read_csv(HERE / "results" / "fcr_search" / "confirmation.csv").iloc[0]
    return [float(top[f"w{i}"]) for i in range(4)]


FIXED = {"SJF": [-1, 0, 0, 0], "LJF": [1, 0, 0, 0], "FIFO": [0, 1, 0, 0], "PriorityOnly": [0, 0, 1, 0]}
POLICIES = ["FCR", *FIXED, "Historical", "Random", *(f"PPO{k}" for k in range(5))]


@cache
def load_ppo(k):
    """Trained model k with its frozen observation statistics, acting deterministically."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    run = HERE / "runs" / f"seed{k}"
    model = PPO.load(run / "final_model.zip", device="cpu",
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
    with open(run / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)                         # frozen observation statistics
    return lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]


def make_policy(name, region, seed):
    """A function obs -> weights for one policy."""
    if name in FIXED or name in ("FCR", "Historical"):
        weights = np.array(FIXED.get(name) or (fcr() if name == "FCR" else historical(region)), dtype=np.float32)
        return lambda obs: weights
    if name == "Random":
        rng = np.random.default_rng(RANDOM_SEED_BASE + 1000 * list(C.REGIONS).index(region) + seed)
        return lambda obs: rng.uniform(-1, 1, 4).astype(np.float32)
    return load_ppo(int(name.removeprefix("PPO")))


def run_episode(job, **mechanism):
    """Play one episode and return its metrics.

    job is (policy, region, seed) or (policy, region, seed, scenario); mechanism overrides config.MECHANISM.
    Times are in days. Waits and completions start when a house becomes eligible (end of its delay);
    makespan and the mean recovery day count from the earthquake (2018-08-19).
    - holding_cost: sum over houses of priority weight x (completion day - eligible day), divided by N
    - auc: share of the recovery area (houses x days up to the makespan) that is already recovered
    - mean_wait, pl2_response: eligible -> first workers, all houses and priority-level-2 houses
    - mean_completion, pl2_completion(_p90): eligible -> repaired (mean, and 90th percentile for PL2)
    """
    policy, region, seed = job[:3]
    scenario = job[3] if len(job) > 3 else region_scenario(region, seed)
    act = make_policy(policy, region, seed)
    env = HouseRecoveryEnv(scenario, **mechanism)
    obs, _ = env.reset(seed=seed)
    buildings = hashlib.sha256(b"".join(a.tobytes() for a in (env.dl, env.pl, env.work, env.arrive, env.elig))).hexdigest()[:16]
    total_reward, actions, terminated, truncated = 0.0, [], False, False
    while not (terminated or truncated):
        actions.append(np.clip(act(obs), -1, 1))
        obs, reward, terminated, truncated, _ = env.step(actions[-1])
        total_reward += reward
    if truncated:
        raise RuntimeError(f"episode hit the {C.MAX_DAYS}-day cap: {job[:3]}")
    wait = env.start - env.elig           # eligibility (end of the pre-construction delay) -> first workers
    delay = env.done - env.elig           # eligibility -> repair completed
    pl2 = env.pl == 2
    makespan = env.done.max()
    return {"policy": policy, "region": region, "seed": seed,
            "holding_cost": -total_reward,
            "auc": np.maximum(makespan - env.done, 0).sum() / (env.n * makespan),
            "makespan": makespan,
            "mean_recovery_day": env.done.mean(),    # days after the earthquake, delay included
            "mean_wait": wait.mean(),
            "mean_completion": delay.mean(),
            "pl2_response": wait[pl2].mean(),
            "pl2_completion": delay[pl2].mean(),
            "pl2_completion_p90": np.quantile(delay[pl2].astype(float), 0.9),
            **{f"completion_{x}": delay[env.dl == k].mean() for k, x in enumerate(["minor", "moderate", "major"])},
            **{k: (np.asarray(w)[getattr(env, attr)] * delay).sum() / env.n for k, (attr, w) in ALT_HC.items()},
            **dict(zip(WEIGHTS, np.asarray(actions, dtype=np.float64).mean(axis=0))),
            "buildings": buildings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    jobs = [(p, r, s) for p in POLICIES for r in C.REGIONS for s in SEEDS]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_episode, jobs))
    df = pd.DataFrame(rows).sort_values(["policy", "region", "seed"])
    assert (df.groupby(["region", "seed"]).buildings.nunique() == 1).all(), "initial buildings differ across policies"
    out = HERE / "results" / "episodes.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(rows)} episodes, none truncated -> {out}")


if __name__ == "__main__":
    main()

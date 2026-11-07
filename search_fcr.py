"""Search the Fixed Composite Rule (FCR): one fixed weight vector for the whole episode.

The four weights [work, wait, priority, damage level] are searched jointly in [-1, 1] with no sparsity
penalty. Three differential-evolution runs (population 20, at most 10 generations) minimize the mean
holding cost on 8 synthetic search scenarios; coordinate probes (+-0.05) then test the best vector. The
three run optima, the best probe and the starting rule [-1, 0, 1, 0] (less repair work first plus social
priority; it seeds two of the three runs) are ranked on 32 other synthetic
scenarios, and the best of them becomes the FCR. Each scenario keeps one batch schedule, delay, absence
and noise stream for all vectors, so the vectors are compared under the same random conditions.

Usage: python search_fcr.py [--workers 14]        (about 20 min)
Writes results/fcr_search/{candidates.csv, endpoints.json, confirmation.csv}.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc

import config as C
from env import HouseRecoveryEnv
from train import scenarios

OUT = Path(__file__).resolve().parent / "results" / "fcr_search"
PRIOR = np.array([-1.0, 0.0, 1.0, 0.0])   # starting rule: less repair work first, plus social priority
# Three runs: a Latin-hypercube start, and two starts that mix in perturbations of PRIOR (sd 0.25 and 0.05).
STARTS = [("global", 2026106001, None), ("mixed_025", 2026106002, 0.25), ("mixed_005", 2026106003, 0.05)]


def normalize(vector):
    """Executed action: divided by its largest |weight| (rankings are scale-free), as float32."""
    vector = np.asarray(vector, dtype=float)
    top = np.abs(vector).max()
    return (vector / top if top > 0 else vector).astype(np.float32)


def key(action):
    """Cache key of an executed action."""
    action = action.copy()
    action[action == 0] = 0                        # -0.0 and 0.0 run identically
    return hashlib.sha256(action.tobytes()).hexdigest()


def holding_cost(action, scenario):
    """Undiscounted holding cost of one episode with a fixed action."""
    env = HouseRecoveryEnv(scenario)
    env.reset(seed=scenario.seed)
    hc, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(action)
        hc -= reward
    if truncated:
        raise RuntimeError(f"scenario {scenario.seed} hit the day cap")
    return hc


def mean_holding_cost(vector, cases):
    """Objective of the search: mean holding cost over a list of scenarios."""
    action = normalize(vector)
    return float(np.mean([holding_cost(action, s) for s in cases]))


class CachedMap:
    """DE `workers` hook: runs each distinct executed action once, in parallel, in first-seen order.

    The cache belongs to one objective, so DE's wrapped copy of it (first argument) is ignored."""

    def __init__(self, pool, objective):
        self.pool, self.objective, self.cost, self.action = pool, objective, {}, {}

    def __call__(self, _, vectors):
        actions = [normalize(v) for v in vectors]
        keys = [key(a) for a in actions]
        new = {k: a for k, a in zip(keys, actions) if k not in self.cost}
        self.cost.update(zip(new, self.pool.map(self.objective, new.values())))
        self.action.update(new)
        return [self.cost[k] for k in keys]

    def frame(self):
        """Every vector evaluated so far with its mean holding cost."""
        return pd.DataFrame([{"key": k, **dict(zip(["w0", "w1", "w2", "w3"], self.action[k].tolist())),
                              "mean_holding_cost": c} for k, c in self.cost.items()])


def initial_population(seed, scale):
    """The prior rule, then Latin-hypercube points (and Gaussian perturbations of the prior)."""
    rng = np.random.default_rng(seed)
    rows = [PRIOR[None, :], 2 * qmc.LatinHypercube(d=4, rng=rng).random(19 if scale is None else 9) - 1]
    if scale is not None:
        rows.append(np.clip(rng.normal(PRIOR, scale, size=(10, 4)), -1, 1))
    return np.vstack(rows)


def run_de(objective, mapper, name, seed, scale):
    """One differential-evolution run; returns its best vector and holding cost."""
    history = []

    def plateau(intermediate_result):   # from generation 6: stop if the last 3 improved the best by <= 0.1%
        history.append(float(intermediate_result.fun))
        print(f"{name} generation {intermediate_result.nit}: best HC {history[-1]:.6f}", flush=True)
        return len(history) >= 6 and (history[-4] - history[-1]) / max(abs(history[-4]), 1e-12) <= 0.001

    result = differential_evolution(
        objective, bounds=[(-1, 1)] * 4, strategy="best1bin", maxiter=10,
        init=initial_population(seed, scale), rng=np.random.default_rng(seed), mutation=(0.5, 1.0),
        recombination=0.7, tol=0, atol=0, polish=False, updating="deferred", workers=mapper,
        callback=plateau)
    return {"run": name, "weights": normalize(result.x).tolist(), "holding_cost": float(result.fun),
            "generations": int(result.nit), "evaluations": int(result.nfev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    C.check_calibration()
    OUT.mkdir(parents=True, exist_ok=True)
    search = partial(mean_holding_cost, cases=scenarios(2026104000, [2, 2, 2, 1, 1]))
    confirm = partial(mean_holding_cost, cases=scenarios(2026105000, [8, 8, 8, 4, 4]))
    with ProcessPoolExecutor(args.workers) as pool:
        mapper = CachedMap(pool, search)
        endpoints = [run_de(search, mapper, *start) for start in STARTS]
        best = np.array(min(endpoints, key=lambda e: e["holding_cost"])["weights"])
        probes = [np.clip(best + sign * 0.05 * np.eye(4)[i], -1, 1) for i in range(4) for sign in (-1, 1)]
        best_probe = normalize(probes[int(np.argmin(mapper(search, probes)))])
        mapper.frame().to_csv(OUT / "candidates.csv", index=False)
        print(f"{len(mapper.cost)} distinct vectors evaluated in search and probes")

        finalists = [np.array(e["weights"]) for e in endpoints] + [best_probe, PRIOR]
        final = CachedMap(pool, confirm)
        final(confirm, finalists)
    ranking = final.frame().sort_values(["mean_holding_cost", "key"])
    ranking.to_csv(OUT / "confirmation.csv", index=False)
    (OUT / "endpoints.json").write_text(json.dumps(
        {"runs": endpoints, "best_probe": best_probe.tolist()}, indent=2) + "\n")
    top = ranking.iloc[0]
    print(f"FCR = {final.action[top.key].tolist()}, confirmation HC {top.mean_holding_cost}")


if __name__ == "__main__":
    main()

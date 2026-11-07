"""Tune PPO for a seed whose training raised a red flag (successive halving, one Hyperband bracket).

Usage: python asha.py --seed K [--parallel 4]

Nine PPO settings (train.py's current settings first, then eight sampled from SPACE) train for 1/9 of
the full budget; the best three continue to 1/3 and the best one to the full budget. Each rung starts
from scratch on the seed's own scenario pool. Trials are ranked by the probe holding cost at the end of
their rung, trials with a red flag after those without. A trial folder that already
holds a final model is scored again instead of retrained, so an interrupted search can resume.
The full-budget winner goes through the regular 20% check and final health; if it passes, it is copied
to runs/seedK, and if it fails, asha.py reports the failure and leaves runs/seedK unchanged. Writes runs/asha/seedK/{trials.csv, best.json} and one folder per trial.
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from health import final_health, load, red_flags

HERE = Path(__file__).resolve().parent
SPACE = dict(ent_coef=[0.0, 0.001, 0.005, 0.01], n_steps=[1024, 2048, 4096], batch_size=[64, 128, 256],
             n_epochs=[5, 10], clip_range=[0.1, 0.2, 0.3], gae_lambda=[0.9, 0.95, 0.98])


def sample(rng):
    """One PPO setting: a value from each SPACE list and a log-uniform learning rate in [3e-5, 1e-3]."""
    config = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    config["lr_start"] = float(10 ** rng.uniform(np.log10(3e-5), np.log10(1e-3)))
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in config.items()}


def trial(seed, name, config, budget, root, stop):
    """Train (or reuse) one trial with train.py and return its score and red flags."""
    run = root / name
    code = 0
    if not (run / "final_model.zip").exists():
        (root / f"{name}.json").write_text(json.dumps(config) + "\n")
        cmd = [sys.executable, "-W", "ignore", str(HERE / "train.py"), "--seed", str(seed), "--total-timesteps",
               str(budget), "--hparams", str(root / f"{name}.json"), "--run", str(run)] + ([] if stop else ["--no-stop"])
        code = subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=open(root / f"{name}.log", "w")).returncode
    flags = red_flags(load(run))
    score = -float(pd.read_csv(run / "probe.csv").hc.iloc[-1])
    return {"trial": name, "budget": budget, "exit": code, "score": score, "red_flags": "; ".join(flags), **config}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    root = HERE / "runs" / "asha" / f"seed{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7_000 + args.seed)
    from train import LR_START, PPO_SETTINGS
    current = {k: v for k, v in PPO_SETTINGS.items() if k in SPACE} | {"lr_start": LR_START}
    configs = {"c0": current} | {f"c{i}": sample(rng) for i in range(1, 9)}
    rows, alive = [], list(configs)
    for rung, (budget, keep) in enumerate([(args.total_timesteps // 9, 3), (args.total_timesteps // 3, 1)]):
        with ThreadPoolExecutor(args.parallel) as ex:
            results = list(ex.map(lambda c: trial(args.seed, f"r{rung}_{c}", configs[c], budget, root, False), alive))
        rows += [{**r, "config": c} for r, c in zip(results, alive)]
        ranked = sorted(zip(results, alive), key=lambda rc: (bool(rc[0]["red_flags"]), -rc[0]["score"]))
        alive = [c for _, c in ranked[:keep]]
        pd.DataFrame(rows).to_csv(root / "trials.csv", index=False)
        print(f"rung {rung}: kept {alive}", flush=True)
    best = configs[alive[0]]
    (root / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    final = trial(args.seed, f"r2_{alive[0]}", best, args.total_timesteps, root, True)
    rows.append({**final, "config": alive[0]})
    pd.DataFrame(rows).to_csv(root / "trials.csv", index=False)
    run = root / f"r2_{alive[0]}"
    if final["exit"] != 0 or not (run / "final_model.zip").exists():
        raise SystemExit(f"seed {args.seed}: the tuned run raised red flags again: {final['red_flags']}")
    health = final_health(run)
    print(f"seed {args.seed}: tuned run healthy={health['healthy']} {health['flags']}", flush=True)
    if health["healthy"]:
        target = HERE / "runs" / f"seed{args.seed}"
        if target.exists():
            shutil.move(target, root / "failed_default_run")
        shutil.copytree(run, target)


if __name__ == "__main__":
    main()

"""Sensitivity analysis: do the main conclusions hold when uncertain parameters change?

The main conclusions are the seven checks in CHECKS (C1-C7), written down before the first run.

Each setting changes one or two parameters around the calibrated simulator: the workforce K (x0.5, x1.5),
the days and shares of the information batches, the absence rate, the return probability and the progress
noise. Every policy is evaluated in the seven regions; the trained models are evaluated, not retrained.
A setting starts with seeds 1000-1009 and gets 10 more seeds (up to 30) while any check has a paired difference
with a standard error above a quarter of its size. Holding cost under other cost weights is recomputed from the
main evaluation (evaluate.ALT_HC) and needs no extra runs.

Usage: OMP_NUM_THREADS=1 python sensitivity.py [--workers 14] [--plot-only]   (about 1 h on 14 cores)
Writes results/sensitivity/{episodes.csv, checks.csv, summary.md} and figures/sensitivity_C*.png.
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from env import region_scenario
from evaluate import POLICIES, run_episode
from report import block_ci

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results" / "sensitivity", HERE / "figures"
PPO = [f"PPO{k}" for k in range(5)]
# The conclusions to check, written down before the first sensitivity run: (name, metric, policy A, policy B).
# A conclusion holds in a setting when the paired difference A - B has a 95% interval entirely below 0.
CHECKS = [("C1 HC: FCR < Learned", "holding_cost", "FCR", "Learned"),
          ("C2 HC: Learned < Random", "holding_cost", "Learned", "Random"),
          ("C3 HC: SJF < Historical", "holding_cost", "SJF", "Historical"),
          ("C4 Makespan: LJF < SJF", "makespan", "LJF", "SJF"),
          ("C5 HC: SJF < LJF", "holding_cost", "SJF", "LJF"),
          ("C6 PL2 RT: Learned < SJF", "pl2_response", "Learned", "SJF"),
          ("C7 Mean wait: FCR < Learned", "mean_wait", "FCR", "Learned")]


def settings():
    """The 44 settings: name -> workforce scale, batch schedule and mechanism overrides."""
    days = [d for d, _ in C.BATCHES]
    base = {"scale": 1.0, "batches": C.BATCHES, "mech": {}}
    out = {"base": base}
    for f in (0.5, 1.5):
        out[f"K x{f}"] = {**base, "scale": f}
    for f2, f3, s1 in itertools.product((0.5, 1, 1.5), (0.5, 1, 1.5), (0.1, 0.3, 0.5)):
        if (f2, f3, s1) != (1, 1, 0.3):
            sched = (days[0], round(days[0] + f2 * (days[1] - days[0])), round(days[0] + f3 * (days[2] - days[0])))
            out[f"d2 x{f2}, d3 x{f3}, first {s1}"] = {**base, "batches": tuple(zip(sched, (s1, (1 - s1) / 2, (1 - s1) / 2)))}
    for rho in (0.025, 0.10):
        out[f"rho {rho}"] = {**base, "mech": {"absence_rate": rho}}
    for p in (1.0, 0.77, 0.33):
        out[f"p {p}"] = {**base, "mech": {"return_prob": p}}
    for noise in (0.0, 0.30):
        out[f"noise {noise}"] = {**base, "mech": {"noise": noise}}
    for f, rho in itertools.product((0.5, 1.5), (0.025, 0.10)):
        out[f"K x{f}, rho {rho}"] = {**base, "scale": f, "mech": {"absence_rate": rho}}
    for p, noise in itertools.product((1.0, 0.33), (0.0, 0.30)):
        out[f"p {p}, noise {noise}"] = {**base, "mech": {"return_prob": p, "noise": noise}}
    return out


def job(args):
    """One episode of one policy in one setting."""
    name, s, policy, region, seed = args
    row = run_episode((policy, region, seed, region_scenario(region, seed, s["batches"], s["scale"])), **s["mech"])
    return {"setting": name, **row}


def learned(df):
    """Adds the 5-model mean as policy 'Learned'."""
    mean = df[df.policy.isin(PPO)].groupby(["setting", "region", "seed"], as_index=False).mean(numeric_only=True)
    return pd.concat([df, mean.assign(policy="Learned")])


def check(df):
    """Paired difference A - B and its 95% interval for every check in every setting."""
    rows = []
    for (setting, (name, metric, a, b)) in itertools.product(df.setting.unique(), CHECKS):
        f = df[df.setting == setting].set_index(["policy", "region", "seed"])[metric]
        d = (f.loc[a] - f.loc[b]).rename(metric).reset_index()
        m, lo, hi = block_ci(d, metric)
        blocks = d.groupby("seed")[metric].mean()
        rows.append({"setting": setting, "check": name, "diff": m, "lo": lo, "hi": hi,
                     "se": blocks.std(ddof=1) / np.sqrt(len(blocks)), "n_seeds": len(blocks), "holds": hi < 0})
    return pd.DataFrame(rows)


def run(workers):
    """Run all settings, adding 10 seeds to a setting while any check is too noisy (at most 30 seeds)."""
    S, frames = settings(), []
    with ProcessPoolExecutor(workers) as pool:
        todo = {name: range(1000, 1010) for name in S}
        while todo:
            jobs = [(n, S[n], p, r, s) for n, ss in todo.items() for p in POLICIES for r in C.REGIONS for s in ss]
            frames.append(pd.DataFrame(pool.map(job, jobs, chunksize=4)))
            df = learned(pd.concat(frames))
            df.to_csv(OUT / "episodes.csv", index=False)
            c = check(df[df.setting.isin(todo)])
            noisy = c[(c.se > c["diff"].abs() / 4) & (c.n_seeds < 30)].setting.unique()
            todo = {n: range(1000 + c[c.setting == n].n_seeds.max(), 1010 + c[c.setting == n].n_seeds.max()) for n in noisy}
            print(f"{len(jobs)} episodes; {len(todo)} settings need more seeds", flush=True)


def summarize():
    """Share of settings in which each check holds, the difference table, and the heat maps."""
    df = pd.read_csv(OUT / "episodes.csv")
    c = check(df)
    c.to_csv(OUT / "checks.csv", index=False)
    share = c.groupby("check").holds.mean()
    with open(OUT / "summary.md", "w") as f:
        f.write("# Sensitivity analysis: share of settings where each declared check holds\n\n")
        f.write(share.rename("share of settings").to_frame().to_markdown() + "\n\n")
        f.write("Paired difference A - B (95% interval over seed blocks); a check holds when the upper bound < 0.\n\n")
        f.write(c.pivot(index="setting", columns="check", values="diff").round(3).to_markdown() + "\n\n")
        f.write("Seeds per setting:\n\n" + c.groupby("setting").n_seeds.max().to_frame().to_markdown() + "\n")
    print(share.to_string())
    heatmaps(c)


def heatmaps(c):
    """For each check, the paired difference on the three two-parameter grids."""
    grids = [("K scale", "absence rate", [0.5, 1, 1.5], [0.025, 0.05, 0.10],
              lambda x, y: "base" if (x, y) == (1, 0.05) else f"K x{x}" if y == 0.05 else f"rho {y}" if x == 1 else f"K x{x}, rho {y}"),
             ("return prob p", "progress noise", [1.0, 0.5, 0.33], [0.0, 0.15, 0.30],
              lambda x, y: "base" if (x, y) == (0.5, 0.15) else f"p {x}" if y == 0.15 else f"noise {y}" if x == 0.5 else f"p {x}, noise {y}"),
             ("d2 scale", "d3 scale", [0.5, 1, 1.5], [0.5, 1, 1.5],
              lambda x, y: "base" if (x, y) == (1, 1) else f"d2 x{x}, d3 x{y}, first 0.3")]
    for name, *_ in CHECKS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        sub = c[c.check == name].set_index("setting")
        for ax, (xl, yl, xs, ys, key) in zip(axes, grids):
            z = np.array([[sub.loc[key(x, y), "diff"] for x in xs] for y in ys])
            ok = np.array([[sub.loc[key(x, y), "holds"] for x in xs] for y in ys])
            lim = np.abs(z).max()
            im = ax.imshow(z, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            for i, j in itertools.product(range(len(ys)), range(len(xs))):
                ax.text(j, i, f"{z[i, j]:.2f}{'' if ok[i, j] else ' ×'}", ha="center", va="center", fontsize=9)
            ax.set_xticks(range(len(xs)), xs)
            ax.set_yticks(range(len(ys)), ys)
            ax.set(xlabel=xl, ylabel=yl)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"{name}: paired difference A - B (× = 95% interval does not exclude 0 on the claimed side)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(FIG / f"sensitivity_{name.split(':')[0].replace(' ', '_')}.png", dpi=170)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    if not args.plot_only:
        run(args.workers)
    summarize()


if __name__ == "__main__":
    main()

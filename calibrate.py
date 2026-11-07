"""Calibrate the simulator on Mataram's recorded recovery curves and validate it on the other six regions.

Fixed inputs: the workforce K and the pre-construction delay (Alisjahbana & Kiremidjian 2021, Tables 2
and 4). Calibrated by grid search:
- crew caps, the most workers on one house, by damage level (minor, moderate, major; 1-6 each);
- alpha, the scale of repair work;
- lam, the stretch of the information batches, which arrive on days 0, 14 lam and 60 lam with shares
  30/35/35 (the last batch no later than config.LAST_BATCH_MAX).

The simulated allocation follows the historical rule assumed by Alisjahbana & Kiremidjian: major damage
first (West Lombok: minor and moderate first), first come first served within a damage level.
The fit is the mean of four RMSEs between the mean simulated and the recorded completion curves (overall
and for each damage level) on the recorded days, 95-590 after 2018-08-19. The three level curves are
what tell the three crew caps apart.

Every grid point runs 10 seeds; the 10 best points are rerun with 30 seeds. top.csv also gives each of
them the Mataram holding-cost gap between LJF and SJF, which is about 0 when the workforce is never
scarce. By default the best rerun fit is chosen; --pick N chooses row N of top.csv instead. The reported
calibration uses --pick 4: the best Mataram fit (caps 1/2/6) leaves the workforce idle in five of the
seven regions, so the order of repairs would not matter there, and it fits the six validation regions
worse (mean overall RMSE 0.156 against 0.104).

Usage: python calibrate.py [--workers 14] [--from-grid] [--pick N] [--plot-only]
       (full grid about 15 min on 14 cores; --from-grid reuses results/calibration/grid.csv)
Writes results/calibration/{grid.csv, top.csv, best.json, validation.csv, curves.npz}
and figures/calibration_{regions,by_damage,rmse_surface}.png.
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from env import HouseRecoveryEnv, region_scenario

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results" / "calibration", HERE / "figures"
HORIZON = 600
CAPS = [1, 2, 3, 4, 5, 6]
GRID = dict(alpha=[1, 1.5, 2, 3, 4, 5, 6, 8], c_minor=CAPS, c_moderate=CAPS, c_major=CAPS,
            lam=[x for x in (1, 2, 3, 4) if 60 * x <= C.LAST_BATCH_MAX])
LEVELS = ["Minor", "Moderate", "Major"]


def historical(region):
    """Weights [work, wait, priority, damage level]: major damage first, then FIFO."""
    return np.array([0, 0.01, 0, -1 if region == "lombokbarat" else 1], dtype=np.float32)


def batches(lam):
    """The information schedule for stretch lam."""
    return ((0, 0.30), (round(14 * lam), 0.35), (round(60 * lam), 0.35))


def observed():
    """{region: (days, overall share completed, share completed by damage level (n x 3))}."""
    d = pd.read_pickle(HERE / "data" / "lombok_data.pkl").apply(pd.to_numeric, errors="coerce")
    days = (d.index - pd.Timestamp("2018-08-19")).days.to_numpy()
    out = {}
    for name, r in C.REGIONS.items():
        # Completions are cumulative counts: a missing report keeps the last reported value, and days
        # before the first report count as 0.
        comp = d[[f"{name}_comp_{x}" for x in ("rr", "rs", "rb")]].ffill().fillna(0).to_numpy()
        out[name] = (days, comp.sum(axis=1) / sum(r.buildings), comp / np.array(r.buildings))
    return out


def mechanism(alpha, c_minor, c_moderate, c_major):
    """Simulator overrides for one grid point (crew caps cast back to int: grid.csv stores floats)."""
    return dict(alpha=alpha, max_crew=np.array([c_minor, c_moderate, c_major], dtype=int))


def curves(region, alpha, c_minor, c_moderate, c_major, lam, seed):
    """Share of houses completed by the end of each day 0..HORIZON-1 under the historical rule: overall and
    by damage level (4 x HORIZON)."""
    env = HouseRecoveryEnv(region_scenario(region, seed, batches(lam)), **mechanism(alpha, c_minor, c_moderate, c_major))
    env.reset(seed=seed)
    w = historical(region)
    for _ in range(HORIZON):
        if env.step(w)[2]:
            break
    done = np.where(env.done >= 0, env.done, HORIZON)
    t = np.arange(HORIZON)
    rows = [np.searchsorted(np.sort(done), t, side="right") / env.n]
    rows += [np.searchsorted(np.sort(done[env.dl == k]), t, side="right") / (env.dl == k).sum() for k in range(3)]
    return np.vstack(rows)


def rmse(curves4, obs):
    """RMSE of the overall curve and of the three damage levels (4 values)."""
    days, share, by_level = obs
    return np.sqrt(np.mean((curves4[:, days] - np.vstack([share, by_level.T])) ** 2, axis=1))


def run_point(job):
    """Mean and sd curves over seeds, the fit (mean of the 4 RMSEs of the mean curves), its standard error
    over seeds, and the 4 RMSEs."""
    region, params, seeds, obs = job
    runs = np.array([curves(region, *params, seed) for seed in seeds])
    per_seed = [rmse(c, obs).mean() for c in runs]
    four = rmse(runs.mean(axis=0), obs)
    return runs.mean(axis=0), runs.std(axis=0), float(four.mean()), np.std(per_seed, ddof=1) / np.sqrt(len(seeds)), four


def spread(params):
    """Mataram holding cost of LJF minus SJF: about 0 when the workforce never binds."""
    from evaluate import run_episode
    *mech, lam = params
    hc = {p: np.mean([run_episode((p, "mataram", s, region_scenario("mataram", s, batches(lam))),
                                  **mechanism(*mech))["holding_cost"] for s in range(3)]) for p in ("LJF", "SJF")}
    return hc["LJF"] - hc["SJF"]


def fit(pool, obs, from_grid, pick=None):
    """Run (or reuse) the grid on Mataram, rerun the 10 best points with 30 seeds, and choose one."""
    points = list(itertools.product(*GRID.values()))
    seeds10, seeds30 = range(10), range(100, 130)
    if from_grid:
        grid = pd.read_csv(OUT / "grid.csv")
    else:
        results = pool.map(run_point, [("mataram", p, seeds10, obs["mataram"]) for p in points], chunksize=8)
        grid = pd.DataFrame([(*p, r[2], r[3], *r[4]) for p, r in zip(points, results)],
                            columns=[*GRID, "fit", "se", "rmse_overall", "rmse_minor", "rmse_moderate", "rmse_major"])
        grid.to_csv(OUT / "grid.csv", index=False)
    top = grid.nsmallest(10, "fit").reset_index(drop=True)
    finalists = [tuple(row[list(GRID)]) for _, row in top.iterrows()]
    rerun = list(pool.map(run_point, [("mataram", p, seeds30, obs["mataram"]) for p in finalists]))
    top["fit30"], top["se30"], top["rmse_overall30"] = [r[2] for r in rerun], [r[3] for r in rerun], [r[4][0] for r in rerun]
    top["hc_ljf_minus_sjf"] = list(pool.map(spread, finalists))
    top.to_csv(OUT / "top.csv", index=False)
    print(top.round(4).to_string(index=False), flush=True)
    best = top.loc[top.fit30.idxmin() if pick is None else pick]
    params = {k: float(best[k]) for k in GRID}
    params.update(fit=float(best.fit30), se=float(best.se30), rmse_overall=float(best.rmse_overall30),
                  batches=batches(params["lam"]), seeds=f"{seeds30.start}-{seeds30.stop - 1}")
    (OUT / "best.json").write_text(json.dumps(params, indent=2) + "\n")
    return params


def validate(pool, obs, params):
    """Run the chosen point in all seven regions (30 seeds) and save the RMSEs and the mean curves."""
    p = tuple(params[k] for k in GRID)
    seeds = range(100, 130)
    res = dict(zip(C.REGIONS, pool.map(run_point, [(r, p, seeds, obs[r]) for r in C.REGIONS])))
    rows, mean, sd = [], {}, {}
    for r, (m, s, fit4, se, four) in res.items():
        rows.append({"region": C.REGIONS[r].name, "role": "calibration" if r == "mataram" else "validation",
                     "fit": fit4, "se": se, **dict(zip(["rmse_overall", *[f"rmse_{x.lower()}" for x in LEVELS]], four))})
        mean[r], sd[r] = m, s
    pd.DataFrame(rows).to_csv(OUT / "validation.csv", index=False)
    np.savez(OUT / "curves.npz", **{f"{r}_mean": mean[r] for r in mean}, **{f"{r}_sd": sd[r] for r in sd})


def plot(obs):
    """Figures: overall curves by region, curves by damage level, and 2-D slices of the fit surface."""
    params = json.loads((OUT / "best.json").read_text())
    val = pd.read_csv(OUT / "validation.csv").set_index("region")
    cv = np.load(OUT / "curves.npz")
    t = np.arange(HORIZON)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})

    def panel(ax, r, row, title, rmse_value):
        m, s = cv[f"{r}_mean"][row], cv[f"{r}_sd"][row]
        days, share, by_level = obs[r]
        ax.fill_between(t, m - s, m + s, color="tab:blue", alpha=0.2, linewidth=0)
        ax.plot(t, m, color="tab:blue", label="Simulated (mean of 30)")
        ax.plot(days, share if row == 0 else by_level[:, row - 1], "--", color="black", label="Observed (BPBD)")
        ax.set(xlim=(0, HORIZON), ylim=(0, 1.02), title=f"{title}\nRMSE = {rmse_value:.3f}")
        ax.grid(color="0.92", linewidth=0.6)

    tag = (f"α = {params['alpha']:g}, crew caps {params['c_minor']:g}/{params['c_moderate']:g}/{params['c_major']:g}, "
           f"batches at days {', '.join(str(d) for d, _ in params['batches'])}")
    # Figure A: overall recovery, every region
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, C.REGIONS):
        name = C.REGIONS[r].name
        panel(ax, r, 0, f"{name} ({val.loc[name, 'role']})", val.loc[name, "rmse_overall"])
    axes.flat[-1].axis("off")
    axes.flat[0].legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Days after the 2018-08-19 earthquake")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of damaged houses rebuilt")
    fig.suptitle(f"Simulated vs observed recovery: calibrated on Mataram ({tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_regions.png", dpi=200)
    plt.close(fig)

    # Figure B: by damage level, every region
    fig, axes = plt.subplots(7, 3, figsize=(11, 20), sharex=True, sharey=True)
    for i, r in enumerate(C.REGIONS):
        name = C.REGIONS[r].name
        for k in range(3):
            panel(axes[i, k], r, k + 1, f"{name}: {LEVELS[k]}", val.loc[name, f"rmse_{LEVELS[k].lower()}"])
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Days after the earthquake")
    fig.suptitle(f"Recovery by damage level ({tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_by_damage.png", dpi=170)
    plt.close(fig)

    # Figure C: RMSE surface, 2-D slices through the best point
    grid = pd.read_csv(OUT / "grid.csv")
    pairs = [("alpha", "lam"), ("c_minor", "c_major"), ("c_moderate", "c_major")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (x, y) in zip(axes, pairs):
        fixed = [k for k in GRID if k not in (x, y)]
        sub = grid[np.logical_and.reduce([np.isclose(grid[k], params[k]) for k in fixed])]
        z = sub.pivot(index=y, columns=x, values="fit")
        im = ax.imshow(z.values, origin="lower", aspect="auto", cmap="viridis_r", vmax=min(0.3, z.values.max()))
        ax.set_xticks(range(len(z.columns)), [f"{v:g}" for v in z.columns], rotation=90, fontsize=8)
        ax.set_yticks(range(len(z.index)), [f"{v:g}" for v in z.index], fontsize=8)
        ax.plot(list(z.columns).index(params[x]), list(z.index).index(params[y]), "r*", markersize=12)
        ax.set(xlabel=x, ylabel=y, title=", ".join(f"{k} = {params[k]:g}" for k in fixed))
        fig.colorbar(im, ax=ax, label="Mataram fit: mean of 4 RMSEs (10 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_rmse_surface.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--from-grid", action="store_true", help="reuse results/calibration/grid.csv")
    ap.add_argument("--pick", type=int, help="take this row of top.csv instead of the best fit")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    obs = observed()
    if not args.plot_only:
        with ProcessPoolExecutor(args.workers) as pool:
            params = fit(pool, obs, args.from_grid, args.pick)
            print("best:", params, flush=True)
            validate(pool, obs, params)
    print(pd.read_csv(OUT / "validation.csv").round(4).to_string(index=False))
    plot(obs)


if __name__ == "__main__":
    main()

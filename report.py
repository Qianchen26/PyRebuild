"""Tables and figures of the results, from results/episodes.csv and runs/seed*/.

Usage: python report.py        (a few seconds)
Writes results/table5.csv (Table 5), results/metrics_table.{csv,md} (every metric with its 95% interval,
overall and by region), results/metrics_by_region.csv, results/paired.csv (each trained model and the learned policy,
minus FCR and SJF), figures/training_progress.{pdf,png}, figures/training_signals.png, figures/policy_weights.png and
results/learned_weights.csv.

Averaging: within each environment seed the seven regions get equal weight, and the ten seed means are
then averaged. The 95% t intervals use the ten seed means (9 degrees of freedom). The learned policy is the
mean of the five trained models.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import config as C
from evaluate import FIXED, WEIGHTS, fcr

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results", HERE / "figures"
METRICS = {"holding_cost": "HC", "makespan": "Makespan", "mean_recovery_day": "Mean recovery day",
           "mean_wait": "Mean wait",
           "mean_completion": "Mean completion", "pl2_response": "PL2 RT", "pl2_completion": "PL2 completion",
           "pl2_completion_p90": "PL2 completion P90", "completion_minor": "Completion minor",
           "completion_moderate": "Completion moderate", "completion_major": "Completion major"}
PPO = [f"PPO{k}" for k in range(5)]
ORDER = ["FCR", "SJF", "FIFO", "PriorityOnly", "Historical", "Random", "LJF", *PPO]


def block_ci(frame, col):
    """Mean and 95% t interval over seed blocks (regions averaged within each seed)."""
    blocks = frame.groupby("seed")[col].mean()
    m, se = blocks.mean(), blocks.std(ddof=1) / np.sqrt(len(blocks))
    h = stats.t.ppf(0.975, len(blocks) - 1) * se
    return m, m - h, m + h


def tables(df):
    """Table 5 and the full metrics table; returns the episodes with the 5-model mean added as a policy."""
    df = pd.concat([df, df[df.policy.isin(PPO)].assign(policy="Learned (5-model mean)")
                    .groupby(["policy", "region", "seed"], as_index=False).mean(numeric_only=True)])
    order = [*ORDER, "Learned (5-model mean)"]
    ci = {p: {name: block_ci(f, col) for col, name in METRICS.items()} for p, f in df.groupby("policy")}
    means = pd.DataFrame({p: {k: v[0] for k, v in row.items()} for p, row in ci.items()}).T.loc[order]
    means.to_csv(OUT / "table5.csv")
    cells = pd.DataFrame({p: {k: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in row.items()}
                          for p, row in ci.items()}).T.loc[order]
    by_region = (df.groupby(["policy", "region"])[list(METRICS)].mean().rename(columns=METRICS)
                 .reset_index().assign(region=lambda d: d.region.map(lambda r: C.REGIONS[r].name)))
    cells.to_csv(OUT / "metrics_table.csv")
    by_region.to_csv(OUT / "metrics_by_region.csv", index=False)
    with open(OUT / "metrics_table.md", "w") as f:
        f.write("# Evaluation metrics (7 regions x seeds 1000-1009; mean [95% CI over seed blocks])\n\n")
        f.write("Units: HC = priority-weighted building-days per building; times in days.\n\n" + cells.to_markdown() + "\n\n")
        for col in ["HC", "Mean wait", "PL2 RT", "Makespan"]:
            f.write(f"## {col} by region\n\n" + by_region.pivot(index="policy", columns="region", values=col)
                    .loc[order].round(3).to_markdown() + "\n\n")
    pd.set_option("display.width", 250)
    print(means.round(3).to_string())
    return df


def paired(df):
    """Learned policy (each model and the 5-model mean) minus FCR and minus SJF, matched by region and seed."""
    key, cols = ["region", "seed"], {"holding_cost": "HC", "mean_wait": "Mean wait", "pl2_response": "PL2 RT",
                                     "pl2_completion": "PL2 completion", "makespan": "Makespan"}
    rows = []
    for ref in ("FCR", "SJF"):
        base = df[df.policy == ref].set_index(key)[list(cols)]
        for k in [*PPO, "Learned (5-model mean)"]:
            d = (df[df.policy == k].set_index(key)[list(cols)] - base).reset_index()
            for col, name in cols.items():
                m, lo, hi = block_ci(d, col)
                rows.append({"policy": k, "reference": ref, "metric": name, "diff": m, "lo": lo, "hi": hi,
                             "excludes_0": lo > 0 or hi < 0})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paired.csv", index=False)
    print(out[out.policy == "Learned (5-model mean)"].round(3).to_string(index=False))


def training_progress():
    """(a) raw episode reward as logged; (b) probe holding cost relative to the Random policy (health.py)."""
    curves = [pd.read_csv(HERE / f"runs/seed{k}/progress.csv").dropna(subset=["rollout/ep_rew_mean"]) for k in range(5)]
    grid = np.linspace(0, min(c["time/total_timesteps"].max() for c in curves), 200)
    returns = np.array([np.interp(grid, c["time/total_timesteps"], c["rollout/ep_rew_mean"]) for c in curves])
    probes = [pd.read_csv(HERE / f"runs/seed{k}/probe.csv") for k in range(5)]
    refs = [json.loads((HERE / f"runs/seed{k}/probe_reference.json").read_text())["random_hc"] for k in range(5)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .8, "savefig.facecolor": "white",
                         "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.subplots_adjust(left=.08, right=.985, bottom=.16, top=.84, wspace=.28)
    for values in returns:
        ax.plot(grid / 1e6, values, color="0.72", linewidth=.9, zorder=2)
    ax.plot(grid / 1e6, returns.mean(axis=0), color="black", linewidth=2, zorder=3)
    ax.set(xlabel="Environment steps (millions)", ylabel="Mean cumulative episode reward",
           title="(a) Logged episode reward (last 100 episodes)")
    for probe, ref in zip(probes, refs):
        bx.plot(probe.timesteps / 1e6, probe.hc / ref, color="0.6", marker="o", markersize=3, linewidth=.9)
    bx.axhline(1, color="black", linewidth=1, linestyle="--")
    bx.text(1.0, 1.005, "Random policy", ha="right", va="bottom", fontsize=10)
    bx.set(xlabel="Environment steps (millions)", ylabel="Probe holding cost / Random",
           title="(b) Deterministic policy on 5 fixed probe scenarios")
    for a in (ax, bx):
        a.grid(axis="y", color="0.90", linewidth=.6, zorder=0)
        a.title.set_fontsize(11.5)
    fig.legend([Line2D([], [], color="black", linewidth=2), Line2D([], [], color="0.72", linewidth=1)],
               ["Mean across seeds", "Individual seeds (n = 5)"],
               loc="upper center", bbox_to_anchor=(.5, .995), ncol=2, frameon=False, fontsize=11)
    fig.savefig(FIG / "training_progress.png", dpi=300)
    fig.savefig(FIG / "training_progress.pdf")
    plt.close(fig)


def training_signals():
    """The training signals checked by health.py, per PPO update, for every seed."""
    panels = [("train/approx_kl", "(a) approx_kl", 0.2), ("train/entropy_loss", "(b) Policy entropy", None),
              ("train/explained_variance", "(c) Explained variance", 0.5), ("rollout/ep_rew_mean", "(d) Episode reward", None),
              ("train/std", "(e) Policy std", None), ("train/clip_fraction", "(f) Clip fraction", None)]
    logs = [pd.read_csv(HERE / f"runs/seed{k}/progress.csv") for k in range(5)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title, line) in zip(axes.flat, panels):
        for k, d in enumerate(logs):
            d = d.dropna(subset=[col])
            y = -d[col] if col == "train/entropy_loss" else d[col]
            ax.plot(d["time/total_timesteps"] / 1e6, y, linewidth=1.1, label=f"seed {k}")
        if line is not None:
            ax.axhline(line, color="black", linestyle="--", linewidth=.9)
        ax.set(title=title, xlabel="Environment steps (millions)")
        ax.grid(color="0.9", linewidth=.6)
    axes[0, 0].set_yscale("log")
    axes[0, 0].text(0.02, 0.2, "red-flag level 0.2", transform=axes[0, 0].get_yaxis_transform(), va="bottom", fontsize=9)
    axes[0, 2].text(0.02, 0.5, "health threshold 0.5 (after 300k steps)", transform=axes[0, 2].get_yaxis_transform(), va="bottom", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "training_signals.png", dpi=200)
    plt.close(fig)


def weight_comparison(df):
    """Weights of the learned policy next to SJF and FCR, in two panels: (a) the mean over all decision days,
    i.e. what the models usually output; (b) the mean over contested days, when that weight can change who
    gets workers (evaluate.py, env.contested_attributes), i.e. how the models score under competition.
    Each model's value averages its 70 episode means; the bar is the mean of the five models.
    Writes results/learned_weights.csv with both means and the share of contested days per model."""
    panels = {"all": WEIGHTS, "contested": [f"{w}_contested" for w in WEIGHTS]}
    ppo = df[df.policy.isin(PPO)].groupby("policy")
    table = pd.concat({k: ppo[cols].mean().set_axis(WEIGHTS, axis=1) for k, cols in panels.items()}, axis=1)
    days = ppo[[f"days_{w[2:]}_contested" for w in WEIGHTS]].sum()
    table = table.join(pd.concat({"contested share": days.div(ppo.decision_days.sum(), axis=0).set_axis(WEIGHTS, axis=1)}, axis=1))
    table.to_csv(OUT / "learned_weights.csv")
    print("\nLearned weights [work, wait, priority, damage]: all days, contested days, contested share\n"
          + table.round(3).to_string())
    names, colors = ["SJF", "FCR", "Learned policy"], ["#D9D9D9", "#969696", "#404040"]
    labels = ["Repair work", "Days since\neligibility", "Social priority", "Damage level"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "axes.linewidth": .7,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    fig.subplots_adjust(left=.17, right=.985, bottom=.14, top=.80, wspace=.08)
    groups, offsets = np.arange(4), [-.27, 0., .27]
    for ax, (key, title) in zip(axes, [("all", "(a) All decision days"), ("contested", "(b) Contested days")]):
        model_means = table[key].loc[PPO].to_numpy()
        weights = np.array([FIXED["SJF"], fcr(), model_means.mean(axis=0)], dtype=float)
        ax.axvline(0, color="black", linewidth=1., zorder=2)
        for values, offset, color in zip(weights, offsets, colors):
            ax.barh(groups + offset, values, height=.22, color=color, edgecolor="#303030", linewidth=.55, zorder=3)
        for k, offset in enumerate(np.linspace(-.10, .10, 5)):
            ax.scatter(model_means[k], groups + offsets[2] + offset, s=12, facecolor="white",
                       edgecolor="black", linewidth=.7, zorder=6)
        for boundary in np.arange(.5, 3.5):
            ax.axhline(boundary, color="#E5E5E5", linewidth=.65, zorder=0)
        ax.set_title(title, fontsize=14, loc="left")
        ax.set_yticks(groups, labels, fontsize=13.5)
        ax.tick_params(axis="y", length=0, pad=14)
        ax.set_ylim(3.57, -.57)
        ax.set_xlim(-1.12, 1.12)
        ax.set_xticks([-1., -.5, 0., .5, 1.], ["\u22121.0", "\u22120.5", "0", "0.5", "1.0"])
        ax.tick_params(axis="x", labelsize=13)
        ax.grid(axis="x", color="#EEEEEE", linewidth=.6)
        ax.set_axisbelow(True)
    fig.supxlabel("Dispatching weight (attributes min-max scaled among candidates)", y=.02, fontsize=12)
    fig.legend([Patch(facecolor=c, edgecolor="#303030", linewidth=.55) for c in colors], names,
               ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.58, .995), fontsize=12.5)
    fig.legend([Line2D([], [], marker="o", markersize=4, markerfacecolor="white", markeredgecolor="black",
                       linestyle="none")], ["Learned-policy points: one mean per training seed"],
               frameon=False, loc="upper center", bbox_to_anchor=(.58, .935), fontsize=11.5)
    fig.savefig(FIG / "policy_weights.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    episodes = tables(pd.read_csv(OUT / "episodes.csv"))
    paired(episodes)
    training_progress()
    training_signals()
    weight_comparison(episodes)

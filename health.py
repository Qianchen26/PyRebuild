"""Health criteria of a PPO training run, written before the reported runs were trained.

A run counts as successful when its training signals are healthy; it is not required to beat a
baseline policy, because whether the learned policy beats the fixed rules is what evaluate.py tests. The signals come from runs/<run>/progress.csv, one row per PPO update.

Red flags, checked once at 20% of training (any one stops the run, and run_training.sh starts asha.py):
  R1 NaN or inf in a logged training metric
  R2 approx_kl above 0.2 in 3 consecutive updates
  R3 policy standard deviation outside [0.05, 5]
  R4 mean explained variance of the last 5 updates below 0.2
  R5 rising entropy: mean policy standard deviation of the last 5 updates more than 5% above the first 5
Final health (all must hold): R1-R3 never triggered; mean explained variance after 300,000 steps at least
0.5; mean standard deviation of the last 10% of updates at most 5% above the first 10%.
The 5% tolerance keeps update-to-update noise (about 1%) from counting as a rise in entropy.

The logged episode reward is a poor learning signal under domain randomization: the first episodes to
finish are the short scenarios, so early rewards look better than they are. Training therefore also
logs probe.csv, the holding cost of the deterministic policy on five fixed scenarios (one per scenario type in config.SCENARIO_TYPES)
at the start and every 10% of training, with the Random policy's cost as a reference
(probe_reference.json). The probe is reported and ranks the asha.py trials; it is not a red flag.
"""
import json

import numpy as np
import pandas as pd

CHECK_AT = 0.2
KL, KL_RUN, STD, ENTROPY_TOL = 0.2, 3, (0.05, 5.0), 1.05


def load(run):
    """The rows of progress.csv that follow a PPO update (the first row is logged before any update)."""
    path = run / "progress.csv"
    df = pd.read_csv(path) if path.stat().st_size else pd.DataFrame()
    return df[df["train/approx_kl"].notna()].reset_index(drop=True) if "train/approx_kl" in df else df


def _common(df):
    """R1-R3, which must hold over the whole run."""
    flags = []
    metrics = df[[c for c in df if c.startswith("train/")]].to_numpy(dtype=float)
    if not np.isfinite(metrics).all():
        flags.append("R1 non-finite training metric")
    high = (df["train/approx_kl"] > KL).astype(int).to_numpy()
    if np.convolve(high, np.ones(KL_RUN, dtype=int), "valid").max(initial=0) >= KL_RUN:
        flags.append(f"R2 approx_kl > {KL} in {KL_RUN} consecutive updates")
    if not df["train/std"].between(*STD).all():
        flags.append(f"R3 policy std outside {STD}: {df['train/std'].min():.3g}-{df['train/std'].max():.3g}")
    return flags


def red_flags(df):
    """R1-R5 on the updates logged so far (used at 20% of training)."""
    flags = _common(df)
    if df["train/explained_variance"].tail(5).mean() < 0.2:
        flags.append(f"R4 explained variance {df['train/explained_variance'].tail(5).mean():.3f} < 0.2")
    std = df["train/std"].to_numpy()
    if std[-5:].mean() > ENTROPY_TOL * std[:5].mean():
        flags.append(f"R5 entropy rising: std {std[:5].mean():.3f} -> {std[-5:].mean():.3f}")
    return flags


def final_health(run):
    """Final verdict with the numbers behind it; written to runs/<run>/health.json by train.py."""
    df, probe = load(run), pd.read_csv(run / "probe.csv")
    reference = json.loads((run / "probe_reference.json").read_text())["random_hc"]
    flags = _common(df)
    ev = df.loc[df["time/total_timesteps"] >= 300_000, "train/explained_variance"].mean()
    if not ev >= 0.5:
        flags.append(f"explained variance after 300k = {ev:.3f} < 0.5")
    k = max(1, len(df) // 10)
    std = df["train/std"].to_numpy()
    if std[-k:].mean() > ENTROPY_TOL * std[:k].mean():
        flags.append(f"entropy rising: std first 10% {std[:k].mean():.3f} -> last 10% {std[-k:].mean():.3f}")
    return {"healthy": not flags, "flags": flags, "ev_after_300k": float(ev), "probe_random_hc": reference,
            "probe_hc": dict(zip(probe.timesteps.astype(int).astype(str), probe.hc.round(4))),
            "kl_max": float(df["train/approx_kl"].max()),
            "std_range": [float(df["train/std"].min()), float(df["train/std"].max())],
            "entropy_loss_first_last": [float(df["train/entropy_loss"].iloc[0]), float(df["train/entropy_loss"].iloc[-1])]}


def _hc(scenario, act):
    """Holding cost of one probe episode; act(obs) returns the day's weights."""
    from env import HouseRecoveryEnv
    env = HouseRecoveryEnv(scenario)
    obs, _ = env.reset(seed=scenario.seed)
    hc, end = 0.0, False
    while not end:
        obs, reward, terminated, truncated, _ = env.step(act(obs))
        hc, end = hc - reward, terminated or truncated
    return hc


def reference_hc(scenarios, run=None):
    """Mean holding cost of Random (U(-1, 1) weights each day, stream 12345 + j for probe scenario j)."""
    rngs = [np.random.default_rng(12345 + j) for j in range(len(scenarios))]
    value = float(np.mean([_hc(s, lambda _, r=r: r.uniform(-1, 1, 4).astype(np.float32)) for s, r in zip(scenarios, rngs)]))
    if run is not None:
        (run / "probe_reference.json").write_text(json.dumps({"random_hc": value}) + "\n")
    return value


def probe_hc(model, scenarios):
    """Mean holding cost of the deterministic policy, with the training observation statistics."""
    norm = model.get_env()
    return float(np.mean([_hc(s, lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0])
                          for s in scenarios]))


def callback(run, total, probes, stop=True):
    from stable_baselines3.common.callbacks import BaseCallback

    class HealthCallback(BaseCallback):
        """Probes every 10%; checks the red flags once, at the first update after 20%; stops on a flag if `stop`."""

        def __init__(self):
            super().__init__()
            self.checked, self.red_flags, self.rows, self.next_probe, self.reference = False, [], [], 0.0, None

        def _probe(self):
            self.rows.append({"timesteps": self.num_timesteps, "hc": probe_hc(self.model, probes)})
            pd.DataFrame(self.rows).to_csv(run / "probe.csv", index=False)
            self.logger.record("probe/hc", self.rows[-1]["hc"])
            self.next_probe += 0.1

        def _on_training_start(self):
            self.reference = reference_hc(probes, run)
            self._probe()

        def _on_rollout_end(self):
            if self.num_timesteps >= self.next_probe * total:
                self._probe()
            if not self.checked and self.num_timesteps >= CHECK_AT * total:
                df = load(run)
                if len(df) < 5:            # too few updates logged yet: check at the next one
                    return
                self.checked = True
                self.red_flags = red_flags(df)
                (run / "health_20pct.json").write_text(json.dumps(
                    {"timesteps": int(self.num_timesteps), "probe_hc": self.rows, "probe_random_hc": self.reference,
                     "red_flags": self.red_flags},
                    indent=2) + "\n")

        def _on_training_end(self):   # after the last update, i.e. the saved model
            self._probe()

        def _on_step(self):
            return not (stop and self.red_flags)

    return HealthCallback()

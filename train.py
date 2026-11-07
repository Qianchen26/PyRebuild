"""Train one PPO policy with domain randomization over a pool of 200 synthetic scenarios.

Usage: python train.py --seed K        (K = 0-4; about 1e6 environment steps, 20 min on 8 cores)

Seed K uses scenario seeds 100000 + 1000 K + (0..199). The five pools do not overlap each other or the
seeds used for evaluation (1000-1009) and for the FCR search. Each of the eight parallel environments
owns every eighth scenario of the pool and plays its scenarios in shuffled passes, so every scenario is
trained and the counts within an environment differ by at most one. Every episode draws a new batch
schedule, new delays, absences and progress noise, while a scenario fixes its houses (damage level, repair
work, priority) and its workforce.

At 20% of training the red flags of health.py are checked; a red flag stops the run with exit code 3
(run_training.sh then starts asha.py). A run that trains to the end but fails the final health check
exits with code 4; its model is kept for diagnosis (asha.py trials, run with --no-stop, exit with 0). Writes runs/seedK/: progress.csv and TensorBoard events,
probe.csv, checkpoints every 10%, final_model.zip, vecnormalize.pkl, scenario_counts.csv, health.json.
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

import config as C
from env import HouseRecoveryEnv, Scenario

RUNS = Path(__file__).resolve().parent / "runs"
GAMMA = 0.99   # horizon 1/(1 - gamma) = 100 days covers the median time a crew stays on a house (96 days)
LR_START, LR_END = 5.174327057823035e-4, 1e-5   # linear schedule
# Selected by asha.py. With the Stable-Baselines3 defaults and an entropy coefficient of 0.01, the policy
# entropy kept rising during training (the entropy bonus outweighed the policy gradient).
PPO_SETTINGS = dict(n_steps=1024, batch_size=128, n_epochs=5, gae_lambda=0.9,
                    clip_range=0.2, ent_coef=0.001, net_arch=[256, 256])


def scenario_pool(base_seed):
    """The 200 scenarios [(type, Scenario)]; a scenario's seed fixes its size, damage mix, workforce and houses."""
    pool = []
    kinds = [kind for kind, (_, count) in C.SCENARIO_TYPES.items() for _ in range(count)]
    for j, kind in enumerate(kinds):
        seed = base_seed + j
        rng = np.random.default_rng((seed, 1))
        n = int(rng.integers(*C.SYNTH_BUILDINGS))
        counts = tuple(int(c) for c in rng.multinomial(n, C.SCENARIO_TYPES[kind][0]))
        workers = max(1, int(n * rng.uniform(*C.SYNTH_WORKER_RATIO)))   # whole workers, rounded down
        n_infra = int(n * rng.uniform(*C.SYNTH_INFRA_RATIO))
        hospitals = max(1, int(n_infra * 0.3))
        pool.append((kind, Scenario(counts, workers, hospitals, n_infra - hospitals, seed, None)))
    return pool


def scenarios(base_seed, per_type):
    """The first `count` scenarios of each damage type in the pool starting at base_seed."""
    pool = scenario_pool(base_seed)
    return [s for kind, count in zip(C.SCENARIO_TYPES, per_type) for s in [s for k, s in pool if k == kind][:count]]


class CyclingScenarioEnv(gym.Env):
    """One training environment: plays its scenarios in shuffled passes and counts how often each was played."""
    observation_space = HouseRecoveryEnv.observation_space
    action_space = HouseRecoveryEnv.action_space

    def __init__(self, pool, indices, env_seed):
        self.pool, self.indices, self.rng = pool, indices, np.random.default_rng(env_seed)
        self.queue, self.counts = [], np.zeros(len(pool), dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        if not self.queue:
            self.queue = list(self.rng.permutation(self.indices))
        j = self.queue.pop()
        self.counts[j] += 1
        self.env = HouseRecoveryEnv(self.pool[j][1])
        return self.env.reset(seed=int(self.rng.integers(2**63)))

    def step(self, action):
        return self.env.step(action)


def make_env(pool, indices, env_seed):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(CyclingScenarioEnv(pool, indices, env_seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--hparams", help="JSON with PPO settings that replace the defaults (from asha.py)")
    ap.add_argument("--run", help="output folder (default runs/seedK)")
    ap.add_argument("--no-stop", action="store_true", help="record red flags but train to the end (asha.py trials)")
    args = ap.parse_args()
    C.check_calibration()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from health import callback, final_health

    torch.set_num_threads(1)
    settings = {**PPO_SETTINGS, "gamma": GAMMA, "lr_start": LR_START}
    if args.hparams:
        settings.update(json.loads(Path(args.hparams).read_text()))
    run = Path(args.run) if args.run else RUNS / f"seed{args.seed}"
    if (run / "final_model.zip").exists():
        raise SystemExit(f"{run} already has a final model; move it before retraining")
    run.mkdir(parents=True, exist_ok=True)
    (run / "settings.json").write_text(json.dumps({**settings, **vars(args)}, indent=2) + "\n")

    pool = scenario_pool(100_000 + 1000 * args.seed)
    gamma, lr0 = settings.pop("gamma"), settings.pop("lr_start")
    env = VecNormalize(SubprocVecEnv([partial(make_env, pool, list(range(i, len(pool), args.n_envs)), 1000 * args.seed + i)
                                     for i in range(args.n_envs)]),
                       norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=gamma)
    net = settings.pop("net_arch")
    model = PPO("MlpPolicy", env, learning_rate=lambda left: LR_END + (lr0 - LR_END) * left, gamma=gamma,
                seed=args.seed, verbose=0, policy_kwargs={"net_arch": net}, **settings)
    model.set_logger(configure(str(run), ["csv", "tensorboard"]))
    every = max(args.total_timesteps // 10 // args.n_envs, 1)
    probes = scenarios(100_000 + 1000 * args.seed, [1] * len(C.SCENARIO_TYPES))   # one per damage type
    health = callback(run, args.total_timesteps, probes, stop=not args.no_stop)
    model.learn(args.total_timesteps, callback=[CheckpointCallback(every, str(run / "checkpoints"),
                                                                   save_vecnormalize=True), health])
    per_env = np.array(env.get_attr("counts"))
    counts = per_env.sum(axis=0)
    np.savetxt(run / "scenario_counts.csv", per_env.T, fmt="%d", delimiter=",",
               header=",".join(f"env{i}" for i in range(args.n_envs)))
    if health.red_flags and not args.no_stop:
        env.close()
        print(f"seed {args.seed}: red flags at 20%: {health.red_flags}", flush=True)
        sys.exit(3)
    model.save(run / "final_model")
    env.save(run / "vecnormalize.pkl")
    env.close()
    report = final_health(run)
    report["scenarios_never_trained"] = int((counts == 0).sum())
    report["per_env_count_spread"] = int(max(np.ptp(c[i::args.n_envs]) for i, c in enumerate(per_env)))
    (run / "health.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"seed {args.seed}: healthy={report['healthy']}", flush=True)
    if not report["healthy"] and not args.no_stop:
        sys.exit(4)


if __name__ == "__main__":
    main()

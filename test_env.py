"""Self-check of the simulator: python test_env.py   (prints "ok" in about one second)

Each block tests one property of env.py: the holding-cost identity, eligibility and same-day dispatch,
crews that never shrink, the absence model, the ranking, that no information leaks, the random
schedules, the delay distribution, and the split between house and episode random numbers.
"""
import hashlib

import numpy as np

import config as C
from env import HouseRecoveryEnv, Scenario, random_schedule, region_scenario

SMALL = Scenario((1200, 500, 300), 900, 6, 20, 7, ((0, 0.3), (14, 0.35), (60, 0.35)))
NO_DELAY = dict(delay_median=np.full(3, 1e-9))   # every house eligible on its information day


def episode(env, weights, seed=0, every_day=None):
    env.reset(seed=seed)
    hc, terminated = 0.0, False
    while not terminated:
        _, reward, terminated, truncated, _ = env.step(np.asarray(weights, dtype=np.float32))
        assert not truncated
        hc -= reward
        if every_day:
            every_day(env)
    return hc


def buildings_hash(env):
    return hashlib.sha256(b"".join(a.tobytes() for a in (env.dl, env.pl, env.work))).hexdigest()


# Holding cost = sum_i w_i (done_i - elig_i) / N with random schedules and delays; no house gets workers
# before it is eligible; houses eligible on the first decision day can get workers that day
env = HouseRecoveryEnv(SMALL._replace(batches=None))
hc = episode(env, [0, 1, 0, 0])
assert abs(hc - (env.weight * (env.done - env.elig)).sum() / env.n) < 1e-10
assert (env.start >= env.elig).all()
for region in C.REGIONS:
    env = HouseRecoveryEnv(region_scenario(region, 1000))
    env.reset(seed=1000)
    first = env.day
    env.step(np.array([0, 0, 1, 0], dtype=np.float32))       # PriorityOnly, first decision day
    assert (env.start[(env.pl == 2) & (env.elig == first)] == first).all(), region

# Crews never shrink before completion, no house restarts, and every worker is in a crew or the free pool
env = HouseRecoveryEnv(SMALL, **NO_DELAY)
last_crew, last_start = np.zeros(sum(SMALL.counts), dtype=int), np.full(sum(SMALL.counts), -1)


def check(env):
    global last_crew, last_start
    unfinished = env.done < 0
    assert (env.crew[unfinished] >= last_crew[unfinished]).all()
    assert ((last_start < 0) | (env.start == last_start)).all()
    assert env.crew.sum() + env.free + env.free_away == env.K
    last_crew, last_start = env.crew.copy(), env.start.copy()


episode(env, [1, 0, 0, 0], every_day=check)                  # LJF

# Absences: long-run rate rho, mean absence 1/p days, and p = 1 gives a daily absent share ~ U(0, 2 rho)
LONG = Scenario((1, 0, 0), 10_000, 0, 0, 1, ((0, 1.0),))      # one building that never finishes
for p in (0.5, 1.0):
    env = HouseRecoveryEnv(LONG, alpha=1e9, return_prob=p)
    env.reset(seed=3)
    absent = []
    for _ in range(2000):
        env.step(np.zeros(4, dtype=np.float32))
        absent.append((env.away.sum() + env.free_away) / env.K)
    absent = np.array(absent[50:])
    assert abs(absent.mean() - 0.05) < 0.003, (p, absent.mean())
    if p == 1.0:
        assert abs(absent.std() - 0.0289) < 0.003, absent.std()
env = HouseRecoveryEnv(LONG._replace(workers=200_000), alpha=1e9, absence_rate=0.05, return_prob=0.5)
env.reset(seed=4)
env.onset, total = 0.0, [env.free_away]
for _ in range(60):
    env.step(np.zeros(4, dtype=np.float32))
    total.append(env.free_away)
assert abs(sum(total) / total[0] - 2.0) < 0.1, sum(total) / total[0]

# A single-attribute rule ranks exactly by that raw attribute (first day, not every candidate served)
for weights, key in [([-1, 0, 0, 0], lambda e: e.work), ([1, 0, 0, 0], lambda e: -e.work),
                     ([0, 0, 1, 0], lambda e: -e.pl), ([0, 0, 0, 1], lambda e: -e.dl)]:
    env = HouseRecoveryEnv(SMALL._replace(workers=300), **NO_DELAY)
    env.reset(seed=5)
    env.step(np.array(weights, dtype=np.float32))
    started, waiting = env.start == 0, (env.arrive == 0) & (env.start < 0)
    assert key(env)[started].max() <= key(env)[waiting].min(), weights

# No information leak: extra houses in a later batch leave every earlier observation unchanged
def add_late(env, m, day):
    for name, value in [("dl", 0), ("pl", 0), ("work", 30.0), ("arrive", day), ("elig", day), ("cap", 2), ("weight", 1.0),
                        ("rem", 30.0), ("start", -1), ("done", -1), ("crew", 0), ("away", 0)]:
        a = getattr(env, name)
        setattr(env, name, np.concatenate([a, np.full(m, value, dtype=a.dtype)]))
    env.n += m
    env.eligible = env.elig <= env.day


action = np.array([-0.3, 0.5, 1.0, 0.2], dtype=np.float32)
a, b = HouseRecoveryEnv(SMALL, **NO_DELAY), HouseRecoveryEnv(SMALL, **NO_DELAY)
obs_a, _ = a.reset(seed=6)
obs_b, _ = b.reset(seed=6)
add_late(b, 5000, 60)
for day in range(60):
    assert (obs_a == obs_b).all() and obs_a.shape == (17,) and np.isfinite(obs_a).all(), day
    obs_a, obs_b = a.step(action)[0], b.step(action)[0]
assert not (obs_a == obs_b).all()
# ... and a known house whose delay ends later changes nothing before the earlier end date
a, b = HouseRecoveryEnv(SMALL), HouseRecoveryEnv(SMALL)
obs_a, _ = a.reset(seed=8)
obs_b, _ = b.reset(seed=8)
waiting = np.flatnonzero((b.arrive <= b.day) & (b.elig > b.day + 30))[:200]
b.elig[waiting] += 100
cutoff = b.elig[waiting].min() - 100
while a.day < cutoff:
    assert (obs_a == obs_b).all(), a.day
    assert obs_a[16] == np.float32(((a.arrive <= a.day) & (a.elig > a.day)).sum() / a.K)
    obs_a, obs_b = a.step(action)[0], b.step(action)[0]
    assert a.day == b.day

# Random schedules stay within their bounds and keep the calibrated mean shares
rng = np.random.default_rng(0)
draws = [random_schedule(rng) for _ in range(10_000)]
days = np.array([[d for d, _ in s] for s in draws])
shares = np.array([[w for _, w in s] for s in draws])
base_days, base_shares = np.array(C.BATCHES).T
gaps = np.diff(base_days, prepend=0)
assert (days >= np.cumsum(0.5 * gaps) - 0.5).all() and (days <= np.cumsum(1.5 * gaps) + 0.5).all()
assert np.abs(shares.mean(axis=0) - base_shares).max() < 0.01
assert np.abs(shares.std(axis=0) - 0.1).max() < 0.02, shares.std(axis=0)

# Delays by damage level match Alisjahbana & Kiremidjian (2021): medians 275/300/225 days, sigma 0.3
env = HouseRecoveryEnv(region_scenario("lombokbarat", 1))
env.reset(seed=1)
for k, median in enumerate(C.MECHANISM["delay_median"]):
    log_delay = np.log(np.maximum(env.elig - env.arrive, 1)[env.dl == k])
    assert abs(np.exp(np.median(log_delay)) / median - 1) < 0.03 and abs(log_delay.std() - 0.3) < 0.02, k

# Houses are fixed by the scenario seed; absences and noise come from the episode seed
env = HouseRecoveryEnv(SMALL)
env.reset(seed=1)
h1, s1 = buildings_hash(env), env.np_random.random()
env.reset(seed=2)
assert buildings_hash(env) == h1 and env.np_random.random() != s1
print("ok")

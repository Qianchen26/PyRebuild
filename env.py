"""Simulator of daily workforce allocation to damaged houses (a Gymnasium environment).

The model has three stages for every house:
1. Damage information arrives in batches (config.BATCHES).
2. After the information arrives, the house waits through its own pre-construction delay (verification,
   funding, forming the community group). No workers are held during the delay.
3. When the delay ends the house becomes eligible. The policy then decides when the house gets workers,
   and the crew stays until the house is repaired.

One step is one day:
- houses whose delay ends today become candidates, and workers' absences start or end;
- the policy picks four dispatching weights;
- free workers who are present first fill crews that started short of their cap, then go to candidates in
  score order, each candidate getting up to its crew cap;
- present crew members repair; finished houses release their crew;
- the holding cost of the day is charged.

Days on which no house is eligible or under repair need no decision and cost nothing, so they are skipped.

Action: weights in [-1, 1] on [repair work, days since eligibility, social priority level, damage level]. Each
attribute is min-max scaled among the day's candidates (an attribute equal for all candidates counts 0).

Observation (17 values; VecNormalize standardizes them during training):
- the shares of candidates in the 9 damage x priority groups;
- candidate repair work / K, candidate crew caps / K;
- free present workers / K, crew members / K, absent workers / K;
- remaining work of houses under repair / K;
- days since the disaster;
- houses whose information has arrived but whose delay has not ended / K.
Nothing depends on houses whose information has not arrived, or on when a delay will end.

Reward: minus the priority-weighted number of eligible unfinished houses, divided by N. The episode
holding cost is therefore sum_i w_i (done_i - eligible_i) / N; the delay itself is not charged.
"""
from typing import NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C


class Scenario(NamedTuple):
    counts: tuple            # houses by damage level (minor, moderate, major)
    workers: int             # workforce K
    hospitals: int           # houses with priority level 2
    schools: int             # houses with priority level 1
    seed: int                # fixes the house attributes (work, priority, order)
    batches: tuple | None    # ((day, share), ...); None draws a new schedule every episode


def region_scenario(name, seed, batches=C.BATCHES, workforce_scale=1.0):
    """One of the seven Lombok regions; workforce_scale multiplies K (sensitivity runs only)."""
    r = C.REGIONS[name]
    return Scenario(r.buildings, int(workforce_scale * r.workers), r.hospitals, r.schools, seed, batches)


def random_schedule(rng):
    """A batch schedule around the calibrated one, for training and the FCR search.

    The first day and each gap between batches are the calibrated values times U(0.5, 1.5), and no batch
    comes after LAST_BATCH_MAX. The shares follow a Dirichlet distribution with mean equal to the
    calibrated shares (concentration 20, i.e. a standard deviation of about 0.1 per batch).
    """
    days, shares = np.array(C.BATCHES).T
    gaps = np.diff(days, prepend=0.0) * rng.uniform(0.5, 1.5, len(days))
    days = np.minimum(np.cumsum(gaps), C.LAST_BATCH_MAX).round().astype(int)
    return tuple(zip(days.tolist(), rng.dirichlet(20 * shares).tolist()))


class HouseRecoveryEnv(gym.Env):
    observation_space = spaces.Box(0.0, np.inf, (17,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def __init__(self, scenario, **mechanism):
        self.scenario, self.m = scenario, {**C.MECHANISM, **mechanism}
        rho, p = self.m["absence_rate"], self.m["return_prob"]
        # Mean daily chance that a present worker starts an absence, chosen so that the long-run share
        # of absent workers equals rho: rho * p = (1 - rho) * onset.
        self.onset = rho * p / (1 - rho)

    def reset(self, *, seed=None, options=None):
        """Draw a new episode. Houses come from the scenario seed; the episode seed (self.np_random)
        draws the batch membership, delays, absences and progress noise."""
        super().reset(seed=seed)
        s, m, rng = self.scenario, self.m, self.np_random
        houses = np.random.default_rng((s.seed, 0))
        n = sum(s.counts)
        dl = np.repeat([0, 1, 2], s.counts)
        work = houses.lognormal(np.log(m["alpha"] * m["repair_median"][dl]), m["repair_sigma"])
        pl = np.zeros(n, dtype=np.int64)
        chosen = houses.permutation(n)
        pl[chosen[:s.hospitals]] = 2
        pl[chosen[s.hospitals:s.hospitals + s.schools]] = 1
        order = houses.permutation(n)             # a random fixed order breaks ranking ties
        self.dl, self.pl, self.work = dl[order], pl[order], work[order]
        self.schedule = s.batches or random_schedule(rng)
        days, shares = zip(*self.schedule)
        self.arrive = np.asarray(days)[rng.choice(len(days), n, p=np.asarray(shares) / sum(shares))]
        delay = rng.lognormal(np.log(np.asarray(m["delay_median"])[self.dl]), m["delay_sigma"])
        self.elig = self.arrive + delay.round().astype(np.int64)

        self.n, self.K = n, s.workers
        self.cap = np.asarray(m["max_crew"])[self.dl]
        self.weight = np.asarray(m["holding_cost"])[self.pl]
        self.rem = self.work.copy()
        self.start = np.full(n, -1)               # day the house got its first workers
        self.done = np.full(n, -1)                # day the house was finished
        self.crew = np.zeros(n, dtype=np.int64)   # crew members, absent ones included
        self.away = np.zeros(n, dtype=np.int64)   # crew members who are absent
        self.free, self.free_away = self.K, 0     # workers in no crew: present, absent
        self.day = 0
        self._begin_day()
        self._skip_idle()
        return self._obs(), {}

    def step(self, action):
        self._dispatch(np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0))
        self._repair()
        cost = self.weight[self.eligible & (self.done < 0)].sum() / self.n
        self.day += 1
        self._begin_day()
        self._skip_idle()
        terminated = bool((self.done >= 0).all())
        return self._obs(), -float(cost), terminated, not terminated and self.day >= C.MAX_DAYS, {}

    def _begin_day(self):
        """Mark the eligible houses, then update absences.

        One chance h_t ~ U(0, 2 onset) is drawn for the day and shared by all workers: each worker
        present at the start of the day leaves with chance h_t, and each absent worker returns with
        chance p. Absences are not covered by other workers; an absent crew member stays in the crew.
        """
        self.eligible = self.elig <= self.day
        self.active = np.flatnonzero((self.start >= 0) & (self.done < 0))
        rng, p = self.np_random, self.m["return_prob"]
        h = rng.uniform(0.0, 2.0 * self.onset)
        a = self.active
        self.away[a] += rng.binomial(self.crew[a] - self.away[a], h) - rng.binomial(self.away[a], p)
        leave, back = rng.binomial(self.free, h), rng.binomial(self.free_away, p)
        self.free += back - leave
        self.free_away += leave - back

    def _skip_idle(self):
        """Jump over days with no eligible candidate and no house under repair (nothing to decide)."""
        while not len(self.active) and not (self.eligible & (self.start < 0)).any() and (self.done < 0).any():
            self.day += 1
            self._begin_day()

    def _dispatch(self, weights):
        """Give today's free present workers to houses."""
        free = self.free
        if free <= 0:
            return
        # First, crews that started below their cap (too few free workers on their first day),
        # earliest start first.
        short = self.active[self.crew[self.active] < self.cap[self.active]]
        if len(short):
            short = short[np.argsort(self.start[short], kind="stable")]
            give = self._fill(free, self.cap[short] - self.crew[short])
            self.crew[short] += give
            free -= give.sum()
        # Then candidates, in descending score.
        cand = np.flatnonzero(self.eligible & (self.start < 0))
        if free > 0 and len(cand):
            x = self._attributes(cand)
            lo, span = x.min(axis=0), np.ptp(x, axis=0)
            score = ((x - lo) / np.where(span > 0, span, 1.0)) @ weights
            # At most `free` candidates can start today, so sorting only those (plus every candidate
            # tied with the last of them) gives the same order as sorting all candidates.
            k = min(len(cand), free)
            if k < len(cand):
                keep = score >= np.partition(score, len(score) - k)[len(score) - k]
                cand, score = cand[keep], score[keep]
            order = cand[np.argsort(-score, kind="stable")]
            give = self._fill(free, self.cap[order])
            started = order[give > 0]
            self.crew[started] = give[give > 0]
            self.start[started] = self.day
            self.active = np.union1d(self.active, started)
            free -= give.sum()
        self.free = free

    def _attributes(self, cand):
        """The four raw ranking attributes of the candidates: repair work, days since eligibility, priority, damage."""
        return np.column_stack([self.work[cand], self.day - self.elig[cand], self.pl[cand], self.dl[cand]])

    def contested_attributes(self):
        """Which of today's four weights can change who gets workers (read-only; call before step).

        A weight matters only on a contested day: after short crews are filled, some free present workers
        remain but fewer than the crew caps of all candidates, and at least two candidates compete. On
        other days every weight vector gives the same dispatch. On a contested day, the weight of an
        attribute that is equal for all candidates still has no effect (it is scaled to 0).
        """
        a = self.active
        free = self.free - (self.cap[a] - self.crew[a]).sum()
        cand = np.flatnonzero(self.eligible & (self.start < 0))
        if free <= 0 or len(cand) < 2 or free >= self.cap[cand].sum():
            return np.zeros(4, dtype=bool)
        return np.ptp(self._attributes(cand), axis=0) > 0

    @staticmethod
    def _fill(free, demand):
        """Hand out `free` workers in order: each house gets its demand until the workers run out."""
        return np.clip(free - (np.cumsum(demand) - demand), 0, demand)

    def _repair(self):
        """Present crew members repair; a finished house sends its crew (present and absent) back to the free pool."""
        a = self.active
        present = self.crew[a] - self.away[a]
        a, present = a[present > 0], present[present > 0]
        progress = np.maximum(0.0, present + self.np_random.normal(0.0, self.m["noise"] * present))
        self.rem[a] -= progress
        fin = a[self.rem[a] <= 0]
        self.done[fin] = self.day
        self.free += int((self.crew[fin] - self.away[fin]).sum())
        self.free_away += int(self.away[fin].sum())
        self.crew[fin] = self.away[fin] = 0

    def _obs(self):
        cand = self.eligible & (self.start < 0)
        groups = np.bincount(self.dl[cand] * 3 + self.pl[cand], minlength=9)
        a, K = self.active, self.K
        return np.array([*(groups / max(groups.sum(), 1)),
                         self.work[cand].sum() / K,
                         self.cap[cand].sum() / K,
                         self.free / K,
                         self.crew.sum() / K,
                         (self.away.sum() + self.free_away) / K,
                         self.rem[a].sum() / K,
                         self.day,
                         ((self.arrive <= self.day) & ~self.eligible).sum() / K], dtype=np.float32)

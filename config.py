"""Inputs of the simulator: regional data, calibrated values, mechanism parameters, synthetic scenarios.

Every simulator input used by the other scripts is defined here once. Sources are given next to each
value. The mechanism values without a source (absence rate, progress noise, holding-cost weights) are design
choices that the sensitivity analysis (sensitivity.py) varies.
"""
from typing import NamedTuple

import numpy as np


class Region(NamedTuple):
    name: str
    buildings: tuple[int, int, int]  # damaged houses by level: minor, moderate, major
    workers: int                     # workforce K
    hospitals: int                   # medical facilities: houses assigned priority level 2
    schools: int                     # houses assigned priority level 1


# Damaged houses: BPBD NTB records as reported by Alisjahbana & Kiremidjian (2021), Table 1.
# Workforce: their Table 2 (their public notebook uses 45,208 for West Lombok; we use the table).
# Medical facilities and schools: BPS NTB 2018 statistics. The order of the regions fixes Random's seeds.
REGIONS = {
    "mataram": Region("Mataram", (9500, 3672, 1345), 9917, 44, 118),
    "lombokbarat": Region("West Lombok", (45218, 13556, 14069), 45028, 80, 210),
    "lombokutara": Region("North Lombok", (8889, 4772, 42049), 22996, 39, 85),
    "lomboktengah": Region("Central Lombok", (16639, 3096, 4483), 15048, 124, 523),
    "lomboktimur": Region("East Lombok", (12209, 4657, 10104), 15404, 122, 514),
    "sumbawabarat": Region("West Sumbawa", (13078, 3803, 1283), 10200, 38, 54),
    "sumbawa": Region("Sumbawa", (9652, 2756, 1374), 10360, 120, 169),
}

# Calibrated on Mataram's recovery curves (calibrate.py --from-grid --pick 4; results/calibration/best.json).
# Among the ten best Mataram fits, this is the one in which the workforce is scarce in all seven regions
# and the six validation regions fit best. Mataram fit 0.067 (overall RMSE 0.064); validation RMSE 0.104.
ALPHA = 6.0                                     # repair-work scale
BATCHES = ((0, 0.30), (14, 0.35), (60, 0.35))   # damage information: (day after 2018-08-19, share of houses)
LAST_BATCH_MAX = 350   # the last batch arrives by day 350, i.e. 365 days after the 2018-08-05 mainshock:
                       # the latest end of damage verification documented for comparable earthquakes

# Mechanism parameters. HouseRecoveryEnv(scenario, **overrides) replaces any of them.
MECHANISM = dict(
    alpha=ALPHA,
    repair_median=np.array([30.0, 40.0, 50.0]),  # worker-days at alpha = 1 (Alisjahbana & Kiremidjian 2021, Table 4)
    repair_sigma=0.4,                            # log-space sd of repair work (same source)
    max_crew=np.array([2, 2, 6]),                # most workers on one house, by damage level (calibrated)
    noise=0.15,                                  # daily progress ~ N(w, (0.15 w)^2) for w present workers
    absence_rate=0.05,                           # long-run share of workers absent
    return_prob=0.5,                             # daily chance that an absent worker returns (mean absence 2 days,
                                                 # derived from Sichani 2010, pp. 48-49)
    holding_cost=np.array([1.0, 2.0, 5.0]),      # daily cost of an unfinished house, by priority level
    delay_median=np.array([275.0, 300.0, 225.0]),  # pre-construction delay in days, by damage level
    delay_sigma=0.3,                               # (Alisjahbana & Kiremidjian 2021, Table 4)
)
MAX_DAYS = 5000                                  # stops an episode that would never end; never reached

# Synthetic scenarios for PPO training and the FCR search: damage shares and how many of the 200 per pool.
SCENARIO_TYPES = {
    "minor_dominated": ((0.70, 0.20, 0.10), 50),
    "major_dominated": ((0.10, 0.20, 0.70), 50),
    "balanced": ((0.34, 0.33, 0.33), 50),
    "extreme_minor": ((0.90, 0.05, 0.05), 25),
    "extreme_major": ((0.05, 0.05, 0.90), 25),
}
SYNTH_BUILDINGS = (10_000, 80_000)
SYNTH_WORKER_RATIO = (0.10, 0.80)   # workers = houses x ratio, rounded down
SYNTH_INFRA_RATIO = (0.002, 0.03)   # share of hospitals and schools (about 30% hospitals)

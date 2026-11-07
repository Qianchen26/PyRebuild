# PyRebuild

Code and data for the manuscript *A Sequential Decision Framework for Representing and Comparing Workforce Allocation Policies in Post-Disaster Building Recovery* (Qianchen Yu, Zhaoxi Zhang, and Vivian W. H. Wong; under review).

## Overview

Post-disaster building recovery requires allocating a limited workforce as damage information arrives and workforce availability changes. Previous studies have developed fixed rules and policies learned through reinforcement learning (RL) for this allocation and compared their recovery outcomes. However, few have examined how the scheduling criteria of learned policies differ from those of fixed rules. Therefore, this repository implements a sequential decision framework that expresses these policies through parameterization, lets them use the same input, and then compares their output. Decision makers can then use the framework to choose a policy based on the trade-offs that matter for their objectives.

The case study is the housing recovery after the 2018 Lombok, Indonesia earthquakes: seven regions with 13,782 to 72,843 damaged houses each.

## How the simulator works

The simulator (`env.py`) follows each damaged house through three stages:

1. **Damage information arrives in batches.** The batches arrive on days 0, 14 and 60 after the 2018-08-19 earthquake, and the policy does not know how many houses the next batch will bring.
2. **Each house waits through a pre-construction delay** (verification, funding, forming the community group). No workers are held during this delay.
3. **Eligible houses wait in a queue for workers.** Houses queue when the free workers cannot give every eligible house its crew. Every day each candidate (an eligible house without workers) gets a score from four attributes (repair work, days since eligibility, social priority and damage level), weighted by the policy. Free workers first fill crews that started below their cap, and the rest are then assigned from the highest score to the lowest, up to each house's crew cap.

A house that has started keeps its crew until it is finished and does not go back to the queue. Workers can be absent, and nobody replaces them. When an absent worker comes back, the worker rejoins the crew if the house is not finished; otherwise the worker goes to the free pool.

Every policy uses the same scoring and allocation mechanism, so policies differ only in their weights. The fixed rules (Shortest-Job-First (SJF), Longest-Job-First (LJF), First-In-First-Out (FIFO) and PriorityOnly) use one weight each. The Fixed Composite Rule (FCR) is a weight vector selected by search. Random draws new weights every day, and Historical is the major-damage-first rule assumed in the calibration. The learned policy is trained with Proximal Policy Optimization (PPO) and chooses the weights every day from aggregate recovery information. Because all weights are explicit, the scheduling criteria of the learned policy can be inspected next to its outcomes.

## Main results

The table below gives the averages over the seven regions and ten environment seeds; the learned policy is the mean of five trained models. Holding cost is the priority-weighted days that eligible houses wait for completion, divided by the number of houses.

| Policy | Holding cost | Makespan (days) | Medical facilities: days to first workers | Mean wait (days) |
|---|---|---|---|---|
| FCR | **138.8** | 1,175.3 | **0.0** | 39.9 |
| SJF | 139.3 | 1,176.0 | 37.0 | **39.8** |
| Learned policy | 142.3 | 1,153.8 | 17.9 | 42.7 |
| PriorityOnly | 158.7 | 1,147.7 | **0.0** | 59.8 |
| Random | 159.0 | 1,129.1 | 1.6 | 60.0 |
| FIFO | 161.3 | 1,159.8 | 61.9 | 61.5 |
| Historical (major damage first) | 165.2 | 1,154.9 | 65.3 | 65.2 |
| LJF | 185.9 | **1,128.7** | 88.7 | 85.6 |

No policy is best on every metric. FCR has the lowest holding cost and gives medical facilities workers on the day they become eligible. The learned policy finishes all repairs about 21 days earlier than FCR, but its houses wait longer on average. The full metrics table with 95% intervals is `results/metrics_table.md`.

## Reproduce the results

Install the dependencies (Python 3.13; versions in `requirements.txt`) and run the self-check:

```
pip install -r requirements.txt
python test_env.py
```

**`reproduce.ipynb` rebuilds every table and figure from the saved outputs in this repository in a few minutes.** It can also rerun the evaluation (set `RERUN_EVALUATION = True`; about one minute on 14 cores).

Times in the table below are for a 14-core laptop; `OMP_NUM_THREADS=1` keeps the parallel workers from competing for threads.

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Calibrate on Mataram, validate on six regions | `OMP_NUM_THREADS=1 python calibrate.py --pick 4` | 15 min | `results/calibration/`, `figures/calibration_*.png` |
| 2. Train five PPO models | `bash run_training.sh` | 25 min | `runs/seed0`–`seed4` |
| 3. Search the FCR | `OMP_NUM_THREADS=1 python search_fcr.py` | 20 min | `results/fcr_search/` |
| 4. Evaluate all policies | `OMP_NUM_THREADS=1 python evaluate.py` | 1 min | `results/episodes.csv` |
| 5. Tables and figures | `python report.py` | 10 s | `results/table5.csv`, `results/metrics_table.md`, `results/paired.csv`, `figures/` |
| 6. Sensitivity analysis | `OMP_NUM_THREADS=1 python sensitivity.py` | 1 h | `results/sensitivity/`, `figures/sensitivity_*.png` |

Steps 2 and 3 are independent and can run at the same time. In step 2, a seed whose training raises a red flag at 20% (see `health.py`) is tuned automatically with `asha.py`.

## Repository structure

```
config.py         regional data, calibrated values, mechanism parameters (with sources)
env.py            the simulator (Gymnasium environment)
calibrate.py      grid-search calibration against the recorded recovery curves
train.py          PPO training with domain randomization over synthetic scenarios
health.py         training health criteria (red flags at 20%, final check)
asha.py           successive-halving tuning for a seed with a red flag
run_training.sh   trains seeds 0-4 in parallel and calls asha.py when needed
search_fcr.py     differential-evolution search of the Fixed Composite Rule
evaluate.py       evaluation of all policies in the seven regions
report.py         tables, paired comparisons and figures
sensitivity.py    sensitivity analysis with pre-declared checks
test_env.py       self-check of the simulator
reproduce.ipynb   rebuilds the reported tables and figures
data/             recorded recovery data (see Data and sources)
results/          outputs of the steps above
figures/          figures of the steps above
runs/seedK/       trained models, training logs and health reports
```

Each file starts with a docstring that explains what it does; the scripts that you run also say how to run them and what they write.

## Data and sources

**`data/lombok_data.pkl`** contains the housing recovery records of the provincial disaster management agency of Nusa Tenggara Barat (BPBD NTB), as compiled by Alisjahbana and Kiremidjian (2021). It is a pandas DataFrame with one row per reporting date from 2018-11-22 to 2020-03-31 (449 dates). The columns are `{region}_{field}_{level}`:
- `region`: `mataram`, `lombokbarat` (West Lombok), `lombokutara` (North Lombok), `lomboktengah` (Central Lombok), `lomboktimur` (East Lombok), `sumbawabarat` (West Sumbawa), `sumbawa`;
- `field`: `comp` (houses completed, cumulative), `process` (in process), `target`, `plan`;
- `level`: `rr` (minor damage), `rs` (moderate), `rb` (major).

We use only the `comp` columns. A missing report keeps the last reported value, and dates before the first report count as zero. The file was obtained from the authors' public repository, [queue-recovery](https://github.com/ialisjahbana/queue-recovery). Their paper is:

> Alisjahbana, I., and Kiremidjian, A. (2021). Modeling housing recovery after the 2018 Lombok earthquakes using a stochastic queuing model. *Earthquake Spectra*, 37(2), 587–611. https://doi.org/10.1177/8755293020970972

**Regional inputs** (`config.REGIONS`):
- Damaged houses by level come from the same paper (Table 1), and the workforce of each region from its Table 2.
- The numbers of medical facilities (hospitals and public health centers; priority level 2) and schools (priority level 1), which give the social priority, come from the 2018 statistics of BPS-Statistics Indonesia, Nusa Tenggara Barat province.

**Mechanism parameters** (`config.MECHANISM`):
- The pre-construction delays (medians of 275, 300 and 225 days for minor, moderate and major damage; log-space standard deviation 0.3) and the base repair work (30, 40 and 50 worker-days) come from Alisjahbana and Kiremidjian (2021), Table 4.
- The mean absence of two days (an absent worker returns with probability 0.5 each day) comes from the absence durations reported by Sichani (2010, pp. 48–49).
- The latest information batch arrives no later than day 350 (365 days after the 2018-08-05 mainshock), the latest end of damage verification documented for comparable earthquakes. Lombok itself finished earlier: 98.6% of heavily damaged houses had a district decree 134 days after the mainshock (IFRC appeal MDRID013, revised emergency plan of action, p. 5).
- The crew caps, the repair-work scale and the batch days are calibrated (`calibrate.py`).
- The absence rate (5%) and the progress noise (15% standard deviation of daily progress) are design choices, and `sensitivity.py` varies them.

## Citation

If you use this code, please cite the manuscript (citation details will be added after review) and, for the data, Alisjahbana and Kiremidjian (2021).

## License

No license has been chosen for the code yet. The data file `data/lombok_data.pkl` is redistributed under GPL-3.0, as in its source repository.

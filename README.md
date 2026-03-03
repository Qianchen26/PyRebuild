# PyRebuild

**Deep Reinforcement Learning for Post-Disaster Housing Recovery Scheduling**

A Gymnasium-based simulation environment and PPO training pipeline for optimizing post-disaster housing recovery, calibrated against the **2018 Lombok, Indonesia earthquake** (7 regions, 200k+ damaged buildings).

## Repository Structure

```
├── code/                  # Python package
│   ├── config.py          # All parameters with literature citations
│   ├── env.py             # HouseRecoveryEnv (Gymnasium)
│   ├── baselines.py       # 5 baseline scheduling policies
│   ├── metrics.py         # Evaluation metric functions
│   ├── evaluate.py        # Multi-seed evaluation runner
│   ├── train.py           # PPO training with SB3
│   ├── scenarios.py       # Scenario generation & persistence
│   └── tests/             # Unit tests (26 tests)
│       └── test_env.py
├── output/                # Training & evaluation outputs
│   ├── eval/              # Evaluation results (JSON + CSV)
│   ├── models/            # Saved PPO model + VecNormalize stats
│   ├── scenarios/         # Pre-generated scenario files
│   └── tb_logs/           # TensorBoard training logs
└── README.md
```

## MDP Formulation

| Component | Description |
|-----------|-------------|
| **State** | 34-dim: 9 queues × 3 features + 7 global features |
| **Action** | 5-dim continuous dispatching weights (PDR) |
| **Reward** | Holding cost: priority-weighted count of unfinished buildings |
| **Termination** | All revealed buildings repaired |

## Quick Start

### Install Dependencies

```bash
pip install gymnasium numpy stable-baselines3 pandas
```

### Run Baselines Evaluation

```bash
python -m code.evaluate --tag baseline_eval --n-seeds 10
```

### Train PPO Agent

```bash
python -m code.train
```

### Run Tests

```bash
python -m pytest code/tests/ -v
```

## Baseline Policies

| Policy | Strategy | Source |
|--------|----------|--------|
| LJF | Longest Job First | Lv et al. (2025) |
| SJF | Shortest Job First | Lv et al. (2025) |
| StrictPriority | Priority-level first | BNPB PERKA No.4/2012 |
| FIFO | First-in first-out | Standard fairness baseline |
| Random | Uniform random weights | Lower-bound reference |

## Data Sources

- **Building counts & recovery parameters**: Alisjahbana & Kiremidjian (2021)
- **Infrastructure data**: BPS NTB 2018 (hospitals, schools)
- **Batch arrival timeline**: BNPB 2018 disaster reports
- **Crew capacity**: FEMA P-58, RSMeans 2024, Indonesia POKMAS

# PA-CSAC Open-Source Release

This repository contains a cleaned open-source release of **PA-CSAC** (Probabilistic-Aware Constrained Soft Actor-Critic) for hybrid electric vehicle (HEV) eco-cruise control.

## Overview

PA-CSAC integrates:

- a probabilistic prediction module for preceding-vehicle behavior,
- uncertainty-aware state enhancement,
- constrained reinforcement learning,
- and a dynamic safety shield for closed-loop action correction.

The release is organized to support a reproducible workflow from prediction data preparation to control training and result validation, while excluding large temporary outputs and debugging artifacts from the original research workspace.

## Repository Layout

```text
PA_CSAC_OPEN_SOURCE/
├── algos/PA_CSAC/                 # Core PA-CSAC implementation (incl. PPO-Lagrangian baseline,
│                                   # sigma-source and shield on/off ablation switches)
├── prediction/
│   ├── data/                      # Raw trajectory data
│   ├── results/csv/               # Included processed control datasets + prediction-side
│   │                               # experiment summaries (ablation, min-gap grid, paper metrics)
│   ├── README.md                  # Prediction module notes
│   └── transformer_traffic_predict_optimized.py
├── results/
│   ├── seed22/                    # Archived experiment outputs (five random seeds)
│   ├── seed32/
│   ├── seed42/
│   ├── seed52/
│   ├── seed62/
│   ├── reeval_summary/            # Cross-seed re-evaluation mean±std summary tables
│   ├── reward_hparam_sensitivity_aggregated.csv
│   ├── split_conformal_analysis.csv
│   └── compute_cost.csv
├── scripts/
│   ├── run_quick.py               # Quick and paper-style experiment entry
│   ├── run_full.py                # Full single-seed training entry
│   ├── workflow_manager.py        # Data/output validation entry
│   ├── run_reward_hparam_sensitivity.py   # Reward hyper-parameter sensitivity analysis
│   ├── run_safe_rl_baseline.py            # Safe-RL baseline (PPO-Lagrangian) evaluation
│   ├── reeval_perscenario.py              # Per-scenario re-evaluation across seeds
│   ├── reeval_comp_ablation.py            # Component-ablation re-evaluation
│   ├── reeval_shieldoff_baselines.py      # Shield on/off ablation re-evaluation
│   ├── reeval_sigma_source.py             # Sigma-source ablation re-evaluation
│   ├── split_conformal_analysis.py        # Split conformal prediction analysis
│   ├── aggregate_main_table.py            # Cross-seed main-table aggregation
│   ├── plot_convergence_multiseed.py      # Multi-seed convergence figure
│   ├── plot_ablation_bars.py              # Ablation bar figure
│   └── collect_compute_cost.py            # Compute-cost statistics collection
├── utils/                         # Validation and utility functions
├── assets/
│   └── best_params.json           # Reference tuned parameters for PA-CSAC
├── config.yaml                    # Supplementary configuration snapshot
├── requirements.txt
├── REPRODUCE.md
└── LICENSE
```

## Environment Setup

Recommended environment:

- Python 3.12.5
- A PyTorch / torchvision build compatible with Python 3.12.5 and your local CUDA or CPU setup

Install dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Data Files

This release includes both:

- the raw trajectory file: `prediction/data/Zong_B_length_70.csv`
- the processed control datasets:
  - `prediction/results/csv/pcc_rl_prediction_dataset_for_control.csv`
  - `prediction/results/csv/pcc_rl_prediction_dataset.csv`

The processed control dataset is included to support quick reproduction. If needed, it can be regenerated from the raw data using the prediction script.

## Included Archived Results

This release also includes the previously generated experiment outputs for:

- `results/seed42`
- `results/seed52`
- `results/seed62`

These folders are the paper-aligned archived runs retained for inspection, comparison, and validation. They are stored as slimmed public-release archives, and their provenance plus retained-file scope are documented in `results/README.md`. Newly generated outputs are still written to `outputs/`.

## Quick Start

Run a quick validation:

```bash
python scripts/run_quick.py --smoke --seeds 42
```

Run the paper-style five-seed experiment:

```bash
python scripts/run_quick.py --paper --seeds 22,32,42,52,62
```

Run a single full training job (writes to `results/seed{N}`, overwriting the archived reference outputs; restore them with `git restore results/`):

```bash
python scripts/run_full.py --seed 42
```

Reproduce the cross-seed analyses after training (expects model checkpoints under `results/seed{N}/models/`):

```bash
python scripts/reeval_perscenario.py          # per-scenario re-evaluation + mean/std summaries
python scripts/aggregate_main_table.py        # cross-seed main table
python scripts/run_reward_hparam_sensitivity.py
python scripts/split_conformal_analysis.py
python scripts/collect_compute_cost.py
```

## Validation

Validate the input dataset and one archived result directory:

```bash
python scripts/workflow_manager.py ^
  --data_path prediction/results/csv/pcc_rl_prediction_dataset_for_control.csv ^
  --result_dir results/seed42
```

## Notes

- New experiment outputs are written to `outputs/` and are intentionally ignored by Git.
- Archived results for `seed42/52/62` are included because they correspond to the retained paper-result archives.
- The manuscript main-results table uses three-seed statistics aggregated from `seed42`, `seed52`, and `seed62`, rather than any single archived run alone.
- `config.yaml` is retained as a reference snapshot; the main public entry points are the scripts under `scripts/`.

## Citation

If you use this repository, please cite the corresponding paper and provide a link to this release.

## License

This project is released under the MIT License. See `LICENSE` for details.

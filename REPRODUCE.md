# Reproduction Guide

## 1. Install Dependencies

Use the fleet Python environment:

```bash
python --version
```

Expected target environment:

```text
Python 3.12.5
```

Install a PyTorch / torchvision build compatible with Python 3.12.5 first, then install the remaining dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## 2. Optional: Regenerate the Processed Control Dataset

The repository already includes the processed control dataset used by the public scripts:

- `prediction/results/csv/pcc_rl_prediction_dataset_for_control.csv`

If you want to regenerate the prediction outputs from the raw trajectory file, run:

```bash
python prediction/transformer_traffic_predict_optimized.py
```

After regeneration, confirm that `prediction/results/csv/` contains the control dataset.

## 3. Run Quick Validation

```bash
python scripts/run_quick.py --smoke --seeds 42
```

## 4. Reproduce the Paper-Style Main Results

```bash
python scripts/run_quick.py --paper --seeds 22,32,42,52,62
```

Outputs are written under `outputs/quick/`.

## 5. Run a Full Single-Seed Experiment

```bash
python scripts/run_full.py --seed 42
```

Outputs are written under `results/seed42/` (this overwrites the archived reference outputs for that seed; restore them with `git restore results/`). Model checkpoints are saved under `results/seed{N}/models/` and are not tracked by Git.

## 6. Re-evaluation and Cross-Seed Analyses

After training the five seeds, the cross-seed statistics in `results/reeval_summary/` can be regenerated (expects model checkpoints under `results/seed{N}/models/`):

```bash
python scripts/reeval_perscenario.py                 # per-scenario re-evaluation + mean/std summaries
python scripts/run_safe_rl_baseline.py               # PPO-Lagrangian baseline rows
python scripts/reeval_comp_ablation.py               # component ablation re-evaluation
python scripts/reeval_shieldoff_baselines.py         # shield on/off ablation
python scripts/reeval_sigma_source.py                # sigma-source ablation
python scripts/aggregate_main_table.py               # cross-seed main table
python scripts/run_reward_hparam_sensitivity.py      # reward hyper-parameter sensitivity
python scripts/split_conformal_analysis.py           # split conformal analysis
python scripts/collect_compute_cost.py               # compute-cost statistics
python scripts/plot_convergence_multiseed.py         # multi-seed convergence figure
python scripts/plot_ablation_bars.py                 # ablation bar figure
```

## 7. Validate Outputs

```bash
python scripts/workflow_manager.py ^
  --data_path prediction/results/csv/pcc_rl_prediction_dataset_for_control.csv ^
  --result_dir results/seed42
```

Validation reports are generated under `reports/`.

## 8. Included Archived Runs

The release already contains archived outputs for:

- `results/seed22`, `results/seed32`, `results/seed42`, `results/seed52`, `results/seed62`
- `results/reeval_summary/` (cross-seed mean±std re-evaluation summaries)
- `results/reward_hparam_sensitivity_aggregated.csv`, `results/split_conformal_analysis.csv`, `results/compute_cost.csv`

These are the paper-aligned archived runs included with the release. They are stored as slimmed public-release archives, and their source mapping is documented in `results/README.md`.

Important note:

- The manuscript main-results table is based on the five-seed statistics aggregated from `22,32,42,52,62`.
- A single archived `benchmark_summary.csv` corresponds to one seed only and should not be interpreted as the final aggregated paper result by itself.

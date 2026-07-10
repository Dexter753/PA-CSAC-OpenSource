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
python scripts/run_quick.py --paper --seeds 42,52,62
```

Outputs are written under `outputs/quick/`.

## 5. Run a Full Single-Seed Experiment

```bash
python scripts/run_full.py --seed 42
```

Outputs are written under `outputs/full/seed42/`.

## 6. Validate Outputs

```bash
python scripts/workflow_manager.py ^
  --data_path prediction/results/csv/pcc_rl_prediction_dataset_for_control.csv ^
  --result_dir results/seed42
```

Validation reports are generated under `reports/`.

## 7. Included Archived Runs

The release already contains archived outputs for:

- `results/seed42`
- `results/seed52`
- `results/seed62`

These are the paper-aligned archived runs included with the release. They are stored as slimmed public-release archives, and their source mapping is documented in `results/README.md`.

Important note:

- The manuscript main-results table is based on the three-seed statistics aggregated from `42,52,62`.
- A single archived `benchmark_summary.csv` corresponds to one seed only and should not be interpreted as the final aggregated paper result by itself.

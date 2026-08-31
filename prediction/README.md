# Prediction Module

This folder contains the probabilistic prediction component used by PA-CSAC.

## Included Files

```text
prediction/
├── transformer_traffic_predict_optimized.py
├── data/
│   └── Zong_B_length_70.csv
└── results/
    └── csv/
        ├── pcc_rl_prediction_dataset.csv
        ├── pcc_rl_prediction_dataset_for_control.csv
        ├── ablation_experiment.csv
        ├── min_gap_grid_experiment.csv
        ├── paper_ablation_summary.csv
        ├── paper_min_gap_summary.csv
        ├── paper_step_metrics.csv
        └── paper_summary_metrics.csv
```

## Purpose

- `Zong_B_length_70.csv` is the raw trajectory source file.
- `pcc_rl_prediction_dataset_for_control.csv` is the processed control dataset used directly by the public training scripts.
- `pcc_rl_prediction_dataset.csv` is also included for compatibility with existing utilities.
- `ablation_experiment.csv` / `paper_ablation_summary.csv` are the prediction-module ablation results (attention / architecture variants).
- `min_gap_grid_experiment.csv` / `paper_min_gap_summary.csv` are the minimum-time-gap grid sensitivity results.
- `paper_step_metrics.csv` / `paper_summary_metrics.csv` are the prediction-side step-level and summary metrics used in the paper.

## Usage

To regenerate prediction outputs from raw data:

```bash
python prediction/transformer_traffic_predict_optimized.py
```

The open-source release does not track temporary prediction logs, model checkpoints, or visualization outputs.

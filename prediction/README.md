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
        └── pcc_rl_prediction_dataset_for_control.csv
```

## Purpose

- `Zong_B_length_70.csv` is the raw trajectory source file.
- `pcc_rl_prediction_dataset_for_control.csv` is the processed control dataset used directly by the public training scripts.
- `pcc_rl_prediction_dataset.csv` is also included for compatibility with existing utilities.

## Usage

To regenerate prediction outputs from raw data:

```bash
python prediction/transformer_traffic_predict_optimized.py
```

The open-source release does not track temporary prediction logs, model checkpoints, or visualization outputs.

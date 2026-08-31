"""
Split-conformal calibration analysis for the deployed predictor's intervals.

Addresses the reviewer's Minor-4 point: the deployed Transformer+TFG+GMH
predictor reports PICP = 1.000 on the held-out test split (over-coverage).
This script quantifies how a post-hoc split-conformal recalibration of the
dispersion channel would tighten the intervals, using the same control-side
dataset that feeds the closed-loop evaluation:

  calibration split : rows whose 'split' column equals 'validation'
                      (held out from predictor training, used only here)
  evaluation split  : rows whose 'split' column equals 'test'
  conformity score  : s_i = |y_i - mu_i| / sigma_i   (normalized, per horizon)
  conformal quantile: q_hat = ceil((n+1)*(1-alpha)) / n empirical quantile of s
  recalibrated CI   : [mu - q_hat*sigma, mu + q_hat*sigma]
  target alpha      : 0.05 (95% intervals, matching the deployed 1.96*sigma)

Outputs
-------
results/split_conformal_analysis.csv  (per-horizon raw vs calibrated metrics)
stdout summary for the paper.

Usage (Windows):
  C:/Users/hp/anaconda3/envs/fleet/python.exe scripts/split_conformal_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv"
OUT_CSV = PROJECT_ROOT / "results" / "split_conformal_analysis.csv"
HORIZONS = [1, 3, 5]
ALPHA = 0.05
Z95 = 1.959963984540054  # exact two-sided 95% normal quantile


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: ceil((n+1)(1-alpha))-th order stat."""
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(s[rank - 1])


def interval_metrics(y, mu, half_width):
    covered = np.abs(y - mu) <= half_width
    picp = float(np.mean(covered))
    mpiw = float(np.mean(2.0 * half_width))
    return picp, mpiw


def main():
    head = pd.read_csv(CSV_PATH, nrows=2)
    have_pa = all(f"pa_pred_v_lead_t{s}" in head.columns and f"pa_std_v_lead_t{s}" in head.columns for s in HORIZONS)
    prefix = "pa_" if have_pa else ""
    pred_cols = [f"{prefix}pred_v_lead_t{s}" for s in HORIZONS]
    std_cols = [f"{prefix}std_v_lead_t{s}" for s in HORIZONS]
    cols = ["Vehicle_ID", "split", "v_lead"] + pred_cols + std_cols
    df = pd.read_csv(CSV_PATH, usecols=cols)
    print("using prefix:", repr(prefix))
    print("split counts:", df["split"].value_counts().to_dict())

    calib_mask = df["split"].astype(str).str.lower().str.startswith("val")
    test_mask = df["split"].astype(str).str.lower() == "test"
    if not calib_mask.any():
        raise ValueError("no validation rows found for split-conformal calibration")
    print(f"calibration rows: {int(calib_mask.sum())}, test rows: {int(test_mask.sum())}")

    rows = []
    for s in HORIZONS:
        mu_col = f"{prefix}pred_v_lead_t{s}"
        sd_col = f"{prefix}std_v_lead_t{s}"

        def build(frame):
            g = frame.sort_values(["Vehicle_ID", "Timestamp"]) if "Timestamp" in frame.columns else frame
            y = g.groupby("Vehicle_ID")["v_lead"].shift(-s)
            mu = g[mu_col]
            sd = g[sd_col]
            ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-6)
            return y[ok].to_numpy(float), mu[ok].to_numpy(float), sd[ok].to_numpy(float)

        y_c, mu_c, sd_c = build(df[calib_mask])
        y_t, mu_t, sd_t = build(df[test_mask])

        # Raw deployed intervals: mu +/- 1.96 * sigma
        picp_raw, mpiw_raw = interval_metrics(y_t, mu_t, Z95 * sd_t)

        # Split-conformal recalibration on the validation split
        scores = np.abs(y_c - mu_c) / sd_c
        q_hat = conformal_quantile(scores, ALPHA)
        picp_cal, mpiw_cal = interval_metrics(y_t, mu_t, q_hat * sd_t)

        # Equivalent sigma rescale factor relative to the deployed 1.96 level
        scale = q_hat / Z95
        rows.append({
            "horizon_s": s,
            "n_calib": len(y_c), "n_test": len(y_t),
            "picp_raw": round(picp_raw, 4), "mpiw_raw_mps": round(mpiw_raw, 4),
            "q_hat": round(q_hat, 4),
            "picp_calibrated": round(picp_cal, 4),
            "mpiw_calibrated_mps": round(mpiw_cal, 4),
            "sigma_scale_vs_deployed": round(scale, 4),
            "mpiw_reduction_pct": round(100.0 * (1.0 - mpiw_cal / mpiw_raw), 2),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()

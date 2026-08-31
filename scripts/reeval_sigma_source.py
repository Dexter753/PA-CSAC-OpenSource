# -*- coding: utf-8 -*-
"""
M3 sigma-source ablation: learned dispersion vs. empirical residual statistics.

Reviewer concern: the dominant channel is sigma, and sigma does not require a
GMM-Transformer -- per-horizon binned empirical residual std over the
calibration (validation) split provides a near-zero-cost dispersion signal.

This script (zero training):
  1. computes the per-horizon empirical residual std
        s_h = std( v_lead(t+h) - mu_h )   on the validation split
     from the deployed prediction dataset (same file as
     split_conformal_analysis.py);
  2. replays the five PA-CSAC seed checkpoints through the unified
     per-scenario pipeline with
        prediction_sigma_source = "empirical_residual"
     (mu stays learned; sigma is replaced by mean(s_{t+1}, s_{t+3}, s_{t+5}),
      matching the deployment convention sigma_raw = mean(std_t1, std_t3, std_t5));
  3. aggregates against the learned-sigma baseline (inj_base row of the
     unified re-evaluation).

Outputs:
  results/reeval_perscenario/sigma_source_summary.csv   (per-seed)
  results/reeval_perscenario/sigma_source_mean_std.csv  (aggregates + learned ref)

Idempotent: per-seed detail CSVs are reused when present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "algos" / "PA_CSAC"))

import reeval_perscenario as rp  # noqa: E402

import torch  # noqa: E402

CSV_PATH = PROJECT_ROOT / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv"
HORIZONS = [1, 3, 5]
BASELINE_ROW = "ErrorInjection_inj_base"  # learned-sigma reference in reeval_injection_mean_std.csv


def empirical_sigma_table() -> dict:
    """Per-horizon residual std over the validation (calibration) split."""
    head = pd.read_csv(CSV_PATH, nrows=2)
    prefix = "pa_" if all(f"pa_pred_v_lead_t{s}" in head.columns for s in HORIZONS) else ""
    cols = ["Vehicle_ID", "split", "v_lead"] + [f"{prefix}pred_v_lead_t{s}" for s in HORIZONS]
    df = pd.read_csv(CSV_PATH, usecols=cols)
    val = df[df["split"].astype(str).str.lower().str.startswith("val")].copy()
    if val.empty:
        raise ValueError("no validation rows in prediction dataset")
    table = {}
    for s in HORIZONS:
        mu = val[f"{prefix}pred_v_lead_t{s}"]
        y = val.sort_values(["Vehicle_ID"]).groupby("Vehicle_ID")["v_lead"].shift(-s)
        y = y.reindex(val.index)
        ok = np.isfinite(y) & np.isfinite(mu)
        table[f"t+{s}"] = float(np.std((y - mu)[ok].values, ddof=1))
    return table


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = empirical_sigma_table()
    print("empirical residual std (validation split):", {k: round(v, 4) for k, v in table.items()})
    sigma_emp = float(np.mean([table[f"t+{s}"] for s in HORIZONS]))
    print(f"deployment-convention sigma_emp = mean of the three bins = {sigma_emp:.4f} m/s")

    extra = {
        "prediction_sigma_source": "empirical_residual",
        "empirical_sigma_table": table,
    }

    rows = []
    out_dir = rp.OUT_DIR / "sigma_source" / "empirical"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in rp.SEEDS:
        ckpt = rp.PROJECT_ROOT / "results" / f"seed{seed}" / "models" / "pa_csac_best.pt"
        if not ckpt.exists():
            print(f"[seed {seed}][SigmaEmpirical] checkpoint missing: {ckpt}")
            continue
        rp.set_seed(int(seed))
        probe_env = rp.make_env(device, extra_params=dict(extra))
        obs_dim = int(probe_env.observation_space.shape[0])
        agent = rp.load_agent("PACSAC", ckpt, obs_dim, device, rp.PACSAC_KWARGS)
        policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
        df = rp.eval_method_seed("SigmaEmpirical", policy_fn, device, seed, out_dir,
                                 extra_params=dict(extra))
        rows.append(rp._summarize_seed(df, "SigmaEmpirical", seed))
        print(f"[seed {seed}][SigmaEmpirical] VSR cur={rows[-1]['vsr_current']:.3f} "
              f"fuel(cur)={rows[-1]['fuel_current_mean']:.2f}")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(rp.OUT_DIR / "sigma_source_summary.csv", index=False, encoding="utf-8-sig")

    agg = pd.DataFrame([{
        "method": "sigma_empirical", "n_seeds": len(rows),
        "sigma_emp_deploy": sigma_emp,
        "fuel_mean": sdf["fuel_current_mean"].mean() if rows else np.nan,
        "fuel_std": sdf["fuel_current_mean"].std(ddof=1) if rows else np.nan,
        "gap_mean": sdf["gap_rmse_current_mean"].mean() if rows else np.nan,
        "gap_std": sdf["gap_rmse_current_mean"].std(ddof=1) if rows else np.nan,
        "vsr_mean": sdf["vsr_current"].mean() if rows else np.nan,
        "vsr_std": sdf["vsr_current"].std(ddof=1) if rows else np.nan,
        "vr_mean": sdf["vr_hard_current_mean"].mean() if rows else np.nan,
        "vr_std": sdf["vr_hard_current_mean"].std(ddof=1) if rows else np.nan,
    }])

    # learned-sigma reference from the unified injection baseline (inj_base)
    ref = pd.read_csv(rp.OUT_DIR / "reeval_injection_mean_std.csv")
    ref = ref[ref["method"] == BASELINE_ROW]
    if not ref.empty:
        r = ref.iloc[0]
        agg = pd.concat([agg, pd.DataFrame([{
            "method": "sigma_learned", "n_seeds": int(r["n_seeds"]),
            "sigma_emp_deploy": np.nan,
            "fuel_mean": r["fuel_mean"], "fuel_std": r["fuel_std"],
            "gap_mean": r["gap_mean"], "gap_std": r["gap_std"],
            "vsr_mean": r["vsr_mean"], "vsr_std": r["vsr_std"],
            "vr_mean": r["vr_mean"], "vr_std": r["vr_std"],
        }])], ignore_index=True)

    agg.to_csv(rp.OUT_DIR / "sigma_source_mean_std.csv", index=False, encoding="utf-8-sig")
    print("\n==== Sigma-source ablation aggregates ====")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

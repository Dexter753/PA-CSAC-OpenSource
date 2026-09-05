"""Closed-loop re-evaluation of the MPC-L baseline (MPC with learned mean preview).

MPC-L is identical to the MPC baseline of the main comparison (same beam
search, weights, action set, dispersion channel, shield, and validity
protocol); the only difference is the horizon reference of the
preceding-vehicle speed: instead of holding the observed v_lead constant over
the 8-step horizon (persistence preview), each horizon step h uses the
deployed predictor's mean forecast, obtained by linear interpolation of the
t+1 / t+3 / t+5 predictions (prob9 columns 0..2 of the control-side dataset)
and holding the t+5 value beyond.

This isolates the mean-preview channel on the optimal-control consumer and
completes the 2x2 {persistence, learned} x {MPC, PA-CSAC} preview-value
matrix discussed in the main text.

The rollout protocol, env construction, validity gates, and per-scenario
output format replicate reeval_perscenario.py exactly; the rollout loop is
duplicated here (rather than reusing eval_method_seed) because the MPC-L
policy must read the live rollout environment's prediction columns at each
step, and eval_method_seed constructs its own env internally.

Output: results/reeval_perscenario/MPC-L_seed0_perscenario.csv
        results/reeval_perscenario/mpc_l_summary.txt

Usage:
  python scripts\reeval_mpc_l.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: train.py imports pyplot at module level

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_reeval_path = PROJECT_ROOT / "scripts" / "reeval_perscenario.py"
_spec = importlib.util.spec_from_file_location("reeval_perscenario", str(_reeval_path))
reeval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reeval)

_PROGRESS_LOG = PROJECT_ROOT / "reports" / "mpc_l_progress.log"


def _progress(msg):
    with open(_PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


_PROGRESS_LOG.write_text("module reeval imported\n", encoding="utf-8")

baseline_controller = reeval.baseline_controller
make_env = reeval.make_env
build_reset_scenarios = reeval.build_reset_scenarios
rollout_episode = reeval.rollout_episode
episode_stats = reeval.episode_stats
classify = reeval.classify
set_seed = reeval.set_seed
OUT_DIR = reeval.OUT_DIR
MIN_STEPS_RATIO = reeval.MIN_STEPS_RATIO

MPC_HORIZON = 8  # identical to the MPC baseline's horizon
_PREVIEW_KNOTS = np.array([1.0, 3.0, 5.0])  # t+1 / t+3 / t+5 forecast knots


def make_mpc_l_policy(env):
    """MPC-L policy reading the live rollout env's learned mean forecasts."""
    def policy(obs):
        preview = None
        try:
            prob9 = env.current_group_data["prob9"]
            idx = int(np.clip(int(env.step_idx), 0, len(prob9) - 1))
            mu = np.asarray(prob9[idx][0:3], dtype=np.float64)
            steps = np.arange(1, MPC_HORIZON + 1, dtype=np.float64)
            # linear interpolation between the knots; holds mu[2] beyond t+5
            preview = np.interp(steps, _PREVIEW_KNOTS, mu)
        except Exception:
            preview = None
        return baseline_controller(
            "MPC-L", obs, dt=float(getattr(env, "dt_episode", 1.0)),
            v_lead_preview=preview)
    return policy


def main():
    device = "cpu"  # deterministic controller; the env is CSV-driven
    set_seed(42)    # same deterministic seed as the other traditional baselines
    _progress("main: start")
    env = make_env(device)
    _progress("main: env created")
    policy_fn = make_mpc_l_policy(env)
    scenarios = build_reset_scenarios(env)
    min_steps = max(10, int(getattr(env, "episode_len", 70) * MIN_STEPS_RATIO))

    # resumable rollout: persist each scenario to a partial CSV so repeated
    # invocations continue where the previous run stopped (the sandboxed
    # execution environment enforces a wall-clock limit per process)
    out_csv = OUT_DIR / "MPC-L_seed0_perscenario.csv"
    partial_csv = OUT_DIR / "MPC-L_seed0_partial.csv"
    rows = []
    done_idx = set()
    if partial_csv.exists():
        try:
            df_old = pd.read_csv(partial_csv, encoding="utf-8-sig")
            rows = df_old.to_dict("records")
            done_idx = {int(r["scenario_idx"]) for r in rows}
            _progress(f"main: resumed {len(done_idx)} scenarios")
        except Exception:
            rows, done_idx = [], set()

    for k, sc in enumerate(scenarios):
        if k in done_idx:
            continue
        records = rollout_episode(env, policy_fn, sc)
        st = episode_stats(records)
        _progress(f"main: scenario {k} done ({st['steps']} steps, "
                  f"fuel={st['fuel_l_per_100km']:.3f})")
        st["scenario_idx"] = k
        st["group_idx"] = sc["group_idx"]
        v_leg, v_cur, chans = classify(st, min_steps)
        st["valid_legacy"] = v_leg
        st["valid_current"] = v_cur
        st["fail_channels"] = chans
        rows.append(st)
        pd.DataFrame(rows).to_csv(partial_csv, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows).sort_values("scenario_idx").reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---- summary (same aggregation as _summarize_seed, conditional basis) ----
    mask = df["valid_current"].astype(bool)
    lines = ["MPC-L (learned mean preview) - per-scenario replay summary",
             f"scenarios: {len(df)}   valid (current protocol): {int(mask.sum())}",
             "",
             "conditional (valid-only) means:",
             f"  fuel_l_per_100km : {df.loc[mask, 'fuel_l_per_100km'].mean():.4f}",
             f"  gap_rmse         : {df.loc[mask, 'gap_rmse'].mean():.4f}",
             f"  jerk_rmse        : {df.loc[mask, 'jerk_rmse'].mean():.4f}",
             f"  vr_hard          : {df.loc[mask, 'vr_hard'].mean():.4f}",
             "",
             "all-scenario means:",
             f"  fuel_l_per_100km : {df['fuel_l_per_100km'].mean():.4f}",
             f"  gap_rmse         : {df['gap_rmse'].mean():.4f}",
             f"  jerk_rmse        : {df['jerk_rmse'].mean():.4f}",
             f"  vr_hard          : {df['vr_hard'].mean():.4f}",
             "",
             "per-scenario rows:",
             df[["scenario_idx", "steps", "terminated_reason", "valid_current",
                 "fuel_l_per_100km", "gap_rmse", "jerk_rmse", "vr_hard"]].to_string(index=False)]
    report = "\n".join(lines)
    (OUT_DIR / "mpc_l_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved -> {out_csv}")
    print(f"saved -> {OUT_DIR / 'mpc_l_summary.txt'}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
        (PROJECT_ROOT / "reports" / "mpc_l_error.log").write_text(err, encoding="utf-8")
        raise

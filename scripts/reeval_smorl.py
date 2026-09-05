"""Closed-loop re-evaluation of the SMORL baseline (Zhu et al., 2022) under
the unified per-scenario protocol.

The rollout protocol, env construction, validity gates, and per-scenario
output format replicate reeval_perscenario.py exactly; the rollout loop is
duplicated here (rather than reusing eval_method_seed) because the SMORL
policy is a receding-horizon trajectory optimizer that must read the live
rollout environment's prediction columns (prob9 t+1/t+3/t+5 mean forecasts,
identical to MPC-L) and dt at each step.

Expected checkpoints: results/seed{N}/models/smorl.pt  (N in 22/32/42/52/62),
produced by scripts/run_smorl_baseline.py.

Output: results/reeval_perscenario/SMORL_seed{N}_perscenario.csv
        results/reeval_perscenario/smorl_summary.txt

Usage:
  python scripts\reeval_smorl.py
  python scripts\reeval_smorl.py --seeds 42 52 62
"""
import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: train.py imports pyplot at module level

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_reeval_path = PROJECT_ROOT / "scripts" / "reeval_perscenario.py"
_spec = importlib.util.spec_from_file_location("reeval_perscenario", str(_reeval_path))
reeval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reeval)

from algos.PA_CSAC.smorl import SMORL  # noqa: E402

_PROGRESS_LOG = PROJECT_ROOT / "reports" / "smorl_reeval_progress.log"
SEEDS = [22, 32, 42, 52, 62]
PLAN_HORIZON = 8
PREVIEW_KNOTS = np.array([1.0, 3.0, 5.0])


def _progress(msg):
    with open(_PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def make_smorl_policy(agent, env):
    """SMORL policy: trajectory optimizer reading the live env's learned
    mean forecasts and dt (preview construction identical to MPC-L)."""
    def policy(obs):
        preview = None
        try:
            prob9 = env.current_group_data["prob9"]
            idx = int(np.clip(int(env.step_idx), 0, len(prob9) - 1))
            mu = np.asarray(prob9[idx][0:3], dtype=np.float64)
            steps = np.arange(1, PLAN_HORIZON + 1, dtype=np.float64)
            preview = np.interp(steps, PREVIEW_KNOTS, mu)
        except Exception:
            preview = None
        return agent.select_action(
            obs, deterministic=True,
            dt=float(getattr(env, "dt_episode", 1.0)),
            v_lead_preview=preview)
    return policy


def run_seed(seed, device):
    out_csv = reeval.OUT_DIR / f"SMORL_seed{seed}_perscenario.csv"
    partial_csv = reeval.OUT_DIR / f"SMORL_seed{seed}_partial.csv"
    ckpt = PROJECT_ROOT / "results" / f"seed{seed}" / "models" / "smorl.pt"
    if not ckpt.exists():
        _progress(f"seed {seed}: checkpoint missing: {ckpt}")
        print(f"[skip] seed {seed}: checkpoint missing: {ckpt}")
        return None

    reeval.set_seed(int(seed))
    env = reeval.make_env(device)
    agent = SMORL(obs_dim=int(env.observation_space.shape[0]), act_dim=1, device=device,
                  env_params=env.params, horizon=PLAN_HORIZON)
    agent.load(str(ckpt))
    # 重评估阶段规划器的物理模型参数与重评估环境保持一致
    agent.env_params = env.params
    policy_fn = make_smorl_policy(agent, env)

    scenarios = reeval.build_reset_scenarios(env)
    min_steps = max(10, int(getattr(env, "episode_len", 70) * reeval.MIN_STEPS_RATIO))

    # resumable rollout: persist each scenario to a partial CSV so repeated
    # invocations continue where the previous run stopped
    rows = []
    done_idx = set()
    if partial_csv.exists():
        try:
            df_old = pd.read_csv(partial_csv, encoding="utf-8-sig")
            rows = df_old.to_dict("records")
            done_idx = {int(r["scenario_idx"]) for r in rows}
            _progress(f"seed {seed}: resumed {len(done_idx)} scenarios")
        except Exception:
            rows, done_idx = [], set()

    for k, sc in enumerate(scenarios):
        if k in done_idx:
            continue
        records = reeval.rollout_episode(env, policy_fn, sc)
        st = reeval.episode_stats(records)
        _progress(f"seed {seed}: scenario {k} done ({st['steps']} steps, "
                  f"fuel={st['fuel_l_per_100km']:.3f})")
        st["scenario_idx"] = k
        st["group_idx"] = sc["group_idx"]
        v_leg, v_cur, chans = reeval.classify(st, min_steps)
        st["valid_legacy"] = v_leg
        st["valid_current"] = v_cur
        st["fail_channels"] = chans
        rows.append(st)
        pd.DataFrame(rows).to_csv(partial_csv, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows).sort_values("scenario_idx").reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if partial_csv.exists():
        partial_csv.unlink()
    return reeval._summarize_seed(df, "SMORL", seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reeval.OUT_DIR.mkdir(parents=True, exist_ok=True)
    _progress(f"main: start (seeds={args.seeds}, device={device})")

    summary_rows = []
    for seed in args.seeds:
        row = run_seed(int(seed), device)
        if row is not None:
            summary_rows.append(row)
            print(f"[SMORL][seed={seed}] VSR legacy={row['vsr_legacy']:.3f} "
                  f"current={row['vsr_current']:.3f} "
                  f"fuel(cur)={row['fuel_current_mean']:.2f}")

    if not summary_rows:
        print("no seed completed; nothing to summarize")
        return
    sdf = pd.DataFrame(summary_rows)
    agg = sdf.agg({
        "fuel_current_mean": ["mean", "std"],
        "gap_rmse_current_mean": ["mean", "std"],
        "jerk_rmse_current_mean": ["mean", "std"],
        "vr_hard_current_mean": ["mean", "std"],
        "vsr_current": ["mean", "std"],
    }).round(4)
    lines = ["SMORL (Zhu et al., 2022) - unified per-scenario replay summary",
             f"seeds: {list(sdf['seed'])}",
             "",
             "conditional (valid-only) mean +/- std over seeds:",
             agg.to_string(),
             "",
             "per-seed rows:",
             sdf[["seed", "vsr_current", "n_valid_current", "fuel_current_mean",
                  "gap_rmse_current_mean", "jerk_rmse_current_mean",
                  "vr_hard_current_mean"]].to_string(index=False)]
    report = "\n".join(lines)
    (reeval.OUT_DIR / "smorl_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved -> {reeval.OUT_DIR / 'smorl_summary.txt'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
        (PROJECT_ROOT / "reports" / "smorl_reeval_error.log").write_text(err, encoding="utf-8")
        raise

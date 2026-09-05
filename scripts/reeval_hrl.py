"""Closed-loop re-evaluation of the HRL baseline (Zhang et al., 2023) under
the unified per-scenario protocol.

The rollout protocol, env construction, validity gates, and per-scenario
output format replicate reeval_perscenario.py exactly; the rollout loop is
duplicated here (rather than reusing eval_method_seed) because the HRL
policy is a hierarchical controller whose macro-period bookkeeping (goal
re-planning every K=10 steps at the boundary, linear goal interpolation
within the period) must follow the live rollout environment's step counter,
and the controller state must be reset at episode boundaries.

Expected checkpoints: results/seed{N}/models/hrl.pt  (N in 22/32/42/52/62),
produced by scripts/run_hrl_baseline.py.

Output: results/reeval_perscenario/HRL_seed{N}_perscenario.csv
        results/reeval_perscenario/hrl_summary.txt

Usage:
  python scripts\reeval_hrl.py
  python scripts\reeval_hrl.py --seeds 42
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

from algos.PA_CSAC.hrl import HRL, HRLController  # noqa: E402

_PROGRESS_LOG = PROJECT_ROOT / "reports" / "hrl_reeval_progress.log"
SEEDS = [22, 32, 42, 52, 62]
MACRO_PERIOD = 10  # identical to run_hrl_baseline.py


def _progress(msg):
    with open(_PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def make_hrl_policy(agent, env):
    """HRL policy: hierarchical execution with non-hierarchical deployment.

    The controller re-plans its (SOC, time-headway) goal at every macro-period
    boundary (K=10 steps) and interpolates linearly within the period; the
    controller state is reset when a new episode begins (env.step_idx==0)."""
    controller = HRLController(agent, macro_period=MACRO_PERIOD)

    def policy(obs):
        if int(getattr(env, "step_idx", 0)) == 0 and controller.step != 0:
            controller.reset()
        return controller.act(obs, deterministic=True)

    return policy


def run_seed(seed, device):
    out_csv = reeval.OUT_DIR / f"HRL_seed{seed}_perscenario.csv"
    partial_csv = reeval.OUT_DIR / f"HRL_seed{seed}_partial.csv"
    ckpt = PROJECT_ROOT / "results" / f"seed{seed}" / "models" / "hrl.pt"
    if not ckpt.exists():
        _progress(f"seed {seed}: checkpoint missing: {ckpt}")
        print(f"[skip] seed {seed}: checkpoint missing: {ckpt}")
        return None

    reeval.set_seed(int(seed))
    env = reeval.make_env(device)
    agent = HRL(obs_dim=int(env.observation_space.shape[0]), act_dim=1, device=device,
                macro_period=MACRO_PERIOD)
    agent.load(str(ckpt))
    policy_fn = make_hrl_policy(agent, env)

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
    return reeval._summarize_seed(df, "HRL", seed)


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
            print(f"[HRL][seed={seed}] VSR legacy={row['vsr_legacy']:.3f} "
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
    lines = ["HRL (Zhang et al., 2023) - unified per-scenario replay summary",
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
    (reeval.OUT_DIR / "hrl_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved -> {reeval.OUT_DIR / 'hrl_summary.txt'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
        (PROJECT_ROOT / "reports" / "hrl_reeval_error.log").write_text(err, encoding="utf-8")
        raise

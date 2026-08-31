"""
Unified-protocol per-scenario benchmark re-evaluation (all methods, all seeds).

Background
----------
The five-seed main experiment was produced across two environment code versions:
  - seeds 42/52/62 (legacy env): the per-step ``violation`` flag ORs the
    car-following upper-bound breach (viol_upper) into the hard-constraint
    event set, so a loose-following episode inflates violation_rate;
  - seeds 22/32 (current env): viol_upper is tracked separately (upper_rate
    gate <= 0.20) and violation_rate contains only lower/jerk/ttc events.

For an unimpeachable main table, every method x seed must be scored under one
protocol. This script reloads the trained checkpoints and replays the same 17
NGSIM test scenarios under the CURRENT environment code, logging per-scenario:
  - vr_hard  : mean(lower | jerk | ttc)          (current gate, <= 0.20)
  - vr_wide  : mean(lower | jerk | ttc | upper)  (legacy gate, <= 0.20)
  - upper_rate, gap_rmse, fuel, steps, termination
  - validity under both protocols

Outputs
-------
results/reeval_perscenario/{method}_seed{N}_perscenario.csv  (per-scenario)
results/reeval_perscenario/reeval_summary.csv               (aggregated)
results/reeval_perscenario/reeval_main_table_mean_std.csv   (mean +/- std, current protocol)

Usage (Windows cmd):
  python scripts\reeval_perscenario.py
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_ENV_PATH = PROJECT_ROOT / "algos" / "PA_CSAC" / "env.py"
_MODEL_PATH = PROJECT_ROOT / "algos" / "PA_CSAC" / "model.py"
_TRAIN_PATH = PROJECT_ROOT / "algos" / "PA_CSAC" / "train.py"

env_spec = importlib.util.spec_from_file_location("env_reeval", str(_ENV_PATH))
env_module = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(env_module)
CloudPCCEnv = env_module.CloudPCCEnv

model_spec = importlib.util.spec_from_file_location("model_reeval", str(_MODEL_PATH))
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)

train_spec = importlib.util.spec_from_file_location("train_reeval", str(_TRAIN_PATH))
train_module = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train_module)
baseline_controller = train_module.baseline_controller

from utils.utils import set_seed, summarize_metrics  # noqa: E402

import torch  # noqa: E402

# --- Fixed configuration (mirrors run_config_snapshot.csv / run_full.py) ----
CSV_PATH = str(PROJECT_ROOT / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv")
SEEDS = [22, 32, 42, 52, 62]
ENV_PARAMS = {"lower_violation_ratio": 0.94, "upper_cost_weight": 0.40}
VR_LIMIT = 0.20
UPPER_LIMIT = 0.20
GAP_LIMIT = 50.0
MIN_STEPS_RATIO = 0.40
OUT_DIR = PROJECT_ROOT / "results" / "reeval_perscenario"

LEARNERS = [
    ("PA-CSAC", "pa_csac_best.pt", "PACSAC", dict(cost_limit=0.30, use_prob_embedding=True, use_cost_constraint=True)),
    ("DDPG", "ddpg_best.pt", "DDPG", {}),
    ("TD3", "td3_best.pt", "TD3", {}),
    ("SAC", "sac_best.pt", "SAC", {}),
    ("PPO", "ppo.pt", "PPO", {}),
    ("PPO-Lag", "ppo_lagrangian.pt", "PPOLagrangian", dict(cost_limit=0.30)),
]
TRADITIONAL = ["ACC", "MPC", "LQR", "IDM"]

# --- Ablation variants (mirrors train.py STAGE 3, lines 3610-3760) -----------
# All variants share the PACSAC agent structure; only the env feature_mode and
# env params differ. Base params = ablation_env_params + info_fixed_params.
PACSAC_KWARGS = dict(cost_limit=0.30, use_prob_embedding=True, use_cost_constraint=True)
_ABLATION_BASE = {
    "upper_cost_weight": 0.40,
    "lower_violation_ratio": 0.94,
    "sigma_fixed_no_prediction": 1.8,
    "sigma_target_scale": 0.55,
    "sigma_target_bias": 0.30,
    "weight_mode": "fixed",
    "w_energy_fixed": 0.50,
    "w_safe_fixed": 0.50,
}
_ADAPTIVE = {"weight_mode": "adaptive", "sigma_ref": 1.8, "sigma_sharpness": 2.2,
             "w_safe_min": 0.35, "w_safe_max": 0.65}
_FIXED = {"weight_mode": "fixed", "w_energy_fixed": 0.50, "w_safe_fixed": 0.50}
ABLATIONS = [
    # (row_name, ckpt, feature_mode, extra_env_params)
    ("Ablation_no_prediction", "ablation_no_prediction_best.pt", "no_prediction", {}),
    ("Ablation_mean_prediction", "ablation_mean_prediction_best.pt", "mean_prediction", {}),
    ("Ablation_pa_csac_no_adaptive", "ablation_pa_csac_no_adaptive_best.pt", "pa_csac", {"ablation_sigma_mean": 1.8}),
    ("Ablation_pa_csac", "ablation_pa_csac_best.pt", "pa_csac", {"ablation_sigma_mean": 1.8, **_ADAPTIVE}),
    ("Ablation_mechanism_pa_adaptive", "ablation_mechanism_pa_adaptive_best.pt", "pa_csac", dict(_ADAPTIVE)),
    ("Ablation_mechanism_w_o_adaptive_weights", "ablation_mechanism_w_o_adaptive_weights_best.pt", "pa_csac", dict(_FIXED)),
    ("Ablation_mechanism_w_o_dyn_dsafe", "ablation_mechanism_w_o_dyn_dsafe_best.pt", "pa_csac", {**_ADAPTIVE, "k_sigma_dsafe": 0.0}),
    ("Ablation_mechanism_w_o_ada_and_dsafe", "ablation_mechanism_w_o_ada_and_dsafe_best.pt", "pa_csac", {**_FIXED, "k_sigma_dsafe": 0.0}),
]

# --- Prediction-error injections (mirrors train.py lines 3990-4007) ----------
# Uses the main PA-CSAC checkpoint; the three knobs are set explicitly in every
# case (unset ones take the neutral value) so runs are version-independent.
_INJ_NEUTRAL = {"pred_error_residual_scale": 1.00, "prediction_error_bias_mps": 0.00, "prediction_sigma_scale": 1.00}
def _inj(name, **kw):
    p = dict(_INJ_NEUTRAL); p.update(kw); return (f"ErrorInjection_{name}", "pa_csac_best.pt", "pa_csac", p)
INJECTIONS = [
    _inj("inj_base"),
    _inj("inj_residual_x0p5", pred_error_residual_scale=0.50),
    _inj("inj_residual_x1p3", pred_error_residual_scale=1.30),
    _inj("inj_residual_x1p6", pred_error_residual_scale=1.60),
    _inj("inj_residual_x2p0", pred_error_residual_scale=2.00),
    _inj("inj_residual_x3p0", pred_error_residual_scale=3.00),
    _inj("inj_bias_p05", prediction_error_bias_mps=0.50),
    _inj("inj_bias_p10", prediction_error_bias_mps=1.00),
    _inj("inj_bias_n05", prediction_error_bias_mps=-0.50),
    _inj("inj_bias_n10", prediction_error_bias_mps=-1.00),
    _inj("inj_sigma_x0p5", prediction_sigma_scale=0.50),
    _inj("inj_sigma_x1p5", prediction_sigma_scale=1.50),
    _inj("inj_sigma_x2p0", prediction_sigma_scale=2.00),
    _inj("inj_combined_mild", pred_error_residual_scale=1.30, prediction_error_bias_mps=0.30, prediction_sigma_scale=1.30),
    _inj("inj_combined", pred_error_residual_scale=1.50, prediction_error_bias_mps=0.50, prediction_sigma_scale=1.60),
    _inj("inj_combined_severe", pred_error_residual_scale=2.00, prediction_error_bias_mps=1.00, prediction_sigma_scale=2.00),
    # M3.3 persistence closed-loop contrast: residual scale 0 collapses the
    # predicted mean to the last observed lead speed at every horizon
    # (mu = v_lead), while the dispersion channel keeps the learned sigma.
    _inj("persistence", pred_error_residual_scale=0.00),
]


def build_reset_scenarios(env):
    n_groups = int(len(getattr(env, "processed_groups", [])))
    return [{"group_idx": int(i), "deterministic_reset": True, "soc0": 0.60} for i in range(n_groups)]


def make_env(device, feature_mode="pa_csac", extra_params=None):
    env = CloudPCCEnv(
        CSV_PATH,
        device=device,
        feature_mode=feature_mode,
        split_mode="test",
        strict_prediction_columns=True,
        strict_dedicated_prediction_columns=False,
    )
    params = dict(ENV_PARAMS)
    if extra_params:
        params.update(extra_params)
    env.params.update(params)
    return env


def rollout_episode(env, policy_fn, reset_option):
    obs, _ = env.reset(options=reset_option)
    done = False
    records = []
    while not done:
        action = policy_fn(obs)
        next_obs, reward, done, _, info = env.step(action)
        records.append(info)
        obs = next_obs
    return records


def episode_stats(records):
    n = len(records)
    lower = np.array([float(r.get("viol_lower", 0.0)) for r in records])
    upper = np.array([float(r.get("viol_upper", 0.0)) for r in records])
    jerk_v = np.array([float(r.get("viol_jerk", 0.0)) for r in records])
    ttc_v = np.array([float(r.get("viol_ttc", 0.0)) for r in records])
    gap_err = np.array([float(r.get("gap_error", 0.0)) for r in records])
    shield_act = np.array([float(r.get("shield_active", 0.0)) for r in records])
    rate_act = np.array([float(r.get("rate_limit_active", 0.0)) for r in records])
    sh_delta = np.array([float(r.get("shield_delta_total", 0.0)) for r in records])
    hard = ((lower + jerk_v + ttc_v) > 0).astype(float)
    wide = ((lower + jerk_v + ttc_v + upper) > 0).astype(float)
    m = summarize_metrics(records)
    return {
        "steps": n,
        "terminated_reason": str(records[-1].get("terminated_reason", "unknown")) if records else "empty",
        "collision": int(any(float(r.get("collision", 0.0)) > 0.5 for r in records)),
        "numeric_invalid": int(any(float(r.get("numeric_invalid", 0.0)) > 0.5 for r in records)),
        "vr_hard": float(np.mean(hard)) if n else float("nan"),
        "vr_wide": float(np.mean(wide)) if n else float("nan"),
        "upper_rate": float(np.mean(upper)) if n else float("nan"),
        "lower_rate": float(np.mean(lower)) if n else float("nan"),
        "jerk_rate": float(np.mean(jerk_v)) if n else float("nan"),
        "ttc_rate": float(np.mean(ttc_v)) if n else float("nan"),
        "gap_rmse": float(np.sqrt(np.mean(gap_err ** 2))) if n else float("nan"),
        "shield_rate": float(np.mean(shield_act)) if n else float("nan"),
        "rate_limit_rate": float(np.mean(rate_act)) if n else float("nan"),
        "shield_delta_abs_mean": float(np.mean(np.abs(sh_delta))) if n else float("nan"),
        "fuel_l_per_100km": float(m.get("fuel_l_per_100km", float("nan"))),
        "distance_km": float(m.get("distance_km", float("nan"))),
        "jerk_rmse": float(m.get("jerk_rmse", float("nan"))),
    }


def classify(row, min_steps):
    hard_fail = []
    if row["steps"] < min_steps:
        hard_fail.append("too_short")
    if row["terminated_reason"] == "dropout":
        hard_fail.append("dropout")
    if row["collision"] > 0:
        hard_fail.append("collision")
    if row["numeric_invalid"] > 0:
        hard_fail.append("numeric")
    legacy_gate = (row["vr_wide"] <= VR_LIMIT) and (row["gap_rmse"] <= GAP_LIMIT)
    current_gate = (row["vr_hard"] <= VR_LIMIT) and (row["upper_rate"] <= UPPER_LIMIT) and (row["gap_rmse"] <= GAP_LIMIT)
    valid_legacy = (not hard_fail) and legacy_gate
    valid_current = (not hard_fail) and current_gate
    chans = list(hard_fail)
    if not legacy_gate and not hard_fail:
        chans.append("legacy:" + ("vr_wide" if row["vr_wide"] > VR_LIMIT else "gap"))
    if not current_gate and not hard_fail:
        if row["vr_hard"] > VR_LIMIT:
            chans.append("cur:vr_hard")
        if row["upper_rate"] > UPPER_LIMIT:
            chans.append("cur:upper")
        if row["gap_rmse"] > GAP_LIMIT:
            chans.append("cur:gap")
    return valid_legacy, valid_current, ";".join(chans) if chans else "ok"


def _compat_rebuild_prob_embedding(agent, sd):
    """Rebuild prob_embedding when the checkpoint carries extra layers.

    Old-code checkpoints (seeds 42/52/62) were trained with an extra
    LayerNorm(emb_dim) between the last Linear and the final Tanh:
        old:  [Linear, LN, ReLU, Dropout, Linear, LN, ReLU, Linear(7), LN(8), Tanh(9)]
        new:  [Linear, LN, ReLU, Dropout, Linear, LN, ReLU, Linear(7), Tanh(8)]
    (verified against the pre-refactor model.py in the parent project).
    Extra param modules are inferred from weight ndim: 2D -> Linear,
    1D -> LayerNorm; inserted after the last shared param module, before the
    tail activation, then loaded with strict=True as a loud safety net.
    """
    import torch.nn as nn

    cur_seq = list(agent.prob_embedding.embedding)
    cur_param_idx = [i for i, m in enumerate(cur_seq) if hasattr(m, "weight")]
    ckpt_param_idx = sorted({int(k.split(".")[1]) for k in sd if k.endswith(".weight")})
    if ckpt_param_idx == cur_param_idx:
        return False
    extra = [i for i in ckpt_param_idx if i not in cur_param_idx]
    last_cp = cur_param_idx[-1]
    skeleton = cur_seq[: last_cp + 1]
    tail = cur_seq[last_cp + 1:]
    extras = []
    for i in extra:
        w = sd[f"embedding.{i}.weight"]
        if w.ndim == 2:
            extras.append(nn.Linear(int(w.shape[1]), int(w.shape[0])))
        elif w.ndim == 1:
            extras.append(nn.LayerNorm(int(w.shape[0])))
        else:
            raise ValueError(f"unsupported extra param shape at embedding.{i}: {tuple(w.shape)}")
    agent.prob_embedding.embedding = nn.Sequential(*(skeleton + extras + tail))
    return True


def load_agent(cls_name, ckpt_path, obs_dim, device, kwargs):
    cls = getattr(model_module, cls_name)
    agent = cls(obs_dim=obs_dim, act_dim=1, act_limit=2.0, device=device, **kwargs)
    try:
        agent.load(str(ckpt_path))
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "prob_embedding" in ckpt and hasattr(agent, "prob_embedding"):
            _compat_rebuild_prob_embedding(agent, ckpt["prob_embedding"])
            agent.load(str(ckpt_path))
        elif isinstance(ckpt, dict) and "pi" in ckpt:
            # PPO / PPO-Lag checkpoints store raw module state dicts.
            agent.pi.load_state_dict(ckpt["pi"])
            if "v" in ckpt:
                agent.v.load_state_dict(ckpt["v"])
            if "vc" in ckpt and hasattr(agent, "vc"):
                agent.vc.load_state_dict(ckpt["vc"])
        else:
            raise
    agent.device = torch.device(device)
    return agent


def eval_method_seed(method, policy_fn, device, seed, out_dir, feature_mode="pa_csac", extra_params=None):
    out_csv = out_dir / f"{method}_seed{seed}_perscenario.csv"
    # 断点续跑：若该 method×seed 的重放明细已存在且列齐全，直接复用，不重复 rollout
    if out_csv.exists():
        try:
            df_old = pd.read_csv(out_csv, encoding="utf-8-sig")
            if {"valid_legacy", "valid_current", "fuel_l_per_100km"} <= set(df_old.columns) and len(df_old) > 0:
                print(f"[skip] {method}_seed{seed} 已有重放明细，复用 {out_csv.name}")
                return df_old
        except Exception:
            pass
    set_seed(int(seed))
    env = make_env(device, feature_mode=feature_mode, extra_params=extra_params)
    scenarios = build_reset_scenarios(env)
    min_steps = max(10, int(getattr(env, "episode_len", 70) * MIN_STEPS_RATIO))
    rows = []
    for k, sc in enumerate(scenarios):
        records = rollout_episode(env, policy_fn, sc)
        st = episode_stats(records)
        st["scenario_idx"] = k
        st["group_idx"] = sc["group_idx"]
        v_leg, v_cur, chans = classify(st, min_steps)
        st["valid_legacy"] = v_leg
        st["valid_current"] = v_cur
        st["fail_channels"] = chans
        rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def _summarize_seed(df, method, seed):
    def _opt(col, only_valid=True):
        # 兼容旧明细 CSV（无盾统计列）时返回 NaN，不阻塞聚合
        if col not in df.columns:
            return float("nan")
        mask = df["valid_current"] if only_valid else pd.Series(True, index=df.index)
        return float(df.loc[mask, col].mean()) if mask.any() else float("nan")

    return {
        "method": method,
        "seed": seed,
        "vsr_legacy": float(df["valid_legacy"].mean()),
        "vsr_current": float(df["valid_current"].mean()),
        "n_valid_current": int(df["valid_current"].sum()),
        "fuel_current_mean": float(df.loc[df["valid_current"], "fuel_l_per_100km"].mean()) if df["valid_current"].any() else float("nan"),
        "gap_rmse_current_mean": float(df.loc[df["valid_current"], "gap_rmse"].mean()) if df["valid_current"].any() else float("nan"),
        "jerk_rmse_current_mean": float(df.loc[df["valid_current"], "jerk_rmse"].mean()) if df["valid_current"].any() else float("nan"),
        "vr_hard_current_mean": float(df.loc[df["valid_current"], "vr_hard"].mean()) if df["valid_current"].any() else float("nan"),
        "shield_rate_valid_mean": _opt("shield_rate"),
        "shield_rate_all_mean": _opt("shield_rate", only_valid=False),
        "rate_limit_rate_all_mean": _opt("rate_limit_rate", only_valid=False),
        "shield_delta_abs_all_mean": _opt("shield_delta_abs_mean", only_valid=False),
    }


def _aggregate(summary_rows, out_csv, title):
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT_DIR / out_csv.replace("_mean_std", ""), index=False, encoding="utf-8-sig")
    agg_spec = dict(
        n_seeds=("seed", "count"),
        fuel_mean=("fuel_current_mean", "mean"),
        fuel_std=("fuel_current_mean", "std"),
        gap_mean=("gap_rmse_current_mean", "mean"),
        gap_std=("gap_rmse_current_mean", "std"),
        jerk_mean=("jerk_rmse_current_mean", "mean"),
        jerk_std=("jerk_rmse_current_mean", "std"),
        vr_mean=("vr_hard_current_mean", "mean"),
        vr_std=("vr_hard_current_mean", "std"),
        vsr_mean=("vsr_current", "mean"),
        vsr_std=("vsr_current", "std"),
    )
    # 盾统计列（旧明细缺列时为 NaN，聚合仍可执行）
    for col in ("shield_rate_valid_mean", "shield_rate_all_mean", "rate_limit_rate_all_mean"):
        if col in sdf.columns and sdf[col].notna().any():
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")
    agg = sdf.groupby("method").agg(**agg_spec).reset_index()
    agg.to_csv(OUT_DIR / out_csv, index=False, encoding="utf-8-sig")
    print(f"\n==== {title} (current protocol, mean±std over seeds) ====")
    print(agg.to_string(index=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", default=["ablation", "injection", "shield"],
                        choices=["main", "ablation", "injection", "shield"],
                        help="要重放的实验部分（main=主表，已跑过可跳过）")
    args = parser.parse_args()
    parts = set(args.parts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Part 1: main comparison ----------
    if "main" in parts:
        summary_rows = []
        for seed in SEEDS:
            for method, ckpt_name, cls_name, kwargs in LEARNERS:
                ckpt = PROJECT_ROOT / "results" / f"seed{seed}" / "models" / ckpt_name
                if not ckpt.exists():
                    print(f"[seed {seed}][{method}] checkpoint missing: {ckpt}")
                    continue
                set_seed(int(seed))
                probe_env = make_env(device)
                obs_dim = int(probe_env.observation_space.shape[0])
                agent = load_agent(cls_name, ckpt, obs_dim, device, kwargs)
                policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
                df = eval_method_seed(method, policy_fn, device, seed, OUT_DIR)
                summary_rows.append(_summarize_seed(df, method, seed))
                print(f"[seed {seed}][{method}] VSR legacy={df['valid_legacy'].mean():.3f} "
                      f"current={df['valid_current'].mean():.3f} "
                      f"fuel(cur)={summary_rows[-1]['fuel_current_mean']:.2f}")

        for method in TRADITIONAL:
            set_seed(42)
            env = make_env(device)
            policy_fn = lambda o, n=method, e=env: baseline_controller(n, o, dt=float(getattr(e, "dt_episode", 1.0)))
            df = eval_method_seed(method, policy_fn, device, 0, OUT_DIR)
            summary_rows.append(_summarize_seed(df, method, 0))
            print(f"[{method}] (deterministic) VSR legacy={df['valid_legacy'].mean():.3f} "
                  f"current={df['valid_current'].mean():.3f}")

        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(OUT_DIR / "reeval_summary.csv", index=False, encoding="utf-8-sig")
        learn = sdf[sdf["seed"] > 0]
        agg = learn.groupby("method").agg(
            n_seeds=("seed", "count"),
            fuel_mean=("fuel_current_mean", "mean"), fuel_std=("fuel_current_mean", "std"),
            gap_mean=("gap_rmse_current_mean", "mean"), gap_std=("gap_rmse_current_mean", "std"),
            jerk_mean=("jerk_rmse_current_mean", "mean"), jerk_std=("jerk_rmse_current_mean", "std"),
            vr_mean=("vr_hard_current_mean", "mean"), vr_std=("vr_hard_current_mean", "std"),
            vsr_mean=("vsr_current", "mean"), vsr_std=("vsr_current", "std"),
        ).reset_index()
        trad = sdf[sdf["seed"] == 0].copy()
        trad_rows = pd.DataFrame({
            "method": trad["method"], "n_seeds": 0,
            "fuel_mean": trad["fuel_current_mean"], "fuel_std": 0.0,
            "gap_mean": trad["gap_rmse_current_mean"], "gap_std": 0.0,
            "jerk_mean": trad["jerk_rmse_current_mean"], "jerk_std": 0.0,
            "vr_mean": trad["vr_hard_current_mean"], "vr_std": 0.0,
            "vsr_mean": trad["vsr_current"], "vsr_std": 0.0,
        })
        full = pd.concat([agg, trad_rows], ignore_index=True)
        full.to_csv(OUT_DIR / "reeval_main_table_mean_std.csv", index=False, encoding="utf-8-sig")
        print("\n==== CROSS-METHOD SUMMARY (current protocol, mean±std over 5 seeds) ====")
        print(full.to_string(index=False))

    # ---------- Part 2: ablation variants ----------
    if "ablation" in parts:
        rows = []
        for seed in SEEDS:
            for method, ckpt_name, feature_mode, extra in ABLATIONS:
                ckpt = PROJECT_ROOT / "results" / f"seed{seed}" / "models" / ckpt_name
                if not ckpt.exists():
                    print(f"[seed {seed}][{method}] checkpoint missing: {ckpt}")
                    continue
                set_seed(int(seed))
                env_params = dict(_ABLATION_BASE)
                env_params.update(extra)
                probe_env = make_env(device, feature_mode=feature_mode, extra_params=env_params)
                obs_dim = int(probe_env.observation_space.shape[0])
                agent = load_agent("PACSAC", ckpt, obs_dim, device, PACSAC_KWARGS)
                policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
                df = eval_method_seed(method, policy_fn, device, seed, OUT_DIR,
                                      feature_mode=feature_mode, extra_params=env_params)
                rows.append(_summarize_seed(df, method, seed))
                print(f"[seed {seed}][{method}] VSR cur={rows[-1]['vsr_current']:.3f} "
                      f"fuel(cur)={rows[-1]['fuel_current_mean']:.2f}")
        if rows:
            _aggregate(rows, "reeval_ablation_mean_std.csv", "ABLATION SUMMARY")

    # ---------- Part 3: prediction-error injections ----------
    if "injection" in parts:
        rows = []
        for seed in SEEDS:
            for method, ckpt_name, feature_mode, inj_params in INJECTIONS:
                ckpt = PROJECT_ROOT / "results" / f"seed{seed}" / "models" / ckpt_name
                if not ckpt.exists():
                    print(f"[seed {seed}][{method}] checkpoint missing: {ckpt}")
                    continue
                set_seed(int(seed))
                probe_env = make_env(device, feature_mode=feature_mode, extra_params=inj_params)
                obs_dim = int(probe_env.observation_space.shape[0])
                agent = load_agent("PACSAC", ckpt, obs_dim, device, PACSAC_KWARGS)
                policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
                df = eval_method_seed(method, policy_fn, device, seed, OUT_DIR,
                                      feature_mode=feature_mode, extra_params=inj_params)
                rows.append(_summarize_seed(df, method, seed))
                print(f"[seed {seed}][{method}] VSR cur={rows[-1]['vsr_current']:.3f} "
                      f"fuel(cur)={rows[-1]['fuel_current_mean']:.2f}")
        if rows:
            _aggregate(rows, "reeval_injection_mean_std.csv", "ERROR-INJECTION SUMMARY")

    # ---------- Part 4: shield-off ablation (execution-time safety shield) ----------
    if "shield" in parts:
        rows = []
        for seed in SEEDS:
            ckpt = PROJECT_ROOT / "results" / f"seed{seed}" / "models" / "pa_csac_best.pt"
            if not ckpt.exists():
                print(f"[seed {seed}][ShieldOff] checkpoint missing: {ckpt}")
                continue
            set_seed(int(seed))
            probe_env = make_env(device, extra_params={"disable_shield": True})
            obs_dim = int(probe_env.observation_space.shape[0])
            agent = load_agent("PACSAC", ckpt, obs_dim, device, PACSAC_KWARGS)
            policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
            df = eval_method_seed("ShieldOff", policy_fn, device, seed, OUT_DIR,
                                  extra_params={"disable_shield": True})
            rows.append(_summarize_seed(df, "ShieldOff", seed))
            print(f"[seed {seed}][ShieldOff] VSR cur={rows[-1]['vsr_current']:.3f} "
                  f"fuel(cur)={rows[-1]['fuel_current_mean']:.2f}")
        if rows:
            _aggregate(rows, "reeval_shieldoff_mean_std.csv", "SHIELD-OFF ABLATION SUMMARY")


if __name__ == "__main__":
    main()

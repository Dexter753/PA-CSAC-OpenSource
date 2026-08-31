import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.PA_CSAC.env import CloudPCCEnv
from algos.PA_CSAC.train import _build_eval_reset_scenarios, evaluate, train_pa_csac
from utils.utils import set_seed


def resolve_csv_path(project_root: Path) -> Path:
    data_csv_for_control = project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv"
    data_csv_full = project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset.csv"
    if data_csv_for_control.exists():
        return data_csv_for_control
    if data_csv_full.exists():
        return data_csv_full
    raise FileNotFoundError(
        f"找不到决策数据集: {data_csv_for_control} 或 {data_csv_full}"
    )


def build_cases():
    return [
        {
            "group": "follow_far_coef",
            "label": "follow_far_coef_x0.5",
            "env_params": {"follow_far_coef": 0.15 * 0.5},
        },
        {
            "group": "follow_far_coef",
            "label": "follow_far_coef_x1.0",
            "env_params": {"follow_far_coef": 0.15 * 1.0},
        },
        {
            "group": "follow_far_coef",
            "label": "follow_far_coef_x1.5",
            "env_params": {"follow_far_coef": 0.15 * 1.5},
        },
        {
            "group": "k_sigma_dsafe",
            "label": "k_sigma_dsafe_x0.5",
            "env_params": {"k_sigma_dsafe": 0.9 * 0.5},
        },
        {
            "group": "k_sigma_dsafe",
            "label": "k_sigma_dsafe_x1.0",
            "env_params": {"k_sigma_dsafe": 0.9 * 1.0},
        },
        {
            "group": "k_sigma_dsafe",
            "label": "k_sigma_dsafe_x1.5",
            "env_params": {"k_sigma_dsafe": 0.9 * 1.5},
        },
        {
            "group": "shield_cost_scale",
            "label": "shield_cost_scale_x0.5",
            "env_params": {
                "shield_cost_push_coef": 0.18 * 0.5,
                "shield_cost_pull_coef": 0.08 * 0.5,
            },
        },
        {
            "group": "shield_cost_scale",
            "label": "shield_cost_scale_x1.0",
            "env_params": {
                "shield_cost_push_coef": 0.18 * 1.0,
                "shield_cost_pull_coef": 0.08 * 1.0,
            },
        },
        {
            "group": "shield_cost_scale",
            "label": "shield_cost_scale_x1.5",
            "env_params": {
                "shield_cost_push_coef": 0.18 * 1.5,
                "shield_cost_pull_coef": 0.08 * 1.5,
            },
        },
    ]


METRIC_COLS = ["fuel_l_per_100km", "gap_rmse", "jerk_rmse", "violation_rate", "valid_episode_ratio"]
METRIC_PREC = {
    "fuel_l_per_100km": 2,
    "gap_rmse": 2,
    "jerk_rmse": 2,
    "violation_rate": 3,
    "valid_episode_ratio": 2,
}


def run_seed(seed: int, train_steps: int, eval_episodes: int, two_stage: bool, force: bool) -> Path:
    """跑单个seed的全部case；已有结果的case自动跳过（断点续跑）。"""
    project_root = ROOT
    csv_path = resolve_csv_path(project_root)
    save_dir = project_root / "results" / f"seed{seed}" / "reward_hparam_sensitivity"
    model_dir = save_dir / "models"
    trace_dir = save_dir / "traces"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)
    result_csv = save_dir / "reward_hparam_sensitivity.csv"

    cases = build_cases()
    rows = []
    done_labels = set()
    if result_csv.exists() and not force:
        try:
            existing = pd.read_csv(result_csv)
            done_labels = {str(x) for x in existing["case_label"].tolist()}
            rows = existing.to_dict(orient="records")
            print(f"[Sensitivity][seed{seed}] 断点续跑: 已完成 {len(done_labels)}/9 个case, 跳过")
        except Exception as exc:
            print(f"[Sensitivity][seed{seed}] 读取已有结果失败({exc}), 从头开始")
            rows, done_labels = [], set()

    pending = [c for c in cases if str(c["label"]) not in done_labels]
    if not pending:
        print(f"[Sensitivity][seed{seed}] 全部case已完成, 无需重跑")
        return result_csv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(seed))

    base_env = CloudPCCEnv(
        str(csv_path),
        device=device,
        feature_mode="pa_csac",
        split_mode="test",
        strict_prediction_columns=False,
        strict_dedicated_prediction_columns=False,
    )
    eval_reset_scenarios = _build_eval_reset_scenarios(base_env, int(eval_episodes), soc0=0.60, seed=int(seed))

    for case in pending:
        label = str(case["label"])
        env_params = dict(case["env_params"])
        print(f"[Sensitivity][seed{seed}] Running {label} ...")

        agent, _, _ = train_pa_csac(
            str(csv_path),
            total_steps=int(train_steps),
            save_dir=str(model_dir),
            feature_mode="pa_csac",
            model_name=f"{label}.pt",
            history_tag=label,
            env_params_override=env_params,
            seed=int(seed),
            best_eval_episodes=16,
            strict_prediction_columns=False,
            strict_dedicated_prediction_columns=False,
            constraint_method="penalty",
            penalty_weight=1.0,
            prob_emb_lr=1e-3,
            two_stage=bool(two_stage),
            reward_scale=5.0,
            reward_bias=0.15,
            alpha_min=0.02,
            alpha_max=0.05,
            phase2_lr_ratio=0.025,
            shield_mismatch_coef=0.18,
        )

        env_eval = CloudPCCEnv(
            str(csv_path),
            device=device,
            feature_mode="pa_csac",
            split_mode="test",
            strict_prediction_columns=False,
            strict_dedicated_prediction_columns=False,
        )
        env_eval.params.update(env_params)
        metrics, _ = evaluate(
            env_eval,
            lambda o, a=agent: a.select_action(o, deterministic=True),
            episodes=len(eval_reset_scenarios),
            save_dir=str(save_dir),
            name=f"Sens_{label}",
            reset_options=list(eval_reset_scenarios),
            trace_dir=str(trace_dir),
        )

        row = {
            "seed": int(seed),
            "case_group": str(case["group"]),
            "case_label": label,
            "train_steps": int(train_steps),
            "eval_episodes": int(len(eval_reset_scenarios)),
            "fuel_l_per_100km": float(metrics.get("fuel_l_per_100km", float("nan"))),
            "gap_rmse": float(metrics.get("gap_rmse", float("nan"))),
            "jerk_rmse": float(metrics.get("jerk_rmse", float("nan"))),
            "violation_rate": float(metrics.get("violation_rate", float("nan"))),
            "valid_episode_ratio": float(metrics.get("valid_episode_ratio", float("nan"))),
            "paper_valid": bool(metrics.get("paper_valid", False)),
        }
        row.update(env_params)
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_csv, index=False, encoding="utf-8-sig")

    print(f"[Sensitivity][seed{seed}] 完成, 结果保存至: {result_csv}")
    return result_csv


def _fmt(mean, std, prec: int) -> str:
    """单seed时只输出数值；多seed时输出 mean $\\pm$ std。"""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"


def _level_text(case: dict) -> str:
    """生成表格 Level 列文本, 如 $\\times 0.5$ (0.075) 或 (0.09, 0.04)。"""
    label = str(case["label"])
    level = label.split("_")[-1]
    values = list(case["env_params"].values())
    if len(values) == 1:
        val_text = f"({values[0]:g})"
    else:
        val_text = "(" + ", ".join(f"{v:g}" for v in values) + ")"
    return f"$\\times {level[1:]}$ {val_text}"


GROUP_TEX = {
    "follow_far_coef": r"$\alpha_{\text{far}}$ (car-following)",
    "k_sigma_dsafe": r"$k_{\text{uncert}}$ (safety--uncertainty)",
    "shield_cost_scale": r"\multirow{3}{*}{\shortstack[l]{Shield cost scale\\$(k_{\text{push}}^{c}, k_{\text{pull}}^{c})$}}",
}


def aggregate(seed_csvs, out_path: Path) -> pd.DataFrame:
    """跨seed聚合 mean±std, 并打印可直接粘贴进论文 tab:reward_sensitivity 的行。"""
    frames = []
    for p in seed_csvs:
        if not Path(p).exists():
            print(f"[Sensitivity][aggregate] 缺少文件, 跳过: {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise RuntimeError("没有可聚合的结果文件")
    all_df = pd.concat(frames, ignore_index=True)

    cases = build_cases()
    agg_rows = []
    for case in cases:
        label = str(case["label"])
        sub = all_df[all_df["case_label"] == label]
        if sub.empty:
            continue
        row = {"case_group": case["group"], "case_label": label, "n_seeds": int(len(sub))}
        for m in METRIC_COLS:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Sensitivity][aggregate] 聚合结果已保存: {out_path}")

    print("% ==== LaTeX 表格体(可直接替换 tab:reward_sensitivity 的数据行) ====")
    by_group = {}
    for case, row in zip([c for c in cases if c["label"] in set(agg["case_label"])], agg_rows):
        by_group.setdefault(case["group"], []).append((case, row))
    first_group = True
    for group, items in by_group.items():
        if not first_group:
            print("            \\midrule")
        first_group = False
        for i, (case, row) in enumerate(items):
            group_cell = GROUP_TEX.get(group, group) if i == 0 else ""
            cells = " & ".join(
                _fmt(row[f"{m}_mean"], row[f"{m}_std"], METRIC_PREC[m]) for m in METRIC_COLS
            )
            if group_cell:
                if group == "shield_cost_scale":
                    print(f"            {group_cell}")
                    print(f"            & {_level_text(case)} & {cells} \\\\")
                else:
                    print(f"            \\multirow{{3}}{{*}}{{{group_cell}}}")
                    print(f"            & {_level_text(case)} & {cells} \\\\")
            else:
                print(f"            & {_level_text(case)} & {cells} \\\\")
    print("% ==========================================================")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62], help="要运行的seed列表")
    parser.add_argument("--train_steps", type=int, default=12000)
    parser.add_argument("--eval_episodes", type=int, default=17)
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略已有结果, 强制重跑全部case")
    parser.add_argument("--aggregate_only", action="store_true", help="只做跨seed聚合, 不训练")
    args = parser.parse_args()

    if args.aggregate_only:
        seed_csvs = [ROOT / "results" / f"seed{s}" / "reward_hparam_sensitivity" / "reward_hparam_sensitivity.csv" for s in args.seeds]
        aggregate(seed_csvs, ROOT / "results" / "reward_hparam_sensitivity_aggregated.csv")
        return

    seed_csvs = []
    for seed in args.seeds:
        seed_csvs.append(run_seed(int(seed), int(args.train_steps), int(args.eval_episodes), bool(args.two_stage), bool(args.force)))

    aggregate(seed_csvs, ROOT / "results" / "reward_hparam_sensitivity_aggregated.csv")


if __name__ == "__main__":
    main()

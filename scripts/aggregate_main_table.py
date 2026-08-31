# -*- coding: utf-8 -*-
"""主对比表聚合脚本（单一数据源，全表 mean±std 自动生成）。

数据来源：results/seed{...}/benchmark_summary.csv（每方法一行，主管线 evaluate() 实测）
输出：
  1) results/main_table_aggregate.csv（mean/std/逐种子原始值，全留痕可溯源）
  2) 控制台 LaTeX 表格行（直接粘贴进论文主表）
  3) 传统控制器跨种子确定性校验（应完全一致，max|Δ|≈0）

用法：
  python scripts/aggregate_main_table.py --seeds 42 52 62
  python scripts/aggregate_main_table.py --seeds 22 32 42 52 62
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# 主表方法行（与 benchmark_summary.csv 行标签一致）
METHODS = ["ACC", "MPC", "LQR", "IDM", "DDPG", "TD3", "SAC", "PPO", "PPO-Lag", "PA-CSAC"]
TRADITIONAL = ["ACC", "MPC", "LQR", "IDM"]
# 主表五列指标
METRICS = ["fuel_l_per_100km", "gap_rmse", "jerk_rmse", "violation_rate", "valid_episode_ratio"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--out", type=str, default="results/main_table_aggregate.csv")
    args = parser.parse_args()

    frames = {}
    for s in args.seeds:
        p = RESULTS / f"seed{s}" / "benchmark_summary.csv"
        if not p.exists():
            print(f"[warn] 缺少 {p}，跳过 seed{s}")
            continue
        frames[s] = pd.read_csv(p, index_col=0, encoding="utf-8-sig")
    if not frames:
        raise SystemExit("[error] 没有可用的 benchmark_summary.csv")

    rows = []
    for m in METHODS:
        vals = {met: [] for met in METRICS}
        for s, df in frames.items():
            if m not in df.index:
                print(f"[warn] seed{s} 缺少方法行 {m}")
                continue
            r = df.loc[m]
            for met in METRICS:
                vals[met].append(float(r[met]))
        n = len(vals[METRICS[0]])
        if n == 0:
            print(f"[warn] 方法 {m} 在所有种子中均缺失，跳过")
            continue
        row = {"method": m, "n_seeds": n}
        for met in METRICS:
            arr = np.array(vals[met], dtype=float)
            row[f"{met}_mean"] = round(float(arr.mean()), 4)
            row[f"{met}_std"] = round(float(arr.std(ddof=1)), 4) if n > 1 else 0.0
            row[f"{met}_values"] = ";".join(f"{v:.4f}" for v in arr)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = ROOT / args.out
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_path}（含逐种子原始值列，可溯源）")

    # 传统控制器确定性校验：同一方法跨种子应为逐位一致（无学习随机性）
    print("\n== 传统控制器跨种子确定性校验（max |Δ|） ==")
    for m in TRADITIONAL:
        for met in ["fuel_l_per_100km", "gap_rmse"]:
            per_seed = [float(frames[s].loc[m][met]) for s in frames if m in frames[s].index]
            if len(per_seed) > 1:
                max_diff = float(np.abs(np.diff(np.array(per_seed))).max())
                flag = "OK" if max_diff < 1e-9 else "[!] 存在差异，请核查"
                print(f"  {m:5s} {met:22s} max|Δ|={max_diff:.2e}  {flag}")

    # LaTeX 表格行（fuel/gap/jerk/VSR 两位小数，violation 三位小数，与论文主表格式一致）
    print("\n== LaTeX 表格行（mean $\\pm$ std over seeds） ==")
    for r in rows:
        fuel = f"{r['fuel_l_per_100km_mean']:.2f} $\\pm$ {r['fuel_l_per_100km_std']:.2f}"
        gap = f"{r['gap_rmse_mean']:.2f} $\\pm$ {r['gap_rmse_std']:.2f}"
        jerk = f"{r['jerk_rmse_mean']:.2f} $\\pm$ {r['jerk_rmse_std']:.2f}"
        viol = f"{r['violation_rate_mean']:.3f} $\\pm$ {r['violation_rate_std']:.3f}"
        vsr = f"{r['valid_episode_ratio_mean']:.2f} $\\pm$ {r['valid_episode_ratio_std']:.2f}"
        print(f"            {r['method']} & ${fuel}$ & ${gap}$ & ${jerk}$ & ${viol}$ & ${vsr}$ \\\\")

    # 每种子数值清单（审计用）
    print("\n== 逐种子原始值（审计留痕） ==")
    for r in rows:
        print(f"  {r['method']:8s} fuel: {r['fuel_l_per_100km_values']} | VSR: {r['valid_episode_ratio_values']}")


if __name__ == "__main__":
    main()

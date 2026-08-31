# -*- coding: utf-8 -*-
"""PPO-Lagrangian 对偶变量轨迹图（约束机制实证，供 3.3 节/补充材料引用）。

数据来源：results/seed{...}/histories/ppo_lagrangian_lambda.csv
  （train_ppo_lagrangian 每 epoch 记录：epoch, step, jc_hat, dual_lam）

输出：results/fig_lambda_trajectory.png（双子图）
  (a) λ 轨迹：单调上升、无 settle epoch（对偶无界增长的直接证据）
  (b) Ĵ_c 与阈值 d=0.30（对数轴）：约束回报估计始终高于阈值 20-80 倍

用法：python scripts/plot_lambda_trajectory.py --seeds 42 52 62
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COST_LIMIT = 0.30  # 与 PA-CSAC 的 C_lim 统一（论文 3.1 节）
COLORS = {22: "#9467bd", 32: "#ff7f0e", 42: "#1f77b4", 52: "#d62728", 62: "#2ca02c"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 32, 42, 52, 62])
    parser.add_argument("--out", type=str, default="results/fig_lambda_trajectory.png")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)

    plotted = 0
    for s in args.seeds:
        p = RESULTS / f"seed{s}" / "histories" / "ppo_lagrangian_lambda.csv"
        if not p.exists():
            print(f"[warn] 缺少 {p}，跳过 seed{s}")
            continue
        df = pd.read_csv(p)
        c = COLORS.get(s)
        axes[0].plot(df["epoch"], df["dual_lam"], marker="o", ms=3, lw=1.4, color=c, label=f"seed {s}")
        axes[1].plot(df["epoch"], df["jc_hat"].clip(lower=1e-3), marker="o", ms=3, lw=1.4, color=c, label=f"seed {s}")
        plotted += 1

    if plotted == 0:
        raise SystemExit("[error] 没有可用的 lambda 轨迹文件")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Dual variable $\lambda$")
    axes[0].set_title(r"(a) $\lambda$ trajectory")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(COST_LIMIT, color="k", ls="--", lw=1.2, label=r"threshold $d = 0.30$")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"Constraint return estimate $\hat{J}_c$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"(b) $\hat{J}_c$ vs. threshold")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.3, which="both")

    out_path = ROOT / args.out
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}（{plotted} seeds）")
    # 打印摘要统计，便于论文引用数字
    for s in args.seeds:
        p = RESULTS / f"seed{s}" / "histories" / "ppo_lagrangian_lambda.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        print(f"  seed{s}: epochs={len(df)}, lambda {df['dual_lam'].iloc[0]:.3f} -> {df['dual_lam'].iloc[-1]:.3f}, "
              f"Jc range [{df['jc_hat'].min():.2f}, {df['jc_hat'].max():.2f}], "
              f"min(Jc/d)={df['jc_hat'].min() / COST_LIMIT:.1f}x")


if __name__ == "__main__":
    main()

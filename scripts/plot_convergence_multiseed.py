# -*- coding: utf-8 -*-
"""绘制全种子（22/32/42/52/62）训练收敛对比图，用于论文图9（fig:training_convergence）。

数据来源：results/seed{22,32,42,52,62}/histories/{method}_history.csv（列：episode_reward, step）
输出：results/fig9.png（论文正文引用的文件名，直接覆盖）

图中元素（与论文 3.5 节描述一致）：
  - 实线：各步骤上五种子的均值
  - 阴影带：mean ± 1σ（种子间离散度）

配色与绘制层次（审稿可读性设计）：
  - Okabe-Ito 色盲安全调色板，五色相距最大化
  - 两遍绘制：先画全部 σ 带（低 zorder），再画全部均值线，
    PA-CSAC 最后绘制并置于最高 zorder，确保 "ours" 不被覆盖
  - 不画逐种子细线：σ 带已编码种子离散度，25 条半透明细线
    会与带叠印产生混色，降低可读性

运行：python scripts/plot_convergence_multiseed.py
"""
from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_SEEDS = [22, 32, 42, 52, 62]
# 增大平滑窗口至 200 回合（≈14000 步，约占训练 23%），
# 有效消除 episode 内噪声导致的虚假带宽，保留真实的种子间策略差异
SMOOTH_WIN = 200

METHODS = [
    ("pa_csac", "PA-CSAC (ours)", "#D55E00"),  # vermillion，最高 zorder
    ("sac", "SAC", "#0072B2"),                 # blue
    ("td3", "TD3", "#009E73"),                 # bluish green
    ("ddpg", "DDPG", "#CC79A7"),               # reddish purple
    ("ppo", "PPO", "#E69F00"),                 # orange
]


def load_smoothed(method: str, seed: int) -> pd.DataFrame:
    path = RESULTS / f"seed{seed}" / "histories" / f"{method}_history.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.sort_values("step").reset_index(drop=True)
    df["reward_smooth"] = (
        df["episode_reward"].rolling(SMOOTH_WIN, min_periods=1, center=True).mean()
    )
    return df[["step", "reward_smooth"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="绘图的种子列表，默认 22 32 42 52 62")
    args = parser.parse_args()
    seeds = args.seeds

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    stats = []
    for method, label, color in METHODS:
        curves = []
        for seed in seeds:
            try:
                curves.append(load_smoothed(method, seed))
            except FileNotFoundError:
                print(f"[warn] 缺少 {method} seed{seed} 历史文件，跳过该种子")
        if not curves:
            print(f"[warn] {method} 无任何种子数据，跳过")
            continue

        step_max = min(c["step"].max() for c in curves)
        grid = np.arange(curves[0]["step"].min(), step_max + 1, 70.0)
        y = np.vstack([
            np.interp(grid, c["step"], c["reward_smooth"]) for c in curves
        ])
        mean = y.mean(axis=0)
        std = y.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        stats.append((method, label, color, grid / 1000.0, mean, std))

        tail = slice(-max(10, len(grid) // 50), None)
        print(f"{label:16s} 末期回报 mean={mean[tail].mean():8.2f}  "
              f"range=[{y.min(axis=0)[tail].min():8.2f}, "
              f"{y.max(axis=0)[tail].max():8.2f}]")

    # 第一遍：全部 σ 带（低 zorder，避免叠印遮盖均值线）
    for _, _, c, x, mean, std in stats:
        ax.fill_between(x, mean - std, mean + std,
                        color=c, alpha=0.18, lw=0, zorder=2)
    # 第二遍：全部均值线；PA-CSAC（stats[0]）最后画、最高 zorder
    for _, label, c, x, mean, std in reversed(stats):
        is_ours = label.startswith("PA-CSAC")
        ax.plot(x, mean, color=c, lw=2.5 if is_ours else 2.0,
                zorder=6 if is_ours else 4, label=label)

    ax.set_xlabel(r"Training steps ($\times 10^3$)")
    ax.set_ylabel("Smoothed episode return")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(alpha=0.2, ls="--")
    ax.set_xlim(left=0)
    fig.tight_layout()

    out = RESULTS / "fig9.png"
    fig.savefig(out, dpi=300)
    print(f"\n[done] 已保存 {out}")


if __name__ == "__main__":
    main()

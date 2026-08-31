# -*- coding: utf-8 -*-
"""闭环消融配置柱状图（论文 4.3 节 fig:ablation_prediction_module，fig7.png）。

单一数据源：results/reeval_perscenario/reeval_ablation_mean_std.csv
（reeval_perscenario.py 从 5 种子 checkpoint 统一重放 17 场景后的聚合输出，
与正文 Table 8/Table 9 及补充材料 Table S7 的 mean±std 完全一致）。

绘制 7 个入论文配置（预测信息族 3 个 + 自适应机制族 4 个）：
  2×2 面板：Fuel / Gap RMSE / Jerk RMSE / Valid Scenario Ratio
  柱高 = 五种子均值，误差棒 = ±1σ；每配置独立配色（冷色=预测信息族，暖色=自适应机制族），
  族间另以竖直分隔线区分。
  注意：reeval CSV 中的 Ablation_pa_csac_no_adaptive 为旧口径筛选遗留配置，
  未进入当前论文的 Table 8/9/S7，故不绘制。

运行：python scripts/plot_ablation_bars.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "reeval_perscenario" / "reeval_ablation_mean_std.csv"
OUT = ROOT / "results" / "fig7.png"

# (CSV method, 显示名, 柱色)  颜色按族区分：冷色系=预测信息族（Table 8），暖色系=自适应机制族（Table 9），
# 每个配置独立配色，7 色互异且色盲可辨（深->浅编码消融程度）
CONFIGS = [
    ("Ablation_pa_csac",                        "Full dist.\ninput",        "#1D3557"),
    ("Ablation_mean_prediction",                "Mean only",                "#457B9D"),
    ("Ablation_no_prediction",                  "No future info",           "#2A9D8F"),
    ("Ablation_mechanism_pa_adaptive",          "Complete\nmodel",          "#E9C46A"),
    ("Ablation_mechanism_w_o_adaptive_weights", "w/o adaptive\nweights",    "#F4A261"),
    ("Ablation_mechanism_w_o_dyn_dsafe",        "w/o dynamic\nsafe dist.",  "#E76F51"),
    ("Ablation_mechanism_w_o_ada_and_dsafe",    "w/o both",                 "#9B2226"),
]

# (列前缀, 面板标题, y轴标签, 值格式, y上限余量)
PANELS = [
    ("fuel", "Fuel (L/100km)",      "Fuel (L/100km)",      "{:.2f}", 1.15),
    ("gap",  "Gap RMSE (m)",        "Gap RMSE (m)",        "{:.1f}", 1.15),
    ("jerk", "Jerk RMSE (m/s$^3$)", "Jerk RMSE (m/s$^3$)", "{:.2f}", 1.25),
    ("vsr",  "Valid Scenario Ratio", "Valid Scenario Ratio", "{:.2f}", 1.12),
]


def main() -> None:
    df = pd.read_csv(AGG).set_index("method")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
    })

    labels = [c[1] for c in CONFIGS]
    colors = [c[2] for c in CONFIGS]
    x = np.arange(len(CONFIGS))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for ax, (prefix, title, ylabel, fmt, headroom) in zip(axes.flat, PANELS):
        mean = np.array([df.loc[m, f"{prefix}_mean"] for m, _, _ in CONFIGS])
        std = np.array([df.loc[m, f"{prefix}_std"] for m, _, _ in CONFIGS])

        ax.bar(x, mean, width=0.62, color=colors, edgecolor="black", linewidth=0.6,
               yerr=std, error_kw=dict(ecolor="black", elinewidth=1.0, capsize=3.0))
        for xi, mu, sd in zip(x, mean, std):
            ax.text(xi, mu + sd + 0.03 * mean.max(), fmt.format(mu),
                    ha="center", va="bottom", fontsize=8.5)

        # 预测信息族 vs 自适应机制族 分隔线（第3、4柱之间）
        ax.axvline(2.5, color="gray", ls="--", lw=0.9, alpha=0.7)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max((mean + std).max() * headroom, 1e-6))
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"[OK] 写出 {OUT}")


if __name__ == "__main__":
    main()

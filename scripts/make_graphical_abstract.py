# -*- coding: utf-8 -*-
"""生成 Elsevier 投稿用图形摘要（Graphical Abstract）。

规范要点：
  - 最小宽度 531 像素，推荐宽高比约 5:4，300 dpi（输出 1950x1560 px）
  - 概念性流程图，文字精简，突出方法闭环与核心结果
  - 输出：results/fig_graphical_abstract.png（投稿系统单独上传，不入正文）

运行：python scripts/make_graphical_abstract.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fig_graphical_abstract.png"

# ---- 全局样式（与正文图一致：衬线字体、简洁风格）----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

C_PRED = "#457B9D"     # 预测模块蓝
C_AGENT = "#E63946"    # PA-CSAC 红
C_PLANT = "#2A9D8F"    # 车辆/环境绿
C_RESULT = "#6D597A"   # 结果紫
C_GRAY = "#4A4E69"


def box(ax, x, y, w, h, text, fc, fs=9.5, tc="white", bold=True, style="round,pad=0.02,rounding_size=0.025"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        facecolor=fc, edgecolor="none", mutation_aspect=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p0, p1, color=C_GRAY, lw=2.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
        linestyle=ls, zorder=1, shrinkA=2, shrinkB=2))


def gmm_icon(ax, x0, y0, w, h):
    """小型 GMM 分布示意：两个分量 + 混合密度 + 95% 区间阴影。"""
    t = np.linspace(-3, 3, 200)
    g1 = np.exp(-(t + 1.1) ** 2 / (2 * 0.55 ** 2))
    g2 = 0.65 * np.exp(-(t - 1.3) ** 2 / (2 * 0.75 ** 2))
    mix = g1 + g2
    x = x0 + (t + 3) / 6 * w
    y = y0 + mix / mix.max() * h
    # 95% 区间阴影
    ax.fill_between(x, y0, y, color=C_PRED, alpha=0.18, zorder=1)
    ax.plot(x, y, color=C_PRED, lw=2.0, zorder=2)
    ax.plot(x0 + (t + 3) / 6 * w, y0 + g1 / mix.max() * h,
            color=C_PRED, lw=1.0, ls="--", alpha=0.65, zorder=2)
    ax.plot(x0 + (t + 3) / 6 * w, y0 + g2 / mix.max() * h,
            color=C_PRED, lw=1.0, ls=":", alpha=0.65, zorder=2)
    # 区间端线
    for q in (0.1, 0.9):
        ax.plot([x0 + q * w] * 2, [y0, y0 + 0.32 * h],
                color=C_PRED, lw=1.2, alpha=0.9, zorder=2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ============ 顶部标题条（一句话方法概括）============
    ax.text(5.0, 7.62, "PA-CSAC: Probabilistic-Aware Constrained SAC for HEV Eco-Cruise Control",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#1D3557")

    # ============ 左：概率预测模块 ============
    box(ax, 0.30, 5.30, 2.55, 0.62, "NGSIM preceding-vehicle\ntrajectories", C_PRED, fs=8.5)
    box(ax, 0.30, 4.05, 2.55, 0.95, "Transformer probabilistic\nprediction (TFG + GMH)", C_PRED, fs=8.5)
    gmm_icon(ax, 0.62, 2.05, 1.95, 1.35)
    ax.text(1.58, 1.72, r"Gaussian-mixture $\mu,\ \sigma$", ha="center",
            fontsize=8, color=C_PRED)
    ax.text(1.58, 1.38, "future-speed distribution", ha="center",
            fontsize=8, color=C_PRED)
    arrow(ax, (1.58, 5.28), (1.58, 5.03))
    arrow(ax, (1.58, 4.03), (1.58, 3.46))

    # ============ 中：PA-CSAC 智能体（三层堆叠 + 闭环）============
    box(ax, 3.72, 5.55, 2.85, 0.85, "Probabilistic feature\nembedding (20-D state)", C_AGENT, fs=8.5)
    box(ax, 3.72, 4.28, 2.85, 0.85, "SAC policy +\nfixed-penalty constraint", C_AGENT, fs=8.5)
    box(ax, 3.72, 3.01, 2.85, 0.85, "Uncertainty-aware\nsafety shield", C_AGENT, fs=8.5)
    arrow(ax, (5.14, 5.53), (5.14, 5.15))
    arrow(ax, (5.14, 4.26), (5.14, 3.88))
    ax.text(5.14, 2.55, "PA-CSAC agent", ha="center",
            fontsize=9.5, fontweight="bold", color=C_AGENT)
    arrow(ax, (2.87, 3.44), (3.70, 4.70), rad=-0.12)          # 预测→嵌入

    # ============ 右：结果面板 ============
    box(ax, 7.28, 4.42, 2.42, 2.60, "", C_RESULT, fs=9)
    ax.text(8.49, 6.68, "Outcomes (5 seeds)", ha="center",
            fontsize=9, fontweight="bold", color="white", zorder=3)
    ax.text(8.49, 6.06, "Fuel\n6.43 L/100km", ha="center",
            fontsize=9.5, fontweight="bold", color="white", zorder=3)
    ax.text(8.49, 5.28, "Valid scenario ratio\n0.92 / 17 scenarios", ha="center",
            fontsize=9, color="white", zorder=3)
    ax.text(8.49, 4.62, "Gap RMSE 21.14 m", ha="center",
            fontsize=9, color="white", zorder=3)
    arrow(ax, (6.59, 4.70), (7.26, 5.30))

    # ============ 下：HEV 动力总成 + 闭环反馈 ============
    box(ax, 3.72, 0.30, 2.85, 0.80, "HEV powertrain\n+ traffic environment", C_PLANT, fs=8.5)
    arrow(ax, (5.14, 2.99), (5.14, 1.14))                       # 盾→车辆（修正动作）
    ax.text(5.32, 2.05, "action", fontsize=8, color=C_GRAY)
    arrow(ax, (3.70, 0.70), (1.58, 0.70), rad=0.0)              # 车辆→数据（左向）
    arrow(ax, (1.58, 0.70), (1.58, 1.26))                       # 数据→GMM 图标
    ax.text(2.60, 0.47, "closed-loop state feedback", fontsize=8, color=C_GRAY)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"[done] saved {OUT}")


if __name__ == "__main__":
    main()

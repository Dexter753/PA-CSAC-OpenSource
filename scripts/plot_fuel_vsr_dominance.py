# -*- coding: utf-8 -*-
"""油耗-场景有效率（fuel-VSR）支配散点图（论文 3.2 节 fig:fuel_vsr_dominance）。

单一数据源：results/reeval_perscenario/ 下的统一重评估输出
（reeval_perscenario.py 重载全部10方法×5种子 checkpoint，在当前口径下重放17场景）：
  - reeval_main_table_mean_std.csv : 各方法 mean±std（学习方法5种子，传统控制器单次确定性评估）
  - reeval_summary.csv             : 学习方法逐种子原始值（用于小标记）

输出：results/fig6.png（论文正文引用的文件名，直接覆盖）
  - x 轴：Valid Scenario Ratio（右=更好）；y 轴反转，上=更省油
  - 实心大标记=学习方法均值，小标记=逐种子值，空心=传统控制器
  - 学习方法逐方法独立配色（Okabe-Ito，与图9一致），传统控制器灰色空心
  - 着色区域=联合支配区 {VSR ≥ 0.8, fuel ≤ IDM}（以最强传统控制器的油耗为界）
  - PA-CSAC 用星形高亮：唯一进入支配区的学习方法

运行：python scripts/plot_fuel_vsr_dominance.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
REEVAL_DIR = ROOT / "results" / "reeval_perscenario"
AGG = REEVAL_DIR / "reeval_main_table_mean_std.csv"
PERSEED = REEVAL_DIR / "reeval_summary.csv"
OUT = ROOT / "results" / "fig6.png"

VSR_GATE = 0.8
XMAX = 1.04
YMAX = 27.5  # 容纳 PPO-Lag 的 +1σ 误差棒（18.99+7.65=26.64）

# method -> (显示名, 是否学习方法, 标记, 颜色)
# 学习方法逐方法独立配色（Okabe-Ito 色盲安全调色板，与图9训练收敛图保持一致），
# 传统控制器统一灰色空心标记，PA-CSAC 用 vermillion 星形高亮
STYLE = {
    "ACC":     ("ACC", False, "o", "#7f7f7f"),
    "MPC":     ("MPC", False, "o", "#7f7f7f"),
    "LQR":     ("LQR", False, "o", "#7f7f7f"),
    "IDM":     ("IDM", False, "o", "#7f7f7f"),
    "DDPG":    ("DDPG", True, "o", "#CC79A7"),
    "TD3":     ("TD3", True, "o", "#009E73"),
    "SAC":     ("SAC", True, "o", "#0072B2"),
    "PPO":     ("PPO", True, "o", "#E69F00"),
    "PPO-Lag": ("PPO-Lagrangian", True, "o", "#56B4E9"),
    "PA-CSAC": ("PA-CSAC (ours)", True, "*", "#D55E00"),
}


def main() -> None:
    df = pd.read_csv(AGG).set_index("method")
    per = pd.read_csv(PERSEED)
    fuel_gate = float(df.loc["IDM", "fuel_mean"])  # 支配区油耗界=最强传统控制器

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # 联合支配区 {VSR >= 0.8, fuel <= IDM}：y 轴反转后为右上角
    ax.add_patch(Rectangle((VSR_GATE, 0), XMAX - VSR_GATE, fuel_gate,
                           facecolor="#2A9D8F", alpha=0.10, lw=0, zorder=0))
    ax.axhline(fuel_gate, color="#2A9D8F", ls=":", lw=1.0, alpha=0.8)
    ax.axvline(VSR_GATE, color="#2A9D8F", ls=":", lw=1.0, alpha=0.8)
    ax.text(VSR_GATE - 0.015, 0.6,
            rf"joint region $\{{$VSR $\geq$ 0.8, fuel $\leq$ IDM ({fuel_gate:.1f})$\}}$",
            ha="right", va="top", fontsize=9, color="#1D6A5E")

    # 逐种子小标记（仅学习方法，seed>0），颜色与所属方法一致
    for method, is_rl, _, color in STYLE.values():
        if not is_rl:
            continue
        sub = per[(per["method"] == method) & (per["seed"] > 0)]
        if sub.empty:
            continue
        ax.scatter(sub["vsr_current"], sub["fuel_current_mean"],
                   s=14, color=color, alpha=0.35, lw=0, zorder=2)

    # 均值±σ 大标记
    for method, row in df.iterrows():
        if method not in STYLE:
            continue
        label, is_rl, marker, color = STYLE[method]
        ours = method == "PA-CSAC"
        has_err = bool(is_rl) and row["fuel_std"] > 0
        ax.errorbar(row["vsr_mean"], row["fuel_mean"],
                    xerr=row["vsr_std"] if has_err else 0,
                    yerr=row["fuel_std"] if has_err else 0,
                    fmt=marker, ms=15 if ours else 7,
                    mfc=(color if is_rl else "none"), mec=color, ecolor=color,
                    elinewidth=1.1, capsize=2.5, lw=0,
                    zorder=5 if ours else 3)

    # 逐点直接标注（y 轴反转：dy<0 = 显示在点上方）
    x = {m: df.loc[m, "vsr_mean"] for m in STYLE if m in df.index}
    y = {m: df.loc[m, "fuel_mean"] for m in STYLE if m in df.index}
    ANN = {  # method: (dx, dy, ha)
        "SAC":     (-0.022, 0.0, "right"),    # 左侧（TD3 在其右上）
        "TD3":     (0.022, 0.0, "left"),      # 右侧
        "DDPG":    (0.024, 0.0, "left"),      # 右侧（避开自身竖棒与SAC种子点）
        "PA-CSAC": (-0.030, 0.55, "right"),   # 星形左下，留在支配区内侧
        "IDM":     (-0.022, 0.0, "right"),    # 四个传统控制器均位于 x=1.0，左侧标注
        "ACC":     (-0.022, 0.0, "right"),
        "LQR":     (-0.022, 0.0, "right"),
        "MPC":     (-0.022, 0.0, "right"),
        "PPO":     (0.024, 0.0, "left"),
        "PPO-Lag": (0.024, 0.0, "left"),
    }
    for method, (dx, dy, ha) in ANN.items():
        ours = method == "PA-CSAC"
        color = STYLE[method][3]
        ax.annotate(STYLE[method][0], (x[method] + dx, y[method] + dy),
                    ha=ha, va="center", fontsize=9.5,
                    color=color, fontweight="bold" if ours else "normal")

    ax.set_xlabel("Valid Scenario Ratio (mean over five seeds)")
    ax.set_ylabel(r"Fuel consumption (L/100km, mean over five seeds)")
    ax.set_xlim(-0.03, XMAX)
    ax.set_ylim(0, YMAX)
    ax.invert_yaxis()  # 上=更省油，右上=支配角
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}")
    print(f"[gate] VSR>= {VSR_GATE}, fuel <= IDM ({fuel_gate:.2f} L/100km)")

    # 打印图中数字，供论文核对
    for method in STYLE:
        if method in df.index:
            r = df.loc[method]
            print(f"  {method:8s} VSR={r['vsr_mean']:.2f}±{r['vsr_std']:.2f}  "
                  f"fuel={r['fuel_mean']:.2f}±{r['fuel_std']:.2f}")


if __name__ == "__main__":
    main()

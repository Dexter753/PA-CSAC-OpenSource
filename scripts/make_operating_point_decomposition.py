# -*- coding: utf-8 -*-
"""工况点分解：条件油耗 = 匀速巡航下限 + 瞬态跟踪成本
（论文 fig:op_decomposition / tab:op_decomposition，输出 results/fig7.png）。

单一数据源：results/reeval_perscenario/speed_fuel_table.csv 的 valid（条件）口径，
与正文 Table 1 的条件统计完全同基（学习方法 = 各自有效场景子集上的逐种子均值，
确定性控制器 = 单次评估）。

方法：对每个方法（学习型逐种子），取其条件平均车速 v̄，用与闭环评估完全一致的
ECMS 物理模型（utils.utils.hev_parallel_energy：同一初始 SOC=soc_target、同一
回合时长、同一 33 MJ/L 等效能折算）计算"以 v̄ 匀速巡航"的每公里油耗下限
（steady-cruise floor）；条件油耗减去该下限即瞬态跟踪成本
（transient tracking cost：实际跟驰中加减速吞吐与间距调节的额外能耗）。

解读要点（脚本运行后在终端打印）：
  - 松跟驰家族（DDPG/TD3/SAC/SMORL/HRL）的巡航下限（~2.9-3.2 L/100km）
    并不低于 PA-CSAC（~2.74），其低油耗完全来自瞬态项的缺失（不执行跟踪任务）；
  - 在真正执行跟踪任务的方法中，PA-CSAC 的瞬态成本最小（约为 IDM 的一半）。

输出：
  results/operating_point_decomposition.csv : 全部 13 方法的分解统计（mean±std）
  results/fig7.png                          : 堆叠柱状图（灰=巡航下限，彩色=瞬态成本）

运行：python scripts/make_operating_point_decomposition.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import hev_parallel_energy, industry_hev_params  # noqa: E402

SPEED_TABLE = ROOT / "results" / "reeval_perscenario" / "speed_fuel_table.csv"
OUT_CSV = ROOT / "results" / "operating_point_decomposition.csv"
OUT_FIG = ROOT / "results" / "fig7.png"

# 等效能 -> 升折算，与 utils.py 中 fuel_l_per_100km 的 33e6 J/L 完全一致
J_PER_LITRE = 33e6

# 显示名与配色和 fig6（plot_fuel_vsr_dominance.py）保持一致
STYLE = {
    "ACC":     ("ACC", "#7f7f7f"),
    "MPC":     ("MPC", "#7f7f7f"),
    "MPC-L":   ("MPC-L", "#7f7f7f"),
    "LQR":     ("LQR", "#7f7f7f"),
    "IDM":     ("IDM", "#7f7f7f"),
    "DDPG":    ("DDPG", "#CC79A7"),
    "TD3":     ("TD3", "#009E73"),
    "SAC":     ("SAC", "#0072B2"),
    "PPO":     ("PPO", "#E69F00"),
    "PPO-Lag": ("PPO-Lagrangian", "#56B4E9"),
    "SMORL":   ("SMORL", "#8C564B"),
    "HRL":     ("HRL", "#999933"),
    "PA-CSAC": ("PA-CSAC (ours)", "#D55E00"),
}
LEARNING = {"DDPG", "TD3", "SAC", "PPO", "PPO-Lag", "SMORL", "HRL", "PA-CSAC"}


def cruise_floor_l_per_100km(v_kmh: float, t_s: float, params: dict) -> float:
    """以恒定速度 v_kmh 巡航 t_s 秒的每公里等效油耗下限（同一 ECMS 模型与折算）。"""
    v = float(v_kmh) / 3.6
    steps = int(round(float(t_s)))
    soc = float(params["soc_target"])
    energy_j = 0.0
    for _ in range(steps):
        e, soc = hev_parallel_energy(v, 0.0, soc, dt=1.0, params=params)
        energy_j += e
    dist_km = v * steps / 1000.0
    return (energy_j / J_PER_LITRE) / dist_km * 100.0


def main() -> None:
    params = industry_hev_params()
    d = pd.read_csv(SPEED_TABLE)
    d = d[d["basis"] == "valid"].copy()
    d["floor"] = [cruise_floor_l_per_100km(v, t, params)
                  for v, t in zip(d["v_mean_kmh"], d["t_mean_s"])]
    d["transient"] = d["fuel_l_100km"] - d["floor"]

    rows = []
    for m, sub in d.groupby("method"):
        n = len(sub)
        rows.append({
            "method": m, "n_seeds": n,
            "v_mean": sub["v_mean_kmh"].mean(),
            "v_std": sub["v_mean_kmh"].std(ddof=1) if n > 1 else 0.0,
            "floor_mean": sub["floor"].mean(),
            "floor_std": sub["floor"].std(ddof=1) if n > 1 else 0.0,
            "transient_mean": sub["transient"].mean(),
            "transient_std": sub["transient"].std(ddof=1) if n > 1 else 0.0,
            "total_mean": sub["fuel_l_100km"].mean(),
            "total_std": sub["fuel_l_100km"].std(ddof=1) if n > 1 else 0.0,
        })
    agg = pd.DataFrame(rows).set_index("method").sort_values("total_mean")
    agg.to_csv(OUT_CSV, encoding="utf-8-sig")
    print("=== operating-point decomposition (conditional/valid basis, L/100km) ===")
    print(agg.round(3).to_string())

    # ---------------- fig7：堆叠柱状图 ----------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(agg))
    floors = agg["floor_mean"].to_numpy()
    transients = agg["transient_mean"].to_numpy()
    colors = [STYLE[m][1] for m in agg.index]

    ax.bar(x, floors, width=0.62, color="#cfcfcf", edgecolor="#7f7f7f",
           lw=0.5, zorder=2)
    ax.bar(x, transients, width=0.62, bottom=floors, color=colors,
           edgecolor="black", lw=0.4, zorder=2)

    # 总量标注 + ±1σ 误差棒（学习方法）
    for i, m in enumerate(agg.index):
        tot, std = agg.loc[m, "total_mean"], float(agg.loc[m, "total_std"])
        has_err = (m in LEARNING) and std > 0
        if has_err:
            ax.errorbar(i, tot, yerr=std, fmt="none", ecolor="black",
                        elinewidth=1.0, capsize=2.5, zorder=4)
        ax.text(i, tot + (std if has_err else 0.0) + 0.30, f"{tot:.1f}",
                ha="center", va="bottom", fontsize=8.5)

    handles = [
        Patch(facecolor="#cfcfcf", edgecolor="#7f7f7f", lw=0.5,
              label="Steady-cruise floor at own mean speed"),
        Patch(facecolor="#7f7f7f", edgecolor="black", lw=0.4,
              label="Transient tracking cost"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[m][0] for m in agg.index],
                       rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Conditional fuel consumption (L/100km)")
    ax.set_ylim(0, 27.5)  # 与 fig6 一致，容纳 PPO-Lag 的 +1σ
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_FIG}")


if __name__ == "__main__":
    main()

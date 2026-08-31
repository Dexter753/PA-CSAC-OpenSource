# -*- coding: utf-8 -*-
"""计算成本分析数据收集（全部来自真实文件/实测，禁止虚构）。

来源：
  1) 参数量：加载 results/seed42/models/*.pt，递归统计 checkpoint 内所有张量的参数量
     （即部署时需要加载的参数总量；ACC/MPC/LQR/IDM 为模型式方法，无学习参数）
  2) 推理延迟：results/seed{22,32,42,52,62}/benchmark_summary.csv 的 infer_time_ms 列
     （主管线 evaluate() 实测，五种子 mean±std，与论文 "five benchmark passes" 口径一致）
  3) 训练耗时（可选 --time-segment N，默认关闭）：两点差分法测稳态训练速率——
     每方法实测 N//2 与 N 两段（用 train.py 内置 train_time_seconds 精确计时，
     已剔除数据加载/CUDA初始化等一次性开销），差分得到 s/千步并线性外推
     60k 步小时数（论文中标注 extrapolated）。建议 N=8000。
     注意：不带 --time-segment 重跑时，训练耗时列自动沿用 CSV 中已有的实测值，
     不会覆盖为空（防止训练耗时数据源断链）。

输出：results/compute_cost.csv + 控制台 LaTeX 表格行
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
SEEDS = [22, 32, 42, 52, 62]  # 推理延迟统计用全部五个种子（与论文 five benchmark passes 口径一致）
TRADITIONAL = ["ACC", "MPC", "LQR", "IDM"]
RL_CKPTS = {
    "PA-CSAC": "pa_csac.pt",
    "SAC": "sac.pt",
    "TD3": "td3.pt",
    "DDPG": "ddpg.pt",
    "PPO": "ppo.pt",
    "PPO-Lag": "ppo_lagrangian.pt",
}


def count_params(ckpt_path: Path) -> float:
    """递归统计 checkpoint 内所有张量的参数总量（百万）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def _count(obj):
        total = 0
        if torch.is_tensor(obj):
            return obj.numel()
        if isinstance(obj, dict):
            for v in obj.values():
                total += _count(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                total += _count(v)
        return total

    return _count(ckpt) / 1e6


def collect_params() -> dict:
    out = {}
    models_dir = RESULTS / "seed42" / "models"
    for name, fn in RL_CKPTS.items():
        p = models_dir / fn
        if p.exists():
            try:
                out[name] = count_params(p)
            except Exception as e:
                print(f"[warn] {name}: 参数统计失败 {e}")
        else:
            print(f"[warn] {name}: 未找到 {p}")
    return out


def collect_latency(seeds) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        p = RESULTS / f"seed{seed}" / "benchmark_summary.csv"
        if not p.exists():
            print(f"[warn] 缺少 {p}")
            continue
        df = pd.read_csv(p, index_col=0, encoding="utf-8-sig")
        if "infer_time_ms" in df.columns:
            for algo, v in df["infer_time_ms"].items():
                rows.append({"algo": str(algo), "infer_time_ms": float(v)})
    if not rows:
        return pd.DataFrame()
    lat = pd.DataFrame(rows)
    return lat.groupby("algo")["infer_time_ms"].agg(["mean", "std", "count"]).reset_index()


def _run_segment(fn, tmp_dir: str, steps: int) -> float:
    """跑一段训练，返回内部精确计时 train_time_seconds（剔除CUDA初始化等一次性开销）。"""
    agent = fn(tmp_dir, int(steps))
    # 各 train_* 返回 (agent, history, steps) 或 agent 本身
    ag = agent[0] if isinstance(agent, tuple) else agent
    return float(getattr(ag, "train_time_seconds", 0.0))


def collect_train_time(segment_steps: int) -> dict:
    """两点差分法测各方法稳态训练速率，外推 60k 步小时数。

    方法学（透明可复现）：
      - 每方法跑两个等比段：short = segment_steps//2 与 long = segment_steps，
        使用内部 train_time_seconds（train.py 已内置，从数据加载/CUDA初始化之后起算）。
      - steady_s_per_step = (T_long - T_short) / (long - short)，
        差分消去一次性初始化开销；基线的 ACC 热启动按 total//3 比例混入少量
        纯前向步（无梯度），使估计略偏乐观，表注须披露。
      - hours_60k = steady_s_per_step * 60000 / 3600（线性外推，论文标注 extrapolated）。
      - PA-CSAC 显式 two_stage=False，与主管线实测口径一致（run_full.py）。
    """
    from algos.PA_CSAC.train import train_ddpg, train_pa_csac, train_ppo, train_ppo_lagrangian, train_sac, train_td3

    csv_path = None
    for name in ("pcc_rl_prediction_dataset_for_control.csv", "pcc_rl_prediction_dataset.csv"):
        p = ROOT / "prediction" / "results" / "csv" / name
        if p.exists():
            csv_path = str(p)
            break
    if csv_path is None:
        print("[warn] 找不到数据集，跳过训练耗时实测")
        return {}

    jobs = {
        "DDPG": lambda d, s: train_ddpg(csv_path, total_steps=s, save_dir=d, seed=999),
        "TD3": lambda d, s: train_td3(csv_path, total_steps=s, save_dir=d, seed=999),
        "SAC": lambda d, s: train_sac(csv_path, total_steps=s, save_dir=d, seed=999),
        "PPO": lambda d, s: train_ppo(csv_path, total_steps=s, save_dir=d, seed=999),
        "PPO-Lag": lambda d, s: train_ppo_lagrangian(csv_path, total_steps=s, save_dir=d, seed=999),
        "PA-CSAC": lambda d, s: train_pa_csac(csv_path, total_steps=s, save_dir=d, seed=999, two_stage=False),
    }

    long_steps = int(segment_steps)
    short_steps = long_steps // 2
    if short_steps < 2000:
        print(f"[warn] --time-segment 过短（{long_steps}），至少 4000 才能保证差分段进入稳定训练区")
        return {}

    out = {}
    for name, fn in jobs.items():
        t_short = t_long = None
        try:
            tmp_short = Path(tempfile.mkdtemp(prefix=f"cost_{name}_s_"))
            t_short = _run_segment(fn, str(tmp_short), short_steps)
        except Exception as e:
            print(f"[warn] {name}: 短段({short_steps}步)计时失败: {e}")
        finally:
            shutil.rmtree(tmp_short, ignore_errors=True)
        try:
            tmp_long = Path(tempfile.mkdtemp(prefix=f"cost_{name}_l_"))
            t_long = _run_segment(fn, str(tmp_long), long_steps)
        except Exception as e:
            print(f"[warn] {name}: 长段({long_steps}步)计时失败: {e}")
        finally:
            shutil.rmtree(tmp_long, ignore_errors=True)

        if t_short is not None and t_long is not None and t_long > t_short > 0:
            steady = (t_long - t_short) / max(long_steps - short_steps, 1)
            out[name] = {
                "s_per_1k": steady * 1000.0,
                "hours_60k_extrapolated": steady * 60000.0 / 3600.0,
                "t_short_s": round(t_short, 1),
                "t_long_s": round(t_long, 1),
            }
            print(f"[Time] {name:8s}: T({short_steps})={t_short:.1f}s, T({long_steps})={t_long:.1f}s "
                  f"-> 稳态 {out[name]['s_per_1k']:.2f} s/千步, 外推60k ≈ {out[name]['hours_60k_extrapolated']:.2f} h")
        else:
            print(f"[warn] {name}: 差分无效 (t_short={t_short}, t_long={t_long})，跳过")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-segment", type=int, default=0,
                        help=">0 时用两点差分法实测稳态训练速率并外推60k（建议 8000，"
                             "即每方法各跑 4000 与 8000 步两个段）")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="参与推理延迟统计的种子列表（如 --seeds 22 32 42 52 62）")
    args = parser.parse_args()

    params = collect_params()
    latency = collect_latency(args.seeds)
    train_time = collect_train_time(args.time_segment) if args.time_segment > 0 else {}
    if args.time_segment <= 0:
        # 不带 --time-segment 重跑（例如仅刷新推理延迟口径）时，沿用已有实测训练耗时，
        # 避免把 train_s_per_1k / train_hours 列覆盖为空导致论文数据源断链
        prev_csv = RESULTS / "compute_cost.csv"
        if prev_csv.exists():
            try:
                old = pd.read_csv(prev_csv)
                carried = 0
                for _, r in old.iterrows():
                    if pd.notna(r.get("train_s_per_1k")):
                        train_time[str(r["algo"])] = {
                            "s_per_1k": float(r["train_s_per_1k"]),
                            "hours_60k_extrapolated": float(r["train_hours_60k_extrapolated"]),
                            "t_short_s": float(r.get("timing_short_segment_s", float("nan"))),
                            "t_long_s": float(r.get("timing_long_segment_s", float("nan"))),
                        }
                        carried += 1
                if carried:
                    print(f"[Info] 未指定 --time-segment：训练耗时列沿用已有实测值（{carried} 个方法）；"
                          f"如需重测请加 --time-segment 8000")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 沿用已有训练耗时失败（训练耗时列将为空）: {e}")

    # 汇总表
    algos = TRADITIONAL + [n for n in RL_CKPTS if n in params or n in train_time or (latency.shape[0] and n in latency["algo"].values)]
    rows = []
    lat_idx = {str(r["algo"]): r for _, r in latency.iterrows()} if latency.shape[0] else {}
    for a in algos:
        row = {
            "algo": a,
            "params_M": round(params[a], 3) if a in params else None,
            "infer_ms_mean": round(float(lat_idx[a]["mean"]), 4) if a in lat_idx else None,
            "infer_ms_std": round(float(lat_idx[a]["std"]), 4) if a in lat_idx and pd.notna(lat_idx[a]["std"]) else 0.0,
            "train_s_per_1k": round(train_time[a]["s_per_1k"], 2) if a in train_time else None,
            "train_hours_60k_extrapolated": round(train_time[a]["hours_60k_extrapolated"], 2) if a in train_time else None,
            "timing_short_segment_s": train_time[a].get("t_short_s") if a in train_time else None,
            "timing_long_segment_s": train_time[a].get("t_long_s") if a in train_time else None,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "compute_cost.csv", index=False, encoding="utf-8-sig")
    print("\n===== compute_cost.csv =====")
    print(out.to_string(index=False))

    # LaTeX 行（粘贴进论文表格）
    print("\n===== LaTeX 表格行 =====")
    for _, r in out.iterrows():
        p = f"{r['params_M']:.2f}" if pd.notna(r["params_M"]) else "--"
        inf = f"{r['infer_ms_mean']:.3f}$\\pm${r['infer_ms_std']:.3f}" if pd.notna(r["infer_ms_mean"]) else "--"
        rate = f"{r['train_s_per_1k']:.2f}" if pd.notna(r["train_s_per_1k"]) else "--"
        tr = f"{r['train_hours_60k_extrapolated']:.2f}" if pd.notna(r["train_hours_60k_extrapolated"]) else "--"
        print(f"{r['algo']} & {p} & {inf} & {rate} & {tr} \\\\")


if __name__ == "__main__":
    main()

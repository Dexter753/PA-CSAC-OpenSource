# -*- coding: utf-8 -*-
"""PPO-Lagrangian 安全RL基线：训练 + 与主实验同协议评估 + 追加至各种子 benchmark_summary.csv。

协议对齐（与 algos/PA_CSAC/train.py 主管线完全一致）：
  - 训练步数与其他RL基线相同（默认 60000，主管线 baseline_steps=max(train_steps,12000)）
  - 评估场景：_build_eval_reset_scenarios(env, 0, soc0=0.60, seed=种子)（全测试场景）
  - 评估环境参数：lower_violation_ratio=0.94, upper_cost_weight=0.4（与主管线 STAGE 2 基线评估的 global_env_params 一致）
  - 训练环境参数：与环境默认值一致（0.92/0.20），与 DDPG/TD3/SAC/PPO 基线训练协议对齐
  - 约束阈值：cost_limit=0.30（与 PA-CSAC 固定惩罚阈值 C_lim 统一）

用法：
  python scripts/run_safe_rl_baseline.py --seeds 42 52 62 --steps 60000
  python scripts/run_safe_rl_baseline.py --seeds 42 --eval-only   # 仅评估已有checkpoint
"""
import argparse
import atexit
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.PA_CSAC.train import (  # noqa: E402
    CloudPCCEnv,
    PPOLagrangian,
    _build_eval_reset_scenarios,
    evaluate,
    set_seed,
    train_ppo_lagrangian,
)

GLOBAL_ENV_PARAMS = {"lower_violation_ratio": 0.94, "upper_cost_weight": 0.4}
ALGO_NAME = "PPO-Lag"


class _Tee:
    """将输出同时写入终端与日志文件（与 run_quick.py 的 _Tee 机制一致）。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _enable_run_log(log_path: Path) -> Path:
    """启用完整训练过程终端日志（txt 留痕，便于事后定位训练问题）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    atexit.register(lambda: (f.flush(), f.close()))
    print(f"[RunLog] Full terminal log -> {log_path}")
    return log_path


def resolve_data_csv() -> Path:
    for name in ("pcc_rl_prediction_dataset_for_control.csv", "pcc_rl_prediction_dataset.csv"):
        p = ROOT / "prediction" / "results" / "csv" / name
        if p.exists():
            return p
    raise FileNotFoundError("找不到决策数据集（prediction/results/csv/）")


def append_benchmark_row(seed_dir: Path, metrics: dict) -> None:
    """将 PPO-Lag 行追加/覆盖到 benchmark_summary.csv（列结构与主管线输出一致）。"""
    csv_path = seed_dir / "benchmark_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
    else:
        df = pd.DataFrame()
    df.loc[ALGO_NAME] = pd.Series(metrics)
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")


def run_seed(seed: int, csv_path: Path, steps: int, eval_only: bool, cost_limit: float, lam_lr: float) -> None:
    seed_dir = ROOT / "results" / f"seed{seed}"
    model_dir = seed_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))

    if not eval_only:
        print(f"\n[{ALGO_NAME}] 训练 seed={seed}, steps={steps}")
        train_ppo_lagrangian(
            str(csv_path),
            total_steps=int(steps),
            save_dir=str(model_dir),
            seed=int(seed) + 60,  # 主管线基线种子偏移：DDPG+20/TD3+30/SAC+40/PPO+50
            feature_mode="pa_csac",
            cost_limit=float(cost_limit),
            lam_lr=float(lam_lr),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CloudPCCEnv(str(csv_path), device=device, feature_mode="pa_csac", split_mode="test")
    env.params.update(GLOBAL_ENV_PARAMS)
    scenarios = _build_eval_reset_scenarios(env, 0, soc0=0.60, seed=int(seed))
    if not scenarios:
        raise ValueError(f"seed={seed}: 无可用测试场景")

    agent = PPOLagrangian(obs_dim=int(env.observation_space.shape[0]), act_dim=1, act_limit=2.0, device=device)
    agent.load(str(model_dir / "ppo_lagrangian.pt"))

    print(f"[{ALGO_NAME}] 评估 seed={seed}, scenarios={len(scenarios)}")
    metrics, _ = evaluate(
        env,
        lambda o: agent.select_action(o, deterministic=True),
        episodes=len(scenarios),
        save_dir=str(seed_dir),
        name=ALGO_NAME,
        reset_options=scenarios,
        trace_dir=str(seed_dir / "traces"),
    )
    append_benchmark_row(seed_dir, metrics)

    fuel = float(metrics.get("fuel_l_per_100km", float("nan")))
    vsr = float(metrics.get("valid_episode_ratio", float("nan")))
    vr = float(metrics.get("violation_rate", float("nan")))
    gap = float(metrics.get("gap_rmse", float("nan")))
    print(f"[{ALGO_NAME}][seed={seed}] fuel={fuel:.2f} L/100km | VSR={vsr:.2f} | VR={vr:.3f} | gapRMSE={gap:.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--cost-limit", type=float, default=0.30)
    parser.add_argument("--lam-lr", type=float, default=5e-3)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    csv_path = resolve_data_csv()
    log_path = _enable_run_log(ROOT / "results" / "logs" / f"ppo_lag_terminal_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"[INFO] 数据集: {csv_path}")
    print(f"[INFO] PPO-Lag 运行日志: {log_path}")
    for seed in args.seeds:
        run_seed(int(seed), csv_path, int(args.steps), bool(args.eval_only), float(args.cost_limit), float(args.lam_lr))


if __name__ == "__main__":
    main()

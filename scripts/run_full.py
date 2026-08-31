import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（scripts 的上一级）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.PA_CSAC.train import run_all_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_component_ablation", action="store_true")
    args = parser.parse_args()

    project_root = ROOT
    data_csv_for_control = project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv"
    data_csv_full = project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset.csv"
    data_csv = data_csv_for_control if data_csv_for_control.exists() else data_csv_full

    # 每个种子写入独立子目录 results/seed{N}/，避免多个种子的运行互相覆盖/清理
    # （run_all_experiments 开头的 _cleanup_result_artifacts 会删除 save_dir 根目录下的
    #   benchmark_summary.csv 等文件——若多个种子共用 results/ 根目录，后一个种子
    #   启动时会清掉前一个种子的全部汇总数据）
    save_dir = project_root / "results" / f"seed{int(args.seed)}"

    if not data_csv.exists():
        raise FileNotFoundError(
            f"找不到决策数据集: {data_csv_for_control} 或 {data_csv_full}。请先运行预测脚本生成输出。"
        )

    print(f"[INFO] 项目根目录: {project_root}")
    print(f"[INFO] 使用数据集: {data_csv}")
    print(f"[INFO] 结果保存目录: {save_dir}")

    # PPO-Lagrangian 行保护：run_safe_rl_baseline.py 会把 PPO-Lag 评估行追加到
    # 各种子的 benchmark_summary.csv；而本次 run_all_experiments 的清理与重写
    # 会将其抹掉。这里在运行前暂存该行，运行结束后若缺失则自动补回，
    # 避免重跑某个种子后需要人工重跑 PPO-Lag 评估。
    import pandas as pd
    ppolag_row = None
    benchmark_csv = save_dir / "benchmark_summary.csv"
    if benchmark_csv.exists():
        try:
            _df = pd.read_csv(benchmark_csv, index_col=0, encoding="utf-8-sig")
            if "PPO-Lagrangian" in _df.index:
                ppolag_row = _df.loc[["PPO-Lagrangian"]]
                print("[INFO] 已暂存已有的 PPO-Lagrangian 评估行（运行结束后自动补回）")
        except Exception as e:  # noqa: BLE001
            print(f"[Warn] 读取已有 benchmark_summary.csv 失败（忽略）: {e}")

    run_all_experiments(
        csv_path=str(data_csv),
        save_dir=str(save_dir),
        train_only=False,
        train_steps=60000,
        eval_episodes=0,
        ablation_train_steps=60000,
        global_seed=int(args.seed),
        run_component_ablation=bool(not args.no_component_ablation),
        lower_violation_ratio=0.94,   # 与 results/seed*/run_config_snapshot.csv 实测数据一致
        upper_cost_weight=0.4,        # （run_all_experiments 的函数默认值 0.92/0.20 未被主实验采用）
        two_stage=False,  # 与 results/seed*/ 实测数据一致：主实验为单阶段60k连续训练
                            # （判据：two_stage=True 会在 models/ 留下 pa_csac_phase1_best.pt，实测数据无此文件）
    )

    # 运行结束：若最终 benchmark 缺少 PPO-Lagrangian 行而运行前存在，则补回
    if ppolag_row is not None and benchmark_csv.exists():
        try:
            _df = pd.read_csv(benchmark_csv, index_col=0, encoding="utf-8-sig")
            if "PPO-Lagrangian" not in _df.index:
                _df = pd.concat([_df, ppolag_row])
                _df.to_csv(benchmark_csv, encoding="utf-8-sig")
                print("[INFO] 已自动补回 PPO-Lagrangian 评估行")
        except Exception as e:  # noqa: BLE001
            print(f"[Warn] 补回 PPO-Lagrangian 行失败: {e}")
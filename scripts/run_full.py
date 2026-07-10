import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_component_ablation", action="store_true")
    args = parser.parse_args()
    from algos.PA_CSAC.train import run_all_experiments

    project_root = ROOT
    data_csv_for_control = project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv"
    data_csv_full = project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset.csv"
    data_csv = data_csv_for_control if data_csv_for_control.exists() else data_csv_full

    save_dir = project_root / "outputs" / "full" / f"seed{int(args.seed)}"

    if not data_csv.exists():
        raise FileNotFoundError(
            f"找不到决策数据集: {data_csv_for_control} 或 {data_csv_full}。请先运行预测脚本生成输出。"
        )

    print(f"[INFO] 项目根目录: {project_root}")
    print(f"[INFO] 使用数据集: {data_csv}")
    print(f"[INFO] 结果保存目录: {save_dir}")

    run_all_experiments(
        csv_path=str(data_csv),
        save_dir=str(save_dir),
        train_only=False,
        train_steps=60000,
        eval_episodes=0,
        ablation_train_steps=60000,
        global_seed=int(args.seed),
        run_component_ablation=bool(not args.no_component_ablation),
    )

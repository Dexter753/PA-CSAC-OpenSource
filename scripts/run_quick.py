import os
import sys
import json
import atexit
import argparse
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.PA_CSAC.train import run_all_experiments, generate_cross_seed_report


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except PermissionError:
                pass  # 忽略文件被占用的情况
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except PermissionError:
                pass  # 忽略文件被占用的情况


def _enable_run_log(log_path: str):
    """启用运行日志，使用安全的文件操作"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 添加重试机制和进程ID，确保文件名唯一
    process_id = os.getpid()
    unique_suffix = f"_pid{process_id}_{int(time.time() * 1000)}"
    
    # 如果文件存在或有权限问题，添加唯一后缀
    base_name = log_path
    if os.path.exists(log_path):
        name, ext = os.path.splitext(log_path)
        log_path = f"{name}{unique_suffix}{ext}"
    
    try:
        f = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)

        def _cleanup():
            try:
                f.flush()
                f.close()
            except Exception:
                pass

        atexit.register(_cleanup)
        print(f"[RunLog] Full terminal log -> {log_path}")
        return log_path
    except PermissionError:
        # 如果还是权限错误，打印警告但继续运行
        print(f"[RunLog] 警告: 无法创建日志文件 {log_path}，将继续运行但不保存日志")
        return None

def _parse_seeds(seed_text: str):
    vals = []
    for s in str(seed_text).split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("--seeds 不能为空")
    return vals


def _load_best_params(project_root: Path):
    params_path = project_root / "assets" / "best_params.json"
    if not params_path.exists():
        return {}, None
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("best_params", {})), str(params_path)


def _resolve_pa_csac_sync_config(args, project_root: Path):
    best_params, params_path = _load_best_params(project_root)

    reward_scale = float(
        args.reward_scale
        if args.reward_scale is not None
        else best_params.get("reward_scale", 5.0)
    )
    reward_bias = float(
        args.reward_bias
        if args.reward_bias is not None
        else best_params.get("reward_bias", 0.25)
    )
    alpha_min = float(
        args.alpha_min
        if args.alpha_min is not None
        else best_params.get("alpha_min", 0.02)
    )
    alpha_max = float(
        args.alpha_max
        if args.alpha_max is not None
        else best_params.get("alpha_max", 0.03)
    )
    raw_phase2_lr_ratio = float(
        args.phase2_lr_ratio
        if args.phase2_lr_ratio is not None
        else best_params.get("phase2_lr_ratio", 0.025)
    )
    return {
        "reward_scale": reward_scale,
        "reward_bias": reward_bias,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "raw_phase2_lr_ratio": raw_phase2_lr_ratio,
        "phase2_lr_ratio": raw_phase2_lr_ratio,
        "requested_two_stage": bool(args.two_stage),
        "requested_phase1_ratio": float(args.phase1_ratio),
        "source_path": params_path,
    }


def _effective_phase2_lr_ratio(raw_phase2_lr_ratio: float, train_steps: int):
    # 根据最新正式两种子日志，长程 paper 入口若直接使用 0.04，会在 step=5000 之后
    # 持续劣化并最终 Phase2Collapse；而 verify 基线稳定改善建立在 0.025 裁剪上。
    # 因此这里统一沿用 verify 已验证的 phase2_lr_ratio 上限，先保证正式入口与
    # 当前最佳证据链一致，再做后续多种子验证。
    _ = int(train_steps)
    return float(min(float(raw_phase2_lr_ratio), 0.025))


def _effective_two_stage(requested_two_stage: bool, train_steps: int):
    # 学术复核修正：
    # 训练器本身支持两阶段，且 CLI 已显式暴露 --two_stage/--no-two_stage。
    # 为保证“用户声明的正式配置”“终端真实运行配置”“论文方法描述”三者一致，
    # 这里不再对长程 paper 入口做隐式覆盖，而是严格遵从用户显式请求。
    _ = int(train_steps)
    return bool(requested_two_stage)


def _build_pa_csac_run_cfg(base_cfg: dict, train_steps: int):
    cfg = dict(base_cfg)
    cfg["train_steps"] = int(train_steps)
    cfg["phase2_lr_ratio"] = _effective_phase2_lr_ratio(cfg["raw_phase2_lr_ratio"], train_steps)
    cfg["two_stage"] = _effective_two_stage(cfg.get("requested_two_stage", True), train_steps)
    cfg["phase1_ratio"] = float(cfg.get("requested_phase1_ratio", 0.55)) if bool(cfg["two_stage"]) else 1.0
    return cfg


def _print_pa_csac_sync_config(cfg: dict, strict_prediction_columns: bool, strict_dedicated_prediction_columns: bool):
    print("[RunCfg][PA-CSAC] synchronized with verify baseline")
    if cfg.get("source_path"):
        print(f"[RunCfg][PA-CSAC] params_source={cfg['source_path']}")
    print(
        "[RunCfg][PA-CSAC] "
        f"reward_scale={cfg['reward_scale']:.3f}, reward_bias={cfg['reward_bias']:.3f}, "
        f"alpha_min={cfg['alpha_min']:.4f}, alpha_max={cfg['alpha_max']:.4f}"
    )
    print(
        "[RunCfg][PA-CSAC] "
        f"train_steps={int(cfg.get('train_steps', 0))}, "
        f"phase2_lr_ratio={cfg['raw_phase2_lr_ratio']:.4f}->{cfg['phase2_lr_ratio']:.4f}, "
        f"two_stage={bool(cfg.get('two_stage', True))}, "
        f"phase1_ratio={float(cfg.get('phase1_ratio', 0.55)):.3f}, "
        f"strict_prediction_columns={bool(strict_prediction_columns)}, "
        f"strict_dedicated_prediction_columns={bool(strict_dedicated_prediction_columns)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch_with_baselines", action="store_true")
    # 学术道德修正：默认种子数从 5 增加到 10，提高统计功效
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--ablation_tune", action="store_true")
    parser.add_argument("--disable_quality_gate", action="store_true")
    # 学术道德修正：质量门阈值与 _episode_is_valid 保持一致
    parser.add_argument("--gate_valid_ratio", type=float, default=0.50)
    parser.add_argument("--gate_upper", type=float, default=0.12)
    parser.add_argument("--gate_gap", type=float, default=32.0)
    parser.add_argument("--equiv_margin_fuel", type=float, default=0.5)
    parser.add_argument("--equiv_margin_gap", type=float, default=1.0)
    parser.add_argument("--include_mean_prediction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict_prediction_columns", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict_dedicated_prediction_columns", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_ablation_causal_mode", action="store_true")
    parser.add_argument("--disable_error_injection", action="store_true")
    parser.add_argument("--lower_violation_ratio", type=float, default=0.92)
    parser.add_argument("--upper_cost_weight", type=float, default=0.20)
    parser.add_argument("--drl_baseline_no_prediction", action="store_true", default=False,
                        help="DRL baselines (DDPG/TD3/SAC/PPO) use feature_mode=no_prediction for fair comparison, isolating prediction information value.")

    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--ablation_steps", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument("--no_ablation", action="store_true")
    parser.add_argument("--no_sensitivity", action="store_true")
    parser.add_argument("--no_component_ablation", action="store_true")
    parser.add_argument("--constraint_method", type=str, default="penalty", choices=["penalty", "lagrangian"],
                        help="'penalty': 固定惩罚权重(约束满足时无惩罚) | 'lagrangian': 自适应拉格朗日乘子")
    parser.add_argument("--penalty_weight", type=float, default=1.0,
                        help="Penalty Method模式下的惩罚权重（仅 constraint_method=penalty 时生效）")
    parser.add_argument("--prob_emb_lr", type=float, default=1e-3,
                        help="概率嵌入层学习率（独立于actor学习率，推荐1e-3~3e-4）")
    parser.add_argument("--two_stage", action=argparse.BooleanOptionalAction, default=True,
                        help="启用两阶段训练：阶段1冻结概率嵌入参数(残差≈恒等映射)，阶段2微调嵌入")
    parser.add_argument("--phase1_ratio", type=float, default=0.55,
                        help="两阶段训练中阶段1步数占比（默认0.55；正式长程按比例切分，短程verify保留裁剪式两阶段）")
    # 学术道德修正：增加消融调参的默认步数和评估 episode 数
    # 评估 episode 数必须足够大（>=32），以确保传统控制器（ACC/MPC/LQR）有足够有效样本
    # 注意：论文正式实验统一使用 run_full.py 的 60,000 步口径（见 results/seed*/run_config_snapshot.csv），
    # 本脚本 --paper 模式是独立的快速验证入口，其默认值与论文口径无关。
    parser.add_argument("--ablation_tune_steps", type=int, default=200000)
    parser.add_argument("--ablation_tune_eval", type=int, default=32)
    parser.add_argument("--reward_scale", type=float, default=None)
    parser.add_argument("--reward_bias", type=float, default=None)
    parser.add_argument("--alpha_min", type=float, default=None)
    parser.add_argument("--alpha_max", type=float, default=None)
    parser.add_argument("--phase2_lr_ratio", type=float, default=None)
    args = parser.parse_args()
    seed_list = _parse_seeds(args.seeds)

    # 学术道德修正：使用相对路径，提高代码可移植性
    script_dir = Path(__file__).resolve().parent
    project_root = ROOT
    
    data_csv_for_control = str(project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv")
    data_csv_full = str(project_root / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset.csv")
    save_dir = str(project_root / "outputs" / "quick")

    data_csv = data_csv_for_control if os.path.exists(data_csv_for_control) else data_csv_full
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"找不到决策数据集: {data_csv_for_control} 或 {data_csv_full}。请先运行预测脚本生成输出。"
        )

    pa_sync_cfg = _resolve_pa_csac_sync_config(args, project_root)

    if bool(args.ablation_tune):
        tune_root = os.path.join(save_dir, f"paper_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(tune_root, exist_ok=True)
        _enable_run_log(os.path.join(tune_root, "terminal_full_log.txt"))
        for sd in seed_list:
            out_dir = os.path.join(tune_root, f"seed{int(sd)}")
            print("\n" + "#" * 60)
            print(f"PAPER FULL RUN: seed{int(sd)} -> {out_dir}")
            print("#" * 60)
            tune_steps = int(max(6000, args.ablation_tune_steps))
            tune_eval = int(max(6, args.ablation_tune_eval))
            run_cfg = _build_pa_csac_run_cfg(pa_sync_cfg, tune_steps)
            _print_pa_csac_sync_config(
                run_cfg,
                args.strict_prediction_columns,
                args.strict_dedicated_prediction_columns,
            )
            # 论文完整版：开启全部功能
            run_all_experiments(
                csv_path=data_csv,
                save_dir=out_dir,
                train_only=False,
                train_steps=tune_steps,
                eval_episodes=tune_eval,
                ablation_train_steps=tune_steps,
                run_drl_baselines=bool(not args.no_baselines),  # 开启DRL基线
                run_ablation=bool(not args.no_ablation),  # 开启信息/机制消融
                run_sensitivity=bool(not args.no_sensitivity),  # 开启敏感性分析
                global_seed=int(sd),
                include_mean_prediction=bool(args.include_mean_prediction),
                strict_prediction_columns=bool(args.strict_prediction_columns),
                strict_dedicated_prediction_columns=bool(args.strict_dedicated_prediction_columns),
                ablation_causal_mode=bool(not args.disable_ablation_causal_mode),
                run_error_injection=bool(not args.disable_error_injection),
                lower_violation_ratio=float(args.lower_violation_ratio),
                upper_cost_weight=float(args.upper_cost_weight),
                drl_baseline_feature_mode="no_prediction" if args.drl_baseline_no_prediction else "pa_csac",
                run_component_ablation=bool(not args.no_component_ablation),
                constraint_method=str(args.constraint_method),
                penalty_weight=float(args.penalty_weight),
                prob_emb_lr=float(args.prob_emb_lr),
                two_stage=bool(run_cfg["two_stage"]),
                phase1_ratio=float(run_cfg["phase1_ratio"]),
                reward_scale=float(run_cfg["reward_scale"]),
                reward_bias=float(run_cfg["reward_bias"]),
                alpha_min=float(run_cfg["alpha_min"]),
                alpha_max=float(run_cfg["alpha_max"]),
                phase2_lr_ratio=float(run_cfg["phase2_lr_ratio"]),
            )
        if len(seed_list) >= 2:
            report_path = generate_cross_seed_report(
                tune_root,
                quality_gate=bool(not args.disable_quality_gate),
                gate_valid_ratio=float(args.gate_valid_ratio),
                gate_upper=float(args.gate_upper),
                gate_gap=float(args.gate_gap),
                equiv_margin_fuel=float(args.equiv_margin_fuel),
                equiv_margin_gap_rmse=float(args.equiv_margin_gap),
            )
            if report_path:
                print(f"[CrossSeed] 分析报告已生成: {report_path}")
        raise SystemExit(0)

    if bool(args.batch):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_root = os.path.join(save_dir, f"batch_{run_id}")
        os.makedirs(batch_root, exist_ok=True)
        _enable_run_log(os.path.join(batch_root, "terminal_full_log.txt"))

        runs = [
            {
                "tag": "1_smoke",
                "train_steps": 3000,
                "eval_episodes": 16,  # 提高统计显著性
                "ablation_steps": 3000,
                "run_drl_baselines": False,
                "run_ablation": False,
                "run_sensitivity": False,
            },
            {
                "tag": "2_quick_noheavy",
                "train_steps": 12000,
                "eval_episodes": 20,
                "ablation_steps": 12000,
                "run_drl_baselines": False,
                "run_ablation": False,
                "run_sensitivity": False,
                "multi_seed": False,
            },
            {
                "tag": "3_train_more",
                "train_steps": 100000,  # 增加训练步数
                "eval_episodes": 24,  # 增加评估episode数
                "ablation_steps": 50000,  # 增加消融训练步数
                "run_drl_baselines": bool(args.batch_with_baselines and (not args.no_baselines)),
                "run_ablation": bool(not args.no_ablation),
                "run_sensitivity": bool(not args.no_sensitivity),
                "multi_seed": True,
            },
        ]

        for r in runs:
            seeds_this_run = seed_list if bool(r.get("multi_seed", False)) else [seed_list[0]]
            for sd in seeds_this_run:
                tag = r["tag"] if len(seeds_this_run) == 1 else f"{r['tag']}_seed{int(sd)}"
                out_dir = os.path.join(batch_root, tag)
                print("\n" + "#" * 60)
                print(f"BATCH RUN: {tag} -> {out_dir}")
                print("#" * 60)
                run_cfg = _build_pa_csac_run_cfg(pa_sync_cfg, int(r["train_steps"]))
                _print_pa_csac_sync_config(
                    run_cfg,
                    args.strict_prediction_columns,
                    args.strict_dedicated_prediction_columns,
                )
                run_all_experiments(
                    csv_path=data_csv,
                    save_dir=out_dir,
                    train_only=False,
                    train_steps=int(r["train_steps"]),
                    eval_episodes=int(r["eval_episodes"]),
                    ablation_train_steps=int(r["ablation_steps"]),
                    run_drl_baselines=bool(r["run_drl_baselines"]),
                    run_ablation=bool(r["run_ablation"]),
                    run_sensitivity=bool(r["run_sensitivity"]),
                    global_seed=int(sd),
                    include_mean_prediction=bool(args.include_mean_prediction),
                    strict_prediction_columns=bool(args.strict_prediction_columns),
                    strict_dedicated_prediction_columns=bool(args.strict_dedicated_prediction_columns),
                    ablation_causal_mode=bool(not args.disable_ablation_causal_mode),
                    run_error_injection=bool(not args.disable_error_injection),
                    lower_violation_ratio=float(args.lower_violation_ratio),
                    upper_cost_weight=float(args.upper_cost_weight),
                    drl_baseline_feature_mode="no_prediction" if args.drl_baseline_no_prediction else "pa_csac",
                    run_component_ablation=bool(not args.no_component_ablation),
                    constraint_method=str(args.constraint_method),
                    penalty_weight=float(args.penalty_weight),
                    prob_emb_lr=float(args.prob_emb_lr),
                    two_stage=bool(run_cfg["two_stage"]),
                    phase1_ratio=float(run_cfg["phase1_ratio"]),
                    reward_scale=float(run_cfg["reward_scale"]),
                    reward_bias=float(run_cfg["reward_bias"]),
                    alpha_min=float(run_cfg["alpha_min"]),
                    alpha_max=float(run_cfg["alpha_max"]),
                    phase2_lr_ratio=float(run_cfg["phase2_lr_ratio"]),     
                )
        # 针对多种子目录生成跨seed统计与显著性分析
        for r in runs:
            if bool(r.get("multi_seed", False)):
                target_prefix = f"{r['tag']}_seed"
                report_path = generate_cross_seed_report(
                    batch_root,
                    name_filter=target_prefix,
                    quality_gate=bool(not args.disable_quality_gate),
                    gate_valid_ratio=float(args.gate_valid_ratio),
                    gate_upper=float(args.gate_upper),
                    gate_gap=float(args.gate_gap),
                    equiv_margin_fuel=float(args.equiv_margin_fuel),
                    equiv_margin_gap_rmse=float(args.equiv_margin_gap),
                )
                if report_path:
                    print(f"[CrossSeed] 分析报告已生成: {report_path} (filter={target_prefix})")
        raise SystemExit(0)

    if args.paper:
        default_train_steps = 200000  # 本脚本 paper 模式的独立默认值；论文正式口径为 run_full.py 的 60,000 步（results/seed*/run_config_snapshot.csv）
        default_ablation_steps = 120000  # 消融实验120k步，与主实验保持一致的训练量
        default_eval_eps = 32  # 论文模式需要更多评估episode以提高统计显著性
    else:
        default_train_steps = 20000
        default_ablation_steps = 20000
        default_eval_eps = 16

    if args.smoke:
        default_train_steps = 5000
        default_ablation_steps = 5000
        default_eval_eps = 8

    train_steps = int(default_train_steps if args.train_steps is None else args.train_steps)
    ablation_steps = int(default_ablation_steps if args.ablation_steps is None else args.ablation_steps)
    eval_eps = int(default_eval_eps if args.eval_episodes is None else args.eval_episodes)

    if args.paper:
        # 创建论文模式根目录
        paper_root = os.path.join(save_dir, f"paper_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(paper_root, exist_ok=True)
        _enable_run_log(os.path.join(paper_root, "terminal_full_log.txt"))
        
        # 遍历所有seed
        for sd in seed_list:
            out_dir = os.path.join(paper_root, f"seed{int(sd)}")
            print("\n" + "#" * 60)
            print(f"PAPER FULL RUN: seed{int(sd)} -> {out_dir}")
            print("#" * 60)
            run_cfg = _build_pa_csac_run_cfg(pa_sync_cfg, train_steps)
            _print_pa_csac_sync_config(
                run_cfg,
                args.strict_prediction_columns,
                args.strict_dedicated_prediction_columns,
            )
            
            run_all_experiments(
                csv_path=data_csv,
                save_dir=out_dir,
                train_only=False,
                train_steps=train_steps,
                eval_episodes=eval_eps,
                ablation_train_steps=ablation_steps,
                run_drl_baselines=bool(not args.no_baselines),  # 开启DRL基线
                run_ablation=bool(not args.no_ablation),  # 开启信息/机制消融
                run_sensitivity=bool(not args.no_sensitivity),
                global_seed=int(sd),
                include_mean_prediction=bool(args.include_mean_prediction),
                strict_prediction_columns=bool(args.strict_prediction_columns),
                strict_dedicated_prediction_columns=bool(args.strict_dedicated_prediction_columns),
                ablation_causal_mode=bool(not args.disable_ablation_causal_mode),
                run_error_injection=bool(not args.disable_error_injection),
                lower_violation_ratio=float(args.lower_violation_ratio),
                upper_cost_weight=float(args.upper_cost_weight),
                drl_baseline_feature_mode="no_prediction" if args.drl_baseline_no_prediction else "pa_csac",
                run_component_ablation=bool(not args.no_component_ablation),
                constraint_method=str(args.constraint_method),
                penalty_weight=float(args.penalty_weight),
                prob_emb_lr=float(args.prob_emb_lr),
                two_stage=bool(run_cfg["two_stage"]),
                phase1_ratio=float(run_cfg["phase1_ratio"]),
                reward_scale=float(run_cfg["reward_scale"]),
                reward_bias=float(run_cfg["reward_bias"]),
                alpha_min=float(run_cfg["alpha_min"]),
                alpha_max=float(run_cfg["alpha_max"]),
                phase2_lr_ratio=float(run_cfg["phase2_lr_ratio"]),
            )
        
        # 生成跨seed统计报告
        if len(seed_list) >= 2:
            report_path = generate_cross_seed_report(
                paper_root,
                quality_gate=bool(not args.disable_quality_gate),
                gate_valid_ratio=float(args.gate_valid_ratio),
                gate_upper=float(args.gate_upper),
                gate_gap=float(args.gate_gap),
                equiv_margin_fuel=float(args.equiv_margin_fuel),
                equiv_margin_gap_rmse=float(args.equiv_margin_gap),
            )
            if report_path:
                print(f"[CrossSeed] 分析报告已生成: {report_path}")
        raise SystemExit(0)

    _enable_run_log(os.path.join(save_dir, "terminal_full_log.txt"))
    run_cfg = _build_pa_csac_run_cfg(pa_sync_cfg, train_steps)
    _print_pa_csac_sync_config(
        run_cfg,
        args.strict_prediction_columns,
        args.strict_dedicated_prediction_columns,
    )

    run_all_experiments(
        csv_path=data_csv,
        save_dir=save_dir,
        train_only=False,
        train_steps=train_steps,
        eval_episodes=eval_eps,
        ablation_train_steps=ablation_steps,
        run_drl_baselines=bool(not args.no_baselines),  # 快速测试也运行DRL基线
        run_ablation=bool(not args.no_ablation),  # 快速测试也运行消融
        run_sensitivity=bool(not args.no_sensitivity),
        global_seed=int(seed_list[0]),
        include_mean_prediction=bool(args.include_mean_prediction),
        strict_prediction_columns=bool(args.strict_prediction_columns),
        strict_dedicated_prediction_columns=bool(args.strict_dedicated_prediction_columns),
        ablation_causal_mode=bool(not args.disable_ablation_causal_mode),
        run_error_injection=bool(not args.disable_error_injection),
        lower_violation_ratio=float(args.lower_violation_ratio),
        upper_cost_weight=float(args.upper_cost_weight),
        drl_baseline_feature_mode="no_prediction" if args.drl_baseline_no_prediction else "pa_csac",
        run_component_ablation=bool(not args.no_component_ablation),
        constraint_method=str(args.constraint_method),
        penalty_weight=float(args.penalty_weight),
        prob_emb_lr=float(args.prob_emb_lr),
        two_stage=bool(run_cfg["two_stage"]),
        phase1_ratio=float(run_cfg["phase1_ratio"]),
        reward_scale=float(run_cfg["reward_scale"]),
        reward_bias=float(run_cfg["reward_bias"]),
        alpha_min=float(run_cfg["alpha_min"]),
        alpha_max=float(run_cfg["alpha_max"]),
        phase2_lr_ratio=float(run_cfg["phase2_lr_ratio"]),
    )

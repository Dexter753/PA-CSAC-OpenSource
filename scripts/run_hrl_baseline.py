# -*- coding: utf-8 -*-
"""HRL 基线（Zhang et al., 2023, Appl. Energy 333, 120599）：训练 + 与主实验同协议评估
+ 追加至各种子 benchmark_summary.csv。

协议对齐（与 algos/PA_CSAC/train.py 主管线及 run_safe_rl_baseline.py 完全一致）：
  - 训练步数与其他RL基线相同（默认 60000）
  - 训练环境参数：与环境默认值一致（0.92/0.20），与 DDPG/TD3/SAC/PPO/PPO-Lag/SMORL 对齐
  - 评估场景：_build_eval_reset_scenarios(env, 0, soc0=0.60, seed=种子)（全测试场景）
  - 评估环境参数：lower_violation_ratio=0.94, upper_cost_weight=0.4（与主管线
    STAGE 2 基线评估的 global_env_params 一致）
  - 种子偏移：DDPG+20/TD3+30/SAC+40/PPO+50/PPO-Lag+60/SMORL+70 之后，HRL 使用 seed+80
  - 网络结构/优化器/目标网络更新与 DDPG 基线一致（唯一差异为分层结构本身）

HRL 专属协议（源自 Zhang et al., 2023 摘要可核实内容）：
  - 分层策略 + 非分层执行：上层仅在宏周期边界（K=10 步）规划 (SOC, 时距) 目标，
    目标在宏周期内线性插值，下层每步以目标为条件输出加速度
  - 上层奖励 = 宏周期内环境奖励累积（经济性目标）；下层奖励 = 环境奖励 + 目标跟踪塑形
  - 自学习交互（无监督热启动，原文为 GPS 数据构建的跟驰场景中自学习）
  - best checkpoint：训练内定期以确定性分层策略在训练集场景上评估平均回报选取

用法：
  python scripts/run_hrl_baseline.py --seeds 42 52 62 --steps 60000
  python scripts/run_hrl_baseline.py --seeds 42 --eval-only   # 仅评估已有checkpoint
"""
import argparse
import atexit
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: train.py imports pyplot at module level

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.PA_CSAC.train import (  # noqa: E402
    CloudPCCEnv,
    _build_eval_reset_scenarios,
    _check_convergence,
    _save_history_csv,
    evaluate,
    set_seed,
)
from algos.PA_CSAC.hrl import HRL, HRLController, macro_state_from_obs  # noqa: E402
from utils.utils import ReplayBuffer  # noqa: E402

GLOBAL_ENV_PARAMS = {"lower_violation_ratio": 0.94, "upper_cost_weight": 0.4}
ALGO_NAME = "HRL"
SEED_OFFSET = 80  # 主管线基线种子偏移序列：DDPG+20/.../PPO-Lag+60/SMORL+70
MACRO_PERIOD = 10
GOAL_NOISE = 0.1  # 上层目标探索噪声（unit 空间标准差）


class _Tee:
    """将输出同时写入终端与日志文件（与 run_safe_rl_baseline.py 一致）。"""

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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1)  # 行缓冲：崩溃时保留已输出内容
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
    csv_path = seed_dir / "benchmark_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
        df.loc[ALGO_NAME] = pd.Series(metrics)
    else:
        df = pd.DataFrame(metrics, index=[ALGO_NAME])
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")


def make_eval_policy(agent: HRL, env):
    """评估用分层策略闭包：通过 env.step_idx==0 检测回合边界并复位控制器。"""
    controller = HRLController(agent)

    def policy(obs):
        if int(getattr(env, "step_idx", 0)) == 0 and controller.step != 0:
            controller.reset()
        return controller.act(obs, deterministic=True)

    return policy


def train_hrl(csv_path: str, total_steps: int, save_dir: str, seed: int,
              feature_mode: str = "pa_csac") -> HRL:
    set_seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CloudPCCEnv(str(csv_path), device=device, feature_mode=feature_mode, split_mode="train")
    obs_dim = int(env.observation_space.shape[0])
    agent = HRL(obs_dim=obs_dim, act_dim=1, device=device, macro_period=MACRO_PERIOD)
    controller = HRLController(agent, macro_period=MACRO_PERIOD)
    controller.rng = np.random.default_rng(int(seed))

    lower_replay = ReplayBuffer(max_size=400000, obs_dim=obs_dim + 2, act_dim=1, seed=int(seed))
    upper_replay = ReplayBuffer(max_size=100000, obs_dim=8, act_dim=2, seed=int(seed))
    _rng = np.random.default_rng(int(seed))

    model_dir = Path(save_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "hrl.pt"
    best_path = model_dir / "hrl_best.pt"
    hist_path = model_dir.parent / "histories" / "hrl_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = int(total_steps)
    start_steps = int(min(3000, max(500, total_steps * 0.25)))
    update_after = int(min(2000, max(300, total_steps * 0.15)))
    update_every = 2
    batch_size = 256
    explore_noise = 0.1
    eval_interval = int(max(2500, total_steps // 6))
    start_time = time.time()

    # 学习率调度器（与 train_offpolicy_agent 的 baseline 调度一致）
    def _lr_lambda(step):
        decay_steps = max(1, total_steps // 3)
        return 0.5 ** min(step // decay_steps, 2)

    lower_actor_sched = torch.optim.lr_scheduler.LambdaLR(agent.lower_actor_opt, lr_lambda=_lr_lambda)
    lower_critic_sched = torch.optim.lr_scheduler.LambdaLR(agent.lower_critic_opt, lr_lambda=_lr_lambda)
    upper_actor_sched = torch.optim.lr_scheduler.LambdaLR(agent.upper_actor_opt, lr_lambda=_lr_lambda)
    upper_critic_sched = torch.optim.lr_scheduler.LambdaLR(agent.upper_critic_opt, lr_lambda=_lr_lambda)

    print(f"[TrainCfg][{ALGO_NAME}] total_steps={total_steps}, start_steps={start_steps}, "
          f"update_after={update_after}, noise={explore_noise}, macro_period={MACRO_PERIOD}, "
          f"goal_noise={GOAL_NOISE}")

    eval_env = CloudPCCEnv(str(csv_path), device=device, feature_mode=feature_mode, split_mode="train")

    def _score(n_episodes=4):
        total = 0.0
        ctrl = HRLController(agent, macro_period=MACRO_PERIOD)
        for g in range(n_episodes):
            o, _ = eval_env.reset(options={"group_idx": g, "deterministic_reset": True, "soc0": 0.60})
            done, ep_r = False, 0.0
            ctrl.reset()
            while not done:
                a = ctrl.act(o, deterministic=True)
                o, r, done, _, _ = eval_env.step(a)
                ep_r += float(r)
            total += ep_r
        return total / max(n_episodes, 1)

    obs, _ = env.reset(options={"group_idx": 0})
    group_idx = 0
    ep_reward = 0.0
    history, hist_steps = [], []
    best_score = None

    # 宏周期簿记：本期起始 boundary 的 macro_obs、本期规划目标 g_end_b、期内环境奖励累积
    macro_obs_b = None
    g_end_b = None
    macro_R = 0.0
    controller.random_goal = True  # 探索期（start_steps 内）上层均匀随机目标

    for t in range(total_steps):
        # ---- 宏周期边界：闭合上一期的上层转移（next 即当前 obs） ----
        boundary_now = (controller.step % MACRO_PERIOD == 0)
        if boundary_now and g_end_b is not None:
            upper_replay.add(macro_obs_b, g_end_b, macro_R, 0.0,
                             macro_state_from_obs(obs), 0.0)
            macro_R = 0.0

        # ---- 分层动作选择 ----
        if t < start_steps:
            g_ref = controller.current_goal(obs)
            act = np.array([float(_rng.uniform(-3.0, 2.0))], dtype=np.float32)
        else:
            controller.random_goal = False
            noise_current = float(explore_noise) * max(0.2, 1.0 - t / max(total_steps, 1))
            goal_noise_current = float(GOAL_NOISE) * max(0.2, 1.0 - t / max(total_steps, 1))
            g_ref = controller.current_goal(obs, noise_std=goal_noise_current)
            act = agent.select_action(obs, g_ref, deterministic=True)
            act = act + _rng.normal(size=act.shape).astype(np.float32) * noise_current
            act = np.clip(act, env.action_space.low, env.action_space.high).astype(np.float32)

        if boundary_now:
            macro_obs_b = macro_state_from_obs(obs)
            g_end_b = controller.g_end.copy()

        # ---- 环境交互 ----
        obs_g = np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                                np.asarray(g_ref, dtype=np.float32).reshape(-1)])
        next_obs, rew, done, _, info = env.step(act)
        exec_act = np.array([float(info["acc"])], dtype=np.float32)
        ep_reward += float(rew)
        macro_R += float(rew)

        # 下层转移：目标在步内视为常数（标准 goal-conditioned 约定）
        next_obs_g = np.concatenate([np.asarray(next_obs, dtype=np.float32).reshape(-1),
                                     np.asarray(g_ref, dtype=np.float32).reshape(-1)])
        r_lower = controller.lower_reward(float(rew), next_obs, g_ref)
        lower_replay.add(obs_g, exec_act, r_lower, float(info["cost"]), next_obs_g, float(done))

        controller.step += 1
        obs = next_obs

        if done:
            # 闭合未完结宏周期（done=1）
            if g_end_b is not None:
                upper_replay.add(macro_obs_b, g_end_b, macro_R, 0.0,
                                 macro_state_from_obs(next_obs), 1.0)
            history.append(ep_reward)
            hist_steps.append(t + 1)
            ep_reward = 0.0
            group_idx = (group_idx + 1) % len(env.processed_groups)
            obs, _ = env.reset(options={"group_idx": group_idx})
            controller.reset()
            controller.random_goal = (t < start_steps)
            macro_obs_b = None
            g_end_b = None
            macro_R = 0.0

        # ---- 离策更新 ----
        if t >= update_after and t % update_every == 0 and len(lower_replay) >= batch_size:
            agent.update_lower(lower_replay.sample(batch_size))
            lower_actor_sched.step()
            lower_critic_sched.step()
        if t >= update_after and t % MACRO_PERIOD == 0 and len(upper_replay) >= batch_size:
            agent.update_upper(upper_replay.sample(batch_size))
            upper_actor_sched.step()
            upper_critic_sched.step()

        # ---- 定期评估选 best checkpoint ----
        if t > 0 and t % eval_interval == 0:
            try:
                sc = _score()
                if best_score is None or sc > best_score:
                    best_score = sc
                    agent.save(str(best_path))
                    print(f"[BestCkpt][{ALGO_NAME}] step={t} score={best_score:.2f}")
            except Exception as e:
                print(f"[Eval Warning][{ALGO_NAME}] step={t}: {e}")

        if t > 0 and t % 2000 == 0:
            print(f"[{ALGO_NAME}] step={t}/{total_steps} lower_buf={len(lower_replay)} "
                  f"upper_buf={len(upper_replay)} elapsed={time.time() - start_time:.0f}s")

    if ep_reward != 0.0:
        history.append(ep_reward)
        hist_steps.append(int(total_steps))

    if best_path.is_file():
        try:
            agent.load(str(best_path))
            print(f"[BestCkpt][{ALGO_NAME}] Loaded best model with score={best_score:.2f}")
        except Exception as e:
            print(f"[BestCkpt Warning][{ALGO_NAME}] load failed: {e}")
    agent.save(str(save_path))
    _save_history_csv(history, str(hist_path), hist_steps)

    convergence_status = _check_convergence(history, hist_steps, total_steps)
    if not convergence_status["converged"]:
        print(f"[Convergence Warning][{ALGO_NAME}]: {convergence_status['reason']}")
    else:
        print(f"[Convergence OK][{ALGO_NAME}]: {convergence_status['message']}")

    agent.train_time_seconds = float(time.time() - start_time)
    if torch.cuda.is_available():
        agent.train_memory_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        torch.cuda.reset_peak_memory_stats()
    else:
        agent.train_memory_mb = 0.0
    print(f"[Efficiency][{ALGO_NAME}]: train_time={agent.train_time_seconds:.1f}s, "
          f"peak_memory={agent.train_memory_mb:.1f}MB")
    return agent


def run_seed(seed: int, csv_path: Path, steps: int, eval_only: bool) -> None:
    seed_dir = ROOT / "results" / f"seed{seed}"
    model_dir = seed_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))

    if not eval_only:
        print(f"\n[{ALGO_NAME}] 训练 seed={seed}, steps={steps}")
        train_hrl(
            str(csv_path),
            total_steps=int(steps),
            save_dir=str(model_dir),
            seed=int(seed) + SEED_OFFSET,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CloudPCCEnv(str(csv_path), device=device, feature_mode="pa_csac", split_mode="test")
    env.params.update(GLOBAL_ENV_PARAMS)
    scenarios = _build_eval_reset_scenarios(env, 0, soc0=0.60, seed=int(seed))
    if not scenarios:
        raise ValueError(f"seed={seed}: 无可用测试场景")

    agent = HRL(obs_dim=int(env.observation_space.shape[0]), act_dim=1, device=device,
                macro_period=MACRO_PERIOD)
    agent.load(str(model_dir / "hrl.pt"))

    print(f"[{ALGO_NAME}] 评估 seed={seed}, scenarios={len(scenarios)}")
    policy_fn = make_eval_policy(agent, env)
    metrics, _ = evaluate(
        env,
        policy_fn,
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
    print(f"[{ALGO_NAME}][seed={seed}] fuel={fuel:.2f} L/100km | VSR={vsr:.2f} | "
          f"VR={vr:.3f} | gapRMSE={gap:.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 32, 42, 52, 62])
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    csv_path = resolve_data_csv()
    log_path = _enable_run_log(
        ROOT / "results" / "logs" / f"hrl_terminal_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"[INFO] 数据集: {csv_path}")
    print(f"[INFO] HRL 运行日志: {log_path}")
    for seed in args.seeds:
        run_seed(int(seed), csv_path, int(args.steps), bool(args.eval_only))


if __name__ == "__main__":
    main()

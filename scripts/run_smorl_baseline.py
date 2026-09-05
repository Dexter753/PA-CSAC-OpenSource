# -*- coding: utf-8 -*-
"""SMORL 基线（Zhu et al., 2022, arXiv:2105.11640）：训练 + 与主实验同协议评估
+ 追加至各种子 benchmark_summary.csv。

协议对齐（与 algos/PA_CSAC/train.py 主管线及 run_safe_rl_baseline.py 完全一致）：
  - 训练步数与其他RL基线相同（默认 60000）
  - 训练环境参数：与环境默认值一致（0.92/0.20），与 DDPG/TD3/SAC/PPO/PPO-Lag 对齐
  - 评估场景：_build_eval_reset_scenarios(env, 0, soc0=0.60, seed=种子)（全测试场景）
  - 评估环境参数：lower_violation_ratio=0.94, upper_cost_weight=0.4（与主管线
    STAGE 2 基线评估的 global_env_params 一致）
  - 种子偏移：DDPG+20/TD3+30/SAC+40/PPO+50/PPO-Lag+60 之后，SMORL 使用 seed+70

SMORL 专属协议（源自 Algorithm 1）：
  - 初始化：保守控制器（IDM）收集 30 条成功轨迹（Algorithm 1 的 N_0），
    仅成功轨迹进入经验回放与安全集池
  - 每步：求解重择域轨迹优化得到动作；以概率 eps=0.2 探索（Tab. II）
  - 每步：离策更新双 critic / 扰动网络 / VAE（BCQ 式，Eqns. (17)-(23)）
  - 回合成功（完整走完、无碰撞/掉队/数值失效）才入缓冲区，并累积安全集样本
  - best checkpoint：训练内定期以确定性策略在训练集场景上评估平均回报选取

用法：
  python scripts/run_smorl_baseline.py --seeds 42 52 62 --steps 60000
  python scripts/run_smorl_baseline.py --seeds 42 --eval-only   # 仅评估已有checkpoint

预期运行时长：轨迹优化每步约需数百次物理模型评估，单种子 60000 步约需数小时，
建议不同种子分终端并行运行。
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
    baseline_controller,
    evaluate,
    set_seed,
)
from algos.PA_CSAC.smorl import ACT_SET, SMORL  # noqa: E402
from utils.utils import ReplayBuffer  # noqa: E402

GLOBAL_ENV_PARAMS = {"lower_violation_ratio": 0.94, "upper_cost_weight": 0.4}
ALGO_NAME = "SMORL"
SEED_OFFSET = 70  # 主管线基线种子偏移序列：DDPG+20/TD3+30/SAC+40/PPO+50/PPO-Lag+60
PLAN_HORIZON = 8  # 与 MPC 基线一致的时域步长
PREVIEW_KNOTS = np.array([1.0, 3.0, 5.0])  # t+1 / t+3 / t+5 预测节点（与 MPC-L 一致）

# Algorithm 1 初始化：保守控制器收集的成功轨迹数 N_0
WARMSTART_TARGET_EPISODES = 30
WARMSTART_MAX_ATTEMPTS = 200


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
    csv_path = seed_dir / "benchmark_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
        df.loc[ALGO_NAME] = pd.Series(metrics)
    else:
        df = pd.DataFrame(metrics, index=[ALGO_NAME])
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")


def _preview_from_env(env, horizon=PLAN_HORIZON):
    """从环境的 prob9 列构造 t+1/t+3/t+5 线性插值预览（与 MPC-L 一致）。"""
    try:
        prob9 = env.current_group_data["prob9"]
        idx = int(np.clip(int(env.step_idx), 0, len(prob9) - 1))
        mu = np.asarray(prob9[idx][0:3], dtype=np.float64)
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return np.interp(steps, PREVIEW_KNOTS, mu)
    except Exception:
        return None


def _agent_policy(agent, env):
    def policy(obs):
        return agent.select_action(
            obs, deterministic=True,
            dt=float(getattr(env, "dt_episode", 1.0)),
            v_lead_preview=_preview_from_env(env),
        )
    return policy


def train_smorl(csv_path: str, total_steps: int, save_dir: str, seed: int,
                feature_mode: str = "pa_csac") -> SMORL:
    set_seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CloudPCCEnv(str(csv_path), device=device, feature_mode=feature_mode, split_mode="train")
    obs_dim = int(env.observation_space.shape[0])
    agent = SMORL(obs_dim=obs_dim, act_dim=1, device=device,
                  env_params=env.params, horizon=PLAN_HORIZON)
    replay = ReplayBuffer(max_size=200000, obs_dim=obs_dim, act_dim=1, seed=int(seed))
    rng = np.random.default_rng(int(seed))

    model_dir = Path(save_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "smorl.pt"
    best_path = model_dir / "smorl_best.pt"
    hist_path = model_dir.parent / "histories" / "smorl_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- Algorithm 1 初始化：IDM 保守控制器收集 N_0 条成功轨迹 ----------------
    successes, attempts = 0, 0
    while successes < WARMSTART_TARGET_EPISODES and attempts < WARMSTART_MAX_ATTEMPTS:
        attempts += 1
        group_idx = attempts % len(env.processed_groups)
        obs, _ = env.reset(options={"group_idx": group_idx})
        ep_buf, ep_obs, done = [], [], False
        info = {}
        while not done:
            act = baseline_controller("IDM", obs, dt=float(env.dt_episode))
            next_obs, rew, done, _, info = env.step(act)
            exec_act = np.array([float(info["acc"])], dtype=np.float32)
            ep_buf.append((obs, exec_act, float(rew), float(info["cost"]), next_obs, float(done)))
            ep_obs.append(np.asarray(obs, dtype=np.float32).copy())
            obs = next_obs
        if str(info.get("terminated_reason", "")) == "timeout":
            successes += 1
            for tr in ep_buf:
                replay.add(*tr)
            agent.observe_safe_states(np.stack(ep_obs))
    print(f"[SMORL] warm start: {successes} successful IDM episodes "
          f"(attempts={attempts}), buffer={len(replay)}, "
          f"safe pool={agent.safe_pool_size} states")
    # 冻结离散化边界并完成首次安全集拟合 + delta 校准
    agent.fit_safe_set()

    # ---------------- 训练主循环（Algorithm 1） ----------------
    best_score = None
    eval_interval = int(max(2500, total_steps // 6))
    eval_env = CloudPCCEnv(str(csv_path), device=device, feature_mode=feature_mode, split_mode="train")

    def _score(n_episodes=4):
        total = 0.0
        for g in range(n_episodes):
            o, _ = eval_env.reset(options={"group_idx": g, "deterministic_reset": True, "soc0": 0.60})
            done, ep_r = False, 0.0
            while not done:
                a = agent.select_action(
                    o, deterministic=True, dt=float(eval_env.dt_episode),
                    v_lead_preview=_preview_from_env(eval_env))
                o, r, done, _, _ = eval_env.step(a)
                ep_r += float(r)
            total += ep_r
        return total / max(n_episodes, 1)

    obs, _ = env.reset(options={"group_idx": 0})
    group_idx = 0
    ep_buf, ep_obs, ep_reward = [], [], 0.0
    history, hist_steps = [], []
    n_success_ep, n_fail_ep = 0, 0
    start_time = time.time()
    print(f"[TrainCfg][SMORL] total_steps={total_steps}, horizon={PLAN_HORIZON}, "
          f"eps={agent.epsilon}, eval_interval={eval_interval}")

    for t in range(int(total_steps)):
        if rng.random() < agent.epsilon:
            # 探索（Tab. II: eps=0.2）：在离散控制格点上均匀采样
            act = np.array([float(rng.choice(ACT_SET))], dtype=np.float32)
        else:
            act = agent.select_action(
                obs, deterministic=True, dt=float(env.dt_episode),
                v_lead_preview=_preview_from_env(env))
        next_obs, rew, done, _, info = env.step(act)
        exec_act = np.array([float(info["acc"])], dtype=np.float32)
        ep_buf.append((obs, exec_act, float(rew), float(info["cost"]), next_obs, float(done)))
        ep_obs.append(np.asarray(obs, dtype=np.float32).copy())
        obs = next_obs
        ep_reward += float(rew)

        if len(replay) >= 256:
            agent.update(replay.sample(256))

        if done:
            history.append(ep_reward)
            hist_steps.append(t + 1)
            reason = str(info.get("terminated_reason", "unknown"))
            if reason == "timeout":
                # 仅成功轨迹进入缓冲区与安全集 [Algorithm 1]
                for tr in ep_buf:
                    replay.add(*tr)
                agent.observe_safe_states(np.stack(ep_obs))
                n_success_ep += 1
            else:
                n_fail_ep += 1
            ep_buf, ep_obs, ep_reward = [], [], 0.0
            group_idx = (group_idx + 1) % len(env.processed_groups)
            obs, _ = env.reset(options={"group_idx": group_idx})

        if t > 0 and t % eval_interval == 0:
            try:
                sc = _score()
                if best_score is None or sc > best_score:
                    best_score = sc
                    agent.save(str(best_path))
                    print(f"[BestCkpt][SMORL] step={t} score={best_score:.2f}")
            except Exception as e:
                print(f"[Eval Warning][SMORL] step={t}: {e}")

        if t > 0 and t % 2000 == 0:
            print(f"[SMORL] step={t}/{total_steps} buffer={len(replay)} "
                  f"safe_pool={agent.safe_pool_size} succ_ep={n_success_ep} "
                  f"fail_ep={n_fail_ep} elapsed={time.time() - start_time:.0f}s")

    if ep_reward != 0.0:
        history.append(ep_reward)
        hist_steps.append(int(total_steps))

    if best_path.is_file():
        try:
            agent.load(str(best_path))
            print(f"[BestCkpt][SMORL] Loaded best model with score={best_score:.2f}")
        except Exception as e:
            print(f"[BestCkpt Warning][SMORL] load failed: {e}")
    agent.save(str(save_path))
    pd.DataFrame({"step": hist_steps, "reward": history}).to_csv(hist_path, index=False, encoding="utf-8-sig")

    agent.train_time_seconds = float(time.time() - start_time)
    if torch.cuda.is_available():
        agent.train_memory_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        torch.cuda.reset_peak_memory_stats()
    else:
        agent.train_memory_mb = 0.0
    print(f"[Efficiency][SMORL]: train_time={agent.train_time_seconds:.1f}s, "
          f"peak_memory={agent.train_memory_mb:.1f}MB")
    return agent


def run_seed(seed: int, csv_path: Path, steps: int, eval_only: bool) -> None:
    seed_dir = ROOT / "results" / f"seed{seed}"
    model_dir = seed_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))

    if not eval_only:
        print(f"\n[{ALGO_NAME}] 训练 seed={seed}, steps={steps}")
        train_smorl(
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

    agent = SMORL(obs_dim=int(env.observation_space.shape[0]), act_dim=1, device=device,
                  env_params=env.params, horizon=PLAN_HORIZON)
    agent.load(str(model_dir / "smorl.pt"))
    # 评估阶段规划器的物理模型参数与评估环境保持一致
    agent.env_params = env.params

    print(f"[{ALGO_NAME}] 评估 seed={seed}, scenarios={len(scenarios)}")
    policy_fn = _agent_policy(agent, env)
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
        ROOT / "results" / "logs" / f"smorl_terminal_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"[INFO] 数据集: {csv_path}")
    print(f"[INFO] SMORL 运行日志: {log_path}")
    for seed in args.seeds:
        run_seed(int(seed), csv_path, int(args.steps), bool(args.eval_only))


if __name__ == "__main__":
    main()

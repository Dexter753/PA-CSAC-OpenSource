import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

import copy
import collections
import numpy as np
import os
import tempfile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from constraint_handler import ConstraintHandler
from shield import shield_action_from_obs_vectorized as _shield_action_from_obs

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _is_torch_zip_save_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "inline_container" in text
        or "unexpected pos" in text
        or "[errno 22]" in text
        or "invalid argument" in text
    )


def _atomic_torch_save(obj, path, retries=3):
    """Save checkpoints robustly on Windows.

    PyTorch's zip-based serializer can occasionally fail on Windows with
    errors such as "unexpected pos" / "Invalid argument" while finalizing
    the archive. To avoid leaving truncated checkpoint files behind, this
    helper writes to a temp file in the same directory, fsyncs it, then
    atomically replaces the target file. If zip serialization fails, it
    retries with the legacy serializer as a fallback.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    strategies = [True] + [False] * max(int(retries) - 1, 0)
    last_exc = None

    for attempt_idx, use_new_zip in enumerate(strategies, start=1):
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
                dir=directory,
            )
            with os.fdopen(fd, "wb") as f:
                fd = None
                torch.save(
                    obj,
                    f,
                    _use_new_zipfile_serialization=use_new_zip,
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)
            return
        except Exception as exc:
            last_exc = exc
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            should_retry = attempt_idx < len(strategies)
            if should_retry and _is_torch_zip_save_error(exc):
                print(
                    f"[CheckpointSave Warning] save failed for {path} "
                    f"(attempt {attempt_idx}/{len(strategies)}): {exc}. "
                    "Retrying with legacy serialization."
                )
                time.sleep(0.2 * attempt_idx)
                continue
            raise

    raise RuntimeError(f"Failed to save checkpoint: {path}") from last_exc

class ProbEmbeddingDiagnostic:
    """概率嵌入层诊断器 - 用于追踪prob_embedding是否真正在学习"""
    
    def __init__(self):
        self.enabled = True
        self.history = {
            'param_change': [],
            'grad_magnitude': [],
            'output_std': [],
        }
        self._initial_params = None
        self._last_params = None
        
    def capture_initial(self, prob_embedding):
        """捕获初始参数"""
        if self._initial_params is None and prob_embedding is not None:
            self._initial_params = {
                n: p.detach().clone().cpu().numpy()
                for n, p in prob_embedding.named_parameters()
            }
            self._last_params = self._initial_params.copy()
            print(f"[DIAG] prob_embedding初始参数已捕获")
    
    def compute_grad_magnitude(self, prob_embedding):
        """计算prob_embedding的梯度幅度"""
        if prob_embedding is None:
            return 0.0
        total_norm = 0.0
        for p in prob_embedding.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return np.sqrt(total_norm)
    
    def compute_param_change(self, prob_embedding):
        """计算参数变化量"""
        if self._last_params is None or prob_embedding is None:
            return 0.0
        total_change = 0.0
        for n, p in prob_embedding.named_parameters():
            if n in self._last_params:
                change = np.abs(p.detach().cpu().numpy() - self._last_params[n]).mean()
                total_change += change
        return total_change
    
    def update_params(self, prob_embedding):
        """更新最后参数记录"""
        if prob_embedding is not None:
            self._last_params = {
                n: p.detach().clone().cpu().numpy()
                for n, p in prob_embedding.named_parameters()
            }
    
    def record(self, prob_embedding, prob_input, update_step, env_step=None, phase_name=None, mix_ratio=None):
        """记录诊断信息"""
        if not self.enabled or prob_embedding is None:
            return
            
        grad_mag = self.compute_grad_magnitude(prob_embedding)
        param_change = self.compute_param_change(prob_embedding)
        
        with torch.no_grad():
            output = prob_embedding(prob_input)
            output_std = float(output.std().item())
        
        self.history['grad_magnitude'].append(grad_mag)
        self.history['param_change'].append(param_change)
        self.history['output_std'].append(output_std)
        
        self.update_params(prob_embedding)
        
        if update_step % 500 == 0 and update_step > 0:
            self._print_report(
                update_step,
                env_step=env_step,
                phase_name=phase_name,
                mix_ratio=mix_ratio,
            )
    
    def _print_report(self, update_step, env_step=None, phase_name=None, mix_ratio=None):
        """打印诊断报告"""
        recent_n = min(10, len(self.history['grad_magnitude']))
        if recent_n == 0:
            return
            
        recent_grad = np.mean(self.history['grad_magnitude'][-recent_n:])
        recent_change = np.mean(self.history['param_change'][-recent_n:])
        recent_std = np.mean(self.history['output_std'][-recent_n:])
        
        phase_label = str(phase_name) if phase_name is not None else "unknown"
        env_label = str(int(env_step)) if env_step is not None else "N/A"
        mix_label = f"{float(mix_ratio):.2f}" if mix_ratio is not None else "N/A"
        print(f"\n{'='*70}")
        print(
            f"[PROB_EMB_DIAG] UpdateStep {int(update_step)} "
            f"(env_step={env_label}, phase={phase_label}, mix={mix_label})"
        )
        print(f"{'='*70}")
        print(f"[DIAG] prob_embedding梯度幅度(最近{recent_n}步平均): {recent_grad:.6f}")
        print(f"[DIAG] prob_embedding参数变化(最近{recent_n}步平均): {recent_change:.6f}")
        print(f"[DIAG] prob_embedding输出标准差(最近{recent_n}步平均): {recent_std:.6f}")
        
        if recent_grad < 1e-8:
            print(f"[CRITICAL] prob_embedding几乎没有收到梯度！")
        elif recent_change < 1e-6:
            print(f"[CRITICAL] prob_embedding参数几乎没有变化！")
        elif recent_std < 0.01:
            print(f"[WARNING] prob_embedding输出接近常数！")
        print(f"{'='*70}\n")

def build_mlp(dims, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(dims) - 1):
        activation = act if i < len(dims) - 2 else out_act
        layer = nn.Linear(dims[i], dims[i + 1])
        # 权重初始化优化：使用 Xavier 初始化提升收敛速度
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
        layers.append(layer)
        layers.append(activation())
    return nn.Sequential(*layers)

class ProbFeatureEmbedding(nn.Module):
    """概率特征嵌入层
    
    设计：output = Tanh(embedding(x))，输出范围[-1, 1]。
    - Phase1（冻结+透传）：直接使用原始概率特征，Actor/Critic在原始特征空间学习
    - Phase2（解冻+嵌入）：使用Tanh嵌入特征，微调概率表征
    
    日志证据（trial_030）：
    - Phase1结束时: valid=7/8, fuel=10.649
    - Phase2早期(500步内): valid=8/8, fuel=9.397 ← Phase2确实改善了策略
    - Phase2后期(4000步): valid=3/8, fuel=17.369 ← 过拟合退化
    - 结论：Tanh嵌入设计有效，但Phase2需要早停防止过拟合
    """
    def __init__(self, in_dim=8, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Tanh()
        )
        for i, m in enumerate(self.embedding.modules()):
            if isinstance(m, nn.Linear):
                gain = 0.1  # 减小初始化增益：降低初始输出方差，避免早期噪声干扰训练
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        emb = self.embedding(x)
        return emb

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.body = build_mlp([obs_dim, 256, 256, 256], act=nn.ReLU, out_act=nn.ReLU)
        self.mu_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)
        
        # 优化：Actor 输出层初始化为较小值，防止训练初期动作过于激进
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic=False, with_logprob=True):
        hidden = self.body(obs)
        mu = self.mu_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        logp_pi = None
        if with_logprob:
            logp_pi = dist.log_prob(action).sum(dim=-1, keepdim=True)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

        action = torch.tanh(action)
        return self.act_limit * action, logp_pi

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, deterministic, False)
            return a

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, out_act='tanh', q_scale=10.0):
        """Critic网络
        out_act: 'tanh' (旧版, 输出[-q_scale,q_scale]), 
                 'linear' (PA-CSAC V7: 线性输出, 无激活, 标准SAC做法)
        q_scale: 输出缩放因子（仅用于tanh模式）
        """
        super().__init__()
        act_map = {'tanh': nn.Tanh, 'linear': nn.Identity}
        # V11 数据驱动（V6-V10五版本日志 + cost约束修复）：
        #
        # V6(/10): Q∈[-5,0], 熵占17%→Q翻正, fuel=14.62
        # V7(/1):  Q∈[-60,-144], Actor激进→碰撞77%, valid=0/8
        # V8(/3):  Q∈[-6,-26], Alpha=0.02时Q坍缩→fuel=14.0
        # V9(/4):  Q∈[-14,-16], Alpha=0.1时熵占7%→Q翻正
        # V10(/2): Q稳定-32~-65, 但viol_rate=77.6%（约束失效！）
        #   根因：cost_excess = relu(qc_pi/EMA - 0.5) = relu(负值) = 0
        #
        # V11: cost_excess = relu(qc_pi.abs() - cost_limit)
        #   修复：使用绝对值表示不安全程度，当|qc_pi| > 0.3时触发约束
        #   Linear Critic + 梯度裁剪(2.0) + clamp[-1000,0] + 修复cost约束
        self.q = build_mlp([obs_dim + act_dim, 256, 128, 1], act=nn.ReLU, out_act=act_map[out_act])
        self.q_scale = float(q_scale)
        self._out_act = out_act

    def forward(self, obs, act):
        raw = self.q(torch.cat([obs, act], dim=-1))
        if self._out_act == 'linear':
            # 直接输出，无激活、无缩放
            # Q值由TD学习自然驱动到负值（所有target≤0）
            return raw
        return raw * self.q_scale

class PACSAC:
    """
    PA-CSAC: Constrained SAC (双Q + 成本Q + 约束处理)
    适配 Cloud-PCC 的安全约束学习
    核心改进：使用 ProbFeatureEmbedding 对 8 维概率嵌入进行可学习映射，
    替代 env 中的手工特征工程，使网络能够端到端学习最优概率表征。
    
    约束策略通过 ConstraintHandler 策略模式注入：
    - PenaltyHandler: 固定惩罚权重，约束满足时无惩罚，仅在违反时施加
    - LagrangianHandler: 自适应拉格朗日乘子
    
    组件消融模式：
    - use_cost_constraint=False: 去掉约束 + qc 网络，退化为标准 SAC+shield
    - use_prob_embedding=False: 去掉可学习嵌入层，直接使用原始概率特征
    """
    def __init__(
        self,
        obs_dim,
        act_dim=1,
        act_limit=2.0,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        cost_limit=0.08,
        device="cpu",
        use_prob_embedding=True,
        use_cost_constraint=True,
        actor_lr=3e-4,
        critic_lr=3e-4,
        constraint_method='penalty',
        penalty_weight=1.0,
        prob_emb_lr=3e-4,
        reward_scale=5.0,
        alpha_min=0.02,
        alpha_max=0.05,
        shield_mismatch_coef=0.18,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.cost_limit = float(cost_limit)
        self.target_entropy = -float(act_dim)
        self.use_prob_embedding = bool(use_prob_embedding)
        self.use_cost_constraint = bool(use_cost_constraint)
        self.constraint_method = str(constraint_method)
        self.penalty_weight = float(penalty_weight)
        self.reward_scale = float(reward_scale)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.shield_mismatch_coef = float(shield_mismatch_coef)
        
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.update_step = 0
        self.policy_delay = 2
        self.cost_q_ema = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.cost_limit_norm = self.cost_limit * 3.3333
        self.q_loss_history = []
        self.q_loss_warning_threshold = 25000.0  # V10 rew/2: Q≈-320级，MSE正常范围

        self.prob_embedding = ProbFeatureEmbedding(in_dim=8, emb_dim=8).to(self.device) if self.use_prob_embedding else None
        
        # PA-CSAC V8: Critic使用linear(线性输出，无激活)
        # rew/3归一化：V6(/10)Q太弱→fuel=14.62, V7(/1)Q太强→碰撞77%
        # V8折中：Q≈-48, 范围~25, 83x V6信号, 不淹没安全约束
        self.actor = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.q1 = Critic(obs_dim, act_dim, out_act='linear').to(self.device)
        self.q2 = Critic(obs_dim, act_dim, out_act='linear').to(self.device)
        self.qc = Critic(obs_dim, act_dim, out_act='linear').to(self.device) if self.use_cost_constraint else None
        
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()
        self.qc_target = copy.deepcopy(self.qc).eval() if self.use_cost_constraint else None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.prob_emb_opt = None
        if self.use_prob_embedding:
            self.prob_emb_opt = torch.optim.Adam(self.prob_embedding.parameters(), lr=float(prob_emb_lr))
            self.prob_emb_diagnostic = ProbEmbeddingDiagnostic()
            self.prob_emb_diagnostic.capture_initial(self.prob_embedding)
        
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        if self.use_cost_constraint:
            critic_params += list(self.qc.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)
        
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=8e-4)

        if self.use_cost_constraint:
            self._constraint = ConstraintHandler.create(
                method=self.constraint_method,
                penalty_weight=self.penalty_weight,
                device=str(self.device),
            )
        else:
            self._constraint = None

        self._null_multiplier = torch.tensor(0.0, device=self.device)
        self._embedding_frozen_for_phase1 = False  # 两阶段训练控制标志
        self._prob_embedding_mix = 1.0 if self.use_prob_embedding else 0.0
        self._train_env_step = None
        self._train_phase_name = "phase1"

    @property
    def lagrangian_lambda(self):
        if self._constraint is None:
            return self._null_multiplier
        return self._constraint.get_multiplier()

    @property
    def temperature(self):
        return torch.exp(self.log_alpha).clamp(1e-5, 1.0)

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.use_prob_embedding:
            obs_t = self._embed_obs(obs_t)
        action = self.actor.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def _embed_obs(self, obs):
        """将 20 维观察中的后 8 维概率特征通过可学习嵌入层映射
        
        两阶段训练：
        - Phase1（冻结）：透传原始概率特征，等效于use_prob_embedding=False
          Actor/Critic在原始特征空间学习稳定策略
        - Phase2（解冻）：使用Tanh嵌入特征，微调概率表征
          trial_030数据证明：Phase2早期(500步)确实改善了策略(7/8→8/8)
        """
        base = obs[:, :12]
        prob = obs[:, 12:]
        if self._embedding_frozen_for_phase1:
            # Phase1：透传原始概率特征
            return torch.cat([base, prob], dim=-1)
        prob_emb = self.prob_embedding(prob)
        mix = float(np.clip(getattr(self, "_prob_embedding_mix", 1.0), 0.0, 1.0))
        # Phase2 不直接从“原始特征”硬切到“嵌入特征”，而是渐进混合，
        # 避免 verify_log 中在 Phase2 入口出现的输入分布突变和策略掉队。
        prob_feat = (1.0 - mix) * prob + mix * prob_emb
        return torch.cat([base, prob_feat], dim=-1)

    def freeze_prob_embedding(self):
        """冻结概率嵌入层 + 切换透传模式：两阶段训练阶段1使用，先学习稳定策略"""
        if self.prob_embedding is not None:
            for p in self.prob_embedding.parameters():
                p.requires_grad = False
            self._embedding_frozen_for_phase1 = True
            self._prob_embedding_mix = 0.0
            if hasattr(self, 'prob_emb_diagnostic') and self.prob_emb_diagnostic is not None:
                self.prob_emb_diagnostic.enabled = False

    def unfreeze_prob_embedding(self):
        """解冻概率嵌入层 + 恢复嵌入模式：两阶段训练阶段2使用，在稳定策略上微调嵌入"""
        if self.prob_embedding is not None:
            for p in self.prob_embedding.parameters():
                p.requires_grad = True
            self._embedding_frozen_for_phase1 = False
            # 解冻后默认从“接近透传”开始，由训练循环逐步提升混合比例。
            self._prob_embedding_mix = 0.0

    def set_prob_embedding_mix(self, mix_ratio):
        """设置原始概率特征与嵌入特征的混合比例。"""
        self._prob_embedding_mix = float(np.clip(mix_ratio, 0.0, 1.0))

    def set_train_step_context(self, env_step=None, phase_name=None):
        """记录训练上下文，便于把 update_step 与 env_step/phase 对齐到日志。"""
        self._train_env_step = None if env_step is None else int(env_step)
        if phase_name is not None:
            self._train_phase_name = str(phase_name)

    def reset_optimizer_states(self):
        """在回载 checkpoint 或阶段切换后清空优化器动量，避免旧状态驱动新参数继续更新。"""
        for opt in [self.actor_opt, self.critic_opt, self.alpha_opt, self.prob_emb_opt]:
            if opt is not None:
                # PyTorch Optimizer.state 需要保持 defaultdict(dict) 语义；
                # 若直接赋普通 dict，会在 step() 首次访问新参数状态时触发 KeyError。
                opt.state = collections.defaultdict(dict)

    def _polyak_update(self, source_net, target_net):
        for p, p_t in zip(source_net.parameters(), target_net.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch):
        self.update_step += 1
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        # V17 数据驱动修复：reward缩放归一化（基于V12~V16完整日志对比确诊）
        # 使用可配置的reward_scale参数，支持贝叶斯优化自动搜索
        rew = rew / self.reward_scale
        cost = torch.as_tensor(batch["cost"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        # === DEBUG: 记录奖励和成本统计 ===
        debug_info = {
            'rew_min': float(rew.min().item()),
            'rew_max': float(rew.max().item()),
            'rew_mean': float(rew.mean().item()),
            'cost_min': float(cost.min().item()),
            'cost_max': float(cost.max().item()),
            'cost_mean': float(cost.mean().item()),
            'batch_act_abs_mean': float(act.abs().mean().item()),
            'batch_act_abs_max': float(act.abs().max().item()),
        }

        if self.use_prob_embedding:
            obs_emb_for_critic = self._embed_obs(obs)
            next_obs_emb = self._embed_obs(next_obs)
        else:
            obs_emb_for_critic = obs
            next_obs_emb = next_obs

        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs_emb)
            next_act_exec = _shield_action_from_obs(next_obs_emb, next_act)
            q1_t = self.q1_target(next_obs_emb, next_act_exec)
            q2_t = self.q2_target(next_obs_emb, next_act_exec)
            next_q = torch.min(q1_t, q2_t)
            
            # === DEBUG: 记录裁剪前的next_q ===
            debug_info['next_q_before_clamp_min'] = float(next_q.min().item())
            debug_info['next_q_before_clamp_max'] = float(next_q.max().item())
            debug_info['next_q_before_clamp_mean'] = float(next_q.mean().item())
            
            next_q = torch.clamp(next_q, -1000.0, 0.0)  # V10 rew/2: Q≈-320级，clamp留3x裕量
            
            # === DEBUG: 记录裁剪后的next_q ===
            debug_info['next_q_after_clamp_min'] = float(next_q.min().item())
            debug_info['next_q_after_clamp_max'] = float(next_q.max().item())
            
            target_q = rew + self.gamma * (1.0 - done) * (next_q - self.temperature.detach() * next_logp)

            if self.use_cost_constraint:
                next_qc = self.qc_target(next_obs_emb, next_act_exec)
                debug_info['next_qc_before_clamp_min'] = float(next_qc.min().item())
                debug_info['next_qc_before_clamp_max'] = float(next_qc.max().item())
                next_qc = torch.clamp(next_qc, -1000.0, 0.0)  # V10: 与主Q一致
                target_qc = cost + self.gamma * (1.0 - done) * next_qc

        # === DEBUG: 记录裁剪前的target_q ===
        debug_info['target_q_before_clamp_min'] = float(target_q.min().item())
        debug_info['target_q_before_clamp_max'] = float(target_q.max().item())
        debug_info['target_q_before_clamp_mean'] = float(target_q.mean().item())
        
        target_q = torch.clamp(target_q, -1000.0, 0.0)  # V10 rew/2: 宽clamp
        
        # === DEBUG: 记录裁剪后的target_q ===
        debug_info['target_q_after_clamp_min'] = float(target_q.min().item())
        debug_info['target_q_after_clamp_max'] = float(target_q.max().item())
        
        if self.use_cost_constraint:
            target_qc = torch.clamp(target_qc, -1000.0, 0.0)
        
        # === DEBUG: 记录当前Q网络输出 ===
        q1_pred = self.q1(obs_emb_for_critic.detach(), act)
        q2_pred = self.q2(obs_emb_for_critic.detach(), act)
        debug_info['q1_pred_min'] = float(q1_pred.min().item())
        debug_info['q1_pred_max'] = float(q1_pred.max().item())
        debug_info['q1_pred_mean'] = float(q1_pred.mean().item())
        debug_info['q2_pred_min'] = float(q2_pred.min().item())
        debug_info['q2_pred_max'] = float(q2_pred.max().item())
        debug_info['q2_pred_mean'] = float(q2_pred.mean().item())
        
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        qc_loss = F.mse_loss(self.qc(obs_emb_for_critic.detach(), act), target_qc) if self.use_cost_constraint else torch.tensor(0.0, device=self.device)
        self.critic_opt.zero_grad()
        total_critic_loss = q1_loss + q2_loss + qc_loss
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=2.0)  # 增强梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=2.0)
        if self.use_cost_constraint:
            torch.nn.utils.clip_grad_norm_(self.qc.parameters(), max_norm=2.0)
        self.critic_opt.step()

        if self.use_prob_embedding:
            obs_emb = self._embed_obs(obs)
        else:
            obs_emb = obs

        sampled_act, logp = self.actor(obs_emb)
        sampled_exec = _shield_action_from_obs(obs_emb, sampled_act)
        shield_push = torch.relu(sampled_act - sampled_exec)
        shield_pull = torch.relu(sampled_exec - sampled_act)
        shield_regularizer = shield_push + 0.6 * shield_pull
        debug_info['sampled_act_mean'] = float(sampled_act.mean().item())
        debug_info['sampled_act_std'] = float(sampled_act.std(unbiased=False).item())
        debug_info['sampled_exec_mean'] = float(sampled_exec.mean().item())
        debug_info['sampled_exec_std'] = float(sampled_exec.std(unbiased=False).item())
        debug_info['logp_mean'] = float(logp.mean().item())
        debug_info['shield_push_mean'] = float(shield_push.mean().item())
        debug_info['shield_pull_mean'] = float(shield_pull.mean().item())
        debug_info['shield_reg_mean'] = float(shield_regularizer.mean().item())
        # 关键修复：q_pi不detach，让梯度流向prob_embedding
        q_pi = torch.min(self.q1(obs_emb, sampled_exec), self.q2(obs_emb, sampled_exec))
        
        # === DEBUG: 记录策略Q值 ===
        debug_info['q_pi_min'] = float(q_pi.min().item())
        debug_info['q_pi_max'] = float(q_pi.max().item())
        debug_info['q_pi_mean'] = float(q_pi.mean().item())

        if self.use_cost_constraint:
            # 关键修复：使用qc_pi.abs()而不是qc_norm
            # 原因：qc_pi使用Linear激活输出负值，qc_norm = qc_pi/EMA = 负值
            # 原问题：cost_excess = relu(负值 - 0.5) = 0，约束完全失效
            # 修复：使用qc_pi.abs()表示不安全程度，当|qc_pi| > cost_limit时触发
            qc_pi = self.qc(obs_emb, sampled_exec)
            cost_excess = torch.relu(qc_pi.abs() - self.cost_limit)
            debug_info['cost_excess_max'] = float(cost_excess.max().item())
            debug_info['qc_pi_abs_mean'] = float(qc_pi.abs().mean().item())
            debug_info['qc_pi_abs_max'] = float(qc_pi.abs().max().item())
            qc_norm = qc_pi.abs()  # 用于约束损失计算
        else:
            cost_excess = torch.zeros_like(q_pi)
            qc_norm = torch.zeros(1, device=self.device)

        # Alpha使用自动熵调节，但设置合理范围
        # V11：保持Alpha自动调节，因为策略需要熵来探索
        # 注意：如果策略锁定在激进模式，需要调整奖励函数或约束机制
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # 限制Alpha范围，防止极端值
        # V13：上限0.10→0.05
        # 证据：V12 Step 6000 Alpha=0.10时Q从-26骤降到-9→8000翻正(+21.6)
        #   Alpha=0.10 → 熵bonus≈+0.23 → TD target出现正值 → Q翻正
        #   0.05 → 熵bonus≈+0.12 → 占Q的比例从3%降到1.3%
        with torch.no_grad():
            # 使用可配置的Alpha范围，支持贝叶斯优化自动搜索
            self.log_alpha.clamp_min_(np.log(self.alpha_min))
            self.log_alpha.clamp_max_(np.log(self.alpha_max))

        if self._constraint is not None:
            lambda_loss = self._constraint.compute_loss(qc_norm, self.cost_limit_norm)
            if lambda_loss.requires_grad:
                self._constraint.zero_grad()
                lambda_loss.backward()
                self._constraint.step()
        else:
            lambda_loss = torch.tensor(0.0, device=self.device)

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.update_step % self.policy_delay == 0:
            for p in self.q1.parameters(): p.requires_grad = False
            for p in self.q2.parameters(): p.requires_grad = False
            if self.use_cost_constraint:
                for p in self.qc.parameters(): p.requires_grad = False
            
            # V15 数据驱动修复：clamp q_pi到非正，防止Actor被正Q值误导
            # 日志证据：V12 Step 8000 q_pi_max=+19.770 → Actor优化正Q → 策略激进 → fuel=15~16
            # 数学保证：奖励始终为负 → 真实Q值必为负 → 正Q是Critic估计错误
            # 效果：即使Critic暂时预测正Q，Actor也不会被误导，只依赖负Q信号学习
            q_pi_safe = torch.clamp(q_pi, -1000.0, 0.0)
            actor_loss = (
                self.temperature.detach() * logp
                - q_pi_safe
                + self.lagrangian_lambda.detach() * cost_excess
                + self.shield_mismatch_coef * shield_regularizer
            ).mean()
            
            self.actor_opt.zero_grad()
            if self.prob_emb_opt is not None:
                self.prob_emb_opt.zero_grad()
            actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            # V12修复：prob_embedding max_norm 2.0→5.0
            # 日志证据：Phase 2 prob_emb梯度全程被裁剪到2.0上限
            #   裁剪截断梯度信息→Q值在[-25,-71]振荡→训练不稳定→策略锁定
            #   增大到5.0后梯度不被截断→信息完整→Q值稳定→actor正常学习
            if self.prob_emb_opt is not None:
                torch.nn.utils.clip_grad_norm_(self.prob_embedding.parameters(), max_norm=5.0)
            self.actor_opt.step()
            if self.prob_emb_opt is not None:
                self.prob_emb_opt.step()
                if hasattr(self, 'prob_emb_diagnostic') and self.prob_emb_diagnostic is not None:
                    prob_input_for_diag = obs[:, 12:]
                    self.prob_emb_diagnostic.record(
                        self.prob_embedding,
                        prob_input_for_diag,
                        self.update_step,
                        env_step=self._train_env_step,
                        phase_name=self._train_phase_name,
                        mix_ratio=getattr(self, "_prob_embedding_mix", None),
                    )
            
            for p in self.q1.parameters(): p.requires_grad = True
            for p in self.q2.parameters(): p.requires_grad = True
            if self.use_cost_constraint:
                for p in self.qc.parameters(): p.requires_grad = True

        self._polyak_update(self.q1, self.q1_target)
        self._polyak_update(self.q2, self.q2_target)
        if self.use_cost_constraint:
            self._polyak_update(self.qc, self.qc_target)

        q_loss_total = (q1_loss + q2_loss).item()
        self.q_loss_history.append(q_loss_total)
        if len(self.q_loss_history) > 100:
            self.q_loss_history.pop(0)
        
        is_q_loss_exploding = False
        if len(self.q_loss_history) >= 10:
            recent_mean = float(np.mean(self.q_loss_history[-10:]))
            if recent_mean > self.q_loss_warning_threshold:
                is_q_loss_exploding = True
        
        # === 合并调试信息到返回结果 ===
        stats = {
            "q_loss": q_loss_total,
            "qc_loss": qc_loss.item(),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.temperature.item(),
            "lambda": self.lagrangian_lambda.item() if self.use_cost_constraint else 0.0,
            "is_q_loss_exploding": is_q_loss_exploding,
        }
        # 添加调试信息
        stats.update(debug_info)
        
        return stats

    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha,
            "update_step": self.update_step,
            "constraint_handler": self._constraint.state_dict() if self._constraint is not None else {},
        }
        if self.use_cost_constraint:
            ckpt["qc"] = self.qc.state_dict()
            ckpt["qc_target"] = self.qc_target.state_dict()
        if self.use_prob_embedding:
            ckpt["prob_embedding"] = self.prob_embedding.state_dict()
        ckpt["_embedding_frozen_for_phase1"] = self._embedding_frozen_for_phase1
        ckpt["_prob_embedding_mix"] = float(getattr(self, "_prob_embedding_mix", 1.0))
        ckpt["actor_opt"] = self.actor_opt.state_dict()
        ckpt["critic_opt"] = self.critic_opt.state_dict()
        ckpt["alpha_opt"] = self.alpha_opt.state_dict()
        if self.prob_emb_opt is not None:
            ckpt["prob_emb_opt"] = self.prob_emb_opt.state_dict()
        _atomic_torch_save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        if "q1_target" in ckpt:
            self.q1_target.load_state_dict(ckpt["q1_target"])
            self.q2_target.load_state_dict(ckpt["q2_target"])
        else:
            self.q1_target = copy.deepcopy(self.q1)
            self.q2_target = copy.deepcopy(self.q2)
        if self.use_cost_constraint and "qc" in ckpt:
            self.qc.load_state_dict(ckpt["qc"])
            if "qc_target" in ckpt:
                self.qc_target.load_state_dict(ckpt["qc_target"])
            else:
                self.qc_target = copy.deepcopy(self.qc)
            if self._constraint is not None and "constraint_handler" in ckpt:
                self._constraint.load_state_dict(ckpt["constraint_handler"])
        if self.use_prob_embedding and "prob_embedding" in ckpt:
            self.prob_embedding.load_state_dict(ckpt["prob_embedding"])
        self._embedding_frozen_for_phase1 = bool(ckpt.get("_embedding_frozen_for_phase1", False))
        self._prob_embedding_mix = float(np.clip(ckpt.get("_prob_embedding_mix", 1.0), 0.0, 1.0))
        self.log_alpha.data = ckpt["log_alpha"].to(self.device)
        if "update_step" in ckpt:
            self.update_step = int(ckpt["update_step"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "alpha_opt" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        if self.prob_emb_opt is not None and "prob_emb_opt" in ckpt:
            self.prob_emb_opt.load_state_dict(ckpt["prob_emb_opt"])


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = build_mlp([obs_dim, 256, 256, 256, act_dim], act=nn.ReLU, out_act=nn.Identity)
        last = self.net[-2] if len(self.net) >= 2 else None
        if isinstance(last, nn.Linear):
            nn.init.uniform_(last.weight, -3e-3, 3e-3)
            nn.init.constant_(last.bias, 0.0)
        self.act_limit = float(act_limit)

    def forward(self, obs):
        return self.act_limit * torch.tanh(self.net(obs))


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v = build_mlp([obs_dim, 256, 256, 1], act=nn.ReLU, out_act=nn.Identity)

    def forward(self, obs):
        return self.v(obs)


def _to_tensors(batch, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["act"], dtype=torch.float32, device=device)
    rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=device).unsqueeze(-1)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
    done = torch.as_tensor(batch["done"], dtype=torch.float32, device=device).unsqueeze(-1)
    return obs, act, rew, next_obs, done


class SAC:
    def __init__(
        self,
        obs_dim,
        act_dim=1,
        act_limit=2.0,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device="cpu",
        auto_alpha=True,
    ):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.act_limit = float(act_limit)
        self.target_entropy = -float(act_dim)
        self.auto_alpha = bool(auto_alpha)

        self.actor = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.q1 = Critic(obs_dim, act_dim).to(self.device)
        self.q2 = Critic(obs_dim, act_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=1e-3)

    @property
    def temperature(self):
        return torch.exp(self.log_alpha).clamp(1e-5, 1.0)

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def _polyak_update(self, source_net, target_net):
        for p, p_t in zip(source_net.parameters(), target_net.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch):
        obs, act, rew, next_obs, done = _to_tensors(batch, self.device)

        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            next_exec = _shield_action_from_obs(next_obs, next_act)
            q1_t = self.q1_target(next_obs, next_exec)
            q2_t = self.q2_target(next_obs, next_exec)
            next_q = torch.min(q1_t, q2_t)
            target_q = rew + self.gamma * (1.0 - done) * (next_q - self.temperature.detach() * next_logp)

        q1_loss = F.mse_loss(self.q1(obs, act), target_q)
        q2_loss = F.mse_loss(self.q2(obs, act), target_q)

        self.critic_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10.0)
        self.critic_opt.step()

        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        sampled_act, logp = self.actor(obs)
        sampled_exec = _shield_action_from_obs(obs, sampled_act)
        q_pi = torch.min(self.q1(obs, sampled_exec), self.q2(obs, sampled_exec))
        actor_loss = (self.temperature.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        self._polyak_update(self.q1, self.q1_target)
        self._polyak_update(self.q2, self.q2_target)

        return {
            "q_loss": (q1_loss + q2_loss).item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.temperature.item(),
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else float(alpha_loss),
        }

    def save(self, path):
        _atomic_torch_save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "log_alpha": self.log_alpha,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.log_alpha.data = ckpt["log_alpha"].to(self.device)
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()


class DDPG:
    def __init__(self, obs_dim, act_dim=1, act_limit=2.0, gamma=0.99, tau=0.005, device="cpu"):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.act_limit = float(act_limit)

        self.actor = DeterministicActor(obs_dim, act_dim, act_limit).to(self.device)
        self.q = Critic(obs_dim, act_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.q_target = copy.deepcopy(self.q).eval()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.q.parameters(), lr=1e-3)

    def select_action(self, obs, deterministic=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(obs_t)
        return a.cpu().numpy()[0]

    def _polyak_update(self, source_net, target_net):
        for p, p_t in zip(source_net.parameters(), target_net.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch):
        obs, act, rew, next_obs, done = _to_tensors(batch, self.device)

        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            next_act = _shield_action_from_obs(next_obs, next_act)
            target_q = rew + self.gamma * (1.0 - done) * self.q_target(next_obs, next_act)

        q_loss = F.mse_loss(self.q(obs, act), target_q)
        self.critic_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.critic_opt.step()

        for p in self.q.parameters():
            p.requires_grad = False
        a_pi = self.actor(obs)
        a_exec = _shield_action_from_obs(obs, a_pi)
        actor_loss = -(self.q(obs, a_exec)).mean() + 2e-2 * (a_pi ** 2).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()
        for p in self.q.parameters():
            p.requires_grad = True

        self._polyak_update(self.actor, self.actor_target)
        self._polyak_update(self.q, self.q_target)

        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item()}

    def save(self, path):
        _atomic_torch_save(
            {
                "actor": self.actor.state_dict(),
                "q": self.q.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q.load_state_dict(ckpt["q"])
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.q_target = copy.deepcopy(self.q).eval()


class TD3:
    def __init__(
        self,
        obs_dim,
        act_dim=1,
        act_limit=2.0,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.act_limit = float(act_limit)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.update_it = 0

        self.actor = DeterministicActor(obs_dim, act_dim, act_limit).to(self.device)
        self.q1 = Critic(obs_dim, act_dim).to(self.device)
        self.q2 = Critic(obs_dim, act_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=1e-3)

    def select_action(self, obs, deterministic=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(obs_t)
        return a.cpu().numpy()[0]

    def _polyak_update(self, source_net, target_net):
        for p, p_t in zip(source_net.parameters(), target_net.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch):
        self.update_it += 1
        obs, act, rew, next_obs, done = _to_tensors(batch, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs) + noise).clamp(-self.act_limit, self.act_limit)
            next_act = _shield_action_from_obs(next_obs, next_act)
            target_q = rew + self.gamma * (1.0 - done) * torch.min(
                self.q1_target(next_obs, next_act), self.q2_target(next_obs, next_act)
            )

        q1_loss = F.mse_loss(self.q1(obs, act), target_q)
        q2_loss = F.mse_loss(self.q2(obs, act), target_q)
        self.critic_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10.0)
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.update_it % self.policy_delay == 0:
            for p in self.q1.parameters():
                p.requires_grad = False
            a_pi = self.actor(obs)
            a_exec = _shield_action_from_obs(obs, a_pi)
            actor_loss = -(self.q1(obs, a_exec)).mean() + 5e-3 * (a_pi ** 2).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()
            for p in self.q1.parameters():
                p.requires_grad = True

            self._polyak_update(self.actor, self.actor_target)
            self._polyak_update(self.q1, self.q1_target)
            self._polyak_update(self.q2, self.q2_target)
        else:
            self._polyak_update(self.q1, self.q1_target)
            self._polyak_update(self.q2, self.q2_target)

        return {
            "q_loss": (q1_loss + q2_loss).item(),
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else float(actor_loss),
        }

    def save(self, path):
        _atomic_torch_save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()


class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = build_mlp([obs_dim, 256, 256, act_dim], act=nn.Tanh, out_act=nn.Identity)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.act_limit = float(act_limit)

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std).clamp(1e-4, 10.0)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        pre_tanh = dist.rsample()
        logp = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        logp -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1, keepdim=True)
        a = torch.tanh(pre_tanh) * self.act_limit
        return a, logp

    def act(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        if deterministic:
            a = mu
        else:
            a = Normal(mu, std).sample()
        a = torch.tanh(a) * self.act_limit
        a = _shield_action_from_obs(obs, a)
        return a


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim=1,
        act_limit=2.0,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_iters=80,
        target_kl=0.02,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_ratio = float(clip_ratio)
        self.train_iters = int(train_iters)
        self.target_kl = float(target_kl)
        self.act_limit = float(act_limit)

        self.pi = PPOActor(obs_dim, act_dim, act_limit).to(self.device)
        self.v = ValueNet(obs_dim).to(self.device)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=vf_lr)

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.pi.act(obs_t, deterministic=deterministic)
        return a.cpu().numpy()[0]

    def compute_gae(self, rewards, values, dones, last_value):
        """
        学术道德修正：正确处理 episode 边界。
        当 done=True 时，gae 应重置为 0，避免跨 episode 的 bootstrap 错误。
        """
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t] > 0.5:
                gae = 0.0
            next_v = 0.0 if dones[t] > 0.5 else (last_value if t == len(rewards) - 1 else values[t + 1])
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        ret = adv + values
        return adv, ret

    def update(self, data):
        obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(data["act"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(data["adv"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        ret = torch.as_tensor(data["ret"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        logp_old = torch.as_tensor(data["logp"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pi_loss_v = 0.0
        v_loss_v = 0.0
        kl_v = 0.0

        for _ in range(self.train_iters):
            mu, std = self.pi(obs)
            dist = Normal(mu, std)
            pre_tanh = torch.atanh((act / self.act_limit).clamp(-0.999, 0.999))
            logp = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
            logp -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1, keepdim=True)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()

            v_pred = self.v(obs)
            v_loss = F.mse_loss(v_pred, ret)

            self.pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=10.0)
            self.pi_opt.step()

            self.v_opt.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=10.0)
            self.v_opt.step()

            with torch.no_grad():
                kl = (logp_old - logp).mean().item()
            pi_loss_v = float(pi_loss.item())
            v_loss_v = float(v_loss.item())
            kl_v = float(kl)
            if kl > 1.5 * self.target_kl:
                break

        return {"pi_loss": pi_loss_v, "v_loss": v_loss_v, "kl": kl_v}

    def save(self, path):
        _atomic_torch_save({"pi": self.pi.state_dict(), "v": self.v.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.pi.load_state_dict(ckpt["pi"])
        self.v.load_state_dict(ckpt["v"])

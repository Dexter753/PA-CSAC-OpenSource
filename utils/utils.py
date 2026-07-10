import random
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def set_seed(seed=42):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

class ReplayBuffer:
    def __init__(self, max_size=300000, obs_dim=None, act_dim=1, seed=None):
        if obs_dim is None:
            raise ValueError("ReplayBuffer: obs_dim must be explicitly provided (current environment uses 20-dim observation space)")
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)
        # 使用固定大小的 NumPy 数组预分配内存，大幅提升采样速度
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.cost_buf = np.zeros(max_size, dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)

    def add(self, obs, act, rew, cost, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        if self.size == 0:
            raise ValueError("ReplayBuffer is empty, cannot sample")
        indices = self._rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs_buf[indices],
            "act": self.act_buf[indices],
            "rew": self.rew_buf[indices],
            "cost": self.cost_buf[indices],
            "next_obs": self.next_obs_buf[indices],
            "done": self.done_buf[indices],
        }

    def __len__(self):
        return self.size

_HEV_PARAMS_CACHE = {
    "mass_kg": 1500.0,
    "wheel_radius_m": 0.31,
    "air_density": 1.225,
    "drag_coeff": 0.29,
    "frontal_area_m2": 2.2,
    "rolling_resist_coeff": 0.01,
    "gravity": 9.81,
    "battery_capacity_kwh": 1.8,
    "soc_min": 0.35,
    "soc_max": 0.85,
    "soc_target": 0.60,
    "engine_eff": 0.35,
    "motor_eff": 0.92,
    "regen_eff": 0.70,
    "fuel_lhv_j_per_kg": 42.6e6,
    "elec_eq_factor": 2.8,
    "max_gap_m": 120.0,
    "hard_upper_safe_ratio": 1.40,  # 适度提高上限比例
    "hard_upper_maxgap_ratio": 0.60,
    "lower_violation_ratio": 0.92,  # 适度放宽下限
    "upper_cost_weight": 0.20,  # 上界仅作软代价，不再主导策略
    "jerk_limit": 3.0,
    # 动态权重参数
    "weight_mode": "dynamic",
    "w_energy_base": 0.50,
    "w_safe_base": 0.50,
    "safety_weight_factor": 0.5,
    "uncertainty_weight_factor": 0.3,
    "soc_weight_factor": 0.2,
    "w_safe_min_dynamic": 0.40,
    "w_safe_max_dynamic": 0.70,
}

def industry_hev_params():
    """
    行业常用HEV参数 (Toyota Prius 级别)
    注意：返回字典的副本，避免修改全局缓存导致环境参数不一致
    """
    import copy
    return copy.deepcopy(_HEV_PARAMS_CACHE)

def hev_parallel_energy(v_ego, acc, soc, dt=1.0, params=None):
    """
    学术级：基于逻辑门限的并联混动能耗物理模型 (ECMS V5 物理回归版)
    完全使用真实物理参数，杜绝任何人工缩放
    """
    p = industry_hev_params() if params is None else params

    v = float(max(float(v_ego), 0.0)) if np.isfinite(v_ego) else 0.0
    a = float(acc) if np.isfinite(acc) else 0.0
    dt = float(dt) if np.isfinite(dt) else 1.0
    dt = float(np.clip(dt, 1e-3, 10.0))

    if v < 0.2 and a < 0.0:
        a = 0.0

    v_avg = float(max(0.0, v + 0.5 * a * dt))

    f_roll = p["mass_kg"] * p["gravity"] * p["rolling_resist_coeff"]
    f_drag = 0.5 * p["air_density"] * p["drag_coeff"] * p["frontal_area_m2"] * v_avg**2
    f_inertial = p["mass_kg"] * a

    if v_avg <= 1e-6:
        p_wheel = 0.0
    else:
        p_wheel = (f_roll + f_drag + f_inertial) * v_avg

    p_aux = 1000.0
    p_total_demand = p_wheel + p_aux

    fuel_energy_j = 0.0
    elec_energy_j = 0.0

    p_ev_max = 15000.0
    if p_total_demand > 0.0:
        if p_total_demand < p_ev_max and soc > p["soc_min"]:
            elec_energy_j = (p_total_demand / p["motor_eff"]) * dt
        else:
            p_eng = p_total_demand * 0.8
            p_mot = p_total_demand * 0.2
            fuel_energy_j = (p_eng / p["engine_eff"]) * dt
            elec_energy_j = (p_mot / p["motor_eff"]) * dt
    else:
        if v > 0.5:
            p_brake = -p_wheel
            p_regen_recov = min(p_brake, p_ev_max) * p["regen_eff"]
        else:
            p_regen_recov = 0.0
        elec_power = p_aux - p_regen_recov
        elec_energy_j = elec_power * dt

    battery_cap_j = p["battery_capacity_kwh"] * 3.6e6
    soc_next = soc - (elec_energy_j / battery_cap_j)
    soc_next = float(np.clip(soc_next, p["soc_min"], p["soc_max"]))

    s_penalty = 1.0 + 10.0 * (p["soc_target"] - soc)**3
    s_penalty = float(max(s_penalty, 0.1))
    net_elec_j = (soc - soc_next) * battery_cap_j
    total_energy_j = fuel_energy_j + net_elec_j * p["elec_eq_factor"] * s_penalty
    total_energy_j = float(max(float(total_energy_j), 0.0))

    return float(total_energy_j), soc_next

def dynamic_safe_distance(v_ego, v_lead, sigma_v, base_gap=5.0, tau=1.2, b=4.0, k_sigma=0.9):
    delta_v = max(v_ego - v_lead, 0.0)
    d_safe = base_gap + tau * v_ego + (delta_v**2) / (2 * b + 1e-6) + k_sigma * sigma_v
    return float(d_safe)

def dynamic_reward_weights(
    d_gap, d_safe, sigma_mean, soc, v_ego, v_lead=None, params=None
):
    """
    动态权重调整算法：根据系统实时状态自动调整节能与安全权重
    
    参数：
        d_gap: 当前车距 (m)
        d_safe: 安全车距 (m)
        sigma_mean: 预测不确定性 (m/s)
        soc: 当前SOC值
        v_ego: 自车速度 (m/s)
        v_lead: 前车速度 (m/s)
        params: 配置参数
    
    返回：
        w_energy: 节能权重 [0, 1]
        w_safe: 安全权重 [0, 1]
        diagnostics: 权重调整诊断信息
    """
    p = params if params is not None else {}
    
    # 基础配置
    w_energy_base = float(p.get("w_energy_base", 0.50))
    w_safe_base = float(p.get("w_safe_base", 0.50))
    
    # 安全风险评估：车距越近风险越高
    gap_ratio = float(d_gap / max(d_safe, 1e-6))
    safety_risk = 1.0 - min(gap_ratio, 1.5) / 1.5  # [0, 1]
    
    # TTC风险评估：TTC<5s开始增加风险
    closing_speed = max(float(v_ego - v_lead), 0.0) if v_lead is not None else 0.0
    ttc = float(d_gap / (closing_speed + 1e-6)) if closing_speed > 1e-6 else float("inf")
    ttc_risk = max(0.0, 1.0 - ttc / 5.0)
    
    # 预测不确定性风险
    sigma_ref = float(p.get("sigma_ref", 1.8))
    uncertainty_risk = min(float(sigma_mean / sigma_ref), 1.0)
    
    # SOC风险（电量越低越需要节能）
    soc_target = float(p.get("soc_target", 0.60))
    soc_deviation = abs(soc - soc_target)
    soc_risk = min(soc_deviation / 0.3, 1.0)
    
    # 综合风险计算
    safety_weight_factor = float(p.get("safety_weight_factor", 0.6))
    uncertainty_weight_factor = float(p.get("uncertainty_weight_factor", 0.2))
    soc_weight_factor = float(p.get("soc_weight_factor", 0.2))
    
    combined_safety_risk = (
        safety_weight_factor * (safety_risk * 0.7 + ttc_risk * 0.3) +
        uncertainty_weight_factor * uncertainty_risk +
        soc_weight_factor * soc_risk
    )
    
    # 动态调整权重
    w_safe_min = float(p.get("w_safe_min_dynamic", 0.35))
    w_safe_max = float(p.get("w_safe_max_dynamic", 0.70))
    w_safe = w_safe_min + (w_safe_max - w_safe_min) * combined_safety_risk
    w_energy = 1.0 - w_safe
    
    # 确保权重在合理范围
    w_energy = max(0.25, min(0.75, w_energy))
    w_safe = max(0.25, min(0.75, w_safe))
    
    diagnostics = {
        "safety_risk": float(safety_risk),
        "ttc_risk": float(ttc_risk),
        "uncertainty_risk": float(uncertainty_risk),
        "soc_risk": float(soc_risk),
        "combined_safety_risk": float(combined_safety_risk),
    }
    
    return w_energy, w_safe, diagnostics


def adaptive_reward_weights(
    sigma_mean,
    sigma_ref=1.8,
    sharpness=2.2,
    w_safe_min=0.42,
    w_safe_max=0.68,
):
    """
    基于预测不确定性的动态权重切换（退饱和版本）
    - 使用相对 sigma 偏差，减弱不同 mode 下 sigma 偏置导致的顶格现象
    - 约束 w_safe 区间，避免长期锁死在极端值
    """
    sigma = float(np.clip(float(sigma_mean), 0.0, 10.0))
    sigma_ref = float(max(float(sigma_ref), 1e-4))
    sharpness = float(np.clip(float(sharpness), 0.3, 8.0))
    w_safe_min = float(np.clip(float(w_safe_min), 0.05, 0.95))
    w_safe_max = float(np.clip(float(w_safe_max), w_safe_min + 1e-3, 0.98))

    # 以相对偏差作为输入，减少绝对量纲对饱和的影响
    rel_sigma = (sigma - sigma_ref) / sigma_ref
    w_safe_base = 1.0 / (1.0 + np.exp(-sharpness * rel_sigma))

    w_safe = w_safe_min + (w_safe_max - w_safe_min) * w_safe_base
    w_safe = float(np.clip(w_safe, w_safe_min, w_safe_max))
    w_energy = float(1.0 - w_safe)

    # 消融实验公平性修正：当 sigma 为固定参考值（如1.8）时，
    # 确保 w_safe 不会过度偏向极端，保持合理的节能-安全平衡
    if abs(float(sigma) - float(sigma_ref)) < 0.05:
        w_safe = float(w_safe_min + (w_safe_max - w_safe_min) * 0.5)
        w_safe = float(np.clip(w_safe, w_safe_min, w_safe_max))
        w_energy = float(1.0 - w_safe)

    return w_energy, w_safe

def reward_and_constraint(v_ego, acc, jerk, d_gap, d_safe, soc, sigma_mean, v_lead=None, dt=1.0, params=None, debug=False):
    """多目标统一奖励：节能+安全+舒适+SOC，并显式抑制过大车距"""
    p = industry_hev_params() if params is None else params

    # 权重调整模式选择
    # mode: 'fixed' -> 固定权重; 'adaptive' -> 基于sigma的自适应; 'dynamic' -> 多因素动态调整
    raw_weight_mode = str(p.get("weight_mode", "dynamic")) if isinstance(p, dict) else "dynamic"
    weight_mode = raw_weight_mode.lower().strip()
    sigma_mean = float(sigma_mean) if np.isfinite(sigma_mean) else 0.0
    sigma_mean = float(np.clip(sigma_mean, 0.0, 10.0))
    
    if weight_mode == "dynamic":
        w_energy, w_safe, weight_diagnostics = dynamic_reward_weights(
            d_gap, d_safe, sigma_mean, soc, v_ego, v_lead, p
        )
    elif weight_mode == "adaptive":
        sigma_ref = float(p.get("sigma_ref", 1.8)) if isinstance(p, dict) else 1.8
        sharpness = float(p.get("sigma_sharpness", 2.2)) if isinstance(p, dict) else 2.2
        w_safe_min = float(p.get("w_safe_min", 0.40)) if isinstance(p, dict) else 0.40
        w_safe_max = float(p.get("w_safe_max", 0.60)) if isinstance(p, dict) else 0.60
        w_energy, w_safe = adaptive_reward_weights(
            sigma_mean,
            sigma_ref=sigma_ref,
            sharpness=sharpness,
            w_safe_min=w_safe_min,
            w_safe_max=w_safe_max,
        )
        weight_diagnostics = {"mode": "adaptive"}
    else:
        # 固定权重模式 - 使用更平衡的权重
        w_energy = float(p.get("w_energy_fixed", 0.50)) if isinstance(p, dict) else 0.50
        w_safe = float(p.get("w_safe_fixed", 0.50)) if isinstance(p, dict) else 0.50
        s = float(w_energy + w_safe)
        if (not np.isfinite(s)) or (s <= 1e-8):
            w_energy, w_safe = 0.50, 0.50
        else:
            w_energy = float(w_energy / s)
            w_safe = float(w_safe / s)
        weight_diagnostics = {"mode": "fixed"}
        if weight_mode not in {"fixed", "adaptive", "dynamic"}:
            weight_diagnostics["invalid_weight_mode"] = raw_weight_mode
        weight_mode = "fixed"

    dt = float(dt) if np.isfinite(dt) else 1.0
    dt = float(np.clip(dt, 1e-3, 10.0))

    energy_j, soc_next = hev_parallel_energy(v_ego, acc, soc, dt=dt, params=p)
    base_energy_j, _ = hev_parallel_energy(v_ego, 0.0, soc, dt=dt, params=p)

    energy_cost_j = float(max(float(energy_j), 0.0)) if np.isfinite(energy_j) else 0.0
    base_energy_cost_j = float(max(float(base_energy_j), 0.0)) if np.isfinite(base_energy_j) else 0.0
    energy_saving_j = base_energy_cost_j - energy_cost_j

    # 1) 节能项 - 减小奖励幅度，增加稳定性
    r_energy = 0.8 * np.tanh(energy_saving_j / 8000.0)  # 范围 [-0.8, 0.8]

    # 2) 安全项 - 大幅减小惩罚幅度，避免Q值爆炸
    # 安全目标区：[d_safe, 1.2 * d_safe]
    if d_gap < d_safe:
        deficit = float(min(d_safe - d_gap, 10.0))  # 限制最大惩罚
        r_safe = -0.8 * np.tanh(deficit / 2.0)  # 范围 [-0.8, 0]
    else:
        safety_zone_upper = 1.2 * d_safe
        if d_gap <= safety_zone_upper:
            r_safe = 0.1 * (1.0 - np.tanh(max(0.0, d_gap - d_safe) / 3.0))  # 小正奖励
        else:
            r_safe = -0.08 * np.tanh(max(0.0, d_gap - safety_zone_upper) / 10.0)  # 轻微惩罚
        
        if v_lead is not None and np.isfinite(v_lead):
            closing_speed = max(float(v_ego - v_lead), 0.0)
            near_margin = 2.0
            near = max(0.0, near_margin - float(d_gap - d_safe))
            if near > 0 and closing_speed > 1.0:
                r_safe = float(r_safe - 0.35 * np.tanh(near / 0.8) * np.tanh(closing_speed / 0.6))

    # 3) 跟驰目标带
    # 终端证据：warmstart 后 700 步最好，但 2000/4949 步 lower 与 jerk 明显抬升，
    # 说明“缩小车距”驱动力仍偏强。这里改为非对称跟驰项：
    # - gap 偏大时仅温和鼓励跟进，避免一开始就猛追；
    # - gap 偏小时更明确惩罚，抑制激进贴近前车。
    sigma_target_scale = float(p.get("sigma_target_scale", 0.55)) if isinstance(p, dict) else 0.55
    sigma_target_bias = float(p.get("sigma_target_bias", 0.3)) if isinstance(p, dict) else 0.3
    target_gap = d_safe + sigma_target_scale * float(sigma_mean) + sigma_target_bias
    gap_delta = float(d_gap - target_gap)
    gap_err_abs = abs(gap_delta)
    # 19:51 的终端证据显示：上一轮“反保守”修改把策略从大间距保守
    # 直接推到了高 violation / 高 rate-limit 的另一端。
    # 这里回收到中间区：仍然惩罚长期掉队，但不再像上一轮那样过猛追车。
    follow_far_coef = float(p.get("follow_far_coef", 0.15)) if isinstance(p, dict) else 0.15
    follow_far_scale = float(p.get("follow_far_scale", 7.6)) if isinstance(p, dict) else 7.6
    follow_close_coef = float(p.get("follow_close_coef", 0.42)) if isinstance(p, dict) else 0.42
    follow_close_scale = float(p.get("follow_close_scale", 3.6)) if isinstance(p, dict) else 3.6
    if gap_delta >= 0.0:
        r_follow = -follow_far_coef * np.tanh(gap_delta / max(follow_far_scale, 1e-6))
    else:
        r_follow = -follow_close_coef * np.tanh((-gap_delta) / max(follow_close_scale, 1e-6))
    # 在目标跟驰带内提供小幅正奖励，避免策略只有“少犯错”而缺少“学对了”的信号
    target_band_m = float(p.get("target_band_m", 3.2)) if isinstance(p, dict) else 3.2
    if gap_err_abs <= target_band_m:
        r_follow = float(r_follow + 0.18 * (1.0 - gap_err_abs / max(target_band_m, 1e-6)))
    gap_hi = max(0.0, float(d_gap - (target_gap + 10.0)))
    gap_hi_coef = float(p.get("gap_hi_coef", 0.006)) if isinstance(p, dict) else 0.006
    r_follow = float(r_follow - gap_hi_coef * gap_hi)
    max_gap_m = float(p.get("max_gap_m", 120.0)) if isinstance(p, dict) else 120.0
    max_gap_m = float(np.clip(max_gap_m, 50.0, 1000.0))
    # 软上界应明显宽于目标跟驰带，只在“长期明显掉队”时启动，而不是一开始就主导奖励
    d_upper_soft = max(1.45 * target_gap + 3.0, 0.45 * max_gap_m)
    gap_excess = max(0.0, float(d_gap - d_upper_soft))
    gap_upper_tanh_coef = float(p.get("gap_upper_tanh_coef", 0.27)) if isinstance(p, dict) else 0.27
    gap_upper_lin_coef = float(p.get("gap_upper_lin_coef", 0.006)) if isinstance(p, dict) else 0.006
    gap_upper_quad_coef = float(p.get("gap_upper_quad_coef", 0.0002)) if isinstance(p, dict) else 0.0002
    r_gap_upper = (
        -gap_upper_tanh_coef * np.tanh(gap_excess / 4.5)
        - gap_upper_lin_coef * gap_excess
        - gap_upper_quad_coef * (gap_excess ** 2)
    )

    if v_lead is not None and np.isfinite(v_lead):
        # 速度匹配仅在“确实偏远、需要追赶”时增强；
        # 接近目标带后应明显减弱，避免继续加速把策略推向 lower/jerk 风险。
        catch_need = float(np.tanh(max(gap_delta - target_band_m, 0.0) / 6.0))
        match_scale = 0.35 + 0.65 * catch_need
        r_v_match = -0.18 * np.tanh(abs(v_ego - v_lead) / 3.2) * match_scale
        r_stop = -0.40 if (v_lead > 2.0 and v_ego < 0.5) else 0.0
        gap_factor = float(np.tanh(gap_excess / 12.0))
        vel_factor = float(np.tanh((float(v_lead) - float(v_ego)) / 1.5))
        gap_recover = float(np.tanh(max(gap_delta - target_band_m, 0.0) / 9.5))
        r_catch = 0.35 * max(gap_factor, gap_recover) * vel_factor
        r_brake_behind = -0.12 * max(-float(acc), 0.0) * gap_factor * max(vel_factor, 0.0)
    else:
        r_v_match = 0.0
        r_stop = 0.0
        r_catch = 0.0
        r_brake_behind = 0.0

    # 若已经贴近目标/安全边界，继续加速应被明确惩罚，防止“越追越近、越近越冲”。
    close_gap = max(0.0, float(target_gap - d_gap))
    approach_acc_penalty = -0.12 * max(float(acc), 0.0) * np.tanh(close_gap / 2.5)
    if v_lead is not None and np.isfinite(v_lead):
        closing_speed = max(float(v_ego - v_lead), 0.0)
        approach_acc_penalty -= 0.08 * np.tanh(close_gap / 2.0) * np.tanh(closing_speed / 0.8)

    comfort_jerk_coef = float(p.get("r_comfort_jerk_coef", 0.20)) if isinstance(p, dict) else 0.20
    comfort_acc_coef = float(p.get("r_comfort_acc_coef", 0.12)) if isinstance(p, dict) else 0.12
    r_comfort = -comfort_jerk_coef * np.tanh(abs(jerk) / 1.4) - comfort_acc_coef * np.tanh(abs(acc) / 2.2)
    r_brake = -0.06 * max(-acc, 0.0)

    # 5) SOC 保持
    r_soc = -0.14 * np.tanh(abs(soc_next - p["soc_target"]) / 0.018)

    # V19修复：平衡探索与跟车激励，reward_bias可配置（支持贝叶斯优化自动搜索）
    # 日志证据：V18移除偏置后奖励始终≤0，策略坍缩到保守动作（viol_rate=80%）
    # +0.15偏置：提供正反馈激励跟车，但不足以让Q翻正
    # 数据对比：trial_030 reward max = +0.029~+0.054（有正奖励），说明clip(+0.20)是有效的
    # verify_20000 reward max = -0.045（无正奖励），根因是Phase1过长导致策略固化，不是clip问题
    reward_bias = float(p.get("reward_bias", 0.15)) if isinstance(p, dict) else 0.15
    raw_reward = 0.95 * (w_energy * r_energy + w_safe * r_safe) + r_follow + r_gap_upper + r_v_match + r_stop + r_catch + r_brake_behind + approach_acc_penalty + r_comfort + r_brake + r_soc + reward_bias
    # 非对称裁剪：允许少量正奖励，上限由reward_bias+0.05决定
    reward = float(np.clip(raw_reward, -10.0, reward_bias + 0.05))

    hard_upper_safe_ratio = float(p.get("hard_upper_safe_ratio", 1.40)) if isinstance(p, dict) else 1.40
    hard_upper_maxgap_ratio = float(p.get("hard_upper_maxgap_ratio", 0.60)) if isinstance(p, dict) else 0.60
    jerk_limit = float(p.get("jerk_limit", 3.0)) if isinstance(p, dict) else 3.0
    upper_cost_weight = float(p.get("upper_cost_weight", 0.20)) if isinstance(p, dict) else 0.20

    d_gap_hard_upper = max(hard_upper_safe_ratio * d_safe, hard_upper_maxgap_ratio * max_gap_m)
    # 离散时间与传感抖动下，lower 违约使用可配置容忍比例，避免过严误报
    lower_violation_ratio = float(p.get("lower_violation_ratio", 0.92)) if isinstance(p, dict) else 0.92
    lower_violation_ratio = float(np.clip(lower_violation_ratio, 0.90, 1.00))
    viol_lower = bool(d_gap < lower_violation_ratio * d_safe)
    viol_jerk = bool(abs(jerk) > jerk_limit)
    viol_upper = bool(d_gap > d_gap_hard_upper)

    closing_speed = max(float(v_ego - v_lead), 0.0) if v_lead is not None else 0.0
    ttc = float(d_gap / (closing_speed + 1e-6)) if closing_speed > 1e-6 else float("inf")
    viol_ttc = bool(ttc < 1.0)

    # 上界偏大属于效率/跟驰质量问题，不应与碰撞、过大jerk、TTC风险等硬安全事件混为一谈。
    # 否则 evaluate() 中 violation_rate 与 upper_rate 会对同一问题进行双重惩罚，诱导策略过度保守。
    violation_event = float(viol_lower or viol_jerk or viol_ttc)
    # 上界违约保留为软成本，并继续单独统计 avg_viol_upper
    lower_margin = float(lower_violation_ratio * d_safe)
    lower_soft_shortfall = max(0.0, lower_margin - float(d_gap))
    lower_soft_cost = float(min(1.0, lower_soft_shortfall / max(0.20 * max(float(d_safe), 1.0), 1.0)))
    jerk_soft_cost = float(min(1.0, max(0.0, abs(float(jerk)) - 0.85 * jerk_limit) / max(0.35 * jerk_limit, 1e-6)))
    cost = float(
        min(
            1.0,
            float(viol_lower or viol_jerk or viol_ttc)
            + upper_cost_weight * float(viol_upper)
            + 0.35 * lower_soft_cost
            + 0.12 * jerk_soft_cost,
        )
    )

    terms = {
        "r_energy": float(r_energy),
        "r_safe": float(r_safe),
        "r_follow": float(r_follow),
        "r_gap_upper": float(r_gap_upper),
        "r_v_match": float(r_v_match),
        "r_stop": float(r_stop),
        "r_catch": float(r_catch),
        "r_brake_behind": float(r_brake_behind),
        "approach_acc_penalty": float(approach_acc_penalty),
        "r_comfort": float(r_comfort),
        "r_brake": float(r_brake),
        "r_soc": float(r_soc),
        "w_energy": float(w_energy),
        "w_safe": float(w_safe),
        "target_gap": float(target_gap),
        "d_upper_soft": float(d_upper_soft),
        "d_hard_upper": float(d_gap_hard_upper),
        "viol_lower": float(viol_lower),
        "viol_jerk": float(viol_jerk),
        "viol_upper": float(viol_upper),
        "ttc": float(ttc),
        "viol_ttc": float(viol_ttc),
        "violation_event": float(violation_event),
        "violation_cost": float(cost),
        "lower_soft_cost": float(lower_soft_cost),
        "jerk_soft_cost": float(jerk_soft_cost),
        # 添加权重诊断信息
        "weight_mode": str(weight_mode),
    }
    # 合并动态权重诊断信息
    terms.update(weight_diagnostics)

    return float(reward), float(cost), float(energy_j), soc_next, terms

def _configure_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.8,
    })

PAPER_COLORS = {
    'PA-CSAC': '#1f77b4',
    'ACC': '#d62728',
    'MPC': '#2ca02c',
    'LQR': '#ff7f0e',
    'IDM': '#8c564b',
    'DDPG': '#9467bd',
    'TD3': '#e377c2',
    'SAC': '#17becf',
    'PPO': '#7f7f7f',
    'Baseline_no_prediction': '#bcbd22',
    'Baseline_mean_prediction': '#e7ba52',
    'Baseline_lstm_prediction': '#aec7e8',
    'Baseline_transformer_prediction': '#98df8a',
    'Baseline_pa_csac': '#1f77b4',
}

PAPER_LINESTYLES = {
    'PA-CSAC': '-',
    'ACC': '--',
    'MPC': '-.',
    'LQR': ':',
    'IDM': (0, (3, 1, 1, 1)),
    'DDPG': '-',
    'TD3': '--',
    'SAC': '-.',
    'PPO': ':',
}

COMPONENT_ABLATION_COLORS = {
    'pa_csac_full': '#2ca02c',
    'pa_csac_lagrangian': '#1f77b4',
    'no_lagrangian': '#d62728',
    'no_embedding': '#ff7f0e',
    'no_lagrangian_no_embedding': '#9467bd',
}

COMPONENT_ABLATION_LABELS = {
    'pa_csac_full': 'PA-CSAC\n(Penalty)',
    'pa_csac_lagrangian': 'PA-CSAC\n(Lagrangian)',
    'no_lagrangian': 'w/o\nCostConstraint',
    'no_embedding': 'w/o Prob\nEmbedding',
    'no_lagrangian_no_embedding': 'w/o\nBoth',
}


def plot_multi_algo_comparison(all_results_dict, save_path):
    if not all_results_dict: return
    _configure_paper_style()
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)
    
    for name, records in all_results_dict.items():
        if not records: continue
        steps = np.arange(len(records))
        v_ego = np.array([r["v_ego"] for r in records], dtype=float)
        d_gap = np.array([r["d_gap"] for r in records], dtype=float)
        soc = np.array([r["soc"] for r in records], dtype=float)
        fuel_j = np.array([r["fuel"] for r in records], dtype=float)
        color = PAPER_COLORS.get(name, 'gray')
        ls = PAPER_LINESTYLES.get(name, '-')
        
        axs[0, 0].plot(steps, v_ego, color=color, linestyle=ls, label=name)
        axs[0, 1].plot(steps, d_gap, color=color, linestyle=ls, label=name)
        axs[1, 0].plot(steps, soc, color=color, linestyle=ls, label=name)
        cum_fuel_g = np.cumsum(np.maximum(fuel_j, 0.0)) / 42600.0
        axs[1, 1].plot(steps, cum_fuel_g, color=color, linestyle=ls, label=name)

    axs[0, 0].set_title('(a) Velocity Tracking')
    axs[0, 0].set_ylabel('Velocity (m/s)')
    axs[0, 0].set_xlabel('Time Step')
    first_records = next((v for v in all_results_dict.values() if v), [])
    if first_records:
        axs[0, 0].plot(np.arange(len(first_records)), [r["v_lead"] for r in first_records],
                       color='black', linestyle=(0, (1, 2)), linewidth=1.2, label='Lead Vehicle', alpha=0.55)
    axs[0, 0].legend(loc='best', framealpha=0.85)

    axs[0, 1].set_title('(b) Inter-Vehicle Gap')
    axs[0, 1].set_ylabel('Gap (m)')
    axs[0, 1].set_xlabel('Time Step')
    if 'PA-CSAC' in all_results_dict:
        pa_records = all_results_dict['PA-CSAC']
        d_safe_vals = [r["d_safe"] for r in pa_records]
        axs[0, 1].plot(np.arange(len(pa_records)), d_safe_vals,
                       color='black', linestyle=(0, (1, 2)), linewidth=1.2, label='Safe Distance', alpha=0.55)
    axs[0, 1].legend(loc='best', framealpha=0.85)

    axs[1, 0].set_title('(c) Battery SOC')
    axs[1, 0].set_ylabel('SOC')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].axhline(y=0.6, color='black', linestyle=':', linewidth=1.0, alpha=0.4)
    all_socs = [r["soc"] for recs in all_results_dict.values() for r in recs]
    if all_socs:
        soc_min, soc_max = min(all_socs), max(all_socs)
        pad = max(0.005, (soc_max - soc_min) * 0.5)
        axs[1, 0].set_ylim(soc_min - pad, soc_max + pad)

    axs[1, 1].set_title('(d) Cumulative Fuel Consumption')
    axs[1, 1].set_ylabel('Cum. Fuel (g)')
    axs[1, 1].set_xlabel('Time Step')

    plt.savefig(save_path)
    plt.close()

def _estimate_engine_points(records, params=None):
    p = industry_hev_params() if params is None else params
    if not records:
        return np.array([]), np.array([])
    v = np.maximum(np.array([r["v_ego"] for r in records], dtype=float), 0.3)
    a = np.array([r.get("acc", 0.0) for r in records], dtype=float)
    om_w = v / p["wheel_radius_m"]
    gear, eta = 9.5, 0.9
    om_e = np.clip(om_w * gear, 80, 480)
    f_roll = p["mass_kg"] * p["gravity"] * p["rolling_resist_coeff"]
    f_drag = 0.5 * p["air_density"] * p["drag_coeff"] * p["frontal_area_m2"] * v**2
    f_iner = p["mass_kg"] * a
    t_w = np.maximum((f_roll + f_drag + f_iner), 0.0) * p["wheel_radius_m"]
    t_e = np.clip(t_w / (gear * eta + 1e-6), 0, 120)
    return om_e, t_e

def plot_map_style_figures(all_results_dict, save_dir):
    if not all_results_dict:
        return
    os.makedirs(save_dir, exist_ok=True)
    _configure_paper_style()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    all_om, all_tq = [], []
    for recs in all_results_dict.values():
        om_e, tq_e = _estimate_engine_points(recs)
        all_om.extend(om_e.tolist())
        all_tq.extend(tq_e.tolist())

    if all_om:
        h = axs[0, 0].hist2d(all_om, all_tq, bins=[30, 26], cmap='YlGnBu')
        fig.colorbar(h[3], ax=axs[0, 0], label='Counts')

    for name, recs in all_results_dict.items():
        om_e, tq_e = _estimate_engine_points(recs)
        if len(om_e) > 0:
            axs[0, 0].scatter(om_e, tq_e, s=14, alpha=0.55, c=PAPER_COLORS.get(name, 'gray'),
                              edgecolors='none', label=name)

    axs[0, 0].set_title('(a) Data-driven Working Point Map')
    axs[0, 0].set_xlabel('Engine Speed (rad/s)')
    axs[0, 0].set_ylabel('Engine Torque (Nm)')
    axs[0, 0].legend(fontsize=8, loc='upper right', framealpha=0.85, ncol=2)

    pa = all_results_dict.get('PA-CSAC', next(iter(all_results_dict.values())))
    gap = np.array([r['d_gap'] for r in pa], dtype=float)
    soc_arr = np.array([r['soc'] for r in pa], dtype=float)
    h2 = axs[0, 1].hist2d(gap, soc_arr, bins=[20, 16], cmap='YlOrRd')
    fig.colorbar(h2[3], ax=axs[0, 1], label='Counts')
    axs[0, 1].set_title('(b) SOC-Gap Density (PA-CSAC)')
    axs[0, 1].set_xlabel('Gap (m)')
    axs[0, 1].set_ylabel('SOC')

    for name, recs in all_results_dict.items():
        soc_vals = np.array([r['soc'] for r in recs], dtype=float)
        axs[1, 0].plot(soc_vals, lw=1.8, label=name, color=PAPER_COLORS.get(name, 'gray'))
    axs[1, 0].axhspan(0.55, 0.65, color='mediumpurple', alpha=0.10)
    axs[1, 0].set_title('(c) SOC Trajectories')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('SOC')
    axs[1, 0].legend(fontsize=8, loc='best', framealpha=0.85, ncol=2)

    for name, recs in all_results_dict.items():
        fuel_j = np.array([r['fuel'] for r in recs], dtype=float)
        axs[1, 1].hist(fuel_j / 1000.0, bins=18, alpha=0.35, label=name, color=PAPER_COLORS.get(name, 'gray'))
    axs[1, 1].set_title('(d) Equivalent Energy Distribution')
    axs[1, 1].set_xlabel('Energy per Step (kJ)')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend(fontsize=8, loc='best', framealpha=0.85, ncol=2)

    plt.savefig(os.path.join(save_dir, 'Map_Style_Research_Figure.png'))
    plt.close()


def plot_soc_comparison(all_results_dict, save_path):
    if not all_results_dict:
        return
    _configure_paper_style()
    plt.figure(figsize=(10, 5))
    for name, recs in all_results_dict.items():
        if not recs:
            continue
        soc = np.array([r['soc'] for r in recs], dtype=float)
        plt.plot(np.arange(len(soc)), soc, label=name,
                 color=PAPER_COLORS.get(name, 'gray'),
                 linestyle=PAPER_LINESTYLES.get(name, '-'))
    plt.axhspan(0.58, 0.62, color='mediumpurple', alpha=0.10, label='SOC Target')
    plt.axhline(0.60, color='black', linestyle=':', alpha=0.6)
    plt.title('SOC Comparison Across Controllers')
    plt.xlabel('Time Step')
    plt.ylabel('SOC')
    plt.legend(fontsize=9, loc='best', framealpha=0.85, ncol=2)
    plt.savefig(save_path)
    plt.close()

def plot_paper_ready_results(records, save_dir, name="PA-CSAC"):
    os.makedirs(save_dir, exist_ok=True)
    if not records:
        return
    _configure_paper_style()
    steps = np.arange(len(records))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    v_ego = np.array([r["v_ego"] for r in records], dtype=float)
    v_lead = np.array([r.get("v_lead", 0) for r in records], dtype=float)
    d_gap = np.array([r["d_gap"] for r in records], dtype=float)
    d_safe = np.array([r.get("d_safe", 0) for r in records], dtype=float)
    soc_vals = np.array([r["soc"] for r in records], dtype=float)
    acc_vals = np.array([r["acc"] for r in records], dtype=float)
    fuel_vals = np.array([r["fuel"] for r in records], dtype=float)

    axs[0, 0].plot(steps, v_ego, color='#1f77b4', label='Ego', linewidth=2.0)
    axs[0, 0].plot(steps, v_lead, color='#d62728', linestyle='--', linewidth=1.5, label='Lead', alpha=0.8)
    axs[0, 0].set_title('(a) Velocity Profile')
    axs[0, 0].set_ylabel('Velocity (m/s)')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].legend(loc='best', framealpha=0.85)

    d_safe_alpha = 0.7
    axs[0, 1].fill_between(steps, 0, d_safe, color='red', alpha=0.08, label='Unsafe Region')
    axs[0, 1].plot(steps, d_gap, color='#2ca02c', linewidth=2.0, label='Actual Gap')
    axs[0, 1].plot(steps, d_safe, color='#d62728', linestyle='--', linewidth=1.5, alpha=d_safe_alpha, label='Safe Threshold')
    axs[0, 1].set_title('(b) Inter-Vehicle Gap vs Safe Distance')
    axs[0, 1].set_ylabel('Distance (m)')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].legend(loc='best', framealpha=0.85)

    axs[1, 0].plot(steps, soc_vals, color='#9467bd', linewidth=2.0)
    axs[1, 0].axhline(0.60, linestyle=':', color='black', alpha=0.5, linewidth=1.2)
    axs[1, 0].axhspan(0.58, 0.62, color='mediumpurple', alpha=0.08)
    axs[1, 0].set_title('(c) Battery SOC')
    axs[1, 0].set_ylabel('SOC')
    axs[1, 0].set_xlabel('Time Step')

    colors_bar = np.where(acc_vals >= 0, '#2ca02c', '#d62728')
    axs[1, 1].bar(steps, acc_vals, color=colors_bar, alpha=0.7, width=0.8, label='Acceleration')
    ax2 = axs[1, 1].twinx()
    cum_fuel_g = np.cumsum(np.maximum(fuel_vals, 0.0)) / 42600.0
    ax2.plot(steps, cum_fuel_g, color='#ff7f0e', linewidth=2.2, label='Cum. Fuel')
    axs[1, 1].set_title('(d) Control Action & Cumulative Fuel')
    axs[1, 1].set_ylabel(r'Acceleration (m/s$^2$)')
    axs[1, 1].set_xlabel('Time Step')
    ax2.set_ylabel('Cum. Fuel (g)')
    lines1, labels1 = axs[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.85, fontsize=8)

    plt.savefig(os.path.join(save_dir, f"Paper_Ready_{name}.png"))
    plt.savefig(os.path.join(save_dir, f"plot_benchmark_{name}.png"))
    plt.savefig(os.path.join(save_dir, f"{name}_Trajectory.png"))
    plt.close()

def plot_training_comparison(history_dict, save_path, x_max_steps=None):
    _configure_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'PA-CSAC (Ours)': '#1f77b4', 'SAC': '#17becf', 'PPO': '#7f7f7f',
        'DDPG': '#9467bd', 'TD3': '#e377c2'
    }

    all_smooth_values = []
    for name, hist in history_dict.items():
        if hist is None:
            continue
        if isinstance(hist, dict):
            rewards = hist.get('rewards', [])
            steps = hist.get('steps', None)
        else:
            rewards = hist
            steps = None

        if not rewards:
            continue

        raw = np.array(rewards, dtype=float)
        window = int(max(1, min(len(raw), max(10, len(raw) // 40))))
        smooth = np.convolve(raw, np.ones(window) / window, mode='valid')
        raw_sq = np.convolve(raw**2, np.ones(window) / window, mode='valid')
        smooth_std = np.sqrt(np.maximum(raw_sq - smooth**2, 0.0))
        ci = 1.96 * smooth_std / np.sqrt(window)

        if steps is not None and len(steps) == len(rewards):
            x_s = np.array(steps, dtype=float)[window - 1:]
            n = min(len(x_s), len(smooth))
            x_plot, y_plot = x_s[:n], smooth[:n]
            ci_plot = ci[:n]
            if x_max_steps is not None and n > 0:
                mask = x_plot <= float(x_max_steps)
                x_plot, y_plot = x_plot[mask], y_plot[mask]
                ci_plot = ci_plot[mask]
        else:
            x_plot = np.arange(len(smooth), dtype=float)
            y_plot = smooth
            ci_plot = ci

        color = colors.get(name, None)
        ax.plot(x_plot, y_plot, label=name, color=color, linewidth=2.2)
        ax.fill_between(x_plot, y_plot - ci_plot, y_plot + ci_plot, color=color, alpha=0.10)
        all_smooth_values.extend(y_plot.tolist())

    if all_smooth_values:
        y_min = np.percentile(all_smooth_values, 2)
        y_max = np.percentile(all_smooth_values, 98)
        ax.set_ylim(y_min - abs(y_min) * 0.1, y_max + abs(y_max) * 0.1)

    if x_max_steps is not None:
        ax.set_xlim(0, float(x_max_steps))

    ax.set_title('Training Convergence Comparison')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Smoothed Episode Reward (95% CI)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
    plt.savefig(save_path)
    plt.close()


def plot_component_ablation_results(comp_df, save_path):
    if comp_df is None or comp_df.empty:
        return
    _configure_paper_style()
    metrics = ['fuel_l_per_100km', 'gap_rmse', 'jerk_rmse', 'violation_rate']
    metric_labels = ['Fuel (L/100km)', 'Gap RMSE (m)', 'Jerk RMSE', 'Violation Rate']
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.38, wspace=0.30)

    variants = comp_df['variant'].tolist()
    x = np.arange(len(variants))
    bar_width = 0.55

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axs[idx // 2, idx % 2]
        vals = comp_df[metric].to_numpy(dtype=float)
        colors = [COMPONENT_ABLATION_COLORS.get(v, '#7f7f7f') for v in variants]
        bar_labels = [COMPONENT_ABLATION_LABELS.get(v, v) for v in variants]
        bars = ax.bar(x, vals, bar_width, color=colors, edgecolor='white', linewidth=0.8, alpha=0.9)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            if np.isfinite(v):
                offset = max(abs(v) * 0.01, 0.01)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(f'({chr(97 + idx)}) {label}')
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=0, fontsize=9)
        ax.set_ylabel(label)

    axs[0, 1].set_title('(b) Gap RMSE (m)')
    axs[1, 0].set_title('(c) Jerk RMSE')
    axs[1, 1].set_title('(d) Violation Rate')

    plt.savefig(save_path)
    plt.close()


def apply_safety_shield(acc_cmd, d_gap, d_safe, rel_speed, a_low=-3.0, a_high=2.0, dt_pred=1.0):
    from shield import shield_action_scalar
    a = float(np.clip(acc_cmd, a_low, a_high))
    a = shield_action_scalar(a, d_gap, d_safe, rel_speed, dt_pred)
    return a

def summarize_metrics(records):
    """
    汇总指标：增加焦耳到百公里油耗的换算，提升学术可读性
    """
    if not records: return {}

    # 能量单位：Joules
    energy_j = np.array([r.get("fuel", 0.0) for r in records], dtype=float)
    energy_j = np.nan_to_num(energy_j, nan=0.0, posinf=0.0, neginf=0.0)
    energy_j_pos = np.maximum(energy_j, 0.0)
    gap_error = np.nan_to_num(np.array([r.get("gap_error", 0.0) for r in records], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    jerk = np.nan_to_num(np.array([r.get("jerk", 0.0) for r in records], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    collision = np.nan_to_num(np.array([r.get("collision", 0.0) for r in records], dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    violation_event = np.nan_to_num(np.array([r.get("violation", 0.0) for r in records], dtype=float), nan=1.0, posinf=1.0, neginf=0.0)
    violation_cost = np.nan_to_num(np.array([r.get("violation_cost", r.get("cost", r.get("violation", 0.0))) for r in records], dtype=float), nan=1.0, posinf=1.0, neginf=0.0)
    ttc = np.nan_to_num(np.array([r.get("ttc", float("inf")) for r in records], dtype=float), nan=float("inf"), posinf=float("inf"), neginf=0.0)
    viol_ttc = np.nan_to_num(np.array([r.get("viol_ttc", 0.0) for r in records], dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    numeric_invalid = np.nan_to_num(np.array([r.get("numeric_invalid", 0.0) for r in records], dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    infer_ms = np.nan_to_num(np.array([r.get("infer_ms", 0.0) for r in records], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    brake = np.array([r.get("brake", 0.0) for r in records])
    brake_intensity = np.array([r.get("brake_intensity", 0.0) for r in records])
    soc = np.array([r.get("soc", 0.0) for r in records], dtype=float)
    
    # 计算百公里等效油耗 (L/100km)
    # 假设汽油热值 33MJ/L
    total_energy_j = float(np.sum(energy_j_pos))
    if "x_ego" in records[-1]:
        x_series = np.array([float(r.get("x_ego", 0.0)) for r in records], dtype=float)
        # 仅累计正向位移增量，避免异常回退污染里程
        total_dist_km = float(np.sum(np.maximum(np.diff(x_series, prepend=x_series[0]), 0.0))) / 1000.0
    else:
        total_dist_km = 0.0

    # 回退方案：按 v*dt 进行物理积分，避免默认 dt=1s 的量纲偏差
    if total_dist_km <= 1e-9:
        v_series = np.array([float(r.get("v_ego", 0.0)) for r in records], dtype=float)
        dt_series = np.array([float(r.get("dt", 1.0)) for r in records], dtype=float)
        dt_series = np.clip(np.nan_to_num(dt_series, nan=1.0, posinf=1.0, neginf=1.0), 1e-3, 10.0)
        total_dist_km = float(np.sum(np.maximum(v_series, 0.0) * dt_series)) / 1000.0

    violation_rate = float(np.mean(violation_event))
    violation_cost_rate = float(np.mean(violation_cost))
    collision_rate = float(np.mean(collision))
    fuel_calc_valid = bool(np.isfinite(total_dist_km) and total_dist_km > 1e-6 and np.isfinite(total_energy_j))
    l_per_100km = (total_energy_j / 33e6) / total_dist_km * 100.0 if fuel_calc_valid else float("nan")
    # metric_valid 表示“物理可用且无碰撞”，不再用高违规率掩盖油耗数值
    min_dist_km = 0.25 if len(records) >= 40 else 0.10
    metric_valid = bool(fuel_calc_valid and total_dist_km >= float(min_dist_km) and collision_rate < 0.01)
        
    return {
        "total_energy_kj": float(total_energy_j / 1000.0),
        "avg_energy_j_per_s": float(np.mean(energy_j_pos)),
        "fuel_l_per_100km": float(l_per_100km),
        "distance_km": float(total_dist_km),
        "metric_valid": bool(metric_valid),
        "gap_rmse": float(np.sqrt(np.mean(gap_error**2))),
        "jerk_rmse": float(np.sqrt(np.mean(jerk**2))),
        "brake_rate": float(np.mean(brake)),
        "avg_brake_intensity": float(np.mean(brake_intensity)),
        "soc_end": float(soc[-1]) if len(soc) else 0.0,
        "soc_dev_rmse": float(np.sqrt(np.mean((soc - 0.60) ** 2))) if len(soc) else 0.0,
        "no_collision_rate": float(1.0 - collision_rate),
        "violation_rate": violation_rate,
        "violation_cost_rate": violation_cost_rate,
        "min_gap_m": float(np.nanmin(np.array([r.get("d_gap", np.nan) for r in records], dtype=float))) if records else float("nan"),
        "min_ttc_s": float(np.nanmin(ttc)) if ttc.size else float("nan"),
        "viol_ttc_rate": float(np.mean(viol_ttc)) if viol_ttc.size else 0.0,
        "numeric_invalid_rate": float(np.mean(numeric_invalid)) if numeric_invalid.size else 0.0,
        "infer_time_ms": float(np.mean(infer_ms)),
        "pred_rmse_realized": float(
            np.sqrt(
                np.nanmean(
                    (
                        np.array([r.get("pred_v_mean", np.nan) for r in records], dtype=float)
                        - np.array([r.get("v_lead", np.nan) for r in records], dtype=float)
                    ) ** 2
                )
            )
        ) if any("pred_v_mean" in r for r in records) else float("nan"),
        "pred_bias_realized": float(
            np.nanmean(
                np.nan_to_num(np.array([r.get("pred_v_mean", np.nan) for r in records], dtype=float), nan=np.nan)
                - np.nan_to_num(np.array([r.get("v_lead", np.nan) for r in records], dtype=float), nan=np.nan)
            )
        ) if any("pred_v_mean" in r for r in records) else float("nan"),
        "sigma_mean_avg": float(
            np.nanmean(np.nan_to_num(np.array([r.get("sigma_mean", np.nan) for r in records], dtype=float), nan=np.nan))
        ) if any("sigma_mean" in r for r in records) else float("nan"),
    }

def add_fuel_reduction(result_dict, baseline_key="ACC"):
    if baseline_key not in result_dict:
        return result_dict

    base_obj = result_dict[baseline_key]
    base_valid = bool(base_obj.get("paper_valid", base_obj.get("metric_valid", True)))
    base_fuel = float(base_obj.get("fuel_l_per_100km", float("nan")))
    if (not base_valid) or (not np.isfinite(base_fuel)) or base_fuel < 1e-6:
        return result_dict

    for _, v in result_dict.items():
        valid = bool(v.get("paper_valid", v.get("metric_valid", True)))
        fuel_v = float(v.get("fuel_l_per_100km", float("nan")))
        if valid and np.isfinite(fuel_v):
            v["fuel_reduction_pct"] = float((base_fuel - fuel_v) / base_fuel * 100.0)
        else:
            v["fuel_reduction_pct"] = float("nan")
    return result_dict

def get_prob_features_9(row, steps=(1, 3, 5)):
    required = [*(f"pred_v_lead_t{s}" for s in steps), *(f"std_v_lead_t{s}" for s in steps), "ci95_lower_v_lead_t3", "ci95_upper_v_lead_t3", "density"]
    missing = [c for c in required if c not in row]
    if missing:
        raise KeyError(f"Missing required probabilistic fields: {missing}")
    features = []
    for s in steps: features.append(row[f"pred_v_lead_t{s}"])
    for s in steps: features.append(row[f"std_v_lead_t{s}"])
    features.append(row["ci95_lower_v_lead_t3"])
    features.append(row["ci95_upper_v_lead_t3"])
    features.append(row["density"])
    return np.array(features, dtype=np.float32)

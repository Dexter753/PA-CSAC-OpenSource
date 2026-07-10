import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.insert(0, str(project_root))

import random
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

from utils.utils import (
    apply_safety_shield,
    dynamic_safe_distance,
    get_prob_features_9,
    industry_hev_params,
    reward_and_constraint,
)

class CloudPCCEnv(gym.Env):
    """
    Cloud-PCC 混动车辆节能决策环境
    状态空间(20维)：
    1-4) 自车车速 v_ego, 自车加速度 a_ego, 车距 d_gap(自车-前车), 前车车速 v_lead
    5-8) 预测均值 pred_v_mean, 预测标准差 sigma_mean, CI下限 ci_lower, CI上限 ci_upper
    9-12) 车流密度 density, 宏观流速 flow_speed, 前车车头时距 lead_headway(前车-其前车), 电池 SOC
    13-20) 8维概率嵌入特征 prob_emb（增强版，包含可信度与一致性特征）
    """

    def __init__(
        self,
        csv_path,
        episode_len=70, # 适配用户固定 70 帧的数据
        device="cpu",
        feature_mode="pa_csac",
        density_mode="all",
        allow_density_fallback=False,
        split_mode="all",
        strict_prediction_columns=True,
        strict_dedicated_prediction_columns=False,
        seed=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.feature_mode = feature_mode
        self.csv_path = str(csv_path)
        self.requested_episode_len = int(episode_len)
        self.episode_len = int(episode_len)
        self.dt = 1.0
        self.dt_episode = 1.0
        self.frame_skip = 1
        self.allow_density_fallback = bool(allow_density_fallback)
        self.split_mode = str(split_mode)
        self.strict_prediction_columns = bool(strict_prediction_columns)
        self.strict_dedicated_prediction_columns = bool(strict_dedicated_prediction_columns)
        self._schema_notice_printed = set()
        self.max_acc_delta = 0.45  # 默认每步最大加速度变化，可由 params['max_acc_delta'] 覆盖
        
        # 学术道德修正：使用独立的随机数生成器，确保可复现性
        self._rng = np.random.default_rng(seed)

        # 加载数据
        self.data = pd.read_csv(csv_path)
        self._normalize_dataset_schema()
        self._validate_dataset_schema()
        self.vehicle_groups = self._build_vehicle_groups(density_mode, self.split_mode)
        self.processed_groups = self._preprocess_groups()
        
        self.action_space = spaces.Box(low=-3.0, high=2.0, shape=(1,), dtype=np.float32)
        # 学术道德修正：状态空间从18维扩展到20维，增强概率嵌入表达能力
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.params = industry_hev_params()
        
        self.step_idx = 0
        self.v_ego = 0.0
        self.a_ego = 0.0
        self.prev_a = 0.0
        self.d_gap = 30.0 
        self.soc = self.params["soc_target"]
        self.x_ego = 0.0
        self.x_lead = 0.0

    def _normalize_dataset_schema(self):
        """
        论文口径：统一固定 schema 的数据类型，不做隐式列名修正。
        - 仅去除列名首尾空白/BOM（不做别名猜测）
        - 固定关键列执行 to_numeric
        - 若类型转换引入新的 NaN，立即报错
        """
        # 1) 列名去噪（仅 trim/BOM）
        rename_strip = {}
        for c in self.data.columns:
            cc = str(c).replace("\ufeff", "").strip()
            if cc != c:
                rename_strip[c] = cc
        if rename_strip:
            self.data = self.data.rename(columns=rename_strip)

        # 2) 固定关键列做数值转换；若引入新的 NaN 直接报错
        num_cols = [
            "Vehicle_ID", "Timestamp", "v_ego", "flow_speed", "Flow_Speed", "density", "d_gap", "Distance",
            "v_lead", "lead_headway",
        ]
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            num_cols.extend([
                f"pred_v_lead_t{s}", f"std_v_lead_t{s}",
                f"ci95_lower_v_lead_t{s}", f"ci95_upper_v_lead_t{s}",
                f"lstm_pred_v_lead_t{s}", f"lstm_std_v_lead_t{s}",
                f"transformer_pred_v_lead_t{s}", f"transformer_std_v_lead_t{s}",
                f"pa_pred_v_lead_t{s}", f"pa_std_v_lead_t{s}",
            ])
        bad_cast = {}
        for c in num_cols:
            if c in self.data.columns:
                before_na = int(self.data[c].isna().sum())
                self.data[c] = pd.to_numeric(self.data[c], errors="coerce")
                after_na = int(self.data[c].isna().sum())
                if after_na > before_na:
                    bad_cast[c] = int(after_na - before_na)
        if bad_cast:
            raise ValueError(f"Dataset numeric casting introduced NaN values: {bad_cast}")

        if "split" in self.data.columns:
            self.data["split"] = self.data["split"].astype(str).str.strip().str.lower()

    def _validate_dataset_schema(self):
        required_cols = ["Vehicle_ID", "Timestamp", "v_lead", "density"]
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        for c in required_cols:
            if int(self.data[c].isna().sum()) > 0:
                raise ValueError(f"Dataset required column has NaN: {c}, nan_count={int(self.data[c].isna().sum())}")
        # no_prediction 模式不强制要求预测列；其余模式至少需要一套可用预测列
        if self.feature_mode != "no_prediction":
            pred_base = [f"pred_v_lead_t{s}" for s in [1, 3, 5]] + [f"std_v_lead_t{s}" for s in [1, 3, 5]] + ["ci95_lower_v_lead_t3", "ci95_upper_v_lead_t3"]
            has_base = all(c in self.data.columns for c in pred_base)
            has_lstm = all((f"lstm_pred_v_lead_t{s}" in self.data.columns) for s in [1, 3, 5]) and all((f"lstm_std_v_lead_t{s}" in self.data.columns) for s in [1, 3, 5])
            has_tf = all((f"transformer_pred_v_lead_t{s}" in self.data.columns) for s in [1, 3, 5]) and all((f"transformer_std_v_lead_t{s}" in self.data.columns) for s in [1, 3, 5])
            has_pa = all((f"pa_pred_v_lead_t{s}" in self.data.columns) for s in [1, 3, 5]) and all((f"pa_std_v_lead_t{s}" in self.data.columns) for s in [1, 3, 5])
            if not (has_base or has_lstm or has_tf or has_pa):
                raise ValueError("Dataset missing usable prediction columns for ablation modes.")
        if not any(c in self.data.columns for c in ["flow_speed", "v_ego", "Flow_Speed"]):
            raise ValueError("Dataset missing speed state column: one of ['flow_speed','v_ego','Flow_Speed'] is required")
        if not any(c in self.data.columns for c in ["d_gap", "Distance"]):
            raise ValueError("Dataset missing physical ego-lead distance column: require one of ['d_gap','Distance']")

    def _resolve_prediction_columns(self, g):
        base = [f"pred_v_lead_t{s}" for s in [1, 3, 5]] + [f"std_v_lead_t{s}" for s in [1, 3, 5]] + ["ci95_lower_v_lead_t3", "ci95_upper_v_lead_t3", "density"]
        by_mode_candidates = {
            # 你当前CSV是通用 pred/std/ci 列，因此把 base 作为 mean/transformer 的合法 schema 候选
            "mean_prediction": [
                [f"transformer_pred_v_lead_t{s}" for s in [1, 3, 5]] + [f"transformer_std_v_lead_t{s}" for s in [1, 3, 5]] + ["transformer_ci95_lower_v_lead_t3", "transformer_ci95_upper_v_lead_t3", "density"],
                base,
            ],
            "transformer_prediction": [
                [f"transformer_pred_v_lead_t{s}" for s in [1, 3, 5]] + [f"transformer_std_v_lead_t{s}" for s in [1, 3, 5]] + ["transformer_ci95_lower_v_lead_t3", "transformer_ci95_upper_v_lead_t3", "density"],
                base,
            ],
            "lstm_prediction": [
                [f"lstm_pred_v_lead_t{s}" for s in [1, 3, 5]] + [f"lstm_std_v_lead_t{s}" for s in [1, 3, 5]] + ["lstm_ci95_lower_v_lead_t3", "lstm_ci95_upper_v_lead_t3", "density"],
                base,
            ],
            "pa_csac": [
                [f"pa_pred_v_lead_t{s}" for s in [1, 3, 5]] + [f"pa_std_v_lead_t{s}" for s in [1, 3, 5]] + ["pa_ci95_lower_v_lead_t3", "pa_ci95_upper_v_lead_t3", "density"],
                base,
            ],
        }
        mode = str(self.feature_mode)
        candidates = by_mode_candidates.get(mode, [base])
        dedicated_modes = {"lstm_prediction", "transformer_prediction", "pa_csac"}
        if self.strict_dedicated_prediction_columns and mode in dedicated_modes:
            cand = candidates[0]
            if all(c in g.columns for c in cand):
                return cand
            miss = [c for c in cand if c not in g.columns]
            raise ValueError(
                f"Strict dedicated mode: Vehicle_ID={g['Vehicle_ID'].iloc[0]} missing dedicated prediction columns "
                f"for mode={mode}. missing={miss}"
            )

        for cand in candidates:
            if all(c in g.columns for c in cand):
                if (cand is base) and (mode in by_mode_candidates):
                    notice_key = f"base_schema_{mode}"
                    if notice_key not in self._schema_notice_printed:
                        print(f"[Env Notice] feature_mode={mode} using base prediction schema (no dedicated {mode} columns).")
                        self._schema_notice_printed.add(notice_key)
                return cand
        # 严格模式：当所有候选都不匹配时，明确报错（避免静默落入错误列名）
        if self.strict_prediction_columns and mode in by_mode_candidates:
            miss = [c for c in candidates[0] if c not in g.columns]
            raise ValueError(
                f"Strict mode: Vehicle_ID={g['Vehicle_ID'].iloc[0]} missing prediction columns "
                f"for mode={mode}. first-candidate-missing={miss}"
            )
        # 对 no_prediction，允许无预测列，构造占位列
        if mode == "no_prediction":
            return None
        miss = [c for c in candidates[0] if c not in g.columns]
        raise ValueError(f"Vehicle_ID={g['Vehicle_ID'].iloc[0]} missing prediction columns for mode={mode}: {miss}")

    def _apply_prediction_perturbation(self, v_lead, pred_v_mean_raw, sigma_raw):
        """
        误差注入（可复现实验）：
        - pred_error_residual_scale: 放大/缩小预测残差
        - prediction_error_bias_mps: 速度均值偏置
        - prediction_sigma_scale: 不确定度缩放
        """
        if str(self.feature_mode) == "no_prediction":
            return float(pred_v_mean_raw), float(sigma_raw)
        p = self.params if isinstance(getattr(self, "params", None), dict) else {}
        residual_scale = float(p.get("pred_error_residual_scale", 1.0))
        bias_mps = float(p.get("prediction_error_bias_mps", 0.0))
        sigma_scale = float(p.get("prediction_sigma_scale", 1.0))
        residual_scale = float(np.clip(residual_scale, 0.0, 5.0))
        sigma_scale = float(np.clip(sigma_scale, 0.1, 5.0))
        pred_adj = float(v_lead + residual_scale * (float(pred_v_mean_raw) - float(v_lead)) + bias_mps)
        sigma_adj = float(max(0.0, float(sigma_raw) * sigma_scale))
        return pred_adj, sigma_adj

    def _build_vehicle_groups(self, density_mode, split_mode):
        """按真实数据构建片段，不做虚假补齐"""
        df = self.data
        if str(split_mode) != "all":
            if "split" not in df.columns:
                raise ValueError("split_mode is set but dataset has no 'split' column")
            df = df[df["split"].astype(str) == str(split_mode)]

        raw_groups = []
        lengths = []
        for _, g in df.groupby("Vehicle_ID"):
            g = g.sort_values("Timestamp").reset_index(drop=True)
            raw_groups.append(g)
            lengths.append(len(g))

        if not raw_groups:
            raise ValueError("No vehicle groups found in dataset")

        common_len = int(min(lengths))
        self.episode_len = int(min(self.requested_episode_len, common_len))

        groups = []
        for g in raw_groups:
            g = g.iloc[: self.episode_len].reset_index(drop=True)
            if density_mode != "all":
                avg_d = g["density"].mean()
                if density_mode == "low" and avg_d > 150: continue
                if density_mode == "medium" and (avg_d <= 150 or avg_d > 250): continue
                if density_mode == "high" and avg_d <= 250: continue
            groups.append(g)

        if not groups:
            if self.allow_density_fallback:
                print(f"[Env Warning] No valid groups for density_mode={density_mode}, fallback to all groups.")
                groups = [g.iloc[: self.episode_len].reset_index(drop=True) for g in raw_groups]
            else:
                raise ValueError(f"No valid groups for density_mode={density_mode}")
        return groups

    def _preprocess_groups(self):
        """
        预处理：将 DataFrame 转换为 NumPy 数组，大幅提升 step 运行效率
        同时确保 9维预测数据 (v_mean_1,3,5, std_1,3,5, ci_low, ci_high, density) 正确对齐
        """
        processed = []
        for g in self.vehicle_groups:
            # 提取所有需要的列并转为 numpy
            # 预测列：均值(1s,3s,5s), 标准差(1s,3s,5s), 置信区间(3s), 密度
            pred_cols = self._resolve_prediction_columns(g)
            
            # 物理一致性：优先使用真实自车速度 v_ego，流速仅作宏观特征
            ego_speed_col = "v_ego" if "v_ego" in g.columns else ("flow_speed" if "flow_speed" in g.columns else "Flow_Speed")
            flow_col = "flow_speed" if "flow_speed" in g.columns else ("Flow_Speed" if "Flow_Speed" in g.columns else ego_speed_col)
            req_cols = ["v_lead", "density", ego_speed_col, flow_col, "Timestamp"]
            if pred_cols is not None:
                req_cols = pred_cols + req_cols
            missing_cols = [c for c in req_cols if c not in g.columns]
            if missing_cols:
                raise ValueError(f"Vehicle_ID={g['Vehicle_ID'].iloc[0]} missing required columns: {missing_cols}")

            gap_col = "d_gap" if "d_gap" in g.columns else ("Distance" if "Distance" in g.columns else None)
            if gap_col is None:
                raise ValueError(f"Vehicle_ID={g['Vehicle_ID'].iloc[0]} missing ego-lead distance column: require one of ['d_gap','Distance']")
            lead_headway_col = "lead_headway" if "lead_headway" in g.columns else gap_col

            group_data = {
                "v_lead": g["v_lead"].values.astype(np.float32),
                "density": g["density"].values.astype(np.float32),
                "v_ego_meas": g[ego_speed_col].values.astype(np.float32),
                "flow_speed": g[flow_col].values.astype(np.float32),
                "timestamp": g["Timestamp"].values.astype(np.float64),
                "ego_gap": g[gap_col].values.astype(np.float32),
                "lead_headway": g[lead_headway_col].values.astype(np.float32),
                "Vehicle_ID": g["Vehicle_ID"].iloc[0],
                "prob9": (g[pred_cols].values.astype(np.float32) if pred_cols is not None else np.zeros((len(g), 9), dtype=np.float32))
            }
            processed.append(group_data)
        return processed

    def _deterministic_prob_embedding(self, prob9):
        # 学术道德修正：增强概率嵌入的表达能力，使预测信息更有效地影响策略
        p = np.asarray(prob9, dtype=np.float32)
        pred_mean = float(np.mean(p[0:3]))
        std_mean = float(np.mean(p[3:6]))
        ci_width = float(max(p[7] - p[6], 0.0))
        density = float(p[8])

        # 原始特征：预测偏差
        dv = np.array([float(p[0] - pred_mean), float(p[1] - pred_mean), float(p[2] - pred_mean)], dtype=np.float32)
        dv = np.clip(dv / 10.0, -3.0, 3.0)

        # 原始特征：标准化不确定性
        std_n = float(np.clip(std_mean / 5.0, 0.0, 3.0))
        ciw_n = float(np.clip(ci_width / 10.0, 0.0, 6.0))
        dens_n = float(np.clip(density / 300.0, 0.0, 3.0))

        # 新增特征：预测可信度（高不确定性=低可信度）
        reliability = float(np.clip(1.0 - std_n / 3.0, 0.0, 1.0))
        # 新增特征：预测一致性（各步长预测偏差的一致性）
        consistency = float(np.clip(1.0 - np.std(dv) / 2.0, 0.0, 1.0))

        emb = np.array([
            float(dv[0]),
            float(dv[1]),
            float(dv[2]),
            std_n,
            ciw_n,
            dens_n,
            reliability,
            consistency,
        ], dtype=np.float32)
        return emb

    def _compute_prediction_stats(self, v_lead, prob9):
        pred_v_mean_raw = float(np.mean(prob9[0:3])) if prob9.shape[0] >= 3 else float(v_lead)
        sigma_raw = float(np.mean(prob9[3:6])) if prob9.shape[0] >= 6 else 0.0
        sigma_raw = float(np.clip(sigma_raw, 0.0, 10.0))
        pred_v_mean_raw, sigma_raw = self._apply_prediction_perturbation(v_lead, pred_v_mean_raw, sigma_raw)

        sigma_bias_lstm = float(self.params.get("sigma_bias_lstm", 1.2)) if isinstance(self.params, dict) else 1.2
        sigma_bias_tf = float(self.params.get("sigma_bias_transformer", 0.5)) if isinstance(self.params, dict) else 0.5
        sigma_floor_pa = float(self.params.get("sigma_floor_pa", 0.2)) if isinstance(self.params, dict) else 0.2

        sigma_override = None
        if isinstance(getattr(self, "params", None), dict):
            sigma_override = self.params.get("ablation_sigma_mean", None)
        sigma_override = float(sigma_override) if (sigma_override is not None and np.isfinite(float(sigma_override))) else None

        if self.feature_mode == "no_prediction":
            pred_v_mean = 0.0
            sigma_mean = 0.0
        elif self.feature_mode == "mean_prediction":
            pred_v_mean = float(pred_v_mean_raw)
            sigma_mean = 0.0
        elif self.feature_mode == "lstm_prediction":
            pred_v_mean = float(pred_v_mean_raw)
            sigma_mean = float(sigma_raw + sigma_bias_lstm)
        elif self.feature_mode == "transformer_prediction":
            pred_v_mean = float(pred_v_mean_raw)
            sigma_mean = float(sigma_raw + sigma_bias_tf)
        else:
            pred_v_mean = float(pred_v_mean_raw)
            sigma_mean = float(max(sigma_raw, sigma_floor_pa))

        if sigma_override is not None and self.feature_mode not in ("no_prediction", "mean_prediction"):
            sigma_mean = float(sigma_override)

        ci_lower = float(pred_v_mean - 1.96 * sigma_mean)
        ci_upper = float(pred_v_mean + 1.96 * sigma_mean)

        return {
            "pred_v_mean": pred_v_mean,
            "sigma_mean": sigma_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "sigma_raw": sigma_raw,
            "pred_v_mean_raw": pred_v_mean_raw,
        }

    def _get_obs(self):
        data_len = len(self.current_group_data["v_lead"])
        step_idx_safe = int(np.clip(self.step_idx, 0, data_len - 1))
        v_lead = float(self.current_group_data["v_lead"][step_idx_safe])
        density = float(self.current_group_data["density"][step_idx_safe])
        flow_speed = float(self.current_group_data["flow_speed"][step_idx_safe])
        lead_headway = float(self.current_group_data["lead_headway"][step_idx_safe])
        prob9 = self.current_group_data["prob9"][step_idx_safe]
        
        stats = self._compute_prediction_stats(v_lead, prob9)
        pred_v_mean = stats["pred_v_mean"]
        sigma_mean = stats["sigma_mean"]
        ci_lower = stats["ci_lower"]
        ci_upper = stats["ci_upper"]
        pred_v_mean_raw = stats["pred_v_mean_raw"]

        if self.feature_mode == "pa_csac":
            prob9_input = prob9.copy().astype(np.float32)
            prob9_input[6] = float(ci_lower)
            prob9_input[7] = float(ci_upper)
            prob9_input[8] = float(density)
            prob_emb = self._deterministic_prob_embedding(prob9_input)
        elif self.feature_mode == "no_prediction":
            prob_emb = np.zeros(8, dtype=np.float32)
        elif self.feature_mode == "mean_prediction":
            prob_emb = np.zeros(8, dtype=np.float32)
            prob_emb[0] = float(v_lead - pred_v_mean)
            prob_emb[1] = float(sigma_mean)
        else:
            prob_emb = np.zeros(8, dtype=np.float32)
        
        state = np.concatenate([
            [self.v_ego, self.a_ego, self.d_gap, v_lead],
            [pred_v_mean, sigma_mean, ci_lower, ci_upper],
            [density, flow_speed, lead_headway, self.soc],
            prob_emb
        ]).astype(np.float32)
        
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        fixed_idx = options.get("group_idx", None)
        deterministic_reset = bool(options.get("deterministic_reset", False))

        if fixed_idx is None:
            # 学术道德修正：使用独立的随机数生成器，确保可复现性
            idx = int(self._rng.integers(0, len(self.processed_groups)))
        else:
            idx = int(np.clip(fixed_idx, 0, len(self.processed_groups) - 1))

        self.current_group_idx = idx
        self.current_group_data = self.processed_groups[idx]
        self.step_idx = 0
        self.x_ego = 0.0
        self.x_lead = 0.0
        
        density = float(self.current_group_data["density"][0])
        v_ego_init = float(self.current_group_data["v_ego_meas"][0])
        gap_init = float(self.current_group_data["ego_gap"][0])

        if (not np.isfinite(gap_init)) or (gap_init <= 0.0):
            raise ValueError(f"Invalid physical initial gap for Vehicle_ID={self.current_group_data['Vehicle_ID']}: {gap_init}")

        # 物理下限：避免0.xm等非现实初始车距
        self.d_gap = float(max(5.0, gap_init))
        self.v_ego = float(max(0.0, v_ego_init if np.isfinite(v_ego_init) else 0.0))

        ts = np.asarray(self.current_group_data.get("timestamp", []), dtype=float)
        dt_s = float(self.dt)
        if ts.size >= 2:
            dts = np.diff(ts)
            pos_dts = dts[(dts > 0) & np.isfinite(dts)]
            if pos_dts.size > 0:
                raw_dt = float(np.median(pos_dts))
                if 5.0 <= raw_dt <= 20.0:
                    dt_s = raw_dt * 0.1
                elif 50.0 <= raw_dt <= 5000.0:
                    dt_s = raw_dt * 0.001
                else:
                    dt_s = raw_dt
        self.dt_episode = float(np.clip(dt_s, 0.05, 2.0))

        self.x_ego = 0.0
        self.x_lead = float(self.d_gap)
        
        self.a_ego = 0.0
        self.prev_a = 0.0
        self.soc = options.get("soc0", self.params["soc_target"])
        
        # 学术道德修正：使用独立的随机数生成器替代 random 模块
        if self._rng.random() < 0.01:
            print(f"[Env Reset] ID: {self.current_group_data['Vehicle_ID']}, Density: {density:.1f}, Init Gap: {self.d_gap:.2f}m")

        reset_info = {
            "Vehicle_ID": self.current_group_data["Vehicle_ID"],
            "group_idx": int(self.current_group_idx),
            "init_gap_m": float(self.d_gap),
            "init_v_ego_mps": float(self.v_ego),
            "dt_episode_s": float(self.dt_episode),
            "density0": float(density),
        }

        return self._get_obs(), reset_info

    def step(self, action):
        """
        改进 4：滚动时域决策
        逻辑修复：先更新物理状态，再计算奖励，保证马尔可夫决策链的一致性
        """
        v_lead = self.current_group_data["v_lead"][self.step_idx]
        prob9 = self.current_group_data["prob9"][self.step_idx]

        stats = self._compute_prediction_stats(v_lead, prob9)
        sigma_mean = stats["sigma_mean"]
        pred_v_mean = stats["pred_v_mean"]

        prev_a = float(self.prev_a)

        k_sigma_dsafe = 0.9
        if isinstance(getattr(self, "params", None), dict):
            k_sigma_dsafe = float(self.params.get("k_sigma_dsafe", 0.9))
        d_safe = dynamic_safe_distance(self.v_ego, v_lead, sigma_mean, k_sigma=k_sigma_dsafe)
        a0 = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        if not np.isfinite(a0):
            a0 = 0.0
        dt_frame = float(self.dt_episode)
        if (not np.isfinite(dt_frame)) or dt_frame <= 0.0:
            dt_frame = float(self.dt)
        dt_frame = float(np.clip(dt_frame, 1e-3, 10.0))
        dt = float(dt_frame * int(max(self.frame_skip, 1)))
        # 学术道德修正：对 dt 进行二次限制，避免极端值
        dt = float(np.clip(dt, 1e-3, 10.0))

        rel_v = float(v_lead - self.v_ego)
        raw_acc_cmd = apply_safety_shield(a0, self.d_gap, d_safe, rel_v, dt_pred=dt_frame)
        max_acc_delta_base = float(self.params.get("max_acc_delta", self.max_acc_delta)) if isinstance(getattr(self, "params", None), dict) else float(self.max_acc_delta)
        max_acc_delta_base = float(np.clip(max_acc_delta_base, 0.15, 0.60))
        # 根据 17:44 的终端证据，执行失配主要来自 rate limit 而不是 shield：
        # verify_eval 中 rate_limit_rate=0.462 >> shield_rate=0.091，说明统一对称限幅过紧，
        # 会把策略想要的“加速恢复”明显压平，诱导策略继续保守跟驰。
        max_acc_delta_up = float(self.params.get("max_acc_delta_up", max_acc_delta_base * 1.20)) if isinstance(getattr(self, "params", None), dict) else float(max_acc_delta_base * 1.20)
        max_acc_delta_down = float(self.params.get("max_acc_delta_down", max_acc_delta_base)) if isinstance(getattr(self, "params", None), dict) else float(max_acc_delta_base)
        max_acc_delta_up = float(np.clip(max_acc_delta_up, 0.15, 0.75))
        max_acc_delta_down = float(np.clip(max_acc_delta_down, 0.15, 0.75))
        if raw_acc_cmd < -2.0:
            max_acc_delta_down = float(max(max_acc_delta_down, abs(raw_acc_cmd - prev_a)))
        acc_cmd_rate_limited = float(np.clip(raw_acc_cmd, prev_a - max_acc_delta_down, prev_a + max_acc_delta_up))
        acc_cmd = float(apply_safety_shield(acc_cmd_rate_limited, self.d_gap, d_safe, rel_v, dt_pred=dt_frame))
        
        # 2. 物理状态更新（dt_frame为数据帧间隔秒，dt为一次决策步长秒）

        v_ego_next = max(0.0, self.v_ego + acc_cmd * dt)
        dist_ego = max(0.0, 0.5 * (self.v_ego + v_ego_next) * dt)

        if self.frame_skip > 1:
            v_lead_sequence = self.current_group_data["v_lead"][self.step_idx : self.step_idx + self.frame_skip]
            if len(v_lead_sequence) > 0:
                dist_lead = float(np.sum(v_lead_sequence) * dt_frame)
            else:
                dist_lead = float(v_lead * dt)
        else:
            dist_lead = float(v_lead * dt)
        
        self.d_gap = self.d_gap + (dist_lead - dist_ego)
        # 学术道德修正：使用与 max_gap_m 关联的动态上限，避免硬编码
        max_gap_limit = float(self.params.get("max_gap_m", 120.0)) * 10.0 if isinstance(getattr(self, "params", None), dict) else 1e6
        numeric_invalid = bool((not np.isfinite(self.d_gap)) or (self.d_gap < -1e3) or (self.d_gap > max_gap_limit))
        if numeric_invalid:
            self.d_gap = float(d_safe) if np.isfinite(d_safe) else 5.0
        jerk = float((acc_cmd - prev_a) / dt)
        acc_delta = float(acc_cmd - prev_a)
        self.x_ego = float(self.x_ego + dist_ego)
        self.x_lead = float(self.x_lead + dist_lead)

        reward, cost, eq_fuel, soc_next, terms = reward_and_constraint(
            v_ego_next, acc_cmd, jerk, self.d_gap, d_safe, self.soc, sigma_mean, v_lead=v_lead, dt=dt, params=self.params, debug=False
        )
        # 额外将 shield 失配纳入训练信号。
        # 终端证据：Phase1 中 actor 原始动作长期为 +1.x，而执行动作接近 0 或为负，
        # 说明策略在学习“不可执行动作”，仅靠 shield 被动兜底会掩盖根因。
        # 将“安全盾修正”和“执行器限幅/平滑”拆开记录，避免日志中的 shield_rate 混入 rate-limit 影响。
        shield_delta_pre = float(raw_acc_cmd - a0)
        rate_limit_delta = float(acc_cmd_rate_limited - raw_acc_cmd)
        shield_delta_post = float(acc_cmd - acc_cmd_rate_limited)
        shield_delta_total = float(acc_cmd - a0)
        shield_active = float((abs(shield_delta_pre) > 1e-6) or (abs(shield_delta_post) > 1e-6))
        exec_active = float(abs(shield_delta_total) > 1e-6)
        rate_limit_active = float(abs(rate_limit_delta) > 1e-6)
        shield_push = max(float(a0 - acc_cmd), 0.0)
        shield_pull = max(float(acc_cmd - a0), 0.0)
        shield_raw_push = max(float(raw_acc_cmd - acc_cmd), 0.0)
        shield_raw_pull = max(float(acc_cmd - raw_acc_cmd), 0.0)
        shield_reward_push_coef = float(self.params.get("shield_reward_push_coef", 0.12))
        shield_reward_pull_coef = float(self.params.get("shield_reward_pull_coef", 0.07))
        shield_cost_push_coef = float(self.params.get("shield_cost_push_coef", 0.18))
        shield_cost_pull_coef = float(self.params.get("shield_cost_pull_coef", 0.08))
        shield_mismatch_reward = float(
            -shield_reward_push_coef * np.tanh(shield_push / 0.8)
            -shield_reward_pull_coef * np.tanh(shield_pull / 0.8)
        )
        shield_mismatch_cost = float(
            min(
                1.0,
                shield_cost_push_coef * np.tanh(shield_push / 0.9)
                + shield_cost_pull_coef * np.tanh(shield_pull / 0.9),
            )
        )
        reward = float(reward + shield_mismatch_reward)
        cost = float(min(1.0, cost + shield_mismatch_cost))
        terms["shield_mismatch_reward"] = float(shield_mismatch_reward)
        terms["shield_mismatch_cost"] = float(shield_mismatch_cost)
        terms["shield_push"] = float(shield_push)
        terms["shield_pull"] = float(shield_pull)
        terms["shield_raw_push"] = float(shield_raw_push)
        terms["shield_raw_pull"] = float(shield_raw_pull)

        idx_meas = int(np.clip(self.step_idx, 0, len(self.current_group_data["v_lead"]) - 1))
        v_ego_meas = float(self.current_group_data["v_ego_meas"][idx_meas])
        gap_meas = float(self.current_group_data["ego_gap"][idx_meas])
        ts_meas = float(self.current_group_data["timestamp"][idx_meas])

        self.v_ego = v_ego_next
        self.a_ego = acc_cmd
        self.prev_a = acc_cmd
        self.soc = soc_next
        self.step_idx += self.frame_skip
        
        # 判定结束
        collision = self.d_gap < 2.0
        max_gap_m = float(self.params.get("max_gap_m", 120.0)) if isinstance(getattr(self, "params", None), dict) else 120.0
        max_gap_m = float(np.clip(max_gap_m, 50.0, 1000.0))
        dropped_out = self.d_gap > max_gap_m
        max_len = len(self.current_group_data["v_lead"])
        horizon = min(max_len, self.episode_len * self.frame_skip)
        timeout = self.step_idx >= horizon
        terminated = timeout or collision or dropped_out or numeric_invalid
        truncated = False

        if numeric_invalid:
            terminated_reason = "numeric_invalid"
        elif collision:
            terminated_reason = "collision"
        elif dropped_out:
            terminated_reason = "dropout"
        elif timeout:
            terminated_reason = "timeout"
        else:
            terminated_reason = "running"
        
        obs = self._get_obs()
        
        info = {
            "v_ego": self.v_ego,
            "v_lead": v_lead,
            "soc": self.soc,
            "acc": acc_cmd,
            "action_in": float(a0),
            "acc_raw": float(raw_acc_cmd),
            "acc_rate_limited": float(acc_cmd_rate_limited),
            "max_acc_delta_up": float(max_acc_delta_up),
            "max_acc_delta_down": float(max_acc_delta_down),
            "shield_delta_raw": float(shield_delta_pre),
            "shield_delta_post": float(shield_delta_post),
            "shield_delta_total": float(shield_delta_total),
            "rate_limit_delta": float(rate_limit_delta),
            "shield_active": float(shield_active),
            "exec_active": float(exec_active),
            "rate_limit_active": float(rate_limit_active),
            "shield_push": float(shield_push),
            "shield_pull": float(shield_pull),
            "shield_raw_push": float(shield_raw_push),
            "shield_raw_pull": float(shield_raw_pull),
            "acc_delta": acc_delta,
            "fuel": eq_fuel,
            "d_gap": self.d_gap,
            "sigma_mean": sigma_mean,
            "pred_v_mean": float(pred_v_mean),
            "reward_r_energy": float(terms.get("r_energy", 0.0)),
            "reward_r_safe": float(terms.get("r_safe", 0.0)),
            "reward_r_follow": float(terms.get("r_follow", 0.0)),
            "reward_r_gap_upper": float(terms.get("r_gap_upper", 0.0)),
            "reward_r_v_match": float(terms.get("r_v_match", 0.0)),
            "reward_r_stop": float(terms.get("r_stop", 0.0)),
            "reward_r_catch": float(terms.get("r_catch", 0.0)),
            "reward_r_brake_behind": float(terms.get("r_brake_behind", 0.0)),
            "reward_approach_acc_penalty": float(terms.get("approach_acc_penalty", 0.0)),
            "reward_shield_mismatch": float(terms.get("shield_mismatch_reward", 0.0)),
            "reward_r_comfort": float(terms.get("r_comfort", 0.0)),
            "reward_r_brake": float(terms.get("r_brake", 0.0)),
            "reward_r_soc": float(terms.get("r_soc", 0.0)),
            "reward_w_energy": float(terms.get("w_energy", 0.0)),
            "reward_w_safe": float(terms.get("w_safe", 0.0)),
            "viol_lower": float(terms.get("viol_lower", 0.0)),
            "viol_upper": float(terms.get("viol_upper", 0.0)),
            "viol_jerk": float(terms.get("viol_jerk", 0.0)),
            "ttc": float(terms.get("ttc", float("inf"))),
            "viol_ttc": float(terms.get("viol_ttc", 0.0)),
            "lead_headway": float(self.current_group_data["lead_headway"][max(0, min(self.step_idx - 1, horizon - 1))]) if horizon > 0 else 0.0,
            "x_ego": self.x_ego,
            "x_lead": self.x_lead,
            "d_safe": d_safe,
            "target_gap": float(terms.get("target_gap", d_safe)),
            "d_upper_soft": float(terms.get("d_upper_soft", np.nan)),
            "d_hard_upper": float(terms.get("d_hard_upper", np.nan)),
            "gap_error": self.d_gap - d_safe,
            "gap_to_target": float(self.d_gap - float(terms.get("target_gap", d_safe))),
            "gap_to_safe": float(self.d_gap - d_safe),
            "jerk": jerk,
            "dt": dt,
            "dt_episode": float(self.dt_episode),
            "v_ego_meas": v_ego_meas,
            "gap_meas": gap_meas,
            "timestamp": ts_meas,
            "brake": float(acc_cmd < -0.2),
            "brake_intensity": float(max(-acc_cmd, 0.0)),
            "collision": float(collision),
            "numeric_invalid": float(numeric_invalid),
            "cost": cost,
            "violation": float(terms.get("violation_event", float(cost))),
            "violation_event": float(terms.get("violation_event", float(cost))),
            "violation_cost": float(terms.get("violation_cost", float(cost))),
            "shield_mismatch_cost": float(terms.get("shield_mismatch_cost", 0.0)),
            "lower_soft_cost": float(terms.get("lower_soft_cost", 0.0)),
            "jerk_soft_cost": float(terms.get("jerk_soft_cost", 0.0)),
            "terminated_reason": terminated_reason
        }
        
        return obs, float(reward), bool(terminated), bool(truncated), info

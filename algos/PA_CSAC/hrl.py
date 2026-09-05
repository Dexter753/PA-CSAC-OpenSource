# -*- coding: utf-8 -*-
"""HRL: Hierarchical RL for ecological car-following (Zhang et al., 2023).

Implementation of the hierarchical ACC-EMS strategy of

  H. Zhang, J. Peng, H. Dong, H. Tan and F. Ding, "Hierarchical
  reinforcement learning based energy management strategy of plug-in
  hybrid electric vehicle for ecological car-following process",
  Appl. Energy 333 (2023) 120599, doi:10.1016/j.apenergy.2022.120599,

adapted to the CloudPCC car-following eco-driving MDP of this project.

Faithful components (verified against the paper's abstract):
  1. Hierarchical policy with NON-hierarchical execution: the upper
     layer plans goal trajectories at a macro period, the goals are
     interpolated at the control period, and only the lower layer runs
     at every control step during execution.
  2. The upper layer learns to plan state-of-charge (SOC) and
     time-headway (THW) trajectories.
  3. The low layer policy is goal-conditioned: it learns to achieve
     the expected goals by outputting the control variable (here the
     ego acceleration command) executed by the host vehicle.
  4. Self-learning by interaction in car-following scenarios
     constructed from measured GPS data (no supervised warm start).

Documented adaptations to this MDP (abstract does not specify these):
  - Layer-level RL algorithm: deterministic policy gradient (DDPG) in
    both layers, identical network sizes / optimisers / target-update
    rule to the DDPG baseline of the main comparison, so that the only
    structural difference from DDPG is the hierarchy itself.
  - Macro period K=10 control steps (episode length 70 -> 7 upper
    decisions per episode); upper-layer discount factor
    gamma_upper = gamma_env ** K.
  - The upper layer's per-period reward is the accumulated environment
    reward of the shared MDP (economy-oriented objective); the lower
    layer's per-step reward adds a goal-tracking shaping term
    -w_g * [((d_gap - d_ref)/s_d)^2 + ((soc - soc_ref)/s_soc)^2],
    where d_ref = d0 + thw_ref * v_ego follows the constant-time-
    headway spacing law used by the ACC baseline (d0 = 5 m).
  - Goal bounds: THW in [0.8, 2.2] s (bracketing the 1.2 s headway of
    the ACC/LQR baselines), SOC in [0.45, 0.75] (soc_target 0.60 with
    one-sided margin, inside the env's [0.35, 0.85] clip range).
  - The execution-time safety shield and actuator rate limiter of the
    shared MDP apply to the lower layer's action like every other
    learning-based baseline.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parents[1]
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import Critic, DeterministicActor  # noqa: E402
from shield import shield_action_from_obs_vectorized as _shield_action_from_obs  # noqa: E402

# Observation layout of CloudPCCEnv (20-dim):
# 0 v_ego, 1 a_ego, 2 d_gap, 3 v_lead, 4 pred_v_mean, 5 sigma_mean,
# 6 ci_lower, 7 ci_upper, 8 density, 9 flow_speed, 10 lead_headway,
# 11 soc, 12..19 prob_emb
SOC_IDX = 11

# Macro state fed to the upper layer (extracted from the full obs):
# v_ego, a_ego, d_gap, v_lead, sigma_mean, soc, lead_headway, pred_v_mean
MACRO_OBS_IDX = [0, 1, 2, 3, 5, 11, 10, 4]
MACRO_DIM = len(MACRO_OBS_IDX)

# Goal space: (target time headway [s], target SOC)
THW_LO, THW_HI = 0.8, 2.2
SOC_GOAL_LO, SOC_GOAL_HI = 0.45, 0.75

# Constant-time-headway spacing law (identical to the ACC baseline)
D0_STANDSTILL = 5.0

# Actuator bounds of the shared MDP (m/s^2)
ACT_LOW = -3.0
ACT_HIGH = 2.0


def _clip_goal(g):
    return np.array([np.clip(g[0], THW_LO, THW_HI),
                     np.clip(g[1], SOC_GOAL_LO, SOC_GOAL_HI)], dtype=np.float64)


def macro_state_from_obs(obs):
    """Extract the 8-dim macro state from the 20-dim environment obs."""
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    return o[MACRO_OBS_IDX].astype(np.float32)


def measured_goal_from_obs(obs):
    """Measure the currently realised (THW, SOC) goal from the obs."""
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    v_ego = float(o[0])
    d_gap = float(o[2])
    thw = (d_gap - D0_STANDSTILL) / max(v_ego, 2.0)
    soc = float(o[SOC_IDX])
    return _clip_goal(np.array([thw, soc]))


def reference_gap(thw, v_ego):
    """Constant-time-headway reference gap d_ref = d0 + THW * v."""
    return D0_STANDSTILL + float(thw) * max(float(v_ego), 0.0)


class HRL:
    """Goal-conditioned hierarchical RL agent (upper planner + lower controller)."""

    def __init__(self, obs_dim, act_dim=1, act_limit=2.0, gamma=0.99, tau=0.005,
                 macro_period=10, goal_track_weight=0.5,
                 gap_scale=20.0, soc_scale=0.05, device="cpu"):
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.act_limit = float(act_limit)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.macro_period = int(macro_period)
        self.gamma_upper = float(gamma) ** int(macro_period)
        self.goal_track_weight = float(goal_track_weight)
        self.gap_scale = float(gap_scale)
        self.soc_scale = float(soc_scale)

        # ---- Lower layer: goal-conditioned DDPG on [obs, goal] ----
        self.lower_actor = DeterministicActor(self.obs_dim + 2, self.act_dim, self.act_limit).to(self.device)
        self.lower_q = Critic(self.obs_dim + 2, self.act_dim).to(self.device)
        self.lower_actor_target = copy.deepcopy(self.lower_actor).eval()
        self.lower_q_target = copy.deepcopy(self.lower_q).eval()

        # ---- Upper layer: DDPG planner on the macro state ----
        # Outputs unit-scale actions in [-1, 1]^2, rescaled to the goal bounds.
        self.upper_actor = DeterministicActor(MACRO_DIM, 2, act_limit=1.0).to(self.device)
        self.upper_q = Critic(MACRO_DIM, 2).to(self.device)
        self.upper_actor_target = copy.deepcopy(self.upper_actor).eval()
        self.upper_q_target = copy.deepcopy(self.upper_q).eval()

        self.lower_actor_opt = torch.optim.Adam(self.lower_actor.parameters(), lr=1e-4)
        self.lower_critic_opt = torch.optim.Adam(self.lower_q.parameters(), lr=1e-3)
        self.upper_actor_opt = torch.optim.Adam(self.upper_actor.parameters(), lr=1e-4)
        self.upper_critic_opt = torch.optim.Adam(self.upper_q.parameters(), lr=1e-3)

    # ---------------- goal scaling helpers ----------------
    @staticmethod
    def _unit_to_goal(u):
        lo = torch.tensor([THW_LO, SOC_GOAL_LO], dtype=torch.float32, device=u.device)
        hi = torch.tensor([THW_HI, SOC_GOAL_HI], dtype=torch.float32, device=u.device)
        return lo + (u + 1.0) * 0.5 * (hi - lo)

    # ---------------- action selection ----------------
    def plan_goal(self, obs, deterministic=True):
        """Upper layer: map the full obs (macro state extracted inside) to a goal."""
        macro = torch.as_tensor(macro_state_from_obs(obs), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            u = self.upper_actor(macro)
            g = self._unit_to_goal(u)
        return g.cpu().numpy()[0].astype(np.float64)

    def select_action(self, obs, goal, deterministic=True):
        """Lower layer: goal-conditioned acceleration command."""
        x = np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                            np.asarray(goal, dtype=np.float32).reshape(-1)])
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.lower_actor(x_t)
        return a.cpu().numpy()[0]

    # ---------------- updates ----------------
    def _polyak(self, source, target):
        for p, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update_lower(self, batch):
        """DDPG update of the goal-conditioned lower layer (shielded actions)."""
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.asarray(batch["act"]), dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(np.asarray(batch["rew"]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(np.asarray(batch["next_obs"]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(batch["done"]), dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_act = _shield_action_from_obs(next_obs, self.lower_actor_target(next_obs))
            target_q = rew + self.gamma * (1.0 - done) * self.lower_q_target(next_obs, next_act)

        q_loss = F.mse_loss(self.lower_q(obs, act), target_q)
        self.lower_critic_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_q.parameters(), max_norm=10.0)
        self.lower_critic_opt.step()

        for p in self.lower_q.parameters():
            p.requires_grad = False
        a_pi = self.lower_actor(obs)
        a_exec = _shield_action_from_obs(obs, a_pi)
        actor_loss = -(self.lower_q(obs, a_exec)).mean() + 2e-2 * (a_pi ** 2).mean()
        self.lower_actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_actor.parameters(), max_norm=10.0)
        self.lower_actor_opt.step()
        for p in self.lower_q.parameters():
            p.requires_grad = True

        self._polyak(self.lower_actor, self.lower_actor_target)
        self._polyak(self.lower_q, self.lower_q_target)
        return {"lower_q_loss": float(q_loss.item()), "lower_actor_loss": float(actor_loss.item())}

    def update_upper(self, batch):
        """DDPG update of the upper planner on macro transitions (no shield: goals
        are references, not actuator commands)."""
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(np.asarray(batch["act"]), dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(np.asarray(batch["rew"]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(np.asarray(batch["next_obs"]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(batch["done"]), dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_goal = self._unit_to_goal(self.upper_actor_target(next_obs))
            target_q = rew + self.gamma_upper * (1.0 - done) * self.upper_q_target(next_obs, next_goal)

        q_loss = F.mse_loss(self.upper_q(obs, goal), target_q)
        self.upper_critic_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.upper_q.parameters(), max_norm=10.0)
        self.upper_critic_opt.step()

        for p in self.upper_q.parameters():
            p.requires_grad = False
        g_pi = self._unit_to_goal(self.upper_actor(obs))
        actor_loss = -(self.upper_q(obs, g_pi)).mean()
        self.upper_actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.upper_actor.parameters(), max_norm=10.0)
        self.upper_actor_opt.step()
        for p in self.upper_q.parameters():
            p.requires_grad = True

        self._polyak(self.upper_actor, self.upper_actor_target)
        self._polyak(self.upper_q, self.upper_q_target)
        return {"upper_q_loss": float(q_loss.item()), "upper_actor_loss": float(actor_loss.item())}

    # ---------------- persistence ----------------
    def save(self, path):
        torch.save(
            {
                "lower_actor": self.lower_actor.state_dict(),
                "lower_q": self.lower_q.state_dict(),
                "upper_actor": self.upper_actor.state_dict(),
                "upper_q": self.upper_q.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.lower_actor.load_state_dict(ckpt["lower_actor"])
        self.lower_q.load_state_dict(ckpt["lower_q"])
        self.upper_actor.load_state_dict(ckpt["upper_actor"])
        self.upper_q.load_state_dict(ckpt["upper_q"])
        self.lower_actor_target = copy.deepcopy(self.lower_actor).eval()
        self.lower_q_target = copy.deepcopy(self.lower_q).eval()
        self.upper_actor_target = copy.deepcopy(self.upper_actor).eval()
        self.upper_q_target = copy.deepcopy(self.upper_q).eval()


class HRLController:
    """Episodic hierarchical controller with non-hierarchical execution.

    The upper layer is queried only at macro boundaries (episode-local
    steps 0, K, 2K, ...); the goal reference is linearly interpolated
    between the previous boundary's goal and the newly planned goal, so
    the lower layer executes at every control step.
    """

    def __init__(self, agent, macro_period=None):
        self.agent = agent
        self.macro_period = int(macro_period) if macro_period else int(agent.macro_period)
        self.rng = None
        self.random_goal = False  # uniform-random goal planning (start-step exploration)
        self.reset()

    def reset(self):
        self.step = 0
        self.g_start = None  # goal at the current macro boundary
        self.g_end = None    # goal planned for the next macro boundary

    def _plan(self, obs, noise_std=0.0):
        if self.random_goal and self.rng is not None:
            lo = np.array([THW_LO, SOC_GOAL_LO])
            hi = np.array([THW_HI, SOC_GOAL_HI])
            return _clip_goal(lo + self.rng.random(2) * (hi - lo))
        g = self.agent.plan_goal(obs, deterministic=True)
        if noise_std > 0.0 and self.rng is not None:
            # exploration in unit space, then clipped back to the goal bounds
            u = 2.0 * (g - np.array([THW_LO, SOC_GOAL_LO])) / \
                np.array([THW_HI - THW_LO, SOC_GOAL_HI - SOC_GOAL_LO]) - 1.0
            u = u + self.rng.normal(size=u.shape) * noise_std
            lo = np.array([THW_LO, SOC_GOAL_LO])
            hi = np.array([THW_HI, SOC_GOAL_HI])
            g = lo + (np.clip(u, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)
        return _clip_goal(g)

    def current_goal(self, obs, noise_std=0.0):
        """Return the interpolated goal reference for the current step."""
        k = self.step % self.macro_period
        if k == 0:
            if self.g_end is None:
                # episode start: anchor the interpolation at the measured goal
                self.g_start = measured_goal_from_obs(obs)
            else:
                self.g_start = self.g_end
            self.g_end = self._plan(obs, noise_std=noise_std)
        alpha = float(k) / float(self.macro_period)
        g_ref = (1.0 - alpha) * self.g_start + alpha * self.g_end
        return _clip_goal(g_ref)

    def act(self, obs, deterministic=True):
        """One control step: interpolated goal -> lower-layer action."""
        g_ref = self.current_goal(obs)
        a = self.agent.select_action(obs, g_ref, deterministic=deterministic)
        self.step += 1
        return a

    # ---------------- training helpers ----------------
    def lower_reward(self, env_reward, obs, goal):
        """Env reward + goal-tracking shaping for the lower layer."""
        ag = self.agent
        o = np.asarray(obs, dtype=np.float64).reshape(-1)
        d_ref = reference_gap(goal[0], o[0])
        gap_err = (float(o[2]) - d_ref) / ag.gap_scale
        soc_err = (float(o[SOC_IDX]) - float(goal[1])) / ag.soc_scale
        track = gap_err * gap_err + soc_err * soc_err
        return float(env_reward) - ag.goal_track_weight * track

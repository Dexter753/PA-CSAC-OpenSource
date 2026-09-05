# -*- coding: utf-8 -*-
"""SMORL: Safe Model-based Off-policy Reinforcement Learning (Zhu et al., 2022).

Implementation of Algorithm 1 of

  Z. Zhu, N. Pivaro, S. Gupta, A. Gupta and M. Canova, "Safe Model-based
  Off-policy Reinforcement Learning for Eco-Driving in Connected and
  Automated Hybrid Electric Vehicles", arXiv:2105.11640v2, 2022,

adapted to the CloudPCC car-following eco-driving MDP of this project.

Faithful components (paper reference in brackets):
  1. Receding-horizon trajectory optimization over a known physics-based
     vehicle/energy model [Eqn. (15), Sec. V-A].  The horizon dynamics are
     deterministic (persistence or learned preview of the preceding-vehicle
     speed), the control set is discretized (9 accelerations, identical to
     the MPC baseline of the main comparison), and the search is a
     deterministic dynamic programming over the discretized control grid,
     implemented as beam search [Sec. V-A].  Terminal cost: BCQ-constrained
     minimisation of the learned double Q [Eqns. (15)-(18)] plus a safe-set
     penalty P_N on the terminal state [Eqn. (28)].
  2. Off-policy Q learning with the Batch-Constrained correction of BCQ
     [Eqns. (17)-(23)]: twin critics, a conditional VAE G^w over the
     state-action distribution of the replay buffer (latent 5, 2x300 MLPs
     [Appendix A]), and a perturbation network xi^phi clipped to
     [-Phi, Phi] trained by the deterministic policy gradient [Eqn. (21)].
     The critic target uses the pessimistic max over the two target critics
     [Eqn. (17)], and the bootstrapped action is the argmin over n VAE
     samples plus perturbation [Eqn. (18)].
  3. Safe-set approximation by an autoregressive LSTM density over
     discretized states [Sec. V-C, Appendix B]: per-dimension one-hot
     inputs fed sequentially to an LSTM(50) with per-dimension masked
     softmax heads, trained by maximum likelihood on the states of
     successful (feasible) episodes only [Eqn. (32)]; a state is safe when
     p_psi(x) >= delta [Eqn. (26)].

Hyperparameters follow the paper's Table II where the mapping is direct:
gamma=0.995, Adam lr=1e-4, replay buffer 2e5, batch 256, tau=1e-3,
exploration rate eps=0.2, n=10 VAE-sampled actions, LSTM size 50.
Phi=30 Nm is mapped to 0.30 m/s^2 (~6% of the [-3, 2] m/s^2 action range,
matching the paper's ~30 Nm over a several-hundred-Nm torque range).

Adaptations to this MDP (documented deviations):
  - The stage cost is the negated environment reward of the shared
    car-following MDP (identical reward interface for every compared
    method), evaluated inside the planner with the same physics-based
    model (`reward_and_constraint`) that the environment itself uses
    [Sec. II-A1].  Actuator rate limits and the execution-time safety
    shield are treated as unmodelled execution layers, consistent with the
    MPC baseline, which models only a first-order actuator lag (tau_a).
  - The paper's spatial-domain transcription (200 m horizon, H_s=20,
    Delta_s=10 m) becomes a time-domain horizon of H=8 steps with
    dt = env.dt_episode, because the constraints of this MDP (gap safety,
    dispersion-scaled safe distance) are time-indexed rather than
    road-feature-indexed.
  - The receding-horizon reference of the preceding-vehicle speed uses
    the deployed predictor's t+1/t+3/t+5 mean forecasts (linear
    interpolation, identical to MPC-L) when provided; otherwise the
    observed v_lead is held constant (persistence).
  - Only successfully completed episodes (full horizon, no collision /
    dropout / numeric invalidation) are pushed to the replay buffer and
    the safe-set pool [Algorithm 1, Sec. V-C].
  - delta is recalibrated after each safe-set refit as the 5th percentile
    of p_psi over the current safe-state pool (implementation choice; the
    paper leaves the calibration of delta open).
  - Domain randomization of the paper (Sec. V-B, every 50 steps) is
    omitted: initial states come from the measured scenarios (shared
    protocol across all compared methods).
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.utils import dynamic_safe_distance, industry_hev_params, reward_and_constraint  # noqa: E402

# Actuator bounds of the shared MDP (m/s^2) and the discretized control
# grid, identical to the MPC baseline's act set.
ACT_LOW = -3.0
ACT_HIGH = 2.0
ACT_SET = np.array([-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0], dtype=np.float64)

# Observation layout of CloudPCCEnv (20-dim):
# 0 v_ego, 1 a_ego, 2 d_gap, 3 v_lead, 4 pred_v_mean, 5 sigma_mean,
# 6 ci_lower, 7 ci_upper, 8 density, 9 flow_speed, 10 lead_headway,
# 11 soc, 12..19 prob_emb
SOC_IDX = 11


class QMLP(nn.Module):
    """Critic MLP of size (200, 100, 50) [Sec. V-B, Fig. 4]."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class PerturbationNet(nn.Module):
    """Perturbation network xi^phi [Eqns. (18), (21), (23)], output clipped
    to [-Phi, Phi].  MLP of size (200, 100, 50) [Sec. V-B, Fig. 4]."""

    def __init__(self, obs_dim, act_dim, phi=0.30):
        super().__init__()
        self.phi = float(phi)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, act_dim),
        )

    def forward(self, obs, act):
        return self.phi * torch.tanh(self.net(torch.cat([obs, act], dim=-1)))


class BCQVAE(nn.Module):
    """Conditional VAE G^w over the buffer's state-action distribution
    [Eqn. (18), Appendix A]: latent dimension 5, encoder/decoder 2x300 MLPs.

    Encoder takes (obs, act) -> (mu, log_std); decoder takes (obs, z) -> act.
    """

    def __init__(self, obs_dim, act_dim, latent_dim=5, hidden=300):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.enc = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def encode(self, obs, act):
        h = self.enc(torch.cat([obs, act], dim=-1))
        mu, log_std = h.chunk(2, dim=-1)
        return mu, torch.clamp(log_std, -6.0, 2.0)

    def decode(self, obs, z):
        return torch.clamp(self.dec(torch.cat([obs, z], dim=-1)), ACT_LOW, ACT_HIGH)

    def loss(self, obs, act):
        mu, log_std = self.encode(obs, act)
        z = mu + torch.randn_like(mu) * log_std.exp()
        rec = self.dec(torch.cat([obs, z], dim=-1))
        recon = F.mse_loss(rec, act)
        kl = 0.5 * (mu.pow(2) + (2.0 * log_std).exp() - 1.0 - 2.0 * log_std).sum(-1).mean()
        return recon + kl, float(recon.item()), float(kl.item())

    def sample_actions(self, obs, n):
        """obs (B, obs_dim) -> n candidate actions per state, (B, n, act_dim)."""
        b = obs.shape[0]
        z = torch.randn(b, n, self.latent_dim, device=obs.device, dtype=obs.dtype)
        obs_rep = obs.unsqueeze(1).expand(b, n, obs.shape[-1]).reshape(b * n, obs.shape[-1])
        return self.decode(obs_rep, z.reshape(b * n, -1)).reshape(b, n, -1)


class SafeSetAR(nn.Module):
    """Autoregressive LSTM density over discretized states [Sec. V-C, App. B].

    Each state dimension is discretized into ``n_bins`` bins (one-hot);
    the one-hot of dimension i-1 is fed at LSTM step i (causal shift), and
    a per-dimension linear head maps the hidden state to a softmax over
    that dimension's bins.  log p(x) = sum_i log p(x_i | x_{<i}).
    """

    def __init__(self, obs_dim, n_bins=20, hidden=50):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_bins = int(n_bins)
        self.lstm = nn.LSTM(self.n_bins, int(hidden), batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, self.n_bins) for _ in range(self.obs_dim)])

    def log_prob(self, bins):
        """bins: (B, obs_dim) long tensor in [0, n_bins) -> (B,) log density."""
        b, d = bins.shape
        onehot = F.one_hot(bins, self.n_bins).to(torch.float32)  # (B, D, n_bins)
        inp = torch.zeros_like(onehot)
        inp[:, 1:] = onehot[:, :-1]
        h, _ = self.lstm(inp)  # (B, D, hidden)
        logp = torch.zeros(b, device=bins.device, dtype=torch.float32)
        for i in range(d):
            logits = self.heads[i](h[:, i, :])
            logp = logp + F.log_softmax(logits, dim=-1).gather(1, bins[:, i : i + 1]).squeeze(1)
        return logp


class SMORL:
    """Safe Model-based Off-policy RL agent (Algorithm 1 of Zhu et al. 2022).

    Execution policy: receding-horizon trajectory optimization whose
    terminal cost is the BCQ-constrained double-Q estimate plus a safe-set
    penalty.  Learning: off-policy twin-critic Q learning with VAE action
    generation and a perturbation network (BCQ), and an autoregressive
    LSTM safe set fitted on successful episodes only.
    """

    def __init__(
        self,
        obs_dim,
        act_dim=1,
        act_limit=2.0,
        gamma=0.995,
        tau=1e-3,
        lr=1e-4,
        phi=0.30,
        n_candidate=10,
        n_bins=20,
        lstm_hidden=50,
        horizon=8,
        beam_width=14,
        epsilon=0.2,
        env_params=None,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_candidate = int(n_candidate)
        self.horizon = int(horizon)
        self.beam_width = int(beam_width)
        self.epsilon = float(epsilon)
        self.n_bins = int(n_bins)

        # planner penalty weights: gap-safety deficit weight matches the MPC
        # baseline (w_safe = 10.0); safe-set log-density barrier weight.
        self.w_gap_safe = 10.0
        self.kappa_safe_set = 25.0
        self.tau_a = 0.5  # first-order actuator lag, identical to the MPC baseline

        self.env_params = dict(env_params) if env_params else industry_hev_params()

        # --- networks [Sec. IV, Algorithm 1] ---
        self.q1 = QMLP(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QMLP(self.obs_dim, self.act_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()
        self.vae = BCQVAE(self.obs_dim, self.act_dim).to(self.device)
        self.perturb = PerturbationNet(self.obs_dim, self.act_dim, phi=phi).to(self.device)
        self.perturb_target = copy.deepcopy(self.perturb).eval()
        self.safeset = SafeSetAR(self.obs_dim, n_bins=self.n_bins, hidden=lstm_hidden).to(self.device)

        # single learning rate alpha = 1e-4 for every module [Tab. II]
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=float(lr))
        self.perturb_opt = torch.optim.Adam(self.perturb.parameters(), lr=float(lr))
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=float(lr))
        self.safeset_opt = torch.optim.Adam(self.safeset.parameters(), lr=float(lr))

        # safe-set state
        self.bin_edges = None            # (obs_dim, n_bins+1) numpy
        self.log_delta = None            # scalar, log of delta [Eqn. (26)]
        self.safe_set_ready = False
        self._safe_pool = []             # list of (T_i, obs_dim) arrays
        self._pool_episodes_at_fit = 0
        self.refit_every_episodes = 10
        self.safe_fit_steps = 200
        self._safe_fit_rng = np.random.default_rng(0)

        self.train_time_seconds = 0.0
        self.train_memory_mb = 0.0
        self._update_it = 0

    # ------------------------------------------------------------------
    # safe-set management [Sec. V-C]
    # ------------------------------------------------------------------
    @property
    def safe_pool_size(self):
        return int(sum(a.shape[0] for a in self._safe_pool))

    def observe_safe_states(self, states, force_fit=False):
        """Append the state array of one successful episode; refit the LSTM
        density when enough new episodes have accumulated [Algorithm 1]."""
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return
        self._safe_pool.append(arr)
        n_ep = len(self._safe_pool)
        if force_fit or (n_ep - self._pool_episodes_at_fit) >= self.refit_every_episodes:
            self.fit_safe_set()

    def _compute_bins(self, pool):
        """Per-dimension percentile bin edges, frozen after warm start."""
        edges = np.empty((self.obs_dim, self.n_bins + 1), dtype=np.float64)
        qs = np.linspace(0.5, 99.5, self.n_bins + 1)
        for i in range(self.obs_dim):
            col = pool[:, i].astype(np.float64)
            e = np.percentile(col, qs)
            e = np.unique(e)
            if e.size < 2:
                lo, hi = float(e[0]) - 1.0, float(e[0]) + 1.0
                e = np.linspace(lo, hi, self.n_bins + 1)
            elif e.size < self.n_bins + 1:
                e = np.linspace(e[0], e[-1], self.n_bins + 1)
            edges[i] = e
        return edges

    def _digitize_np(self, x):
        """x: (B, obs_dim) float -> (B, obs_dim) int bin indices."""
        bins = np.empty(x.shape, dtype=np.int64)
        for i in range(self.obs_dim):
            inner = self.bin_edges[i][1:-1]
            bins[:, i] = np.clip(np.searchsorted(inner, x[:, i], side="right"), 0, self.n_bins - 1)
        return bins

    def fit_safe_set(self):
        """MLE fit of the autoregressive density on the safe pool [Eqn. (32)],
        then delta recalibration (5th percentile of p_psi over the pool)."""
        if not self._safe_pool:
            return
        pool = np.concatenate(self._safe_pool, axis=0)
        if self.bin_edges is None:
            self.bin_edges = self._compute_bins(pool)
        # subsample for a bounded fit cost
        if pool.shape[0] > 20000:
            sel = self._safe_fit_rng.choice(pool.shape[0], size=20000, replace=False)
            pool = pool[sel]
        x = torch.as_tensor(pool, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(self._digitize_np(pool), dtype=torch.long, device=self.device)
        self.safeset.train()
        n = b.shape[0]
        for _ in range(self.safe_fit_steps):
            idx = torch.as_tensor(
                self._safe_fit_rng.integers(0, n, size=min(256, n)), device=self.device, dtype=torch.long)
            loss = -self.safeset.log_prob(b[idx]).mean()
            self.safeset_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.safeset.parameters(), max_norm=10.0)
            self.safeset_opt.step()
        self.safeset.eval()
        # delta calibration: 5th percentile of log p over the pool
        with torch.no_grad():
            logp_chunks = []
            for s in range(0, n, 4096):
                logp_chunks.append(self.safeset.log_prob(b[s : s + 4096]).cpu().numpy())
            logp = np.concatenate(logp_chunks)
        self.log_delta = float(np.percentile(logp, 5.0))
        self.safe_set_ready = True
        self._pool_episodes_at_fit = len(self._safe_pool)

    # ------------------------------------------------------------------
    # trajectory optimization [Eqn. (15), Sec. V-A]
    # ------------------------------------------------------------------
    def _terminal_value(self, x_term):
        """BCQ-constrained terminal cost [Eqns. (15)-(18)] plus safe-set
        penalty P_N [Eqn. (28)].  x_term: (B, obs_dim) numpy -> (B,) numpy."""
        with torch.no_grad():
            x = torch.as_tensor(x_term, dtype=torch.float32, device=self.device)
            bsz = x.shape[0]
            n = self.n_candidate
            a_vae = self.vae.sample_actions(x, n)  # (B, n, A)
            x_rep = x.unsqueeze(1).expand(bsz, n, self.obs_dim).reshape(bsz * n, self.obs_dim)
            a_flat = a_vae.reshape(bsz * n, self.act_dim)
            pert = self.perturb(x_rep, a_flat).reshape(bsz, n, self.act_dim)
            a_cand = torch.clamp(a_vae + pert, ACT_LOW, ACT_HIGH)
            q_cand = torch.max(
                self.q1_target(x_rep, a_cand.reshape(bsz * n, self.act_dim)),
                self.q2_target(x_rep, a_cand.reshape(bsz * n, self.act_dim)),
            ).reshape(bsz, n)
            v = q_cand.min(dim=1).values  # (B,)
            if self.safe_set_ready:
                bins = torch.as_tensor(
                    self._digitize_np(np.asarray(x_term, dtype=np.float64)),
                    dtype=torch.long, device=self.device)
                logp = self.safeset.log_prob(bins).to(v.dtype)
                v = v + self.kappa_safe_set * torch.clamp(self.log_delta - logp, min=0.0)
        return v.cpu().numpy()

    def select_action(self, obs, deterministic=True, dt=1.0, v_lead_preview=None):
        """Execution policy: solve the receding-horizon trajectory
        optimization and return its first control [Algorithm 1]."""
        obs64 = np.asarray(obs, dtype=np.float64).reshape(-1)
        dt = float(np.clip(float(dt), 1e-3, 2.0))
        v_ego = float(obs64[0])
        a_ego = float(obs64[1])
        d_gap = float(obs64[2])
        v_lead = float(obs64[3])
        sigma_mean = float(np.clip(float(obs64[5]), 0.0, 10.0))
        soc = float(obs64[SOC_IDX])
        params = self.env_params
        horizon = self.horizon

        def _preview_at(h):
            if v_lead_preview is not None and len(v_lead_preview) >= h:
                return float(v_lead_preview[h - 1])
            return v_lead

        # beam = (accumulated stage cost, v, a, d_gap, soc, first control u)
        beams = [(0.0, v_ego, a_ego, d_gap, soc, None)]
        for h_step in range(horizon):
            v_lead_h = _preview_at(h_step + 1)
            cands = []
            for cost, v, a, d, s, first_u in beams:
                rel_v = float(v_lead_h - v)
                for u in ACT_SET.tolist():
                    # first-order actuator lag, identical to the MPC baseline
                    a_next = float(np.clip(a + (u - a) * (dt / self.tau_a), ACT_LOW, ACT_HIGH))
                    v_next = float(max(0.0, v + a_next * dt))
                    d_next = float(d + rel_v * dt)
                    jerk = float((a_next - a) / max(dt, 1e-6))
                    d_safe_next = dynamic_safe_distance(v_next, v_lead_h, sigma_mean)
                    try:
                        reward, _, _, soc_next, _ = reward_and_constraint(
                            v_next, a_next, jerk, d_next, d_safe_next, s, sigma_mean,
                            v_lead=v_lead_h, dt=dt, params=params, debug=False)
                    except Exception:
                        reward, soc_next = 0.0, s
                    # stage cost = negated environment reward [Eqn. (15)]
                    stage = -float(reward)
                    # hard-constraint penalty P^k: gap-safety deficit
                    deficit = float(max(0.0, 1.02 * d_safe_next - d_next))
                    stage = float(stage + self.w_gap_safe * deficit * deficit)
                    cands.append((cost + stage, v_next, a_next, d_next, soc_next,
                                  float(u) if first_u is None else first_u))
            cands.sort(key=lambda c: c[0])
            beams = cands[: self.beam_width] if cands else beams

        if not beams:
            return np.array([0.0], dtype=np.float32)

        # terminal cost over the surviving beams
        v_lead_end = _preview_at(horizon)
        x_term = np.stack([self._terminal_state(obs64, b, v_lead_end) for b in beams])
        v_term = self._terminal_value(x_term)
        totals = np.array([b[0] for b in beams], dtype=np.float64) + (self.gamma ** horizon) * v_term
        best = int(np.argmin(totals))
        best_u = float(beams[best][5]) if beams[best][5] is not None else 0.0
        return np.array([np.clip(best_u, ACT_LOW, ACT_HIGH)], dtype=np.float32)

    def _terminal_state(self, obs64, beam, v_lead_end):
        """G mapping [Sec. V-A]: planner state -> predicted terminal full
        state; exogenous features held at their observed values."""
        _, v, a, d, s, _ = beam
        x_h = obs64.copy()
        x_h[0] = float(v)
        x_h[1] = float(a)
        x_h[2] = float(d)
        x_h[3] = float(v_lead_end)
        x_h[SOC_IDX] = float(s)
        return x_h

    # ------------------------------------------------------------------
    # off-policy Q learning [Eqns. (17)-(23)]
    # ------------------------------------------------------------------
    def _polyak(self, src, dst):
        for p, pt in zip(src.parameters(), dst.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch):
        """One gradient step of the BCQ-style off-policy updates."""
        self._update_it += 1
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.asarray(batch["act"]), dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(np.asarray(batch["rew"]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.asarray(batch["next_obs"]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(batch["done"]), dtype=torch.float32, device=self.device)
        bsz = obs.shape[0]
        cost = -rew  # cost-to-go formulation (minimisation)

        # ---- critic target [Eqns. (17)-(18)] ----
        with torch.no_grad():
            n = self.n_candidate
            a_vae = self.vae.sample_actions(next_obs, n)  # (B, n, A)
            x_rep = next_obs.unsqueeze(1).expand(bsz, n, self.obs_dim).reshape(bsz * n, self.obs_dim)
            a_flat = a_vae.reshape(bsz * n, self.act_dim)
            pert = self.perturb(x_rep, a_flat).reshape(bsz, n, self.act_dim)
            a_cand = torch.clamp(a_vae + pert, ACT_LOW, ACT_HIGH)
            ac_flat = a_cand.reshape(bsz * n, self.act_dim)
            q_cand = torch.max(self.q1_target(x_rep, ac_flat), self.q2_target(x_rep, ac_flat)).reshape(bsz, n)
            k_best = q_cand.argmin(dim=1)  # (B,)
            u_next = a_cand[torch.arange(bsz, device=self.device), k_best]  # (B, A)
            q_next = torch.max(self.q1_target(next_obs, u_next), self.q2_target(next_obs, u_next))
            y = cost + self.gamma * (1.0 - done) * q_next

        q_loss = F.mse_loss(self.q1(obs, act), y) + F.mse_loss(self.q2(obs, act), y)
        self.q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10.0)
        self.q_opt.step()

        # ---- perturbation network [Eqn. (21)] ----
        a_pert = torch.clamp(act + self.perturb(obs, act), ACT_LOW, ACT_HIGH)
        p_loss = self.q1(obs, a_pert).mean()
        self.perturb_opt.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.perturb.parameters(), max_norm=10.0)
        self.perturb_opt.step()

        # ---- VAE [Eqn. (30), Appendix A] ----
        v_loss, rec, kl = self.vae.loss(obs, act)
        self.vae_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=10.0)
        self.vae_opt.step()

        # ---- target updates [Eqn. (22)] ----
        self._polyak(self.q1, self.q1_target)
        self._polyak(self.q2, self.q2_target)
        self._polyak(self.perturb, self.perturb_target)

        return {"q_loss": float(q_loss.item()), "p_loss": float(p_loss.item()),
                "vae_loss": float(v_loss.item()), "vae_rec": rec, "vae_kl": kl}

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path):
        torch.save({
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "vae": self.vae.state_dict(),
            "perturb": self.perturb.state_dict(),
            "perturb_target": self.perturb_target.state_dict(),
            "safeset": self.safeset.state_dict(),
            "bin_edges": self.bin_edges,
            "log_delta": self.log_delta,
            "safe_set_ready": self.safe_set_ready,
            "config": {
                "obs_dim": self.obs_dim, "act_dim": self.act_dim, "gamma": self.gamma,
                "tau": self.tau, "n_candidate": self.n_candidate, "n_bins": self.n_bins,
                "horizon": self.horizon, "beam_width": self.beam_width,
                "epsilon": self.epsilon, "w_gap_safe": self.w_gap_safe,
                "kappa_safe_set": self.kappa_safe_set,
            },
        }, str(path))

    def load(self, path):
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.vae.load_state_dict(ckpt["vae"])
        self.perturb.load_state_dict(ckpt["perturb"])
        self.perturb_target.load_state_dict(ckpt["perturb_target"])
        self.safeset.load_state_dict(ckpt["safeset"])
        self.bin_edges = ckpt["bin_edges"]
        self.log_delta = ckpt["log_delta"]
        self.safe_set_ready = bool(ckpt["safe_set_ready"])

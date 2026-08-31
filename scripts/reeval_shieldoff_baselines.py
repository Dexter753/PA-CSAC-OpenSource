# -*- coding: utf-8 -*-
"""
M8: extend the shield-off attribution to all learning-based baselines.

The paper's shield-off replay (reeval_perscenario.py Part 4) covers PA-CSAC
only. This script replays DDPG / TD3 / SAC / PPO / PPO-Lagrangian from their
retained best checkpoints under disable_shield=True (actuator rate limiting
retained), completing the attribution matrix: which methods rely on the
corrective layer versus learned compliance.

Outputs:
  results/reeval_perscenario/shieldoff_baselines_summary.csv   (per-seed)
  results/reeval_perscenario/shieldoff_baselines_mean_std.csv (aggregates)

Idempotent: per-seed detail CSVs are reused when present.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "algos" / "PA_CSAC"))

import reeval_perscenario as rp  # noqa: E402

import torch  # noqa: E402

# checkpoint names per method (same registry as rp.LEARNERS; on-policy keeps final)
CKPT_NAMES = {
    "DDPG": "ddpg_best.pt",
    "TD3": "td3_best.pt",
    "SAC": "sac_best.pt",
    "PPO": "ppo.pt",
    "PPO-Lag": "ppo_lagrangian.pt",
}

CLASS_OF = {"DDPG": "DDPG", "TD3": "TD3", "SAC": "SAC", "PPO": "PPO",
            "PPO-Lag": "PPOLagrangian"}

KWARGS_OF = {"PPO-Lag": dict(cost_limit=0.30)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = rp.OUT_DIR / "shieldoff_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, ckpt_name in CKPT_NAMES.items():
        cls_name = CLASS_OF[method]
        for seed in rp.SEEDS:
            ckpt = rp.PROJECT_ROOT / "results" / f"seed{seed}" / "models" / ckpt_name
            if not ckpt.exists():
                print(f"[{method}][seed {seed}] checkpoint missing: {ckpt}")
                continue
            rp.set_seed(int(seed))
            probe_env = rp.make_env(device, extra_params={"disable_shield": True})
            obs_dim = int(probe_env.observation_space.shape[0])
            agent = rp.load_agent(cls_name, ckpt, obs_dim, device, KWARGS_OF.get(method, {}))
            policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
            df = rp.eval_method_seed(f"ShieldOff_{method}", policy_fn, device, seed, out_dir,
                                     extra_params={"disable_shield": True})
            rows.append(rp._summarize_seed(df, f"ShieldOff_{method}", seed))
            print(f"[{method}][seed {seed}] VSR cur={rows[-1]['vsr_current']:.3f} "
                  f"fuel(cur)={rows[-1]['fuel_current_mean']:.2f}")

    if not rows:
        print("no rows produced; check checkpoint availability")
        return
    sdf = pd.DataFrame(rows)
    sdf.to_csv(rp.OUT_DIR / "shieldoff_baselines_summary.csv", index=False, encoding="utf-8-sig")

    agg = sdf.groupby("method").agg(
        n_seeds=("seed", "count"),
        fuel_mean=("fuel_current_mean", "mean"), fuel_std=("fuel_current_mean", "std"),
        gap_mean=("gap_rmse_current_mean", "mean"), gap_std=("gap_rmse_current_mean", "std"),
        jerk_mean=("jerk_rmse_current_mean", "mean"), jerk_std=("jerk_rmse_current_mean", "std"),
        vr_mean=("vr_hard_current_mean", "mean"), vr_std=("vr_hard_current_mean", "std"),
        vsr_mean=("vsr_current", "mean"), vsr_std=("vsr_current", "std"),
    ).reset_index()
    agg.to_csv(rp.OUT_DIR / "shieldoff_baselines_mean_std.csv", index=False, encoding="utf-8-sig")
    print("\n==== Shield-off baseline aggregates ====")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Component-ablation completion training for seeds 22 and 32.

The constraint-handling x embedding cross (Table 5 of the paper) was run at a
development stage on three seeds (42, 52, 62). This script completes the
cross to the full five-seed set (22, 32, 42, 52, 62) by training the five
component-ablation variants for the two missing seeds, with the identical
configuration snapshot recorded for the original runs
(results/seed42/run_config_snapshot.csv):

  - train_steps / ablation_train_steps = 60000
  - strict_prediction_columns = True
  - env: lower_violation_ratio=0.94, upper_cost_weight=0.4
  - comp-ablation exploration: policy_noise_init=0.03, noise_min=0.002
  - best_eval_episodes = 16
  - per-variant seed = global_seed + 200 (as in train.py STAGE 3.5)
  - penalty_weight = 1.0, prob_emb_lr = 1e-3, two_stage = True,
    phase1_ratio = 0.55, reward_scale = 5.0, reward_bias = 0.15,
    alpha_min/max = 0.02/0.05, phase2_lr_ratio = 0.025,
    shield_mismatch_coef = 0.18

Idempotent: a variant whose checkpoint already exists is skipped.

Usage:
  C:/Users/hp/anaconda3/envs/fleet/python.exe scripts/train_comp_ablation_seeds2232.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "algos" / "PA_CSAC"))

from train import train_pa_csac  # noqa: E402

CSV_PATH = PROJECT_ROOT / "prediction" / "results" / "csv" / "pcc_rl_prediction_dataset_for_control.csv"

TRAIN_STEPS = 60000
ENV_PARAMS = {"lower_violation_ratio": 0.94, "upper_cost_weight": 0.4}

# Same variant set as train.py STAGE 3.5 (comp_variants)
VARIANTS = [
    {"name": "pa_csac_full", "use_cost_constraint": True, "use_prob_embedding": True, "constraint_method": "penalty"},
    {"name": "pa_csac_lagrangian", "use_cost_constraint": True, "use_prob_embedding": True, "constraint_method": "lagrangian"},
    {"name": "no_lagrangian", "use_cost_constraint": False, "use_prob_embedding": True, "constraint_method": "penalty"},
    {"name": "no_embedding", "use_cost_constraint": True, "use_prob_embedding": False, "constraint_method": "penalty"},
    {"name": "no_lagrangian_no_embedding", "use_cost_constraint": False, "use_prob_embedding": False, "constraint_method": "penalty"},
]

SEEDS = [22, 32]


def main():
    for global_seed in SEEDS:
        model_dir = PROJECT_ROOT / "results" / f"seed{global_seed}" / "models"
        for comp in VARIANTS:
            name = comp["name"]
            save_dir = model_dir / f"comp_ablation_{name}"
            ckpt = save_dir / f"{name}.pt"
            if ckpt.exists():
                print(f"[skip] seed{global_seed} {name} already trained: {ckpt}")
                continue
            print(f"[train] seed{global_seed} {name} "
                  f"(cost={comp['use_cost_constraint']}, embed={comp['use_prob_embedding']}, "
                  f"method={comp['constraint_method']})")
            train_pa_csac(
                str(CSV_PATH),
                total_steps=TRAIN_STEPS,
                save_dir=str(save_dir),
                feature_mode="pa_csac",
                model_name=f"{name}.pt",
                history_tag=f"comp_{name}",
                env_params_override=dict(ENV_PARAMS),
                seed=int(global_seed) + 200,
                policy_noise_init=0.03,
                policy_noise_min=0.002,
                best_eval_episodes=16,
                strict_prediction_columns=True,
                strict_dedicated_prediction_columns=False,
                use_cost_constraint=comp["use_cost_constraint"],
                use_prob_embedding=comp["use_prob_embedding"],
                constraint_method=comp["constraint_method"],
                penalty_weight=1.0,
                prob_emb_lr=1e-3,
                two_stage=True,
                phase1_ratio=0.55,
                reward_scale=5.0,
                reward_bias=0.15,
                alpha_min=0.02,
                alpha_max=0.05,
                phase2_lr_ratio=0.025,
                shield_mismatch_coef=0.18,
            )
            print(f"[done] seed{global_seed} {name} -> {ckpt}")


if __name__ == "__main__":
    main()

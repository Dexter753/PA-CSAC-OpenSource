"""Five-seed unified replay of the constraint-handling x embedding screen.

Replays the five component-ablation checkpoints (pa_csac_full, pa_csac_lagrangian,
no_lagrangian, no_embedding, no_lagrangian_no_embedding) for all five seeds
(22, 32, 42, 52, 62) through the identical per-scenario re-evaluation protocol
of the main comparison (current dual-threshold gate, hard-violation-rate
column), replacing the development-stage three-seed soft-constraint screen of
the constraint-handling table. No training is performed; existing checkpoints
are replayed deterministically.

Outputs (under results/reeval_perscenario/):
    comp/                              per-seed per-scenario detail
    reeval_comp_ablation_summary.csv   per-seed summary rows
    reeval_comp_ablation_mean_std.csv  five-seed aggregates

Run:  python scripts/reeval_comp_ablation.py
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import reeval_perscenario as rp  # noqa: E402

import torch  # noqa: E402

# (row_name, ckpt_filename, agent_kwargs) -- feature_mode is pa_csac for all
# variants (the embedding toggle is an agent-side switch, not an env switch).
VARIANTS = [
    ("CompAblation_full_penalty_embed",
     "comp_ablation_pa_csac_full/pa_csac_full_best.pt",
     dict(cost_limit=0.30, use_prob_embedding=True, use_cost_constraint=True,
          constraint_method="penalty")),
    ("CompAblation_lagrangian_embed",
     "comp_ablation_pa_csac_lagrangian/pa_csac_lagrangian_best.pt",
     dict(cost_limit=0.30, use_prob_embedding=True, use_cost_constraint=True,
          constraint_method="lagrangian")),
    ("CompAblation_no_constraint_embed",
     "comp_ablation_no_lagrangian/no_lagrangian_best.pt",
     dict(cost_limit=0.30, use_prob_embedding=True, use_cost_constraint=False,
          constraint_method="penalty")),
    ("CompAblation_penalty_no_embed",
     "comp_ablation_no_embedding/no_embedding_best.pt",
     dict(cost_limit=0.30, use_prob_embedding=False, use_cost_constraint=True,
          constraint_method="penalty")),
    ("CompAblation_no_constraint_no_embed",
     "comp_ablation_no_lagrangian_no_embedding/no_lagrangian_no_embedding_best.pt",
     dict(cost_limit=0.30, use_prob_embedding=False, use_cost_constraint=False,
          constraint_method="penalty")),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = rp.OUT_DIR / "comp"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in rp.SEEDS:
        for row_name, ckpt_rel, kwargs in VARIANTS:
            ckpt = rp.PROJECT_ROOT / "results" / f"seed{seed}" / "models" / ckpt_rel
            if not ckpt.exists():
                print(f"[seed {seed}][{row_name}] checkpoint missing: {ckpt}")
                continue
            rp.set_seed(int(seed))
            probe_env = rp.make_env(device)
            obs_dim = int(probe_env.observation_space.shape[0])
            agent = rp.load_agent("PACSAC", ckpt, obs_dim, device, kwargs)
            policy_fn = lambda o, a=agent: a.select_action(o, deterministic=True)
            df = rp.eval_method_seed(row_name, policy_fn, device, seed, out_dir)
            rows.append(rp._summarize_seed(df, row_name, seed))
            print(f"[seed {seed}][{row_name}] VSR cur={rows[-1]['vsr_current']:.3f} "
                  f"fuel(cur)={rows[-1]['fuel_current_mean']:.2f}")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(rp.OUT_DIR / "reeval_comp_ablation_summary.csv", index=False, encoding="utf-8-sig")
    agg = sdf.groupby("method").agg(
        n_seeds=("seed", "count"),
        fuel_mean=("fuel_current_mean", "mean"), fuel_std=("fuel_current_mean", "std"),
        gap_mean=("gap_rmse_current_mean", "mean"), gap_std=("gap_rmse_current_mean", "std"),
        jerk_mean=("jerk_rmse_current_mean", "mean"), jerk_std=("jerk_rmse_current_mean", "std"),
        vr_mean=("vr_hard_current_mean", "mean"), vr_std=("vr_hard_current_mean", "std"),
        vsr_mean=("vsr_current", "mean"), vsr_std=("vsr_current", "std"),
    ).reset_index()
    agg.to_csv(rp.OUT_DIR / "reeval_comp_ablation_mean_std.csv", index=False, encoding="utf-8-sig")
    print("\n==== COMP-ABLATION five-seed aggregates (current protocol) ====")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

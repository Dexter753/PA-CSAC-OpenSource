"""Diagnostic: verify PPO-Lagrangian seed 52/62 checkpoint independence.

Computes pairwise actor-parameter L2 distances across the five seeds and,
for seeds 52 vs 62, the per-scenario rollout agreement already present in
results/reeval_perscenario/. Read-only; writes a small report to stdout.
"""
import hashlib
import itertools
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1] / "results"
SEEDS = [22, 32, 42, 52, 62]


def sha256_prefix(seed, n=16):
    p = ROOT / f"seed{seed}" / "models" / "ppo_lagrangian.pt"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_pi(seed):
    ck = torch.load(
        ROOT / f"seed{seed}" / "models" / "ppo_lagrangian.pt",
        map_location="cpu",
        weights_only=False,
    )
    return ck["pi"]


def l2(sd_a, sd_b):
    d2 = sum(((sd_a[k].float() - sd_b[k].float()) ** 2).sum().item() for k in sd_a)
    n2 = sum((sd_b[k].float() ** 2).sum().item() for k in sd_b)
    return d2 ** 0.5, n2 ** 0.5


def main():
    print("SHA-256 prefixes (first 16 hex chars) of the saved checkpoints:")
    for s in SEEDS:
        print(f"  seed {s}: {sha256_prefix(s)}")
    print()

    pis = {s: load_pi(s) for s in SEEDS}

    for s in SEEDS:
        norm, _ = l2(pis[s], {k: torch.zeros_like(v) for k, v in pis[s].items()})
        print(f"||theta_pi(seed {s})|| = {norm:.2f}")

    print()
    for a, b in itertools.combinations(SEEDS, 2):
        d, nb = l2(pis[a], pis[b])
        print(f"L2(pi_{a}, pi_{b}) = {d:8.2f}   (relative {d / nb:.2f})")

    # Per-scenario agreement between seeds 52 and 62
    print()
    cols = ["fuel_l_per_100km", "gap_rmse", "jerk_rmse", "vr_hard",
            "distance_km", "rate_limit_rate", "shield_rate", "upper_rate",
            "lower_rate", "steps"]
    for pair in [(52, 62), (22, 32), (22, 52)]:
        a, b = pair
        try:
            da = pd.read_csv(ROOT / "reeval_perscenario" / f"PPO-Lag_seed{a}_perscenario.csv")
            db = pd.read_csv(ROOT / "reeval_perscenario" / f"PPO-Lag_seed{b}_perscenario.csv")
        except FileNotFoundError:
            print(f"seed pair {pair}: per-scenario csv missing, skipped")
            continue
        print(f"--- rollout agreement seed {a} vs seed {b} (17 scenarios) ---")
        for c in cols:
            diff = (da[c] - db[c]).abs()
            print(f"  {c:>18}: max|diff| = {diff.max():.3e}   mean|diff| = {diff.mean():.3e}")
        flags_same = bool((da["valid_current"] == db["valid_current"]).all())
        print(f"  validity flags identical: {flags_same}")


if __name__ == "__main__":
    main()

"""Speed-/time-normalised comparison table for the main text (reviewer item 3).

Reads the unified per-scenario re-evaluation CSVs already produced by
reeval_perscenario.py (no new rollouts) and computes, for every method:

  - mean ego speed (km/h)         = distance_km / (steps * dt) * 3600
  - mean travel time per scenario (s) = steps * dt   (dt = 1 s)
  - mean delay vs lead vehicle (km/h) = lead mean speed - ego mean speed
        (lead speed taken from the test-split scenario data itself)
  - fuel per unit time (L/h)      = fuel_l_per_100km * v_kmh / 100
  - fuel per distance  (L/100km)  = as recorded

Two aggregation bases, mirroring the conditional / unconditional distinction
of the main text:
  - 'valid'  : conditional on each seed's own valid scenarios (Table 1 basis)
  - 'all'    : all 17 scenarios per seed (Table S9 basis)

Learning methods aggregate mean +/- std over the five seeds; deterministic
controllers are single evaluations. Outputs:
  results/reeval_perscenario/speed_fuel_table.csv   (long format, per seed)
  results/reeval_perscenario/speed_fuel_table.tex   (LaTeX table body)
"""
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = r"I:\资源汇总\强化学习车队节能控制项目-python\algos\PA_CSAC\prediction\results\csv\pcc_rl_prediction_dataset_for_control.csv"

LEARN = ["PA-CSAC", "DDPG", "TD3", "SAC", "PPO", "PPO-Lag", "SMORL", "HRL"]
TRAD = ["ACC", "MPC", "MPC-L", "LQR", "IDM"]
SEEDS = [22, 32, 42, 52, 62]
DT = 1.0  # s, unified control period (main text Sec. dataset_scenarios)
OUT_DIR = PROJECT_ROOT / "results" / "reeval_perscenario"


def load_env():
    """Instantiate the test-split env to read per-scenario lead-speed stats."""
    _ENV_PATH = PROJECT_ROOT / "algos" / "PA_CSAC" / "env.py"
    spec = importlib.util.spec_from_file_location("env_speedtbl", str(_ENV_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CloudPCCEnv(CSV_PATH, device="cpu", feature_mode="pa_csac",
                           split_mode="test", strict_prediction_columns=True,
                           strict_dedicated_prediction_columns=False)


def lead_speed_per_group(env):
    """Mean lead speed (km/h) of every test group, indexed by group_idx."""
    out = {}
    for g in env.processed_groups:
        v = np.asarray(g["v_lead"], dtype=float) * 3.6
        out[int(g["group_idx"] if "group_idx" in g else 0)] = float(v.mean())
    # processed_groups is a list ordered by construction; group_idx = list order
    return {i: float(np.asarray(g["v_lead"], dtype=float).mean() * 3.6)
            for i, g in enumerate(env.processed_groups)}


def per_seed_stats(df, lead_kmh):
    """Aggregate one seed's per-scenario rows into speed/time/delay/fuel stats."""
    res = {}
    for basis, mask in (("valid", df["valid_current"].astype(bool)),
                        ("all", pd.Series(True, index=df.index))):
        d = df[mask]
        if len(d) == 0:
            continue
        t_s = d["steps"].astype(float) * DT
        dist_km = d["distance_km"].astype(float)
        v_kmh = dist_km / t_s * 3600.0
        l_per_h = d["fuel_l_per_100km"].astype(float) * v_kmh / 100.0
        lead = d["group_idx"].astype(int).map(lead_kmh)
        delay = lead - v_kmh
        res[basis] = {
            "n_scen": int(len(d)),
            "v_mean_kmh": float(v_kmh.mean()),
            "t_mean_s": float(t_s.mean()),
            "delay_kmh": float(delay.mean()),
            "fuel_l_per_h": float(l_per_h.mean()),
            "fuel_l_100km": float(d["fuel_l_per_100km"].mean()),
        }
    return res


def fmt_pm(mean, std, prec=2):
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"


def main():
    env = load_env()
    lead_kmh = lead_speed_per_group(env)
    print(f"[env] test groups: {len(lead_kmh)}, "
          f"lead speed range {min(lead_kmh.values()):.1f}-{max(lead_kmh.values()):.1f} km/h, "
          f"corpus mean {np.mean(list(lead_kmh.values())):.2f} km/h")

    rows = []
    for m in LEARN:
        for s in SEEDS:
            p = OUT_DIR / f"{m}_seed{s}_perscenario.csv"
            if not p.exists():
                print(f"[warn] missing {p.name}")
                continue
            st = per_seed_stats(pd.read_csv(p), lead_kmh)
            for basis, v in st.items():
                v.update({"method": m, "seed": s, "basis": basis})
                rows.append(v)
    for m in TRAD:
        p = OUT_DIR / f"{m}_seed0_perscenario.csv"
        if not p.exists():
            print(f"[warn] missing {p.name}")
            continue
        st = per_seed_stats(pd.read_csv(p), lead_kmh)
        for basis, v in st.items():
            v.update({"method": m, "seed": 0, "basis": basis})
            rows.append(v)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "speed_fuel_table.csv", index=False, encoding="utf-8-sig")

    # ---- aggregate and build LaTeX body ----
    order = TRAD + LEARN
    label = {"ACC": "ACC", "MPC": "MPC", "MPC-L": "MPC-L", "LQR": "LQR", "IDM": "IDM",
             "DDPG": "DDPG", "TD3": "TD3", "SAC": "SAC", "PPO": "PPO",
             "PPO-Lag": "PPO-Lagrangian", "SMORL": "SMORL", "HRL": "HRL",
             "PA-CSAC": r"\textbf{PA-CSAC}"}
    tex_bodies = {"valid": [], "all": []}
    print("\n==== speed / time / delay / fuel table (mean over seeds) ====")
    hdr = ("method        basis   n_scen  v(kmh)        t(s)    delay(kmh)     "
           "L/h           L/100km")
    for basis in ("valid", "all"):
        print(f"\n--- basis: {basis} ---")
        print(hdr)
        for m in order:
            sub = df[(df["method"] == m) & (df["basis"] == basis)]
            if len(sub) == 0:
                continue
            if m in TRAD:
                r = sub.iloc[0]
                tex = (f"        {label[m]} & {r['v_mean_kmh']:.2f} & "
                       f"{r['t_mean_s']:.1f} & {r['delay_kmh']:.2f} & "
                       f"{r['fuel_l_per_h']:.2f} & {r['fuel_l_100km']:.2f} \\\\")
                print(f"{m:<13} {basis:<6} {int(r['n_scen']):>3}  "
                      f"{r['v_mean_kmh']:>8.2f}  {r['t_mean_s']:>8.1f}  "
                      f"{r['delay_kmh']:>8.2f}  {r['fuel_l_per_h']:>8.2f}  "
                      f"{r['fuel_l_100km']:>8.2f}")
            else:
                if len(sub) < 2:
                    continue
                agg = sub.select_dtypes(include=[float, int]).mean()
                std = sub[["v_mean_kmh", "delay_kmh", "fuel_l_per_h",
                           "fuel_l_100km"]].std()
                tex = (f"        {label[m]} & {fmt_pm(agg['v_mean_kmh'], std['v_mean_kmh'])} & "
                       f"{agg['t_mean_s']:.1f} & "
                       f"{fmt_pm(agg['delay_kmh'], std['delay_kmh'])} & "
                       f"{fmt_pm(agg['fuel_l_per_h'], std['fuel_l_per_h'])} & "
                       f"{fmt_pm(agg['fuel_l_100km'], std['fuel_l_100km'])} \\\\")
                print(f"{m:<13} {basis:<6} {agg['n_scen']:>5.1f}  "
                      f"{agg['v_mean_kmh']:>8.2f}±{std['v_mean_kmh']:.2f}  "
                      f"{agg['t_mean_s']:>8.1f}  "
                      f"{agg['delay_kmh']:>8.2f}±{std['delay_kmh']:.2f}  "
                      f"{agg['fuel_l_per_h']:>8.2f}±{std['fuel_l_per_h']:.2f}  "
                      f"{agg['fuel_l_100km']:>8.2f}±{std['fuel_l_100km']:.2f}")
            tex_bodies[basis].append(tex)
    for basis in ("valid", "all"):
        body = "\n".join(tex_bodies[basis])
        (OUT_DIR / f"speed_fuel_table_{basis}.tex").write_text(body + "\n",
                                                              encoding="utf-8")
    print(f"\nsaved -> {OUT_DIR / 'speed_fuel_table.csv'}")
    print(f"saved -> {OUT_DIR / 'speed_fuel_table_valid.tex'}")
    print(f"saved -> {OUT_DIR / 'speed_fuel_table_all.tex'}")


if __name__ == "__main__":
    main()

# Archived Result Provenance

The archived result folders in this release are copied from the paper-aligned experiment archives in the original research workspace (`PA_CSAC_CLEAN/results/`):

- `results/seed22` <- `PA_CSAC_CLEAN/results/seed22`
- `results/seed32` <- `PA_CSAC_CLEAN/results/seed32`
- `results/seed42` <- `PA_CSAC_CLEAN/results/seed42`
- `results/seed52` <- `PA_CSAC_CLEAN/results/seed52`
- `results/seed62` <- `PA_CSAC_CLEAN/results/seed62`
- `results/reeval_summary/` <- `PA_CSAC_CLEAN/results/reeval_perscenario/` (mean±std and summary tables only)
- `results/reward_hparam_sensitivity_aggregated.csv`, `results/split_conformal_analysis.csv`, `results/compute_cost.csv` <- `PA_CSAC_CLEAN/results/`

These five folders are the retained paper-result archives for the public release (five-seed protocol: 22, 32, 42, 52, 62), updated with the re-evaluated benchmark, ablation (prediction / adaptive-weight / mechanism), component-ablation, error-injection, and sensitivity-analysis outputs.

To keep the GitHub release lightweight, each archived `seed` folder is a slimmed paper-result package. The following bulky artifacts were intentionally removed from the public release:

- model checkpoints under `models/`
- raw rollout files under `traces/`
- training-history directories such as `histories/`
- intermediate `*_Trajectory.png` and `plot_benchmark_*` image exports (the paper-level `Paper_Ready_*` figures are retained)

Each archived `seed` folder retains:

- core result tables in CSV format (benchmark, ablation, component ablation, error injection, sensitivity, weight sensitivity)
- the run configuration snapshot and robustness summary text
- paper-level summary figures for comparison, ablation, robustness, SOC, training, and sensitivity visualization, plus the paper-ready per-controller trajectory figures and error-injection / sensitivity scenario figures

`results/reeval_summary/` retains the cross-seed mean±std re-evaluation tables for the main comparison, ablations, component ablations, error injection, shield on/off, and sigma-source analyses. The per-scenario re-evaluation details and the corresponding scripts' regenerated outputs live under `results/reeval_perscenario/` in the original workspace and are not part of the slimmed release.

Important note:

- The values reported in the manuscript main-results table are five-seed statistics aggregated from `seed22`, `seed32`, `seed42`, `seed52`, and `seed62` (see `reeval_main_table_mean_std.csv`).
- Therefore, the metrics in a single `benchmark_summary.csv` file do not equal the aggregated paper values by themselves.
- For paper reproduction, keep the seed set fixed to `22,32,42,52,62`.

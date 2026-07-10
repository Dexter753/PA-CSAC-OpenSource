# Archived Result Provenance

The archived result folders in this release are copied from the paper-aligned experiment archives in the original research workspace:

- `results/seed42` <- `PA_CSAC_CLEAN/results_quick/paper_full_20260624_111825/seed42`
- `results/seed52` <- `PA_CSAC_CLEAN/results_quick/paper_full_20260628_091121/seed52`
- `results/seed62` <- `PA_CSAC_CLEAN/results_quick/paper_full_20260630_012509/seed62`

These three folders are the retained paper-result archives for the public release.

To keep the GitHub release lightweight, each archived `seed` folder is a slimmed paper-result package. The following bulky artifacts were intentionally removed from the public release:

- model checkpoints under `models/`
- raw rollout files under `traces/`
- training-history directories such as `histories/`
- repetitive per-controller trajectory figures
- repetitive `Paper_Ready_*` and `plot_benchmark_*` image exports

Each archived `seed` folder now retains only:

- core result tables in CSV format
- the run configuration snapshot and robustness summary text
- a small set of paper-level summary figures for comparison, ablation, robustness, SOC, training, and sensitivity visualization

Important note:

- The values reported in the manuscript main-results table are three-seed statistics aggregated from `seed42`, `seed52`, and `seed62`.
- Therefore, the metrics in a single `benchmark_summary.csv` file do not equal the aggregated paper values by themselves.
- For paper reproduction, keep the seed set fixed to `42,52,62`.

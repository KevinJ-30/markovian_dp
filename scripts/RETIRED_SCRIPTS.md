# Scripts removable after unified-runner consolidation

These scripts are superseded by `scripts/run_experiments.py`. Retirement requires migrated active callers/imports, successful runner verification, and no live legacy jobs depending on the old files. Historical results, cached datasets, and third-party code are not deleted or rewritten.

Deletion is currently deferred: the active `progap_hidden128_standard8_tuning_20261002T213302Z` study still imports the old modules and launches the old worker. Keep these original files intact until that controller and its workers finish; use the new runner for subsequent experiments.

## Removable after replacement and drain

- `scripts/full_matrix.sh` — fixed grid replaced by `configs/main.json` and the unified runner.
- `scripts/full_matrix_queue.py` — queue and scheduling replaced by the unified runner.
- `scripts/full_matrix_campaign.py` — campaign controller replaced by unified scheduling, logs, state, and resume.
- `scripts/full_matrix_records.py` — fixed registry and sealed-manifest/hash machinery retired; necessary scientific behavior remains in the worker, ordinary configs/results, and migrated analysis consumers.
- `scripts/ideation_study.py` — grid/launch behavior replaced by `configs/gender_physics.json`; dataset presets moved into `scripts/run_experiment.py`, and validation-based selection into `scripts/summarize_results.py`.
- `scripts/sparse_ablation_grid.py` — generators replaced by `configs/sparse_ablation.json` and `configs/depth_ablation.json`; analysis no longer imports the old generator.
- `scripts/sparse_ablation_ofat.sh` — launch wrapper replaced by the unified runner using `configs/sparse_ablation.json`.
- `scripts/sparse_ablation_paper.sh` — experiment launches replaced by the two ablation configs; invoke the retained `scripts/sparse_ablation.py` separately for figures.

## Old filenames removable; implementation retained under new names

- `scripts/full_matrix_run.py` → `scripts/run_experiment.py`: retains training, privacy calibration, metrics, and native artifacts. ProGAP's default depth is now **3**, with four native training stages.
- `scripts/full_matrix_runtime.py` → `scripts/runner_runtime.py`: retains GPU authorization, process ownership/cleanup, locking, and supervision; obsolete provenance machinery is removed.

The new path contains no compatibility launchers or re-export aliases. The original legacy files remain solely to avoid breaking the active study; remove them after it drains.

## Kept

- `scripts/summarize_results.py` — table generation, historical CSV support, and validation-based configuration selection.
- `scripts/sparse_ablation.py` — analysis/figure entrypoint using ordinary result indexes rather than deleted registry modules.
- `scripts/sparsegnn_final_*` — separately documented archived initial-tuning utilities, unchanged by this consolidation.
- Other scripts have not been established as redundant by this consolidation; do not remove them based on this inventory.

This is an inventory, not a deletion script. Future experiments should use JSON configs with the unified runner; see [usage](README.md#unified-experiment-runner).

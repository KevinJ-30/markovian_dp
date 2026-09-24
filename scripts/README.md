# Supporting scripts

Training entry points are `python -m src.experiments.sparse` and
`python -m src.experiments.run`. The legacy local experiment drivers and
study-specific figure recipes have been removed.

## Reusable utilities

| Script | Purpose |
|---|---|
| `setup_graphsaint.sh` | Extract and check manually downloaded GraphSAINT datasets. |
| `calibrate_grid.py` | Emit noise multipliers for a grid of privacy targets. |
| `ceiling_fullbatch.py` | Run a non-private full-batch classification comparison. |
| `summarize_sweep.py` | Select a validation-best step and report seed-averaged test results; one configuration per child directory. |
| `summarize_matched_eps.py` | Summarize matched-budget studies using their expected directory naming conventions. |
| `plot_frontier.py` | Plot privacy–utility curves from an explicit CSV glob. |
| `plot_sparse_frontier.py` | Plot a compatible SparseGNN CSV containing epsilon values. |

The Python utilities expose `--help`. GraphSAINT setup accepts an input ZIP
directory and an optional destination; see the root README.

## Cluster helpers

- `_ice_env.sh`: sourced by the ICE Slurm launchers to configure their environment.
- `_matched_eps_grid.sh`: sourced by `sbatch/graphsaint_meps.sbatch` to run its
  matched-budget grid.

These are source-only helpers, not standalone training commands.

## Study-specific campaign helpers

`sparsegnn_final_sweep.py`, `sparsegnn_final_reports.py`,
`sparsegnn_final_verification.py`, and `sparsegnn_partial_nonprivate.py` are
retained for the initial-tuning study. They depend on that study's Python
modules and original layout; they are not general-purpose experiment commands.
Moving the study under `results/old_stuff/` does not automatically migrate
their imports or paths.

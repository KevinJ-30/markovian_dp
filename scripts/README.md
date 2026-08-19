# Experiment entry points

    _dataset_settings.sh     per-dataset config sourced by every driver
    ladder_stage01.sh        S0 baselines + S1 sparsification sweep (no DP)
    ladder_stage2.sh         S2 clip+noise sweep, then S3 post-hoc epsilon
    orientation_ablation.sh  in- vs out-expansion (directed graphs only)
    relbench_f1.sh           RelBench entity-task ladder
    sweep.sh <axis>          one-axis tuning sweeps (lr, momentum, clip,
                             batch, k, optimizer)
    diagnose.sh <what>       gradnorm / metrics diagnostics
    ceiling_fullbatch.py     non-DP utility ceiling, full-batch
    summarize_sweep.py       best-checkpoint table for a sweep directory
    plot_*.py                figures

Long runs should be detached so neither a closed terminal nor idle sleep kills
them:

    nohup caffeinate -i ./scripts/sweep.sh lr > results/logs/sweep_lr.log 2>&1 &

Set `OMP_NUM_THREADS=1` when running many cells concurrently; each cell is
launch-latency bound on small subgraphs, so parallel processes beat threads.

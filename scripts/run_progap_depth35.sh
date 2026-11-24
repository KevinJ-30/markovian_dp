#!/usr/bin/env bash
# Run the depth-specific tuning, validation selection, and five-seed repeats.
set -euo pipefail
cd "$(dirname "$0")/.."

runner=/usr/scratch/asaha92/envs/graph_subsampling/bin/python
tune=main_r1_eps1258_tune_progap_depth35
repeats=main_r1_eps1258_retuned_repeats_progap_depth35
repeat_config="results/$tune/repeats.json"

run_sweep() {
    local config=$1 name=$2
    local options=(--gpus 0,1,2,3,4,5,6,7 --max-jobs-per-gpu 3 --progap-python "$runner")
    if [[ -f "results/$name/state.json" ]]; then
        options+=(--resume --retry-failed)
    fi
    "$runner" scripts/run_experiments.py "$config" "${options[@]}"
}

run_sweep "configs/tuning/$tune.json" "$tune"
if [[ ! -f "$repeat_config" ]]; then
    "$runner" scripts/make_repeat_config.py configs/tuning/main_r1_eps1258_selection_progap_depth35.json
fi
run_sweep "$repeat_config" "$repeats"

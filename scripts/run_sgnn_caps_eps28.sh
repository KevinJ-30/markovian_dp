#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python=/usr/scratch/asaha92/envs/graph_subsampling/bin/python
study=sgnn_k5p02_k1p1_eps28
tune=${study}_tune
repeat_config="results/$tune/repeats.json"

run_sweep() {
    local config=$1 name=$2
    local options=(--gpus 0,1,2,3,4,5,6,7)
    if [[ -f "results/$name/state.json" ]]; then
        options+=(--resume --retry-failed)
    fi
    "$python" scripts/run_experiments.py "$config" "${options[@]}"
}

run_sweep "configs/tuning/$tune.json" "$tune"
if [[ ! -f "$repeat_config" ]]; then
    "$python" scripts/make_repeat_config.py "configs/tuning/${study}_selection.json"
fi
run_sweep "$repeat_config" "${study}_seeds6_10"

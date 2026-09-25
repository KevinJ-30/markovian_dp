#!/usr/bin/env bash
# Sequential, configuration-isolated final experiment grid. No GPU scheduling.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/full_matrix.sh [--dry-run]

Run the complete matrix, then write summary.csv and summary.md under OUT_ROOT.
--dry-run prints shell-escaped commands and the run count without loading data,
creating directories, allocating GPUs, or invoking training/the summarizer.

Environment (space-separated values for grid variables):
  PYTHON             Python executable with the main training dependencies [python]
  PROGAP_PYTHON      Python executable with ProGAP dependencies [same as PYTHON]
  DEVICE             Training device [cuda]
  OUT_ROOT           Result root [<repository>/results/full_matrix]
  DATASETS           Protocols [8 standard + 3 all-but-two domain protocols]
  METHODS            mlp graphsage gin dp_mlp progap dpar dp_gnn_sage dp_gnn_gin
                     sparse_sage sparse_gin
  EPSILONS           Private targets [2 8]; non-private methods run only once
  SEEDS              Training seeds [0]; graph split stays fixed
  LEARNING_RATES     [0.01 0.001]
  BATCH_SIZES        [256 1024]
  EPOCHS             [20]; ProGAP uses this many epochs per stage
  P2_VALUES         SparseGNN edge sampling probabilities [0.5 0.1]
  BOOTSTRAP_RESAMPLES [1000]; final test bootstrap, confidence 95%

Fixed: MLP/DP-MLP hidden64; graph methods hidden128; dropout0.5.
DPAR retains its native 70-PPR-root limit and default graph sampling rate.
Existing per-epoch validation selects the final test checkpoint.
Use a new OUT_ROOT for a new campaign; existing run folders are never replaced.
HELP
}

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (($#)); then
    usage >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
PYTHON="${PYTHON:-python}"
PROGAP_PYTHON="${PROGAP_PYTHON:-$PYTHON}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/full_matrix}"
# Relative output roots are always relative to the repository, including when
# the launcher is invoked by an absolute path from another directory.
[[ "$OUT_ROOT" = /* ]] || OUT_ROOT="$ROOT/$OUT_ROOT"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-1000}"

read -r -a datasets <<< "${DATASETS:-ogbn-arxiv ogbn-products saint-reddit saint-yelp saint-amazon twitch-allbut2 facebook100-allbut2 mag-allbut2}"
read -r -a methods <<< "${METHODS:-mlp graphsage gin dp_mlp progap dpar dp_gnn_sage dp_gnn_gin sparse_sage sparse_gin}"
read -r -a epsilons <<< "${EPSILONS:-2 8}"
read -r -a seeds <<< "${SEEDS:-0}"
read -r -a rates <<< "${LEARNING_RATES:-0.01 0.001}"
read -r -a batches <<< "${BATCH_SIZES:-256 1024}"
read -r -a epochs_grid <<< "${EPOCHS:-20}"
read -r -a p2_values <<< "${P2_VALUES:-0.5 0.1}"

for method in "${methods[@]}"; do
    case "$method" in
        mlp|graphsage|gin|dp_mlp|progap|dpar|dp_gnn_sage|dp_gnn_gin|sparse_sage|sparse_gin) ;;
        *) printf 'Unknown method: %s\n' "$method" >&2; exit 2 ;;
    esac
done

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

count=0
for dataset in "${datasets[@]}"; do
    dataset_slug="${dataset//[^[:alnum:]._-]/_}"
    for method in "${methods[@]}"; do
        case "$method" in
            mlp|graphsage|gin) privacy_values=(non-private) ;;
            *) privacy_values=("${epsilons[@]}") ;;
        esac
        case "$method" in
            sparse_sage|sparse_gin) sampling_values=("${p2_values[@]}") ;;
            *) sampling_values=(none) ;;
        esac
        for epsilon in "${privacy_values[@]}"; do
            privacy_tag=non-private
            [[ "$epsilon" == non-private ]] || privacy_tag="eps${epsilon}"
            for lr in "${rates[@]}"; do
                for batch in "${batches[@]}"; do
                    for epochs in "${epochs_grid[@]}"; do
                        for p2 in "${sampling_values[@]}"; do
                            regime="lr${lr}_b${batch}_e${epochs}"
                            [[ "$p2" == none ]] || regime+="_p2${p2}"
                            for seed in "${seeds[@]}"; do
                                run_dir="$OUT_ROOT/runs/$dataset_slug/$method/$privacy_tag/$regime/seed${seed}"
                                command=("$PYTHON" "$ROOT/scripts/full_matrix_run.py"
                                    --dataset "$dataset" --method "$method"
                                    --lr "$lr" --batch-size "$batch" --epochs "$epochs"
                                    --seed "$seed" --dropout 0.5 --mlp-hidden 64 --gnn-hidden 128
                                    --device "$DEVICE" --out-dir "$run_dir"
                                    --bootstrap-resamples "$BOOTSTRAP_RESAMPLES")
                                [[ "$epsilon" == non-private ]] || command+=(--epsilon "$epsilon")
                                [[ "$p2" == none ]] || command+=(--p2 "$p2")
                                [[ "$method" != progap ]] || command+=(--progap-python "$PROGAP_PYTHON")
                                count=$((count + 1))
                                print_command "${command[@]}"
                                if (( ! DRY_RUN )); then
                                    if [[ -e "$run_dir" ]]; then
                                        printf 'Refusing existing run directory: %s\nUse a fresh OUT_ROOT.\n' "$run_dir" >&2
                                        exit 2
                                    fi
                                    mkdir -p -- "$OUT_ROOT/logs"
                                    log="$OUT_ROOT/logs/${dataset_slug}__${method}__${privacy_tag}__${regime}__seed${seed}.log"
                                    "${command[@]}" 2>&1 | tee "$log"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

summary=("$PYTHON" "$ROOT/scripts/summarize_results.py"
    "$OUT_ROOT/runs/**/result.csv" --bootstrap --best --out "$OUT_ROOT/summary")
print_command "${summary[@]}"
if (( DRY_RUN )); then
    printf 'Dry run: %d training runs; no experiments executed.\n' "$count"
else
    "${summary[@]}"
    printf 'Completed %d training runs. Results: %s/summary.csv and %s/summary.md\n' "$count" "$OUT_ROOT" "$OUT_ROOT"
fi

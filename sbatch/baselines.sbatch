#!/bin/bash
#SBATCH -Jbaseline
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t8:00:00
#SBATCH -oReport-baseline-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── src/experiments/run.py baselines (BaselineTrainer / DPAR / DP-GNN / upstream) ──
#
#   CONFIG=configs/flickr_mlp.json sbatch --export=ALL sbatch/baselines.sbatch
#   CONFIG=configs/flickr_mlp.json METHOD=graphsage sbatch --export=ALL ...
#
# GPU: sbatch flags override the #SBATCH lines, e.g.
#   sbatch -p ice-gpu --gres=gpu:1 --export=ALL sbatch/baselines.sbatch
#
# Different code path from graphsaint_meps.sbatch: run.py -> BaselineTrainer,
# with its own accountant (Opacus PRV, add/remove) and delta.  Compare utility
# sigma-for-sigma, NOT epsilon-for-epsilon -- always pin noise_multiplier
# rather than a target epsilon, or the baseline gets ~2x less noise.
#
# Results land in results/inductive/<dataset>/<method>.json.

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

CONFIG=${CONFIG:?set CONFIG, e.g. configs/flickr_mlp.json}
[ -f "$CONFIG" ] || { echo "FATAL: no such config: $CONFIG" >&2; exit 1; }

ARGS=(--config "$CONFIG")
[ -n "${DATASET:-}" ] && ARGS+=(--dataset "$DATASET")
[ -n "${METHOD:-}" ]  && ARGS+=(--method "$METHOD")
[ -n "${OUT:-}" ]     && ARGS+=(--out "$OUT")

echo "=== baseline  $(date) ==="
echo "    config=$CONFIG ${DATASET:+dataset=$DATASET} ${METHOD:+method=$METHOD}"
$PY -c "import json,sys; d=json.load(open('$CONFIG'));
print('   ', d['dataset'], d['method'], json.dumps(d.get('parameters',{})))"

$PY -u -m src.experiments.run "${ARGS[@]}"

echo
echo "=== complete $(date) ==="

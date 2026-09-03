#!/bin/zsh
# rel-trial/study-outcome privacy-utility ladder (binary node classification,
# AUROC).  ~12k labelled roots over a 5.4M-node relational graph; r=2 so a row
# root reaches its trial's related entities.  K_in=20, K_out=3 -- under
# in-expansion epsilon is priced by K_out, so this FK graph is cheap.
#
# Each cell is tracked every 25 steps, so one run yields a whole eps curve.
# Non-DP reference (p2 1.0 and 0.1) plus a DP p2 x sigma grid.
#
#   nohup caffeinate -i ./scripts/_reltrial_ladder.sh > results/logs/reltrial.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python
export RELBENCH_CACHE_DIR=/private/tmp/claude-501/-Users-kevinjacob-markovian-dp-copy/101f6985-da05-419f-9855-65ff68fe6783/scratchpad/relbench_cache
DS="relbench:rel-trial/study-outcome"
DELTA=1e-5
COMMON=(--dataset $DS --inductive --direction in --model binary_gnn --aggr mean \
        --p1 0.05 --r 2 --num_layers 2 --K_in 20 --K_out 3 --clip 1.0 \
        --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 3 \
        --T 500 --track_every 25)

run_cell() {   # out_dir, then extra flags
  local OUT=$1; shift
  if [[ -s "$OUT/sparse_gnn_relbench_rel-trial_study-outcome_dp_results.csv" \
     || -s "$OUT/sparse_gnn_relbench_rel-trial_study-outcome_results.csv" ]]; then
    echo "  [skip] $OUT"; return 0
  fi
  $PY -u -m src.sparse.run $COMMON "$@" --out_dir $OUT
}

mkdir -p results/logs results/relbench/reltrial

echo "=== non-DP reference $(date) ==="
for P2 in 1.0 0.1; do
  run_cell results/relbench/reltrial/nodp_p${P2} --p2 $P2 --lr 0.01
done

echo "=== DP grid $(date) ==="
for P2 in 1.0 0.1; do
  for SIGMA in 2.0 5.0 20.0; do
    OUT=results/relbench/reltrial/dp_p${P2}_s${SIGMA}
    run_cell $OUT --dp --p2 $P2 --sigma $SIGMA --lr 0.3
    $PY -u -m src.sparse.compute_epsilon \
        --csv $OUT/sparse_gnn_relbench_rel-trial_study-outcome_dp_results.csv \
        --delta $DELTA | tail -2 || true
  done
done

echo "=== summary $(date) ==="
$PY scripts/summarize_sweep.py results/relbench/reltrial --metric test_acc || true
echo "=== RELTRIAL LADDER COMPLETE $(date) ==="

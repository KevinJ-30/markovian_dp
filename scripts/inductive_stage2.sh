#!/bin/zsh
# Stage 2 (clip + noise) and Stage 3 (post-hoc epsilon).
#
#   ./scripts/inductive_stage2.sh                       # arxiv + flickr
#   DATASETS="ppi" ./scripts/inductive_stage2.sh        # PPI only
#
#   nohup caffeinate -i ./scripts/inductive_stage2.sh > results/inductive_stage2.log 2>&1 &
#
# Grid per dataset and per r: p2 {1.0,0.5,0.25,0.1} x sigma {2,5,10,20} x 3 seeds
# = 48 runs.  CSVs are flushed per row, so a killed run keeps what it finished.
#
# p1 is the SAME as the Stage 0-1 runs, so this frontier is directly readable
# against that ceiling.  lr differs by design: Adam at 0.01 for non-DP, SGD at
# 1.0 for DP, where the noisy sum is divided by the expected batch.
#
# Stage 3 attaches epsilon from Theorem 6.4 (node substitution), because these
# runs use --direction in.  The tighter Theorem 4.5 marked pair exists only for
# out-expansion; scripts/orientation_ablation.sh reports both side by side.

set -e
cd "$(dirname "$0")/.."
PY=(/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python -u)
DATASETS=(${=DATASETS:-ogbn-arxiv flickr})

# Resume support: skip any block whose output CSV already exists and is
# non-empty.  A lid-close pause never needs this (macOS suspends and resumes the
# processes intact), but if the job is killed you can rerun the same command and
# it picks up at the first unfinished block.  FORCE=1 redoes everything.
done_already() {
  if [[ -z "$FORCE" && -s "$1" ]]; then
    echo "  [skip] already have $1"
    return 0
  fi
  return 1
}


for DS in $DATASETS; do
  source scripts/_dataset_settings.sh $DS
  echo "\n########## $DS ##########"

  for R in $R_VALUES; do
    OUT=results/inductive_stage2_${TAG}_r$R
    echo "=== [S2] DP sigma sweep, r=$R (L=$R), capped ==="
    done_already $OUT/sparse_gnn_${TAG}_dp_results.csv || \
    $PY -m src.sparse.run --dataset $DS $INDUCTIVE --direction in --dp \
        $MODEL $CAP $REG --p1 $P1 --p2 $P2_GRID --r $R --num_layers $L \
        --sigma $SIGMA_GRID --clip $CLIP --lr $LR_DP \
        --T $T --seeds $SEEDS --roots_from train --out_dir $OUT

    echo "=== [S3] post-hoc epsilon (Theorem 6.4), r=$R ==="
    $PY -m src.sparse.compute_epsilon \
        --csv $OUT/sparse_gnn_${TAG}_dp_results.csv --delta $DELTA

    # Ceiling line comes from inductive_stage01.sh; skipped quietly if absent.
    $PY scripts/plot_sparse_frontier.py \
        --csv $OUT/sparse_gnn_${TAG}_dp_results_with_eps.csv \
        --ceiling_csv results/inductive_stage1_${TAG}_r$R/sparse_gnn_${TAG}_results.csv \
        --out results/inductive_frontier_${TAG}_r$R.png || true
  done
done

echo "\n=== Stage 2-3 complete ==="

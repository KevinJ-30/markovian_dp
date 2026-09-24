#!/bin/zsh
# Non-private ceiling and sparsification-only runs. These establish the utility
# ladder that the private runs degrade from.
#
#   ./scripts/ladder_stage01.sh                       # arxiv + flickr
#   DATASETS="ppi-large" ./scripts/ladder_stage01.sh  # PPI-large only
#   DATASETS="ogbn-arxiv flickr ppi-large" ./scripts/ladder_stage01.sh
#
#   nohup caffeinate -i ./scripts/ladder_stage01.sh > results/logs/ladder_stage01.log 2>&1 &
#
# Every dataset is a plain member of DATASETS; all per-dataset differences live
# in scripts/_dataset_settings.sh so all the ladder scripts agree.
#
# Every run uses the fixed two-layer GNN (--num_layers $L); r sweeps the
# expansion depth / privacy radius independently (see _dataset_settings.sh).
#
# NOTE: anything recorded before 2026-08-12 used --aggr gcn and/or --direction
# out, both of which make the rooted computation disagree with full-graph
# evaluation.  Those CSVs are not comparable with these; regenerate, don't mix.

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
  COMMON=(--dataset $DS --direction in --p1 $P1 --T $T \
          --lr $LR_NONDP $REG --seeds $SEEDS )


  # S0 ceiling. At p2=1, no cap, and r=L the per-root computation is
  # full-graph inference (mean aggregation), so full-batch reaches the same
  # number in ~1 min instead of ~4 h on PPI.  SLOW_CEILING=1 forces the per-root
  # path if you want it measured through the sampling loop.
  echo "=== [S0] ceiling: all edges, no cap, r=$CEIL_R (L=$CEIL_R) ==="
  if done_already results/inductive_ceiling_$TAG/sparse_gnn_${TAG}_results.csv; then
    :
  elif [[ -n "$SLOW_CEILING" ]]; then
    $PY -m src.experiments.sparse $COMMON $MODEL --p2 1.0 --r $CEIL_R \
        --num_layers $CEIL_R --out_dir results/inductive_ceiling_$TAG
  else
    $PY scripts/ceiling_fullbatch.py --dataset $DS $MODEL $REG \
        --num_layers $CEIL_R --lr $LR_NONDP --epochs 300 --seeds $SEEDS \
        --out_dir results/inductive_ceiling_$TAG
  fi

  for R in $R_VALUES; do
    echo "=== [S1] sparsification sweep, r=$R (L=$R), capped ==="
    done_already results/inductive_stage1_${TAG}_r$R/sparse_gnn_${TAG}_results.csv || \
    $PY -m src.experiments.sparse $COMMON $MODEL $CAP --p2 $P2_GRID --r $R \
        --num_layers $L --out_dir results/inductive_stage1_${TAG}_r$R
  done
done

echo "\n=== Non-private ladder complete ==="

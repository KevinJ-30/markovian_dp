#!/bin/zsh
# Stage 0 (baselines) + Stage 1 (sparsification-only).  No DP yet — this
# establishes the utility ladder the Stage 2 runs degrade from.
#
#   ./scripts/inductive_stage01.sh                       # arxiv + flickr
#   DATASETS="ppi" ./scripts/inductive_stage01.sh        # PPI only
#   DATASETS="ogbn-arxiv flickr ppi" ./scripts/inductive_stage01.sh
#
#   nohup caffeinate -i ./scripts/inductive_stage01.sh > results/inductive_stage01.log 2>&1 &
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
  COMMON=(--dataset $DS $INDUCTIVE --direction in --p1 $P1 --T $T \
          --lr $LR_NONDP $REG --seeds $SEEDS --roots_from train)

  echo "=== [S0a] graph-blind baseline (r=0) ==="
  done_already results/inductive_blind_$TAG/sparse_gnn_${TAG}_results.csv || \
  $PY -m src.sparse.run $COMMON $BLIND --p2 1.0 \
      --out_dir results/inductive_blind_$TAG

  # S0b ceiling.  At p2=1, no cap, and r=L the per-root computation IS
  # full-graph inference (mean aggregation), so full-batch reaches the same
  # number in ~1 min instead of ~4 h on PPI.  SLOW_CEILING=1 forces the per-root
  # path if you want it measured through the sampling loop.
  echo "=== [S0b] ceiling: all edges, no cap, r=$CEIL_R (L=$CEIL_R) ==="
  if done_already results/inductive_ceiling_$TAG/sparse_gnn_${TAG}_results.csv; then
    :
  elif [[ -n "$SLOW_CEILING" ]]; then
    $PY -m src.sparse.run $COMMON $MODEL --p2 1.0 --r $CEIL_R \
        --num_layers $CEIL_R --out_dir results/inductive_ceiling_$TAG
  else
    $PY scripts/ceiling_fullbatch.py --dataset $DS $INDUCTIVE $MODEL $REG \
        --num_layers $CEIL_R --lr $LR_NONDP --epochs 300 --seeds $SEEDS \
        --out_dir results/inductive_ceiling_$TAG
  fi

  for R in $R_VALUES; do
    echo "=== [S1] sparsification sweep, r=$R (L=$R), capped ==="
    done_already results/inductive_stage1_${TAG}_r$R/sparse_gnn_${TAG}_results.csv || \
    $PY -m src.sparse.run $COMMON $MODEL $CAP --p2 $P2_GRID --r $R \
        --num_layers $L --out_dir results/inductive_stage1_${TAG}_r$R
  done
done

echo "\n=== Stage 0-1 complete ==="

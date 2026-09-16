#!/bin/zsh
# Orientation ablation: the experiment behind Sections 5-6 of manuscript v35.
#
#   nohup caffeinate -i ./scripts/orientation_ablation.sh > results/orientation.log 2>&1 &
#
# Runs identical configurations twice, once per expansion orientation:
#
#   --direction in    Algorithm 5. Traverses incoming edges, so the retained arcs
#                     point at the root and message passing delivers neighbor
#                     features to it.
#   --direction out   Legacy Algorithm 2/4. The root sits on the source side of
#                     every arc, so its training representation is isolated.
#
# Accounting always applies the in-expansion Theorem 5.4 dominating pair; this
# script varies orientation only as a utility/sampling ablation.
#
# ONLY DIRECTED GRAPHS ARE INFORMATIVE HERE.  Measured reciprocal-arc fractions:
#
#     flickr            1.0000   undirected -> in and out expansion are IDENTICAL
#     ppi               1.0000   undirected -> identical
#     reddit            (undirected)        -> identical
#     ogbn-arxiv        0.0145   strongly directed
#     relbench rel-f1   0.0000   purely directed (foreign-key arcs)
#
# So the ablation runs on arxiv and RelBench only; adding Flickr/PPI/Reddit would
# just reproduce each run twice.
#
# Read the mean rooted-subgraph size and each orientation's utility above the
# graph-blind MLP baseline. The augmented CSV contains the sole `epsilon` column.
#
# On arxiv, note that capping to K_in=K_out=5 removes most of the 3015-vs-221
# degree asymmetry, so the uncapped ceiling block is the one that discriminates.
# On RelBench the gap is stark: out-expansion leaves row roots with in-degree 0.

set -e
cd "$(dirname "$0")/.."
PY=(/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python -u)
DATASETS=(${=DATASETS:-ogbn-arxiv relbench-f1-top3})
DELTA=${DELTA:-1e-6}

for DS in $DATASETS; do
  source scripts/_dataset_settings.sh $DS
  OUT=results/orientation_$TAG
  echo "\n########## $DS ##########"
  COMMON=(--dataset $DS $INDUCTIVE --T $T --lr $LR_NONDP $REG \
          --seeds $SEEDS --roots_from train)

  echo "=== [0] graph-blind reference (r=0) ==="
  $PY -m src.sparse.run $COMMON $BLIND --p2 1.0 --p1 $P1 \
      --out_dir $OUT/mlp

  for DIR in in out; do
    echo "\n=== [1] uncapped ceiling, direction=$DIR ==="
    $PY -m src.sparse.run $COMMON $MODEL --direction $DIR --p1 $P1 \
        --p2 1.0 --r $CEIL_R --num_layers $CEIL_R --out_dir $OUT/ceiling_$DIR

    echo "=== [2] capped sparsification sweep, direction=$DIR ==="
    $PY -m src.sparse.run $COMMON $MODEL --direction $DIR --p1 $P1 $CAP \
        --p2 $P2_GRID --r $CEIL_R --num_layers $CEIL_R \
        --out_dir $OUT/stage1_$DIR

    echo "=== [3] DP sweep, direction=$DIR ==="
    $PY -m src.sparse.run $COMMON $MODEL --direction $DIR --dp --p1 $P1 $CAP \
        --p2 $P2_GRID --r $CEIL_R --num_layers $CEIL_R \
        --sigma $SIGMA_GRID --clip $CLIP --lr $LR_DP \
        --out_dir $OUT/dp_$DIR

    echo "=== [4] post-hoc epsilon, direction=$DIR ==="
    $PY -m src.sparse.compute_epsilon \
        --csv $OUT/dp_$DIR/sparse_gnn_${TAG}_dp_results.csv --delta $DELTA
  done

  echo "\n=== $DS done: compare $OUT/dp_in vs $OUT/dp_out ==="
done

echo "\n=== orientation ablation complete ==="

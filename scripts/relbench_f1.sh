#!/bin/zsh
# The full ladder on a RelBench entity task (default rel-f1/driver-top3).
#
#   ./scripts/relbench_f1.sh
#   DS="relbench:rel-hm/user-churn" ./scripts/relbench_f1.sh
#
#   nohup caffeinate -i ./scripts/relbench_f1.sh > results/relbench_f1.log 2>&1 &
#
# All parameters come from scripts/_dataset_settings.sh, the same file the other
# ladder scripts source, so nothing here is hardcoded and every run passes
# --num_layers equal to its --r.
#
# RelBench splits are temporal, so the setting is natively inductive: --inductive
# expands training over the graph as of the train cutoff, while evaluation uses
# the full graph so held-out rows keep their real neighbourhoods.  Metric is
# AUROC (driver-top3 is ~17-20% positive, so accuracy is uninformative).
#
# Depth: a root is a prediction row, reaching its entity at r=1 and the entity's
# history at r=2.  Both are swept, and r=1 is the honest "graph barely used" rung.
#
# SCALE CAVEAT.  rel-f1 has 1353 training rows and 92 distinct drivers, so it
# validates the pipeline but its epsilon is not informative.  For DP numbers
# point this at a large task:
#     DS="relbench:rel-stack/user-badge" ./scripts/relbench_f1.sh
#     DS="relbench:rel-hm/user-churn"    ./scripts/relbench_f1.sh
#
# K_OUT CAVEAT.  Under in-expansion the accounting shells are n_d = K_out^d
# (Theorem 6.4, Eq. 44), and an entity feeds one arc per prediction row, so
# K_out decides whether epsilon is finite at all.  Measured on rel-f1 at
# p1=0.05, r=2, sigma=5, T=300:
#     K_out   2      3      5       10      20
#     eps     33.3   78.1   287.3   2185    inf
# Capping K_out keeps every labelled root but strips most roots of their
# history, so it trades utility against epsilon directly.  S1b sweeps it.

set -e
cd "$(dirname "$0")/.."
PY=(/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python -u)
DS=${DS:-relbench-f1-top3}
source scripts/_dataset_settings.sh $DS
OUT=results/relbench_$TAG

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


COMMON=(--dataset $DS $INDUCTIVE --direction in --p1 $P1 --T $T \
        --lr $LR_NONDP $REG --seeds $SEEDS --roots_from train)

echo "=== [S0a] graph-blind baseline (r=0) ==="
done_already $OUT/blind/sparse_gnn_${TAG}_results.csv || \
$PY -m src.sparse.run $COMMON $BLIND --p2 1.0 --out_dir $OUT/blind

echo "=== [S0b] ceiling: all edges, no cap, r=$CEIL_R (L=$CEIL_R) ==="
done_already $OUT/ceiling/sparse_gnn_${TAG}_results.csv || \
$PY -m src.sparse.run $COMMON $MODEL --p2 1.0 --r $CEIL_R \
    --num_layers $CEIL_R --out_dir $OUT/ceiling

for R in $R_VALUES; do
  echo "=== [S1] sparsification sweep, r=$R (L=$R), capped ==="
  done_already $OUT/stage1_r$R/sparse_gnn_${TAG}_results.csv || \
  $PY -m src.sparse.run $COMMON $MODEL $CAP --p2 $P2_GRID --r $R \
      --num_layers $R --out_dir $OUT/stage1_r$R
done

echo "=== [S1b] the K_out utility/privacy knob (no DP yet, r=$CEIL_R) ==="
for KO in 2 3 5 10; do
  done_already $OUT/kout_$KO/sparse_gnn_${TAG}_results.csv || \
  $PY -m src.sparse.run $COMMON $MODEL --K_in 20 --K_out $KO \
      --p2 1.0 --r $CEIL_R --num_layers $CEIL_R --out_dir $OUT/kout_$KO
done

for R in $R_VALUES; do
  echo "=== [S2] DP sigma sweep, r=$R (L=$R) ==="
  done_already $OUT/dp_r$R/sparse_gnn_${TAG}_dp_results.csv || \
  $PY -m src.sparse.run --dataset $DS $INDUCTIVE --direction in --dp \
      $MODEL $CAP $REG --p1 $P1 --p2 $P2_GRID --r $R --num_layers $R \
      --sigma $SIGMA_GRID --clip $CLIP --lr $LR_DP \
      --T $T --seeds $SEEDS --roots_from train --out_dir $OUT/dp_r$R

  echo "=== [S3] post-hoc epsilon (Theorem 6.4), r=$R ==="
  $PY -m src.sparse.compute_epsilon \
      --csv $OUT/dp_r$R/sparse_gnn_${TAG}_dp_results.csv --delta $DELTA
done

echo "=== relbench ladder complete: $OUT ==="

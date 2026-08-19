#!/bin/bash
#SBATCH -Jsparse_rb                       # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                 # 1x H200 GPU
#SBATCH --mem-per-gpu=128G                # rel-hm / rel-stack are multi-GB
#SBATCH -t8:00:00                         # 8 hours walltime
#SBATCH -oReport-sparse-rb-%j.out         # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── SparseGNN on RelBench entity tasks ──
#
# RelBench splits are temporal, so the setting is natively inductive: --inductive
# expands training over the graph as it stood at the train cutoff, while
# evaluation uses the full graph so held-out rows keep their real neighbourhoods.
# Metric is AUROC.
#
# rel-f1 (1353 training rows) is the pipeline check and runs on a laptop via
# scripts/relbench_f1.sh.  This job is for the tasks whose sample size makes
# epsilon meaningful:
#     rel-hm/user-churn      ~1M rows
#     rel-stack/user-badge   ~200k rows
#     rel-avito/user-visits  medium
#
# Depth: a root is a prediction row, reaching its entity at r=1 and the entity's
# history at r=2, so r>=2 with --num_layers 2.
#
# K_out is the parameter that decides whether epsilon is finite: under
# in-expansion the accounting shells are n_d = K_out^d (Theorem 6.4, Eq. 44), and
# an entity feeds one arc per prediction row.  It is swept, not fixed.

cd $SLURM_SUBMIT_DIR

# RelBench caches downloaded databases here; point it at scratch, not $HOME.
export RELBENCH_CACHE_DIR=${SCRATCH:-$HOME/scratch}/relbench_cache
export HF_HOME=${SCRATCH:-$HOME/scratch}/hf_cache

DELTA=1e-6
TASKS=("relbench:rel-stack/user-badge" "relbench:rel-hm/user-churn")

for DS in "${TASKS[@]}"; do
  TAG=$(echo "$DS" | tr '/:' '__')
  OUT=results/relbench_$TAG
  echo "########## $DS ##########"

  echo "=== [S0a] graph-blind MLP baseline ==="
  srun python -u -m src.sparse.run \
      --dataset "$DS" --model mlp --inductive --direction in \
      --r 0 --p2 1.0 --p1 0.01 --T 1000 --lr 0.01 --seeds 3 \
      --roots_from train --out_dir $OUT/mlp

  echo "=== [S0b] GCN ceiling ==="
  srun python -u -m src.sparse.run \
      --dataset "$DS" --model binary_gnn --inductive --direction in \
      --p2 1.0 --r 2 --p1 0.01 --T 1000 --lr 0.01 --seeds 3 --num_layers 2 \
      --roots_from train --out_dir $OUT/ceiling

  echo "=== [S1] sparsification-only sweep ==="
  srun python -u -m src.sparse.run \
      --dataset "$DS" --model binary_gnn --inductive --direction in \
      --K_in 20 --K_out 3 --p2 1.0 0.5 0.25 0.1 --r 2 \
      --p1 0.01 --T 1000 --lr 0.01 --seeds 3 --num_layers 2 \
      --roots_from train --out_dir $OUT/stage1

  echo "=== [S1b] the K_out utility/privacy knob ==="
  for KO in 2 3 5 10; do
    srun python -u -m src.sparse.run \
        --dataset "$DS" --model binary_gnn --inductive --direction in \
        --K_in 20 --K_out $KO --p2 1.0 --r 2 \
        --p1 0.01 --T 1000 --lr 0.01 --seeds 3 --num_layers 2 \
        --roots_from train --out_dir $OUT/kout_$KO
  done

  echo "=== [S2] DP sigma sweep ==="
  srun python -u -m src.sparse.run \
      --dataset "$DS" --model binary_gnn --inductive --direction in --dp \
      --K_in 20 --K_out 3 --p2 1.0 0.5 0.25 0.1 --r 2 \
      --sigma 2.0 5.0 10.0 20.0 --clip 1.0 --lr 1.0 \
      --p1 0.01 --T 1000 --seeds 3 --num_layers 2 \
      --roots_from train --out_dir $OUT/dp

  echo "=== [S3] post-hoc epsilon (Theorem 6.4) ==="
  srun python -u -m src.sparse.compute_epsilon \
      --csv $OUT/dp/sparse_gnn_${TAG}_dp_results.csv --delta $DELTA
done

echo "=== Done ==="

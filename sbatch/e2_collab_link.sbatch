#!/bin/bash
#SBATCH -Je2_collab_lp
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=96G
#SBATCH -t8:00:00
#SBATCH -oReport-e2-collab-lp-%j.out
#SBATCH --mail-type=END,FAIL

# E2 — DP link prediction frontier on ogbl-collab (single-phase Poisson).
# Same spirit as e2_arxiv: bins {4,8,16} × σ grid × q loop (no DP-MLP; link has no baselines).
# If 8h is tight, drop q values or trim SIGMAS.

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbl-collab
echo "OGB data root: $OGB_DATA_ROOT"

SIGMAS="0.5 0.7 1.0 1.5 2.0 4.0"
SEEDS=1
OUT=results/e2_frontier

for Q in 0.05 0.10 0.20; do
  echo "=== ogbl-collab link | q=$Q | Algo 2 / 3 ==="
  srun python run.py \
    --dataset ogbl-collab \
    --task link \
    --algo 2 3 \
    --num-bins 4 8 16 \
    --subsample-prob 0.3 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 300 --patience 15 \
    --tag "e2_collab_link_q${Q//./}" \
    --output-dir "$OUT"
done

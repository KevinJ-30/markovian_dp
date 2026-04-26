#!/bin/bash
#SBATCH -Je2_arxiv
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=96G
#SBATCH -t8:00:00
#SBATCH -oReport-e2-arxiv-%j.out
#SBATCH --mail-type=END,FAIL

# E2 — Privacy/utility frontier on ogbn-arxiv.
# Methods: DP-MLP (= MLP + Algo2 + bins=1), Algo 2, Algo 3 (p=0.3).
# Grid: bins {4,8,16} x sigma {0.5,0.7,1.0,1.5,2.0,4.0} x q {0.03,0.05,0.10,0.20,0.40}.
# Seeds: 3. Accountant: PRV (RDP can be re-derived offline from the rows).
#
# --q is single-valued in run.py, so we loop over q values.
# If 24h is tight, comment out q values (or sigma values inside the srun).

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-arxiv
echo "OGB data root: $OGB_DATA_ROOT"

SIGMAS="0.5 0.7 1.0 1.5 2.0 4.0"
SEEDS=1
OUT=results/e2_frontier

for Q in 0.03 0.05 0.10 0.20 0.40; do
  echo "=== q=$Q : Algo 2 / Algo 3 sweep ==="
  srun python run.py \
    --dataset ogbn-arxiv \
    --task node \
    --algo 2 3 \
    --num-bins 4 8 16 \
    --subsample-prob 0.3 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 300 --patience 15 \
    --tag "e2_arxiv_q${Q//./}" \
    --output-dir "$OUT"

  echo "=== q=$Q : DP-MLP (bins=1) ==="
  srun python run.py \
    --dataset ogbn-arxiv \
    --task node \
    --model mlp \
    --algo 2 \
    --num-bins 1 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 300 --patience 15 \
    --tag "e2_arxiv_dpmlp_q${Q//./}" \
    --output-dir "$OUT"
done

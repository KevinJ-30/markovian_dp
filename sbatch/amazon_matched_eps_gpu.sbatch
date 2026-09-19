#!/bin/bash
#SBATCH -Jamz_gpu
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=180G
#SBATCH -t8:00:00
#SBATCH -oReport-amazon-gpu-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── AmazonProducts on a GPU ──
#
#   EPOCHS=10 BATCH=512 GRID=1e-4 ... sbatch --export=ALL sbatch/amazon_matched_eps_gpu.sbatch
#
# Same grid as amazon_matched_eps.sbatch; src/experiments/sparse.py picks cuda
# automatically when it is visible, so nothing here passes a device flag.
#
# WHAT THE GPU ACTUALLY BUYS.  The DP cells go through the padded batch path
# (PaddedGNNStack + per-sample gradients), which is vectorized and benefits.
# The non-DP ceilings run one forward per root through the unbatched path, so
# they are largely CPU-bound regardless -- do not expect those to speed up.
#
# MEMORY.  180G, not the 128G the CPU script asks for.  128G was already at 90%
# on the CAPPED grid (measured MaxRSS 115.5 GB), and an uncapped ceiling keeps
# all 262.8M arcs instead of the ~39M a K=25 cap leaves -- 6.7x the edges.
# Do NOT add --mem-per-gpu on top of this: Slurm rejects the combination.
#
# UNCAPPED AT r=2 MAY NOT FIT.  Mean degree is ~167, so a 2-hop neighbourhood
# is ~28k nodes per root before the power-law tail. NODP_K=none is safe at r=1;
# at r=2 prefer a generous cap (NODP_K=50) -- which is also closer to what
# GraphSAINT does, since their RW sampler takes num_root=1500 depth=2 rather
# than full neighbourhoods.

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

DS=amazon
NTRAIN=1255968
HIDDEN=${HIDDEN:-512}      # GraphSAINT Appendix C.3 uses 512 for Amazon
DROPOUT=${DROPOUT:-0.1}    # ditto
OUT_ROOT=${OUT_ROOT:-results/amazon_gpu}
TRACK_EVERY=${TRACK_EVERY:-500}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
  echo "WARNING: no GPU visible; this will run on CPU" >&2
$PY -c "import torch; print(f'  torch sees cuda={torch.cuda.is_available()}')"

source scripts/_matched_eps_grid.sh

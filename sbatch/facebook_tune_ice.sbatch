#!/bin/bash
# Facebook DP optimizer/lr tune on ICE, at a fixed decent cell
# (p2=1.0, sigma=5, K=5 -> eps~6): does DP-Adam at ProGAP's lr help over our
# DP-SGD lr=0.3, and was 0.3 leaving accuracy on the table?
#
# Adam is untested in our DP path; it is privacy-sound (Adam post-processes the
# already-clipped+noised gradient, so no accounting change).  matches ProGAP's
# optimizer=adam, lr in {0.01, 0.05}, hidden=16.
#
#   sbatch sbatch/facebook_tune_ice.sbatch

#SBATCH -Jfb_tune
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t4:00:00
#SBATCH -oReport-fb-tune-%j.out
#SBATCH --mail-type=END,FAIL

set -e
cd $SLURM_SUBMIT_DIR

SCRATCH=${SCRATCH:-$HOME/scratch}
PROJECT=${PROJECT:-$SCRATCH/markovian_dp_results}
export FACEBOOK_DATA_ROOT=$SCRATCH/data/facebook100
source $SCRATCH/venvs/markovian/bin/activate
export OMP_NUM_THREADS=1
export TORCH_HOME=$SCRATCH/.torch MPLCONFIGDIR=$SCRATCH/.mpl XDG_CACHE_HOME=$SCRATCH/.cache

OUT_ROOT=$PROJECT/results/facebook_tune
PY="python -u"
DELTA=1e-5
# Fixed cell: p2=1.0, sigma=5, K=5, r=1, hidden=16 (all matched to the earlier
# sparse run and to ProGAP where applicable).  Only optimizer+lr vary.
COMMON="--dataset facebook --direction in --aggr mean --model gnn \
        --p1 0.013 --p2 1.0 --r 1 --num_layers 2 --hidden 16 \
        --dp --sigma 5.0 --clip 1.0 --K_in 5 --K_out 5 \
        --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 3 \
        --T 500 --track_every 25"

run_cell() {   # optimizer lr
  local OPT=$1 LR=$2
  local OUT=$OUT_ROOT/${OPT}_lr${LR}
  if [ -s "$OUT/sparse_gnn_facebook_dp_results.csv" ]; then
    echo "  [skip] $OUT"; return 0
  fi
  echo "=== optimizer=$OPT lr=$LR $(date) ==="
  $PY -m src.sparse.run $COMMON --optimizer $OPT --lr $LR --out_dir $OUT
  $PY -m src.sparse.compute_epsilon \
      --csv $OUT/sparse_gnn_facebook_dp_results.csv --delta $DELTA | tail -2 || true
}

for LR in 0.01 0.05 0.1; do run_cell adam $LR; done
for LR in 0.1 0.3 1.0;  do run_cell sgd  $LR; done

echo "=== summary $(date) ==="
$PY scripts/summarize_sweep.py $OUT_ROOT --metric test_acc || true
echo "=== FB TUNE COMPLETE $(date) ==="

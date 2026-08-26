#!/bin/bash
# PACE-ICE version of facebook_settings.sbatch.
#
# ICE is a SEPARATE cluster from Phoenix: different scheduler config AND a
# different filesystem.  The three #SBATCH lines below and the two paths in the
# body are marked CONFIRM-ON-ICE because I do not know your ICE allocation.
# Find the right values on an ICE login node with:
#
#     sinfo -s                                  # partition names + limits
#     sacctmgr show assoc user=$USER format=account,qos,partition
#     echo $HOME ; ls -d ~/scratch 2>/dev/null  # ICE home / scratch paths
#
# ICE is free and non-preemptible for instructional use, so there is no
# embers/inferno QoS and no billing to worry about — that is the whole reason
# to run here instead of Phoenix embers.

#SBATCH -Jfacebook_dp
#SBATCH -A CONFIRM-ON-ICE                  # your ICE account (sacctmgr, above)
#SBATCH -p CONFIRM-ON-ICE                  # ICE cpu partition (sinfo -s)
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t8:00:00                          # non-preemptible: one arm can be one job
#SBATCH -oReport-facebook-dp-%j.out
#SBATCH --mail-type=END,FAIL
#
#   sbatch --export=ARM=matched sbatch/facebook_settings_ice.sbatch
#   sbatch --export=ARM=sparse  sbatch/facebook_settings_ice.sbatch

set -e
cd $SLURM_SUBMIT_DIR

# CONFIRM-ON-ICE: these paths differ from Phoenix.  ICE storage is typically
# under /storage/ice1/... or your ICE $HOME; set them to wherever you cloned
# the repo and built the venv on ICE.
PROJECT=${PROJECT:-$HOME/markovian_dp_results}
SCRATCH=${SCRATCH:-$HOME/scratch}

export FACEBOOK_DATA_ROOT=$SCRATCH/data/facebook100
source $SCRATCH/venvs/markovian/bin/activate
export OMP_NUM_THREADS=1

OUT_ROOT=$PROJECT/results/facebook
PY="python -u"
DELTA=1e-5
COMMON="--dataset facebook --direction in --aggr mean --model gnn \
        --p1 0.013 --r 1 --num_layers 2 --hidden 16 --clip 1.0 \
        --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 3 \
        --T 500 --track_every 25"

ARM=${ARM:-sparse}

run_cell() {
  local OUT=$1; shift
  if [ -s "$OUT/sparse_gnn_facebook_dp_results.csv" ] || \
     [ -s "$OUT/sparse_gnn_facebook_results.csv" ]; then
    echo "  [skip] $OUT"; return 0
  fi
  $PY -m src.sparse.run $COMMON "$@" --out_dir $OUT
}

if [ "$ARM" = "matched" ]; then
  echo "=== matched cap K=100 (D=100 like ProGAP) ==="
  run_cell $OUT_ROOT/matched_nodp --p2 1.0 --K_in 100 --K_out 100 --lr 0.05
  for SIGMA in 30 50 80; do
    OUT=$OUT_ROOT/matched_s${SIGMA}
    run_cell $OUT --dp --p2 1.0 --K_in 100 --K_out 100 --sigma $SIGMA --lr 0.3
    $PY -m src.sparse.compute_epsilon \
        --csv $OUT/sparse_gnn_facebook_dp_results.csv --delta $DELTA || true
  done
else
  echo "=== sparse K=5 (our operating point) ==="
  for P2 in 1.0 0.1; do
    run_cell $OUT_ROOT/sparse_nodp_p${P2} --p2 $P2 --K_in 5 --K_out 5 --lr 0.05
    for SIGMA in 5 20; do
      OUT=$OUT_ROOT/sparse_p${P2}_s${SIGMA}
      run_cell $OUT --dp --p2 $P2 --K_in 5 --K_out 5 --sigma $SIGMA --lr 0.3
      $PY -m src.sparse.compute_epsilon \
          --csv $OUT/sparse_gnn_facebook_dp_results.csv --delta $DELTA || true
    done
  done
fi

echo "=== FACEBOOK $ARM COMPLETE ==="

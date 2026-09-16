#!/bin/bash
#SBATCH -Jgpu_smoke                       # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                 # 1x H200
#SBATCH --mem-per-gpu=128G                # yelp loads ~2.2 GB raw, more resident
#SBATCH --cpus-per-task=16                # expansion is CPU-side, see PHASE 4
#SBATCH -t1:00:00                         # smoke, not a sweep
#SBATCH -oReport-gpu-smoke-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── Pre-flight smoke for the GPU sweeps ──
#
#   sbatch sbatch/gpu_smoke.sbatch
#
# Answers four questions, cheapest and most load-bearing first, and STOPS at the
# first failure so a broken GPU path cannot be mistaken for a slow one:
#
#   1. does the environment import and does the accountant still reproduce its
#      pinned values?                        (scripts/_ice_env.sh preflight)
#   2. does the CPU test suite still pass on this machine?
#   3. does the DP engine compute the same thing on an H200 as on a CPU?
#      -- the one that decides whether ANY GPU sweep is trustworthy
#   4. how fast is a real cell, and how does that split between the CPU-side
#      expansion and the GPU-side gradient?
#
# WHY THIS EXISTS.  tests/test_vectorized.py pins the vectorized DP path against
# the per-root reference loop, but every one of those tests runs on the CPU;
# there is no CUDA coverage anywhere in tests/.  A GPU sweep whose gradients
# disagree with the loop would report an epsilon describing a mechanism that
# never ran.  PHASE 3 closes that gap.
#
# SBATCH FLAGS.  --gres/--mem-per-gpu copied from sbatch/sparse_inductive.sbatch,
# the proven GPU script in this repo; note it carries neither -A nor -p, unlike
# the ice-cpu scripts which use `-A coc -p ice-cpu`.  If the scheduler rejects
# this, copy the account/partition flags from sbatch/yelp_matched_eps.sbatch.

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

fail() { echo; echo "=== SMOKE FAILED AT: $1 ==="; exit 1; }

echo
echo "############ PHASE 1: environment ############"
nvidia-smi || fail "nvidia-smi (no GPU visible)"
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; \
print('torch sees:', torch.cuda.get_device_name(0))" || fail "torch/CUDA"

echo
echo "############ PHASE 2: CPU test suite ############"
# The full suite is the contract the GPU run has to match.  Fast (~1 min).
$PY -m pytest tests/ -q -x --no-header -p no:cacheprovider || fail "pytest"

echo
echo "############ PHASE 3: CPU-vs-GPU equivalence ############"
# THE load-bearing check.  Compares four arms (cpu/cuda x loop/vectorized)
# trained from identical seeds; sampling and noise are device-independent by
# construction, so only float arithmetic can differ.
$PY scripts/check_device_equivalence.py --steps 60 --nodes 600 --hidden 64 \
    || fail "device equivalence -- DO NOT RUN A GPU SWEEP UNTIL THIS PASSES"

echo
echo "############ PHASE 4: timed end-to-end cell (yelp, 1 epoch) ############"
# One real cell of the real grid at EPOCHS=1, to get a per-step rate to
# extrapolate from and to confirm all four arms train.  EPS_LIST=8 and a single
# CELLS entry keep this to 4 cells: nodp_gnn, nodp_mlp, dpmlp_eps8, gnn_...eps8.
#
# GRID=1e-6 is deliberate and NOT the repo default of 1e-4.  The PLD
# discretization floor is ~T*grid, and T = epochs/p1 now scales with the graph,
# so at yelp's T the 1e-4 default inflates sigma substantially (measured on
# amazon at 10 epochs, eps=2: sigma 6.996 at 1e-4 vs 2.963 at 1e-6).  Calibrating
# the smoke at the coarse default would smoke-test the wrong sigma.
#
# TRACK_EVERY=1050 is one evaluation per epoch (= round(1/p1) for yelp at
# B=512).  Full-graph evaluation over 13.2M arcs is the expensive part of a
# short run, so do not lower it here.
export DS=yelp
export NTRAIN=537635
export OUT_ROOT=${OUT_ROOT:-results/gpu_smoke_yelp}
export HIDDEN=256
export DROPOUT=0.1
export EPOCHS=1
export GRID=1e-6
export TRACK_EVERY=1050
export SEEDS=1
export EPS_LIST="8"
export CELLS="5:0.1:2"

rm -rf "$OUT_ROOT"          # a smoke must not skip cells from a previous smoke
time source scripts/_matched_eps_grid.sh || fail "end-to-end grid"

echo
echo "############ RESULT ############"
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc || true
echo
echo "=== SMOKE PASSED $(date) ==="
echo "Next: check the per-step rate in the PHASE 4 timings before committing"
echo "walltime to the full grid (yelp EPOCHS=20 is 21,001 steps x 18 cells)."

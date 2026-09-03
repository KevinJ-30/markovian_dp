#!/bin/bash
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t8:00:00
#SBATCH -oReport-ppi-meps-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── PPI: DP-GNN vs DP-MLP at MATCHED epsilon ──
#
#   sbatch sbatch/ppi_matched_eps.sbatch
#
# THE QUESTION THIS ANSWERS.  Non-privately the graph is worth +9.4 points on
# PPI (GNN 0.6256 vs graph-blind 0.5314, both at hidden=256/K=5, Stage 1).  But
# at matched epsilon the blind model needs far less noise, because r=0 has no
# expansion and so collapses to a plain subsampled Gaussian.  Measured:
#
#     target eps   blind sigma   GNN sigma (r=1,K=5)   penalty
#              1          7.61                 45.54     5.98x
#              2          4.06                 24.29     5.98x
#              4          2.29                 13.21     5.76x
#              8          1.51                  7.32     4.84x
#
# So the GNN must absorb ~6x the noise to keep a +9.4 point lead.  On facebook a
# 5x sigma increase cost 5-10 points, so this could go either way.  If DP-GNN
# does not beat DP-MLP here, the mechanism does not justify itself -- which is
# why this runs before any wider sweep.
#
# SETTINGS come from Stage 1 (results/ppi_stage1) and GraphSAGE (arXiv 1706.02216):
#   hidden=256  their output dim at every depth; FREE in epsilon (the accountant
#               never sees width, and per-root clipping to C fixes sensitivity),
#               and worth +10 to +14 points over the old hidden=16.
#   K=5         K 5->25 buys +8.7 points but costs 7.2x epsilon; not worth it.
#   r=1         r=2 buys +1.8 points at K=5 and costs 9.4x epsilon.
#   K_in=K_out  keeps cap_mode=auto on the symmetric/undirected path; asking for
#               K_in != K_out silently switches capping algorithms.
#   p1=0.0114   batch 512 / 44,906 train nodes (GraphSAGE's batch size).
#   T=2000      ~23 epochs; their supervised runs used 10.
#
# delta = n^-1.01 with n = 56,944 (full |V|) = 1.57e-5.
#
# grid=1e-5, NOT the 1e-4 default: pessimistic rounding accumulates over
# composition, so the numerical floor is ~T*grid.  At T=2000 that is 0.2 with
# grid=1e-4 -- 20% of an eps=1 target, which inflates the required sigma by ~9%
# and costs accuracy for nothing.  1e-5 puts the floor at 0.02.

set -u
cd $SLURM_SUBMIT_DIR

# Env layout copied from sbatch/facebook_tune_ice.sbatch, the proven ICE setup.
SCRATCH=${SCRATCH:-$HOME/scratch}
source $SCRATCH/venvs/markovian/bin/activate
PY=${PY:-python}
export PPI_DATA_ROOT=$SCRATCH/data/PPI
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCH_HOME=$SCRATCH/.torch MPLCONFIGDIR=$SCRATCH/.mpl \
       XDG_CACHE_HOME=$SCRATCH/.cache

OUT_ROOT=${OUT_ROOT:-results/ppi_matched_eps}
N_NODES=56944
P1=0.0114
T=2000
K=5
HIDDEN=256
SEEDS=${SEEDS:-3}
GRID=1e-5
EPS_LIST="1 2 4 8"
P2_LIST="1.0 0.5 0.25 0.1"

mkdir -p "$OUT_ROOT" results/logs

COMMON=(--dataset ppi --direction in --model multilabel_gnn --aggr mean
        --p1 $P1 --num_layers 2 --hidden $HIDDEN --clip 1.0
        --dropout 0.0 --weight_decay 0.0 --roots_from train
        --seeds $SEEDS --T $T --optimizer sgd --track_every 100)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_ppi*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${COMMON[@]}" "$@" --out_dir "$out"
}

echo "=== PPI matched-epsilon sweep $(date) ==="
echo "    hidden=$HIDDEN K=$K T=$T p1=$P1 seeds=$SEEDS delta=n^-1.01 (n=$N_NODES)"

# ── A. DP-GNN arm: r=1, sigma solved per (p2, eps) ────────────────────────────
echo "--- calibrating GNN arm (r=1) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 $P2_LIST --p1 $P1 --r 1 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma_gnn.txt"
cat "$OUT_ROOT/sigma_gnn.txt"

grep -v '^#' "$OUT_ROOT/sigma_gnn.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/gnn_p2${P2}_eps${EPS}" \
      --p2 "$P2" --r 1 --K_in $K --K_out $K --dp --sigma "$SIGMA" --lr 0.3
done

# ── B. DP-MLP arm: the same mechanism at r=0, so the ONLY difference is the
#      graph.  Its epsilon is far cheaper, hence far less noise at matched eps.
echo "--- calibrating blind arm (r=0) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 $P1 --r 0 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma_blind.txt"
cat "$OUT_ROOT/sigma_blind.txt"

grep -v '^#' "$OUT_ROOT/sigma_blind.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && continue
  run_cell "$OUT_ROOT/blind_eps${EPS}" \
      --p2 1.0 --r 0 --K_in $K --K_out $K --dp --sigma "$SIGMA" --lr 0.3
done

echo
echo "=== PPI matched-epsilon complete $(date) ==="
echo "non-DP references (Stage 1): GNN r=1 0.6256 | blind r=0 0.5314 | trivial 0.4608"
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true

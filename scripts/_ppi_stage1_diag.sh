#!/usr/bin/env bash
# PPI Stage 1: non-DP capacity diagnostic.  Does PPI clear its own trivial
# baseline with room to spare BEFORE any DP noise is added?
#
#   nohup caffeinate -i bash scripts/_ppi_stage1_diag.sh \
#       > results/logs/ppi_stage1.log 2>&1 &
#
# Context.  Every DP result on PPI to date (47 cells, eps 0.65 - 464) sits below
# the trivial micro-F1 baseline of 0.4608, so the DP sweep cannot be the thing
# that is broken.  Reference points, from Hamilton et al. 2017 (GraphSAGE,
# arXiv:1706.02216) Table 1, supervised micro-F1 on PPI:
#
#     Random 0.396 | Raw features 0.422 | SAGE-GCN 0.500
#     SAGE-mean 0.598 | SAGE-LSTM 0.612 | SAGE-pool 0.600
#
# Our non-DP mean-aggregator run is 0.546, i.e. 5 points under SAGE-mean, so the
# setup is close but not matched.  Two candidate gaps, both pulled from that
# paper's appendix:
#
#   hidden  they use output dimension 256 at every depth; we use 16.  Width is
#           FREE in epsilon (the accountant never sees it, and per-root clipping
#           to norm C fixes sensitivity regardless of width), so this is the
#           cheapest lever available.
#   K       they sample S1=25, S2=10 (2-hop budget 250); K=5 gives us 25, and
#           retains only 16% of PPI's edges.  This one does cost epsilon.
#
# Why PPI in particular is fragile: the appendix notes 42% of PPI nodes have no
# non-zero features, "which makes leveraging neighborhood information critical".
# Capping to K=5 removes the only signal those nodes have.
#
# Batch 512 and lr 0.01 also come from that paper (their batch size for all
# methods; the top of their supervised lr sweep).  p1 = 512/44906 = 0.0114.
#
# GATE: hidden=256 with K=25 should land near 0.598.  If it does not, the gap is
# in the mechanism rather than the parameters and no DP budget will rescue PPI.

set -u
cd "$(dirname "$0")/.."

PY=${PY:-/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python}
OUT_ROOT=${OUT_ROOT:-results/ppi_stage1}
SEEDS=${SEEDS:-2}
T=${T:-2000}
P1=${P1:-0.0114}          # 512 roots / 44,906 train nodes

# Non-DP throughout: this stage asks about capacity, not privacy.  adam+lr 0.01
# is the GraphSAGE supervised setting.  dropout/wd 0 per _dataset_settings.sh.
COMMON=(--dataset ppi --direction in --model multilabel_gnn --aggr mean
        --p1 "$P1" --p2 1.0 --num_layers 2 --clip 1.0
        --dropout 0.0 --weight_decay 0.0 --roots_from train
        --seeds "$SEEDS" --T "$T" --optimizer adam --lr 0.01
        --track_every 100)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_ppi*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${COMMON[@]}" "$@" --out_dir "$out"
}

mkdir -p results/logs

echo "=== PPI stage 1 (non-DP capacity diagnostic) $(date) ==="
echo "    p1=$P1 (batch 512)  T=$T  seeds=$SEEDS  trivial baseline = 0.4608"
echo "    targets: SAGE-mean 0.598, raw-features 0.422"
echo

# A. width sweep at the current cap.  Isolates how much of the 5-point gap to
#    SAGE-mean is just model size.  r sweeps inside one invocation.
echo "--- A: hidden sweep, K=5 ---"
for H in 16 64 256; do
  run_cell "$OUT_ROOT/h${H}_K5" --hidden "$H" --K_in 5 --K_out 5 --r 1 2
done

# B. cap sweep at the width that A should show is best.  K_in = K_out keeps
#    cap_mode='auto' on the symmetric/undirected path for PPI; asking for
#    K_in != K_out would silently switch capping algorithms.
echo "--- B: K sweep, hidden=256 ---"
for K in 10 25; do
  run_cell "$OUT_ROOT/h256_K${K}" --hidden 256 --K_in "$K" --K_out "$K" --r 1 2
done

# C. graph-blind reference (r=0).  The non-private analogue of the DP-MLP
#    baseline, and directly comparable to GraphSAGE's "raw features" 0.422.
echo "--- C: graph-blind r=0 reference ---"
run_cell "$OUT_ROOT/blind_h256" --hidden 256 --K_in 5 --K_out 5 --r 0

echo
echo "=== PPI stage 1 complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true

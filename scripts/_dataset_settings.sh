# Shared per-dataset settings, sourced by every ladder script.
#
#   source scripts/_dataset_settings.sh <dataset>
#
# One file so no script special-cases a dataset and every run is reproducible
# from these values alone.
#
# Variables set:
#   MODEL        mechanism flags for the main runs
#   BLIND        flags for the graph-blind baseline (same mechanism at --r 0, so
#                the baseline is measured on the SAME metric as everything else)
#   INDUCTIVE    --inductive, or empty for natively-inductive datasets
#   P1           root-sampling probability, the SAME for DP and non-DP so the DP
#                frontier is readable against its own non-DP ceiling
#   T            training steps
#   CAP          degree-cap flags; K_out is always explicit, because under
#                in-expansion the accounting shells are n_d = K_out^d and so it
#                is K_out that prices epsilon (Theorem 6.4, Eq. 44)
#   R_VALUES     expansion depths to sweep
#   CEIL_R       depth for the uncapped ceiling run
#   REG          --dropout / --weight_decay
#
#
# DEPTH: L = 2, FIXED; r IS THE PRIVACY RADIUS
# --------------------------------------------
# The model is a two-layer GNN everywhere (the paper's base mechanism).  The
# earlier L = r rule forced a one-layer model at r=1, which on PPI sits below
# even the graph-blind two-layer baseline — the rung measured model capacity,
# not sparsification.  Privacy is priced by the EXPANSION depth r alone
# (the accountant sees only the sampling process), so with L fixed at 2:
#
#     r=2 L=2  EXACT    rooted computation = full-graph inference; eps ~ K_out^2
#     r=1 L=2  TRUNC    GraphSAGE-style truncated receptive field in training
#                       (measured eval mismatch on capped arxiv: 0.6% mean,
#                       1.9% max); eps ~ K_out^1 — the cheap-privacy rung.
#
# We sweep r in {1,2}; r=3 would cost K_out^3 in epsilon.
#
#
# REGULARIZATION
# --------------
# run.py's defaults (dropout 0.5, weight_decay 5e-4) are the Planetoid settings
# and they cost PPI 7 points of micro-F1 (0.476 vs 0.546).  With dropout=0 and no
# decay the per-root engine reproduces a full-batch GNN trained on the same
# capped graph to four decimals (0.5463 vs 0.5464) — the check that the engine
# itself is sound.  Applied uniformly so no cross-dataset comparison is
# confounded by a regularization difference.

_ds=$1

REG=(--dropout 0.0 --weight_decay 0.0)
R_VALUES=(1 2)
L=2          # GNN depth, fixed (see DEPTH note above); r sweeps independently
CEIL_R=2

case $_ds in
  ppi)
    # 24 disjoint graphs split 20/2/2 -> natively inductive, --inductive is a
    # no-op.  121-way multilabel, so BCE + micro-F1; plain --model gnn crashes
    # here ("shape '[1]' is invalid for input of size 121").
    # T=2000: the measured learning curve plateaus by step ~1000 (0.4756 at 1k,
    # 0.4712 at 6k), so this is ample.
    MODEL=(--model multilabel_gnn --aggr mean)
    BLIND=(--model multilabel_gnn --aggr mean --r 0)
    INDUCTIVE=()
    P1=0.01; T=2000
    CAP=(--K_in 5 --K_out 5)
    ;;
  relbench*)
    # Temporal splits -> natively inductive, but --inductive still selects the
    # loader's train-cutoff graph.  Binary and imbalanced, so AUROC.  A root is a
    # prediction row: r=1 reaches only its entity, r=2 reaches its history.
    # p1=0.05 (68 of 1353 train rows per step) with T=900 keeps total epochs
    # comparable to the earlier p1=0.2/T=300 while keeping epsilon affordable.
    MODEL=(--model binary_gnn --aggr mean)
    BLIND=(--model binary_gnn --aggr mean --r 0)
    INDUCTIVE=(--inductive)
    P1=0.05; T=900
    CAP=(--K_in 20 --K_out 3)
    ;;
  reddit)
    MODEL=(--aggr mean); BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.002; T=500
    CAP=(--K_in 5 --K_out 5)
    ;;
  *)
    # ogbn-arxiv, flickr, and any other single-label transductive graph converted
    # to inductive via the train-induced subgraph.
    MODEL=(--aggr mean); BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.005; T=500
    CAP=(--K_in 5 --K_out 5)
    ;;
esac

# Shared sweep grids, identical across every ladder script.
P2_GRID=(1.0 0.5 0.25 0.1)
SIGMA_GRID=(2.0 5.0 10.0 20.0)
SEEDS=3
CLIP=1.0
LR_NONDP=0.01      # Adam
LR_DP=1.0          # SGD; the DP path divides the noisy sum by the expected batch
DELTA=${DELTA:-1e-6}

# relbench:<db>/<task> contains characters that are not filename-safe.
TAG=$(echo $_ds | tr '/:' '__')

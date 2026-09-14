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
#   EPOCHS       passes over the training roots (the schedule is specified HERE)
#   T            training steps, DERIVED as round(EPOCHS/P1) at the bottom
#   CAP          degree-cap flags; K_out is always explicit, because under
#                in-expansion the accounting shells are n_d = K_out^d and so it
#                is K_out that prices epsilon (Theorem 6.4, Eq. 44)
#   R_VALUES     expansion depths to sweep
#   CEIL_R       depth for the uncapped ceiling run
#   REG          --dropout / --weight_decay
#
#
# DEPTH: L FIXED AT 2, r SWEPT INDEPENDENTLY
# ------------------------------------------
# r and L are separate knobs and are NOT tied together.  r is the expansion
# depth and prices epsilon as K_out^r; L is the model depth and does not enter
# the accounting at all, so depth is FREE in epsilon.  Tying them throws that
# away: at r=1 it would force a one-layer model, which measures capacity rather
# than sparsification.
#
# r=2 IS REQUIRED for the graph to be worth anything.  Measured non-privately on
# PPI-large at K=25 over 34 epochs, against a graph-blind MLP at 0.5330 micro-F1:
#     r=1 L=1   0.4542   -- 8 points BELOW blind, and flat in T
#     r=2 L=2   0.8227   -- 29 points above, crossing over by ~3 epochs
# One hop carries less than the node's own features on these graphs.
#
# r=2 is also what makes sparsification worth something: sigma for eps=8 at
# T=3000 on PPI-large K=5 is 102.25 at p2=1.0 against 8.20 at p2=0.1, a 12.5x
# saving, where at r=1 the same move bought 1.3-2x.  r=3 would cost K_out^3.
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
R_VALUES=(2)   # r=1 does not beat the graph-blind arm; see DEPTH above
L=2            # GNN depth, fixed and INDEPENDENT of r (free in epsilon)
CEIL_R=2

case $_ds in
  # ── GraphSAINT suite ────────────────────────────────────────────────────
  # Architecture/regularization follow GraphSAINT Table 5 and Appendix C.3
  # (arXiv:1907.04931) so we are not inventing our own settings: hidden 512
  # for PPI/Yelp/Amazon and 128 for Reddit; dropout 0.0 on PPI and 0.1 on the
  # other three.  Their lr=0.01 is for ADAM; ours is SGD, see LR_DP below.
  #
  # p1 = B/N_train at a FIXED batch B=512 (GraphSAGE's default batch size), so
  # the sampling rate falls as the graph grows and epsilon falls with it.  At
  # T=300, p2=0.1, r=2, K=5 the noise needed for eps=2 / eps=8 is:
  #     PPI-large  p1=0.011402   5.25 / 1.83
  #     Reddit     p1=0.003326   1.92 / 1.25
  #     Yelp       p1=0.000952   1.42 / 0.99
  #     Amazon     p1=0.000408   1.23 / 0.87
  # sigma < 1 on Amazon at eps=8 means noise below the clipping norm.
  ppi-large)
    MODEL=(--model multilabel_gnn --aggr mean)
    BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.011402; EPOCHS=${EPOCHS:-10}
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=512; DROPOUT=0.0
    ;;
  saint-reddit)
    MODEL=(--aggr mean); BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.003326; EPOCHS=${EPOCHS:-10}
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=128; DROPOUT=0.1
    ;;
  yelp)
    MODEL=(--model multilabel_gnn --aggr mean)
    BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.000952; EPOCHS=${EPOCHS:-10}
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=512; DROPOUT=0.1
    ;;
  amazon)
    MODEL=(--model multilabel_gnn --aggr mean)
    BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.000408; EPOCHS=${EPOCHS:-10}
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=512; DROPOUT=0.1
    ;;
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
  facebook)
    # FB100 UIllinois20, the GAP/ProGAP comparison graph.  Transductive: it is a
    # single social graph with no natural inductive split, so --inductive is left
    # off and eval_graph=auto scores on the training graph.  p1 and lr come from
    # the tuning sweep in sbatch/facebook_tune_ice.sbatch, not the shared
    # defaults.
    MODEL=(--aggr mean); BLIND=(--model mlp --r 0)
    INDUCTIVE=()
    P1=0.013; T=500
    CAP=(--K_in 5 --K_out 5)
    LR_DP=0.3
    ;;
  reddit)
    MODEL=(--aggr mean); BLIND=(--model mlp --r 0)
    INDUCTIVE=(--inductive)
    P1=0.002; T=500
    CAP=(--K_in 5 --K_out 5)
    LR_DP=0.3
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
HIDDEN=${HIDDEN:-256}
DROPOUT=${DROPOUT:-0.0}

# OPTIMIZER AND LEARNING RATE
# ---------------------------
# ADAM everywhere, DP and non-DP alike, at lr = 0.01 -- ONE value for every
# dataset and both arms.  Three reasons for Adam over SGD:
#   1. Every baseline is Adam (DPAR is DPAdamGaussianOptimizer upstream, ProGAP
#      defaults to it, HeterPoisson uses it), so this is the same-optimizer
#      comparison.
#   2. It lets us take GraphSAINT's published config as a package: their
#      lr=0.01 and per-dataset dropout were grid-searched together under Adam,
#      so borrowing the dropout while running SGD would be incoherent.
#   3. Under DP it is free -- the optimizer is post-processing of the noised
#      gradient, so the accountant never sees it.
#
# lr=0.01 is both GraphSAINT's published value AND our measured DP optimum.
# Swept on PPI-large (p2=0.1, r=2, K=5, T=300, sigma=5.25, eps~2, 1 seed):
#     Adam  0.001 -> 0.4018   0.003 -> 0.4038   0.01 -> 0.4129   0.03 -> 0.4082
#     SGD   0.1   -> 0.4017   0.3   -> 0.4031   1.0  -> 0.4141   3.0  -> 0.4101
# i.e. a tie between the two optimizers at their respective optima, so nothing
# is lost by preferring the one the baselines use.
#
# Adam also removes a scale problem SGD has here: without clipping, the non-DP
# gradient scale is set by the loss, and the multilabel heads average over C
# labels (121 on PPI) making their gradients ~C times smaller than a
# single-label head's.  Under SGD that forced a different lr per task type
# (non-DP PPI was still climbing at lr=20); Adam normalizes it away.
#
# For the SGD ablation, use --optimizer sgd with lr 1.0 (DP).
LR_DP=${LR_DP:-0.01}
LR_NONDP=${LR_NONDP:-0.01}

# STEP COUNT: SET EPOCHS, DERIVE T
# -------------------------------
# T is NOT a free constant -- it is round(EPOCHS/P1), applied at the bottom of
# this file.  A fixed T is a different amount of data on every dataset, because
# P1 = B/N_train shrinks as the graph grows: at the T=300 this file used to
# hardcode, PPI-large saw 3.4 passes over its training nodes and Amazon 0.12.
# Comparing those two arms compares convergence, not privacy-utility.
#
# EPOCHS=10 is the default.  It is affordable everywhere -- measured sigma for
# eps=8, delta=1e-6, p2=0.1, r=2, K=5, B=512, at EQUAL epochs:
#
#     epochs   ppi-large   reddit   yelp   amazon
#          1        1.87     1.43   1.17     1.05
#          5        3.10     1.87   1.37     1.21
#         10        4.36     2.47   1.61     1.37
#         20        6.20     3.48   2.17     1.84
#
# and 10 epochs clears the point where the graph starts paying: the non-private
# PPI-large curve crosses the graph-blind MLP at ~3 epochs.  Note the bigger
# graph is CHEAPER at equal epochs (more data, same batch), so this budget is
# set by the smallest dataset, not the largest.
#
# Raising it is not free in wall-clock: T scales with N_train, so EPOCHS=10 is
# 877 steps on PPI-large and 24,510 on Amazon.
#
# The NON-DP ceiling is not privacy-constrained and should NOT be matched to
# this budget -- run it to convergence and say so.
DELTA=${DELTA:-1e-6}

# T = EPOCHS/P1: one Poisson step draws P1*N_train roots, so 1/P1 steps is one
# pass over the pool.  Scripts that want an explicit step count can still set T
# in the environment, which wins.  src/sparse/run.py --epochs does the same
# conversion internally; this is here because calibrate_grid.py solves sigma out
# of process and has to be handed the identical T.
EPOCHS=${EPOCHS:-10}
T=${T:-$(python3 -c "print(max(1, round($EPOCHS/$P1)))")}

# relbench:<db>/<task> contains characters that are not filename-safe.
TAG=$(echo $_ds | tr '/:' '__')

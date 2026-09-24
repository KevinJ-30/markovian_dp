# Shared per-dataset settings, sourced by every ladder script.
#
#   source scripts/_dataset_settings.sh <dataset>
#
# One file so no script special-cases a dataset and every run is reproducible
# from these values alone.
#
# Variables set:
#   MODEL        mechanism flags for the main runs
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
# DEPTH: L FIXED AT 2, r SWEPT INDEPENDENTLY
# ------------------------------------------
# r and L are separate knobs and are NOT tied together.  r is the expansion
# depth and prices epsilon as K_out^r; L is the model depth and does not enter
# the accounting at all, so depth is FREE in epsilon.  Tying them throws that
# away: at r=1 it would force a one-layer model, which measures capacity rather
# than sparsification.
#
# r=2 is required for the graph to carry useful signal in the measured
# large-graph runs. At K=25 over 34 non-private epochs, r=1/L=1 reached 0.4542
# micro-F1 and stayed flat in T, while r=2/L=2 reached 0.8227.
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
R_VALUES=(2)
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
    P1=0.011402; T=300
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=512; DROPOUT=0.0
    ;;
  saint-reddit)
    MODEL=(--aggr mean)
    P1=0.003326; T=300
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=128; DROPOUT=0.1
    ;;
  saint-yelp)
    MODEL=(--model multilabel_gnn --aggr mean)
    P1=0.000952; T=300
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=512; DROPOUT=0.1
    ;;
  saint-amazon)
    MODEL=(--model multilabel_gnn --aggr mean)
    P1=0.000408; T=300
    CAP=(--K_in 5 --K_out 5)
    HIDDEN=512; DROPOUT=0.1
    ;;
  twitch-explicit)
    # Domain roles come from the loader's benchmark defaults unless the caller
    # adds explicit --*_domains flags. Twitch is binary and reports AUROC.
    MODEL=(--model binary_gnn --aggr mean)
    P1=0.005; T=500
    CAP=(--K_in 5 --K_out 5)
    ;;
  facebook100)
    # The 18-school benchmark is distinct from facebook/UIllinois20. Domain
    # defaults are resolved by the loader; this is a categorical accuracy task.
    MODEL=(--model gnn --aggr mean)
    P1=0.005; T=500
    CAP=(--K_in 5 --K_out 5)
    ;;
  mag-countries)
    # Class 19 remains in the categorical loss and is filtered only from the
    # reported metric. The loader owns the default US -> shared CN protocol.
    MODEL=(--model gnn --aggr mean)
    P1=0.005; T=500
    CAP=(--K_in 5 --K_out 5)
    ;;
  facebook)
    # FB100 UIllinois20 has no native inductive split. The loader's masks define
    # a train-induced graph, matching the SparseGNN training contract. p1 and lr
    # come from the tuning sweep in sbatch/facebook_tune_ice.sbatch.
    MODEL=(--aggr mean)
    P1=0.013; T=500
    CAP=(--K_in 5 --K_out 5)
    LR_DP=0.3
    ;;
  reddit)
    MODEL=(--aggr mean)
    P1=0.002; T=500
    CAP=(--K_in 5 --K_out 5)
    LR_DP=0.3
    ;;
  *)
    # Single-graph datasets use the train-induced graph defined by their masks.
    MODEL=(--aggr mean)
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

# STEP COUNT
# ----------
# T=300 for the DP arms.  sigma grows as sqrt(T), so the cost is steep and the
# old T=2000 was buying very little: on PPI-large at p2=0.1, r=2, eps=2 the
# required sigma is 5.25 at T=300 against 14.20 at T=2000, a 2.7x noise
# penalty for 6.7x the steps.  Measured sigma for eps=2 by T (p2=0.1):
#     T=100  3.11    T=300  5.25    T=1000   9.77
#     T=200  4.29    T=500  6.82    T=2000  14.20
#
# The NON-DP ceiling is not privacy-constrained and should NOT be matched to
# T=300: at batch 512 that is only 3.4 epochs on PPI-large and nowhere near
# converged.  Run the ceiling to convergence and say so.
DELTA=${DELTA:-1e-6}

# Dataset namespaces can contain characters that are not filename-safe.
TAG=$(echo $_ds | tr '/:' '__')

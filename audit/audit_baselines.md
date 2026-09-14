# Baseline audit — `markovian_dp` (branch `sparse_expand_clean`), 2026-09-11

Scope: **baselines only**, and whether the comparison against SparseExpand is apples-to-apples.
`src/sparse/` was read only to establish what OUR runs do.

---

# (A) What the code does

## A.1 Headline finding, stated up front

There are **two disjoint universes of "baseline" in this repo**, and they never meet:

1. **The external-baseline harness** (`src/experiments/` + `third_party/` + `configs/*.json`).
   Four external methods are wired. **None has ever been run.** `results/` contains 420 CSVs
   and **zero `.json` files**; `src/experiments/run.py:135-139` writes only JSON, into
   `results/inductive/<dataset>/<method>.json`, and that directory does not exist.
   No sbatch job, no `scripts/*` driver, and no README section references any of them
   (`grep -rn "dpar|progap|heterpoisson|dp_gnn|graphsage" sbatch/ scripts/ README.md` → no hits).

2. **The "blind arm"** — the *same* SparseExpand mechanism run at `--r 0`
   (`sbatch/ppi_matched_eps.sbatch:100-102`, `sbatch/arxiv_inductive_matched_eps.sbatch:106-110`,
   `sbatch/relbench_regression_meps.sbatch:129-133`). This is the *only* baseline with real
   numbers, and it is what `paper/experiments_current.tex:40` means by "Blind arm = same
   mechanism at r=0, not a separate MLP" and what the row literally labelled `baselines`
   in `paper/experiments_current.tex:77,146` reports (verified: 92.46/92.98/93.70/94.43 on
   rel-amazon/item-ltv exactly reproduces `results/relbench/relamazon_itemltv_meps/blind_eps*`
   best-on-val test MAE).

So the "apples-to-apples" question mostly reduces to *"is the r=0 blind arm a fair floor?"* —
and the answer (§B-1, §B-2) is **no, in two ways that both favour our method**.

## A.2 Baseline inventory

| # | Name | Paper / origin | Privacy notion claimed | Code location | Vendored or reimplemented | Wired into a runnable driver? | Ever run? |
|---|------|----------------|------------------------|---------------|---------------------------|-------------------------------|-----------|
| 1 | **`mlp`** | none (plain 2-layer MLP) | none (non-private) | `src/experiments/baselines.py:35-48,82-148` | first-party | yes — `src/experiments/run.py:93-95`, `configs/initial_mlp.json`, `configs/ppi_native_smoke.json` | no results |
| 2 | **`graphsage`** | Hamilton et al. 2017 (mean-aggregation reimpl., no PyG) | none (non-private) | `src/experiments/baselines.py:51-79` | reimplemented | yes — `run.py:93-95`; only listed in `configs/inductive_comparison.json` (unused file) | no results |
| 3 | **`dp_mlp`** | standard DP-SGD on a graph-blind MLP | Poisson-subsampled Gaussian, **node add/remove**, sensitivity `C` | `baselines.py:118-120,150-174`; accountant `src/experiments/privacy.py:82-94` → `src/sparse/accounting.py:605-622` (Opacus PRV) | first-party | yes — `run.py:93-95` | no results |
| 4 | **`dpar`** | DPAR (WWW'24), decoupled DP-APPR + DP-SGD | DP-PPR Gaussian + DP-SGD, **node add/remove**, then multiplied by a "privacy amplification sampling rate" | `src/experiments/dpar.py` (whole file); accountants `privacy.py:139-234` | **reimplemented in PyTorch**; only the RDP accountant is vendored (`privacy.py:203` loads `third_party/DPAR/dpgnn/privacy_utils/rdp_accountant.py` by file path) | yes — `run.py:91-92`; `configs/{initial_dpar,initial_dpar_pubmed,cora_ml_dpar_eps8,cora_ml_dpar_smoke,ppi_dpar_smoke}.json` | no results |
| 5 | **`dp_gnn`** | Daigavane et al. 2021, *Node-Level DP GNNs* (Google Research) | node add/remove with bounded degree `K`; sensitivity `2(K+1)C`; multi-term hypergeometric RDP | `src/experiments/dpgnn.py` + `dpgnn_adapter.py` | **reimplemented** ("first-party", `dpgnn.py:1-8`, `dpgnn_adapter.py:1-6`); accountant is a line-by-line port of ProGAP's copy of Google's (`dpgnn.py:69-100` ≡ `third_party/ProGAP/core/privacy/algorithms/gnn_sgd.py:8-64`) | yes — `run.py:96-103`; `configs/{cora_ml_dp_gnn_smoke,initial_dp_gnn}.json` | no results |
| 6 | **`progap`** | ProGAP (WSDM'24), Sajadmanesh & Gatica-Perez | node-level: NAP Gaussian (sens `sqrt(K)`) ⊕ Poisson-subsampled DP-SGD, composed in autodp; **add/remove** | `third_party/ProGAP/` + `third_party/ProGAP/inductive_adapter.py`; bridged by `src/experiments/upstream.py:115-185` | **vendored** (rev `3ccad59e`, `upstream.py:23-27`) | wired but **BROKEN** — see §B-4 | no results |
| 7 | **`heterpoisson`** | PNPiGNNs / *Preserving Node-Level Privacy in GNNs* (Xiang et al.) | node-level with heterogeneous Poisson subsampling; custom numerical RDP over a degree-dependent sensitivity mixture; **add/remove** | `third_party/PNPiGNNs/.../` + `inductive_adapter.py`; bridged by `upstream.py:115-185` | **vendored** (rev `9a06332`, `upstream.py:29-33`) | wired; configs point at another user's venv (§B-5) | no results |
| 8 | **`--model mlp` (`MLPMechanism`)** | ours — graph-blind `g0` inside the SparseExpand engine | our Theorem (node **substitution**) at `r=0` | `src/sparse/mlp_mechanism.py` | first-party | yes, but only via `scripts/_dataset_settings.sh:87,94,103` (facebook/reddit/arxiv ladders) — **not** used by any matched-ε sbatch | some non-DP ladder runs (`results/arxiv/inductive_mlp`, `results/ppi/inductive_blind_ppi`) |
| 9 | **"blind arm" = `gnn`/`multilabel_gnn`/`binary_gnn`/`regression_gnn` at `--r 0`** | ours | our Theorem (node substitution) | `src/sparse/run.py` with `--r 0` | first-party | yes — all three matched-ε sbatch files | **YES — the only baseline with results** |
| 10 | **`trivial_baseline`** | label-only predictor (majority class / all-positive / 0.5 AUROC / train-mean MAE) | n/a | `src/sparse/run.py:90-118` | first-party | yes, written into every CSV column `trivial_baseline` | yes |

## A.3 Privacy-notion compatibility (the critical question)

**Ours.** `src/sparse/accounting.py:238-246` builds `P = Σ π_k N(-2k, σ²)`, `Q = Σ π_k N(+2k, σ²)`.
The `±2k` means **node substitution** (sensitivity `2C`). `accounting.py:1-25` and
`sparsegnn_theorem_label` (`accounting.py:437-438`) confirm: `thm6.4-substitution` for
`--direction in`. Substitution is the *strictest* of the three relations.

At `r=0` (the blind arm), `shell_sizes(0,...)=[1]`, `q=[1]`, so `π = Bernoulli(p1)` and the pair
is `(1-p1)N(0)+p1 N(-2)` vs `(1-p1)N(0)+p1 N(+2)` — i.e. a plain Poisson-subsampled Gaussian
**at sensitivity 2C**. That is the same accountant, same δ, same theorem as the GNN arm.
**The blind-arm ε is genuinely on our ε axis. Good.**

Every external baseline is on a **different** axis:

| Baseline | Neighbouring relation implied by its code | Sensitivity used | Accountant | Composition |
|---|---|---|---|---|
| ours (`r≥0`) | **node substitution** | `2C` (`accounting.py:243-245`) | our dominating-pair PLD → Google `dp_accounting.pld` (`accounting.py:188,232-235`) | PLD, `symmetric=True` |
| `dp_mlp` | node **add/remove** | `C` | **Opacus PRV** (`privacy.py:89`, `accounting.py:614-622`) | PRV (RDP fallback) |
| `dpar` (PPR part) | node add/remove, advanced-composition over `ppr_releases`, **then multiplied by `amplification_rate`** | `ppr_clip` | hand-rolled inverse of DPAR's printed formula (`privacy.py:147-196`) | `ε/(2√(m log(e+ε/δ)))` inverted, ×`q` |
| `dpar` (SGD part) | node add/remove (Poisson-subsampled Gaussian), **then ×`amplification_rate`** | `sgd_clip` | **vendored TF-Privacy RDP** (`privacy.py:203-214`) | RDP, orders `1.1…9.9, 12…63` |
| `dp_gnn` | node add/remove, bounded degree `K` | `2(K+1)·threshold` (`dpgnn.py:65-66,202-206`) | `dp_accounting.rdp.RdpAccountant` + hypergeometric amplification (`dpgnn.py:79-100`) | RDP, orders `1.1…9.9` |
| `progap` | node add/remove | NAP `√K`; DP-SGD `C` | **autodp** `ComposeGaussian`/`Composition` (`core/privacy/mechanisms/composed.py:24-28`) | autodp analytic Gaussian composition |
| `heterpoisson` | node add/remove, sensitivity mixture `[0, ½, 1, 2, …, D_out]` (`privacy/mix.py:228-233`) | `C` after a `2×` grad-norm inflation (`train_scheduler.py:262`) | **custom numerical RDP** over a 1e6-point grid (`mix.py:159-206`) | RDP → `eps_from_delta_rdp` (Balle Thm 21) |

**Is any conversion applied? No.** There is no code anywhere that converts between
substitution and add/remove, and no code anywhere that puts a baseline ε next to ours.
`scripts/summarize_sweep.py` reads only our CSVs; `scripts/plot_*.py` and
`scripts/_meeting_figures_20260903.py` load only `sparse_gnn_*_results.csv`;
`src/experiments/run.py` writes JSON that nothing reads. **The comparison point does not exist
yet** — which is good news (nothing is currently wrong in print) and bad news (the moment
anyone builds that table, the axes are incompatible by construction).

Extra wrinkle: `naive_opacus_epsilon`'s own docstring (`accounting.py:609-612`) says
*"it is NOT a valid node-level guarantee; it is a floor showing the price of graph structure"* —
yet `DPMLPAccountant.account` (`privacy.py:85-94`) uses exactly that function to report
`dp_mlp`'s privacy in `baselines.py:137-141`.

## A.4 Hyperparameter fairness table

"tuned?" = evidence in `results/` or a sweep script that the value was searched.

| | **ours (matched-ε)** | `mlp` / `graphsage` | `dp_mlp` | `dpar` | `dp_gnn` | `progap` | `heterpoisson` |
|---|---|---|---|---|---|---|---|
| optimizer | SGD (`--optimizer sgd`, sbatch/ppi_matched_eps.sbatch:71); Adam for non-DP refs | **Adam** (baselines.py:110) | **Adam** (same) | **Adam** (dpar.py:302) | **Adam** (dpgnn.py:220) | Adam (progap/base.py:29) | Adam (inductive_adapter.py:131) |
| lr | **0.3** DP / 0.01 non-DP (ppi_matched_eps.sbatch:101,115) | 1e-2 (baselines.py:23) | 1e-2 | 5e-3 (dpar.py:49) | **3e-3, not configurable** (dpgnn.py:34) | 0.01 (progap/base.py:30) | 1e-3 from config (`configs/initial_heterpoisson.json`) |
| weight decay | **0.0** (sbatch:70) | 5e-4 (baselines.py:24) | 5e-4 | 1e-4 (dpar.py:50) | 0 (none) | 0.0 (progap/base.py:31) | 0 (none) |
| hidden dim | **256** (sbatch:60) | 64 (baselines.py:21) | 64 | 32 (dpar.py:46) | **100, not configurable** (dpgnn.py:35) | **16** (progap/base.py:21) | **128, hardcoded** (inductive_adapter.py:128) |
| layers | 2 (sbatch:69) | 2 (baselines.py:22) | 2 | 2 (dpar.py:47) | fixed 3-linear (dpgnn.py:42-46) | `depth=2` + 1 base + 1 head | `K=1` (one conv!) |
| dropout | **0.0** (sbatch:70) | **0.5** (baselines.py:22) | 0.5 | 0.1 (dpar.py:48) | none | 0.0 | none |
| epochs / steps | `T=2000` PPI / `500` arxiv, batch ≈512 → ~23 epochs | 100 **full-batch steps** (baselines.py:116-125) | 100 × ⌈n/256⌉ steps | 100 epochs × ⌈70/60⌉ | `steps` from config (1 in smoke, 800 in `initial_dp_gnn.json`) | `PROGAP_EPOCHS=100` per stage × (depth+1) stages | `epochs=100` (upstream uses 4–9) |
| batch size | expected `p1·|pool|` = 512 (PPI/arxiv), Poisson | 256 | 256, Poisson (baselines.py:153-154) | 60 default / 70 in config (shuffled, **not Poisson**) | 32–256, **sampled with replacement** (dpgnn.py:227) | 32 (upstream uses 256–4096) | 32 (upstream uses 2048–4096) |
| clip `C` | 1.0, swept `{0.1,0.2,0.5,1.0}` (`results/ppi/ppi_clip`) | — | 1.0 (baselines.py:28) | ppr 0.01 / sgd 1.0 (dpar.py:33,38) | **adaptive, 75th-pct, non-private** (dpgnn.py:36,173-185) | 1.0 (progap/node.py:26) | 1.0 from config |
| σ | **solved per (p2, ε)** by `scripts/calibrate_grid.py` | — | fixed `1.0` (baselines.py:27); **no target-ε path** | solved by `calibrate_dpar_noise` when `target_epsilon` set (dpar.py:309-329) | fixed from config; **no target-ε path** | solved by autodp `calibrate` (progap/node.py:72-75) | solved by bisection (`accounting_analysis.py:441-463`) |
| early stop / model select | none in `run.py` (final model at `T`); `summarize_sweep.py:103-109` picks best-on-val post hoc | **best-val over 100 epochs** (baselines.py:126-132) | same | **best-val over 100 epochs** (dpar.py:348-354) | none (final model) | **best-val per stage** (`core/trainer/trainer.py:103-121`) — but "val" = the train partition (§B-6) | best-val over epochs (`train_scheduler.py:137-147`) |
| δ | `n^-1.01` (calibrate_grid.py:64) → 1.57e-5 PPI, 5.24e-6 arxiv | — | **1e-5** (baselines.py:29) | **5e-4** in configs, split ε/2, δ/2 (privacy.py:276-277) | **1/(10·n_train)**, hardcoded (dpgnn.py:231) | 5e-4 from config | 5e-4 from config |
| seeds | 3 (`--seeds 3`) | 1 (config `seed: 0`) | 1 | 1 | 1 | 1 (upstream paper: 10 repeats) | 1 (upstream: 3–5) |
| tuned per method? | **heavily** — `results/{facebook_tune, ppi_lr_sweep, ppi_optmatch, ppi_tuned, ppi_clip, ppi_clip_lr, ppi_k, ppi_batch, ppi_stage1, facebook_width}` ≈ 60 tuning cells | **no** — `configs/initial_mlp.json` copies the dataclass defaults verbatim | no | **no** — every value in `DPARConfig` is upstream's TF flag default (`third_party/DPAR/main.py:34,47-57,69-75`) | **no, and mostly not even exposed** | **no** — and the config *contradicts* the paper's own grid | **no** — and the config contradicts upstream's `run_*.sh` |

## A.5 Data pipeline: what the adapters translate, and what silently differs

Both universes call the same loader (`src/experiments/run.py:74` and `src/sparse/run.py:356-360`
→ `src.datasets.load_dataset`). After that they diverge completely:

| | ours (`src/sparse/run.py`) | external harness (`src/experiments/`) |
|---|---|---|
| split | the dataset's **native** masks (PPI 20/2/2 disjoint graphs, OGB temporal, RelBench temporal). `--common_inductive_split` exists but no sbatch uses it | a **fresh deterministic 60/20/20 stratified split** (`inductive.py:77-106`), saved under `data/inductive_splits/` |
| training graph | full graph; `--inductive` keeps only train–train arcs (`run.py:417-422`) | three **induced** partitions; every cross-partition edge is deleted in all three (`inductive.py:169-186`) |
| **eval graph** | `--eval_graph auto` → full `data.edge_index` (inductive) or the capped training graph (transductive); the *other* graph is also recorded in `*_alt` columns | the **val/test induced subgraph only** — a held-out node keeps only its edges to other held-out nodes |
| degree cap | `K_in=K_out=5` (PPI/arxiv), `22/6` (rel-amazon), applied to train **and** eval graphs (`run.py:474-481`) | **no `K` cap**; each baseline does its own bounding (DPAR top-k=16 PPR, DP-GNN Bernoulli `max_degree=5`, ProGAP `BoundOutDegree(5)` on train only, HeterPoisson `num_neighbors=1`) |
| feature preprocessing | none | HeterPoisson **z-scores features with train statistics** (`third_party/PNPiGNNs/.../inductive_adapter.py:47-52`); no one else does |
| multilabel / regression | supported (`multilabel_gnn`, `regression_gnn`, `binary_gnn`) | `mlp`/`graphsage`/`dp_mlp` support multilabel+regression; `dpar` multilabel only; `dp_gnn`/`progap`/`heterpoisson` **single-label classification only** (`num_classes = int(train.y.max())+1`) |

The eval-graph row is the big one: for arxiv / facebook / reddit / flickr a random 60/20/20
node partition destroys most of a held-out node's neighbourhood, so the baselines are scored
on badly mutilated graphs while we score on the intact one.

## A.6 Metrics

- `mlp`/`graphsage`/`dp_mlp`/`dpar` share `_task_metric` (`dpar.py:252-257`): accuracy+macro-F1,
  or micro-F1 for multilabel (`dpar.py:205-222` — matches `MultiLabelGNNMechanism`), or MAE
  for regression (`dpar.py:225-231` — matches `RegressionGNNMechanism`). **These are genuinely
  the same metric definitions as ours.**
- **But there is no AUROC path.** For binary RelBench tasks we report AUROC
  (`binary_mechanism.py`, `run.py:397-400`) while these baselines would report plain accuracy
  on a heavily imbalanced split — incomparable.
- `progap` computes accuracy + a hand-rolled macro-F1 over `torch.unique(target)`
  (`inductive_adapter.py:41-50`) — the F1 denominator is `pred.sum()+actual.sum()`, i.e. an
  F1 over *present* classes only, not the same macro-F1 as `dpar.py:193-202`.
- `heterpoisson` reports `hit_accuracy` (micro accuracy) and `mean_f1_s` from its own
  `ClassificationMetrics` (`utils.py:59-81`), and only ever scores the *centre node* of each
  sampled subgraph (`train_scheduler.py:193-195`).
- All of them score on the **induced** val/test graph (§A.5).

## A.7 What has actually been run

| artifact | status |
|---|---|
| `results/**/*.json` | **0 files** — no external baseline has ever produced a result |
| `results/inductive/` | does not exist (the default output path of `src/experiments/run.py:135`) |
| `results/**/*_results.csv` | 335 files, all from `src.sparse.run` |
| `results/**/*_with_eps.csv` | 85 files — **none in any `*_matched_eps` / `*_meps` directory** |
| `configs/inductive_comparison.json` | consumed by nothing (`grep` → no hits) |
| `sbatch/` | 11 jobs, all `src.sparse.run` |
| `scripts/` | 34 files, none mentions any external baseline |
| `ice_status_report.txt` | only `trivial_baseline` lines; no baseline runs |

## A.8 Hardcoded / stale numbers

- `sbatch/ppi_matched_eps.sbatch:20-26` hardcodes a σ table (7.61/45.54 … 1.51/7.32) in a
  comment; `:120` hardcodes `GNN r=1 0.6256 | blind r=0 0.5314 | trivial 0.4608`;
  `sbatch/arxiv_inductive_matched_eps.sbatch:114` hardcodes `GNN r=1 0.5185 | blind r=0 0.4876`.
  These are stale Stage-1 references quoted as prose, not recomputed.
- `paper/experiments_current.tex:72` says "1 seed" for the item-ltv table, but the underlying
  CSVs have **3 seeds**.
- `paper/experiments_current.tex:160-176` has an empty DPAR / DP-MLP / ProGAP table —
  correctly empty, since nothing has been run. Nothing is *fabricated* anywhere. Good.
- No external-paper numbers are copied anywhere. Good.

---

# (B) Open questions / discrepancies / risks / fairness concerns

### B-1. HIGH — the "graph-blind" arm is **not graph-blind at inference**, and its neighbour weights are pure DP noise
The blind arm is `--model multilabel_gnn|gnn|binary_gnn|regression_gnn --r 0`
(`sbatch/ppi_matched_eps.sbatch:68,100-102`; `sbatch/arxiv_inductive_matched_eps.sbatch:108-109`;
`sbatch/relbench_regression_meps.sbatch:131-132`). At `r=0` each rooted subgraph is a single
isolated node, so training never sees an edge. But **evaluation runs full message passing**:
`multilabel_mechanism.py:110` / `gnn_mechanism.py:153` / `regression_mechanism.py:79` /
`binary_mechanism.py:94` all call `self.module(data.x, self.eval_edges(data))`.

With `aggr='mean'` the layer is `SAGEConv` (`layers.py:33`), whose `lin_l.weight` receives
**exactly zero gradient** on an isolated node (verified empirically: `lin_l.weight grad_norm=0.0`,
`lin_r.weight grad_norm=3.1`). In the DP path every parameter still gets Gaussian noise added
(`sparse_gnn.py:98-102`: `p.grad = (acc + z) / denom` for *all* params), so `lin_l.weight`
executes a pure random walk of per-step std `lr·σ·C/denom` and then, at eval, multiplies the
**real neighbour mean**. On PPI (`lr=0.3`, `denom≈512`, `T=2000`) that is a random matrix of
element std ≈ `0.039` at ε=8 and ≈ `0.20` at ε=1, injected into the logits.

Direct proof that the arm is not graph-blind: its metrics **differ between the two eval graphs**
(`results/ppi_matched_eps/blind_eps8.0`, step 2000, seed 0: `test_acc=0.42900` vs
`test_acc_alt=0.42629`; `test_auroc=0.70414` vs `0.71158`). A graph-blind model would give
bit-identical numbers on both.

Corroborating symptom: the arxiv blind arm is **flat** across the whole budget
(0.2821 / 0.2826 / 0.2825 / 0.2825 for ε = 1/2/4/8) — a real DP-MLP going from σ=2.12 to
σ=0.99 should improve visibly.

**Impact:** the floor our headline PPI/RelBench claims are measured against is depressed by an
artefact of our own code. The entire "+1.5 to +4.1 pts" claim
(`paper/experiments_current.tex:49-50`) rests on it.
**Fix:** use `MLPMechanism` (`src/sparse/mlp_mechanism.py`, which evaluates with
`self.module(data.x)` and no edges, line 74) for the blind arm, or zero/freeze `lin_l.weight`
when `r=0`, or add a graph-free eval column.

### B-2. HIGH — the blind arm's *architecture* is inconsistent across the repo
`scripts/_dataset_settings.sh:64,76` defines `BLIND` as `--model multilabel_gnn|binary_gnn --r 0`,
but `:87,94,103` defines it as `--model mlp --r 0` (facebook / reddit / arxiv). Meanwhile the
matched-ε sbatch files ignore `_dataset_settings.sh` entirely and hardcode the GNN-at-`r=0`
form for *every* dataset including arxiv (`results/arxiv_matched_eps/blind_eps8.0/...csv` shows
`model=gnn`). So `results/arxiv/inductive_mlp` (true MLP) and `results/arxiv_matched_eps/blind_*`
(GNN at r=0) are two different baselines both called "blind", and
`scripts/_meeting_figures_20260903.py:143` labels the latter **"graph-blind (DP-MLP)"** in the
figure legend, which is simply wrong.

### B-3. HIGH — nothing puts baseline ε and our ε on the same axis, and if anything did, the axes are incompatible
Our ε is **node substitution** (`accounting.py:243-245`, `±2k`). Every external baseline is
**node add/remove** (§A.3). At equal σ, substitution costs roughly the ε of add/remove at σ/2 —
a large, systematic penalty on *us*, so a naive side-by-side table would understate our method
(and `dp_mlp` in particular, whose ε comes from
`naive_opacus_epsilon` — a function whose own docstring at `accounting.py:609-612` disclaims it
as *not a valid node-level guarantee*). Additionally δ differs three ways: `n^-1.01`
(ours, `calibrate_grid.py:64`), `1e-5` (`baselines.py:29`), `5e-4` (all baseline configs),
`1/(10·n_train)` (`dpgnn.py:231`). And the accountants differ: PLD (ours) vs PRV (dp_mlp)
vs TF-Privacy RDP (dpar) vs `dp_accounting` RDP (dp_gnn) vs autodp (progap) vs a bespoke
numerical RDP (heterpoisson). **No conversion exists anywhere in the code.**

### B-4. HIGH — ProGAP cannot import: `third_party/ProGAP/core/data/` is missing and `.gitignore`d
`third_party/ProGAP/inductive_adapter.py:17` imports `core.methods.progap.node`, which at
`core/methods/progap/node.py:6,11` imports `core.data.loader.node` and
`core.data.transforms.bound_degree`. **`third_party/ProGAP/core/data/` does not exist on disk**
(`find third_party/ProGAP -type f` lists no such path), and `git check-ignore -v` confirms
`.gitignore:48` (`data/`) matches `third_party/ProGAP/core/data/loader/node.py`. The root
`data/` rule silently ate the vendored subtree. `core/datasets/loader.py:10-12` needs three
more missing transforms. **ProGAP is not runnable in this checkout at all**, and
`third_party/ProGAP/test_inductive_adapter.py` would fail at import too (it is outside
`pytest tests/` so nobody notices). The only ProGAP test that *does* run
(`tests/test_upstream_target_adapters.py:22-27`) uses a **stub** adapter script and never
touches ProGAP code.

### B-5. HIGH — baseline configs hardcode another user's Python interpreters
`configs/initial_progap.json:12-13`, `configs/cora_ml_progap_smoke.json`,
`configs/initial_heterpoisson.json`, `configs/cora_ml_heterpoisson_smoke.json` all set
`"command": ["/usr/scratch/asaha92/envs/{progap,heterpoisson}/bin/python", "inductive_adapter.py"]`.
`upstream.py:157` runs that argv with `check=True`. These paths belong to a different
cluster user and do not exist locally. Combined with B-4, the upstream arm is not reproducible
by anyone on this repo today.

### B-6. HIGH — ProGAP is configured well outside its own paper's grid, in ways that cripple it
`third_party/ProGAP/experiments.py:52-55` is ProGAP's own node-level grid:
`max_degree = 100` (facebook), `epochs ∈ {5, 10}`, `batch_size = 256`, `lr ∈ {0.01, 0.05}`,
`base_layers ∈ {1,2}`, `depth ∈ {1..5}`, `repeats = 10`.
`configs/initial_progap.json` instead uses `max_degree=5`, `epochs=100`, `batch_size=32`,
`depth=2`, one seed, no lr search. Since `NoisySGD` composes `epochs·n//batch_size` steps
(`core/privacy/algorithms/noisy_sgd.py:35`) and ProGAP trains `depth+1` stages, 100 epochs at
batch 32 is ~**60× more composition than upstream ever used**, which forces σ up by a large
factor at the same ε. Additionally, `inductive_adapter.py:25-31` sets
`train_mask = val_mask = test_mask = all nodes`, so ProGAP's best-checkpoint selection
(`core/trainer/trainer.py:103-107`, `monitor='val/acc'`) selects on the **training partition** —
no real validation at all.

### B-7. HIGH — HeterPoisson is likewise configured against upstream's own scripts
Upstream `run_facebook.sh`/`run_pubmed.sh`/`run_amazon.sh`: `expected_batchsize=4096`,
`epoch=9` (reddit 4), `lr=0.01`, and sweeps `num_neighbors ∈ {1..5}` *and*
`num_neighbors_test ∈ {1,4,7,10,13}` over 3 seeds.
`configs/initial_heterpoisson.json` uses `expected_batchsize=32`, `epochs=100`, `lr=0.001`,
`K=1`, `num_neighbors=1`, one seed — i.e. ~128× more steps at 128× smaller batch, 10× smaller
lr, a single-layer model seeing exactly one neighbour, and no `num_neighbors_test` knob at all
(the adapter never exposes it; `sampling.py:163-166` therefore uses `num_neighbors=1` at test).

### B-8. HIGH — DPAR's reported ε uses an unsound linear amplification, ours does not
`src/experiments/privacy.py:189` (`epsilon = inverse_composition(...) * amplification_rate`) and
`:216` (`epsilon=float(epsilon * amplification_rate)`) faithfully reproduce upstream
(`third_party/DPAR/main.py:108,137`). Multiplying ε by the subsampling rate `q` is not a valid
amplification theorem outside the `ε→0` regime. With `sampled_train_nodes=70` on a ~1.8k-node
cora-ml train partition, `amplification_rate ≈ 0.04` — a **~25× discount** on the reported ε.
Putting that ε on our (substitution, PLD) axis without comment would give DPAR an enormous
unearned head start.

### B-9. HIGH — DPAR's port trains on only ~70 nodes; upstream trains on all of them
Upstream (`third_party/DPAR/main.py:97,164-173`) computes real PPR rows for the first
`ppr_num` train nodes and gives **every remaining train node an identity row**, so training
still uses the whole `train_index`. Our port (`dpar.py:260-279` `_sample_train_partition`,
called at `dpar.py:305-307`) instead **induces a subgraph on `sampled_train_nodes` nodes** and
trains on those alone — no identity rows, no remaining nodes. With
`configs/cora_ml_dpar_eps8.json` that is a **70-node training set** against our 45k (PPI) /
91k (arxiv). The `for root in range(num_nodes)` ISTA loop (`dpar.py:113`) with
`max_iterations = max(10_000, 10n)` makes anything larger intractable, so this is not easily
raised.

### B-10. HIGH — `dp_gnn` and `dp_mlp` have no target-ε path and cannot be matched to our budget
`BaselineConfig` (`baselines.py:17-32`) has `noise_multiplier` but no `target_epsilon`;
`BaselineTrainer.fit` only ever calls `.account(...)` (`baselines.py:137`), never
`DPMLPAccountant.calibrate` (`privacy.py:96-112`), which is dead code.
`dpgnn_adapter.run_partitioned` (`dpgnn_adapter.py:33-35`) forwards only
`steps / batch_size / noise_multiplier / evaluate_every / seed` — `learning_rate`,
`latent_size`, `max_degree`, `clip_percentile`, `max_subgraph_nodes` (`dpgnn.py:32-36`) are
**unreachable from any config**. So these two can only be run at a σ you guess, and DP-GNN is
permanently pinned at library defaults while our method is swept over ~60 tuning cells.

### B-11. HIGH — baselines evaluate on mutilated held-out graphs, ours evaluates on the intact graph
`src/experiments/inductive.py:169-186` (`_induce`) deletes every cross-partition edge from
**all three** partitions, and every baseline's `_evaluate` scores on `partition.data`
(`baselines.py:100-104`, `dpar.py:289-294`, `dpgnn.py:209-213`,
`ProGAP/inductive_adapter.py:34-50`, `PNPiGNNs/inductive_adapter.py:120-127`). Ours evaluates
on the full `data.edge_index` (`run.py:322-328`, `--eval_graph auto`), and
`src/sparse/run.py:476-481` explicitly refuses to filter the eval graph *precisely because*
"on PPI their mean in-degree drops 29.3 → 0". The baselines get exactly the treatment we
rejected for ourselves. On arxiv/facebook/reddit a random 60/20/20 partition removes the
majority of a test node's neighbours.

### B-12. MED — DP-GNN's clipping thresholds are computed non-privately from real gradients
`dpgnn.py:173-185` takes a 75th-percentile quantile of the true per-root gradient norms on the
first batch and uses it as the clipping norm, with **no noise and no privacy charge**
(`dpgnn.py:224`). This is a leak in the baseline's *favour* (it gets free, well-calibrated
adaptive clipping that our fixed `C=1.0` does not). It also means DP-GNN's reported ε is
optimistic relative to its actual mechanism.

### B-13. MED — the two selection rules in the repo give different, non-ε-matched answers
`scripts/_meeting_figures_20260903.py:88-96` (`final_acc`) reports the **last** checkpoint —
that IS ε-matched, and reproduces the paper's numbers (PPI final step: blind 0.4024/0.4166,
GNN 0.4171/0.4580 → +1.47/+4.14 ≈ the claimed "+1.5 to +4.1").
`scripts/summarize_sweep.py:103-107` (run in every sbatch tail) reports the **best-on-val**
checkpoint, which lands at different steps per arm — PPI ε=1: blind at step **200**, GNN at
step **1400**; item-ltv: blind at step **300**, GNN at **475**. Since ε grows with the step
count and **no `*_with_eps.csv` exists in any `*_matched_eps`/`*_meps` directory**, those
tables silently compare arms at different privacy budgets (the blind arm quoted at a stricter
ε than the GNN arm). Run `src.sparse.compute_epsilon` on the matched-ε dirs, or restrict the
comparison to step `T`.

### B-14. MED — `mlp`/`graphsage` get 100 *full-batch* steps while `dp_mlp` gets 100 *epochs* of minibatches
`baselines.py:116-125`: the non-private branch does one full-batch Adam step per epoch (100
updates total); the `dp_mlp` branch does `steps_per_epoch = ⌈n/256⌉` updates per epoch
(`baselines.py:113,118-120`). The "non-private ceiling" therefore gets ~7–350× fewer updates
than the private run it is supposed to bound. (An analogous symptom already exists in our own
results: `results/relbench/relamazon_itemltv_meps/nodp_blind` final MAE **127.5** vs the DP
blind arm's **93–94** — the non-DP reference is worse than every private run.)

### B-15. MED — DPAR's accountant assumes Poisson sampling; its trainer uses shuffled fixed batches
`dpar.py:338-339` iterates `torch.randperm(...).split(batch_size)`, but
`DPARAccountant.account_training` (`privacy.py:198-223`) calls TF-Privacy `compute_rdp` with
`q = batch_size / ppr_releases`, which is the subsampled-Gaussian (Poisson) bound. Upstream has
the same gap, so this is faithful-but-unsound. Ours genuinely Poisson-samples
(`sparse_expand.py:321-339`), so our ε is the honest one — worth saying out loud in the paper.

### B-16. MED — `dp_mlp` normalizes by the realized Poisson batch size
`baselines.py:173` divides by `selected.numel()` (the realized `|B|`) rather than the expected
batch size; `baselines.py:155-156` also skips the step entirely (no noise) when `|B| = 0`.
Standard DP-SGD requires the expected-batch normalizer. Ours does it correctly
(`sparse_gnn.py:62-68,102`, `denom = expected_batch`). Minor ε leakage in the baseline.

### B-17. MED — `progap` / `heterpoisson` / `dp_gnn` cannot run on PPI or any RelBench task
All three derive `num_classes = int(train.y.max()) + 1`
(`ProGAP/inductive_adapter.py:97`, `PNPiGNNs/inductive_adapter.py:105`,
`dpgnn_adapter.py:22-24`) and use `F.cross_entropy`. PPI is 121-way multilabel and RelBench is
binary/regression. So on our three headline benchmarks only `dpar`, `mlp`, `graphsage`,
`dp_mlp` are even applicable — and per `paper/experiments_current.tex:155-158` even those are
blocked on RelBench because `_native_split_indices` (`inductive.py:124-128`) requires the masks
to cover every node exactly once, while `relbench_data.py:286` leaves unlabeled DB-entity nodes
in no mask. Confirmed structurally.

### B-18. MED — ProGAP trains degree-bounded but predicts unbounded, and adds NAP noise to held-out graphs
`core/methods/progap/node.py:93-95` applies `BoundOutDegree(max_degree)` inside `setup()`, but
`ProGAP/inductive_adapter.py:34-40` deliberately bypasses `setup()` for val/test and only calls
`_prepare()`, which does **not** bound degree. The NAP noise std is calibrated for sensitivity
`√max_degree` (`progap/node.py:44`) yet the prediction-time aggregation runs over the
unbounded degree — a train/test mismatch. Worse, `NAP.forward` (`core/nn/nap.py:15-19`) still
*perturbs* on the held-out graphs, so ProGAP pays accuracy for noise that buys no privacy in
this disjoint-partition setting.

### B-19. MED — `third_party/PNPiGNNs/.../main.py` is broken by local edits
`train_scheduler.trainer.__init__` was modified to require keyword-only `target_delta`,
`degree_bound`, `steps` (`train_scheduler.py:33-34`), but `main.py:28-35` still constructs it
without them. The vendored upstream entry point no longer runs; only the adapter does. This
also means the vendored tree is **not** a pristine copy of revision `9a06332` as
`upstream.py:31` claims.

### B-20. MED — our method is Adam-vs-SGD mismatched against every baseline
Every baseline is unconditionally Adam (§A.4). Our matched-ε runs pin `--optimizer sgd`.
`sbatch/ppi_matched_eps_adam.sbatch:26-28` explicitly flags this ("matching the baseline
harness, which is already unconditionally Adam") and was written to test it — but the
comparison run has no results yet (`results/ppi_matched_eps_adam/` does not exist).

### B-21. LOW — width asymmetry is large and acknowledged as free in ε
The README (`README.md:88-101`) and `sbatch/ppi_matched_eps.sbatch:33-35` state that hidden
width costs nothing in ε and is worth +10–14 points. We run `hidden=256`; the baselines run
16 (ProGAP), 32 (DPAR), 64 (dp_mlp), 100 (DP-GNN), 128 (HeterPoisson). Even a faithful
reproduction of each paper's defaults will look weak against a width our own docs say is free.
Any published table needs either a width-matched arm or an explicit note.

### B-22. LOW — device pins are stale/inconsistent
`src/experiments/run.py:36-44` forbids `cuda:0-3` and auto-selects `cuda:4`, but the baseline
configs pin `cuda:5` (`initial_progap`, `initial_heterpoisson`) and `cuda:6`
(`initial_dpar_pubmed`), while our sbatch jobs run on `ice-cpu`. Cosmetic, but it means the
configs were last exercised on a different machine.

### B-23. LOW — `trivial_baseline` is computed from private training labels
`src/sparse/run.py:116-118` takes the majority class from `y[data.train_mask]` with no noise.
It is only a reported floor, never a released model, but if it appears in a paper table it is
technically a non-private statistic of the training set.

### B-24. LOW — third-party tests are never executed
`third_party/ProGAP/test_inductive_adapter.py` and
`third_party/PNPiGNNs/.../test_inductive_adapter.py` are outside `pytest tests/`. The ProGAP
one would currently fail at import (B-4). The two in-repo baseline tests
(`tests/test_dpgnn_adapter.py`, `tests/test_upstream_target_adapters.py`) test the first-party
DP-GNN port and a **stub** subprocess adapter respectively — neither exercises ProGAP or
HeterPoisson code.

---

## Suggested priority order

1. **B-1 / B-2** — the blind arm is the only baseline with numbers and it is measurably
   corrupted. Everything in `paper/experiments_current.tex` §"Results so far" depends on it.
2. **B-13** — attach per-checkpoint ε to the `*_matched_eps` CSVs so "matched ε" is verifiable.
3. **B-3** — decide and *document* the neighbouring relation for every ε before any table is
   built; substitution-vs-add/remove is a factor-2 in σ.
4. **B-4 / B-5** — ProGAP is un-runnable; fix `.gitignore` (`!third_party/**/data/`) and
   re-vendor `core/data/`, and de-hardcode the interpreter paths.
5. **B-6 / B-7 / B-9 / B-10 / B-21** — if baselines are going to be published, each needs at
   minimum its own paper's hyperparameter grid, matched width, and a target-ε path.

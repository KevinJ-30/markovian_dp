# Audit — base mechanisms, losses, metrics, checkpoint selection

Repo: `/Users/kevinjacob/markovian_dp copy`, branch `sparse_expand_clean`, HEAD `b10a794`.
Scope: task heads / losses / evaluation metrics. Read-only; no files edited.

---

# (A) What the code does

## A.1 Mechanism table

| Class | File | Task type | Loss (exact) | Head shape | Activation | `metric_name` (primary) | Secondary metrics | Datasets |
|---|---|---|---|---|---|---|---|---|
| `GNNMechanism` | `src/sparse/gnn_mechanism.py:44` | single-label node classification | `F.nll_loss(root_logits, root_y)` — 1 root, default `reduction='mean'` over 1 element (`gnn_mechanism.py:82`); batched path uses `reduction='none'` then one loss per root (`:118`) | `[n, num_classes]` | `F.log_softmax(x, dim=1)` (`:41`) | `"accuracy"` (inherited default, `base_mechanism.py:32`) | none | ogbn-arxiv, Flickr, Reddit, facebook, Planetoid, RelBench multiclass |
| `MLPMechanism` | `src/sparse/mlp_mechanism.py:46` | single-label node classification, graph-blind | `F.nll_loss(out, root_y)` (`:69`) | `[1, num_classes]` | `F.log_softmax(x, dim=1)` (`:43`) | `"accuracy"` (inherited) | none | blind/Stage-0 baseline arms on the same datasets, run with `--r 0` |
| `MultiLabelGNNMechanism` | `src/sparse/multilabel_mechanism.py:75` | multilabel node classification | `F.binary_cross_entropy_with_logits(root_logits, root_y)` — default `reduction='mean'`, i.e. **mean over the 121 labels** (`:104`) | `[n, num_labels]` | none — raw logits (`:41`) | `"micro_f1"` (`:82`) | `<split>_auroc` = micro-AUROC (`:118`) | PPI (121 labels) |
| `BinaryGNNMechanism` | `src/sparse/binary_mechanism.py:59` | binary entity classification | `F.binary_cross_entropy_with_logits(root_logit, root_y)`, 1 element (`:86`) | `[n]` (`dims[-1]=1`, `.view(-1)`) | none — raw logit (`:40`) | `"auroc"` (`:66`) | `<split>_bin_acc` = accuracy at logit>0 (`:106`) | RelBench BINARY_CLASSIFICATION (rel-f1/driver-top3, driver-dnf; rel-hm/user-churn; rel-stack/user-badge) |
| `RegressionGNNMechanism` | `src/sparse/regression_mechanism.py:45` | node/entity regression | `F.mse_loss(root_pred, root_y)`, 1 element (`:73`) | `[n]` | none — unbounded (`:42`) | `"mae"` (`:52`) | `<split>_rmse` (`:92`), `<split>_r2` (`:95`) | RelBench REGRESSION (rel-amazon/user-ltv, item-ltv; rel-hm/item-sales; rel-trial/…) |

Shared facts:
- **No class weighting, no `pos_weight`, no label smoothing anywhere** in the five mechanisms (grep over `src/` finds `scale_pos_weight` only in the GAD XGBoost side pipeline, `src/sparse/gad/xgb_graph.py:83-86`).
- Backbone is shared: `build_conv_stack` → `SAGEConv(aggr='mean')` or `GCNConv(add_self_loops=True, normalize=True)` (`src/sparse/layers.py:27-38`). No BatchNorm/LayerNorm anywhere in `src/sparse/` (dropout is the only train/eval-mode-sensitive layer).
- Unsupervised roots return `zero_loss()` — a differentiable zero, `params[0].sum()*0.0` (`base_mechanism.py:147-152`), gated on `train_mask` in all five (`gnn:73`, `mlp:62`, `multilabel:95`, `binary:78`, `regression:65`).
- Only `GNNMechanism` implements the batched `iter_subgraph_loss_batches` fast path (`gnn_mechanism.py:123`); the other four fall back to the base per-subgraph loop (`base_mechanism.py:137-140`).

## A.2 Metric table — exact implementations

| Metric | Where | Implementation | Threshold / baseline |
|---|---|---|---|
| accuracy (single-label) | `gnn_mechanism.py:154-160`, `mlp_mechanism.py:76-82` | hand-rolled: `out.argmax(dim=1)`, `(pred[mask]==y[mask]).sum()/n` | argmax, no threshold |
| micro-F1 (multilabel) | `multilabel_mechanism.py:44-50` (`_micro_f1`) | hand-rolled: pooled tp/fp/fn over all (node,label) pairs, `2tp/(2tp+fp+fn)` | **fixed `logits > 0`** (= sigmoid ≥ 0.5), `:111`. **Not tuned.** NaN when `denom==0` |
| micro-AUROC (multilabel) | `multilabel_mechanism.py:53-72` (`_micro_auroc`) | hand-rolled Mann-Whitney over the **flattened** (node,label) pool → "micro" averaging. **No tie-averaging** (`torch.argsort` + `arange` ranks). float32 | threshold-free. NaN if a split is all-pos or all-neg |
| AUROC (binary) | `binary_mechanism.py:43-56` (`_auroc`) | hand-rolled Mann-Whitney, numpy float64, **with** tie-averaged ranks via `np.unique`/`np.bincount` (`:53-55`). Verified `isclose` to `sklearn.roc_auc_score` (`tests/test_mechanisms.py:59-65`) | threshold-free. NaN if single-class |
| bin_acc (binary secondary) | `binary_mechanism.py:106-108` | hand-rolled `((scores>0).astype(y.dtype) == y).mean()` | **fixed logit > 0** (= prob > 0.5) |
| MAE (regression) | `regression_mechanism.py:90-91` | hand-rolled: `((pred-target)*target_std).abs().mean()` | reported in the label's original units (× `target_std`) |
| RMSE | `regression_mechanism.py:92` | hand-rolled `residual.pow(2).mean().sqrt()` | original units |
| R² | `regression_mechanism.py:93-97` | hand-rolled `1 - SS_res/SS_tot` with **`SS_tot = Σ(y_true − mean(y_true[mask]))²` = the EVALUATED SPLIT's own mean** | matches `sklearn.r2_score`, which is exactly what RelBench's `relbench.metrics.r2` calls. NaN when `ss_tot == 0` |
| AUROC / AUPRC / Rec@K (GAD side pipeline) | `src/sparse/gad/metrics.py:24-45` | **sklearn** `roc_auc_score`, `average_precision_score`; `rec_at_k` hand-rolled with `k = #positives in y_true` (GADBench convention) | AP = average precision. Rec@K uses test-set positive count to set k |
| macro-F1 (baselines path only) | `src/experiments/dpar.py:193-202` | hand-rolled, averaged over `torch.unique(labels)` — i.e. only classes **present in the eval split** (sklearn's default averages over the union of true∪pred) | argmax |
| micro-F1 (baselines path) | `src/experiments/dpar.py:205-222` | duplicate hand-rolled implementation, identical formula to `multilabel_mechanism._micro_f1` | `logits > 0` |
| MAE (baselines path) | `src/experiments/dpar.py:225-231` | `(preds - target).abs().mean()` — **no `target_std` rescaling** | z-space units |

**Trivial baselines ARE computed in code**, not only in prose: `src/sparse/run.py:90-118` (`trivial_baseline`), written to the CSV column `trivial_baseline` for every row (`run.py:545`, `:657`) and printed at startup (`run.py:521`).
- `auroc` → hardcoded 0.5 (`:108`)
- `micro_f1` → `2p/(1+p)` at the **test-split** positive rate p (`:110-112`). Verified numerically to 8 decimals against the all-ones predictor's `_micro_f1`; reproduces README's 0.4608 on PPI.
- `mae` → `mean(|y_test|) * target_std` (`:113-115`) — see risk #2, this is NOT the train-mean predictor.
- `accuracy` → most-frequent **train** class scored on test (`:116-118`).
- `ceiling_fullbatch.py:101` reuses the same function.

## A.3 `_alt` convention and CSV naming

- `_evaluate` in `sparse_gnn.py:107-113` runs `mechanism.evaluate(data)` on the configured graph, then `mechanism.evaluate_on(data, alt_edge_index)` and suffixes **every** returned key with `_alt`.
- `eval_graph='auto'` → `'train'` for transductive, `'full'` for inductive (`run.py:319-326`); the alt graph is then the other one (`run.py:613-617`).
- CSV header (`run.py:535-565`): `train_acc/val_acc/test_acc` hold **whatever `metric_name` says** (accuracy / micro-F1 / AUROC / MAE); `metric` column records which. Alt columns are named `train_acc_alt/val_acc_alt/test_acc_alt` + `*_auroc_alt`, and the writer reads dict keys `train_alt/val_alt/test_alt` + `*_auroc_alt` (`run.py:666-668`). **The mapping is correct**, but the header spelling (`_acc_alt`) and the dict key (`_alt`) differ.
- Primary/alt convention **is consistent across all five mechanisms**: primary always lands in `*_acc`, secondaries always get their own suffixed columns, alt-graph copies always get `_alt`.

## A.4 Evaluation protocol

| Question | Answer |
|---|---|
| `torch.no_grad()`? | Yes — `@torch.no_grad()` on all five `evaluate` methods (`gnn:149`, `mlp:71`, `multilabel:106`, `binary:88`, `regression:75`) |
| Full-batch or batched? | Full-batch: one forward over **all** nodes, then masked per split. Above `_DENSE_MESSAGE_BUDGET = 250M` arc×feature elements it swaps the edge_index for a CSR adjacency (`base_mechanism.py:50, 69-90`); `tests/test_mechanisms.py:156-186` asserts the two agree to 1e-9 |
| Which graph? | The full graph by default; `eval_edge_index` overrides it (`base_mechanism.py:38-43`, `run.py:613-617`). Never the sampled subgraphs. Transductive runs default to the **capped training graph**, inductive to the **full** graph |
| Same aggregator/normalization as training? | Same `nn.Module`, same `aggr`. But training sees rooted, p2-sparsified, degree-capped neighbourhoods while eval sees full ones. For `aggr='mean'` the rooted computation equals full-graph inference at p2=1/uncapped (`layers.py:7-10`); for `aggr='gcn'` it does not (`layers.py:11-14`) |
| `.eval()` mode? | Yes — every `evaluate` calls `self.eval_mode()` → `module.eval()` (`base_mechanism.py:117-118`). Dropout is the only mode-sensitive layer; no BatchNorm exists |
| Does eval consume RNG? | **No.** Dropout is off in eval, so no draw from the global generator; root/edge sampling uses a dedicated `sample_gen` and noise a dedicated `noise_gen` (`sparse_gnn.py:181-182`). The docstring claim at `sparse_gnn.py:163-167` ("a tracked run follows exactly the same trajectory as an untracked one") holds |
| Train mode restored? | Yes — `_step_nondp`/`_step_dp` each call `mechanism.train_mode()` first (`sparse_gnn.py:41, 67`) |

## A.5 Checkpoint selection (`scripts/summarize_sweep.py`)

- `_curve(rows, key)` (`:30-45`) buckets by `step`, averages across seeds, and also collects `epsilon` per step.
- Selection (`:95-116`): `val_key = 'val_' + metric[5:]` when `--metric` starts with `test_`; select `min`/`max` over the **val** curve; report `curve[best]` = the **test** value at that step. Falls back to selecting on test with a printed `WARNING: ... (leakage)` only when there is no val column. **Commit `30ceee7` is genuinely implemented and is the only selection path in the sweep summarizer.**
- Direction (`_lower_is_better`, `:48-64`): special-cases `*_r2`/`r2` → higher-is-better (commit `5978dbb`), then falls back to the CSV's own `metric` column → `mae`/`rmse` are lower-is-better, everything else higher. **Direction is correct for all five mechanisms' primary and secondary columns**: `test_acc` on a `mae` run → lower; `test_rmse` → lower; `test_r2` → higher; `test_bin_acc` on an `auroc` run → higher; `test_auroc` on a `micro_f1` run → higher.
- Callers: `sbatch/*.sbatch` and `scripts/*.sh` all pass `--metric test_acc` (or `$METRIC`, default `test_acc`) — always a `test_` prefix, so the val path is always taken when a val column exists.

## A.6 Metric vs. official benchmark metric

Verified against the installed `relbench` package:
- `relbench.base.task_entity` regression tasks declare `metrics = [r2, mae, rmse]`; classification tasks declare `metrics = [average_precision, accuracy, f1, roc_auc]`.
- `relbench.metrics.r2 = sklearn.r2_score`, `mae = sklearn.mean_absolute_error`, `roc_auc = sklearn.roc_auc_score`, `accuracy` thresholds `pred > 0.5` on a probability, `f1` thresholds `pred >= 0.5`.

| Benchmark | Official headline | What we report | Verdict |
|---|---|---|---|
| RelBench entity classification | AUROC | AUROC (primary) + accuracy (secondary) | ✅ matches; hand-rolled AUROC is tie-corrected and unit-tested against sklearn. Our `bin_acc` (logit>0) is equivalent to RelBench's `accuracy` (prob>0.5). **We do NOT report `average_precision` or `f1`, which RelBench also lists** — AP is the metric most sensitive to the imbalance these tasks have |
| RelBench entity regression | MAE | MAE (primary) + RMSE + R² | ✅ MAE primary is right, and **commit `4562ee7`'s claim is accurate**: R² genuinely is in RelBench's own metric list for regression, and our R² uses sklearn's/RelBench's convention (the evaluated split's own mean). The commit did exactly what it said |
| OGB (ogbn-arxiv) | accuracy | accuracy | ✅ same quantity; we use OGB's official split indices (`src/datasets.py:84`) but compute accuracy ourselves rather than through `ogb.Evaluator` (numerically identical for top-1 accuracy) |
| PPI | micro-F1 | micro-F1 (primary) + micro-AUROC (secondary) | ✅ micro-F1 at sigmoid≥0.5 matches the GraphSAGE/GAT convention. Micro-AUROC is an extra the benchmark does not report — correctly labelled secondary |

`b10a794` ("Add secondary accuracy metric to BinaryGNNMechanism") did what its message says in `binary_mechanism.py` (8 lines: the `bin_acc` block) and `src/sparse/run.py` (the three new columns), but it **also** silently added regression support to `src/experiments/baselines.py` / `dpar.py` / `inductive.py` / `experiments/run.py` — see risk #1, that half of the commit carries the bug.

---

# (B) Open questions / discrepancies / risks

### 1. HIGH — baselines pick the **worst** checkpoint on regression tasks (direction not flipped)
`src/experiments/baselines.py:114` initialises `best_val = float("-inf")` and `:127` keeps the state whenever `validation > best_val`. For `regression=True` the "validation" value is **MAE** (`dpar.py:225-231`, `baselines.py:103-104`), where lower is better — so the saved best checkpoint is the one with the **highest** MAE, i.e. the worst model, and that is the state loaded at `:132` and reported as `test_accuracy`. Same pattern at `src/experiments/dpar.py:334, 349-351`.
Commit `b10a794` added the `regression` flag to `BaselineConfig` and threaded it through loss/metric but never touched the selection direction. Any regression number produced through the `mlp`/`dp_mlp`/`graphsage` baseline path is wrong (systematically bad), which would make our method look better than it is. `scripts/summarize_sweep.py` got this right (`:48-64`); this path did not.

### 2. HIGH — the regression target is never mean-centred, so the MAE "trivial baseline" is the **zero** predictor, not the train-mean predictor
`src/sparse/relbench_data.py:263-269` does `y = y / target_std` only — **the train mean is never subtracted**. Three places assert the opposite:
- `src/sparse/regression_mechanism.py:8-13`: *"Targets are expected in Z-SCORED form (train-split mean subtracted, train-split std divided out) — `load_relbench` does this"*.
- `src/sparse/relbench_data.py:257-262`: *"'predict the train mean' is exactly 'predict 0' in z-space, which is what the trivial-baseline computation in run.py relies on"*.
- `src/sparse/run.py:96-99`: the `mae` branch of `trivial_baseline` is documented as *"MAE of 'always predict the train mean' on test"* but computes `mean(|y_test|) * target_std` (`:115`) = MAE of the **all-zero** predictor.

RelBench regression targets (LTV, sales) are non-negative with large positive means, so the zero predictor is far worse than the train-mean predictor and the printed floor is inflated. Every `"trivial baseline (mae) on test: …"` line in `ice_status_report.txt` (11.93 / 0.0761 / 16.78 / 77.13 / 0.0522) is the wrong baseline, and every `<-- BELOW TRIVIAL BASELINE` / "clears it" judgement built on it is unreliable. **R² is unaffected** (it is invariant to the missing translation), so `test_r2` is currently the only trustworthy "does this beat trivial" signal for regression — and it is negative in several logged runs.
Secondary consequence: training MSE on an un-centred target forces the net to learn a large intercept, which under DP clipping is exactly the kind of signal that gets clipped away.

### 3. HIGH — `summarize_sweep.py` silently selects the **first** checkpoint when the val curve contains a NaN
`scripts/summarize_sweep.py:105-107` uses `min/max(val_curve, key=val_curve.get)`. Python's `max` seeds with the first element and only replaces on `>`; every comparison against NaN is False. Verified: `max({10: nan, 20: 0.5, 30: 0.9}, key=…)` returns `10`. So a **single leading NaN** (let alone an all-NaN column) makes the summarizer report the earliest tracked step as "best" with no warning at all — the `WARNING: … (leakage)` branch never fires because `val_curve` is truthy. NaN val is reachable: `_auroc` returns NaN on a single-class val split (`binary_mechanism.py:47-48`), `_micro_auroc` likewise (`multilabel_mechanism.py:67-68`), `r2` when `ss_tot==0` (`regression_mechanism.py:97`), and any metric when a split mask is empty. Small RelBench tasks are exactly where this bites.

### 4. MED — `_micro_auroc` has no tie handling; the README's "AUROC 0.4955" is an artefact
`src/sparse/multilabel_mechanism.py:69-72` assigns ranks `1..N` in `argsort` order with no averaging within ties, unlike the binary `_auroc` which explicitly does average ties (`binary_mechanism.py:52-55`). Consequence: a constant predictor scores ≠ 0.5. Measured on a synthetic PPI-shaped target: all-ones predictor → `_micro_auroc = 0.4988` (should be exactly 0.5). This is where `README.md:165-166`'s "AUROC 0.4955" and `scripts/diagnose.sh`'s `all-ones (eps=0)` row come from — the docstring at `multilabel_mechanism.py:59` itself says the right answer is 0.5. Two AUROC implementations in the same repo disagree on ties, and only the binary one is unit-tested for it (`tests/test_mechanisms.py:50-52` covers `_auroc`, nothing covers `_micro_auroc`). Impact on real (non-tied) logits is negligible; impact on the quoted floor is real.

### 5. MED — RelBench AP/F1 are never computed, so the imbalanced-task story rests on AUROC alone
RelBench's own binary tasks list `[average_precision, accuracy, f1, roc_auc]`. We compute roc_auc and accuracy; AP and F1 are absent from `BinaryGNNMechanism.evaluate` (`binary_mechanism.py:88-109`) and from the CSV schema. On tasks with an ~17-20% (or ~82%) positive rate, AP is the discriminating number, and it is the one a RelBench reviewer will look for. `sklearn.average_precision_score` is already imported elsewhere in the repo (`src/sparse/gad/metrics.py:11`), so the only reason it is missing is that nobody added it.

### 6. MED — the baselines path reports regression MAE in **z-space** while our mechanism reports it in **original units**
`src/experiments/dpar.py:230` computes `(preds - target).abs().mean()` with no `target_std` multiplier, while `regression_mechanism.py:90-91` multiplies the residual by `self._target_std`. Its own docstring (`dpar.py:226-228`) claims *"matches src.sparse.regression_mechanism's metric, so a baseline and the SparseGNN mechanism are judged the same way"* — they are not, they differ by a factor of `target_std` (e.g. ~40× on rel-amazon/user-ltv). Any table putting the two side by side is comparing different units. (Mitigating: the two pipelines use different loaders and may never be tabled together — worth confirming.)

### 7. MED — `trivial_baseline` is computed from **test** labels
`src/sparse/run.py:109-118`: the `micro_f1` positive rate, the `mae` magnitude, and the majority-class accuracy are all evaluated on `data.test_mask`. It is a label-only reference, not a model decision, so it cannot leak into training — but it is a test-set statistic printed and recorded as the bar results are judged against, and the `accuracy` branch is inconsistent with the others (it takes the majority class from **train**, `:116-117`, then scores on test; the other three read test directly). Worth making all four train-derived for consistency.

### 8. MED — `_accuracy_and_macro_f1` averages only over classes present in the eval split
`src/experiments/dpar.py:197` iterates `torch.unique(labels)` (ground-truth labels only). sklearn's `f1_score(average='macro')` averages over the union of true and predicted labels. On a split missing a class, or when the model predicts a class that never appears in the split, our macro-F1 is higher than sklearn's. Only affects the `experiments/` baselines path, not the sparse mechanisms.

### 9. MED — regression `_alt`, `_rmse_alt`, `_r2_alt`, `_bin_acc_alt` are computed and then thrown away
`sparse_gnn.py:110-112` suffixes **every** key of the alt evaluation, so `train_rmse_alt`, `test_r2_alt`, `test_bin_acc_alt` all exist in the returned dict — but `run.py:666-668` only writes `train_alt/val_alt/test_alt` and `*_auroc_alt`. The alt-graph secondary metrics are silently dropped, so the "measure the cap gap rather than assume it" claim (`README.md:161-163`) holds only for the primary metric and AUROC, not for RMSE/R²/bin_acc.

### 10. MED — the regression guard from `e4363e5` is correct but **incomplete**
`src/sparse/run.py:390-396` raises `SystemExit` when `'REGRESSION' in task_type.upper()` and `args.model != 'regression_gnn'`. What it guards: a classification/binary mechanism being pointed at float targets, which would otherwise die inside `nll_loss`/`cross_entropy` with an opaque dtype error. Three gaps:
  1. **It only fires when `task_type` exists.** `task_type` is only set by `_RelBenchDataset` (`relbench_data.py:291`). A non-RelBench float-target dataset gets `task_type == ''` and the guard is a no-op.
  2. **The converse is unguarded**: `--model regression_gnn` on a *classification* dataset is never rejected. It would silently train MSE against class indices and report "MAE" — a wrong number rather than a crash, which is the worse failure mode.
  3. **The `experiments/` path has no equivalent guard at all.** `src/experiments/run.py:64-89` (added by `b10a794`) passes `regression` only to `{mlp, dp_mlp, graphsage}`; `DPARConfig` has no `regression` field, so `method="dpar"` with `regression: true` reaches `_task_loss(..., multilabel)` (`dpar.py:346, 377`) with `regression` defaulting to False → `F.cross_entropy` on float targets → exactly the opaque crash `e4363e5` set out to eliminate, in the sibling driver. The exclusion is documented as deliberate in a comment (`experiments/run.py:63-67`) but is not enforced by a raise.

### 11. LOW — CSV column `train_acc_alt` vs dict key `train_alt`
`run.py:564` names the header `train_acc_alt/val_acc_alt/test_acc_alt`; `run.py:666` reads `train_alt/val_alt/test_alt`. Functionally correct, but the two spellings disagree, and the `_acc` in the header is doubly misleading because the column holds MAE on a regression run. Same pre-existing wart as `test_acc` holding AUROC/MAE for the non-accuracy mechanisms (documented at `run.py:533-534`, `summarize_sweep.py:56-60`).

### 12. LOW — per-root multilabel loss is a **mean over 121 labels**, the full-batch ceiling is a mean over (nodes × labels)
`multilabel_mechanism.py:104` uses default `reduction='mean'`, so each root contributes `(1/121)·Σ_labels BCE`; the summed batch objective is `Σ_roots (1/121)·Σ_labels`. `scripts/ceiling_fullbatch.py:58` uses `Σ_{nodes,labels}/(|nodes|·121)`. The two objectives differ by a factor of the batch size, i.e. a different effective learning rate. The script claims the two agree to four decimals on PPI (`ceiling_fullbatch.py:11-12`) — plausible because Adam is roughly scale-invariant — but the claim is empirical and untested in CI. It also changes how hard the DP clip at C bites, since the per-root gradient norm is 121× smaller than a summed one.

### 13. LOW — `ceiling_fullbatch.py` contradicts itself about the degree cap and hardcodes higher-is-better
It has no `--K_in`/cap option at all and evaluates on the uncapped `data.edge_index`, yet the docstring says *"Measured on PPI with the same capped graph"* (`:11`). Also `:145` uses `mean > trivial` unconditionally; harmless today only because `MECHANISMS` (`:46-50`) excludes `regression_gnn`, but it will silently invert if that changes. Its CSV header (`:136-141`, 29 columns) is called *"Same schema as src.sparse.run"* (`:135`) but is a strict subset of run.py's 65-column header — `summarize_sweep` survives only because the `metric` column happens to be present.

### 14. LOW — `summarize_sweep.py` minor sharp edges
- `:117-118` always prints a column literally headed `auroc` pulled from `test_auroc`; for a regression sweep it is always `-` and the more useful `test_r2` is never shown.
- `:119` `if eps.get(best)` treats `epsilon == 0.0` as missing.
- `:83-84` takes `csvs[0]` only — a cell directory with more than one results CSV silently ignores the rest.
- `:108-109` `if best not in curve: continue` drops the whole cell with no message.
- `--metric val_acc` (no `test_` prefix) triggers the "leakage" warning even though selecting on val and reporting val is fine (`:95-96`).

### 15. LOW — GAD side pipeline has no validation split and uses test positives to set k
`src/sparse/gad/run.py:126-131` evaluates only on `test_mask`; there is no val curve and no checkpoint selection (XGBoost fits once). `rec_at_k(yt, st)` with `k=None` defaults to the number of positives in `y_true` (`gad/metrics.py:38-39`) — GADBench's convention, but it is a test-set statistic. Also note the asymmetry with the main pipeline: the GAD XGBoost detector *does* correct class imbalance (`xgb_graph.py:83-86`, `scale_pos_weight = n_neg/n_pos` from the train split only), while none of the five DP mechanisms do.

---

## Direct answers to the numbered questions

**Q7 — is any metric computed on TEST during training in a way that influences a decision?**
In the **sparse** pipeline: no. `train_sparse_gnn` computes test at every tracked checkpoint (`sparse_gnn.py:208-213`) and in the verbose print (`:215-218`), but nothing consumes it — there is no early stopping, no best-state save, no lr selection in `src/sparse/run.py`, and the final returned metrics are the last step's. `checkpoint_callback` is never wired to a decision by any caller. The console sweep table sorts by test (`run.py:690-692`) but is display-only. Downstream, `summarize_sweep.py` selects on val.
In the **experiments** pipeline: selection is on val (`baselines.py:127`, `dpar.py:349`) — correct in principle, but with the inverted direction of risk #1 for regression.
Residual test-set exposure: `trivial_baseline` reads test labels (risk #7), and `rec_at_k`'s k comes from test positives (risk #15).

**Q8 — trivial-baseline floor: code or prose?**
Code. `src/sparse/run.py:90-118`, recorded per-row in the `trivial_baseline` CSV column and printed at run start; reused by `scripts/ceiling_fullbatch.py:101` and cross-checked by `scripts/diagnose.sh metrics`. The PPI 0.4608 figure is reproduced exactly by the `micro_f1` branch. **Class imbalance is not handled** anywhere in the five mechanisms — no `pos_weight`, no class weights, for either our method or the MLP/blind baselines, so the two arms are symmetric on that axis. The only asymmetry is the GAD XGBoost detector's `scale_pos_weight` (risk #15), which is a different pipeline.

**Q9 — what the `e4363e5` guard actually guards, and is it complete?**
See risk #10. Short version: it converts an opaque `cross_entropy`-on-floats dtype crash into a clear `SystemExit`, for RelBench regression tasks reached through `src/sparse/run.py` only. It is incomplete in three ways — it depends on a `task_type` attribute only RelBench sets, it does not catch the reverse mispairing (`regression_gnn` on a classification task, which fails silently rather than loudly), and the sibling `src/experiments/run.py` driver has no equivalent guard for DPAR.

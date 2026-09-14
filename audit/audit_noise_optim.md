# Audit: DP-SGD mechanics (clipping, noise, optimizer, LR, training loop)

Repo: `/Users/kevinjacob/markovian_dp copy`, branch `sparse_expand_clean`, HEAD `b10a794`.
Scope: DP-SGD mechanics only. Accounting math and dataset construction are out of scope.
Read-only: no files were modified. Empirical probes were run in-process (no files written to the repo).

---

# (A) What the code does

## 1. Clipping

**Per-rooted-subgraph, global L2 over the flattened full gradient, clip-if-exceeds.**

- The clipping call sits *inside* the per-loss loop, one call per sampled root:
  `src/sparse/sparse_gnn.py:85-93`
  ```python
  for i, loss_H in enumerate(losses):
      grads = torch.autograd.grad(loss_H, params, retain_graph=i < len(losses) - 1, allow_unused=True)
      grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
      clipped = mechanism.clip_flat_grad(grads, C)
      for acc, g in zip(grad_accum, clipped):
          acc.add_(g)
  ```
- The norm is the **global** L2 across *all* parameter tensors concatenated (not per-tensor):
  `src/sparse/base_mechanism.py:156-164`
  ```python
  total_sq = torch.stack([g.pow(2).sum() for g in grads]).sum()
  coef = (C / (total_sq.sqrt() + 1e-12)).clamp(max=1.0)
  return [g * coef for g in grads]
  ```
  `.clamp(max=1.0)` makes this `min(1, C/||g||)` — **clip-if-exceeds**, not scale-to-C. A gradient with norm < C passes through unchanged (asserted at `tests/test_dp_mechanics.py:76-83`).
- `+1e-12` in the denominator makes the effective cap infinitesimally below C (conservative direction; harmless).
- Per-subgraph (not per-batch) is verified by test: `tests/test_dp_mechanics.py:86-99` (4 aligned roots sum to 4C, not C).
- The non-DP path performs **no clipping at all** (`sparse_gnn.py:39-54`).

## 2. Noise

- Drawn once per step, after the whole per-root sum is accumulated: `sparse_gnn.py:98-99`.
- Distribution: `std = sigma * C`, drawn on CPU with the seeded generator, then moved to device — `base_mechanism.py:166-197`.
- **Division happens AFTER noise** (`sparse_gnn.py:101-102`):
  ```python
  for p, acc, z in zip(params, grad_accum, noise):
      p.grad = (acc + z) / denom
  ```
  So the released update is `(Σ_i clip(g_i) + N(0,(σC)²I)) / E[B]` — the standard DP-SGD ordering. Effective per-coordinate noise std in the optimizer-facing gradient is `σC/E[B]`, matching a mean-gradient with noise-to-signal `σC/B`. (Dividing *before* the noise would have meant `E[B]×` too much noise; that is not what happens.)
- On an **empty root set** the DP step still runs and still adds noise (`sparse_gnn.py:72-79`, comment at `196-200`), which is what the accountant's fixed-T composition needs. Tested at `tests/test_dp_mechanics.py:162-193, 263-269`.
- RNG: two independent CPU generators seeded from the run seed — sampling `seed`, noise `seed + 10_000` (`sparse_gnn.py:181-182`). Noise is reproducible across runs at a fixed seed, independent of the sampling stream (`tests/test_dp_mechanics.py:226-241`), and fresh every step (`214-223`).
- **However the noise is NOT i.i.d. across parameter tensors — see risk HIGH-1.**

## 3. Sum-vs-mean (the classic leak)

**The denominator is safe — it is `p1 × |pool|`, not the realized `|V_root|`.**

`src/sparse/sparse_gnn.py:184-186`
```python
pool_size = (num_nodes if candidate_nodes is None else int(candidate_nodes.numel()))
expected_batch = p1 * pool_size
```
`src/sparse/sparse_gnn.py:70`
```python
denom = max(float(expected_batch), 1.0)
```
`candidate_nodes` is fixed before training (`run.py:509-511`, `torch.where(data.train_mask)[0]`), so `denom` is constant over all T steps and independent of which roots were drawn. There is an explicit regression test for this: `tests/test_dp_mechanics.py:305-319`. **No data-dependent normalization in the SparseGNN engine.** (Residual caveat: `|train pool|` is itself treated as public, the standard DP-SGD assumption.)

The same cannot be said of the DP-MLP **baseline** — see MED-4.

## 4. Optimizer

`base_mechanism.py:97-112` is the only construction site for the engine:
```python
if kind == "adam":  torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
elif kind == "sgd": torch.optim.SGD(self.module.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
```
Selection in `run.py:618-619`:
```python
opt_kind = (args.optimizer if args.optimizer != 'auto' else ('sgd' if args.dp else 'adam'))
```
- `--optimizer` default `auto` (`run.py:224-229`): **Adam for non-DP, SGD for DP**.
- Momentum reaches SGD only; Adam ignores it. Default `0.0` (`run.py:231`). Only `scripts/sweep.sh:54` ever sets it non-zero (momentum axis, M ∈ {0.0, 0.9}).
- All mechanisms share this one code path — there is no per-mechanism optimizer.
- Drivers that **pin** the optimizer: `sbatch/ppi_matched_eps.sbatch:71` (`sgd`), `sbatch/ppi_matched_eps_adam.sbatch:64` (`adam`), `sbatch/relamazon_scope.sbatch:98,99,113`, `sbatch/relbench_regression_meps.sbatch:120,121,132,144`, `sbatch/arxiv_inductive_matched_eps.sbatch:82,85,97,109`, `scripts/_coverage_sweep.sh:95,97`, `scripts/_facebook_width.sh:78,95`, `scripts/_arxiv_r2_diag.sh:39`, `scripts/_ppi_stage1_diag.sh:51`, `scripts/sweep.sh:84`.
- Drivers that **do not** pin it (so `auto` decides): `scripts/ladder_stage01.sh`, `ladder_stage2.sh`, `orientation_ablation.sh`, `relbench_f1.sh`, `sbatch/sparse_inductive.sbatch`, `sbatch/sparse_relbench.sbatch`, `sbatch/reddit_settings.sbatch`, `sbatch/facebook_settings*.sbatch`, `scripts/run_sparse_inductive_ablation.py`.
- Baseline harness: unconditionally Adam — `src/experiments/baselines.py:110`, `src/experiments/dpar.py:302`, `src/experiments/dpgnn.py:220`. `scripts/ceiling_fullbatch.py:110-111` is hardcoded `kind='adam'`.

### `ppi_matched_eps_adam.sbatch` vs `ppi_matched_eps.sbatch`
Diff is exactly two things, exactly as the header claims:
| | `ppi_matched_eps.sbatch` | `ppi_matched_eps_adam.sbatch` |
|---|---|---|
| `--optimizer` | `sgd` (:71) | `adam` (:64) |
| `--lr` | `0.3` (:101, :115) | `0.01` (`ADAM_LR`, :57, :90, :103) |
| `OUT_ROOT` | `results/ppi_matched_eps` | `results/ppi_matched_eps_adam` |
| everything else (P1 0.0114, T 2000, K 5, HIDDEN 256, GRID 1e-5, EPS_LIST, P2_LIST, N_NODES, dropout 0.0, wd 0.0, clip 1.0, seeds, track_every 100) | identical | identical |
Both call `calibrate_grid.py` with identical arguments, so σ per cell is identical. This is a clean one-variable (optimizer+its lr) comparison, as advertised.

## 5. Learning rate

- CLI default `--lr 0.01` (`run.py:230`). **No scheduler, no decay, no warmup anywhere** — `grep -rn "lr_scheduler|StepLR|Cosine" --include=*.py` returns nothing outside `third_party/`.
- lr is applied identically in the DP and non-DP paths, but the **gradient scale differs by ~E[B]** between them (see HIGH-2), so identical `--lr` means very different step sizes.
- Shared defaults in `scripts/_dataset_settings.sh:115-118`:
  ```
  LR_NONDP=0.01      # Adam
  LR_DP=${LR_DP:-1.0}   # SGD
  ```
  with per-dataset overrides `LR_DP=0.3` for facebook (`:91`) and reddit (`:98`).
- Observed DP lr in drivers: **0.3** everywhere except `sbatch/sparse_inductive.sbatch:55,86` and `sbatch/sparse_relbench.sbatch:77` (**1.0**), and `scripts/run_sparse_inductive_ablation.py` (**0.01**, the CLI default, never overridden).
- Observed non-DP lr: **0.01** (Adam) everywhere except facebook's `--lr 0.05` (`sbatch/facebook_settings.sbatch:59,69`, `facebook_settings_ice.sbatch:67,77`).
- Baselines: `baselines.py:23` lr 1e-2; `dpgnn.py:34` lr 3e-3; `dpar.py:49` lr 5e-3. All Adam. So the DP-vs-baseline comparison is our-SGD-at-0.3 vs their-Adam-at-0.003…0.01.

## 6. Other regularization

| Mechanism | Present? | Where / default |
|---|---|---|
| Dropout | yes | `--dropout` default **0.5** (`run.py:223`); applied between conv layers only, not after the last (`gnn_mechanism.py:36-41`, mirrored in the other three GNN mechanisms + `mlp_mechanism.py:37-43`). Every shipped driver sets `--dropout 0.0` (`scripts/_dataset_settings.sh:51`, all sbatch COMMON blocks) **except** `scripts/run_sparse_inductive_ablation.py`. |
| Weight decay | yes | `--weight_decay` default **5e-4** (`run.py:234`), passed straight to Adam/SGD (`base_mechanism.py:102,108`). Applied by the optimizer to the *already-divided* gradient (post-processing, no privacy cost). Drivers set 0.0. |
| Momentum | yes (SGD only) | `--momentum` default **0.0** (`run.py:231`). |
| BatchNorm / LayerNorm | **none** | no normalization layers in `layers.py`, `_NodeGNN`, `_MultiLabelGNN`, `_BinaryGNN`, `_RegressionGNN`, `_MLP`. |
| Early stopping | **none in the training loop** | but the reported number is the best *tracked* checkpoint chosen on validation, post-hoc, in `scripts/summarize_sweep.py:103-107`. See MED-3. |
| Gradient accumulation | **none** (in the DP sense) | forward passes are chunked (`gnn_mechanism.py:123-142`) but every root still gets its own `autograd.grad` + its own clip. |
| Gradient/global-norm clipping of the *aggregate* | none | only the per-root clip. |

## 7. Batching

**Genuinely Poisson, one optimizer step per sampled set, no microbatches.**

Sampling line — `src/sparse/sparse_expand.py:338-344`:
```python
pool = (torch.arange(num_nodes) if candidate_nodes is None else candidate_nodes.cpu())
n = int(pool.numel())
if p1 >= 1.0:
    return pool.clone()
keep = torch.rand(n, generator=generator) < p1
return pool[keep]
```
Independent Bernoulli(p1) per node — **not** a fixed-size sample, so the amplification claim is not invalidated by the sampler. Verified empirically by test (`tests/test_dp_mechanics.py:274-282` asserts Binomial variance `np(1-p)`, which a fixed-size sampler would fail), and independence across steps at `285-290`.

Batch composition: `sparse_gnn.py:190-194` draws the roots and expands each one, then hands the whole list to one step. Inside the DP step, `iter_subgraph_loss_batches` groups roots into **forward-pass chunks** for speed:
- `GNNMechanism` overrides it (`gnn_mechanism.py:123-142`) to pack disconnected subgraphs into one GNN forward, bounded by `max_batched_subgraph_nodes` (default **8192**, `gnn_mechanism.py:58`, **not exposed on the CLI**). Because the components are disconnected, the per-root `autograd.grad` still yields the exact per-root gradient — asserted at `tests/test_sparse_batching.py:53-72`.
- Every other mechanism (`multilabel_gnn`, `binary_gnn`, `regression_gnn`, `mlp`) falls back to `base_mechanism.py:137-140`, which yields **all** losses in one list (one forward per root, all graphs retained). No correctness issue, memory issue — MED-8.

Either way this is *forward* batching, not DP microbatching: clipping granularity is always one root.

## 8. Step count T, and what a step is

- `sparse_gnn.py:189` `for t in range(1, T + 1)` — **one Poisson draw + one optimizer step per t**. A "step" is one gradient update, not an epoch.
- DP path: **exactly T** optimizer steps, always (empty batches still step on noise, `sparse_gnn.py:196-202`).
- Non-DP path: **≤ T**, because `sparse_gnn.py:204-205` does `if roots.numel() == 0: continue`. Irrelevant at realistic `p1·|pool|` but a documented asymmetry.
- `--track_every` (`run.py:292`, default 0) triggers `_evaluate` at `sparse_gnn.py:208-213`. **It does not change the trajectory** — evaluation is `@torch.no_grad()`, runs in `eval()` mode so `F.dropout(..., training=False)` draws nothing, and the conv layers consume no RNG. Verified empirically: with `track_every ∈ {0,5}` × `dropout ∈ {0.0,0.5}` × `dp ∈ {False,True}`, final parameters were **bit-identical** (max|Δ| = 0.000e+00) in all four cases. The docstring claim at `sparse_gnn.py:164-167` holds.
- `--eval_every` (default 50) / `--progress_every` only print, and only under `--verbose` (`sparse_gnn.py:215-218`).

## 9. The non-DP path

**A separate implementation** (`_step_nondp`, `sparse_gnn.py:39-54`), not "the DP path with σ=0". Differences:

| | non-DP `_step_nondp` | DP `_step_dp` |
|---|---|---|
| gradient | `Σ_H loss_H` → **one** `backward()` | per-root `autograd.grad`, clipped, summed |
| clipping | none | per-root to C |
| normalization | **none — raw sum** (`:47-52`) | `/ max(p1·pool, 1)` (`:70,102`) |
| empty batch | skip the step (`:204-205`) | noise-only step (`:72-79`) |
| optimizer under `auto` | Adam | SGD |

The mathematical content is otherwise identical (same per-root losses, same batched forward), so with C=∞, σ=0 and the same normalization the two would agree. As shipped they differ in gradient scale by a factor of `E[B]` and in optimizer — see HIGH-2.

## 10. Numerics

- Everything is float32 (`torch.get_default_dtype()`); the noise staging buffer is allocated with `torch.get_default_dtype()` and keyed only on `(shape, device)`, **not dtype** (`base_mechanism.py:180-188`) — fine today, fragile if a mechanism ever holds mixed dtypes.
- In-place ops: `acc.add_(g)` operates on a fresh `zeros_like` accumulator with `g` a fresh tensor from `clip_flat_grad` (`g * coef`) — safe. `p.grad = (acc + z) / denom` materializes a new tensor before `z` can be reused — safe.
- `torch.autograd.grad` never touches `.grad`, so `opt.zero_grad()` at `:100` (after accumulation) is harmless; no stale-gradient contamination.
- Clipping is strictly before the step; nothing clips after `opt.step()`.
- `float(running)` at `sparse_gnn.py:104` forces a device sync every step (perf only).
- `clip_flat_grad` keeps `coef` as a 0-dim tensor (no `float()` sync) — an improvement over the pre-`ff98371` version.

## Hyperparameter table (every `src/sparse/run.py` CLI flag)

| Flag | run.py line | Default | Overridden where |
|---|---|---|---|
| `--dataset` | 167 | `citeseer` | every driver |
| `--model` | 169 | `gnn` | `multilabel_gnn` (ppi), `binary_gnn`/`regression_gnn` (relbench), `mlp` (blind arm) |
| `--aggr` | 180 | `mean` | never (always `mean`) |
| `--relbench_root` | 185 | `row` | relbench sbatch |
| `--relbench_reverse_edges` | 188 | off | — |
| `--inductive` | 190 | off | relbench/arxiv/reddit/flickr drivers |
| `--common_inductive_split` | 195 | off | `run_sparse_inductive_ablation.py:43` |
| `--split_root` | 198 | `data/inductive_splits` | — |
| `--split_seed` | 200 | 0 | — |
| `--direction` | 202 | `in` | `out` only in `orientation_ablation.sh` |
| `--p1` | 209 | `[0.5]` | 0.0114 (ppi), 0.013 (fb), 0.002 (reddit), 0.005 (default ds), 0.05 (relbench) |
| `--p2` | 215 | `[0.5]` | grids `1.0 0.5 0.25 0.1` |
| `--r` | 215 | `[2]` | 0 / 1 / 2 |
| `--T` | 218 | 200 | 2000 (ppi), 900 (relbench), 500–1000 |
| `--hidden` | 221 | 64 | 256 (ppi, relbench), 16 (facebook) |
| `--num_layers` | 222 | 2 | always 2 |
| `--dropout` | 223 | **0.5** | **0.0** in every driver except `run_sparse_inductive_ablation.py` |
| `--optimizer` | 224 | **`auto`** (adam if non-DP, sgd if DP) | pinned in ~10 drivers; left `auto` in ~9 |
| `--lr` | 230 | **0.01** | 0.3 DP / 0.01 non-DP mostly; 1.0 (sparse_inductive, sparse_relbench); 0.05 (facebook non-DP) |
| `--momentum` | 231 | 0.0 | 0.9 only in `sweep.sh:54` |
| `--weight_decay` | 234 | **5e-4** | **0.0** in every driver except `run_sparse_inductive_ablation.py` |
| `--roots_from` | 235 | `train` | — |
| `--dp` | 238 | off | DP arms |
| `--clip` | 239 | 1.0 | `sweep.sh` clip axis (0.1–1.0) |
| `--sigma` | 241 | `[1.0]` if unset | per-cell from `calibrate_grid.py`; grids `2 5 10 20` |
| `--target_epsilon` / `--target_delta` | 244/247 | None | unused by shipped drivers (they pre-calibrate σ instead) |
| `--accounting_theorem` | 249 | `auto` | — |
| `--accounting_grid` | 252 | 1e-4 | 1e-5 via `calibrate_grid.py --grid` |
| `--calibration_rtol` / `--calibration_atol` | 254/256 | 1e-3 / 1e-6 | — |
| `--K_in` | 258 | **None** | 5 (ppi/fb/reddit/default), 20 (relbench) |
| `--K_out` | 262 | None → `K_in` | 5 / 3 |
| `--cap_mode` | 264 | `auto` | — |
| `--cap_seed` | 272 | None → run seed | — |
| `--eval_graph` | 279 | `auto` (`train` transductive, `full` inductive) | — |
| `--track_every` | 292 | 0 | 100 (ppi), 50 (`sweep.sh`) |
| `--seeds` | 299 | 3 | 1–3 |
| `--out_dir` | 300 | `results` | per cell |
| `--plot` / `--verbose` | 301/303 | off | — |
| `--progress_every` | 304 | None → `eval_every` | 500 |
| `--eval_every` | 307 | 50 | — |
| *(not a flag)* `max_batched_subgraph_nodes` | `gnn_mechanism.py:58` | 8192 | never |

---

# (B) Open questions / discrepancies / risks

## HIGH-1 — DP noise is NOT independent across same-shaped parameter tensors; a noise-free linear functional of the clipped gradient sum is released every step
`src/sparse/base_mechanism.py:175-196`

The per-step noise buffer cache is keyed on `(shape, device)` and the **cached tensor object itself is appended to the output list**:
```python
key = (tuple(grad.shape), grad.device)
buffers = cache.get(key)
...
cpu.normal_(generator=generator)
if device is cpu: cpu.mul_(std)
else: device.copy_(cpu, non_blocking=True); device.mul_(std)
noise.append(device)          # <-- same object for every grad with the same shape
```
Two parameter tensors of the same shape therefore receive **the identical noise draw** (the last one written), not two independent draws.

Confirmed empirically on the real models (probe run in-process, `aggr` default `mean` = `SAGEConv`):

| config | params | unique noise tensors | aliased |
|---|---|---|---|
| `aggr=mean, L=2` | 6 | 4 | **2** (`convs.0.lin_l.weight` ≡ `convs.0.lin_r.weight`; `convs.1.lin_l.weight` ≡ `convs.1.lin_r.weight`) |
| `aggr=mean, L=3` | 9 | 5 | **4** (also `convs.0.lin_l.bias` ≡ `convs.1.lin_l.bias`) |
| `aggr=gcn, L=2` | 4 | 4 | 0 |
| `aggr=gcn, L=3` | 6 | 5 | 1 (`convs.0.bias` ≡ `convs.1.bias`) |

`SAGEConv` always has `lin_l.weight` and `lin_r.weight` of identical shape `[out, in]`, so **every layer of every `aggr=mean` model is affected**, and `mean` is the default and is what every driver uses. On PPI (in=50, hidden=256, out=121) that is 43,776 of ~87,900 coordinates — about half the model — carrying duplicated noise.

End-to-end demonstration through the real `_step_dp` with `MultiLabelGNNMechanism`, σ=100, C=1:
```
per-coordinate released-grad std:            45.73
std of (lin_l.grad - lin_r.grad):             0.0442   # would be ~70.7 if independent
max| noisy_diff - noiseless_diff |:           6.7e-06  # i.e. zero, up to float32
```
The projection `Σ_i clip(g_i)|_{lin_l} − Σ_i clip(g_i)|_{lin_r}` is released **in the clear**, every step, for T steps. Its per-root sensitivity is up to 2C. The released vector's noise covariance is singular, so the Gaussian-mechanism premise (`N(0,(σC)²I)` on the full flattened gradient, `base_mechanism.py:169`, and the paper's Assumption 6.3) does not hold and **every reported ε is invalid for the mechanism actually run**, regardless of whether the accounting math is right.

Utility is essentially unaffected — the *marginal* per-coordinate law is still `N(0,(σC)²)` — so this will not show up as an anomaly in any results CSV.

**Regression, not a long-standing bug.** `git show ff98371^:src/sparse/base_mechanism.py` shows the previous implementation was correct:
```python
return [(torch.randn(g.shape, generator=generator) * std).to(g.device) for g in grads]
```
Commit `ff98371` "Restore archived DP experiments and runtime sources" (2026-09-08 23:12) introduced the buffer cache. **98 of 335 result CSVs postdate that commit** (all of `results/relbench/rel{hm,amazon,arxiv,avito}_*_meps/`, written Sep 9–10), and they all used `--aggr mean`. Anything else was produced by the correct code.

Fix is one line: return a copy (`noise.append(device.clone())` defeats the point of the cache; better to key the cache on `(shape, device, index)` or just drop the cache and go back to `torch.randn`). Note the aliasing also makes the *seeded noise stream* differ from the pre-`ff98371` stream even after a fix, so old and new runs will not be bitwise comparable.

**Test-coverage gap that let this through:** every mechanism in `tests/test_dp_mechanics.py` is `_FixedGradMechanism` = `nn.Linear(dim, 1, bias=False)` — a **single** parameter tensor (`:37-38`), so the aliasing path is never exercised. All 31 tests in `test_dp_mechanics.py` + `test_sparse_batching.py` pass on the buggy code.

## HIGH-2 — DP and non-DP runs differ by an ~E[B]× gradient rescale *and* an optimizer swap, so "DP vs non-DP" is a three-variable comparison
`sparse_gnn.py:47-53` vs `sparse_gnn.py:70,102`; `run.py:618-619`

Non-DP takes the **raw sum** of per-root losses with no normalization; DP divides by `E[B] = p1·|pool|`. On PPI `E[B] ≈ 512`, so at the same `--lr` the non-DP path takes ~512× larger steps. On top of that `--optimizer auto` gives Adam to non-DP and SGD to DP. The drivers partly compensate by hand (`LR_NONDP=0.01` Adam / `LR_DP=0.3` SGD, `_dataset_settings.sh:115-118`), but that is a tuned patch, not an equalization: nothing in the repo checks that the two are comparable.

Concretely, the PPI headline in `sbatch/ppi_matched_eps.sbatch:120` — *"non-DP references (Stage 1): GNN r=1 0.6256 | blind r=0 0.5314"* — comes from `scripts/_ppi_stage1_diag.sh:51` (`--optimizer adam --lr 0.01`, summed gradient) and is compared against DP cells run with `--optimizer sgd --lr 0.3` and a `/512` gradient. Any gap attributed to "the noise" also contains the optimizer change and a ~15,000× difference in nominal step scale.

`ppi_matched_eps_adam.sbatch` is the right instinct and removes the optimizer confound *within the DP arm*, but the non-DP reference it is compared to is still the Adam/summed-gradient Stage-1 number. The cleanest fix is to normalize `_step_nondp` by the same `denom` so the two paths are on one scale, then compare at matched optimizer (which `--optimizer sgd` already allows).

## MED-3 — Reported numbers are a best-of-20 checkpoint selected on validation, with no privacy cost charged for the selection
`scripts/summarize_sweep.py:91-121`, driven by `--track_every 100` at `run.py:292`

`summarize_sweep.py` picks `best = max(val_curve, ...)` over every tracked checkpoint and reports test *and* ε at that step (`:103-121`). Two issues:
1. The selection uses validation labels, which under node-level DP are part of the protected data. That is post-hoc early stopping on private data; its cost is not in the accountant. (The module docstring at `:12-16` correctly identifies test-selection as leakage but treats val-selection as safe — under node-DP it is not obviously safe, since the val nodes are nodes of the same protected graph.)
2. The ε shown is ε(t*) for the chosen step, but the mechanism actually *ran* to T and every intermediate model was materialized; pairing "best of 20" with "ε at that one step" understates the released budget. (This one overlaps the accounting agent's scope — flagging so it is not dropped between us.)

## MED-4 — The DP-MLP baseline *does* have the data-dependent-denominator bug
`src/experiments/baselines.py:173`
```python
parameter.grad = (accumulator + noise) / selected.numel()
```
`selected` is the realized Poisson sample (`:154`), so the denominator is the realized batch size — exactly the leak the SparseGNN engine correctly avoids. The baseline's accountant (`:143-147`) charges plain Poisson-amplified DP-SGD, which does not model the data-dependent scaling. Same pattern in `src/experiments/dpar.py:385-390` (`/ len(roots)`), though there `roots` comes from a fixed-size `randperm` so the denominator is at least constant.

Not our mechanism, but it is the baseline our mechanism is compared against, and the asymmetry cuts in our disfavour in the write-up only if someone notices.

## MED-5 — The DP-MLP baseline skips the step entirely on an empty Poisson sample
`src/experiments/baselines.py:155-156` (`if not selected.numel(): return`) — no noise, no step, while the accountant at `:146` charges `epochs * steps_per_epoch` steps regardless. The SparseGNN engine gets this right (`sparse_gnn.py:196-202`). Low practical impact at batch 256 / N large, but it is the same class of mismatch.

## MED-6 — The DPAR baseline uses fixed-size shuffled minibatches but is accounted with Poisson amplification
`src/experiments/dpar.py:338` (`torch.randperm(...)`) vs `dpar.py:404` (`sample_rate=min(config.batch_size / ppr_releases, 1.0)`). Shuffling is not Poisson subsampling, so the amplification is not licensed. The code comment says this matches upstream DPAR, so it may be a deliberate fidelity choice — but if the paper reports DPAR at a stated ε next to SparseGNN at an ε derived from a genuinely-Poisson sampler, that asymmetry should be stated. Our own sampler is genuinely Poisson (`sparse_expand.py:343`), so this is a baseline-side caveat only.

## MED-7 — The degree-cap RNG and the root-sampling RNG are seeded with the same integer, so they draw the identical number sequence
`run.py:444` `cap_gen = torch.Generator().manual_seed(int(cap_seed))` with `cap_seed = s` (`run.py:501`), and `sparse_gnn.py:181` `sample_gen = _make_generator(seed)` with `seed = s` (`run.py:631`). Both are fresh `torch.Generator`s from the same seed, so `torch.rand(E)` in the cap and `torch.rand(|pool|)` in the first root draw consume the *same* uniform values. Which arcs survived the cap and which nodes are roots at step 1 are therefore deterministically coupled. The accounting assumes the Bernoulli root draws are independent of everything else. The noise generator is safely offset (`seed + 10_000`, `sparse_gnn.py:182`), so this does not touch the noise — but it is free to fix (offset `cap_seed` the same way) and the sampling independence is load-bearing for the amplification argument.

## MED-8 — Every mechanism except `GNNMechanism` retains all per-root autograd graphs simultaneously
`base_mechanism.py:137-140` yields `self.subgraph_losses(subgraphs)` — all losses computed eagerly, all graphs alive — while `GNNMechanism` streams bounded chunks (`gnn_mechanism.py:123-142`). PPI (`--model multilabel_gnn`, the headline experiment) therefore holds ~512 independent 787-node forward graphs per step. This is the likely cause of the "~2.4 s/step" noted in `scripts/ceiling_fullbatch.py:16` and of the wallclock timeout described in `sbatch/ppi_matched_eps.sbatch:88-91`. Correctness is unaffected; a `MultiLabelGNNMechanism` batched-chunk override would be a large speedup for free.

## MED-9 — `run.py`'s regularization defaults disagree with every driver, and one driver silently uses the defaults
`run.py:223` `--dropout 0.5` and `run.py:234` `--weight_decay 5e-4` are the Planetoid values. `scripts/_dataset_settings.sh:40-47` documents that these cost PPI 7 points of micro-F1 and that all runs use 0.0/0.0. But `scripts/run_sparse_inductive_ablation.py:41-52` builds its command line without `--dropout`, `--weight_decay`, `--optimizer`, or `--lr`, so that ablation runs with dropout 0.5, wd 5e-4, SGD, **lr 0.01** — a 30× smaller DP learning rate than every other DP cell in the repo. Any cross-comparison of `results/inductive/sparse_ablation` against the sbatch results is confounded four ways.

## MED-10 — Pinned-buffer/async-copy race on GPU
`base_mechanism.py:183-195`: the CPU staging buffer is pinned and copied with `non_blocking=True`, then reused on the next step by a host-side `cpu.normal_()` with no `torch.cuda.synchronize()` or event guard. If the CUDA queue is deep, the host can overwrite the pinned buffer before the H2D copy executes, corrupting that step's noise. The shipped sbatch jobs target `ice-cpu` (`#SBATCH -p ice-cpu`) where `device is cpu` and this cannot fire, but any GPU run is exposed. (A correct fix for HIGH-1 that drops the cache removes this too.)

## LOW-11 — Weight decay is applied to the `1/E[B]`-scaled gradient
`base_mechanism.py:102,108` hands `weight_decay` to the optimizer, which adds `wd·θ` to the *already divided* gradient. So the decay-to-signal ratio in DP runs is `E[B]×` larger than the same `--weight_decay` would give in a mean-gradient setup. Harmless with the drivers' `0.0`, but a trap for anyone who runs with defaults. No privacy cost (data-independent).

## LOW-12 — `denom = max(E[B], 1.0)` silently changes the effective lr for small pools
`sparse_gnn.py:70`. When `p1·|pool| < 1` the update is the raw noisy sum rather than a mean. Fine for every shipped config (smallest is `p1·|pool| ≈ 68` for relbench), but the clamp is undocumented in the docstring at `:62-65`.

## LOW-13 — `--roots_from all` divides by `p1·N` while unlabeled roots contribute exactly zero
`sparse_gnn.py:184-186` uses the full node count when `candidate_nodes is None`, but roots outside `train_mask` return `zero_loss()` (`gnn_mechanism.py:72-74` and the three siblings). The effective step is therefore scaled down by the labeled fraction. Correct for privacy (the denominator stays public), but it means `--roots_from all` and `--roots_from train` are not on the same learning-rate scale. Default is `train`, so this is latent.

## LOW-14 — The non-DP path takes fewer than T updates
`sparse_gnn.py:204-205`. At `p1·|pool| ≈ 512` the probability of an empty draw is ~0, so this never bites in practice, but the DP and non-DP arms nominally run different numbers of updates at the same `--T`.

## LOW-15 — `scripts/ceiling_fullbatch.py` claims a schema it no longer has, and optimizes a mean loss
`ceiling_fullbatch.py:135` says *"Same schema as src.sparse.run so downstream analysis is unchanged"*, but its header (`:136-141`, 29 columns) is missing `optimizer`, `target_epsilon`, `roots_from`, `hidden`, `dropout`, `weight_decay`, `seeds`, `cap_seed`, `K_*_achieved`, the `_rmse`/`_r2`/`_bin_acc` columns and all `_alt` columns that `run.py:535-565` writes. Separately, `node_loss` (`:53-59`) uses `F.nll_loss`/`binary_cross_entropy_with_logits` at default `reduction='mean'`, whereas the engine sums per-root losses — so the "ceiling" is trained at a `|train|×` different gradient scale from the thing it is a ceiling for. Adam largely absorbs this, and the docstring's measured agreement (0.5463 vs 0.5464) suggests it does, but the claim of equivalence is stronger than what the code establishes.

## LOW-16 — The `retain_graph` flag can be spent on a `zero_loss` graph
`sparse_gnn.py:87`: `retain_graph = i < len(losses) - 1`. If the last loss in a chunk is a `zero_loss()` (unlabeled root — its own tiny graph, `base_mechanism.py:147-152`), the final `autograd.grad(retain_graph=False)` frees that graph while the shared batched-forward graph stays alive until Python drops the `losses` list. Memory only, and only reachable with `--roots_from all`.

## LOW-17 — The DP-GNN baseline clips **per parameter tensor**, ours clips the flat vector
`src/experiments/dpgnn.py:198-207` clips each tensor to its own adaptive `threshold` and adds noise scaled per-tensor; `src/experiments/dpgnn_adapter.py:68` describes this as *"per-parameter clipping and Gaussian DP-Adam"*. Ours is a single global-norm clip. Both are valid mechanisms, but "same C" does not mean "same sensitivity budget" across the two, which matters if any table presents them at a matched clip norm.

## LOW-18 — Noise-buffer cache keyed on `(shape, device)` but allocated with `torch.get_default_dtype()`
`base_mechanism.py:180-188`. A float64 or half-precision parameter of a shape already in the cache would get the wrong-dtype buffer, or a dtype mismatch on `copy_`. Nothing in the repo does this today.

## LOW-19 — `float(running)` syncs the device every step
`sparse_gnn.py:104`, purely to produce a number that is only consumed by the `--verbose` print at `:217-218`. Cheap to make lazy.

## LOW-20 — `train_sparse_gnn_with_budget` is effectively dead code
`sparse_gnn.py:227-272` is exported (`src/sparse/__init__.py:25,45`) and tested (`tests/test_sparse_privacy_calibration.py:27-43`), but `run.py` calibrates inline at `:571-579` and calls `train_sparse_gnn` directly at `:626`. Two calibration call sites that can drift apart; the shipped drivers use neither (they pre-solve σ with `scripts/calibrate_grid.py` and pass `--sigma`).

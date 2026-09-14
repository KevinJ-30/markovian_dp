# Codebase audit — running list

Started 2026-09-11 on branch `sparse_expand_clean` @ `b10a794`.

A live inventory of *how the pipeline actually behaves*, and where that
disagrees with the README, the drivers, or `paper/experiments_current.tex`.
Every claim carries a `file:line`. Items are not fixes — they are things to
decide on.

Severity: **HIGH** = affects a reported number or a privacy claim ·
**MED** = affects comparability across runs · **LOW** = documentation drift.

**Scope separation.** §0–§8 are *our* method (`src/sparse/`). §9 is the
baselines (`src/experiments/`, `third_party/`), kept separate on purpose.

Everything below was either verified directly against the code or measured.
Where a number came from a probe rather than a read, it says "measured".

Longer backing detail — full formula walk-throughs, per-flag tables, and the
lower-severity items that did not make this list — is in `audit/`:
`audit_accounting.md`, `audit_noise_optim.md`, `audit_gnn_expand.md`,
`audit_datasets.md`, `audit_metrics.md`, `audit_baselines.md` (~2,400 lines).

## Triage

Things that change a claim in the paper:

| # | Issue | Hits |
|---|---|---|
| §0 | **DP noise aliased across same-shaped params** — ε invalid for the mechanism that ran | every `results/relbench/*_meps/` run (84 CSVs, all RelBench numbers in the tex) |
| §4d | **The "graph-blind" arm is not blind** — r=0 GNN evaluates with untrained neighbour weights | RelBench margins (+6 pt → +1.4 pt); PPI survives (verified) |
| §8a | **Regression targets never mean-centred** → "trivial MAE" is the zero predictor | every "beats trivial" claim on LTV/sales |
| §5 | **Union-graph gap confirmed against v42** — Assumption 5.2 bounds g ∪ g′, we cap only g | every ε: **1.35×–2.69×**, depending on p2 and r; the sparse arm pays least |
| §2b-i | **Blank `direction` silently switches adjacency relation** in post-hoc ε | archived CSVs; any future re-run of them |
| §1 | **Seven δ conventions** across method + baselines | cross-run and cross-method comparability |
| §6b-i/ii | **RelBench temporal leakage** — train roots read label-defining rows; val is leaky and val picks checkpoints | all RelBench results |
| §6b-iii | **RelBench features z-scored on the whole DB** — leakage + un-accounted node-DP channel | all RelBench results |

Things that block a claim you want to make:

| # | Issue |
|---|---|
| §9a | The external-baseline harness has **never been run** — 0 JSON files in `results/` |
| §9c | **ProGAP cannot import** (`ModuleNotFoundError: No module named 'core.data'`) |
| §3 | Matched-ε ε **never verified post hoc** — no `_with_eps.csv` in any matched-ε dir |
| §6b | Our `--inductive` and the baselines' split are **different evaluation protocols** |

**Status 2026-09-12:** items marked FIXED below have code changes landed and
tested (178 tests pass). Each carries an `ADDENDUM` block saying what changed,
what was verified, and what is still outstanding. Nothing that requires a
judgement call (δ value, union-graph default, which datasets to rerun) was
changed.

Cheapest high-value actions, in order:

1. Fix §0 (one line), add a two-same-shaped-tensor regression test, rerun the
   RelBench `*_meps` families.
2. Run `compute_epsilon --delta … --grid 1e-5` over the matched-ε dirs (§3, §2).
3. Pick one δ and stick to it (§1).
4. Decide the union-graph question (§5) — now priced: option (A) costs ~2× ε, option (B) costs 44% of PPI's neighbourhood.
5. Switch the blind arm to `--model mlp`, or score it with an empty
   `edge_index` (§4d).

---

## ⚠️ 0. CRITICAL — the DP noise is not independent across same-shaped parameters

**This invalidates ε for every run produced after 2026-09-08 23:12.** I verified
it three ways: by reading the code, by running a probe against the real
mechanism, and against git history.

`src/sparse/base_mechanism.py:175-196` caches the noise staging buffer keyed on
`(shape, device)` — and then appends **the cached tensor object itself** to the
output list:

```python
key = (tuple(grad.shape), grad.device)      # :180
buffers = cache.get(key)                     # :181
...
cpu.normal_(generator=generator)             # :190  fresh draw, same buffer
...
noise.append(device)                         # :196  SAME OBJECT appended again
```

Two parameters of the same shape therefore receive **the identical draw**, not
two independent ones. Probe on a real `MultiLabelGNNMechanism` (`aggr=mean`,
L=2, the PPI configuration):

```
convs.0.lin_l.weight  (16, 50)   id 6151054608
convs.0.lin_r.weight  (16, 50)   id 6151054608   <-- same tensor
convs.1.lin_l.weight  (121, 16)  id 6151059984
convs.1.lin_r.weight  (121, 16)  id 6151059984   <-- same tensor

6 parameters → 4 distinct noise tensors.  max|z_l − z_r| = 0.000e+00
```

`SAGEConv` *always* has `lin_l.weight` and `lin_r.weight` of identical shape, so
**every layer of every `aggr=mean` model is affected** — and `mean` is the
default and what every driver uses.

**Why it breaks the guarantee.** The released update is
`(Σ_i clip(g_i) + Z) / E[B]` with `Z` singular: `z_l − z_r = 0` exactly. So the
linear functional `Σ_i clip(g_i)|_{lin_l} − Σ_i clip(g_i)|_{lin_r}` — a
statistic of the private data with per-root sensitivity up to `2C` — is
**released in the clear, every step, for T steps**. The Gaussian-mechanism
premise (noise `N(0,(σC)²I)` on the full flattened gradient,
`base_mechanism.py:169`, and the paper's Assumption 6.3) does not hold, so the
accounting is accounting for a different mechanism than the one that ran. This
is independent of whether the dominating-pair math is correct.

Utility is unaffected — the *marginal* per-coordinate law is still `N(0,(σC)²)` —
so nothing in any results CSV looks wrong.

**It is a regression, and the blast radius is bounded.** `git show
ff98371^:src/sparse/base_mechanism.py` shows the previous version was correct:

```python
return [(torch.randn(g.shape, generator=generator) * std).to(g.device)
        for g in grads]
```

`ff98371` "Restore archived DP experiments and runtime sources"
(2026-09-08 23:12) introduced the cache. **84 of 246 DP result CSVs postdate
it**, and they are exactly the RelBench matched-ε families:

```
results/relbench/relamazon_itemltv_meps/     relamazon_userltv_meps/
                 relarxiv_authorcategory_meps/  relavito_adctr_meps/
                 relf1_driverposition_meps/     relhm_itemsales_meps/
                 relhm_userchurn_meps/
```

That is **every RelBench number in `paper/experiments_current.tex`**.
`results/ppi_matched_eps/`, `results/arxiv_matched_eps/`,
`results/facebook_width/` all predate `ff98371` and were produced by the correct
noise code — those ε values are unaffected by this.

**Why the tests did not catch it.** Every mechanism in `tests/test_dp_mechanics.py`
is `_FixedGradMechanism` = `nn.Linear(dim, 1, bias=False)` (`:37-38`) — a
*single* parameter tensor, so the aliasing path is never exercised. All 176
tests pass on the buggy code.

**Fix:** drop the cache and restore the `torch.randn`-per-grad version, or key
it on `(shape, device, index)`. Note that either way the seeded noise stream
changes, so post-fix reruns will not be bitwise comparable to pre-`ff98371`
runs. Add a regression test with a two-same-shaped-tensor model.

---


> **ADDENDUM — FIXED 2026-09-12.** `base_mechanism.py:gaussian_noise_like` now
> returns one independent `torch.randn` draw per gradient tensor; the
> `(shape, device)` buffer cache is gone, with a comment saying why it must not
> come back. Verified: the probe that showed 6 parameters → 4 distinct noise
> tensors now shows **6 → 6**. Two regression tests added in
> `tests/test_dp_mechanics.py` using a deliberately two-same-shaped-tensor
> model (`test_noise_is_independent_across_same_shaped_parameters`,
> `test_noise_across_calls_is_fresh_for_same_shapes`) — the existing tests all
> use a single-tensor `nn.Linear`, which is why this went undetected. Suite:
> 178 passed.
> **Still to do:** rerun the 84 affected `results/relbench/*_meps/` CSVs. The
> noise stream has changed, so post-fix runs are not bitwise comparable to
> pre-`ff98371` ones.

## A. Orientation — the files to read, in order

Read these twelve and you understand the whole pipeline. Everything else is a
driver, a plot, or a test.

### The mechanism (our method)

| File | Lines | What lives here |
|---|---|---|
| `src/sparse/run.py` | 709 | **The CLI and the policy layer.** Every default, the dataset→loader dispatch, the degree-capping call site, the `eval_graph` decision, the seed loop, and the CSV writer. If you read one file, read this one. |
| `src/sparse/sparse_gnn.py` | 272 | **The training engine** (Algorithm 1). Root sampling, the per-step loop, the DP branch (clip → sum → noise → step) and the non-DP branch. |
| `src/sparse/sparse_expand.py` | 344 | **SparseExpand** (Algorithm 5/2) + `cap_degrees*` + `dedup_arcs` + Poisson root sampling. The graph-side of the mechanism. |
| `src/sparse/base_mechanism.py` | 197 | The `g0` interface: clipping, noise, optimizer construction, evaluation helpers. Shared by all five heads. |
| `src/sparse/layers.py` | 39 | The message-passing stack. `aggr='mean'` vs `aggr='gcn'` — the exactness argument lives in this docstring. |
| `src/sparse/accounting.py` | 622 | **All the privacy math.** Dominating pairs (Eq. 43–47), the `dp_accounting` handoff, both theorem branches. |
| `src/sparse/compute_epsilon.py` | 185 | Post-hoc ε for a results CSV. The *only* place a finished run gets an ε. |

### Task heads (pick the one for your dataset)

`src/sparse/gnn_mechanism.py` (single-label) · `multilabel_mechanism.py` (PPI) ·
`binary_mechanism.py` (RelBench classification) · `regression_mechanism.py`
(RelBench regression) · `mlp_mechanism.py` (graph-blind arm).

### Data

| File | What lives here |
|---|---|
| `src/datasets.py` (603) | Planetoid / OGB / Reddit / Flickr / PPI loaders, split masks, symmetrization. |
| `src/sparse/relbench_data.py` (326) | RelBench `Database` → homogeneous directed graph. Row nodes, `_encode_table`, temporal cutoffs. |
| `src/experiments/inductive.py` (301) | The *shared* graph-disjoint split used by the baseline harness (and by `--common_inductive_split`). |

### Where the numbers come from

| File | What lives here |
|---|---|
| `scripts/_dataset_settings.sh` | **The per-dataset hyperparameter table.** Sourced by every ladder driver. Read the header comment — it documents the L=2 decision and the regularization decision. |
| `scripts/calibrate_grid.py` | Solves σ for a target ε offline. Feeds the matched-ε sbatch jobs. |
| `scripts/summarize_sweep.py` | Best-checkpoint table for a sweep directory. |
| `sbatch/ppi_matched_eps.sbatch` | The headline experiment, and the best-documented driver in the repo. |
| `paper/experiments_current.tex` | The current claims ledger. Supersedes `experiments_plan.tex` / `experiments_section.tex`. |

### Tests worth reading as documentation

`tests/test_dp_mechanics.py` (356) *measures* the DP path rather than asserting
on it · `tests/test_theorem_numerical.py` (179) checks the dominating pair
against the real mechanism's hockey-stick divergence on a star graph ·
`tests/test_accounting.py` (495).

---

## 1. δ — four conventions in simultaneous use (seven, counting baselines)

**HIGH.** There is no single δ. Four different values are live:

| δ | Where | Which results |
|---|---|---|
| `1e-6` | `scripts/_dataset_settings.sh:119`, `scripts/_coverage_sweep.sh:54`, `sbatch/sparse_inductive.sbatch:30`, `sbatch/reddit_settings.sbatch:56`, `sbatch/sparse_relbench.sbatch:37`, `scripts/_ppi_pareto_grid.sh:23` | the whole ladder (`ladder_stage2`, `sweep`, `relbench_f1`, `orientation_ablation`), `results/ppi/pareto/*`, reddit, coverage |
| `1e-5` | `sbatch/facebook_settings.sbatch:39`, `facebook_settings_ice.sbatch:48`, `facebook_tune_ice.sbatch:34`, `scripts/_reltrial_ladder.sh:16` | `results/facebook_ice/*`, `results/facebook_tune/*`, `results/relbench/reltrial/*` |
| `n^-1.01` | `--delta_from_n` in `sbatch/ppi_matched_eps.sbatch:94`, `arxiv_inductive_matched_eps.sbatch:90`, `scripts/_facebook_width.sh:88`, `relbench_regression_meps.sbatch:125`, `relamazon_scope.sbatch:107` | **every matched-ε result**, i.e. the headline numbers |
| `1e-5` (silent default) | `src/sparse/compute_epsilon.py:47` | anything run without an explicit `--delta` |

Consequences:

1. **HIGH** — `results/ppi_matched_eps/` (δ = 1.574e-5) and `results/ppi/pareto/`
   (δ = 1e-6) are on different privacy scales. Any figure that draws both is
   mixing them. Same for facebook: `facebook_width` (n^-1.01) vs `facebook_ice`
   (1e-5).
2. **HIGH** — `n^-1.01` is a weak δ. For PPI, n = 56 944 → δ = 1.574e-5 and
   **δ·n = 0.896**, i.e. δ ≈ 0.9/n. For arxiv, n = 169 343 → δ = 5.2e-6,
   δ·n = 0.885. The usual requirement is δ ≪ 1/n; at δ ≈ 1/n a mechanism can
   satisfy the definition while releasing a random record in the clear. A
   reviewer will ask. `n^-1.1` (PPI: 3.4e-6) or a flat `1e-6` both clear it.
3. **MED** — `compute_epsilon.py:47` defaults to `1e-5`, which matches *none*
   of the three intentional conventions. A post-hoc ε run that forgets
   `--delta` silently reports at the wrong δ.
4. **LOW** — `paper/experiments_current.tex:34` states "δ = n^-1.01, full node
   count" as though it were universal. True only of the matched-ε runs.

And the baselines add three more, so a filled-in comparison table would span
**seven** δ conventions:

| δ | Method | Where |
|---|---|---|
| `1e-5` | DP-MLP / GraphSAGE | `src/experiments/baselines.py:29` |
| `1e-4` + `1e-3` | DPAR (ppr + sgd, total ≈ 1e-3) | `src/experiments/dpar.py:34,44` |
| `1/(10n)` | DP-GNN | `src/experiments/dpgnn.py:231` |

DPAR's total δ ≈ 1e-3 is **3.3/n** on citeseer (n=3327) — weaker than releasing a
random node in the clear.

**Decide:** one δ for the paper, applied everywhere, and re-derive σ for any
run that used a different one.

---

## 2. The accounting grid is set in one place and defaulted in another

**MED.** `--grid` is the PLD discretization interval; the numerical floor on ε
is ≈ `T · grid`.

- `scripts/calibrate_grid.py:50` default `1e-4`; the matched-ε jobs override to
  `1e-5` (`ppi_matched_eps.sbatch:93`), correctly — at T = 2000, `1e-4` puts the
  floor at 0.2, i.e. 20 % of an ε = 1 target.
- `src/sparse/compute_epsilon.py:53` default `1e-4` — **not** overridden by any
  driver that calls it.
- `src/sparse/run.py:252` `--accounting_grid` default `1e-4`, used by the
  in-process `--target_epsilon` calibration path.

So σ was *solved* at grid = 1e-5 but any post-hoc ε would be *reported* at
grid = 1e-4, at a floor 10× coarser. The two would not agree, and the
disagreement is largest exactly where it matters (small ε, large T).

**Also MED:** there are two calibration paths — offline `scripts/calibrate_grid.py`
and in-process `run.py --target_epsilon` (`run.py:571` → `calibrate_sparsegnn_noise`).
Nothing checks they agree. Every landed result used the offline one.

---

## 2b. The accounting core is sound — record this so it is not re-audited

The dominating-pair construction was checked analytically **and** numerically: the
PLD handed to `dp_accounting` dominates the analytic pair (single-step
`get_delta_for_epsilon` vs fine quadrature, ratios 1.0004–1.035, always ≥ 1).

- Accountant is **PLD**, not RDP, with `pessimistic_estimate=True` and
  `math.ceil` rounding.
- **Every numerical knob pushes ε up**: `grid`, `n_sigma=10`,
  `atoms_per_sigma=400`, the `1e-14`/`1e-18` floors, and `dp_accounting`'s
  `tail_mass_truncation`.
- `symmetric=True` is legitimate for the substitution pair (`Q(x) = P(−x)`);
  the Thm-4.5 branch builds both orientations and takes the max.
- `C` and `L` correctly do **not** enter ε.
- Shells are `n_d = K_out^d` for `direction='in'` and `K_in^d` for `'out'`
  (`accounting.py:18, 83-97`) — the out-cap prices in-expansion, as documented.
- The per-checkpoint schedule (`accounting.py:270-289`) is a valid guarantee for
  releasing `θ_1..θ_t`.

**One genuine (tiny) under-estimate:** `accounting.py:344-345` drops Thm-4.5
fibers below `1e-14` *without* routing their mass to an infinite-loss outcome,
unlike the substitution path which does. Measured worst case ≈ 5e-14 × T=2000 ≈
1e-10 against δ=1e-6 — irrelevant today, but it scales with `K^r` and it is the
only non-conservative direction found.

### 2b-i. HIGH — a blank `direction` column silently changes the adjacency relation

`src/sparse/compute_epsilon.py:79`:

```python
direction = row.get('direction') or 'out'
```

A missing **or blank** cell routes the row to Theorem 4.5 — **insertion/removal**
instead of substitution. That is a different neighbour relation, not just a
different number. Measured on one synthetic row
(`p1=.05 p2=.5 r=1 K=5 σ=5 T=50`): `direction=in` → ε = 4.8491
(`thm6.4-substitution`); column deleted or blank → ε = 7.7864
(`thm4.5-insertion-removal`). **No warning either way.** The archived CSVs in
`results/archive/` genuinely lack the column, so this path is live.

At `p1=0.3` the ordering even flips (thm4.5 = 114.98 < substitution = 241.50), so
you cannot even reason about which direction the error goes.

Make a blank `direction` a hard error.


> **ADDENDUM — FIXED 2026-09-12.** `compute_epsilon.py` no longer defaults a
> blank `direction` to `'out'`. A missing/blank value is now a hard error
> naming the consequence (substitution vs insertion/removal), and a new
> `--assume_direction {in,out}` flag exists for the archived pre-orientation-fix
> CSVs that genuinely lack the column. Verified end to end.

### 2b-ii. HIGH — the orientation ablation plots two different adjacency notions on one axis

`scripts/orientation_ablation.sh:72-79` runs `compute_epsilon` with the default
`--theorem auto` on **both** `dp_in/` and `dp_out/`, then invites you to compare
them. Under `auto`, `in` gets substitution-ε and `out` gets add/remove-ε.

All six plotting scripts read the `epsilon` column
(`plot_frontier.py:53`, `plot_ppi_frontier.py:52`, `plot_ppi_pareto.py:54`,
`plot_relbench_report.py:57`, `summarize_sweep.py:42`). The
`epsilon_substitution` column exists precisely to make this comparison honest
(`compute_epsilon.py:107-110`) — **no plotting script uses it.**
`plot_sparse_frontier.py:38` knows about an alternative column, but it is the
*legacy* `epsilon_thm4`.

### 2b-iii. MED — `K_in` is NOT free in ε at r ≥ 2

`accounting.py:111` sets `K = min(K_in, K_out)`, which feeds the `K^(l-1)` path
count at `accounting.py:61`. Measured at `r=2, K_out=5`: raising `K_in` from
1 → 5 moves ε from **10.07 → 29.30** (~3×), then saturates.

So `K_in` is free **only at r=1**, or in the saturated regime `K_in ≥ K_out`.
(This is consistent with the earlier 2026-09-01 measurement — r=2, K_out=5,
p2=0.25: K_in=2 → 6.84, K_in ∈ {5,20,100} → 13.49 flat — so it is a
sharpening, not a contradiction. The short form "K_in is free in ε" is what
over-generalizes.)

The saturated regime happens to be where the RelBench figure sits (`K_out=3`,
`K_in` swept 2–20), so `scripts/plot_relbench_report.py:182`'s caption ("ε barely
moves with K_in") is true of what it measured but reads as a general claim.

### 2b-iv. MED/LOW — other accounting-path issues

- **`compute_epsilon` does not validate δ at all.** `--delta 1.5` → writes
  `eps=0.0000` for every row (plus a **negative** `eps_naive` from Opacus);
  `--delta 0` → writes `eps=inf`. `calibrate_sparsegnn_noise` *does* validate
  (`accounting.py:528-530`); the post-hoc path does not.
- **No dataset awareness in δ.** Run on reddit (n=232,965) without `--delta`, the
  `1e-5` default is ≈ 2.3/n — formally meaningless for node-level DP — and it is
  written into the CSV with no warning.
- **`compute_epsilon` never reads the columns `run.py:539-541` wrote**
  (`target_delta`, `calibrated_epsilon`, `accounting_theorem`,
  `accounting_grid`), so a `calibrated_epsilon` and a post-hoc `epsilon` computed
  at a *different* δ and grid can sit in the same row uncross-checked. None of the
  16 `compute_epsilon` call sites passes `--grid`.
- With `--track_every`, every checkpoint row carries the **final-step**
  `target_epsilon`/`calibrated_epsilon` (they are per-cell constants,
  `run.py:593-598`), so a step-500-of-2000 row claims `calibrated_epsilon=1.0`.
- **`--dp` without `--K_in`** writes the graph's *observed* max degrees into the
  `K_in`/`K_out` columns (`run.py:484-489`), and `compute_epsilon` consumes them
  with no way to tell they were observed rather than enforced. The observed max
  degree is itself data-dependent, and under substitution the neighbouring graph
  may exceed it. On arxiv the uncapped max degrees are 3015/221.
- **Dead duplicate:** `_substitution_pld` is defined at `accounting.py:254` and
  **immediately redefined** at `:262`. Verified — the first is dead. Bodies are
  equivalent today, so nothing is wrong now, but an edit to the shadowed copy
  would be silently ignored. Likewise `thm4_fiber_weights`
  (`accounting.py:136-142`) re-implements the `q_d` recurrence instead of calling
  `_q_products` (`:50-64`) — two copies of the core privacy formula.
- **Duplicated dispatch:** `compute_epsilon.py:112-128` hand-rolls the `auto`
  mapping *and* the label strings, while `resolve_sparsegnn_theorem`
  (`accounting.py:414-429`) and `sparsegnn_epsilon_schedule` (`:441-464`) exist
  for exactly that and the latter is **called nowhere**.
- `epsilon_naive_opacus` is not a guarantee for this mechanism but sits in a
  column named like one.
- The grid used to compute an ε is **not recorded** in the output CSV.

---


> **ADDENDUM — PARTLY FIXED 2026-09-12.**
> - `--delta` is now **required** (no silent `1e-5`) and validated to lie in
>   `(0, 1)`. Previously `--delta 1.5` wrote `eps=0.0000` for every row and
>   `--delta 0` wrote `inf`; both now exit with a message. Every shipped driver
>   already passes `--delta`, so nothing breaks.
> - The discretization is now recorded per row as a new `epsilon_grid` column,
>   and `compute_epsilon` **warns** when a row's recorded `accounting_grid`
>   differs from the grid it is computing at — the calibrate-at-1e-5 /
>   report-at-1e-4 mismatch is now visible instead of silent.
> - `run.py`'s post-hoc hint now prints `--delta <delta> --grid <grid>` rather
>   than a command that would fail.
> - **Not fixed (decisions, not bugs):** which δ to standardise on; the
>   duplicate `_substitution_pld` definition; the hand-rolled theorem dispatch.

## 3. Matched-ε results have never had ε verified post hoc

**HIGH (verification gap, not necessarily a bug).**

- 246 DP result CSVs exist; only **85** have a `_with_eps.csv` sibling.
- **No** matched-ε directory has one: `results/ppi_matched_eps/`,
  `results/arxiv_matched_eps/`, `results/facebook_width/`,
  `results/relbench/*_meps/` all lack it.

For those runs the ε is the *target* handed to `calibrate_grid.py`, preserved
only in the directory name and in `sigma_gnn.txt` / `sigma_blind.txt`. The
README's stated workflow (train → attach ε) was not run on the headline results.

I checked the chain by hand for PPI and it is **internally consistent**:
`results/ppi_matched_eps/sigma_gnn.txt` header records
`p1=0.0114 r=1 K=5 T=2000 delta=1.574e-05 direction=in grid=1e-05`, and the σ
recorded in each CSV matches its row exactly (p2=0.1/ε=1 → 11.359375;
p2=1.0/ε=8 → 7.277344; blind ε=1 → 7.574219). So the numbers are defensible —
they are just not *checked* by anything in the repo.

**Do:** run `compute_epsilon.py --delta 1.574e-5 --grid 1e-5` over the
matched-ε dirs and confirm the reported ε lands on the target. This is the
cheapest high-value thing on this list. (Note the `--grid` — see §2.)

*Minor:* `sbatch/ppi_matched_eps.sbatch:29` quotes blind σ = 7.61 at ε = 1;
the calibrator produced 7.574219. Doc drift only.

---

## 4. Degree capping — what is capped, and what is scored

Answering the direct question: **the cap is applied to one graph, and *which*
graph is scored depends on a flag most drivers do not pass.**

Mechanics (`src/sparse/run.py:434–495`):

1. `_simplify_and_cap` = `dedup_arcs` **then** cap. Dedup first, so parallel
   arcs cannot win the survival lottery twice (`run.py:437`, motivated at
   `run.py:424` — "path counts, Lemma 20, assumes graphs without parallel edges").
2. Cap mode is `auto` → `undirected` iff `K_in == K_out` **and** the arc set is
   symmetric, else `directed` (`run.py:446–448`). Capping a symmetric graph in
   `directed` mode destroys symmetry — "~2/3 of surviving arcs lose their
   reverse at K=5" (`run.py:454–457`). **Asking for `K_in != K_out` on a
   symmetric graph silently switches algorithms.**
3. Truncation is **random**, seeded by `cap_seed` (`run.py:444`), and the graph
   is **re-capped per seed** unless `--cap_seed` pins it (`run.py:500–505`). So
   the reported spread includes cap variance. Good.
4. `K_in` defaults to **`None` = no cap** (`run.py:258`). With `--dp` and no
   `--K_in`, it warns and prices ε off the graph's raw max degree
   (`run.py:484–489`).

Which graph gets scored (`run.py:322–328`, `613–617`):

| `--inductive` | `eval_graph` (auto) | primary metric on | `_alt` metric on |
|---|---|---|---|
| yes | `full` | full uncapped graph | full graph, **capped** |
| no | `train` | **capped** training graph | full **uncapped** graph |

**HIGH / MED — the consequence.** `scripts/_dataset_settings.sh:65` sets
`INDUCTIVE=()` for PPI ("natively inductive, `--inductive` is a no-op"). That is
true for *edge dropping* — PPI's splits are disjoint components, so there are no
crossing edges to drop — but it is **not** a no-op for `eval_graph`, which falls
through to `train`. Verified in the CSVs: every
`results/ppi_matched_eps/*` and `results/ppi_stage1/*` row records
`eval_graph=train`, `inductive=False`.

So the headline PPI micro-F1 is measured on a **K=5 degree-capped** version of
the test graphs, not the real ones. The full-graph score is in `*_acc_alt`.
Same for facebook (`results/facebook_width/*`, `results/coverage_facebook/*`).

This is a *defensible* choice — `run.py:323–327` argues the model never saw a
node of degree > K, so the uncapped graph is off-distribution — but it is the
opposite of what both docs claim:

- `README.md:162` — "evaluation defaults to the full one"
- `paper/experiments_current.tex:35` — "evaluation defaults to the full graph"

Both are true only for the `--inductive` datasets (arxiv, flickr, reddit,
relbench). **LOW→HIGH depending on whether the paper number is the capped one.**

**MED — facebook was run under both conventions.** `results/facebook_ice/*`
and `results/facebook_tune/*` record `eval_graph=full`; `results/facebook_width/*`
and `results/coverage_facebook/*` record `eval_graph=train`. Same dataset, same
`inductive=False`, different scoring graph. Those two families are not
comparable.

**MED — regression has no `_alt`.** The `_alt` columns only exist for `acc` and
`auroc` (`run.py:564–565`). RMSE/R²/`bin_acc` have no alt, so the
capped-vs-full gap is unmeasured for every RelBench regression result.

---


> **ADDENDUM — POLICY CHANGED 2026-09-13.** Evaluation is now **always on the
> unprocessed full graph**. `--eval_graph auto` resolves to `'full'` for
> transductive and inductive runs alike, so no training-side preprocessing
> (degree cap, dedup, split filter) touches the graph a result is measured on.
>
> This deliberately overrides the argument the old code made — that a
> transductive model never saw a node of degree > K, so the uncapped graph is
> off-distribution. True, but it bought in-distribution evaluation by
> preprocessing the test set, and it contradicted what both `README.md:162` and
> `experiments_current.tex:35` already claimed. The capped-graph number is still
> recorded as the `*_alt` columns, so the gap stays measurable.
>
> Verified on citeseer: `eval_graph=full` where it previously printed
> `eval_graph=train`. **Consequence:** PPI, facebook and every other
> transductive family now report the number that used to sit in `*_acc_alt`.
> `--eval_graph train` still exists to reproduce old runs.

### 4b. Degree capping — the remaining mechanics, verified

- **Capping happens at graph-construction time only.** SparseExpand itself never
  truncates — there is no per-level fan-out cap anywhere. An uncapped graph gives
  unbounded expansion fan-out.
- **Truncation is uniformly random, not first-K.** `cap_degrees._cap`
  (`sparse_expand.py:208-217`) shuffles, stable-sorts, keeps rank < K. Measured on
  a 10-arc star at `K_in=1` over 3000 seeds: survivor histogram
  `[282,308,280,317,319,294,293,303,300,304]` — uniform, no index bias.
- `cap_degrees` caps in-degree first then out-degree (`sparse_expand.py:219-222`);
  since both only remove arcs the second pass cannot re-violate the first.
- **MED — `--K_out N` without `--K_in` applies no cap at all.** `run.py:441`
  returns early on `K_in_req is None`, so the out-cap is silently ignored. Under
  `--dp` this also trips the `run.py:484-489` raw-max-degree path.


> **ADDENDUM — FIXED 2026-09-12.** `--K_out N` without `--K_in` now exits with
> a message instead of silently applying no cap at all (verified: exit code 1).


> **ADDENDUM 2 — POLICY CHANGED 2026-09-13.** `--cap_mode auto` now resolves to
> **`directed` for every graph**, symmetric or not. An undirected graph is
> treated as a directed arc set: dropping an arc does not oblige us to drop its
> reverse, and in/out degree are capped independently — which is exactly the
> pair of bounds Assumption 5.2 states. The undirected variant is still
> available as `--cap_mode undirected`.
>
> This reverses the old preference for symmetry-preserving capping. Symmetry was
> never required by the mechanism; it only made in- and out-expansion coincide.
> Verified on citeseer (a symmetric graph): `mode=directed`, max (in,out)
> 99,99 → 5,5.
>
> **Consequence:** every symmetric dataset (PPI, facebook, reddit, flickr) now
> trains on a different capped graph than before, so numbers will move. On a
> symmetric graph directed capping loses the reverse of roughly two-thirds of
> surviving arcs at K=5, so effective neighbourhoods shrink.

### 4c. The aggregator, and the "EXACT" claim

Both aggregators are **stock PyG**, no subclassing (`layers.py:27-39`) — "our
averaging" is `SAGEConv(aggr='mean')`:

```
mean:  h_i = W_l · (1/|N_in(i)| · Σ_{j∈N_in(i)} h_j) + b + W_r · h_i
       normalizer = target's in-degree on the supplied edge set, no self-loop, no +1
       empty in-neighbourhood → zero aggregate
gcn:   h_i = Σ_{j∈N_in(i)∪{i}} (d̂_j d̂_i)^{-1/2} Θ h_j + b,   d̂_v = 1+|N_in(v)|
       PyG uses the IN-degree at both ends, so on directed graphs (arxiv,
       RelBench) this is not textbook symmetric normalization
```

Stack: `conv → ReLU → dropout` between layers, none after the last; no output L2
normalization (original GraphSAGE has it), no dropout on the input features
(Kipf's GCN has it), no residuals, no norm layers.

**MED (as a claim) — the README's "EXACT" statement is missing two
preconditions.** `README.md:155-160` and `layers.py:8-13` say mean makes
rooted-subgraph inference exactly equal to full-graph inference. Measured
`max|rooted − full|` over 60 roots:

| aggr | L | r=1 | r=2 | r=3 | r=4 |
|---|---|---|---|---|---|
| mean | 1 | **0** | **0** | **0** | – |
| mean | 2 | 1.2e-1 | **0** | **0** | – |
| mean | 3 | 1.4e-1 | 4.4e-2 | **0** | – |
| gcn | 1 | 2.6e+0 | **0** | **0** | **0** |
| gcn | 2 | 3.0e+0 | 6.0e-1 | **0** | **0** |

The real rule: **mean is exact iff `p2 == 1` and `L ≤ r`; gcn is exact iff
`p2 == 1` and `L ≤ r − 1`.** So gcn is not intrinsically inexact — it just needs
one more shell, which costs `K_out^{L+1}` in ε. At `p2=0.5, L=r=2` mean's error is
max 0.51 / mean 0.12. **The shipped experiments violate both preconditions**
(`p2 ∈ {1, .5, .25, .1}`, and `L=2` with `r=1`).

**MED — at the shipped `p2=0.1, K=5`, most roots aggregate nothing.**
`P(empty) ≥ (1−p2)^K = 0.59`. Those roots train as MLPs but are evaluated with a
real neighbour mean. `run.py:66-88`'s warning threshold (<1.05 nodes) does not
catch this.

**No off-by-one in `r`.** Verified on a 5-chain: `r` levels materialize the
radius-`r` ball plus arcs at depth 1..r — exactly an `r`-layer mean-GNN's
computation tree.

**No L-vs-r guard anywhere.** `L > r` is privacy-safe (ε is priced by `r` alone)
but creates a train/eval mismatch; `L < r` silently discards hops you paid ε for.
`ladder_stage01.sh:65` echoes "L=$R" while passing `--num_layers 2`.

### 4d. HIGH — the "graph-blind" arm is not blind

`_dataset_settings.sh:64,76` and `sbatch/ppi_matched_eps.sbatch:98-103` define the
blind arm as **the same GNN mechanism at `--r 0`** — not as an MLP. At `r=0` the
rooted subgraph is one node with no edges, so the SAGE aggregate is identically
zero and `lin_l.weight` receives **exactly zero gradient**. But `evaluate()` runs a
full forward over a *real* graph, multiplying real neighbour means by those
never-trained weights.

Verified by my own probe:

```
r=0 subgraph edge counts: [0,0,0,0,0,0,0,0,0,0]
convs.0.lin_l.weight  |grad| = 0.000000e+00      <- never trains
convs.0.lin_r.weight  |grad| = 8.906471e+00
convs.1.lin_l.weight  |grad| = 0.000000e+00
eval forward WITH edges vs WITHOUT: max|diff| = 0.5846   <- eval uses the graph
```

Non-privately `lin_l` stays at random init. **Under `--dp` it becomes a pure
Gaussian random walk** — zero signal plus fresh noise every step. At PPI's blind
settings (σ=7.57, C=1, lr=0.3, denom=p1·pool≈512, T=2000) the accumulated
std is `lr·σC/denom·√T ≈ 0.20`, against a Glorot init std of ≈0.081 for the
first layer — so after training the neighbour weight is **several times larger
than its initialization**, and entirely noise.

The measured consequence on `results/relbench/relhm_userchurn_meps/` (test AUROC,
last step):

| arm | primary (full graph) | `_alt` (capped graph) |
|---|---|---|
| blind `r=0`, ε=1 | 0.5093 | **0.6022** |
| GNN `r=2 p2=0.1`, ε=1 | 0.5582 | 0.6154 |
| blind `r=0`, non-DP | 0.4834 | 0.5926 |
| GNN `r=2`, non-DP | 0.5867 | 0.6295 |

A genuinely blind model would score *identically* on both graphs. The 9-point
swing is direct proof that the arm consumes graph structure. And
`paper/experiments_current.tex:85`'s claim — "GNN AUROC 0.55–0.56 vs blind
0.49–0.51 (chance)" — is a **+6 pt** margin on the primary graph but **+1.4 pt**
on the alt graph. Same pattern on `rel-f1/driver-position` (23.45 vs 8.81 MAE) and
`rel-amazon/item-ltv` (93.6 vs 87.2). PPI is affected far less (0.4215 vs 0.4194).

**The PPI headline survives this, though.** I recomputed the PPI matched-ε margin
on both graphs (best-on-validation, averaged over seeds):

| ε | blind primary | GNN primary | margin (capped) | margin (uncapped `_alt`) |
|---|---|---|---|---|
| 1 | 0.4136 | 0.4234 | +0.98 pt | **+1.53 pt** |
| 2 | 0.4208 | 0.4405 | +1.97 pt | **+2.51 pt** |
| 4 | 0.4199 | 0.4519 | +3.20 pt | **+3.58 pt** |
| 8 | 0.4184 | 0.4580 | +3.96 pt | **+4.16 pt** |

The margin is *larger* on the uncapped graph, so PPI's conclusion is robust to
which graph is scored. Note the paper's quoted "+1.5 to +4.1 pts"
(`experiments_current.tex:57-58`) matches the **`_alt` (uncapped)** column, not
the primary one — worth pinning down which column the paper means, given §4's
finding that PPI's primary column is the capped graph.

RelBench is the family where the blind arm's eval graph changes the conclusion.

**Two incompatible definitions of "blind" coexist in the repo:**
`_dataset_settings.sh:64,76` (r=0 GNN) vs `sbatch/sparse_relbench.sbatch:47`
(`--model mlp`). The `mlp` mechanism has no neighbour weight at all and is
genuinely blind; arxiv/flickr/reddit/facebook use it, PPI and RelBench do not.

Fix direction: score the blind arm with an empty `edge_index`, or switch it to
`--model mlp` everywhere.

---


> **ADDENDUM — GUARDED 2026-09-12.** `run.py` now prints a loud warning when
> `--r 0` is combined with any GNN mechanism, stating that the arm is not
> blind and pointing at `--model mlp`. Verified: warns for `--model gnn`,
> silent for `--model mlp`. Left as a warning rather than an error because the
> existing PPI/RelBench results were produced this way and the flag still needs
> to run for reproduction. **Still to do:** switch the drivers in
> `_dataset_settings.sh:64,76` and the matched-ε sbatch files to `--model mlp`.

## 5. The union-graph gap — confirmed against v42, and now priced

**HIGH.** Re-checked 2026-09-12 against
`~/Downloads/Markovian_DP_Subsampling (42).pdf`. **Assumption 5.2 is unchanged
and the K that enters Theorem 5.4 is a bound on the union, not on your graph.**

### What the assumption actually says

v42 §5.1, just before Assumption 5.2:

> "Fix neighboring graphs g ≃_G g′ and let s be the substituted vertex. Write
> **H = g ∨ g′ = (V, E ∪ E′)** for their directed union."

> **Assumption 5.2 (Degree bounds).** "*The union graph H* has maximum in-degree
> K_in and maximum out-degree K_out. We write K = min{K_in, K_out}."

It is stated a second time on p.10: "We will use degree bounds for the union of
neighboring graphs… We assume that g ∪ g′ has maximum in-degree K_in and maximum
out-degree K_out."

And Lemma 5 uses it in both places the shells come from — every quantity is
measured in H:

> "A directed path of length ℓ from s to v has ℓ−1 internal vertices. Tracing
> the path forward from s gives |P^(ℓ)_{s,v}| ≤ K_out^{ℓ−1}, whereas tracing it
> backward from v gives ≤ K_in^{ℓ−1}."
>
> "every vertex at distance d from s is the endpoint of a directed path of
> length d starting at s. There are at most K_out^d such paths."

with `d_H(s,v)` the distance **in H** and `P^(ℓ)_{s,v}` the paths **in H**
(Lemma 4's coupling is explicitly over `Ω = {0,1}^{E_H}`, `E_H = E ∪ E′`).

### So what is the gap, exactly

The theorem is conditional on the *pair*: it holds for every `g ≃_G g′` **whose
union** is (K_in, K_out)-bounded. `run.py:434-465` enforces the bound on `g` —
the one graph it holds. It cannot enforce anything about `g′`, which is a
counterfactual.

If `g` and `g′` are both legal K-capped graphs, their union is **not**
K-bounded — it is 2K-bounded at `s`. Verified empirically on Facebook
(2026-09-01, K=5, 300 random substitutions with both graphs legal): the union
realized max in-degree 10 and max out-degree 10, exactly 2K.

**Capping the released graph at K therefore licenses Theorem 5.4 at
K_union = 2K, not at K.** That is the whole gap. It is not an error in the
theorem and not an error in `cap_degrees` — it is a mismatch about which graph
the symbol `K` refers to.

The code has no union graph at all: `grep -rni union src/ tests/` returns two
hits, both PPI's disjoint-union loader (`datasets.py:129,150`), and there is no
factor of 2 on K anywhere. So `paper/experiments_current.tex:33` — "K_in, K_out
capped on both the training graph and the union graph" — describes something
that is not implemented, and could not be.

### The cost is much smaller than doubling K

This is the new part. Def 5.1 says `E △ E′ ⊆ ({s}×V) ∪ (V×{s})`, so **for
u ≠ s and w ≠ s, the arc (u,w) is in E iff it is in E′.** Every arc of H between
two non-`s` vertices is a common arc. The union can only inflate degree *at s*.

Shortest paths from `s` never revisit `s`, so every step after the first sees
the un-inflated degree. Recounting Lemma 5 under that:

```
n_d  ≤ 2·K_out^d                                    (was K_out^d)
|P^(ℓ)|  ≤ min( 2·K_out^(ℓ-1),  K_in^(ℓ-1) )        (was K^(ℓ-1))
```

The backward trace from `v` visits only non-`s` vertices, so it carries **no
factor of 2** — which means that whenever `K_in ≤ K_out` `q_d` is **completely
unchanged** and only `n_d` doubles. The correction is one factor of 2 per shell,
not `2^d`.

That covers every symmetric config (PPI, facebook, reddit, arxiv all run
`K_in = K_out = 5`). **RelBench is the exception** — `K_in=20, K_out=3`, so
`K_in > K_out` and its `q_d` does change too, which is why its ratio (2.84×) is
the largest in the table below. Verified: `q_d` is bit-identical between the two
formulas iff `K_in ≤ K_out`.

### The cost is not a flat 2× — it depends on p2 and r

*(Corrected 2026-09-12 after collaborator pushback. An earlier draft of this
section reported "~2×" from the r=1, p2=1 cell and over-generalized it.)*

**Only the shells `d ≥ 1` are doubled.** The `d = 0` term — the substituted
vertex itself, `n_0 = 1`, `q_0 = 1` — is untouched, because `s` is a single
vertex in both graphs. So the expected number of affected roots goes from
`p1(1 + A)` to `p1(1 + 2A)`, where

```
A = Σ_{d≥1} n_d q_d
(1 + 2A)/(1 + A)  →  2  as A → ∞   (dense, deep)
                  →  1  as A → 0   (aggressive sparsification)
```

and ε tracks that ratio closely, slightly superlinearly. Since `q_d ~ p2^d`,
**sparsification shrinks `A` and therefore absorbs the correction.** Measured
with the repo's accountant at `grid=1e-5`:

| cell | p2 | A | λ ratio | ε now | ε with 2K | ratio |
|---|---|---|---|---|---|---|
| facebook r=1, σ=5, T=500 | 1.0 | 5.00 | 1.83 | 7.19 | 14.81 | 2.06× |
| facebook r=1 | 0.5 | 2.50 | 1.71 | 3.90 | 7.19 | 1.85× |
| facebook r=1 | 0.25 | 1.25 | 1.56 | 2.39 | 3.90 | 1.63× |
| facebook r=1 | 0.1 | 0.50 | 1.33 | 1.54 | 2.10 | **1.37×** |
| facebook r=2 | 1.0 | 30.0 | 1.97 | 57.6 | 154.9 | **2.69×** |
| facebook r=2 | 0.1 | 1.95 | 1.66 | 3.22 | 5.69 | 1.77× |
| PPI r=1, σ=45.4, T=2000 | 1.0 | 5.00 | 1.83 | 1.00 | 1.95 | 1.95× |
| PPI r=1, σ=45.4 | 0.1 | 0.50 | 1.33 | 0.227 | 0.307 | **1.35×** |

**Range: 1.35× to 2.69×.** Two things follow, and they cut in opposite
directions:

- The **sparse arm pays least** (≈1.35–1.4× at p2=0.1, r=1). The arm the paper
  advocates is the one the correction hurts least.
- At **r=2 with p2=1 it is 2.69×, worse than doubling** — ε is superlinear in λ
  there. So "much less than 2×" is not universal either; it is a statement about
  the small-`A` regime.

### An exact identity at r=1: the fix costs one step down the p2 grid

At `r=1` the mixture depends on the shells only through `A = n_1 q_1 = K·p2`.
So doubling `K` and halving `p2` leave `A` — and therefore ε — **exactly**
unchanged. Confirmed numerically: facebook r=1, corrected at p2=0.5 gives
**7.192**; uncorrected at p2=1.0 gives **7.192**. Same for PPI (1.000 both).

Since p2 is swept anyway, the union correction simply shifts the frontier by one
grid point at r=1. It does not hold exactly at r≥2, where `q_d` also depends on
`K`.

### The number that matters is the σ increase, not the ε increase

"ε roughly doubles" is a misleading frame. At matched ε, what the correction
actually costs is more noise — and since ε ≈ λ/σ in this regime, the σ ratio is
**exactly** `(1+2A)/(1+A)`:

| cell | p2 | σ now | σ corrected | ratio |
|---|---|---|---|---|
| facebook r=1 | 1.0 | 5.00 | 9.17 | 1.83× |
| facebook r=1 | 0.1 | 5.00 | 6.66 | **1.33×** |
| PPI ε=1 | 1.0 | 45.44 | 83.30 | 1.83× |
| PPI ε=1 | 0.1 | 11.36 | 15.15 | **1.33×** |

Converting that to utility via PPI's own measured σ-vs-micro-F1 curve at p2=0.1
(σ = 1.96 / 3.31 / 6.05 / 11.36 → F1 = .4580 / .4519 / .4405 / .4234), a 1.33×
σ bump costs:

- **ε=8: ≈0.33 points** against the measured +4.16 margin — negligible
- **ε=1: ≈0.77 points** against the measured +1.53 margin — eats half of it

So the correction is comfortably survivable at ε ≥ 2 and is a real problem only
at ε=1.

### Validated against a closed form

The ratio is not an artifact of the PLD machinery. In the degenerate case
p1=1, p2=1 the mixture collapses to a point mass, the pair becomes a plain
Gaussian mechanism with Δ = 4k√T, and the repo's accountant can be checked
against the exact Balle–Wang formula:

| K_out | k | repo ε | exact ε | repo/exact |
|---|---|---|---|---|
| 5 | 6 | 0.29590 | 0.29553 | 1.001 |
| 10 | 11 | 0.56572 | 0.56512 | 1.001 |

Agreement to 0.1%, and **doubling the shell gives ratio 1.912 by the repo and
1.912 exactly.** The accountant is right and only mildly conservative; the cost
is a property of the mechanism, not of the implementation. It is group privacy:
doubling how many roots one node can influence doubles the group size.

### Implementation

One line in `shell_sizes` (`accounting.py:96`): return `2 * base**d` for
`d ≥ 1`, **leaving `n_0 = 1` untouched**. Nothing else in the accountant moves.
Comment it as the union-graph correction to Assumption 5.2.

Two implementation warnings:

1. **Do not multiply the whole returned list.** `shell_sizes` returns
   `[1] + [base**d ...]`; a blanket `2 *` would also scale `n_0`, which is wrong
   (`s` is one vertex in both graphs) and would push the cost back toward 2×
   in exactly the sparse regime where the saving lives.
2. **Do not implement it as `K_out → 2·K_out`.** That yields `(2K)^d`, i.e.
   `2^d·K^d` — correct at r=1, but 2× too pessimistic at r=2 (facebook r=2:
   394 naive vs 155 tight).

### The two honest options

**(A) Keep the cap at K, fix the accountant** to `n_d = 2·K_out^d`. Costs
~2× ε. All reported ε values move; no rerun needed (ε is post-hoc), but every
matched-ε run would need σ re-solved to land back on its target.

**(B) Halve the cap, keep the accountant.** Capping `g` at `k` makes the union
`2k`-bounded, so today's accounting at K=5 is valid if you cap at **2**. ε is
unchanged, and the cost is all in utility. Measured on PPI:

| cap | arcs kept | mean in-degree at train roots | E[1-hop subgraph] |
|---|---|---|---|
| K=5 | 195,664 | 3.435 | 4.435 |
| K=3 | 124,252 | 2.182 | 3.182 |
| K=2 | 85,194 | 1.497 | 2.497 |

Going 5 → 2 drops the average root's neighbourhood by **44 %**. Since the graph
is only worth ~+4 pts at ε=8 on PPI to begin with (§4d), (B) would likely eat
most of the effect the paper is trying to demonstrate. **(A) looks like the
better trade** — a 2× ε is a much smaller claim change than losing the result.

**(C)** remains: tighten Assumption 5.2 to bound `g` and redo Lemma 5 carrying
the slack explicitly. That is the same arithmetic as (A), just placed in the
theorem rather than the code — and it is the honest version if you want `K` in
the paper to mean the thing you actually enforce.

**Do not silently change the accountant.** Scripts for all of this are in
`audit/union_cost.py` (first pass) and `audit/union_cost2.py` (four variants
side by side, with the derivation in the module docstring).

---

## 6. Theorem/assumption numbering is inconsistent across the source

**LOW**, but it is the kind of thing a referee notices. Three numbering schemes
coexist:

| Object | Names used in code |
|---|---|
| in-expansion substitution | "Theorem 5.4" (`accounting.py:6`, `compute_epsilon.py:9`, `README.md:113`) **and** "Theorem 6.4" (`run.py:205,260`, `accounting.py:89,333`, `compute_epsilon.py:51`, `_dataset_settings.sh:18`) |
| out-expansion mirror | "Theorem 1/2" (`accounting.py:92,333`) |
| out-expansion insert/remove | "Theorem 4.5" (`accounting.py:22,125,343,380,402,427`) — `accounting.py:7` says it "has not been restated under any number in the current draft" |
| degree bound | "Assumption 3.1 / 6.2" (`sparse_expand.py:184`) **vs** "Assumption 5.2" (`run.py:487`) |
| noisy base mechanism | "Assumption 3.2" (`base_mechanism.py:4,160`, `__init__.py:6`) **vs** "Assumption 6.3" (`sparse_gnn.py:15,198`) |

**The schemes are dateable**, which makes this cheap to fix rather than
mysterious. Reconciling against the manuscript versions:

| scheme | manuscript | what it calls things |
|---|---|---|
| `3.x` | pre-v36 | Assumption 3.1 (degree bound), 3.2 (noisy base mechanism), Algorithm 2 (SparseExpand) |
| `6.x` | v36 | Theorem 6.4 (substitution), Assumption 6.2 / 6.3, Algorithm 5 |
| `5.x` | **v39, the live draft** | Theorem 5.4 / 5.6, Assumption 5.2, Algorithm 3 (`SparseExpand_in`) |

So the "5.x" citations are the *current* ones and the "3.x"/"6.x" are strata from
two earlier drafts. `accounting.py:4-11` is honest about the v36 renumbering but
predates v39. One pass to move everything to v39 numbering would settle it.

Note also that `paper/` in this repo is **not** the live manuscript — v39 lives
outside the repo (`~/Downloads/Markovian_DP_Subsampling (39).pdf`). The
`paper/*.tex` experiment write-ups are current; the bundled
`Markovian_DP_Subsampling (26).pdf` is many revisions stale.

**MED** — the `--direction out` ablation is accounted by a theorem the current
theory doc does not state (`accounting.py:7-10`, `README.md:118-121`). Any
orientation-ablation result carries an unconfirmed guarantee.

---

## 6b. Datasets, splits, and the four different inductive paths

15 dataset keys (`src/datasets.py:12-41`) plus arbitrary `relbench:<db>/<task>`.
Three keys are **dead**: `bluesky` (stub raises, `datasets.py:379-384`),
`ogbl-collab` (all-ones masks, no link-prediction mechanism, `:117-121`),
`cora-ml` (no masks, crashes `src.sparse.run` at `run.py:511`).

### There are four inductive paths, not two

| Path | Where | Semantics |
|---|---|---|
| (a) natively inductive loaders | PPI disjoint union (`datasets.py:167-185`), RelBench temporal (`relbench_data.py:271-273`) | measured 0 cross-partition arcs |
| (b) `--inductive` | `run.py:402-422` | drops cross-split arcs **for training**, restores all of them **for scoring** |
| (c) `--common_inductive_split` | `run.py:364-381` | cuts `data.edge_index` itself — permanently |
| (d) the baseline harness | `inductive.py:169-186` → `baselines.py`, `dpar.py`, `upstream.py` | three materialised cross-edge-free partitions |

**(a) ≡ (c) ≡ (d). (b) is not.** Path (b) trains on the cut graph and evaluates on
the *uncut* one; every baseline is evaluated on the **test-induced subgraph**,
where (measured) 60–76 % of arcs are gone on arxiv/flickr and test nodes keep only
test–test edges.

**HIGH — so our numbers and the baseline numbers are on different evaluation
protocols.** SparseExpand results from `ladder_stage01.sh` / `*_matched_eps.sbatch`
cannot be tabulated against baseline results from `src.experiments.run`. The fair
path *exists* — `--common_inductive_split`, and `configs/inductive_comparison.json`
+ `configs/sparse_inductive_ablation.json` are built to pair on it — but **no
campaign in `results/` uses it**. Mitigating: the head-to-head rows in
`paper/experiments_current.tex:168-171` are still empty, so nothing unfair is
published yet.

### 6b-i. HIGH — RelBench training reads the rows that define its own labels

`relbench_data.py:271-273` cuts the training graph at a **per-split** timestamp
(the max over all train rows), not per row. A train row's label is computed from
rows in `(t_row, t_row + Δ]`, and those rows are ≤ `train_end`, so they are in the
graph.

Measured on `rel-f1/driver-top3` at `direction=in, r=2, p2=1`: a train root's
rooted subgraph averages 269.5 nodes, of which **122.8 (46 %) are strictly later
than the root's own timestamp**. The `results` table's `position`/`points` columns
are node features — the answer is literally two hops away during training.

Test rows *are* clean (0 future nodes reachable; `get_db()` truncates at
`test_timestamp`), so the reported test metric is not inflated. But the model is
trained on a shortcut that vanishes at test time — a plausible mechanism for the
"ε trend runs backwards" anomaly at `experiments_current.tex:86-89`.

The module's own caveat (`relbench_data.py:27-30`) calls this "leakage between
training examples only", which understates it.

### 6b-ii. HIGH — the RelBench validation metric is leaky, and validation now picks checkpoints

`data.edge_index` is the test-cutoff graph, used unchanged for **all three** splits
(`base_mechanism.py:75-76`). Measured: a val row reaches on average **123.5 nodes
dated after its own timestamp**. No val-cutoff graph is ever built
(`relbench_data.py:280-281` builds only `train_edge_index`). Since `30ceee7`,
`summarize_sweep.py` selects the best checkpoint on validation — i.e. on a
contaminated signal.

### 6b-iii. HIGH — RelBench feature statistics are fit on the whole database

`relbench_data.py:97-99` computes `mu`/`sd` over every row of every column, and
`:82-92` builds the categorical vocabulary the same way — including the entire val
window and everything up to the test cutoff. Two distinct problems:

1. *Leakage*: val/test-period feature distributions shape the representation the
   model trains on.
2. **Un-accounted node-DP channel**: the features the mechanism consumes are a
   data-dependent function of **every** node. `accounting.py` charges nothing for
   it, so the released model depends on all nodes through a path outside the
   guarantee.

The **target** scaling at `:257-269` correctly uses train rows only; the feature
path does not.

### 6b-iv. Splits and reproducibility

- No unseeded randomness in loading.
- **MED — the baselines re-draw their split per seed; our method does not.**
  `experiments/run.py:70` passes the run seed as the split seed; `run.py:200-201`
  has a separate `--split_seed` (default 0) and `--seeds` is only a count. The
  reported ± are not the same quantity on the two sides.
- **MED — `facebook`'s split is a hard-coded `seed=0`, never varied**
  (`datasets.py:231, 586`). Every facebook run shares one 75/10/15 split, so the
  seed spread excludes split variance. ProGAP's protocol — cited at
  `datasets.py:249-250` — averages over random splits.
- **MED — no `to_undirected`**, so ogbn-arxiv is used directed (which is why
  `cap_mode=auto` picks `directed` there — consistent, but worth stating).
- MED — `data/inductive_splits/*.pt` caching validates **only node count**
  (`inductive.py:265-278`).
- MED — `--common_inductive_split` + RelBench is silently wrong: `run.py:364-381`
  rewrites `data.edge_index` but leaves `data.train_edge_index`, and
  `run.py:403-410` then prefers the latter.
- LOW — nothing is version-pinned except cora-ml (`datasets.py:207-208`). This
  matters most for RelBench, where node ids come from `db.table_dict` order.

### 6b-v. MED — the README's edge-drop numbers are off for Reddit

`README.md:168-170` says the cross-split drop is "68–76 % of edges on ogbn-arxiv,
Flickr, and Reddit". Measured: arxiv 67.9 %, flickr 75.8 %, **reddit 54.4 %**.

---

## 7. Hyperparameters — the defaults, and what actually overrides them

### `src/sparse/run.py` defaults

| Flag | Default | Note |
|---|---|---|
| `--aggr` | `mean` | `:180` |
| `--direction` | `in` | `:202` |
| `--p1` / `--p2` | `[0.5]` / `[0.5]` | `:209,212` — never used; every driver sets them |
| `--hidden` | `64` | `:221` |
| `--dropout` | **`0.5`** | `:223` — Planetoid default |
| `--weight_decay` | **`5e-4`** | `:234` — Planetoid default |
| `--lr` | `0.01` | `:230` |
| `--momentum` | `0.0` | `:231` |
| `--optimizer` | `auto` | `:224` → **`sgd` if `--dp` else `adam`** (`:618-619`) |
| `--clip` | `1.0` | `:239` |
| `--K_in` / `--K_out` | `None` / `= K_in` | `:258,262` — **no cap by default** |
| `--cap_seed` | `None` | `:272` — re-cap per seed |
| `--accounting_grid` | `1e-4` | `:252` |
| `--track_every` | `0` | `:292` |
| `--eval_every` | `50` | `:307` |

**LOW** — `README.md:65-69`'s example command passes neither `--dropout` nor
`--weight_decay`, so it runs at 0.5 / 5e-4 — the settings
`_dataset_settings.sh:40-47` documents as costing PPI 7 points of micro-F1. The
README example does not reproduce the paper's setup.

### What the drivers actually use (`scripts/_dataset_settings.sh`)

Uniform: `--dropout 0` `--weight_decay 0` (`:51`), `CLIP=1.0` (`:114`),
`SEEDS=3` (`:113`), `P2_GRID=(1.0 0.5 0.25 0.1)` (`:111`),
`SIGMA_GRID=(2 5 10 20)` (`:112`), `--roots_from train` everywhere.

| Dataset | model | p1 | T | K_in/K_out | inductive | lr (DP) |
|---|---|---|---|---|---|---|
| ppi | `multilabel_gnn` | 0.01 | 2000 | 5 / 5 | — | 1.0 (fallback) |
| relbench* | `binary_gnn` | 0.05 | 900 | **20 / 3** | yes | 1.0 (fallback) |
| facebook | gnn | 0.013 | 500 | 5 / 5 | — | 0.3 |
| reddit | gnn | 0.002 | 500 | 5 / 5 | yes | 0.3 |
| *else* (arxiv, flickr) | gnn | 0.005 | 500 | 5 / 5 | yes | 1.0 (fallback) |

### Optimizer and learning rate — the asymmetry

**MED.** Non-DP runs are **Adam @ lr 0.01** (`_dataset_settings.sh:115`); DP runs
are **SGD @ lr 0.3** (facebook/reddit) or **lr 1.0** (everything else, the
`:118` fallback). This is baked into `run.py:618-619`'s `auto`, so it is the
default behaviour, not just a driver choice.

Every "DP vs non-DP ceiling" gap therefore confounds *privacy noise* with
*optimizer change*. `scripts/sweep.sh:13` shows awareness ("non-DP SGD
reference, so the DP gap excludes the optimizer change") but that arm is one
sweep, not the standard. `results/facebook_tune/` has the direct comparison
(adam 0.01/0.05/0.1 vs sgd 0.1/0.3/1.0, all DP).

`sbatch/ppi_matched_eps_adam.sbatch` (**untracked**) is the clean A/B: identical
grid and identical calibrated σ, `--optimizer adam --lr 0.01` vs the SGD run's
`--lr 0.3`. Not yet in git; no `results/ppi_matched_eps_adam/` yet.

**MED** — the PPI matched-ε run uses `--lr 0.3`, but the PPI branch of
`_dataset_settings.sh` sets no `LR_DP`, so the ladder's PPI DP runs use the
**1.0** fallback. PPI DP results exist at both learning rates.

### L vs r — the policy changed and two drivers did not follow

**MED.** `_dataset_settings.sh:24-37` documents the current rule: **L = 2 fixed,
r sweeps independently**, and explains the old `L = r` rule was wrong ("the rung
measured model capacity, not sparsification"). But:

| Driver | what it passes |
|---|---|
| `scripts/ladder_stage2.sh:47` | `--num_layers $L` ✅ current |
| `scripts/ladder_stage01.sh:70` | `--num_layers $L` ✅ |
| `sbatch/ppi_matched_eps.sbatch:69` etc. | `--num_layers 2` ✅ |
| `scripts/relbench_f1.sh:72,86` | `--num_layers $R` ❌ old L=r rule (and `:11` documents it) |
| `scripts/orientation_ablation.sh:60,64,69` | `--num_layers $CEIL_R` ❌ |

Visible in the results: `results/ppi/inductive_stage1_ppi_r1` and
`inductive_stage2_ppi_r1` record `L=1`; everything newer records `L=2`. The
RelBench ladder and the orientation ablation are still on the superseded policy.


> **ADDENDUM — POLICY CHANGED 2026-09-13.** Main experiments now use
> **r = L = 2**. `scripts/_dataset_settings.sh` sets `R_VALUES=(2)` with `L=2`,
> and the DEPTH header is rewritten: matching them is what makes the rooted
> computation exact, since an L-layer mean-GNN reads exactly the radius-L ball.
> `r` is varied only in the ablations, with `L` moving alongside it.
>
> `run.py` now prints a NOTE whenever `r != L`, naming which way the mismatch
> cuts (`r > L` pays ε for hops the model never reads; `L > r` trains on a
> truncated neighbourhood and evaluates on a full one). A warning, not an error,
> because ablations vary them on purpose. Verified.
>
> This also resolves the inconsistency below: `relbench_f1.sh` and
> `orientation_ablation.sh` pass `--num_layers $R`, which now agrees with the
> policy rather than contradicting it.
>
> **Still to do — 19 call sites pin `--r 1` with `--num_layers 2`** and are now
> off-policy: `sbatch/ppi_matched_eps.sbatch`, `ppi_matched_eps_adam.sbatch`,
> `arxiv_inductive_matched_eps.sbatch`, `facebook_settings*.sbatch`,
> `reddit_settings.sbatch`, `sparse_inductive.sbatch`, and
> `scripts/_facebook_width.sh`, `_ppi_pareto_grid.sh`, `_ppi_gcn_arm.sh`,
> `_ppi_stage1_diag.sh`. These were left alone deliberately: moving them to r=2
> costs roughly **8–9× ε** (measured facebook r=1 7.19 → r=2 57.57), and
> `ppi_matched_eps.sbatch:31` documents the r=1 choice as buying "+1.8 points at
> K=5 for 9.4x epsilon". Their σ would be re-solved automatically by
> `calibrate_grid.py`, so the change is mechanically safe — but it is a
> deliberate privacy-budget decision, not a config tidy-up.

### Width

**MED.** `hidden` is not uniform: 16 for `facebook_settings*.sbatch:42,50`,
`facebook_tune_ice.sbatch:38`, `_coverage_sweep.sh:64`; 256 for PPI/arxiv/
relbench matched-ε and `_facebook_width.sh`. Cross-dataset comparisons at
different widths. `results/facebook_width/` exists to quantify exactly this.

**LOW** — older CSVs have no `hidden` column at all (schema grew), so provenance
for `results/facebook_ice/*`, `results/reddit/*`, several `results/ppi/*` is
partially lost.

---

## 7b. DP mechanics — what is correct, and what is not

### Correct, and verified by measurement (not just by reading)

- **Clipping is per-rooted-subgraph**, inside the per-loss loop
  (`sparse_gnn.py:85-93`), on the **global L2 norm across all parameter tensors
  concatenated** (`base_mechanism.py:162-164`), and it is *clip-if-exceeds*
  (`min(1, C/‖g‖)` via `.clamp(max=1.0)`), not scale-to-C.
- **Noise is drawn once per step on the summed gradient, and the division
  happens after** — `p.grad = (acc + z) / denom` (`sparse_gnn.py:101-102`). This
  is the correct DP-SGD ordering. (Dividing first would have meant `E[B]×` too
  much noise.)
- **The denominator is safe.** `denom = max(p1 · pool_size, 1.0)`
  (`sparse_gnn.py:70, 184-186`), where `pool_size` is fixed before training
  (`run.py:509-511`). It is *not* the realized `|V_root|` — the classic DP leak
  is avoided, and there is a regression test (`test_dp_mechanics.py:305-319`).
- **The sampler is genuinely Poisson.** `sparse_expand.py:338-344`:
  `keep = torch.rand(n, generator=generator) < p1`, independent Bernoulli per
  node, not a fixed-size sample. The amplification claim is not undermined by
  the sampler. Test asserts Binomial variance (`test_dp_mechanics.py:274-282`).
- **Empty root sets still take a noise-only step** (`sparse_gnn.py:72-79`,
  `196-202`), which is what fixed-T composition requires.
- **`--track_every` does not change the trajectory.** Verified empirically:
  final parameters bit-identical (max|Δ| = 0) across
  `track_every ∈ {0,5} × dropout ∈ {0,0.5} × dp ∈ {F,T}`. The claim at
  `sparse_gnn.py:164-167` holds.
- Separate RNG streams for sampling (`seed`) and noise (`seed + 10_000`)
  (`sparse_gnn.py:181-182`).

### 7b-i. MED — "DP vs non-DP" is a *three*-variable comparison

Beyond the optimizer swap in §7, the two paths differ in **gradient scale**.
`_step_nondp` (`sparse_gnn.py:39-54`) takes the **raw sum** of per-root losses
with no normalization; `_step_dp` divides by `E[B]`. On PPI `E[B] ≈ 512`, so at
the same `--lr` the non-DP path takes ~512× larger steps. Stacked with
Adam-vs-SGD and 0.01-vs-0.3, the PPI headline comparison —
`sbatch/ppi_matched_eps.sbatch:120` quotes Stage-1's 0.6256 / 0.5314 against DP
cells — differs in optimizer, learning rate, *and* nominal gradient scale.

Cleanest fix: normalize `_step_nondp` by the same `denom`, then compare at
matched optimizer (`--optimizer sgd` already allows this).

`_step_nondp` is also a **separate implementation**, not "the DP path at σ=0",
so nothing structurally forces the two to agree.

### 7b-ii. MED — the degree-cap RNG and the root-sampling RNG use the same seed

`run.py:444` `cap_gen = torch.Generator().manual_seed(int(cap_seed))` with
`cap_seed = s` (`run.py:501`), and `sparse_gnn.py:181` `sample_gen =
_make_generator(seed)` with the same `s`. Two fresh generators from the same
seed draw the **same uniform sequence**, so which arcs survived the cap and
which nodes are roots at step 1 are deterministically coupled. The amplification
argument wants the Bernoulli root draws independent of the graph construction.
The noise stream is safely offset; this one is free to fix the same way.


> **ADDENDUM — FIXED 2026-09-12.** The degree-cap generator is now seeded
> `cap_seed + 20_000`, so it no longer draws the same uniform sequence as the
> root sampler (which uses `seed`). Note this changes the capped graph for a
> given seed, so numbers will shift slightly from previously-recorded runs.

### 7b-iii. MED — the reported number is a best-of-N checkpoint chosen on private labels

`--track_every 100` at T=2000 gives 20 checkpoints; `summarize_sweep.py:103-107`
picks the best on **validation** and reports test at that step. Under node-level
DP the validation nodes are nodes of the same protected graph, so this is
post-hoc early stopping on protected data and its cost is not in the accountant.
The module docstring (`:12-16`) correctly identifies *test*-selection as leakage
but treats val-selection as free. Worth a sentence in the paper either way.

### 7b-iv. MED — `run_sparse_inductive_ablation.py` silently runs at CLI defaults

`scripts/run_sparse_inductive_ablation.py:41-52` builds its command line without
`--dropout`, `--weight_decay`, `--optimizer`, or `--lr`, so that ablation runs
at dropout 0.5, wd 5e-4, SGD, **lr 0.01** — a 30× smaller DP learning rate than
every other DP cell in the repo. Confounded four ways against anything else.

### 7b-v. Lower-severity

- **GPU only:** `base_mechanism.py:183-195` reuses a pinned CPU buffer copied
  with `non_blocking=True` and no sync, so a deep CUDA queue could let the host
  overwrite it before the copy runs. The sbatch jobs are all `ice-cpu`, so this
  cannot fire today. A correct fix for §0 removes it too.
- Every mechanism **except** `GNNMechanism` holds all per-root autograd graphs
  simultaneously (`base_mechanism.py:137-140` vs `gnn_mechanism.py:123-142`).
  PPI — the headline experiment — keeps ~512 forward graphs alive per step.
  Likely the cause of the 8 h wallclock timeout described at
  `ppi_matched_eps.sbatch:88-91`. Correctness unaffected; a batched-chunk
  override for `MultiLabelGNNMechanism` is a free speedup.
- Weight decay is applied to the already-divided gradient, so its ratio to the
  signal is `E[B]×` larger in DP runs than a mean-gradient setup would give.
  Harmless at the drivers' 0.0.
- `max_batched_subgraph_nodes = 8192` (`gnn_mechanism.py:58`) is not exposed on
  the CLI.
- `train_sparse_gnn_with_budget` (`sparse_gnn.py:227-272`) is exported and
  tested but unused — `run.py` calibrates inline at `:571-579`. Two calibration
  code paths that can drift.

---

## 8. Task heads, losses and metrics

Test suite state: **176 passed, 0 failed** (25 s, `PytorchEnv`). Note
`README.md:44` still says "106 tests". The suite is green *and* the items below
are real — the tests cover the DP mechanics and the accounting thoroughly, and
the regression/metric semantics barely at all.

### The five heads

| Class | Task | Loss | Primary metric | Secondary |
|---|---|---|---|---|
| `GNNMechanism` | single-label | `F.nll_loss` on `log_softmax` | accuracy | — |
| `MLPMechanism` | single-label, graph-blind | `F.nll_loss` | accuracy | — |
| `MultiLabelGNNMechanism` | multilabel (PPI) | `BCEWithLogits`, **mean over 121 labels** | micro-F1 @ logit>0 | micro-AUROC |
| `BinaryGNNMechanism` | binary (RelBench) | `BCEWithLogits`, 1 logit | AUROC | `bin_acc` @ logit>0 |
| `RegressionGNNMechanism` | regression (RelBench) | `F.mse_loss` | MAE (original units) | RMSE, R² |

All metrics are hand-rolled except the GAD side pipeline (sklearn). **No
`pos_weight`, no class weighting, no label smoothing anywhere** — so our method
and the blind arm are symmetric on that axis. Backbone is stock PyG:
`SAGEConv(aggr='mean')` or `GCNConv(add_self_loops=True, normalize=True)`
(`src/sparse/layers.py:27-38`) — "our averaging" *is* GraphSAGE-mean, not a
custom aggregator. No BatchNorm/LayerNorm anywhere.

**Benchmark alignment is good** (verified against the installed `relbench`):
regression tasks declare `metrics = [r2, mae, rmse]` and
`relbench.metrics.r2 = sklearn.r2_score`, computed against the evaluated
split's own mean — which is what `regression_mechanism.py:93-97` does. Commits
`4562ee7` and `b10a794` did what their messages claim. OGB accuracy ✅,
PPI micro-F1 ✅, RelBench AUROC ✅.

**Evaluation protocol is clean.** All five heads `@torch.no_grad()` + `.eval()`,
full-batch with a CSR fallback above 250M arc×feature elements (unit-tested to
1e-9), dedicated `sample_gen`/`noise_gen` so evaluation consumes no global RNG —
the "a tracked run follows the same trajectory as an untracked one" claim
(`sparse_gnn.py:163-167`) **holds**.

**Checkpoint selection is correct where it matters.** `summarize_sweep.py:95-116`
selects on the *val* curve and reports the *test* value at that step — commit
`30ceee7` is genuinely implemented and is the only selection path. Direction
(`:48-64`) is right for every metric: MAE/RMSE lower, R² higher, everything else
higher. Nothing in `src/sparse/` consumes a test metric for any decision (no
early stopping, no best-state save).

### 8a. HIGH — regression targets are scaled but never mean-centred

`src/sparse/relbench_data.py:264-269` does `y = y / target_std` and **nothing
else**. Verified by reading it. Three places state the opposite:

- `relbench_data.py:256-262` — "'predict the train mean' is exactly 'predict 0'
  in z-space, which is what the trivial-baseline computation in run.py relies on"
- `regression_mechanism.py:8-13` — "Targets are expected in Z-SCORED form
  (train-split mean subtracted, …)"
- `run.py:96-99` — "mae → MAE of 'always predict the train mean' on test"

`run.py:113-115` actually computes `mean(|y_test|) * target_std`, which is the
MAE of the **all-zero** predictor. The comment at `relbench_data.py:258-260` is
self-contradictory: MAE is translation-invariant *as a function of residuals*,
but the trivial predictor's own value is not.

Consequence: on RelBench's non-negative heavy-tailed targets (LTV, sales) the
zero predictor is far worse than the train-mean predictor, so the printed floor
is **inflated** and "beats trivial" is too easy. `paper/experiments_current.tex:73`
("GNN beats trivial (77.13)") rests on this bar. Every
`trivial baseline (mae) on test: …` line in `ice_status_report.txt` is the wrong
reference.

**R² is unaffected** (translation-invariant), so `test_r2` is currently the only
trustworthy "does it beat trivial" signal for regression — and it is negative in
several logged runs. Secondary effect: MSE on an un-centred target forces a large
learned intercept, which is exactly the signal per-root clipping at `C` removes.


> **ADDENDUM — FIXED 2026-09-12.** `trivial_baseline`'s `mae` branch now
> computes `mean(|y_test − mean(y_train)|) × target_std` — the actual
> train-mean predictor — instead of `mean(|y_test|) × target_std`, which was
> the all-zero predictor. I fixed the *baseline* rather than centring the
> target, because centring would change training and invalidate every existing
> regression run; this changes only the reported bar. The three docstrings that
> claimed the target was z-scored (`relbench_data.py`, `regression_mechanism.py`,
> `run.py`) now say plainly that it is scaled but **not** centred, and note the
> consequence for the learned intercept under clipping.
> **Still to do:** the RelBench regression rows should be re-read against the
> corrected bar; the old `trivial baseline (mae)` lines in `ice_status_report.txt`
> are still the wrong reference.

### 8b. HIGH — the baseline harness selects the *worst* checkpoint on regression

`src/experiments/baselines.py:114` `best_val = float("-inf")`, `:127`
`if validation > best_val`. For `regression=True`, `_evaluate` →
`_task_metric(..., regression=True)` → `_regression_mae` (`dpar.py:252-255`),
so "validation" is **MAE, lower-is-better** — and the kept state is the one with
the *highest* MAE. Same pattern at `dpar.py:334, 349-351`. Verified by reading
all three sites.

Commit `b10a794` threaded a `regression` flag through loss and metric but never
flipped the selection direction. Every regression number from the
`mlp`/`dp_mlp`/`graphsage` baseline arm is the worst checkpoint — **which
flatters our method**. `summarize_sweep.py` got this right; this path did not.


> **ADDENDUM — FIXED 2026-09-12.** `baselines.py` now seeds `best_val` with
> `+inf` and tests `validation < best_val` when `config.regression` is set,
> so regression runs keep the lowest-MAE checkpoint rather than the highest.
> `dpar.py` needed no change — `DPARConfig` has no `regression` field, so it
> can never run regression — but that exclusion was only a comment, so
> `experiments/run.py` now **raises** when `regression=true` is paired with any
> method other than `mlp`/`dp_mlp`/`graphsage`. That closes the third gap listed
> in §8e (a `dpar` + `regression` config previously died inside
> `cross_entropy` on float targets).

### 8c. HIGH — `summarize_sweep.py` picks the first checkpoint on a NaN val curve

`scripts/summarize_sweep.py:105-107` uses `max(val_curve, key=val_curve.get)`.
`max` seeds with the first element and every comparison against NaN is False, so
one leading NaN makes it report the **earliest tracked step** as best, silently —
the leakage `WARNING` never fires because the dict is non-empty. Reachable:
AUROC → NaN on a single-class val split (`binary_mechanism.py:47-48`,
`multilabel_mechanism.py:67-68`), R² → NaN when `ss_tot==0`, any metric on an
empty mask. Small RelBench tasks are exactly where this bites.


> **ADDENDUM — FIXED 2026-09-12.** `summarize_sweep.py` now filters NaN out of
> the curve before `min`/`max` (via `v == v`), warns how many checkpoints were
> dropped, and skips the cell entirely with a message if every checkpoint is
> NaN. Also fixed the adjacent `if eps.get(best)` test, which printed `-` for a
> genuine ε of exactly 0.0.

### 8d. MED — `_micro_auroc` has no tie handling

`multilabel_mechanism.py:69-72` assigns ranks in `argsort` order with no
averaging within ties, unlike the binary `_auroc` which explicitly averages tied
ranks (`binary_mechanism.py:52-55`). A constant predictor therefore does not
score 0.5 — measured 0.4988 on a PPI-shaped target. This is the origin of
`README.md:166`'s "AUROC 0.4955". Only `_auroc` has a tie test.


> **ADDENDUM — FIXED 2026-09-12.** `_micro_auroc` now averages ranks within
> ties, matching `binary_mechanism._auroc`. Verified: all-ones, all-zeros and
> constant-0.3 predictors now score **exactly 0.5** (was 0.4988), and on
> untied scores it agrees with `sklearn.roc_auc_score` to machine precision
> (0.501998135309467 vs 0.5019981353094671).
> **Note:** README's quoted PPI floor of "AUROC 0.4955" was an artefact of the
> old implementation and should be restated as 0.5.

### 8e. MED — other gaps

- RelBench also lists `average_precision` and `f1` for binary tasks; we compute
  neither. **AP is the imbalance-sensitive metric** a RelBench reviewer looks
  for, and `sklearn.average_precision_score` is already imported in
  `gad/metrics.py:11`.
- The baselines path reports regression MAE in **z-space**
  (`dpar.py:230`, no `target_std`) while `regression_mechanism.py:90-91` reports
  **original units**. `dpar.py:226-228` claims they match. They differ by
  `target_std` (~40× on rel-amazon/user-ltv). Do not table them side by side.
- `trivial_baseline` reads **test** labels for micro-F1/MAE, but the majority
  class for accuracy comes from **train** (`run.py:109-118`). Inconsistent.
- The `e4363e5` regression guard (`run.py:390-396`) only fires when `task_type`
  exists (RelBench only), does **not** catch the reverse mispairing
  (`regression_gnn` on a classification task → silent bogus "MAE"), and
  `src/experiments/run.py` has no equivalent — `method="dpar"` with
  `regression: true` still hits the crash the guard was written to prevent.
- Alt-graph secondaries are computed then dropped: `sparse_gnn.py:110-112`
  produces `test_rmse_alt`/`test_r2_alt`/`test_bin_acc_alt`, but
  `run.py:666-668` writes only `*_acc_alt` and `*_auroc_alt`.

---


> **ADDENDUM — PARTLY FIXED 2026-09-12.**
> - **Alt-graph secondaries no longer dropped.** `run.py` now writes
>   `train/val/test` × `rmse_alt`, `r2_alt`, `bin_acc_alt` — nine columns that
>   `_evaluate` was already computing and the writer discarded. Smoke-tested:
>   header and rows both 71 columns.
> - **The DPAR regression guard** is fixed — see the §8b addendum.
> - **Not fixed:** average precision and F1 are still not computed for RelBench
>   binary tasks; `trivial_baseline` still reads test labels for micro-F1/MAE
>   while taking the majority class from train; the reverse mispairing
>   (`regression_gnn` on a classification task) is still unguarded.

## 9. Baselines — kept separate

**The headline: there are two disjoint "baseline" universes, and they have never
met.**

### 9a. HIGH — the external-baseline harness has never been run

`src/experiments/run.py:135-139` writes JSON to
`results/inductive/<dataset>/<method>.json`. `results/` contains **420 CSVs and
zero JSON files**; that directory does not exist. No sbatch job, no `scripts/*`
driver, and no README line references `dpar|progap|heterpoisson|dp_gnn|graphsage`.
`configs/inductive_comparison.json` is consumed by nothing.

| Baseline | Where | Status |
|---|---|---|
| `mlp`, `graphsage`, `dp_mlp` | `src/experiments/baselines.py` | first-party, wired, never run |
| `dpar` | `src/experiments/dpar.py` | reimplemented in PyTorch; only the RDP accountant is vendored |
| `dp_gnn` | `src/experiments/dpgnn.py` | reimplemented |
| `progap`, `heterpoisson` | `third_party/`, via `upstream.py:115-185` subprocess bridge | vendored; **ProGAP cannot import** (see 9c) |

So every "baseline" number that exists today is **our own mechanism at `--r 0`**
— see §4d for why that arm is not blind. The row literally labelled `baselines`
at `experiments_current.tex:77` reproduces
`results/relbench/relamazon_itemltv_meps/blind_eps*` exactly.

**Good news: nothing is fabricated.** The DPAR/ProGAP/DP-MLP table at
`experiments_current.tex:160-176` is correctly left empty, and no baseline number
is copied from a paper. The only staleness is prose in sbatch headers.

### 9b. HIGH — the privacy notions are not comparable, and no conversion exists

Ours is node **substitution** (`accounting.py:243-245`, the `±2k` pair). **Every**
baseline is add/remove. On top of that: three δ conventions across the harness
(`n^-1.01` / `1e-5` / `5e-4` / `1/(10n)`) and **five different accountants**
(PLD, PRV, TF-Privacy RDP, `dp_accounting` RDP, autodp, plus a bespoke one).

No conversion is applied anywhere. Mitigating: **no comparison point exists yet**,
so nothing is wrong in print — but the moment that table is filled in, ε=8 will
mean six different things across six rows.

### 9c. HIGH — the vendored baselines cannot run as configured

- **ProGAP cannot import.** `third_party/ProGAP/core/data/` is missing on disk;
  `git check-ignore` confirms the root `.gitignore:48` (`data/`) swallowed it.
  `progap/node.py:6,11` needs it.
- ProGAP/HeterPoisson configs hardcode
  `/usr/scratch/asaha92/envs/.../bin/python`.
- ProGAP/HeterPoisson/DP-GNN are **single-label only** — they cannot run on PPI or
  RelBench at all, which is most of the current experimental surface.

### 9d. HIGH — the baselines are run far off their own papers' settings

| | our config | the paper's |
|---|---|---|
| ProGAP | `epochs=100, batch=32, max_degree=5` | `epochs ∈ {5,10}, batch=256, max_degree=100` (`ProGAP/experiments.py:52-55`) |
| HeterPoisson | `batch=32, epochs=100, lr=1e-3, num_neighbors=1` | `4096 / 9 / 0.01 / sweep 1–5` |

That is ~60× the composition for ProGAP. Also: the DPAR port trains on a
**70-node induced subgraph**, while upstream gives the remaining train nodes
identity PPR rows and trains on all of them (`DPAR/main.py:168-173`).

### 9e. HIGH — the baselines are scored on mutilated held-out graphs

`inductive.py:169-186` deletes every cross-partition edge from **all three**
partitions, so baselines evaluate on test-induced subgraphs. This is precisely the
treatment `src/sparse/run.py:476-481` explicitly rejects for *our* method — its
comment notes that filtering "would leave held-out nodes with no edges at all (on
PPI their mean in-degree drops 29.3 → 0)". We gave ourselves the un-filtered eval
graph and the baselines the filtered one. See also §6b.

### 9f. Hyperparameter asymmetries

| | ours | baselines |
|---|---|---|
| optimizer | **SGD** (DP) / Adam (non-DP) | **Adam**, unconditionally (`baselines.py:110`, `dpar.py:302`, `dpgnn.py:220`) |
| lr | 0.3 (DP) | 1e-2 (`baselines.py:23`), 3e-3 (`dpgnn.py:34`), 5e-3 (`dpar.py:49`) |
| hidden | **256** | 16 / 32 / 64 / 100 / 128 |
| clipping | one global-norm clip | DP-GNN clips **per parameter tensor** (`dpgnn.py:198-207`) |
| target-ε path | yes | `dp_mlp` and `dp_gnn` have **none** |
| steps | T=2000 | `mlp`/`graphsage` get 100 full-batch steps; `dp_mlp` gets 100 epochs of minibatches |

Our own README calls width "free in ε" — so running baselines at 16–128 while we
run at 256 is a gap we have explicitly argued costs nothing to close.

**Also:** DP-GNN's clip threshold is a **non-private 75th percentile of real
gradients** (`dpgnn.py`) — an unaccounted data-dependent choice in the baseline.

### 9g. MED — two DP-correctness bugs on the baseline side

These mirror problems our own engine gets right (§7b):

- `baselines.py:173` divides by `selected.numel()` — the **realized** Poisson
  batch size. That is the data-dependent-denominator leak our engine avoids.
- `baselines.py:155-156` **skips the step entirely** on an empty Poisson sample
  (no noise, no step), while the accountant at `:146` charges
  `epochs × steps_per_epoch` steps regardless.
- `dpar.py:338` draws fixed-size `randperm` minibatches but `dpar.py:404`
  accounts them with **Poisson amplification** (`sample_rate = batch/N`).
  Shuffling is not Poisson subsampling. (May be deliberate upstream fidelity —
  but it should be stated if the numbers are ever printed.)

---

## 10. Results provenance — which directories are on which policy

Built by reading one row from each of 144 result CSVs. The columns that matter
for comparability are `L`, `eval_graph`, `optimizer`/`lr`, `hidden`, `cap_mode`.

| Family | L | eval_graph | opt / lr | hidden | cap_mode | Comparable with |
|---|---|---|---|---|---|---|
| `ppi_matched_eps/*` | 2 | **train** | sgd 0.3 | 256 | undirected | `ppi_stage1/*` |
| `ppi_stage1/*` | 2 | train | adam 0.01 (non-DP) | 16–256 | undirected | `ppi_matched_eps/*` |
| `ppi/*` (older) | 1–2 | **full** | mixed | — | undirected | *not* with the above |
| `arxiv_matched_eps/*` | 2 | full | sgd 0.3 / adam 0.01 | 256 | **directed** | internally |
| `facebook_width/*` | 2 | **train** | sgd 0.3 | 256 | undirected | `coverage_facebook/*` |
| `facebook_ice/*`, `facebook_tune/*` | 2 | **full** | sgd 0.3 / adam 0.05 | — | undirected | *not* with `facebook_width` |
| `coverage_facebook/*` | 1–2 | train | sgd 0.3 | 16 | undirected | — |
| `reddit/transductive_*` | 2 | full | sgd 0.3 | — | undirected | internally |
| `relbench/relf1_*` | 2 | full | — | — | directed | internally |

Also: `results/relbench/relf1_regression_smoke` has **empty K_in/K_out** — an
uncapped run (non-DP, so harmless, but the pattern is the one `run.py:484-489`
warns about).

---

## 11. Open questions for the paper text

1. `paper/experiments_current.tex:100` labels a row "baselines" in the item-ltv
   table, while §"Baselines (RelBench)" says the baseline harness is **blocked**
   on RelBench ("split builder requires every node labeled"). So "baselines" in
   the results tables means the **r=0 blind arm**, not DPAR/ProGAP/DP-MLP. Two
   different things are called "baseline" in one document.
2. RelBench results are **1 seed** (stated for item-ltv and author-category).
3. user-churn's ε trend runs backwards (0.5622 at ε=1 → 0.5570 at ε=8,
   confirmed in the summarize output). The tex hypothesizes DP noise acting as
   regularizer under `--dropout 0 --weight_decay 0`. Worth also checking whether
   best-checkpoint selection over a tracked trajectory is picking noise.
4. The grid tables in §"The grid" are empty — most cells unrun.

---

## 12. Training procedure — the model, and the graph pipeline

Reference section rather than a list of problems. Everything here was read off
the code, not inferred.

### 12a. The GNN

One class, `_NodeGNN` (`gnn_mechanism.py:24-41`), duplicated essentially
verbatim in `multilabel_mechanism.py:36-41`, `binary_mechanism.py:36-41`, and
`regression_mechanism.py`. The layer widths:

```python
dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
self.convs = build_conv_stack(dims, aggr=aggr)
```

So `--num_layers 2 --hidden 256` on PPI is exactly `50 → 256 → 121`: **two conv
layers, one hidden layer.** `L` counts convolutions, not hidden layers.

The forward pass (`gnn_mechanism.py:35-41`):

```python
for i, conv in enumerate(self.convs):
    x = conv(x, edge_index)
    if i < len(self.convs) - 1:
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
return F.log_softmax(x, dim=1)      # multilabel/binary/regression: raw output
```

The convolutions are **stock PyG**, no subclassing (`layers.py:27-39`):

| `--aggr` | layer | formula |
|---|---|---|
| `mean` (default) | `SAGEConv(aggr="mean")` | `h_i = W_l·mean_{j∈N_in(i)} h_j + b + W_r·h_i` |
| `gcn` | `GCNConv(add_self_loops=True, normalize=True)` | `h_i = Σ_{j∈N_in(i)∪{i}} (d̂_j d̂_i)^{-1/2} Θ h_j + b` |

Parameters at `L=2, aggr=mean`: six tensors — `convs.{0,1}.lin_l.weight`,
`convs.{0,1}.lin_l.bias`, `convs.{0,1}.lin_r.weight`. Note `lin_r` carries **no
bias**. (This six-tensor layout is what §0's noise bug keys on: `lin_l.weight`
and `lin_r.weight` are the same shape at every layer.)

**What the stack does NOT have** — worth stating because reviewers assume some
of these:

- no BatchNorm, no LayerNorm anywhere in `src/`
- no residual connections, no jumping knowledge
- no dropout on the *input* features (Kipf's GCN has it); dropout sits only
  between conv layers
- no output L2 normalization (the original GraphSAGE L2-normalizes each layer)
- **no ReLU and no dropout after the final conv** — those are gated on
  `i < len(self.convs) - 1`. What *does* follow the last conv is the task's
  output transform, which differs per head (table below)
- neighbour sampling is Bernoulli(`p2`) once, in SparseExpand — **not**
  GraphSAGE's fixed fan-out re-sampled per layer per epoch

**Output transform and loss, per head.** Only the single-label heads apply an
output activation; the other three return the final conv's raw output and fold
the link function into the loss:

| head | after the last conv | loss |
|---|---|---|
| `gnn` (single-label) | `F.log_softmax(x, dim=1)` (`gnn_mechanism.py:41`) | `F.nll_loss` — together these are cross-entropy |
| `mlp` (blind) | `F.log_softmax(x, dim=1)` | `F.nll_loss` |
| `multilabel` (PPI) | **none** — `return x` (`multilabel_mechanism.py:41`) | `F.binary_cross_entropy_with_logits` (sigmoid fused into the loss) |
| `binary` (RelBench) | **none** — `return x.view(-1)` (`binary_mechanism.py:40`) | `F.binary_cross_entropy_with_logits` |
| `regression` (RelBench) | **none** — `return x.view(-1)`, unbounded (`regression_mechanism.py:42`) | `F.mse_loss` |

This is why the multilabel/binary metrics threshold at `logits > 0` rather than
`p > 0.5` — the two are the same point, just on opposite sides of the sigmoid
that was never applied. And it is why the regression head emits an unbounded
scalar, which is what makes the un-centred target in §8a bite: the network has
to learn a large intercept with no output transform to absorb it.

The loss is taken at the **root only** — local index 0 by `RootedSubgraph`
convention (`gnn_mechanism.py:79-82`). Every other node in the sampled subgraph
exists only to compute the root's representation. Roots without a training label
return `zero_loss()` (`:73-74`).

### 12b. The graph pipeline, in execution order

This is the complete list of what happens to `edge_index` between the loader and
training. There is nothing else.

1. **`load_dataset`** (`src/datasets.py`) — the raw PyG/OGB loader.
   **No PyG transforms are applied anywhere**: `grep transform src/datasets.py`
   finds no `transform=` or `pre_transform=`. Graphs and features arrive exactly
   as the upstream dataset ships them.
2. **`--common_inductive_split`** (optional, `run.py:364-381`) — replaces
   `data.edge_index` with only within-partition arcs, permanently.
3. **`--inductive`** (optional, `run.py:402-422`) — either uses the loader's
   `train_edge_index` (RelBench: everything at or before the train cutoff) or
   keeps only arcs with **both** endpoints in `train_mask`.
4. **`dedup_arcs`** (`run.py:437` → `sparse_expand.py:226-241`) — removes
   parallel arcs, because the path counts in the accounting assume a simple
   graph. Implemented as an int64 key `u*n + v` plus `torch.unique` rather than
   `torch.unique(dim=1)`, which would need several copies of a `[2, E]` tensor —
   on Reddit (114.6M arcs) that alone exhausts 16 GB. Returns the original
   tensor untouched when there are no duplicates.
5. **Degree capping** (`run.py:452` or `:458`) — see 12c.
6. **`build_adjacency`** (`run.py:494` → `sparse_expand.py:71-86`) — CSR over
   CPU. `direction='in'` keys rows by `edge_index[1]` (the target) and stores
   sources, so `adj.neighbors(u)` returns **the sources of arcs into u**.
   `'out'` swaps the rows.

**What is deliberately not done, and matters:**

- **No `to_undirected`.** ogbn-arxiv and RelBench stay directed. The undirected
  datasets (Planetoid, Flickr, Reddit, PPI) are undirected only because the
  upstream loader already stores both arcs.
- **No feature normalization.** Planetoid's usual `NormalizeFeatures()`
  row-normalization is absent. RelBench is the sole exception — it z-scores
  features at `relbench_data.py:97-99`, fit over the whole database (see §6b-iii).
- **No self-loop addition.** `GCNConv` adds them internally per layer;
  SAGE-mean never sees one.
- **No isolated-node removal**, no coalescing beyond the dedup above.
- **Facebook is the one dataset with its own processing**
  (`datasets.py:230-251`): one-hot features, class filtering by count, and
  removal of self-loops and isolated nodes — replicating ProGAP's pre_transform.

### 12c. Degree capping, exactly

Which algorithm runs is decided by `cap_mode` (`run.py:446-448`): `auto` picks
`undirected` iff `K_in == K_out` **and** the arc set is symmetric, else
`directed`.

**`cap_degrees_undirected`** (`sparse_expand.py:258-311`) — collapses arc pairs
to undirected edges, walks a uniformly random edge order greedily keeping an
edge when both endpoints are still under `K`, then re-emits both arcs. Result is
symmetric, so in- and out-expansion coincide on it. Some nodes finish *below* K,
which still satisfies the bound. A self-loop consumes one unit of its node's
capacity and is emitted as a single arc.

**`cap_degrees`** (`sparse_expand.py:177-224`) — caps in-degree first (`row=1`),
then out-degree (`row=0`). Both passes only remove arcs, so the second cannot
re-violate the first. On a symmetric graph this treats the two arcs of an edge
independently and **destroys symmetry** — measured, only ~1/3 of surviving arcs
keep their reverse at K=5. That is why `auto` avoids it there.

Shared properties:

- **Truncation is uniformly random**, not first-K: shuffle, then stable-sort by
  key, so ties land in uniformly random order (`:206-217`). Measured on a 10-arc
  star at `K_in=1` over 3000 seeds, the survivor histogram is flat.
- **Dedup happens first** (`run.py:437` before `:452`), so duplicate arcs cannot
  win the survival lottery twice.
- **Capped once per seed, not per step** (`run.py:500-505`), with
  `cap_seed = seed` unless `--cap_seed` pins one graph for all seeds. So the
  reported spread across seeds includes the cap's own variance.
- The realized maxima are recorded per run as `K_in_achieved` / `K_out_achieved`.

---

# Appendix — the union-graph problem, stated on its own

*Self-contained restatement. Nothing here depends on the rest of the document.*

## The mechanism and what it assumes

One training step samples each node as a root with probability `p1`, grows each
root's subgraph by walking incoming edges `r` levels and keeping each examined
arc with probability `p2`, computes one clipped gradient per root, sums them,
and adds Gaussian noise.

Theorem 5.4 bounds the privacy loss of that step by reducing it to two 1-D
Gaussian mixtures, built from two counts:

```
n_d = K_out^d                                     how many roots sit d hops from s
q_d = 1 - ∏_{ℓ=d..r} (1 - p2^ℓ)^(K^(ℓ-1))         how much one such root can change
```

## The problem

Both counts are counts of **paths and distances in a graph**. The proof requires
that graph to be `H = g ∪ g′` — the union of the two neighbouring graphs.

That is not a technicality. Lemma 4 establishes `q_d` by coupling: it places one
Bernoulli(`p2`) coin on every edge of `E_H = E ∪ E′` and runs the expansion
under `g` and under `g′` off the same coins. Under that coupling, a root `v`'s
subgraph differs between the two worlds only if some path from `s` to `v`
survived — **and that path may use an edge present in only one of the two
graphs.** Mixed paths are a real channel, so the counting must happen in the
union. v42 states this as Assumption 5.2, twice.

The code enforces a degree bound on `g` — the graph it holds. It cannot enforce
anything about `g′`, which is a counterfactual with nothing to call `cap` on.

**So the symbol `K` refers to two different objects.** The `K` enforced by
`cap_degrees`, and the `K` consumed by Eq (7)/(8), which is a property of the
union. If `g` and `g′` are both legal `K`-capped graphs, then at the substituted
vertex `s`:

```
out-arcs of s in H  =  (≤K from g) ∪ (≤K from g′)   →  up to 2K
```

Measured on Facebook (K=5, 300 random substitutions with both graphs legal): the
union realized max in-degree and max out-degree of exactly 10.

**Capping the held graph at `K` licenses Theorem 5.4 at `K_union = 2K`, not at
`K`.** There is no error in the theorem and no error in `cap_degrees`. The two
simply name different graphs.

## Why the correction is one factor of 2, not `2^d`

Definition 5.1 restricts the difference to arcs touching `s`:
`E △ E′ ⊆ ({s}×V) ∪ (V×{s})`. So for `u ≠ s` and `w ≠ s`, the arc `(u,w)` is in
`E` iff it is in `E′` — **every arc of `H` between two non-`s` vertices is common
to both graphs.** The union is inflated at exactly one place: `s` itself.

A shortest path from `s` never returns to `s`, so only its *first* step sees the
inflated degree; every later step is an ordinary `≤ K_out` step. Recounting:

```
n_d      ≤ 2·K_out · K_out^(d-1)  =  2·K_out^d
|P^(ℓ)|  ≤ min( 2·K_out^(ℓ-1), K_in^(ℓ-1) )
```

The backward path count never steps *out of* `s`, so it carries no factor of 2 —
hence when `K_in ≤ K_out`, `q_d` is unchanged and only `n_d` doubles. Verified:
`q_d` is bit-identical between the two formulas iff `K_in ≤ K_out`.

The factor of 2 is **tight, not slack**: `s` may have `K_out` out-arcs in `g` and
a disjoint set of `K_out` out-arcs in `g′`. The only way to remove it is to
weaken the neighbouring relation — e.g. let only `s`'s *features* change while
holding its edges fixed — which would give up exactly the protection the paper
is claiming.

## Where the 2 can be paid

These are not competing assumptions. They are the same guarantee at two
operating points:

| | cap enforced | formula fed | ε (facebook r=1) | mean root neighbourhood (PPI) |
|---|---|---|---|---|
| pay in ε | 5 | `2·K_out^d` | 7.19 → **14.81** | 4.435 (unchanged) |
| pay in utility | 2 | `K_out^d` | 7.19 (unchanged) | 4.435 → **1.497** |

Full cost of paying in ε: **1.4–2.1× at r=1, ~2.8× at r=2.** Smaller `p2` is
penalized *less* (PPI ε=1 at `p2=0.1` goes to 1.37×, at `p2=1.0` to 1.95×),
because `q_d` damping partly absorbs the larger shell.

## The cleanest fix for the paper

Restate the assumption on the individual graphs and carry the 2 into Eq (8):

> **Assumption 5.2′.** Each of `g`, `g′` has maximum in-degree `K_in` and
> maximum out-degree `K_out`.
>
> **Eq (8′).** `n_0 = 1`, `n_d = 2·K_out^d` for `1 ≤ d ≤ r`.

Same arithmetic as paying in ε, but the assumption becomes one an implementer
can actually verify — you can check `deg(g) ≤ K` on the graph in front of you;
you can never check a property of a pair. In the code this is a **one-line
change** to `shell_sizes` (`accounting.py:96`), returning `2 * base**d`. Nothing
else in the accountant moves.

## Separate, and worth not conflating

`cap_degrees` is randomized, so `cap(g)` and `cap(g′)` can differ at vertices
*other than* `s` — which threatens Definition 5.1's neighbouring relation
itself, not just the degree bound. The docstring at `sparse_expand.py:191-195`
flags this as the known Daigavane et al. caveat. Both readings of where capping
sits still land on the factor of 2, so it does not change the analysis above —
but it is a second question, not this one.

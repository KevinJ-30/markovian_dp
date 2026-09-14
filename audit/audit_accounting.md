# Audit: privacy accounting and epsilon computation

Repo: `/Users/kevinjacob/markovian_dp copy`, branch `sparse_expand_clean`
Scope: accounting / epsilon only. Read-only; nothing was edited.
Environment note: `dp_accounting` is **not** importable from the default `python`
(`/Users/kevinjacob/anaconda3/bin/python`). It lives in
`/Users/kevinjacob/anaconda3/envs/PytorchEnv` (`dp_accounting` 
at `.../PytorchEnv/lib/python3.11/site-packages/dp_accounting`). All numeric checks below
were run with that interpreter.

---

# PART A — What the code does (factual)

## A1. The dominating-pair construction

Everything lives in `/Users/kevinjacob/markovian_dp copy/src/sparse/accounting.py`.

### A1.1 Path-retention bounds `q_d` (`accounting.py:50-64`)

```
K   = min(K_in, K_out)                                   # accounting.py:111, :134
q_0 = 1
q_d = 1 - prod_{l=d..r} (1 - p2^l)^(K^(l-1))   for d>=1  # accounting.py:61-63
```
Computed in log space (`math.log1p(-(p2**l))`, `accounting.py:61`) with an explicit
`p2 >= 1.0 -> q_d = 1.0` short circuit (`accounting.py:58-60`) that avoids
`log1p(-1) -> ValueError`. `p2 = 0` gives `q_d = 0` for all `d>=1`.

Interpretation (my reconstruction, consistent with the docstring): `q_d` upper-bounds the
probability that at least one surviving path of length `l ∈ [d, r]` connects the substituted
vertex `s` to a shell-`d` vertex `v`. The exponent `K^(l-1)` bounds the number of length-`l`
paths; `min(K_in, K_out)` is a legitimate bound because counting forward gives `K_out^(l-1)`
and counting backward gives `K_in^(l-1)`, and the min of two valid bounds is valid. The
product form (rather than a union bound) is a valid **upper** bound on `q_d` by
Harris/FKG: path-survival events are increasing in the edge-retention variables, so they
are positively correlated and `P(no path survives) >= prod_i P(path_i broken)`.

### A1.2 Shell sizes `n_d` (`accounting.py:83-97`)

```
n_0 = 1                                     # s itself as a root
n_d = K_out^d   for direction='in'          # accounting.py:96 (base = K_out)
n_d = K_in^d    for direction='out'         # accounting.py:96 (base = K_in)
```
`r+1` shells, `d ∈ [0, r]`; **exponent is `d`, top exponent is `r`**. No off-by-one, and
`L`/`num_layers` never appears. Test anchor: `tests/test_accounting.py:126-130`.

### A1.3 Substitution mixture (`accounting.py:100-120`)

`pi = law of J = sum_{d=0..r} Binomial(n_d, p1 * q_d)`, built by successive
`np.convolve` of exact `scipy.stats.binom` pmfs (`accounting.py:115-117`), then
clipped and renormalized (`accounting.py:118-119`). Length is `1 + sum_d n_d`.

Pair: `P = sum_k pi_k N(-2k, sigma^2)`, `Q = sum_k pi_k N(+2k, sigma^2)`
(`accounting.py:246-250`). Units are multiples of `C`: a node substitution moves one
affected root's clipped gradient by `<= 2C`, and the injected noise is
`sigma*C` (`src/sparse/base_mechanism.py:174`), so `C` cancels.

### A1.4 Thm-4.5 insertion/removal mixture (`accounting.py:123-146`, `342-357`)

`pi = law of sum_{d=1..r} Binomial(K_in^d, p1*q_d)` — note `d` starts at **1** here
(`accounting.py:136`), no `d=0` shell, and the shell base is always `K_in`
(`accounting.py:143`) regardless of `direction`. The `q_d` recurrence is **re-implemented
inline** at `accounting.py:136-142` rather than calling `_q_products`; the two copies are
currently identical.

Marked pair, per fiber `j` (`accounting.py:348-353`):
```
P_j = N(-j, sigma^2)
Q_j = (1-p1) N(+j, sigma^2) + p1 N(+j+1, sigma^2)
```
Both orientations (`swap=False/True`, `accounting.py:347`) are built and the **max** is
taken after composition (`accounting.py:386`, `:410-411`, `:559-560`).

### A1.5 Which theorem each branch claims

`accounting.py:414-438` (`resolve_sparsegnn_theorem`, `sparsegnn_theorem_label`):

| direction | theorem='auto' resolves to | reported label |
|---|---|---|
| `in`  | `substitution` | `thm6.4-substitution` |
| `out` | `thm45`        | `thm4.5-insertion-removal` |
| `out` + `theorem='substitution'` | `substitution` | `thm1.2-substitution` |
| `in` + `theorem='thm45'` | **raises ValueError** (`accounting.py:425-428`) |

The module docstring (`accounting.py:1-36`) explicitly says the numbering is against
manuscript v36 and that the current theory doc renumbered in-expansion substitution to
Theorem 5.4 and has **not restated any out-expansion theorem at all**, so
`--direction out` accounting is self-declared "unconfirmed". The code labels still say
"thm6.4"/"thm4.5"/"thm1.2" and those strings are written into result CSVs
(`src/sparse/compute_epsilon.py:124`, `:127-128`).

I verified the formulas match the docstring line by line. I did **not** verify the
theorem statements against the manuscript — `paper/` is flagged stale in project memory,
and I did not read it. **Unverified: whether Theorem 5.4/4.5 as currently written say what
this code implements.**

### A1.6 From analytic pair to `dp_accounting` PLD (`accounting.py:172-238`)

Per fiber:
1. Grid `[lo, hi] = [min(all means) - 10*sigma, max(all means) + 10*sigma]`
   (`accounting.py:198-199`), `n_atoms = clamp((hi-lo)*400/sigma, 2000, 2_000_000)`
   (`accounting.py:200-201`).
2. `num_mass[i] = w_f * (CDF(edge_{i+1}) - CDF(edge_i))` — exact Gaussian-mixture CDF
   differences (`accounting.py:204`).
3. `edge_loss = logpdf_num - logpdf_den` at the edges; a **global monotonicity assert**
   (`accounting.py:208-211`) then `cell_loss[i] = max(edge_loss[i], edge_loss[i+1])`
   (`accounting.py:212`).
4. Numerator mass outside `[lo, hi]` and cells below `1e-18` go to a `+inf`-loss outcome
   (`accounting.py:215-218`, `:226-227`).
5. Denominator mass per cell is set to `num_mass * exp(-cell_loss)` (`accounting.py:223-224`),
   which is `<=` the true denominator mass; the shortfall becomes a `'rest'` outcome with
   loss `-inf` (`accounting.py:228-230`), which contributes 0 to the hockey stick.
6. Handed to `PLD.from_two_probability_mass_functions(..., pessimistic_estimate=True,
   value_discretization_interval=grid, symmetric=True)` (`accounting.py:236-238`).

**This is a sound upper bound.** The math checks out: on each cell,
`q(x) = p(x) e^{-loss(x)} >= p(x) e^{-cell_loss}`, so
`∫(p - αq)_+ <= m_i (1 - α e^{-cell_loss})_+`; grid tails and dropped components are
routed to `+inf` loss, which is the maximal contribution. The within-cell monotonicity the
per-cell max needs is not just asserted — it is provable for both families: for a Gaussian
location mixture `(log P)'(x) = (E[mu|x] - x)/sigma^2` with `E[mu|x]` nondecreasing, and for
the substitution pair `loss'(x) = (E[mu|x] + E[mu|-x])/sigma^2 <= 0` since all `P` means are
`<= 0`; for the thm-4.5 fibers `loss'(x) = (-j - E_den[mu|x])/sigma^2 <= 0`.

**Empirically confirmed**: single-step `pld.get_delta_for_epsilon(eps)` vs a fine-quadrature
hockey stick on the analytic pair, at `grid=1e-5`:

| config | eps | PLD delta | analytic delta | ratio |
|---|---|---|---|---|
| p1=.05 p2=.5 r=1 K=(5,5) s=2 | 1.0 | 4.9042e-3 | 4.8906e-3 | 1.0028 |
| p1=.2  p2=1  r=1 K=(4,4) s=1 | 1.0 | 5.1984e-1 | 5.1961e-1 | 1.0004 |
| p1=.01 p2=1  r=2 K=(20,3) s=5| 2.0 | 1.8310e-14| 1.7690e-14| 1.0351 |

Upper bound in every case, and tight (0.04 % – 3.5 %).

### A1.7 Composition

- **Accountant: PLD** (`dp_accounting.pld.privacy_loss_distribution`), not RDP.
  `naive_opacus_epsilon` (`accounting.py:605-622`) is the only RDP/PRV path and is
  explicitly labelled not a valid node-level guarantee.
- Final-iterate: `pld.self_compose(steps).get_epsilon_for_delta(delta)`
  (`accounting.py:339`, `:410-411`, `:550`, `:559-560`).
- Checkpoint schedule: `_compose_schedule` (`accounting.py:270-289`) advances in lockstep,
  one `self_compose(gap)` + one `compose` per checkpoint.
- `value_discretization_interval = grid`, default `1e-4` everywhere
  (`accounting.py:302`, `:328`, `:377`, `:399`, `:453`, `:479`, `:522`;
  `compute_epsilon.py:53`; `run.py:252`; `calibrate_grid.py:50`;
  `src/experiments/privacy.py:47`, `:74`).
- Rounding is `math.ceil` under `pessimistic_estimate=True`
  (`.../dp_accounting/pld/privacy_loss_distribution.py:426`), so composition
  **over**-reports by up to `T*grid`.
- `self_compose`/`compose` default `tail_mass_truncation=1e-15`; the truncated right
  (high-loss) tail is folded into `infinity_mass` (`.../pld/pld_pmf.py:413-414`), i.e.
  pessimistic; the left tail sits below any eps of interest and contributes 0.

Measured discretization cost at `p1=.0114 p2=1 r=1 K=5 sigma=10 T=2000 delta=1e-6`:

| knob | eps |
|---|---|
| grid=1e-3 | 7.2177 |
| grid=1e-4 (**default**) | 6.3180 |
| grid=1e-5 | 6.2282 |
| atoms_per_sigma=100 | 6.5225 |
| atoms_per_sigma=400 (**default**) | 6.3180 |
| atoms_per_sigma=1600 | 6.2670 |
| n_sigma=6 / 10 / 14 | 6.3184 / 6.3180 / 6.3180 |

Every knob moves eps in the **conservative** direction. The default `grid=1e-4` inflates
eps by ~1.4 % at T=2000; `grid=1e-3` inflates by ~16 %.

### A1.8 Calibration (`accounting.py:517-602`)

Doubling bracket from `sigma=1` up to `max_sigma` then bisection to
`max(sigma_atol, sigma_rtol*sigma)` (`accounting.py:573-590`); sigma-independent mixture
weights computed once (`accounting.py:542`, `:552`). `clip` is used **only** to report
`noise_std = sigma*clip` (`accounting.py:592`) — it never touches epsilon. Input validation
is thorough (`accounting.py:526-538`).

## A2. Direction handling

| | `--direction in` | `--direction out` |
|---|---|---|
| expansion | incoming arcs (`sparse_expand.py:137`, `:163-164`) | outgoing arcs |
| default theorem | substitution | thm4.5 insertion/removal |
| adjacency relation | **node substitution** | **node add/remove** |
| shell base | `K_out` | `K_in` |
| `K` in `q_d` | `min(K_in,K_out)` | `min(K_in,K_out)` |
| pair symmetric? | yes | no |
| max over orientations? | not needed | yes (`accounting.py:386`, `:410`) |

The substitution pair is genuinely self-reverse: `Q(x) = P(-x)`, so `L_{P,Q}(x) = -L_{P,Q}(-x)`
and `L_{Q,P}` under `Q` has the same law as `L_{P,Q}` under `P`. `symmetric=True` is therefore
correct there. For thm-4.5, `symmetric=True` is applied to each single-orientation PLD and the
two are max'd after composition — also correct.

`sparsegnn_substitution_epsilon` is always computed for **every** row regardless of
direction and written to `epsilon_substitution` (`compute_epsilon.py:107-110`, `:152`) so the
two orientations can be compared under one adjacency notion.

Numerically confirmed (`p1=.005 p2=.5 r=1 K=5 sigma=5 T=500 delta=1e-6`):
`sub_in == sub_out == 1.3891` exactly when `K_in == K_out`; `thm4.5 = 6.6227`.
At `p1=0.3`, `thm4.5 = 114.98 < sub = 241.50` — the two adjacency notions cross over.

## A3. Every hardcoded / defaulted delta

| value | file:line | what it is |
|---|---|---|
| `1e-5` | `src/sparse/compute_epsilon.py:47` | **CLI default** for post-hoc epsilon |
| `1e-5` | `src/experiments/baselines.py:29` | `BaselineConfig.delta` (DP-MLP / GraphSAGE baselines) |
| `1e-4` | `src/experiments/dpar.py:34` | `DPARConfig.ppr_delta` |
| `1e-3` | `src/experiments/dpar.py:44` | `DPARConfig.sgd_delta` |
| `1/(10n)` | `src/experiments/dpgnn.py:231` | DP-GNN (Daigavane) delta, computed from `train.num_nodes` |
| `n^-1.01` | `scripts/calibrate_grid.py:64` | `--delta_from_n`, the "agreed convention" per its docstring (`calibrate_grid.py:19-21`) |
| `n^-1.01` | `scripts/relbench_scope.py:85`, `:89` | same convention |
| `1e-6` | `scripts/_dataset_settings.sh:119` | `DELTA=${DELTA:-1e-6}` (main sweep driver) |
| `1e-6` | `scripts/_coverage_sweep.sh:54`, `scripts/orientation_ablation.sh:44` | |
| `1e-6` | `sbatch/sparse_inductive.sbatch:30`, `sbatch/reddit_settings.sbatch:56`, `sbatch/sparse_relbench.sbatch:37` | |
| `1e-6` | `scripts/_ppi_pareto_grid.sh:23` (inline `--delta 1e-6`) | |
| `1e-5` | `sbatch/facebook_settings.sbatch:39`, `sbatch/facebook_settings_ice.sbatch:48`, `sbatch/facebook_tune_ice.sbatch:34` | |
| `1e-5` | `scripts/_reltrial_ladder.sh:16` | |
| `1e-6` | `scripts/plot_relbench_report.py:151` (+ axis labels at `:84`, `:187`) | figure recomputes epsilon at 1e-6 |
| `1e-5` / `1e-6` / `5e-4` | tests, throughout `tests/test_accounting.py`, `tests/test_sparse_privacy_calibration.py` | |

`--delta`/`--delta_from_n` in `calibrate_grid.py` is a `required=True` mutually exclusive
group (`calibrate_grid.py:55-58`) — no silent default there. `run.py --target_delta` is
required alongside `--target_epsilon` (`run.py:309-314`).

Node counts and what the deltas actually are:

| dataset | n | 1/n | n^-1.01 | n^-1.1 | delta used |
|---|---|---|---|---|---|
| citeseer | 3,327 | 3.01e-4 | 2.77e-4 | 1.34e-4 | 1e-5 / 1e-6 |
| facebook | 26,406 | 3.79e-5 | 3.42e-5 | 1.37e-5 | **1e-5** |
| ppi | 56,944 | 1.76e-5 | **1.57e-5** | 5.88e-6 | **1e-6** (frontier) and **1.57e-5** (matched-eps) |
| flickr | 89,250 | 1.12e-5 | 1.00e-5 | 3.58e-6 | 1e-6 |
| ogbn-arxiv | 169,343 | 5.90e-6 | 5.24e-6 | 1.77e-6 | 1e-6 |
| reddit | 232,965 | 4.29e-6 | 3.79e-6 | 1.25e-6 | 1e-6 |

Recorded deltas in shipped `*_with_eps.csv`: facebook = `1e-05` (all 12 dirs I sampled),
reddit/arxiv/ppi/relbench/coverage = `1e-06`, archived citeseer = `1e-05`.

## A4. Which parameters enter epsilon (as implemented)

Verified by signature + call site, not by docs.

| parameter | enters? | where |
|---|---|---|
| `p1` | **yes** | `accounting.py:117` (`p1*q_d`), `:352` |
| `p2` | **yes** | `accounting.py:61` |
| `r` | **yes** | shells + `q_d` product range |
| `sigma` | **yes** | `accounting.py:204-206` |
| `T`/`steps` | **yes** | `self_compose(steps)` |
| `delta` | **yes** | `get_epsilon_for_delta` |
| `K_out` | **yes** | shell base for `'in'` (`accounting.py:96`), and `min()` at `:111` |
| `K_in` | **yes, but only via `K = min(K_in,K_out)` under `'in'`** — and **only when `r >= 2`** | `accounting.py:111`, `:134` |
| `C` / `clip` | **no** | only `accounting.py:592` (reporting) |
| `L` / `num_layers` | **no** | never referenced in `accounting.py` or `compute_epsilon.py` |
| batch size / #roots | **no** | `expected_batch` is used only as a post-processing divisor (`sparse_gnn.py:70`, `:102`) |
| `cap_mode`, `cap_seed`, lr, momentum, optimizer, dropout, hidden | **no** | |

Measured `K_in` dependence (`direction=in`, `K_out=5`, `p1=.01 p2=.5 sigma=5 T=500 delta=1e-6`):

- `r=2`: K_in = 1/2/3/5/8/20 → eps = 10.07 / 16.42 / 21.70 / 29.30 / 29.30 / 29.30.
  So **raising K_in raises epsilon** until it saturates at `K_in = K_out`.
- `r=1`: K_in = 1/2/5/20 → eps = 2.933258 identically (because `q_1 = 1-(1-p2)^{K^0} = p2`,
  free of `K`).

Measured `K_out` dependence at `K_in=5`, `r=2`: 1.49 / 3.57 / 8.05 / 29.30 / 94.31 for
`K_out = 1/2/3/5/8` — the `K_out^r` blow-up.

## A5. Degree cap vs the accounting

- `run.py:434-465` deduplicates arcs then caps. `cap_mode='auto'` picks
  `cap_degrees_undirected(..., K_in)` iff `K_in == K_out` **and** the arc set is symmetric,
  else `cap_degrees(..., K_in, K_out)` (both directions capped independently,
  `sparse_expand.py:219-222`).
- The cap is applied **only to `g`**, the single training graph. The capped graph is
  re-drawn per seed unless `--cap_seed` pins it (`run.py:500-505`).
- `K_in`/`K_out` written to the CSV are the **requested** caps (`run.py:646-647` reads
  `gph['K_in']`, set from `K_in_req` at `run.py:483`); the **achieved** max degrees go to
  separate `K_in_achieved`/`K_out_achieved` columns (`run.py:654-655`) that `compute_epsilon`
  never reads. Since capping only removes arcs, achieved <= requested, so using requested is
  conservative.
- If `--dp` is passed **without** `--K_in`, `run.py:484-489` prints a warning and writes the
  graph's **raw observed max degrees** into the `K_in`/`K_out` columns; `compute_epsilon`
  then accounts with those.
- `K_out^d` with exponent `d ∈ [1, r]` — README claim confirmed (`accounting.py:96-97`,
  `README.md:127`). The expansion loop is `for _ell in range(r)` (`sparse_expand.py:144`),
  so nodes are at distance `<= r`; no off-by-one, `L` is irrelevant.

## A6. Post-hoc workflow (`src/sparse/compute_epsilon.py`)

Reads the CSV with `csv.DictReader` (`:65-66`). Config key is
`(direction, p1, p2, r, sigma, T, K_in, K_out)` (`:79-82`); step is `step` or, if absent/blank,
`T` (`:84-87`). Required columns and failure modes, as measured:

| column | missing | blank |
|---|---|---|
| `p1,p2,r,sigma,T` | `KeyError` (crash) | `ValueError` (crash) |
| `K_in` | `SystemExit` with a helpful message (`:90-94`) | same `SystemExit` |
| `K_out` | `KeyError` | **`ValueError` crash** (confirmed by running it) |
| `direction` | **silently defaults to `'out'`** (`:79`) | **silently defaults to `'out'`** |
| `step` | falls back to `T` (`:86-87`) | falls back to `T` |
| `dp` | only `rows[0]` is checked (`:69-71`), and only warns | |
| `test_acc` | `KeyError` in the summary table (`:168`) | `ValueError` |

Columns written back: `step`, `epsilon`, `epsilon_theorem`, `epsilon_substitution`,
`epsilon_naive_opacus`, `delta` (`:149-154`). `delta` **overwrites** any existing value.
The grid actually used is **not** recorded.

---

# PART B — Open questions, discrepancies, risks

Severity is about the risk to a published epsilon number, not code aesthetics.

---

### B1. `direction` silently defaults to `'out'`, which switches the *adjacency relation* — HIGH
`src/sparse/compute_epsilon.py:79` — `direction = row.get('direction') or 'out'`

A missing **or blank** `direction` cell silently routes the row to Theorem 4.5
(insertion/removal). Measured on a synthetic CSV (`p1=.05 p2=.5 r=1 K=5 sigma=5 T=50`):
with `direction=in` → `eps=4.8491 (thm6.4-substitution)`; with the column deleted or left
blank → `eps=7.7864 (thm4.5-insertion-removal)`. No warning is printed either way. The
archived CSVs in `results/archive/` genuinely lack the column, so this path is live.

Worse than the number changing: the two branches report epsilon under **different
neighbour relations** (substitution vs add/remove). They are not comparable, and at
`p1=0.3` the ordering even flips (`thm4.5 = 114.98 < sub = 241.50`). A blank cell therefore
silently converts a substitution-DP claim into a weaker add/remove-DP claim.
Suggested fix direction: make a missing/blank `direction` a hard error, or at minimum
print a loud warning.

---

### B2. The orientation ablation compares epsilons under two different adjacency notions — HIGH
`scripts/orientation_ablation.sh:72-76` runs `compute_epsilon` with default
`--theorem auto` on both `dp_in/` and `dp_out/`, then says
"compare $OUT/dp_in vs $OUT/dp_out" (`:79`). Every plotting script reads the
`epsilon` column (`scripts/plot_frontier.py:53`, `plot_ppi_frontier.py:52`,
`plot_ppi_pareto.py:54`, `plot_relbench_report.py:57`, `summarize_sweep.py:42`), i.e. the
`auto` column. So an in-vs-out frontier plot puts substitution-eps and add/remove-eps on
one axis.

The `epsilon_substitution` column exists precisely to make this comparison honest
(`compute_epsilon.py:107-110`, `:152`) — but no plotting script uses it. Only
`plot_sparse_frontier.py:38` even knows about an alternative column, and that is the
*legacy* `epsilon_thm4`, not `epsilon_substitution`.

---

### B3. Union-graph gap: degrees are capped on `g` only; nothing compensates — HIGH
`src/sparse/run.py:434-465` caps the single training graph. `src/sparse/accounting.py`
contains **no** factor of 2 anywhere (`grep -rn "2K\|2 \* K\|union"` over `src/` returns
nothing relevant). If Assumption 5.2 bounds degrees on `g ∪ g'`, then the shells should be
built from `2K_out`, not `K_out`, and since epsilon scales roughly as `K_out^r`, correct
accounting would be dramatically larger (measured: `K_out` 3→5 at `r=2` moves eps
8.05 → 29.30; 5→8 moves 29.30 → 94.31; i.e. doubling `K_out` at `r=2` is a ~10x eps hit).
This matches the OPEN item already in project memory; **nothing in the code has changed
to address it**. This is the single largest open correctness question in the accounting path.

Related and separate: `cap_degrees`'s own docstring (`sparse_expand.py:190-195`) admits
the capping randomness at a surviving node can depend on the inserted node's arcs, so the
capped graphs of `g` and `g'` may differ at nodes **other than** `s`. The code follows
Daigavane et al. and caps once, accounting on the result. That is a documented assumption,
not a proof.

---

### B4. Checkpoint selection on validation is not accounted — HIGH
`scripts/summarize_sweep.py:103-109` picks `argmax_t val(t)` and then reports
`curve[best]` with `eps[best]` (`:119-121`). The docstring (`:12-16`) correctly fixed
*test*-selection leakage but the same argument applies to privacy: to choose `t*` you must
have run to `T` and evaluated val at every checkpoint. Under node-DP the val nodes are part
of the private dataset, so (a) the trajectory that was actually produced costs `eps(T)`,
not `eps(t*)`, and (b) the selection itself is a data-dependent release that no accountant
covers. Reporting `eps(t*)` understates on both counts.

The per-checkpoint schedule itself is *correct* (`accounting.py:270-289`) — `eps(t)` is a
valid guarantee for releasing `θ_1..θ_t`. The mismatch is in how it is consumed.
Minimum honest fix: report `eps(T)` for a best-on-val checkpoint, or fix `t*` up front.

---

### B5. Four incompatible delta conventions coexist across the comparison suite — HIGH
Baselines and the proposed method are accounted at different deltas, so "matched epsilon"
tables comparing them are not matched:

- SparseGNN: `1e-6` (most drivers) or `n^-1.01` (matched-eps drivers) or `1e-5`
  (all facebook drivers, and the `compute_epsilon` default).
- DP-MLP / GraphSAGE baselines: `1e-5` (`src/experiments/baselines.py:29`).
- DPAR: `ppr_delta=1e-4` + `sgd_delta=1e-3` (`src/experiments/dpar.py:34,44`) — a total
  delta of order `1e-3`, which for citeseer (n=3327) is **3.3/n**, i.e. weaker than
  "release a random node in the clear".
- DP-GNN: `1/(10n)` (`src/experiments/dpgnn.py:231`).

And within SparseGNN itself: PPI is reported at `1e-6` in `results/ppi/frontier/` but at
`n^-1.01 = 1.57e-5` in `sbatch/ppi_matched_eps.sbatch:94` — a **15x** difference in delta
on the same dataset, which changes epsilon materially. `results/facebook_*` at `1e-5`
against `results/reddit|arxiv|relbench` at `1e-6` is the same problem across datasets.

This is the unresolved "delta convention decision" from project memory; the code still has
both conventions wired in and in active use.

---

### B6. `compute_epsilon` defaults `--delta 1e-5` with no dataset awareness — MED
`src/sparse/compute_epsilon.py:47`. Running it without `--delta` on reddit (n=232,965)
produces epsilons at `delta = 1e-5 ≈ 2.3/n` — i.e. formally meaningless for node-level DP —
and writes `delta=1e-05` into the CSV with no warning. There is no check that
`delta << 1/n`, nor any access to `n` (the dataset column is present but never consulted).
`calibrate_grid.py` got this right by making delta mandatory; `compute_epsilon` did not.

---

### B7. `compute_epsilon` accepts and silently uses nonsensical delta — MED
No validation of `--delta` at all. Measured:
- `--delta 1.5` → prints and writes `eps=0.0000` for every row, plus
  `eps_naive=-7.4028` (a **negative** epsilon from Opacus), no error.
- `--delta 0` → `eps=inf` written to the CSV.

`calibrate_sparsegnn_noise` does validate (`accounting.py:528-530`); the post-hoc path does not.

---

### B8. Post-hoc epsilon can disagree with the calibration that produced the run — MED
`compute_epsilon` never reads the `target_delta`, `target_epsilon`, `calibrated_epsilon`,
`accounting_theorem` or `accounting_grid` columns that `run.py:539-541` writes. So an
`--target_epsilon 1.0 --target_delta 1e-5` run, post-processed with the default
`--delta 1e-5 --grid 1e-4`, ends up with an `epsilon` column computed at a different delta
and a different grid from `calibrated_epsilon`, sitting side by side in the same row with
no cross-check. Concretely, the matched-eps drivers calibrate at `grid=1e-5`
(`sbatch/ppi_matched_eps.sbatch:62`, `arxiv_inductive_matched_eps.sbatch:52`,
`relbench_regression_meps.sbatch:74`, `scripts/_facebook_width.sh:48`), while **every**
`compute_epsilon` invocation in `scripts/` and `sbatch/` uses the default `grid=1e-4`
(none of the 16 call sites passes `--grid`). At T=2000 that is a 1.4 % difference; at
T=2000 with `grid=1e-3` it would be 16 %.

Also: with `--track_every`, every checkpoint row carries the *final-step*
`target_epsilon`/`calibrated_epsilon` (they are per-cell constants,
`run.py:593-598`), so a step-500-of-2000 row claims `calibrated_epsilon=1.0`.

---

### B9. `--dp` without `--K_in` computes epsilon from the graph's observed max degree — MED
`src/sparse/run.py:484-489`. The warning is printed once, but the raw max degrees are then
written into the `K_in`/`K_out` columns and `compute_epsilon` consumes them with no way to
tell they were observed rather than enforced. Two problems: (a) the observed max degree is a
**data-dependent** quantity that is itself released via the CSV, and (b) under node
substitution the neighbouring graph may have a larger max degree, so the bound does not
hold uniformly over the adjacency class. On arxiv the uncapped max degrees are 3015/221
(per `scripts/orientation_ablation.sh:36`), so this is not hypothetical.
`compute_epsilon` could detect it (`cap_mode` is blank in exactly that case,
`run.py:441-442`) but does not look.

---

### B10. Theorem selection logic is duplicated and can drift — MED
`src/sparse/accounting.py:414-429` (`resolve_sparsegnn_theorem`) and
`accounting.py:432-438` (`sparsegnn_theorem_label`) are the canonical resolvers, but
`src/sparse/compute_epsilon.py:112-128` **re-implements both by hand** — the `auto` mapping
*and* the label strings `'thm4.5-insertion-removal'` / `'thm6.4-substitution'` /
`'thm1.2-substitution'`. They agree today. `sparsegnn_epsilon_schedule`
(`accounting.py:441-464`), the function that exists to do exactly this dispatch, is
exported in `src/sparse/__init__.py:30` and **called nowhere in the repo**.

---

### B11. `K_in` is not free in epsilon at `r >= 2`, contrary to the standing note — MED
`accounting.py:111` (`K = min(K_in, K_out)`) feeds the `K^(l-1)` path count at
`accounting.py:61`. Measured above (A4): at `r=2, K_out=5`, raising `K_in` from 1 to 5
raises eps from 10.07 to 29.30 — a ~3x effect — after which it saturates. At `r=1` it is
exactly free. The project-memory note "K_in free in ε" and
`scripts/plot_relbench_report.py:182` ("ε barely moves with K_in") are true only in the
saturated regime `K_in >= K_out`, which is where the relbench figure happens to sit
(`K_out=3`, `K_in` swept 2..20). The figure's caption generalizes beyond what it measures.

---

### B12. Dead duplicate function in the accounting path — MED (drift risk, not a bug today)
`src/sparse/accounting.py:254-259` defines `_substitution_pld`, and
`accounting.py:262-267` immediately **redefines the same name**. The first is dead. The two
bodies are semantically identical today (one inlines the weight computation, one binds it
to a local first), so nothing is wrong now — but a future edit to the first, shadowed copy
would be silently ignored. Introduced by `f2a8a88`/`33cf1ec`/`ff98371`.

Similarly, `thm4_fiber_weights` (`accounting.py:136-142`) re-implements the `q_d` recurrence
inline instead of calling `_q_products` (`accounting.py:50-64`) — two copies of the core
privacy formula.

---

### B13. Thm-4.5 drops low-weight fibers without routing their mass to infinity — LOW
`accounting.py:344-345` filters `pi_j >= 1e-14` and passes only the survivors as fibers.
`_pld_from_fibers` only accounts each *surviving* fiber's own grid tail
(`accounting.py:215`), so the dropped fiber mass simply vanishes from the numerator instead
of becoming an infinite-loss outcome. Since `H_α(P||Q) = Σ_j π_j H_α(P_j||Q_j)`, the reported
delta is under-stated by up to the dropped mass. This is a genuine (if tiny)
**under-estimate** path.

The substitution path does **not** have this bug: `accounting.py:250` passes `w_f = 1.0`, so
`inf_mass += max(0, 1.0 - num_mass.sum())` at `accounting.py:215` catches the dropped
components correctly.

Measured dropped mass: 0 at `(p1=.05,p2=.5,r=1,K=5)`; 2.6e-15 at `r=2,K=5`; 1.9e-14 at
`(p1=.3,p2=1,r=2,K=20)`; 5.3e-14 at `(p1=.05,p2=1,r=2,K=100)`. Times T=2000 that is at worst
~1e-10 against deltas of 1e-6 — **numerically irrelevant today**, but it is the only
under-estimate direction I found in the whole construction, and it scales with `K^r`.

---

### B14. `np.convolve` makes the substitution weights `O((K^r)^2)` — LOW
`accounting.py:117` convolves length-`n_d+1` pmfs with `np.convolve` (direct, not FFT).
At `K=100, r=2` the final length is ~10,101 (fine, measured). At `K=100, r=3` it would be
~10^6 and the convolution becomes ~10^12 operations, i.e. a hang, not an error. Similarly
`_binom_pmf` (`accounting.py:67-80`) materializes a length-`K^r+1` array. Nothing in the
repo currently runs `r=3` at large `K`, but there is no guard.

---

### B15. `epsilon_naive_opacus` is a non-guarantee sitting in a column named `epsilon_*` — LOW
`accounting.py:605-622` / `compute_epsilon.py:134-137`. Correctly documented as invalid at
`accounting.py:610-612` and `compute_epsilon.py:21-24`, printed in the frontier table as
`eps_naive` (`compute_epsilon.py:181`). The risk is purely that a downstream reader picks the
wrong column. Also `naive_opacus_epsilon` swallows *all* exceptions and silently falls back
from PRV to RDP (`accounting.py:619-622`), and `compute_epsilon.py:136-137` swallows again
into `NaN`.

---

### B16. The grid used to compute an epsilon is not recorded — LOW
`compute_epsilon.py:147-154` writes `delta` but not `grid`. A `*_with_eps.csv` cannot be
audited for whether its epsilon came from `grid=1e-4` or `grid=1e-3` (a 16 % difference at
T=2000). `run.py:539-541` does record `accounting_grid`, but only for `--target_epsilon` runs.

---

### B17. Theorem labels cite v36 numbering that the current draft has renumbered — LOW
`accounting.py:1-11` is explicit that the labels are stale and that `--direction out`
accounting has **no** counterpart in the current theory doc. Those stale strings
(`thm6.4-substitution`, `thm4.5-insertion-removal`, `thm1.2-substitution`) are written into
every results CSV (`compute_epsilon.py:124`, `:127-128`) and into figure axis labels
(`plot_relbench_report.py:84` says "Theorem 6.4"). Cosmetic until a reader tries to match a
CSV against the paper.

---

### B18. Things I checked that are FINE (recording so they are not re-audited)

- The PLD construction is a genuine, tight upper bound (A1.6): verified analytically and
  numerically against fine-quadrature hockey sticks at three configs / three alphas.
- Every numerical knob (`grid`, `n_sigma`, `atoms_per_sigma`, the `1e-14`/`1e-18` floors,
  `tail_mass_truncation`) moves epsilon **up**, except B13.
- `symmetric=True` is legitimate for the substitution pair (`Q(x) = P(-x)`); for thm-4.5
  both orientations are built and max'd.
- Composition is pessimistic PLD self-composition, `math.ceil` rounding.
- `C` correctly does not enter epsilon; noise is `sigma*C` (`base_mechanism.py:174`),
  clipping is per-subgraph (`base_mechanism.py:156-164`), and the `/expected_batch`
  normalization is applied to signal and noise alike (`sparse_gnn.py:102`) — post-processing.
- `L` does not enter, and cannot widen the radius (`sparse_expand.py:144`; test at
  `tests/test_dp_mechanics.py:325`).
- The DP step runs even on an empty root batch (`sparse_gnn.py:196-202`), which is what the
  Poisson accounting requires.
- The monotonicity assert at `accounting.py:208-211` did not fire across a 48-config
  edge-case sweep (p1 ∈ {0.01,1}, p2 ∈ {0,1}, r ∈ {0,1,2}, K ∈ {(5,5),(20,3)},
  sigma ∈ {0.5,50}, all three branches).
- The `_ice_env.sh:106-118` / `_local_env.sh:39-51` accountant regression gate reproduces:
  `EPS(p1=.013,p2=1,r=1,K=5,sigma=5,T=500,delta=1e-6,grid=1e-4) = 7.214349` vs expected
  `7.2143`.
- `calibrate_grid.py --K $SCOPE_K_OUT` in `relbench_regression_meps.sbatch:125,137` sets
  `K_in = K_out = K_out` while training caps at `K_in = SCOPE_K_IN`. This looks like a
  mismatch but is **safe**: under `direction='in'` the shell base is `K_out` either way, and
  `K = min(K_in,K_out)` can only increase when `K_in` is raised to `K_out`, so the
  calibration is at worst conservative. Obscure, but not a bug.
- `blank K_out` and `missing/blank K_in` both fail loudly rather than defaulting.

---

### B19. Unverified / out of scope

- Whether the current manuscript's Theorem 5.4 (and any out-expansion successor to 4.5)
  states what this code implements. I did not read `paper/` (flagged stale in memory).
- Whether the "independent paths" / `K^(l-1)` path-count bound is what the theorem's proof
  actually uses, or whether the theorem needs a different exponent.
- Whether Assumption 5.2 is stated over `g ∪ g'` (B3 is written conditionally on that
  being true, from the task brief and project memory, not from reading the paper).
- Whether the `epsilon`/`epsilon_substitution` columns in the shipped
  `results/**/*_with_eps.csv` were produced by the current `accounting.py` or an older one.
  The archived citeseer/arxiv CSVs use the legacy `epsilon_thm4` column name and lack
  `direction`, so at least those predate the current code.

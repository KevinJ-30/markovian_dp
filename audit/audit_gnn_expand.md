# Audit — graph machinery (layers, aggregation, SparseExpand, degree capping)

Repo: `/Users/kevinjacob/markovian_dp copy`, branch `sparse_expand_clean`, read-only.
Scope: message passing, aggregation, SparseExpand, degree capping. Not accounting, datasets, optimizer.
PyG 2.7.0 (`/Users/kevinjacob/anaconda3/lib/python3.10/site-packages/torch_geometric`).

All numeric claims below were measured by running the repo's own code (probe scripts in this
scratchpad: `probe.py`, `probe2.py`, `probe3.py`, `probe5.py`).

---

# (A) What the code does

## A.1 The aggregator — exact formulas

`src/sparse/layers.py:27-39` is the only place a conv is constructed. Every mechanism
(`gnn`, `multilabel_gnn`, `binary_gnn`, `regression_gnn`) calls it with
`dims = [in] + [hidden]*(L-1) + [out]` (e.g. `gnn_mechanism.py:32`).

### `--aggr mean` → `SAGEConv(d_in, d_out, aggr="mean")` (`layers.py:31-35`)

PyG defaults in force: `root_weight=True`, `bias=True`, `normalize=False`, `project=False`
(`sage_conv.py:66-75`).

```
                    1                                                 (l-1)
h_i^(l)  =  W_l^(l) ---  sum_{j in N_in(i)} h_j^(l-1)   +  b^(l)  +  W_r^(l) h_i
                  |N_in(i)|
```

* **Normalizer**: `|N_in(i)|` — the in-degree of the TARGET, counted on the *edge_index actually
  passed to `forward`* (the rooted subgraph during training, the full/capped graph at eval).
* **Self-loop**: none added. `SAGEConv` has no `add_self_loops` option.
* **Root feature**: a separate weight `W_r` (`lin_r`, `bias=False`), **summed**, not concatenated.
  Algebraically identical to the original GraphSAGE concat form `[W_l | W_r] · [agg ; h_i]`
  (`sage_conv.py:132-138`).
* **Empty in-neighbourhood** → aggregate is the **zero vector**, so
  `h_i = b^(l) + W_r h_i^(l-1)` (an MLP layer). Verified finite, `probe.py` step 4.

### `--aggr gcn` → `GCNConv(d_in, d_out, add_self_loops=True, normalize=True)` (`layers.py:36-39`)

PyG defaults in force: `improved=False`, `cached=False` (good — the subgraph changes every root),
`bias=True` (`gcn_conv.py:178-191`).

```
h_i^(l)  =  sum_{j in N_in(i) U {i}}  ------1------  Theta^(l) h_j^(l-1)   +  b^(l)
                                      sqrt(d^_j d^_i)

            d^_v  =  1 + |N_in(v)|            (self-loop included)
```

* Is it `D^-1/2 A D^-1/2`? **Yes in form, but `D` is the IN-degree matrix at BOTH ends.**
  `gcn_norm` computes `deg = scatter(w, col)` with `col = edge_index[1]` for
  `flow='source_to_target'`, then `w_ji = deg^-1/2[row] * deg^-1/2[col]`
  (`gcn_conv.py:203-211`). On a symmetric arc set in-deg = out-deg = deg, so it reduces to
  textbook GCN. On a **directed** graph (ogbn-arxiv, RelBench) the source term uses the source's
  *in*-degree, which is not the textbook symmetric normalization at all.
* **Which graph?** Whichever edge_index is handed to `forward`. Training: the rooted subgraph's
  local edge_index (`gnn_mechanism.py:77-78`), so all degrees are subgraph-local. Evaluation:
  full or capped graph (`gnn_mechanism.py:153` → `base_mechanism.py:69-90`).

### The README "EXACT" claim, verified

`README.md:155-160` and `layers.py:8-13` claim mean makes rooted-subgraph inference *exactly*
equal full-graph inference because its normalizer reads only the target's in-neighbourhood,
"which SparseExpand always materializes in full".

Measured `max |rooted(root) − full(root)|` over 60 roots, random directed graph, `dropout=0`:

| aggr | L | r=1 | r=2 | r=3 | r=4 |
|---|---|---|---|---|---|
| mean | 1 | **0** | **0** | **0** | – |
| mean | 2 | 1.2e-1 | **0** | **0** | – |
| mean | 3 | 1.4e-1 | 4.4e-2 | **0** | – |
| gcn | 1 | 2.6e+0 | **0** | **0** | **0** |
| gcn | 2 | 3.0e+0 | 6.0e-1 | **0** | **0** |
| gcn | 3 | 2.2e+0 | 6.1e-1 | 1.0e-1 | **0** |

The true rule is **`mean` is exact iff `p2 == 1` and `L <= r`; `gcn` is exact iff `p2 == 1` and
`L <= r - 1`.** (Induction: `h^(l)_u` on H is exact iff every in-arc of every node within
`l` hops of `u` is present; mean needs in-arcs of nodes at distance ≤ l−1, gcn additionally needs
the in-degree of their sources, i.e. one hop further.)

At `p2 = 0.5, L = r = 2`: mean error `max 5.1e-1 / mean 1.2e-1`. So under actual sparsified
training neither aggregator is exact — the rooted computation equals inference on the *sampled*
structure, not on the full graph. Degree capping does not break exactness per se (the capped graph
is simply "the graph"), but it does mean train and eval see different edge sets unless
`--eval_graph train` (see A.4). Deduplication is a no-op for exactness (parallel arcs are removed
before anything else). Isolated roots are handled consistently (zero aggregate / self-loop only)
in both the rooted and the full forward, so they do not break exactness.

**Verdict: the claim holds exactly only at `p2 = 1` with `L <= r`. Both preconditions are omitted
from the README, and the shipped experiments violate both** (see B1, B2, B3).

## A.2 Is this a standard GCN / GraphSAGE?

The convs themselves are stock PyG (no subclassing, no monkey-patching). The stack around them
(`gnn_mechanism.py:35-41`, identical in `multilabel_mechanism.py:36-41`,
`binary_mechanism.py:36-41`, `regression_mechanism.py`):

```python
for i, conv in enumerate(self.convs):
    x = conv(x, edge_index)
    if i < len(self.convs) - 1:
        x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
return F.log_softmax(x, dim=1)          # multilabel/binary return raw logits
```

Differences from reference implementations:

| item | this repo | reference |
|---|---|---|
| bias | present, one per layer (`lin_l` bias for SAGE; `bias` for GCN) | same |
| root weight | SAGE: yes, separate `lin_r` (sum). GCN: none, self-loop instead | same as PyG |
| output L2 normalization | **off** (`normalize=False`) | original GraphSAGE L2-normalizes each layer |
| neighbour sampling | Bernoulli(p2) per arc, once, in SparseExpand | GraphSAGE: fixed fan-out re-sampled per layer per epoch |
| GCN norm | `D_in^-1/2 (A+I) D_in^-1/2`, recomputed per subgraph, `cached=False` | Kipf: same on undirected; PyG's in-degree convention differs on directed graphs |
| activation | ReLU between layers only, none after last | same |
| dropout | between layers only — **not on the input features** | Kipf's GCN also drops the input layer |
| residuals / JK / norm layers | none anywhere (`grep BatchNorm/LayerNorm` → 0 hits in `src/`) | – |
| final layer | plain conv → `log_softmax` (single-label) or raw logits (multilabel/binary/regression) | same |
| depth | `L = num_layers`, decoupled from `r` | – |

So: stock PyG convs, a thin standard stack, no exotic tricks. The non-standard part is entirely in
*what graph* is fed to it.

## A.3 SparseExpand as implemented (`sparse_expand.py:106-174`)

1. **Roots**: `sample_roots` (`:321-344`) — independent Bernoulli(p1) over `candidate_nodes`
   (default = train nodes, `run.py:510-511`). Poisson subsampling, no fixed batch size.
   Driven once per step at `sparse_gnn.py:190`.
2. **Adjacency**: `build_adjacency(edge_index, n, direction)` (`:71-86`). `direction='in'` keys CSR
   by `edge_index[1]` (target) and stores `edge_index[0]` (source) — so `adj.neighbors(u)` returns
   the **sources of arcs into u**. `direction='out'` swaps the rows (`:77`).
3. **Expansion**: `for _ell in range(r)` (`:144`). For each `u` in the frontier, ALL of `u`'s
   neighbours are examined at once and each arc is kept independently with prob p2
   (`:147-148`, `_bernoulli_keep` `:94-103`). `if not keep.any(): continue` (`:149-150`) is a pure
   short-circuit. For each kept `w`: if new, assign a local index and push to `next_frontier`
   (`:155-158`); then record the arc **regardless of novelty** (`:154`, Alg-5 line 8 before line 9)
   — so E_v can contain arcs into already-discovered vertices, including back-arcs and self-loops.
4. **Orientation**: `edges_local.append([visited[w], u_local] if expand_in else [u_local, visited[w]])`
   (`:163-164`). Under `'in'` the traversed arc `(w,u)` keeps its original orientation, so messages
   flow *toward* the root. This is the only place the flag changes the recorded geometry; the other
   two are `build_adjacency` (`:77`) and the accounting shell choice
   (`accounting.py` via `direction=`). A mismatch between adjacency and expansion direction raises
   (`:133-136`), tested at `test_sparse_expand.py:142-146`.
5. **Termination**: `frontier = next_frontier; if not frontier: break` (`:165-167`). Each node is
   expanded at most once (only *new* nodes enter the frontier), so **each arc gets at most one
   Bernoulli draw** — matching the accounting's per-arc examination model.
6. **Local indexing**: `nodes[0] == root` always (`:139-140`), which is what
   `subgraph_loss` relies on (`gnn_mechanism.py:79-80`).

**Off-by-one check: none.** Measured on a 5-chain `4→3→2→1→0`, root 0, p2=1:

```
r=0: nodes=[0]            edges=[]
r=1: nodes=[0,1]          edges=[(1,0)]
r=2: nodes=[0,1,2]        edges=[(1,0),(2,1)]
r=3: nodes=[0,1,2,3]      edges=[(1,0),(2,1),(3,2)]
```

`r` levels materialize node set = ball of radius `r`, and arcs at depth 1..`r` (i.e. all in-arcs of
every node at distance ≤ r−1; distance-r nodes are never expanded). That is precisely the
computation tree an `r`-layer mean-GNN needs — so `r` is the *number of hops of edges*, not the
number of expanded shells minus one. Correct.

## A.4 Degree capping

**Where.** Graph-construction time only, inside `run.py:434-465` (`_simplify_and_cap`), called from
`_build_graphs` (`:467-495`). **SparseExpand itself never truncates** — no per-level fan-out cap
exists anywhere. If the graph is uncapped, expansion fan-out is unbounded.

**Order.** `dedup_arcs` first (`run.py:437`), cap second (`:452` or `:458`). So the cap is applied
**after** deduplication — correct, and the docstring explains why (`sparse_expand.py:227-233`,
"duplicates also get outsized survival odds under capping").

**Both directions?** Yes, when `K_in` is given:
```python
ei = cap_degrees(ei, n_nodes, K_in=K_in_req, K_out=K_out_req, generator=cap_gen)   # run.py:458
```
with `K_out_req = args.K_out if args.K_out is not None else args.K_in` (`run.py:431`).
`cap_degrees` caps in-degree first (`row=1`), then out-degree (`row=0`) (`sparse_expand.py:219-222`);
since both passes only remove arcs, the second cannot re-violate the first. `cap_mode=auto`
(`run.py:446-459`) picks `cap_degrees_undirected` when the arc set is symmetric and `K_in == K_out`,
which caps the *undirected* degree at K and re-emits both arcs, preserving symmetry
(`sparse_expand.py:258-311`).

**Random or first-K?** Random, both variants.
`cap_degrees._cap` shuffles then stable-sorts by key so ties are in uniformly random order, then
keeps rank < K (`sparse_expand.py:208-217`). Measured on a 10-arc star with `K_in=1` over 3000 seeds:
survivor histogram `[282,308,280,317,319,294,293,303,300,304]` — uniform, **no index bias**.
`cap_degrees_undirected` is greedy over a uniformly random edge permutation (`:291-304`), so it is
random but *not* a uniform K-subset (some nodes finish below K; that still satisfies the bound).

**Re-sampled per step?** No. One capped graph per *seed*, built once before training
(`run.py:500-505`); `--cap_seed` pins it so every seed shares one graph. The known-caveat comment is
at `sparse_expand.py:186-195`.

**Training vs evaluation graph** (`run.py:322-328`, `613-617`):

| run | `eval_graph` (auto) | model evaluated on | recorded as `*_alt` |
|---|---|---|---|
| transductive | `train` | **capped + deduped training graph** | uncapped `data.edge_index` |
| inductive | `full` | **uncapped, unfiltered `data.edge_index`** | capped full graph (`eval_capped`) |

So transductive runs evaluate on capped degrees (matching training); **inductive runs evaluate on
uncapped degrees** while having trained on ≤ K. Both numbers are always recorded, so the gap is
measurable, but the headline column for inductive runs is the off-distribution one.
`README.md:161-163` ("evaluation defaults to the full one") is stale for the transductive case.

## A.5 L vs r

No guard, no assert, no warning anywhere (`grep` over `src/`, `scripts/`: the only mention is the
docstring `gnn_mechanism.py:50-52` "== max SparseExpand distance r ... though not enforced").
`--num_layers` and `--r` are fully independent CLI knobs (`run.py:215-217, 222`).

* **L > r**: the model still only reads nodes inside H, so *privacy is unaffected* — epsilon is
  priced by `(p1, p2, r, K_in, K_out, T)` only (`accounting.py:517-524`), and
  `tests/test_dp_mechanics.py:322-356` verifies depth cannot widen the radius. What breaks is
  *fidelity*: the nodes at distance r have no in-arcs in H, so their layer-≥1 representations are
  computed as if isolated, while the evaluation forward computes them from their real neighbours.
  Measured train-vs-eval logit gap at L=2, r=1: 1.2e-1 max on a random graph; the repo's own
  measurement on capped arxiv is "0.6% mean, 1.9% max"
  (`scripts/_dataset_settings.sh:33-35`). This is a *deliberate* choice there (L fixed at 2, r
  swept in {1,2}) and is documented in that file only.
* **L < r**: the extra hops are literally discarded — measured `mean L=1, r=2` and `r=3` both give
  exactly the full-graph answer, i.e. identical to `r=1`. But epsilon *is* charged for them
  (shell `K_out^r`). Pure waste. Exception: for `--aggr gcn`, `r = L+1` is *not* waste — it is
  exactly what buys exactness (A.1).

## A.6 Self-loops, isolated roots, empty subgraphs

* `nodes` always contains the root (`sparse_expand.py:139-140`), so a subgraph is never empty; the
  worst case is 1 node / 0 edges. `edge_index` falls back to `zeros((2,0))` (`:173`).
* Such a root is **not skipped**: it flows through `iter_subgraph_loss_batches` →
  `_batched_loss_chunk` like any other and yields a finite loss and a real gradient
  (`gnn_mechanism.py:87-121`; `tests/test_sparse_batching.py:24-27` uses exactly such a subgraph).
  With mean the aggregate is the zero vector so the root's output is `b + W_r x_root`; with gcn the
  self-loop gives `Θ x_root + b`. The effective batch size is therefore *not* changed by empty
  subgraphs, and the Poisson assumption is intact.
* Roots outside `train_mask` return `zero_loss()` — a differentiable zero with exactly-zero gradient
  (`gnn_mechanism.py:72-74`, `base_mechanism.py:147-152`). They still occupy a slot in the batch
  (harmless; clipped to 0). `--roots_from train` (the default) avoids them anyway.
* An empty *root set* is handled explicitly: the DP path runs a noise-only step
  (`sparse_gnn.py:72-79`, `196-202`), the non-DP path `continue`s (`:205`).
* Self-loops present in G are traversed (root is its own in-neighbour), recorded as a local `(0,0)`
  arc, and do not extend the frontier. Under mean, such a root is counted *twice*: once inside the
  mean and again through `W_r`.

## A.7 Batching across subgraphs

Only `GNNMechanism` batches (`gnn_mechanism.py:87-147`); `multilabel`, `binary`, `regression`, and
`mlp` inherit `BaseMechanism.subgraph_losses` (`base_mechanism.py:133-140`), i.e. one forward per
root — so PPI and RelBench runs have no batching at all.

The batched path is a **disjoint union**, not a merge:
* per-subgraph node offsets `offset += subgraph.num_nodes` (`:97-99`),
* `edge_parts.append(subgraph.edge_index + offset)` (`:108`),
* one forward over the concatenated block (`:114`),
* `F.nll_loss(out[offsets], labels, reduction='none')` → one loss per root (`:118`).

Because each subgraph occupies its own index block and blocks share no arcs, message passing cannot
cross between roots; a node appearing in two subgraphs is materialized as two independent rows with
independent hidden states. Degree normalizers (both mean and `gcn_norm`) are computed per node and
therefore per block. Chunking (`:123-142`, `max_batched_subgraph_nodes=8192`) does not change any
value. `tests/test_sparse_batching.py:54-72` asserts batched == sequential for both losses and
clipped gradients. **No per-subgraph sensitivity leak from batching.**

## A.8 Feature handling

* Features are a plain gather `self.data.x[nodes]` (`gnn_mechanism.py:76`, `:114`) — no
  renormalization, no rescaling, no per-subgraph statistics.
* **No BatchNorm / LayerNorm / `F.normalize` anywhere in `src/`** (grep: the single `normalize=True`
  hit is `GCNConv`'s adjacency normalization, `layers.py:37`). SAGE's `normalize` (output L2) is off.
* Dropout is element-wise on the concatenated block — independent per element, no cross-root
  coupling.
* The one cross-subgraph statistic is upstream of this code: RelBench features are z-scored per
  column over the whole table (`relbench_data.py:98`) and regression targets are z-scored on
  train-split statistics (`relbench_data.py:257`, consumed at `regression_mechanism.py:61`).

## A.9 The GAD side pipeline (`src/sparse/gad/neighbor_aggregation.py`)

Parameter-free feature builder for XGBoost, not a trained GNN:

```
h^(0) = x ;  h^(l)_v = Aggr{ h^(l-1)_u : (u,v) in E } ;  feat(v) = [h^0 || h^1 || ... || h^L]
```

Differences from the main stack:
* **No weights, no bias, no activation** — pure aggregation; the root's own features are kept by
  **concatenation** of every level (`:63`, `:104`), where SAGE sums a learned `W_r` and GCN uses a
  self-loop.
* Same in-orientation convention (`scatter(h[src], dst)` — dst absorbs src, `:61`, `:101`) and the
  same zero-vector-for-empty-in-neighbourhood behaviour (`:52`).
* Two builders: `aggregate_features` runs on **one globally Bernoulli-sparsified edge set**
  (`sparsify_edges_bernoulli`, `:27-44`) shared by *all* roots — this is NOT the analyzed rooted
  mechanism (the arc draws are correlated across roots), and it is the **default**
  (`gad/run.py`, `xgb_graph.py:59-65`). `aggregate_features_expand` (`:66-105`) is the faithful
  per-root SparseExpand version.
* `aggregate_features_expand` inherits the same `L <= r` condition as the main stack: it runs r
  rounds over the full subgraph arc set and reads `h[0]` at each level, so level ℓ is exact iff
  ℓ ≤ r. `tests/test_gad.py:64-71` only covers `L == r`.
* **No degree capping, no p1 root sampling, no DP noise** anywhere in the GAD pipeline
  (`grep cap_degrees src/sparse/gad/` → no hits). It is a utility-degradation study over p2 only.

---

# (B) Open questions / discrepancies / risks

### B1. `--r 0` with a GNN mechanism leaves the neighbour weight untrained but still uses it at evaluation — **HIGH**

`scripts/_dataset_settings.sh:63` (PPI) and `:75` (RelBench) define the "graph-blind" baseline as
`BLIND=(--model multilabel_gnn --aggr mean --r 0)` / `(--model binary_gnn --aggr mean --r 0)`.
`sbatch/ppi_matched_eps.sbatch:98-103` uses the same at `--r 0` for the headline matched-epsilon
blind arm.

At `r = 0` every subgraph is `nodes=[root], edges=[]`, so the SAGE aggregate is identically zero and
`lin_l.weight` receives **exactly zero gradient**. Measured (`probe3.py`):

```
convs.0.lin_l.weight   |grad| = 0.000e+00      <- neighbour weight, no signal
convs.0.lin_l.bias     |grad| = 1.341e-01
convs.0.lin_r.weight   |grad| = 3.752e-01      <- root weight, trained
convs.1.lin_l.weight   |grad| = 0.000e+00
```

But `evaluate()` runs a **full-graph** forward (`gnn_mechanism.py:153`), so that never-trained
matrix multiplies real neighbour means. Non-DP, it stays at random init. **Under `--dp` it is a pure
Gaussian random walk**: `_step_dp` noises every parameter (`sparse_gnn.py:98-103`), measured
`||Δ|| = 0.30` vs `||init|| = 1.62` after only 20 steps at σ=5. For the PPI arm
(`lr=0.3`, `denom = p1·|pool| ≈ 512`, `T=2000`) the per-element drift is
`lr·σ/denom·sqrt(T) ≈ 0.026σ`, versus a Glorot init of ≈ 0.073 — i.e. at σ≈10 the neighbour weight
ends up ~3.6× larger than its initialization and entirely noise.

Measured impact (CiteSeer, non-DP, 100 steps, `probe5.py`): reported test 0.4730 vs 0.4970 for the
function actually trained (−2.4 pts, non-DP; far worse under DP).

Consequence: the "does the graph help?" comparison is biased **in favour of the graph arm**, because
the baseline it is compared against is a GNN with a noise-driven neighbour term rather than a blind
model. `--model mlp --r 0` (used for arxiv/flickr/reddit/facebook, `_dataset_settings.sh:87,94,103`)
is genuinely blind — `MLPMechanism.evaluate` ignores edges entirely (`mlp_mechanism.py:75`).
Open question: were PPI/RelBench blind numbers produced with `multilabel_gnn`/`binary_gnn`? If so
they need re-running, or evaluating with an empty edge set.

### B2. The "EXACT" claim is stated without its two preconditions — **HIGH (as a claim)**

`README.md:155-160`, `layers.py:8-13`, `run.py:180-184` all assert mean makes rooted inference
*exactly* equal full-graph inference, unconditionally. Measured truth (A.1): exact iff **`p2 == 1`
AND `L <= r`**. The shipped experiments violate both — `p2 ∈ {1.0, 0.5, 0.25, 0.1}`
(`sbatch/ppi_matched_eps.sbatch:64`) and `L=2, r=1` (`_dataset_settings.sh:53`, `R_VALUES=(1 2)`).
At `p2=0.5, L=r=2` the measured rooted-vs-full error is `max 5.1e-1 / mean 1.2e-1` on logits.
The defensible claim is narrower: *mean's normalizer is a function of the target's own arc draws, so
rooting introduces no additional boundary error beyond the sparsification itself* — which is true
and is the real reason mean beats gcn.

### B3. Train/eval aggregation mismatch at small p2: most roots aggregate nothing — **HIGH**

With mean, a root whose arcs are all dropped gets the **zero vector**, not "the mean of its
neighbours". `P(no in-arc survives) = (1-p2)^{deg_in} ≥ (1-p2)^K`:

| K=5 | p2=0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|
| P(root aggregates nothing) | 0.774 | **0.590** | 0.237 | 0.031 |

At the shipped `p2=0.1, K=5` grid point, ~59% of training roots are trained as pure MLPs, while
every root is evaluated with a full mean aggregate. This is a systematic train/test shift that is
*not* the same thing as sparsification noise — the zero vector is a specific point in feature space,
not an unbiased estimate of the neighbour mean. `run.py:66-88` warns only when mean subgraph size
< 1.05, which misses this regime entirely (at p2=0.1, K=5 the mean size is ≈ 1.5). Worth measuring:
does a "no neighbours" indicator feature, or skipping the aggregate term, close the gap?

### B4. `--K_out N` without `--K_in` silently applies **no cap at all** — **MED**

```python
if K_in_req is None:
    return ei, ''                      # run.py:441-442
```
`K_out_req` is computed (`run.py:431`) but never used on this path. `cap_degrees(ei, n, K_in=None,
K_out=3)` works correctly in isolation (verified: caps out-degree to 3), so this is purely the
`run.py` early return. Under `--dp` the run then falls into `k_in, k_out = max_degrees(train_ei)`
(`run.py:484-489`) and prints a warning, so the *reported* epsilon stays honest (it uses the raw max
degrees), but the user asked for `K_out=5` and got an uncapped graph priced at the raw degree. Since
under in-expansion it is `K_out` that prices the guarantee (`README.md:127`), `--K_out`-only is
exactly the invocation someone would try.

### B5. `gcn` is not intrinsically inexact — it needs `r = L+1` — **MED**

`layers.py:11-13` and `README.md:158-160` attribute a ~27-point PPI accuracy loss to the gcn
aggregator. Measured (A.1), gcn is bit-exact against full-graph inference at `r ≥ L+1` and only
wrong at `r = L`. The 27-point loss is therefore an artifact of running gcn at `r = L`, not a
property of symmetric normalization. The honest framing: *gcn costs one extra expansion shell
(`K_out^{L+1}` instead of `K_out^L`) to be exact, which is why mean is the right default* — an
epsilon argument, not a correctness argument.

### B6. No L-vs-r guard, and one script mislabels its own runs — **MED**

Nothing anywhere checks `num_layers` against `r`. `scripts/ladder_stage01.sh:65` echoes
`"[S1] sparsification sweep, r=$R (L=$R), capped"` but line 68 passes `--num_layers $L` with
`L=2` fixed (`_dataset_settings.sh:53`) — so the `r=1` rung is logged as "L=1" while actually
running L=2. Logs and CSV (`run.py:650` records the real `args.num_layers`) disagree. A one-line
warning in `run.py` when `num_layers > r` (fidelity) or `num_layers < r` (wasted epsilon) would
catch both this and B1 (`r=0` with a GNN mechanism).

### B7. Inductive runs evaluate on an uncapped graph they never trained on — **MED**

`run.py:322-328` sets `eval_graph='full'` for inductive runs → `data.edge_index`, uncapped and
unfiltered, while training used degrees ≤ K. With mean the shift is bounded (a mean over more
neighbours) but real, and it compounds with B3. The capped counterpart is recorded under `*_alt`
(`run.py:479-481, 616`), so the comparison exists in the CSVs — but the headline column is the
off-distribution one. `README.md:161-163` describes the *opposite* default for transductive runs
and is stale.

### B8. Global feature statistics cross subgraph boundaries — **MED** (upstream of this scope)

`relbench_data.py:98` z-scores each feature column over the whole table and `:257` z-scores
regression targets on train-split statistics. Every per-root forward therefore reads features whose
normalization depends on every other node's data (and on labels, for the target scale). This is the
standard, usually-tolerated preprocessing caveat, but it is a genuine cross-subgraph dependency that
the per-subgraph sensitivity argument does not cover. Flagging because it is the only global
statistic in the feature path — the model itself is clean (no BatchNorm anywhere).

### B9. Each undirected edge gets two independent p2 coins — **LOW**

On a symmetric arc set, expanding the root draws for `(w,root)` and expanding `w` later draws
independently for `(root,w)`. Consistent with treating the graph as directed with 2 arcs (which is
what the accounting does), but means an "edge" survives with probability `p2` per direction, not
`p2` per edge. Worth confirming the theory statement matches.

### B10. The headline aggregator claim has no test — **LOW**

`tests/test_sparse_expand.py:219-236` only checks that a single `GCNConv` responds to a
root-directed arc on a 2-node graph. Nothing tests `rooted(root) == full(root)` for the mean
aggregator at `L <= r` — the property the whole "train sparsified, evaluate full" design rests on,
and the one that silently fails at `L > r`. A 5-line parametrized test over `(aggr, L, r)` would
pin down the exact table in A.1.

### B11. Degree-cap tests live in the wrong file per the README — **LOW**

`README.md:150-151` says `test_sparse_expand.py` covers "expansion, orientation, degree capping".
It covers no capping at all; the cap tests are in `tests/test_accounting.py:438-490`. Those tests
check bounds, subset, determinism and symmetry — but **not uniformity of the truncation**
(I verified uniformity empirically; nothing would catch a regression to first-K-by-index, which
would be a systematic bias toward low-id neighbours).

### B12. `dedup_arcs` returns a differently-ordered tensor depending on whether duplicates existed — **LOW**

`sparse_expand.py:238-241`: no duplicates → the original (unsorted) tensor; duplicates → a
sorted rebuild. The degree cap consumes randomness in arc order, so two graphs differing by one
parallel arc produce unrelated capped graphs from the same `cap_seed`. Harmless in practice
(all shipped loaders are simple graphs), but it makes `--cap_seed` reproducibility contingent on the
input's duplicate status.

### B13. A real self-loop is double-counted under mean — **LOW**

SAGE adds no self-loop, but if G contains `(v,v)` (preserved by `cap_degrees_undirected`,
`sparse_expand.py:306-311`), `v` appears both inside its own mean and through `W_r`. GCN's
`add_remaining_self_loops` sets the existing loop's weight to 1 rather than adding a second, so gcn
is unaffected. Only matters for graphs that ship self-loops.

### B14. `_report_subgraph_size` probes only the seed-0 graph — **LOW**

`run.py:507, 513-515` uses `graphs[0]['adj']`, and its warning threshold (mean size < 1.05) is far
below the regime where the aggregate is usually empty (B3). It also probes at `max(p2), max(r)`,
i.e. the *widest* cell of the sweep, so the narrow cells that actually degenerate are never
reported.

### B15. The GAD default sparsifier is not the analyzed mechanism — **LOW** (documented)

`xgb_graph.py:59-65` defaults to `sparsifier="global"`: a single Bernoulli(p2) draw over the whole
edge set, shared by every root. The per-root independence that SparseExpand provides is absent.
`neighbor_aggregation.py:11-16` documents the distinction honestly ("expand ... matches the DP
mechanism exactly"), and the GAD pipeline claims no DP guarantee (no capping, no p1, no noise), so
this is only a risk if GAD numbers are ever quoted next to an epsilon.

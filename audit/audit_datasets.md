# Dataset / preprocessing / split audit — `markovian_dp copy` @ `sparse_expand_clean`

Scope: datasets, preprocessing, splits, transductive/inductive distinction.
Read-only. All statistics below marked **(measured)** were produced by loading the
dataset through `src/datasets.py` in `PytorchEnv` (torch 2.8.0 / PyG 2.7.0) —
not copied from the literature.

---

# (A) What the code does

## A.1 Dataset inventory

`SUPPORTED_DATASETS` is declared at `src/datasets.py:12-41`; dispatch is
`src/datasets.py:518-603`.

| key | loader | N | arcs (directed rows of `edge_index`) | sym? | F | C | task | inductive? |
|---|---|---|---|---|---|---|---|---|
| `cora` | `datasets.py:600-603` Planetoid | 2 708 | 10 556 | yes | 1 433 | 7 | single-label | transductive; `--inductive` **degenerate** (42 arcs survive) |
| `citeseer` | same | 3 327 | 9 104 | yes | 3 703 | 6 | single-label | same (16 arcs survive) |
| `pubmed` | same | 19 717 | 88 648 | yes | 500 | 3 | single-label | same (**0** arcs survive) |
| `cora-ml` | `datasets.py:193-227` | 2 995 | 8 416 | **no** (246 in / 72 out) | 2 879 | 7 | single-label | **no masks at all** — see B4 |
| `ogbn-arxiv` | `datasets.py:69-89` | 169 343 | 1 166 243 | **no** (13 155 in / 436 out) | 128 | 40 | single-label | made inductive by `--inductive` (32.1 % of arcs survive) |
| `ogbn-products` | same | ~2.45 M | ~123 M | yes | 100 | 47 | single-label | made inductive (not downloaded here) |
| `reddit` | `datasets.py:564-569` | 232 965 | 114 615 892 | yes | 602 | 41 | single-label | made inductive (45.6 % of arcs survive) |
| `flickr` | `datasets.py:571-578` | 89 250 | 899 756 | yes | 500 | 7 | single-label | made inductive (24.2 % of arcs survive) |
| `ppi` | `datasets.py:149-190` | 56 944 | 1 587 264 | yes | 50 | 121 | **multilabel** (`y` float `[N,121]`) | **natively** inductive (24 disjoint graphs, 20/2/2) |
| `facebook` | `datasets.py:230-337` | 26 406 | 2 117 924 | yes | 501 | 6 | single-label | transductive by design (`_dataset_settings.sh:81-92`) |
| `tolokers` | `datasets.py:499-515` | 11 758 | 1 038 000 | yes | 10 | 2 | binary (anomaly) | transductive; GAD side pipeline only |
| `questions` | same | 48 921 | 307 080 | yes | 301 | 2 | binary (anomaly) | transductive; GAD only |
| `ogbl-collab` | `datasets.py:92-122` | ~235 k | ~2.4 M | yes | 128 | n/a | **link prediction** | listed but unusable — see B12 |
| `bluesky` | `datasets.py:477-496` | — | — | — | — | — | — | **stub, raises `NotImplementedError`** (`datasets.py:379-384`) |
| `relbench:<db>/<task>` | `src/sparse/relbench_data.py:107-293` | e.g. `rel-f1/driver-top3`: 76 730 | 172 088 | **no** (3 701 in / 105 out) | 82 | 2 | binary / multiclass / **regression** | **natively** inductive (temporal) |

All counts, symmetry and max-degree figures are **(measured)** and agree exactly
with the table in the `degree-cap-by-graph-type` memory note.

Task-type routing is enforced in `src/sparse/run.py:386-400`: multilabel ⇒
`multilabel_gnn` (hard error), `REGRESSION` ⇒ `regression_gnn` (hard error),
`BINARY` ⇒ `binary_gnn` (warning only).

RelBench shorthands registered: `relbench-f1-top3`, `relbench-f1-dnf`
(`datasets.py:39-40`); any pair may also be named directly as
`relbench:<database>/<task>` (`datasets.py:537-543`, parser at
`relbench_data.py:315-326`).

## A.2 The three inductive paths

### Path (a) — natively inductive loaders
* **PPI** (`datasets.py:149-190`): the 24 PPI graphs are concatenated with node
  offsets (`:167-182`) and masks assigned by source graph (`:183-185`). Components
  are disconnected, so a train root provably cannot reach a val/test node.
  **(measured)** cross-partition arcs = 0.
* **RelBench** (`relbench_data.py:271-286`): `train_end` = max timestamp over the
  train rows; `data.train_edge_index` keeps arcs whose *both* endpoints have
  `node_time <= train_end`. `data.edge_index` is the full graph, which is the
  test-cutoff graph because `relbench.get_dataset(...).get_db()` defaults to
  `upto_test_timestamp=True`. **(measured)** zero train-graph arcs touch a val/test
  row node.

### Path (b) — `--inductive` on a single-graph dataset
`src/sparse/run.py:402-422`. Two branches:
* if the loader supplied `train_edge_index` (RelBench only) → use it verbatim
  (`:404-410`);
* otherwise → keep only arcs with **both** endpoints in `train_mask` (`:411-422`).

Evaluation is *not* filtered: `--eval_graph auto` resolves to `full`
(`run.py:322-328`), i.e. `data.edge_index` with every cross-split arc restored.

**(measured)** fraction of arcs surviving the train-induced filter: arxiv 32.1 %,
flickr 24.2 %, reddit 45.6 %, facebook 56.2 %, ppi 77.3 %, cora 0.4 %,
citeseer 0.2 %, pubmed 0.0 %.

### Path (c) — `--common_inductive_split` (shared with the baselines)
`src/sparse/run.py:364-381` calls `load_or_create_inductive_split`
(`src/experiments/inductive.py:228-301`), overwrites `train/val/test_mask` with a
**deterministic stratified 60/20/20 re-split** (`inductive.py:77-106`), and then
deletes every inter-partition arc from `data.edge_index` itself (`run.py:374-379`).
Because `data.edge_index` is the graph both `--inductive` and `--eval_graph full`
read from, training *and* evaluation then happen on the same 3-component graph the
baselines see. Only `scripts/run_sparse_inductive_ablation.py:41-49` uses this
path.

### Path (d) — the baseline harness
`src/experiments/run.py:74-83` always builds an `InductiveSplit` and hands each
trainer three *materialised, relabelled, cross-edge-free* induced graphs
(`inductive.py:169-186`). Both portable baselines
(`baselines.py:100-105`) and DPAR (`dpar.py:289-296, 348-356`) evaluate on
`split.val.data` / `split.test.data`, and external baselines (ProGAP, DP-GNN,
HeterPoisson) receive the same three `.pt` files via
`upstream.py:37-49`.

**Semantic equivalence.** (a) ≡ (c) ≡ (d). Path (b) is **not** equivalent to any of
them: it removes cross-split arcs from training only, and restores them for
evaluation. See B1.

### A.3 Which graph is used for train vs eval

| | training graph | primary metric graph | `_alt` metric graph |
|---|---|---|---|
| transductive (`--inductive` absent) | dedup + degree-capped **full** graph (`run.py:432-465`) | **same capped training graph** (`eval_graph='train'`, `run.py:613-615`) | raw uncapped `data.edge_index` (`run.py:615`) |
| inductive (`--inductive`) | dedup + capped **train-induced** graph | **raw uncapped full `data.edge_index`** (`eval_graph='full'`, `eval_edge_index` left `None`, `base_mechanism.py:75-76`) | capped-but-**unfiltered** graph `eval_capped` (`run.py:479-482, 617`) |

`_alt` is attached in `sparse_gnn.py:107-113` and written to the
`train_acc_alt/val_acc_alt/test_acc_alt` + `*_auroc_alt` columns
(`run.py:564-565, 664-668`).

So the README's "evaluation defaults to the full one" (README:162-163) and
`paper/experiments_current.tex:37-38` describe only the **inductive** half of the
policy. For transductive runs the default is the *training* graph. (The
`eval-graph-and-depth-policy` memory note has it right; the README and the tex do
not.)

`MLPMechanism.evaluate` ignores the graph entirely (`mlp_mechanism.py:72-83`), so
for `--model mlp` arms the `_alt` columns are numerically identical to the primary
ones. Every other mechanism routes through `eval_edges`
(`gnn_mechanism.py:153`, `multilabel_mechanism.py:110`, `binary_mechanism.py:94`,
`regression_mechanism.py:79`).

### A.4 Splits, per dataset

| dataset | source of masks | fixed or per-seed |
|---|---|---|
| Planetoid ×3 | PyG's public split (20/class, 500 val, 1000 test) | fixed; **masks cover only 1 640/2 708 nodes on cora** |
| OGB node | official `get_idx_split()`, validated for disjointness + full coverage (`datasets.py:44-66`) | fixed |
| Flickr / Reddit | PyG's shipped GraphSAINT `role.json` masks | fixed |
| PPI | assigned by source graph index (`datasets.py:169-185`) | fixed |
| Tolokers / Questions | 1 of 10 shipped split columns, `split_idx` (`datasets.py:509-514`) | **always column 0** from `src.sparse.run` — that CLI never passes `split_idx` |
| facebook | random 75/10/15, `torch.Generator().manual_seed(seed)` with `seed=0` **hard-defaulted** (`datasets.py:231, 297-305`) and never overridden (`datasets.py:586`) | fixed for all runs and all seeds |
| cora-ml | **none** | — |
| RelBench | `task.get_table(split)` — RelBench's own temporal windows | fixed |
| `--common_inductive_split` / baselines | stratified 60/20/20 (`inductive.py:77-106`), seeded by `--split_seed` (default 0) for our method, by `config["seed"]` for the baselines (`experiments/run.py:70, 75-83`) | our method: fixed; baselines: **re-drawn per seed** |

**RelBench temporal boundaries.** `relbench_data.py:142-143` pulls RelBench's own
train/val/test tables. The only boundary the code computes itself is `train_end`
= max timestamp over *train rows* (`:271-272`); arcs are admitted to
`train_edge_index` iff both endpoints satisfy `node_time <= train_end` (`:273`).
There is no val-cutoff graph.

## A.5 Preprocessing

| step | where | before or after split |
|---|---|---|
| feature normalisation | **none** for any PyG/OGB/Planetoid/Flickr/Reddit/PPI/facebook dataset — no `NormalizeFeatures`/`to_undirected` transform anywhere in `src/` | n/a |
| RelBench per-column z-scoring | `relbench_data.py:93-101` (`mu`/`sd` over the whole column) | **after** the split is known but fit on **all** rows |
| RelBench categorical one-hot | `relbench_data.py:82-92`, vocabulary = all rows | all rows |
| RelBench datetime → z-scored epoch-years | `relbench_data.py:68-72, 93-101` | all rows |
| RelBench **target** z-scoring | `relbench_data.py:257-269` — **train rows only**, scale saved as `target_std` | correct |
| symmetrisation (`to_undirected`) | **never called** | — |
| self-loop addition | never at the data level; `GCNConv(add_self_loops=True)` internally (`layers.py:37`). **(measured)** every shipped dataset has 0 self-loops | — |
| edge dedup | `run.py:437-440` → `dedup_arcs` (`sparse_expand.py:226-241`) | after the inductive filter |
| degree cap | `run.py:441-465`, one draw per seed (`run.py:500-505`) | after dedup |
| self-loop + isolated-node removal | `datasets.py:321-332` (facebook only) | **after** the split was drawn (matches ProGAP) |
| label encoding | facebook `datasets.py:276-279` (`pd.Categorical` codes, shifted if 0 present) then re-encoded after `FilterClassByCount` (`:308-319`) | split drawn before the class filter, then subset |

The `symmetric=True` "gotcha" in my memory index is in `accounting.py:232-238`
(out of scope here). The *dataset-side* symmetry switch is `--cap_mode auto` (A.6).

## A.6 RelBench flattening

`relbench_data.py:107-293`.

* **Nodes** — one per row of every table in `db.table_dict` (`:147-153`), plus one
  "row node" per task row when `root='row'` (default, `:164-166`). `root='entity'`
  instead collapses to one node per (entity, split) with `label_agg` in
  {`last`,`max`} (`:158-163`) and discards ~93 % of rel-f1's supervision.
* **Edges** — one arc per foreign key, oriented **child → parent** (`:206-217`);
  plus **entity → row-node** (`:225-231`) so that in-expansion from a row root
  reaches its entity at depth 1 and the entity's history at depth 2.
  `reverse_edges=True` mirrors every arc (`:235-236`).
* **Columns** (`_encode_table`, `:58-104`) — pkey and fkey columns skipped
  (`:172-173`); numerics z-scored; datetimes → epoch-years, z-scored;
  non-numeric with `nunique <= max_categories` (32) → one-hot; anything wider, or
  any list-valued cell, dropped. NaNs map to 0 (i.e. the column mean).
* **Layout** — block-diagonal per table plus a node-type one-hot
  (`:168-191`); a row node's only own feature is its timestamp (`:176-180`).
* **Timestamps** — `node_time` per node, `-inf` for static tables (`:194-204`);
  stored on `data.node_time` (`:282`).
* **Direction is load-bearing.** With `--direction in` the root's neighbourhood is
  built along incoming arcs, so `row → entity → entity's child rows`; `r >= 2` is
  required to see any history. **(measured)** max in-degree 3 701 vs out-degree 105
  on rel-f1, confirming parents are the hubs. `--K_in 20 --K_out 3`
  (`_dataset_settings.sh:79`) therefore truncates each entity's history to 20 rows.

## A.7 Undirected vs directed

The only switch is `--cap_mode` (`run.py:264-271`), resolved at `run.py:446-448`:

```python
mode = ('undirected' if K_in_req == K_out_req and
        edge_set_is_symmetric(ei, n_nodes) else 'directed')
```

It switches **only which capping routine runs**:
* `undirected` → `cap_degrees_undirected` (`sparse_expand.py:258-311`): collapse arc
  pairs, greedy over a random edge order keeping every endpoint's degree ≤ K,
  re-emit both arcs. Result stays symmetric.
* `directed` → `cap_degrees` (`sparse_expand.py:177-223`): independent in-pass then
  out-pass; on a symmetric graph this leaves only ~1/3 of arcs reciprocated.

It does **not** touch `--direction` (expansion orientation), which is an
independent flag defaulting to `in`. **(measured)** auto picks `undirected` for
cora/citeseer/pubmed/flickr/reddit/ppi/facebook/tolokers/questions and `directed`
for ogbn-arxiv, cora-ml and RelBench.

## A.8 Reproducibility surface

* `_set_seed` seeds `random` + `torch` only (`run.py:53-55`); numpy is never
  seeded, but no loader or capping routine uses `np.random`.
* The degree cap is redrawn per seed unless `--cap_seed` is pinned
  (`run.py:272-278, 500-505`) — deliberate, documented.
* `_report_subgraph_size` uses its own generator seeded 999 (`run.py:77`), so the
  probe does not perturb training.
* No dataset version is pinned anywhere except cora-ml, which pins an exact commit
  (`datasets.py:207-208`). README:52 pins nothing.
* `torch.load` is globally monkey-patched to `weights_only=False` around OGB loads
  (`datasets.py:74-80, 95-105`); restored in `finally`.

---

# (B) Open questions / discrepancies / risks

### B1. HIGH — `--inductive` trains on the cut graph but evaluates on the uncut graph; the baselines never get that
`src/sparse/run.py:402-422` vs `run.py:322-328`. Our method drops cross-split arcs
for training and restores **all** of them for scoring. Every baseline
(`experiments/run.py:74-83` → `inductive.py:169-186` → `baselines.py:100-105`,
`dpar.py:289-296`, `upstream.py:37-49`) is scored on the *test-induced subgraph*,
where **(measured)** 60–76 % of arcs are gone on arxiv/flickr and test nodes
retain only test–test edges.

Defensible as "inductive inference on arriving nodes", but it is a different
evaluation protocol from the one every baseline runs, so SparseExpand numbers from
`ladder_stage01.sh` / `*_matched_eps.sbatch` **cannot be tabulated against**
baseline numbers from `src.experiments.run`. The fair path exists —
`--common_inductive_split` (`run.py:364-381`) — and `configs/inductive_comparison.json`
+ `configs/sparse_inductive_ablation.json` are designed to pair on it, but no
campaign in `results/` uses it. The head-to-head rows in
`paper/experiments_current.tex:168-171` (DPAR / DP-MLP / ProGAP) are still empty,
so nothing is published on the unfair comparison **yet**.

### B2. HIGH — the `--r 0` "graph-blind" arm is evaluated **with** message passing, using untrained neighbour weights
`_dataset_settings.sh:64, 76` defines the blind arm as the *same* GNN mechanism at
`--r 0`. At `r=0` a rooted subgraph is one node with no edges, so
`SAGEConv.lin_l` receives **exactly zero gradient** — verified:
`lin_l` grad norm 0.0, `lin_r` grad norm 3.38 on an empty `edge_index`. Evaluation
then runs `self.module(data.x, self.eval_edges(data))`
(`multilabel_mechanism.py:110`, `binary_mechanism.py:94`,
`regression_mechanism.py:79`) over a real graph, injecting the **random-init**
neighbour weights into every prediction.

Measured consequence on `results/relbench/relhm_userchurn_meps/`:

| arm | test AUROC on full graph (primary) | on capped graph (`_alt`) |
|---|---|---|
| blind `r=0`, ε=1 | 0.5093 | **0.6022** |
| GNN `r=2 p2=0.1`, ε=1 | 0.5582 | 0.6154 |
| blind `r=0`, non-DP | 0.4834 | 0.5926 |
| GNN `r=2`, non-DP | 0.5867 | 0.6295 |

The blind arm gains ~10 AUROC points purely from the choice of eval graph. The
headline claim in `paper/experiments_current.tex:85` ("GNN AUROC 0.55–0.56 vs.
blind 0.49–0.51 (chance)") is a **+6 pt** margin on the primary graph and a
**+1.4 pt** margin on the alt graph. Same pattern on `rel-f1/driver-position`
(`r=0`: 23.45 vs 8.81 MAE) and `rel-amazon/item-ltv` (93.6 vs 87.2). PPI's blind
arm is affected too but far less (0.4215 vs 0.4194).
Fix direction: score the blind arm with an empty `edge_index`, or use `--model mlp`
(as `sbatch/sparse_relbench.sbatch:47` already does) — the two definitions of
"blind" currently coexist in the repo.

### B3. HIGH — RelBench training lets a train root read the rows that define its own label
`relbench_data.py:271-273` cuts the training graph at a **per-split** timestamp
(max over all train rows), not per row. The module's own caveat
(`relbench_data.py:27-30`) calls this "leakage between training examples only",
which understates it: a train row's label is computed from rows in
`(t_row, t_row + timedelta]`, and those rows are ≤ `train_end`, hence present.

**(measured)** on `rel-f1/driver-top3`, `direction=in, r=2, p2=1`: a train root's
rooted subgraph averages 269.5 nodes, of which **122.8 (46 %) are strictly later
than the root's own timestamp**. The `results` table's `position`/`points` columns
are node features, so the answer is literally two hops away during training.
Test rows are clean (0 future nodes reachable — `get_db()` truncates at
`test_timestamp`), so this does not inflate the reported test metric; it means the
model is trained on a shortcut that vanishes at test time, which is a plausible
mechanism for the "ε trend runs backwards" anomaly noted at
`experiments_current.tex:86-89`.

### B4. HIGH — RelBench validation metric is temporally leaky, and val selects checkpoints
`data.edge_index` is the test-cutoff graph, used unchanged for *all three* splits
(`base_mechanism.py:75-76`). **(measured)** a val row reaches on average 123.5
nodes dated after its own timestamp. Since `summarize_sweep.py` now picks the best
checkpoint on validation (commit `30ceee7`), selection is made on a contaminated
signal. No val-cutoff graph is ever constructed (`relbench_data.py:280-281` builds
only `train_edge_index`).

### B5. HIGH — RelBench feature statistics are fit on the whole database
`relbench_data.py:97-99` computes `mu`/`sd` over every row of every column, and
`:82-92` builds the categorical vocabulary the same way. These rows include the
entire val window and everything up to the test cutoff. Two separate problems:
1. *Leakage*: val/test-period feature distributions inform the representation the
   model trains on.
2. *Node-DP*: the features consumed by the mechanism are a data-dependent function
   of **all** nodes. `accounting.py` charges nothing for it, so the released model
   depends on every node's data through a channel outside the guarantee. Contrast
   with the target scaling at `:257-269`, which correctly uses train rows only.

### B6. MED — the primary metric is measured on a different graph for transductive vs inductive runs
`run.py:322-328`. Transductive → capped training graph (in-distribution).
Inductive → raw uncapped graph (off-distribution). **(measured)** the gap is large:
user-churn GNN 0.5582 (full) vs 0.6154 (capped); reddit 0.9377 vs 0.8893. Any table
that puts a transductive dataset (facebook) next to inductive ones (arxiv, ppi at
`inductive=False` but relbench at `inductive=True`) — e.g.
`experiments_current.tex:109-118` — is mixing two protocols in one row.

### B7. MED — README and the paper describe the eval-graph policy incorrectly
README:162-163 and `paper/experiments_current.tex:37-38` both say "evaluation
defaults to the full one". True only when `--inductive` is set
(`run.py:322-328`). Also README:168-170 says the cross-split drop is "68–76 % of
edges on ogbn-arxiv, Flickr, and Reddit"; **(measured)** arxiv 67.9 %, flickr
75.8 %, **reddit 54.4 %**.

### B8. MED — the baselines re-draw their split per seed; our method does not
`experiments/run.py:70, 75-83` passes the run seed as the split seed, so seed
variation includes split variation. `src/sparse/run.py:200-201` has a separate
`--split_seed` (default 0) while `--seeds` is only a count. Reported ± are
therefore not the same quantity on the two sides.

### B9. MED — `facebook`'s split is a hard-coded `seed=0` and is never varied
`datasets.py:231` (`seed=0` default) and `datasets.py:586` (`_load_facebook()` with
no args). Every facebook run in `results/` shares one random 75/10/15 split, so the
reported seed spread excludes split variance entirely. Compare ProGAP, whose
protocol (cited at `datasets.py:249-250`) averages over random splits.

### B10. MED — `--common_inductive_split` + RelBench is silently wrong
`run.py:364-381` replaces `data.edge_index` with within-partition arcs, but
`data.train_edge_index` is left untouched; `run.py:403-410` then prefers
`train_edge_index`. On a RelBench binary task the combination trains on the
loader's train-cutoff graph while the masks come from a stratified re-split of
*all* nodes — including the ~74 000 unlabeled DB rows, which get `y = 0`
(`relbench_data.py:241`) and land in train/val/test. Result: bogus supervision and
a training graph where train roots *can* reach "val/test" nodes. Guard-railed only
by the `_split_indices` float check, which RelBench binary tasks pass.

### B11. MED — `data/inductive_splits/*.pt` cache is keyed on node count alone
`inductive.py:265-278` validates only `payload["num_nodes"]` (and
`split_strategy` for native). A stale file from an earlier split policy, an earlier
dataset version, or a different ratio is silently reused. There is no policy
version stamp and the requested `seed` is not cross-checked against the payload's.

### B12. MED — several registered datasets cannot actually run
* `bluesky`: `_read_bluesky_raw` raises (`datasets.py:379-384`).
* `ogbl-collab`: `train/val/test_mask` are all-ones (`datasets.py:117-121`), so
  every mechanism would report train == val == test; no link-prediction mechanism
  exists (`relbench_data.py:41-44` explains why LP does not fit the g0 shape).
* `cora-ml`: `_load_cora_ml` returns no masks at all (`datasets.py:225-227`), so
  `src.sparse.run` dies at `run.py:511` / `run.py:116` unless
  `--common_inductive_split` is passed.
None of these are gated at `load_dataset`.

### B13. MED — Planetoid + `--inductive` is degenerate, and the ladder's default branch would apply it
`_dataset_settings.sh:100-107` sends every unrecognised dataset down the
`INDUCTIVE=(--inductive)` branch. **(measured)** the train-induced subgraph has
42 arcs on cora, 16 on citeseer and **0** on pubmed — the GNN becomes an MLP.
`run.py:86-87` would print the "roots are effectively isolated" warning, but
nothing refuses the configuration. Related: Planetoid masks cover only
1 640/2 708 nodes, so `split_strategy="native"` is unavailable for them
(`inductive.py:127-128`).

### B14. MED — `--inductive` also swaps which graph is *evaluated* and *capped*, contradicting the PPI docstring
`datasets.py:158-159` says `--inductive` "is not needed or has any effect" on PPI.
It has no effect on *training* (correct), but it flips `eval_graph` from `train` to
`full` (`run.py:322-328`) and changes what gets capped (`run.py:432, 474`).
**(measured)** on PPI it removes 22.7 % of arcs — all val/test-internal — which is
exactly why `run.py:476-478` has to build a separate unfiltered `eval_capped`.
The docstring should say "no effect on training".

### B15. MED — no `to_undirected`, so ogbn-arxiv is used directed
`degree-cap-by-graph-type` notes this deliberately, and it is the right choice for
the theory (`sparse_expand.py:17-18`), but the OGB leaderboard convention is to
symmetrise arxiv. **(measured)** the raw graph is 13 155-in / 436-out, so the
`directed` cap path applies and in-expansion follows incoming citations. Any
comparison to published arxiv numbers is therefore not like-for-like and should say
so.

### B16. MED — Planetoid features are not row-normalised
No `NormalizeFeatures` anywhere in `src/`. Cora/CiteSeer/PubMed numbers in
`results/archive/` are therefore on raw bag-of-words counts, unlike essentially
every published Planetoid number.

### B17. LOW — `split_idx` is unreachable from `src.sparse.run`
`datasets.py:509-514` supports 10 shipped splits for Tolokers/Questions;
`run.py:356-360` never forwards `split_idx`, so only column 0 is ever used there.
The GAD CLI does expose it (`gad/run.py:79-80`).

### B18. LOW — `_induce` does not subset non-standard node tensors
`inductive.py:169-186` re-indexes `x` and `y` only. `node_time`,
`train_edge_index` and `target_std` survive `data.clone()` at full length and
become misaligned. Latent today because RelBench cannot reach that code path
(`experiments_current.tex:155-158`), but it will bite the moment it can.

### B19. LOW — hard-coded `/tmp` roots for Planetoid and the heterophilous graphs
`datasets.py:507` and `datasets.py:601` use `root=f'/tmp/{canonical}'` with no env
override, unlike every other loader (`OGB_DATA_ROOT`, `PPI_DATA_ROOT`,
`FLICKR_DATA_ROOT`, `REDDIT_DATA_ROOT`, `CORA_ML_DATA_ROOT`,
`FACEBOOK_DATA_ROOT`, `BLUESKY_DATA_ROOT`). On a cluster `/tmp` is purged and
per-node, so these silently re-download.

### B20. LOW — no dataset/library version pinning
`relbench.get_dataset(..., download=True)` (`relbench_data.py:137-138`) and
`PygNodePropPredDataset` (`datasets.py:78`) are unpinned, and README:52 pins
nothing. For RelBench specifically, the node-id space and the feature column layout
are derived from `db.table_dict` iteration order and per-column `nunique`
(`relbench_data.py:141, 150-153, 82-92`), so a library or data revision silently
renumbers every node and reshapes `x`. Only cora-ml is pinned
(`datasets.py:207-208`).

### B21. LOW — `num_classes` for RelBench is read off train ∪ val ∪ test labels
`relbench_data.py:288-289`. Standard practice, but it is a (tiny) use of test
labels at model-construction time.

### B22. LOW — stale in-code line references
`sbatch/arxiv_inductive_matched_eps.sbatch:34` cites "run.py:331-337" for the
train-induced filter; it now lives at `run.py:411-422`.

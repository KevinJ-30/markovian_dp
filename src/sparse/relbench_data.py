"""
RelBench entity tasks as a homogeneous directed graph for SparseGNN.

RelBench ships relational databases plus temporal entity-prediction tasks.  This
module turns one (dataset, task) pair into the single `Data` object the
SparseGNN engine already consumes, so no part of the mechanism, the DP path, or
the accounting needs to know that the source was a relational database.

Why build the graph from the raw `Database` instead of RelBench's own
`make_pkey_fkey_graph`
----------------------------------------------------------------------------
`make_pkey_fkey_graph` produces a HeteroData whose per-type features are
`torch_frame` TensorFrames, which additionally drags in a text embedder for text
columns.  Our engine is homogeneous and our accounting is stated for a single
directed graph with degree bounds (K_in, K_out).  Walking `db.table_dict`
directly gives exactly that, with no extra dependencies.

Graph construction
------------------
Nodes
    One node per row of every table, plus one node per row of the task table
    ("row nodes").  A row node is the ROOT of a prediction: it carries a single
    label at a single timestamp, which is what `subgraph_loss` expects.

    Using row nodes rather than entity nodes as roots matters on rel-f1: the
    task has 1353 training rows but only 92 distinct drivers, so entity roots
    would throw away 93% of the supervision.  Pass root='entity' to get the
    entity framing instead (one root per driver, label aggregated over rows).

Edges
    One arc per foreign key, oriented CHILD -> PARENT (e.g. results -> drivers),
    plus one arc entity_row -> row_node.  Information therefore flows

        results / standings / qualifying  ->  driver  ->  prediction row

    so under the corrected in-expansion (Algorithm 5) a row root reaches its
    driver at depth 1 and that driver's history at depth 2.  Note the row arc
    runs PARENT -> CHILD, the opposite of the foreign-key arcs: a row node is a
    readout, so it is fed by its entity rather than referencing it.  (With the
    arc the other way a row root has in-degree 0 and expansion reaches nothing —
    the mean-subgraph-size diagnostic in run.py catches exactly that.)

    `reverse_edges=True` additionally adds the mirrored arcs, which enriches
    neighbourhoods but inflates K_out and hence epsilon (Theorem 6.4, Eq. 44) —
    it is off by default.

    Depth matters here: r=1 gives a root only its entity, so r>=2 is required
    for a RelBench root to see any history at all.

Features
    Per table: z-scored numerics, datetimes as z-scored epoch-years, and
    one-hot categoricals up to `max_categories`.  Free-text and high-cardinality
    identifier columns are dropped.  Each table's block is written into its own
    column range of a shared `x` (block-diagonal), and a node-type one-hot is
    appended, so a single shared input layer acts as a per-type encoder and
    `GNNMechanism` needs no changes.

Time
    Splits are temporal.  Two edge sets are attached:

        data.edge_index        graph at the TEST cutoff — used by `evaluate`,
                               so held-out rows have their real neighbourhoods
        data.train_edge_index   graph at the TRAIN cutoff — used for training
                               expansion, so training can never read an edge
                               that did not exist yet

    `src.sparse.run --inductive` picks up `train_edge_index` automatically.

    Caveat: the cutoff is per SPLIT, not per row.  Inside the training window a
    row dated 1998 may reach a row dated 2003.  That is future leakage between
    training examples only — never into val/test — and it is the coarse version
    of what RelBench's own neighbour sampler does per seed time.  Per-root time
    filtering would require SparseExpand to consult a timestamp while expanding;
    it is not implemented here.

Scale caveat
    rel-f1 is the smallest RelBench database (1353 training rows).  It is the
    right target for validating the pipeline, but epsilon at that sample size is
    not informative — for DP numbers point this at a large task (rel-hm
    user-churn, rel-stack user-badge) via --dataset relbench:<db>/<task>.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# Columns that are pure identifiers or free text carry no signal for us and
# would blow up the one-hot width.
_MAX_CATEGORIES = 32


def _encode_table(df: pd.DataFrame, skip_cols: set, max_categories: int
                  ) -> np.ndarray:
    """Dense float matrix for one table: numerics, datetimes, small categoricals."""
    blocks = []
    for col in df.columns:
        if col in skip_cols:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(dtype=np.float64, na_value=np.nan)
        elif pd.api.types.is_datetime64_any_dtype(s):
            # Epoch years: a smooth, comparable encoding of time.
            v = s.astype('int64').to_numpy(dtype=np.float64)
            v[s.isna().to_numpy()] = np.nan
            v = v / (365.25 * 24 * 3600 * 1e9)
        else:
            if s.nunique(dropna=True) > max_categories:
                continue                       # identifier / free text
            codes = pd.Categorical(s).codes    # -1 for NaN
            n_cat = int(codes.max()) + 1
            if n_cat <= 1:
                continue
            onehot = np.zeros((len(s), n_cat), dtype=np.float64)
            valid = codes >= 0
            onehot[np.arange(len(s))[valid], codes[valid]] = 1.0
            blocks.append(onehot)
            continue
        v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
        finite = np.isfinite(v)
        if not finite.any():
            continue
        mu = v[finite].mean()
        sd = v[finite].std()
        v = (v - mu) / (sd if sd > 0 else 1.0)
        v[~finite] = 0.0
        blocks.append(v)
    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float64)
    return np.concatenate(blocks, axis=1)


def load_relbench(dataset_name: str, task_name: str, *,
                  root: str = 'row',
                  label_agg: str = 'last',
                  reverse_edges: bool = False,
                  max_categories: int = _MAX_CATEGORIES):
    """Build the homogeneous graph for a RelBench entity task.

    Args:
        dataset_name:  e.g. 'rel-f1'.
        task_name:     e.g. 'driver-top3'.
        root:          'row' (default) — one root per task row, all supervision
                       kept; 'entity' — one root per entity, labels aggregated.
        label_agg:     for root='entity': 'last' (label at the latest row in the
                       split) or 'any' (max over the split's rows).
        reverse_edges: also add PARENT -> CHILD arcs (raises K_out, and so eps).
        max_categories: one-hot width cap for categorical columns.

    Returns:
        (dataset, data) where `dataset` exposes num_features / num_classes and
        `data` is a PyG Data with x, y, edge_index, train_edge_index and
        train/val/test masks.
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    if root not in ('row', 'entity'):
        raise ValueError(f"root must be 'row' or 'entity', got {root!r}")
    if label_agg not in ('last', 'any'):
        raise ValueError(f"label_agg must be 'last' or 'any', got {label_agg!r}")

    ds = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)
    db = ds.get_db()

    tables = list(db.table_dict.items())
    split_dfs = {s: task.get_table(s, mask_input_cols=False).df
                 for s in ('train', 'val', 'test')}
    target_col, entity_col, time_col = (task.target_col, task.entity_col,
                                        task.time_col)

    # ── node index space: one block per table, then the task-row block ────────
    offsets, sizes = {}, {}
    n_nodes = 0
    for name, tbl in tables:
        offsets[name] = n_nodes
        sizes[name] = len(tbl.df)
        n_nodes += len(tbl.df)

    rows = pd.concat([split_dfs[s].assign(_split=i)
                      for i, s in enumerate(('train', 'val', 'test'))],
                     ignore_index=True)
    if root == 'entity':
        # Collapse to one labelled record per (entity, split).
        rows = rows.sort_values(time_col)
        agg = 'last' if label_agg == 'last' else 'max'
        rows = (rows.groupby([entity_col, '_split'], as_index=False)
                    .agg({target_col: agg, time_col: 'last'}))
    row_offset = n_nodes
    n_row_nodes = 0 if root == 'entity' else len(rows)
    n_nodes += n_row_nodes

    # ── features: block-diagonal per table, plus a node-type one-hot ──────────
    n_types = len(tables) + (1 if root == 'row' else 0)
    feat_blocks, widths = [], []
    for name, tbl in tables:
        skip = {tbl.pkey_col, *tbl.fkey_col_to_pkey_table}
        skip.discard(None)
        feat_blocks.append(_encode_table(tbl.df, skip, max_categories))
        widths.append(feat_blocks[-1].shape[1])
    if root == 'row':
        # A row node's own features: its timestamp only.  Anything else about
        # the row IS the label.
        feat_blocks.append(_encode_table(rows[[time_col]], set(), max_categories))
        widths.append(feat_blocks[-1].shape[1])

    total_width = sum(widths) + n_types
    x = np.zeros((n_nodes, total_width), dtype=np.float32)
    col = 0
    for t, block in enumerate(feat_blocks):
        start = offsets[tables[t][0]] if t < len(tables) else row_offset
        n = block.shape[0]
        if block.shape[1]:
            x[start:start + n, col:col + block.shape[1]] = block
        col += block.shape[1]
        x[start:start + n, sum(widths) + t] = 1.0        # node-type one-hot

    # ── node timestamps (NaT / static tables -> -inf, always available) ───────
    node_time = np.full(n_nodes, -np.inf, dtype=np.float64)
    for name, tbl in tables:
        if tbl.time_col is None:
            continue
        ts = tbl.df[tbl.time_col]
        v = ts.astype('int64').to_numpy(dtype=np.float64)
        v[ts.isna().to_numpy()] = -np.inf
        node_time[offsets[name]:offsets[name] + sizes[name]] = v
    if root == 'row':
        node_time[row_offset:row_offset + n_row_nodes] = (
            rows[time_col].astype('int64').to_numpy(dtype=np.float64))

    # ── edges: foreign keys, oriented child -> parent ─────────────────────────
    src_list, dst_list = [], []
    for name, tbl in tables:
        for fkey_col, parent in tbl.fkey_col_to_pkey_table.items():
            parent_df = db.table_dict[parent].df
            pkey = db.table_dict[parent].pkey_col
            pos = pd.Series(np.arange(len(parent_df)), index=parent_df[pkey])
            child_local = np.arange(len(tbl.df))
            parent_local = tbl.df[fkey_col].map(pos).to_numpy()
            ok = ~pd.isna(parent_local)
            src_list.append(child_local[ok] + offsets[name])
            dst_list.append(parent_local[ok].astype(np.int64) + offsets[parent])

    entity_table = task.entity_table
    entity_df = db.table_dict[entity_table].df
    entity_pkey = db.table_dict[entity_table].pkey_col
    entity_pos = pd.Series(np.arange(len(entity_df)), index=entity_df[entity_pkey])
    row_entity_local = rows[entity_col].map(entity_pos).to_numpy()

    if root == 'row':
        # entity -> row (parent -> child): the row node is a readout that its
        # entity feeds, so in-expansion from the root reaches the entity and,
        # one hop further, the entity's history.
        ok = ~pd.isna(row_entity_local)
        src_list.append(row_entity_local[ok].astype(np.int64) + offsets[entity_table])
        dst_list.append(np.arange(len(rows))[ok] + row_offset)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    if reverse_edges:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])

    # ── labels, masks, and the two time-filtered edge sets ────────────────────
    y = np.zeros(n_nodes, dtype=np.int64)
    masks = {s: np.zeros(n_nodes, dtype=bool) for s in ('train', 'val', 'test')}
    if root == 'row':
        node_of_row = np.arange(len(rows)) + row_offset
    else:
        keep = ~pd.isna(row_entity_local)
        node_of_row = np.where(
            keep, np.nan_to_num(row_entity_local, nan=0).astype(np.int64)
            + offsets[entity_table], -1)
    labels = rows[target_col].to_numpy()
    for i, split in enumerate(('train', 'val', 'test')):
        sel = (rows['_split'].to_numpy() == i) & (node_of_row >= 0)
        nodes = node_of_row[sel]
        y[nodes] = labels[sel].astype(np.int64)
        masks[split][nodes] = True

    train_end = float(rows.loc[rows['_split'] == 0, time_col]
                      .astype('int64').max())
    edge_ok_train = (node_time[src] <= train_end) & (node_time[dst] <= train_end)

    data = Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
    )
    data.train_edge_index = torch.from_numpy(
        np.stack([src[edge_ok_train], dst[edge_ok_train]])).long()
    data.node_time = torch.from_numpy(node_time)
    for split in ('train', 'val', 'test'):
        setattr(data, f'{split}_mask', torch.from_numpy(masks[split]))

    num_classes = int(y[masks['train'] | masks['val'] | masks['test']].max()) + 1
    dataset = _RelBenchDataset(data, total_width, num_classes,
                               task=task, task_type=str(task.task_type))
    return dataset, data


class _RelBenchDataset:
    """Minimal dataset wrapper exposing num_features / num_classes."""

    def __init__(self, data, num_features, num_classes, **extra):
        self._data = data
        self.num_features = num_features
        self.num_classes = num_classes
        for k, v in extra.items():
            setattr(self, k, v)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("single-graph dataset")
        return self._data


def parse_relbench_name(name: str):
    """'relbench:rel-f1/driver-top3' -> ('rel-f1', 'driver-top3').

    Also accepts the registered shorthands in src/datasets.py.
    """
    body = name.split(':', 1)[1]
    if '/' not in body:
        raise ValueError(
            f"expected relbench:<database>/<task>, got {name!r} "
            "(e.g. relbench:rel-f1/driver-top3)")
    db_name, task_name = body.split('/', 1)
    return db_name, task_name

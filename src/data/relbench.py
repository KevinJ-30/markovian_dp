"""
RelBench entity tasks as a homogeneous directed graph for SparseGNN.

Turns one (database, task) pair into the single `Data` object the engine
consumes, so no part of the mechanism, DP path, or accounting needs to know the
source was relational.  Built from the raw `Database` rather than RelBench's
`make_pkey_fkey_graph`, which produces a HeteroData of torch_frame TensorFrames
and pulls in a text embedder.

Nodes    one per table row, plus one "row node" per task row (root='row', the
         default) carrying a single label at a single timestamp.  root='entity'
         instead roots one node per entity with labels aggregated, which on
         rel-f1 discards ~93% of the supervision.
Edges    one arc per foreign key, child -> parent, plus entity -> row_node so
         that in-expansion from a row root reaches its entity at depth 1 and
         that entity's history at depth 2 (so r >= 2 is required to see any
         history).  reverse_edges=True mirrors every arc, which enriches
         neighbourhoods but raises K_out and hence epsilon.
Features per table: z-scored numerics, datetimes as z-scored epoch-years,
         one-hot categoricals under `max_categories`; free text and
         high-cardinality identifiers dropped.  Blocks are laid out
         block-diagonally with a node-type one-hot appended.
Time     `data.edge_index` is the graph at the TEST cutoff (used by evaluate);
         `data.train_edge_index` is the TRAIN cutoff, selected automatically by
         `src.sparse.run`.

Two caveats.  The cutoff is per split, not per row, so inside the training
window an early row may reach a later one — leakage between training examples
only, never into val/test.  And rel-f1 has 1353 training rows, enough to
validate the pipeline but too few for a meaningful epsilon; use a large task
(rel-hm user-churn, rel-stack user-badge) for DP numbers.

Task types.  RelBench ships REGRESSION, BINARY_CLASSIFICATION,
MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION, and LINK_PREDICTION
(`relbench.base.TaskType`).  The first three map onto this module's existing
`y` handling (BinaryGNNMechanism / GNNMechanism) with no change here.
REGRESSION targets are scaled by TRAIN-split statistics and the scale recorded
as `data.target_std` / `dataset.target_std`.  Metrics are reported in that
scaled space; multiply by target_std for the label's original units.
LINK_PREDICTION is not wired: it predicts a (src, dst) pair rather than a
single node's label, which does not fit the one-root-one-scalar-loss shape
every mechanism here assumes (see BaseMechanism.subgraph_loss) — supporting it
means deciding what SparseExpand roots on for a pair task, which changes the
accounting's shell structure, not just adding a new mechanism class.
"""

import zlib

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# Columns that are pure identifiers or free text carry no signal for us and
# would blow up the one-hot width.
_MAX_CATEGORIES = 32


def _hash_block(s: pd.Series, n_buckets: int) -> np.ndarray:
    """One-hot over a stable hash of the cell, for text / high-cardinality
    columns that would otherwise be dropped.  Mirrors torch_frame's
    HashTextEmbedder; crc32 rather than hash() because str hashing is salted
    per process, so hash() would give a different encoding every run.
    """
    if n_buckets <= 0:
        return np.zeros((len(s), 0), dtype=np.float64)   # disabled
    out = np.zeros((len(s), n_buckets), dtype=np.float64)
    for i, value in enumerate(s):
        if value is None or (np.isscalar(value) and pd.isna(value)):
            continue
        out[i, zlib.crc32(str(value).encode('utf-8')) % n_buckets] = 1.0
    return out


# >= 0.5 distinct values per row means an identifier, not a category. Hashing
# those hands the model a per-row code to memorize -- measured on rel-f1, doing
# so cost 20 AUROC points -- so they stay dropped.
_MAX_UNIQUE_RATIO = 0.5


def _encode_table(df: pd.DataFrame, skip_cols: set, max_categories: int,
                  n_hash: int = 16, stat_mask: np.ndarray | None = None
                  ) -> np.ndarray:
    """Dense float matrix for one table.

    stat_mask selects the rows whose statistics may set the scale (the
    pre-cutoff rows); None uses every row.
    """
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
            try:
                n_unique = s.nunique(dropna=True)
            except (TypeError, ValueError):
                # List/array cells (e.g. rel-amazon's product categories) are
                # unhashable for nunique(); hash their string form instead.
                blocks.append(_hash_block(s, n_hash))
                continue
            if n_unique > max_categories:
                if n_unique > _MAX_UNIQUE_RATIO * len(s):
                    continue                            # identifier
                blocks.append(_hash_block(s, n_hash))   # wide categorical
                continue
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
        ref = finite.ravel()
        if stat_mask is not None and (ref & stat_mask).any():
            ref = ref & stat_mask
        mu = v[ref, 0].mean()
        sd = v[ref, 0].std()
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
                  max_categories: int = _MAX_CATEGORIES,
                  n_hash: int = 0,
                  hubs: str = 'keep'):
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
        hubs:          'keep' | 'drop' | 'replicate'. Hub tables are non-entity
                       tables with no foreign keys. 'drop' removes their arcs;
                       'replicate' gives val/test their own copies.

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
    if hubs not in ('keep', 'drop', 'replicate'):
        raise ValueError(f"hubs must be 'keep', 'drop' or 'replicate', got {hubs!r}")

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

    # Feature scaling must not see past the training cutoff, so hoist it here.
    train_end = float(rows.loc[rows['_split'] == 0, time_col]
                      .astype('int64').max())

    # ── hubs: split owning every row, plus hub copies ────────────────────────
    entity_table = task.entity_table
    entity_df = db.table_dict[entity_table].df
    entity_pkey = db.table_dict[entity_table].pkey_col
    entity_pos = pd.Series(np.arange(len(entity_df)), index=entity_df[entity_pkey])
    row_entity_local = rows[entity_col].map(entity_pos).to_numpy()

    hub_tables: set = set()
    row_split = {}
    copy_offsets = {}
    if hubs != 'keep':
        hub_tables = {name for name, tbl in tables
                      if not tbl.fkey_col_to_pkey_table and name != entity_table}
        labelled = ~pd.isna(row_entity_local)
        ent_split = pd.DataFrame({
            'e': row_entity_local[labelled].astype(np.int64),
            's': rows['_split'].to_numpy()[labelled]}).drop_duplicates()
        if ent_split['e'].duplicated().any():
            raise ValueError(
                f"hubs={hubs!r} needs every {entity_table} entity in one split, "
                f"but {int(ent_split['e'].duplicated().sum())} appear in "
                f"several; use an entity-disjoint split instead")
        # Unlabelled entities join train.
        entity_split = np.zeros(len(entity_df), dtype=np.int64)
        entity_split[ent_split['e'].to_numpy()] = ent_split['s'].to_numpy()

        # Owning entity of every row, via foreign keys.
        owner = {entity_table: np.arange(len(entity_df))}
        changed = True
        while changed:
            changed = False
            for name, tbl in tables:
                if name in owner or name in hub_tables:
                    continue
                fkeys = sorted(tbl.fkey_col_to_pkey_table.items(),
                               key=lambda kv: kv[1] != entity_table)
                for fkey_col, parent in fkeys:
                    if parent not in owner:
                        continue
                    parent_df = db.table_dict[parent].df
                    pos = pd.Series(np.arange(len(parent_df)),
                                    index=parent_df[db.table_dict[parent].pkey_col])
                    local = tbl.df[fkey_col].map(pos).to_numpy()
                    ok = ~pd.isna(local)
                    own = np.full(len(tbl.df), -1, dtype=np.int64)
                    own[ok] = owner[parent][local[ok].astype(np.int64)]
                    owner[name] = own
                    changed = True
                    break
        for name, tbl in tables:
            if name in hub_tables:
                row_split[name] = np.zeros(len(tbl.df), dtype=np.int64)
            elif name in owner:
                own = owner[name]
                row_split[name] = np.where(
                    own >= 0, entity_split[np.clip(own, 0, None)], 0)
            else:
                raise ValueError(
                    f"hubs={hubs!r}: table {name!r} reaches neither the entity "
                    f"table {entity_table!r} nor a hub table; cannot assign a split")
        if hubs == 'replicate':
            for name in sorted(hub_tables):
                copy_offsets[name] = [n_nodes, n_nodes + sizes[name]]
                n_nodes += 2 * sizes[name]
        print(f"  hubs={hubs}: hub tables {sorted(hub_tables)} "
              f"({sum(sizes[h] for h in hub_tables)} rows"
              f"{', x3 copies' if hubs == 'replicate' else ', arcs dropped'})")

    # ── features: block-diagonal per table, plus a node-type one-hot ──────────
    n_types = len(tables) + (1 if root == 'row' else 0)
    feat_blocks, widths = [], []
    for name, tbl in tables:
        skip = {tbl.pkey_col, *tbl.fkey_col_to_pkey_table}
        skip.discard(None)
        if tbl.time_col is None:
            stat_mask = None                   # static table: no cutoff to apply
        else:
            ts = tbl.df[tbl.time_col]
            stat_mask = (ts.astype('int64').to_numpy() <= train_end)
            stat_mask |= ts.isna().to_numpy()  # undated rows are not "future"
        feat_blocks.append(_encode_table(
            tbl.df, skip, max_categories, n_hash=n_hash, stat_mask=stat_mask))
        widths.append(feat_blocks[-1].shape[1])
    if root == 'row':
        # A row node's own features: its timestamp only.  Anything else about
        # the row IS the label.
        feat_blocks.append(_encode_table(
            rows[[time_col]], set(), max_categories, n_hash=n_hash,
            stat_mask=(rows['_split'].to_numpy() == 0)))
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
    for name, (val_off, test_off) in copy_offsets.items():
        block = x[offsets[name]:offsets[name] + sizes[name]]
        x[val_off:val_off + sizes[name]] = block
        x[test_off:test_off + sizes[name]] = block

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
    for name, (val_off, test_off) in copy_offsets.items():
        block = node_time[offsets[name]:offsets[name] + sizes[name]]
        node_time[val_off:val_off + sizes[name]] = block
        node_time[test_off:test_off + sizes[name]] = block

    # ── edges: foreign keys, oriented child -> parent ─────────────────────────
    src_list, dst_list = [], []
    for name, tbl in tables:
        for fkey_col, parent in tbl.fkey_col_to_pkey_table.items():
            if hubs == 'drop' and parent in hub_tables:
                continue
            parent_df = db.table_dict[parent].df
            pkey = db.table_dict[parent].pkey_col
            pos = pd.Series(np.arange(len(parent_df)), index=parent_df[pkey])
            child_local = np.arange(len(tbl.df))
            parent_local = tbl.df[fkey_col].map(pos).to_numpy()
            ok = ~pd.isna(parent_local)
            src_list.append(child_local[ok] + offsets[name])
            if hubs == 'replicate' and parent in hub_tables:
                base = np.array([offsets[parent], *copy_offsets[parent]])
                dst_list.append(parent_local[ok].astype(np.int64)
                                + base[row_split[name][ok]])
            else:
                dst_list.append(parent_local[ok].astype(np.int64) + offsets[parent])

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
    from relbench.base import TaskType
    is_regression = task.task_type == TaskType.REGRESSION
    y = np.zeros(n_nodes, dtype=np.float64 if is_regression else np.int64)
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
        y[nodes] = labels[sel].astype(y.dtype)
        masks[split][nodes] = True

    # One node, one label: an entity in two splits keeps the LAST split's label
    # while the earlier split's mask stays set, i.e. trains on test labels.
    # rel-hm/item-sales hit this on 105,542 of 105,542 entities.
    overlap = (masks['train'] & (masks['val'] | masks['test'])).sum()
    if overlap:
        raise ValueError(
            f"root='entity' is unusable for {dataset_name}/{task_name}: "
            f"{int(overlap)} entities appear in more than one split, so their "
            f"label would be the later split's while train_mask stays set. "
            f"Use root='row'.")

    # Regression targets are SCALED by TRAIN-split statistics only (val/test
    # rows never inform the scale a model trains against).  Metrics stay in
    # this scaled space; target_std is recorded so the original units are a
    # single multiply away.
    #
    # Consequence, because it has bitten once: the target is NOT centred, so
    # the train mean is NOT 0 here.  "Predict the train mean" is therefore not
    # "predict 0", and run.py's trivial_baseline must (and now does) subtract
    # mean(y_train) explicitly rather than take |y_test|.  The network also has
    # to learn the intercept itself, with no output transform to absorb it.
    target_std = 1.0
    if is_regression:
        train_vals = y[masks['train']]
        target_std = float(train_vals.std())
        if target_std <= 0:
            target_std = 1.0
        y = y / target_std

    edge_ok_train = (node_time[src] <= train_end) & (node_time[dst] <= train_end)

    # Disjointness check: no arc may join two splits.
    node_split = np.zeros(n_nodes, dtype=np.int64)
    if hubs != 'keep':
        for name, _ in tables:
            node_split[offsets[name]:offsets[name] + sizes[name]] = row_split[name]
        for name, (val_off, test_off) in copy_offsets.items():
            node_split[val_off:val_off + sizes[name]] = 1
            node_split[test_off:test_off + sizes[name]] = 2
        if root == 'row':
            node_split[row_offset:row_offset + n_row_nodes] = rows['_split'].to_numpy()
        crossing = int((node_split[src] != node_split[dst]).sum())
        if crossing:
            raise AssertionError(
                f"hubs={hubs!r}: {crossing} arcs join different splits")
    # Nodes of the training-time graph, for delta.
    n_train_nodes = int(((node_split == 0) & (node_time <= train_end)).sum())

    # edge_index is unfiltered on purpose: get_db() defaults to
    # upto_test_timestamp=True, so no post-cutoff row exists to reach.
    # Filtering again would drop legitimate edges.  Checked by
    # scripts/relbench_leakage_check.py.
    data = Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y).float() if is_regression else torch.from_numpy(y),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
    )
    data.train_edge_index = torch.from_numpy(
        np.stack([src[edge_ok_train], dst[edge_ok_train]])).long()
    data.node_time = torch.from_numpy(node_time)
    data.node_split = torch.from_numpy(node_split)
    data.n_train_nodes = n_train_nodes
    if is_regression:
        data.target_std = target_std
    for split in ('train', 'val', 'test'):
        setattr(data, f'{split}_mask', torch.from_numpy(masks[split]))

    num_classes = (1 if is_regression else
                  int(y[masks['train'] | masks['val'] | masks['test']].max()) + 1)
    dataset = _RelBenchDataset(data, total_width, num_classes,
                               task=task, task_type=str(task.task_type),
                               target_std=target_std, hubs=hubs,
                               n_train_nodes=n_train_nodes)
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

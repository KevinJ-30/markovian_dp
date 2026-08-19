"""
Unified dataset loading for Planetoid, OGB, and PyG benchmark datasets.
"""

import os

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


SUPPORTED_DATASETS = {
    # Planetoid (transductive node classification)
    'cora': 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed',
    # OGB node classification
    'ogbn-products': 'ogbn-products',
    'ogbn-arxiv': 'ogbn-arxiv',
    # PyG Reddit (large transductive node classification)
    'reddit': 'Reddit',
    # Inductive node classification benchmarks
    'flickr': 'Flickr',
    'ppi': 'PPI',
    # Heterophilous graphs (GADBench: binary anomaly node classification)
    'tolokers': 'Tolokers',
    'questions': 'Questions',
    # OGB link property prediction
    'ogbl-collab': 'ogbl-collab',
    # GraphBench Bluesky (temporal-split node classification, inductive)
    'bluesky': 'Bluesky',
    # RelBench entity tasks (temporal, natively inductive).  Shorthands for the
    # generic form `relbench:<database>/<task>`, which accepts any RelBench pair.
    'relbench-f1-top3': 'relbench:rel-f1/driver-top3',
    'relbench-f1-dnf': 'relbench:rel-f1/driver-dnf',
}


def _load_ogb_node(name):
    """Load an OGB node-property dataset, returning (dataset, data) with bool masks."""
    from ogb.nodeproppred import PygNodePropPredDataset
    # PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks
    # OGB's internal loading of PyG objects. Allow unsafe load for OGB.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
    root = os.environ.get('OGB_DATA_ROOT', f'data/{name}')
    try:
        dataset = PygNodePropPredDataset(name=name, root=root)
    finally:
        torch.load = _orig_load
    data = dataset[0]
    # OGB node labels are (N, 1) — squeeze to (N,)
    data.y = data.y.squeeze(-1)
    split_idx = dataset.get_idx_split()
    num_nodes = data.x.size(0)
    for split_name in ['train', 'val', 'test']:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[split_idx[split_name if split_name != 'val' else 'valid']] = True
        setattr(data, f'{split_name}_mask', mask)
    return dataset, data


def _load_ogbl_collab():
    """Load ogbl-collab with positive/negative edge splits attached to data."""
    from ogb.linkproppred import PygLinkPropPredDataset
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
    root = os.environ.get('OGB_DATA_ROOT', 'data/ogbl-collab')
    try:
        dataset = PygLinkPropPredDataset(name='ogbl-collab', root=root)
        # PyTorch 2.6+ default weights_only=True also affects OGB's
        # split files (train.pt/valid.pt/test.pt), so keep the override
        # active while loading split edges as well.
        split_edge = dataset.get_edge_split()
    finally:
        torch.load = _orig_load
    data = dataset[0]
    # ogbl-collab edge tensors are shape [E, 2] — store as (2, E) for consistency
    # with edge_index conventions used elsewhere.
    data.train_pos_edge = split_edge['train']['edge'].t().contiguous()
    data.valid_pos_edge = split_edge['valid']['edge'].t().contiguous()
    data.valid_neg_edge = split_edge['valid']['edge_neg'].t().contiguous()
    data.test_pos_edge = split_edge['test']['edge'].t().contiguous()
    data.test_neg_edge = split_edge['test']['edge_neg'].t().contiguous()
    # All nodes are potentially "active" for link prediction (their embeddings
    # contribute to any incident edge). The trainer's active_mask machinery
    # then composes naturally with bin assignment and Poisson subsampling.
    num_nodes = data.x.size(0)
    all_true = torch.ones(num_nodes, dtype=torch.bool)
    data.train_mask = all_true
    data.val_mask = all_true.clone()
    data.test_mask = all_true.clone()
    return dataset, data


class _SimpleDataset:
    """Minimal dataset wrapper exposing num_features / num_classes.

    Mirrors the shape of a PyG/OGB dataset object for graphs we assemble
    ourselves (PPI's disjoint union, RelBench's relational graph) rather than
    load through torch_geometric.datasets.
    """

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


def _load_ppi():
    """PPI as one disjoint-union graph with split masks by source graph.

    PPI ships 24 protein-protein interaction graphs already partitioned into
    20 train / 2 val / 2 test.  Concatenating them (offsetting each graph's
    edge_index by the running node count) gives a single Data object that the
    SparseGNN engine consumes unchanged, while keeping the splits genuinely
    INDUCTIVE: the components are disconnected, so SparseExpand from a training
    root provably cannot reach a val/test node no matter how large r is.  That
    makes PPI the cleanest node-DP story available — no train-induced-subgraph
    surgery (`--inductive`) is needed or has any effect.

    Labels are 121-way MULTILABEL, so pair this with `--model multilabel_gnn`;
    `num_classes` is the number of label columns and metrics are micro-F1.
    """
    from torch_geometric.datasets import PPI

    root = os.environ.get('PPI_DATA_ROOT', 'data/PPI')
    xs, ys, edge_indices, split_of_node = [], [], [], []
    offset = 0
    for split in ('train', 'val', 'test'):
        for graph in PPI(root=root, split=split):
            xs.append(graph.x)
            ys.append(graph.y)
            edge_indices.append(graph.edge_index + offset)
            n = int(graph.num_nodes)
            split_of_node.append(torch.full((n,), {'train': 0, 'val': 1,
                                                   'test': 2}[split],
                                            dtype=torch.long))
            offset += n

    data = Data(x=torch.cat(xs, dim=0),
                y=torch.cat(ys, dim=0).float(),
                edge_index=torch.cat(edge_indices, dim=1))
    where = torch.cat(split_of_node)
    for i, split in enumerate(('train', 'val', 'test')):
        setattr(data, f'{split}_mask', where == i)

    num_features = int(data.x.size(1))
    num_classes = int(data.y.size(1))
    return _SimpleDataset(data, num_features, num_classes,
                          multilabel=True), data


class _BlueskyDataset:
    """Minimal dataset wrapper exposing num_features / num_classes for make_model.

    Mirrors the shape of PyG Dataset / OGB dataset objects used elsewhere in
    this module without requiring a full Dataset subclass — Bluesky is loaded
    from raw files (or a vendored package), not via torch_geometric.datasets.
    """

    def __init__(self, data, num_features, num_classes):
        self._data = data
        self.num_features = num_features
        self.num_classes = num_classes

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("Bluesky is a single-graph dataset")
        return self._data


def _read_bluesky_raw(root):
    """Read raw Bluesky data from disk and return the fields needed for the
    temporal split. Stubbed — plug in once we know GraphBench's exact format.

    Expected return: a dict with the following keys (torch.Tensor values
    unless noted):
      - 'x':           [N, F] float, node features
      - 'y':           [N]    long,  node labels (for node classification)
      - 'edge_index':  [2, E] long,  full directed edge tensor
      - 'edge_time':   [E]    numeric, per-edge timestamp (any monotone type)
      - 'node_time':   [N]    numeric, OPTIONAL — per-node first-appearance
                              timestamp. If absent, train/val/test masks fall
                              back to a fixed-seed random split.
      - 'num_classes': int,   number of classification classes

    GraphBench distributes datasets as PyG `InMemoryDataset` objects with HDF5
    raw files; the actual reader will likely be a single torch.load() (if
    they ship a .pt) or an h5py read. Plug it in here when the data is in
    hand at `root`.
    """
    raise NotImplementedError(
        "Bluesky raw reader is stubbed. Set BLUESKY_DATA_ROOT and implement "
        "_read_bluesky_raw() in src/datasets.py — see its docstring for the "
        "expected return shape. Source: https://zenodo.org/records/11082879 "
        "and https://graphbench.github.io/website/"
    )


def _temporal_split(raw, t_train=None, t_val=None):
    """Build train/eval edge_index views and node masks from raw temporal data.

    Pure function — extracted so it can be unit-tested by monkeypatching
    `_read_bluesky_raw` without needing the actual dataset on disk.

    Args:
        raw: dict from `_read_bluesky_raw`. See its docstring for keys.
        t_train: training cutoff timestamp. Edges with edge_time <= t_train
            form the training graph view. If None, defaults to the 60th
            percentile of edge_time.
        t_val: val cutoff. Edges with edge_time <= t_val form the eval graph
            view. If None, defaults to the 80th percentile of edge_time.

    Returns:
        (data, num_features, num_classes) where `data` is a PyG Data object
        with:
          - x, y: from raw
          - edge_index: edges with edge_time <= t_train (training view)
          - eval_edge_index: edges with edge_time <= t_val (eval view; the
            extra t_val..t_test edges only matter if we later support link
            prediction, where they'd be the val edges to score)
          - train_mask, val_mask, test_mask: from node_time if present, else
            a fixed-seed random split (logged at load time)
    """
    x = raw['x']
    y = raw['y']
    edge_index = raw['edge_index']
    edge_time = raw['edge_time']
    num_classes = raw['num_classes']
    num_nodes = x.size(0)

    edge_time_t = torch.as_tensor(edge_time)
    if t_train is None:
        t_train = torch.quantile(edge_time_t.float(), 0.60).item()
    if t_val is None:
        t_val = torch.quantile(edge_time_t.float(), 0.80).item()
    if t_val < t_train:
        raise ValueError(
            f"BLUESKY_T_VAL ({t_val}) must be >= BLUESKY_T_TRAIN ({t_train})"
        )

    train_edge_mask = edge_time_t <= t_train
    eval_edge_mask = edge_time_t <= t_val

    data = Data(x=x, y=y)
    data.edge_index = edge_index[:, train_edge_mask]
    data.eval_edge_index = edge_index[:, eval_edge_mask]
    data.num_nodes = num_nodes

    node_time = raw.get('node_time')
    if node_time is not None:
        node_time_t = torch.as_tensor(node_time)
        train_mask = node_time_t <= t_train
        val_mask = (node_time_t > t_train) & (node_time_t <= t_val)
        test_mask = node_time_t > t_val
        print(f"  bluesky: temporal node split via node_time "
              f"(train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
              f"test={int(test_mask.sum())})")
    else:
        # Fallback: random split with a fixed seed. Edge-level inductive split
        # still holds (training cannot see post-cutoff edges), but node-level
        # train/val/test sets are not temporally separated.
        rng = torch.Generator()
        rng.manual_seed(0)
        perm = torch.randperm(num_nodes, generator=rng)
        n_train = int(0.60 * num_nodes)
        n_val = int(0.20 * num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True
        print(f"  bluesky: no node_time in raw data, using fixed-seed random "
              f"split (train={n_train}, val={n_val}, test={num_nodes - n_train - n_val}). "
              f"Edge-level inductive split still holds via edge_time filter.")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"  bluesky: t_train={t_train:.4g}, t_val={t_val:.4g}; "
          f"train edges={int(train_edge_mask.sum())}, "
          f"eval edges={int(eval_edge_mask.sum())} (full={edge_index.size(1)})")

    num_features = x.size(1)
    return data, num_features, num_classes


def _load_bluesky():
    """Load GraphBench Bluesky as a temporal-split (inductive) graph.

    Reads raw data from BLUESKY_DATA_ROOT (default 'data/bluesky') via the
    stubbed `_read_bluesky_raw`, then applies `_temporal_split` to produce
    the train-time `edge_index` and eval-time `eval_edge_index` views.
    Cutoffs default to the 60th/80th percentile of edge_time and can be
    overridden via BLUESKY_T_TRAIN / BLUESKY_T_VAL env vars.
    """
    root = os.environ.get('BLUESKY_DATA_ROOT', 'data/bluesky')
    raw = _read_bluesky_raw(root)

    t_train_env = os.environ.get('BLUESKY_T_TRAIN')
    t_val_env = os.environ.get('BLUESKY_T_VAL')
    t_train = float(t_train_env) if t_train_env is not None else None
    t_val = float(t_val_env) if t_val_env is not None else None

    data, num_features, num_classes = _temporal_split(raw, t_train, t_val)
    dataset = _BlueskyDataset(data, num_features, num_classes)
    return dataset, data


def _load_heterophilous(canonical, split_idx=0):
    """Load a HeterophilousGraphDataset (GADBench GAD datasets) with 1-D masks.

    These datasets ship 10 pre-defined splits: train/val/test_mask each have shape
    [N, num_splits]. We select column `split_idx` and expose the usual 1-D bool
    masks so the rest of the pipeline is unchanged. Labels are binary (anomaly=1).
    """
    from torch_geometric.datasets import HeterophilousGraphDataset
    dataset = HeterophilousGraphDataset(root=f'/tmp/{canonical}', name=canonical)
    data = dataset[0]
    num_splits = data.train_mask.size(1)
    if not (0 <= split_idx < num_splits):
        raise ValueError(f"split_idx {split_idx} out of range [0, {num_splits}) "
                         f"for {canonical}")
    for split in ('train', 'val', 'test'):
        setattr(data, f'{split}_mask', getattr(data, f'{split}_mask')[:, split_idx])
    return dataset, data


def load_dataset(name, device='cpu', split_idx=0, **relbench_kwargs):
    """
    Load a dataset by name.

    Args:
        name: One of the keys in SUPPORTED_DATASETS (case-insensitive), or a
            RelBench pair written as 'relbench:<database>/<task>'.
        device: Device to move data to.
        split_idx: For datasets with multiple predefined splits (Tolokers,
            Questions), which split column to use. Ignored otherwise.
        **relbench_kwargs: forwarded to src.sparse.relbench_data.load_relbench
            (root, label_agg, reverse_edges, max_categories) for RelBench names.

    Returns:
        (dataset, data) tuple.
    """
    key = name.lower()
    # RelBench pairs may be named directly as relbench:<database>/<task>, or via
    # one of the shorthands in SUPPORTED_DATASETS.
    spec = SUPPORTED_DATASETS.get(key, name)
    if isinstance(spec, str) and spec.startswith('relbench:'):
        from src.sparse.relbench_data import load_relbench, parse_relbench_name
        db_name, task_name = parse_relbench_name(spec)
        dataset, data = load_relbench(db_name, task_name, **relbench_kwargs)
        data = data.to(device)
        return dataset, data

    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: "
                         f"{list(SUPPORTED_DATASETS.keys())} or "
                         f"relbench:<database>/<task>")

    if key in ('tolokers', 'questions'):
        dataset, data = _load_heterophilous(SUPPORTED_DATASETS[key], split_idx=split_idx)
        data = data.to(device)
        return dataset, data

    if key in ('ogbn-products', 'ogbn-arxiv'):
        dataset, data = _load_ogb_node(key)
        data = data.to(device)
        return dataset, data

    if key == 'reddit':
        from torch_geometric.datasets import Reddit
        root = os.environ.get('REDDIT_DATA_ROOT', 'data/Reddit')
        dataset = Reddit(root=root)
        data = dataset[0].to(device)
        return dataset, data

    if key == 'flickr':
        # Single graph with train/val/test masks (GraphSAINT's inductive
        # benchmark); pass --inductive to train on the train-induced subgraph.
        from torch_geometric.datasets import Flickr
        root = os.environ.get('FLICKR_DATA_ROOT', 'data/Flickr')
        dataset = Flickr(root=root)
        data = dataset[0].to(device)
        return dataset, data

    if key == 'ppi':
        dataset, data = _load_ppi()
        data = data.to(device)
        return dataset, data

    if key == 'ogbl-collab':
        dataset, data = _load_ogbl_collab()
        data = data.to(device)
        return dataset, data

    if key == 'bluesky':
        dataset, data = _load_bluesky()
        data = data.to(device)
        return dataset, data

    canonical = SUPPORTED_DATASETS[key]
    dataset = Planetoid(root=f'/tmp/{canonical}', name=canonical)
    data = dataset[0].to(device)
    return dataset, data

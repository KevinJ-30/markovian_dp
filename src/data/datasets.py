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
    'cora-ml': 'Cora-ML',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed',
    # OGB node classification
    'ogbn-products': 'ogbn-products',
    'ogbn-arxiv': 'ogbn-arxiv',
    # PyG Reddit (large transductive node classification)
    'reddit': 'Reddit',
    # Inductive node classification benchmarks
    'flickr': 'Flickr',
    # GraphLand scalar regression, fixed random 80/10/10 node splits.
    'hm-prices': 'hm-prices',
    'avazu-ctr': 'avazu-ctr',
    # GAP/ProGAP's Facebook: the UIllinois20 FB100 network, year label filtered
    # to classes with >=1000 nodes.  For head-to-head comparison with those
    # papers on a dataset where the graph actually carries signal.
    'facebook': 'Facebook',
    # Provenance-specific domain-disjoint node-classification benchmarks.
    'twitch-explicit': 'Twitch-Explicit',
    'facebook100': 'Facebook100',
    'mag-countries': 'MAG-Countries',
    # GraphSAINT benchmark graphs (Zeng et al., ICLR 2020), loaded from the
    # authors' released files under their inductive protocol.  Shorthands for
    # the generic form `graphsaint:<name>`.  NOTE these are NOT the same graphs
    # as the bare 'reddit' / 'flickr' keys above, which are PyG's
    # versions -- GraphSAINT's Reddit is ~5x sparser and splits differ.
    'ppi-large': 'graphsaint:ppi-large',
    'saint-flickr': 'graphsaint:flickr',
    'saint-reddit': 'graphsaint:reddit',
    'saint-yelp': 'graphsaint:yelp',
    'saint-amazon': 'graphsaint:amazon',
}


def _validated_ogb_node_split_indices(name, split_idx, num_nodes):
    """Validate OGB's complete, disjoint official node split."""
    if not isinstance(split_idx, dict):
        raise ValueError(f"OGB dataset {name} has an invalid official split: expected a mapping")
    indices, membership = {}, torch.zeros(num_nodes, dtype=torch.bool)
    for local_name, ogb_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        if ogb_name not in split_idx:
            raise ValueError(f"OGB dataset {name} has an invalid official split: missing {ogb_name}")
        index = split_idx[ogb_name]
        if not isinstance(index, torch.Tensor) or index.ndim != 1:
            raise ValueError(f"OGB dataset {name} has an invalid official split: {ogb_name} must be one-dimensional")
        if index.dtype.is_floating_point or index.dtype == torch.bool:
            raise ValueError(f"OGB dataset {name} has an invalid official split: {ogb_name} must contain integer node IDs")
        index = index.detach().cpu().to(torch.long)
        if torch.any(index < 0) or torch.any(index >= num_nodes):
            raise ValueError(f"OGB dataset {name} has an invalid official split: {ogb_name} contains out-of-range node IDs")
        if torch.any(membership[index]):
            raise ValueError(f"OGB dataset {name} has an invalid official split: node IDs overlap")
        membership[index] = True
        indices[local_name] = index
    if not torch.all(membership):
        raise ValueError(f"OGB dataset {name} has an invalid official split: node IDs do not cover every node")
    return indices


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
    indices = _validated_ogb_node_split_indices(name, dataset.get_idx_split(), data.x.size(0))
    for split_name, index in indices.items():
        mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        mask[index] = True
        setattr(data, f'{split_name}_mask', mask)
    return dataset, data


class _SimpleDataset:
    """Minimal dataset wrapper exposing num_features / num_classes.

    Mirrors the shape of a PyG/OGB dataset object for graphs we assemble
    ourselves (GraphSAINT releases and domain-disjoint graphs) rather than
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


def _load_graphland(name, root=None):
    """GraphLand regression with a fixed seed-0 random 80/10/10 split.

    Reuse PyG's full-graph feature preprocessing, but not its published split
    or target scaling. Targets are standardized using only our training nodes.
    These two releases require a finite target at every node so native
    graph-disjoint training and its supervised population remain well-defined.
    """
    from torch_geometric.datasets import GraphLandDataset

    root = root or os.environ.get('GRAPHLAND_DATA_ROOT', 'data/graphland')
    source = GraphLandDataset(
        root=os.fspath(root), name=name, split='RH',
        regression_targets_transform=None)
    data = source[0]
    targets = data.y.reshape(-1).float()
    n = int(data.num_nodes)
    if targets.numel() != n or not torch.isfinite(targets).all():
        raise ValueError(
            f"GraphLand {name} requires one finite regression target per node")
    n_train, n_val = 8 * n // 10, n // 10
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError(
            f"GraphLand {name} needs at least 10 nodes for an 80/10/10 split")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    for role, indices in (
            ('train', order[:n_train]),
            ('val', order[n_train:n_train + n_val]),
            ('test', order[n_train + n_val:])):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[indices] = True
        setattr(data, f'{role}_mask', mask)

    train_targets = targets[data.train_mask].double()
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std(unbiased=False))
    if target_std == 0:
        target_std = 1.0
    data.y = ((targets.double() - target_mean) / target_std).float()
    return _SimpleDataset(
        data, data.num_node_features, 1,
        task_type='REGRESSION', primary_metric='r2', multilabel=False,
        split_strategy='native', split_seed=0,
        target_mean=target_mean, target_std=target_std), data


GRAPHSAINT_DATASETS = {
    # name -> (is_multilabel, human note).  Statistics are GraphSAINT Table 1,
    # reproduced exactly by this loader (see _load_graphsaint's docstring).
    'ppi-large': (True,  '56,944 nodes / 818,716 edges / 50 feat / 121 labels'),
    'flickr':    (False, '89,250 nodes / 899,756 edges / 500 feat / 7 classes'),
    'reddit':    (False, '232,965 nodes / 11,606,919 edges / 602 feat / 41 classes'),
    'yelp':      (True,  '716,847 nodes / 6,977,410 edges / 300 feat / 100 labels'),
    'amazon':    (True,  '1,598,960 nodes / 132,169,734 edges / 200 feat / 107 labels'),
}


def _load_graphsaint(name, root=None):
    """A GraphSAINT benchmark graph, from the authors' own released files.

    Named `graphsaint:<name>` with <name> in GRAPHSAINT_DATASETS.  These are the
    graphs and splits of Zeng et al., ICLR 2020 (arXiv:1907.04931), used under
    their setting: "Experiments are under the inductive, supervised learning
    setting", where test nodes are invisible during training.

    Why not PyG's versions.  They are not the same graphs.  PyG's `Reddit` has
    57,307,946 undirected edges; GraphSAINT's has 11,606,919 -- about 5x
    sparser -- and the splits differ too (153,932/23,699/55,334 here against
    PyG's 153,431/23,831/55,703).  Loading the authors' files is what makes our
    numbers comparable to their published baselines.

    Directory layout, one per dataset, as distributed:

        <root>/<name>/adj_full.npz     scipy CSR, ROW-NORMALIZED, self-loops
                     /adj_train.npz    same, restricted to training nodes
                     /feats.npy        [N, F] float
                     /role.json        {"tr": [...], "va": [...], "te": [...]}
                     /class_map.json   {node_id: label | multi-hot list}
                     /labels.npy       [N, C] float, amazon only (faster than
                                       its 523 MB class_map.json)

    Three preprocessing steps are REQUIRED and are what reconcile the files with
    the paper's Table 1 and training protocol:

      * BINARIZE.  The stored matrices hold 1/deg, not 1 -- they are the
        row-normalized adjacency, which is also why `A != A.T` before
        binarizing.  After binarizing they are exactly symmetric.
      * DROP SELF-LOOPS.  PPI-large carries 25,084 of them.  The paper's "Edges"
        column counts them: 793,632 undirected edges + 25,084 self-loops =
        818,716, its stated figure.  Reddit has none, and its 23,213,838 arcs
        are exactly 2 x 11,606,919.  We drop them because the accounting counts
        paths in a simple graph and degree capping would otherwise spend a node's
        budget on an arc to itself.
      * STANDARDIZE FEATURES.  Match GraphSAINT's loader by fitting a
        StandardScaler on nodes present in the training adjacency and applying
        it to every split.

    `data.edge_index` is the full graph and `data.train_edge_index` the
    train-induced one. `src.experiments.sparse` uses the latter for training and passes
    the former separately for evaluation.
    """
    import json
    import numpy as np
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler

    if name not in GRAPHSAINT_DATASETS:
        raise ValueError(
            f"unknown GraphSAINT dataset {name!r}; expected one of "
            f"{sorted(GRAPHSAINT_DATASETS)}")
    multilabel, _note = GRAPHSAINT_DATASETS[name]

    root = root or os.environ.get('GRAPHSAINT_DATA_ROOT', 'data/graphsaint')
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} not found. Download the GraphSAINT data (Google Drive link in "
            f"github.com/GraphSAINT/GraphSAINT) and extract so that "
            f"{os.path.join(d, 'adj_full.npz')} exists, or set "
            f"GRAPHSAINT_DATA_ROOT to the directory holding <name>/ folders.")

    def _arcs(path):
        """CSR -> edge_index plus nodes present before self-loop removal."""
        a = sp.load_npz(path).tocoo()
        active_nodes = np.unique(a.row)
        keep = a.row != a.col                      # drop self-loops
        row, col = a.row[keep], a.col[keep]
        edge_index = torch.from_numpy(np.stack([row, col])).long()
        return edge_index, active_nodes

    edge_index, _ = _arcs(os.path.join(d, 'adj_full.npz'))
    train_edge_index, train_nodes = _arcs(os.path.join(d, 'adj_train.npz'))

    features = np.load(os.path.join(d, 'feats.npy'))
    scaler = StandardScaler(copy=False).fit(features[train_nodes])
    x = torch.from_numpy(scaler.transform(features)).float()
    n = int(x.size(0))

    role = json.load(open(os.path.join(d, 'role.json')))
    masks = {}
    for key, split in (('tr', 'train'), ('va', 'val'), ('te', 'test')):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.tensor(role[key], dtype=torch.long)] = True
        masks[split] = m

    labels_npy = os.path.join(d, 'labels.npy')
    cache = os.path.join(d, '_labels_cache.pt')
    if multilabel and os.path.exists(labels_npy):
        # amazon ships this alongside a 523 MB class_map.json; same content.
        y = torch.from_numpy(np.load(labels_npy)).float()
    elif os.path.exists(cache):
        y = torch.load(cache)
    else:
        # yelp's class_map.json is 367 MB and amazon's 523 MB; a sweep invokes
        # run.py once per cell, so parse once and cache the tensor beside it.
        cm = json.load(open(os.path.join(d, 'class_map.json')))
        first = next(iter(cm.values()))
        if multilabel:
            C = len(first)
            y = torch.zeros(n, C, dtype=torch.float)
            for k, v in cm.items():
                y[int(k)] = torch.tensor(v, dtype=torch.float)
        else:
            y = torch.zeros(n, dtype=torch.long)
            for k, v in cm.items():
                y[int(k)] = int(v)
        try:
            torch.save(y, cache)
        except OSError:
            pass          # read-only data dir is fine, just slower next time

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_edge_index = train_edge_index
    for split, m in masks.items():
        setattr(data, f'{split}_mask', m)

    num_classes = int(y.size(1)) if multilabel else int(y.max()) + 1
    return _SimpleDataset(data, int(x.size(1)), num_classes,
                          multilabel=multilabel), data


def _load_cora_ml():
    """Load the Cora-ML sparse graph distributed with DPAR.

    The upstream archive is intentionally used rather than silently substituting
    Planetoid Cora, which is a different graph and feature matrix.
    """
    import numpy as np
    import scipy.sparse as sp
    from torch_geometric.data import download_url

    root = os.environ.get('CORA_ML_DATA_ROOT', 'data/cora_ml')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, 'cora_ml.npz')
    if not os.path.exists(path):
        download_url('https://raw.githubusercontent.com/Emory-AIMS/DPAR/'
                     'b31f371522af8a5142f4c6b34f712cff30623b31/data/cora_ml.npz', root)
    with np.load(path, allow_pickle=True) as archive:
        raw = dict(archive)

    def csr(prefix):
        for separator in ('.', '_'):
            key = f'{prefix}{separator}data'
            if key in raw:
                return sp.csr_matrix((raw[key], raw[f'{prefix}{separator}indices'],
                                      raw[f'{prefix}{separator}indptr']),
                                     shape=raw[f'{prefix}{separator}shape'])
        raise KeyError(f'Cora-ML archive lacks {prefix} CSR fields')

    adjacency = csr('adj_matrix') if 'adj_matrix.data' in raw or 'adj_matrix_data' in raw else csr('adj')
    attributes = csr('attr_matrix') if 'attr_matrix.data' in raw or 'attr_matrix_data' in raw else csr('attr')
    labels = torch.from_numpy(raw['labels']).long()
    coo = adjacency.tocoo()
    data = Data(x=torch.from_numpy(attributes.toarray()).float(), y=labels,
                edge_index=torch.from_numpy(np.vstack((coo.row, coo.col))).long())
    return _SimpleDataset(data, int(data.x.size(1)), int(labels.max()) + 1), data


def _load_facebook(name='UIllinois20', target='year', min_count=1000,
                   val_ratio=0.10, test_ratio=0.15, seed=0):
    """GAP/ProGAP's Facebook: one FB100 university network, node classification.

    Replicates core/datasets/facebook.py + its pre_transform in the ProGAP repo
    (github.com/sisaman/ProGAP):

      * download <name>.mat from sisaman/pyg-datasets (features in `local_info`,
        adjacency in `A`);
      * label y = `target` column (default 'year'); one-hot the other five
        categorical attributes as features, treating value 0 as missing;
      * split 75/val/test at random over ALL nodes, THEN keep only classes with
        >= `min_count` members and drop the rest (FilterClassByCount), which is
        what reduces UIllinois20's years to ~6 classes / ~26k nodes;
      * remove self-loops and isolated nodes.

    The D=100 degree bound is NOT applied here — it is a training-time cap,
    matched by `--K_out 100` in the SparseGNN pipeline.

    The split is random (their protocol), not their exact split; they report a
    mean over random splits, so a fixed-seed 75/10/15 split is comparable.
    """
    import ssl
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    from torch_geometric.utils import subgraph
    from torch_geometric.data import download_url

    targets = ['status', 'gender', 'major', 'minor', 'housing', 'year']
    root = os.environ.get('FACEBOOK_DATA_ROOT', 'data/facebook100')
    os.makedirs(root, exist_ok=True)
    mat_path = os.path.join(root, f'{name}.mat')
    if not os.path.exists(mat_path):
        ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            download_url('https://github.com/sisaman/pyg-datasets/raw/main/'
                         f'datasets/facebook100/{name}.mat', root)
        finally:
            ssl._create_default_https_context = ctx

    mat = loadmat(mat_path)
    feats = pd.DataFrame(mat['local_info'][:, :-1], columns=targets)

    # label: LabelEncoder == sorted-unique codes; shift to 0-based if 0 present.
    y_codes = pd.Categorical(feats[target]).codes.astype(np.int64)
    y = torch.from_numpy(y_codes)
    if (feats[target] == 0).any():
        y = y - 1

    # features: one-hot the other attributes, value 0 -> missing (no column).
    x_df = feats.drop(columns=target).replace({0: None})
    x = torch.tensor(pd.get_dummies(x_df).values, dtype=torch.float)

    from torch_geometric.utils import from_scipy_sparse_matrix
    edge_index = from_scipy_sparse_matrix(mat['A'])[0]

    # drop unlabeled, relabel
    keep = y >= 0
    edge_index, _ = subgraph(keep, edge_index, relabel_nodes=True,
                             num_nodes=len(y))
    x, y = x[keep], y[keep]
    n = int(y.numel())

    # 75/10/15 random split over all nodes (their RandomNodeSplit order: before
    # the class filter).
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val, n_test = int(val_ratio * n), int(test_ratio * n)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[perm[:n_test]] = True
    val_mask[perm[n_test:n_test + n_val]] = True
    train_mask[perm[n_test + n_val:]] = True

    # FilterClassByCount(min_count, remove_unlabeled=True): keep classes with
    # >= min_count members, drop the rest, relabel classes to 0..C-1.
    onehot = torch.nn.functional.one_hot(y)
    counts = onehot.sum(0)
    onehot = onehot[:, counts >= min_count]
    row_keep = onehot.sum(1).bool()
    y = onehot.argmax(1)
    idx = torch.where(row_keep)[0]
    edge_index, _ = subgraph(row_keep, edge_index, relabel_nodes=True,
                             num_nodes=n)
    x, y = x[row_keep], y[row_keep]
    train_mask, val_mask, test_mask = (train_mask[row_keep], val_mask[row_keep],
                                       test_mask[row_keep])

    # remove self-loops, then isolated nodes
    sl = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, sl]
    deg = torch.zeros(int(y.numel()), dtype=torch.long)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    not_iso = deg > 0
    edge_index, _ = subgraph(not_iso, edge_index, relabel_nodes=True,
                             num_nodes=int(y.numel()))
    x, y = x[not_iso], y[not_iso]
    train_mask, val_mask, test_mask = (train_mask[not_iso], val_mask[not_iso],
                                       test_mask[not_iso])

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    num_classes = int(y.max()) + 1
    return _SimpleDataset(data, int(x.size(1)), num_classes), data


def load_dataset(name, device='cpu', domain_split=None, *, root=None):
    """
    Load a dataset by name.

    Args:
        name: One of the keys in SUPPORTED_DATASETS (case-insensitive), or
            'graphsaint:<name>' for a registered GraphSAINT release.
        device: Device to move data to.
        domain_split: Optional train/validation/test domain selection for the
            domain-disjoint datasets. Supplying any role requires all three.
        root: Optional cache directory for domain, GraphSAINT, and GraphLand datasets.

    Returns:
        (dataset, data) tuple.
    """
    key = name.lower()
    spec = SUPPORTED_DATASETS.get(key, name)
    if key in ('twitch-explicit', 'facebook100', 'mag-countries'):
        from src.data.domain_datasets import load_domain_dataset
        data, metadata = load_domain_dataset(
            key, domain_split=domain_split, root=root)
        dataset = _SimpleDataset(data, **metadata)
        return dataset, data.to(device)
    if domain_split is not None:
        raise ValueError(
            f"domain_split is only supported for domain datasets, not '{name}'")
    # GraphSAINT graphs are named graphsaint:<name>.  Deliberately NOT folded
    # into the bare 'reddit'/'flickr' keys: those are the PyG versions, which are
    # different graphs with different splits (see _load_graphsaint).
    if isinstance(spec, str) and spec.startswith('graphsaint:'):
        dataset, data = _load_graphsaint(
            spec.split(':', 1)[1], root=root)
        return dataset, data.to(device)

    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: "
                         f"{list(SUPPORTED_DATASETS.keys())} or "
                         f"graphsaint:<name>")
    if key in ('hm-prices', 'avazu-ctr'):
        dataset, data = _load_graphland(key, root=root)
        return dataset, data.to(device)
    if key == 'cora-ml':
        dataset, data = _load_cora_ml()
        return dataset, data.to(device)

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
        # Single graph with train/val/test masks; src.experiments.sparse automatically
        # builds the train-induced graph.
        from torch_geometric.datasets import Flickr
        root = os.environ.get('FLICKR_DATA_ROOT', 'data/Flickr')
        dataset = Flickr(root=root)
        data = dataset[0].to(device)
        return dataset, data

    if key == 'facebook':
        dataset, data = _load_facebook()
        data = data.to(device)
        return dataset, data

    canonical = SUPPORTED_DATASETS[key]
    dataset = Planetoid(root=f'/tmp/{canonical}', name=canonical)
    data = dataset[0].to(device)
    return dataset, data

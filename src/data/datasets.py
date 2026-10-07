"""
Unified dataset loading for OGB, GraphSAINT, and domain-disjoint benchmarks.
"""

import os
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.processing.cache import cache_access_lock, cache_creation_lock


def _cached_pyg_dataset(dataset_class, cache_root, required_files, **kwargs):
    """Allow concurrent cached constructors; isolate upstream in-place writers."""
    cache_root = Path(cache_root)

    def ready():
        return all((cache_root / path).is_file() for path in required_files)

    # Keep the lock outside the root: OGB replaces that directory on download.
    with cache_access_lock(cache_root, ready=ready):
        return dataset_class(**kwargs)


SUPPORTED_DATASETS = {
    # OGB node classification
    'ogbn-products': 'ogbn-products',
    'ogbn-arxiv': 'ogbn-arxiv',
    # Provenance-specific domain-disjoint node-classification benchmarks.
    'facebook100-year': 'Facebook100-Year',
    'mag-countries': 'MAG-Countries',
    # GraphSAINT benchmark graphs (Zeng et al., ICLR 2020), loaded from the
    # authors' released files under their inductive protocol.  Shorthands for
    # the generic form `graphsaint:<name>`.
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
    directory = Path(root) / name.replace('-', '_')
    if directory.with_name(directory.name + '_pyg').exists():
        directory = directory.with_name(directory.name + '_pyg')
    try:
        dataset = _cached_pyg_dataset(
            PygNodePropPredDataset, directory,
            ('processed/geometric_data_processed.pt', 'raw/edge.csv.gz',
             'raw/node-feat.csv.gz', 'RELEASE_v1.txt'),
            name=name, root=root)
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




GRAPHSAINT_DATASETS = {
    # name -> (is_multilabel, human note).  Statistics are GraphSAINT Table 1,
    # reproduced exactly by this loader (see _load_graphsaint's docstring).
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
      * DROP SELF-LOOPS.  The accounting counts paths in a simple graph and
        degree capping would otherwise spend a node's budget on an arc to
        itself. Reddit's 23,213,838 arcs are exactly 2 x 11,606,919 edges.
      * STANDARDIZE FEATURES.  Match GraphSAINT's loader by fitting a
        StandardScaler on nodes present in the training adjacency and applying
        it to every split.

    `data.edge_index` is the full graph and `data.train_edge_index` the
    train-induced one. Native train/val/test masks define the inductive partitions.
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
    if multilabel and os.path.exists(labels_npy):
        # amazon ships this alongside a 523 MB class_map.json; same content.
        y = torch.from_numpy(np.load(labels_npy)).float()
    else:
        y = _graphsaint_labels(d, n, multilabel)

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_edge_index = train_edge_index
    for split, m in masks.items():
        setattr(data, f'{split}_mask', m)

    num_classes = int(y.size(1)) if multilabel else int(y.max()) + 1
    return _SimpleDataset(data, int(x.size(1)), num_classes,
                          multilabel=multilabel), data


def _graphsaint_labels(directory, num_nodes, multilabel):
    import json

    directory = Path(directory)
    cache = directory / '_labels_cache.pt'
    if cache.exists():
        return torch.load(cache)

    def parse_labels():
        # Yelp/Amazon class maps are hundreds of MB: cache the parsed tensor.
        with (directory / 'class_map.json').open() as stream:
            class_map = json.load(stream)
        if multilabel:
            labels = torch.zeros(
                num_nodes, len(next(iter(class_map.values()))), dtype=torch.float)
            for key, value in class_map.items():
                labels[int(key)] = torch.tensor(value, dtype=torch.float)
        else:
            labels = torch.zeros(num_nodes, dtype=torch.long)
            for key, value in class_map.items():
                labels[int(key)] = int(value)
        return labels

    labels = None
    temporary = None
    try:
        with cache_creation_lock(cache):
            if not cache.exists():
                labels = parse_labels()
                with tempfile.NamedTemporaryFile(
                        prefix=f".{cache.name}.", suffix=".tmp",
                        dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    torch.save(labels, stream)
                os.replace(temporary, cache)
    except OSError:
        # A read-only data directory still supports uncached loading.
        return labels if labels is not None else parse_labels()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return labels if labels is not None else torch.load(cache)


def load_dataset(name, device='cpu', domain_split=None, *, root=None):
    """
    Load a dataset by name.

    Args:
        name: One of the keys in SUPPORTED_DATASETS (case-insensitive), or
            'graphsaint:<name>' for a registered GraphSAINT release.
        device: Device to move data to.
        domain_split: Optional train/validation/test domain selection for the
            domain-disjoint datasets. Supplying any role requires all three.
        root: Optional cache directory for domain and GraphSAINT datasets.

    Returns:
        (dataset, data) tuple.
    """
    key = name.lower()
    spec = SUPPORTED_DATASETS.get(key, name)
    from src.data.domain_datasets import DOMAIN_DATASET_NAMES, load_domain_dataset
    if key in DOMAIN_DATASET_NAMES:
        data, metadata = load_domain_dataset(
            key, domain_split=domain_split, root=root)
        dataset = _SimpleDataset(data, **metadata)
        return dataset, data.to(device)
    if domain_split is not None:
        raise ValueError(
            f"domain_split is only supported for domain datasets, not '{name}'")
    # GraphSAINT graphs use provenance-specific aliases or graphsaint:<name>.
    if isinstance(spec, str) and spec.startswith('graphsaint:'):
        dataset, data = _load_graphsaint(
            spec.split(':', 1)[1], root=root)
        return dataset, data.to(device)

    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: "
                         f"{list(SUPPORTED_DATASETS.keys())} or "
                         f"graphsaint:<name>")
    dataset, data = _load_ogb_node(key)
    return dataset, data.to(device)

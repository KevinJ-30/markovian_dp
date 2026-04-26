"""
Unified dataset loading for Planetoid, OGB, and PyG benchmark datasets.
"""

import os

import torch
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
    # OGB link property prediction
    'ogbl-collab': 'ogbl-collab',
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
    finally:
        torch.load = _orig_load
    data = dataset[0]
    split_edge = dataset.get_edge_split()
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


def load_dataset(name, device='cpu'):
    """
    Load a dataset by name.

    Args:
        name: One of the keys in SUPPORTED_DATASETS (case-insensitive).
        device: Device to move data to.

    Returns:
        (dataset, data) tuple.
    """
    key = name.lower()
    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {list(SUPPORTED_DATASETS.keys())}")

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

    if key == 'ogbl-collab':
        dataset, data = _load_ogbl_collab()
        data = data.to(device)
        return dataset, data

    canonical = SUPPORTED_DATASETS[key]
    dataset = Planetoid(root=f'/tmp/{canonical}', name=canonical)
    data = dataset[0].to(device)
    return dataset, data

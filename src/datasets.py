"""
Unified dataset loading for Planetoid and OGB datasets.
"""

import torch
from torch_geometric.datasets import Planetoid


SUPPORTED_DATASETS = {
    'cora': 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed',
    'ogbn-products': 'ogbn-products',
}


def load_dataset(name, device='cpu'):
    """
    Load a dataset by name.

    Args:
        name: One of 'cora', 'citeseer', 'pubmed', 'ogbn-products' (case-insensitive).
        device: Device to move data to.

    Returns:
        (dataset, data) tuple.
    """
    key = name.lower()
    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {list(SUPPORTED_DATASETS.keys())}")

    if key == 'ogbn-products':
        from ogb.nodeproppred import PygNodePropPredDataset
        # PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks
        # OGB's internal loading of PyG objects. Allow unsafe load for OGB.
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        try:
            dataset = PygNodePropPredDataset(name='ogbn-products', root='/tmp/ogbn-products')
        finally:
            torch.load = _orig_load
        data = dataset[0]
        # ogbn-products labels are (N, 1) — squeeze to (N,)
        data.y = data.y.squeeze(-1)
        # Convert index splits to masks
        split_idx = dataset.get_idx_split()
        num_nodes = data.x.size(0)
        for split_name in ['train', 'val', 'test']:
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[split_idx[split_name if split_name != 'val' else 'valid']] = True
            setattr(data, f'{split_name}_mask', mask)
        data = data.to(device)
        return dataset, data

    canonical = SUPPORTED_DATASETS[key]
    dataset = Planetoid(root=f'/tmp/{canonical}', name=canonical)
    data = dataset[0].to(device)
    return dataset, data

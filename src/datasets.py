"""
Unified dataset loading for Planetoid datasets.
"""

from torch_geometric.datasets import Planetoid


SUPPORTED_DATASETS = {
    'cora': 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed',
}


def load_dataset(name, device='cpu'):
    """
    Load a Planetoid dataset by name.

    Args:
        name: One of 'cora', 'citeseer', 'pubmed' (case-insensitive).
        device: Device to move data to.

    Returns:
        (dataset, data) tuple.
    """
    key = name.lower()
    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {list(SUPPORTED_DATASETS.keys())}")
    canonical = SUPPORTED_DATASETS[key]
    dataset = Planetoid(root=f'/tmp/{canonical}', name=canonical)
    data = dataset[0].to(device)
    return dataset, data

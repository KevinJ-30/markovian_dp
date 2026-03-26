"""
Model factory.
"""

from src.models.gcn import SubgraphGCN
from src.models.mlp import NodeMLP


def make_model(dataset, model_type='gcn', hidden_channels=64):
    """
    Create a model by type.

    Args:
        dataset: PyG dataset (used for num_features, num_classes).
        model_type: 'gcn' or 'mlp'.
        hidden_channels: Hidden layer size.

    Returns:
        nn.Module
    """
    if model_type == 'gcn':
        return SubgraphGCN(
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
        )
    elif model_type == 'mlp':
        return NodeMLP(
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: gcn, mlp")

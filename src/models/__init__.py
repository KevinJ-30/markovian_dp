"""
Model factory.
"""

from src.models.gcn import SubgraphGCN
from src.models.mlp import NodeMLP
from src.models.link_predictor import LinkPredGCN


def make_model(dataset, model_type='gcn', hidden_channels=64):
    """
    Create a model by type.

    Args:
        dataset: PyG dataset. For node-classification models, used for
                 num_features and num_classes. For link-prediction models,
                 only num_features is used; out_channels = hidden_channels.
        model_type: 'gcn', 'mlp', or 'link_pred_gcn'.
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
    elif model_type == 'link_pred_gcn':
        return LinkPredGCN(
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: gcn, mlp, link_pred_gcn"
        )

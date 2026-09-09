import json

import torch
from torch_geometric.data import Data

from src.experiments.dpgnn_adapter import run_partitioned
from src.experiments.inductive import load_or_create_inductive_split
from src.experiments.upstream import export_partitions


def test_dpgnn_partition_adapter_smoke(tmp_path):
    nodes = torch.arange(30)
    edge_index = torch.stack((nodes, torch.roll(nodes, -1)))
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    data = Data(x=torch.randn(30, 5), y=torch.arange(30) // 10, edge_index=edge_index)
    split = load_or_create_inductive_split(data, "dpg nn-unit", root=tmp_path / "splits", seed=0)
    manifest = export_partitions(split, tmp_path / "partitions")
    result = run_partitioned(manifest, tmp_path / "result.json", steps=1, batch_size=4,
                             noise_multiplier=2.0, seed=0)
    assert result["method"] == "dp_gnn"
    assert 0.0 <= result["validation_accuracy"] <= 1.0
    assert json.loads((tmp_path / "result.json").read_text())["test_accuracy"] == result["test_accuracy"]


def test_dpgnn_adapter_is_first_party(tmp_path):
    nodes = torch.arange(12)
    data = Data(
        x=torch.randn(12, 3),
        y=torch.arange(12) % 3,
        edge_index=torch.stack((nodes, torch.roll(nodes, -1))),
    )
    split = load_or_create_inductive_split(data, "dpg nn-first-party",
                                           root=tmp_path / "splits", seed=0)
    manifest = export_partitions(split, tmp_path / "partitions")
    result = run_partitioned(manifest, tmp_path / "result.json", steps=2,
                             batch_size=4, noise_multiplier=2.0, seed=0)

    assert result["implementation"]["source"] == "src.experiments.dpgnn"
    assert result["privacy"]["accountant"] == "first_party.dpgnn.multiterm_rdp"

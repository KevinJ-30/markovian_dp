import json
import math

import torch
from torch_geometric.data import Data

from src.experiments.dpgnn_adapter import _load_partitions, run_partitioned
from src.processing.splits import load_or_create_inductive_split
from src.experiments.upstream import export_partitions


def test_dpgnn_partition_adapter_smoke(tmp_path):
    nodes = torch.arange(30)
    edge_index = torch.stack((nodes, torch.roll(nodes, -1)))
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    data = Data(x=torch.randn(30, 5), y=torch.arange(30) // 10, edge_index=edge_index)
    split = load_or_create_inductive_split(data, "dpg nn-unit", root=tmp_path / "splits", seed=0)
    manifest = export_partitions(split, tmp_path / "partitions")
    loaded, task = _load_partitions(manifest)
    assert task["primary_metric"] == "accuracy"
    assert all(
        loaded[name].eval_mask.dtype == torch.bool
        and loaded[name].eval_mask.numel() == loaded[name].num_nodes
        for name in ("train", "val", "test")
    )
    result = run_partitioned(
        manifest, tmp_path / "result.json", steps=1, batch_size=4,
        noise_multiplier=2.0, seed=0, clip=0.7, max_private_batch_nodes=1,
        architecture="graphsage")
    assert result["method"] == "dp_gnn"
    assert result["implementation"]["architecture"] == "graphsage"
    persisted = json.loads((tmp_path / "result.json").read_text())
    for metric in ("validation_accuracy", "test_accuracy"):
        assert 0.0 <= result[metric] <= 1.0
        assert persisted[metric] == result[metric]
    assert math.isfinite(result["privacy"]["epsilon"])
    assert result["privacy"]["epsilon"] > 0
    assert result["privacy"]["composition_count"] == 1
    assert persisted["privacy"] == result["privacy"]




def test_dpgnn_adapter_consumes_binary_manifest_metadata(tmp_path):
    nodes = torch.arange(40)
    edge_index = torch.stack((nodes, torch.roll(nodes, -1)))
    data = Data(
        x=torch.randn(40, 4),
        y=nodes.remainder(2),
        edge_index=torch.cat((edge_index, edge_index.flip(0)), dim=1),
    )
    split = load_or_create_inductive_split(
        data,
        "dpg-nn-binary",
        root=tmp_path / "splits",
        seed=2,
        primary_metric="auroc",
        binary=True,
    )
    manifest = export_partitions(split, tmp_path / "partitions")
    result = run_partitioned(
        manifest,
        tmp_path / "binary-result.json",
        steps=1,
        batch_size=4,
        noise_multiplier=2.0,
        max_private_batch_nodes=8,
    )
    assert result["metric"] == "auroc"
    assert set(result) >= {"validation_auroc", "test_auroc"}
    assert "validation_accuracy" not in result

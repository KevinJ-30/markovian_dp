from argparse import Namespace

import torch
from torch_geometric.data import Data

from privacy import accounting_analysis
from privacy import sampling


def _args():
    return Namespace(K=1, num_neighbors=1, seed=3, device="cpu")


def test_partition_samplers_are_local_and_cpu_cached(tmp_path):
    args = _args()
    partitions = {
        "train": Data(x=torch.ones(3, 2), y=torch.tensor([0, 1, 0]), edge_index=torch.tensor([[0, 1], [1, 2]])),
        "val": Data(x=torch.ones(2, 2), y=torch.tensor([1, 0]), edge_index=torch.tensor([[0], [1]])),
        "test": Data(x=torch.ones(2, 2), y=torch.tensor([0, 1]), edge_index=torch.empty((2, 0), dtype=torch.long)),
    }
    datasets = []
    for name, graph in partitions.items():
        dataset = sampling.subgraph_sampler(
            K=1, num_neighbors=1, neighbor_num_constrain_for_training_for_memory=1,
            out_degree_inverse=sampling.compute_out_degree_inverse(
                graph.edge_index, graph.num_nodes, tmp_path / f"{name}-degree.pt",
            ),
            graph_data=graph, graph_data_name=name, mask=torch.ones(graph.num_nodes, dtype=torch.bool),
            setting="inductive", dataset_mode=name if name != "train" else "train",
            device="cpu", args=args, cache_file_path=tmp_path / name,
        )
        datasets.append(dataset)
        sample = dataset[0]
        assert sample[3].min() >= 0 and sample[3].max() < graph.num_nodes
    assert len({str(dataset.cache_file_path) for dataset in datasets}) == 3
    assert sampling.collate_subgraphs([]) is None


def test_native_calibration_uses_exact_steps_and_safe_high_sigma(monkeypatch):
    captured = {}

    class FakeComputer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def eps_from_noise(self, sigma, delta, show_flag=False):
            return 10.0 / sigma, 2.0

    monkeypatch.setattr(accounting_analysis.mix, "divergence_computer", FakeComputer)
    sigma, achieved = accounting_analysis.get_std_node_dp(
        q=0.25, steps=6, D_out=8, M_train=1, epsilon=2.0, delta=5e-4,
    )
    assert captured["steps"] == 6
    assert captured["D_out"] == 8
    assert achieved <= 2.0
    assert sigma >= 5.0

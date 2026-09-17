from argparse import Namespace

import torch
from torch_geometric.data import Data

from inductive_adapter import _sampler
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
        dataset = _sampler(graph, name=name, mode=name, args=args, cache_dir=tmp_path)
        datasets.append(dataset)
        sample = dataset[0]
        assert sample[3].min() >= 0 and sample[3].max() < graph.num_nodes
    assert len({str(dataset.cache_file_path) for dataset in datasets}) == 3
    assert sampling.collate_subgraphs([]) is None


def test_inverse_degree_counts_unique_incoming_sources(tmp_path):
    edges = torch.tensor([[0, 0, 1, 2], [2, 2, 2, 0]])
    sources = [set() for _ in range(4)]
    for source, target in edges.t().tolist():
        sources[target].add(source)
    expected = torch.tensor([1 / len(nodes) if nodes else 0 for nodes in sources])
    inverse = sampling.compute_in_degree_inverse(edges, 4, tmp_path / "inverse.pt")
    torch.testing.assert_close(inverse, expected)


def test_inverse_degree_handles_empty_edges_and_population(tmp_path):
    edges = torch.empty((2, 0), dtype=torch.long)
    for num_nodes in (3, 0):
        inverse = sampling.compute_in_degree_inverse(edges, num_nodes, tmp_path / f"{num_nodes}.pt")
        torch.testing.assert_close(inverse, torch.zeros(num_nodes, dtype=torch.float))


def test_adapter_ignores_old_outgoing_degree_cache(tmp_path):
    graph = Data(
        x=torch.ones(4, 2), y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 0, 1, 2], [2, 2, 2, 0]]),
    )
    old_path = tmp_path / "train-degree-inverse.pt"
    torch.save(torch.tensor([0.5, 1, 1, 0]), old_path)
    old_contents = old_path.read_bytes()
    for _ in range(2):
        dataset = _sampler(graph, name="train", mode="train", args=_args(), cache_dir=tmp_path)
        torch.testing.assert_close(dataset.out_degree_inverse, torch.tensor([1, 0, 0.5, 0]))
    torch.testing.assert_close(
        torch.load(tmp_path / "train-in-degree-inverse.pt", weights_only=False),
        torch.tensor([1, 0, 0.5, 0]),
    )
    assert old_path.read_bytes() == old_contents


def test_adapter_caps_training_prefix_but_preserves_evaluation_sampling(tmp_path):
    leaves = torch.arange(1, 502)
    centers = torch.zeros_like(leaves)
    graph = Data(
        x=torch.arange(1, 503, dtype=torch.float).view(-1, 1),
        y=torch.zeros(502, dtype=torch.long),
        edge_index=torch.stack((torch.cat((centers, leaves)), torch.cat((leaves, centers)))),
    )
    for mode, candidate_count in (("train", 500), ("val", 501), ("test", 501)):
        dataset = _sampler(graph, name=mode, mode=mode, args=_args(), cache_dir=tmp_path)
        assert set(dataset.dict_of_nodes_neighbors[0].tolist()) == set(range(1, candidate_count + 1))
        node_ids = dataset[0][3].reshape(-1).tolist()
        if mode == "test":
            assert len(node_ids) == 2
            assert node_ids[0] == 0 and node_ids[1] in range(1, 502)
        else:
            assert len(node_ids) == candidate_count + 1
            assert set(node_ids) == set(range(candidate_count + 1))

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.processing.graphs import (
    max_degrees,
    preprocess_edges,
    preprocess_graph,
    preprocess_inductive_split,
)
from src.processing.splits import GraphPartition, InductiveSplit


def _edge_set(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


def _random_graph(n=200, m=3000, seed=0):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randint(0, n, (m,), generator=generator)
    target = torch.randint(0, n, (m,), generator=generator)
    return torch.stack((source, target)), n


def _random_undirected_graph(n=150, m=1200, seed=0):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randint(0, n, (m,), generator=generator)
    target = torch.randint(0, n, (m,), generator=generator)
    keep = source != target
    source, target = source[keep], target[keep]
    edges = torch.stack(
        (torch.cat((source, target)), torch.cat((target, source)))
    )
    return torch.unique(edges, dim=1), n


def test_preprocess_edges_is_nonmutating_bidirectional_and_simple():
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 2, 1]],
        dtype=torch.int32,
    )
    original = edge_index.clone()

    result = preprocess_edges(edge_index, 3)

    assert torch.equal(edge_index, original)
    assert result.dtype == torch.long
    assert result.device.type == "cpu"
    assert _edge_set(result) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert result.size(1) == 4


def test_preprocess_edges_replaces_input_diagonals_with_exact_loops():
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 2, 2], [0, 0, 1, 1, 2, 2]]
    )

    result = preprocess_edges(edge_index, 4, add_self_loops=True)

    loops = result[:, result[0] == result[1]]
    assert loops.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert _edge_set(result[:, result[0] != result[1]]) == {(0, 1), (1, 0)}


def test_preprocess_edges_handles_empty_graphs():
    empty = torch.empty((2, 0), dtype=torch.long)
    assert preprocess_edges(empty, 0).shape == (2, 0)
    assert preprocess_edges(empty, 3).shape == (2, 0)
    assert preprocess_edges(empty, 3, add_self_loops=True).tolist() == [
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert max_degrees(empty, 0) == (0, 0)


@pytest.mark.parametrize(
    ("edge_index", "num_nodes", "kwargs", "message"),
    [
        (torch.tensor([0, 1]), 2, {}, "shape"),
        (torch.zeros((3, 1), dtype=torch.long), 2, {}, "shape"),
        (torch.tensor([[0], [-1]]), 2, {}, "outside"),
        (torch.tensor([[0], [2]]), 2, {}, "outside"),
        (torch.tensor([[0.0], [1.0]]), 2, {}, "integer"),
        (torch.empty((2, 0), dtype=torch.long), -1, {}, "nonnegative"),
        (torch.empty((2, 0), dtype=torch.long), 2, {"max_in_degree": 0}, "positive"),
        (torch.empty((2, 0), dtype=torch.long), 2, {"max_out_degree": True}, "positive"),
        (torch.empty((2, 0), dtype=torch.long), 2, {"degree_cap_mode": "other"}, "degree_cap_mode"),
        (
            torch.empty((2, 0), dtype=torch.long),
            2,
            {"degree_cap_mode": "undirected"},
            "requires equal",
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            2,
            {
                "max_in_degree": 1,
                "max_out_degree": 2,
                "degree_cap_mode": "undirected",
            },
            "requires equal",
        ),
    ],
)
def test_preprocess_edges_rejects_invalid_inputs(edge_index, num_nodes, kwargs, message):
    with pytest.raises(ValueError, match=message):
        preprocess_edges(edge_index, num_nodes, **kwargs)


def test_directed_degree_capping_respects_both_bounds_and_seed():
    edge_index, num_nodes = _random_graph()
    first = preprocess_edges(
        edge_index,
        num_nodes,
        max_in_degree=7,
        max_out_degree=9,
        generator=torch.Generator().manual_seed(3),
    )
    second = preprocess_edges(
        edge_index,
        num_nodes,
        max_in_degree=7,
        max_out_degree=9,
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.equal(first, second)
    max_in, max_out = max_degrees(first, num_nodes)
    assert max_in <= 7
    assert max_out <= 9
    input_off_diagonal = _edge_set(edge_index[:, edge_index[0] != edge_index[1]])
    bidirectional_input = input_off_diagonal | {
        (target, source) for source, target in input_off_diagonal
    }
    assert _edge_set(first) <= bidirectional_input


def test_undirected_degree_capping_is_symmetric_bounded_and_seeded():
    edge_index, num_nodes = _random_undirected_graph(seed=7)

    def capped():
        return preprocess_edges(
            edge_index,
            num_nodes,
            max_in_degree=4,
            max_out_degree=4,
            degree_cap_mode="undirected",
            generator=torch.Generator().manual_seed(11),
        )

    first = capped()
    edges = _edge_set(first)
    assert torch.equal(first, capped())
    assert all((target, source) in edges for source, target in edges)
    max_in, max_out = max_degrees(first, num_nodes)
    assert max_in <= 4
    assert max_out <= 4
    assert edges <= _edge_set(edge_index)


def test_degree_caps_apply_before_exact_loop_insertion():
    nodes = torch.arange(5)
    source = nodes.repeat_interleave(5)
    target = nodes.repeat(5)
    edge_index = torch.stack((source, target))

    result = preprocess_edges(
        edge_index,
        5,
        max_in_degree=1,
        max_out_degree=1,
        add_self_loops=True,
        generator=torch.Generator().manual_seed(4),
    )

    off_diagonal = result[:, result[0] != result[1]]
    assert max_degrees(off_diagonal, 5) <= (1, 1)
    loops = result[:, result[0] == result[1]]
    assert loops.tolist() == [nodes.tolist(), nodes.tolist()]


def _partition(offset):
    data = Data(
        x=torch.arange(6, dtype=torch.float32).reshape(3, 2) + offset,
        y=torch.tensor([0, 1, 0]),
        edge_index=torch.tensor(
            [[0, 0, 0, 1, 1, 2], [0, 1, 1, 0, 0, 2]]
        ),
        num_nodes=3,
    )
    node_ids = torch.arange(offset, offset + 3)
    return GraphPartition(data=data, node_ids=node_ids, stats={"edges": 99})


def test_preprocess_inductive_split_preserves_partition_metadata_and_rebuilds_stats():
    masks = {
        "train": torch.tensor([True, True, True, False, False, False, False, False, False]),
        "val": torch.tensor([False, False, False, True, True, True, False, False, False]),
        "test": torch.tensor([False, False, False, False, False, False, True, True, True]),
    }
    split = InductiveSplit(
        train=_partition(0),
        val=_partition(3),
        test=_partition(6),
        masks=masks,
        num_classes=2,
        path=Path("split.pt"),
    )

    result = preprocess_inductive_split(split)

    assert result.masks is masks
    assert result.num_classes == split.num_classes
    assert result.path == split.path
    for name in ("train", "val", "test"):
        original = getattr(split, name)
        transformed = getattr(result, name)
        assert torch.equal(transformed.node_ids, original.node_ids)
        assert transformed.data is not original.data
        assert _edge_set(transformed.data.edge_index) == {(0, 1), (1, 0)}
        assert transformed.stats["edges"] == transformed.data.edge_index.size(1) == 2
        assert original.stats["edges"] == 99


def test_preprocess_graph_clones_without_mutating_input():
    data = _partition(0).data
    original_edges = data.edge_index.clone()

    result = preprocess_graph(data)

    assert result is not data
    assert torch.equal(data.edge_index, original_edges)
    assert result.edge_index.device == data.edge_index.device

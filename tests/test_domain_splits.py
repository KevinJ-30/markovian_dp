import pytest
import torch
from torch_geometric.data import Data

from src.processing.graphs import preprocess_inductive_split
from src.processing.splits import load_or_create_inductive_split


def _domain_graph(split_id="toy-fingerprint"):
    data = Data(
        x=torch.arange(18, dtype=torch.float32).reshape(6, 3),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 3, 4, 5, 2, 1],
                [1, 0, 2, 3, 4, 5, 2, 5, 4],
            ]
        ),
        num_nodes=6,
    )
    data.train_mask = torch.tensor([True, True, False, False, False, False])
    data.val_mask = torch.tensor([False, False, True, False, True, False])
    data.test_mask = torch.tensor([False, False, False, True, False, True])
    data.domain_id = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
    data.domain_names = ["source", "target"]
    data.domain_split = {
        "train": ["source"],
        "val": ["target"],
        "test": ["target"],
        "seed": 4,
        "val_ratio": 0.5,
    }
    data.domain_split_id = split_id
    return data


def _edge_tuples(edge_index):
    return {tuple(edge) for edge in edge_index.t().tolist()}


def test_domain_split_keeps_full_shared_target_context_and_local_score_masks(tmp_path):
    split = load_or_create_inductive_split(
        _domain_graph(),
        "toy-domain",
        root=tmp_path,
        split_strategy="domain",
        primary_metric="auroc",
        binary=True,
        metric_ignore_label=19,
    )

    assert split.train.node_ids.tolist() == [0, 1]
    assert split.val.node_ids.tolist() == split.test.node_ids.tolist() == [2, 3, 4, 5]
    assert split.train.eval_mask.tolist() == [True, True]
    assert split.val.eval_mask.tolist() == [True, False, True, False]
    assert split.test.eval_mask.tolist() == [False, True, False, True]
    assert torch.logical_xor(split.val.eval_mask, split.test.eval_mask).all()
    assert _edge_tuples(split.val.data.edge_index) == _edge_tuples(split.test.data.edge_index)
    assert _edge_tuples(split.val.data.edge_index) == {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 3),
    }
    assert split.train.stats["evaluated_nodes"] == 2
    assert split.val.stats["nodes"] == split.test.stats["nodes"] == 4
    assert split.val.stats["evaluated_nodes"] == split.test.stats["evaluated_nodes"] == 2
    assert not ({2, 3, 4, 5} & set(split.train.node_ids.tolist()))


def test_domain_metadata_and_masks_survive_preprocessing_and_device_movement(tmp_path):
    split = load_or_create_inductive_split(
        _domain_graph(),
        "toy-domain",
        root=tmp_path,
        split_strategy="domain",
        primary_metric="auroc",
        binary=True,
        metric_ignore_label=19,
    )
    processed = preprocess_inductive_split(split, make_bidirectional=False).to("cpu")

    assert processed.primary_metric == "auroc"
    assert processed.binary is True
    assert processed.metric_ignore_label == 19
    assert processed.domain_split == split.domain_split
    assert processed.domain_split_id == "toy-fingerprint"
    assert processed.path == split.path
    for name in ("train", "val", "test"):
        before = getattr(split, name)
        after = getattr(processed, name)
        assert torch.equal(after.node_ids, before.node_ids)
        assert torch.equal(after.eval_mask, before.eval_mask)
        assert after.stats["evaluated_nodes"] == int(after.eval_mask.sum())


def test_domain_cache_path_and_payload_are_bound_to_split_id(tmp_path):
    first = load_or_create_inductive_split(
        _domain_graph("fingerprint-a"),
        "toy-domain",
        root=tmp_path,
        split_strategy="domain",
    )
    reloaded = load_or_create_inductive_split(
        _domain_graph("fingerprint-a"),
        "toy-domain",
        root=tmp_path,
        split_strategy="domain",
    )
    second = load_or_create_inductive_split(
        _domain_graph("fingerprint-b"),
        "toy-domain",
        root=tmp_path,
        split_strategy="domain",
    )

    assert first.path == reloaded.path
    assert first.path != second.path
    assert first.path.name == "toy-domain-domain-fingerprint-a.pt"
    assert second.path.name == "toy-domain-domain-fingerprint-b.pt"
    payload = torch.load(first.path, map_location="cpu")
    assert payload["domain_split_id"] == "fingerprint-a"
    assert payload["domain_split"] == first.domain_split
    assert all(torch.equal(payload["masks"][name], first.masks[name]) for name in ("train", "val", "test"))


def test_domain_split_rejects_dataset_metadata_that_disagrees_with_cache_identity(tmp_path):
    data = _domain_graph()
    load_or_create_inductive_split(
        data, "toy-domain", root=tmp_path, split_strategy="domain"
    )
    changed = data.clone()
    changed.val_mask = torch.tensor([False, False, False, True, True, False])
    changed.test_mask = torch.tensor([False, False, True, False, False, True])

    with pytest.raises(ValueError, match="does not match current dataset metadata"):
        load_or_create_inductive_split(
            changed, "toy-domain", root=tmp_path, split_strategy="domain"
        )


def test_domain_split_strictly_validates_context_metadata(tmp_path):
    overlapping_source = _domain_graph()
    overlapping_source.domain_split = {
        **overlapping_source.domain_split,
        "val": ["source", "target"],
    }
    with pytest.raises(ValueError, match="training domains must be disjoint"):
        load_or_create_inductive_split(
            overlapping_source,
            "invalid-overlap",
            root=tmp_path,
            split_strategy="domain",
        )

    malformed_ids = _domain_graph()
    malformed_ids.domain_id = malformed_ids.domain_id.to(torch.int32)
    with pytest.raises(ValueError, match="domain_id must be a one-dimensional long tensor"):
        load_or_create_inductive_split(
            malformed_ids,
            "invalid-domain-id",
            root=tmp_path,
            split_strategy="domain",
        )


def test_existing_split_strategies_get_all_true_local_eval_masks(tmp_path):
    data = _domain_graph()
    native = load_or_create_inductive_split(
        data, "native-toy", root=tmp_path, split_strategy="native"
    )

    for partition in (native.train, native.val, native.test):
        assert partition.eval_mask.dtype == torch.bool
        assert partition.eval_mask.all()
        assert partition.stats["evaluated_nodes"] == partition.stats["nodes"]
    assert native.primary_metric == "accuracy"
    assert native.binary is False
    assert native.metric_ignore_label is None
    assert native.domain_split is None
    assert native.domain_split_id is None

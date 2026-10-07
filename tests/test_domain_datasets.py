import hashlib
import io
import urllib.error

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy.io import loadmat, savemat
from torch_geometric.data import Data

from src.data import datasets
from src.data import domain_datasets as domain
from src.processing.splits import load_or_create_inductive_split


def _split(train, val, test, seed=0, val_ratio=0.2):
    return {
        "train": train,
        "val": val,
        "test": test,
        "seed": seed,
        "val_ratio": val_ratio,
    }


def test_normalization_defaults_equal_explicit_and_are_registry_ordered():
    dataset_name = "facebook100-year"
    normalized, split_id = domain.normalize_domain_split(dataset_name)
    explicit, explicit_id = domain.normalize_domain_split(
        dataset_name,
        _split(
            ["caltech36", "johns-hopkins55", "amherst41"],
            ["yale4", "cornell5"],
            ["texas80", "brown11", "penn94"],
        ),
    )

    assert normalized == explicit
    assert split_id == explicit_id
    assert normalized["train"] == [
        "amherst41",
        "johns-hopkins55",
        "caltech36",
    ]
    assert normalized["val"] == ["cornell5", "yale4"]
    assert normalized["test"] == ["penn94", "brown11", "texas80"]
    _, different_id = domain.normalize_domain_split(
        dataset_name, {**explicit, "seed": 1}
    )
    assert different_id != split_id


@pytest.mark.parametrize(
    ("bad_split", "error"),
    [
        ({"train": ["us"]}, "provide train, val, and test together"),
        (_split([], ["cn"], ["de"]), "must not be empty"),
        (_split(["US"], ["cn"], ["de"]), "Unknown mag-countries domain"),
        (_split(["us", "us"], ["cn"], ["de"]), "duplicate domains"),
        (_split(["us"], ["us"], ["de"]), "Training domains must be disjoint"),
        (_split(["us"], ["cn"], ["de"], val_ratio=0), "0 < val_ratio < 1"),
        (_split(["us"], ["cn"], ["de"], val_ratio=1), "0 < val_ratio < 1"),
    ],
)
def test_invalid_domain_splits_fail_clearly(bad_split, error):
    with pytest.raises((TypeError, ValueError), match=error):
        domain.normalize_domain_split("mag-countries", bad_split)


def test_shared_domain_masks_are_stratified_deterministic_and_exhaustive():
    normalized, split_id = domain.normalize_domain_split(
        "mag-countries", _split(["us"], ["cn"], ["cn"], seed=41, val_ratio=0.2)
    )
    graphs = {
        "us": Data(
            x=torch.zeros((2, 3)),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        ),
        "cn": Data(
            x=torch.zeros((20, 3)),
            y=torch.tensor([0] * 10 + [1] * 10),
            edge_index=torch.tensor([[0, 19], [19, 0]]),
        ),
    }

    first, _ = domain._assemble_domains(
        "mag-countries", normalized, split_id, graphs
    )
    second, _ = domain._assemble_domains(
        "mag-countries", normalized, split_id, graphs
    )

    assert torch.equal(first.val_mask, second.val_mask)
    assert torch.equal(first.test_mask, second.test_mask)
    assert first.val_mask[:2].sum() == 0
    assert first.test_mask[:2].sum() == 0
    assert first.val_mask[2:12].sum() == 2
    assert first.val_mask[12:].sum() == 2
    assert first.test_mask[2:12].sum() == 8
    assert first.test_mask[12:].sum() == 8
    membership = first.train_mask.int() + first.val_mask.int() + first.test_mask.int()
    assert torch.equal(membership, torch.ones_like(membership))
    assert not bool((first.val_mask & first.test_mask).any())
    assert first.edge_index.tolist() == [[0, 1, 2, 21], [1, 0, 21, 2]]


def _write_fb100_year_fixture(root, *, include_last_class=True):
    root.mkdir(exist_ok=True)
    for index, filename in enumerate(domain.FB100_FILES.values()):
        years = [0, 2004, 1900, 2008, 2009 if include_last_class else 2007, 2010]
        info = np.array([[index + 1, gender, 1, 0, 1, year, index + 10]
                         for gender, year in zip([1, 0, 1, 2, 1, 2], years)])
        info[[0, 2], 2] = [777, 778]
        adjacency = sp.csr_matrix(
            (np.ones(8), ([0, 1, 1, 2, 3, 4, 4, 5], [1, 3, 4, 3, 4, 1, 5, 4])),
            shape=(6, 6),
        )
        savemat(root / filename, {"A": adjacency, "local_info": info})


def test_facebook100_year_cohorts_induce_edges_without_target_features(tmp_path):
    _write_fb100_year_fixture(tmp_path)
    requested = _split(["penn94"], ["amherst41"], ["cornell5"])
    dataset, data = datasets.load_dataset("facebook100-year", domain_split=requested, root=tmp_path)
    # Stable year identities, not a per-school relabeling of present classes.
    assert data.y.tolist() == [0, 4, 5] * 3
    assert dataset.num_classes == 6
    assert data.domain_names == ["penn94", "amherst41", "cornell5"]
    assert data.domain_id.dtype == torch.long
    assert data.domain_id.tolist() == [0] * 3 + [1] * 3 + [2] * 3
    # Excluded nodes and all 18 schools still define the feature vocabulary.
    expected_features = torch.zeros(9, 44)
    school_ids = torch.arange(3).repeat_interleave(3)
    expected_features[torch.arange(9), school_ids] = 1
    expected_features[torch.arange(9), torch.tensor([18, 20, 19] * 3)] = 1
    expected_features[:, 21] = 1
    expected_features[torch.arange(9), 26 + school_ids] = 1
    torch.testing.assert_close(data.x, expected_features)
    assert dataset.num_features == 44
    assert dataset.domain_dataset is True
    assert dataset.task_type == "MULTICLASS"
    assert dataset.primary_metric == "accuracy"
    assert dataset.metric_ignore_label is None
    assert dataset.domain_split == data.domain_split == requested
    assert dataset.domain_split_id == data.domain_split_id
    assert dataset.label_metadata == data.label_metadata
    assert data.label_metadata["raw_to_class"] == {
        str(year): year - 2004 for year in range(2004, 2010)
    }
    assert dataset.provenance == data.provenance
    assert data.provenance["revision"] == domain.FB100_REVISION
    assert data.provenance["feature_columns"] == [0, 1, 2, 3, 4, 6]
    assert data.provenance["feature_vocabulary_domains"] == list(domain.FB100_DOMAINS)
    assert data.train_mask.tolist() == [True] * 3 + [False] * 6
    assert data.val_mask.tolist() == [False] * 3 + [True] * 3 + [False] * 3
    assert data.test_mask.tolist() == [False] * 6 + [True] * 3
    expected = {(offset + u, offset + v) for offset in (0, 3, 6)
                for u, v in ((0, 1), (0, 2), (1, 2), (2, 0))}
    assert set(map(tuple, data.edge_index.t().tolist())) == expected
    assert dataset.domain_node_counts == {
        name: {"raw": 6, "retained": 3, "excluded": 3}
        for name in ("penn94", "amherst41", "cornell5")
    }
    # Changing retained target values must change labels, never input features.
    for filename in domain.FB100_FILES.values():
        raw = loadmat(tmp_path / filename)
        raw["local_info"][[1, 4], 5] = [2009, 2004]
        savemat(tmp_path / filename, {"A": raw["A"], "local_info": raw["local_info"]})
    _, changed = datasets.load_dataset("facebook100-year", domain_split=requested, root=tmp_path)
    assert changed.y.tolist() == [5, 4, 0] * 3
    assert torch.equal(changed.x, data.x)
    assert torch.equal(changed.edge_index, data.edge_index)


def test_facebook100_year_preserves_six_class_head_through_cache(tmp_path):
    raw = tmp_path / "raw"
    _write_fb100_year_fixture(raw, include_last_class=False)
    requested = _split(["penn94"], ["amherst41"], ["cornell5"])
    data, _ = domain.load_domain_dataset("facebook100-year", requested, root=raw)
    year = load_or_create_inductive_split(
        data, "facebook100-year", root=tmp_path / "splits", split_strategy="domain"
    )
    assert year.num_classes == 6  # Year 2009 is absent in every fixture school.
    for name in ("train", "val", "test"):
        part = getattr(year, name)
        assert part.data.y.tolist() == [0, 4, 3]
        assert part.node_ids.numel() == 3
        assert bool(part.eval_mask.all())
    reloaded = load_or_create_inductive_split(
        data, "facebook100-year", root=tmp_path / "splits", split_strategy="domain"
    )
    assert reloaded.num_classes == 6
    assert reloaded.train.data.label_metadata["raw_to_class"]["2009"] == 5


def _save_mag(path, label_offset=0):
    graph = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        y=torch.tensor([label_offset, 19], dtype=torch.long),
    )
    torch.save(graph, path)
    payload = path.read_bytes()
    return path.name, len(payload), hashlib.md5(payload).hexdigest()


def test_mag_checks_integrity_before_loading_and_exposes_metric_rule(tmp_path, monkeypatch):
    patched = dict(domain.MAG_ARTIFACTS)
    for name, offset in (("us", 0), ("cn", 1), ("de", 2)):
        filename = domain.MAG_ARTIFACTS[name][0]
        patched[name] = _save_mag(tmp_path / filename, offset)
    monkeypatch.setattr(domain, "MAG_ARTIFACTS", patched)

    events = []
    original_check = domain._check_file
    original_load = torch.load

    def recording_check(*args, **kwargs):
        events.append("check")
        return original_check(*args, **kwargs)

    def recording_load(*args, **kwargs):
        events.append("load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(domain, "_check_file", recording_check)
    monkeypatch.setattr(torch, "load", recording_load)
    data, metadata = domain.load_domain_dataset(
        "mag-countries",
        _split(["us"], ["cn"], ["de"]),
        root=tmp_path,
    )

    assert events == ["check", "load", "check", "load", "check", "load"]
    assert data.x.shape == (6, 2)
    assert data.edge_index.tolist() == [
        [0, 1, 2, 3, 4, 5],
        [1, 0, 3, 2, 5, 4],
    ]
    assert data.domain_names == ["us", "cn", "de"]
    assert metadata["num_classes"] == 20
    assert metadata["metric_ignore_label"] == 19
    assert data.metric_ignore_label == 19


def test_mag_bad_checksum_never_reaches_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "US_labels_20.pt"
    path.write_bytes(b"not a trusted pickle")
    monkeypatch.setitem(
        domain.MAG_ARTIFACTS,
        "us",
        (path.name, path.stat().st_size, "0" * 32),
    )
    loaded = False

    def forbidden_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("unverified content was deserialized")

    def failed_download(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(domain, "_download_atomic", failed_download)
    with pytest.raises(RuntimeError, match="offline"):
        domain._load_mag_domain(tmp_path, "us")
    assert loaded is False


def test_atomic_download_is_reused_and_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_success(url, timeout):
        calls.append((url, timeout))
        return Response(b"complete artifact")

    monkeypatch.setattr(domain.urllib.request, "urlopen", open_success)
    destination = tmp_path / "cache" / "artifact.bin"
    domain._ensure_file("https://example.test/artifact", destination)
    domain._ensure_file("https://example.test/artifact", destination)
    assert destination.read_bytes() == b"complete artifact"
    assert calls == [("https://example.test/artifact", 60)]
    assert list(destination.parent.glob("*.tmp")) == []

    def open_failure(url, timeout):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(domain.urllib.request, "urlopen", open_failure)
    failed = tmp_path / "failed" / "artifact.bin"
    with pytest.raises(RuntimeError, match="https://example.test/missing"):
        domain._download_atomic("https://example.test/missing", failed)
    assert not failed.exists()
    assert list(failed.parent.glob("*.tmp")) == []

import hashlib
import io
import json
import urllib.error

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy.io import savemat
from torch_geometric.data import Data

from src.data import datasets
from src.data import domain_datasets as domain


def _write_twitch_domain(root, name, node_rows, edges, features):
    upper = name.upper()
    folder = root / upper
    folder.mkdir(parents=True)
    prefix = f"musae_{upper}"
    target = folder / f"{prefix}_target.csv"
    target.write_text(
        "id,days,mature,views,partner,new_id\n"
        + "".join(
            f"{source_id},1,{str(mature)},1,False,{graph_id}\n"
            for source_id, graph_id, mature in node_rows
        ),
        encoding="utf-8",
    )
    (folder / f"{prefix}_edges.csv").write_text(
        "from,to\n" + "".join(f"{source},{target}\n" for source, target in edges),
        encoding="utf-8",
    )
    (folder / f"{prefix}_features.json").write_text(
        json.dumps({str(node): values for node, values in features.items()}),
        encoding="utf-8",
    )


def _three_domain_twitch_fixture(root):
    _write_twitch_domain(
        root,
        "de",
        [(1000, 30, False), (1001, 10, True)],
        [(30, 10)],
        {30: [2, 100, 2], 10: [7]},
    )
    _write_twitch_domain(
        root,
        "engb",
        [(2000, 9, True), (2001, 7, False)],
        [(7, 9)],
        {9: [3], 7: []},
    )
    _write_twitch_domain(
        root,
        "es",
        [(3000, 400, False), (3001, 100, True)],
        [(400, 100)],
        {400: [4], 100: [5]},
    )


def _split(train, val, test, seed=0, val_ratio=0.2):
    return {
        "train": train,
        "val": val,
        "test": test,
        "seed": seed,
        "val_ratio": val_ratio,
    }


def test_twitch_noncontiguous_ids_disconnected_offsets_and_public_loading(tmp_path):
    _three_domain_twitch_fixture(tmp_path)
    requested = _split(["de"], ["engb"], ["es"])

    dataset, data = datasets.load_dataset(
        "twitch-explicit", domain_split=requested, root=tmp_path
    )

    assert data.domain_names == ["de", "engb", "es"]
    assert data.domain_id.dtype == torch.long
    assert data.domain_id.tolist() == [0, 0, 1, 1, 2, 2]
    # Node order follows target rows, while raw graph IDs are mapped explicitly.
    assert data.edge_index.tolist() == [[0, 3, 4], [1, 2, 5]]
    assert data.y.tolist() == [0, 1, 1, 0, 0, 1]
    assert data.x.shape == (6, domain.TWITCH_NUM_FEATURES)
    assert data.x[0, 2] == 1 and data.x[0, 100] == 1
    assert data.x[1, 7] == 1 and data.x[3].sum() == 0
    assert data.train_mask.tolist() == [True, True, False, False, False, False]
    assert data.val_mask.tolist() == [False, False, True, True, False, False]
    assert data.test_mask.tolist() == [False, False, False, False, True, True]
    assert dataset.num_features == domain.TWITCH_NUM_FEATURES
    assert dataset.num_classes == 2
    assert dataset.domain_dataset is True
    assert dataset.task_type == "BINARY"
    assert dataset.primary_metric == "auroc"
    assert dataset.metric_ignore_label is None
    assert dataset.domain_split == data.domain_split == requested
    assert dataset.domain_split_id == data.domain_split_id
    assert len(dataset.domain_split_id) == 64


def test_twitch_rejects_unknown_edge_ids_and_conflicting_duplicate_nodes(tmp_path):
    _write_twitch_domain(
        tmp_path,
        "de",
        [(1, 10, False), (2, 10, True)],
        [],
        {10: [0]},
    )
    with pytest.raises(ValueError, match="Conflicting duplicate Twitch node ID"):
        domain._parse_twitch_domain(tmp_path, "de")

    other = tmp_path / "other"
    _write_twitch_domain(
        other,
        "de",
        [(1, 10, False)],
        [(10, 99)],
        {10: [0]},
    )
    with pytest.raises(ValueError, match="Unknown Twitch node ID"):
        domain._parse_twitch_domain(other, "de")


def test_normalization_defaults_equal_explicit_and_are_registry_ordered():
    normalized, split_id = domain.normalize_domain_split("facebook100")
    explicit, explicit_id = domain.normalize_domain_split(
        "facebook100",
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
        "facebook100", {**explicit, "seed": 1}
    )
    assert different_id != split_id


@pytest.mark.parametrize(
    ("bad_split", "error"),
    [
        ({"train": ["de"]}, "provide train, val, and test together"),
        (_split([], ["engb"], ["es"]), "must not be empty"),
        (_split(["DE"], ["engb"], ["es"]), "Unknown twitch-explicit domain"),
        (_split(["de", "de"], ["engb"], ["es"]), "duplicate domains"),
        (_split(["de"], ["de"], ["es"]), "Training domains must be disjoint"),
        (_split(["de"], ["engb"], ["es"], val_ratio=0), "0 < val_ratio < 1"),
        (_split(["de"], ["engb"], ["es"], val_ratio=1), "0 < val_ratio < 1"),
    ],
)
def test_invalid_domain_splits_fail_clearly(bad_split, error):
    with pytest.raises((TypeError, ValueError), match=error):
        domain.normalize_domain_split("twitch-explicit", bad_split)


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


def _write_fb100_fixture(root):
    root.mkdir(exist_ok=True)
    for index, filename in enumerate(domain.FB100_FILES.values()):
        local_info = np.array(
            [
                [index + 1, 0, 1, 0, 1, 2005, index + 10],
                [index + 1, 2, 2, 3, 0, 2006, 0],
            ],
            dtype=np.int64,
        )
        adjacency = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8))
        savemat(root / filename, {"A": adjacency, "local_info": local_info})


def test_facebook100_uses_all_school_vocabulary_and_missing_gender_rule(tmp_path):
    _write_fb100_fixture(tmp_path)
    requested = _split(["penn94"], ["amherst41"], ["cornell5"])

    data, metadata = domain.load_domain_dataset(
        "facebook100", requested, root=tmp_path
    )

    # Match GraphOOD/sklearn label_binarize: binary feature columns occupy one
    # positive-class column, while multiclass columns include raw category 0.
    assert data.x.shape == (6, 41)
    assert data.y.tolist() == [0, 1, 0, 1, 0, 1]
    assert set(map(tuple, data.edge_index.t().tolist())) == {
        (0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4),
    }
    assert data.x[0].sum() == 3
    assert data.x[1].sum() == 5
    # The same non-school categories occupy the same columns across schools,
    # while school-specific status categories remain distinct.
    assert torch.equal(data.x[0, 18:22], data.x[2, 18:22])
    assert data.x[0, 0] == 1 and data.x[2, 1] == 1
    assert data.domain_names == ["penn94", "amherst41", "cornell5"]
    assert metadata["task_type"] == "MULTICLASS"
    assert metadata["primary_metric"] == "accuracy"
    assert metadata["metric_ignore_label"] is None


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


def test_existing_facebook_name_keeps_its_original_loader(monkeypatch):
    expected_dataset = object()

    class DataStub:
        def to(self, device):
            assert device == "cpu"
            return self

    expected_data = DataStub()
    monkeypatch.setattr(
        datasets,
        "_load_facebook",
        lambda: (expected_dataset, expected_data),
    )

    actual_dataset, actual_data = datasets.load_dataset("facebook")

    assert actual_dataset is expected_dataset
    assert actual_data is expected_data

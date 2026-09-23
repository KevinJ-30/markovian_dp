import json

import numpy as np
import pytest
import scipy.sparse as sp

from src.data import datasets



@pytest.mark.parametrize(
    ("public_name", "raw_name"),
    [("saint-yelp", "yelp"), ("saint-amazon", "amazon")],
)
def test_graphsaint_public_aliases_dispatch_to_raw_names(
        public_name, raw_name, monkeypatch):
    calls = []
    dataset = object()

    class DataStub:
        def to(self, device):
            calls.append(("device", device))
            return self

    data = DataStub()

    def fake_load(name, root=None):
        calls.append(("load", name, root))
        return dataset, data

    monkeypatch.setattr(datasets, "_load_graphsaint", fake_load)
    actual_dataset, actual_data = datasets.load_dataset(
        public_name, device="cpu")

    assert actual_dataset is dataset
    assert actual_data is data
    assert calls == [("load", raw_name, None), ("device", "cpu")]


@pytest.mark.parametrize("legacy_name", ["yelp", "amazon"])
def test_unprefixed_graphsaint_aliases_are_rejected(legacy_name):
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_dataset(legacy_name)

def test_graphsaint_loader_standardizes_from_training_adjacency(tmp_path, monkeypatch):
    folder = tmp_path / "fixture"
    folder.mkdir()
    features = np.array(
        [[1.0, 10.0], [3.0, 14.0], [5.0, 18.0], [70.0, 220.0], [90.0, 260.0]],
        dtype=np.float64,
    )
    np.save(folder / "feats.npy", features)

    rows = np.array([0, 1, 2, 3, 4])
    columns = np.array([1, 0, 2, 4, 3])
    full = sp.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(5, 5))
    train = sp.csr_matrix(
        (np.ones(3), (np.array([0, 1, 2]), np.array([1, 0, 2]))),
        shape=(5, 5),
    )
    sp.save_npz(folder / "adj_full.npz", full)
    sp.save_npz(folder / "adj_train.npz", train)
    (folder / "role.json").write_text(json.dumps({"tr": [0, 1, 2], "va": [3], "te": [4]}))
    (folder / "class_map.json").write_text(json.dumps({str(i): i % 2 for i in range(5)}))

    monkeypatch.setitem(datasets.GRAPHSAINT_DATASETS, "fixture", (False, "test fixture"))
    _, data = datasets._load_graphsaint("fixture", root=tmp_path)

    training_features = features[:3]
    expected = (features - training_features.mean(axis=0)) / training_features.std(axis=0)
    np.testing.assert_allclose(data.x.numpy(), expected.astype(np.float32), rtol=1e-6, atol=1e-6)

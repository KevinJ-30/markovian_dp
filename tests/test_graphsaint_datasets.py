import json

import numpy as np
import scipy.sparse as sp

from src.data import datasets


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

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.datasets import load_dataset
from src.experiments.run import _resolve_task_metadata


def _raw_graphland(root, name, targets):
    folder = root / name / "raw" / name
    folder.mkdir(parents=True)
    n = len(targets)
    (folder / "info.yaml").write_text(yaml.safe_dump({
        "task": "regression",
        "target_name": "target",
        "numerical_features_names": ["numeric", "fraction"],
        "fraction_features_names": ["fraction"],
        "categorical_features_names": ["category"],
    }))
    numeric = np.arange(n, dtype=np.float32)
    numeric[2] = np.nan
    pd.DataFrame({
        "numeric": numeric,
        "fraction": np.linspace(0, 1, n),
        "category": np.arange(n) % 3,
    }).to_csv(folder / "features.csv")
    pd.DataFrame({"target": targets}).to_csv(folder / "targets.csv")
    pd.DataFrame({
        "source": np.arange(n), "target": np.roll(np.arange(n), -1),
    }).to_csv(folder / "edgelist.csv", index=False)
    # Deliberately unlike our random 80/10/10 split. The loader must not fit
    # target scaling on this upstream RH training set.
    pd.DataFrame({
        "train": np.arange(n) < n // 2,
        "val": (np.arange(n) >= n // 2) & (np.arange(n) < 3 * n // 4),
        "test": np.arange(n) >= 3 * n // 4,
    }).to_csv(folder / "split_masks_RH.csv")


@pytest.mark.parametrize("name", ["hm-prices", "avazu-ctr"])
def test_graphland_random_split_is_complete_fixed_and_regression(tmp_path, name):
    targets = np.arange(30, dtype=np.float32) ** 2 + 0.25
    _raw_graphland(tmp_path, name, targets)
    dataset, data = load_dataset(name, root=tmp_path)
    masks = [getattr(data, f"{role}_mask") for role in ("train", "val", "test")]
    assert [int(mask.sum()) for mask in masks] == [24, 3, 3]
    assert torch.all(sum(mask.int() for mask in masks) == 1)
    assert torch.isfinite(data.x).all()
    assert _resolve_task_metadata(dataset, {})["regression"]
    assert dataset.num_classes == 1
    assert data.y.dtype == torch.float32
    torch.testing.assert_close(data.y[data.train_mask].mean(), torch.tensor(0.), atol=1e-6, rtol=0)
    torch.testing.assert_close(data.y[data.train_mask].std(unbiased=False), torch.tensor(1.))
    torch.testing.assert_close(
        data.y * dataset.target_std + dataset.target_mean,
        torch.from_numpy(targets),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(937)
        _, reloaded = load_dataset(name, root=tmp_path)
    for role, mask in zip(("train", "val", "test"), masks):
        assert torch.equal(mask, getattr(reloaded, f"{role}_mask"))
    torch.testing.assert_close(data.y, reloaded.y)


def test_graphland_held_out_targets_cannot_affect_training_scaling(tmp_path):
    targets = np.linspace(1, 30, 30, dtype=np.float32)
    _raw_graphland(tmp_path / "original", "hm-prices", targets)
    dataset, data = load_dataset("hm-prices", root=tmp_path / "original")
    changed_targets = targets.copy()
    changed_targets[~data.train_mask.numpy()] += 10000
    _raw_graphland(tmp_path / "changed", "hm-prices", changed_targets)
    changed_dataset, changed = load_dataset("hm-prices", root=tmp_path / "changed")
    torch.testing.assert_close(data.y[data.train_mask], changed.y[changed.train_mask])
    assert dataset.target_mean == changed_dataset.target_mean
    assert dataset.target_std == changed_dataset.target_std
    assert not torch.equal(data.y[data.test_mask], changed.y[changed.test_mask])


def test_graphland_constant_targets_remain_finite(tmp_path):
    _raw_graphland(tmp_path, "avazu-ctr", np.full(30, 0.125))
    dataset, data = load_dataset("avazu-ctr", root=tmp_path)
    assert torch.equal(data.y, torch.zeros(30))
    torch.testing.assert_close(
        data.y * dataset.target_std + dataset.target_mean,
        torch.full((30,), 0.125),
    )


def test_graphland_rejects_unlabeled_training_population(tmp_path):
    targets = np.arange(30, dtype=np.float32)
    targets[0] = np.nan
    _raw_graphland(tmp_path, "hm-prices", targets)
    with pytest.raises(ValueError, match="one finite regression target per node"):
        load_dataset("hm-prices", root=tmp_path)

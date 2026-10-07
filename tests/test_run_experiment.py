"""Scientific parameter, data-isolation, and native worker contracts."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_experiment as worker


def _parameters(**overrides):
    return {"dataset": "fixture", "method": "mlp", "lr": 0.01,
            "batch_size": 32, "epochs": 1, "bootstrap_resamples": 0, **overrides}


def _arguments(output, **overrides):
    command = []
    for key, value in _parameters(**overrides).items():
        command.extend(("--" + key.replace("_", "-"),
                        json.dumps(value) if isinstance(value, dict) else str(value)))
    return [*command, "--device", "cpu", "--out-dir", str(output)]


@pytest.mark.parametrize("change", [
    {"epochs": True}, {"batch_size": 1.0}, {"seed": False}, {"seed": 2**32},
    {"bootstrap_resamples": -1}, {"gnn_hidden": 0}, {"degree_bound": True},
    {"lr": float("nan")}, {"lr": float("inf")}, {"lr": True}, {"lr": 0},
    {"dropout": 1}, {"dropout": -0.01}, {"dropout": float("-inf")},
    {"method": "missing"}, {"dataset": ""}, {"split_root": ""},
    {"device": "cpu"}, {"out_dir": "output"}, {"prepared_protocol": "old"},
    {"campaign_request": "old"}, {"unknown": 1},
])
def test_normalization_rejects_invalid_scientific_parameters(change):
    with pytest.raises(ValueError):
        worker.normalize_parameters(_parameters(**change))


@pytest.mark.parametrize("method,extra", [
    ("mlp", {"epsilon": 8}), ("mlp", {"p2": 0.5}),
    ("mlp", {"progap_depth": 1}), ("sparse_sage", {"progap_depth": 3}),
    ("sparse_sage", {"gin_pooling": "sum"}), ("progap", {"sparse_radius": 1}),
    ("dpar", {"dpgnn_radius": 1}), ("mlp", {"progap_python": sys.executable}),
])
def test_explicit_method_parameters_are_not_silently_ignored(method, extra):
    settings = {"method": method}
    if method not in worker.NONPRIVATE:
        settings["epsilon"] = 8
    if method.startswith("sparse_"):
        settings["p2"] = 0.5
    with pytest.raises(ValueError, match="not applicable"):
        worker.normalize_parameters(_parameters(**settings, **extra))


@pytest.mark.parametrize("change", [
    {"method": "dp_mlp"}, {"method": "dp_mlp", "epsilon": 0},
    {"method": "dp_mlp", "epsilon": float("nan")},
    {"method": "sparse_sage", "epsilon": 8},
    {"method": "sparse_sage", "epsilon": 8, "p2": 0},
    {"method": "sparse_sage", "epsilon": 8, "p2": 1.01},
    {"method": "sparse_sage", "epsilon": 8, "p2": 0.5, "sparse_radius": True},
    {"method": "sparse_sage", "epsilon": 8, "p2": 0.5,
     "degree_bound": 5, "sparse_degree_cap": 10},
    {"method": "progap", "epsilon": 8, "progap_depth": 0},
    {"method": "progap", "epsilon": 8, "progap_depth": False},
    {"method": "progap", "epsilon": 8, "progap_python": ""},
])
def test_private_parameter_boundaries(change):
    with pytest.raises(ValueError):
        worker.normalize_parameters(_parameters(**change))


def test_omitted_depth_does_not_enter_nonprogap_identity():
    for method in worker.METHODS:
        options = {"method": method}
        if method not in worker.NONPRIVATE:
            options["epsilon"] = 8
        if method.startswith("sparse_"):
            options["p2"] = 0.5
        parameters = worker.normalize_parameters(_parameters(**options))
        assert ("progap_depth" in parameters) == (method == "progap")
        assert worker.normalize_parameters(parameters) == parameters


@pytest.mark.parametrize("method", worker.METHODS)
@pytest.mark.parametrize("weight_decay", [-1, float("nan"), float("inf"), True])
def test_weight_decay_rejects_invalid_values_for_every_method(method, weight_decay):
    options = {"method": method, "weight_decay": weight_decay}
    if method not in worker.NONPRIVATE:
        options["epsilon"] = 8
    if method.startswith("sparse_"):
        options["p2"] = 0.5
    with pytest.raises(ValueError, match="weight_decay"):
        worker.normalize_parameters(_parameters(**options))


def test_normalization_does_not_import_training_or_probe_paths(tmp_path):
    code = """
import builtins
import json
import pathlib
from scripts import run_experiment as worker
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith(('torch.', 'src.')):
        raise AssertionError('training import during normalization: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
def forbidden(*args, **kwargs):
    raise AssertionError('filesystem/interpreter probe during normalization')
pathlib.Path.exists = pathlib.Path.stat = pathlib.Path.mkdir = forbidden
worker.shutil.which = forbidden
values = worker.normalize_parameters({
    'dataset': 'facebook100-year', 'method': 'progap', 'epsilon': 8,
    'lr': .01, 'batch_size': 32, 'epochs': 1, 'split_root': 'no-such-cache',
    'progap_python': 'not-installed-python',
    'domain_split': {'train': ['caltech36'], 'val': ['cornell5'], 'test': ['penn94']},
})
assert values['progap_depth'] == 3
assert values['domain_split']['train'] == ['caltech36']
assert values['progap_python'] == 'not-installed-python'
print(json.dumps(values))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=worker.REPO_ROOT,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["split_root"] == str(worker.REPO_ROOT / "no-such-cache")


@pytest.mark.parametrize("split", [
    [], {"unknown": 1}, {"train": ["us"]},
    {"train": "us", "val": ["cn"], "test": ["de"]},
    {"train": ["us", "us"], "val": ["cn"], "test": ["de"]},
    {"train": ["us"], "val": ["us"], "test": ["de"]},
    {"seed": True}, {"val_ratio": float("nan")}, {"val_ratio": 1},
])
def test_invalid_domain_split_structure_is_rejected_without_loading(split):
    with pytest.raises(ValueError):
        worker.normalize_parameters(_parameters(dataset="mag-countries", domain_split=split))


def test_custom_domains_cannot_replace_named_presets_or_nondomain_data():
    split = {"train": ["us"], "val": ["cn"], "test": ["de"]}
    for dataset in ("mag-allbut2", "fb100-year-6", "ogbn-arxiv"):
        with pytest.raises(ValueError, match="domain_split"):
            worker.normalize_parameters(_parameters(dataset=dataset, domain_split=split))


def test_custom_domain_loading_validates_membership_before_data_access(tmp_path):
    split = {"train": ["missing-domain"], "val": ["cn"], "test": ["de"]}
    with pytest.raises(ValueError, match="Unknown mag-countries domain"):
        worker._load_split("mag-countries", tmp_path, split)


def test_custom_domain_split_builds_separate_graphs_and_preserves_metadata(tmp_path, monkeypatch):
    import numpy as np
    import scipy.sparse as sp
    from scipy.io import savemat
    from src.data import datasets
    from src.data.domain_datasets import FB100_FILES, FB100_YEAR_CLASSES, normalize_domain_split

    raw = tmp_path / "raw"
    raw.mkdir()
    for index, filename in enumerate(FB100_FILES.values()):
        info = np.array([[index + 1, 1, 2, 3, 4, year, 5] for year in FB100_YEAR_CLASSES])
        adjacency = sp.csr_matrix(
            (np.ones(6), (np.arange(6), np.roll(np.arange(6), -1))), shape=(6, 6))
        savemat(raw / filename, {"A": adjacency, "local_info": info})
    original_load = datasets.load_dataset
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs:
                        original_load(*args, **kwargs, root=raw))
    requested = {"train": ["cornell5"], "val": ["penn94"], "test": ["amherst41"],
                 "seed": 17, "val_ratio": 0.3}
    dataset, split, task, strategy = worker._load_split("facebook100-year", tmp_path / "splits", requested)
    expected, split_id = normalize_domain_split("facebook100-year", requested)
    assert dataset == "facebook100-year" and strategy == "domain"
    assert not task["binary"] and task["primary_metric"] == "accuracy"
    assert split.domain_split == expected and split.domain_split_id == split_id
    assert split.train.node_ids.tolist() == list(range(12, 18))
    assert split.val.node_ids.tolist() == list(range(6))
    assert split.test.node_ids.tolist() == list(range(6, 12))
    for partition in (split.train, split.val, split.test):
        assert partition.data.num_nodes == 6
        assert partition.data.edge_index.max().item() == 5
        assert partition.eval_mask.tolist() == [True] * 6


@pytest.mark.parametrize("protocol,canonical,train", [
    ("mag-allbut2", "mag-countries", {"us", "fr", "ru", "jp"}),
    ("fb100-year-6", "facebook100-year",
     {"johns-hopkins55", "caltech36", "amherst41", "reed98", "brandeis99", "princeton12"}),
])
def test_named_protocols_preserve_domain_membership(protocol, canonical, train):
    dataset, roles = worker._protocol(protocol)
    assert dataset == canonical
    assert set(roles["train"]) == train
    assert not (train & (set(roles["val"]) | set(roles["test"])))


def test_worker_occupied_output_is_untouched(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_text('{"evidence":"original"}')
    assert worker.main(_arguments(output)) == 1
    assert (output / "result.json").read_text() == '{"evidence":"original"}'
    assert not (output / "worker_error.json").exists()


def test_worker_memory_error_publishes_observed_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(worker.REPO_ROOT)
    output = tmp_path / "output"

    def exhausted(args):
        raise MemoryError("fixture allocation failure")

    monkeypatch.setattr(worker, "run", exhausted)
    assert worker.main(_arguments(output)) == 86
    assert json.loads((output / "worker_error.json").read_text()) == {
        "kind": "host_oom", "exception_class": "MemoryError", "message": "fixture allocation failure"}
    assert not (output / "result.json").exists()


def test_training_seed_does_not_change_shared_split_and_outputs_are_complete(tmp_path, monkeypatch):
    import torch
    from torch_geometric.data import Data
    from src.data import datasets

    monkeypatch.chdir(worker.REPO_ROOT)
    count = 40
    nodes = torch.arange(count)
    graph = Data(x=torch.arange(count * 4).reshape(count, 4).float() / (count * 4),
                 y=nodes % 2, edge_index=torch.stack((nodes, (nodes + 1) % count)), num_nodes=count)
    dataset = SimpleNamespace(task_type="MULTICLASS", primary_metric="accuracy", num_classes=2)
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: (dataset, graph.clone()))
    results = []
    with torch.random.fork_rng(devices=[]):
        for seed in (1, 17):
            output = tmp_path / f"output-{seed}"
            assert worker.main(_arguments(output, seed=seed, split_root=str(tmp_path / "splits"), mlp_hidden=4)) == 0
            result = json.loads((output / "result.json").read_text())
            config = json.loads((output / "config.json").read_text())
            with (output / "result.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 1 and rows[0]["status"] == "completed"
            assert float(rows[0]["test_metric"]) == result["test_metric"]
            assert math.isfinite(result["test_metric"]) and math.isfinite(result["validation_metric"])
            assert result["selection"]["split"] == "validation"
            assert result["seed"] == seed and result["split_seed"] == 0
            assert config["requested"]["seed"] == seed
            assert not (output / "worker_error.json").exists()
            results.append(result)
    assert results[0]["split_file"] == results[1]["split_file"]


@pytest.mark.parametrize("architecture", ["GraphSAGE", "GIN"])
def test_bounded_full_neighbor_inference_preserves_logits(architecture, monkeypatch):
    import torch
    from src.models import baselines

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = getattr(baselines, architecture)(4, 3, hidden=7, layers=2, dropout=0.5).eval()
        features = torch.randn(9, 4)
        edges = torch.tensor([[1, 2, 2, 3, 4, 5, 5, 6, 7, 0],
                              [0, 0, 0, 1, 1, 2, 2, 2, 3, 0]])
        expected = model(features, edges).detach()
        monkeypatch.setattr(baselines, "_MESSAGE_BYTES", 32)
        with torch.no_grad():
            actual = model(features, edges)
            isolated = model(features, torch.empty((2, 0), dtype=torch.long))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[8], isolated[8], rtol=0, atol=0)


def test_progap_preparation_preserves_topology_with_auxiliary_indices():
    code = """
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor
import inductive_adapter
edges = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 1, 2]])
data = Data(x=torch.arange(9).reshape(3, 3).float(), y=torch.tensor([0, 1, 0]),
            edge_index=edges, train_edge_index=edges.clone(),
            edge_weight=torch.tensor([1., 2., 3., 4., 5.]),
            eval_mask=torch.tensor([True, False, True]))
reference = ToSparseTensor(layout=torch.sparse_csr)(
    Data(edge_index=edges, edge_weight=data.edge_weight, num_nodes=3))
actual = inductive_adapter._prepare(data)
torch.testing.assert_close(actual.adj_t.to_dense(), reference.adj_t.to_dense())
torch.testing.assert_close(actual.x, data.x)
assert torch.equal(actual.eval_mask, data.eval_mask)
assert torch.equal(data.edge_index, edges)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60,
                   cwd=worker.REPO_ROOT / "third_party/ProGAP")


def native_progap_fixture(root):
    """Direct deterministic graph fixture, independent of any prepared manifests."""
    import torch
    from torch_geometric.data import Data
    from src.processing.splits import GraphPartition, InductiveSplit, graph_statistics

    features = torch.arange(160 * 8, dtype=torch.float32).reshape(160, 8) / (160 * 8)
    labels = torch.arange(160) % 2
    parts, masks = {}, {}
    for name, lo, hi in (("train", 0, 128), ("val", 128, 144), ("test", 144, 160)):
        count = hi - lo
        source = torch.arange(count)
        target = (source + 1) % count
        edges = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
        data = Data(x=features[lo:hi].clone(), y=labels[lo:hi].clone(), edge_index=edges,
                    num_nodes=count, eval_mask=torch.ones(count, dtype=torch.bool))
        parts[name] = GraphPartition(data, torch.arange(lo, hi), graph_statistics(data), data.eval_mask)
        masks[name] = torch.zeros(160, dtype=torch.bool)
        masks[name][lo:hi] = True
    split = InductiveSplit(**parts, masks=masks, num_classes=2, path=root / "fixture_split")
    task = {"binary": False, "multilabel": False, "regression": False,
            "metric_ignore_label": None, "primary_metric": "accuracy"}
    return split, task


@pytest.mark.parametrize("depth_argument,expected_depth", [(1, 1), (2, 2), (None, 3)])
def test_progap_depth_controls_native_stages_and_privacy_composition(tmp_path, depth_argument, expected_depth):
    split, task = native_progap_fixture(tmp_path)
    settings = {"method": "progap", "epsilon": 8, "gnn_hidden": 8}
    if depth_argument is not None:
        settings["progap_depth"] = depth_argument
    args = worker.parser().parse_args(_arguments(tmp_path / "output", **settings))
    worker._check_args(args)
    native, parameters = worker._progap(args, split, task, 32, 1 / 128)
    assert parameters["depth"] == expected_depth
    assert parameters["stages"] == expected_depth + 1
    assert native["completed_updates"] == 4 * (expected_depth + 1)
    assert {stage["stage"] for stage in native["selection"]["stage_selections"]} == set(range(expected_depth + 1))
    privacy = native["privacy"]["total"]
    assert privacy["parameters"]["component_coefficients"] == [expected_depth, expected_depth + 1]
    assert privacy["composition_count"] == 2 * expected_depth + 1
    assert privacy["epsilon"] <= 8 + 1e-6

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.processing.splits import load_or_create_inductive_split
from src.experiments.upstream import UpstreamBaseline, export_partitions


def _graph():
    nodes = torch.arange(30)
    edge_index = torch.stack((nodes, torch.roll(nodes, shifts=-1)))
    return Data(x=torch.randn(30, 3), y=torch.arange(30) // 10, edge_index=edge_index)


@pytest.fixture
def adapter_source(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "privacy = {'total': {'epsilon': 1.0, 'delta': 0.0005, 'accountant': 'fixture', 'noise_multiplier': 1.0, 'sampling_probability': 0.1, 'composition_count': 1, 'parameters': {}}}\n"
        "calibration = {'target_epsilon': 8.0, 'target_delta': 0.0005, 'achieved_epsilon': 1.0, 'noise_std': 1.0}\n"
        "result = {'privacy': privacy, 'calibration': calibration}\n"
        "if os.environ.get('PROGAP_BINARY') == '1' and not os.environ.get('FIXTURE_LEGACY_BINARY'):\n"
        "    result.update(metric='auroc', validation_auroc=0.5, test_auroc=0.6)\n"
        "    if os.environ.get('FIXTURE_BINARY_ALIASES'):\n"
        "        result.update(validation_accuracy=0.5, test_accuracy=0.6)\n"
        "else:\n"
        "    result.update(validation_accuracy=0.5, validation_macro_f1=0.4, test_accuracy=0.6, test_macro_f1=0.5)\n"
        "Path(os.environ['RESULT_PATH']).write_text(json.dumps(result))\n"
    )
    return source


def _config(source, **overrides):
    parameters = {
        "target_epsilon": 8.0, "target_delta": 0.0005,
        "epochs": 1, "batch_size": 8, "max_degree": 5, "depth": 1,
    }
    config = {
        "source_dir": str(source),
        "command": [sys.executable, "adapter.py"],
        "seed": 17,
        "parameters": parameters,
        "environment": {"RUNTIME_DEVICE": "cpu"},
    }
    config.update(overrides)
    return config


def _task_split(split, *, binary=False, primary_metric="accuracy",
                metric_ignore_label=None, domain_split=None, domain_split_id=None):
    return SimpleNamespace(
        train=split.train,
        val=split.val,
        test=split.test,
        num_classes=split.num_classes,
        primary_metric=primary_metric,
        binary=binary,
        metric_ignore_label=metric_ignore_label,
        domain_split=domain_split,
        domain_split_id=domain_split_id,
    )


def test_partition_export_v2_carries_task_domain_and_local_masks(tmp_path):
    masks = {
        "train": torch.tensor([True, True, True]),
        "val": torch.tensor([True, False, True]),
        "test": torch.tensor([False, True, False]),
    }
    partitions = {}
    for name, eval_mask in masks.items():
        data = Data(
            x=torch.randn(3, 2),
            y=torch.tensor([0, 1, 0]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        )
        partitions[name] = SimpleNamespace(
            data=data,
            node_ids=torch.arange(3),
            eval_mask=eval_mask,
            stats={"nodes": 3},
        )
    domain_split = {
        "train": ["de"],
        "val": ["engb"],
        "test": ["engb"],
        "seed": 7,
        "val_ratio": 0.2,
    }
    split = SimpleNamespace(
        **partitions,
        num_classes=2,
        primary_metric="auroc",
        binary=True,
        metric_ignore_label=None,
        domain_split=domain_split,
        domain_split_id="split-fingerprint",
    )

    manifest_path = export_partitions(split, tmp_path / "export")
    manifest = json.loads(manifest_path.read_text())

    assert manifest == {
        "format": 2,
        "num_classes": 2,
        "primary_metric": "auroc",
        "binary": True,
        "metric_ignore_label": None,
        "domain_split": domain_split,
        "domain_split_id": "split-fingerprint",
        "partitions": {
            "train": "train.pt",
            "val": "val.pt",
            "test": "test.pt",
        },
    }
    for name, expected_mask in masks.items():
        payload = torch.load(
            manifest_path.parent / manifest["partitions"][name],
            map_location="cpu",
            weights_only=False,
        )
        assert torch.equal(payload["data"].eval_mask, expected_mask)
        assert payload["data"].eval_mask.dtype == torch.bool
        assert not hasattr(partitions[name].data, "eval_mask")


def test_progap_binary_rejects_accuracy_named_legacy_results(adapter_source, tmp_path):
    base_split = load_or_create_inductive_split(
        _graph(), "binary-legacy", root=tmp_path, seed=0
    )
    split = _task_split(base_split, binary=True, primary_metric="auroc")
    config = _config(
        adapter_source,
        environment={"RUNTIME_DEVICE": "cpu", "FIXTURE_LEGACY_BINARY": "1"},
    )

    with pytest.raises(ValueError, match="must report metric 'auroc'"):
        UpstreamBaseline("progap", config).run(split)


def test_progap_binary_rejects_auroc_stored_in_accuracy_fields(
        adapter_source, tmp_path):
    base_split = load_or_create_inductive_split(
        _graph(), "binary-aliases", root=tmp_path, seed=0
    )
    split = _task_split(base_split, binary=True, primary_metric="auroc")
    config = _config(
        adapter_source,
        environment={"RUNTIME_DEVICE": "cpu", "FIXTURE_BINARY_ALIASES": "1"},
    )

    with pytest.raises(ValueError, match="must not store auroc in accuracy fields"):
        UpstreamBaseline("progap", config).run(split)


def test_removed_upstream_method_is_rejected():
    with pytest.raises(ValueError, match="unsupported upstream method"):
        UpstreamBaseline("heterpoisson", {})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda config: config["parameters"].pop("target_delta"), "target_delta"),
        (lambda config: config["parameters"].update(target_epsilon=0), "target_epsilon"),
        (lambda config: config["environment"].update(PROGAP_EPSILON="7"), "environment"),
        (lambda config: config["environment"].update(PROGAP_BINARY="1"),
         "resolved from the dataset"),
    ],
)
def test_target_contract_rejects_incomplete_invalid_or_legacy_values(adapter_source, tmp_path, mutate, message):
    split = load_or_create_inductive_split(_graph(), "bridge-progap", root=tmp_path, seed=0)
    config = _config(adapter_source)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        UpstreamBaseline("progap", config).run(split)


@pytest.mark.parametrize("parameter,value", [
    ("hidden_dim", 3.5), ("eval_chunk_size", 0), ("learning_rate", float("nan")),
    ("max_grad_norm", float("inf")), ("dropout", 1.0), ("weight_decay", -0.1),
    ("optimizer", "rmsprop"), ("unknown_model_knob", 10),
])
def test_progap_rejects_invalid_or_ignored_constructor_controls(
    adapter_source, tmp_path, parameter, value
):
    split = load_or_create_inductive_split(_graph(), "invalid-progap-knob", root=tmp_path, seed=0)
    config = _config(adapter_source)
    config["parameters"][parameter] = value
    with pytest.raises(ValueError):
        UpstreamBaseline("progap", config).run(split)

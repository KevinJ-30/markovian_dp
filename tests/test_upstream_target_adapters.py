import json
import sys

import pytest
import torch
from torch_geometric.data import Data

from src.experiments.inductive import load_or_create_inductive_split
from src.experiments.upstream import UpstreamBaseline


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
        "captured = {key: value for key, value in os.environ.items() if key.startswith(('PROGAP_', 'HETERPOISSON_', 'RUNTIME_'))}\n"
        "Path(os.environ['RESULT_PATH']).write_text(json.dumps({'validation_accuracy': 0.5, 'validation_macro_f1': 0.4, 'test_accuracy': 0.6, 'test_macro_f1': 0.5, 'privacy': {'total': {'epsilon': 1.0, 'delta': 0.0005, 'accountant': 'fixture', 'noise_multiplier': 1.0, 'sampling_probability': 0.1, 'composition_count': 1, 'parameters': {}}}, 'calibration': {'target_epsilon': 8.0, 'target_delta': 0.0005, 'achieved_epsilon': 1.0, 'noise_std': 1.0}, 'captured': captured}))\n"
    )
    return source


def _config(source, method="progap", **overrides):
    parameters = {"target_epsilon": 8.0, "target_delta": 0.0005}
    if method == "progap":
        parameters.update(epochs=1, batch_size=8, max_degree=5, depth=1)
    else:
        parameters.update(epochs=1, expected_batchsize=8, K=1, num_neighbors=1,
                          clip_norm=1.0, learning_rate=0.001, degree_bound=8)
    config = {
        "source_dir": str(source),
        "command": [sys.executable, "adapter.py"],
        "seed": 17,
        "parameters": parameters,
        "environment": {"RUNTIME_DEVICE": "cpu"},
    }
    config.update(overrides)
    return config


def test_progap_target_pair_is_forwarded_without_runtime_loss(adapter_source, tmp_path):
    split = load_or_create_inductive_split(_graph(), "bridge", root=tmp_path, seed=0)
    result = UpstreamBaseline("progap", _config(adapter_source)).run(split)

    assert result["captured"] == {
        "PROGAP_BATCH_SIZE": "8",
        "PROGAP_DEPTH": "1",
        "PROGAP_EPOCHS": "1",
        "PROGAP_MAX_DEGREE": "5",
        "PROGAP_SEED": "17",
        "PROGAP_TARGET_DELTA": "0.0005",
        "PROGAP_TARGET_EPSILON": "8.0",
        "RUNTIME_DEVICE": "cpu",
    }
    assert result["privacy"]["total"]["delta"] == 0.0005


@pytest.mark.parametrize(
    "method, mutate, message",
    [
        ("progap", lambda config: config["parameters"].pop("target_delta"), "target_delta"),
        ("progap", lambda config: config["parameters"].update(target_epsilon=0), "target_epsilon"),
        ("progap", lambda config: config["environment"].update(PROGAP_EPSILON="7"), "environment"),
        ("heterpoisson", lambda config: config["parameters"].pop("degree_bound"), "degree_bound"),
    ],
)
def test_target_contract_rejects_incomplete_invalid_or_legacy_values(adapter_source, tmp_path, method, mutate, message):
    split = load_or_create_inductive_split(_graph(), f"bridge-{method}", root=tmp_path, seed=0)
    config = _config(adapter_source, method)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        UpstreamBaseline(method, config).run(split)


def test_heterpoisson_target_pair_is_forwarded(adapter_source, tmp_path):
    split = load_or_create_inductive_split(_graph(), "heter-bridge", root=tmp_path, seed=0)
    result = UpstreamBaseline("heterpoisson", _config(adapter_source, "heterpoisson")).run(split)
    assert result["captured"]["HETERPOISSON_TARGET_EPSILON"] == "8.0"
    assert result["captured"]["HETERPOISSON_TARGET_DELTA"] == "0.0005"
    assert result["captured"]["HETERPOISSON_DEGREE_BOUND"] == "8"

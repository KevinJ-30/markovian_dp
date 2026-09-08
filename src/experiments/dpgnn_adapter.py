"""Partition adapter retaining Google's DP-GNN sampler, trainer, and accountant."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def _source_root() -> Path:
    return Path(__file__).parents[3] / "google-research"


def _imports():
    root = str(_source_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from differentially_private_gnns import input_pipeline, privacy_accountants, train
    import jax
    import jax.numpy as jnp
    import ml_collections
    return input_pipeline, privacy_accountants, train, jax, jnp, ml_collections


def _graph(data: Any, config: Any, rng: Any, input_pipeline: Any):
    """Use upstream conversion, edge sampler, normalization, and self-loops."""
    edge_index = data.edge_index.cpu().numpy()
    raw = SimpleNamespace(
        senders=edge_index[0].astype(np.int32), receivers=edge_index[1].astype(np.int32),
        node_features=data.x.cpu().numpy().astype(np.float32),
        node_labels=data.y.cpu().numpy().astype(np.int64),
        train_nodes=np.arange(data.num_nodes, dtype=np.int32),
        validation_nodes=np.arange(data.num_nodes, dtype=np.int32),
        test_nodes=np.arange(data.num_nodes, dtype=np.int32),
        num_nodes=lambda: int(data.num_nodes), num_edges=lambda: int(edge_index.shape[1]),
    )
    raw = input_pipeline.add_reverse_edges(raw)
    raw = input_pipeline.subsample_graph(raw, config.max_degree, rng)
    graph, labels = input_pipeline.convert_to_graphstuple(raw)
    graph = input_pipeline.add_self_loops(graph)
    graph = input_pipeline.normalizations.normalize_edges_with_mask(
        graph, mask=None, adjacency_normalization=config.adjacency_normalization)
    return graph, labels


def _config(num_classes: int, steps: int, batch_size: int, noise_multiplier: float,
            evaluate_every: int, seed: int):
    _, _, _, _, _, ml_collections = _imports()
    config = ml_collections.ConfigDict()
    config.pad_subgraphs_to = 100
    config.multilabel = False
    config.adjacency_normalization = "inverse-degree"
    config.model = "gcn"
    config.latent_size = 100
    config.num_encoder_layers = 1
    config.num_message_passing_steps = 1
    config.num_decoder_layers = 1
    config.activation_fn = "tanh"
    config.num_classes = num_classes
    config.max_degree = 5
    config.differentially_private_training = True
    config.num_estimation_samples = min(100, batch_size)
    config.l2_norm_clip_percentile = 75
    config.training_noise_multiplier = noise_multiplier
    config.num_training_steps = steps
    config.max_training_epsilon = None
    config.evaluate_every_steps = max(1, evaluate_every)
    config.checkpoint_every_steps = max(1, evaluate_every)
    config.rng_seed = seed
    config.optimizer = "adam"
    config.learning_rate = 3e-3
    config.batch_size = batch_size
    config.dataset = "partitioned"
    config.dataset_path = ""
    return config


def run_partitioned(manifest: str | Path, result_path: str | Path, *, steps: int = 1,
                    batch_size: int = 32, noise_multiplier: float = 2.0,
                    evaluate_every: int = 50, seed: int = 0) -> dict[str, Any]:
    """Fit DP-GNN on train.pt only, then evaluate the immutable state per graph.

    The upstream ``train_and_evaluate`` loop is retained. Its data loader is
    temporarily replaced only at the call boundary with the train partition.
    """
    input_pipeline, privacy_accountants, train, jax, jnp, _ = _imports()
    manifest = Path(manifest)
    payload = json.loads(manifest.read_text())
    parts = {name: torch.load(manifest.parent / file, map_location="cpu", weights_only=False)["data"]
             for name, file in payload["partitions"].items()}
    config = _config(int(payload["num_classes"]), steps, batch_size,
                     noise_multiplier, evaluate_every, seed)
    dataset_rng = jax.random.PRNGKey(seed + 1)
    train_graph, train_labels = _graph(parts["train"], config, dataset_rng, input_pipeline)
    train_masks = {name: np.ones(parts["train"].num_nodes, dtype=bool)
                   for name in ("train", "validation", "test")}
    class _NoopWriter:
        def write_hparams(self, *_args, **_kwargs):
            pass

        def write_scalars(self, *_args, **_kwargs):
            pass

        def flush(self):
            pass

    class _NoopCheckpoint:
        def __init__(self, *_args, **_kwargs):
            pass

        def restore_or_initialize(self, state):
            return state

        def save(self, *_args, **_kwargs):
            pass

    original_dataset = input_pipeline.get_dataset
    original_writer = train.metric_writers.create_default_writer
    original_checkpoint = train.checkpoint.Checkpoint
    input_pipeline.get_dataset = lambda _config, _rng: (train_graph, train_labels, train_masks)
    # CLU's TensorFlow async writer and checkpoint implementation are
    # incompatible with the current TensorFlow profiler API. Both are ancillary
    # persistence/logging facilities, not DP-GNN algorithm code.
    train.metric_writers.create_default_writer = lambda *_args, **_kwargs: _NoopWriter()
    train.checkpoint.Checkpoint = _NoopCheckpoint
    try:
        state = train.train_and_evaluate(config, str(manifest.parent / "dpg nn-workdir"))
    finally:
        input_pipeline.get_dataset = original_dataset
        train.metric_writers.create_default_writer = original_writer
        train.checkpoint.Checkpoint = original_checkpoint
    metrics = {}
    for offset, name in enumerate(("val", "test"), start=2):
        graph, labels = _graph(parts[name], config, jax.random.PRNGKey(seed + offset), input_pipeline)
        logits = train.compute_logits(state, jax.tree.map(jnp.asarray, graph))
        one_hot = jax.nn.one_hot(jnp.asarray(labels), config.num_classes)
        _, accuracy = train.evaluate_predictions(logits, one_hot, jnp.ones(len(labels), dtype=bool))
        metrics[f"{name}_accuracy"] = float(accuracy)
    accountant = privacy_accountants.get_training_privacy_accountant(
        config, parts["train"].num_nodes, train.compute_max_terms_per_node(config))
    epsilon = float(accountant(steps))
    # This is the upstream GCN convention in get_training_privacy_accountant.
    delta = 1.0 / (10 * parts["train"].num_nodes)
    result = {"method": "dp_gnn", "validation_accuracy": metrics["val_accuracy"],
              "test_accuracy": metrics["test_accuracy"],
              "privacy": {"epsilon": epsilon, "delta": delta,
                          "accountant": "upstream.differentially_private_gnns.privacy_accountants",
                          "noise_multiplier": noise_multiplier, "composition_count": steps},
              "upstream": {"source": str(_source_root()), "retained": ["sampler.py", "train.py", "privacy_accountants.py"]}}
    Path(result_path).write_text(json.dumps(result, indent=2) + "\n")
    return result

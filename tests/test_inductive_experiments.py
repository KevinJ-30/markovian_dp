import math
from types import SimpleNamespace

import torch
import pytest
from torch_geometric.data import Data

from src.training.baselines import (
    BaselineConfig,
    BaselineTrainer,
    _LayerwiseNeighborSampler,
)
import src.experiments.run as run_module
import src.training.dpar as dpar_module
from src.experiments.run import _resolve_task_metadata
from src.models.objectives import _binary_auroc, _task_loss
from src.models.baselines import GraphSAGE
from src.training.dpar import (
    DPARConfig,
    DPARTrainer,
    _dpar_adjacency,
    _sample_train_partition,
    private_ista_ppr,
)
from src.processing.splits import load_or_create_inductive_split


def _graph():
    # Three classes, ten nodes each: all split partitions are non-empty.
    nodes = torch.arange(30)
    edge_index = torch.stack((nodes, torch.roll(nodes, shifts=-1)))
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return Data(x=torch.randn(30, 5), y=torch.arange(30) // 10, edge_index=edge_index)


def test_saved_partition_removes_cross_edges(tmp_path):
    split = load_or_create_inductive_split(_graph(), "unit", root=tmp_path, seed=9)
    reloaded = load_or_create_inductive_split(_graph(), "unit", root=tmp_path, seed=9)
    assert split.path == reloaded.path
    assert sum(part.stats["nodes"] for part in (split.train, split.val, split.test)) == 30
    assert all(part.node_ids.numel() == part.data.num_nodes for part in (split.train, split.val, split.test))


def test_private_baselines_run_on_train_partition(tmp_path):
    split = load_or_create_inductive_split(_graph(), "unit", root=tmp_path, seed=0)
    dpar = DPARTrainer(
        DPARConfig(
            epochs=1,
            hidden_size=4,
            topk=2,
            sampled_train_rate=None,
            sampled_train_nodes=2,
            ppr_num=1,
            batch_size=64,
            dropout=0.0,
            dp_sgd=True,
        ),
        "cpu",
    ).fit(split)
    mlp = BaselineTrainer(BaselineConfig(method="dp_mlp", epochs=1, hidden_size=4,
                                          batch_size=64, noise_multiplier=1.0), "cpu").fit(split)
    assert 0.0 <= dpar["test_accuracy"] <= 1.0
    assert 0.0 <= mlp["test_accuracy"] <= 1.0
    assert dpar["privacy"]["training"] is not None
    assert dpar["privacy"]["training"]["accountant"] == "dpar.upstream_rdp_accountant"
    assert mlp["privacy"]["epsilon"] is not None


def test_sampled_train_rate_rounds_up_and_separates_ppr_roots():
    data = _graph()
    partition = SimpleNamespace(data=data)
    sampled, roots, statistics = _sample_train_partition(
        partition,
        DPARConfig(sampled_train_rate=0.21, ppr_num=2),
        torch.Generator().manual_seed(3),
        torch.device("cpu"),
    )
    assert sampled.num_nodes == math.ceil(0.21 * data.num_nodes) == 7
    assert roots.numel() == 2
    assert torch.unique(roots).numel() == 2
    assert statistics["nodes"] == 7


def test_dpar_adjacency_replaces_diagonals_without_mutating_edges():
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 1, 2, 1, 2]]
    )
    original = edge_index.clone()
    adjacency = _dpar_adjacency(edge_index, 3, torch.device("cpu")).coalesce()
    assert torch.equal(edge_index, original)
    assert set(map(tuple, adjacency.indices().t().tolist())) == {
        (0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)
    }
    diagonal = adjacency.indices()[0] == adjacency.indices()[1]
    assert int(diagonal.sum()) == 3
    assert torch.all(adjacency.values() == 1)


def test_private_ista_releases_only_selected_rows_with_full_feature_context():
    data = _graph()
    roots = torch.tensor([7, 1, 4])
    ppr = private_ista_ppr(
        data.edge_index,
        data.num_nodes,
        roots,
        DPARConfig(topk=2),
        torch.device("cpu"),
    ).coalesce()
    assert ppr.shape == (roots.numel(), data.num_nodes)
    row_counts = torch.bincount(ppr.indices()[0], minlength=roots.numel())
    assert torch.all(row_counts <= 2)
    nonreleased = torch.ones(data.num_nodes, dtype=torch.bool)
    nonreleased[roots] = False
    neighbour_features = nonreleased.float().unsqueeze(1)
    assert torch.any(torch.sparse.mm(ppr, neighbour_features) > 0)
    column_mass = torch.zeros(data.num_nodes)
    column_mass.scatter_add_(0, ppr.indices()[1], ppr.values().abs())
    assert torch.allclose(
        column_mass[column_mass > 0],
        torch.ones_like(column_mass[column_mass > 0]),
        atol=1e-7,
        rtol=0.0,
    )


def _literal_released_ista(edge_index, num_nodes, roots, config):
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    adjacency[edge_index[0], edge_index[1]] = 1
    adjacency.fill_diagonal_(1)
    out_degree = (adjacency > 0).sum(dim=1).to(torch.float64)
    inverse_degree = adjacency.sum(dim=1).clamp_min(1e-12).reciprocal()
    dense = torch.zeros((roots.numel(), num_nodes), dtype=torch.float64)
    for row, root in enumerate(roots.tolist()):
        p_old = torch.zeros(num_nodes, dtype=torch.float64)
        p_new = torch.zeros(num_nodes, dtype=torch.float64)
        residual_old = torch.zeros(num_nodes, dtype=torch.float64)
        residual_old[root] = -config.alpha * inverse_degree[root]
        while residual_old.abs().max() > (1 + config.ista_epsilon) * config.rho * config.alpha:
            active = torch.where(p_old - residual_old >= config.rho * config.alpha)[0]
            active_set = set(active.tolist())
            delta_pk = -(residual_old[active] + config.rho * config.alpha)
            p_new[active] = p_old[active] + delta_pk
            residual_new = residual_old
            delta_full = torch.zeros(num_nodes, dtype=torch.float64)
            delta_full[active] = delta_pk
            for node in active.tolist():
                neighbours = torch.where(adjacency[node] > 0)[0].tolist()
                message = sum(
                    float(delta_full[other] / out_degree[other])
                    for other in neighbours
                    if other in active_set
                )
                residual_new[node] = (
                    (1 - 1 / out_degree[node]) * residual_old[node]
                    - config.rho * config.alpha / out_degree[node]
                    - 0.5 * (1 - config.alpha) * delta_full[node] / out_degree[node]
                    - 0.5 * (1 - config.alpha) * message / out_degree[node]
                )
            neighbour_set = {
                neighbour
                for node in active.tolist()
                for neighbour in torch.where(adjacency[node] > 0)[0].tolist()
                if neighbour not in active_set
            }
            for node in neighbour_set:
                neighbours = torch.where(adjacency[node] > 0)[0].tolist()
                message = sum(
                    float(delta_full[other] / out_degree[other])
                    for other in neighbours
                    if other in active_set
                )
                residual_new[node] = (
                    residual_old[node]
                    - 0.5 * (1 - config.alpha) * message / out_degree[node]
                )
            residual_old = residual_new
            p_old = p_new
        nonzero = torch.where(p_old != 0)[0]
        chosen = nonzero[torch.argsort(p_old[nonzero])[-min(config.topk, nonzero.numel()):]]
        dense[row, chosen] = p_old[chosen]
    mass = dense.abs().sum(dim=0)
    dense[:, mass > 0] /= mass[mass > 0]
    return dense


def test_private_ista_matches_released_aliased_recurrence():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
    )
    roots = torch.tensor([3, 1])
    config = DPARConfig(topk=4, rho=0.02, ista_epsilon=1e-4)
    actual = private_ista_ppr(
        edge_index, 4, roots, config, torch.device("cpu")
    ).to_dense()
    expected = _literal_released_ista(edge_index, 4, roots, config)
    assert torch.allclose(actual.to(torch.float64), expected, atol=1e-6, rtol=1e-6)


def test_dpar_target_budget_samples_graph_and_composes_total(tmp_path):
    split = load_or_create_inductive_split(_graph(), "target", root=tmp_path, seed=0)
    target = DPARTrainer(
        DPARConfig(
            epochs=1, hidden_size=4, topk=2,
            sampled_train_rate=None, sampled_train_nodes=8, ppr_num=2,
            batch_size=3, dropout=0.0,
            target_epsilon=8.0, target_delta=5e-4,
        ), "cpu",
    ).fit(split)
    privacy = target["privacy"]
    assert target["sampled_train_graph"]["nodes"] == 8
    assert privacy["ppr"]["accountant"] == "dpar.upstream_ppr_formula"
    assert privacy["ppr"]["composition_count"] == 2
    assert privacy["training"]["accountant"] == "dpar.upstream_rdp_accountant"
    assert privacy["training"]["sampling_probability"] == 1.0
    assert privacy["training"]["composition_count"] == 1
    assert privacy["total"]["epsilon"] <= 8.0
    assert privacy["total"]["delta"] == 5e-4
    assert target["calibration"]["sampled_train_nodes"] == 8
    assert target["calibration"]["ppr_releases"] == 2
    assert target["calibration"]["target_delta"] == 5e-4
    assert target["config"]["dp_ppr"] and target["config"]["dp_sgd"]

    fixed = DPARTrainer(
        DPARConfig(
            epochs=1, hidden_size=4, topk=2,
            sampled_train_rate=None, sampled_train_nodes=8, ppr_num=2,
            batch_size=3, dropout=0.0, dp_ppr=True, ppr_noise=0.7,
            ppr_delta=2e-5, dp_sgd=True, sgd_noise=1.2, sgd_delta=7e-4,
        ), "cpu",
    ).fit(split)
    assert fixed["config"]["ppr_noise"] == 0.7
    assert fixed["config"]["sgd_noise"] == 1.2
    assert fixed["privacy"]["training"]["delta"] == 7e-4
    assert fixed["privacy"]["training"]["sampling_probability"] == 1.0
    assert fixed["privacy"]["training"]["composition_count"] == 1
    assert "calibration" not in fixed


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_dpar_supervises_roots_only_but_uses_nonroot_features(monkeypatch, private, batch_size):
    roots = torch.tensor([3, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data = Data(
        x=torch.tensor([[1.0, -1.0], [2.0, 1.0], [-1.0, 3.0], [1.0, 2.0]]),
        y=torch.tensor([0, 1, 1, 0]),
        edge_index=edge_index,
    )
    held_out = SimpleNamespace(data=data.clone())

    def sample_roots(partition, config, generator, device):
        sampled = partition.data.clone().to(device)
        return sampled, roots.to(device), {"nodes": sampled.num_nodes}

    monkeypatch.setattr(dpar_module, "_sample_train_partition", sample_roots)

    class ObservedTrainer(DPARTrainer):
        def _evaluate(self, model, partition):
            result = super()._evaluate(model, partition)
            self.state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            with torch.no_grad():
                self.predictions = dpar_module.propagate_logits(
                    model(partition.data.x), partition.data.edge_index,
                    self.config.alpha, self.config.inference_steps,
                ).clone()
            return result

    def train(train_data):
        split = SimpleNamespace(
            train=SimpleNamespace(data=train_data, stats={"nodes": train_data.num_nodes}),
            val=held_out, test=held_out, num_classes=2,
        )
        trainer = ObservedTrainer(DPARConfig(
            epochs=3, hidden_size=8, topk=4, sampled_train_rate=1.0,
            ppr_num=2, batch_size=batch_size, dropout=0.0,
            learning_rate=0.02, dp_sgd=private, sgd_noise=0.1, seed=7,
        ))
        trainer.fit(split)
        return trainer

    original = train(data)
    changed_labels = data.clone()
    changed_labels.y[torch.tensor([0, 2])] = 1 - changed_labels.y[torch.tensor([0, 2])]
    relabeled = train(changed_labels)
    assert all(torch.equal(value, relabeled.state[name]) for name, value in original.state.items())
    assert torch.equal(original.predictions, relabeled.predictions)
    if batch_size >= roots.numel():
        roots = roots.flip(0)
        reordered = train(data)
        roots = roots.flip(0)
        assert all(
            torch.allclose(value, reordered.state[name])
            for name, value in original.state.items()
        )
        assert torch.allclose(original.predictions, reordered.predictions)

    changed_features = data.clone()
    changed_features.x[2] = torch.tensor([5.0, -7.0])
    new_context = train(changed_features)
    assert any(
        not torch.allclose(value, new_context.state[name])
        for name, value in original.state.items()
    )
    assert not torch.allclose(original.predictions, new_context.predictions)

    changed_root_labels = data.clone()
    changed_root_labels.y[roots] = 1 - changed_root_labels.y[roots]
    supervised = train(changed_root_labels)
    assert not torch.allclose(original.predictions, supervised.predictions)


def test_binary_objective_is_tie_correct_and_uses_one_logit():
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert _binary_auroc(labels, scores) == pytest.approx(0.75)
    logits = scores[:, None].requires_grad_()
    loss = _task_loss(logits, labels, multilabel=False, binary=True)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        scores, labels.float())
    torch.testing.assert_close(loss, expected)

    trainer = BaselineTrainer(BaselineConfig(method="mlp", binary=True))
    model = trainer._model(Data(x=torch.randn(4, 3)), num_classes=2)
    assert model(torch.randn(4, 3)).shape == (4, 1)


def test_baseline_scores_eval_mask_and_ignore_label_after_full_forward():
    observed = {}

    class FixedLogits(torch.nn.Module):
        def forward(self, x, edge_index=None):
            observed["nodes"] = x.size(0)
            logits = torch.zeros((x.size(0), 20), device=x.device)
            logits[1, 1] = 5.0
            logits[2, 0] = 5.0
            return logits

    data = Data(
        x=torch.randn(4, 3),
        y=torch.tensor([0, 1, 19, 0]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    partition = SimpleNamespace(
        data=data, eval_mask=torch.tensor([False, True, True, False]))
    trainer = BaselineTrainer(BaselineConfig(
        method="graphsage", metric_ignore_label=19))
    accuracy, macro_f1 = trainer._evaluate(FixedLogits(), partition)
    assert observed["nodes"] == 4
    assert accuracy == 1.0
    assert macro_f1 == 1.0


def test_hierarchical_graphsage_matches_untrimmed_sampled_loss():
    edge_index = torch.tensor([
        [1, 2, 3, 4, 5, 5, 6, 7, 7, 0],
        [0, 0, 0, 1, 1, 2, 2, 3, 4, 4],
    ])
    sampler = _LayerwiseNeighborSampler(edge_index, num_nodes=8)
    sampled = sampler.sample(
        torch.tensor([0, 2]),
        fanouts=[2, 2],
        generator=torch.Generator().manual_seed(4),
    )
    assert sampled.num_sampled_nodes[0] == 2
    assert len(sampled.num_sampled_nodes) == 3
    assert len(sampled.num_sampled_edges) == 2

    offset = 0
    for hop, edge_count in enumerate(sampled.num_sampled_edges):
        hop_edges = sampled.edge_index[:, offset:offset + edge_count]
        counts = torch.bincount(
            hop_edges[1], minlength=sampled.node_ids.numel())
        frontier_start = sum(sampled.num_sampled_nodes[:hop])
        frontier_size = sampled.num_sampled_nodes[hop]
        if frontier_size:
            assert int(
                counts[frontier_start:frontier_start + frontier_size].max()
            ) <= 2
        offset += edge_count
    model = GraphSAGE(inputs=3, classes=1, hidden=5, layers=2, dropout=0.0)
    model.eval()
    features = torch.randn(sampled.node_ids.numel(), 3)
    neighbor_logits = model.forward_sampled(
        features,
        sampled.edge_index,
        sampled.num_sampled_nodes,
        sampled.num_sampled_edges,
        hierarchical=False,
    )
    hierarchical_logits = model.forward_sampled(
        features,
        sampled.edge_index,
        sampled.num_sampled_nodes,
        sampled.num_sampled_edges,
        hierarchical=True,
    )
    torch.testing.assert_close(hierarchical_logits, neighbor_logits)



@pytest.mark.parametrize("sampling", ["neighbor", "hierarchical"])
def test_graphsage_sampling_modes_train(sampling, tmp_path):
    split = load_or_create_inductive_split(
        _graph(), f"graphsage-{sampling}", root=tmp_path, seed=3
    )
    result = BaselineTrainer(
        BaselineConfig(
            method="graphsage",
            graphsage_sampling=sampling,
            max_fanout=2,
            layers=2,
            hidden_size=4,
            batch_size=4,
            epochs=1,
            dropout=0.0,
        ),
        "cpu",
    ).fit(split)
    assert result["config"]["graphsage_sampling"] == sampling
    assert 0.0 <= result["validation_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0

def test_dataset_task_metadata_is_authoritative():
    dataset = SimpleNamespace(
        task_type="BINARY", primary_metric="auroc", metric_ignore_label=None)
    resolved = _resolve_task_metadata(dataset, {})
    assert resolved["binary"]
    assert resolved["primary_metric"] == "auroc"
    with pytest.raises(ValueError, match="conflicts"):
        _resolve_task_metadata(dataset, {"binary": False})


def test_domain_dataset_rejects_unsupported_heterpoisson(monkeypatch):
    dataset = SimpleNamespace(
        domain_dataset=True,
        task_type="MULTICLASS",
        primary_metric="accuracy",
        metric_ignore_label=None,
    )
    data = Data(
        x=torch.zeros((3, 2)),
        y=torch.tensor([0, 1, 0]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    monkeypatch.setattr(
        run_module, "load_dataset", lambda *args, **kwargs: (dataset, data))
    with pytest.raises(ValueError, match="does not support domain datasets"):
        run_module.run({
            "dataset": "facebook100",
            "method": "heterpoisson",
            "device": "cpu",
        })

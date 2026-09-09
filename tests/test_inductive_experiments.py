import torch
from torch_geometric.data import Data

from src.experiments.baselines import BaselineConfig, BaselineTrainer
from src.experiments.dpar import DPARConfig, DPARTrainer, private_ista_ppr
from src.experiments.inductive import load_or_create_inductive_split


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
    dpar = DPARTrainer(DPARConfig(epochs=1, hidden_size=4, topk=2, sampled_train_nodes=2,
                                   batch_size=64, dropout=0.0, dp_sgd=True), "cpu").fit(split)
    mlp = BaselineTrainer(BaselineConfig(method="dp_mlp", epochs=1, hidden_size=4,
                                          batch_size=64, noise_multiplier=1.0), "cpu").fit(split)
    assert 0.0 <= dpar["test_accuracy"] <= 1.0
    assert 0.0 <= mlp["test_accuracy"] <= 1.0
    assert dpar["privacy"]["training"] is not None
    assert dpar["privacy"]["training"]["accountant"] == "dpar.upstream_rdp_accountant"
    assert mlp["privacy"]["epsilon"] is not None


def test_private_ista_returns_topk_rows():
    data = _graph()
    ppr = private_ista_ppr(data.edge_index, data.num_nodes,
                           DPARConfig(topk=2), torch.device("cpu"))
    counts = torch.bincount(ppr.coalesce().indices()[0], minlength=data.num_nodes)
    assert torch.all(counts <= 2)


def test_private_ista_releases_every_row_and_column_clips():
    data = _graph()
    ppr = private_ista_ppr(
        data.edge_index, data.num_nodes,
        DPARConfig(topk=2, ppr_column_clip=0.05), torch.device("cpu"),
    ).coalesce()
    row_counts = torch.bincount(ppr.indices()[0], minlength=data.num_nodes)
    column_mass = torch.zeros(data.num_nodes)
    column_mass.scatter_add_(0, ppr.indices()[1], ppr.values().abs())
    assert torch.all(row_counts > 0)
    assert torch.all(row_counts <= 2)
    assert torch.all(column_mass <= 0.05 + 1e-7)


def test_dpar_target_budget_samples_graph_and_composes_total(tmp_path):
    split = load_or_create_inductive_split(_graph(), "target", root=tmp_path, seed=0)
    target = DPARTrainer(
        DPARConfig(
            epochs=1, hidden_size=4, topk=2, sampled_train_nodes=8,
            batch_size=8, dropout=0.0, ppr_column_clip=0.05,
            target_epsilon=8.0, target_delta=5e-4,
        ), "cpu",
    ).fit(split)
    privacy = target["privacy"]
    assert target["sampled_train_graph"]["nodes"] == 8
    assert privacy["ppr"]["accountant"] == "dpar.upstream_ppr_formula"
    assert privacy["training"]["accountant"] == "dpar.upstream_rdp_accountant"
    assert privacy["total"]["accountant"] == "dpar.paper_theorem2_composition"
    assert privacy["total"]["epsilon"] <= 8.0
    assert privacy["total"]["delta"] == 5e-4
    assert target["calibration"]["ppr_releases"] == 8
    assert target["calibration"]["target_delta"] == 5e-4
    assert target["config"]["dp_ppr"] and target["config"]["dp_sgd"]

    fixed = DPARTrainer(
        DPARConfig(
            epochs=1, hidden_size=4, topk=2, sampled_train_nodes=8,
            batch_size=8, dropout=0.0, dp_ppr=True, ppr_noise=0.7,
            ppr_delta=2e-5, dp_sgd=True, sgd_noise=1.2, sgd_delta=7e-4,
        ), "cpu",
    ).fit(split)
    assert fixed["config"]["ppr_noise"] == 0.7
    assert fixed["config"]["sgd_noise"] == 1.2
    assert fixed["privacy"]["training"]["delta"] == 7e-4
    assert "calibration" not in fixed

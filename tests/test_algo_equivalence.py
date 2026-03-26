"""
Test that Algorithms 1 and 2 produce identical gradients.

Sinks in Algorithm 1 receive messages but never send any (no outgoing edges
in the directed subgraph). Since loss is computed only on bin-k training nodes,
sink nodes cannot affect the forward pass, loss, or gradients. Therefore
running the same bin assignment through both algorithms must yield the same
per-parameter gradients.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from src.models.gcn import SubgraphGCN


def test_algo1_algo2_gradient_equivalence():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    num_nodes = data.num_nodes
    edge_index = data.edge_index
    num_bins = 4

    # Fix a single bin assignment so both algorithms see the same randomness
    torch.manual_seed(123)
    bin_assignments = torch.randint(0, num_bins, (num_nodes,))

    # --- Algorithm 1: edges where source is in bin k ---
    def algo1_subgraphs(bin_assignments, edge_index, num_bins):
        result = []
        for k in range(num_bins):
            bin_mask = (bin_assignments == k)
            edge_mask = bin_mask[edge_index[0]]
            result.append((bin_mask, edge_index[:, edge_mask]))
        return result

    # --- Algorithm 2: edges where BOTH endpoints are in bin k ---
    def algo2_subgraphs(bin_assignments, edge_index, num_bins):
        result = []
        for k in range(num_bins):
            bin_mask = (bin_assignments == k)
            edge_mask = bin_mask[edge_index[0]] & bin_mask[edge_index[1]]
            result.append((bin_mask, edge_index[:, edge_mask]))
        return result

    partitions_1 = algo1_subgraphs(bin_assignments, edge_index, num_bins)
    partitions_2 = algo2_subgraphs(bin_assignments, edge_index, num_bins)

    # Both should have the same bin_masks
    for k in range(num_bins):
        assert torch.equal(partitions_1[k][0], partitions_2[k][0]), \
            f"Bin masks differ for bin {k}"

    # Algo 1 should have >= as many edges per bin (it keeps sink edges)
    for k in range(num_bins):
        n1 = partitions_1[k][1].size(1)
        n2 = partitions_2[k][1].size(1)
        assert n1 >= n2, f"Bin {k}: algo1 has {n1} edges < algo2 has {n2}"
        print(f"  Bin {k}: algo1 {n1} edges, algo2 {n2} edges (diff: {n1-n2} sink edges)")

    # Now check gradient equivalence: train the same model on each partition
    # and compare gradients. Use eval mode to disable dropout so the test is
    # purely deterministic (the equivalence holds with dropout too, but would
    # require identical RNG state which is hard to guarantee across different-
    # sized edge_index tensors).
    for k in range(num_bins):
        bin_mask_k = partitions_1[k][0]
        train_in_bin = bin_mask_k & data.train_mask
        if not train_in_bin.any():
            continue

        # Create identical models
        torch.manual_seed(42)
        model1 = SubgraphGCN(dataset.num_features, 64, dataset.num_classes)
        torch.manual_seed(42)
        model2 = SubgraphGCN(dataset.num_features, 64, dataset.num_classes)

        # Use eval mode to disable dropout (deterministic comparison)
        model1.eval()
        model2.eval()

        # Forward + backward through bin k with Algorithm 1
        out1 = model1(data.x, partitions_1[k][1])
        loss1 = F.nll_loss(out1[train_in_bin], data.y[train_in_bin], reduction='sum')
        loss1.backward()

        # Forward + backward through bin k with Algorithm 2
        out2 = model2(data.x, partitions_2[k][1])
        loss2 = F.nll_loss(out2[train_in_bin], data.y[train_in_bin], reduction='sum')
        loss2.backward()

        # Losses must match
        assert torch.allclose(loss1, loss2, atol=1e-5), \
            f"Bin {k}: losses differ — algo1={loss1.item():.6f}, algo2={loss2.item():.6f}"

        # Gradients must match for every parameter
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5), \
                f"Bin {k}, param {n1}: gradient mismatch. " \
                f"Max diff = {(p1.grad - p2.grad).abs().max().item():.2e}"

        print(f"  Bin {k}: loss1={loss1.item():.4f}, loss2={loss2.item():.4f} -- match")

    print("PASSED: Algorithms 1 and 2 produce identical losses and gradients.")


def test_algo3_subsampling():
    """Verify Algorithm 3 drops nodes and produces fewer edges than Algorithm 2."""
    from src.algorithms.algo2 import RemoveSinks
    from src.algorithms.algo3 import RemoveSinksSubsampled

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    num_bins = 4
    device = data.x.device

    # Algo 2 (no subsampling)
    torch.manual_seed(99)
    parts_2 = RemoveSinks().partition(data.edge_index, data.num_nodes, num_bins, device)

    # Algo 3 with high p_perp — should drop many nodes
    torch.manual_seed(99)
    parts_3 = RemoveSinksSubsampled(subsample_prob=0.5).partition(
        data.edge_index, data.num_nodes, num_bins, device
    )

    total_nodes_2 = sum(m.sum().item() for m, _ in parts_2)
    total_nodes_3 = sum(m.sum().item() for m, _ in parts_3)
    total_edges_2 = sum(ei.size(1) for _, ei in parts_2)
    total_edges_3 = sum(ei.size(1) for _, ei in parts_3)

    assert total_nodes_3 < total_nodes_2, \
        f"Algo 3 should have fewer nodes ({total_nodes_3} >= {total_nodes_2})"
    assert total_edges_3 < total_edges_2, \
        f"Algo 3 should have fewer edges ({total_edges_3} >= {total_edges_2})"

    # With p_perp=0, Algo 3 should behave identically to Algo 2
    torch.manual_seed(99)
    parts_3_noop = RemoveSinksSubsampled(subsample_prob=0.0).partition(
        data.edge_index, data.num_nodes, num_bins, device
    )
    torch.manual_seed(99)
    parts_2_same = RemoveSinks().partition(data.edge_index, data.num_nodes, num_bins, device)

    for k in range(num_bins):
        assert torch.equal(parts_3_noop[k][0], parts_2_same[k][0]), \
            f"Bin {k}: masks differ with p_perp=0"
        assert torch.equal(parts_3_noop[k][1], parts_2_same[k][1]), \
            f"Bin {k}: edges differ with p_perp=0"

    print(f"PASSED: Algo 3 subsampling works "
          f"(nodes: {total_nodes_2} -> {total_nodes_3}, "
          f"edges: {total_edges_2} -> {total_edges_3}). "
          f"p_perp=0 matches Algo 2 exactly.")


if __name__ == '__main__':
    test_algo1_algo2_gradient_equivalence()
    print()
    test_algo3_subsampling()
    print("\nAll tests passed.")

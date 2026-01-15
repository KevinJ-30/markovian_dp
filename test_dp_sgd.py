"""
Test script for DP-SGD GNN implementation.

Tests each component incrementally to identify issues early.
"""

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from src.models import NodeGCN
from src.dp_sgd import DPSGD_GNN


def test_single_batch():
    """Test one batch through the entire pipeline."""
    print("=" * 70)
    print("DP-SGD GNN Component Testing")
    print("=" * 70)

    print("\n[1/7] Loading dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    print(f"  ✓ Loaded Cora: {data.num_nodes} nodes, {data.num_edges} edges")

    print("\n[2/7] Creating NeighborLoader...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=4,  # Small batch for testing
        input_nodes=data.train_mask,
        shuffle=False,
    )
    print("  ✓ Created loader with batch_size=4, num_neighbors=[10, 5]")

    print("\n[3/7] Creating model...")
    device = 'cpu'
    model = NodeGCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes
    ).to(device)
    print(f"  ✓ Created NodeGCN model on {device}")

    print("\n[4/7] Creating DP trainer...")
    dp_trainer = DPSGD_GNN(
        model=model,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        device=device
    )
    print("  ✓ Created DPSGD_GNN trainer")

    print("\n[5/7] Getting first batch...")
    batch = next(iter(train_loader))
    print(f"  ✓ Got batch:")
    print(f"    batch_size: {batch.batch_size}")
    print(f"    num_nodes: {batch.num_nodes}")
    print(f"    num_edges: {batch.num_edges}")
    print(f"    has n_id: {hasattr(batch, 'n_id')}")
    print(f"    has batch: {hasattr(batch, 'batch')}")
    print(f"    has y: {hasattr(batch, 'y')}")

    print("\n[6/7] Testing stack_subgraphs...")
    try:
        x_stacked, edge_idx_stacked, num_nodes, targets = dp_trainer.stack_subgraphs(
            batch, max_nodes=100, max_edges=200
        )
        print("  ✓ stack_subgraphs succeeded:")
        print(f"    x_stacked shape: {x_stacked.shape}")
        print(f"    edge_idx_stacked shape: {edge_idx_stacked.shape}")
        print(f"    num_nodes: {num_nodes.tolist()}")
        print(f"    targets shape: {targets.shape}")
        print(f"    targets: {targets.tolist()}")
    except Exception as e:
        print(f"  ✗ stack_subgraphs failed: {e}")
        raise

    print("\n[7/7] Testing full training step...")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = dp_trainer.step(batch, optimizer)
        print(f"  ✓ Training step succeeded:")
        print(f"    Loss: {loss:.4f}")
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        raise

    print("\n" + "=" * 70)
    print("✓ All Tests Passed!")
    print("=" * 70)

    return True


def test_multiple_batches():
    """Test multiple batches to ensure consistency."""
    print("\n" + "=" * 70)
    print("Testing Multiple Batches")
    print("=" * 70)

    print("\nLoading dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    print("Creating loader...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=8,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    print("Creating model...")
    device = 'cpu'
    model = NodeGCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes
    ).to(device)

    print("Creating DP trainer...")
    dp_trainer = DPSGD_GNN(
        model=model,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        device=device
    )

    print("Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nProcessing batches...")
    num_batches_to_test = 5
    losses = []

    for i, batch in enumerate(train_loader):
        if i >= num_batches_to_test:
            break

        try:
            loss = dp_trainer.step(batch, optimizer)
            losses.append(loss)
            print(f"  Batch {i+1}/{num_batches_to_test}: loss = {loss:.4f}")
        except Exception as e:
            print(f"  ✗ Batch {i+1} failed: {e}")
            raise

    print(f"\n✓ Successfully processed {num_batches_to_test} batches")
    print(f"  Average loss: {sum(losses)/len(losses):.4f}")
    print(f"  Loss range: [{min(losses):.4f}, {max(losses):.4f}]")

    return True


if __name__ == "__main__":
    try:
        # Run basic tests
        test_single_batch()

        # Run multi-batch tests
        test_multiple_batches()

        print("\n" + "=" * 70)
        print("✓✓✓ All Tests Completed Successfully! ✓✓✓")
        print("=" * 70)
        print("\nYou can now run train.py to train the model with DP-SGD!")

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗✗✗ Tests Failed ✗✗✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

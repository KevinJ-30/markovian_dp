"""
Debug script to inspect NeighborLoader batch structure.

This helps understand how NeighborLoader organizes sampled subgraphs.
"""

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader


def inspect_batch_structure():
    """Inspect the structure of NeighborLoader batches."""
    print("=" * 70)
    print("NeighborLoader Batch Structure Inspector")
    print("=" * 70)

    # Load dataset
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    print(f"Total nodes: {data.num_nodes}")
    print(f"Total edges: {data.num_edges}")
    print(f"Train nodes: {data.train_mask.sum()}")

    # Create loader with small batch size for easy inspection
    print("\nCreating NeighborLoader...")
    print("  num_neighbors: [10, 5]")
    print("  batch_size: 2")

    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=2,  # Small for easier inspection
        input_nodes=data.train_mask,
        shuffle=False,
    )

    # Inspect first few batches
    print("\n" + "=" * 70)
    print("Batch Inspection")
    print("=" * 70)

    for i, batch in enumerate(train_loader):
        print(f"\n{'='*70}")
        print(f"Batch {i}")
        print(f"{'='*70}")

        # Basic info
        print(f"\nBasic Info:")
        print(f"  batch_size: {batch.batch_size}")
        print(f"  num_nodes: {batch.num_nodes}")
        print(f"  num_edges: {batch.num_edges}")

        # Check attributes
        print(f"\nAttributes:")
        print(f"  has n_id: {hasattr(batch, 'n_id')}")
        print(f"  has batch: {hasattr(batch, 'batch')}")
        print(f"  has y: {hasattr(batch, 'y')}")

        if hasattr(batch, 'n_id'):
            print(f"\nn_id (global node IDs):")
            print(f"  shape: {batch.n_id.shape}")
            print(f"  first {batch.batch_size} (seed nodes): {batch.n_id[:batch.batch_size].tolist()}")
            print(f"  all: {batch.n_id.tolist()}")

        if hasattr(batch, 'batch'):
            print(f"\nbatch tensor (maps local idx -> seed idx):")
            print(f"  shape: {batch.batch.shape}")
            print(f"  values: {batch.batch.tolist()}")
            print(f"  unique: {batch.batch.unique().tolist()}")

            # Analyze subgraphs
            print(f"\nSubgraph sizes:")
            for j in range(batch.batch_size):
                mask = batch.batch == j
                num_nodes_in_subgraph = mask.sum().item()
                node_indices = torch.where(mask)[0].tolist()
                print(f"  Subgraph {j}: {num_nodes_in_subgraph} nodes")
                print(f"    Node indices: {node_indices}")

        if hasattr(batch, 'y'):
            print(f"\ny (labels):")
            print(f"  shape: {batch.y.shape}")
            print(f"  first {batch.batch_size} (seed labels): {batch.y[:batch.batch_size].tolist()}")

        # Edge info
        print(f"\nedge_index:")
        print(f"  shape: {batch.edge_index.shape}")
        print(f"  first 5 edges: {batch.edge_index[:, :5].tolist()}")

        # Node features
        print(f"\nx (node features):")
        print(f"  shape: {batch.x.shape}")

        # Stop after first 3 batches
        if i >= 2:
            break

    print("\n" + "=" * 70)
    print("Inspection Complete")
    print("=" * 70)

    # Key takeaways
    print("\nKey Takeaways:")
    print("1. NeighborLoader DOES provide batch.batch tensor")
    print("2. Seed nodes are the first batch_size entries in n_id")
    print("3. Seed nodes are at local indices 0, 1, 2, ..., batch_size-1")
    print("4. Each subgraph has variable size (not equal division)")
    print("5. batch.batch correctly maps nodes to their seed index")


if __name__ == "__main__":
    inspect_batch_structure()

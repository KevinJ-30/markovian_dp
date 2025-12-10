import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

from src.models import NodeGCN
from src.dp_sgd import DPSGD_GNN


def train_dp(
    data,
    model,
    dp_trainer,
    optimizer,
    train_loader,
    num_epochs: int = 10
):
    """
    Train with DP-SGD.
    
    Args:
        data: Full graph data
        model: GNN model
        dp_trainer: DPSGD_GNN trainer
        optimizer: PyTorch optimizer
        train_loader: NeighborLoader
        num_epochs: Number of epochs
    """
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            loss = dp_trainer.step(batch, optimizer)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


@torch.no_grad()
def test(data, model):
    """
    Evaluate model.
    
    Args:
        data: Full graph data
        model: GNN model
        
    Returns:
        train_acc, test_acc
    """
    model.eval()
    
    # Full graph forward pass
    out = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long))
    pred = out.argmax(dim=1)
    
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    
    return train_acc.item(), test_acc.item()


def main():
    """
    Main training loop.
    """
    print("=" * 70)
    print("DP-SGD for GNNs - Node-Level Privacy")
    print("=" * 70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load dataset (Cora - standard citation network)
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {dataset.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train nodes: {data.train_mask.sum()}")
    print(f"  Test nodes: {data.test_mask.sum()}")
    
    # Create NeighborLoader (following Jan's suggestion)
    print("\nCreating NeighborLoader...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],  # 2-hop sampling: 10 neighbors in layer 1, 5 in layer 2
        batch_size=32,
        input_nodes=data.train_mask,  # Only sample from training nodes
        shuffle=True,
        #disjoint=True,
    )
    print(f"  Sampling: 2-hop neighborhoods")
    print(f"  num_neighbors: [10, 5]")
    print(f"  Batch size: 32")
    
    # Create model
    print("\nInitializing model...")
    model = NodeGCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes
    ).to(device)
    
    # DP-SGD parameters
    max_grad_norm = 1.0
    noise_multiplier = 1.0
    learning_rate = 0.01
    
    print("\nDP-SGD configuration:")
    print(f"  Clipping norm (C): {max_grad_norm}")
    print(f"  Noise multiplier (σ): {noise_multiplier}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create DP trainer
    dp_trainer = DPSGD_GNN(
        model=model,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Baseline accuracy (before training) - shows random initialization performance
    # Useful for debugging but can be removed if desired
    print("\n" + "=" * 70)
    print("Before Training (Random Initialization)")
    print("=" * 70)
    train_acc, test_acc = test(data, model)
    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Train with DP-SGD
    print("\n" + "=" * 70)
    print("Training with DP-SGD")
    print("=" * 70)
    train_dp(
        data=data,
        model=model,
        dp_trainer=dp_trainer,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=10
    )
    
    # Final accuracy
    print("\n" + "=" * 70)
    print("After Training")
    print("=" * 70)
    train_acc, test_acc = test(data, model)
    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Implement privacy accounting (compute ε, δ)")
    print("  2. Compare DP vs non-DP baseline")
    print("  3. Experiment with different C and σ")
    print("  4. Test on other datasets")


if __name__ == "__main__":
    main()
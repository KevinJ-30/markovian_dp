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
    """Main training loop."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=32,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    
    model = NodeGCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes
    ).to(device)
    
    max_grad_norm = 1.0
    noise_multiplier = 1.0
    learning_rate = 0.01
    
    dp_trainer = DPSGD_GNN(
        model=model,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Before Training:")
    train_acc, test_acc = test(data, model)
    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    train_dp(
        data=data,
        model=model,
        dp_trainer=dp_trainer,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=10
    )
    
    print("\nAfter Training:")
    train_acc, test_acc = test(data, model)
    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
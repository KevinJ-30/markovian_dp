import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from typing import Dict, Optional
from torch_geometric.loader import NeighborLoader


class DPSGD_GNN:
    """
    Differentially Private SGD for Graph Neural Networks.
    
    Uses torch.func (functional_call + grad + vmap) to compute per-sample gradients.
    Follows the stacking approach: convert batched PyG graph -> stacked tensors -> vmap.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Initialize DP-SGD trainer.
        
        Args:
            model: PyTorch Geometric GNN model
            max_grad_norm: Clipping threshold C
            noise_multiplier: Noise multiplier σ
            device: Device to run on
        """
        self.model = model.to(device)
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.device = device
        
        self.params = {k: v.detach() for k, v in model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in model.named_buffers()}
        
        print(f"Initialized DP-SGD:")
        print(f"  Clipping norm (C): {max_grad_norm}")
        print(f"  Noise multiplier (σ): {noise_multiplier}")
        print(f"  Parameters: {sum(p.numel() for p in self.params.values())}")
    
    def stack_subgraphs(self, batch, max_nodes: int, max_edges: int):
        """
        Stack batched PyG graph into separate tensors for vmap.
        
        Converts:
            batch.x: [total_nodes, features]
            batch.batch: [total_nodes] (which seed each node belongs to)
        
        To:
            x_stacked: [batch_size, max_nodes, features]
            edge_index_stacked: [batch_size, 2, max_edges]
        
        Args:
            batch: PyG Batch from NeighborLoader
            max_nodes: Maximum nodes to pad to
            max_edges: Maximum edges to pad to
            
        Returns:
            x_stacked, edge_index_stacked, num_nodes_per_graph, targets
        """
        batch_size = batch.batch_size
        num_features = batch.x.size(1)
        device = batch.x.device
        
        # Initialize stacked tensors
        x_stacked = torch.zeros(
            (batch_size, max_nodes, num_features), 
            device=device
        )
        edge_index_stacked = torch.zeros(
            (batch_size, 2, max_edges), 
            dtype=torch.long, 
            device=device
        )
        num_nodes_per_graph = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            mask = (batch.batch == i)
            nodes_in_subgraph = mask.nonzero().squeeze(-1)
            num_nodes = len(nodes_in_subgraph)
            num_nodes_per_graph[i] = num_nodes
            
            if num_nodes > 0:
                x_stacked[i, :num_nodes] = batch.x[nodes_in_subgraph]
            
            edge_index = batch.edge_index
            src_in = torch.isin(edge_index[0], nodes_in_subgraph)
            dst_in = torch.isin(edge_index[1], nodes_in_subgraph)
            edge_mask = src_in & dst_in
            edges_in_subgraph = edge_index[:, edge_mask]
            
            if edges_in_subgraph.numel() > 0:
                mapping = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
                mapping[nodes_in_subgraph] = torch.arange(num_nodes, device=device)
                edges_remapped = mapping[edges_in_subgraph]
                
                num_edges = min(edges_remapped.size(1), max_edges)
                edge_index_stacked[i, :, :num_edges] = edges_remapped[:, :num_edges]
        
        targets = batch.y[:batch_size] if hasattr(batch, 'y') else None
        
        return x_stacked, edge_index_stacked, num_nodes_per_graph, targets
    
    def compute_loss_single_subgraph(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x_single: torch.Tensor,
        edge_index_single: torch.Tensor,
        num_nodes: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for ONE subgraph (one row from stacked tensors).
        
        This function will be vectorized by vmap.
        
        Args:
            params: Model parameters
            buffers: Model buffers
            x_single: [max_nodes, features] - node features (padded)
            edge_index_single: [2, max_edges] - edges (padded)
            num_nodes: scalar - actual number of nodes
            target: scalar - label
            
        Returns:
            loss: scalar
        """
        x = x_single[:num_nodes]
        
        valid_mask = (edge_index_single[0] < num_nodes) & (edge_index_single[1] < num_nodes)
        edge_index = edge_index_single[:, valid_mask]
        
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        
        output = functional_call(
            self.model,
            (params, buffers),
            args=(x, edge_index, batch)
        )
        
        loss = F.cross_entropy(output, target.unsqueeze(0))
        return loss
    
    def compute_per_sample_gradients(self, batch, max_nodes: int = 100, max_edges: int = 500):
        """
        Compute per-sample gradients using vmap + stacking.
        
        Args:
            batch: PyG Batch from NeighborLoader
            max_nodes: Max nodes to pad to
            max_edges: Max edges to pad to
            
        Returns:
            per_sample_grads: Dict[str, Tensor] with shape [batch_size, *param_shape]
        """
        x_stacked, edge_idx_stacked, num_nodes, targets = self.stack_subgraphs(
            batch, max_nodes, max_edges
        )
        
        grad_fn = grad(self.compute_loss_single_subgraph)
        vmap_grad_fn = vmap(grad_fn, in_dims=(None, None, 0, 0, 0, 0))
        
        per_sample_grads = vmap_grad_fn(
            self.params,
            self.buffers,
            x_stacked,
            edge_idx_stacked,
            num_nodes,
            targets
        )
        
        return per_sample_grads
    
    def clip_gradients(self, per_sample_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip each sample's gradient to max norm C.
        
        Args:
            per_sample_grads: Dict with shape [batch_size, *param_shape]
            
        Returns:
            clipped_grads: Dict with same shape
        """
        batch_size = next(iter(per_sample_grads.values())).shape[0]
        
        sq_norms = torch.zeros(batch_size, device=self.device)
        for grads in per_sample_grads.values():
            flat = grads.reshape(batch_size, -1)
            sq_norms += (flat ** 2).sum(dim=1)
        
        norms = torch.sqrt(sq_norms)
        clip_factors = self.max_grad_norm / (norms + 1e-6)
        clip_factors = torch.clamp(clip_factors, max=1.0)
        
        clipped_grads = {}
        for name, grads in per_sample_grads.items():
            shape = [batch_size] + [1] * (grads.ndim - 1)
            clipped_grads[name] = grads * clip_factors.view(shape)
        
        return clipped_grads
    
    def add_noise(self, grads: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Add calibrated Gaussian noise.
        
        Args:
            grads: Aggregated gradients
            batch_size: Batch size
            
        Returns:
            noisy_grads: Gradients with noise added
        """
        noise_scale = self.noise_multiplier * self.max_grad_norm / batch_size
        
        noisy_grads = {}
        for name, grad in grads.items():
            noise = torch.randn_like(grad) * noise_scale
            noisy_grads[name] = grad + noise
        
        return noisy_grads
    
    def step(self, batch, optimizer) -> float:
        """
        Perform one DP-SGD training step.
        
        Args:
            batch: PyG Batch from NeighborLoader
            optimizer: PyTorch optimizer
            
        Returns:
            avg_loss: Average loss over batch
        """
        batch_size = batch.batch_size
        
        per_sample_grads = self.compute_per_sample_gradients(batch)
        clipped_grads = self.clip_gradients(per_sample_grads)
        aggregated = {k: v.mean(dim=0) for k, v in clipped_grads.items()}
        noisy_grads = self.add_noise(aggregated, batch_size)
        
        optimizer.zero_grad()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.grad = noisy_grads[name]
        optimizer.step()
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(output[:batch_size], batch.y[:batch_size])
            self.model.train()
        
        return loss.item()
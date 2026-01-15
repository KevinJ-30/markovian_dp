"""
DP-SGD for Graph Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch_geometric.loader import NeighborLoader


class DPSGD_GNN:
    """
    Differentially Private SGD for Graph Neural Networks.
    
    Computes per-sample gradients by looping over samples, then applies
    gradient clipping and noise addition for differential privacy.
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
    
    def stack_subgraphs(self, batch, max_nodes: int, max_edges: int):
        """
        Stack batched PyG graph into separate tensors for per-sample processing.
        
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
        
        # Validate required attributes
        if not hasattr(batch, 'n_id') or batch.n_id is None:
            raise ValueError("NeighborLoader must provide 'n_id' attribute. Check your PyG installation.")

        # Get or create batch tensor
        # NeighborLoader structure: first batch_size nodes are seeds, rest are neighbors
        # We need to assign each node to its corresponding seed
        if hasattr(batch, 'batch') and batch.batch is not None:
            batch_tensor = batch.batch
        else:
            # Create batch tensor manually
            # In NeighborLoader, the first batch_size nodes are the seed nodes
            # Each seed node's neighborhood needs to be identified
            # For now, we'll use a simple heuristic: distribute nodes roughly equally
            # This is approximate but should work for most cases
            batch_tensor = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)

            # The first batch_size nodes are seeds, assign them to their respective batches
            batch_tensor[:batch_size] = torch.arange(batch_size, device=device)

            # For the remaining nodes (neighbors), we need to figure out which seed they belong to
            # We'll use edge connectivity: a neighbor belongs to the seed it connects to
            if batch.num_nodes > batch_size:
                remaining_nodes = torch.arange(batch_size, batch.num_nodes, device=device)
                # For each remaining node, find which seed nodes it's connected to
                edge_index = batch.edge_index
                for node_idx in remaining_nodes:
                    # Find edges involving this node
                    edges_to_node = (edge_index[1] == node_idx) | (edge_index[0] == node_idx)
                    if edges_to_node.any():
                        connected_nodes = torch.cat([
                            edge_index[0, edges_to_node],
                            edge_index[1, edges_to_node]
                        ]).unique()
                        # Find first seed node in connected nodes
                        seed_connections = connected_nodes[connected_nodes < batch_size]
                        if len(seed_connections) > 0:
                            batch_tensor[node_idx] = seed_connections[0]
                        else:
                            # If not connected to any seed, assign to first seed (fallback)
                            batch_tensor[node_idx] = 0

        # Seed nodes are the first batch_size nodes in NeighborLoader batches
        # These correspond to the input_nodes we're sampling around
        seed_node_local_indices = torch.arange(batch_size, device=device, dtype=torch.long)
        
        for i in range(batch_size):
            mask = (batch_tensor == i)
            node_indices = torch.where(mask)[0]

            if len(node_indices) == 0:
                num_nodes_per_graph[i] = 0
                continue

            # For node-level DP, we need the seed node to be at index 0 in each stacked subgraph
            # The seed node for batch item i is at local index i (first batch_size nodes are seeds)
            seed_local_idx = i
            seed_in_indices = (node_indices == seed_local_idx).nonzero(as_tuple=True)[0]

            if len(seed_in_indices) == 0:
                # Seed node not in this subgraph (shouldn't happen with NeighborLoader)
                print(f"Warning: Seed node {i} not found in its subgraph")
                num_nodes_per_graph[i] = 0
                continue

            # Reorder so seed node is first
            seed_pos = seed_in_indices[0]
            node_indices_reordered = torch.cat([
                node_indices[seed_pos:seed_pos+1],  # Seed node first
                node_indices[:seed_pos],             # Nodes before seed
                node_indices[seed_pos+1:]            # Nodes after seed
            ])

            num_nodes = min(len(node_indices_reordered), max_nodes)
            nodes_to_use = node_indices_reordered[:num_nodes]
            num_nodes_per_graph[i] = num_nodes

            # Extract node features
            x_stacked[i, :num_nodes] = batch.x[nodes_to_use]

            # Extract edges within this subgraph
            edge_index = batch.edge_index
            src_mask = torch.isin(edge_index[0], nodes_to_use)
            dst_mask = torch.isin(edge_index[1], nodes_to_use)
            edge_mask = src_mask & dst_mask

            if edge_mask.any():
                edges_in_subgraph = edge_index[:, edge_mask]

                # Efficient remapping using searchsorted
                nodes_sorted, sort_indices = nodes_to_use.sort()

                # Find positions in sorted array
                src_positions = torch.searchsorted(nodes_sorted, edges_in_subgraph[0])
                dst_positions = torch.searchsorted(nodes_sorted, edges_in_subgraph[1])

                # Clamp positions to valid range (in case of any edge cases)
                src_positions = torch.clamp(src_positions, 0, len(nodes_to_use) - 1)
                dst_positions = torch.clamp(dst_positions, 0, len(nodes_to_use) - 1)

                # Map positions to local indices using the sort order
                src_remapped = sort_indices[src_positions]
                dst_remapped = sort_indices[dst_positions]

                # Validate remapping (filter out any incorrectly remapped edges)
                valid_remap = (nodes_sorted[src_positions] == edges_in_subgraph[0]) & \
                              (nodes_sorted[dst_positions] == edges_in_subgraph[1])

                if valid_remap.any():
                    edges_remapped = torch.stack([
                        src_remapped[valid_remap],
                        dst_remapped[valid_remap]
                    ])
                    num_edges = min(edges_remapped.size(1), max_edges)
                    edge_index_stacked[i, :, :num_edges] = edges_remapped[:, :num_edges]
        
        # Extract labels for seed nodes (first batch_size nodes)
        if hasattr(batch, 'y') and batch.y is not None:
            targets = batch.y[seed_node_local_indices]
        else:
            raise ValueError("Batch must have labels (y attribute)")
        
        return x_stacked, edge_index_stacked, num_nodes_per_graph, targets
    
    def compute_per_sample_gradients(self, batch, max_nodes: int = 100, max_edges: int = 200):
        """
        Compute per-sample gradients by looping over samples (simple, reliable approach).
        
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
        
        batch_size = x_stacked.size(0)
        per_sample_grads = {}
        
        for name, param in self.model.named_parameters():
            per_sample_grads[name] = torch.zeros(
                (batch_size,) + param.shape,
                device=param.device,
                dtype=param.dtype
            )
        
        for i in range(batch_size):
            x_single = x_stacked[i]
            edge_index_single = edge_idx_stacked[i]
            target = targets[i]
            
            row_sums = x_single.abs().sum(dim=1)
            has_data = row_sums > 0
            num_nodes_actual = has_data.sum().item()
            
            if num_nodes_actual == 0:
                continue
            
            x = x_single[:num_nodes_actual]
            valid_edges = (edge_index_single[0] < num_nodes_actual) & (edge_index_single[1] < num_nodes_actual)
            edge_index = edge_index_single[:, valid_edges]
            batch_tensor = torch.zeros(num_nodes_actual, dtype=torch.long, device=x.device)
            
            output = self.model(x, edge_index, batch_tensor)
            loss = F.nll_loss(output[0:1], target.unsqueeze(0))
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    per_sample_grads[name][i] = param.grad.clone()
                    param.grad.zero_()
        
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
        # Validate batch structure
        if not hasattr(batch, 'n_id') or batch.n_id is None:
            raise ValueError("Batch missing 'n_id' attribute")
        if not hasattr(batch, 'y') or batch.y is None:
            raise ValueError("Batch missing 'y' attribute")

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
        
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        
        with torch.no_grad():
            self.model.eval()
            if hasattr(batch, 'batch') and batch.batch is not None:
                batch_tensor = batch.batch if isinstance(batch.batch, torch.Tensor) else torch.tensor(batch.batch, device=self.device)
            else:
                batch_tensor = torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)
            
            if hasattr(batch, 'n_id') and batch.n_id is not None:
                n_id = batch.n_id if isinstance(batch.n_id, torch.Tensor) else torch.tensor(batch.n_id, device=self.device)
                seed_node_indices = list(range(min(batch_size, len(n_id))))
            else:
                seed_node_indices = list(range(min(batch_size, batch.num_nodes)))
            
            output = self.model(batch.x, batch.edge_index, batch_tensor)
            if hasattr(batch, 'y') and batch.y is not None and len(seed_node_indices) > 0:
                targets = batch.y[:batch_size] if batch.y.size(0) >= batch_size else batch.y
                output_seed = output[:len(targets)]
                loss = F.nll_loss(output_seed, targets)
            else:
                loss = torch.tensor(0.0, device=self.device)
            self.model.train()
        
        return loss.item()
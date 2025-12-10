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
        
        # Extract parameters as dict (needed for functional_call)
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
            
        Following Jan's suggestion using torch.scatter.
        
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
        
        # Stack each subgraph
        # batch.batch may be None if disjoint=False (which we're using to avoid pyg-lib issues)
        # So we construct it manually using BFS from each seed node
        if not hasattr(batch, 'batch') or batch.batch is None:
            # Construct batch assignment manually
            # First batch_size nodes are seed nodes (0, 1, 2, ..., batch_size-1)
            batch_tensor = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
            batch_tensor[:batch_size] = torch.arange(batch_size, device=device)
            
            # Build undirected adjacency list for BFS
            edge_index = batch.edge_index
            # Make undirected (add reverse edges)
            edge_index_undir = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            
            # BFS from each seed to assign nodes to neighborhoods
            for seed_idx in range(batch_size):
                visited = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
                queue = [seed_idx]
                visited[seed_idx] = True
                
                while queue:
                    current = queue.pop(0)
                    # Find all neighbors (undirected)
                    mask = (edge_index_undir[0] == current)
                    neighbors = edge_index_undir[1, mask].unique()
                    
                    for neighbor in neighbors:
                        neighbor = neighbor.item()
                        if neighbor < batch.num_nodes and not visited[neighbor]:
                            visited[neighbor] = True
                            # Assign to this seed if not already assigned (first seed wins)
                            if batch_tensor[neighbor] == -1:
                                batch_tensor[neighbor] = seed_idx
                            queue.append(neighbor)
            
            # Any unassigned nodes (shouldn't happen) go to seed 0
            unassigned = (batch_tensor == -1)
            if unassigned.any():
                batch_tensor[unassigned] = 0
        else:
            # batch.batch exists (from disjoint=True if pyg-lib works)
            batch_tensor = batch.batch
            if not isinstance(batch_tensor, torch.Tensor):
                batch_tensor = torch.tensor(batch_tensor, dtype=torch.long, device=device)
            
            # Ensure it's 1D
            if batch_tensor.dim() == 0:
                batch_tensor = batch_tensor.expand(batch.num_nodes)
            elif batch_tensor.dim() > 1:
                batch_tensor = batch_tensor.flatten()
        
        for i in range(batch_size):
            # Find nodes belonging to seed i
            # Use torch.eq for proper tensor comparison
            mask = torch.eq(batch_tensor, i)
            
            # Get indices where mask is True using torch.where
            nodes_in_subgraph = torch.where(mask)[0]
            num_nodes = len(nodes_in_subgraph)
            
            # Truncate to max_nodes if necessary
            num_nodes_to_use = min(num_nodes, max_nodes)
            num_nodes_per_graph[i] = num_nodes_to_use
            
            # Extract and place node features (only up to max_nodes)
            if num_nodes_to_use > 0:
                x_stacked[i, :num_nodes_to_use] = batch.x[nodes_in_subgraph[:num_nodes_to_use]]
            
            # Find edges within this subgraph
            edge_index = batch.edge_index
            src_in = torch.isin(edge_index[0], nodes_in_subgraph)
            dst_in = torch.isin(edge_index[1], nodes_in_subgraph)
            edge_mask = src_in & dst_in
            edges_in_subgraph = edge_index[:, edge_mask]
            
            if edges_in_subgraph.numel() > 0:
                # Remap global indices to local [0, num_nodes_to_use-1]
                # Only consider nodes that we're actually using (truncated to max_nodes)
                nodes_to_use = nodes_in_subgraph[:num_nodes_to_use]
                mapping = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
                mapping[nodes_to_use] = torch.arange(num_nodes_to_use, device=device)
                
                # Filter edges to only those within the nodes we're using
                src_valid = torch.isin(edges_in_subgraph[0], nodes_to_use)
                dst_valid = torch.isin(edges_in_subgraph[1], nodes_to_use)
                valid_edges = edges_in_subgraph[:, src_valid & dst_valid]
                
                if valid_edges.numel() > 0:
                    edges_remapped = mapping[valid_edges]
                    num_edges = min(edges_remapped.size(1), max_edges)
                    edge_index_stacked[i, :, :num_edges] = edges_remapped[:, :num_edges]
        
        # Extract targets (first batch_size entries are seed nodes)
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
        # Use masking instead of slicing (vmap doesn't support data-dependent slicing)
        # Create mask for valid nodes
        max_nodes = x_single.shape[0]
        node_mask = torch.arange(max_nodes, device=x_single.device) < num_nodes
        
        # Mask out padded nodes (set to zero)
        x = x_single * node_mask.unsqueeze(-1)
        
        # Keep fixed-size edge_index (vmap doesn't support boolean operations that change shapes)
        # Mask invalid edges using arithmetic operations (no boolean indexing)
        max_edges = edge_index_single.shape[1]
        
        # Create mask using arithmetic: 1 if valid, 0 if invalid
        # Valid if both src and dst are < num_nodes
        src_valid = (edge_index_single[0] < num_nodes).long()  # Convert to long for arithmetic
        dst_valid = (edge_index_single[1] < num_nodes).long()
        valid_mask = src_valid * dst_valid  # Both must be valid
        
        # Mask edges using arithmetic: invalid edges point to dummy node to avoid scatter issues
        # Using arithmetic instead of torch.where to be vmap-compatible
        dummy_node = max_nodes - 1
        valid_mask = src_valid * dst_valid  # [max_edges] with 1 for valid, 0 for invalid
        
        # Valid edges keep original indices, invalid edges become dummy_node
        edge_src = edge_index_single[0] * valid_mask + dummy_node * (1 - valid_mask)
        edge_dst = edge_index_single[1] * valid_mask + dummy_node * (1 - valid_mask)
        
        # Pre-add self-loops for all valid nodes to avoid GCNConv's add_remaining_self_loops
        # which uses boolean masking incompatible with vmap
        node_indices = torch.arange(max_nodes, device=x_single.device, dtype=torch.long)
        self_loop_valid = (node_indices < num_nodes).long()  # 1 if valid, 0 if invalid
        
        # Valid nodes get self-loops, invalid nodes get dummy node self-loops
        self_loop_src = node_indices * self_loop_valid + dummy_node * (1 - self_loop_valid)
        self_loop_dst = self_loop_src  # Self-loops: src == dst
        self_loops = torch.stack([self_loop_src, self_loop_dst], dim=0)
        
        # Concatenate regular edges and self-loops
        # Truncate to max_edges to fit in fixed-size tensor
        edge_index_full = torch.cat([torch.stack([edge_src, edge_dst], dim=0), self_loops], dim=1)
        # Keep only first max_edges edges (prioritizes regular edges, then self-loops)
        edge_index = edge_index_full[:, :max_edges]
        
        # Create batch assignment (fixed size, all zeros since single graph)
        batch = torch.zeros(max_nodes, dtype=torch.long, device=x.device)
        
        # Run model using functional_call
        output = functional_call(
            self.model,
            (params, buffers),
            args=(x, edge_index, batch)
        )
        
        # Extract prediction for seed node (first node, index 0)
        # output is [max_nodes, out_channels], but seed node is always at index 0
        seed_node_output = output[0]  # First node is the seed node
        
        # Compute loss (target is already a scalar tensor)
        # Use NLL loss since model outputs log_softmax
        loss = F.nll_loss(seed_node_output.unsqueeze(0), target.unsqueeze(0))
        return loss
    
    def compute_per_sample_gradients(self, batch, max_nodes: int = 200, max_edges: int = 1000):
        """
        Compute per-sample gradients using vmap + stacking.
        
        Args:
            batch: PyG Batch from NeighborLoader
            max_nodes: Max nodes to pad to
            max_edges: Max edges to pad to
            
        Returns:
            per_sample_grads: Dict[str, Tensor] with shape [batch_size, *param_shape]
        """
        # Stack subgraphs into tensors
        x_stacked, edge_idx_stacked, num_nodes, targets = self.stack_subgraphs(
            batch, max_nodes, max_edges
        )
        
        # Create gradient function
        grad_fn = grad(self.compute_loss_single_subgraph)
        
        # Vectorize with vmap
        # in_dims: (None, None, 0, 0, 0, 0) means:
        #   params, buffers: same for all (None)
        #   x, edge_idx, num_nodes, targets: vectorize over dim 0
        vmap_grad_fn = vmap(grad_fn, in_dims=(None, None, 0, 0, 0, 0))
        
        # Compute per-sample gradients (this is where vmap magic happens!)
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
        
        # Compute L2 norm for each sample across ALL parameters
        sq_norms = torch.zeros(batch_size, device=self.device)
        for grads in per_sample_grads.values():
            flat = grads.reshape(batch_size, -1)
            sq_norms += (flat ** 2).sum(dim=1)
        
        norms = torch.sqrt(sq_norms)
        
        # Compute clipping factors
        clip_factors = self.max_grad_norm / (norms + 1e-6)
        clip_factors = torch.clamp(clip_factors, max=1.0)
        
        # Apply clipping
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
        
        # 1. Compute per-sample gradients (with vmap!)
        per_sample_grads = self.compute_per_sample_gradients(batch)
        
        # 2. Clip
        clipped_grads = self.clip_gradients(per_sample_grads)
        
        # 3. Aggregate
        aggregated = {k: v.mean(dim=0) for k, v in clipped_grads.items()}
        
        # 4. Add noise
        noisy_grads = self.add_noise(aggregated, batch_size)
        
        # 5. Update
        optimizer.zero_grad()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.grad = noisy_grads[name]
        optimizer.step()
        
        # Update params and buffers after optimizer step
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        
        # Compute loss for logging
        with torch.no_grad():
            self.model.eval()
            output = self.model(batch.x, batch.edge_index, batch.batch)
            # Extract seed node predictions (first batch_size nodes are seed nodes)
            seed_outputs = output[:batch_size]
            loss = F.nll_loss(seed_outputs, batch.y[:batch_size])
            self.model.train()
        
        return loss.item()
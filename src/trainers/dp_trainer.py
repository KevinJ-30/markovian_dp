"""
DP-SGD for Graph Neural Networks using torch.func approach.
"""

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
        self.model = model.to(device)
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.device = device

        self.params = {k: v.detach() for k, v in model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in model.named_buffers()}

        print(f"Initialized DP-SGD:")
        print(f"  Clipping norm (C): {max_grad_norm}")
        print(f"  Noise multiplier (sigma): {noise_multiplier}")
        print(f"  Parameters: {sum(p.numel() for p in self.params.values())}")

    def stack_subgraphs(self, batch, max_nodes: int, max_edges: int):
        """
        Stack batched PyG graph into separate tensors for vmap.
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

        if not hasattr(batch, 'n_id') or batch.n_id is None:
            raise ValueError("NeighborLoader must provide 'n_id' attribute.")

        if hasattr(batch, 'batch') and batch.batch is not None:
            batch_tensor = batch.batch
        else:
            batch_tensor = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
            batch_tensor[:batch_size] = torch.arange(batch_size, device=device)

            if batch.num_nodes > batch_size:
                remaining_nodes = torch.arange(batch_size, batch.num_nodes, device=device)
                edge_index = batch.edge_index
                for node_idx in remaining_nodes:
                    edges_to_node = (edge_index[1] == node_idx) | (edge_index[0] == node_idx)
                    if edges_to_node.any():
                        connected_nodes = torch.cat([
                            edge_index[0, edges_to_node],
                            edge_index[1, edges_to_node]
                        ]).unique()
                        seed_connections = connected_nodes[connected_nodes < batch_size]
                        if len(seed_connections) > 0:
                            batch_tensor[node_idx] = seed_connections[0]
                        else:
                            batch_tensor[node_idx] = 0

        seed_node_local_indices = torch.arange(batch_size, device=device, dtype=torch.long)

        for i in range(batch_size):
            mask = (batch_tensor == i)
            node_indices = torch.where(mask)[0]

            if len(node_indices) == 0:
                num_nodes_per_graph[i] = 0
                continue

            seed_local_idx = i
            seed_in_indices = (node_indices == seed_local_idx).nonzero(as_tuple=True)[0]

            if len(seed_in_indices) == 0:
                print(f"Warning: Seed node {i} not found in its subgraph")
                num_nodes_per_graph[i] = 0
                continue

            seed_pos = seed_in_indices[0]
            node_indices_reordered = torch.cat([
                node_indices[seed_pos:seed_pos+1],
                node_indices[:seed_pos],
                node_indices[seed_pos+1:]
            ])

            num_nodes = min(len(node_indices_reordered), max_nodes)
            nodes_to_use = node_indices_reordered[:num_nodes]
            num_nodes_per_graph[i] = num_nodes

            x_stacked[i, :num_nodes] = batch.x[nodes_to_use]

            edge_index = batch.edge_index
            src_mask = torch.isin(edge_index[0], nodes_to_use)
            dst_mask = torch.isin(edge_index[1], nodes_to_use)
            edge_mask = src_mask & dst_mask

            if edge_mask.any():
                edges_in_subgraph = edge_index[:, edge_mask]

                nodes_sorted, sort_indices = nodes_to_use.sort()

                src_positions = torch.searchsorted(nodes_sorted, edges_in_subgraph[0])
                dst_positions = torch.searchsorted(nodes_sorted, edges_in_subgraph[1])

                src_positions = torch.clamp(src_positions, 0, len(nodes_to_use) - 1)
                dst_positions = torch.clamp(dst_positions, 0, len(nodes_to_use) - 1)

                src_remapped = sort_indices[src_positions]
                dst_remapped = sort_indices[dst_positions]

                valid_remap = (nodes_sorted[src_positions] == edges_in_subgraph[0]) & \
                              (nodes_sorted[dst_positions] == edges_in_subgraph[1])

                if valid_remap.any():
                    edges_remapped = torch.stack([
                        src_remapped[valid_remap],
                        dst_remapped[valid_remap]
                    ])
                    num_edges = min(edges_remapped.size(1), max_edges)
                    edge_index_stacked[i, :, :num_edges] = edges_remapped[:, :num_edges]

        if hasattr(batch, 'y') and batch.y is not None:
            targets = batch.y[seed_node_local_indices]
        else:
            raise ValueError("Batch must have labels (y attribute)")

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
        max_nodes = x_single.size(0)
        num_nodes_actual = num_nodes

        edge_index = edge_index_single
        valid_edge_mask = (edge_index[0] < num_nodes_actual) & (edge_index[1] < num_nodes_actual)

        edge_index_masked = torch.where(
            valid_edge_mask.unsqueeze(0).expand(2, -1),
            edge_index,
            torch.zeros_like(edge_index)
        )

        batch = torch.zeros(max_nodes, dtype=torch.long, device=x_single.device)

        output = functional_call(
            self.model,
            (params, buffers),
            args=(x_single, edge_index_masked, batch)
        )

        loss = F.nll_loss(output[0:1], target.unsqueeze(0), reduction='mean')
        loss = torch.where(num_nodes_actual > 0, loss, torch.tensor(0.0, device=x_single.device))

        return loss

    def compute_per_sample_gradients(self, batch, max_nodes: int = 100, max_edges: int = 200):
        x_stacked, edge_idx_stacked, num_nodes, targets = self.stack_subgraphs(
            batch, max_nodes, max_edges
        )

        grad_fn = grad(self.compute_loss_single_subgraph)
        vmap_grad_fn = vmap(grad_fn, in_dims=(None, None, 0, 0, 0, 0), randomness='different')

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
        noise_scale = self.noise_multiplier * self.max_grad_norm / batch_size

        noisy_grads = {}
        for name, g in grads.items():
            noise = torch.randn_like(g) * noise_scale
            noisy_grads[name] = g + noise

        return noisy_grads

    def step(self, batch, optimizer) -> float:
        if not hasattr(batch, 'n_id') or batch.n_id is None:
            raise ValueError("Batch missing 'n_id' attribute")
        if not hasattr(batch, 'y') or batch.y is None:
            raise ValueError("Batch missing 'y' attribute")

        batch_size = batch.batch_size

        per_sample_grads = self.compute_per_sample_gradients(batch, max_nodes=100, max_edges=200)
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
                batch_tensor = batch.batch
            else:
                batch_tensor = torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)

            output = self.model(batch.x, batch.edge_index, batch_tensor)
            targets = batch.y[:batch_size] if batch.y.size(0) >= batch_size else batch.y
            output_seed = output[:len(targets)]
            loss = F.nll_loss(output_seed, targets)
            self.model.train()

        return loss.item()

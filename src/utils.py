"""
Shared utilities.
"""

import torch


def compute_full_degrees(edge_index, num_nodes) -> torch.Tensor:
    """Count source occurrences in edge_index (= undirected degree for symmetric graphs)."""
    src = edge_index[0]
    return torch.bincount(src, minlength=num_nodes).float()


def sparsify_by_degree(edge_index, num_nodes, max_in_degree, generator=None):
    """
    Cap in-degree by uniformly sampling max_in_degree incoming edges per node.

    For each destination node with in-degree > max_in_degree, keep a random
    sample of max_in_degree incoming edges. Nodes already at or below the cap
    are unchanged. Done once at load time (not per-step), so the resulting
    public edge_index is what every algorithm and step sees.

    This is the standard DP-GNN technique for bounding per-node sensitivity:
    removing one node now changes at most max_in_degree edges, so node-DP
    sensitivity is finite even on high-degree graphs (Reddit, ogbn-products).

    Args:
        edge_index: [2, E] LongTensor.
        num_nodes: total node count (used for bincount).
        max_in_degree: per-destination cap.
        generator: optional torch.Generator for reproducibility.

    Returns:
        [2, E'] LongTensor with E' <= E. Edges may be in arbitrary order.
    """
    if max_in_degree is None or max_in_degree <= 0:
        return edge_index

    device = edge_index.device
    dst = edge_index[1]
    in_deg = torch.bincount(dst, minlength=num_nodes)

    # Fast path: nothing to drop.
    if int(in_deg.max().item()) <= max_in_degree:
        return edge_index

    # Sort edges by destination so each dst's edges occupy a contiguous block.
    order = torch.argsort(dst, stable=True)
    sorted_ei = edge_index[:, order]
    sorted_dst = sorted_ei[1]
    in_deg_sorted = torch.bincount(sorted_dst, minlength=num_nodes)

    # For each edge, compute (rank-within-its-dst-block, drawn random key).
    # Keeping the max_in_degree edges with the smallest random keys is
    # equivalent to a uniform random sample without replacement per dst.
    keys = torch.rand(sorted_ei.size(1), device=device, generator=generator)

    # Need rank of each edge within its destination block, ordered by key.
    # Compute block start offset per dst, then per-edge offset within block.
    block_starts = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    block_starts[1:] = torch.cumsum(in_deg_sorted, dim=0)

    # Per-edge index within its block (0..deg-1).
    edge_block_idx = (
        torch.arange(sorted_ei.size(1), device=device)
        - block_starts[sorted_dst]
    )

    # For each block, find the threshold: keys below threshold are kept.
    # Approach: sort keys within each block via segment sort using
    # (dst, key) lex sort. Argsort on (dst*BIG + key) is unsafe for floats,
    # so do it in two passes: stable sort by key then by dst.
    by_key = torch.argsort(keys)  # global sort by key
    stably_by_dst = sorted_dst[by_key]
    final_perm = by_key[torch.argsort(stably_by_dst, stable=True)]

    # final_perm[i] gives the i-th edge in (dst-asc, key-asc) order.
    # Within each block we now have edges sorted by random key.
    # Compute per-edge rank in its block under this ordering.
    rank_in_block = torch.empty_like(edge_block_idx)
    rank_in_block[final_perm] = edge_block_idx

    keep_mask = rank_in_block < max_in_degree
    return sorted_ei[:, keep_mask]

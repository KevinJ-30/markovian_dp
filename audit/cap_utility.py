import sys, torch
sys.path.insert(0, '/Users/kevinjacob/markovian_dp copy')
from src.datasets import load_dataset
from src.sparse.sparse_expand import (cap_degrees_undirected, dedup_arcs,
                                      build_adjacency, edge_set_is_symmetric)

ds, data = load_dataset('ppi', device='cpu')
n = int(data.num_nodes)
ei = dedup_arcs(data.edge_index, n)
print(f"PPI: n={n}  arcs={ei.size(1)}  symmetric={edge_set_is_symmetric(ei, n)}")

tr = torch.where(data.train_mask)[0]
for K in (5, 3, 2):
    g = torch.Generator().manual_seed(0)
    c = cap_degrees_undirected(ei, n, K, generator=g)
    adj = build_adjacency(c, n, direction='in')
    indeg = torch.bincount(c[1], minlength=n).float()
    print(f"  cap K={K:>2}: arcs {c.size(1):>8}  "
          f"mean in-deg (train roots) {indeg[tr].mean():.3f}  "
          f"E[1-hop subgraph size] {1+indeg[tr].mean():.3f}")

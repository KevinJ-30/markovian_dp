"""
Diagnostic: mean rooted-subgraph size vs p2 on inductive+capped ogbn-arxiv.

Explains why the utility ladder is flat — even at p2=1.0 the out-neighborhoods
are tiny, because we expand along OUT-edges and arxiv's mass is in IN-edges
(max in-degree >> max out-degree). Motivates the directed-graph question.

  python scripts/plot_subgraph_size.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.datasets import load_dataset  # noqa: E402
from src.sparse.sparse_expand import (  # noqa: E402
    build_out_adjacency, cap_degrees, max_degrees, sparse_expand)


def main():
    _, data = load_dataset('ogbn-arxiv')
    ei = data.edge_index
    tr = data.train_mask
    ei = ei[:, tr[ei[0]] & tr[ei[1]]]
    in_raw, out_raw = max_degrees(ei, int(data.num_nodes))
    g = torch.Generator().manual_seed(12345)
    ei = cap_degrees(ei, int(data.num_nodes), K_in=5, K_out=5, generator=g)
    adj = build_out_adjacency(ei, int(data.num_nodes))

    roots = torch.where(tr)[0]
    gg = torch.Generator().manual_seed(0)
    samp = roots[torch.randperm(roots.numel(), generator=gg)[:3000]].tolist()

    p2s = [0.1, 0.25, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for r, style in ((1, 'o-'), (2, 's--')):
        means = []
        for p2 in p2s:
            gen = torch.Generator().manual_seed(1)
            sizes = [sparse_expand(adj, v, p2, r, generator=gen).num_nodes
                     for v in samp]
            means.append(sum(sizes) / len(sizes))
        ax.plot(p2s, means, style, label=f'r={r}')

    ax.set_xlabel(r'edge-sampling probability $p_2$')
    ax.set_ylabel('mean nodes per rooted subgraph')
    ax.set_title('ogbn-arxiv (inductive, $K{=}5$): rooted-subgraph size\n'
                 f'expanding OUT-edges  |  raw max in-deg {in_raw} vs out-deg {out_raw}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.annotate('even at $p_2{=}1$ the root sees only\n'
                '~1.6 out-neighbors: little to sparsify',
                xy=(1.0, 2.59), xytext=(0.15, 3.6),
                fontsize=8, arrowprops=dict(arrowstyle='->', alpha=0.6))
    fig.tight_layout()
    out = 'results/inductive_subgraph_size.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    os.makedirs('paper/figures/experiments', exist_ok=True)
    fig.savefig('paper/figures/experiments/inductive_subgraph_size.png',
                dpi=150, bbox_inches='tight')
    print(f'wrote {out} (+ copy under paper/figures/experiments/)')


if __name__ == '__main__':
    main()

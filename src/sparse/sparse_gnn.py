"""
Algorithm 1: SparseGNN — the model-agnostic training engine.

    for t = 1..T:
        V_root <- { v : B_v = 1 },  B_v ~ Bernoulli(p1)          (root sampling)
        S_t    <- { SparseExpand(G, v, p2, r) : v in V_root }    (Algorithm 2)
        theta  <- Alg(theta, S_t)                                (Alg adds noise)

`Alg` is realized here in two modes:

  * non-DP (default):  loss = sum_{H in S_t} g0_loss(H); a single backward gives
    the summed gradient G(y) = sum_v g0(y_v); optimizer.step().

  * DP (dp=True): per-subgraph backward, clip each g0(H) to L2 norm C, sum,
    add Gaussian noise N(0, (sigma*C)^2 I) (Assumption 6.3), then step.
    Accounted post-hoc by the dominating pairs in accounting.py.

The engine only talks to a BaseMechanism, so it is identical for the GNN node
classifier and a future non-GNN anomaly detector.
"""

from typing import Dict, List, Optional

import torch

from .base_mechanism import BaseMechanism
from .sparse_expand import build_adjacency, sample_roots, sparse_expand


def _make_generator(seed):
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _step_nondp(mechanism: BaseMechanism, subgraphs: List) -> float:
    """Non-DP update: sum per-subgraph losses, one backward, one step."""
    mechanism.train_mode()
    opt = mechanism.optimizer
    opt.zero_grad()

    total = mechanism.zero_loss()
    n_supervised = 0
    for H in subgraphs:
        loss_H = mechanism.subgraph_loss(H)
        total = total + loss_H
        n_supervised += 1
    if n_supervised == 0:
        return 0.0

    total.backward()
    opt.step()
    return float(total.detach())


def _step_dp(mechanism: BaseMechanism, subgraphs: List, *, C: float,
             sigma: float, noise_gen: torch.Generator,
             expected_batch: float = 1.0) -> float:
    """DP update: per-subgraph clip to C, sum, add N(0,(sigma*C)^2 I), step.

    The noisy sum is divided by `expected_batch` (E[|V_root|] = p1 * |pool|)
    before the optimizer step — standard DP-SGD normalization.  This is
    post-processing of the Gaussian mechanism, so it has no privacy cost, but
    it decouples the learning rate from the batch size.
    """
    mechanism.train_mode()
    params = mechanism.parameters()
    grad_accum = [torch.zeros_like(p) for p in params]

    running = 0.0
    for H in subgraphs:
        loss_H = mechanism.subgraph_loss(H)
        if float(loss_H.detach()) == 0.0 and not loss_H.requires_grad:
            continue
        grads = torch.autograd.grad(loss_H, params, retain_graph=False,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        clipped = mechanism.clip_flat_grad(grads, C)
        for acc, g in zip(grad_accum, clipped):
            acc.add_(g)
        running += float(loss_H.detach())

    noise = mechanism.gaussian_noise_like(grad_accum, sigma, C, generator=noise_gen)
    opt = mechanism.optimizer
    opt.zero_grad()
    denom = max(float(expected_batch), 1.0)
    for p, acc, z in zip(params, grad_accum, noise):
        p.grad = (acc + z) / denom
    opt.step()
    return running


def _evaluate(mechanism, data, alt_edge_index):
    """Metrics on the configured graph, plus `<key>_alt` on `alt_edge_index`."""
    out = dict(mechanism.evaluate(data))
    if alt_edge_index is not None:
        for k, v in mechanism.evaluate_on(data, alt_edge_index).items():
            out[f'{k}_alt'] = v
    return out


def train_sparse_gnn(
    mechanism: BaseMechanism,
    data,
    *,
    p1: float,
    p2: float,
    r: int,
    T: int,
    adj: Optional[List[torch.Tensor]] = None,
    direction: str = 'in',
    candidate_nodes: Optional[torch.Tensor] = None,
    dp: bool = False,
    clip: Optional[float] = None,
    sigma: Optional[float] = None,
    seed: int = 0,
    eval_every: int = 0,
    track_every: int = 0,
    eval_alt_edge_index=None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run T steps of SparseGNN and return the final evaluation metrics.

    Args:
        mechanism:       a BaseMechanism (GNN or anomaly detector).
        data:            PyG Data (must expose num_nodes, edge_index; masks used
                         by the mechanism's evaluate).
        p1, p2, r, T:    paper parameters (root prob, edge prob, distance, steps).
        adj:             optional precomputed adjacency from
                         `build_adjacency(..., direction)`; built if None.
        direction:       'in' (Algorithm 5, expansion along incoming edges — the
                         orientation a message-passing GNN needs) or 'out' (the
                         legacy Algorithm 2/4 orientation, for the ablation).
        candidate_nodes: optional pool of eligible roots (defaults to all nodes).
                         For per-root supervised training, restricting this to
                         training nodes avoids wasting steps on unlabeled roots.
        dp:              enable the DP clip+noise path (default False).
        clip, sigma:     DP clipping norm C and noise multiplier sigma (required
                         when dp=True; sigma scales noise std = sigma*C).
        seed:            base seed for reproducible root/edge sampling.
        eval_every:      if >0 and verbose, evaluate every `eval_every` steps.
        eval_alt_edge_index: if given, every evaluation is also run against
                         this adjacency and reported under `<key>_alt`.  Used to
                         record utility on both the training graph and the full
                         one, which differ by the degree cap.
        track_every:     if >0, evaluate every `track_every` steps and return
                         the checkpoints under the 'history' key (a list of
                         {'step': t, <metrics>} dicts).  Evaluation draws no
                         sampling randomness, so a tracked run follows exactly
                         the same trajectory as an untracked one.  Each
                         checkpoint pairs with the epsilon of composing the
                         first t steps (see compute_epsilon --track support).

    Returns:
        dict of metrics from mechanism.evaluate(data); plus 'history' when
        track_every > 0.
    """
    num_nodes = int(data.num_nodes)
    if adj is None:
        adj = build_adjacency(data.edge_index, num_nodes, direction=direction)

    if dp:
        if clip is None or sigma is None:
            raise ValueError("dp=True requires both `clip` (C) and `sigma`.")

    sample_gen = _make_generator(seed)
    noise_gen = _make_generator(seed + 10_000 if seed is not None else None)

    pool_size = (num_nodes if candidate_nodes is None
                 else int(candidate_nodes.numel()))
    expected_batch = p1 * pool_size

    history: List[Dict[str, float]] = []
    for t in range(1, T + 1):
        roots = sample_roots(num_nodes, p1, generator=sample_gen,
                             candidate_nodes=candidate_nodes)
        subgraphs = [sparse_expand(adj, int(v), p2, r, generator=sample_gen,
                                   direction=direction)
                     for v in roots.tolist()]

        if dp:
            # Run the DP step even when no root was sampled: the analyzed base
            # mechanism (Assumption 3.2 / 6.3) adds Gaussian noise to G(y)
            # unconditionally, including on the all-empty batch, so a
            # noise-only update is what matches the accounting.
            loss = _step_dp(mechanism, subgraphs, C=clip, sigma=sigma,
                            noise_gen=noise_gen, expected_batch=expected_batch)
        else:
            if roots.numel() == 0:
                continue
            loss = _step_nondp(mechanism, subgraphs)

        if track_every and (t % track_every == 0 or t == T):
            history.append({'step': t, **_evaluate(mechanism, data,
                                                   eval_alt_edge_index)})

        if verbose and eval_every and (t % eval_every == 0 or t == 1):
            accs = mechanism.evaluate(data)
            print(f"  step {t:4d}/{T}  |V_root|={roots.numel():4d}  "
                  f"loss={loss:.4f}  val={accs['val']:.4f}  test={accs['test']:.4f}")

    final = _evaluate(mechanism, data, eval_alt_edge_index)
    if track_every:
        final = dict(final)
        final['history'] = history
    return final

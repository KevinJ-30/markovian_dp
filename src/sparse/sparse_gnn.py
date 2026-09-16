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

    Two implementations of that SAME mechanism.  `_step_dp` is the reference:
    a Python loop calling autograd once per root.  `_step_dp_vectorized` pads
    the roots into one batch and takes per-root gradients with torch.func.vmap
    (see vectorized.py -- stock library pieces, no hand-derived gradients).
    The fast path is default; mechanisms decline it when it cannot reproduce
    their loss exactly, and --no_vectorized forces the loop.

The engine only talks to a BaseMechanism, so it is identical for the GNN node
classifier and a future non-GNN anomaly detector.
"""

from typing import Any, Callable, Dict, List, Optional

import torch

from .base_mechanism import BaseMechanism
from .sparse_expand import SparseAdjacency, build_adjacency, sample_roots, sparse_expand
from .accounting import SparseGNNNoiseCalibration, calibrate_sparsegnn_noise


def _make_generator(seed):
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _step_nondp(mechanism: BaseMechanism, subgraphs: List,
                expected_batch: float = 1.0) -> float:
    """Non-DP update: mean of per-subgraph losses, one backward, one step.

    Divided by the SAME data-independent `expected_batch = p1 * |pool|` the DP
    path uses, so both paths present the optimizer with a mean gradient and a
    single learning rate transfers between them.  Previously this took the raw
    sum while the DP path took the mean, so at equal --lr the non-DP path
    stepped ~E[B] times further (~512x on PPI) and the DP-vs-ceiling gap
    confounded privacy noise with a gradient rescale.
    """
    mechanism.train_mode()
    opt = mechanism.optimizer
    opt.zero_grad()

    losses = mechanism.subgraph_losses(subgraphs)
    total = mechanism.zero_loss()
    for loss_H in losses:
        total = total + loss_H
    if not losses:
        return 0.0

    (total / max(float(expected_batch), 1.0)).backward()
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
    opt = mechanism.optimizer
    denom = max(float(expected_batch), 1.0)

    if not subgraphs:
        noise = mechanism.gaussian_noise_like(
            params, sigma, C, generator=noise_gen)
        opt.zero_grad()
        for p, z in zip(params, noise):
            p.grad = z / denom
        opt.step()
        return 0.0

    grad_accum = [torch.zeros_like(p) for p in params]
    running = None
    for losses in mechanism.iter_subgraph_loss_batches(subgraphs):
        batch_running = None
        for i, loss_H in enumerate(losses):
            grads = torch.autograd.grad(
                loss_H, params, retain_graph=i < len(losses) - 1,
                allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, params)]
            clipped = mechanism.clip_flat_grad(grads, C)
            for acc, g in zip(grad_accum, clipped):
                acc.add_(g)
            batch_running = (loss_H.detach() if batch_running is None
                             else batch_running + loss_H.detach())
        running = (batch_running if running is None else running + batch_running)

    noise = mechanism.gaussian_noise_like(
        grad_accum, sigma, C, generator=noise_gen)
    opt.zero_grad()
    for p, acc, z in zip(params, grad_accum, noise):
        p.grad = (acc + z) / denom
    opt.step()
    return float(running) if running is not None else 0.0


def _step_dp_vectorized(mechanism: BaseMechanism, subgraphs: List, *, C: float,
                        sigma: float, noise_gen: torch.Generator, cfg: dict,
                        mirror, expected_batch: float = 1.0) -> float:
    """`_step_dp` with the per-root loop replaced by torch.func.vmap.

    Same mechanism, same guarantee: one gradient per root, clipped to C before
    summing, then a single N(0, (sigma*C)^2 I) draw.  Only the route to the
    per-root gradients differs -- see vectorized.py, which uses stock
    torch.func + PyG layers rather than any hand-derived gradient math.
    """
    from .vectorized import build_padded_batch, clipped_grad_sum, per_sample_grads

    mechanism.train_mode()
    params = mechanism.parameters()
    opt = mechanism.optimizer
    denom = max(float(expected_batch), 1.0)

    if not subgraphs:
        noise = mechanism.gaussian_noise_like(params, sigma, C, generator=noise_gen)
        opt.zero_grad()
        for p, z in zip(params, noise):
            p.grad = z / denom
        opt.step()
        return 0.0

    data = mechanism.data
    names = [n for n, _ in mechanism.module.named_parameters()]
    batch = build_padded_batch(
        subgraphs, data.x, data.y,
        supervised=getattr(mechanism, '_train_mask', None),
        device=mechanism.device)
    per_root, total_loss = per_sample_grads(
        dict(mechanism.module.named_parameters()), batch, mirror,
        loss_tail=cfg['loss_tail'], kind=cfg['kind'])

    missing = [n for n in names if n not in per_root]
    if missing:
        # A parameter the mirror does not cover would silently receive only
        # noise, which is a different mechanism from the one accounted.
        raise RuntimeError(
            f"vectorized DP step produced no gradient for {missing}; this "
            f"mechanism should not have declared vectorized_tail")

    # Clip per root, then sum -- `grad_accum` is already parameter-shaped, so
    # the noise drawn from it matches the parameters (per_root entries carry a
    # leading batch axis and must NOT be used for the noise shape).
    grad_accum = clipped_grad_sum(per_root, names, C)
    noise = mechanism.gaussian_noise_like(grad_accum, sigma, C, generator=noise_gen)
    opt.zero_grad()
    for p, acc, z in zip(params, grad_accum, noise):
        p.grad = (acc + z) / denom
    opt.step()
    return total_loss


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
    adj: Optional[SparseAdjacency] = None,
    direction: str = 'in',
    candidate_nodes: Optional[torch.Tensor] = None,
    dp: bool = False,
    clip: Optional[float] = None,
    sigma: Optional[float] = None,
    seed: int = 0,
    eval_every: int = 0,
    track_every: int = 0,
    eval_alt_edge_index=None,
    vectorized: bool = True,
    verbose: bool = False,
    checkpoint_callback: Optional[Callable[[Dict[str, float]], None]] = None,
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

    # Resolve the fast DP path once.  None -> the per-root loop, which is always
    # correct; the mechanism declines whenever the dense form cannot reproduce
    # its `subgraph_loss` exactly (see BaseMechanism.vectorized_config).
    dp_cfg = (mechanism.vectorized_config() if dp and vectorized else None)
    dp_mirror = None
    if dp_cfg is not None:
        from .vectorized import build_mirror
        # Built ONCE per run, not per step: it holds no trained state (every
        # step supplies the mechanism's live parameters via functional_call),
        # so rebuilding it would be pure overhead.
        dp_mirror = build_mirror(mechanism.module, dp_cfg['kind'],
                                 dp_cfg['dropout']).to(mechanism.device)
    if verbose:
        print(f"  DP gradient path: "
              f"{'vectorized (torch.func.vmap)' if dp_cfg else 'per-root loop'}")

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
            if dp_cfg is not None:
                loss = _step_dp_vectorized(
                    mechanism, subgraphs, C=clip, sigma=sigma,
                    noise_gen=noise_gen, cfg=dp_cfg, mirror=dp_mirror,
                    expected_batch=expected_batch)
            else:
                loss = _step_dp(mechanism, subgraphs, C=clip, sigma=sigma,
                                noise_gen=noise_gen,
                                expected_batch=expected_batch)
        else:
            if roots.numel() == 0:
                continue
            loss = _step_nondp(mechanism, subgraphs, expected_batch)

        if track_every and (t % track_every == 0 or t == T):
            checkpoint = {'step': t, **_evaluate(mechanism, data,
                                                 eval_alt_edge_index)}
            history.append(checkpoint)
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint)

        if verbose and eval_every and (t % eval_every == 0 or t == 1):
            accs = mechanism.evaluate(data)
            print(f"  step {t:4d}/{T}  |V_root|={roots.numel():4d}  "
                  f"loss={loss:.4f}  val={accs['val']:.4f}  test={accs['test']:.4f}")

    final = _evaluate(mechanism, data, eval_alt_edge_index)
    if track_every:
        final = dict(final)
        final['history'] = history
    return final


def train_sparse_gnn_with_budget(
    mechanism: BaseMechanism,
    data,
    *,
    target_epsilon: float,
    target_delta: float,
    K_in: int,
    K_out: int,
    p1: float,
    p2: float,
    r: int,
    T: int,
    clip: float,
    direction: str = "in",
    theorem: str = "auto",
    accounting_grid: float = 1e-4,
    calibration_rtol: float = 1e-3,
    calibration_atol: float = 1e-6,
    max_sigma: float = 1e6,
    adj=None,
    candidate_nodes=None,
    seed: int = 0,
    eval_every: int = 0,
    track_every: int = 0,
    eval_alt_edge_index=None,
    vectorized: bool = True,
    verbose: bool = False,
    checkpoint_callback=None,
) -> tuple[dict[str, Any], SparseGNNNoiseCalibration]:
    """Calibrate one certified noise multiplier, then train with it once."""

    calibration = calibrate_sparsegnn_noise(
        target_epsilon=target_epsilon, target_delta=target_delta,
        p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, steps=T, clip=clip,
        direction=direction, theorem=theorem, grid=accounting_grid,
        sigma_rtol=calibration_rtol, sigma_atol=calibration_atol,
        max_sigma=max_sigma,
    )
    metrics = train_sparse_gnn(
        mechanism, data, p1=p1, p2=p2, r=r, T=T, adj=adj,
        direction=direction, candidate_nodes=candidate_nodes, dp=True,
        clip=clip, sigma=calibration.noise_multiplier, seed=seed,
        eval_every=eval_every, track_every=track_every,
        eval_alt_edge_index=eval_alt_edge_index, verbose=verbose,
        checkpoint_callback=checkpoint_callback,
    )
    return metrics, calibration

"""
Algorithm 1: SparseGNN — the model-agnostic training engine.

    for t = 1..T:
        V_root <- { v : B_v = 1 },  B_v ~ Bernoulli(p1)          (root sampling)
        S_t    <- { SparseExpand(G, v, p2, r) : v in V_root }    (Algorithm 2)
        theta  <- Alg(theta, S_t)                                (Alg adds noise)

`Alg` is realized here in two modes:

  * non-DP (default):  loss = sum_{H in S_t} g0_loss(H); a single backward gives
    the summed gradient G(y) = sum_v g0(y_v); optimizer.step().

  * DP (dp=True): a padded root-first batch gives Opacus one gradient sample
    per rooted subgraph; Opacus globally clips each to C, sums, adds Gaussian
    noise N(0, (sigma*C)^2 I), and applies the optimizer.

The engine only talks to a BaseMechanism, so the training loop is independent
of the concrete node-classification model.
"""

import math
from typing import Any, Callable, Dict, List, Optional

import torch

from src.models.base_mechanism import BaseMechanism
from src.models.bootstrap import BootstrapConfig, BootstrapMetrics
from src.privacy.accounting import SparseGNNNoiseCalibration, calibrate_sparsegnn_noise
from src.processing.padded import iter_padded_root_batches
from src.processing.sparse_expand import (
    SparseAdjacency, batch_sparse_expand, build_adjacency, sample_roots,
)


def _make_generator(seed, device="cpu"):
    if seed is None:
        return None
    g = torch.Generator(device=device)
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


class OpacusPrivateUpdate:
    """One logical SparseGNN Gaussian update, possibly in physical chunks."""

    def __init__(self, mechanism: BaseMechanism, *, C: float, sigma: float,
                 expected_batch: float, noise_gen: torch.Generator):
        if mechanism.optimizer is None:
            raise ValueError("build the base optimizer before enabling DP")
        from opacus.grad_sample import GradSampleModule
        from opacus.optimizers import DPOptimizer

        self.mechanism = mechanism
        self.private_module = GradSampleModule(
            mechanism.build_private_module(), batch_first=True,
            loss_reduction="mean", strict=True)
        self.optimizer = DPOptimizer(
            optimizer=mechanism.optimizer,
            noise_multiplier=float(sigma),
            max_grad_norm=float(C),
            expected_batch_size=max(float(expected_batch), 1.0),
            loss_reduction="mean",
            generator=noise_gen,
            secure_mode=False,
        )
        mechanism.optimizer = self.optimizer

    def _process(self, batch, *, final: bool) -> None:
        losses = self.mechanism.private_losses(self.private_module, batch)
        if losses.ndim != 1 or losses.numel() != batch.batch_size:
            raise RuntimeError("private_losses must return one scalar per sample")
        losses.mean().backward()
        if not final:
            self.optimizer.signal_skip_step(True)
        self.optimizer.step()

    def step(self, subgraphs: List) -> None:
        self.mechanism.train_mode()
        self.private_module.train()
        self.optimizer.zero_grad()
        batches = iter_padded_root_batches(
            subgraphs,
            x=self.mechanism.data.x,
            y=self.mechanism.data.y,
            train_mask=self.mechanism.data.train_mask,
            device=self.mechanism.device,
            max_padded_nodes=self.mechanism.max_private_batch_nodes,
        )
        current = next(iter(batches))
        for following in batches:
            self._process(current, final=False)
            # On a skipped physical step Opacus retains summed_grad while
            # clearing grad_sample for the next chunk.
            self.optimizer.zero_grad()
            current = following
        self._process(current, final=True)


def _evaluate(mechanism, test_data, bootstrap=None):
    if bootstrap is None or bootstrap.n_resamples == 0:
        return dict(mechanism.evaluate(test_data))
    task = mechanism.metric_name
    if task == "auroc":
        metrics = ("auroc", "accuracy")
    elif task == "micro_f1":
        metrics = ("micro_f1", "micro_auroc")
    else:
        metrics = (task,)
    accumulator = BootstrapMetrics(task, bootstrap, metrics=metrics)
    result = dict(mechanism.evaluate(test_data, bootstrap=accumulator))
    result["test_confidence_intervals"] = accumulator.compute()
    return result


def train_sparse_gnn(
    mechanism: BaseMechanism,
    train_data,
    test_data,
    *,
    p1: float,
    p2: float,
    r: int,
    T: int,
    adj: Optional[SparseAdjacency] = None,
    direction: str = 'in',
    dp: bool = False,
    clip: Optional[float] = None,
    sigma: Optional[float] = None,
    seed: int = 0,
    eval_every: int = 0,
    track_every: int = 0,
    progress_every: Optional[int] = None,
    verbose: bool = False,
    checkpoint_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    bootstrap: Optional[BootstrapConfig] = None,
) -> Dict[str, Any]:
    """Run all T updates, then evaluate the best validation-primary-metric model.

    Args:
        mechanism:       a BaseMechanism built against `train_data`.
        train_data:      PyG training graph. Its train mask defines eligible
                         roots and its edges are the only edges expanded.
        test_data:       separate PyG graph used for every evaluation.
        p1, p2, r, T:    paper parameters (root prob, edge prob, distance, steps).
        adj:             optional precomputed adjacency from
                         `build_adjacency(..., direction)`; built if None.
        direction:       'in' (Algorithm 5, expansion along incoming edges — the
                         orientation a message-passing GNN needs) or 'out' (the
                         legacy Algorithm 2/4 orientation, for the ablation).
        dp:              enable the DP clip+noise path (default False).
        clip, sigma:     clipping norm C and Opacus noise multiplier (required
                         when dp=True; absolute noise std is sigma*C).
        seed:            base seed for reproducible root/edge sampling.
        bootstrap:       optional node bootstrap for the final test evaluation;
                         None or n_resamples=0 disables confidence intervals.
        eval_every:      validation/selection interval, independent of verbosity.
                         Zero uses ceil(1/p1), one expected epoch; the final step
                         is always a candidate. With p1=0, select at the final step.
        track_every:     if >0, evaluate every `track_every` steps and return
                         the checkpoints under the 'history' key (a list of
                         {'step': t, <metrics>} dicts).  Evaluation draws no
                         sampling randomness, so a tracked run follows exactly
                         the same trajectory as an untracked one.  Each
                         checkpoint pairs with the epsilon of composing the
                         first t steps (see compute_epsilon --track support).
        progress_every:  optional logging interval when verbose; None uses the
                         effective validation interval and zero disables logging.
                         Logging does not add selection candidates.

    Returns:
        Metrics from the restored model and 'selection' metadata; plus 'history'
        when track_every > 0. History records the models at those actual steps.
    """
    if eval_every < 0:
        raise ValueError("eval_every must be nonnegative")
    if track_every < 0:
        raise ValueError("track_every must be nonnegative")
    if progress_every is not None and progress_every < 0:
        raise ValueError("progress_every must be nonnegative")
    evaluate_every = eval_every or (math.ceil(1 / p1) if p1 > 0 else max(1, T))
    if progress_every is None:
        progress_every = evaluate_every

    num_nodes = int(train_data.num_nodes)
    if adj is None:
        adj = build_adjacency(
            train_data.edge_index, num_nodes, direction=direction)

    if dp:
        if clip is None or sigma is None:
            raise ValueError("dp=True requires both `clip` (C) and `sigma`.")

    sample_gen = _make_generator(seed)

    candidate_nodes = torch.where(train_data.train_mask.detach().cpu())[0]
    expected_batch = p1 * int(candidate_nodes.numel())
    private_update = None
    if dp:
        params = mechanism.parameters()
        if not params:
            raise ValueError("DP training requires trainable parameters")
        noise_gen = _make_generator(
            seed + 10_000 if seed is not None else None,
            device=params[0].device)
        private_update = OpacusPrivateUpdate(
            mechanism, C=clip, sigma=sigma, expected_batch=expected_batch,
            noise_gen=noise_gen)

    history: List[Dict[str, float]] = []
    best_state = None
    best_val = None
    best_step = 0
    for t in range(1, T + 1):
        roots = sample_roots(num_nodes, p1, generator=sample_gen,
                             candidate_nodes=candidate_nodes)
        subgraphs = batch_sparse_expand(
            adj, roots, p2, r, generator=sample_gen, direction=direction)

        if dp:
            # Always execute the logical mechanism.  An empty root draw becomes
            # a masked zero-signal batch and therefore a noise-only update.
            private_update.step(subgraphs)
            loss = None
        else:
            loss = (_step_nondp(mechanism, subgraphs, expected_batch)
                    if roots.numel() else 0.0)

        select = t % evaluate_every == 0 or t == T
        track = track_every > 0 and (t % track_every == 0 or t == T)
        progress = (verbose and progress_every > 0
                    and (t % progress_every == 0 or t == 1 or t == T))
        if select or track or progress:
            # Reuse one forward when selection and explicitly requested tracking
            # coincide. Ordinary validation/logging never reduces test metrics.
            accs = (_evaluate(mechanism, test_data) if track else
                    mechanism.evaluate(test_data, splits=("val",)))
        if select:
            validation = float(accs["val"])
            if best_state is None or (
                    not math.isnan(validation)
                    and (best_val is None or validation > best_val)):
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in mechanism.module.state_dict().items()
                }
                best_val = None if math.isnan(validation) else validation
                best_step = t

        if track:
            checkpoint = {'step': t, **accs}
            history.append(checkpoint)
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint)

        if progress:
            loss_text = "private" if loss is None else f"{loss:.4f}"
            print(f"  step {t:4d}/{T}  |V_root|={roots.numel():4d}  "
                  f"loss={loss_text}  val={accs['val']:.4f}")

    if best_state is not None:
        mechanism.module.load_state_dict(best_state)
    final = _evaluate(mechanism, test_data, bootstrap)
    if best_state is None:
        # Preserve evaluation-only runs (T=0) without inventing a training step.
        validation = float(final["val"])
        best_val = None if math.isnan(validation) else validation
    final["selection"] = {
        "metric": mechanism.metric_name,
        "step": best_step,
        "validation_score": best_val,
        "evaluate_every": evaluate_every,
    }
    if track_every:
        final['history'] = history
    return final


def train_sparse_gnn_with_budget(
    mechanism: BaseMechanism,
    train_data,
    test_data,
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
    accounting_grid: float = 1e-3,
    union_safe: bool = True,
    calibration_rtol: float = 1e-3,
    calibration_atol: float = 1e-6,
    max_sigma: float = 1e6,
    adj=None,
    seed: int = 0,
    eval_every: int = 0,
    track_every: int = 0,
    progress_every: Optional[int] = None,
    verbose: bool = False,
    checkpoint_callback=None,
    bootstrap: Optional[BootstrapConfig] = None,
) -> tuple[dict[str, Any], SparseGNNNoiseCalibration]:
    """Calibrate noise and train once, optionally bootstrapping final test metrics."""

    calibration = calibrate_sparsegnn_noise(
        target_epsilon=target_epsilon, target_delta=target_delta,
        p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, steps=T, clip=clip,
        grid=accounting_grid, sigma_rtol=calibration_rtol,
        sigma_atol=calibration_atol, max_sigma=max_sigma,
        union_safe=union_safe,
    )
    metrics = train_sparse_gnn(
        mechanism, train_data, test_data, p1=p1, p2=p2, r=r, T=T, adj=adj,
        direction=direction, dp=True, clip=clip,
        sigma=calibration.noise_multiplier, seed=seed,
        eval_every=eval_every, track_every=track_every, verbose=verbose,
        progress_every=progress_every,
        checkpoint_callback=checkpoint_callback, bootstrap=bootstrap,
    )
    return metrics, calibration

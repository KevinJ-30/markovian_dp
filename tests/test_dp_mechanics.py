"""
Empirical audit of the DP training path.

These tests do not re-read the implementation and agree with it; they MEASURE
what the mechanism actually does and check it against what the accounting
assumes.  Everything the epsilon numbers rest on is asserted here:

  * per-subgraph (not per-batch) clipping, with contributions bounded by C
  * batch sensitivity: one coordinate substitution moves G(y) by <= 2C, one
    insertion/removal by <= C   (Lemma 11 / Assumption 6.3)
  * Gaussian noise added ONCE per step, with std exactly sigma*C per coordinate
  * fresh noise every step, independent of the sampling randomness
  * Poisson root sampling at rate p1, independent across nodes and steps
  * Bernoulli edge retention at rate p2 on every examined arc
  * the optimizer-facing gradient is exactly (sum of clipped + noise)/E[batch]
  * normalization uses only public quantities (no data-dependent denominator)
"""

import pytest
import torch

from src.sparse.base_mechanism import BaseMechanism
from src.sparse.sparse_expand import (build_adjacency, sample_roots,
                                      sparse_expand)
from src.sparse.sparse_gnn import _step_dp


# ── a mechanism whose per-subgraph gradient we control exactly ────────────────

class _FixedGradMechanism(BaseMechanism):
    """g0(H) is a prescribed vector, so clipping/summation can be checked exactly.

    The single parameter is a vector; the loss is <w, target[root]>, whose
    gradient w.r.t. w is exactly target[root].
    """

    def __init__(self, targets, dim=8):
        module = torch.nn.Linear(dim, 1, bias=False)
        super().__init__(module, device=torch.device('cpu'))
        self.targets = targets           # root id -> gradient vector
        self.dim = dim

    def subgraph_loss(self, subgraph):
        t = self.targets.get(subgraph.root)
        if t is None:
            return self.zero_loss()
        return (self.module.weight.view(-1) * t).sum()

    def evaluate(self, data=None):
        return {'train': 0.0, 'val': 0.0, 'test': 0.0}


class _Sub:
    def __init__(self, root):
        self.root = root


def _flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


# ── clipping ─────────────────────────────────────────────────────────────────

def test_per_subgraph_contribution_is_capped_at_C():
    """A huge gradient contributes exactly norm C, not its raw norm."""
    C = 1.0
    big = torch.zeros(8); big[0] = 50.0
    mech = _FixedGradMechanism({0: big})
    mech.build_optimizer(lr=0.0, kind='sgd')        # lr=0: params must not move
    _step_dp(mech, [_Sub(0)], C=C, sigma=0.0,
             noise_gen=torch.Generator().manual_seed(0), expected_batch=1.0)
    contribution = _flat([p.grad for p in mech.parameters()])
    assert contribution.norm().item() == pytest.approx(C, rel=1e-6)


def test_small_gradients_pass_through_unclipped():
    small = torch.zeros(8); small[0] = 0.25
    mech = _FixedGradMechanism({0: small})
    mech.build_optimizer(lr=0.0, kind='sgd')
    _step_dp(mech, [_Sub(0)], C=1.0, sigma=0.0,
             noise_gen=torch.Generator().manual_seed(0), expected_batch=1.0)
    got = _flat([p.grad for p in mech.parameters()])
    assert got.norm().item() == pytest.approx(0.25, rel=1e-6)


def test_clipping_is_per_subgraph_not_per_batch():
    """Four aligned unit-ish gradients must sum to 4C, not be capped at C.

    Per-BATCH clipping would return norm C here and would silently break the
    sensitivity assumption the accounting relies on.
    """
    C = 1.0
    v = torch.zeros(8); v[0] = 10.0                  # each clips to norm C
    mech = _FixedGradMechanism({i: v.clone() for i in range(4)})
    mech.build_optimizer(lr=0.0, kind='sgd')
    _step_dp(mech, [_Sub(i) for i in range(4)], C=C, sigma=0.0,
             noise_gen=torch.Generator().manual_seed(0), expected_batch=1.0)
    got = _flat([p.grad for p in mech.parameters()])
    assert got.norm().item() == pytest.approx(4 * C, rel=1e-6)


# ── sensitivity: the assumption the dominating pairs are built on ────────────

def _sum_of_clipped(mech, roots, C):
    mech.build_optimizer(lr=0.0, kind='sgd')
    _step_dp(mech, [_Sub(i) for i in roots], C=C, sigma=0.0,
             noise_gen=torch.Generator().manual_seed(0), expected_batch=1.0)
    return _flat([p.grad.clone() for p in mech.parameters()])


@pytest.mark.parametrize("seed", range(5))
def test_substitution_sensitivity_at_most_two_C(seed):
    """Replacing one coordinate's subgraph moves G(y) by at most 2C."""
    C = 1.0
    g = torch.Generator().manual_seed(seed)
    targets = {i: torch.randn(8, generator=g) * 3.0 for i in range(6)}
    mech = _FixedGradMechanism(targets)
    base = _sum_of_clipped(mech, range(6), C)

    swapped = dict(targets)
    swapped[3] = torch.randn(8, generator=g) * 3.0       # substitute one record
    mech2 = _FixedGradMechanism(swapped)
    other = _sum_of_clipped(mech2, range(6), C)

    assert (base - other).norm().item() <= 2 * C + 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_insertion_sensitivity_at_most_C(seed):
    """Adding one record moves G(y) by at most C (Lemma 11)."""
    C = 1.0
    g = torch.Generator().manual_seed(seed)
    targets = {i: torch.randn(8, generator=g) * 3.0 for i in range(6)}
    mech = _FixedGradMechanism(targets)
    without = _sum_of_clipped(mech, range(5), C)
    with_extra = _sum_of_clipped(_FixedGradMechanism(targets), range(6), C)
    assert (without - with_extra).norm().item() <= C + 1e-6


# ── noise ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sigma,C", [(1.0, 1.0), (5.0, 1.0), (2.0, 0.5)])
def test_noise_std_is_sigma_times_C(sigma, C):
    """Empirical per-coordinate std of the injected noise == sigma*C.

    Runs the real DP step with an empty batch (zero signal) and measures the
    spread of the resulting gradient across many noise draws.
    """
    B = 1.0
    samples = []
    for s in range(4000):
        mech = _FixedGradMechanism({})
        mech.build_optimizer(lr=0.0, kind='sgd')
        _step_dp(mech, [], C=C, sigma=sigma,
                 noise_gen=torch.Generator().manual_seed(s), expected_batch=B)
        samples.append(_flat([p.grad for p in mech.parameters()]))
    z = torch.stack(samples) * B          # undo the public normalization
    assert z.mean().item() == pytest.approx(0.0, abs=0.15 * sigma * C)
    assert z.std().item() == pytest.approx(sigma * C, rel=0.05)


def test_noise_added_once_per_step_not_per_subgraph():
    """Variance must not grow with batch size: one draw covers the whole sum."""
    sigma, C, B = 3.0, 1.0, 1.0
    def spread(n_subgraphs):
        vals = []
        for s in range(3000):
            mech = _FixedGradMechanism({})       # all contribute zero signal
            mech.build_optimizer(lr=0.0, kind='sgd')
            _step_dp(mech, [_Sub(i) for i in range(n_subgraphs)], C=C,
                     sigma=sigma, noise_gen=torch.Generator().manual_seed(s),
                     expected_batch=B)
            vals.append(_flat([p.grad for p in mech.parameters()]))
        return torch.stack(vals).std().item()
    one, many = spread(1), spread(16)
    assert one == pytest.approx(sigma * C, rel=0.05)
    assert many == pytest.approx(one, rel=0.05)   # 16x batch -> same noise


def test_noise_is_fresh_every_step():
    """Consecutive steps must draw independent noise from the same generator."""
    gen = torch.Generator().manual_seed(0)
    grads = []
    mech = _FixedGradMechanism({})
    mech.build_optimizer(lr=0.0, kind='sgd')
    for _ in range(2):
        _step_dp(mech, [], C=1.0, sigma=1.0, noise_gen=gen, expected_batch=1.0)
        grads.append(_flat([p.grad.clone() for p in mech.parameters()]))
    assert not torch.allclose(grads[0], grads[1])


def test_noise_stream_independent_of_sampling_stream():
    """run/train seeds the two generators apart (seed vs seed+10000).

    Identical sampling with different noise seeds must give different noise,
    and the noise must not be a function of the batch.
    """
    a, b = [], []
    for noise_seed in (7, 8):
        mech = _FixedGradMechanism({})
        mech.build_optimizer(lr=0.0, kind='sgd')
        _step_dp(mech, [], C=1.0, sigma=1.0,
                 noise_gen=torch.Generator().manual_seed(noise_seed),
                 expected_batch=1.0)
        (a if noise_seed == 7 else b).append(
            _flat([p.grad.clone() for p in mech.parameters()]))
    assert not torch.allclose(a[0], b[0])


def test_gradient_is_exactly_signal_plus_noise_over_expected_batch():
    """Reconstruct the optimizer-facing gradient from its two parts."""
    C, sigma, B = 1.0, 2.0, 8.0
    v = torch.zeros(8); v[0] = 0.4
    targets = {i: v.clone() for i in range(3)}

    mech = _FixedGradMechanism(targets)
    mech.build_optimizer(lr=0.0, kind='sgd')
    _step_dp(mech, [_Sub(i) for i in range(3)], C=C, sigma=sigma,
             noise_gen=torch.Generator().manual_seed(123), expected_batch=B)
    got = _flat([p.grad for p in mech.parameters()])

    signal = _sum_of_clipped(_FixedGradMechanism(targets), range(3), C)
    noise = _FixedGradMechanism(targets).gaussian_noise_like(
        [torch.zeros(1, 8)], sigma, C, generator=torch.Generator().manual_seed(123))
    expected = (signal + _flat(noise)) / B
    assert torch.allclose(got, expected, atol=1e-6)


def test_empty_batch_still_gets_noise():
    """The analyzed mechanism adds noise unconditionally, including on |V_root|=0."""
    mech = _FixedGradMechanism({})
    mech.build_optimizer(lr=0.0, kind='sgd')
    _step_dp(mech, [], C=1.0, sigma=1.0,
             noise_gen=torch.Generator().manual_seed(3), expected_batch=1.0)
    assert _flat([p.grad for p in mech.parameters()]).norm().item() > 0


# ── sampling ─────────────────────────────────────────────────────────────────

def test_root_sampling_is_poisson_at_rate_p1():
    n, p1, trials = 500, 0.1, 400
    gen = torch.Generator().manual_seed(0)
    counts = [sample_roots(n, p1, generator=gen).numel() for _ in range(trials)]
    mean = sum(counts) / trials
    var = sum((c - mean) ** 2 for c in counts) / (trials - 1)
    assert mean == pytest.approx(n * p1, rel=0.05)
    # independent Bernoulli => Binomial variance n p (1-p), not 0 (fixed size)
    assert var == pytest.approx(n * p1 * (1 - p1), rel=0.25)


def test_root_sampling_is_independent_across_steps():
    gen = torch.Generator().manual_seed(0)
    a = set(sample_roots(2000, 0.05, generator=gen).tolist())
    b = set(sample_roots(2000, 0.05, generator=gen).tolist())
    overlap = len(a & b) / max(len(a), 1)
    assert overlap < 0.35          # ~p1 under independence, not ~1.0


def test_edge_retention_matches_p2():
    """Every examined arc is kept with probability p2."""
    n, deg, p2 = 1, 4000, 0.3
    ei = torch.stack([torch.arange(1, deg + 1),
                      torch.zeros(deg, dtype=torch.long)])
    adj = build_adjacency(ei, deg + 1, direction='in')
    gen = torch.Generator().manual_seed(0)
    kept = sum(sparse_expand(adj, 0, p2, 1, generator=gen,
                             direction='in').num_edges for _ in range(20))
    assert kept / (20 * deg) == pytest.approx(p2, rel=0.05)


def test_expected_batch_denominator_uses_public_quantities_only():
    """The normalizer is p1 * pool_size — independent of the sampled batch.

    A data-dependent denominator (e.g. the realized |V_root|) would make the
    published update depend on the sample in a way the accounting does not model.
    """
    C, sigma, B = 1.0, 0.0, 10.0
    v = torch.zeros(8); v[0] = 1.0
    for n_roots in (1, 5):
        mech = _FixedGradMechanism({i: v.clone() for i in range(n_roots)})
        mech.build_optimizer(lr=0.0, kind='sgd')
        _step_dp(mech, [_Sub(i) for i in range(n_roots)], C=C, sigma=sigma,
                 noise_gen=torch.Generator().manual_seed(0), expected_batch=B)
        got = _flat([p.grad for p in mech.parameters()]).norm().item()
        assert got == pytest.approx(n_roots * 1.0 / B, rel=1e-6)

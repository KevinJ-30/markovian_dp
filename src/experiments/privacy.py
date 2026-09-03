"""Small interoperability layer for method-specific privacy accountants.

The wrappers retain each method's native accounting convention rather than
pretending that every mechanism is DP-SGD.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PrivacyResult:
    epsilon: float | None
    delta: float | None
    accountant: str
    rdp_orders: tuple[float, ...] = ()
    rdp_values: tuple[float, ...] = ()
    noise_multiplier: float | None = None
    sampling_probability: float | None = None
    composition_count: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrivacyAccountant:
    """Protocol-like base class used by the experiment runner."""

    def account(self, **kwargs: Any) -> PrivacyResult:
        raise NotImplementedError

    def calibrate(self, target_epsilon: float, delta: float, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError


class SparseGNNAccountant(PrivacyAccountant):
    """Adapter around the repository's certified dominating-pair accountant."""

    def account(self, *, p1: float, p2: float, radius: int, k_in: int, k_out: int,
                sigma: float, steps: int, delta: float, direction: str = "in",
                theorem: str = "auto", grid: float = 1e-4) -> PrivacyResult:
        from src.sparse.accounting import (
            sparsegnn_epsilon, sparsegnn_theorem_label,
        )

        epsilon = sparsegnn_epsilon(
            p1, p2, radius, k_in, sigma, steps, delta, K_out=k_out,
            direction=direction, theorem=theorem, grid=grid,
        )
        return PrivacyResult(
            epsilon=float(epsilon), delta=delta, accountant="markovian_dp.dominating_pairs",
            noise_multiplier=sigma, sampling_probability=p1, composition_count=steps,
            parameters={"edge_retention_probability": p2, "radius": radius,
                        "k_in": k_in, "k_out": k_out, "direction": direction,
                        "theorem": sparsegnn_theorem_label(direction, theorem),
                        "grid": grid},
        )

    def calibrate(self, target_epsilon: float, delta: float,
                  **kwargs: Any) -> Mapping[str, Any]:
        from src.sparse.accounting import calibrate_sparsegnn_noise

        calibration = calibrate_sparsegnn_noise(
            target_epsilon=target_epsilon, target_delta=delta,
            p1=kwargs["p1"], p2=kwargs["p2"], r=kwargs["radius"],
            K_in=kwargs["k_in"], K_out=kwargs["k_out"], steps=kwargs["steps"],
            clip=kwargs.get("clip", 1.0), direction=kwargs.get("direction", "in"),
            theorem=kwargs.get("theorem", "auto"), grid=kwargs.get("grid", 1e-4),
            sigma_rtol=kwargs.get("sigma_rtol", 1e-3),
            sigma_atol=kwargs.get("sigma_atol", 1e-6),
            max_sigma=kwargs.get("max_sigma", 1e6),
        )
        return calibration.as_dict()


class DPMLPAccountant(PrivacyAccountant):
    """Adapter around the standard Poisson-subsampled Gaussian accountant."""

    def account(self, *, noise_multiplier: float, sample_rate: float, steps: int,
                delta: float, mechanism: str = "prv") -> PrivacyResult:
        from src.sparse.accounting import naive_opacus_epsilon

        epsilon = naive_opacus_epsilon(noise_multiplier, sample_rate, steps, delta, mechanism)
        return PrivacyResult(
            epsilon=float(epsilon), delta=delta, accountant=f"opacus.{mechanism}",
            noise_multiplier=noise_multiplier, sampling_probability=sample_rate,
            composition_count=steps,
        )

    def calibrate(self, target_epsilon: float, delta: float, **kwargs: Any) -> Mapping[str, Any]:
        sample_rate = float(kwargs["sample_rate"])
        steps = int(kwargs["steps"])
        lo, hi = 1e-3, 1.0
        while self.account(noise_multiplier=hi, sample_rate=sample_rate, steps=steps,
                           delta=delta).epsilon > target_epsilon:
            hi *= 2.0
            if hi > 1e6:
                raise RuntimeError("failed to bracket a DP-SGD noise multiplier")
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.account(noise_multiplier=mid, sample_rate=sample_rate, steps=steps,
                            delta=delta).epsilon > target_epsilon:
                lo = mid
            else:
                hi = mid
        return {"noise_multiplier": hi}


@dataclass(frozen=True)
class DPARNoiseCalibration:
    """Paper-aligned DPAR noise parameters for one target privacy budget."""

    ppr_noise_std: float
    ppr_noise_variance: float
    ppr_epsilon: float
    ppr_delta: float
    ppr_delta_per_release: float
    sgd_noise_multiplier: float
    sgd_noise_std: float
    sgd_noise_variance: float
    sgd_epsilon: float
    sgd_delta: float
    target_epsilon: float
    target_delta: float
    amplification_rate: float
    ppr_releases: int
    sgd_evaluations: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DPARAccountant(PrivacyAccountant):
    """Faithful wrapper of the privacy arithmetic printed by upstream DPAR.

    This intentionally preserves the repository's stated amplification and
    composition convention, including its separate DP-PPR and DP-SGD reports.
    It does not claim a joint composition theorem that upstream does not supply.
    """

    @staticmethod
    def _inverse_composition(value: float, releases: int, delta: float) -> float:
        if value <= 0:
            return 0.0
        if releases <= 0 or delta <= 0:
            raise ValueError("releases and delta must be positive")

        def composed(epsilon: float) -> float:
            return epsilon / (2.0 * math.sqrt(releases * math.log(math.e + epsilon / delta)))

        lo, hi = 0.0, max(1.0, value)
        while composed(hi) < value:
            hi *= 2.0
            if hi > 1e12:
                raise RuntimeError("failed to bracket DPAR composition inverse")
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if composed(mid) < value:
                lo = mid
            else:
                hi = mid
        return hi

    def account(self, *, ppr_releases: int, amplification_rate: float, delta: float,
                ppr_clip: float | None = None, ppr_noise: float | None = None,
                em_epsilon: float | None = None, topk: int | None = None,
                report_value_epsilon: float = 0.0, **_: Any) -> PrivacyResult:
        if not 0.0 < amplification_rate <= 1.0:
            raise ValueError("DPAR amplification_rate must lie in (0, 1]")
        if ppr_noise is not None and ppr_clip is not None:
            primitive = math.sqrt(2.0 * math.log(1.25 / delta)) * ppr_clip / ppr_noise
            mechanism = "gaussian_dp_ppr"
        elif em_epsilon is not None and topk is not None:
            primitive = min(
                topk * em_epsilon,
                topk * em_epsilon * (math.exp(em_epsilon) - 1.0) / (math.exp(em_epsilon) + 1.0)
                + em_epsilon * math.sqrt(2.0 * topk * math.log(1.0 / delta)),
            ) + report_value_epsilon
            mechanism = "exponential_dp_ppr"
        else:
            return PrivacyResult(None, None, "dpar.none", parameters={"dp_ppr": False})
        base_delta = 2.0 * delta * ppr_releases
        epsilon = self._inverse_composition(primitive, ppr_releases, base_delta) * amplification_rate
        return PrivacyResult(
            epsilon=epsilon, delta=base_delta * amplification_rate,
            accountant="dpar.upstream_ppr_formula", noise_multiplier=ppr_noise,
            sampling_probability=amplification_rate, composition_count=ppr_releases,
            parameters={"mechanism": mechanism, "topk": topk, "ppr_clip": ppr_clip,
                        "report_value_epsilon": report_value_epsilon},
        )

    def account_training(self, *, noise_multiplier: float, sample_rate: float, steps: int,
                         delta: float, amplification_rate: float) -> PrivacyResult:
        """Use the vendored released DPAR RDP accountant without TensorFlow."""
        if not 0.0 < amplification_rate <= 1.0:
            raise ValueError("DPAR amplification_rate must lie in (0, 1]")
        source = Path(__file__).parents[2] / "third_party" / "DPAR" / "dpgnn" / "privacy_utils" / "rdp_accountant.py"
        spec = importlib.util.spec_from_file_location("_dpar_upstream_rdp", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load DPAR RDP accountant at {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        orders = tuple([1.0 + order / 10.0 for order in range(1, 100)] + list(range(12, 64)))
        rdp = module.compute_rdp(sample_rate, noise_multiplier, steps, orders)
        # main.py uses delta_sgd=0.001 / amplification_rate before multiplying
        # both outputs by that rate. ``delta`` is therefore the final report delta.
        base_delta = delta / amplification_rate
        epsilon, _, _ = module.get_privacy_spent(orders, rdp, target_delta=base_delta)
        return PrivacyResult(
            epsilon=float(epsilon * amplification_rate), delta=delta,
            accountant="dpar.upstream_rdp_accountant", rdp_orders=orders,
            rdp_values=tuple(float(value) for value in rdp),
            noise_multiplier=noise_multiplier, sampling_probability=sample_rate,
            composition_count=steps,
            parameters={"amplification_rate": amplification_rate,
                        "source": "third_party/DPAR/dpgnn/privacy_utils/rdp_accountant.py"},
        )

    def calibrate(self, target_epsilon: float, delta: float, **kwargs: Any) -> Mapping[str, Any]:
        return calibrate_dpar_noise(
            target_epsilon=target_epsilon, target_delta=delta,
            train_nodes=kwargs["train_nodes"], ppr_releases=kwargs["ppr_releases"],
            ppr_clip=kwargs["ppr_clip"], sgd_clip=kwargs["sgd_clip"],
            batch_size=kwargs["batch_size"], steps=kwargs["steps"],
            sigma_rtol=kwargs.get("sigma_rtol", 1e-3),
            sigma_atol=kwargs.get("sigma_atol", 1e-6),
            max_noise_multiplier=kwargs.get("max_noise_multiplier", 1e6),
        ).as_dict()


def _dpar_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _dpar_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def calibrate_dpar_noise(
    *, target_epsilon: float, target_delta: float, train_nodes: int, ppr_releases: int,
    ppr_clip: float, sgd_clip: float, batch_size: int, steps: int,
    sigma_rtol: float = 1e-3, sigma_atol: float = 1e-6,
    max_noise_multiplier: float = 1e6,
) -> DPARNoiseCalibration:
    """Calibrate Gaussian DP-APPR and DP-SGD to equal final DPAR sub-budgets."""
    target_epsilon = _dpar_positive_finite("target_epsilon", target_epsilon)
    target_delta = float(target_delta)
    if not math.isfinite(target_delta) or not 0.0 < target_delta < 1.0:
        raise ValueError("target_delta must lie in (0, 1)")
    train_nodes = _dpar_positive_int("train_nodes", train_nodes)
    ppr_releases = _dpar_positive_int("ppr_releases", ppr_releases)
    batch_size = _dpar_positive_int("batch_size", batch_size)
    steps = _dpar_positive_int("steps", steps)
    ppr_clip = _dpar_positive_finite("ppr_clip", ppr_clip)
    sgd_clip = _dpar_positive_finite("sgd_clip", sgd_clip)
    sigma_rtol = _dpar_positive_finite("sigma_rtol", sigma_rtol)
    sigma_atol = _dpar_positive_finite("sigma_atol", sigma_atol)
    max_noise_multiplier = _dpar_positive_finite("max_noise_multiplier", max_noise_multiplier)
    if ppr_releases > train_nodes:
        raise ValueError("ppr_releases must not exceed train_nodes")
    if batch_size > ppr_releases:
        raise ValueError("batch_size must not exceed ppr_releases")

    amplification_rate = ppr_releases / train_nodes
    ppr_epsilon = sgd_epsilon = target_epsilon / 2.0
    ppr_delta = sgd_delta = target_delta / 2.0
    epsilon_g = ppr_epsilon / amplification_rate
    delta_g = ppr_delta / amplification_rate
    ppr_delta_per_release = delta_g / (2.0 * ppr_releases)
    if ppr_delta_per_release >= 1.25:
        raise ValueError("DPAR per-release PPR delta must be less than 1.25")
    epsilon_0 = epsilon_g / (
        2.0 * math.sqrt(ppr_releases * math.log(math.e + epsilon_g / delta_g))
    )
    ppr_noise_std = ppr_clip * math.sqrt(2.0 * math.log(1.25 / ppr_delta_per_release)) / epsilon_0

    accountant = DPARAccountant()
    candidates: dict[float, PrivacyResult] = {}

    def candidate(multiplier: float) -> PrivacyResult:
        if multiplier not in candidates:
            candidates[multiplier] = accountant.account_training(
                noise_multiplier=multiplier, sample_rate=batch_size / ppr_releases,
                steps=steps, delta=sgd_delta, amplification_rate=amplification_rate,
            )
        return candidates[multiplier]

    high = min(1.0, max_noise_multiplier)
    if candidate(high).epsilon is None:
        raise RuntimeError("DPAR DP-SGD accounting did not return epsilon")
    while candidate(high).epsilon > sgd_epsilon:
        if high >= max_noise_multiplier:
            raise RuntimeError(
                "failed to bracket a DPAR DP-SGD noise multiplier at "
                f"max_noise_multiplier={max_noise_multiplier}"
            )
        high = min(high * 2.0, max_noise_multiplier)
    low = 0.0
    while high - low > max(sigma_atol, sigma_rtol * high):
        midpoint = (low + high) / 2.0
        if candidate(midpoint).epsilon > sgd_epsilon:
            low = midpoint
        else:
            high = midpoint
    safe = candidate(high)
    return DPARNoiseCalibration(
        ppr_noise_std=ppr_noise_std, ppr_noise_variance=ppr_noise_std ** 2,
        ppr_epsilon=ppr_epsilon, ppr_delta=ppr_delta,
        ppr_delta_per_release=ppr_delta_per_release,
        sgd_noise_multiplier=high, sgd_noise_std=high * sgd_clip,
        sgd_noise_variance=(high * sgd_clip) ** 2,
        sgd_epsilon=float(safe.epsilon), sgd_delta=sgd_delta,
        target_epsilon=target_epsilon, target_delta=target_delta,
        amplification_rate=amplification_rate, ppr_releases=ppr_releases,
        sgd_evaluations=len(candidates),
    )



def account(method: str, **kwargs: Any) -> PrivacyResult:
    """Dispatch the common ``account(...)`` surface without hidden defaults."""
    accountants = {
        "sparse_gnn": SparseGNNAccountant(),
        "dp_mlp": DPMLPAccountant(),
        "dpar": DPARAccountant(),
    }
    try:
        return accountants[method].account(**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown accountant {method!r}") from exc

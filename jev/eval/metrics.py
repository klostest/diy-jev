"""Calibration metrics, shared by the toy model and the Jev probe.

Every metric here reduces to one of two input shapes:

  - (confidence, correct) pairs -- confidence in the *chosen* answer, and
    whether that answer was right. This is what ECE, the risk-coverage
    curve, and the "top-1" Brier decomposition consume; it works identically
    for noul/choice/score once the caller has reduced its answer to a
    right/wrong outcome.
  - (p_true, p_model) distributions over the full label set -- consumed only
    by posterior_kl, which needs the full vector because it is the one
    metric that requires owning the data-generating process (see
    dgp/generator.py) rather than merely observing outcomes.

See docs/explainers.md for the derivation of each metric; this module is
implementation only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


Array = np.ndarray
BinningMethod = Literal["equal_width", "equal_mass"]


def _as_array(x) -> Array:
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def _bin_edges(confidences: Array, n_bins: int, method: BinningMethod) -> Array:
    if method == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if method == "equal_mass":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confidences, qs)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        return edges
    raise ValueError(f"unknown binning method '{method}'")


def _bin_assignments(confidences: Array, edges: Array) -> Array:
    # right-closed bins except the last edge is inclusive on both ends
    idx = np.searchsorted(edges, confidences, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx


@dataclass(frozen=True)
class BinStats:
    edges: Array
    n_bins: int
    counts: Array          # n_k
    mean_confidence: Array  # pbar_k (nan where n_k == 0)
    mean_outcome: Array     # obar_k (nan where n_k == 0)


def compute_bin_stats(confidences: Sequence[float], outcomes: Sequence[float],
                       n_bins: int = 15, method: BinningMethod = "equal_width") -> BinStats:
    conf, out = _as_array(confidences), _as_array(outcomes)
    edges = _bin_edges(conf, n_bins, method)
    k = len(edges) - 1
    assign = _bin_assignments(conf, edges)
    counts = np.zeros(k)
    mean_conf = np.full(k, np.nan)
    mean_out = np.full(k, np.nan)
    for b in range(k):
        mask = assign == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_conf[b] = conf[mask].mean()
            mean_out[b] = out[mask].mean()
    return BinStats(edges=edges, n_bins=k, counts=counts, mean_confidence=mean_conf, mean_outcome=mean_out)


# ---------------------------------------------------------------------------
# ECE + noise floor
# ---------------------------------------------------------------------------


def expected_calibration_error(confidences: Sequence[float], outcomes: Sequence[float],
                                n_bins: int = 15, method: BinningMethod = "equal_width") -> float:
    """Standard binned ECE: sum over non-empty bins of (n_k/N)*|pbar_k - obar_k|."""
    stats = compute_bin_stats(confidences, outcomes, n_bins, method)
    n = stats.counts.sum()
    if n == 0:
        return 0.0
    nonempty = stats.counts > 0
    weights = stats.counts[nonempty] / n
    gaps = np.abs(stats.mean_confidence[nonempty] - stats.mean_outcome[nonempty])
    return float(np.sum(weights * gaps))


def ece_noise_floor(confidences: Sequence[float], n_boot: int = 200, n_bins: int = 15,
                     method: BinningMethod = "equal_width", seed: int = 0) -> Tuple[float, float]:
    """ECE is upward-biased at finite n even for a perfectly calibrated source
    (binning + sampling noise). Simulate that source at these EXACT
    confidences -- outcome_i ~ Bernoulli(confidence_i) -- and report the
    resulting ECE's mean and std across n_boot draws: the floor any real ECE
    should be compared against, not zero.
    """
    conf = _as_array(confidences)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        sim_outcomes = rng.binomial(1, conf)
        vals[i] = expected_calibration_error(conf, sim_outcomes, n_bins, method)
    return float(vals.mean()), float(vals.std())


# ---------------------------------------------------------------------------
# Brier score and the Murphy/Brier decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrierDecomposition:
    brier_raw: float       # the actual score: mean((p_i - o_i)^2) with raw p_i
    brier_binned: float    # mean((pbar_k - o_i)^2): what the decomposition reconstructs exactly
    reliability: float     # (1/N) sum_k n_k (pbar_k - obar_k)^2 -- the calibration gap
    resolution: float      # (1/N) sum_k n_k (obar_k - obar)^2 -- how much confidence sorts outcomes
    uncertainty: float      # obar*(1-obar) -- the irreducible floor at this base rate
    grouping_loss: float    # brier_raw - brier_binned; -> 0 as n_bins -> n (finer bins)


def brier_decomposition(confidences: Sequence[float], outcomes: Sequence[float],
                         n_bins: int = 15, method: BinningMethod = "equal_width") -> BrierDecomposition:
    """Murphy (1973) decomposition: brier_binned = reliability - resolution + uncertainty,
    EXACT for any bin-constant representative value (see docs/explainers.md for
    the derivation) -- which is why this reports brier_binned, not brier_raw,
    as the quantity the identity reconstructs, alongside the raw score you'd
    actually quote and the gap between them.
    """
    conf, out = _as_array(confidences), _as_array(outcomes)
    n = len(conf)
    obar = out.mean()
    uncertainty = obar * (1.0 - obar)

    stats = compute_bin_stats(conf, out, n_bins, method)
    nonempty = stats.counts > 0
    n_k = stats.counts[nonempty]
    pbar_k = stats.mean_confidence[nonempty]
    obar_k = stats.mean_outcome[nonempty]

    reliability = float(np.sum(n_k * (pbar_k - obar_k) ** 2) / n)
    resolution = float(np.sum(n_k * (obar_k - obar) ** 2) / n)
    brier_binned = reliability - resolution + uncertainty
    brier_raw = float(np.mean((conf - out) ** 2))

    return BrierDecomposition(
        brier_raw=brier_raw, brier_binned=brier_binned,
        reliability=reliability, resolution=resolution, uncertainty=float(uncertainty),
        grouping_loss=brier_raw - brier_binned,
    )


# ---------------------------------------------------------------------------
# Temperature scaling (fit on recovered logits)
# ---------------------------------------------------------------------------


def logits_from_probs(probs: Array, eps: float = 1e-12) -> Array:
    """log(p) is an exact logit vector for `probs` under softmax: softmax is
    shift-invariant, so softmax(log p) == p exactly (no information lost, no
    approximation) -- this is what lets temperature scaling apply to a model
    that only ever returns probabilities, like Jev, with no separate access
    to logits."""
    return np.log(np.clip(_as_array(probs), eps, 1.0))


def _softmax(logits: Array, axis: int = -1) -> Array:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature_multiclass(probs: Array, labels: Sequence[int], t_bounds: Tuple[float, float] = (0.05, 20.0)) -> float:
    """Fits scalar T minimizing NLL of softmax(logit(probs)/T) against hard
    labels. T > 1 shrinks confidence toward uniform (fixes overconfidence);
    T < 1 sharpens it (fixes underconfidence)."""
    probs = _as_array(probs)
    labels = np.asarray(labels, dtype=np.int64)
    logits = logits_from_probs(probs)

    def nll(t: float) -> float:
        scaled = _softmax(logits / t, axis=-1)
        chosen = scaled[np.arange(len(labels)), labels]
        return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))

    res = minimize_scalar(nll, bounds=t_bounds, method="bounded")
    return float(res.x)


def fit_temperature_binary(probs: Sequence[float], outcomes: Sequence[float], t_bounds: Tuple[float, float] = (0.05, 20.0)) -> float:
    """Binary case: probs are P(outcome=1); fits T for sigmoid(logit(p)/T)."""
    p = _as_array(probs)
    y = _as_array(outcomes)
    logit = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1 - 1e-12))

    def nll(t: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-logit / t))
        scaled = np.clip(scaled, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled)))

    res = minimize_scalar(nll, bounds=t_bounds, method="bounded")
    return float(res.x)


# ---------------------------------------------------------------------------
# Posterior divergence -- requires owning the true generative posterior
# ---------------------------------------------------------------------------


def kl_divergence(p_true: Array, p_model: Array, eps: float = 1e-12) -> Array:
    """Per-item KL(p_true || p_model), vectorized over the last axis.
    0 * log(0/x) is treated as 0 by convention (support of p_true only)."""
    p_true = _as_array(p_true)
    p_model = np.clip(_as_array(p_model), eps, 1.0)
    log_true = np.log(np.clip(p_true, eps, 1.0))
    terms = np.where(p_true > 0, p_true * (log_true - np.log(p_model)), 0.0)
    return terms.sum(axis=-1)


def mean_posterior_divergence(p_true: Array, p_model: Array) -> dict:
    kl = kl_divergence(p_true, p_model)
    l1 = np.abs(_as_array(p_true) - _as_array(p_model)).sum(axis=-1)
    return {
        "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
        "mean_l1": float(l1.mean()), "median_l1": float(np.median(l1)),
    }


# ---------------------------------------------------------------------------
# Selective prediction: risk-coverage curve + AURC
# ---------------------------------------------------------------------------


def risk_coverage_curve(confidences: Sequence[float], correctness: Sequence[float]) -> Tuple[Array, Array]:
    conf, correct = _as_array(confidences), _as_array(correctness)
    order = np.argsort(-conf, kind="stable")  # most confident first
    sorted_correct = correct[order]
    n = len(conf)
    cum_errors = np.cumsum(1.0 - sorted_correct)
    coverage = np.arange(1, n + 1) / n
    risk = cum_errors / np.arange(1, n + 1)
    return coverage, risk


def area_under_risk_coverage(coverage: Array, risk: Array) -> float:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz
    return float(trapezoid(risk, coverage))


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return float(center - half), float(center + half)


def bootstrap_ci(values: Sequence[float], statistic=np.mean, n_boot: int = 1000,
                  ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    values = _as_array(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        stats[i] = statistic(sample)
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return float(np.percentile(stats, lo)), float(np.percentile(stats, hi))

"""Suite tests use controlled test doubles with a KNOWN, injected
calibration property (never a claim about real Jev) to verify each suite's
parsing and metric arithmetic actually detects what it's supposed to --
the same oracle/detector pattern as test_metrics.py, one level up the stack.
"""
from __future__ import annotations

import numpy as np
import pytest

from jev.dgp.generator import CATEGORY_LABELS, URGENCY_LEVELS
from jev.probe.client import BaseJevClient, JevResponse
from jev.probe.suites import (
    CATEGORY_QUESTION,
    HP_CHOICE_QUESTION,
    HP_EXPEDITE_LABELS,
    HP_NOUL_QUESTION,
    HP_SCORE_QUESTION,
    run_s1_resolvable,
    run_s2_known_ambiguous,
    run_s3_unanswerable,
    run_s4_type_consistency,
    run_s5_confidence_vs_probabilities,
    run_s6_prior_sensitivity,
)


def _sharpen(probs: dict, factor: float) -> dict:
    keys = list(probs.keys())
    logits = np.array([np.log(max(probs[k], 1e-9)) for k in keys])
    scaled = np.exp(logits * factor)
    scaled = scaled / scaled.sum()
    return {k: float(v) for k, v in zip(keys, scaled)}


class OracleCategoryClient(BaseJevClient):
    """Answers CATEGORY choice questions with the EXACT true posterior
    (looked up by exact text match against items the test generated),
    optionally sharpened by a known temperature factor."""

    def __init__(self, items, sharpen: float = 1.0, model_name: str = "oracle-test-double"):
        self.by_text = {it.text: it.posteriors["category"] for it in items}
        self.sharpen = sharpen
        self.model_name = model_name

    def ask(self, state, questions):
        assert "category" in questions
        true_post = self.by_text[state]
        probs = _sharpen(true_post, self.sharpen) if self.sharpen != 1.0 else dict(true_post)
        chosen = max(probs, key=probs.get)
        answer = {"type": "choice", "choice": chosen, "confidence": probs[chosen], "probabilities": probs}
        raw = {"model": self.model_name, "answers": {"category": answer}, "usage": {}}
        return JevResponse(model=self.model_name, answers={"category": answer}, usage={}, raw=raw)


def test_s1_oracle_client_is_near_perfect_on_resolvable_items():
    from jev.dgp.scenarios import build_resolvable
    items = build_resolvable("category", n=200, seed=1, min_confidence=0.97)
    client = OracleCategoryClient(items)
    # run_s1_resolvable regenerates its own item set internally with the SAME seed/params,
    # so it independently reconstructs the same resolvable pool the oracle was built from.
    result = run_s1_resolvable(client, n=200, seed=1)
    assert result.metrics["accuracy"] > 0.9
    assert result.metrics["mean_confidence"] > 0.9
    assert result.model_versions == {"oracle-test-double"}
    assert result.n_items == 200


def test_s2_oracle_client_recovers_target_posteriors_almost_exactly():
    from jev.dgp.scenarios import build_posterior_sweep
    levels = (0.3, 0.5, 0.7, 0.9)
    sweep = build_posterior_sweep("category", positive_labels=["billing"], levels=levels, n_per_level=20, seed=2)
    client = OracleCategoryClient([s.item for s in sweep])
    result = run_s2_known_ambiguous(client, levels=levels, n_per_level=20, seed=2)
    assert result.metrics["mean_kl"] < 1e-6
    assert abs(result.metrics["mean_signed_error_billing"]) < 1e-6


def test_s2_detects_a_known_sharpening_factor():
    from jev.dgp.scenarios import build_posterior_sweep
    levels = (0.3, 0.5, 0.7, 0.9)
    sweep = build_posterior_sweep("category", positive_labels=["billing"], levels=levels, n_per_level=20, seed=3)
    client = OracleCategoryClient([s.item for s in sweep], sharpen=3.0)
    result = run_s2_known_ambiguous(client, levels=levels, n_per_level=20, seed=3)
    # sharpening pushes probabilities of the already-favored label further from truth
    assert result.metrics["mean_kl"] > 0.05
    assert result.metrics["mean_abs_error_billing"] > 0.03


class FixedConfidenceUnanswerableClient(BaseJevClient):
    """Ignores the text entirely and always answers with a fixed confident
    choice for HIDDEN_PRIORITY -- the injected version of the real published
    'confident on an unanswerable question' failure mode."""

    def __init__(self, fixed_choice: str = "high", fixed_confidence: float = 0.75, model_name: str = "overconfident-test-double"):
        self.fixed_choice = fixed_choice
        self.fixed_confidence = fixed_confidence
        self.model_name = model_name

    def ask(self, state, questions):
        other = (1.0 - self.fixed_confidence) / 3
        probs = {l: (self.fixed_confidence if l == self.fixed_choice else other) for l in URGENCY_LEVELS}
        answer = {"type": "choice", "choice": self.fixed_choice, "confidence": self.fixed_confidence, "probabilities": probs}
        raw = {"model": self.model_name, "answers": {"priority": answer}, "usage": {}}
        return JevResponse(model=self.model_name, answers={"priority": answer}, usage={}, raw=raw)


def test_s3_detects_injected_overconfidence_on_unanswerable_question():
    client = FixedConfidenceUnanswerableClient(fixed_choice="high", fixed_confidence=0.75)
    result = run_s3_unanswerable(client, n=300, seed=4)
    # true prior max is 0.4 ("low") but the client always answers "high" at 0.75 confidence
    assert result.metrics["bayes_optimal_accuracy"] == pytest.approx(0.4, abs=1e-9)
    assert result.metrics["accuracy"] < 0.35  # "high" is not the modal true label
    assert result.metrics["overconfidence_gap"] > 0.2  # confident well beyond what it could possibly know


def test_s3_well_calibrated_client_shows_no_overconfidence_gap():
    class PriorAwareClient(BaseJevClient):
        """Answers HIDDEN_PRIORITY honestly with the true prior every time --
        the well-calibrated counterfactual to the test above."""

        def ask(self, state, questions):
            from jev.dgp.generator import HIDDEN_PRIORITY
            probs = dict(HIDDEN_PRIORITY.prior)
            chosen = max(probs, key=probs.get)
            answer = {"type": "choice", "choice": chosen, "confidence": probs[chosen], "probabilities": probs}
            raw = {"model": "prior-aware-test-double", "answers": {"priority": answer}, "usage": {}}
            return JevResponse(model="prior-aware-test-double", answers={"priority": answer}, usage={}, raw=raw)

    result = run_s3_unanswerable(PriorAwareClient(), n=300, seed=5)
    assert abs(result.metrics["overconfidence_gap"]) < 0.05


class TypeConsistentClient(BaseJevClient):
    """Answers all three HIDDEN_PRIORITY views from ONE shared probability,
    by construction internally consistent."""

    def __init__(self, p_expedite: float = 0.3, model_name: str = "consistent-test-double"):
        self.p = p_expedite
        self.model_name = model_name

    def ask(self, state, questions):
        answers = {}
        if "noul" in questions:
            answers["noul"] = {"type": "noul", "noul": self.p}
        if "choice" in questions:
            rest = (1 - self.p) / 2
            probs = {"low": rest, "medium": rest, "high": self.p / 2, "critical": self.p / 2}
            answers["choice"] = {"type": "choice", "choice": "critical", "confidence": self.p / 2, "probabilities": probs}
        if "score" in questions:
            rest = (1 - self.p) / 2
            probs = {"low": rest, "medium": rest, "high": self.p / 2, "critical": self.p / 2}
            answers["score"] = {"type": "score", "score": 2.0, "confidence": 0.5, "probabilities": probs,
                                 "legend": {"0": "low", "1": "medium", "2": "high", "3": "critical"}}
        raw = {"model": self.model_name, "answers": answers, "usage": {}}
        return JevResponse(model=self.model_name, answers=answers, usage={}, raw=raw)


class TypeInconsistentClient(BaseJevClient):
    """Deliberately returns DIFFERENT expedite-probabilities per view."""

    def __init__(self, model_name: str = "inconsistent-test-double"):
        self.model_name = model_name

    def ask(self, state, questions):
        answers = {
            "noul": {"type": "noul", "noul": 0.20},
            "choice": {"type": "choice", "choice": "high", "confidence": 0.5,
                       "probabilities": {"low": 0.25, "medium": 0.25, "high": 0.30, "critical": 0.20}},
            "score": {"type": "score", "score": 2.5, "confidence": 0.6,
                      "probabilities": {"low": 0.05, "medium": 0.05, "high": 0.30, "critical": 0.60},
                      "legend": {"0": "low", "1": "medium", "2": "high", "3": "critical"}},
        }
        raw = {"model": self.model_name, "answers": answers, "usage": {}}
        return JevResponse(model=self.model_name, answers=answers, usage={}, raw=raw)


def test_s4_consistent_client_shows_near_zero_disagreement():
    result = run_s4_type_consistency(TypeConsistentClient(p_expedite=0.3), n=50, seed=6)
    assert result.metrics["mean_abs_noul_vs_choice"] < 1e-6
    assert result.metrics["mean_abs_noul_vs_score"] < 1e-6
    assert result.metrics["true_p_expedite"] == pytest.approx(0.3, abs=1e-9)


def test_s4_inconsistent_client_is_detected():
    # choice: high+critical = 0.30+0.20=0.50; score: 0.30+0.60=0.90; noul: 0.20
    result = run_s4_type_consistency(TypeInconsistentClient(), n=50, seed=7)
    assert result.metrics["mean_abs_noul_vs_choice"] == pytest.approx(0.30, abs=1e-6)
    assert result.metrics["mean_abs_noul_vs_score"] == pytest.approx(0.70, abs=1e-6)
    assert result.metrics["mean_abs_choice_vs_score"] == pytest.approx(0.40, abs=1e-6)


class ConfidenceEqualsMaxProbClient(BaseJevClient):
    """confidence == max(probabilities) exactly, but VARIES across items
    (keyed off a hash of the state text) so the correlation this test checks
    is actually well-defined rather than degenerate."""

    def ask(self, state, questions):
        top = 0.4 + 0.5 * ((hash(state) % 1000) / 1000.0)  # varies in [0.4, 0.9)
        rest = (1.0 - top) / 3
        probs = {"billing": top, "technical": rest, "sales": rest, "spam": rest}
        answer = {"type": "choice", "choice": "billing", "confidence": top, "probabilities": probs}
        raw = {"model": "s5-double", "answers": {"category": answer}, "usage": {}}
        return JevResponse(model="s5-double", answers={"category": answer}, usage={}, raw=raw)


def test_s5_runs_and_reports_finite_correlation():
    result = run_s5_confidence_vs_probabilities(ConfidenceEqualsMaxProbClient(), n=40, seed=8)
    # confidence == max(probabilities) exactly here -> perfect correlation, identical AURC
    assert result.metrics["correlation_confidence_vs_max_probability"] == pytest.approx(1.0, abs=1e-6)
    assert result.metrics["aurc_confidence"] == pytest.approx(result.metrics["aurc_max_probability"], abs=1e-9)


class IgnoresStatedPriorClient(BaseJevClient):
    """Returns the identical probabilities regardless of the instructions
    text stating a base rate -- the injected 'ignores explicit prior' case."""

    def ask(self, state, questions):
        probs = {"billing": 0.25, "technical": 0.25, "sales": 0.25, "spam": 0.25}
        answer = {"type": "choice", "choice": "billing", "confidence": 0.25, "probabilities": probs}
        raw = {"model": "s6-double", "answers": {"category": answer}, "usage": {}}
        return JevResponse(model="s6-double", answers={"category": answer}, usage={}, raw=raw)


def test_s6_detects_zero_responsiveness_to_stated_prior():
    result = run_s6_prior_sensitivity(IgnoresStatedPriorClient(), n=30, seed=9)
    assert result.metrics["mean_observed_shift"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["mean_bayes_correct_shift"] > 0.05  # the Bayes target DOES move
    assert result.metrics["mean_shift_deficit"] > 0.05

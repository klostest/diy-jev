"""The six probe suites, S1-S6. Each pre-registers a hypothesis and a
falsification condition as a module-level constant, before any response is
looked at, per the plan's research-practices section.

A note on schema uncertainty: TypeSafe's documentation is four days old as
of this writing and doesn't give a worked score-type example. The fixture in
probe/fixtures/ is this project's best-guess encoding (probabilities keyed
by level name, plus a legend); _extract_probabilities() below handles that
convention and a plausible index-keyed alternative, and raises clearly if
neither matches -- inspect the first live response and tighten this if it's
wrong, rather than silently mis-parsing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from jev.dgp.generator import (
    CATEGORY,
    CATEGORY_LABELS,
    HIDDEN_PRIORITY,
    URGENCY_LEVELS,
    bayes_error_rate,
    binary_posterior,
    posterior as exact_posterior,
)
from jev.dgp.scenarios import (
    build_ambiguous,
    build_posterior_sweep,
    build_resolvable,
    build_unanswerable,
    generate_fixed_combo_batch,
)
from jev.eval import metrics as m
from jev.probe.client import BaseJevClient, Question


# ---------------------------------------------------------------------------
# Question bank: one Question per concept/view.
# ---------------------------------------------------------------------------

CATEGORY_CRITERIA = {
    "billing": "About invoices, payments, refunds, or subscription costs.",
    "technical": "About bugs, crashes, error codes, or app malfunctions.",
    "sales": "About pricing, plan features, or discounts.",
    "spam": "Unsolicited marketing, or an unsubscribe request.",
}
CATEGORY_QUESTION = Question(
    type="choice", instructions="Classify this support email into exactly one category.",
    criteria=CATEGORY_CRITERIA,
)

URGENCY_QUESTION = Question(
    type="score", instructions="Rate how urgently this ticket needs a response.",
    criteria=list(URGENCY_LEVELS),
)

IS_BUSINESS_QUESTION = Question(
    type="noul", instructions="Was this email sent in a professional/business capacity rather than a personal one?",
    criteria={"true": "Sent in a professional/business capacity.", "false": "Sent in a personal capacity."},
)

# HIDDEN_PRIORITY: three views of one underlying (textually undetectable) latent.
HP_NOUL_QUESTION = Question(
    type="noul", instructions="Should this ticket be expedited, i.e. treated as high or critical priority?",
    criteria={"true": "Should be treated as high or critical priority.", "false": "Standard priority is fine."},
)
HP_CHOICE_QUESTION = Question(
    type="choice", instructions="What priority level should this ticket be assigned?",
    criteria={l: l.capitalize() + " priority" for l in URGENCY_LEVELS},
)
HP_SCORE_QUESTION = Question(
    type="score", instructions="Rate the priority level this ticket should be assigned.",
    criteria=list(URGENCY_LEVELS),
)


def _extract_probabilities(answer: dict, level_order: Sequence[str]) -> np.ndarray:
    """Handles both plausible score-type conventions (see module docstring)."""
    probs = answer.get("probabilities")
    if probs and all(l in probs for l in level_order):
        return np.array([probs[l] for l in level_order], dtype=np.float64)
    legend = answer.get("legend")
    if probs and legend:
        name_to_idx = {v: k for k, v in legend.items()}
        if all(l in name_to_idx and name_to_idx[l] in probs for l in level_order):
            return np.array([probs[name_to_idx[l]] for l in level_order], dtype=np.float64)
    raise ValueError(f"could not parse probabilities for levels {level_order} from answer {answer}")


# ---------------------------------------------------------------------------
# Pre-registered hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hypothesis:
    suite_id: str
    prediction: str
    falsification: str


HYPOTHESES = {
    "S1": Hypothesis("S1", "On resolvable items (true posterior > 0.97 one-hot), Jev's accuracy and mean "
                           "confidence both exceed 0.9.",
                           "Accuracy or mean confidence on this set falls below 0.85."),
    "S2": Hypothesis("S2", "On items with a known true posterior near 0.7, Jev's reported probability for the "
                           "same label sits within 0.15 of 0.7 on average (mean posterior KL < 0.15 nats).",
                           "Mean posterior KL exceeds 0.3 nats, or the reported probability is systematically "
                           "on one side of the true value across the sweep (a temperature-fixable bias)."),
    "S3": Hypothesis("S3", "On HIDDEN_PRIORITY items (true posterior == prior, no evidence in the text by "
                           "construction), Jev's accuracy sits at or below the Bayes rate for guessing the "
                           "prior, but its mean reported confidence in the chosen answer exceeds that Bayes-"
                           "optimal accuracy -- i.e. it is confident on a question it cannot know the answer to, "
                           "replicating the published 44.7%-accuracy/0.74-confidence finding on an independently-"
                           "constructed unanswerable item.",
                           "Mean confidence sits within 0.05 of the Bayes-optimal accuracy (i.e. Jev correctly "
                           "reports low confidence proportional to its irreducible uncertainty)."),
    "S4": Hypothesis("S4", "Asking about the SAME underlying HIDDEN_PRIORITY draw as noul/choice/score gives "
                           "inconsistent implied probabilities across the three framings (e.g. the noul "
                           "P(expedite) disagrees with choice_P(high)+choice_P(critical) by more than 0.1), "
                           "replicating the published finding that miscalibration sign differs by question type.",
                           "The three views agree with each other (and with the true prior) to within 0.05."),
    "S5": Hypothesis("S5", "Jev's `confidence` field provides no material discrimination advantage over "
                           "max(probabilities) alone (AURC using confidence is not meaningfully lower than "
                           "AURC using max-probability, and the two are highly correlated) -- consistent with "
                           "TypeSafe's docs not specifying what confidence estimates beyond the distribution.",
                           "AURC(confidence) is meaningfully lower than AURC(max-probability) (a real, "
                           "distinct, and useful discrimination signal), or the two are weakly correlated "
                           "(confidence is answering a genuinely different question)."),
    "S6": Hypothesis("S6", "Restating an explicit base rate in the question's `instructions` (holding the "
                           "email text, i.e. the surface evidence, fixed at one ambiguous combo) shifts Jev's "
                           "reported probability by LESS than the Bayes-correct amount -- i.e. it under-weights "
                           "an explicitly stated prior relative to the text.",
                           "The shift in Jev's reported probability across the two stated base rates matches "
                           "the exact Bayes-correct shift (computed from the generator's own posterior formula) "
                           "within 0.1."),
}


# ---------------------------------------------------------------------------
# Suite result container
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    suite_id: str
    hypothesis: Hypothesis
    model_versions: set = field(default_factory=set)
    n_items: int = 0
    metrics: dict = field(default_factory=dict)
    records: list = field(default_factory=list)  # per-item audit trail


# ---------------------------------------------------------------------------
# S1: resolvable
# ---------------------------------------------------------------------------


def run_s1_resolvable(client: BaseJevClient, n: int, seed: int) -> SuiteResult:
    items = build_resolvable("category", n=n, seed=seed, min_confidence=0.97)
    confidences, correct, model_versions, records = [], [], set(), []
    for item in items:
        resp = client.ask(item.text, {"category": CATEGORY_QUESTION})
        ans = resp.answers["category"]
        model_versions.add(resp.model)
        chosen = ans["choice"]
        conf = ans.get("confidence", max(ans["probabilities"].values()))
        confidences.append(conf)
        correct.append(1.0 if chosen == item.labels["category"] else 0.0)
        records.append({"text": item.text, "true_label": item.labels["category"], "answer": ans})

    confidences, correct = np.array(confidences), np.array(correct)
    metrics = {
        "accuracy": float(correct.mean()),
        "accuracy_ci": m.wilson_ci(int(correct.sum()), len(correct)),
        "mean_confidence": float(confidences.mean()),
        "ece": m.expected_calibration_error(confidences, correct),
    }
    return SuiteResult("S1", HYPOTHESES["S1"], model_versions, len(items), metrics, records)


# ---------------------------------------------------------------------------
# S2: known-ambiguous sweep
# ---------------------------------------------------------------------------


def run_s2_known_ambiguous(client: BaseJevClient, levels: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
                            n_per_level: int = 25, seed: int = 0) -> SuiteResult:
    sweep = build_posterior_sweep("category", positive_labels=["billing"], levels=levels,
                                   n_per_level=n_per_level, seed=seed)
    p_true, p_model, model_versions, records = [], [], set(), []
    for s in sweep:
        resp = client.ask(s.item.text, {"category": CATEGORY_QUESTION})
        ans = resp.answers["category"]
        model_versions.add(resp.model)
        probs = np.array([ans["probabilities"][l] for l in CATEGORY_LABELS])
        true_vec = np.array([s.item.posteriors["category"][l] for l in CATEGORY_LABELS])
        p_true.append(true_vec)
        p_model.append(probs)
        records.append({"text": s.item.text, "target_p": s.target_p, "achieved_p": s.achieved_p, "answer": ans})

    p_true, p_model = np.array(p_true), np.array(p_model)
    div = m.mean_posterior_divergence(p_true, p_model)
    p_billing_true = p_true[:, CATEGORY_LABELS.index("billing")]
    p_billing_model = p_model[:, CATEGORY_LABELS.index("billing")]
    metrics = {**div, "mean_abs_error_billing": float(np.mean(np.abs(p_billing_true - p_billing_model))),
               "mean_signed_error_billing": float(np.mean(p_billing_model - p_billing_true))}
    return SuiteResult("S2", HYPOTHESES["S2"], model_versions, len(sweep), metrics, records)


# ---------------------------------------------------------------------------
# S3: unanswerable
# ---------------------------------------------------------------------------


def run_s3_unanswerable(client: BaseJevClient, n: int, seed: int) -> SuiteResult:
    items = build_unanswerable(n=n, seed=seed)
    confidences, correct, model_versions, records = [], [], set(), []
    for item in items:
        resp = client.ask(item.text, {"priority": HP_CHOICE_QUESTION})
        ans = resp.answers["priority"]
        model_versions.add(resp.model)
        chosen = ans["choice"]
        conf = ans.get("confidence", max(ans["probabilities"].values()))
        confidences.append(conf)
        correct.append(1.0 if chosen == item.labels["hidden_priority"] else 0.0)
        records.append({"text": item.text, "true_label": item.labels["hidden_priority"], "answer": ans})

    confidences, correct = np.array(confidences), np.array(correct)
    bayes_acc = 1.0 - bayes_error_rate(HIDDEN_PRIORITY)  # == max(prior) by construction
    metrics = {
        "accuracy": float(correct.mean()),
        "accuracy_ci": m.wilson_ci(int(correct.sum()), len(correct)),
        "mean_confidence": float(confidences.mean()),
        "bayes_optimal_accuracy": bayes_acc,
        "overconfidence_gap": float(confidences.mean() - bayes_acc),
        "ece": m.expected_calibration_error(confidences, correct),
    }
    return SuiteResult("S3", HYPOTHESES["S3"], model_versions, len(items), metrics, records)


# ---------------------------------------------------------------------------
# S4: type consistency
# ---------------------------------------------------------------------------


HP_EXPEDITE_LABELS = ["high", "critical"]


def run_s4_type_consistency(client: BaseJevClient, n: int, seed: int) -> SuiteResult:
    items = build_unanswerable(n=n, seed=seed)  # any item works: HIDDEN_PRIORITY carries zero evidence everywhere
    noul_p, choice_expedite_p, score_expedite_p = [], [], []
    model_versions, records = set(), []
    for item in items:
        resp = client.ask(item.text, {"noul": HP_NOUL_QUESTION, "choice": HP_CHOICE_QUESTION, "score": HP_SCORE_QUESTION})
        model_versions.add(resp.model)
        noul_ans = resp.answers["noul"]
        choice_ans = resp.answers["choice"]
        score_ans = resp.answers["score"]

        p_noul = noul_ans["noul"]
        p_choice = sum(choice_ans["probabilities"].get(l, 0.0) for l in HP_EXPEDITE_LABELS)
        score_probs = _extract_probabilities(score_ans, URGENCY_LEVELS)
        p_score = float(sum(score_probs[URGENCY_LEVELS.index(l)] for l in HP_EXPEDITE_LABELS))

        noul_p.append(p_noul)
        choice_expedite_p.append(p_choice)
        score_expedite_p.append(p_score)
        records.append({"text": item.text, "noul": noul_ans, "choice": choice_ans, "score": score_ans})

    noul_p, choice_expedite_p, score_expedite_p = map(np.array, (noul_p, choice_expedite_p, score_expedite_p))
    true_p = binary_posterior(exact_posterior(HIDDEN_PRIORITY, {}), HP_EXPEDITE_LABELS)

    metrics = {
        "true_p_expedite": true_p,
        "mean_noul_p": float(noul_p.mean()), "mean_choice_p": float(choice_expedite_p.mean()),
        "mean_score_p": float(score_expedite_p.mean()),
        "mean_abs_noul_vs_choice": float(np.mean(np.abs(noul_p - choice_expedite_p))),
        "mean_abs_noul_vs_score": float(np.mean(np.abs(noul_p - score_expedite_p))),
        "mean_abs_choice_vs_score": float(np.mean(np.abs(choice_expedite_p - score_expedite_p))),
    }
    return SuiteResult("S4", HYPOTHESES["S4"], model_versions, len(items), metrics, records)


# ---------------------------------------------------------------------------
# S5: confidence vs probabilities
# ---------------------------------------------------------------------------


def run_s5_confidence_vs_probabilities(client: BaseJevClient, n: int, seed: int) -> SuiteResult:
    # Mix resolvable + ambiguous so there's real variation in correctness to rank.
    items = build_resolvable("category", n=n // 2, seed=seed, min_confidence=0.9) + \
            [s.item for s in build_ambiguous("category", ["billing"], target=0.6, n=n // 2, seed=seed + 1, tol=0.25)]
    confidences, max_probs, correct, model_versions, records = [], [], [], set(), []
    for item in items:
        resp = client.ask(item.text, {"category": CATEGORY_QUESTION})
        ans = resp.answers["category"]
        model_versions.add(resp.model)
        chosen = ans["choice"]
        max_p = max(ans["probabilities"].values())
        conf = ans.get("confidence", max_p)
        confidences.append(conf)
        max_probs.append(max_p)
        correct.append(1.0 if chosen == item.labels["category"] else 0.0)
        records.append({"text": item.text, "true_label": item.labels["category"], "answer": ans})

    confidences, max_probs, correct = map(np.array, (confidences, max_probs, correct))
    cov_c, risk_c = m.risk_coverage_curve(confidences, correct)
    cov_p, risk_p = m.risk_coverage_curve(max_probs, correct)
    corr = float(np.corrcoef(confidences, max_probs)[0, 1]) if np.std(confidences) > 0 and np.std(max_probs) > 0 else float("nan")
    metrics = {
        "aurc_confidence": m.area_under_risk_coverage(cov_c, risk_c),
        "aurc_max_probability": m.area_under_risk_coverage(cov_p, risk_p),
        "correlation_confidence_vs_max_probability": corr,
    }
    return SuiteResult("S5", HYPOTHESES["S5"], model_versions, len(items), metrics, records)


# ---------------------------------------------------------------------------
# S6: prior sensitivity via explicit base-rate restatement
# ---------------------------------------------------------------------------


def _bayes_shift_target(realizations: Dict[str, str], positive_labels: Sequence[str], alt_prior: Dict[str, float]) -> float:
    """Exact Bayes-correct posterior under a DIFFERENT prior, same evidence
    (same realizations, hence the same likelihood ratios) -- reuses
    generator's own naive-Bayes formula with alt_prior swapped in."""
    unnorm = {}
    for y in CATEGORY.labels:
        p = alt_prior[y]
        for slot in CATEGORY.slots:
            p *= slot.emissions[y][realizations[slot.name]]
        unnorm[y] = p
    z = sum(unnorm.values())
    post = {y: p / z for y, p in unnorm.items()}
    return binary_posterior(post, positive_labels)


def run_s6_prior_sensitivity(client: BaseJevClient, n: int, seed: int) -> SuiteResult:
    ambiguous = build_ambiguous("category", ["spam"], target=0.5, n=n, seed=seed, tol=0.3)
    low_prior = {"billing": 0.30, "technical": 0.30, "sales": 0.30, "spam": 0.10}
    high_prior = {"billing": 0.15, "technical": 0.15, "sales": 0.10, "spam": 0.60}
    q_low = Question(type="choice",
                      instructions="Context: historically, only about 10% of tickets in this queue turn out "
                                   "to be spam. Classify this support email into exactly one category.",
                      criteria=CATEGORY_CRITERIA)
    q_high = Question(type="choice",
                       instructions="Context: historically, about 60% of tickets in this queue turn out to "
                                    "be spam. Classify this support email into exactly one category.",
                       criteria=CATEGORY_CRITERIA)

    observed_shift, bayes_shift, model_versions, records = [], [], set(), []
    for s in ambiguous:
        resp_low = client.ask(s.item.text, {"category": q_low})
        resp_high = client.ask(s.item.text, {"category": q_high})
        model_versions.update([resp_low.model, resp_high.model])
        p_low = resp_low.answers["category"]["probabilities"]["spam"]
        p_high = resp_high.answers["category"]["probabilities"]["spam"]
        observed_shift.append(p_high - p_low)

        target_low = _bayes_shift_target(s.combo, ["spam"], low_prior)
        target_high = _bayes_shift_target(s.combo, ["spam"], high_prior)
        bayes_shift.append(target_high - target_low)
        records.append({"text": s.item.text, "p_low": p_low, "p_high": p_high,
                         "bayes_target_low": target_low, "bayes_target_high": target_high})

    observed_shift, bayes_shift = np.array(observed_shift), np.array(bayes_shift)
    metrics = {
        "mean_observed_shift": float(observed_shift.mean()),
        "mean_bayes_correct_shift": float(bayes_shift.mean()),
        "mean_shift_deficit": float((bayes_shift - observed_shift).mean()),
        "fraction_shifting_correct_direction": float(np.mean(np.sign(observed_shift) == np.sign(bayes_shift))),
    }
    return SuiteResult("S6", HYPOTHESES["S6"], model_versions, len(ambiguous), metrics, records)

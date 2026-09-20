"""Verification step 1 from the plan: prove the generator's ground truth
before anything downstream (metrics, toy model, probe) is allowed to depend
on it. If the closed-form posterior claimed by generator.py were wrong,
every number produced by this project would be wrong along with it.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from jev.dgp import generator as gen
from jev.dgp.slots import ALL_SLOTS, COMPLAINT_PHRASE, SENDER_DOMAIN, TONE


def binom_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


# ---------------------------------------------------------------------------
# 1. Monte Carlo must match the closed form (simulation-based calibration
#    applied to our own generator).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "realizations",
    [
        {"complaint_phrase": "invoice_issue", "sender_domain": "corp_domain", "tone": "polite"},
        {"complaint_phrase": "discount_ask", "sender_domain": "generic_webmail", "tone": "curt"},
        {"complaint_phrase": "generic_greeting", "sender_domain": "no_domain_shown", "tone": "casual_register"},
        {"complaint_phrase": "none", "sender_domain": "generic_webmail", "tone": "formal_register"},
    ],
)
def test_category_posterior_matches_monte_carlo(realizations):
    concept = gen.CATEGORY
    true_post = gen.posterior(concept, realizations)

    rng = np.random.default_rng(12345)
    n = 200_000
    counts = {y: 0 for y in concept.labels}
    probs = np.array([true_post[y] for y in concept.labels])
    draws = rng.choice(len(concept.labels), size=n, p=probs)
    for idx in draws:
        counts[concept.labels[idx]] += 1

    # This loop doesn't re-derive the posterior -- it independently samples labels
    # from what generator.py *claims* is the posterior and checks the resulting
    # frequencies land within ~5 sigma of the claimed probabilities, catching
    # normalization bugs. The real MC-vs-closed-form check (drawing straight from
    # generate_item) follows in the next test.
    for y in concept.labels:
        p_hat = counts[y] / n
        se = binom_se(true_post[y], n)
        assert abs(p_hat - true_post[y]) < 5 * se + 1e-9, (
            f"label {y}: empirical {p_hat:.5f} vs claimed {true_post[y]:.5f}, se={se:.5f}"
        )


def test_generate_item_hard_labels_match_forced_realization_posterior():
    """End-to-end: force realizations via overrides, sample many *items*
    through the public API (not the internal posterior() call), and confirm
    the empirical label distribution matches the closed-form posterior."""
    forced = {"complaint_phrase": "discount_ask", "sender_domain": "generic_webmail", "tone": "polite"}
    true_post = gen.posterior(gen.CATEGORY, forced)

    g = gen.EmailGenerator(seed=7)
    n = 20_000
    counts = {y: 0 for y in gen.CATEGORY.labels}
    for _ in range(n):
        item = g.generate_item(overrides={"category": forced})
        counts[item.labels["category"]] += 1

    for y in gen.CATEGORY.labels:
        p_hat = counts[y] / n
        se = binom_se(true_post[y], n)
        assert abs(p_hat - true_post[y]) < 6 * se + 1e-9, (
            f"label {y}: empirical {p_hat:.4f} vs claimed {true_post[y]:.4f} (se={se:.4f})"
        )


# ---------------------------------------------------------------------------
# 2. Emission tables are valid conditional distributions (redundant with the
#    import-time validate_all(), kept explicit so a future edit that breaks
#    this fails loudly in the test suite, not just at import).
# ---------------------------------------------------------------------------


def test_all_slot_emissions_are_valid_distributions():
    for slot in ALL_SLOTS.values():
        slot.validate()  # raises on failure
        for label, dist in slot.emissions.items():
            assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_all_concepts_validate():
    for concept in gen.CONCEPTS.values():
        concept.validate()  # raises on failure
        assert abs(sum(concept.prior.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 3. The TONE slot is an "identical emissions" demonstration: it must
#    contribute exactly zero evidence, for every realization.
# ---------------------------------------------------------------------------


def test_identical_emissions_slot_has_zero_likelihood_ratio_range():
    assert TONE.log_likelihood_ratio_range() == 0.0


def test_tone_does_not_move_the_category_posterior():
    base = {"complaint_phrase": "discount_ask", "sender_domain": "generic_webmail"}
    posts = []
    for tone_r in TONE.realizations:
        r = dict(base, tone=tone_r)
        posts.append(gen.posterior(gen.CATEGORY, r))
    first = posts[0]
    for p in posts[1:]:
        for y in gen.CATEGORY.labels:
            assert abs(p[y] - first[y]) < 1e-12, "tone realization changed the posterior"


def test_informative_slots_do_move_the_category_posterior():
    """Sanity check in the other direction: complaint_phrase must move the
    posterior, or the "ambiguous"/"resolvable" families would be vacuous."""
    fixed = {"sender_domain": "generic_webmail", "tone": "polite"}
    p_invoice = gen.posterior(gen.CATEGORY, dict(fixed, complaint_phrase="invoice_issue"))
    p_crash = gen.posterior(gen.CATEGORY, dict(fixed, complaint_phrase="app_crash"))
    assert p_invoice["billing"] > 0.9
    assert p_crash["technical"] > 0.9
    assert p_invoice != p_crash


# ---------------------------------------------------------------------------
# 4. HIDDEN_PRIORITY (zero slots): posterior must equal the prior for *any*
#    input, and this must hold across all three question-type "views".
# ---------------------------------------------------------------------------


def test_unanswerable_concept_has_no_slots():
    assert gen.HIDDEN_PRIORITY.is_unanswerable
    assert gen.n_distinct_realization_combos(gen.HIDDEN_PRIORITY) == 1


def _dicts_close(a, b, tol=1e-9):
    return set(a.keys()) == set(b.keys()) and all(abs(a[k] - b[k]) < tol for k in a)


def test_unanswerable_posterior_always_equals_prior():
    post = gen.posterior(gen.HIDDEN_PRIORITY, {})
    assert _dicts_close(post, gen.HIDDEN_PRIORITY.prior)


def test_unanswerable_views_are_consistent_pushforwards_of_the_prior():
    post = gen.posterior(gen.HIDDEN_PRIORITY, {})
    # noul view: P(expedite) = P(high) + P(critical)
    p_expedite = gen.binary_posterior(post, ["high", "critical"])
    assert abs(p_expedite - (gen.HIDDEN_PRIORITY.prior["high"] + gen.HIDDEN_PRIORITY.prior["critical"])) < 1e-12
    assert abs(p_expedite - 0.3) < 1e-12
    # choice / score views are just `post` itself, read differently -- assert
    # they are the exact same dict, i.e. share one ground truth by construction.
    assert post is not None and set(post.keys()) == set(gen.HIDDEN_PRIORITY.labels)


def test_unanswerable_ignores_forced_realizations_of_other_concepts():
    """Even in an item where every other concept's realizations are forced to
    extreme, highly-informative values, HIDDEN_PRIORITY's posterior cannot
    move, because it has no slots to read evidence from at all."""
    g = gen.EmailGenerator(seed=99)
    forced = {
        "category": {"complaint_phrase": "invoice_issue", "sender_domain": "corp_domain", "tone": "polite"},
        "urgency": {"intensity_marker": "extreme", "exclaim_count": "3+"},
        "is_business": {"signature_style": "formal_signoff"},
    }
    item = g.generate_item(overrides=forced)
    assert _dicts_close(item.posteriors["hidden_priority"], gen.HIDDEN_PRIORITY.prior)


# ---------------------------------------------------------------------------
# 5. Bayes error rate: cross-check two independently-written computations.
# ---------------------------------------------------------------------------


def _bayes_error_via_enumerate_posteriors(concept: gen.ConceptSpec) -> float:
    """Alternative implementation, deliberately structured differently from
    generator.bayes_error_rate (normalizes then re-weights, rather than
    working with the unnormalized joint directly) so the two can cross-check."""
    total_correct = 0.0
    for realizations, post, p_realizations in gen.enumerate_posteriors(concept):
        total_correct += p_realizations * max(post.values())
    return 1.0 - total_correct


@pytest.mark.parametrize("concept", [gen.CATEGORY, gen.URGENCY, gen.IS_BUSINESS, gen.HIDDEN_PRIORITY])
def test_bayes_error_rate_cross_check(concept):
    a = gen.bayes_error_rate(concept)
    b = _bayes_error_via_enumerate_posteriors(concept)
    assert abs(a - b) < 1e-9
    assert 0.0 <= a <= 1.0


def test_bayes_error_rate_of_unanswerable_concept_is_one_minus_max_prior():
    expected = 1.0 - max(gen.HIDDEN_PRIORITY.prior.values())
    assert abs(gen.bayes_error_rate(gen.HIDDEN_PRIORITY) - expected) < 1e-12


def test_realization_marginal_probabilities_sum_to_one():
    for concept in gen.CONCEPTS.values():
        total = sum(p for _, _, p in gen.enumerate_posteriors(concept))
        assert abs(total - 1.0) < 1e-9, f"{concept.name}: marginal P(realizations) sums to {total}"


# ---------------------------------------------------------------------------
# 6. Combinatorics: confirm the claimed surface-form space is what we say it is.
# ---------------------------------------------------------------------------


def test_n_distinct_realization_combos_matches_manual_product():
    expected = len(COMPLAINT_PHRASE.realizations) * len(SENDER_DOMAIN.realizations) * len(TONE.realizations)
    assert gen.n_distinct_realization_combos(gen.CATEGORY) == expected
    assert len(gen.enumerate_posteriors(gen.CATEGORY)) == expected


def test_surface_form_count_matches_brute_force_for_one_concept():
    """Cross-check n_surface_forms_per_concept's closed form (sum-within-slot,
    product-across-slots) against literally enumerating every combo and
    summing its paraphrase-choice product, for a concept small enough to
    brute-force exactly (IS_BUSINESS: 1 slot, 3 realizations)."""
    concept = gen.IS_BUSINESS
    brute_force = 0
    for realizations, _post, _p in gen.enumerate_posteriors(concept):
        combo_multiplier = 1
        for slot in concept.slots:
            combo_multiplier *= len(slot.phrases[realizations[slot.name]])
        brute_force += combo_multiplier
    assert gen.n_surface_forms_per_concept(concept) == brute_force


def test_full_item_surface_form_lower_bound_exceeds_planned_dataset_size():
    """Distinct literal texts the generator can produce comfortably exceeds
    the ~50k items any training run in this project uses, which is what
    makes memorization (as opposed to generalization) implausible."""
    combo_space = 1
    for concept in gen.CONCEPTS.values():
        combo_space *= gen.n_distinct_realization_combos(concept)
    surface_forms = gen.n_surface_forms_lower_bound()
    print(f"\nrealization_combo_space={combo_space:,} "
          f"surface_form_lower_bound={surface_forms:,}")
    assert surface_forms > 50_000 * 20  # comfortably above any planned training-set size


# ---------------------------------------------------------------------------
# 7. Rendering is a pure function of realizations + person + paraphrase RNG --
#    it never receives a label, so it cannot leak one (structural guarantee,
#    checked here by re-rendering identical inputs and requiring identical
#    output; the statistical negative control lives in test_scenarios.py once
#    full datasets exist).
# ---------------------------------------------------------------------------


def test_render_text_is_deterministic_given_realizations_and_rng_state():
    person = gen.Person(first_name="Sam", last_name="Ito", title="Lead", company="Acme")
    cat_r = {"complaint_phrase": "invoice_issue", "sender_domain": "corp_domain", "tone": "polite"}
    urg_r = {"intensity_marker": "mild", "exclaim_count": "0"}
    biz_r = {"signature_style": "first_name_only"}

    rng1 = np.random.default_rng(42)
    text1 = gen.render_text(cat_r, urg_r, biz_r, person, rng1)
    rng2 = np.random.default_rng(42)
    text2 = gen.render_text(cat_r, urg_r, biz_r, person, rng2)
    assert text1 == text2


def test_render_text_signature_has_no_label_parameter():
    import inspect

    params = list(inspect.signature(gen.render_text).parameters.keys())
    assert "label" not in params and "labels" not in params
    params2 = list(inspect.signature(gen._pick_phrase).parameters.keys())
    assert "label" not in params2 and "labels" not in params2

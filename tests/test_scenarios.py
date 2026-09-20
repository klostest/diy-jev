"""Tests for the family builders, compositional split, and negative controls.
These are the "best practice" checks the user asked for explicitly: they
exist to catch a broken generator or a leaking pipeline before any
calibration number is trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from jev.dgp import generator as gen
from jev.dgp import scenarios as sc


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------


def test_build_resolvable_is_near_one_hot():
    items = sc.build_resolvable("category", n=200, seed=1, min_confidence=0.97)
    for it in items:
        post = it.posteriors["category"]
        assert max(post.values()) >= 0.97


def test_build_unanswerable_posterior_is_always_the_prior():
    items = sc.build_unanswerable(n=100, seed=2)
    for it in items:
        post = it.posteriors["hidden_priority"]
        for y in gen.HIDDEN_PRIORITY.labels:
            assert abs(post[y] - gen.HIDDEN_PRIORITY.prior[y]) < 1e-9


def test_build_ambiguous_hits_target_within_tolerance():
    # billing vs {technical, sales, spam}: known to have overlap via generic_greeting
    sweep = sc.build_ambiguous("category", positive_labels=["billing"], target=0.4, n=10, seed=3, tol=0.15)
    assert len(sweep) == 10
    for s in sweep:
        assert abs(s.achieved_p - s.target_p) <= 0.15
        # the item's OWN posterior for the forced combo must match achieved_p exactly
        post = s.item.posteriors["category"]
        assert abs(gen.binary_posterior(post, ["billing"]) - s.achieved_p) < 1e-9


def test_build_ambiguous_raises_when_target_unreachable():
    with pytest.raises(ValueError):
        sc.build_ambiguous("category", positive_labels=["billing"], target=0.4, n=1, seed=0, tol=1e-6)


def test_posterior_sweep_covers_a_spread_of_achieved_values():
    sweep = sc.build_posterior_sweep(
        "category", positive_labels=["sales", "spam"], levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_per_level=5, seed=4,
    )
    achieved = sorted({round(s.achieved_p, 6) for s in sweep})
    assert len(achieved) >= 3, f"sweep collapsed onto too few distinct combos: {achieved}"


# ---------------------------------------------------------------------------
# Compositional split
# ---------------------------------------------------------------------------


def test_compositional_split_has_disjoint_combos():
    g = gen.EmailGenerator(seed=5)
    items = g.generate_batch(500)
    train, test = sc.compositional_split(items, "category", test_fraction=0.25, seed=5)
    train_combos = {sc.combo_key(it, "category") for it in train}
    test_combos = {sc.combo_key(it, "category") for it in test}
    assert train_combos.isdisjoint(test_combos)
    assert len(train) + len(test) == len(items)
    assert len(test) > 0 and len(train) > 0


def test_compositional_split_is_deterministic_given_seed():
    g = gen.EmailGenerator(seed=6)
    items = g.generate_batch(200)
    train1, test1 = sc.compositional_split(items, "urgency", seed=42)
    train2, test2 = sc.compositional_split(items, "urgency", seed=42)
    assert [it.text for it in train1] == [it.text for it in train2]
    assert [it.text for it in test1] == [it.text for it in test2]


def test_compositional_split_falls_back_to_random_split_for_unanswerable_concept():
    g = gen.EmailGenerator(seed=7)
    items = g.generate_batch(200)
    train, test = sc.compositional_split(items, "hidden_priority", test_fraction=0.3, seed=1)
    assert len(train) + len(test) == 200
    assert len(test) == int(200 * 0.3)


# ---------------------------------------------------------------------------
# Negative control 1: label permutation must be an honest shuffle
# ---------------------------------------------------------------------------


def test_permute_hard_labels_preserves_text_and_marginal_distribution():
    g = gen.EmailGenerator(seed=8)
    items = g.generate_batch(300)
    permuted = sc.permute_hard_labels(items, "category", seed=1)

    assert [it.text for it in items] == [it.text for it in permuted]
    assert [it.posteriors["category"] for it in items] == [it.posteriors["category"] for it in permuted]

    orig_labels = sorted(it.labels["category"] for it in items)
    perm_labels = sorted(it.labels["category"] for it in permuted)
    assert orig_labels == perm_labels  # same multiset, different assignment

    same_position = sum(
        1 for a, b in zip(items, permuted) if a.labels["category"] == b.labels["category"]
    )
    assert same_position < len(items)  # a real shuffle, not a no-op


# ---------------------------------------------------------------------------
# Negative control 2: paraphrase must not leak the label
# ---------------------------------------------------------------------------


def test_paraphrase_does_not_leak_label_at_a_fixed_ambiguous_combo():
    """The critical negative control from the plan: at a FIXED realization
    combo (so every item's true evidence is identical), a classifier reading
    only the rendered text must not beat the majority-class baseline by more
    than sampling noise -- any edge would mean paraphrase choice is leaking
    the label the generator is supposed to keep hidden behind it.
    """
    sweep = sc.build_ambiguous("category", positive_labels=["sales", "spam"], target=0.55, n=400, seed=9, tol=0.2)
    items = [s.item for s in sweep]

    result = sc.paraphrase_leakage_score(items, "category")
    slack = 3.0 / (result["n"] ** 0.5)  # ~3 CV-fold binomial-ish slack at this n
    assert result["cv_accuracy"] <= result["majority_baseline"] + slack, (
        f"paraphrase-only classifier reached {result['cv_accuracy']:.3f} accuracy vs "
        f"majority baseline {result['majority_baseline']:.3f} -- possible label leakage "
        f"via paraphrase selection"
    )


def test_paraphrase_leakage_score_requires_single_combo():
    g = gen.EmailGenerator(seed=10)
    items = g.generate_batch(20)  # varied combos
    with pytest.raises(ValueError):
        sc.paraphrase_leakage_score(items, "category")


# ---------------------------------------------------------------------------
# Negative control 3: no predictor -- including the true posterior itself --
# exceeds the exact Bayes error rate.
# ---------------------------------------------------------------------------


def test_bayes_optimal_predictor_achieves_the_bayes_rate_empirically():
    """The Bayes-optimal rule (predict argmax of the TRUE posterior) sampled
    against items drawn from the generator's own process must empirically
    match bayes_error_rate(concept) -- the ceiling no model may cross."""
    concept = gen.CATEGORY
    theoretical = gen.bayes_error_rate(concept)

    g = gen.EmailGenerator(seed=11)
    n = 20_000
    errors = 0
    for _ in range(n):
        item = g.generate_item()
        post = item.posteriors["category"]
        pred = max(post, key=post.get)
        if pred != item.labels["category"]:
            errors += 1
    empirical = errors / n
    se = (theoretical * (1 - theoretical) / n) ** 0.5
    assert abs(empirical - theoretical) < 5 * se + 0.01, (
        f"empirical Bayes-rule error {empirical:.4f} vs theoretical {theoretical:.4f} (se={se:.4f})"
    )

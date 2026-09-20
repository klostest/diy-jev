"""Dataset builders on top of generator.py: the three item families, the
target-posterior sweep, compositional train/test splits, and the negative
controls that catch a broken generator before it corrupts a downstream
number.

Everything here composes generator.py's exact posterior arithmetic; nothing
here estimates anything.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from jev.dgp.generator import (
    CONCEPTS,
    ConceptSpec,
    EmailGenerator,
    Item,
    binary_posterior,
    enumerate_posteriors,
)


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------


def generate_fixed_combo_batch(
    concept_name: str, realizations: Dict[str, str], n: int, seed: int
) -> List[Item]:
    """n items sharing one forced realization-combo for `concept_name` (other
    concepts vary freely). The building block every family below reduces to."""
    g = EmailGenerator(seed=seed)
    return [g.generate_item(overrides={concept_name: realizations}) for _ in range(n)]


def build_resolvable(concept_name: str, n: int, seed: int, min_confidence: float = 0.97) -> List[Item]:
    """Family 1: near one-hot. Filters the concept's enumerated combos to
    those whose max posterior probability clears `min_confidence`, then
    samples items from that filtered set (weighted uniformly over combos, not
    by their natural marginal probability, so rare-but-resolvable combos are
    still represented)."""
    concept = CONCEPTS[concept_name]
    candidates = [
        (r, post) for r, post, _p_marginal in enumerate_posteriors(concept)
        if max(post.values()) >= min_confidence
    ]
    if not candidates:
        raise ValueError(f"no combo for '{concept_name}' reaches min_confidence={min_confidence}")
    rng = random.Random(seed)
    g = EmailGenerator(seed=seed)
    items = []
    for _ in range(n):
        r, _post = candidates[rng.randrange(len(candidates))]
        items.append(g.generate_item(overrides={concept_name: r}))
    return items


def build_unanswerable(n: int, seed: int) -> List[Item]:
    """Family 3: HIDDEN_PRIORITY has zero slots, so every ordinarily-generated
    item already qualifies -- no override needed, no filtering possible."""
    g = EmailGenerator(seed=seed)
    return g.generate_batch(n)


def enumerate_binary_combos(
    concept: ConceptSpec, positive_labels: Sequence[str]
) -> List[Tuple[Dict[str, str], float, float]]:
    """[(realizations, P(positive|realizations), P(realizations))], exact."""
    out = []
    for realizations, post, p_marginal in enumerate_posteriors(concept):
        out.append((realizations, binary_posterior(post, positive_labels), p_marginal))
    return out


def _nearest_combo_index(
    combos: List[Tuple[Dict[str, str], float, float]], target: float
) -> int:
    return min(range(len(combos)), key=lambda i: abs(combos[i][1] - target))


@dataclass(frozen=True)
class SweepItem:
    item: Item
    concept_name: str
    target_p: float
    achieved_p: float  # exact true posterior of the combo actually used
    combo: Dict[str, str]


def build_posterior_sweep(
    concept_name: str,
    positive_labels: Sequence[str],
    levels: Sequence[float],
    n_per_level: int,
    seed: int,
) -> List[SweepItem]:
    """Family 2 (ambiguous), generalized: for each target level in [0, 1], find
    the enumerated combo whose exact binary posterior is closest to it, then
    generate n_per_level items forcing that combo. `achieved_p` is the exact
    ground truth for whatever combo was actually used -- always report that,
    not `target_p`, as the number a calibration check is graded against.
    """
    concept = CONCEPTS[concept_name]
    combos = enumerate_binary_combos(concept, positive_labels)
    g = EmailGenerator(seed=seed)
    out = []
    for level in levels:
        idx = _nearest_combo_index(combos, level)
        realizations, achieved_p, _p_marginal = combos[idx]
        for _ in range(n_per_level):
            item = g.generate_item(overrides={concept_name: realizations})
            out.append(SweepItem(item=item, concept_name=concept_name, target_p=level,
                                  achieved_p=achieved_p, combo=realizations))
    return out


def build_ambiguous(
    concept_name: str, positive_labels: Sequence[str], target: float, n: int, seed: int,
    tol: float = 0.05,
) -> List[SweepItem]:
    """Convenience single-level case of build_posterior_sweep, with a hard
    check that the achievable grid actually reaches close to `target` --
    raises rather than silently returning a misleadingly-labeled dataset."""
    sweep = build_posterior_sweep(concept_name, positive_labels, [target], n, seed)
    achieved = sweep[0].achieved_p
    if abs(achieved - target) > tol:
        raise ValueError(
            f"nearest achievable posterior for target={target} is {achieved:.3f}, "
            f"outside tol={tol}. Widen the slot vocabulary or relax tol."
        )
    return sweep


# ---------------------------------------------------------------------------
# Compositional train/test split (holds out realization-combos, not rows)
# ---------------------------------------------------------------------------


def combo_key(item: Item, concept_name: str) -> Tuple:
    return tuple(sorted(item.realizations[concept_name].items()))


def compositional_split(
    items: List[Item], concept_name: str, test_fraction: float = 0.2, seed: int = 0
) -> Tuple[List[Item], List[Item]]:
    """Splits by realization-combo, not by item, so test items use slot-value
    combinations never seen in training -- the only way to tell memorization
    apart from generalization (see docs/explainers.md).

    HIDDEN_PRIORITY has exactly one combo (itself, `{}`), so there is no
    combinatorial structure to hold out; falls back to a plain random
    item-level split for that concept only.
    """
    concept = CONCEPTS[concept_name]
    if concept.is_unanswerable:
        rng = random.Random(seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n_test = int(len(items) * test_fraction)
        test_idx = set(idx[:n_test])
        train = [it for i, it in enumerate(items) if i not in test_idx]
        test = [it for i, it in enumerate(items) if i in test_idx]
        return train, test

    combos = sorted({combo_key(it, concept_name) for it in items})
    rng = random.Random(seed)
    rng.shuffle(combos)
    n_test_combos = max(1, int(len(combos) * test_fraction))
    test_combo_set = set(combos[:n_test_combos])

    train, test = [], []
    for it in items:
        (test if combo_key(it, concept_name) in test_combo_set else train).append(it)
    return train, test


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def permute_hard_labels(items: List[Item], concept_name: str, seed: int) -> List[Item]:
    """A shuffled-label control set: text and true posteriors are untouched,
    only the *hard* label used as a training/eval target is permuted across
    items. Any model trained against this must land at chance -- if it
    doesn't, something in the pipeline (not the generator) is leaking.
    Never use this for the real soft-target training; only for the control.
    """
    rng = random.Random(seed)
    labels = [it.labels[concept_name] for it in items]
    permuted = labels[:]
    rng.shuffle(permuted)
    out = []
    for it, new_label in zip(items, permuted):
        new_labels = dict(it.labels)
        new_labels[concept_name] = new_label
        out.append(dataclasses.replace(it, labels=new_labels))
    return out


def paraphrase_leakage_score(items: List[Item], concept_name: str) -> Dict[str, float]:
    """Fits a bag-of-words classifier from raw text -> hard label using items
    that all share ONE fixed realization-combo for `concept_name` (build with
    generate_fixed_combo_batch). Since every item's evidence is identical by
    construction, any predictive power above the majority-class baseline can
    only come from the paraphrase strings chosen -- i.e. exactly the leak
    this control is built to catch. Returns cv accuracy and the baseline to
    compare it against; see test_scenarios.py for the pass/fail threshold.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    combos = {combo_key(it, concept_name) for it in items}
    if len(combos) != 1:
        raise ValueError(
            f"paraphrase_leakage_score requires one shared combo, got {len(combos)} distinct combos"
        )
    texts = [it.text for it in items]
    labels = [it.labels[concept_name] for it in items]
    label_set = sorted(set(labels))
    if len(label_set) < 2:
        return {"cv_accuracy": 1.0, "majority_baseline": 1.0, "n": len(items), "n_classes": len(label_set)}

    label_idx = {y: i for i, y in enumerate(label_set)}
    y = np.array([label_idx[l] for l in labels])
    X = TfidfVectorizer(min_df=2).fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return {"cv_accuracy": float("nan"), "majority_baseline": float(np.bincount(y).max() / len(y)),
                "n": len(items), "n_classes": len(label_set)}
    scores = cross_val_score(clf, X, y, cv=n_splits)
    majority = np.bincount(y).max() / len(y)
    return {
        "cv_accuracy": float(scores.mean()),
        "majority_baseline": float(majority),
        "n": len(items),
        "n_classes": len(label_set),
    }

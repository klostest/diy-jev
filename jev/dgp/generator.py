"""The data-generating process: latent labels -> text, with exact posteriors.

Design in one paragraph: each *concept* (CATEGORY, URGENCY, IS_BUSINESS,
HIDDEN_PRIORITY) has a prior over its label and a list of independent
`SlotSpec`s (slots.py). To generate an item we sample the concept's label
from its prior, then sample each slot's *realization* from
`P(realization | label)`, then render the realizations to natural-language
text via label-independent paraphrase choices. Because generation is
naive-Bayes by construction (not assumed post-hoc), the posterior given the
*realizations that were actually drawn* is exactly

    P(y | r_1..r_k) proportional to prior(y) * prod_i P(r_i | y)

with no estimation step. See docs/explainers.md for the derivation and the
log-odds / likelihood-ratio form.

HIDDEN_PRIORITY has zero slots: it is drawn independently of the text and by
construction its posterior given any text is exactly its prior. This is the
"unanswerable" family, and it is used through three question-type "views"
(noul / choice / score) of the *same* underlying draw, which is what makes
suite S4 (type-consistency) and S3 (unanswerable) share ground truth.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from jev.dgp.slots import (
    ALL_SLOTS,
    BUSINESS_LABELS,
    CATEGORY_LABELS,
    COMPLAINT_PHRASE,
    EXCLAIM_COUNT,
    INTENSITY_MARKER,
    SENDER_DOMAIN,
    SIGNATURE_STYLE,
    TONE,
    URGENCY_LEVELS,
    SlotSpec,
)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    labels: List[str]
    prior: Dict[str, float]
    slots: List[SlotSpec]
    ordinal: bool = False

    def validate(self) -> None:
        if set(self.prior.keys()) != set(self.labels):
            raise ValueError(f"concept '{self.name}': prior keys != labels")
        if abs(sum(self.prior.values()) - 1.0) > 1e-9:
            raise ValueError(f"concept '{self.name}': prior sums to {sum(self.prior.values())}")
        for slot in self.slots:
            if set(slot.labels) != set(self.labels):
                raise ValueError(
                    f"concept '{self.name}': slot '{slot.name}' labels {set(slot.labels)} "
                    f"!= concept labels {set(self.labels)}"
                )

    @property
    def is_unanswerable(self) -> bool:
        """True iff the label is drawn independently of every slot -- i.e. the
        text contains, by construction, zero evidence about it."""
        return len(self.slots) == 0

    def prior_vector(self) -> np.ndarray:
        return np.array([self.prior[y] for y in self.labels], dtype=np.float64)


CATEGORY = ConceptSpec(
    name="category",
    labels=CATEGORY_LABELS,
    prior={y: 0.25 for y in CATEGORY_LABELS},
    slots=[COMPLAINT_PHRASE, SENDER_DOMAIN, TONE],
)

URGENCY = ConceptSpec(
    name="urgency",
    labels=URGENCY_LEVELS,
    prior={"low": 0.4, "medium": 0.3, "high": 0.2, "critical": 0.1},
    slots=[INTENSITY_MARKER, EXCLAIM_COUNT],
    ordinal=True,
)

IS_BUSINESS = ConceptSpec(
    name="is_business",
    labels=BUSINESS_LABELS,
    prior={"business": 0.5, "personal": 0.5},
    slots=[SIGNATURE_STYLE],
)

# Zero slots: drawn independently of the text, prior == posterior always.
# Same label set and prior shape as URGENCY on purpose -- a model that
# spuriously reuses urgency evidence for this concept is exactly the failure
# mode this concept is built to expose.
HIDDEN_PRIORITY = ConceptSpec(
    name="hidden_priority",
    labels=URGENCY_LEVELS,
    prior={"low": 0.4, "medium": 0.3, "high": 0.2, "critical": 0.1},
    slots=[],
    ordinal=True,
)

CONCEPTS: Dict[str, ConceptSpec] = {
    c.name: c for c in [CATEGORY, URGENCY, IS_BUSINESS, HIDDEN_PRIORITY]
}
for _c in CONCEPTS.values():
    _c.validate()


# ---------------------------------------------------------------------------
# Exact posterior arithmetic (the load-bearing function of the whole project)
# ---------------------------------------------------------------------------


def unnormalized_joint(concept: ConceptSpec, realizations: Dict[str, str]) -> Dict[str, float]:
    """{label: prior(label) * prod_slot P(realization_slot | label)}."""
    out = {}
    for y in concept.labels:
        p = concept.prior[y]
        for slot in concept.slots:
            p *= slot.emissions[y][realizations[slot.name]]
        out[y] = p
    return out


def posterior(concept: ConceptSpec, realizations: Dict[str, str]) -> Dict[str, float]:
    """Exact P(label | realizations), by Bayes' rule over the naive-Bayes joint.

    If concept.slots is empty this is, by construction, exactly concept.prior
    regardless of `realizations` (which will be `{}`).
    """
    unnorm = unnormalized_joint(concept, realizations)
    z = sum(unnorm.values())
    if z <= 0.0:
        raise ValueError(f"concept '{concept.name}': zero-probability realizations {realizations}")
    return {y: p / z for y, p in unnorm.items()}


def binary_posterior(posterior_dict: Dict[str, float], positive_labels: Sequence[str]) -> float:
    return float(sum(posterior_dict[y] for y in positive_labels))


def bayes_error_rate(concept: ConceptSpec) -> float:
    """Exact irreducible error rate: 1 - E_x[max_y P(y|x)], enumerated exactly
    over the (small, finite) realization space -- no Monte Carlo.

    Uses max_y P(combo, y) = P(combo) * max_y P(y|combo), so we can sum the
    unnormalized joint's max directly without normalizing per combo.
    """
    if concept.is_unanswerable:
        return 1.0 - max(concept.prior.values())
    realization_lists = [slot.realizations for slot in concept.slots]
    total = 0.0
    for combo in itertools.product(*realization_lists):
        realizations = {slot.name: r for slot, r in zip(concept.slots, combo)}
        unnorm = unnormalized_joint(concept, realizations)
        total += max(unnorm.values())
    return 1.0 - total


def enumerate_posteriors(concept: ConceptSpec) -> List[Tuple[Dict[str, str], Dict[str, float], float]]:
    """All (realizations, posterior, P(realizations)) triples, exact.

    P(realizations) = sum_y prior(y) * prod P(r|y) is the marginal probability
    of an item ever landing on this exact combo -- used to weight sweeps and
    to re-derive bayes_error_rate as a sanity cross-check (see tests).
    """
    if concept.is_unanswerable:
        return [({}, dict(concept.prior), 1.0)]
    realization_lists = [slot.realizations for slot in concept.slots]
    out = []
    for combo in itertools.product(*realization_lists):
        realizations = {slot.name: r for slot, r in zip(concept.slots, combo)}
        unnorm = unnormalized_joint(concept, realizations)
        z = sum(unnorm.values())
        out.append((realizations, {y: p / z for y, p in unnorm.items()}, z))
    return out


def n_distinct_realization_combos(concept: ConceptSpec) -> int:
    if concept.is_unanswerable:
        return 1
    n = 1
    for slot in concept.slots:
        n *= len(slot.realizations)
    return n


def n_surface_forms_per_concept(concept: ConceptSpec) -> int:
    """Distinct (realization-combo, paraphrase-choice) strings this concept's
    slots alone can produce.

    Exact, via the distributive law over independent per-slot choices: summing
    bank sizes *within* a slot (the paraphrase choices available once that
    slot's realization is fixed), then taking the product *across* slots,
    equals summing the per-combo phrase-count product over every combo --
    i.e. this is `sum_combo prod_slot bank_size(r_slot)`, not an overcount.
    """
    if concept.is_unanswerable:
        return 1
    total = 1
    for slot in concept.slots:
        total *= sum(len(slot.phrases[r]) for r in slot.realizations)
    return total


def n_surface_forms_lower_bound() -> int:
    """A genuine (conservative) lower bound on distinct full-item texts:
    the product, across concepts, of each concept's own surface-form count,
    times the always-present, concept-independent flavor draws (first/last
    name and subject line, which render in every item regardless of any
    concept's realizations). Excludes title/company, which only appear under
    one is_business realization, so as not to overcount.
    """
    total = 1
    for concept in CONCEPTS.values():
        total *= n_surface_forms_per_concept(concept)
    total *= len(FIRST_NAMES) * len(LAST_NAMES) * len(SUBJECT_BANK)
    return total


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_label(concept: ConceptSpec, rng: np.random.Generator) -> str:
    idx = rng.choice(len(concept.labels), p=concept.prior_vector())
    return concept.labels[idx]


def sample_realizations(concept: ConceptSpec, label: str, rng: np.random.Generator) -> Dict[str, str]:
    out = {}
    for slot in concept.slots:
        dist = slot.emissions[label]
        realizations = list(dist.keys())
        probs = np.array([dist[r] for r in realizations], dtype=np.float64)
        idx = rng.choice(len(realizations), p=probs)
        out[slot.name] = realizations[idx]
    return out


# ---------------------------------------------------------------------------
# Rendering: realizations -> natural-language text (label-independent)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Maria", "James", "Aisha", "Wei", "Sofia", "Daniel", "Priya", "Lucas",
    "Emma", "Noah", "Fatima", "Kenji", "Olivia", "Hassan", "Grace", "Tomas",
]
LAST_NAMES = [
    "Alvarez", "Chen", "Okafor", "Nowak", "Silva", "Kowalski", "Haddad",
    "Nguyen", "Berg", "Costa",
]
TITLES = ["Operations Manager", "Procurement Lead", "Account Owner", "Team Lead"]
COMPANIES = ["Meridian Foods", "Alder & Co", "Riverstone Logistics", "Kestrel Studio"]
SUBJECT_BANK = ["Support request", "Question", "Need some help", "Following up", "Quick note"]


@dataclass(frozen=True)
class Person:
    first_name: str
    last_name: str
    title: str
    company: str

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def local_part(self) -> str:
        return f"{self.first_name.lower()}.{self.last_name.lower()}"


def _pick_phrase(slot: SlotSpec, realization: str, paraphrase_rng: np.random.Generator) -> str:
    """Choose a surface string for `realization`. Signature only ever sees the
    realization key and an RNG stream reserved for paraphrase choice -- it has
    no access to any label, so it structurally cannot leak one."""
    bank = slot.phrases[realization]
    idx = paraphrase_rng.integers(0, len(bank))
    return bank[idx]


def _finalize_sentence(text: str) -> str:
    """Strip, and ensure terminal punctuation -- used so an empty realization
    (e.g. complaint_phrase="none") cleanly disappears rather than leaving a
    stray leading space or a doubled '.!' when another slot's text follows."""
    text = text.strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def render_text(
    category_realizations: Dict[str, str],
    urgency_realizations: Dict[str, str],
    business_realizations: Dict[str, str],
    person: Person,
    paraphrase_rng: np.random.Generator,
) -> str:
    tone_phrase = _pick_phrase(TONE, category_realizations["tone"], paraphrase_rng)
    complaint_phrase = _pick_phrase(COMPLAINT_PHRASE, category_realizations["complaint_phrase"], paraphrase_rng)
    domain = _pick_phrase(SENDER_DOMAIN, category_realizations["sender_domain"], paraphrase_rng)

    urgency_phrase = _pick_phrase(INTENSITY_MARKER, urgency_realizations["intensity_marker"], paraphrase_rng)
    exclaim = _pick_phrase(EXCLAIM_COUNT, urgency_realizations["exclaim_count"], paraphrase_rng)

    sig_template = _pick_phrase(SIGNATURE_STYLE, business_realizations["signature_style"], paraphrase_rng)
    signature = sig_template.format(
        name=person.name, first_name=person.first_name, title=person.title, company=person.company
    )

    subject = SUBJECT_BANK[paraphrase_rng.integers(0, len(SUBJECT_BANK))]

    complaint_core = complaint_phrase.rstrip()
    if complaint_core.endswith("."):
        complaint_core = complaint_core[:-1]
    if complaint_core:
        # exclaim marks punctuate the complaint sentence when there is one.
        complaint_sentence = _finalize_sentence(complaint_core + exclaim)
        urgency_sentence = _finalize_sentence(_capitalize(urgency_phrase)) if urgency_phrase else ""
    else:
        # nothing to punctuate -- fold the exclaim marks into the urgency
        # phrase instead of leaving them as an orphaned "!" sentence.
        complaint_sentence = ""
        urgency_sentence = _finalize_sentence(_capitalize(urgency_phrase) + exclaim) if urgency_phrase else (
            _finalize_sentence(exclaim) if exclaim else ""
        )
    body_sentence = " ".join(p for p in (complaint_sentence, urgency_sentence) if p)

    from_line = f"From: {person.local_part}{domain}" if domain.startswith("@") else f"From: {domain}"
    lines = [
        from_line,
        f"Subject: {subject}",
        "",
    ]
    if tone_phrase:
        lines.append(tone_phrase)
    if body_sentence.strip():
        lines.append(body_sentence)
    if signature.strip():
        lines.append("")
        lines.append(signature)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    text: str
    labels: Dict[str, str]  # concept_name -> sampled label (hard label)
    posteriors: Dict[str, Dict[str, float]]  # concept_name -> exact P(y|x)
    realizations: Dict[str, Dict[str, str]]  # concept_name -> {slot_name: realization}
    person: Person
    meta: Dict[str, object] = field(default_factory=dict)


class EmailGenerator:
    """Samples full items: one draw per concept, independently, rendered into
    a single email-shaped text. Concepts are independent of each other by
    design (no cross-concept correlation) -- see docs/explainers.md for why
    this simplification does not weaken any of the calibration claims tested
    here, and for how to lift it.

    Two independent RNG streams are used throughout the project, always: one
    for latents/realizations (`rng`), one for paraphrase/surface-form choice
    and "flavor" draws like names (`paraphrase_rng`). Keeping them separate
    is what makes "paraphrase never sees the label" a structural fact you can
    grep for, not just a claim.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.paraphrase_rng = np.random.default_rng(seed + 1_000_000)

    def _sample_person(self) -> Person:
        r = self.paraphrase_rng
        return Person(
            first_name=FIRST_NAMES[r.integers(0, len(FIRST_NAMES))],
            last_name=LAST_NAMES[r.integers(0, len(LAST_NAMES))],
            title=TITLES[r.integers(0, len(TITLES))],
            company=COMPANIES[r.integers(0, len(COMPANIES))],
        )

    def generate_item(self, overrides: Optional[Dict[str, Dict[str, str]]] = None) -> Item:
        """Draw one item. `overrides`, if given, is {concept_name: realizations}
        to force specific slot realizations (used to build the resolvable /
        ambiguous / unanswerable sweeps in scenarios.py) -- the label is still
        drawn fresh from the concept's true posterior given those realizations,
        so hard labels stay consistent with the ground-truth posterior even
        when realizations are forced rather than sampled top-down.
        """
        overrides = overrides or {}
        labels: Dict[str, str] = {}
        realizations: Dict[str, Dict[str, str]] = {}
        posteriors: Dict[str, Dict[str, float]] = {}

        for concept in CONCEPTS.values():
            if concept.name in overrides:
                r = overrides[concept.name]
            elif concept.is_unanswerable:
                r = {}
            else:
                y_top = sample_label(concept, self.rng)
                r = sample_realizations(concept, y_top, self.rng)
            post = posterior(concept, r)
            y = concept.labels[self.rng.choice(len(concept.labels), p=[post[l] for l in concept.labels])]
            labels[concept.name] = y
            realizations[concept.name] = r
            posteriors[concept.name] = post

        person = self._sample_person()
        text = render_text(
            realizations["category"], realizations["urgency"], realizations["is_business"],
            person, self.paraphrase_rng,
        )
        return Item(text=text, labels=labels, posteriors=posteriors, realizations=realizations, person=person)

    def generate_batch(self, n: int, overrides: Optional[Dict[str, Dict[str, str]]] = None) -> List[Item]:
        return [self.generate_item(overrides=overrides) for _ in range(n)]

"""Slot definitions: the emission tables that make the DGP's posterior exact.

A *concept* (e.g. CATEGORY) has a label and a prior over that label. A *slot*
is an independent evidence channel: given the concept's label, the generator
draws a *realization* from the slot's emission table `P(realization | label)`,
then renders that realization into natural-language text via a bank of
label-independent surface paraphrases.

Because realizations are drawn independently given the label (naive-Bayes
structure) and paraphrase choice never sees the label, the posterior over the
label given a set of *observed realizations* is exactly:

    P(y | r_1, ..., r_k)  proportional to  prior(y) * prod_i P(r_i | y)

This module holds only the emission tables and phrase banks -- the arithmetic
lives in generator.py. See docs/explainers.md ("Naive Bayes and why the
posterior stays closed-form") for the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


Emissions = Dict[str, Dict[str, float]]  # label -> {realization: prob}
PhraseBank = Dict[str, List[str]]  # realization -> surface-form alternatives


@dataclass(frozen=True)
class SlotSpec:
    """One evidence channel for one concept.

    `emissions[label]` must be a valid probability distribution over
    `realizations` (checked by `validate()`). `phrases[realization]` is a
    list of >=1 label-independent surface strings; the renderer picks among
    them uniformly using a paraphrase-only RNG stream that never sees the
    label, so paraphrase choice is label-independent by construction, not
    just in expectation.
    """

    name: str
    emissions: Emissions
    phrases: PhraseBank

    @property
    def labels(self) -> List[str]:
        return list(self.emissions.keys())

    @property
    def realizations(self) -> List[str]:
        # Union over labels, order-stable, taken from the first label's keys
        # (all labels must share the same realization set; validate() checks this).
        first = next(iter(self.emissions.values()))
        return list(first.keys())

    def validate(self, atol: float = 1e-9) -> None:
        realization_sets = {label: set(dist.keys()) for label, dist in self.emissions.items()}
        base = realization_sets[next(iter(realization_sets))]
        for label, rset in realization_sets.items():
            if rset != base:
                raise ValueError(
                    f"slot '{self.name}': label '{label}' has realizations {rset}, "
                    f"expected {base} -- every label must define the same realization set"
                )
        for label, dist in self.emissions.items():
            total = sum(dist.values())
            if abs(total - 1.0) > atol:
                raise ValueError(
                    f"slot '{self.name}': P(.|{label}) sums to {total}, not 1.0"
                )
            for r, p in dist.items():
                if p < 0.0:
                    raise ValueError(f"slot '{self.name}': P({r}|{label})={p} < 0")
        missing_phrases = set(self.realizations) - set(self.phrases.keys())
        if missing_phrases:
            raise ValueError(f"slot '{self.name}': no phrase bank for {missing_phrases}")
        for r, bank in self.phrases.items():
            if len(bank) == 0:
                raise ValueError(f"slot '{self.name}': empty phrase bank for realization '{r}'")

    def log_likelihood_ratio_range(self) -> float:
        """max over label pairs of |log P(r|y_a) - log P(r|y_b)| across realizations.

        A diagnostic, not used in the generative process: 0.0 means the slot
        is an "identical emissions" (zero-evidence) slot for every realization.
        """
        import math

        labels = self.labels
        worst = 0.0
        for r in self.realizations:
            for i, a in enumerate(labels):
                for b in labels[i + 1 :]:
                    pa, pb = self.emissions[a][r], self.emissions[b][r]
                    if pa > 0 and pb > 0:
                        worst = max(worst, abs(math.log(pa) - math.log(pb)))
                    elif pa != pb:
                        worst = float("inf")
        return worst


# ---------------------------------------------------------------------------
# CATEGORY concept: choice among {billing, technical, sales, spam}
# ---------------------------------------------------------------------------

CATEGORY_LABELS = ["billing", "technical", "sales", "spam"]

COMPLAINT_PHRASE = SlotSpec(
    name="complaint_phrase",
    emissions={
        "billing": {
            "invoice_issue": 0.35, "payment_declined": 0.30, "generic_greeting": 0.05,
            "app_crash": 0.0, "error_code": 0.0, "feature_request": 0.0,
            "pricing_question": 0.0, "discount_ask": 0.0, "unsubscribe_now": 0.0,
            "none": 0.30,
        },
        "technical": {
            "invoice_issue": 0.0, "payment_declined": 0.0, "generic_greeting": 0.05,
            "app_crash": 0.35, "error_code": 0.30, "feature_request": 0.0,
            "pricing_question": 0.0, "discount_ask": 0.0, "unsubscribe_now": 0.0,
            "none": 0.30,
        },
        "sales": {
            "invoice_issue": 0.0, "payment_declined": 0.0, "generic_greeting": 0.05,
            "app_crash": 0.0, "error_code": 0.0, "feature_request": 0.20,
            "pricing_question": 0.25, "discount_ask": 0.20, "unsubscribe_now": 0.0,
            "none": 0.30,
        },
        "spam": {
            "invoice_issue": 0.0, "payment_declined": 0.0, "generic_greeting": 0.15,
            "app_crash": 0.0, "error_code": 0.0, "feature_request": 0.0,
            "pricing_question": 0.0, "discount_ask": 0.25, "unsubscribe_now": 0.30,
            "none": 0.30,
        },
    },
    phrases={
        "invoice_issue": [
            "My invoice amount looks wrong.",
            "There's an error on my latest invoice.",
            "The invoice I received doesn't match what I expected.",
            "Something is off with the invoice you sent me.",
        ],
        "payment_declined": [
            "My payment was declined again.",
            "I keep getting a payment failure message.",
            "The card on file was declined for the third time.",
            "Payment isn't going through on my end.",
        ],
        "app_crash": [
            "The app keeps crashing on launch.",
            "Every time I open the app it crashes.",
            "The application closes itself right after I start it.",
            "It crashes consistently within a few seconds of opening.",
        ],
        "error_code": [
            "I'm seeing error code 402 repeatedly.",
            "Getting a strange error code when I try to log in.",
            "There's an error code popping up that I don't recognize.",
            "An unfamiliar error code shows up every time I retry.",
        ],
        "feature_request": [
            "Could you add a dark mode option?",
            "It would be great if you supported CSV export.",
            "Any chance of adding keyboard shortcuts?",
            "I'd love to see a bulk-edit feature added.",
        ],
        "pricing_question": [
            "What's included in the pro plan?",
            "Can you clarify your pricing tiers?",
            "I'm trying to understand the difference between your plans.",
            "Could you break down what each tier includes?",
        ],
        "discount_ask": [
            "Do you have any discount codes available?",
            "Is there a promo running right now?",
            "Any chance of a discount for a longer commitment?",
            "Are there current deals I could take advantage of?",
        ],
        "unsubscribe_now": [
            "Remove me from this list immediately.",
            "Stop sending me these emails.",
            "Please take me off this mailing list.",
            "I did not sign up for this -- unsubscribe me now.",
        ],
        "generic_greeting": [
            "Hope you're doing well.",
            "Just wanted to reach out.",
            "Hope this message finds you well.",
            "Reaching out about something.",
        ],
        "none": [""],
    },
)

SENDER_DOMAIN = SlotSpec(
    name="sender_domain",
    emissions={
        "billing": {"corp_domain": 0.5, "generic_webmail": 0.3, "no_domain_shown": 0.2},
        "technical": {"corp_domain": 0.5, "generic_webmail": 0.3, "no_domain_shown": 0.2},
        "sales": {"corp_domain": 0.3, "generic_webmail": 0.5, "no_domain_shown": 0.2},
        "spam": {"corp_domain": 0.05, "generic_webmail": 0.75, "no_domain_shown": 0.2},
    },
    phrases={
        "corp_domain": ["@acme-corp.com", "@northwind-inc.com", "@bluecrest.co", "@vantacore.com"],
        "generic_webmail": ["@gmail.com", "@yahoo.com", "@outlook.com", "@icloud.com"],
        "no_domain_shown": ["(sender address withheld)"],
    },
)

# Deliberately IDENTICAL across every label: contributes log-likelihood-ratio
# exactly 0 to the CATEGORY posterior for every realization. Kept
# stylistically distinct from the urgency slots below (no "!", no "ASAP")
# so it cannot act as an accidental confound for the URGENCY concept.
TONE = SlotSpec(
    name="tone",
    emissions={
        label: {"polite": 0.30, "curt": 0.20, "formal_register": 0.20, "casual_register": 0.30}
        for label in CATEGORY_LABELS
    },
    phrases={
        "polite": ["Thanks so much for your help.", "I really appreciate you looking into this."],
        "curt": ["Let me know.", "Awaiting your reply."],
        "formal_register": ["Dear Support Team,", "To whom it may concern,"],
        "casual_register": ["Hey,", "Hi there,"],
    },
)

# ---------------------------------------------------------------------------
# URGENCY concept: ordinal score among {low, medium, high, critical}
# ---------------------------------------------------------------------------

URGENCY_LEVELS = ["low", "medium", "high", "critical"]

INTENSITY_MARKER = SlotSpec(
    name="intensity_marker",
    emissions={
        "low": {"none": 0.55, "mild": 0.30, "strong": 0.12, "extreme": 0.03},
        "medium": {"none": 0.20, "mild": 0.45, "strong": 0.25, "extreme": 0.10},
        "high": {"none": 0.08, "mild": 0.22, "strong": 0.45, "extreme": 0.25},
        "critical": {"none": 0.03, "mild": 0.10, "strong": 0.27, "extreme": 0.60},
    },
    phrases={
        "none": [""],
        "mild": ["whenever you get a chance", "no particular rush"],
        "strong": ["this is fairly urgent", "I'd appreciate a quick response"],
        "extreme": ["this needs to be resolved immediately", "URGENT -- please treat as a priority"],
    },
)

EXCLAIM_COUNT = SlotSpec(
    name="exclaim_count",
    emissions={
        "low": {"0": 0.85, "1": 0.12, "2": 0.02, "3+": 0.01},
        "medium": {"0": 0.55, "1": 0.30, "2": 0.10, "3+": 0.05},
        "high": {"0": 0.30, "1": 0.30, "2": 0.25, "3+": 0.15},
        "critical": {"0": 0.15, "1": 0.20, "2": 0.30, "3+": 0.35},
    },
    phrases={"0": [""], "1": ["!"], "2": ["!!"], "3+": ["!!!"]},
)

# ---------------------------------------------------------------------------
# IS_BUSINESS concept: boolean (noul)
# ---------------------------------------------------------------------------

BUSINESS_LABELS = ["business", "personal"]

SIGNATURE_STYLE = SlotSpec(
    name="signature_style",
    emissions={
        "business": {"formal_signoff": 0.5, "first_name_only": 0.35, "no_signoff": 0.15},
        "personal": {"formal_signoff": 0.1, "first_name_only": 0.40, "no_signoff": 0.50},
    },
    phrases={
        "formal_signoff": ["Best regards,\n{name}\n{title}, {company}"],
        "first_name_only": ["Thanks,\n{first_name}"],
        "no_signoff": [""],
    },
)

ALL_SLOTS: Dict[str, SlotSpec] = {
    s.name: s
    for s in [COMPLAINT_PHRASE, SENDER_DOMAIN, TONE, INTENSITY_MARKER, EXCLAIM_COUNT, SIGNATURE_STYLE]
}


def validate_all() -> None:
    for slot in ALL_SLOTS.values():
        slot.validate()


validate_all()

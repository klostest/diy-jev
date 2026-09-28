"""Check what Jev's `confidence` field is, for `choice` answers.

Hypothesis (from S5): confidence = (top - 1/K) / (1 - 1/K), K = number of
options -- i.e. the top probability rescaled so a uniform guess reads 0.

All the S1-S6 and head-to-head questions had K = 4, which can't test the 1/K
part. This script (1) asks Jev category-style questions with K = 2, 3, 5 and 6
options on 100 held-out test emails (one request per email, all four
questions together; cached, so reruns are free), then (2) scores the
hypothesis and some alternatives on EVERY cached `choice` answer, grouped by K.

    set -a; source .env; set +a
    uv run python scripts/check_confidence_formula.py

Writes outputs/confidence_check/results.json.
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np

from jev.dgp.generator import EmailGenerator
from jev.dgp.scenarios import compositional_split
from jev.probe.client import CachingJevClient, JevAPIError, LiveJevClient, Question, ReplayJevClient

CACHE = Path("outputs/jev_cache.jsonl")
OUT = Path("outputs/confidence_check/results.json")
N_EMAILS = 100

BILLING = "About invoices, payments, refunds, or subscription costs."
TECHNICAL = "About bugs, crashes, error codes, or app malfunctions."
SALES = "About pricing, plan features, or discounts."
SPAM = "Unsolicited marketing, or an unsubscribe request."
INSTR = "Classify this support email into exactly one category."
QUESTIONS = {
    "k2": Question("choice", INSTR, {"billing": BILLING, "other": "Anything that is not about billing."}),
    "k3": Question("choice", INSTR, {"billing": BILLING, "technical": TECHNICAL,
                                     "other": "Anything that is not billing or technical."}),
    "k5": Question("choice", INSTR, {"billing": BILLING, "technical": TECHNICAL, "sales": SALES, "spam": SPAM,
                                     "account": "About logging in, passwords, or account access."}),
    "k6": Question("choice", INSTR, {"billing": BILLING, "technical": TECHNICAL, "sales": SALES, "spam": SPAM,
                                     "account": "About logging in, passwords, or account access.",
                                     "feedback": "General feedback, praise, or a thank-you."}),
}


def collect():
    pool = EmailGenerator(seed=100).generate_batch(30_000)
    _, test_items = compositional_split(pool, "category", test_fraction=0.2, seed=100)
    texts = [it.text for it in test_items[:N_EMAILS]]
    replay, live, n_live = ReplayJevClient([CACHE]), None, 0
    for i, text in enumerate(texts):
        try:
            replay.ask(text, QUESTIONS)
        except JevAPIError:
            live = live or CachingJevClient(LiveJevClient(), CACHE)
            for attempt in range(5):
                try:
                    live.ask(text, QUESTIONS)
                    break
                except Exception as e:  # transient network/API errors
                    if attempt == 4:
                        raise
                    print(f"  email {i}: {type(e).__name__}: {e} -- retrying")
                    time.sleep(2 ** attempt)
            n_live += 1
    print(f"asked {len(texts)} emails ({n_live} new live calls)")


def formulas(P, K):
    top = P[:, 0]
    H = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0), axis=1)
    return {
        "(top - 1/K) / (1 - 1/K)": (top - 1 / K) / (1 - 1 / K),
        "(top - 1/4) / (1 - 1/4)  [K fixed at 4]": (top - 0.25) / 0.75,
        "top": top,
        "top - second": top - P[:, 1],
        "1 - entropy / ln K": 1 - H / np.log(K),
    }


def analyze():
    by_k = collections.defaultdict(list)
    for line in open(CACHE):
        for a in json.loads(line)["response"]["answers"].values():
            if a.get("type") == "choice" and "confidence" in a:
                probs = sorted(a["probabilities"].values(), reverse=True)
                by_k[len(probs)].append((probs, a["confidence"]))
    results = {}
    for K in sorted(by_k):
        rows = by_k[K]
        width = max(len(p) for p, _ in rows)
        P = np.array([p + [0.0] * (width - len(p)) for p, _ in rows])
        conf = np.array([c for _, c in rows])
        print(f"\nK = {K}: {len(rows)} answers   (2-decimal rounding alone allows |err| up to "
              f"{0.005 / (1 - 1 / K) + 0.005:.3f})")
        results[K] = {"n": len(rows)}
        for name, pred in formulas(P, K).items():
            err = np.abs(conf - pred)
            results[K][name] = {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
                                "frac_within_0.02": float((err <= 0.02).mean())}
            print(f"  {name:42s} max|err| {err.max():.3f}  mean|err| {err.mean():.4f}  "
                  f"within 0.02: {(err <= 0.02).mean():.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    collect()
    analyze()

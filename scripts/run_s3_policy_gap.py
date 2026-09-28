"""S3 follow-up: does telling Jev that priority depends on information it
can't see change its answers?

Same 200 emails as the original S3 run (build_unanswerable(n=200, seed=1000)),
same answer options, with one sentence added to the front of the question,
in the same style as S6's base-rate sentence. Each email's answer is compared
with its original S3 answer (from outputs/probe_results.json), so the
comparison is paired: only the question wording differs.

The exact answer for every email is still the base rates (low 0.40, medium
0.30, high 0.20, critical 0.10): priority never affects the text.

    set -a; source .env; set +a
    uv run python scripts/run_s3_policy_gap.py

Live calls are cached in outputs/jev_cache.jsonl and replayed on reruns.
Writes outputs/s3_policy_gap/results.json.
"""
from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

import numpy as np

from jev.dgp.generator import URGENCY_LEVELS
from jev.dgp.scenarios import build_unanswerable
from jev.dgp.slots import INTENSITY_MARKER
from jev.eval import metrics as m
from jev.probe.client import CachingJevClient, JevAPIError, LiveJevClient, Question, ReplayJevClient
from jev.probe.suites import HP_CHOICE_QUESTION

CACHE = Path("outputs/jev_cache.jsonl")
ORIGINAL = Path("outputs/probe_results.json")
OUT = Path("outputs/s3_policy_gap/results.json")
POLICY_SENTENCE = ("Context: priority is set by our team's internal account policy, "
                   "which is not visible in this email. ")
POLICY_QUESTION = Question(type="choice",
                           instructions=POLICY_SENTENCE + HP_CHOICE_QUESTION.instructions,
                           criteria=HP_CHOICE_QUESTION.criteria)
BAYES_ACC = 0.40  # always guess "low"


def ask_all(texts):
    replay, live, out, n_live = ReplayJevClient([CACHE]), None, [], 0
    for i, text in enumerate(texts):
        q = {"priority": POLICY_QUESTION}
        try:
            resp = replay.ask(text, q)
        except JevAPIError:
            live = live or CachingJevClient(LiveJevClient(), CACHE)
            for attempt in range(5):
                try:
                    resp = live.ask(text, q)
                    break
                except Exception as e:  # transient network/API errors
                    if attempt == 4:
                        raise
                    print(f"  item {i}: {type(e).__name__}: {e} -- retrying")
                    time.sleep(2 ** attempt)
            n_live += 1
        out.append((resp.model, resp.answers["priority"]))
    print(f"{len(out)} answers ({n_live} new live calls), models: {sorted({mdl for mdl, _ in out})}")
    return [a for _, a in out]


def urgency_phrase(text: str) -> str:
    for level in ["extreme", "strong", "mild"]:
        if any(p and p.lower() in text.lower() for p in INTENSITY_MARKER.phrases[level]):
            return level
    return "none"


def summarize(answers, true_labels, texts):
    conf = np.array([a.get("confidence", max(a["probabilities"].values())) for a in answers])
    correct = np.array([a["choice"] == y for a, y in zip(answers, true_labels)], dtype=float)
    probs = np.array([[a["probabilities"].get(l, 0.0) for l in URGENCY_LEVELS] for a in answers])
    hi = probs[:, 2] + probs[:, 3]
    by_urgency = {}
    for level in ["none", "mild", "strong", "extreme"]:
        mask = np.array([urgency_phrase(t) == level for t in texts])
        by_urgency[level] = {"n": int(mask.sum()), "mean_p_high_or_critical": float(hi[mask].mean()),
                             "accuracy": float(correct[mask].mean())}
    return {
        "accuracy": float(correct.mean()),
        "accuracy_ci": m.wilson_ci(int(correct.sum()), len(correct)),
        "mean_confidence": float(conf.mean()),
        "median_confidence": float(np.median(conf)),
        "frac_confidence_ge_0.9": float((conf >= 0.9).mean()),
        "overconfidence_gap": float(conf.mean() - BAYES_ACC),
        "mean_probabilities": dict(zip(URGENCY_LEVELS, probs.mean(axis=0).round(3).tolist())),
        "choice_counts": {l: int(sum(a["choice"] == l for a in answers)) for l in URGENCY_LEVELS},
        "by_urgency_phrase": by_urgency,
        "_conf": conf, "_hi": hi,
    }


def main():
    items = build_unanswerable(n=200, seed=1000)
    original = json.load(open(ORIGINAL))["S3"]["records"]
    assert all(it.text == r["text"] for it, r in zip(items, original)), "emails differ from original S3"
    texts = [it.text for it in items]
    labels = [it.labels["hidden_priority"] for it in items]

    before = summarize([r["answer"] for r in original], labels, texts)
    after = summarize(ask_all(texts), labels, texts)

    drop = before["_conf"] - after["_conf"]
    paired = {"frac_confidence_lower": float((drop > 0).mean()),
              "frac_confidence_higher": float((drop < 0).mean()),
              "mean_confidence_change": float(-drop.mean()),
              "mean_change_p_high_or_critical": float((after["_hi"] - before["_hi"]).mean())}

    for name, s in [("original S3", before), ("with policy sentence", after)]:
        print(f"\n== {name} ==")
        for k in ["accuracy", "accuracy_ci", "mean_confidence", "median_confidence",
                  "frac_confidence_ge_0.9", "overconfidence_gap", "mean_probabilities", "choice_counts"]:
            v = s[k]
            print(f"  {k}: {tuple(round(x, 3) for x in v) if isinstance(v, tuple) else (round(v, 3) if isinstance(v, float) else v)}")
        for level, d in s["by_urgency_phrase"].items():
            print(f"  urgency {level:8s} n={d['n']:3d}  P(high or critical) {d['mean_p_high_or_critical']:.2f}  acc {d['accuracy']:.2f}")
    print("\n== paired (same email, before -> after) ==")
    for k, v in paired.items():
        print(f"  {k}: {v:+.3f}" if "change" in k else f"  {k}: {v:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}
    json.dump({"policy_sentence": POLICY_SENTENCE, "n": len(items),
               "original": clean(before), "policy_gap": clean(after), "paired": paired},
              open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()

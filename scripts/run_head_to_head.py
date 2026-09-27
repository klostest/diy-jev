"""Head-to-head: the real Jev (raw and recalibrated with your own labels)
vs. toy models trained from scratch on your own labels, on the SAME emails.

The practitioner's question: with some labeled historical data, is it better
to post-process Jev's probabilities or to train your own model -- and how many
labels does each need? Everything is scored on one evaluation set drawn from
the toy experiment's held-out test split (clue combinations no toy model
trained on).

Stages (each resumable; run in order, `jev` and `toy` can run concurrently):

  uv run python scripts/run_head_to_head.py jev      # ask Jev about all 6,440 test emails (cached)
  uv run python scripts/run_head_to_head.py toy      # train hard-label toys on N labels (+ soft reference)
  uv run python scripts/run_head_to_head.py analyze  # recalibrate Jev on N labels, score everything

The `jev` stage needs JEV_API_KEY etc. in the environment (see probe/client.py);
e.g. `set -a; source .env; set +a` first. Every live call is appended to
outputs/jev_cache.jsonl and replayed on reruns, so nothing is paid for twice.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from jev.dgp.generator import CATEGORY_LABELS, URGENCY_LEVELS, EmailGenerator, HIDDEN_PRIORITY
from jev.dgp.scenarios import compositional_split

OUT_DIR = Path("outputs/head_to_head")
CACHE = Path("outputs/jev_cache.jsonl")
DATA_SEED = 100          # same pool/split as scripts/run_toy_experiment.py
N_POOL = 30_000
SPLIT_SEED = 7           # calibration-pool vs evaluation-set split of the test emails
N_CALIB_POOL = 2_000
TOY_NS = [50, 200, 1_000, 5_000, None]   # None = every training email
TOY_SEEDS = [100, 101, 102]
CALIB_NS = [50, 200, 1_000, 2_000]
CALIB_REPEATS = 3
JEV_FLOOR = 0.005        # Jev rounds to 0.01, so a reported 0 means "below 0.005"
PRIORITY_LABELS = URGENCY_LEVELS  # hidden_priority uses the same four level names


def load_split():
    g = EmailGenerator(seed=DATA_SEED)
    pool = g.generate_batch(N_POOL)
    train_items, test_items = compositional_split(pool, "category", test_fraction=0.2, seed=DATA_SEED)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(test_items))
    calib_idx, eval_idx = order[:N_CALIB_POOL], order[N_CALIB_POOL:]
    return train_items, test_items, calib_idx, eval_idx


# ---------------------------------------------------------------------------
# Stage 1: Jev
# ---------------------------------------------------------------------------


def stage_jev():
    from jev.probe.client import CachingJevClient, JevAPIError, LiveJevClient, ReplayJevClient
    from jev.probe.suites import CATEGORY_QUESTION, HP_CHOICE_QUESTION

    _, test_items, _, _ = load_split()
    questions = {"category": CATEGORY_QUESTION, "priority": HP_CHOICE_QUESTION}
    replay = ReplayJevClient([CACHE])
    live = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, n_live, t0 = [], 0, time.time()
    for i, item in enumerate(test_items):
        try:
            resp = replay.ask(item.text, questions)
        except JevAPIError:
            live = live or CachingJevClient(LiveJevClient(), CACHE)
            for attempt in range(5):
                try:
                    resp = live.ask(item.text, questions)
                    break
                except Exception as e:  # transient network/API errors: back off and retry
                    if attempt == 4:
                        raise
                    print(f"  item {i}: {type(e).__name__}: {e} -- retrying")
                    time.sleep(2 ** attempt)
            n_live += 1
        rows.append({"idx": i, "model": resp.model,
                     "category": resp.answers["category"], "priority": resp.answers["priority"]})
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(test_items)} ({n_live} live calls, {(time.time() - t0) / 60:.1f} min)", flush=True)
    with open(OUT_DIR / "jev_answers.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"done: {len(rows)} answers, {n_live} new live calls, models seen: {sorted({r['model'] for r in rows})}")


# ---------------------------------------------------------------------------
# Stage 2: toy models trained on your own labels
# ---------------------------------------------------------------------------


def _epochs_for(n_train: int) -> int:
    # Roughly the same number of optimizer steps as the full-data 3-epoch run,
    # capped at 40 epochs; the best epoch is then chosen on held-out hard labels.
    return int(min(40, max(3, round(3 * 23_560 / n_train))))


def stage_toy():
    import torch
    from jev.toy.train import TrainConfig, pick_device, predict_all, train_model
    from transformers import AutoTokenizer

    train_items, test_items, _, _ = load_split()
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    device = pick_device()
    runs = [("hard", n, s) for n in TOY_NS for s in TOY_SEEDS] + [("soft", None, TOY_SEEDS[0])]
    for mode, n, seed in runs:
        n_total = n or len(train_items)
        path = OUT_DIR / f"toy_{mode}_n{n_total}_seed{seed}.npz"
        if path.exists():
            print(f"skip {path.name} (exists)")
            continue
        rng = np.random.default_rng(seed)
        subset = [train_items[j] for j in rng.permutation(len(train_items))[:n_total]]
        n_val = max(10, n_total // 5)
        val, fit = subset[:n_val], subset[n_val:]
        cfg = TrainConfig(mode=mode, epochs=_epochs_for(len(fit)), seed=seed, select_best_on_val=True)
        print(f"\n=== {mode} N={n_total} (fit {len(fit)}, val {len(val)}) seed={seed} epochs={cfg.epochs} ===", flush=True)
        t0 = time.time()
        model, history = train_model(fit, cfg, val_items=val)
        preds, _, _ = predict_all(model, test_items, tok, device)
        best = [e["epoch"] for e in history.epoch_summaries if e.get("best_so_far")]
        np.savez(path, category=preds["category"], priority=preds["hidden_priority_choice"],
                 best_epoch=best[-1] if best else -1, minutes=(time.time() - t0) / 60)
        print(f"saved {path.name} in {(time.time() - t0) / 60:.1f} min (best epoch {best[-1] if best else '?'})", flush=True)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Stage 3: recalibrate Jev on N labels, score everything on the eval set
# ---------------------------------------------------------------------------


def _floor_renorm(p: np.ndarray, floor: float) -> np.ndarray:
    q = np.clip(p, floor, 1.0)
    return q / q.sum(axis=1, keepdims=True)


def _jev_matrix(rows, key, labels):
    return np.array([[float(r[key]["probabilities"].get(l, 0.0)) for l in labels] for r in rows])


def _recal_temperature(p_fit, y_fit, p_eval):
    from jev.eval import metrics as m
    t = m.fit_temperature_multiclass(p_fit, y_fit)
    logits = np.log(p_eval) / t
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), {"T": t}


def _recal_isotonic(p_fit, y_fit, p_eval):
    from sklearn.isotonic import IsotonicRegression
    k = p_fit.shape[1]
    out = np.zeros_like(p_eval)
    for c in range(k):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p_fit[:, c], (y_fit == c).astype(float))
        out[:, c] = iso.predict(p_eval[:, c])
    return _floor_renorm(out, 1e-3), {}


def _recal_logistic(p_fit, y_fit, p_eval):
    from sklearn.linear_model import LogisticRegression
    k = p_fit.shape[1]
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(np.log(p_fit), y_fit)
    out = np.full((len(p_eval), k), 1e-3)
    out[:, clf.classes_] = clf.predict_proba(np.log(p_eval))
    return _floor_renorm(out, 1e-3), {}


RECALIBRATORS = {"temperature": _recal_temperature, "isotonic": _recal_isotonic, "logistic": _recal_logistic}


def _score(pred, truth, hard, kl_floor=1e-12):
    from jev.eval import metrics as m
    conf = pred.max(axis=1)
    correct = (pred.argmax(axis=1) == hard).astype(float)
    cov, risk = m.risk_coverage_curve(conf, correct)
    kl = m.kl_divergence(truth, pred, eps=kl_floor)
    return {"accuracy": float(correct.mean()), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
            "ece": float(m.expected_calibration_error(conf, correct)),
            "aurc": float(m.area_under_risk_coverage(cov, risk)), "mean_confidence": float(conf.mean())}


def _agg(dicts):
    keys = dicts[0].keys()
    return {k: {"mean": float(np.mean([d[k] for d in dicts])),
                "std": float(np.std([d[k] for d in dicts], ddof=1)) if len(dicts) > 1 else 0.0,
                "n_runs": len(dicts)} for k in keys}


def stage_analyze():
    train_items, test_items, calib_idx, eval_idx = load_split()
    rows = [json.loads(l) for l in open(OUT_DIR / "jev_answers.jsonl")]
    assert len(rows) == len(test_items)

    truth = {"category": np.array([[it.posteriors["category"][l] for l in CATEGORY_LABELS] for it in test_items]),
             "priority": np.array([[it.posteriors["hidden_priority"][l] for l in PRIORITY_LABELS] for it in test_items])}
    hard = {"category": np.array([CATEGORY_LABELS.index(it.labels["category"]) for it in test_items]),
            "priority": np.array([PRIORITY_LABELS.index(it.labels["hidden_priority"]) for it in test_items])}
    jev_raw = {"category": _jev_matrix(rows, "category", CATEGORY_LABELS),
               "priority": _jev_matrix(rows, "priority", PRIORITY_LABELS)}

    results = {"n_eval": int(len(eval_idx)), "n_calib_pool": int(len(calib_idx)),
               "models_seen": sorted({r["model"] for r in rows}), "category": {}, "priority": {}}
    for q in ["category", "priority"]:
        T, H, E = truth[q], hard[q], eval_idx
        R = results[q]
        R["oracle"] = _score(T[E], T[E], H[E])
        R["jev_raw_preregistered_floor"] = _score(_floor_renorm(jev_raw[q][E], 1e-12), T[E], H[E])
        jev = _floor_renorm(jev_raw[q], JEV_FLOOR)
        R["jev_raw"] = _score(jev[E], T[E], H[E])
        for name, fn in RECALIBRATORS.items():
            for n in CALIB_NS:
                runs = []
                for rep in range(CALIB_REPEATS if n < len(calib_idx) else 1):
                    fit = np.random.default_rng(1000 + rep).choice(calib_idx, size=n, replace=False)
                    pred, _ = fn(jev[fit], H[fit], jev[E])
                    runs.append(_score(pred, T[E], H[E]))
                R[f"jev_{name}_n{n}"] = _agg(runs)
        for mode, ns in [("hard", TOY_NS), ("soft", [None])]:
            for n in ns:
                n_total = n or len(train_items)
                runs = []
                for seed in TOY_SEEDS:
                    path = OUT_DIR / f"toy_{mode}_n{n_total}_seed{seed}.npz"
                    if path.exists():
                        runs.append(_score(np.load(path)[q][E], T[E], H[E]))
                if runs:
                    R[f"toy_{mode}_n{n_total}"] = _agg(runs)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    def fmt(v, k):
        return f"{v[k]['mean']:.4f}±{v[k]['std']:.4f}" if isinstance(v[k], dict) else f"{v[k]:.4f}"
    for q in ["category", "priority"]:
        print(f"\n[{q}]  eval n={len(eval_idx)}")
        print(f"  {'model':34s} {'accuracy':>16s} {'mean_kl':>16s} {'ece':>16s} {'aurc':>16s}")
        for name, v in results[q].items():
            print(f"  {name:34s} " + " ".join(f"{fmt(v, k):>16s}" for k in ["accuracy", "mean_kl", "ece", "aurc"]))
    print(f"\nSaved {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"jev": stage_jev, "toy": stage_toy, "analyze": stage_analyze}[stage]()

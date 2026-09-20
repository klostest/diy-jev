"""The actual experiment: soft-target (KL to the generator's true posterior)
vs hard-label (cross-entropy to one sampled label) training. Same
architecture, data, seed, optimizer, schedule -- only the target differs --
evaluated on a compositionally-held-out split plus the resolvable /
ambiguous / unanswerable family stress sets, plus a shuffled-label negative
control that must land at chance.

Run: uv run python scripts/run_toy_experiment.py
Saves outputs/toy_experiment/{results.json, *.png}.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from jev.dgp.generator import bayes_error_rate, CATEGORY
from jev.dgp.scenarios import (
    build_ambiguous,
    build_resolvable,
    build_unanswerable,
    compositional_split,
    permute_hard_labels,
)
from jev.dgp.generator import EmailGenerator
from jev.eval import metrics as m
from jev.eval import report as rpt
from jev.toy.train import TrainConfig, evaluate, pick_device, predict_all, train_model

OUT_DIR = Path("outputs/toy_experiment")
SEED = 100
N_POOL = 30_000


def strip_for_json(obj):
    if dataclasses.is_dataclass(obj):
        return strip_for_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: strip_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strip_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def summarize_head(name: str, d: dict) -> str:
    b = d["brier"]  # BrierDecomposition dataclass -- attribute access, not subscript
    lines = [
        f"  {name:24s} acc={d['accuracy']:.3f} [{d['accuracy_ci'][0]:.3f},{d['accuracy_ci'][1]:.3f}]"
        f"  ece={d['ece']:.3f} (floor {d['ece_noise_floor']:.3f}+/-{d['ece_noise_floor_std']:.3f})"
        f"  reliability={b.reliability:.4f}  resolution={b.resolution:.4f}"
        f"  mean_kl={d['mean_kl']:.4f}  aurc={d['aurc']:.4f}"
    ]
    if "bayes_error_rate" in d:
        lines.append(f"    (bayes error rate = {d['bayes_error_rate']:.3f}, empirical error = {1 - d['accuracy']:.3f})")
    return "\n".join(lines)


def main():
    print(f"device: {pick_device()}")
    print("Generating dataset...")
    g = EmailGenerator(seed=SEED)
    pool = g.generate_batch(N_POOL)
    train_items, test_items = compositional_split(pool, "category", test_fraction=0.2, seed=SEED)
    print(f"train={len(train_items)} test={len(test_items)}")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    device = pick_device()

    results = {}
    models = {}
    for mode in ["soft", "hard"]:
        print(f"\n=== training mode={mode} ===")
        cfg = TrainConfig(mode=mode, epochs=3, batch_size=32, seed=SEED)
        t0 = time.time()
        model, history = train_model(train_items, cfg, val_items=test_items[:500])
        dt = time.time() - t0
        print(f"trained {mode} in {dt / 60:.1f} min")
        models[mode] = model

        held_out = evaluate(model, test_items, tokenizer, device)
        resolvable_items = build_resolvable("category", 1000, seed=SEED + 1)
        resolvable = evaluate(model, resolvable_items, tokenizer, device)
        unanswerable_items = build_unanswerable(1000, seed=SEED + 2)
        unanswerable = evaluate(model, unanswerable_items, tokenizer, device)
        ambiguous_items = [s.item for s in build_ambiguous(
            "category", ["sales", "spam"], target=0.6, n=1000, seed=SEED + 3, tol=0.2
        )]
        ambiguous = evaluate(model, ambiguous_items, tokenizer, device)

        results[mode] = {
            "held_out": held_out, "resolvable": resolvable,
            "unanswerable": unanswerable, "ambiguous": ambiguous,
            "train_seconds": dt, "history": history.epoch_summaries,
        }

    # --- Negative control: shuffle the category label across items, keep
    # text/posteriors untouched, train a fast hard-label model on it. A
    # correctly-built pipeline must land at chance (== the Bayes rate of a
    # uniform-random 4-way guess, 0.75 error), no matter how well it fits
    # the (uninformative-by-construction) training data.
    print("\n=== shuffled-label negative control ===")
    small_pool = g.generate_batch(6000)
    shuffled = permute_hard_labels(small_pool, "category", seed=SEED)
    shuf_train, shuf_test = compositional_split(shuffled, "category", test_fraction=0.2, seed=SEED)
    cfg_shuf = TrainConfig(mode="hard", epochs=2, batch_size=32, seed=SEED)
    shuf_model, _ = train_model(shuf_train, cfg_shuf)
    results["shuffled_label_control"] = evaluate(shuf_model, shuf_test, tokenizer, device)

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(strip_for_json(results), f, indent=2)

    print("\n\n===== SUMMARY =====")
    for mode in ["soft", "hard"]:
        print(f"\n-- mode={mode} --")
        for split_name in ["held_out", "resolvable", "unanswerable", "ambiguous"]:
            print(f" [{split_name}]")
            for head in ["category", "urgency", "is_business", "hidden_priority_choice"]:
                print(summarize_head(head, results[mode][split_name][head]))

    print("\n-- shuffled-label control (mode=hard, must be near chance) --")
    print(summarize_head("category", results["shuffled_label_control"]["category"]))
    print(f"  (a uniform-random 4-way guess has error rate {1 - 1 / 4:.3f})")

    # --- Plots: category head, held-out split, soft vs hard ---
    curves = {}
    for mode in ["soft", "hard"]:
        preds, targets_soft, targets_hard = predict_all(models[mode], test_items, tokenizer, device)
        conf = preds["category"].max(axis=1)
        correct = (preds["category"].argmax(axis=1) == targets_hard["category"]).astype(float)
        fig = rpt.reliability_diagram(
            conf, correct, n_bins=10, method="equal_mass",
            title=f"Category head ({mode}-trained, held-out)",
        )
        rpt.save_fig(fig, str(OUT_DIR / f"reliability_category_{mode}.png"))
        curves[mode] = m.risk_coverage_curve(conf, correct)

    fig = rpt.risk_coverage_plot(curves, title="Category head: risk-coverage, held-out")
    rpt.save_fig(fig, str(OUT_DIR / "risk_coverage_category.png"))

    print(f"\nSaved report to {OUT_DIR}/")


if __name__ == "__main__":
    main()

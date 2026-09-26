"""The actual experiment: soft-target (KL to the generator's true posterior)
vs hard-label (cross-entropy to one sampled label) training. Same
architecture, optimizer, schedule, and DATA (one fixed pool/split, generated
once) across three training seeds per mode -- per the plan's own research-
practices rule ("report across >=3 seeds for any training claim") -- plus a
shuffled-label negative control that must land at chance.

Run: uv run python scripts/run_toy_experiment.py
Saves outputs/toy_experiment/{results.json, summary.json, *.png}.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from jev.dgp.generator import EmailGenerator
from jev.dgp.scenarios import (
    build_ambiguous,
    build_resolvable,
    build_unanswerable,
    compositional_split,
    permute_hard_labels,
)
from jev.eval import metrics as m
from jev.eval import report as rpt
from jev.toy.train import TrainConfig, evaluate, pick_device, predict_all, train_model

OUT_DIR = Path("outputs/toy_experiment")
DATA_SEED = 100          # fixed: same pool/split for every training seed, so the
                          # comparison isolates training randomness, not data sampling.
TRAIN_SEEDS = [100, 101, 102]
N_POOL = 30_000

HEADS = ["category", "urgency", "is_business", "hidden_priority_choice"]
SPLITS = ["held_out", "resolvable", "unanswerable", "ambiguous"]
HEADLINE_METRICS = ["accuracy", "ece", "aurc", "mean_kl"]


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


def extract_headline(d: dict) -> dict:
    return {k: d[k] for k in HEADLINE_METRICS}


def aggregate_across_seeds(per_seed: list) -> dict:
    """per_seed: list of {metric: value}, one per training seed."""
    out = {}
    for metric in HEADLINE_METRICS:
        vals = [v[metric] for v in per_seed]
        out[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def fmt(agg_metric: dict) -> str:
    return f"{agg_metric['mean']:.4f}+/-{agg_metric['std']:.4f}"


def main():
    device = pick_device()
    print(f"device: {device}")
    print(f"Generating dataset once (data_seed={DATA_SEED}), reused across all training seeds...")
    g = EmailGenerator(seed=DATA_SEED)
    pool = g.generate_batch(N_POOL)
    train_items, test_items = compositional_split(pool, "category", test_fraction=0.2, seed=DATA_SEED)
    print(f"train={len(train_items)} test={len(test_items)}")

    resolvable_items = build_resolvable("category", 1000, seed=DATA_SEED + 1)
    unanswerable_items = build_unanswerable(1000, seed=DATA_SEED + 2)
    ambiguous_items = [s.item for s in build_ambiguous(
        "category", ["sales", "spam"], target=0.6, n=1000, seed=DATA_SEED + 3, tol=0.2
    )]
    eval_sets = {
        "held_out": test_items, "resolvable": resolvable_items,
        "unanswerable": unanswerable_items, "ambiguous": ambiguous_items,
    }

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    per_seed_results = {"soft": {}, "hard": {}}
    first_seed_models = {}

    for mode in ["soft", "hard"]:
        for seed in TRAIN_SEEDS:
            print(f"\n=== mode={mode} train_seed={seed} ===")
            cfg = TrainConfig(mode=mode, epochs=3, batch_size=32, seed=seed)
            t0 = time.time()
            model, history = train_model(train_items, cfg, val_items=test_items[:500])
            dt = time.time() - t0
            print(f"trained {mode}/seed={seed} in {dt / 60:.1f} min")

            per_seed_results[mode][seed] = {
                split: evaluate(model, items, tokenizer, device) for split, items in eval_sets.items()
            }
            per_seed_results[mode][seed]["train_seconds"] = dt
            per_seed_results[mode][seed]["epoch_summaries"] = history.epoch_summaries

            if seed == TRAIN_SEEDS[0]:
                first_seed_models[mode] = model  # kept only for the illustrative plots below

    # --- Negative control: shuffle the category label across items, keep
    # text/posteriors untouched, train a fast hard-label model on it. A
    # correctly-built pipeline must land at chance no matter how well it
    # fits the (uninformative-by-construction) training data. Single run --
    # this is a sanity check, not a headline claim needing seed replication.
    print("\n=== shuffled-label negative control ===")
    small_pool = g.generate_batch(6000)
    shuffled = permute_hard_labels(small_pool, "category", seed=DATA_SEED)
    shuf_train, shuf_test = compositional_split(shuffled, "category", test_fraction=0.2, seed=DATA_SEED)
    cfg_shuf = TrainConfig(mode="hard", epochs=2, batch_size=32, seed=DATA_SEED)
    shuf_model, _ = train_model(shuf_train, cfg_shuf)
    shuffled_control_result = evaluate(shuf_model, shuf_test, tokenizer, device)

    # --- Aggregate across seeds ---
    summary = {}
    for mode in ["soft", "hard"]:
        summary[mode] = {}
        for split in SPLITS:
            summary[mode][split] = {}
            for head in HEADS:
                per_seed_vals = [extract_headline(per_seed_results[mode][s][split][head]) for s in TRAIN_SEEDS]
                summary[mode][split][head] = aggregate_across_seeds(per_seed_vals)

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(strip_for_json({
            "data_seed": DATA_SEED, "train_seeds": TRAIN_SEEDS, "n_pool": N_POOL,
            "per_seed": per_seed_results, "shuffled_label_control": shuffled_control_result,
        }), f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(strip_for_json(summary), f, indent=2)

    # --- Print summary: category head is the headline claim, everything else exploratory ---
    print("\n\n===== MULTI-SEED SUMMARY (mean +/- std across seeds", TRAIN_SEEDS, ") =====")
    for split in SPLITS:
        print(f"\n[{split}]")
        for head in HEADS:
            soft_s, hard_s = summary["soft"][split][head], summary["hard"][split][head]
            print(f"  {head:24s} "
                  f"acc soft={fmt(soft_s['accuracy'])} hard={fmt(hard_s['accuracy'])}  "
                  f"ece soft={fmt(soft_s['ece'])} hard={fmt(hard_s['ece'])}  "
                  f"mean_kl soft={fmt(soft_s['mean_kl'])} hard={fmt(hard_s['mean_kl'])}  "
                  f"aurc soft={fmt(soft_s['aurc'])} hard={fmt(hard_s['aurc'])}")

    print("\n-- shuffled-label control (mode=hard, single run, must be near chance) --")
    d = shuffled_control_result["category"]
    print(f"  category accuracy={d['accuracy']:.3f} (uniform-random 4-way guess = {1 - 1 / 4:.3f} error)")

    # --- Plots: category head, held-out split, soft vs hard -- first seed only, illustrative ---
    curves = {}
    for mode in ["soft", "hard"]:
        preds, targets_soft, targets_hard = predict_all(first_seed_models[mode], test_items, tokenizer, device)
        conf = preds["category"].max(axis=1)
        correct = (preds["category"].argmax(axis=1) == targets_hard["category"]).astype(float)
        fig = rpt.reliability_diagram(
            conf, correct, n_bins=10, method="equal_mass",
            title=f"Category head ({mode}-trained, held-out, seed={TRAIN_SEEDS[0]})",
        )
        rpt.save_fig(fig, str(OUT_DIR / f"reliability_category_{mode}.png"))
        curves[mode] = m.risk_coverage_curve(conf, correct)

    fig = rpt.risk_coverage_plot(curves, title=f"Category head: risk-coverage, held-out (seed={TRAIN_SEEDS[0]})")
    rpt.save_fig(fig, str(OUT_DIR / "risk_coverage_category.png"))

    print(f"\nSaved report to {OUT_DIR}/ (results.json = full per-seed data, summary.json = aggregates)")


if __name__ == "__main__":
    main()

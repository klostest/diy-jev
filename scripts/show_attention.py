"""Show where each question's attention-pooling heads look, for a few emails.

Trains one soft-target toy model on the full training split (same data and
config as scripts/run_toy_experiment.py, seed 100), saves it to
outputs/attention_model.pt (reused on later runs), then for each example
email prints, per question and per head, the tokens with the most weight.

    uv run python scripts/show_attention.py
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from jev.dgp.generator import EmailGenerator
from jev.dgp.scenarios import compositional_split
from jev.toy.model import CONCEPT_ORDER, TypedSystemOneModel
from jev.toy.train import TrainConfig, pick_device, train_model

MODEL_PATH = Path("outputs/attention_model.pt")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4


def example_emails():
    """A few emails chosen to exercise different questions."""
    g = EmailGenerator(seed=42)
    items = [g.generate_item() for _ in range(9)]
    worked_example = items[8]  # the fully worked example in EXPERIMENTAL_SETUP.md, Part 1
    pool = EmailGenerator(seed=7).generate_batch(400)

    def find(pred):
        return next(it for it in pool if pred(it))

    feature = find(lambda it: it.realizations["category"]["complaint_phrase"] == "feature_request"
                   and it.realizations["urgency"]["exclaim_count"] == "0")
    no_complaint = find(lambda it: it.realizations["category"]["complaint_phrase"] == "none"
                        and it.realizations["is_business"]["signature_style"] == "formal_signoff")
    calm = find(lambda it: it.realizations["urgency"]["intensity_marker"] == "mild"
                and it.realizations["category"]["complaint_phrase"] == "app_crash")
    return [("worked example (Part 1)", worked_example), ("feature request", feature),
            ("no complaint sentence", no_complaint), ("crash, but 'no rush'", calm)]


def load_or_train(device):
    model = TypedSystemOneModel(encoder_name=ENCODER).to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"loaded {MODEL_PATH}")
        return model
    pool = EmailGenerator(seed=100).generate_batch(30_000)
    train_items, _ = compositional_split(pool, "category", test_fraction=0.2, seed=100)
    model, _ = train_model(train_items, TrainConfig(mode="soft", epochs=3, seed=100))
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    return model


@torch.no_grad()
def show(model, tok, device, label, item):
    enc = tok(item.text, truncation=True, max_length=64, return_tensors="pt").to(device)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    H = model.encode(enc["input_ids"], enc["attention_mask"])
    out = model(enc["input_ids"], enc["attention_mask"])
    print("\n" + "=" * 78)
    print(f"{label}\n" + "-" * 78 + f"\n{item.text}\n" + "-" * 78)
    print("realizations:", {c: r for c, r in item.realizations.items() if r})
    for c_idx, concept in enumerate(CONCEPT_ORDER):
        q = model.query_embed(torch.full((1, 1), c_idx, dtype=torch.long, device=device))
        _, w = model.cross_attn(q, H, H, need_weights=True, average_attn_weights=False)
        w = w[0, :, 0, :]  # (heads, tokens)
        pred_key = "hidden_priority_choice" if concept == "hidden_priority" else concept
        pred = out[pred_key][0]
        pred_txt = (f"P(business)={pred.item():.2f}" if pred.dim() == 0 else
                    " ".join(f"{p:.2f}" for p in pred.tolist()))
        print(f"\n  [{concept}]  model's answer: {pred_txt}")
        for h in range(w.shape[0]):
            top = torch.topk(w[h], TOP_K)
            desc = ", ".join(f"{tokens[i]!r} {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
            print(f"    head {h}: {desc}")


def main():
    device = pick_device()
    model = load_or_train(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(ENCODER)
    for label, item in example_emails():
        show(model, tok, device, label, item)


if __name__ == "__main__":
    main()

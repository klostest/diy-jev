"""Training loop for the toy model, and the soft-vs-hard ablation that is the
actual experiment (see docs/explainers.md, "soft targets vs label smoothing").

`mode="soft"` trains every head against the generator's exact posterior via a
proper scoring rule (cross-entropy against a soft target, which shares its
gradient with KL since the target doesn't depend on the model's parameters).
`mode="hard"` trains the identical architecture against one sampled label
per item instead. Everything else -- data, splits, architecture, optimizer,
schedule, seed -- is held fixed, so any calibration gap between the two runs
is attributable to the target, not a confound.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from jev.dgp.generator import CATEGORY, HIDDEN_PRIORITY, IS_BUSINESS, URGENCY, bayes_error_rate
from jev.dgp.generator import Item
from jev.eval import metrics as m
from jev.toy.data import ToyDataset
from jev.toy.model import TypedSystemOneModel

TargetMode = Literal["soft", "hard"]

HEADS_MULTICLASS = ["category", "urgency", "hidden_priority_choice", "hidden_priority_score"]
HEADS_BINARY = ["is_business", "hidden_priority_noul"]
SOFT_KEY = {
    "category": "category_soft", "urgency": "urgency_soft",
    "is_business": "is_business_soft", "hidden_priority_noul": "hp_noul_soft",
    "hidden_priority_choice": "hp_choice_soft", "hidden_priority_score": "hp_score_soft",
}
HARD_KEY = {
    "category": "category_hard", "urgency": "urgency_hard",
    "is_business": "is_business_hard", "hidden_priority_noul": "hp_noul_hard",
    "hidden_priority_choice": "hp_choice_hard", "hidden_priority_score": "hp_score_hard",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Losses: soft cross-entropy shares category's/urgency's gradient with KL to
# the true posterior (they differ by the target's own entropy, which does
# not depend on model parameters); hard cross-entropy is the same formula
# against a one-hot / 0-1 target instead.
# ---------------------------------------------------------------------------


def soft_xent(pred: torch.Tensor, target_soft: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(target_soft * torch.log(pred.clamp_min(eps))).sum(dim=-1).mean()


def hard_xent(pred: torch.Tensor, hard_idx: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    chosen = pred.gather(1, hard_idx.unsqueeze(1)).squeeze(1)
    return -torch.log(chosen.clamp_min(eps)).mean()


def soft_bce(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.clamp(eps, 1 - eps)
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], mode: TargetMode):
    if mode == "soft":
        l_cat = soft_xent(outputs["category"], batch["category_soft"])
        l_urg = soft_xent(outputs["urgency"], batch["urgency_soft"])
        l_biz = soft_bce(outputs["is_business"], batch["is_business_soft"])
        l_hpn = soft_bce(outputs["hidden_priority_noul"], batch["hp_noul_soft"])
        l_hpc = soft_xent(outputs["hidden_priority_choice"], batch["hp_choice_soft"])
        l_hps = soft_xent(outputs["hidden_priority_score"], batch["hp_score_soft"])
    else:
        l_cat = hard_xent(outputs["category"], batch["category_hard"])
        l_urg = hard_xent(outputs["urgency"], batch["urgency_hard"])
        l_biz = soft_bce(outputs["is_business"], batch["is_business_hard"])
        l_hpn = soft_bce(outputs["hidden_priority_noul"], batch["hp_noul_hard"])
        l_hpc = hard_xent(outputs["hidden_priority_choice"], batch["hp_choice_hard"])
        l_hps = hard_xent(outputs["hidden_priority_score"], batch["hp_score_hard"])
    total = (l_cat + l_urg + l_biz + l_hpn + l_hpc + l_hps) / 6.0
    parts = {"category": l_cat.item(), "urgency": l_urg.item(), "is_business": l_biz.item(),
             "hp_noul": l_hpn.item(), "hp_choice": l_hpc.item(), "hp_score": l_hps.item()}
    return total, parts


def build_optimizer(model: TypedSystemOneModel, encoder_lr: float = 2e-5, head_lr: float = 2e-4):
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ])


@dataclass
class TrainConfig:
    mode: TargetMode = "soft"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 64
    encoder_lr: float = 2e-5
    head_lr: float = 2e-4
    warmup_ratio: float = 0.1
    seed: int = 0
    log_every: int = 50
    device: Optional[torch.device] = None


@dataclass
class TrainHistory:
    train_loss_by_step: List[float] = field(default_factory=list)
    epoch_summaries: List[Dict[str, float]] = field(default_factory=list)


def train_model(
    train_items: List[Item], config: TrainConfig,
    val_items: Optional[List[Item]] = None,
) -> tuple[TypedSystemOneModel, TrainHistory]:
    set_seed(config.seed)
    device = config.device or pick_device()

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    model = TypedSystemOneModel(encoder_name=config.encoder_name).to(device)

    train_ds = ToyDataset(train_items, tokenizer, max_length=config.max_length)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)

    optimizer = build_optimizer(model, config.encoder_lr, config.head_lr)
    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = TrainHistory()
    model.train()
    step = 0
    for epoch in range(config.epochs):
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss, _parts = compute_loss(outputs, batch, config.mode)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            history.train_loss_by_step.append(loss_val)
            epoch_losses.append(loss_val)
            step += 1
            if step % config.log_every == 0:
                print(f"[{config.mode}] epoch {epoch} step {step} loss {loss_val:.4f}")

        summary = {"epoch": epoch, "mean_train_loss": float(np.mean(epoch_losses))}
        if val_items:
            val_metrics = evaluate(model, val_items, tokenizer, device, max_length=config.max_length)
            summary["val_category_ece"] = val_metrics["category"]["ece"]
            summary["val_category_acc"] = val_metrics["category"]["accuracy"]
        history.epoch_summaries.append(summary)
        print(f"[{config.mode}] epoch {epoch} summary: {summary}")

    return model, history


# ---------------------------------------------------------------------------
# Evaluation: reduces every head to (confidence, correct) for ECE/Brier/AURC,
# and keeps the full distribution for posterior-KL against the exact truth.
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_all(model: TypedSystemOneModel, items: List[Item], tokenizer, device, max_length=64, batch_size=64):
    ds = ToyDataset(items, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds: Dict[str, List[np.ndarray]] = {k: [] for k in
                                           ["category", "urgency", "is_business",
                                            "hidden_priority_noul", "hidden_priority_choice",
                                            "hidden_priority_score"]}
    targets_soft: Dict[str, List[np.ndarray]] = {k: [] for k in SOFT_KEY}
    targets_hard: Dict[str, List[np.ndarray]] = {k: [] for k in HARD_KEY}
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev["input_ids"], batch_dev["attention_mask"])
        for k in preds:
            preds[k].append(out[k].detach().cpu().numpy())
        for k, sk in SOFT_KEY.items():
            targets_soft[k].append(batch[sk].numpy())
        for k, hk in HARD_KEY.items():
            targets_hard[k].append(batch[hk].numpy())
    preds = {k: np.concatenate(v, axis=0) for k, v in preds.items()}
    targets_soft = {k: np.concatenate(v, axis=0) for k, v in targets_soft.items()}
    targets_hard = {k: np.concatenate(v, axis=0) for k, v in targets_hard.items()}
    return preds, targets_soft, targets_hard


def _multiclass_head_metrics(pred: np.ndarray, soft_target: np.ndarray, hard_target: np.ndarray) -> Dict:
    confidence = pred.max(axis=1)
    predicted_idx = pred.argmax(axis=1)
    correct = (predicted_idx == hard_target).astype(np.float64)
    ece = m.expected_calibration_error(confidence, correct)
    floor_mean, floor_std = m.ece_noise_floor(confidence, n_boot=100)
    dec = m.brier_decomposition(confidence, correct)
    div = m.mean_posterior_divergence(soft_target, pred)
    acc = float(correct.mean())
    lo, hi = m.wilson_ci(int(correct.sum()), len(correct))
    coverage, risk = m.risk_coverage_curve(confidence, correct)
    aurc = m.area_under_risk_coverage(coverage, risk)
    return {
        "accuracy": acc, "accuracy_ci": (lo, hi), "ece": ece,
        "ece_noise_floor": floor_mean, "ece_noise_floor_std": floor_std,
        "brier": dec, "aurc": aurc, **div,
    }


def _binary_head_metrics(pred: np.ndarray, soft_target: np.ndarray, hard_target: np.ndarray) -> Dict:
    predicted = (pred >= 0.5).astype(np.float64)
    confidence = np.where(predicted == 1, pred, 1 - pred)
    correct = (predicted == hard_target).astype(np.float64)
    ece = m.expected_calibration_error(confidence, correct)
    floor_mean, floor_std = m.ece_noise_floor(confidence, n_boot=100)
    dec = m.brier_decomposition(confidence, correct)
    acc = float(correct.mean())
    lo, hi = m.wilson_ci(int(correct.sum()), len(correct))
    p_true2 = np.stack([soft_target, 1 - soft_target], axis=1)
    p_model2 = np.stack([pred, 1 - pred], axis=1)
    div = m.mean_posterior_divergence(p_true2, p_model2)
    coverage, risk = m.risk_coverage_curve(confidence, correct)
    aurc = m.area_under_risk_coverage(coverage, risk)
    return {
        "accuracy": acc, "accuracy_ci": (lo, hi), "ece": ece,
        "ece_noise_floor": floor_mean, "ece_noise_floor_std": floor_std,
        "brier": dec, "aurc": aurc, **div,
    }


def evaluate(model: TypedSystemOneModel, items: List[Item], tokenizer, device, max_length: int = 64) -> Dict[str, Dict]:
    preds, targets_soft, targets_hard = predict_all(model, items, tokenizer, device, max_length=max_length)
    out = {}
    for k in HEADS_MULTICLASS:
        out[k] = _multiclass_head_metrics(preds[k], targets_soft[k], targets_hard[k])
    for k in HEADS_BINARY:
        out[k] = _binary_head_metrics(preds[k], targets_soft[k], targets_hard[k])

    out["category"]["bayes_error_rate"] = bayes_error_rate(CATEGORY)
    out["urgency"]["bayes_error_rate"] = bayes_error_rate(URGENCY)
    out["is_business"]["bayes_error_rate"] = bayes_error_rate(IS_BUSINESS)
    out["hidden_priority_choice"]["bayes_error_rate"] = bayes_error_rate(HIDDEN_PRIORITY)
    out["hidden_priority_score"]["bayes_error_rate"] = bayes_error_rate(HIDDEN_PRIORITY)
    return out

"""Wraps generator.Item into tokenized tensors with BOTH soft (true posterior)
and hard (sampled label) targets for every head. train.py picks which target
to use per run -- that choice is the whole soft-vs-hard ablation, so both are
always computed here rather than decided at dataset-construction time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from jev.dgp.generator import (
    BUSINESS_LABELS,
    CATEGORY_LABELS,
    HIDDEN_PRIORITY,
    Item,
    URGENCY_LEVELS,
    binary_posterior,
)

# HIDDEN_PRIORITY's noul view: "should this be expedited", i.e. high or critical.
HP_EXPEDITE_LABELS = ["high", "critical"]


def _vec(post: Dict[str, float], order: List[str]) -> np.ndarray:
    return np.array([post[y] for y in order], dtype=np.float32)


class ToyDataset(Dataset):
    def __init__(self, items: List[Item], tokenizer: PreTrainedTokenizerBase, max_length: int = 64):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        enc = self.tokenizer(
            item.text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        cat_post = item.posteriors["category"]
        urg_post = item.posteriors["urgency"]
        biz_post = item.posteriors["is_business"]
        hp_post = item.posteriors["hidden_priority"]  # == HIDDEN_PRIORITY.prior always, by construction

        sample = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category_soft": torch.from_numpy(_vec(cat_post, CATEGORY_LABELS)),
            "category_hard": torch.tensor(CATEGORY_LABELS.index(item.labels["category"]), dtype=torch.long),
            "urgency_soft": torch.from_numpy(_vec(urg_post, URGENCY_LEVELS)),
            "urgency_hard": torch.tensor(URGENCY_LEVELS.index(item.labels["urgency"]), dtype=torch.long),
            "is_business_soft": torch.tensor(biz_post["business"], dtype=torch.float32),
            "is_business_hard": torch.tensor(1.0 if item.labels["is_business"] == "business" else 0.0),
            "hp_noul_soft": torch.tensor(binary_posterior(hp_post, HP_EXPEDITE_LABELS), dtype=torch.float32),
            "hp_noul_hard": torch.tensor(1.0 if item.labels["hidden_priority"] in HP_EXPEDITE_LABELS else 0.0),
            "hp_choice_soft": torch.from_numpy(_vec(hp_post, URGENCY_LEVELS)),
            "hp_choice_hard": torch.tensor(URGENCY_LEVELS.index(item.labels["hidden_priority"]), dtype=torch.long),
            "hp_score_soft": torch.from_numpy(_vec(hp_post, URGENCY_LEVELS)),
            "hp_score_hard": torch.tensor(URGENCY_LEVELS.index(item.labels["hidden_priority"]), dtype=torch.long),
        }
        return sample

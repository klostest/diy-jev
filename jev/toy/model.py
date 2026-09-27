"""The toy System One model: one encoder pass, parallel typed heads.

    text -> [encoder, ONE forward pass]
                |
       +--------+--------+--------+
       |        |        |        |
   category  urgency  is_business hidden_priority
   (choice)  (choice)   (noul)    (noul+choice+score, 3 heads on 1 pooled vec)

Each concept gets its own learned query embedding that cross-attends over the
encoder's token states (attention pooling). This reproduces the two
architectural properties Jev advertises: asking another question costs one
more small attention op, not another encoder pass, and each question reads
the same state in isolation from the others (no shared decoding context to
degrade). See docs/explainers.md for the transformer background and for why
this is an encoder, not a decoder.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from jev.dgp.generator import BUSINESS_LABELS, CATEGORY_LABELS, URGENCY_LEVELS

CONCEPT_ORDER = ["category", "urgency", "is_business", "hidden_priority"]


class OrdinalHead(nn.Module):
    """Proportional-odds / cumulative-link head (McCullagh 1980): a single
    latent utility `eta` per item, and `n_levels - 1` strictly increasing
    cutpoints shared across items, giving P(Y <= k) = sigmoid(cutpoint_k -
    eta). Level probabilities are the differences of consecutive cumulative
    probabilities -- always non-negative and summing to 1 by construction,
    unlike an unconstrained per-level softmax which does not "know" the
    levels are ordered. See docs/explainers.md.
    """

    def __init__(self, in_dim: int, n_levels: int):
        super().__init__()
        assert n_levels >= 2
        self.n_levels = n_levels
        self.eta = nn.Linear(in_dim, 1)
        self.base_cutpoint = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.zeros(max(n_levels - 2, 0)))

    def cutpoints(self) -> torch.Tensor:
        if self.n_levels == 2:
            return self.base_cutpoint
        increments = F.softplus(self.deltas)
        return torch.cat([self.base_cutpoint, self.base_cutpoint + torch.cumsum(increments, dim=0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta = self.eta(x).squeeze(-1)  # (batch,)
        cutpoints = self.cutpoints()  # (n_levels-1,)
        cum = torch.sigmoid(cutpoints.unsqueeze(0) - eta.unsqueeze(1))  # (batch, n_levels-1), increasing in k
        first = cum[:, :1]
        last = 1.0 - cum[:, -1:]
        if self.n_levels > 2:
            middle = cum[:, 1:] - cum[:, :-1]
            probs = torch.cat([first, middle, last], dim=1)
        else:
            probs = torch.cat([first, last], dim=1)
        probs = torch.clamp(probs, min=1e-8)
        return probs / probs.sum(dim=1, keepdim=True)


class TypedSystemOneModel(nn.Module):
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", n_attn_heads: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.hidden = hidden

        self.query_embed = nn.Embedding(len(CONCEPT_ORDER), hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads=n_attn_heads, batch_first=True)

        self.category_head = nn.Linear(hidden, len(CATEGORY_LABELS))
        # Plain softmax, not OrdinalHead: with the ordinal head, urgency stalled
        # at ~0.20 nats held-out KL after 3 epochs (training loss flat well above
        # its entropy floor), while a softmax head reached ~0.005 under the same
        # config. The proportional-odds form can represent these posteriors
        # (direct fit: 0.011 nats) but trains far more slowly here.
        self.urgency_head = nn.Linear(hidden, len(URGENCY_LEVELS))
        self.is_business_head = nn.Linear(hidden, 1)

        # HIDDEN_PRIORITY: three views of the same pooled vector, mirroring
        # probe suite S4 (type consistency) -- if the model is internally
        # consistent, these three heads should agree with each other exactly
        # as they must agree with the (shared, constant) true prior.
        self.hp_noul_head = nn.Linear(hidden, 1)
        self.hp_choice_head = nn.Linear(hidden, len(URGENCY_LEVELS))
        self.hp_score_head = OrdinalHead(hidden, len(URGENCY_LEVELS))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (batch, seq, hidden)

    def _pool(self, token_states: torch.Tensor, attention_mask: torch.Tensor, concept_idx: int) -> torch.Tensor:
        bsz = token_states.shape[0]
        idx = torch.full((bsz, 1), concept_idx, dtype=torch.long, device=token_states.device)
        query = self.query_embed(idx)  # (batch, 1, hidden)
        key_padding_mask = attention_mask == 0  # True where padded -> excluded from attention
        pooled, _ = self.cross_attn(query, token_states, token_states, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)  # (batch, hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        token_states = self.encode(input_ids, attention_mask)
        cat_p = self._pool(token_states, attention_mask, CONCEPT_ORDER.index("category"))
        urg_p = self._pool(token_states, attention_mask, CONCEPT_ORDER.index("urgency"))
        biz_p = self._pool(token_states, attention_mask, CONCEPT_ORDER.index("is_business"))
        hp_p = self._pool(token_states, attention_mask, CONCEPT_ORDER.index("hidden_priority"))

        return {
            "category": F.softmax(self.category_head(cat_p), dim=-1),
            "urgency": F.softmax(self.urgency_head(urg_p), dim=-1),
            "is_business": torch.sigmoid(self.is_business_head(biz_p)).squeeze(-1),
            "hidden_priority_noul": torch.sigmoid(self.hp_noul_head(hp_p)).squeeze(-1),
            "hidden_priority_choice": F.softmax(self.hp_choice_head(hp_p), dim=-1),
            "hidden_priority_score": self.hp_score_head(hp_p),
        }

"""Structural correctness of the toy model: the ordinal head's probability
simplex is valid and its cutpoints are actually monotonic (not merely
intended to be), and a full forward pass produces the right shapes and valid
distributions everywhere before any training run depends on it.
"""
from __future__ import annotations

import torch

from jev.toy.model import CONCEPT_ORDER, OrdinalHead, TypedSystemOneModel


def test_ordinal_head_cutpoints_are_strictly_increasing():
    head = OrdinalHead(in_dim=8, n_levels=4)
    torch.nn.init.normal_(head.deltas, mean=0.5, std=0.1)  # softplus(x)>0 regardless, but exercise it
    cp = head.cutpoints()
    assert cp.shape == (3,)
    assert torch.all(cp[1:] > cp[:-1])


def test_ordinal_head_output_is_a_valid_simplex():
    head = OrdinalHead(in_dim=8, n_levels=4)
    x = torch.randn(16, 8)
    probs = head(x)
    assert probs.shape == (16, 4)
    assert torch.all(probs >= 0)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(16), atol=1e-5)


def test_ordinal_head_binary_case_reduces_correctly():
    head = OrdinalHead(in_dim=4, n_levels=2)
    x = torch.randn(10, 4)
    probs = head(x)
    assert probs.shape == (10, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)


def test_ordinal_head_respects_eta_ordering():
    """Higher latent utility eta should monotonically shift mass toward
    higher levels -- a basic sanity check on the parameterization's direction."""
    head = OrdinalHead(in_dim=1, n_levels=4)
    with torch.no_grad():
        head.eta.weight.fill_(1.0)
        head.eta.bias.fill_(0.0)
    x_low = torch.tensor([[-5.0]])
    x_high = torch.tensor([[5.0]])
    p_low = head(x_low)[0]
    p_high = head(x_high)[0]
    # expected value of the level index should increase with eta
    levels = torch.arange(4, dtype=torch.float32)
    assert (p_high * levels).sum() > (p_low * levels).sum()


def test_model_forward_pass_shapes_and_valid_distributions():
    model = TypedSystemOneModel()
    model.eval()
    tokenizer_vocab_size = model.encoder.config.vocab_size
    batch, seq = 5, 12
    input_ids = torch.randint(0, tokenizer_vocab_size, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    attention_mask[:, -3:] = 0  # pad the last few positions to exercise masking

    with torch.no_grad():
        out = model(input_ids, attention_mask)

    assert set(out.keys()) == {
        "category", "urgency", "is_business",
        "hidden_priority_noul", "hidden_priority_choice", "hidden_priority_score",
    }
    assert out["category"].shape == (batch, 4)
    assert out["urgency"].shape == (batch, 4)
    assert out["is_business"].shape == (batch,)
    assert out["hidden_priority_noul"].shape == (batch,)
    assert out["hidden_priority_choice"].shape == (batch, 4)
    assert out["hidden_priority_score"].shape == (batch, 4)

    for key in ("category", "urgency", "hidden_priority_choice", "hidden_priority_score"):
        assert torch.allclose(out[key].sum(dim=1), torch.ones(batch), atol=1e-4)
    for key in ("is_business", "hidden_priority_noul"):
        assert torch.all(out[key] >= 0) and torch.all(out[key] <= 1)


def test_concept_order_matches_query_embedding_count():
    model = TypedSystemOneModel()
    assert model.query_embed.num_embeddings == len(CONCEPT_ORDER)

"""Mini-GPT: tokenizer, RoPE, causal attention, sampling, KV cache, a short training run."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm-foundations"))

from llmfound.minigpt import (  # noqa: E402
    MiniGPT,
    apply_rope,
    attention_svg,
    benchmark_kv,
    prepare,
    rope_tables,
    sample_token,
    train_steps,
)
from llmfound.tokenizer import ByteBPE  # noqa: E402


def test_byte_bpe_roundtrips_and_merges_shrink_the_corpus():
    text = Path(__file__).resolve().parents[1].joinpath("llm-foundations/corpus/fenwold.txt").read_text()
    tok = ByteBPE().train(text, num_merges=20)
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(text)) < len(text.encode())
    assert tok.vocab_size == 256 + len(tok.merges) + 3
    assert tok.decode([tok.bos_id] + tok.encode("Gift cards") + [tok.eos_id]) == "Gift cards"


def test_rope_depends_on_relative_position():
    torch.manual_seed(0)
    cos, sin = rope_tables(8, 16)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def dot(m, n):
        qq = apply_rope(q, cos[m: m + 1], sin[m: m + 1])
        kk = apply_rope(k, cos[n: n + 1], sin[n: n + 1])
        return float((qq * kk).sum())

    rotated = apply_rope(q, cos[1:2], sin[1:2])
    assert not torch.allclose(rotated, q)
    # A shift of both positions keeps the relative offset, so the dot product is unchanged.
    assert dot(1, 4) == pytest.approx(dot(5, 8), abs=1e-4)
    # A different relative offset is a different dot product. An identity rotation would fail both.
    assert dot(0, 1) != pytest.approx(dot(0, 5), abs=1e-3)


def test_attention_is_causal_and_the_cache_matches_a_full_forward():
    torch.manual_seed(0)
    tok, ids = prepare(num_merges=8, repeats=1)
    model = MiniGPT(tok.vocab_size, n_embd=32, n_head=4, n_layer=2, block_size=48)
    model.eval()
    idx = ids[:12].view(1, -1)
    logits, _cache, attns = model(idx, return_attn=True)
    weights = attns[0][0, 0]  # layer 0, head 0
    assert weights.shape == (12, 12)
    assert torch.allclose(torch.triu(weights, diagonal=1), torch.zeros_like(weights), atol=1e-6)
    assert torch.allclose(weights.sum(-1), torch.ones(12), atol=1e-5)
    assert float(weights[5, 5].detach()) > 0
    full = logits[:, -1]
    _past, cache, _ = model(idx[:, :-1])
    step, _, _ = model(idx[:, -1:], cache=cache)
    assert torch.allclose(full, step[:, -1], atol=1e-4)
    bench = benchmark_kv(model, idx[:, :6], new_tokens=6)
    assert bench["match"] and bench["new_tokens"] == 6
    svg = attention_svg(weights, [str(i) for i in range(12)])
    assert svg.startswith("<svg") and svg.count("<rect") == 144


def test_sampling_is_greedy_for_top_k_one_and_top_p_keeps_a_token():
    logits = torch.tensor([[0.0, 5.0, 1.0], [2.0, 2.0, 0.0]])
    assert sample_token(logits, temperature=0.0, top_k=0, top_p=1.0).tolist() == [1, 0]
    assert sample_token(logits, temperature=1.0, top_k=1, top_p=1.0).tolist() == [1, 0]
    # Token 2 has enough mass to be sampled, and top-p 0.5 drops it (see the nucleus shift).
    narrow = torch.tensor([[2.0, 2.0, 1.5]])
    open_set = {int(sample_token(narrow, 1.0, 0, 1.0, torch.Generator().manual_seed(s))) for s in range(40)}
    closed = {int(sample_token(narrow, 1.0, 0, 0.5, torch.Generator().manual_seed(s))) for s in range(40)}
    assert 2 in open_set
    assert closed <= {0, 1} and closed


def test_a_few_steps_reduce_the_loss():
    tok, ids = prepare(num_merges=12, repeats=2)
    model = MiniGPT(tok.vocab_size, n_embd=32, n_head=4, n_layer=2, block_size=48)
    losses = train_steps(model, ids, steps=12, batch=8, block=16, lr=3e-3, seed=0)
    assert losses[-1] < losses[0]

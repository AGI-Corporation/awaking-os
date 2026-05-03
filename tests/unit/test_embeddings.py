"""EmbeddingProvider tests."""

from __future__ import annotations

import math

import pytest

from awaking_os.memory.embeddings import FakeEmbeddingProvider


def _l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


async def test_fake_embedding_dim() -> None:
    p = FakeEmbeddingProvider(dim=32)
    assert p.dim == 32
    vec = await p.embed("hello world")
    assert len(vec) == 32


async def test_fake_embedding_is_deterministic() -> None:
    p = FakeEmbeddingProvider()
    a = await p.embed("the quick brown fox")
    b = await p.embed("the quick brown fox")
    assert a == b


async def test_fake_embedding_is_normalized() -> None:
    p = FakeEmbeddingProvider()
    vec = await p.embed("alpha beta gamma")
    assert _l2(vec) == pytest.approx(1.0, abs=1e-6)


async def test_fake_embedding_handles_empty_text() -> None:
    p = FakeEmbeddingProvider(dim=8)
    vec = await p.embed("")
    assert vec == [0.0] * 8


async def test_fake_embedding_overlap_increases_similarity() -> None:
    p = FakeEmbeddingProvider(dim=128)
    a = await p.embed("alpha bravo charlie")
    b = await p.embed("alpha bravo delta")  # 2 of 3 tokens shared
    c = await p.embed("xray yankee zulu")  # 0 shared

    def cos(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))

    sim_ab = cos(a, b)
    sim_ac = cos(a, c)
    assert sim_ab > sim_ac


async def test_fake_embedding_token_order_invariant() -> None:
    p = FakeEmbeddingProvider()
    a = await p.embed("alpha beta gamma")
    b = await p.embed("gamma alpha beta")
    assert a == b  # bag-of-words is order-invariant


def test_fake_embedding_rejects_zero_dim() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dim=0)


def test_sentence_transformer_provider_imports_lazily() -> None:
    """The class should not require sentence-transformers to be installed
    until you instantiate it."""
    from awaking_os.memory.embeddings import SentenceTransformerEmbeddingProvider

    assert SentenceTransformerEmbeddingProvider is not None

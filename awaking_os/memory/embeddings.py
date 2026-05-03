"""Embedding providers.

``FakeEmbeddingProvider`` is the default for tests: deterministic, no
external deps. ``SentenceTransformerEmbeddingProvider`` is the default
for real workloads but requires the ``ml`` extra.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_TOKEN_RE = re.compile(r"\w+")


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words hash embedding.

    Each token contributes a fixed vector derived from its SHA-256 hash;
    contributions are summed and L2-normalized. Identical inputs produce
    identical vectors; semantically overlapping inputs produce
    overlapping vectors. Fast, no model download, suitable for tests.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _token_vector(self, token: str) -> list[float]:
        digest = hashlib.sha256(token.lower().encode()).digest()
        # Stretch the digest to fill the requested dim
        repeats = (self._dim + len(digest) - 1) // len(digest)
        stretched = (digest * repeats)[: self._dim]
        return [(b - 128) / 128.0 for b in stretched]

    async def embed(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            return [0.0] * self._dim
        vec = [0.0] * self._dim
        for tok in tokens:
            for i, x in enumerate(self._token_vector(tok)):
                vec[i] += x
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence-Transformers backed provider. Requires ``awaking-os[ml]``."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed; install with `pip install awaking-os[ml]`"
            ) from e
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        vec = await asyncio.to_thread(self._model.encode, text, normalize_embeddings=True)
        return [float(x) for x in vec]

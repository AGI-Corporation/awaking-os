"""Vector store ABC + InMemory and Chroma implementations."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorHit:
    node_id: str
    score: float  # cosine similarity in [-1, 1]; higher is more similar
    metadata: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self,
        node_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def query(self, embedding: list[float], k: int = 5) -> list[VectorHit]: ...

    @abstractmethod
    def count(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    """Numpy-cosine in-memory store. Good for tests and ephemeral runs."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._meta: dict[str, dict[str, Any]] = {}

    async def upsert(
        self,
        node_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        if node_id in self._meta:
            idx = self._ids.index(node_id)
            self._vectors[idx] = vec
        else:
            self._ids.append(node_id)
            self._vectors.append(vec)
        self._meta[node_id] = dict(metadata or {})

    async def query(self, embedding: list[float], k: int = 5) -> list[VectorHit]:
        if not self._ids:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        matrix = np.vstack(self._vectors)
        norms = np.linalg.norm(matrix, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        scores = (matrix @ q) / (norms * q_norm)
        order = np.argsort(-scores)[:k]
        return [
            VectorHit(
                node_id=self._ids[i],
                score=float(scores[i]),
                metadata=dict(self._meta[self._ids[i]]),
            )
            for i in order
        ]

    def count(self) -> int:
        return len(self._ids)


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed store. Persistent if ``persist_path`` is given."""

    def __init__(
        self,
        persist_path: Path | None = None,
        collection: str = "awaking_nodes",
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("chromadb not installed") from e
        if persist_path is None:
            self._client = chromadb.EphemeralClient()
        else:
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path))
        # Disable Chroma's default embedding function — we supply our own.
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self,
        node_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[node_id],
            embeddings=[embedding],
            metadatas=[metadata or {"_": ""}],
        )

    async def query(self, embedding: list[float], k: int = 5) -> list[VectorHit]:
        if self.count() == 0:
            return []
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        hits: list[VectorHit] = []
        for nid, dist, meta in zip(ids, distances, metadatas, strict=False):
            # Chroma cosine distance is in [0, 2]; similarity = 1 - distance.
            hits.append(
                VectorHit(
                    node_id=nid,
                    score=float(1.0 - dist),
                    metadata=dict(meta or {}),
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

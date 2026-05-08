"""VectorStore tests — InMemory + Chroma (when available)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from awaking_os.memory.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorStore,
)


def _normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


@pytest.fixture(
    params=[
        pytest.param("inmem", id="InMemoryVectorStore"),
        pytest.param("chroma", id="ChromaVectorStore"),
    ]
)
def store(request, tmp_path: Path) -> VectorStore:
    if request.param == "inmem":
        return InMemoryVectorStore()
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    return ChromaVectorStore(persist_path=tmp_path / "chroma")


async def test_empty_query_returns_empty(store: VectorStore) -> None:
    assert await store.query([1.0, 0.0, 0.0, 0.0], k=3) == []
    assert store.count() == 0


async def test_upsert_and_count(store: VectorStore) -> None:
    await store.upsert("a", _normalize([1.0, 0.0, 0.0, 0.0]), {"tag": "alpha"})
    await store.upsert("b", _normalize([0.0, 1.0, 0.0, 0.0]), {"tag": "beta"})
    assert store.count() == 2


async def test_query_nearest_first(store: VectorStore) -> None:
    await store.upsert("near", _normalize([1.0, 0.0, 0.0, 0.0]))
    await store.upsert("far", _normalize([0.0, 0.0, 0.0, 1.0]))
    hits = await store.query(_normalize([0.99, 0.01, 0.0, 0.0]), k=2)
    assert [h.node_id for h in hits] == ["near", "far"]
    assert hits[0].score > hits[1].score


async def test_query_respects_k(store: VectorStore) -> None:
    for i, basis in enumerate(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ):
        await store.upsert(f"v{i}", _normalize(basis))
    hits = await store.query(_normalize([1.0, 1.0, 1.0, 1.0]), k=2)
    assert len(hits) == 2


async def test_upsert_updates_existing_id(store: VectorStore) -> None:
    await store.upsert("x", _normalize([1.0, 0.0, 0.0, 0.0]), {"v": 1})
    await store.upsert("x", _normalize([0.0, 1.0, 0.0, 0.0]), {"v": 2})
    assert store.count() == 1
    hits = await store.query(_normalize([0.0, 1.0, 0.0, 0.0]), k=1)
    assert hits[0].node_id == "x"
    assert hits[0].metadata["v"] == 2


async def test_metadata_roundtrip(store: VectorStore) -> None:
    await store.upsert(
        "node-1",
        _normalize([1.0, 0.0, 0.0, 0.0]),
        {"type": "concept", "created_by": "test"},
    )
    hits = await store.query(_normalize([1.0, 0.0, 0.0, 0.0]), k=1)
    assert hits[0].metadata["type"] == "concept"
    assert hits[0].metadata["created_by"] == "test"


async def test_chroma_persistence_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    path = tmp_path / "chroma_persist"
    s1 = ChromaVectorStore(persist_path=path)
    await s1.upsert("persistent", _normalize([1.0, 0.0, 0.0, 0.0]), {"k": "v"})
    assert s1.count() == 1

    s2 = ChromaVectorStore(persist_path=path)
    assert s2.count() == 1
    hits = await s2.query(_normalize([1.0, 0.0, 0.0, 0.0]), k=1)
    assert hits[0].node_id == "persistent"

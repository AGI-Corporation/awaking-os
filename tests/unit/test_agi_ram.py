"""AGIRam tests."""

from __future__ import annotations

import pytest

from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode


def _node(content: str) -> KnowledgeNode:
    return KnowledgeNode(content=content, created_by="test")


async def test_store_and_get(in_memory_agi_ram: AGIRam) -> None:
    n = _node("hello world")
    nid = await in_memory_agi_ram.store(n)
    fetched = await in_memory_agi_ram.get(nid)
    assert fetched is not None
    assert fetched.content == "hello world"


async def test_retrieve_keyword_match(in_memory_agi_ram: AGIRam) -> None:
    a = await in_memory_agi_ram.store(_node("alpha bravo charlie"))
    b = await in_memory_agi_ram.store(_node("delta echo foxtrot"))
    await in_memory_agi_ram.store(_node("nothing relevant"))

    hits = await in_memory_agi_ram.retrieve("alpha")
    assert [n.id for n in hits] == [a]

    hits = await in_memory_agi_ram.retrieve("foxtrot")
    assert [n.id for n in hits] == [b]


async def test_retrieve_ranks_by_overlap(in_memory_agi_ram: AGIRam) -> None:
    high = await in_memory_agi_ram.store(_node("alpha beta gamma"))
    low = await in_memory_agi_ram.store(_node("alpha unrelated"))
    hits = await in_memory_agi_ram.retrieve("alpha beta")
    assert [n.id for n in hits[:2]] == [high, low]


async def test_retrieve_empty_query_returns_empty(in_memory_agi_ram: AGIRam) -> None:
    await in_memory_agi_ram.store(_node("anything"))
    assert await in_memory_agi_ram.retrieve("") == []


async def test_retrieve_no_match(in_memory_agi_ram: AGIRam) -> None:
    await in_memory_agi_ram.store(_node("alpha"))
    assert await in_memory_agi_ram.retrieve("zeta") == []


async def test_retrieve_respects_k(in_memory_agi_ram: AGIRam) -> None:
    for i in range(5):
        await in_memory_agi_ram.store(_node(f"shared word {i}"))
    hits = await in_memory_agi_ram.retrieve("shared", k=3)
    assert len(hits) == 3


async def test_link(in_memory_agi_ram: AGIRam) -> None:
    a = await in_memory_agi_ram.store(_node("a"))
    b = await in_memory_agi_ram.store(_node("b"))
    await in_memory_agi_ram.link(a, b, "rel")
    assert b in [n.id for n in in_memory_agi_ram.graph.neighbors(a, depth=1)]


async def test_link_missing_raises(in_memory_agi_ram: AGIRam) -> None:
    a = await in_memory_agi_ram.store(_node("a"))
    with pytest.raises(KeyError):
        await in_memory_agi_ram.link(a, "missing", "rel")

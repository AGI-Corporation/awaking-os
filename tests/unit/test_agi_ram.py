"""AGIRam tests."""

from __future__ import annotations

import pytest

from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.desci import DeSciSigner, verify
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


# --- Semantic retrieval (PR 2) -------------------------------------------------


async def test_semantic_enabled_flag(in_memory_agi_ram: AGIRam, semantic_agi_ram: AGIRam) -> None:
    assert in_memory_agi_ram.semantic_enabled is False
    assert semantic_agi_ram.semantic_enabled is True


async def test_store_computes_embedding_when_provider_present(
    semantic_agi_ram: AGIRam,
) -> None:
    n = _node("alpha bravo charlie")
    nid = await semantic_agi_ram.store(n)
    fetched = await semantic_agi_ram.get(nid)
    assert fetched is not None
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 64
    assert semantic_agi_ram.vector_store.count() == 1  # type: ignore[union-attr]


async def test_semantic_retrieve_finds_overlap(semantic_agi_ram: AGIRam) -> None:
    near = await semantic_agi_ram.store(_node("alpha bravo charlie"))
    far = await semantic_agi_ram.store(_node("xray yankee zulu"))

    hits = await semantic_agi_ram.retrieve("alpha bravo", k=2)
    assert hits[0].id == near
    assert {h.id for h in hits} == {near, far}


async def test_semantic_retrieve_respects_k(semantic_agi_ram: AGIRam) -> None:
    for i in range(5):
        await semantic_agi_ram.store(_node(f"shared token{i}"))
    hits = await semantic_agi_ram.retrieve("shared", k=3)
    assert len(hits) == 3


async def test_semantic_retrieve_empty_query(semantic_agi_ram: AGIRam) -> None:
    await semantic_agi_ram.store(_node("anything"))
    assert await semantic_agi_ram.retrieve("   ") == []


async def test_semantic_retrieve_empty_store(semantic_agi_ram: AGIRam) -> None:
    assert await semantic_agi_ram.retrieve("anything") == []


# --- Signing (PR 2) ------------------------------------------------------------


async def test_store_signs_when_signer_present(
    signed_semantic_agi_ram: AGIRam, signer: DeSciSigner
) -> None:
    nid = await signed_semantic_agi_ram.store(_node("signed content"))
    node = await signed_semantic_agi_ram.get(nid)
    assert node is not None
    assert node.attestation is not None
    assert node.attestation.public_key == signer.public_key_hex
    assert verify(node.attestation, node)


async def test_store_does_not_sign_when_signer_absent(
    semantic_agi_ram: AGIRam,
) -> None:
    nid = await semantic_agi_ram.store(_node("unsigned"))
    node = await semantic_agi_ram.get(nid)
    assert node is not None
    assert node.attestation is None


async def test_existing_attestation_is_preserved(
    signed_semantic_agi_ram: AGIRam,
) -> None:
    other_signer = DeSciSigner.from_seed(b"\x02" * 32)
    n = _node("pre-signed")
    n.attestation = other_signer.sign(n)
    nid = await signed_semantic_agi_ram.store(n)
    fetched = await signed_semantic_agi_ram.get(nid)
    assert fetched is not None
    assert fetched.attestation is not None
    assert fetched.attestation.public_key == other_signer.public_key_hex

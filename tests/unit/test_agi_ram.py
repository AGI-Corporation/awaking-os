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


# --- atomic store rollback (PR #1 review follow-up) ---------------------------


from awaking_os.memory.vector_store import VectorStore  # noqa: E402


class _BrokenVectorStore(VectorStore):
    """Minimal VectorStore stand-in whose upsert always raises.

    Subclasses the ABC so future additions to ``VectorStore`` will surface as
    abstract-method errors here, rather than letting the test silently pass on
    an incomplete interface.
    """

    def __init__(self) -> None:
        self.upsert_calls = 0

    async def upsert(self, node_id, embedding, metadata=None) -> None:
        self.upsert_calls += 1
        raise RuntimeError("simulated upsert failure")

    async def query(self, embedding, k=5):
        return []

    def count(self) -> int:
        return 0


async def test_store_rolls_back_graph_when_vector_upsert_fails(
    embedding_provider,
) -> None:
    broken = _BrokenVectorStore()
    ram = AGIRam(embedding_provider=embedding_provider, vector_store=broken)
    node = _node("will fail")

    with pytest.raises(RuntimeError, match="simulated upsert failure"):
        await ram.store(node)

    # The node must NOT remain in the graph after rollback.
    assert node.id not in ram.graph
    assert len(ram) == 0
    assert broken.upsert_calls == 1


async def test_store_succeeds_when_vector_store_works(semantic_agi_ram: AGIRam) -> None:
    # Sanity: the happy path still works after the rollback wrapper.
    nid = await semantic_agi_ram.store(_node("happy path"))
    assert nid in semantic_agi_ram.graph
    assert len(semantic_agi_ram) == 1


async def test_rollback_does_not_swallow_original_error(embedding_provider) -> None:
    """If the rollback itself fails, the original upsert error must still raise."""

    class BrokenGraph:
        def __init__(self) -> None:
            self.added: list[str] = []

        def add(self, node):
            self.added.append(node.id)
            return node.id

        def remove(self, node_id):
            raise RuntimeError("rollback also broken")

    broken_vec = _BrokenVectorStore()
    ram = AGIRam(embedding_provider=embedding_provider, vector_store=broken_vec)
    ram.graph = BrokenGraph()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="simulated upsert failure"):
        await ram.store(_node("doomed"))


# --- on-chain publication wiring (Phase C.1) ---------------------------------


async def test_store_publishes_attestation_when_publisher_wired(
    embedding_provider, vector_store, tmp_path
) -> None:
    """AGIRam with a signer + publisher publishes the attestation after
    a successful store. The receipt is recorded in ram.receipts."""
    from awaking_os.memory.onchain import LocalJSONLPublisher

    signer = DeSciSigner.from_seed(b"\x00" * 32)
    publisher = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    ram = AGIRam(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        signer=signer,
        publisher=publisher,
    )

    nid = await ram.store(_node("publish me"))
    assert nid in ram.receipts
    receipt = ram.receipts[nid]
    assert receipt.block_height == 0
    assert publisher.block_count() == 1


async def test_store_does_not_publish_without_signer(
    embedding_provider, vector_store, tmp_path
) -> None:
    """No signer → no attestation → nothing to publish."""
    from awaking_os.memory.onchain import LocalJSONLPublisher

    publisher = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    ram = AGIRam(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        publisher=publisher,  # but no signer
    )
    nid = await ram.store(_node("no sig"))
    assert nid not in ram.receipts
    assert publisher.block_count() == 0


async def test_store_succeeds_when_publisher_fails(embedding_provider, vector_store) -> None:
    """Publication is best-effort; a broken publisher must not fail store()."""

    class _BrokenPublisher:
        async def publish(self, attestation):
            raise RuntimeError("chain offline")

        async def find(self, node_hash):
            return None

        async def verify_chain(self):
            return False

    signer = DeSciSigner.from_seed(b"\x00" * 32)
    ram = AGIRam(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        signer=signer,
        publisher=_BrokenPublisher(),
    )
    nid = await ram.store(_node("survives chain failure"))
    # Store still succeeded — node is in the graph + vector store.
    assert nid in ram.graph
    # But no receipt was recorded.
    assert nid not in ram.receipts


async def test_published_chain_is_verifiable_after_multiple_stores(
    embedding_provider, vector_store, tmp_path
) -> None:
    from awaking_os.memory.onchain import LocalJSONLPublisher

    signer = DeSciSigner.from_seed(b"\x00" * 32)
    publisher = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    ram = AGIRam(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        signer=signer,
        publisher=publisher,
    )
    for i in range(4):
        await ram.store(_node(f"node-{i}"))

    assert publisher.block_count() == 4
    assert await publisher.verify_chain() is True

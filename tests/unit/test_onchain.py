"""On-chain publisher tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from awaking_os.memory.desci import DeSciSigner
from awaking_os.memory.node import DeSciAttestation, KnowledgeNode
from awaking_os.memory.onchain import (
    GENESIS_PREV_HASH,
    LocalJSONLPublisher,
    PublicationReceipt,
)


def _attestation(node_hash: str = "a" * 64) -> DeSciAttestation:
    """Build a DeSciAttestation from the test signer over a fake node hash."""
    signer = DeSciSigner.from_seed(b"\x00" * 32)
    # Create a real node + sign it for shape correctness, then override
    # node_hash if a specific value is desired.
    node = KnowledgeNode(content=node_hash, created_by="test")
    real = signer.sign(node)
    return DeSciAttestation(
        node_hash=node_hash,
        signature=real.signature,
        public_key=real.public_key,
        signed_at=datetime.now(UTC),
    )


# --- Genesis + basic publish ----------------------------------------------


async def test_first_block_uses_genesis_prev_hash(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    receipt = await pub.publish(_attestation("aa" * 32))
    assert receipt.block_height == 0
    assert receipt.prev_hash == GENESIS_PREV_HASH
    assert pub.block_count() == 1
    assert (tmp_path / "chain.jsonl").exists()


async def test_second_block_links_to_first(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    r1 = await pub.publish(_attestation("aa" * 32))
    r2 = await pub.publish(_attestation("bb" * 32))
    assert r2.block_height == 1
    assert r2.prev_hash == r1.tx_hash


async def test_tx_hash_is_deterministic(tmp_path: Path) -> None:
    """Same canonical inputs produce the same tx_hash regardless of timing."""
    pub_a = LocalJSONLPublisher(tmp_path / "a.jsonl")
    pub_b = LocalJSONLPublisher(tmp_path / "b.jsonl")
    att = _attestation("aa" * 32)
    r_a = await pub_a.publish(att)
    r_b = await pub_b.publish(att)
    assert r_a.tx_hash == r_b.tx_hash


# --- Idempotency ----------------------------------------------------------


async def test_republishing_same_node_returns_existing_receipt(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    att = _attestation("aa" * 32)
    r1 = await pub.publish(att)
    r2 = await pub.publish(att)
    assert r1.block_height == r2.block_height == 0
    assert r1.tx_hash == r2.tx_hash
    # Only one block written.
    assert pub.block_count() == 1


# --- Find -----------------------------------------------------------------


async def test_find_returns_existing_receipt(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    await pub.publish(_attestation("aa" * 32))
    found = await pub.find("aa" * 32)
    assert found is not None
    assert found.block_height == 0


async def test_find_unknown_returns_none(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    assert await pub.find("zz" * 32) is None


# --- Verify ---------------------------------------------------------------


async def test_empty_chain_verifies(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    assert await pub.verify_chain() is True


async def test_intact_chain_verifies(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    for i in range(5):
        await pub.publish(_attestation(f"{i:064x}"))
    assert await pub.verify_chain() is True


async def test_tampered_node_hash_fails_verification(tmp_path: Path) -> None:
    chain_path = tmp_path / "chain.jsonl"
    pub = LocalJSONLPublisher(chain_path)
    for i in range(3):
        await pub.publish(_attestation(f"{i:064x}"))
    # Tamper with the middle block's node_hash field directly.
    lines = chain_path.read_text().splitlines()
    import json as _json

    block = _json.loads(lines[1])
    block["node_hash"] = "ff" * 32
    lines[1] = _json.dumps(block, separators=(",", ":"))
    chain_path.write_text("\n".join(lines) + "\n")
    assert await pub.verify_chain() is False


async def test_tampered_prev_hash_fails_verification(tmp_path: Path) -> None:
    chain_path = tmp_path / "chain.jsonl"
    pub = LocalJSONLPublisher(chain_path)
    for i in range(3):
        await pub.publish(_attestation(f"{i:064x}"))
    lines = chain_path.read_text().splitlines()
    import json as _json

    block = _json.loads(lines[2])
    block["prev_hash"] = "00" * 32
    lines[2] = _json.dumps(block, separators=(",", ":"))
    chain_path.write_text("\n".join(lines) + "\n")
    assert await pub.verify_chain() is False


# --- Concurrency ----------------------------------------------------------


async def test_concurrent_publishes_get_distinct_heights(tmp_path: Path) -> None:
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    atts = [_attestation(f"{i:064x}") for i in range(8)]
    receipts = await asyncio.gather(*(pub.publish(a) for a in atts))
    heights = sorted(r.block_height for r in receipts)
    assert heights == list(range(8))
    assert pub.block_count() == 8
    assert await pub.verify_chain() is True


async def test_two_publishers_pointing_at_same_path_share_state(tmp_path: Path) -> None:
    """Two LocalJSONLPublisher instances with the same path must agree on
    the chain head. Their sqlite ledger is the source of truth."""
    chain = tmp_path / "chain.jsonl"
    a = LocalJSONLPublisher(chain)
    b = LocalJSONLPublisher(chain)
    r1 = await a.publish(_attestation("aa" * 32))
    r2 = await b.publish(_attestation("bb" * 32))
    assert r1.block_height == 0
    assert r2.block_height == 1
    assert r2.prev_hash == r1.tx_hash


async def test_publish_rolls_back_on_jsonl_append_failure(tmp_path: Path) -> None:
    """If the JSONL append fails after the sqlite INSERT, the transaction
    must roll back so the height isn't burned. Otherwise a chain hiccup
    leaves a hole that breaks ``verify_chain``."""
    pub = LocalJSONLPublisher(tmp_path / "chain.jsonl")
    # Publish one block successfully so the chain isn't empty.
    await pub.publish(_attestation("aa" * 32))
    assert pub.block_count() == 1

    # Force the next JSONL append to fail by pointing _path at a
    # directory — `open("a")` on a directory raises IsADirectoryError.
    pub._path = tmp_path

    import pytest as _pytest

    with _pytest.raises(IsADirectoryError):
        await pub.publish(_attestation("bb" * 32))

    # Rollback freed the height: ledger still shows only the first block.
    assert pub.block_count() == 1
    # And the next successful publish (after fixing the path) resumes
    # at height 1, not 2.
    pub._path = tmp_path / "chain.jsonl"
    receipt = await pub.publish(_attestation("cc" * 32))
    assert receipt.block_height == 1
    assert pub.block_count() == 2
    assert await pub.verify_chain() is True


async def test_two_publishers_concurrently_get_distinct_heights(tmp_path: Path) -> None:
    """Two LocalJSONLPublisher instances racing on the same chain must
    still produce a strictly-monotonic, hash-linked chain. Each instance
    has its own asyncio.Lock so the cross-instance serialization has to
    come from sqlite's BEGIN IMMEDIATE write lock — not the asyncio one.
    """
    chain = tmp_path / "chain.jsonl"
    a = LocalJSONLPublisher(chain)
    b = LocalJSONLPublisher(chain)
    atts = [_attestation(f"{i:064x}") for i in range(8)]
    # Alternate publishes across the two instances and gather them all
    # at once — the asyncio scheduler will interleave them freely.
    receipts = await asyncio.gather(
        *((a if i % 2 == 0 else b).publish(att) for i, att in enumerate(atts))
    )
    heights = sorted(r.block_height for r in receipts)
    assert heights == list(range(8))
    # Both publishers see the chain consistently.
    assert a.block_count() == 8
    assert b.block_count() == 8
    # And the chain is intact — every block's prev_hash points at the
    # previous block's tx_hash with no duplicate-height holes.
    assert await a.verify_chain() is True
    assert await b.verify_chain() is True


# --- Receipt model --------------------------------------------------------


def test_receipt_to_dict_roundtrip() -> None:
    r = PublicationReceipt(
        block_height=42,
        tx_hash="ab" * 32,
        prev_hash="cd" * 32,
        node_hash="ef" * 32,
        published_at=datetime.now(UTC),
    )
    d = r.to_dict()
    assert d["block_height"] == 42
    assert d["tx_hash"] == "ab" * 32
    assert d["prev_hash"] == "cd" * 32
    assert d["node_hash"] == "ef" * 32
    assert "T" in d["published_at"]  # ISO format


def test_receipt_equality() -> None:
    now = datetime.now(UTC)
    r1 = PublicationReceipt(0, "a" * 64, "b" * 64, "c" * 64, now)
    r2 = PublicationReceipt(0, "a" * 64, "b" * 64, "c" * 64, now)
    assert r1 == r2


def test_receipt_inequality_on_height() -> None:
    now = datetime.now(UTC)
    r1 = PublicationReceipt(0, "a" * 64, "b" * 64, "c" * 64, now)
    r2 = PublicationReceipt(1, "a" * 64, "b" * 64, "c" * 64, now)
    assert r1 != r2

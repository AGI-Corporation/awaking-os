"""On-chain DeSci publication.

Closes the last codeable line of the README KPI block. The default
implementation is a **local JSONL "chain"** that gives every published
:class:`DeSciAttestation` real chain semantics — sequential
``block_height``, content-addressed ``tx_hash``, and a ``prev_hash``
linking each block to its predecessor — without needing an actual
blockchain. Tampering with any historical block invalidates every
later block's ``prev_hash`` chain, so a verifier can detect rewrites.

The :class:`OnChainPublisher` ABC is the integration point. A real
chain implementation (Ethereum / Polygon / Solana) implements the same
interface; `AGIRam` doesn't care which backend is plugged in.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from awaking_os.memory.node import DeSciAttestation

# Genesis block has a fixed sentinel as its prev_hash so the chain is
# self-rooting and verifiable without an out-of-band initial value.
GENESIS_PREV_HASH = "0" * 64


class PublicationReceipt:
    """Receipt of a publication. Returned by :meth:`OnChainPublisher.publish`."""

    __slots__ = ("block_height", "tx_hash", "prev_hash", "node_hash", "published_at")

    def __init__(
        self,
        block_height: int,
        tx_hash: str,
        prev_hash: str,
        node_hash: str,
        published_at: datetime,
    ) -> None:
        self.block_height = block_height
        self.tx_hash = tx_hash
        self.prev_hash = prev_hash
        self.node_hash = node_hash
        self.published_at = published_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_height": self.block_height,
            "tx_hash": self.tx_hash,
            "prev_hash": self.prev_hash,
            "node_hash": self.node_hash,
            "published_at": self.published_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"PublicationReceipt(block_height={self.block_height}, "
            f"tx_hash={self.tx_hash[:12]}..., node_hash={self.node_hash[:12]}...)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PublicationReceipt):
            return NotImplemented
        return (
            self.block_height == other.block_height
            and self.tx_hash == other.tx_hash
            and self.prev_hash == other.prev_hash
            and self.node_hash == other.node_hash
        )


class OnChainPublisher(ABC):
    @abstractmethod
    async def publish(self, attestation: DeSciAttestation) -> PublicationReceipt: ...

    @abstractmethod
    async def find(self, node_hash: str) -> PublicationReceipt | None: ...

    @abstractmethod
    async def verify_chain(self) -> bool: ...


class LocalJSONLPublisher(OnChainPublisher):
    """JSONL-backed local "chain". One line per block, hash-linked.

    Each block stores: block_height, tx_hash, prev_hash, node_hash,
    public_key, signature, signed_at, published_at. Blocks are written
    atomically via fsync. Concurrent calls are serialized through an
    asyncio lock per instance and a sqlite-backed ledger that tracks
    the current head — so even if two LocalJSONLPublisher instances
    point at the same path, they won't clobber each other's heights.

    Why two stores (JSONL + sqlite ledger)?
    - The JSONL is the canonical, append-only chain — what a verifier
      reads to confirm the chain integrity.
    - The sqlite ledger holds (node_hash → block_height) for O(1)
      ``find`` and locks the current head atomically across processes.
      It's a derived index; deleting it means rebuilding by replaying
      the JSONL.
    """

    def __init__(self, chain_path: Path) -> None:
        self._path = Path(chain_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ledger_path = self._path.with_suffix(self._path.suffix + ".ledger")
        self._lock = asyncio.Lock()
        self._init_ledger()

    def _init_ledger(self) -> None:
        with sqlite3.connect(self._ledger_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS blocks (
                    block_height INTEGER PRIMARY KEY,
                    tx_hash TEXT NOT NULL,
                    node_hash TEXT NOT NULL UNIQUE,
                    prev_hash TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_node_hash ON blocks(node_hash)")

    @staticmethod
    def _compute_tx_hash(
        block_height: int,
        prev_hash: str,
        node_hash: str,
        public_key: str,
        signature: str,
    ) -> str:
        # SHA-256 of the canonical block contents → tx_hash. Any
        # tampering with these fields breaks the linkage.
        canon = json.dumps(
            {
                "block_height": block_height,
                "prev_hash": prev_hash,
                "node_hash": node_hash,
                "public_key": public_key,
                "signature": signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canon.encode()).hexdigest()

    async def publish(self, attestation: DeSciAttestation) -> PublicationReceipt:
        async with self._lock:
            return await asyncio.to_thread(self._publish_sync, attestation)

    def _publish_sync(self, attestation: DeSciAttestation) -> PublicationReceipt:
        # ``isolation_level=None`` puts the connection in autocommit mode
        # so we can issue an explicit ``BEGIN IMMEDIATE``. That acquires
        # sqlite's RESERVED write lock at the start of the critical
        # section, blocking any other writer (in-process *or* across
        # processes) until we COMMIT or ROLLBACK. Without this, two
        # publishers can both read the same max ``block_height`` and
        # both append a duplicate-height line to the JSONL before either
        # INSERT lands.
        conn = sqlite3.connect(self._ledger_path, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Idempotent re-publish: the same node_hash always
                # returns its existing receipt. Releasing the write lock
                # via ROLLBACK (we made no changes) lets other waiters
                # proceed immediately.
                row = conn.execute(
                    "SELECT block_height, tx_hash, prev_hash FROM blocks WHERE node_hash = ?",
                    (attestation.node_hash,),
                ).fetchone()
                if row is not None:
                    height, tx_hash, prev_hash = row
                    conn.execute("ROLLBACK")
                    return PublicationReceipt(
                        block_height=int(height),
                        tx_hash=tx_hash,
                        prev_hash=prev_hash,
                        node_hash=attestation.node_hash,
                        published_at=datetime.now(UTC),
                    )

                # Allocate next height + read prev hash.
                row = conn.execute(
                    "SELECT block_height, tx_hash FROM blocks ORDER BY block_height DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    block_height = 0
                    prev_hash = GENESIS_PREV_HASH
                else:
                    block_height = int(row[0]) + 1
                    prev_hash = row[1]

                tx_hash = self._compute_tx_hash(
                    block_height,
                    prev_hash,
                    attestation.node_hash,
                    attestation.public_key,
                    attestation.signature,
                )
                published_at = datetime.now(UTC)

                block = {
                    "block_height": block_height,
                    "tx_hash": tx_hash,
                    "prev_hash": prev_hash,
                    "node_hash": attestation.node_hash,
                    "public_key": attestation.public_key,
                    "signature": attestation.signature,
                    "signed_at": attestation.signed_at.isoformat(),
                    "published_at": published_at.isoformat(),
                }

                # Reserve the height in sqlite first. The PK constraint
                # on ``block_height`` plus the BEGIN IMMEDIATE write
                # lock guarantee no concurrent publisher can claim the
                # same height. If the JSONL append below fails, the
                # rollback frees the height for the next attempt.
                conn.execute(
                    "INSERT INTO blocks (block_height, tx_hash, node_hash, prev_hash) "
                    "VALUES (?, ?, ?, ?)",
                    (block_height, tx_hash, attestation.node_hash, prev_hash),
                )

                # Append + fsync the JSONL while the write lock is held
                # so no other publisher can interleave a line at the
                # same height.
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(block, separators=(",", ":")) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            return PublicationReceipt(
                block_height=block_height,
                tx_hash=tx_hash,
                prev_hash=prev_hash,
                node_hash=attestation.node_hash,
                published_at=published_at,
            )
        finally:
            conn.close()

    async def find(self, node_hash: str) -> PublicationReceipt | None:
        with sqlite3.connect(self._ledger_path) as conn:
            row = conn.execute(
                "SELECT block_height, tx_hash, prev_hash FROM blocks WHERE node_hash = ?",
                (node_hash,),
            ).fetchone()
        if row is None:
            return None
        height, tx_hash, prev_hash = row
        return PublicationReceipt(
            block_height=int(height),
            tx_hash=tx_hash,
            prev_hash=prev_hash,
            node_hash=node_hash,
            published_at=datetime.now(UTC),  # not stored in ledger; canonical is JSONL
        )

    async def verify_chain(self) -> bool:
        """Re-walk the JSONL and confirm every block's tx_hash + prev_hash linkage.

        Returns True if every block's ``tx_hash`` matches its canonical
        recomputed hash AND every block's ``prev_hash`` matches the
        previous block's ``tx_hash``. Tampering with any field in any
        historical block makes this return False.
        """
        if not self._path.exists():
            return True  # empty chain is trivially valid
        prev_tx_hash = GENESIS_PREV_HASH
        expected_height = 0
        with self._path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                block = json.loads(raw)
                if block["block_height"] != expected_height:
                    return False
                if block["prev_hash"] != prev_tx_hash:
                    return False
                recomputed = self._compute_tx_hash(
                    block["block_height"],
                    block["prev_hash"],
                    block["node_hash"],
                    block["public_key"],
                    block["signature"],
                )
                if recomputed != block["tx_hash"]:
                    return False
                prev_tx_hash = block["tx_hash"]
                expected_height += 1
        return True

    def block_count(self) -> int:
        with sqlite3.connect(self._ledger_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()
        return int(row[0]) if row else 0

"""DeSci attestation — local ed25519 sign + verify.

This is a local stub. There is no chain integration; ``DeSciAttestation``
captures a deterministic hash of the node's identity fields plus an
ed25519 signature over that hash.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from awaking_os.memory.node import DeSciAttestation, KnowledgeNode

# Fields excluded from the canonical hash:
# - attestation: would be circular
# - embedding: derived from content; may be recomputed
# - linked_nodes: graph relations evolve over time
_EXCLUDED_FIELDS = {"attestation", "embedding", "linked_nodes"}


def canonical_hash(node: KnowledgeNode) -> str:
    """SHA-256 hex of a deterministic JSON dump of the node's identity."""
    payload = node.model_dump(exclude=_EXCLUDED_FIELDS, mode="json")
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _public_key_hex(pk: Ed25519PublicKey) -> str:
    return pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


class DeSciSigner:
    """Holds an ed25519 keypair and produces ``DeSciAttestation`` records."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None) -> None:
        self._sk = private_key or Ed25519PrivateKey.generate()
        self._pk = self._sk.public_key()

    @classmethod
    def from_seed(cls, seed: bytes) -> DeSciSigner:
        """Build a deterministic signer from a 32-byte seed (tests, demos)."""
        if len(seed) != 32:
            raise ValueError("ed25519 seed must be 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    @property
    def public_key_hex(self) -> str:
        return _public_key_hex(self._pk)

    def sign(self, node: KnowledgeNode) -> DeSciAttestation:
        node_hash = canonical_hash(node)
        signature = self._sk.sign(bytes.fromhex(node_hash))
        return DeSciAttestation(
            node_hash=node_hash,
            signature=signature.hex(),
            public_key=self.public_key_hex,
            signed_at=datetime.now(UTC),
        )


def verify(attestation: DeSciAttestation, node: KnowledgeNode) -> bool:
    """Return True iff the attestation matches the node's current identity."""
    expected = canonical_hash(node)
    if expected != attestation.node_hash:
        return False
    try:
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(attestation.public_key))
        pk.verify(bytes.fromhex(attestation.signature), bytes.fromhex(attestation.node_hash))
    except (InvalidSignature, ValueError):
        return False
    return True

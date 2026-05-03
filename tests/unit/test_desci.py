"""DeSci attestation tests."""

from __future__ import annotations

import pytest

from awaking_os.memory.desci import DeSciSigner, canonical_hash, verify
from awaking_os.memory.node import KnowledgeNode

_SEED = b"\x00" * 32
_OTHER_SEED = b"\x01" * 32


def _node(content: str = "alpha", **kwargs) -> KnowledgeNode:
    return KnowledgeNode(content=content, created_by="test", **kwargs)


def test_canonical_hash_is_deterministic() -> None:
    n = _node()
    assert canonical_hash(n) == canonical_hash(n)


def test_canonical_hash_excludes_embedding() -> None:
    n1 = _node()
    n2 = n1.model_copy(update={"embedding": [0.1, 0.2, 0.3]})
    assert canonical_hash(n1) == canonical_hash(n2)


def test_canonical_hash_excludes_linked_nodes() -> None:
    n1 = _node()
    n2 = n1.model_copy(update={"linked_nodes": ["other"]})
    assert canonical_hash(n1) == canonical_hash(n2)


def test_canonical_hash_changes_when_content_changes() -> None:
    n1 = _node("alpha")
    n2 = n1.model_copy(update={"content": "bravo"})
    assert canonical_hash(n1) != canonical_hash(n2)


def test_signer_sign_and_verify_roundtrip() -> None:
    signer = DeSciSigner.from_seed(_SEED)
    n = _node()
    att = signer.sign(n)
    assert att.public_key == signer.public_key_hex
    assert att.node_hash == canonical_hash(n)
    assert verify(att, n)


def test_verify_rejects_tampered_content() -> None:
    signer = DeSciSigner.from_seed(_SEED)
    n = _node("alpha")
    att = signer.sign(n)
    tampered = n.model_copy(update={"content": "bravo"})
    assert not verify(att, tampered)


def test_verify_rejects_wrong_signer() -> None:
    a = DeSciSigner.from_seed(_SEED)
    b = DeSciSigner.from_seed(_OTHER_SEED)
    n = _node()
    att = a.sign(n)
    forged = att.model_copy(update={"public_key": b.public_key_hex})
    assert not verify(forged, n)


def test_verify_rejects_garbage_signature() -> None:
    signer = DeSciSigner.from_seed(_SEED)
    n = _node()
    att = signer.sign(n)
    garbage = att.model_copy(update={"signature": "00" * 64})
    assert not verify(garbage, n)


def test_signer_is_deterministic_from_seed() -> None:
    a = DeSciSigner.from_seed(_SEED)
    b = DeSciSigner.from_seed(_SEED)
    assert a.public_key_hex == b.public_key_hex
    n = _node()
    # ed25519 signatures are deterministic for the same key + message
    assert a.sign(n).signature == b.sign(n).signature


def test_signer_random_keys_differ() -> None:
    a = DeSciSigner()
    b = DeSciSigner()
    assert a.public_key_hex != b.public_key_hex


def test_seed_must_be_32_bytes() -> None:
    with pytest.raises(ValueError):
        DeSciSigner.from_seed(b"\x00" * 31)

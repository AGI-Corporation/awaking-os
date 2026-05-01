"""NetworkXKnowledgeGraph tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from awaking_os.memory.knowledge_graph import NetworkXKnowledgeGraph
from awaking_os.memory.node import KnowledgeNode


def _node(content: str) -> KnowledgeNode:
    return KnowledgeNode(content=content, created_by="test")


def test_add_and_get_in_memory() -> None:
    g = NetworkXKnowledgeGraph()
    n = _node("alpha")
    g.add(n)
    assert n.id in g
    assert len(g) == 1
    fetched = g.get(n.id)
    assert fetched is not None
    assert fetched.id == n.id
    assert fetched.content == "alpha"


def test_get_missing_returns_none() -> None:
    g = NetworkXKnowledgeGraph()
    assert g.get("nope") is None


def test_link_requires_both_endpoints() -> None:
    g = NetworkXKnowledgeGraph()
    a = _node("a")
    g.add(a)
    with pytest.raises(KeyError):
        g.link(a.id, "missing", "rel")


def test_neighbors_walks_outgoing_edges() -> None:
    g = NetworkXKnowledgeGraph()
    a, b, c = _node("a"), _node("b"), _node("c")
    for n in (a, b, c):
        g.add(n)
    g.link(a.id, b.id, "next")
    g.link(b.id, c.id, "next")

    one_hop = g.neighbors(a.id, depth=1)
    assert [n.id for n in one_hop] == [b.id]

    two_hop_ids = {n.id for n in g.neighbors(a.id, depth=2)}
    assert two_hop_ids == {b.id, c.id}


def test_neighbors_for_unknown_node_is_empty() -> None:
    g = NetworkXKnowledgeGraph()
    assert g.neighbors("nope", depth=2) == []


def test_persistence_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "g.sqlite"
    g1 = NetworkXKnowledgeGraph(db_path=db)
    a, b = _node("alpha"), _node("beta")
    g1.add(a)
    g1.add(b)
    g1.link(a.id, b.id, "knows")

    g2 = NetworkXKnowledgeGraph(db_path=db)
    assert len(g2) == 2
    assert g2.get(a.id) is not None
    assert g2.get(b.id) is not None
    assert {n.id for n in g2.neighbors(a.id, depth=1)} == {b.id}


def test_all_nodes_returns_every_node() -> None:
    g = NetworkXKnowledgeGraph()
    ids = []
    for content in ("one", "two", "three"):
        n = _node(content)
        g.add(n)
        ids.append(n.id)
    assert {n.id for n in g.all_nodes()} == set(ids)

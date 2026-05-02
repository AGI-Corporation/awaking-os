"""NetworkX-backed knowledge graph with optional sqlite snapshot persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx

from awaking_os.memory.node import KnowledgeNode


class NetworkXKnowledgeGraph:
    """In-memory directed graph; sqlite snapshot is opt-in via ``db_path``.

    Nodes are stored with the full pydantic JSON dump under a ``data``
    attribute; edges carry a ``relation`` string.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._db_path = db_path
        if db_path is not None:
            self._init_db()
            self._load()

    def _init_db(self) -> None:
        assert self._db_path is not None
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    PRIMARY KEY (source, target, relation)
                )"""
            )

    def _load(self) -> None:
        assert self._db_path is not None
        with sqlite3.connect(self._db_path) as conn:
            for node_id, data in conn.execute("SELECT id, data FROM nodes"):
                self._g.add_node(node_id, data=data)
            for source, target, relation in conn.execute(
                "SELECT source, target, relation FROM edges"
            ):
                self._g.add_edge(source, target, key=relation, relation=relation)

    def _persist_node(self, node: KnowledgeNode) -> None:
        if self._db_path is None:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, data) VALUES (?, ?)",
                (node.id, node.model_dump_json()),
            )

    def _persist_edge(self, source: str, target: str, relation: str) -> None:
        if self._db_path is None:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source, target, relation) VALUES (?, ?, ?)",
                (source, target, relation),
            )

    def add(self, node: KnowledgeNode) -> str:
        self._g.add_node(node.id, data=node.model_dump_json())
        self._persist_node(node)
        return node.id

    def get(self, node_id: str) -> KnowledgeNode | None:
        if node_id not in self._g:
            return None
        data = self._g.nodes[node_id].get("data")
        if data is None:
            return None
        return KnowledgeNode.model_validate(json.loads(data))

    def link(self, source: str, target: str, relation: str) -> None:
        if source not in self._g or target not in self._g:
            raise KeyError("Both nodes must exist before linking")
        self._g.add_edge(source, target, key=relation, relation=relation)
        self._persist_edge(source, target, relation)

    def remove(self, node_id: str) -> bool:
        """Remove a node and all its incident edges. Returns True if it existed."""
        if node_id not in self._g:
            return False
        self._g.remove_node(node_id)  # also drops all incident edges
        if self._db_path is not None:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                conn.execute(
                    "DELETE FROM edges WHERE source = ? OR target = ?",
                    (node_id, node_id),
                )
        return True

    def unlink(self, source: str, target: str, relation: str | None = None) -> int:
        """Remove edge(s) between source and target.

        If ``relation`` is given, removes only that edge; otherwise removes
        every edge between the two nodes. Returns the number of edges removed.
        Missing endpoints are a no-op (returns 0).
        """
        if source not in self._g or target not in self._g:
            return 0
        if relation is not None:
            if not self._g.has_edge(source, target, key=relation):
                return 0
            self._g.remove_edge(source, target, key=relation)
            removed = 1
            if self._db_path is not None:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "DELETE FROM edges WHERE source = ? AND target = ? AND relation = ?",
                        (source, target, relation),
                    )
        else:
            edges = list(self._g[source].get(target, {}).keys())
            for key in edges:
                self._g.remove_edge(source, target, key=key)
            removed = len(edges)
            if self._db_path is not None and removed:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "DELETE FROM edges WHERE source = ? AND target = ?",
                        (source, target),
                    )
        return removed

    def neighbors(self, node_id: str, depth: int = 1) -> list[KnowledgeNode]:
        """BFS up to ``depth`` outgoing hops; deterministic order."""
        if node_id not in self._g:
            return []
        seen = {node_id}
        frontier = [node_id]
        out: list[KnowledgeNode] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for current in frontier:
                for nbr in sorted(self._g.successors(current)):
                    if nbr in seen:
                        continue
                    seen.add(nbr)
                    next_frontier.append(nbr)
                    node = self.get(nbr)
                    if node is not None:
                        out.append(node)
            frontier = next_frontier
            if not frontier:
                break
        return out

    def all_nodes(self) -> list[KnowledgeNode]:
        return [self.get(n) for n in self._g.nodes if self.get(n) is not None]  # type: ignore[misc]

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._g

"""AGI-RAM facade — store, retrieve, link knowledge nodes.

PR 1: retrieve uses keyword matching against the graph. Embeddings + Chroma
arrive in PR 2.
"""

from __future__ import annotations

import re
from pathlib import Path

from awaking_os.memory.knowledge_graph import NetworkXKnowledgeGraph
from awaking_os.memory.node import KnowledgeNode

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


class AGIRam:
    def __init__(self, db_path: Path | None = None) -> None:
        self.graph = NetworkXKnowledgeGraph(db_path=db_path)

    async def store(self, node: KnowledgeNode) -> str:
        return self.graph.add(node)

    async def get(self, node_id: str) -> KnowledgeNode | None:
        return self.graph.get(node_id)

    async def link(self, source: str, target: str, relation: str) -> None:
        self.graph.link(source, target, relation)

    async def retrieve(self, query: str, k: int = 5) -> list[KnowledgeNode]:
        """Keyword-overlap ranking. Replaced by semantic search in PR 2."""
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        scored: list[tuple[int, KnowledgeNode]] = []
        for node in self.graph.all_nodes():
            n_tokens = _tokens(node.content)
            overlap = len(q_tokens & n_tokens)
            if overlap > 0:
                scored.append((overlap, node))
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at))
        return [node for _, node in scored[:k]]

    def __len__(self) -> int:
        return len(self.graph)

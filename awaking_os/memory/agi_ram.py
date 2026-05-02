"""AGI-RAM facade — store, retrieve, link knowledge nodes.

When wired with an :class:`EmbeddingProvider` and a :class:`VectorStore`,
``retrieve`` runs semantic similarity search. Without those, it falls
back to keyword-overlap ranking (used by tests that don't need real
embeddings).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from awaking_os.memory.embeddings import EmbeddingProvider
from awaking_os.memory.knowledge_graph import NetworkXKnowledgeGraph
from awaking_os.memory.node import KnowledgeNode
from awaking_os.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


class AGIRam:
    def __init__(
        self,
        db_path: Path | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        signer: object | None = None,  # DeSciSigner; kept loose to avoid import cycle
        publisher: object | None = None,  # OnChainPublisher; loose for the same reason
    ) -> None:
        self.graph = NetworkXKnowledgeGraph(db_path=db_path)
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.signer = signer
        self.publisher = publisher
        # Records publication receipts keyed by node_id, populated as
        # store()'s background publish completes. Useful for tests and
        # for callers that want to surface chain receipts in their UI.
        self.receipts: dict[str, object] = {}

    @property
    def semantic_enabled(self) -> bool:
        return self.vector_store is not None and self.embedding_provider is not None

    async def store(self, node: KnowledgeNode) -> str:
        """Store a node atomically across the graph and vector store.

        If the vector-store upsert fails, the graph add is rolled back so
        the two stores never drift. The original exception is re-raised
        so callers see the failure. After a successful store, if a
        ``publisher`` is wired AND the node has an attestation, the
        attestation is published on-chain (the JSONL chain by default).
        Publication failures are logged but never propagate — the store
        succeeds even when the chain is offline.
        """
        if node.embedding is None and self.embedding_provider is not None:
            node.embedding = await self.embedding_provider.embed(node.content)
        if self.signer is not None and node.attestation is None:
            node.attestation = self.signer.sign(node)  # type: ignore[attr-defined]
        node_id = self.graph.add(node)
        if self.vector_store is not None and node.embedding is not None:
            try:
                await self.vector_store.upsert(
                    node_id=node_id,
                    embedding=node.embedding,
                    metadata={"type": node.type, "created_by": node.created_by},
                )
            except Exception:
                # Compensate so the graph and vector index stay in sync. If
                # the rollback itself fails, log it but raise the original.
                try:
                    self.graph.remove(node_id)
                except Exception:
                    logger.exception(
                        "Failed to roll back graph node %s after vector upsert failure",
                        node_id,
                    )
                raise

        if self.publisher is not None and node.attestation is not None:
            try:
                receipt = await self.publisher.publish(node.attestation)  # type: ignore[attr-defined]
                self.receipts[node_id] = receipt
            except Exception:
                # Publication is a best-effort step — never fail the
                # store on a chain hiccup. Caller can re-publish later.
                logger.exception("Failed to publish attestation for node %s", node_id)
        return node_id

    async def get(self, node_id: str) -> KnowledgeNode | None:
        return self.graph.get(node_id)

    async def link(self, source: str, target: str, relation: str) -> None:
        self.graph.link(source, target, relation)

    async def retrieve(self, query: str, k: int = 5) -> list[KnowledgeNode]:
        if self.semantic_enabled:
            return await self._semantic_retrieve(query, k)
        return await self._keyword_retrieve(query, k)

    async def _semantic_retrieve(self, query: str, k: int) -> list[KnowledgeNode]:
        assert self.embedding_provider is not None and self.vector_store is not None
        if not query.strip():
            return []
        q_emb = await self.embedding_provider.embed(query)
        hits = await self.vector_store.query(q_emb, k=k)
        out: list[KnowledgeNode] = []
        for hit in hits:
            node = self.graph.get(hit.node_id)
            if node is not None:
                out.append(node)
        return out

    async def _keyword_retrieve(self, query: str, k: int) -> list[KnowledgeNode]:
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

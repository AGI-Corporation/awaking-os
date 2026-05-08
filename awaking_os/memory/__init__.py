"""AGI-RAM: knowledge graph, vector store, attestation."""

from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.desci import DeSciSigner, canonical_hash, verify
from awaking_os.memory.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from awaking_os.memory.knowledge_graph import NetworkXKnowledgeGraph
from awaking_os.memory.node import DeSciAttestation, KnowledgeNode
from awaking_os.memory.onchain import (
    GENESIS_PREV_HASH,
    LocalJSONLPublisher,
    OnChainPublisher,
    PublicationReceipt,
)
from awaking_os.memory.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorHit,
    VectorStore,
)

__all__ = [
    "AGIRam",
    "ChromaVectorStore",
    "DeSciAttestation",
    "DeSciSigner",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "GENESIS_PREV_HASH",
    "InMemoryVectorStore",
    "KnowledgeNode",
    "LocalJSONLPublisher",
    "NetworkXKnowledgeGraph",
    "OnChainPublisher",
    "PublicationReceipt",
    "SentenceTransformerEmbeddingProvider",
    "VectorHit",
    "VectorStore",
    "canonical_hash",
    "verify",
]

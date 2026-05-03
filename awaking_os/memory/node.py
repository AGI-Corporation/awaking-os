"""KnowledgeNode + DeSciAttestation models."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

NodeType = Literal["concept", "entity", "event", "research"]


class DeSciAttestation(BaseModel):
    node_hash: str
    signature: str
    public_key: str
    signed_at: datetime


class KnowledgeNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: NodeType = "concept"
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attestation: DeSciAttestation | None = None
    linked_nodes: list[str] = Field(default_factory=list)
    created_by: str = "system"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

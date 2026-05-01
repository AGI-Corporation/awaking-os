"""AgentTask, AgentContext, AgentResult — the kernel's data primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType, TokenBudget


class AgentTask(BaseModel):
    id: str
    priority: int = Field(default=50, ge=0, le=100)
    agent_type: AgentType
    payload: dict[str, Any] = Field(default_factory=dict)
    context_window: TokenBudget = Field(default_factory=TokenBudget)
    ethical_constraints: list[str] = Field(default_factory=list)
    deadline: datetime | None = None


class AgentContext(BaseModel):
    task: AgentTask
    memory: list[KnowledgeNode] = Field(default_factory=list)
    ethical_boundary: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    task_id: str
    agent_id: str
    output: dict[str, Any] = Field(default_factory=dict)
    knowledge_nodes_created: list[str] = Field(default_factory=list)
    phi_contribution: float = 0.0
    elapsed_ms: int = 0

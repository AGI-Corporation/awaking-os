"""AgentTask, AgentContext, AgentResult — the kernel's data primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from awaking_os.kernel.retry import RetryPolicy
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
    retry_policy: RetryPolicy | None = None
    # Number of attempts already made on this task. The kernel bumps
    # this on each retry; agents typically don't read it directly, but
    # idempotent agents can use ``(task.id, task.attempts)`` as a
    # dedupe key if they need attempt-specific behavior.
    attempts: int = Field(default=0, ge=0)


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
